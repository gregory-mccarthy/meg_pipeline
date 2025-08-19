#!/usr/bin/env python3
# ==========================================================
# MNE-PYTHON PREPROCESSING PIPELINE — HYBRID-READY MAIN
# (integrates HybridTransferManager with minimal code churn)
# ==========================================================
"""
This main preserves your local-only flow exactly as before, and
adds optional hybrid behavior:

HYBRID MODE (enabled only when requested in YAML):
  - Prefetch (rsync) raw + any derivatives (parproc) from HPC
    to ./temp/bids using HybridTransferManager.fetch_all_bids_data()
  - Run Stage 1 and/or Stage 2 locally using the LOCAL temp BIDS
  - Push derivatives (including parproc + preproc) back to HPC at end
    using HybridTransferManager.push_results()

LOCAL MODE:
  - No transfer manager is used. Behavior unchanged.

YAML keys (compatible with your earlier examples):
  remote_io:
    enabled: true
    hpc_host: transfer-milgram.ycrc.yale.edu
    hpc_user: gm33
    remote_bids_root: /gpfs/milgram/scratch/mccarthy/gm33/BIDS/epi
    local_temp_dir: ./temp
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import mne
import json
import numpy as np
import matplotlib
# Choose an interactive backend when a display is available; fall back to Agg in headless/SLURM
def _in_slurm() -> bool:
    return any(k in os.environ for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_SUBMIT_DIR"))

def _has_display() -> bool:
    # On macOS, a window server is present when running locally.
    if sys.platform.startswith("darwin"):
        return True
    # On Linux (e.g., OOD), DISPLAY or WAYLAND_DISPLAY indicates a GUI session
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# Allow an explicit override if you ever need it:
_override = os.environ.get("MPLBACKEND_OVERRIDE") or os.environ.get("MPLBACKEND")

if _override:
    matplotlib.use(_override, force=True)
elif _in_slurm() or not _has_display():
    matplotlib.use("Agg", force=True)   # headless: no interactive windows
else:
    matplotlib.use("QtAgg", force=True) # interactive (Mac, OOD desktop)

# ---------- utils import (support v3 or v2) ----------
try:
    import meg_pipeline_utils_v3 as utils
except ImportError:
    import meg_pipeline_utils as utils

# ---------- optional transfer manager (only used in hybrid) ----------
try:
    from transfer_manager import HybridTransferManager
except ImportError:
    HybridTransferManager = None  # OK for local-only mode

# ---------- logging/version ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")
PIPELINE_VERSION = "1.2"
CHECKPOINT_VERSION = "1.0"  # checkpoint format compatibility


# ==========================================================
# PIPELINE STAGE DETECTION (unchanged logic, uses p["bids_root"])
# ==========================================================
def detect_pipeline_stage(yaml_path: str, p: dict, force_stage: str = None):
    """
    Decide whether to start at Stage 1 (no checkpoint) or Stage 2 (resume from checkpoint).
    Looks for parproc fif + manifest under:
      {bids_root}/derivatives/{pipeline_name}/sub-*/ses-*/meg/*_desc-parproc_meg.fif
    """
    if force_stage:
        logger.info(f"🔧 Forcing pipeline stage: {force_stage}")
        if force_stage == "stage1":
            return ("stage1", None)
        elif force_stage != "stage2":
            logger.error(f"Invalid forced stage: {force_stage}")
            sys.exit(1)

    chk_cfg = p.get("checkpoint", {})
    enabled = chk_cfg.get("enabled", True)
    if not enabled and not force_stage:
        logger.info("Checkpoint system disabled in configuration")
        return ("stage1", None)

    derivatives_root = chk_cfg.get("derivatives_root", "derivatives")
    pipeline_name = chk_cfg.get("pipeline_name", "preprocessing")

    bids_root = Path(p["bids_root"])
    deriv_root = bids_root / derivatives_root / pipeline_name

    subject_dir = f"sub-{p['subject']}"
    if p.get("session"):
        checkpoint_dir = deriv_root / subject_dir / f"ses-{p['session']}" / "meg"
    else:
        checkpoint_dir = deriv_root / subject_dir / "meg"

    base_name = f"sub-{p['subject']}"
    if p.get("session"): base_name += f"_ses-{p['session']}"
    if p.get("task"):    base_name += f"_task-{p['task']}"
    if p.get("run"):     base_name += f"_run-{p['run']}"

    parproc_fif = checkpoint_dir / f"{base_name}_desc-parproc_meg.fif"
    manifest_path = parproc_fif.with_name(parproc_fif.stem + "_manifest.yaml")

    if parproc_fif.exists() and manifest_path.exists():
        logger.info(f"🔄 Checkpoint detected: {parproc_fif}")
        # basic version/age check (non-interactive in stage detect)
        try:
            man = utils.load_yaml(str(manifest_path))
            chk_version = man.get("checkpoint_version", "0.0")
            if chk_version != CHECKPOINT_VERSION:
                logger.warning(f"Checkpoint version mismatch (found {chk_version}, expect {CHECKPOINT_VERSION})")
            return ("stage2", parproc_fif)
        except Exception as e:
            logger.warning(f"Manifest read failed, ignoring checkpoint: {e}")
            return ("stage1", None)

    if force_stage == "stage2":
        logger.error(f"Cannot force Stage 2: checkpoint not found at {parproc_fif}")
        sys.exit(1)

    logger.info("🆕 No checkpoint found → Stage 1")
    return ("stage1", None)


def validate_checkpoint_integrity(checkpoint_path: Path, manifest_path: Path) -> bool:
    try:
        if checkpoint_path.stat().st_size < 1024:
            logger.error("Checkpoint file is suspiciously small")
            return False
        raw_test = mne.io.read_raw_fif(str(checkpoint_path), preload=False, verbose='ERROR')
        n_channels = len(raw_test.ch_names)
        duration = raw_test.times[-1]
        logger.info(f"Checkpoint validation: {n_channels} channels, {duration:.1f}s duration")
        del raw_test

        man = utils.load_yaml(str(manifest_path))
        for key in ["stage", "created_utc", "inputs", "artifacts"]:
            if key not in man:
                logger.error(f"Manifest missing required key: {key}")
                return False
        return True
    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        return False


# ==========================================================
# STAGE 1 (unchanged core; writes parproc checkpoint)
# ==========================================================
def run_stage1_pipeline(yaml_path: str, p: dict):
    p["_cfg_path"] = yaml_path
    repo_root = Path(__file__).resolve().parent
    logger.info(f"Repository root: {repo_root}")

    try:
        checked_paths = utils.check_critical_files_exist(p, repo_root)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e)); sys.exit(1)

    for key, path in checked_paths.items():
        logger.info(f"  {key}: {path}")

    subject = p["subject"]
    session = p.get("session")
    task    = p.get("task")
    run     = p.get("run")
    bids_root = p["bids_root"]

    # ---------------- 2. Load Raw ----------------
    utils.log_section("2. Load Raw Data")
    bids_path = utils.BIDSPath(subject=subject, session=session, task=task, run=run,
                               datatype="meg", root=bids_root)
    raw = utils.read_raw_bids_robust(bids_path)
    raw_fif_path = raw.filenames[0]
    meg_dir = os.path.dirname(raw_fif_path)

    expected_pos = utils.get_bids_headpos_path(subject, session, task, run, meg_dir)
    pos_file_requested = expected_pos if os.path.exists(expected_pos) else None
    logger.info(f"Head position file: {pos_file_requested if pos_file_requested else 'Will compute if needed'}")

    # ---------------- 3. Head Movement ----------------
    utils.log_section("3. Compute or Load Head Position")
    head_movement_cfg = p.get("head_movement", {})
    movement_enabled = head_movement_cfg.get("enabled", False)
    head_pos_array = None
    if movement_enabled:
        head_pos_path = utils.get_head_pos_for_maxwell(raw, pos_file=pos_file_requested,
                                                       compute_if_missing=True, logger=logger)
        if head_pos_path and os.path.exists(head_pos_path):
            try:
                head_pos_array = utils.mne.chpi.read_head_pos(head_pos_path)
            except Exception as e:
                logger.warning(f"Could not read head position file: {head_pos_path}\n{e}")

    # Paths for derivatives (local)
    bids_path_deriv = bids_path.copy().update(
        root=Path(bids_root) / "derivatives" / p.get("checkpoint", {}).get("pipeline_name", "preprocessing"),
        suffix="meg", description="preproc", extension=".fif"
    )
    out_fif = bids_path_deriv.fpath
    plots_dir = os.path.join(os.path.dirname(out_fif), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Save head movement plot if available
    if movement_enabled and head_pos_array is not None:
        utils.plot_head_movement(head_pos_array, plots_dir)

    # ---------------- 4. EEG Setup ----------------
    utils.log_section("4. EEG Channel Setup")
    utils.prepare_eeg_channels(raw, str(checked_paths["montage"]), logger)

    # ---------------- 5. Metadata ----------------
    utils.log_section("5. Metadata Repair")
    utils.apply_metadata_repairs(raw, p.get('metadata_fixes', {}))

    # ---------------- 6. QC ----------------
    utils.log_section("6. PSD & RMS Diagnostics (Pre-filtering)")
    utils.qc_meg_raw(raw, plots_dir)
    utils.plot_psd_and_peaks(raw, "Raw PSD Before Maxwell", plots_dir)

    # ---------------- 7. Maxwell ----------------
    utils.log_section("7. Maxwell Filter (tSSS)")
    raw = utils.apply_maxwell_filter(
        raw,
        head_pos=head_pos_array,
        destination=None,
        cal=str(checked_paths["calibration_file"]),
        crosstalk=str(checked_paths["cross_talk_file"])
    )
    utils.plot_psd_and_peaks(raw, "After Maxwell", plots_dir)

    # ---------------- 8. Notch ----------------
    utils.log_section("8. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    picks = utils.mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    raw.notch_filter(notch_freqs, picks=picks, method='fir', filter_length='auto')
    utils.plot_psd_and_peaks(raw, "After Maxwell and Notch", plots_dir)

    # ---------------- 9. AutoReject + checkpoint ----------------
    utils.log_section("9. Bad Channel Detection with AutoReject (checkpoint)")
    ar_cfg = p.get("autoreject", {})

    if not ar_cfg.get("enabled", True):
        logger.info("AutoReject disabled — skipping bad channel detection")
        # But still write checkpoint with all the other Stage 1 processing

    p["_checkpoint_version"] = CHECKPOINT_VERSION
    try:
        status, raw = utils.run_autoreject_with_checkpoint(raw, p, bids_path, logger)
        if status == 'exit':
            logger.info("✅ Stage 1 completed — checkpoint saved; exit as requested")
            return ('exit', raw)
        elif status == 'continue':
            logger.info("✅ Stage 1 completed — checkpoint saved")
            return ('continue', raw)
    except Exception as e:
        logger.error(f"Checkpoint writing failed: {e}")
        resp = input("Checkpoint failed. Continue without checkpoint? (y/n): ").strip().lower()
        if resp != 'y': return ('exit', raw)
        logger.warning("Continuing without checkpoint")

    logger.error("Unexpected Stage 1 flow")
    return ('exit', raw)

# ==========================================================
# STAGE 2 (unchanged core; loads checkpoint and completes)
# ==========================================================
def run_stage2_pipeline(yaml_path: str, p: dict, checkpoint_path: Path):
    p["_cfg_path"] = yaml_path
    p["_checkpoint_version"] = CHECKPOINT_VERSION

    utils.log_section("STAGE 2: Validate + Load")
    manifest_path = checkpoint_path.with_name(checkpoint_path.stem + "_manifest.yaml")
    if not validate_checkpoint_integrity(checkpoint_path, manifest_path):
        logger.error("Checkpoint validation failed")
        resp = input("Restart from Stage 1? (y/n): ").strip().lower()
        if resp == 'y':
            run_stage1_pipeline(yaml_path, p); return
        sys.exit(1)

    # Load checkpoint raw
    raw = mne.io.read_raw_fif(str(checkpoint_path), preload=True, verbose="error")
    logger.info(f"✅ Loaded checkpoint: {len(raw.ch_names)} ch, {raw.times[-1]:.1f}s; "
                f"bads={raw.info.get('bads', [])}")

    # Derivative output paths (use current p['bids_root'])
    utils.log_section("STAGE 2: Paths / Outputs")
    bids_path = utils.BIDSPath(
        subject=p["subject"], session=p.get("session"), task=p.get("task"), run=p.get("run"),
        datatype="meg", root=p["bids_root"]
    )
    deriv_root = Path(p["bids_root"]) / "derivatives" / p.get("checkpoint", {}).get("pipeline_name", "preprocessing")
    bids_path_deriv = bids_path.copy().update(root=deriv_root, suffix="meg", description="preproc", extension=".fif")
    bids_path_ica_eeg = bids_path_deriv.copy().update(description="preprocICAeeg")
    bids_path_ica_meg = bids_path_deriv.copy().update(description="preprocICAmeg")
    out_fif = bids_path_deriv.fpath
    plots_dir = os.path.join(os.path.dirname(out_fif), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 9b. Interactive review (no recompute)
    utils.log_section("9b. Interactive Review (no AR recompute)")
    ar_cfg = p.get("autoreject", {})
    if p.get("interactive_bad_channels", True):
        # Always offer interactive review if interactive_bad_channels is True
        # regardless of whether AutoReject was enabled in Stage 1
        status, raw = utils.run_autoreject_with_checkpoint(raw, p, bids_path, logger)
        if status != 'continue':
            logger.warning(f"Unexpected status from interactive review: {status}")
    else:
        logger.info("Interactive review disabled - proceeding with current bad channels")

    # 9c. Rank (optional)
    utils.log_section("9c. Estimate MEG Rank")
    try:
        empirical_rank = mne.compute_rank(raw); logger.info(f"Empirical rank: {empirical_rank}")
    except Exception as e:
        logger.warning(f"Rank estimate failed: {e}"); empirical_rank = None

    # 10–13. ICA, events, final filter
    utils.log_section("10. ICA: EEG")
    ica_eeg_cfg = p["ica_preprocessing"]["eeg"]; eeg_exclude = []
    try:
        raw, eeg_exclude = utils.run_ica(raw, ica_eeg_cfg, bids_path_ica_eeg.fpath, modality="eeg")
    except Exception as e:
        logger.error(f"EEG ICA failed: {e}")
        if input("Continue without EEG ICA? (y/n): ").strip().lower() != 'y': sys.exit(1)

    utils.log_section("11. ICA: MEG")
    ica_meg_cfg = p["ica_preprocessing"]["meg"]; meg_exclude = []
    try:
        raw, meg_exclude = utils.run_ica(raw, ica_meg_cfg, bids_path_ica_meg.fpath, modality="meg")
    except Exception as e:
        logger.error(f"MEG ICA failed: {e}")
        if input("Continue without MEG ICA? (y/n): ").strip().lower() != 'y': sys.exit(1)

    utils.plot_psd_and_peaks(raw, "After ICA", plots_dir)

    utils.log_section("12. Event Detection")
    event_counts = {}
    try:
        events = utils.bitwise_events(raw)
        if events.size:
            annots = utils.mne.annotations_from_events(events, sfreq=raw.info['sfreq'])
            if raw.annotations is not None and len(raw.annotations) > 0:
                combined = mne.Annotations(
                    onset=list(raw.annotations.onset) + list(annots.onset),
                    duration=list(raw.annotations.duration) + list(annots.duration),
                    description=list(raw.annotations.description) + list(annots.description),
                    orig_time=raw.annotations.orig_time
                )
                raw.set_annotations(combined)
            else:
                raw.set_annotations(annots)
            codes, counts = np.unique(events[:, 2], return_counts=True)
            event_counts = {int(c): int(n) for c, n in zip(codes, counts)}
        else:
            logger.warning("No events detected")
    except Exception as e:
        logger.error(f"Event detection failed: {e}")

    utils.log_section("13. Final Filter + Cleanup")
    try:
        raw = utils.apply_final_filter_and_cleanup(raw, p)
    except Exception as e:
        logger.error(f"Final filtering failed: {e}")
        if input("Save without final filtering? (y/n): ").strip().lower() != 'y': sys.exit(1)

    # 14. Save outputs
    utils.log_section("14. Save Final Data")
    try:
        written_files = utils.write_bids_robust(raw, bids_path_deriv, overwrite=True, verbose=True)
        logger.info(f"✅ Saved: {out_fif}")
    except Exception as e:
        logger.error(f"Failed to save final data: {e}")
        emergency = Path.cwd() / f"emergency_save_{p['subject']}.fif"
        try:
            raw.save(str(emergency), overwrite=True); logger.info(f"Emergency saved: {emergency}")
        except: logger.error("Emergency save failed"); sys.exit(1)

    # 15. Logs
    utils.log_section("15. Write Logs")
    recording_duration_sec = float(raw.times[-1])
    final_filter_cfg = p.get("final_filter", {})
    drop_types = final_filter_cfg.get("drop_channel_types", [])
    dropped_channels = [ch for ch in raw.ch_names
                        if any(ch.upper().startswith(prefix.upper()) for prefix in drop_types)]
    empirical_meg_rank = (mne.compute_rank(raw).get('meg') if 'compute_rank' in dir(mne) else None)
    main_output_files = utils.get_all_bids_split_files(out_fif)
    ica_output_files = []
    for ica_fif in [bids_path_ica_eeg.fpath, bids_path_ica_meg.fpath]:
        if Path(ica_fif).exists():
            ica_output_files += utils.get_all_bids_split_files(ica_fif)
    ica_output_files = list(dict.fromkeys(ica_output_files))

    yaml_log = {
        "pipeline_version": PIPELINE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "pipeline_stage": "stage2_complete",
        "checkpoint_used": str(checkpoint_path),
        "meg_rank_estimate": empirical_meg_rank,
        "input_config": yaml_path,
        "bids_basefile": str(out_fif),
        "main_output_files": main_output_files,
        "ica_output_files": ica_output_files,
        "output_file": str(out_fif),
        "runtime_info": utils.get_runtime_info(),
        "recording_duration_sec": recording_duration_sec,
        "final_filter": {
            "highpass": final_filter_cfg.get("highpass"),
            "lowpass": final_filter_cfg.get("lowpass"),
            "resample_hz": final_filter_cfg.get("resample_hz"),
            "drop_channel_types": drop_types,
            "channels_dropped": dropped_channels
        },
        "ica": {"eeg_excluded": eeg_exclude, "meg_excluded": meg_exclude},
        "event_counts": event_counts,
        "completion_time": datetime.now(timezone.utc).isoformat()
    }
    out_log_yaml = Path(out_fif).with_name(Path(out_fif).stem + "_log.yaml")
    out_log_json = Path(out_fif).with_name(Path(out_fif).stem + "_log.json")
    try:
        utils.save_yaml(str(out_log_yaml), utils.make_serializable(yaml_log))
        with open(out_log_json, "w") as f: json.dump(utils.make_serializable(yaml_log), f, indent=2)
        logger.info(f"Logs → {out_log_yaml} / {out_log_json}")
    except Exception as e:
        logger.error(f"Failed to write logs: {e}")

    # 16. Completion
    utils.log_section("16. Completion")
    if checkpoint_path.exists():
        resp = input("Remove checkpoint files? [y/N] ").strip().lower()
        if resp == 'y':
            try:
                checkpoint_path.unlink()
                manifest_path.unlink()
                logger.info("Checkpoint files removed")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint files: {e}")
    logger.info("🎉 Stage 2 completed successfully.")
    logger.info(f"Final output: {out_fif}")


# ==========================================================
# HYBRID PREFETCH + PUSH GLUE
# ==========================================================
def maybe_hybrid_prefetch(config: dict) -> tuple[dict, bool, bool, str, str]:
    """
    If remote_io.enabled, call HybridTransferManager to pull raw + derivatives.
    Returns: (updated_config, hybrid_used, checkpoint_exists, local_bids_root, remote_bids_root)
    """
    remote_io = config.get("remote_io", {})
    enabled = bool(remote_io.get("enabled", False))

    # Backward-compatible fallbacks if user provided old keys at top level
    hpc_host = remote_io.get("hpc_host", config.get("hpc_host"))
    hpc_user = remote_io.get("hpc_user", config.get("hpc_user"))
    remote_bids_root = remote_io.get("remote_bids_root", config.get("bids_root"))
    local_temp_dir = remote_io.get("local_temp_dir", config.get("temp_dir", "./temp"))

    if not enabled:
        return (config, False, False, "", "")

    if HybridTransferManager is None:
        logger.error("remote_io.enabled=True but transfer_manager not importable")
        sys.exit(1)

    # Instantiate manager and fetch
    logger.info("🌐 HYBRID: Prefetching raw + derivatives to local temp (single Duo auth)")
    tm = HybridTransferManager(hpc_host, hpc_user, local_temp_dir)
    try:
        local_bids_root, checkpoint_exists = tm.fetch_all_bids_data(
            config["subject"], config.get("session"), config.get("task"), config.get("run"),
            remote_bids_root
        )
    except Exception as e:
        logger.exception("Hybrid prefetch failed"); sys.exit(1)

    # Update config to point the pipeline at the LOCAL temp BIDS root
    new_cfg = dict(config)
    new_cfg["bids_root"] = local_bids_root  # <<< CRITICAL: everything else works unchanged
    # Also store for later push
    new_cfg["_hybrid_local_bids_root"] = local_bids_root
    new_cfg["_hybrid_remote_bids_root"] = remote_bids_root
    return (new_cfg, True, checkpoint_exists, local_bids_root, remote_bids_root)


def maybe_hybrid_push(config: dict):
    """
    If we used hybrid prefetch, push derivatives back to HPC.
    """
    if not config.get("_hybrid_local_bids_root") or not config.get("_hybrid_remote_bids_root"):
        return
    if HybridTransferManager is None:
        logger.error("Hybrid push requested but transfer_manager not importable")
        return

    local_bids_root = config["_hybrid_local_bids_root"]
    remote_bids_root = config["_hybrid_remote_bids_root"]
    rio = config.get("remote_io", {})
    hpc_host = rio.get("hpc_host", config.get("hpc_host"))
    hpc_user = rio.get("hpc_user", config.get("hpc_user"))
    local_temp_dir = rio.get("local_temp_dir", config.get("temp_dir", "./temp"))

    tm = HybridTransferManager(hpc_host, hpc_user, local_temp_dir)
    logger.info("☁️  HYBRID: Pushing derivatives (parproc + preproc) back to HPC")
    try:
        exit_code = tm.push_results(local_bids_root, remote_bids_root,
                                    config["subject"], config.get("session"))
        if exit_code == 0:
            logger.info("✅ Hybrid push complete")
        else:
            logger.warning(f"⚠️ Hybrid push finished with exit code {exit_code}")
    except Exception as e:
        logger.exception("Hybrid push failed")


# ==========================================================
# MAIN DISPATCH
# ==========================================================
def run_pipeline(yaml_path: str, force_stage: str = None):
    utils.log_section("1. Load Configuration and Detect Pipeline Stage")
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}"); sys.exit(1)

    try:
        p0 = utils.load_yaml(yaml_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}"); sys.exit(1)

    # Minimal validation
    for key in ["subject", "bids_root"]:
        if key not in p0 and not (key == "bids_root" and p0.get("remote_io", {}).get("enabled")):
            logger.error(f"Missing required configuration key: {key}")
            sys.exit(1)

    # HYBRID PREFETCH (if enabled) — updates p["bids_root"] to local temp
    p, hybrid_used, checkpoint_present, local_bids_root, remote_bids_root = maybe_hybrid_prefetch(p0)

    # Stage detection uses (possibly updated) p["bids_root"]
    stage, checkpoint_path = detect_pipeline_stage(yaml_path, p, force_stage)
    logger.info(f"🎯 Running pipeline in {stage.upper()} mode")

    # Stage 1
    if stage == "stage1":
        status, _ = run_stage1_pipeline(yaml_path, p)
        if status == 'exit':
            logger.info("🎯 Stage 1 completed - exiting as requested"); sys.exit(0)
        elif status == 'continue':
            # rediscover checkpoint under the same (local) bids_root
            stage2, checkpoint_path = detect_pipeline_stage(yaml_path, p, None)
            if stage2 != "stage2" or checkpoint_path is None:
                logger.error("Expected Stage 2 after Stage 1, but no checkpoint found"); sys.exit(1)
            stage = "stage2"

    # Stage 2
    if stage == "stage2":
        run_stage2_pipeline(yaml_path, p, checkpoint_path)
        # If hybrid, push results back to HPC
        if hybrid_used:
            maybe_hybrid_push(p)
        logger.info("🎉 Pipeline completed successfully!")
        return

    logger.error(f"Unknown pipeline stage: {stage}")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MEG/EEG Preprocessing Pipeline with Hybrid Transfer Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local-only (no transfers):
  python pipeline.py config_local.yaml

  # Hybrid: pull raw+derivatives to ./temp/bids, run both stages, push results back:
  python pipeline.py config_hybrid.yaml

  # Force a stage:
  python pipeline.py config.yaml --force-stage stage2
        """
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--force-stage", choices=["stage1", "stage2"], help="Force specific pipeline stage")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--version", action="version", version=f"MEG Pipeline v{PIPELINE_VERSION}")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG); logger.setLevel(logging.DEBUG)

    print("=" * 60)
    print(f"MEG/EEG Preprocessing Pipeline v{PIPELINE_VERSION}")
    print(f"Configuration: {args.config}")
    if args.force_stage: print(f"Forced Stage: {args.force_stage}")
    print("=" * 60, "\n")

    try:
        run_pipeline(args.config, args.force_stage)
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        sys.exit(1)
