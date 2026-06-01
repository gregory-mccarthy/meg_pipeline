#!/usr/bin/env python3
"""
MEG/EEG PREPROCESSING PIPELINE - HYBRID TRANSFER CAPABLE

A two-stage preprocessing pipeline for MEG/EEG data using MNE-Python.
Supports both local processing and hybrid mode with HPC data transfer.

HYBRID MODE:
  - Transfers raw data and derivatives from HPC to local temporary directory
  - Executes preprocessing stages locally using transferred data
  - Transfers processed results back to HPC upon completion

LOCAL MODE:
  - Processes data directly from local BIDS directory
  - No data transfer operations

YAML Configuration Keys:
  remote_io:
    enabled: true
    hpc_host: transfer-milgram.ycrc.yale.edu
    hpc_user: gm33
    remote_bids_root: /gpfs/milgram/scratch/mccarthy/gm33/BIDS/epi
    local_temp_dir: ./temp

PIPELINE STAGES:
  Stage 1: Data loading, Maxwell filtering, bad channel detection, checkpoint creation
  Stage 2: Interactive review, ICA processing, event detection, final filtering

Output Artifacts:
  - Preprocessed MEG/EEG data in BIDS format
  - ICA decomposition files for MEG and EEG
  - Quality control plots and metrics
  - Comprehensive processing logs in YAML and JSON formats
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import json
import numpy as np
import matplotlib  # safe to import; do NOT import pyplot yet
from bids_io_utils import parse_time_window, apply_bids_events_tsv, apply_bids_channels_tsv

# Use distinct aliases (no shadowing)
#import bids_io_utils_v2 as bdu   # BIDS/data utilities

import meg_pipeline_utils as utils

from headpos_utils import (
    prepare_headpos_from_config,
    compute_head_movement_stats,
    compute_head_pos_from_raw,
    plot_head_movement,
    read_head_pos_safe,
    get_bids_headpos_path,
)

from bids_io_utils import parse_time_window

import mne  # OK; avoid mne.viz until after backend is chosen

mne.set_log_level('WARNING')
#mne.set_log_level('ERROR')
#mne.set_log_level('CRITICAL')
#mne.set_log_level('INFO')
#mne.set_log_level('DEBUG')

# --- remove the old local backend logic (_in_slurm/_has_display/matplotlib.use(...)) ---
# We'll call mutils.setup_matplotlib(stage) later, once we know the stage.
try:
    from transfer_manager import HybridTransferManager
except ImportError:
    HybridTransferManager = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")
PIPELINE_VERSION = "1.2"
CHECKPOINT_VERSION = "1.0"

import numpy as np

print("RUNNING SCRIPT:", os.path.abspath(__file__))

def log_meg_stats(raw, tag, logger, tmin=None, tmax=None):
    seg = raw.copy().crop(
        tmin=raw.times[0] if tmin is None else tmin,
        tmax=min(raw.times[-1], (tmin or raw.times[0]) + 10.0) if tmax is None else tmax,
    )
    picks = mne.pick_types(seg.info, meg=True)
    data = seg.get_data(picks=picks)  # shape (n_meg, n_times)

    per_chan_max = np.max(np.abs(data), axis=1)
    logger.info(
        f"[{tag}] MEG | "
        f"median(|max|)={np.median(per_chan_max):.3e}, "
        f"95th={np.percentile(per_chan_max, 95):.3e}, "
        f"max={per_chan_max.max():.3e}"
    )


def detect_pipeline_stage(yaml_path: str, p: dict, force_stage: str = None):
    """
    Determine whether to execute Stage 1 or resume from Stage 2 checkpoint.

    Searches for existing checkpoint files in BIDS derivatives structure:
      {bids_root}/derivatives/{pipeline_name}/sub-*/ses-*/meg/*_desc-parproc_meg.fif

    Returns:
        tuple: (stage_name, checkpoint_path) where stage_name is 'stage1' or 'stage2'
    """
    if force_stage:
        logger.info(f"Pipeline stage forced: {force_stage}")
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
        logger.info(f"Checkpoint detected: {parproc_fif}")
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

    logger.info("No checkpoint found, starting from Stage 1")
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


def run_stage1_pipeline(yaml_path: str, p: dict):
    """
    STAGE 1: DATA LOADING AND MAXWELL FILTERING
    """
    p["_cfg_path"] = yaml_path
    repo_root = Path(__file__).resolve().parent
    logger.info(f"Repository root: {repo_root}")

    try:
        checked_paths = utils.check_critical_files_exist(p, repo_root)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e));
        sys.exit(1)

    for key, path in checked_paths.items():
        logger.info(f"  {key}: {path}")

    subject = p["subject"]
    session = p.get("session")
    task = p.get("task")
    run = p.get("run")
    bids_root = p["bids_root"]

    head_movement_stats = None
    empirical_rank = None
    eeg_setup_results = None
    metadata_repair_results = None

    utils.log_section("2. Load Raw Data")
    bids_path = utils.BIDSPath(subject=subject, session=session, task=task, run=run,
                               datatype="meg", root=bids_root)
    raw = utils.read_raw_bids_robust(bids_path)

    # Extract original recording info BEFORE cropping to preserve true baseline
    raw_fif_path = raw.filenames[0]
    meg_dir = os.path.dirname(raw_fif_path)

    # --- NEW: Apply Annotations from _events.tsv if configured ---
    annot_cfg = p.get("annotations", {})
    if annot_cfg.get("read_events_tsv", False):
        raw = apply_bids_events_tsv(raw, raw_fif_path, logger=logger)

    original_recording_info = {
        "raw_file_path": raw_fif_path,
        "original_sfreq": float(raw.info["sfreq"]),
        "original_duration_sec": float(raw.times[-1]),
        "n_channels_total": len(raw.ch_names),
        "channel_types": {
            "meg": int(np.size(mne.pick_types(raw.info, meg=True))),
            "eeg": int(np.size(mne.pick_types(raw.info, eeg=True))),
            "eog": int(np.size(mne.pick_types(raw.info, eog=True))),
            "ecg": int(np.size(mne.pick_types(raw.info, ecg=True))),
            "stim": int(np.size(mne.pick_types(raw.info, stim=True))),
            "misc": int(np.size(mne.pick_types(raw.info, misc=True))),
        },
        "measurement_date": (
            raw.info.get("meas_date").isoformat()
            if raw.info.get("meas_date") else None
        ),
        "line_frequency": float(raw.info.get("line_freq", 60.0)),
    }

    # Optional sub-segment cropping
    try:
        tw = parse_time_window(p)  # p is your loaded config dict
    except Exception as _e:
        logger.error(f"Invalid time_window in YAML: {_e}");
        sys.exit(1)
    if tw:
        tmin, tmax = tw
        raw.crop(tmin=tmin, tmax=tmax)  # lazy; still not preloaded
        logger.info(
            f"Cropped recording to window: start={tmin if tmin is not None else 0.0:.3f}s, "
            f"end={'end' if tmax is None else f'{tmax:.3f}s'}; duration={raw.times[-1]:.3f}s"
        )
        log_meg_stats(raw, "After crop", logger)


    # Build derivative paths FIRST (for later saving/plots)
    bids_path_deriv = bids_path.copy().update(
        root=Path(bids_root) / "derivatives" / p.get("checkpoint", {}).get("pipeline_name", "preprocessing"),
        suffix="meg",
        description="preproc",
        extension=".fif",
    )
    out_fif = str(bids_path_deriv.fpath)
    os.makedirs(os.path.dirname(out_fif), exist_ok=True)

    plots_dir = os.path.join(os.path.dirname(out_fif), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    utils.log_section("3. Compute or Load Head Position")

    # Head-position config from YAML
    # Prioritizes 'head_movement', falls back to 'head_position_processing'
    if isinstance(p, dict):
        hpp_cfg = p.get("head_movement", p.get("head_position_processing", {}))
    else:
        hpp_cfg = {}

    # Optional: derive a window tag from the global time_window for naming subset .pos
    tw = parse_time_window(p)
    tmin, tmax = (tw if tw else (None, None))
    start_s = 0 if tmin is None else int(tmin)
    end_s = "end" if tmax is None else int(tmax)
    subset_tag = hpp_cfg.get("subset_naming", "desc-crop")
    run_piece = f"_run-{run}" if run else ""

    ck = p.get("checkpoint", {}) if isinstance(p, dict) else {}
    derivatives_root = ck.get("derivatives_root", "derivatives")
    pipeline_name = ck.get("pipeline_name", "preprocessing")

    def _save_subset_pos(hp_abs):
        """Write a window-specific .pos file for this run and return its path."""
        deriv_dir = (
            Path(bids_root)
            / derivatives_root
            / pipeline_name
            / f"sub-{subject}"
            / f"ses-{session}"
            / "meg"
        )
        deriv_dir.mkdir(parents=True, exist_ok=True)
        subset_fname = (
            f"sub-{subject}_ses-{session}_task-{task}{run_piece}_"
            f"{subset_tag}{start_s}-{end_s}_headpos.pos"
        )
        subset_path = deriv_dir / subset_fname
        mne.chpi.write_head_pos(str(subset_path), hp_abs)
        logger.info(f"Wrote head-position subset to: {subset_path}")
        return str(subset_path)

    def _load_reference_headpos(ref_run):
        """Load head_pos array for a reference run (used for source='run' or destination='reference')."""
        ref_run_str = f"{int(ref_run):02d}" if isinstance(ref_run, (int, np.integer, str)) else str(ref_run)
        ref_pos_path = get_bids_headpos_path(
            subject=subject,
            session=session,
            task=task,
            run=ref_run_str,
            meg_dir=str(
                # use the same MEG directory as this run, which is already resolved
                Path(bids_root) / f"sub-{subject}" / f"ses-{session}" / "meg"
            ),
        )
        hp = read_head_pos_safe(str(ref_pos_path), logger=logger)
        return hp

    # 4. EEG Channel Setup
    utils.log_section("4. EEG Channel Setup")
    eeg_setup_results = utils.prepare_eeg_channels(
        raw,
        checked_paths["montage"],
        logger,
    )

    # 5. Metadata Repair
    utils.log_section("5. Metadata Repair")
    metadata_repair_results = utils.apply_metadata_repairs(
        raw,
        p.get("metadata_fixes", {}),
    )

    # 5b. Bad Channel Processing (CRITICAL BEFORE MAXWELL)
    # Target 1: Dynamic sidecars from the annotation step
    tsv_bad_channels = apply_bids_channels_tsv(raw, raw_fif_path, logger=logger)

    # Target 2: Static damaged sensors designated globally in the YAML file
    yaml_bad_channels = p.get("manual_bad_channels", [])
    if yaml_bad_channels:
        logger.info(f"[Channels] Found static bad channels in YAML configuration: {yaml_bad_channels}")

    # Set-union the distinct sources to establish the true bad channel list
    existing_bads = set(raw.info.get('bads', []))
    unified_bads = existing_bads.union(tsv_bad_channels).union(yaml_bad_channels)

    raw.info['bads'] = list(unified_bads)

    logger.info(f"[Channels] Combined Maxwell exclusion list: {raw.info['bads']}")
    logger.info(f"   - (Inherited/Raw: {len(existing_bads)} | TSV: {len(tsv_bad_channels)} | YAML: {len(yaml_bad_channels)})")

    # 6. PSD & RMS Diagnostics (Pre-filtering)
    utils.log_section("6. PSD & RMS Diagnostics (Pre-filtering)")
    utils.qc_meg_raw(raw, plots_dir)
    utils.plot_psd_and_peaks(raw, "Raw PSD Before Maxwell", plots_dir)

    # Pre-Maxwell MEG quality metrics (for later comparison)
    metrics_pre_maxwell = utils.compute_meg_quality_metrics(raw, "pre_maxwell")

    # Determine requested source and movement policy from YAML
    source_cfg = (
        hpp_cfg.get("source", "file")
        if isinstance(hpp_cfg, dict)
        else "file"
    )
    movement_requested = (
        bool(hpp_cfg.get("movement_compensation", True))
        if isinstance(hpp_cfg, dict)
        else True
    )

    # Default .pos path: if source is "file" or "run" and no file_path is
    # given, derive the canonical BIDS headpos sidecar in this run's MEG dir.
    default_pos_path = None
    if isinstance(hpp_cfg, dict) and source_cfg in ("file", "run"):
        if not hpp_cfg.get("file_path"):
            meg_dir = str(Path(bids_root) / f"sub-{subject}" / f"ses-{session}" / "meg")
            try:
                default_pos_path = get_bids_headpos_path(
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    meg_dir=meg_dir,
                )
                logger.info(f"[headpos] Using default BIDS sidecar path: {default_pos_path}")
            except Exception as e:
                logger.warning(
                    f"[headpos] Failed to derive default BIDS headpos path: {e}. "
                    "Will rely on explicit file_path or compute-from-raw."
                )

    headpos_result = prepare_headpos_from_config(
        raw=raw,
        cfg=hpp_cfg if isinstance(hpp_cfg, dict) else {},
        default_pos_path=default_pos_path,
        compute_headpos_fn=compute_head_pos_from_raw,
        load_reference_headpos_fn=_load_reference_headpos,
        save_subset_fn=_save_subset_pos if isinstance(hpp_cfg, dict) and hpp_cfg.get("write_subset",
                                                                                     False) else None,
        logger=logger,
    )

    head_pos_array = headpos_result.get("head_pos_abs", None)
    destination = headpos_result.get("destination", None)
    movement_enabled = bool(headpos_result.get("movement_enabled", False))
    source_used = headpos_result.get("source_used", source_cfg)

    # Fallback compute
    if (
            movement_requested
            and (head_pos_array is None or len(head_pos_array) == 0)
            and source_used in ("file", "run")
    ):
        logger.warning(
            "[headpos] source='%s' did not yield head_pos data; "
            "falling back to source='compute' (compute_head_pos_from_raw).",
            source_used,
        )
        fallback_cfg = dict(hpp_cfg) if isinstance(hpp_cfg, dict) else {}
        fallback_cfg["source"] = "compute"
        headpos_result = prepare_headpos_from_config(
            raw=raw,
            cfg=fallback_cfg,
            default_pos_path=None,
            compute_headpos_fn=compute_head_pos_from_raw,
            load_reference_headpos_fn=_load_reference_headpos,
            save_subset_fn=_save_subset_pos if fallback_cfg.get("write_subset", False) else None,
            logger=logger,
        )
        head_pos_array = headpos_result.get("head_pos_abs", None)
        destination = headpos_result.get("destination", None)
        movement_enabled = bool(headpos_result.get("movement_enabled", False))
        source_used = headpos_result.get("source_used", "compute")

    # Decide what to pass into maxwell_filter as head_pos
    if movement_enabled and head_pos_array is not None and len(head_pos_array) > 0:
        head_pos_for_maxwell = head_pos_array
        logger.info(
            f"Head-pos for Maxwell (source='{source_used}'): "
            f"n={head_pos_array.shape[0]}, "
            f"abs_range=[{head_pos_array[0, 0]:.3f}, {head_pos_array[-1, 0]:.3f}] s "
            "(movement compensation enabled)."
        )
    else:
        head_pos_for_maxwell = None
        if not movement_requested:
            logger.info(
                "movement_compensation=False in YAML – Maxwell will use static origin/destination only."
            )
        else:
            logger.info(
                "No usable head_pos for Maxwell (source='%s') – running tSSS without movement compensation.",
                source_used,
            )

    # Compute and plot head-movement statistics if head_pos is available
    if head_pos_array is not None and len(head_pos_array) > 0:
        head_movement_stats = compute_head_movement_stats(head_pos_array)
        if head_movement_stats is not None:
            logger.info(
                "Head movement: max displacement "
                f"{head_movement_stats['translation_stats_mm']['max_displacement']:.1f} mm, "
                f"max rotation {head_movement_stats['rotation_stats_deg']['max_rotation']:.1f}°"
            )
        plot_head_movement(head_pos_array, plots_dir, logger=logger)
    else:
        head_movement_stats = None
        logger.info("No head position data available; skipping head-movement metrics and plot.")

    # ---------------- 7. Maxwell Filter (tSSS) ----------------
    utils.log_section("7. Maxwell Filter (tSSS)")

    st_duration = (
        float(hpp_cfg.get("st_duration", 10.0))
        if isinstance(hpp_cfg, dict)
        else 10.0
    )
    use_headshape_origin = (
        bool(hpp_cfg.get("use_headshape_origin", True))
        if isinstance(hpp_cfg, dict)
        else True
    )
    use_destination = (
        bool(hpp_cfg.get("use_destination", True))
        if isinstance(hpp_cfg, dict)
        else True
    )

    logger.info(
        "Applying Maxwell filter (tSSS) with "
        f"{'movement compensation' if head_pos_for_maxwell is not None else 'NO movement compensation'}, "
        f"st_duration={st_duration:.1f}s, "
        f"{'headshape origin' if use_headshape_origin else 'auto origin'}, "
        f"{'destination enabled' if use_destination else 'no destination'}."
    )

    hp = None
    if head_pos_for_maxwell is not None and len(head_pos_for_maxwell) > 0:
        hp = np.asarray(head_pos_for_maxwell, float)

    dest_arr = None
    if use_destination:
        if destination is None:
            logger.info("[SSS] No destination computed – proceeding without destination.")
        else:
            Transform = getattr(getattr(mne, "transforms", None), "Transform", None)
            if Transform is not None and isinstance(destination, Transform):
                dest_arr = destination
                logger.info("[SSS] Using destination as mne Transform.")
            elif isinstance(destination, (str, Path)):
                dest_arr = str(destination)
                logger.info(f"[SSS] Using destination transform file: {dest_arr}")
            else:
                dest_arr = np.asarray(destination, float).reshape(3,)
                logger.info(
                    f"[SSS] Using HEAD-frame destination from head_pos: "
                    f"{dest_arr.round(4).tolist()} m"
                )
    else:
        logger.info("[SSS] use_destination=False – proceeding without destination.")

    # Origin: either a headshape-based origin (HEAD frame, meters)
    # or MNE's built-in auto origin if disabled or if fit fails.
    origin_head = None
    if use_headshape_origin:
        try:
            origin_head = utils._fit_origin_from_headshape(raw.info)
            logger.info(
                f"[SSS] Using explicit headshape origin (HEAD): "
                f"{origin_head.round(4).tolist()} m"
            )
        except Exception as e:
            origin_head = None
            logger.warning(
                f"[SSS] Headshape-origin fit failed ({e}); falling back to MNE auto origin."
            )
    else:
        logger.info("[SSS] use_headshape_origin=False – using MNE auto origin.")

    maxwell_kwargs = dict(
        calibration=str(checked_paths["calibration_file"]) if checked_paths.get("calibration_file") else None,
        cross_talk=str(checked_paths["cross_talk_file"]) if checked_paths.get("cross_talk_file") else None,
        head_pos=hp,
        st_duration=st_duration,
        verbose=False,
    )

    if origin_head is not None:
        maxwell_kwargs["origin"] = origin_head
    if dest_arr is not None:
        maxwell_kwargs["destination"] = dest_arr

    try:
        try:
            from mne.transforms import Transform
        except Exception:
            Transform = ()

        logger.info("Maxwell args: calibration=%s", maxwell_kwargs.get("calibration"))
        logger.info("Maxwell args: cross_talk=%s", maxwell_kwargs.get("cross_talk"))
        logger.info("Maxwell args: st_duration=%s", maxwell_kwargs.get("st_duration"))

        if maxwell_kwargs.get("head_pos") is None:
            logger.info("Maxwell args: head_pos=None (no movement compensation)")
        else:
            hp_arr = maxwell_kwargs.get("head_pos")
            try:
                hp_shape = getattr(hp_arr, "shape", None)
                t0 = float(hp_arr[0, 0])
                t1 = float(hp_arr[-1, 0])
                logger.info("Maxwell args: head_pos shape=%s, time=[%.3f, %.3f] sec", hp_shape, t0, t1)
            except Exception:
                logger.info("Maxwell args: head_pos provided (unable to summarize)")

        if "origin" in maxwell_kwargs and maxwell_kwargs["origin"] is not None:
            origin_obj = maxwell_kwargs["origin"]
            try:
                origin_vec = np.asarray(origin_obj, float).reshape(3,)
            except Exception as e:
                raise RuntimeError(f"Invalid Maxwell origin (expected 3-vector in meters): {origin_obj!r}") from e
            maxwell_kwargs["origin"] = origin_vec
            logger.info("Maxwell args: origin=%s (head coords, m)", origin_vec.tolist())
        else:
            logger.info("Maxwell args: origin=None (MNE default)")

        if "destination" in maxwell_kwargs and maxwell_kwargs["destination"] is not None:
            dest_obj = maxwell_kwargs["destination"]

            if isinstance(dest_obj, (str, Path)):
                dest_path = str(dest_obj)
                maxwell_kwargs["destination"] = dest_path
                logger.info("Maxwell args: destination=%s (path-like)", dest_path)

            elif Transform and isinstance(dest_obj, Transform):
                logger.info("Maxwell args: destination=<mne.Transform> (passed through)")

            else:
                try:
                    dest_vec = np.asarray(dest_obj, float).reshape(3,)
                except Exception as e:
                    raise RuntimeError(
                        "Invalid Maxwell destination. Expected None, 3-vector (meters, head coords), "
                        "path-like (FIF transform), or mne.transforms.Transform. "
                        f"Got: {type(dest_obj)} -> {dest_obj!r}"
                    ) from e
                maxwell_kwargs["destination"] = dest_vec
                logger.info("Maxwell args: destination=%s (head coords, m)", dest_vec.tolist())
        else:
            logger.info("Maxwell args: destination=None (no destination transform)")

    except Exception:
        raise

    # --- Enforce default MNE skips ONLY for Maxwell Filtering ---
    # We explicitly DO NOT skip 'BAD_break' here. Skipping internal segments
    # creates massive step-function discontinuities that cause severe ringing
    # artifacts during downstream temporal filtering.

    skip_annots = ['edge', 'bad_acq_skip']  # MNE defaults
    maxwell_kwargs["skip_by_annotation"] = skip_annots
    logger.info("[SSS] Using default skip_by_annotation ('edge', 'bad_acq_skip') to maintain temporal continuity.")

    raw = mne.preprocessing.maxwell_filter(raw, **maxwell_kwargs)

    utils.plot_psd_and_peaks(raw, "After Maxwell", plots_dir)

    metrics_post_maxwell = utils.compute_meg_quality_metrics(raw, "post_maxwell")
    maxwell_quality_metrics = utils.log_maxwell_quality_results(
        metrics_pre_maxwell,
        metrics_post_maxwell,
        logger,
    )

    utils.log_section("7b. Estimate MEG Rank After Maxwell")

    try:
        rank_dict = mne.compute_rank(raw)
        meg_rank = None
        if "meg" in rank_dict:
            meg_rank = rank_dict["meg"]
        else:
            for k, v in rank_dict.items():
                if "meg" in k.lower():
                    meg_rank = v
                    break

        logger.info(f"Estimated MEG rank after Maxwell: {meg_rank}")
        rank_info = {
            "meg_rank": meg_rank,
            "method": "empirical",
            "computed_after": "maxwell_filtering",
        }

    except Exception as e:
        logger.warning(f"Rank estimation failed: {e}")
        rank_info = {"error": str(e), "meg_rank": None}

    raw.load_data()

    utils.log_section("8. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    picks = utils.mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    raw.notch_filter(notch_freqs, picks=picks, method='fir', filter_length='auto')
    utils.plot_psd_and_peaks(raw, "After Maxwell and Notch", plots_dir)

    notch_filter_params = {
        "frequencies_hz": notch_freqs,
        "method": "fir",
        "n_channels_filtered": len(picks)
    }

    utils.log_section("9. Bad Channel Detection with AutoReject (checkpoint)")
    p["_checkpoint_version"] = CHECKPOINT_VERSION

    try:
        checkpoint_path, stage1_ar_metadata = utils.run_autoreject_stage1(raw, p, bids_path, logger)
        logger.info("Stage 1 AutoReject and checkpoint creation completed")

    except Exception as e:
        logger.error(f"Stage 1 AutoReject/checkpoint creation failed: {e}")
        resp = input("AutoReject/checkpoint failed. Continue without checkpoint? (y/n): ").strip().lower()
        if resp != 'y':
            return ('exit', raw)
        logger.warning("Continuing without checkpoint - Stage 2 will not be possible")

        stage1_ar_metadata = {
            "checkpoint_path": None,
            "bads_detected": list(raw.info.get('bads', [])),
            "n_bad_channels": len(raw.info.get('bads', [])),
            "n_annotations": int(len(raw.annotations) if raw.annotations is not None else 0),
            "autoreject_enabled": False,
            "checkpoint_failed": True,
            "error": str(e)
        }
        checkpoint_path = None

    stage1_meta = {
        "artifacts": {
            "parproc_raw_fif": str(checkpoint_path) if checkpoint_path else None,
        },
        "results": stage1_ar_metadata
    }
    utils.log_section("10. Save Stage 1 Manifest")

    quality_metrics = {}
    if 'metrics_pre_maxwell' in locals():
        quality_metrics['pre_maxwell'] = metrics_pre_maxwell
    if 'metrics_post_maxwell' in locals():
        quality_metrics['post_maxwell'] = metrics_post_maxwell
    if 'maxwell_quality_metrics' in locals():
        quality_metrics['maxwell_improvements'] = maxwell_quality_metrics

    ar_results = {
        "bads_detected": list(raw.info.get('bads', [])),
        "n_bad_channels": len(raw.info.get('bads', [])),
        "n_annotations": int(len(raw.annotations) if raw.annotations is not None else 0),
    }
    if stage1_meta and "results" in stage1_meta and "autoreject_details" in stage1_meta["results"]:
        ar_results["autoreject_details"] = stage1_meta["results"]["autoreject_details"]

    checkpoint_file = None
    if stage1_meta and "artifacts" in stage1_meta:
        checkpoint_file = stage1_meta["artifacts"].get("parproc_raw_fif")

    if not checkpoint_file:
        bids_root = Path(p.get("bids_root", bids_path.root))
        derivatives_root = p.get("checkpoint", {}).get("derivatives_root", "derivatives")
        pipeline_name = p.get("checkpoint", {}).get("pipeline_name", "preprocessing")
        deriv_root = bids_root / derivatives_root / pipeline_name

        subj_dir = f"sub-{bids_path.subject}"
        if bids_path.session:
            ses_dir = f"ses-{bids_path.session}"
            parproc_dir = deriv_root / subj_dir / ses_dir / "meg"
        else:
            parproc_dir = deriv_root / subj_dir / "meg"
        parproc_dir.mkdir(parents=True, exist_ok=True)

        base = f"sub-{bids_path.subject}"
        if bids_path.session:
            base += f"_ses-{bids_path.session}"
        if bids_path.task:
            base += f"_task-{bids_path.task}"
        if bids_path.run:
            base += f"_run-{bids_path.run}"

        checkpoint_file = str(parproc_dir / f"{base}_desc-parproc_meg.fif")

    manifest_path = utils.write_stage1_manifest(
        raw=raw,
        cfg=p,
        bids_path=bids_path,
        checkpoint_file=checkpoint_file,
        autoreject_results=ar_results,
        quality_metrics=quality_metrics,
        head_movement_stats=head_movement_stats,
        rank_info=rank_info,
        eeg_setup_results=eeg_setup_results,
        metadata_repair_results=metadata_repair_results,
        original_recording_info=original_recording_info,
        notch_filter_params=notch_filter_params,
        processing_paths={"plots_directory": plots_dir},
        logger=logger,
    )

    exit_after = bool(p.get("checkpoint", {}).get("exit_after_checkpoint", False))
    status = 'exit' if exit_after else 'continue'

    if status == 'exit':
        logger.info("Stage 1 completed, checkpoint and manifest saved, exit as requested")
        return ('exit', raw)
    elif status == 'continue':
        logger.info("Stage 1 completed, checkpoint and manifest saved")
        return ('continue', raw)

    logger.error("Unexpected Stage 1 flow")
    return ('exit', raw)


def run_stage2_pipeline(yaml_path: str, p: dict, checkpoint_path: Path):
    """
    STAGE 2: INTERACTIVE PROCESSING AND FINALIZATION
    """
    p["_cfg_path"] = yaml_path
    p["_checkpoint_version"] = CHECKPOINT_VERSION

    utils.log_section("1. Validate and Load Stage 1 Checkpoint")
    manifest_path = checkpoint_path.with_name(checkpoint_path.stem + "_manifest.yaml")

    if not validate_checkpoint_integrity(checkpoint_path, manifest_path):
        logger.error("Checkpoint validation failed")
        resp = input("Restart from Stage 1? (y/n): ").strip().lower()
        if resp == 'y':
            run_stage1_pipeline(yaml_path, p)
            return
        sys.exit(1)

    try:
        stage1_manifest = utils.load_yaml(str(manifest_path))
        logger.info(f"Loaded Stage 1 manifest: {manifest_path}")

        if "system_info" in stage1_manifest and "stage1" in stage1_manifest["system_info"]:
            stage1_system = stage1_manifest["system_info"]["stage1"]
            logger.info(f"Stage 1 completed on: {stage1_system.get('hostname', 'unknown')} "
                        f"at {stage1_system.get('completed_utc', 'unknown time')}")

    except Exception as e:
        logger.warning(f"Could not read manifest at {manifest_path}: {e}")
        stage1_manifest = {}

    prior_yaml_sha = (stage1_manifest.get("inputs", {}) or {}).get("yaml_sha256")
    current_yaml_sha = None
    try:
        if p.get("_cfg_path"):
            import hashlib
            with open(p["_cfg_path"], 'r') as f:
                current_yaml_sha = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute SHA256 of current YAML: {e}")

    yaml_audit = {
        "prior_yaml_sha256": prior_yaml_sha,
        "current_yaml_sha256": current_yaml_sha,
        "yaml_changed": bool(prior_yaml_sha and current_yaml_sha and prior_yaml_sha != current_yaml_sha)
    }
    if yaml_audit["yaml_changed"]:
        logger.warning("YAML configuration has changed since Stage 1 checkpoint was created")

    utils.log_section("2. Setup Output Paths")
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

    utils.log_section("3. Interactive Bad Channel Review")

    try:
        raw, stage2_ar_metadata = utils.run_interactive_review_stage2(checkpoint_path, p, logger)
        logger.info("Stage 2 interactive review completed")

        if stage2_ar_metadata.get("interactive_review_enabled", False):
            changes = stage2_ar_metadata.get("changes_made", {})
            added = changes.get("bad_channels_added", [])
            removed = changes.get("bad_channels_removed", [])
            if added:
                logger.info(f"   Added bad channels: {added}")
            if removed:
                logger.info(f"   Rescued channels: {removed}")
            if not added and not removed:
                logger.info("   No bad channel changes made")
        else:
            logger.info("   Interactive review was disabled")

    except Exception as e:
        logger.error(f"Interactive review failed: {e}")
        resp = input("Continue without interactive review? (y/n): ").strip().lower()
        if resp != 'y':
            sys.exit(1)

        logger.warning("Loading checkpoint without interactive review")
        try:
            raw = mne.io.read_raw_fif(str(checkpoint_path), preload=True, verbose="error")
            stage2_ar_metadata = {
                "interactive_review_enabled": False,
                "review_failed": True,
                "error": str(e),
                "bads_after_review": list(raw.info.get('bads', [])),
                "n_bad_channels_final": len(raw.info.get('bads', [])),
                "n_annotations_final": int(len(raw.annotations) if raw.annotations is not None else 0)
            }
        except Exception as load_error:
            logger.error(f"Failed to load checkpoint: {load_error}")
            sys.exit(1)

    logger.info(f"Current state: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
    logger.info(f"Bad channels: {len(raw.info.get('bads', []))} - {raw.info.get('bads', [])}")
    logger.info(f"Annotations: {len(raw.annotations) if raw.annotations is not None else 0}")

    utils.log_section("4. MEG Rank Estimation")
    stage2_rank_info = None
    try:
        empirical_rank = mne.compute_rank(raw)
        stage2_rank_info = {
            "meg_rank": int(empirical_rank.get('meg', 0)) if empirical_rank.get('meg') else None,
            "eeg_rank": int(empirical_rank.get('eeg', 0)) if empirical_rank.get('eeg') else None,
            "method": "empirical",
            "computed_after": "stage2_checkpoint_load"
        }
        logger.info(f"Stage 2 rank estimate: MEG={stage2_rank_info['meg_rank']}, "
                    f"EEG={stage2_rank_info['eeg_rank']}")
    except Exception as e:
        logger.warning(f"Stage 2 rank estimation failed: {e}")
        stage2_rank_info = {"error": str(e)}

    utils.log_section("5. ICA: EEG")
    ica_eeg_cfg = p["ica_preprocessing"]["eeg"]
    eeg_exclude = []
    eeg_ica_results = {}

    # --- NEW: Extract flag from YAML ---
    reject_breaks = p.get("annotations", {}).get("reject_bad_breaks", True)

    try:
        raw, eeg_exclude = utils.run_ica(raw, ica_eeg_cfg, bids_path_ica_eeg.fpath, modality="eeg",
                                         reject_by_annotation=reject_breaks)
        eeg_ica_results = {
            "success": True,
            "n_components_excluded": len(eeg_exclude),
            "excluded_components": eeg_exclude,
            "config": ica_eeg_cfg
        }
    except Exception as e:
        logger.error(f"EEG ICA failed: {e}")
        # ... (keep existing except block logic)

    utils.log_section("6. ICA: MEG")
    ica_meg_cfg = p["ica_preprocessing"]["meg"]
    meg_exclude = []
    meg_ica_results = {}
    try:
        raw, meg_exclude = utils.run_ica(raw, ica_meg_cfg, bids_path_ica_meg.fpath, modality="meg",
                                         reject_by_annotation=reject_breaks)
        meg_ica_results = {
            "success": True,
            "n_components_excluded": len(meg_exclude),
            "excluded_components": meg_exclude,
            "config": ica_meg_cfg
        }
    except Exception as e:
        logger.error(f"MEG ICA failed: {e}")
        meg_ica_results = {"success": False, "error": str(e)}
        if input("Continue without MEG ICA? (y/n): ").strip().lower() != 'y':
            sys.exit(1)

    utils.plot_psd_and_peaks(raw, "After ICA", plots_dir)

    utils.log_section("7. Event Detection")
    event_detection_results = {}

    try:
        STIM_CH = "STI101"
        MASK = 0xFFFF
        MAX_SETTLE_MS = 6.0
        REFRACTORY_MS = 20.0

        events = utils.events_from_sti101_early_onset_final_code(
            raw,
            stim_channel=STIM_CH,
            mask=MASK,
            max_settle_ms=MAX_SETTLE_MS,
            refractory_ms=REFRACTORY_MS,
        )

        if events.size:
            utils.annotate_events_from_sti101(
                raw,
                stim_channel=STIM_CH,
                mask=MASK,
                max_settle_ms=MAX_SETTLE_MS,
                refractory_ms=REFRACTORY_MS,
                prefix="TRIG/",
                replace=True,
            )

            codes, counts = np.unique(events[:, 2], return_counts=True)
            event_counts = {int(c): int(n) for c, n in zip(codes, counts)}

            event_detection_results = {
                "success": True,
                "n_events_total": int(events.shape[0]),
                "event_counts": event_counts,
                "unique_event_codes": [int(c) for c in codes],
                "stim_channel": STIM_CH,
                "mask_hex": hex(MASK),
                "max_settle_ms": float(MAX_SETTLE_MS),
                "refractory_ms": float(REFRACTORY_MS),
            }
            logger.info(
                f"[Events] total={events.shape[0]} "
                f"codes={list(map(int, codes))} "
                f"mask={hex(MASK)} settle={MAX_SETTLE_MS}ms refr={REFRACTORY_MS}ms"
            )
        else:
            logger.warning("No events detected on STI101 with current mask/parameters.")
            event_detection_results = {
                "success": True,
                "n_events_total": 0,
                "event_counts": {},
                "unique_event_codes": [],
                "stim_channel": STIM_CH,
                "mask_hex": hex(MASK),
                "max_settle_ms": float(MAX_SETTLE_MS),
                "refractory_ms": float(REFRACTORY_MS),
            }

    except Exception as e:
        logger.error(f"Event detection failed: {e}")
        event_detection_results = {"success": False, "error": str(e)}

    utils.log_section("8. Final Filter and Cleanup")
    final_filter_results = {}
    try:
        pre_filter_info = {
            "n_channels": len(raw.ch_names),
            "sfreq": float(raw.info['sfreq']),
            "duration_sec": float(raw.times[-1])
        }

        raw = utils.apply_final_filter_and_cleanup(raw, p)

        final_filter_results = {
            "success": True,
            "config": p.get("final_filter", {}),
            "pre_filter": pre_filter_info,
            "post_filter": {
                "n_channels": len(raw.ch_names),
                "sfreq": float(raw.info['sfreq']),
                "duration_sec": float(raw.times[-1])
            }
        }
        logger.info("Final filtering completed")
    except Exception as e:
        logger.error(f"Final filtering failed: {e}")
        final_filter_results = {"success": False, "error": str(e)}
        if input("Save without final filtering? (y/n): ").strip().lower() != 'y':
            sys.exit(1)

    utils.log_section("9. Save Final Data")
    try:
        written_files = utils.write_bids_robust(raw, bids_path_deriv, overwrite=True, verbose=True)
        main_output_files = utils.get_all_bids_split_files(out_fif)
        logger.info(f"Saved: {out_fif}")
    except Exception as e:
        logger.error(f"Failed to save final data: {e}")
        emergency = Path.cwd() / f"emergency_save_{p['subject']}.fif"
        try:
            raw.save(str(emergency), overwrite=True)
            logger.info(f"Emergency saved: {emergency}")
            sys.exit(1) # FIX: Added missing exit to prevent stranding
        except:
            logger.error("Emergency save failed")
            sys.exit(1)

    ica_output_files = []
    for ica_fif in [bids_path_ica_eeg.fpath, bids_path_ica_meg.fpath]:
        if Path(ica_fif).exists():
            ica_output_files += utils.get_all_bids_split_files(ica_fif)
    ica_output_files = list(dict.fromkeys(ica_output_files))

    utils.log_section("10. Write Comprehensive Final Logs")

    stage2_system_info = utils.get_runtime_info()
    stage2_system_info["stage"] = "stage2"
    stage2_system_info["completed_utc"] = datetime.now(timezone.utc).isoformat()

    yaml_audit = {
        "prior_yaml_sha256": prior_yaml_sha,
        "current_yaml_sha256": current_yaml_sha,
        "yaml_changed": bool(prior_yaml_sha and current_yaml_sha and prior_yaml_sha != current_yaml_sha)
    }

    stage1_system_info = stage1_manifest.get("system_info", {}).get("stage1", {})
    stage1_processing_results = stage1_manifest.get("processing_results", {})
    stage1_quality_metrics = stage1_manifest.get("quality_metrics", {})
    stage1_final_state = stage1_manifest.get("final_state", {})
    stage1_parameters = stage1_manifest.get("parameters", {})
    stage1_inputs = stage1_manifest.get("inputs", {})

    stage2_processing_results = {
        "interactive_review": stage2_ar_metadata,
        "rank_estimation": stage2_rank_info or {},
        "ica_eeg": eeg_ica_results,
        "ica_meg": meg_ica_results,
        "event_detection": event_detection_results,
        "final_filtering": final_filter_results
    }

    comprehensive_log = {
        "log_version": "3.0",
        "pipeline_version": PIPELINE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "pipeline_stage": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),

        "system_info": {
            "stage1": stage1_system_info,
            "stage2": stage2_system_info
        },

        "inputs": stage1_inputs,

        "parameters": {
            "stage1": stage1_parameters,
            "stage2": {
                "ica_preprocessing": p.get("ica_preprocessing", {}),
                "interactive_bad_channels": p.get("interactive_bad_channels", True),
                "final_filter": p.get("final_filter", {}),
                "checkpoint": p.get("checkpoint", {})
            }
        },

        "processing_results": {
            "stage1": stage1_processing_results,
            "stage2": stage2_processing_results
        },

        "quality_metrics": stage1_quality_metrics,

        "final_state": {
            "recording_duration_sec": float(raw.times[-1]),
            "sampling_frequency": float(raw.info['sfreq']),
            "n_channels_total": len(raw.ch_names),
            "n_bad_channels": len(raw.info.get('bads', [])),
            "bad_channels": list(raw.info.get('bads', [])),
            "n_annotations": int(len(raw.annotations) if raw.annotations is not None else 0),
            "channel_counts": {
                "meg_mag": len(mne.pick_types(raw.info, meg='mag', exclude='bads')),
                "meg_grad": len(mne.pick_types(raw.info, meg='grad', exclude='bads')),
                "eeg": len(mne.pick_types(raw.info, eeg=True, exclude='bads')),
                "eog": len(mne.pick_types(raw.info, eog=True, exclude='bads')),
                "ecg": len(mne.pick_types(raw.info, ecg=True, exclude='bads')),
                "stim": len(mne.pick_types(raw.info, stim=True, exclude='bads')),
            }
        },

        "outputs": {
            "main_output_files": main_output_files,
            "ica_output_files": ica_output_files,
            "primary_output": str(out_fif),
            "plots_directory": plots_dir,
            "checkpoint_artifacts": stage1_manifest.get("artifacts", {})
        },

        "summary": {
            "subject": p["subject"],
            "session": p.get("session"),
            "task": p.get("task"),
            "run": p.get("run"),
            "recording_duration_sec": float(raw.times[-1]),
            "total_bad_channels": len(raw.info.get('bads', [])),
            "maxwell_quality_grade": stage1_quality_metrics.get("maxwell_improvements", {}).get("overall_quality_score",
                                                                                                {}).get("grade"),
            "maxwell_quality_score": stage1_quality_metrics.get("maxwell_improvements", {}).get("overall_quality_score",
                                                                                                {}).get("value"),
            "max_head_movement_mm": (
                stage1_processing_results.get("head_movement", {}).get("translation_stats_mm", {}).get(
                    "max_displacement")),
            "meg_rank": stage1_processing_results.get("rank_estimation", {}).get("meg_rank"),
            "n_events_detected": event_detection_results.get("n_events_total", 0),
            "ica_components_removed": {
                "eeg": len(eeg_exclude),
                "meg": len(meg_exclude)
            }
        },

        "provenance": {
            "yaml_audit": yaml_audit,
            "checkpoint_used": str(checkpoint_path),
            "manifest_used": str(manifest_path)
        }
    }

    out_log_yaml = Path(out_fif).with_name(Path(out_fif).stem + "_log.yaml")
    out_log_json = Path(out_fif).with_name(Path(out_fif).stem + "_log.json")

    try:
        utils.save_yaml(str(out_log_yaml), utils.make_serializable(comprehensive_log))
        with open(out_log_json, "w") as f:
            json.dump(utils.make_serializable(comprehensive_log), f, indent=2)
        logger.info(f"Final logs written:")
        logger.info(f"   YAML: {out_log_yaml}")
        logger.info(f"   JSON: {out_log_json}")

        summary = comprehensive_log["summary"]
        logger.info(f"Processing Summary:")
        logger.info(f"   Quality Grade: {summary.get('maxwell_quality_grade', 'N/A')}")
        logger.info(f"   Bad Channels: {summary.get('total_bad_channels', 0)}")
        logger.info(f"   Head Movement: {summary.get('max_head_movement_mm', 'N/A')} mm")
        logger.info(f"   Events: {summary.get('n_events_detected', 0)}")

    except Exception as e:
        logger.error(f"Failed to write comprehensive logs: {e}")

    utils.log_section("11. Cleanup and Completion")
    if checkpoint_path.exists():
        resp = input("Remove checkpoint files? [y/N] ").strip().lower()
        if resp == 'y':
            try:
                checkpoint_path.unlink()
                manifest_path.unlink()
                logger.info("Checkpoint files removed")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint files: {e}")

    logger.info("Stage 2 completed successfully!")
    logger.info(f"Final output: {out_fif}")
    logger.info(f"Complete logs: {out_log_yaml}")

def maybe_hybrid_prefetch(cfg: dict) -> tuple[dict, bool, bool, str, str]:
    """
    Execute hybrid transfer prefetch operation if enabled in configuration.
    """
    remote_io = cfg.get("remote_io", {})
    enabled = bool(remote_io.get("enabled", False))

    hpc_host = remote_io.get("hpc_host", cfg.get("hpc_host"))
    hpc_user = remote_io.get("hpc_user", cfg.get("hpc_user"))
    remote_bids_root = remote_io.get("remote_bids_root", cfg.get("bids_root"))
    local_temp_dir = remote_io.get("local_temp_dir", cfg.get("temp_dir", "./temp"))

    if not enabled:
        return (cfg, False, False, "", "")

    if HybridTransferManager is None:
        logger.error("remote_io.enabled=True but transfer_manager not importable")
        sys.exit(1)

    logger.info("HYBRID: Prefetching raw + derivatives to local temp (single Duo auth)")
    tm = HybridTransferManager(
        hpc_host, hpc_user, local_temp_dir,
        use_multiplex=True,
        verbose=False,
        dry_run=False,
    )
    try:
        local_bids_root, checkpoint_exists = tm.fetch_all_bids_data(
            cfg["subject"], cfg.get("session"), cfg.get("task"), cfg.get("run"),
            remote_bids_root
        )
    except Exception as e:
        logger.exception("Hybrid prefetch failed");
        sys.exit(1)

    new_cfg = dict(cfg)
    new_cfg["bids_root"] = local_bids_root
    new_cfg["_hybrid_local_bids_root"] = local_bids_root
    new_cfg["_hybrid_remote_bids_root"] = remote_bids_root
    return (new_cfg, True, checkpoint_exists, local_bids_root, remote_bids_root)


def maybe_hybrid_push(cfg: dict):
    """
    Execute hybrid transfer push operation to return processed data to HPC.
    """
    if not cfg.get("_hybrid_local_bids_root") or not cfg.get("_hybrid_remote_bids_root"):
        return
    if HybridTransferManager is None:
        logger.error("Hybrid push requested but transfer_manager not importable")
        return

    local_bids_root = cfg["_hybrid_local_bids_root"]
    remote_bids_root = cfg["_hybrid_remote_bids_root"]
    rio = cfg.get("remote_io", {})
    hpc_host = rio.get("hpc_host", cfg.get("hpc_host"))
    hpc_user = rio.get("hpc_user", cfg.get("hpc_user"))
    local_temp_dir = rio.get("local_temp_dir", cfg.get("temp_dir", "./temp"))

    tm = HybridTransferManager(hpc_host, hpc_user, local_temp_dir)
    logger.info("HYBRID: Pushing derivatives (parproc + preproc) back to HPC")
    try:
        exit_code = tm.push_results(local_bids_root, remote_bids_root,
                                    cfg["subject"], cfg.get("session"))
        if exit_code == 0:
            logger.info("Hybrid push complete")
        else:
            logger.warning(f"Hybrid push finished with exit code {exit_code}")
    except Exception as e:
        logger.exception("Hybrid push failed")


def run_pipeline(yaml_path: str, force_stage: str = None):
    """
    Main pipeline execution function.
    """
    utils.log_section("1. Load Configuration and Detect Pipeline Stage")
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}");
        sys.exit(1)

    try:
        p0 = utils.build_effective_config(
            user_yaml_path=yaml_path,
            lab_defaults_path=os.getenv("LAB_DEFAULTS_YAML"),
            fif_path=None
        )
    except Exception as e:
        logger.error(f"Failed to build effective configuration: {e}")
        sys.exit(1)

    for key in ["subject", "bids_root"]:
        if key not in p0 and not (key == "bids_root" and p0.get("remote_io", {}).get("enabled")):
            logger.error(f"Missing required configuration key: {key}")
            sys.exit(1)

    p, hybrid_used, checkpoint_present, local_bids_root, remote_bids_root = maybe_hybrid_prefetch(p0)

    stage, checkpoint_path = detect_pipeline_stage(yaml_path, p, force_stage)
    logger.info(f"Running pipeline in {stage.upper()} mode")
    try:
        utils.setup_matplotlib(stage)
    except AttributeError:
        pass

    if stage == "stage1":
        status, _ = run_stage1_pipeline(yaml_path, p)
        if status == 'exit':
            logger.info("Stage 1 completed - exiting as requested")
            sys.exit(0)
        elif status == 'continue':
            stage2, checkpoint_path = detect_pipeline_stage(yaml_path, p, None)
            if stage2 != "stage2" or checkpoint_path is None:
                logger.error("Expected Stage 2 after Stage 1, but no checkpoint found")
                sys.exit(1)
            stage = "stage2"
            try:
                utils.setup_matplotlib(stage)
            except AttributeError:
                pass

    if stage == "stage2":
        run_stage2_pipeline(yaml_path, p, checkpoint_path)
        if hybrid_used:
            maybe_hybrid_push(p)
        logger.info("Pipeline completed successfully!")
        return

    logger.error(f"Unknown pipeline stage: {stage}")
    sys.exit(1)

def parse_arguments():
    """
    Parse command line arguments for pipeline execution.
    """
    parser = argparse.ArgumentParser(
        description="MEG/EEG Preprocessing Pipeline with Hybrid Transfer Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local-only processing (no transfers):
  python pipeline.py config_local.yaml

  # Hybrid mode (pull raw+derivatives to ./temp/bids, run both stages, push results back):
  python pipeline.py config_hybrid.yaml

  # Force specific stage execution:
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
        logging.getLogger().setLevel(logging.DEBUG);
        logger.setLevel(logging.DEBUG)

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