#!/usr/bin/env python3
"""
OPM PREPROCESSING PIPELINE

A two-stage preprocessing pipeline tailored for Cerca 64-triaxial OPM data.
Mirrors the architecture of preprocess_meg.py but implements OPM-specific
physics (HFC/mSSS) and hardware adaptations.

PIPELINE STAGES:
  Stage 1: Data loading, event/channel TSV ingestion, coil patching,
           spatial denoising (HFC or mSSS), AutoReject, checkpointing.
  Stage 2: Interactive review, ICA processing, final filtering.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np

import mne

mne.set_log_level('WARNING')

# Core Brain Pipes Infrastructure
from bids_io_utils import (
    parse_time_window,
    apply_bids_events_tsv,
    apply_bids_channels_tsv
)
import meg_pipeline_utils as utils

# OPM Specific Physics and Hardware Handlers
from opm_msss import (
    MSSSConfig,
    apply_hfc,
    apply_python_msss,
    auto_detect_dead_channels,
    patch_cerca_opm_coil_types,
    suggest_two_msss_centers,
    summarize_opm_geometry
)

try:
    from transfer_manager import HybridTransferManager
except ImportError:
    HybridTransferManager = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("opm_pipeline")
PIPELINE_VERSION = "1.2-OPM"
CHECKPOINT_VERSION = "1.0"


def detect_pipeline_stage(yaml_path: str, p: dict, force_stage: str = None):
    """Determine whether to execute Stage 1 or resume from Stage 2 checkpoint."""
    if force_stage:
        logger.info(f"Pipeline stage forced: {force_stage}")
        if force_stage == "stage1":
            return ("stage1", None)
        elif force_stage != "stage2":
            logger.error(f"Invalid forced stage: {force_stage}")
            sys.exit(1)

    chk_cfg = p.get("checkpoint", {})
    if not chk_cfg.get("enabled", True) and not force_stage:
        logger.info("Checkpoint system disabled in configuration")
        return ("stage1", None)

    bids_root = Path(p["bids_root"])
    deriv_root = bids_root / chk_cfg.get("derivatives_root", "derivatives") / chk_cfg.get("pipeline_name",
                                                                                          "preprocessing")

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
        return ("stage2", parproc_fif)

    if force_stage == "stage2":
        logger.error(f"Cannot force Stage 2: checkpoint not found at {parproc_fif}")
        sys.exit(1)

    return ("stage1", None)


def run_stage1_pipeline(yaml_path: str, p: dict):
    """
    STAGE 1: DATA LOADING AND OPM SPATIAL DENOISING
    """
    p["_cfg_path"] = yaml_path

    subject = p["subject"]
    session = p.get("session")
    task = p.get("task")
    run = p.get("run")
    bids_root = p["bids_root"]

    utils.log_section("1. Load Raw OPM Data")
    bids_path = utils.BIDSPath(subject=subject, session=session, task=task, run=run,
                               datatype="meg", root=bids_root)
    raw = utils.read_raw_bids_robust(bids_path)
    raw_fif_path = raw.filenames[0]

    # 2. Ingest Annotations (from cerca_opm_events.py)
    annot_cfg = p.get("annotations", {})
    if annot_cfg.get("read_events_tsv", False):
        raw = apply_bids_events_tsv(raw, raw_fif_path, logger=logger)

    # 3. Bad Channel Union Strategy
    utils.log_section("2. Bad Channel Aggregation")
    yaml_bads = p.get("manual_bad_channels", [])
    tsv_bads = apply_bids_channels_tsv(raw, raw_fif_path, logger=logger)

    dead_threshold = p.get("spatial_denoiser", {}).get("dead_channel_threshold", 1e-14)
    dead_bads = auto_detect_dead_channels(raw, threshold=dead_threshold)

    existing_bads = set(raw.info.get('bads', []))
    unified_bads = existing_bads.union(yaml_bads).union(tsv_bads).union(dead_bads)
    raw.info['bads'] = list(unified_bads)

    logger.info(f"[Channels] Combined OPM exclusion list: {raw.info['bads']}")
    logger.info(
        f"   - (Inherited: {len(existing_bads)} | TSV: {len(tsv_bads)} | YAML: {len(yaml_bads)} | Auto-Dead: {len(dead_bads)})")

    # 4. Hardware Patching
    utils.log_section("3. OPM Hardware Setup")
    n_patched = patch_cerca_opm_coil_types(raw)
    logger.info(f"Patched {n_patched} MEG channel(s) to point magnetometer coil type.")

    geom = summarize_opm_geometry(raw)
    logger.info("Geometry summary:")
    logger.info(json.dumps(geom, indent=2))

    # Optional cropping
    tw = parse_time_window(p)
    if tw:
        raw.crop(tmin=tw[0], tmax=tw[1])
        logger.info(f"Cropped recording to window: {tw}")

    # Build derivative paths
    bids_path_deriv = bids_path.copy().update(
        root=Path(bids_root) / "derivatives" / p.get("checkpoint", {}).get("pipeline_name", "preprocessing"),
        suffix="meg", description="preproc", extension=".fif"
    )
    plots_dir = os.path.join(os.path.dirname(bids_path_deriv.fpath), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    utils.qc_meg_raw(raw, plots_dir)
    utils.plot_psd_and_peaks(raw, "Raw PSD Before Denoising", plots_dir)

    # 5. Spatial Denoising (HFC or mSSS)
    utils.log_section("4. Spatial Denoising")
    denoiser_cfg = p.get("spatial_denoiser", {})
    method = denoiser_cfg.get("method", "hfc").lower()

    if method == "hfc":
        hfc_order = denoiser_cfg.get("hfc_order", 2)
        logger.info(f"Applying HFC with order={hfc_order}")
        raw = apply_hfc(raw, order=hfc_order, copy=False)
        denoiser_summary = {"name": "hfc", "order": hfc_order}

    elif method == "msss":
        auto_centers = denoiser_cfg.get("msss_auto_centers")
        if auto_centers:
            suggested = suggest_two_msss_centers(raw, frame=auto_centers)
            center1 = [float(x) for x in suggested.centers[0]]
            center2 = [float(x) for x in suggested.centers[1]]
            centers_frame = suggested.centers_frame
        else:
            center1 = denoiser_cfg.get("msss_center1")
            center2 = denoiser_cfg.get("msss_center2")
            centers_frame = denoiser_cfg.get("msss_centers_frame", "meg")

        config = MSSSConfig(
            center1=center1,
            center2=center2,
            centers_frame=centers_frame,
            int_order=denoiser_cfg.get("msss_int_order", 8),
            ext_order=denoiser_cfg.get("msss_ext_order", 3),
            threshold=denoiser_cfg.get("msss_threshold", 0.005),
            regularize=None,
            bad_condition="warning",
        )
        logger.info(f"Applying mSSS (centers_frame={centers_frame})...")
        raw, result = apply_python_msss(raw, config, copy=False)
        denoiser_summary = {"name": "python-msss", **result.summary()}
    else:
        logger.error(f"Unknown spatial denoiser method: {method}")
        sys.exit(1)

    utils.plot_psd_and_peaks(raw, f"After {method.upper()} Denoising", plots_dir)

    # 6. Basic Filtering
    utils.log_section("5. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    raw.notch_filter(notch_freqs, picks='meg', method='fir', filter_length='auto')

    # 7. AutoReject & Checkpoint
    utils.log_section("6. Bad Channel Detection with AutoReject (checkpoint)")
    p["_checkpoint_version"] = CHECKPOINT_VERSION
    try:
        checkpoint_path, stage1_ar_metadata = utils.run_autoreject_stage1(raw, p, bids_path, logger)
        logger.info("Stage 1 AutoReject and checkpoint creation completed")
    except Exception as e:
        logger.error(f"Stage 1 AutoReject failed: {e}")
        sys.exit(1)

    ar_results = {
        "bads_detected": list(raw.info.get('bads', [])),
        "n_bad_channels": len(raw.info.get('bads', [])),
    }

    manifest_path = utils.write_stage1_manifest(
        raw=raw,
        cfg=p,
        bids_path=bids_path,
        checkpoint_file=str(checkpoint_path),
        autoreject_results=ar_results,
        quality_metrics={"spatial_denoiser": denoiser_summary},
        head_movement_stats=None,
        rank_info={"method": "opm_hardware_defaults"},
        eeg_setup_results=None,
        metadata_repair_results=None,
        original_recording_info={"original_sfreq": float(raw.info["sfreq"])},
        notch_filter_params={"frequencies_hz": notch_freqs},
        processing_paths={"plots_directory": plots_dir},
        logger=logger,
    )

    if p.get("checkpoint", {}).get("exit_after_checkpoint", False):
        return ('exit', raw)
    return ('continue', raw)


def run_stage2_pipeline(yaml_path: str, p: dict, checkpoint_path: Path):
    """
    STAGE 2: INTERACTIVE PROCESSING AND FINALIZATION
    """
    p["_cfg_path"] = yaml_path
    p["_checkpoint_version"] = CHECKPOINT_VERSION

    utils.log_section("1. Load Stage 1 Checkpoint")
    raw, stage2_ar_metadata = utils.run_interactive_review_stage2(checkpoint_path, p, logger)
    logger.info("Stage 2 interactive review completed")

    bids_path = utils.BIDSPath(
        subject=p["subject"], session=p.get("session"), task=p.get("task"), run=p.get("run"),
        datatype="meg", root=p["bids_root"]
    )
    deriv_root = Path(p["bids_root"]) / "derivatives" / p.get("checkpoint", {}).get("pipeline_name", "preprocessing")
    bids_path_deriv = bids_path.copy().update(root=deriv_root, suffix="meg", description="preproc", extension=".fif")
    out_fif = bids_path_deriv.fpath
    plots_dir = os.path.join(os.path.dirname(out_fif), "plots")

    utils.log_section("2. ICA: MEG")
    ica_meg_cfg = p.get("ica_preprocessing", {}).get("meg")
    meg_exclude = []
    if ica_meg_cfg:
        reject_breaks = p.get("annotations", {}).get("reject_bad_breaks", True)
        bids_path_ica = bids_path_deriv.copy().update(description="preprocICAmeg").fpath
        try:
            raw, meg_exclude = utils.run_ica(raw, ica_meg_cfg, bids_path_ica, modality="meg",
                                             reject_by_annotation=reject_breaks)
        except Exception as e:
            logger.error(f"MEG ICA failed: {e}")

    utils.plot_psd_and_peaks(raw, "After ICA", plots_dir)

    utils.log_section("3. Final Filter and Cleanup")
    raw = utils.apply_final_filter_and_cleanup(raw, p)

    utils.log_section("4. Save Final Data")
    utils.write_bids_robust(raw, bids_path_deriv, overwrite=True, verbose=True)
    logger.info(f"Stage 2 completed successfully! Saved to: {out_fif}")


def run_pipeline(yaml_path: str, force_stage: str = None):
    utils.log_section("1. Load Configuration")
    p = utils.build_effective_config(user_yaml_path=yaml_path, lab_defaults_path=None, fif_path=None)

    stage, checkpoint_path = detect_pipeline_stage(yaml_path, p, force_stage)
    logger.info(f"Running OPM pipeline in {stage.upper()} mode")

    try:
        utils.setup_matplotlib(stage)
    except AttributeError:
        pass

    if stage == "stage1":
        status, _ = run_stage1_pipeline(yaml_path, p)
        if status == 'exit':
            sys.exit(0)

        stage2, checkpoint_path = detect_pipeline_stage(yaml_path, p, None)
        stage = "stage2"

    if stage == "stage2":
        run_stage2_pipeline(yaml_path, p, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cerca OPM Preprocessing Pipeline")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--force-stage", choices=["stage1", "stage2"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"Cerca OPM Preprocessing Pipeline v{PIPELINE_VERSION}")
    print("=" * 60, "\n")

    run_pipeline(args.config, args.force_stage)