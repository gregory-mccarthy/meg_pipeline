#!/usr/bin/env python3
"""
EMPTY ROOM MEG PREPROCESSING PIPELINE

A streamlined preprocessing script specifically for MEG empty room recordings.
Uses the same YAML configuration file as preprocess_meg.py so that identical
Maxwell filter and notch filter parameters are applied. Automatically discovers
the empty room recording (task-noise or task-emptyroom) in the same subject/
session MEG directory.

The covariance output file is named to mirror the task data (e.g.
task-mns_run-01_desc-noise_cov.fif) so downstream inverse modeling code can
locate it using the same BIDS entities. If the empty room was recorded at a
higher sampling rate than the task data, the empty room is downsampled before
computing covariance so the noise estimate matches the task bandwidth.

Safely skips human-specific YAML parameters (head movement, ICA, AutoReject).

Outputs:
  - Preprocessed empty room data (Maxwell filtered, Notch filtered)
  - Task-matched noise covariance matrix (*_desc-noise_cov.fif)
  - JSON and YAML processing logs
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import mne

import meg_pipeline_utils as utils
from bids_io_utils import (parse_time_window, resolve_meg_dir, parse_meg_fname,
                          find_first_file_for_run, build_bids_stem)

mne.set_log_level('WARNING')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("empty_room_pipeline")

# Recognised empty room task labels, checked in priority order
_ER_TASK_LABELS = ("noise", "emptyroom")


def find_empty_room_task(bids_root, subject, session):
    """
    Scan the subject's MEG directory for a file whose task entity is one of
    the recognised empty room labels ('noise', 'emptyroom').

    Returns the task string on success; raises FileNotFoundError otherwise.
    """
    meg_dir = resolve_meg_dir(Path(bids_root), subject, session)
    if not meg_dir.exists():
        raise FileNotFoundError(
            f"MEG directory not found for sub-{subject}"
            + (f"/ses-{session}" if session else "")
            + f": {meg_dir}"
        )

    # Parse every .fif filename and collect the task labels present on disk
    tasks_on_disk = set()
    for fif in meg_dir.glob("*.fif"):
        try:
            info = parse_meg_fname(fif.name)
            if info.get("task"):
                tasks_on_disk.add(info["task"])
        except ValueError:
            continue

    # Return the first recognised empty room label found
    for label in _ER_TASK_LABELS:
        if label in tasks_on_disk:
            logger.info(f"Discovered empty room recording: task-{label}")
            return label

    raise FileNotFoundError(
        f"No empty room recording found in {meg_dir}.\n"
        f"  Looked for task labels: {list(_ER_TASK_LABELS)}\n"
        f"  Task labels on disk:    {sorted(tasks_on_disk) if tasks_on_disk else '(none)'}"
    )


def run_empty_room_pipeline(yaml_path: str, er_file_override: str = None):
    utils.log_section("1. Load Configuration")
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}")
        sys.exit(1)

    p = utils.build_effective_config(
        user_yaml_path=yaml_path,
        lab_defaults_path=os.getenv("LAB_DEFAULTS_YAML"),
        fif_path=None
    )

    repo_root = Path(__file__).resolve().parent
    try:
        checked_paths = utils.check_critical_files_exist(p, repo_root)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    subject = p["subject"]
    session = p.get("session")
    bids_root = p["bids_root"]

    # Read the task and run from the YAML — these identify the task data whose
    # inverse model will use this covariance, and are used for output naming.
    source_task = p.get("task")
    source_run = p.get("run")
    if not source_task:
        logger.error("YAML must contain a 'task' field (e.g. task: mns).")
        sys.exit(1)

    utils.log_section("2. Load Empty Room Data")

    if er_file_override:
        # Explicit file provided — skip discovery, load directly
        er_path = Path(er_file_override).resolve()
        if not er_path.exists():
            logger.error(f"Specified empty room file not found: {er_path}")
            sys.exit(1)
        logger.info(f"Using explicit empty room file: {er_path}")
        raw = mne.io.read_raw_fif(str(er_path), preload=True, verbose=False)

        # Derive the ER task label from the filename for ER output naming
        try:
            er_task = parse_meg_fname(er_path.name).get("task", "noise")
        except ValueError:
            er_task = "noise"
    else:
        # Auto-discover the empty room recording in the subject's MEG directory.
        er_task = find_empty_room_task(bids_root, subject, session)

        er_bids_path = utils.BIDSPath(
            subject=subject, session=session, task=er_task, run=None,
            datatype="meg", root=bids_root
        )
        raw = utils.read_raw_bids_robust(er_bids_path)

    er_sfreq = float(raw.info['sfreq'])

    # Peek at the task data to get its sampling rate
    logger.info(f"Peeking at task-{source_task} data to determine sampling rate...")
    try:
        task_fif, _, _ = find_first_file_for_run(
            bids_root, subject, session, task=source_task, run=source_run
        )
        task_raw = mne.io.read_raw_fif(str(task_fif), preload=False, verbose=False)
        task_sfreq = float(task_raw.info['sfreq'])
        del task_raw
        logger.info(f"Task sampling rate: {task_sfreq} Hz | Empty room sampling rate: {er_sfreq} Hz")
    except Exception as e:
        logger.warning(f"Could not peek at task data ({e}). Using empty room native rate.")
        task_sfreq = er_sfreq

    # Optional cropping
    tw = parse_time_window(p)
    if tw:
        tmin, tmax = tw
        raw.crop(tmin=tmin, tmax=tmax)
        logger.info(f"Cropped empty room recording to: {tmin}s - {tmax}s")

    # Output paths
    deriv_root = Path(bids_root) / "derivatives" / p.get("checkpoint", {}).get("pipeline_name", "preprocessing")

    # Preprocessed ER data: one per session, named with the ER task label
    er_bids_deriv = utils.BIDSPath(
        subject=subject, session=session, task=er_task, run=None,
        datatype="meg", root=str(deriv_root)
    ).update(suffix="meg", description="preproc", extension=".fif")
    out_fif = str(er_bids_deriv.fpath)

    # Covariance file: named to mirror the task data's BIDS entities.
    # Built as a string because 'cov' is not in MNE-BIDS's allowed suffix list.
    cov_dir = os.path.dirname(out_fif)  # same derivatives meg directory
    cov_stem = build_bids_stem(subject, session, source_task, source_run)
    cov_path = os.path.join(cov_dir, f"{cov_stem}_desc-noise_cov.fif")

    os.makedirs(os.path.dirname(out_fif), exist_ok=True)
    plots_dir = os.path.join(os.path.dirname(out_fif), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    utils.qc_meg_raw(raw, plots_dir)
    utils.plot_psd_and_peaks(raw, "ER Raw PSD Before Maxwell", plots_dir)

    utils.log_section("3. Maxwell Filter (Empty Room Mode)")

    # Extract intended duration, but cap it safely for short ER files
    st_duration = float(p.get("head_position_processing", {}).get("st_duration", 10.0))
    data_duration = raw.times[-1]

    if st_duration > data_duration:
        logger.warning(f"[SSS] Requested st_duration ({st_duration}s) exceeds data ({data_duration:.1f}s).")
        if data_duration <= 1.0:
            st_duration = None
            logger.warning("[SSS] Using standard SSS (no temporal extension) due to short data.")
        else:
            st_duration = float(data_duration)
            logger.warning(f"[SSS] Reduced st_duration to {st_duration:.3f}s.")

    maxwell_kwargs = dict(
        calibration=str(checked_paths["calibration_file"]),
        cross_talk=str(checked_paths["cross_talk_file"]),
        coord_frame="meg",
        origin=(0., 0., 0.),
        head_pos=None,
        destination=None,
        st_duration=st_duration,
        verbose=False,
    )

    logger.info("Applying empty room SSS (coord_frame='meg', origin=(0,0,0))")
    raw = mne.preprocessing.maxwell_filter(raw, **maxwell_kwargs)
    utils.plot_psd_and_peaks(raw, "ER After Maxwell", plots_dir)

    utils.log_section("4. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    picks = mne.pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    raw.notch_filter(notch_freqs, picks=picks, method='fir', filter_length='auto')
    utils.plot_psd_and_peaks(raw, "ER After Maxwell and Notch", plots_dir)

    utils.log_section("5. Compute Noise Covariance")

    # Resample the processed empty room data to match the task sampling rate
    # before computing covariance, so the noise estimate reflects the same bandwidth.
    raw_for_cov = raw.copy()
    cov_sfreq = er_sfreq  # track what rate the covariance was computed at

    if task_sfreq < er_sfreq:
        logger.info(f"Downsampling empty room copy from {er_sfreq} Hz to {task_sfreq} Hz for covariance.")
        raw_for_cov.resample(task_sfreq, verbose=False)
        cov_sfreq = task_sfreq
    elif task_sfreq > er_sfreq:
        logger.warning(
            "=" * 60 + "\n"
            "  SAMPLING RATE WARNING\n"
            "  The empty room was recorded at %.0f Hz but the task data is at %.0f Hz.\n"
            "  The noise covariance will underestimate noise power above %.0f Hz\n"
            "  (the empty room Nyquist frequency). For best results, collect empty\n"
            "  room data at a sampling rate >= the highest task sampling rate.\n"
            + "=" * 60, er_sfreq, task_sfreq, er_sfreq / 2.0
        )
    else:
        logger.info(f"Sampling rates match ({er_sfreq} Hz). No resampling needed.")

    logger.info("Computing empirical noise covariance matrix using MEG channels...")
    picks_cov = mne.pick_types(raw_for_cov.info, meg=True, eeg=False, exclude='bads')
    cov = mne.compute_raw_covariance(
        raw_for_cov,
        tmin=0,
        tmax=None,
        method=['shrunk', 'empirical'],
        picks=picks_cov,
        verbose=False
    )
    del raw_for_cov

    # Plot the covariance matrices for visual QC
    try:
        fig_cov, fig_svd = mne.viz.plot_cov(cov, raw.info, show=False)
        fig_cov.savefig(os.path.join(plots_dir, "ER_noise_covariance.png"))
        logger.info("Saved noise covariance plots.")
    except Exception as e:
        logger.warning(f"Failed to plot covariance: {e}")

    utils.log_section("6. Save Outputs")

    try:
        raw.save(out_fif, overwrite=True)
        logger.info(f"Saved preprocessed empty room data: {out_fif}")

        mne.write_cov(cov_path, cov, overwrite=True)
        logger.info(f"Saved noise covariance matrix: {cov_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        sys.exit(1)

    utils.log_section("7. Write Comprehensive Logs")

    empty_room_log = {
        "pipeline_type": "empty_room_preprocessing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_yaml": os.path.abspath(yaml_path),
        "summary": {
            "subject": subject,
            "session": session,
            "er_task": er_task,
            "source_task": source_task,
            "source_run": source_run,
            "er_sampling_rate_hz": er_sfreq,
            "task_sampling_rate_hz": task_sfreq,
            "cov_sampling_rate_hz": cov_sfreq,
            "recording_duration_sec": float(raw.times[-1]),
        },
        "final_state": {
            "n_channels_total": len(raw.ch_names),
            "n_bad_channels": len(raw.info.get('bads', [])),
            "bad_channels": list(raw.info.get('bads', [])),
            "channel_counts": {
                "meg_mag": len(mne.pick_types(raw.info, meg='mag', exclude='bads')),
                "meg_grad": len(mne.pick_types(raw.info, meg='grad', exclude='bads')),
            }
        },
        "parameters": {
            "maxwell_filter": {
                "coord_frame": maxwell_kwargs["coord_frame"],
                "origin": list(maxwell_kwargs["origin"]),
                "st_duration": maxwell_kwargs["st_duration"]
            },
            "notch_filter": {
                "frequencies_hz": notch_freqs
            }
        },
        "outputs": {
            "preprocessed_data": str(out_fif),
            "noise_covariance": str(cov_path),
            "plots_directory": str(plots_dir)
        }
    }

    out_log_yaml = out_fif.replace("_meg.fif", "_log.yaml")
    out_log_json = out_fif.replace("_meg.fif", "_log.json")

    try:
        serializable_log = utils.make_serializable(empty_room_log)

        utils.save_yaml(str(out_log_yaml), serializable_log)
        with open(out_log_json, "w") as f:
            json.dump(serializable_log, f, indent=2)

        logger.info(f"Final logs written:")
        logger.info(f"   YAML: {out_log_yaml}")
        logger.info(f"   JSON: {out_log_json}")
    except Exception as e:
        logger.error(f"Failed to write comprehensive logs: {e}")

    logger.info("Empty room processing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MEG Empty Room Data")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--empty-room-file", type=str, default=None,
                        help="Path to a specific empty room .fif file. "
                             "Bypasses automatic discovery. Use when the session "
                             "contains multiple noise recordings and you need to "
                             "select one explicitly.")
    args = parser.parse_args()

    print("=" * 60)
    print("MEG Empty Room Preprocessing Pipeline")
    print(f"Configuration: {args.config}")
    if args.empty_room_file:
        print(f"Empty room override: {args.empty_room_file}")
    print("=" * 60, "\n")

    try:
        run_empty_room_pipeline(args.config, er_file_override=args.empty_room_file)
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        sys.exit(1)
