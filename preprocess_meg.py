#!/usr/bin/env python3
# ==========================================================
# MNE-PYTHON PREPROCESSING PIPELINE — VERSION 1.0
# by Gregory McCarthy and ChatGPT — 2025-07-28
# ==========================================================
"""
Modular MEG/EEG Preprocessing Pipeline (MNE-Python)

**Pipeline Overview:**
    1. Load YAML config and critical files
    2. Load raw BIDS MEG/EEG data (.fif)
    3. Compute or load head position
    4. EEG channel setup (montage, digitization, renaming)
    5. Metadata repairs (YAML or expert_patch)
    6. PSD & RMS diagnostics (pre-filtering, quality control)
    7. Notch filter for line noise (e.g., 60Hz)
    8. Maxwell filter (tSSS) for MEG (spatial filter)
    9. Bad channel detection (AutoReject/RANSAC for EEG, MAG, GRAD)
   10. ICA for EEG (interactive review)
   11. ICA for MEG (interactive review)
   12. Event/stimulus detection and annotation
   13. Optional final filter, downsampling, channel drop, cleanup
   14. Save final data and YAML/JSON logs
   15. (Optional) Push outputs to remote/HPC (if hybrid workflow)
"""

import os
import sys
import logging
from pathlib import Path
import mne
import meg_pipeline_utils as utils
import json
import numpy as np

import matplotlib
matplotlib.use('QtAgg')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")
PIPELINE_VERSION = "1.0"


def run_pipeline(yaml_path: str):
    # ===================================================
    # 1. Load Config, Set Paths, and Check Critical Files
    # ===================================================
    utils.log_section("1. Load config and check paths and critical files")
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}")
        sys.exit(1)

    p = utils.load_yaml(yaml_path)
    repo_root = Path(__file__).resolve().parent
    logger.info(f"Repo root set to: {repo_root}")

    try:
        checked_paths = utils.check_critical_files_exist(p, repo_root)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("All critical files found:")
    for key, path in checked_paths.items():
        logger.info(f"  {key}: {path}")

    runtime_info = utils.get_runtime_info()
    subject = p["subject"]
    session = p.get("session", None)
    task = p.get("task", None)
    run = p.get("run", None)
    bids_root = p["bids_root"]

    ENV = utils.detect_environment(hpc_hostname_tag="milgram")

    # ---------------------------------------------
    # 1.1 Decide Local vs. Remote Data Handling
    # ---------------------------------------------
    if ENV == "hpc" or bids_root.startswith("/Users"):
        local_bids_root = bids_root
    else:
        temp_dir = p.get("temp_dir", "./temp")
        base_stem = f"sub-{subject}"
        if session: base_stem += f"_ses-{session}"
        if task: base_stem += f"_task-{task}"
        if run: base_stem += f"_run-{run}"
        remote_meg_dir = os.path.join(bids_root, f"sub-{subject}", f"ses-{session}" if session else "", "meg")
        local_bids_root = os.path.join(temp_dir, "bids")
        local_meg_dir = os.path.join(local_bids_root, f"sub-{subject}", f"ses-{session}" if session else "", "meg")
        utils.fetch_bids_data_and_sidecars(p.get("hpc_host"), p.get("hpc_user"), remote_meg_dir, base_stem,
                                           local_meg_dir)

    bids_path = utils.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        root=local_bids_root
    )
    bids_path_deriv = bids_path.copy().update(
        root=Path(local_bids_root) / "derivatives" / "preprocessing",
        suffix="meg",
        description="preproc",
        extension=".fif"
    )
    bids_path_ica_eeg = bids_path_deriv.copy().update(description="preprocICAeeg")
    bids_path_ica_meg = bids_path_deriv.copy().update(description="preprocICAmeg")

    # ==============================
    # 2. Load Raw Data
    # ==============================
    utils.log_section("2. Load Raw Data")
    raw = utils.read_raw_bids_robust(bids_path)
    raw_fif_path = raw.filenames[0]
    meg_dir = os.path.dirname(raw_fif_path)

    expected_pos_file = utils.get_bids_headpos_path(subject, session, task, run, meg_dir)
    pos_file_requested = expected_pos_file if os.path.exists(expected_pos_file) else None
    print(
        f"Using existing head position file: {pos_file_requested}" if pos_file_requested else "No matching headpos.pos file found. Will compute if needed.")

    # ==============================
    # 3. Compute or Load Head Position
    # ==============================
    utils.log_section("3. Compute or Load Head Position")
    new_headpos_computed = False
    head_movement_cfg = p.get("head_movement", {})
    movement_enabled = head_movement_cfg.get("enabled", False)
    head_pos_array = None
    head_movement_log = {
        "enabled": movement_enabled,
        "method": None,
        "file_used": None
    }

    if movement_enabled:
        head_pos_path = utils.get_head_pos_for_maxwell(
            raw, pos_file=pos_file_requested, compute_if_missing=True, logger=logger
        )
        if head_pos_path and os.path.exists(head_pos_path):
            try:
                head_pos_array = utils.mne.chpi.read_head_pos(head_pos_path)
                head_movement_log.update({
                    "method": "computed" if pos_file_requested is None else "user-supplied",
                    "file_used": head_pos_path
                })
            except Exception as e:
                logger.warning(f"Could not read computed .pos file: {head_pos_path}\n{e}")
                head_pos_array = None
        else:
            logger.warning("Head movement was enabled but no .pos file was found or computed.")
    else:
        logger.info("Head movement correction is disabled.")

    if movement_enabled and (pos_file_requested is None):
        if head_pos_path and os.path.exists(head_pos_path):
            new_headpos_computed = True

    # ==============================
    # Define output file and plots directory early
    # ==============================
    out_fif = bids_path_deriv.fpath
    plots_dir = os.path.join(os.path.dirname(out_fif), "plots")

    # ==============================
    # (Optional): Save head movement plot if data is available
    # ==============================
    if movement_enabled and head_pos_array is not None:
        utils.plot_head_movement(head_pos_array, plots_dir)

    # ==============================
    # 4. EEG Channel Setup
    # ==============================
    utils.log_section("4. EEG Channel Setup")
    montage_path = checked_paths["montage"]
    eeg_status = utils.prepare_eeg_channels(raw, str(montage_path), logger)

    # ==============================
    # 5. Metadata Repair
    # ==============================
    utils.log_section("5. Metadata Repair")
    metadata_log = utils.apply_metadata_repairs(raw, p.get('metadata_fixes', {}))

    # ================================================
    # 6. PSD & RMS Diagnostics (Pre-filtering)
    # ================================================
    utils.log_section("6. PSD & RMS Diagnostics (Pre-filtering)")
    utils.qc_meg_raw(raw, plots_dir)
    utils.plot_psd_and_peaks(raw, "Raw PSD Before Maxwell", plots_dir)

    # ==============================
    # 7. Maxwell Filter (tSSS)
    # ==============================
    utils.log_section("7. Maxwell Filter (tSSS)")
    raw = utils.apply_maxwell_filter(
        raw,
        head_pos=head_pos_array,
        destination=None,
        cal=str(checked_paths["calibration_file"]),
        crosstalk=str(checked_paths["cross_talk_file"])
    )
    utils.plot_psd_and_peaks(raw, "After Maxwell", plots_dir)

    # ==============================
    # 8. Notch Filter
    # ==============================
    utils.log_section("8. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    picks = utils.mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    raw.notch_filter(notch_freqs, picks=picks, method='fir', filter_length='auto')
    utils.plot_psd_and_peaks(raw, "After Maxwell and Notch", plots_dir)

    # ==================================================
    # 9. Bad Channel Detection (AutoReject, All Types)
    # ==================================================
    utils.log_section("9. Bad Channel Detection with AutoReject")

    ar_cfg = p.get("autoreject", {})
    ar_enabled = ar_cfg.get("enabled", True)
    ar_types = ar_cfg.get("which_types", ['eeg', 'mag', 'grad'])
    autoreject_filter_settings = ar_cfg.get("filter", {
        'highpass': 1.0,
        'lowpass': 40.0,
        'resample_hz': None
    })
    epoch_settings = ar_cfg.get("epoch", {
        'duration': 2.0,
        'tmin': 0.0,
        'tmax': 2.0
    })
    consensus_thresh = ar_cfg.get("consensus_thresh", 0.3)
    global_epoch_thresh = ar_cfg.get("global_epoch_thresh", 0.3)  # NEW PARAMETER

    if ar_enabled:
        logger.info("==== [9] Running autoreject bad channel detection for all types (EEG, MAG, GRAD) ====")
        bads_dict = utils.find_bad_channels_autoreject_by_type(
            raw,
            which_types=ar_types,
            filter_settings=autoreject_filter_settings,
            epoch_settings=epoch_settings,
            consensus_thresh=consensus_thresh,
            global_epoch_thresh=global_epoch_thresh,  # <-- NEW ARGUMENT
            interactive=True,
            logger=logger
        )
        logger.info(f"Bad channels detected by type: {bads_dict}")
        logger.info(f"All channels marked bad after detection: {raw.info['bads']}")
        for typ in ar_types:
            gb_epochs = bads_dict.get(f"{typ}_global_bad_epochs", [])
            logger.info(f"Detected {len(gb_epochs)} global bad epochs for {typ.upper()} (indices: {gb_epochs})")
    else:
        logger.info("Autoreject bad channel detection is DISABLED in YAML config. Skipping this step.")

    # ==============================
    # 9a. Estimate MEG Data Rank after Bad Channel Marking
    # ==============================
    empirical_rank = mne.compute_rank(raw)
    logger.info(f"Empirical rank after Maxwell filtering and bad channel detection: {empirical_rank}")

    # ==============================
    # 10. ICA: EEG
    # ==============================
    utils.log_section("10. ICA: EEG")
    ica_eeg_cfg = p["ica_preprocessing"]["eeg"]
    raw, eeg_exclude = utils.run_ica(raw, ica_eeg_cfg, bids_path_ica_eeg.fpath, modality="eeg")

    # ==============================
    # 11. ICA: MEG
    # ==============================
    utils.log_section("11. ICA: MEG")
    ica_meg_cfg = p["ica_preprocessing"]["meg"]
    raw, meg_exclude = utils.run_ica(raw, ica_meg_cfg, bids_path_ica_meg.fpath, modality="meg")

    utils.plot_psd_and_peaks(raw, "After ICA", plots_dir)

    # ==============================
    # 12. Event Detection
    # ==============================
    utils.log_section("12. Event Detection")
    events = utils.bitwise_events(raw)
    event_counts = {}
    if events.size:
        annots = utils.mne.annotations_from_events(events, sfreq=raw.info['sfreq'])
        # Combine existing annotations with new event annotations
        if raw.annotations is not None and len(raw.annotations) > 0:
            # Create combined annotations manually to avoid orig_time issues
            combined_annotations = mne.Annotations(
                onset=list(raw.annotations.onset) + list(annots.onset),
                duration=list(raw.annotations.duration) + list(annots.duration),
                description=list(raw.annotations.description) + list(annots.description),
                orig_time=raw.annotations.orig_time
            )
            raw.set_annotations(combined_annotations)
        else:
            raw.set_annotations(annots)

        codes, counts = utils.np.unique(events[:, 2], return_counts=True)
        event_counts = {int(c): int(n) for c, n in zip(codes, counts)}
        for code, n in event_counts.items():
            logger.info(f"Event {code}: {n} occurrences")
    else:
        logger.warning("No events detected")

    # ==============================
    # 13. Optional Final Filter, Downsample, and Cleanup
    # ==============================
    utils.log_section("13. Optional Final Filter, Downsample, and Cleanup")
    raw = utils.apply_final_filter_and_cleanup(raw, p)

    # ==============================
    # 14. Save Final Data
    # ==============================
    utils.log_section("14. Save Final Data")
    written_files = utils.write_bids_robust(raw, bids_path_deriv, overwrite=True, verbose=True)
    logger.info(f"Saved preprocessed data to: {out_fif}")

    # ==============================
    # 15. Log and Upload Outputs (add rank info to logs)
    # ==============================
    final_filter_cfg = p.get("final_filter", {})
    drop_types = final_filter_cfg.get("drop_channel_types", [])
    dropped_channels = [
        ch for ch in raw.ch_names
        if any(ch.upper().startswith(prefix.upper()) for prefix in drop_types)
    ]
    final_filter_log = {
        "highpass": final_filter_cfg.get("highpass", None),
        "lowpass": final_filter_cfg.get("lowpass", None),
        "resample_hz": final_filter_cfg.get("resample_hz", None),
        "drop_channel_types": drop_types,
        "channels_dropped": dropped_channels
    }

    recording_duration_sec = raw.times[-1]
    ica_durations = {
        "eeg_duration_sec": raw.times[-1] if 'eeg_exclude' in locals() else None,
        "meg_duration_sec": raw.times[-1] if 'meg_exclude' in locals() else None
    }

    main_output_files = utils.get_all_bids_split_files(out_fif)
    ica_output_files = []
    for ica_fif in [bids_path_ica_eeg.fpath, bids_path_ica_meg.fpath]:
        ica_output_files += utils.get_all_bids_split_files(ica_fif)
    ica_output_files = list(dict.fromkeys(ica_output_files))

    yaml_log = {
        "meg_rank_estimate": empirical_rank,
        "input_config": yaml_path,
        "bids_basefile": str(out_fif),
        "main_output_files": main_output_files,
        "ica_output_files": ica_output_files,
        "output_file": str(out_fif),
        "runtime_info": runtime_info,
        "recording_duration_sec": float(recording_duration_sec),
        "final_filter": final_filter_log,
        "ica_duration": ica_durations,
        "bad_channels": raw.info.get('bads', []),
        "eeg_channel_preparation": eeg_status,
        "metadata_repairs": metadata_log,
        "head_movement": head_movement_log,
        "ica": {
            "eeg_excluded": eeg_exclude,
            "meg_excluded": meg_exclude,
        },
        "event_counts": event_counts,
        "log_version": PIPELINE_VERSION,
    }

    out_log_yaml = out_fif.with_name(out_fif.stem + "_log.yaml")
    out_log_json = out_fif.with_name(out_fif.stem + "_log.json")

    utils.save_yaml(out_log_yaml, utils.make_serializable(yaml_log))
    logger.info(f"YAML log written to: {out_log_yaml}")

    with open(out_log_json, "w") as f:
        json.dump(utils.make_serializable(yaml_log), f, indent=2)
    logger.info(f"JSON log written to: {out_log_json}")

    # Optionally upload outputs if needed
    ENV = utils.detect_environment(hpc_hostname_tag="milgram")
    main_output_files = utils.write_bids_robust(raw, out_fif, overwrite=True, verbose=True)

    ica_output_files = []
    for ica_fif in [bids_path_ica_eeg.fpath, bids_path_ica_meg.fpath]:
        ica_output_files += utils.get_all_bids_split_files(ica_fif)
    ica_output_files = list(dict.fromkeys(ica_output_files))

    outputs_to_push = [str(out_log_yaml), str(out_log_json)] + main_output_files + ica_output_files
    outputs_to_push = list(dict.fromkeys(outputs_to_push))

    if utils.is_hybrid_workflow(p["bids_root"], out_fif):
        print("\n[User action required]")
        print(
            "Processing complete. Press ENTER when ready to upload all derivatives to the HPC (be ready for DUO prompt).")
        input()
        utils.push_bids_derivatives_rsync(
            local_bids_root="./temp/BIDS",
            remote_bids_root=p["bids_root"],
            hpc_host=p["hpc_host"],
            hpc_user=p["hpc_user"],
            verbose=True
        )
        if new_headpos_computed:
            print(
                "\nA new head position .pos file was computed. Press ENTER to upload it to the HPC raw folder (be ready for DUO prompt).")
            input()
            local_bids_root = os.path.abspath("./temp/BIDS")
            rel_path = os.path.relpath(pos_path, start=local_bids_root)
            remote_file = os.path.join(p["bids_root"], rel_path)
            remote_dir = os.path.dirname(remote_file)
            remote_spec = f"{p['hpc_user']}@{p['hpc_host']}:{remote_dir}/"
            print(f"Pushing new/updated headpos file: {pos_path} -> {remote_spec}")
            import subprocess
            subprocess.run(["scp", "-v", pos_path, remote_spec], check=True)
            print("[info] Head position file upload complete.")
    else:
        print("[info] No remote data used or output—skipping remote upload step.")


# ==========================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python preprocessing_pipeline_v1_0.py config.yaml")
        sys.exit(1)
    else:
        try:
            run_pipeline(sys.argv[1])
        except Exception as e:
            logger.exception(f"Pipeline execution failed: {e}")
            sys.exit(1)
