import os, sys, platform, socket, logging, getpass, gc
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_bids import BIDSPath
from scipy.signal import find_peaks
import numpy as np
from autoreject import Ransac, AutoReject
import logging
from autoreject import Ransac, AutoReject
from collections import Counter

# Logging and version global
logger = logging.getLogger("pipeline")
PIPELINE_VERSION = "1.0"

# YAML support
try:
    from ruamel.yaml import YAML
    _yaml_mode = 'ruamel'
except ImportError:
    import yaml
    _yaml_mode = 'pyyaml'

# (If needed by main)
from bids_io_utils import (
    fetch_bids_data_and_sidecars,
    push_bids_derivatives_rsync,
    detect_environment,
    write_bids_robust,
    read_raw_bids_robust,
    get_all_bids_split_files
)

def load_yaml(path: str) -> dict:
    with open(path, 'r') as fh:
        if _yaml_mode == 'ruamel':
            yaml_obj = YAML()
            return yaml_obj.load(fh)
        else:
            return yaml.safe_load(fh)

def save_yaml(path: str, data: dict) -> None:
    with open(path, 'w') as fh:
        if _yaml_mode == 'ruamel':
            yaml_obj = YAML()
            yaml_obj.default_flow_style = False
            yaml_obj.dump(data, fh)
        else:
            yaml.safe_dump(data, fh, default_flow_style=False)

def make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, bytes):
        return obj.decode()
    elif hasattr(obj, '__str__') and not isinstance(obj, str):
        return str(obj)
    else:
        return obj

def log_section(title: str) -> None:
    banner = f"[ {title} ]"
    logger.info("\n" + "=" * len(banner))
    logger.info(banner)
    logger.info("=" * len(banner) + "\n")

def get_runtime_info() -> Dict[str, Any]:
    return {
        "script_version": PIPELINE_VERSION,
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "mne_version": mne.__version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
    }

def save_plot(fig, out_dir, fname):
    """Save a matplotlib figure and close it."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path

# ========== Workflow Determination ==========

def is_hybrid_workflow(bids_root, out_fif):
    """
    Returns True if the workflow is 'hybrid' (e.g., run locally but data or output is remote), else False.
    You can make this as strict or loose as you want!
    """
    # Example: local is '/Users/', remote is '/gpfs/' (customize for your lab)
    is_local_bids = str(bids_root).startswith("/Users/")
    is_local_out = str(out_fif).startswith("/Users/")
    # Hybrid if data or output is not local
    return not (is_local_bids and is_local_out)

# ========== Check paths and critical files ==========

def find_repo_root(start_path=None, marker='.git'):
    """Walk up from start_path until a directory containing 'marker' is found."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    else:
        start_path = Path(start_path).resolve()
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Repo root not found using marker '{marker}'.")

def resolve_path_from_repo_root(user_path, repo_root):
    """Return absolute Path: if relative, resolve from repo_root."""
    p = Path(user_path)
    return p if p.is_absolute() else repo_root / p

def get_and_check_path(cfg_dict, key, repo_root, description="file"):
    user_path = cfg_dict.get(key)
    if user_path is None:
        raise ValueError(f"No {description} specified in the config under {key}")
    resolved = resolve_path_from_repo_root(user_path, repo_root)
    if not resolved.exists():
        raise FileNotFoundError(f"{description.capitalize()} not found: {resolved}")
    return resolved

def check_critical_files_exist(config, repo_root):
    files_to_check = [
        # Each item: (dict, key, description)
        (config.get("eeg_handling", {}), "montage", "montage file"),
        (config, "calibration_file", "MEG calibration file"),
        (config, "cross_talk_file", "MEG crosstalk file"),
        # Add more as needed
    ]
    checked_paths = {}
    for cfg_dict, key, desc in files_to_check:
        checked_paths[key] = get_and_check_path(cfg_dict, key, repo_root, desc)
    return checked_paths

def get_bids_headpos_path(subject, session, task, run, meg_dir):
    fname = f"sub-{subject}"
    if session:
        fname += f"_ses-{session}"
    if task:
        fname += f"_task-{task}"
    if run:
        fname += f"_run-{run}"
    fname += "_headpos.pos"
    return os.path.join(meg_dir, fname)

# ========== CORE MODULES ==========

def get_head_pos_for_maxwell(raw: mne.io.Raw,
                              pos_file: Optional[str] = None,
                              compute_if_missing: bool = True,
                              logger: Optional[logging.Logger] = None) -> Optional[str]:
    if pos_file and os.path.isfile(pos_file):
        if logger:
            logger.info(f"Using user-specified head position file: {pos_file}")
        return pos_file

    if not compute_if_missing:
        if logger:
            logger.warning("No head pos file specified, and compute_if_missing is False.")
        return None

    try:
        if logger:
            logger.info("Computing head position from continuous HPI...")
        chpi_amps = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amps)
        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)

        if head_pos.shape[0] == 0:
            if logger:
                logger.warning("cHPI present but head_pos is empty. Skipping head movement correction.")
            return None
    except Exception as e:
        if logger:
            logger.error(f"Failed to compute head position from cHPI: {e}")
        return None

    try:
        raw_fname = raw.filenames[0] if hasattr(raw, 'filenames') else None
        if not raw_fname:
            raise ValueError("Cannot determine raw filename to derive output path.")

        raw_base = os.path.splitext(os.path.basename(raw_fname))[0]
        if raw_base.endswith("_meg"):
            raw_base = raw_base[:-4]
        pos_path = os.path.join(os.path.dirname(raw_fname), f"{raw_base}_headpos.pos")

        mne.chpi.write_head_pos(pos_path, head_pos)

        if logger:
            logger.info(f"Computed head position saved to: {pos_path}")
        return pos_path

    except Exception as e:
        if logger:
            logger.error(f"Failed to write computed .pos file: {e}")
        return None

def load_raw_data(config: dict) -> mne.io.Raw:
    bids_path = BIDSPath(
        subject=config['subject'],
        session=config.get('session'),
        task=config.get('task'),
        run=config.get('run'),
        root=config['bids_root'],
        datatype='meg'
    )
    fname = f"sub-{bids_path.subject}"
    if bids_path.session:
        fname += f"_ses-{bids_path.session}"
    if bids_path.task:
        fname += f"_task-{bids_path.task}"
    if bids_path.run:
        fname += f"_run-{bids_path.run}"
    fname += "_meg.fif"

    raw_file = os.path.join(
        bids_path.root,
        f"sub-{bids_path.subject}",
        f"ses-{bids_path.session}" if bids_path.session else "",
        "meg",
        fname
    ).replace("//", "/")

    if not os.path.exists(raw_file):
        logger.error(f"MEG file not found: {raw_file}")
        sys.exit(1)

    logger.info(f"Loading raw data: {raw_file}")
    return mne.io.read_raw_fif(raw_file, preload=True)

def prepare_eeg_channels(raw: mne.io.Raw, montage_path: Optional[str], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
    ch_locs = []
    good_chs, bad_chs = [], []

    for idx in eeg_picks:
        pos = raw.info['chs'][idx]['loc'][:3]
        if not all(abs(x) < 1e-8 for x in pos):
            good_chs.append(raw.ch_names[idx])
            ch_locs.append(pos)
        else:
            bad_chs.append(raw.ch_names[idx])

    status = {
        "n_eeg_channels": len(eeg_picks),
        "n_digitized_eeg": len(good_chs),
        "digitized_used": False,
        "montage_assigned": False,
        "eeg_channel_renaming": {},
        "dropped_channels": [],
        "log_message": None
    }

    if len(good_chs) > 0:
        if bad_chs:
            raw.drop_channels(bad_chs)
            if logger:
                logger.info(f"Dropped EEG channels with (0,0,0) location: {bad_chs}")
            status["dropped_channels"] = bad_chs
        if logger:
            logger.info(f"Retained EEG channels with digitized locations: {good_chs}")
        status["n_eeg_channels"] = len(good_chs)
        status["n_digitized_eeg"] = len(good_chs)
        status["digitized_used"] = True

        if montage_path:
            if montage_path.endswith(('.sfp', '.elc', '.csv')):
                montage = mne.channels.read_custom_montage(montage_path)
            else:
                montage = mne.channels.make_standard_montage(montage_path)
            eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
            raw_names = [raw.ch_names[i] for i in eeg_picks]
            montage_names = montage.ch_names
            n_rename = min(len(raw_names), len(montage_names))
            rename_map = {raw_names[i]: montage_names[i] for i in range(n_rename)}
            raw.rename_channels(rename_map)
            if logger:
                logger.info(f"Renamed EEG channels: {rename_map}")
            status["eeg_channel_renaming"] = rename_map

    elif montage_path:
        if montage_path.endswith(('.sfp', '.elc', '.csv')):
            montage = mne.channels.read_custom_montage(montage_path)
        else:
            montage = mne.channels.make_standard_montage(montage_path)
        raw.set_montage(montage, on_missing='warn')
        if logger:
            logger.info(f"Assigned montage {montage_path} to EEG channels (names and positions).")
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
        raw_names = [raw.ch_names[i] for i in eeg_picks]
        montage_names = montage.ch_names
        n_rename = min(len(raw_names), len(montage_names))
        rename_map = {raw_names[i]: montage_names[i] for i in range(n_rename)}
        raw.rename_channels(rename_map)
        if logger:
            logger.info(f"Renamed EEG channels: {rename_map}")
        status["eeg_channel_renaming"] = rename_map
        status["montage_assigned"] = True

    else:
        if eeg_picks:
            drop_names = [raw.ch_names[i] for i in eeg_picks]
            raw.drop_channels(drop_names)
            if logger:
                logger.warning("No EEG spatial info or montage provided — all EEG channels dropped.")
            status["dropped_channels"] = drop_names
            status["log_message"] = "No EEG spatial info or montage. All EEG channels dropped. Treated as no EEG recorded."
        status["n_eeg_channels"] = 0
        status["n_digitized_eeg"] = 0
        status["digitized_used"] = False
        status["montage_assigned"] = False
        status["eeg_channel_renaming"] = {}

    return status

def apply_metadata_repairs(raw: mne.io.Raw, metadata_cfg: dict) -> Dict[str, Any]:
    repairs = {"non_eeg_channel_repairs": {}, "generic_repairs": {}, "expert_patch_result": None}
    for old, fix in metadata_cfg.get('fix_non_eeg_channels', {}).items():
        if old not in raw.ch_names:
            continue
        new = fix.get('name', old)
        if new != old:
            raw.rename_channels({old: new})
        if 'type' in fix:
            raw.set_channel_types({new: fix['type']})
        repairs["non_eeg_channel_repairs"][old] = fix

    for key, val in metadata_cfg.get('fix_generic', {}).items():
        if key == 'meas_date':
            try:
                raw.info['meas_date'] = datetime.fromisoformat(val) if val else None
            except Exception as e:
                logger.warning(f"Bad meas_date format: {val} ({e})")
                continue
        else:
            raw.info[key] = val
        repairs["generic_repairs"][key] = val

    patch = metadata_cfg.get('expert_patch', None)
    if patch:
        logger.warning("Running expert_patch code block. Use with caution!")
        try:
            exec(patch, globals(), locals())
            repairs["expert_patch_result"] = "executed"
        except Exception as e:
            repairs["expert_patch_result"] = f"error: {e}"
            logger.error(f"Error in expert_patch: {e}")
    return repairs


def find_bad_channels_autoreject_by_type(
        raw,
        which_types=('eeg', 'mag', 'grad'),
        filter_settings=None,
        epoch_settings=None,
        consensus_thresh=0.3,
        global_epoch_thresh=0.3,
        interactive=False,
        logger=None
):
    """
    Detect bad channels (via AutoReject and RANSAC) and globally bad epochs
    on a filtered copy of the data, but transfer all results (bad channels and
    global bad epochs as Annotations) to the original, unfiltered raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Original unfiltered raw data. Will be modified in-place.
    which_types : tuple or list
        Channel types to check ('eeg', 'mag', 'grad').
    filter_settings : dict
        Filtering parameters (keys: highpass, lowpass, resample_hz).
    epoch_settings : dict
        Epoching parameters (keys: duration, tmin, tmax).
    consensus_thresh : float
        Fraction of epochs a channel must be bad to be flagged as bad.
    global_epoch_thresh : float
        Fraction of channels that must be bad in an epoch to mark it globally bad.
    interactive : bool
        If True, launch raw.plot for interactive review of bads and epochs.
    logger : logging.Logger or None
        Logger for progress/info output.

    Returns
    -------
    detected_bads : dict
        Dict containing bad channels and global bad epochs per type.
        (Also updates raw.info['bads'] and raw.annotations in-place.)
    """
    import numpy as np
    from collections import Counter
    import logging
    from mne.annotations import Annotations
    import mne

    if logger is None:
        logger = logging.getLogger("pipeline")

    filter_settings = filter_settings or {'highpass': 1.0, 'lowpass': 40.0}
    epoch_settings = epoch_settings or {'duration': 2.0, 'tmin': 0.0, 'tmax': 2.0}

    # Ensure numeric types
    l_freq = float(filter_settings.get('highpass', 1.0))
    h_freq = float(filter_settings.get('lowpass', 40.0))

    # Handle resampling properly - only resample if explicitly specified
    resample_hz = filter_settings.get('resample_hz')
    if resample_hz:
        resample_hz = float(resample_hz)

    duration = float(epoch_settings.get('duration', 2.0))
    tmin = float(epoch_settings.get('tmin', 0.0))
    tmax = float(epoch_settings.get('tmax', 2.0))

    consensus_thresh = float(consensus_thresh)
    global_epoch_thresh = float(global_epoch_thresh)

    detected_bads = {}
    all_bad_channels = set()
    all_global_bad_epochs = set()

    logger.info(f"Creating filtered copy for bad channel detection: {l_freq}-{h_freq} Hz")
    raw_filt_all = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        verbose='ERROR'
    )

    # Only resample if resample_hz is specified and not None
    if resample_hz:
        logger.info(f"Resampling for bad channel detection to {resample_hz} Hz")
        raw_filt_all.resample(resample_hz, npad="auto")

    # Run bad channel/global bad epoch detection per type
    for typ in which_types:
        logger.info(f"\n=== Running autoreject bad channel detection for {typ.upper()} ===")
        if typ == 'eeg':
            picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        elif typ == 'mag':
            picks = mne.pick_types(raw.info, meg='mag', eeg=False, exclude='bads')
        elif typ == 'grad':
            picks = mne.pick_types(raw.info, meg='grad', eeg=False, exclude='bads')
        else:
            logger.warning(f"Unknown channel type: {typ}, skipping.")
            detected_bads[typ] = []
            detected_bads[f"{typ}_global_bad_epochs"] = []
            continue
        ch_names = [raw.ch_names[i] for i in picks]
        if len(picks) == 0:
            logger.info(f"No {typ.upper()} channels found for autoreject.")
            detected_bads[typ] = []
            detected_bads[f"{typ}_global_bad_epochs"] = []
            continue
        logger.info(f"Channels analyzed ({typ}): {ch_names}")

        # Filtered copy: only relevant channels
        raw_filt = raw_filt_all.copy().pick_channels(ch_names)

        events = mne.make_fixed_length_events(raw_filt, duration=duration)
        epochs = mne.Epochs(
            raw_filt, events, event_id=None,
            tmin=tmin, tmax=tmax,
            picks=np.arange(len(ch_names)),
            baseline=None, detrend=0, preload=True, verbose=False,
            reject_by_annotation=True
        )
        logger.info(
            f"Created {len(epochs)} fixed-length epochs ({duration}s) for {typ.upper()}.")

        # RANSAC for EEG only
        bads_ransac = []
        if typ == 'eeg':
            from autoreject import Ransac
            logger.info("Running RANSAC for EEG bad channel screening...")
            ransac = Ransac(n_jobs=1, random_state=42)
            ransac.fit(epochs)
            bads_ransac = list(ransac.bad_chs_)
            logger.info(f"RANSAC bad EEG channels: {bads_ransac}")

        # Autoreject on good channels
        if typ == 'eeg':
            picks_good = [ch for ch in epochs.ch_names if ch not in bads_ransac]
        else:
            picks_good = epochs.ch_names
        if len(picks_good) < 2:
            logger.warning(f"Not enough good {typ.upper()} channels for autoreject after RANSAC.")
            bads_final = bads_ransac
            detected_bads[f"{typ}_global_bad_epochs"] = []
        else:
            from autoreject import AutoReject
            epochs_good = epochs.copy().pick_channels(picks_good)
            ar = AutoReject(n_interpolate=[0], consensus=None, n_jobs=1, random_state=42)
            ar.fit(epochs_good)
            reject_log = ar.get_reject_log(epochs_good)
            # Per-channel bads
            bad_labels = [lbl for labs in reject_log.labels for lbl in labs if isinstance(lbl, str)]
            n_epochs = len(reject_log.labels)
            bads_ar = {ch for ch, ct in Counter(bad_labels).items() if ct / n_epochs > consensus_thresh}
            logger.info(f"Autoreject (consensus>{consensus_thresh:.2f}) {typ.upper()} bads: {sorted(list(bads_ar))}")
            bads_final = list(set(bads_ransac) | bads_ar)

            # GLOBAL BAD EPOCHS
            reject_labels = np.array(reject_log.labels)  # shape: (n_epochs, n_channels)
            frac_bad_per_epoch = reject_labels.mean(axis=1)
            global_bad_epochs = np.where(frac_bad_per_epoch > global_epoch_thresh)[0].tolist()
            logger.info(f"Global bad epochs ({len(global_bad_epochs)}) for {typ.upper()}: {global_bad_epochs} "
                        f"(threshold={global_epoch_thresh:.2f})")
            detected_bads[f"{typ}_global_bad_epochs"] = global_bad_epochs
            all_global_bad_epochs.update(global_bad_epochs)

        bads_final = [ch for ch in bads_final if isinstance(ch, str)]
        detected_bads[typ] = bads_final
        all_bad_channels.update(bads_final)

    # ========================
    # Transfer to ORIGINAL RAW
    # ========================
    # Update bad channels
    prev_bads = set(raw.info.get('bads', []))
    new_bads = [ch for ch in all_bad_channels if ch not in prev_bads]
    raw.info['bads'] = sorted(prev_bads | set(all_bad_channels))
    logger.info(f"Updated raw.info['bads']: added {new_bads}, all bads: {raw.info['bads']}")

    # --- Remove old BAD_GlobalEpoch annotations from raw and filtered copy ---
    def remove_bad_epoch_anns(raw_obj):
        keep = [ann for ann in raw_obj.annotations if ann['description'] != "BAD_GlobalEpoch"]
        if len(keep) > 0:
            raw_obj.set_annotations(Annotations(
                onset=[ann['onset'] for ann in keep],
                duration=[ann['duration'] for ann in keep],
                description=[ann['description'] for ann in keep]
            ))
        else:
            raw_obj.set_annotations(Annotations([], [], []))

    remove_bad_epoch_anns(raw)
    remove_bad_epoch_anns(raw_filt_all)

    # --- Add new BAD_GlobalEpoch annotations ---
    anns = None
    if len(all_global_bad_epochs) > 0:
        # Time-based mapping: get actual time windows from filtered data, then map to original timeline
        filtered_epochs_times = []
        for idx in sorted(set(all_global_bad_epochs)):
            # Get actual time boundaries from the filtered data timeline
            start_time = raw_filt_all.times[0] + idx * duration
            end_time = start_time + duration
            filtered_epochs_times.append((start_time, end_time))

        # Map these time windows to original data (these times should align since filtering preserves timing)
        epoch_onsets = [start_time for start_time, end_time in filtered_epochs_times]
        durations_list = [end_time - start_time for start_time, end_time in filtered_epochs_times]
        descriptions = ["BAD_GlobalEpoch"] * len(epoch_onsets)
        anns = Annotations(onset=epoch_onsets, duration=durations_list, description=descriptions)
        if raw.annotations is not None and len(raw.annotations) > 0:
            raw.set_annotations(raw.annotations + anns)
        else:
            raw.set_annotations(anns)
        if raw_filt_all.annotations is not None and len(raw_filt_all.annotations) > 0:
            raw_filt_all.set_annotations(raw_filt_all.annotations + anns)
        else:
            raw_filt_all.set_annotations(anns)
        logger.info(f"Added {len(anns)} BAD_GlobalEpoch annotations to raw and raw_filt_all.")

    # --- Debug print before plotting ---
    print("Annotations in raw_filt_all before plot:")
    for ann in raw_filt_all.annotations:
        print(f"  {ann['onset']:.2f}-{ann['onset'] + ann['duration']:.2f}s: {ann['description']}")

    # ================================
    # INTERACTIVE REVIEW (Channels/Epochs)
    # ================================
    if interactive:
        logger.info(
            "Launching interactive viewer for review: bad channels and BAD_GlobalEpoch grayed out (FILTERED data).")
        raw_filt_all.info['bads'] = list(raw.info['bads'])  # Sync bads for GUI

        print("\nInteractive Bad Channel & Epoch Review:")
        print("- Auto-detected bad channels are grayed out")
        print("- BAD_GlobalEpoch epochs are grayed out (can be edited in the browser)")
        print("- Click channel names or annotation bars to toggle bad/good status")
        print("- Close the plot window when finished")
        print(f"Initial bad channels: {sorted(raw_filt_all.info['bads'])}")

        print("RAW_FILT_ALL time range: {:.2f} - {:.2f}".format(raw_filt_all.times[0], raw_filt_all.times[-1]))
        for ann in raw_filt_all.annotations:
            print(f"ANNOT: {ann['description']} onset={ann['onset']:.2f}s duration={ann['duration']:.2f}s")
        print("Bad channels before plot:", raw_filt_all.info['bads'])
        import sys
        sys.stdout.flush()

        fig = raw_filt_all.plot(
            n_channels=min(32, len(raw_filt_all.ch_names)),
            block=True,
            show=True
        )
        input("Review plot, then close the window and press Enter...")

        # --- Transfer user-reviewed bad channels to original raw ---
        final_bads = raw_filt_all.info['bads'].copy()
        raw.info['bads'] = final_bads

        # --- Transfer user-reviewed BAD_GlobalEpoch annotations ---
        gb_annots = [ann for ann in raw_filt_all.annotations if ann['description'] == "BAD_GlobalEpoch"]
        # Convert annotation onset times back to epoch indices for the original data timeline
        first_sample_time = raw.times[0]
        final_bad_epochs = [int(round((a['onset'] - first_sample_time) / duration)) for a in gb_annots]
        logger.info(f"After user review, final global bad epochs: {final_bad_epochs}")

        # Update both raw and detected_bads with final annotations
        remove_bad_epoch_anns(raw)
        if len(final_bad_epochs) > 0:
            # Use time-based mapping for final annotations too
            final_epochs_times = []
            for idx in final_bad_epochs:
                start_time = first_sample_time + idx * duration
                end_time = start_time + duration
                final_epochs_times.append((start_time, end_time))

            epoch_onsets = [start_time for start_time, end_time in final_epochs_times]
            durations_list = [end_time - start_time for start_time, end_time in final_epochs_times]
            descriptions = ["BAD_GlobalEpoch"] * len(epoch_onsets)
            anns = Annotations(onset=epoch_onsets, duration=durations_list, description=descriptions)
            if raw.annotations is not None and len(raw.annotations) > 0:
                # Create combined annotations manually to avoid orig_time issues
                combined_annotations = mne.Annotations(
                    onset=list(raw.annotations.onset) + list(anns.onset),
                    duration=list(raw.annotations.duration) + list(anns.duration),
                    description=list(raw.annotations.description) + list(anns.description),
                    orig_time=raw.annotations.orig_time
                )
                raw.set_annotations(combined_annotations)
            else:
                raw.set_annotations(anns)
            logger.info(f"User-reviewed BAD_GlobalEpoch annotations transferred to unfiltered raw.")

        for typ in which_types:
            detected_bads[f"{typ}_global_bad_epochs"] = final_bad_epochs

        logger.info(f"Final bad channels after user review: {sorted(final_bads)}")

    # =====================
    # CLEAN UP FILTERED COPY
    # =====================
    del raw_filt_all

    return detected_bads

def qc_meg_raw(raw: mne.io.Raw, plots_dir: str, hf_band: tuple = (250, 400)):
    data = raw.get_data()
    rms = np.sqrt((data ** 2).mean(axis=1))
    for typ, label in [("mag", "Magnetometers"), ("grad", "Gradiometers"), ("eeg", "EEG")]:
        picks = mne.pick_types(raw.info, meg=typ if "meg" in typ else False, eeg=(typ == 'eeg'), exclude='bads')
        if len(picks) == 0:
            continue
        vals = rms[picks]
        logger.info(f"{label}: {vals.mean():.2e} ± {vals.std():.2e} (n={len(vals)})")
        fig, ax = plt.subplots()
        ax.hist(vals, bins=40, alpha=0.7)
        ax.set_title(f"RMS {label}")
        fname = f"qc_rms_{label.lower().replace(' ', '_')}.png"
        save_plot(fig, plots_dir, fname)
        if "meg" in typ:
            psd = raw.compute_psd(picks=picks, fmin=hf_band[0], fmax=hf_band[1])
            hf_power = psd.get_data().mean(axis=1)
            logger.info(f"{label} HF noise: {hf_power.mean():.2e} ± {hf_power.std():.2e}")

def plot_psd_and_peaks(raw: mne.io.Raw, title: str, plots_dir: str, fmax: Optional[float] = None, n_peaks: int = 10):
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    if len(picks) == 0:
        return
    sf = raw.info['sfreq']
    fmax = fmax or sf / 2
    psd = raw.compute_psd(picks=picks, fmax=fmax, method='welch')
    data = psd.get_data().mean(axis=0)
    prominence = np.percentile(data, 95) / 5
    peaks, props = find_peaks(data, prominence=prominence)
    top = np.argsort(props['prominences'])[::-1][:n_peaks]
    fig = psd.plot(show=False)
    ax = fig.axes[0]
    ax.scatter(psd.freqs[peaks][top], 10 * np.log10(data[peaks][top] + 1e-25), zorder=5, label='peaks')
    ax.set_title(title)
    ax.legend()
    fname = f"psd_peaks_{title.replace(' ', '_').lower()}.png"
    save_plot(fig, plots_dir, fname)

def plot_head_movement(head_pos_array, plots_dir: str, fname="head_movement_over_time.png"):
    """
    Plot translation (mm) and rotation (deg) over time from head_pos array.
    head_pos_array: N x 10 array as returned by mne.chpi.read_head_pos()
    """
    if head_pos_array is None or head_pos_array.shape[0] == 0:
        logger.warning("No head position data found to plot head movement.")
        return

    t = head_pos_array[:, 0]  # time in seconds
    x, y, z = head_pos_array[:, 1:4].T * 1000  # translation in mm
    rot = head_pos_array[:, 4:7] * (180 / np.pi)  # rotation in degrees

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, x, label='X')
    axes[0].plot(t, y, label='Y')
    axes[0].plot(t, z, label='Z')
    axes[0].set_ylabel('Translation (mm)')
    axes[0].legend()
    axes[0].set_title('Head Translation Over Time')

    axes[1].plot(t, rot[:, 0], label='Yaw')
    axes[1].plot(t, rot[:, 1], label='Pitch')
    axes[1].plot(t, rot[:, 2], label='Roll')
    axes[1].set_ylabel('Rotation (deg)')
    axes[1].legend()
    axes[1].set_title('Head Rotation Over Time')
    axes[1].set_xlabel('Time (s)')

    fig.tight_layout()
    save_plot(fig, plots_dir, fname)

def run_ica(raw: mne.io.Raw, config: dict, output_path: Path, modality: str) -> tuple:
    """
    Run ICA on specified modality (EEG or MEG).

    Args:
        raw: MNE Raw object
        config: ICA configuration dictionary
        output_path: Path to save ICA solution
        modality: Either 'eeg' or 'meg'

    Returns:
        tuple: (cleaned_raw, excluded_components)
    """
    picks = mne.pick_types(
        raw.info,
        meg=(modality == 'meg'),
        eeg=(modality == 'eeg'),
        exclude='bads'
    )
    if len(picks) == 0:
        logger.info(f"No {modality.upper()} channels for ICA.")
        return raw, []

    logger.info(f"Running ICA on {modality.upper()} channels")

    # Create copy for ICA preprocessing
    raw_ica = raw.copy()

    # Apply filtering - always filter for ICA with specified range
    l_freq = config.get('highpass', None)
    h_freq = config.get('lowpass', None)

    # Always apply the filter if any frequency is specified
    if l_freq is not None or h_freq is not None:
        logger.info(f"About to filter {modality.upper()}: {l_freq}-{h_freq} Hz")
        raw_ica.filter(l_freq, h_freq, picks=picks, fir_design='firwin', verbose='error')
        logger.info(f"Filtered {modality.upper()} for ICA: highpass={l_freq}, lowpass={h_freq}")
    else:
        logger.warning(f"No filtering specified for {modality.upper()} ICA - MNE recommends 1 Hz highpass!")

    # Resample if specified
    hz = config.get('resample_hz', None)
    if hz:
        hz = float(hz)  # Ensure numeric type
        raw_ica.resample(hz, npad="auto")
        logger.info(f"Resampled {modality.upper()} for ICA to {hz} Hz")

    # Handle duration cropping
    max_sec = config.get("max_ica_duration_sec", None)
    full_duration_sec = raw_ica.times[-1]
    if max_sec is not None:
        max_sec = float(max_sec)
        if full_duration_sec > max_sec:
            raw_ica.crop(0, max_sec)
            logger.info(f"ICA input cropped to first {max_sec} seconds (of {full_duration_sec:.1f} sec total).")
        else:
            logger.info(f"ICA using full available duration ({full_duration_sec:.1f} sec), shorter than crop limit.")
    else:
        logger.info(f"ICA using full recording duration: {full_duration_sec:.1f} seconds.")

    # Create and fit ICA - MNE automatically handles rank after Maxwell filtering
    ica = ICA(n_components=min(20, len(picks)), method='fastica', random_state=97)
    ica.fit(raw_ica, picks=picks)  # No rank parameter needed

    # Interactive component review
    interactive = config.get("interactive", True)
    if interactive:
        try:
            #ica.plot_components(inst=raw_ica)
            ica.plot_sources(raw_ica)
            input(f"Review {modality.upper()} ICA components. Press Enter to continue...")
        except Exception as e:
            logger.warning(f"Interactive ICA plots failed: {e}")

    # Apply ICA to original data
    exclude = list(ica.exclude)
    raw_clean = ica.apply(raw.copy())

    # Save ICA solution
    output_dir = os.path.dirname(str(output_path))
    os.makedirs(output_dir, exist_ok=True)
    ica.save(str(output_path), overwrite=True)

    logger.info(f"Excluded {modality.upper()} ICA components: {exclude}")
    return raw_clean, exclude

def apply_maxwell_filter(raw: mne.io.Raw, head_pos=None, destination=None, cal=None, crosstalk=None) -> mne.io.Raw:
    logger.info("Applying Maxwell filter (tSSS)...")
    return mne.preprocessing.maxwell_filter(raw, head_pos=head_pos, destination=destination, calibration=cal, cross_talk=crosstalk, st_duration=10.0, coord_frame='head', verbose='error')

def bitwise_events(raw: mne.io.Raw, mask: int = 0xFFFF, min_high: int = 2, min_off: int = 5) -> np.ndarray:
    ST_CH = 'STI101'
    pick = mne.pick_channels(raw.ch_names, [ST_CH])
    if not pick: return np.empty((0, 3), int)
    stim = raw.get_data(picks=pick)[0].astype(np.uint16) & mask
    events = []
    for bit in range(16):
        code = 1 << bit
        if mask & code == 0: continue
        vec = (stim & code) != 0
        onsets = np.flatnonzero(vec)
        if not len(onsets): continue
        runs = np.split(onsets, np.where(np.diff(onsets) > 1)[0] + 1)
        last_rise = -min_off - 1
        for r in runs:
            if len(r) >= min_high and r[0] - last_rise >= min_off:
                events.append([r[0], 0, code])
                last_rise = r[0]
    return np.array(sorted(events), dtype=int)

def apply_final_filter_and_cleanup(raw: mne.io.Raw, config: dict) -> mne.io.Raw:
    """
    Applies optional highpass/lowpass filtering, resampling, and
    drops specified channel types or name patterns before saving.

    Controlled entirely by the 'final_filter' section of the YAML config.
    """
    final_cfg = config.get("final_filter", {})
    raw_proc = raw.copy()

    # 1. Drop unwanted channels by prefix (e.g., HPI*, HLC*, EEG061)
    drop_prefixes = final_cfg.get("drop_channel_types", [])
    drop_chs = [
        ch for ch in raw_proc.ch_names
        if any(ch.upper().startswith(prefix.upper()) for prefix in drop_prefixes)
    ]
    if drop_chs:
        raw_proc.drop_channels(drop_chs)
        logger.info(f"Dropped channels before save: {drop_chs}")

    # 2. Determine signal picks for filtering/resampling
    picks = mne.pick_types(
        raw_proc.info,
        meg=True, eeg=True, eog=True, ecg=False, stim=False, misc=False,
        exclude='bads'
    )

    # 3. Apply filtering if requested
    hp = final_cfg.get("highpass", None)
    lp = final_cfg.get("lowpass", None)
    if hp is not None or lp is not None:
        raw_proc.filter(hp, lp, picks=picks, fir_design='firwin', verbose='error')
        logger.info(f"Applied final filter: highpass={hp}, lowpass={lp}")

    # 4. Apply resampling if requested
    resample_hz = final_cfg.get("resample_hz", None)
    if resample_hz is not None:
        raw_proc.resample(resample_hz, npad="auto")
        logger.info(f"Resampled data to {resample_hz} Hz before save")

    return raw_proc
