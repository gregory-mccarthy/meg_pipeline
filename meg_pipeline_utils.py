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
from collections import Counter
from scipy.stats import kurtosis

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

def check_critical_files_exist(cfg, repo_root):
    checked_paths = {}

    # Handle montage specially - allow None
    eeg_handling = cfg.get("eeg_handling", {})
    montage = eeg_handling.get("montage") if eeg_handling else None
    if montage is not None:
        checked_paths["montage"] = get_and_check_path(eeg_handling, "montage", repo_root, "montage file")
    else:
        checked_paths["montage"] = None

    # Handle required files normally
    checked_paths["calibration_file"] = get_and_check_path(cfg, "calibration_file", repo_root, "MEG calibration file")
    checked_paths["cross_talk_file"] = get_and_check_path(cfg, "cross_talk_file", repo_root, "MEG crosstalk file")

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


# ========== CLEAN SPLIT: Two separate AutoReject functions ==========

def run_autoreject_stage1(raw, cfg, bids_path, logger):
    """
    Stage 1: Run AutoReject headlessly (non-interactive) and save checkpoint.

    This function:
    - Runs RANSAC + AutoReject on filtered/downsampled copy
    - Updates bad channels and annotations on original raw object
    - Saves checkpoint FIF file
    - Returns metadata for manifest writing

    Args:
        raw: MNE Raw object (will be modified in-place)
        cfg: Configuration dictionary
        bids_path: BIDS path object
        logger: Logger instance

    Returns:
        tuple: (checkpoint_path, stage1_metadata)
    """
    from pathlib import Path

    # Get configuration
    chk_cfg = cfg.get("checkpoint", {})
    ar_cfg = cfg.get("autoreject", {})

    if not chk_cfg.get("enabled", True):
        logger.info("[AR Stage 1] Checkpointing disabled - skipping AutoReject")
        return (None, {"autoreject_skipped": True})

    if not ar_cfg.get("enabled", True):
        logger.info("[AR Stage 1] AutoReject disabled - will save checkpoint without AR processing")

    # Build checkpoint path
    bids_root = Path(cfg.get("bids_root", bids_path.root))
    derivatives_root = chk_cfg.get("derivatives_root", "derivatives")
    pipeline_name = chk_cfg.get("pipeline_name", "preprocessing")
    deriv_root = bids_root / derivatives_root / pipeline_name

    subject_dir = f"sub-{bids_path.subject}"
    if bids_path.session:
        session_dir = f"ses-{bids_path.session}"
        parproc_dir = deriv_root / subject_dir / session_dir / "meg"
    else:
        parproc_dir = deriv_root / subject_dir / "meg"
    parproc_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"sub-{bids_path.subject}"
    if bids_path.session:
        base_name += f"_ses-{bids_path.session}"
    if bids_path.task:
        base_name += f"_task-{bids_path.task}"
    if bids_path.run:
        base_name += f"_run-{bids_path.run}"

    checkpoint_path = parproc_dir / f"{base_name}_desc-parproc_meg.fif"

    # Run AutoReject if enabled
    ar_metadata = {}
    if ar_cfg.get("enabled", True):
        logger.info("[AR Stage 1] Running non-interactive AutoReject...")

        # Get AR parameters
        ar_types = ar_cfg.get("which_types", ['eeg', 'mag', 'grad'])
        fset = ar_cfg.get("filter", {'highpass': 1.0, 'lowpass': 40.0, 'resample_hz': None})
        eset = ar_cfg.get("epoch", {'duration': 2.0, 'tmin': 0.0, 'tmax': 2.0})
        consensus_thresh = ar_cfg.get("consensus_thresh", 0.3)
        global_epoch_thresh = ar_cfg.get("global_epoch_thresh", 0.3)

        # Run AutoReject (modifies raw in-place)
        ar_results = find_bad_channels_autoreject_by_type(
            raw,
            which_types=ar_types,
            filter_settings=fset,
            epoch_settings=eset,
            consensus_thresh=consensus_thresh,
            global_epoch_thresh=global_epoch_thresh,
            interactive=False,
            logger=logger,
            cfg=cfg,
        )

        ar_metadata = {
            "autoreject_enabled": True,
            "autoreject_results": ar_results,
            "parameters": {
                "which_types": ar_types,
                "filter_settings": fset,
                "epoch_settings": eset,
                "consensus_thresh": consensus_thresh,
                "global_epoch_thresh": global_epoch_thresh
            }
        }

        logger.info(f"[AR Stage 1] ✅ AutoReject complete. Bad channels: {raw.info.get('bads', [])}")
    else:
        logger.info("[AR Stage 1] AutoReject disabled - proceeding without bad channel detection")
        ar_metadata = {"autoreject_enabled": False}

    # Save checkpoint
    try:
        atomic_writes = chk_cfg.get("atomic_writes", True)
        if atomic_writes:
            tmp_path = checkpoint_path.with_name(checkpoint_path.stem + "_tmp.fif")
            raw.save(str(tmp_path), overwrite=True)
            tmp_path.replace(checkpoint_path)
        else:
            raw.save(str(checkpoint_path), overwrite=True)

        logger.info(f"[AR Stage 1] ✅ Checkpoint saved: {checkpoint_path}")

    except Exception as e:
        logger.error(f"[AR Stage 1] Failed to save checkpoint: {e}")
        raise

    # Prepare metadata for manifest
    stage1_metadata = {
        "checkpoint_path": str(checkpoint_path),
        "bads_detected": list(raw.info.get('bads', [])),
        "n_bad_channels": len(raw.info.get('bads', [])),
        "n_annotations": int(len(raw.annotations) if raw.annotations is not None else 0),
        **ar_metadata
    }

    return (checkpoint_path, stage1_metadata)


def run_interactive_review_stage2(checkpoint_path, cfg, logger):
    """
    Stage 2: Load checkpoint and run interactive bad channel review.

    This function:
    - Loads checkpoint FIF file
    - Creates filtered/downsampled copy for interactive viewing
    - Runs interactive plotting for bad channel review
    - Updates and saves checkpoint with user changes
    - Returns updated raw object and metadata

    Args:
        checkpoint_path: Path to checkpoint FIF file
        cfg: Configuration dictionary
        logger: Logger instance

    Returns:
        tuple: (updated_raw, stage2_metadata)
    """
    import mne
    from pathlib import Path

    # Load checkpoint
    logger.info(f"[AR Stage 2] Loading checkpoint: {checkpoint_path}")
    try:
        raw = mne.io.read_raw_fif(str(checkpoint_path), preload=True, verbose="error")
        logger.info(f"[AR Stage 2] ✅ Loaded: {len(raw.ch_names)} channels, "
                    f"{raw.times[-1]:.1f}s, {len(raw.info.get('bads', []))} bad channels")
    except Exception as e:
        logger.error(f"[AR Stage 2] Failed to load checkpoint: {e}")
        raise

    # Check if interactive review is enabled
    if not cfg.get("interactive_bad_channels", True):
        logger.info("[AR Stage 2] Interactive review disabled - returning checkpoint as-is")
        stage2_metadata = {
            "interactive_review_enabled": False,
            "bads_after_review": list(raw.info.get('bads', [])),
            "n_bad_channels_final": len(raw.info.get('bads', [])),
            "n_annotations_final": int(len(raw.annotations) if raw.annotations is not None else 0)
        }
        return (raw, stage2_metadata)

    # Get filter settings for interactive viewing (same as Stage 1 AR)
    ar_cfg = cfg.get("autoreject", {})
    fset = ar_cfg.get("filter", {'highpass': 1.0, 'lowpass': 40.0, 'resample_hz': None})
    l_freq = float(fset.get('highpass', 1.0))
    h_freq = float(fset.get('lowpass', 40.0))
    resample_hz = fset.get('resample_hz')

    # Create filtered copy for interactive viewing
    logger.info(f"[AR Stage 2] Creating filtered copy for interactive review: {l_freq}-{h_freq} Hz")
    try:
        raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose='ERROR')
        if resample_hz:
            resample_hz = float(resample_hz)
            raw_filtered.resample(resample_hz, npad="auto")
            logger.info(f"[AR Stage 2] Resampled to {resample_hz} Hz for viewing")
    except Exception as e:
        logger.error(f"[AR Stage 2] Failed to create filtered copy: {e}")
        # Fall back to unfiltered if filtering fails
        raw_filtered = raw.copy()

    # Store original state for comparison
    original_bads = set(raw.info.get('bads', []))
    original_n_annotations = len(raw.annotations) if raw.annotations is not None else 0

    # Run interactive review
    logger.info("[AR Stage 2] 🎮 Opening interactive review window...")
    logger.info("Instructions:")
    logger.info("  - Click channel names to mark/unmark as bad")
    logger.info("  - Click time segments to mark/unmark annotations")
    logger.info("  - Use mouse wheel to zoom")
    logger.info("  - Close window when done")

    try:
        n_channels = min(32, len(raw_filtered.ch_names))
        raw_filtered.plot(
            n_channels=n_channels,
            duration=30.0,
            scalings='auto',
            show=True,
            block=True,
            title=f"Stage 2 Review - {len(raw_filtered.info['bads'])} bad channels"
        )

        print("\n" + "=" * 60)
        print("🎯 INTERACTIVE REVIEW COMPLETE")
        print("   - Close the plot window if you haven't already")
        print("   - Changes will be saved to checkpoint")
        print("=" * 60)

        response = input("Press Enter to save changes (or 'abort' to cancel): ").strip().lower()
        if response == 'abort':
            logger.info("[AR Stage 2] ❌ User aborted - keeping original state")
            del raw_filtered
            stage2_metadata = {
                "interactive_review_enabled": True,
                "user_aborted": True,
                "bads_after_review": list(original_bads),
                "n_bad_channels_final": len(original_bads),
                "n_annotations_final": original_n_annotations
            }
            return (raw, stage2_metadata)

    except Exception as e:
        logger.error(f"[AR Stage 2] Interactive plotting failed: {e}")
        logger.info("[AR Stage 2] Continuing with original checkpoint state")
        del raw_filtered
        stage2_metadata = {
            "interactive_review_enabled": True,
            "plotting_failed": True,
            "error": str(e),
            "bads_after_review": list(original_bads),
            "n_bad_channels_final": len(original_bads),
            "n_annotations_final": original_n_annotations
        }
        return (raw, stage2_metadata)

    # Transfer changes from filtered copy back to original raw
    final_bads = set(raw_filtered.info.get('bads', []))
    added_bads = final_bads - original_bads
    removed_bads = original_bads - final_bads

    if added_bads:
        logger.info(f"[AR Stage 2] ✅ User marked as bad: {sorted(added_bads)}")
    if removed_bads:
        logger.info(f"[AR Stage 2] 🔄 User rescued channels: {sorted(removed_bads)}")
    if not added_bads and not removed_bads:
        logger.info("[AR Stage 2] No bad channel changes made")

    # Update original raw with user changes
    raw.info['bads'] = sorted(final_bads)

    # Transfer annotation changes
    if raw_filtered.annotations is not None and len(raw_filtered.annotations) > 0:
        # Note: Annotations need to be scaled back if resampling was applied
        if resample_hz and resample_hz != raw.info['sfreq']:
            scale_factor = raw.info['sfreq'] / resample_hz
            scaled_onsets = [onset * scale_factor for onset in raw_filtered.annotations.onset]
            scaled_durations = [dur * scale_factor for dur in raw_filtered.annotations.duration]

            from mne import Annotations
            scaled_annotations = Annotations(
                onset=scaled_onsets,
                duration=scaled_durations,
                description=list(raw_filtered.annotations.description),
                orig_time=raw.annotations.orig_time if raw.annotations else None
            )
            raw.set_annotations(scaled_annotations)
        else:
            raw.set_annotations(raw_filtered.annotations)

        final_n_annotations = len(raw.annotations)
        annotation_change = final_n_annotations - original_n_annotations
        if annotation_change != 0:
            logger.info(f"[AR Stage 2] Annotation changes: {annotation_change:+d} "
                        f"(total: {final_n_annotations})")
    else:
        final_n_annotations = 0
        if original_n_annotations > 0:
            logger.info(f"[AR Stage 2] All annotations removed")

    # Clean up filtered copy
    del raw_filtered

    # Save updated checkpoint
    chk_cfg = cfg.get("checkpoint", {})
    try:
        atomic_writes = chk_cfg.get("atomic_writes", True)
        if atomic_writes:
            tmp_path = checkpoint_path.with_name(checkpoint_path.stem + "_tmp.fif")
            raw.save(str(tmp_path), overwrite=True)
            tmp_path.replace(checkpoint_path)
        else:
            raw.save(str(checkpoint_path), overwrite=True)

        logger.info(f"[AR Stage 2] ✅ Updated checkpoint saved")

    except Exception as e:
        logger.error(f"[AR Stage 2] Failed to save updated checkpoint: {e}")
        # Continue anyway - we have the updated raw object

    # Prepare Stage 2 metadata
    stage2_metadata = {
        "interactive_review_enabled": True,
        "user_aborted": False,
        "plotting_failed": False,
        "changes_made": {
            "bad_channels_added": sorted(added_bads),
            "bad_channels_removed": sorted(removed_bads),
            "annotation_change": final_n_annotations - original_n_annotations
        },
        "bads_after_review": list(raw.info.get('bads', [])),
        "n_bad_channels_final": len(raw.info.get('bads', [])),
        "n_annotations_final": final_n_annotations,
        "filter_settings_used": fset
    }

    logger.info("[AR Stage 2] 🎉 Interactive review completed successfully!")
    return (raw, stage2_metadata)

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

def load_raw_data(cfg: dict) -> mne.io.Raw:  # CHANGED: config -> cfg
    bids_path = BIDSPath(
        subject=cfg['subject'],           # CHANGED: config -> cfg
        session=cfg.get('session'),       # CHANGED: config -> cfg
        task=cfg.get('task'),             # CHANGED: config -> cfg
        run=cfg.get('run'),               # CHANGED: config -> cfg
        root=cfg['bids_root'],            # CHANGED: config -> cfg
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
        if len(eeg_picks) > 0:
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

#----------- Writing the manifest ---------------

def write_stage1_manifest(raw, cfg, bids_path, checkpoint_file=None,  # CHANGED: config -> cfg
                         autoreject_results=None, quality_metrics=None,
                         head_movement_stats=None, rank_info=None,
                         eeg_setup_results=None, metadata_repair_results=None,
                         original_recording_info=None, notch_filter_params=None,
                         processing_paths=None, logger=None):
    """
    Write comprehensive Stage 1 manifest file with all processing results and metadata.
    This manifest serves as the complete log of Stage 1 processing and provides
    all necessary information for Stage 2 continuation.

    Args:
        raw: MNE Raw object (after all Stage 1 processing)
        config: Pipeline configuration dictionary
        bids_path: BIDS path object
        checkpoint_file: Path to checkpoint FIF file
        autoreject_results: Dictionary of AutoReject results
        quality_metrics: Dictionary of quality metrics (pre/post Maxwell, improvements)
        head_movement_stats: Head movement analysis results
        rank_info: MEG rank estimation results
        eeg_setup_results: EEG channel setup results
        metadata_repair_results: Metadata repair results
        original_recording_info: Original recording metadata
        notch_filter_params: Notch filtering parameters
        processing_paths: Processing paths (plots directory, etc.)
        logger: Logger instance

    Returns:
        Path to manifest file
    """
    if logger is None:
        logger = logging.getLogger("pipeline")

    # Get checkpoint configuration
    chk_cfg = cfg.get("checkpoint", {})
    derivatives_root = chk_cfg.get("derivatives_root", "derivatives")
    pipeline_name = chk_cfg.get("pipeline_name", "preprocessing")

    # Build paths (same as in run_autoreject_with_checkpoint)
    bids_root = Path(cfg.get("bids_root", bids_path.root))
    deriv_root = bids_root / derivatives_root / pipeline_name

    # Build proper subject/session directory structure
    subject_dir = f"sub-{bids_path.subject}"
    if bids_path.session:
        session_dir = f"ses-{bids_path.session}"
        parproc_dir = deriv_root / subject_dir / session_dir / "meg"
    else:
        parproc_dir = deriv_root / subject_dir / "meg"

    # Build checkpoint filename
    base_name = f"sub-{bids_path.subject}"
    if bids_path.session:
        base_name += f"_ses-{bids_path.session}"
    if bids_path.task:
        base_name += f"_task-{bids_path.task}"
    if bids_path.run:
        base_name += f"_run-{bids_path.run}"

    parproc_fif = parproc_dir / f"{base_name}_desc-parproc_meg.fif"
    manifest_path = parproc_fif.with_name(parproc_fif.stem + "_manifest.yaml")

    # Read YAML hash for provenance tracking
    yaml_text = None
    yaml_hash = None
    try:
        yaml_path_local = cfg.get("_cfg_path", "")
        if yaml_path_local and Path(yaml_path_local).exists():
            yaml_text = Path(yaml_path_local).read_text()
            import hashlib
            yaml_hash = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.warning(f"Could not read YAML for manifest: {e}")

    # Get current system info for Stage 1
    stage1_system_info = get_runtime_info()
    stage1_system_info["stage"] = "stage1"
    stage1_system_info["completed_utc"] = datetime.utcnow().isoformat() + "Z"

    # Build comprehensive manifest
    manifest = {
        # Header information
        "manifest_version": "2.0",  # Updated version to indicate enhanced format
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "stage": "stage1_complete",
        "pipeline_version": cfg.get("pipeline_version", "1.2"),
        "checkpoint_version": cfg.get("_checkpoint_version", "1.0"),

        # System information - Stage 1
        "system_info": {
            "stage1": stage1_system_info,
            "stage2": None  # Will be filled by Stage 2
        },

        # Input configuration and provenance
        "inputs": {
            "bids": {
                "subject": bids_path.subject,
                "session": bids_path.session,
                "task": bids_path.task,
                "run": bids_path.run,
            },
            "bids_root": str(bids_root),
            "yaml_path": cfg.get("_cfg_path", None),
            "yaml_sha256": yaml_hash,
            "original_recording": original_recording_info or {}
        },

        # Artifacts and outputs from Stage 1
        "artifacts": {
            "parproc_raw_fif": checkpoint_file or str(parproc_fif),
            "manifest_path": str(manifest_path),
            "plots_directory": processing_paths.get("plots_directory") if processing_paths else None
        },

        # Configuration parameters used in Stage 1
        "parameters": {
            "autoreject": cfg.get("autoreject", {}),
            "checkpoint": chk_cfg,
            "head_movement": cfg.get("head_movement", {}),
            "maxwell_filter": {
                "calibration_file": cfg.get("calibration_file"),
                "cross_talk_file": cfg.get("cross_talk_file"),
            },
            "notch_filter": notch_filter_params or {},
            "line_freq": cfg.get("line_freq", 60.0),
            "interactive_bad_channels": cfg.get("interactive_bad_channels", True),
            "metadata_fixes": cfg.get("metadata_fixes", {}),
            "eeg_handling": cfg.get("eeg_handling", {})
        },

        # Processing results from each stage
        "processing_results": {
            # EEG setup results
            "eeg_setup": eeg_setup_results or {},

            # Metadata repairs
            "metadata_repairs": metadata_repair_results or {},

            # Head movement analysis
            "head_movement": head_movement_stats or {"enabled": False},

            # Rank estimation
            "rank_estimation": rank_info or {},

            # AutoReject results
            "autoreject": {
                "bads_detected": list(raw.info.get('bads', [])),
                "n_bad_channels": len(raw.info.get('bads', [])),
                "n_annotations": int(len(raw.annotations) if raw.annotations is not None else 0),
                **(autoreject_results or {})
            }
        },

        # Quality metrics (comprehensive)
        "quality_metrics": quality_metrics or {},

        # Final state information
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

        # Stage 1 completion flag
        "stage1_complete": True,
        "exit_after_checkpoint": chk_cfg.get("exit_after_checkpoint", False)
    }

    # Save manifest with proper serialization
    try:
        save_yaml(str(manifest_path), make_serializable(manifest))
        logger.info(f"[manifest] ✅ Stage 1 manifest saved: {manifest_path}")

        # Log key metrics for immediate feedback
        if quality_metrics and 'maxwell_improvements' in quality_metrics:
            score_info = quality_metrics['maxwell_improvements'].get('overall_quality_score', {})
            logger.info(f"[manifest] Quality Score: {score_info.get('value', 'N/A')}/100 "
                        f"(Grade: {score_info.get('grade', 'N/A')})")

        if head_movement_stats and 'translation_stats_mm' in head_movement_stats:
            max_disp = head_movement_stats['translation_stats_mm']['max_displacement']
            logger.info(f"[manifest] Max head movement: {max_disp:.1f} mm")

        if rank_info and rank_info.get('meg_rank'):
            logger.info(f"[manifest] MEG rank: {rank_info['meg_rank']}")

        logger.info(f"[manifest] Bad channels: {len(manifest['final_state']['bad_channels'])}")

    except Exception as e:
        logger.error(f"Failed to write Stage 1 manifest: {e}")
        raise

    return manifest_path

# ========== QUALITY METRICS FUNCTIONS ==========
# Add these functions to your meg_pipeline_utils.py file
# Required additional import: from scipy.stats import kurtosis

def compute_meg_quality_metrics(raw: mne.io.Raw, stage_name: str = "raw") -> Dict[str, Any]:
    """
    Compute MEG/EEG quality metrics (SI units) for a single pipeline stage.

    Returns SI units for machine-readability:
      mag=T, grad=T/m, eeg=V. Use separate logging helpers to print fT, fT/cm, µV.
    """
    metrics: Dict[str, Any] = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
    }

    # ---- Recording metadata ----
    try:
        duration_sec = float(raw.n_times) / float(raw.info["sfreq"])
    except Exception:
        duration_sec = float(raw.times[-1]) if len(raw.times) else None

    all_ch_names = list(raw.ch_names)
    bads = list(raw.info.get("bads", [])) if "bads" in raw.info else []

    metrics["recording"] = {
        "duration_sec": duration_sec,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels_total": int(len(all_ch_names)),
        "bad_channels": bads,
        "n_bad": int(len(bads)),
        "n_annotations": int(len(getattr(raw, "annotations", []) or [])),
    }

    # ---- Channel picks ----
    def _pick(raw, meg=None, eeg=False):
        return mne.pick_types(raw.info, meg=meg, eeg=eeg, stim=False, eog=False, ecg=False, seeg=False, misc=False, exclude=[])

    channel_types = {
        "mag":  dict(picks=lambda r: _pick(r, meg='mag')),
        "grad": dict(picks=lambda r: _pick(r, meg='grad')),
        "eeg":  dict(picks=lambda r: _pick(r, meg=False, eeg=True)),
    }

    line_freq = float(raw.info.get("line_freq", 60.0) or 60.0)
    nyq = float(raw.info["sfreq"]) / 2.0

    # ---- Helpers ----
    def _amplitude_stats(data: np.ndarray) -> Dict[str, Any]:
        """RMS/variance on demeaned data; P2P 95% on raw channel P2P (SI)."""
        if data.size == 0:
            return {"rms": None, "peak_to_peak_95pct": None, "variance": None, "n_channels": 0}
        data_demean = data - data.mean(axis=1, keepdims=True)
        rms = float(np.sqrt((data_demean ** 2).mean()))
        ptp95 = float(np.percentile(np.ptp(data, axis=1), 95))
        var = float(data_demean.var())
        return {"rms": rms, "peak_to_peak_95pct": ptp95, "variance": var, "n_channels": int(data.shape[0])}

    def _band_power(raw, picks, fmin, fmax):
        """Mean PSD power over [fmin,fmax] with Nyquist guard (power units)."""
        if fmin >= nyq:
            return None
        try:
            fmax_use = min(fmax, nyq - 1e-6)
            psd = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax_use, verbose='ERROR')
            return float(psd.get_data().mean())
        except Exception:
            return None

    def _one_over_f(raw, picks):
        """1/f slope & intercept on log10 PSD, 1–45 Hz, masking 8–13 Hz (alpha)."""
        fmin, fmax = 1.0, min(45.0, nyq - 1e-6)
        try:
            psd = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose='ERROR')
            freqs = psd.freqs
            data = psd.get_data()
            if data.size == 0:
                return {"slope": None, "intercept": None}
            p = data.mean(axis=0)
            mask = (freqs >= fmin) & (freqs <= fmax) & ~((freqs >= 8.0) & (freqs <= 13.0))
            x = np.log10(freqs[mask])
            y = np.log10(p[mask] + np.finfo(float).tiny)
            A = np.vstack([x, np.ones_like(x)]).T
            sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return {"slope": float(sol[0]), "intercept": float(sol[1])}
        except Exception:
            return {"slope": None, "intercept": None}

    def _line_harmonics(raw, picks, line_freq):
        """Peak power at 1x/2x/3x line freq and peak/sideband ratio (exclude center)."""
        out = {}
        for harm in (1, 2, 3):
            f0 = line_freq * harm
            if f0 >= nyq or f0 <= 0:
                out[f"line_{harm}x_peak"] = None
                out[f"line_{harm}x_ratio"] = None
                continue
            try:
                psd_narrow = raw.compute_psd(picks=picks, fmin=f0 - 0.5, fmax=f0 + 0.5, verbose='ERROR')
                idx0 = np.argmin(np.abs(psd_narrow.freqs - f0))
                peak_power = float(psd_narrow.get_data()[:, idx0].mean())

                side_low  = raw.compute_psd(picks=picks, fmin=max(f0 - 3.0, 0.1), fmax=max(f0 - 1.0, 0.2), verbose='ERROR')
                side_high = raw.compute_psd(picks=picks, fmin=f0 + 1.0,            fmax=min(f0 + 3.0, nyq - 1e-6), verbose='ERROR')
                side_mean = np.mean([side_low.get_data().mean(), side_high.get_data().mean()])
                ratio = float(peak_power / side_mean) if side_mean > 0 else None

                out[f"line_{harm}x_peak"] = peak_power
                out[f"line_{harm}x_ratio"] = ratio
            except Exception:
                out[f"line_{harm}x_peak"] = None
                out[f"line_{harm}x_ratio"] = None
        return out

    # ---- Per-type metrics ----
    for ch_type, cfg in channel_types.items():
        picks = cfg["picks"](raw)
        if picks is None or len(picks) == 0:
            continue

        try:
            data = raw.get_data(picks=picks)
        except Exception:
            data = np.empty((0, 0))

        metrics[f"{ch_type}_amplitude"] = _amplitude_stats(data)

        # Kurtosis (dimensionless)
        try:
            k = kurtosis(data, axis=1, fisher=False, bias=False, nan_policy='omit') if data.size else np.array([])
            metrics[f"{ch_type}_kurtosis"] = {
                "mean": float(np.nanmean(k)) if k.size else None,
                "max":  float(np.nanmax(k))  if k.size else None,
                "std":  float(np.nanstd(k))  if k.size else None,
            }
        except Exception:
            metrics[f"{ch_type}_kurtosis"] = {"mean": None, "max": None, "std": None}

        spectral: Dict[str, Any] = {}
        # High-frequency environmental band (guard for Nyquist)
        spectral["noise_200_250hz"] = _band_power(raw, picks, 200.0, 250.0) if nyq >= 250.0 else None

        # Line noise harmonics
        spectral.update(_line_harmonics(raw, picks, line_freq))

        # Neural bands
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta":  (15.0, 30.0),
            "low_gamma":  (30.0, 50.0),
            "high_gamma": (50.0, 80.0),
        }
        for name, (fmin, fmax) in bands.items():
            spectral[f"band_{name}"] = _band_power(raw, picks, fmin, fmax)

        spectral["one_over_f"] = _one_over_f(raw, picks)
        metrics[f"{ch_type}_spectral"] = spectral

    # ---- cHPI quick status (best-effort) ----
    try:
        chpi_info = raw.info.get("hpi_meas", None)
        hpi_n_coils = None
        if chpi_info and isinstance(chpi_info, list) and len(chpi_info):
            hpi_n_coils = len(chpi_info[0].get("hpi_coils", [])) if isinstance(chpi_info[0], dict) else None
        metrics["chpi"] = {"has_hpi_meas": bool(chpi_info), "n_coils": hpi_n_coils, "active": bool(chpi_info)}
    except Exception:
        metrics["chpi"] = {"has_hpi_meas": None, "n_coils": None, "active": None}

    # ---- SSS/Maxwell parameters if applied ----
    if "proc_history" in raw.info:
        for proc in raw.info["proc_history"]:
            if isinstance(proc, dict) and "max_info" in proc:
                max_info = proc["max_info"]
                metrics["sss_info"] = {
                    "in_order": int(max_info.get("in_order", 0)),
                    "out_order": int(max_info.get("out_order", 0)),
                    "nbad": int(max_info.get("nbad", 0)),
                    "coord_frame": str(max_info.get("coord_frame", "unknown")),
                }
                break

    return metrics

def calculate_maxwell_improvements(metrics_before: Dict[str, Any],
                                   metrics_after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate improvements from Maxwell filtering by comparing metric dictionaries.

    Args:
        metrics_before: Metrics computed before Maxwell
        metrics_after: Metrics computed after Maxwell

    Returns:
        Dictionary containing shielding factors, signal preservation, and quality score
    """
    improvements = {
        "timestamp": datetime.now().isoformat(),
        "stages_compared": f"{metrics_before['stage']}_to_{metrics_after['stage']}"
    }

    # Calculate shielding factors and improvements for each channel type
    for ch_type in ['mag', 'grad']:
        before_spectral = metrics_before.get(f"{ch_type}_spectral", {})
        after_spectral = metrics_after.get(f"{ch_type}_spectral", {})

        if not before_spectral or not after_spectral:
            continue

        improvements[ch_type] = {}

        # 1. Environmental noise reduction (shielding factor)
        noise_key = "noise_200_250hz"
        before_noise = before_spectral.get(noise_key)
        after_noise = after_spectral.get(noise_key)

        if before_noise is not None and after_noise is not None and before_noise > 0 and after_noise > 0:
            factor = before_noise / after_noise
            improvements[ch_type]["environmental_shielding"] = {
                "factor": float(factor),
                "db": float(10 * np.log10(factor)),
                "percent_reduction": float((1 - 1 / factor) * 100),
                "before_power": float(before_noise),
                "after_power": float(after_noise),
                "quality": "excellent" if factor > 100 else
                "good" if factor > 10 else
                "moderate" if factor > 3 else "poor"
            }

        # 2. Line noise reduction
        for harm in [1, 2, 3]:
            key = f"line_{harm}x_peak"
            if key in before_spectral and key in after_spectral:
                before_val = before_spectral[key]
                after_val = after_spectral[key]
                if before_val is not None and after_val is not None and before_val > 0 and after_val > 0:
                    factor = before_val / after_val
                    improvements[ch_type][f"line_{harm}x_reduction"] = {
                        "factor": float(factor),
                        "db": float(10 * np.log10(factor)),
                        "percent_reduction": float((1 - 1 / factor) * 100)
                    }

        # 3. Signal preservation in neural bands
        for band in ['theta', 'alpha', 'beta', 'low_gamma']:
            key = f"{band}_power"
            if key in before_spectral and key in after_spectral:
                before_val = before_spectral[key]
                after_val = after_spectral[key]
                if before_val is not None and before_val > 0:
                    preservation = after_val / before_val if after_val is not None else 0
                    improvements[ch_type][f"{band}_preservation"] = {
                        "ratio": float(preservation),
                        "percent": float(preservation * 100),
                        "assessment": "good" if 0.7 < preservation < 1.3 else
                        "acceptable" if 0.5 < preservation < 1.5 else "poor"
                    }

        # 4. Overall amplitude changes
        before_amp = metrics_before.get(f"{ch_type}_amplitude", {})
        after_amp = metrics_after.get(f"{ch_type}_amplitude", {})
        if before_amp and after_amp:
            before_rms = before_amp.get('rms', 0)
            after_rms = after_amp.get('rms', 0)
            before_var = before_amp.get('variance', 0)
            after_var = after_amp.get('variance', 0)

            improvements[ch_type]["amplitude_change"] = {
                "rms_ratio": float(after_rms / before_rms) if before_rms > 0 else None,
                "variance_ratio": float(after_var / before_var) if before_var > 0 else None
            }

        # 5. Kurtosis change (reduction in spiky artifacts)
        before_kurt = metrics_before.get(f"{ch_type}_kurtosis", {})
        after_kurt = metrics_after.get(f"{ch_type}_kurtosis", {})
        if before_kurt and after_kurt:
            before_mean = before_kurt.get('mean', 0)
            after_mean = after_kurt.get('mean', 0)
            if before_mean != 0:
                improvements[ch_type]["kurtosis_change"] = {
                    "before": float(before_mean),
                    "after": float(after_mean),
                    "reduction": float(before_mean - after_mean),
                    "percent_change": float((after_mean - before_mean) / abs(before_mean) * 100)
                }

    # 6. Calculate overall quality score
    score = calculate_quality_score(improvements)
    improvements['overall_quality_score'] = score

    return improvements


def calculate_quality_score(improvements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a single quality score from Maxwell improvements.

    Args:
        improvements: Dictionary from calculate_maxwell_improvements()

    Returns:
        Dictionary with score, grade, and interpretation
    """
    score = 100.0
    weights = {'mag': 0.6, 'grad': 0.4}  # Magnetometers slightly more important

    details = []  # Track what affected the score

    for ch_type in ['mag', 'grad']:
        if ch_type not in improvements:
            continue

        ch_weight = weights[ch_type]
        ch_improvements = improvements[ch_type]

        # Check shielding factor (most important - 40% of score)
        if 'environmental_shielding' in ch_improvements:
            factor = ch_improvements['environmental_shielding'].get('factor', 1)
            if factor < 3:
                score -= 30 * ch_weight
                details.append(f"{ch_type}: poor shielding (<3x)")
            elif factor < 10:
                score -= 15 * ch_weight
                details.append(f"{ch_type}: moderate shielding (3-10x)")
            elif factor > 100:
                score += 5 * ch_weight  # Bonus for excellent shielding
                details.append(f"{ch_type}: excellent shielding (>100x)")
        else:
            score -= 20 * ch_weight  # No shielding data is concerning
            details.append(f"{ch_type}: no shielding data")

        # Check signal preservation (30% of score)
        signal_preserved = []
        for band in ['alpha', 'beta']:
            key = f"{band}_preservation"
            if key in ch_improvements:
                ratio = ch_improvements[key]['ratio']
                signal_preserved.append(ratio)
                if ratio < 0.5 or ratio > 2.0:
                    score -= 10 * ch_weight
                    details.append(f"{ch_type} {band}: poor preservation")
                elif ratio < 0.7 or ratio > 1.3:
                    score -= 5 * ch_weight
                    details.append(f"{ch_type} {band}: suboptimal preservation")

        # Bonus for excellent signal preservation
        if signal_preserved and all(0.9 < r < 1.1 for r in signal_preserved):
            score += 5 * ch_weight
            details.append(f"{ch_type}: excellent signal preservation")

        # Check line noise reduction (10% of score)
        line_1x = ch_improvements.get('line_1x_reduction', {}).get('factor', 1)
        if line_1x < 2:
            score -= 5 * ch_weight
            details.append(f"{ch_type}: poor line noise reduction")

    # Ensure score is in valid range
    final_score = float(max(0, min(100, score)))

    # Determine grade
    if final_score >= 90:
        grade = "A"
        interpretation = "excellent"
    elif final_score >= 80:
        grade = "B"
        interpretation = "good"
    elif final_score >= 70:
        grade = "C"
        interpretation = "acceptable"
    elif final_score >= 60:
        grade = "D"
        interpretation = "marginal"
    else:
        grade = "F"
        interpretation = "poor"

    return {
        "value": final_score,
        "grade": grade,
        "interpretation": interpretation,
        "details": details[:5]  # Keep top 5 most important factors
    }

def generate_quality_summary(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from log data for executive summary.

    Args:
        log_data: Complete pipeline log dictionary

    Returns:
        Simplified summary with key quality metrics
    """
    quality = log_data.get('quality_metrics', {})
    improvements = quality.get('maxwell_improvements', {})

    # Extract key values safely
    mag_shielding = improvements.get('mag', {}).get('environmental_shielding', {})
    grad_shielding = improvements.get('grad', {}).get('environmental_shielding', {})

    # Calculate average signal preservation
    preservation_values = []
    for ch_type in ['mag', 'grad']:
        for band in ['alpha', 'beta']:
            key = f"{band}_preservation"
            if ch_type in improvements and key in improvements[ch_type]:
                preservation_values.append(improvements[ch_type][key].get('percent', 100))

    avg_preservation = float(np.mean(preservation_values)) if preservation_values else None

    summary = {
        "session": f"{log_data.get('subject', 'unknown')}_{log_data.get('session', '')}_{log_data.get('task', '')}",
        "date": log_data.get('runtime_info', {}).get('timestamp', 'unknown'),
        "quality_grade": improvements.get('overall_quality_score', {}).get('grade', 'N/A'),
        "quality_score": improvements.get('overall_quality_score', {}).get('value', None),
        "mag_shielding_factor_db": mag_shielding.get('db', None),
        "grad_shielding_factor_db": grad_shielding.get('db', None),
        "signal_preservation_percent": avg_preservation,
        "recording_duration_sec": log_data.get('recording_duration_sec', None),
        "n_bad_channels": len(
            log_data.get('quality_metrics', {}).get('post_maxwell', {}).get('recording', {}).get('bad_channels', []))
    }

    return summary


# Add this function to meg_pipeline_utils.py

def log_maxwell_quality_results(metrics_pre: Dict[str, Any],
                                metrics_post: Dict[str, Any],
                                logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Calculate and log Maxwell filtering quality from pre/post metrics.

    Args:
        metrics_pre: Pre-Maxwell metrics from compute_meg_quality_metrics()
        metrics_post: Post-Maxwell metrics from compute_meg_quality_metrics()
        logger: Optional logger instance

    Returns:
        Dictionary containing all metrics and improvements
    """
    if logger is None:
        logger = logging.getLogger("pipeline")

    # Calculate improvements
    logger.info("Calculating Maxwell filter effectiveness...")
    improvements = calculate_maxwell_improvements(metrics_pre, metrics_post)

    # Log summary to console
    quality_score = improvements.get('overall_quality_score', {})
    logger.info(f"📊 Maxwell Quality Score: {quality_score.get('value', 0):.1f}/100 "
                f"(Grade: {quality_score.get('grade', 'N/A')})")

    # Log key metrics per channel type
    for ch_type in ['mag', 'grad']:
        if ch_type not in improvements:
            continue

        # Shielding factor
        shielding = improvements[ch_type].get('environmental_shielding', {})
        if shielding:
            logger.info(f"  {ch_type.upper()} shielding: {shielding.get('db', 0):.1f} dB "
                        f"({shielding.get('quality', 'unknown')})")

        # Signal preservation
        preservation = []
        for band in ['alpha', 'beta']:
            key = f"{band}_preservation"
            if key in improvements[ch_type]:
                pres = improvements[ch_type][key]
                preservation.append(f"{band}={pres['percent']:.0f}%")
        if preservation:
            logger.info(f"  {ch_type.upper()} signal preserved: {', '.join(preservation)}")

    # Log any quality concerns
    if quality_score.get('details'):
        logger.info("  Quality factors: " + "; ".join(quality_score['details'][:3]))

    # Return complete metrics package
    return {
        "pre_maxwell": metrics_pre,
        "post_maxwell": metrics_post,
        "maxwell_improvements": improvements,
        "quality_score": quality_score
    }

# ---------- IMPROVED AUTOREJECT FUNCTION (HEADLESS / STAGE 1 ONLY) ----------
def find_bad_channels_autoreject_by_type(
        raw,
        which_types=('eeg', 'mag', 'grad'),
        filter_settings=None,
        epoch_settings=None,
        consensus_thresh=0.3,
        global_epoch_thresh=0.3,
        interactive=False,   # kept for API compatibility; ignored here
        logger=None,
        cfg=None
):
    """
    Detect bad channels (EEG RANSAC + AutoReject) and globally bad epochs, then
    write results back to the ORIGINAL raw object (bads + BAD_GlobalEpoch annotations).

    Stage 1 (headless) only: `interactive` is ignored; Stage 2 handles UI.

    Returns
    -------
    detected_bads : dict
        {
          'eeg': [...], 'mag': [...], 'grad': [...],
          'eeg_global_bad_epochs': [sample_idx, ...], ...
        }
        Note: *_global_bad_epochs are sample indices in ORIGINAL raw.
    """
    import time
    import numpy as np
    from collections import Counter
    import logging
    import mne
    from mne.annotations import Annotations

    if logger is None:
        logger = logging.getLogger("pipeline")

    t0 = time.perf_counter()

    # Warn and ignore interactive for Stage 1
    if interactive:
        logger.warning(
            "interactive=True was passed to find_bad_channels_autoreject_by_type, "
            "but Stage 1 is headless. Ignoring interactive mode here; "
            "use run_interactive_review_stage2 for UI-driven review."
        )

    # Parse params with safe defaults
    filter_settings = filter_settings or {'highpass': 1.0, 'lowpass': 40.0, 'resample_hz': None}
    epoch_settings = epoch_settings or {'duration': 2.0, 'tmin': 0.0, 'tmax': 2.0}

    l_freq = float(filter_settings.get('highpass', 1.0))
    h_freq = float(filter_settings.get('lowpass', 40.0))
    resample_hz = filter_settings.get('resample_hz')
    resample_hz = float(resample_hz) if resample_hz else None

    duration = float(epoch_settings.get('duration', 2.0))
    tmin = float(epoch_settings.get('tmin', 0.0))
    tmax = float(epoch_settings.get('tmax', 2.0))

    consensus_thresh = float(consensus_thresh)
    global_epoch_thresh = float(global_epoch_thresh)

    # ---- Resolve AutoReject params from cfg (key fix) ----
    ar_cfg = cfg.get("autoreject", {}) if cfg else {}
    ar_n_interpolate = ar_cfg.get("n_interpolate", [0])
    ar_cv = ar_cfg.get("cv_folds", 5)
    ar_method = ar_cfg.get("thresh_method", "random_search")
    ar_n_jobs = int(ar_cfg.get("n_jobs", 1))
    ar_random_state = int(ar_cfg.get("random_state", 42))
    ar_verbose = bool(ar_cfg.get("verbose", False))

    logger.info(
        f"[AutoReject params] cv={ar_cv}, method={ar_method}, "
        f"n_interpolate={ar_n_interpolate}, n_jobs={ar_n_jobs}, random_state={ar_random_state}"
    )
    logger.info(
        f"[AutoReject thresholds] consensus>{consensus_thresh:.2f}, global_epoch>{global_epoch_thresh:.2f}"
    )

    detected_bads = {}
    all_bad_channels = set()
    all_global_bad_epochs = set()  # sample indices in ORIGINAL raw

    # For summary stats (autodetect only = from AutoReject consensus, excluding RANSAC)
    autodetect_bad_counts = {typ: 0 for typ in which_types}
    autodetect_bad_channels_all = set()
    global_bad_epoch_counts = {typ: 0 for typ in which_types}

    # Create filtered copy for detection
    logger.info(f"Creating filtered copy for bad detection: {l_freq}-{h_freq} Hz"
                + (f", resample→{resample_hz} Hz" if resample_hz else ""))
    try:
        raw_filt_all = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose='ERROR')
        if resample_hz:
            original_sfreq = float(raw_filt_all.info['sfreq'])
            raw_filt_all.resample(resample_hz, npad="auto")
            logger.info(f"Resampled from {original_sfreq:.1f} Hz to {float(resample_hz):.1f} Hz")
    except Exception as e:
        logger.error(f"Failed to create filtered copy: {e}")
        raise

    # Helper: pick channels per type (from ORIGINAL raw.info)
    def _picks_for(typ):
        if typ == 'eeg':
            return mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        if typ == 'mag':
            return mne.pick_types(raw.info, meg='mag', eeg=False, exclude='bads')
        if typ == 'grad':
            return mne.pick_types(raw.info, meg='grad', eeg=False, exclude='bads')
        return np.array([], dtype=int)

    # Iterate by type
    for typ in which_types:
        logger.info(f"\n=== AutoReject pass for {typ.upper()} ===")
        picks = _picks_for(typ)
        if picks.size == 0:
            logger.info(f"No {typ.upper()} channels to analyze.")
            detected_bads[typ] = []
            detected_bads[f"{typ}_global_bad_epochs"] = []
            continue

        ch_names = [raw.ch_names[i] for i in picks]
        logger.info(
            f"Analyzing {len(ch_names)} {typ.upper()} channels: {ch_names[:5]}{'...' if len(ch_names) > 5 else ''}"
        )

        # Work on filtered copy containing only these channels
        try:
            raw_filt = raw_filt_all.copy().pick_channels(ch_names)
        except Exception as e:
            logger.error(f"Failed to pick {typ} channels: {e}")
            detected_bads[typ] = []
            detected_bads[f"{typ}_global_bad_epochs"] = []
            continue

        # Build fixed-length epochs on filtered copy
        try:
            events = mne.make_fixed_length_events(raw_filt, duration=duration)
            if events is None or len(events) == 0:
                logger.warning(f"No fixed-length events could be created for {typ.upper()}. Skipping.")
                detected_bads[typ] = []
                detected_bads[f"{typ}_global_bad_epochs"] = []
                continue

            epochs = mne.Epochs(
                raw_filt, events, event_id=None,
                tmin=tmin, tmax=tmax,
                picks=np.arange(len(ch_names)),
                baseline=None, detrend=0, preload=True, verbose=False,
                reject_by_annotation=True
            )
            if len(epochs) == 0:
                logger.warning(f"Zero epochs after creation for {typ.upper()}. Skipping.")
                detected_bads[typ] = []
                detected_bads[f"{typ}_global_bad_epochs"] = []
                continue

            logger.info(f"Created {len(epochs)} fixed-length epochs ({duration:.2f}s) for {typ.upper()}.")
        except Exception as e:
            logger.error(f"Failed to create epochs for {typ}: {e}")
            detected_bads[typ] = []
            detected_bads[f"{typ}_global_bad_epochs"] = []
            continue

        # RANSAC (EEG only)
        bads_ransac = []
        if typ == 'eeg':
            try:
                from autoreject import Ransac
                logger.info("Running RANSAC for EEG channels...")
                ransac = Ransac(n_jobs=ar_n_jobs, random_state=ar_random_state)
                ransac.fit(epochs)
                bads_ransac = list(getattr(ransac, "bad_chs_", []))
                logger.info(f"RANSAC identified {len(bads_ransac)} bad EEG channels: {bads_ransac}")
            except Exception as e:
                logger.warning(f"RANSAC failed for EEG: {e}")
                bads_ransac = []

        # AutoReject on remaining "good" channels
        if typ == 'eeg':
            picks_good = [ch for ch in epochs.ch_names if ch not in bads_ransac]
        else:
            picks_good = epochs.ch_names

        if len(picks_good) < 2:
            logger.warning(f"Not enough good {typ.upper()} channels for AutoReject after RANSAC.")
            bads_final = bads_ransac
            detected_bads[f"{typ}_global_bad_epochs"] = []
        else:
            try:
                from autoreject import AutoReject
                logger.info(
                    f"Running AutoReject on {len(picks_good)} good {typ.upper()} channels "
                    f"(cv={ar_cv}, method={ar_method}, n_interp={ar_n_interpolate})..."
                )

                epochs_good = epochs.copy().pick_channels(picks_good)

                ar = AutoReject(
                    n_interpolate=ar_n_interpolate,
                    cv=ar_cv,
                    thresh_method=ar_method,
                    consensus=None,
                    n_jobs=ar_n_jobs,
                    random_state=ar_random_state,
                    verbose=ar_verbose
                )
                ar.fit(epochs_good)
                reject_log = ar.get_reject_log(epochs_good)

                # --- AutoReject "autodetect" consensus (EXCLUDING RANSAC) ---
                # prefer boolean labels (n_epochs, n_channels); fall back defensively if needed
                bad_consensus = set()
                n_epochs = 0
                if hasattr(reject_log, "labels") and reject_log.labels is not None:
                    labels = np.array(reject_log.labels)
                    if labels.ndim == 2:  # boolean matrix expected
                        n_epochs = labels.shape[0]
                        frac_bad_by_chan = labels.mean(axis=0)
                        bad_consensus = {
                            epochs_good.ch_names[i] for i, frac in enumerate(frac_bad_by_chan)
                            if frac > consensus_thresh
                        }
                    else:
                        # Very defensive fallback (older structures)
                        n_epochs = len(labels)
                        # Keep your original Counter approach for compatibility
                        bad_labels = []
                        for row in labels:
                            for val in row:
                                if isinstance(val, str):
                                    bad_labels.append(val)
                        if n_epochs > 0 and bad_labels:
                            counts = Counter(bad_labels)
                            bad_consensus = {
                                ch for ch, ct in counts.items() if (ct / n_epochs) > consensus_thresh
                            }

                logger.info(
                    f"AutoReject consensus>{consensus_thresh:.2f} → "
                    f"{len(bad_consensus)} {typ.upper()} bads (autodetect): {sorted(bad_consensus)}"
                )

                # --- Global bad epochs for this type (fraction of channels bad per epoch) ---
                global_bad_epochs = []
                if hasattr(reject_log, "labels") and reject_log.labels is not None:
                    labels = np.array(reject_log.labels)
                    if labels.ndim == 2 and labels.shape[0] > 0:
                        frac_bad_per_epoch = labels.mean(axis=1)
                        bad_epoch_indices = np.where(frac_bad_per_epoch > global_epoch_thresh)[0]

                        # Convert epoch start index (in filtered copy) back to ORIGINAL raw sample indices
                        for epoch_idx in bad_epoch_indices:
                            sample_start = int(events[epoch_idx, 0])  # index in filtered copy
                            if resample_hz and raw.info['sfreq'] != resample_hz:
                                scale_factor = float(raw.info['sfreq']) / float(resample_hz)
                                sample_start = int(sample_start * scale_factor)
                            global_bad_epochs.append(sample_start)

                logger.info(
                    f"Global bad epochs for {typ.upper()}: {len(global_bad_epochs)} "
                    f"(threshold={global_epoch_thresh:.2f})"
                )
                detected_bads[f"{typ}_global_bad_epochs"] = global_bad_epochs
                all_global_bad_epochs.update(global_bad_epochs)
                global_bad_epoch_counts[typ] = len(global_bad_epochs)

                # --- Final bad set for this type (RANSAC ∪ AutoReject) ---
                bads_final = list(set(bads_ransac) | set(bad_consensus))

                # --- Summary tallies (autodetect only) ---
                autodetect_bad_counts[typ] = len(bad_consensus)
                autodetect_bad_channels_all.update(bad_consensus)

            except Exception as e:
                logger.warning(f"AutoReject failed for {typ.upper()}: {e}")
                bads_final = bads_ransac
                detected_bads[f"{typ}_global_bad_epochs"] = []

        # Collect type-level bads
        bads_final = [ch for ch in bads_final if isinstance(ch, str)]
        detected_bads[typ] = bads_final
        all_bad_channels.update(bads_final)

    # ---------- Apply results to ORIGINAL raw ----------
    # Update bad channels
    prev_bads = set(raw.info.get('bads', []))
    new_bads = sorted(set(all_bad_channels) - prev_bads)
    raw.info['bads'] = sorted(prev_bads | set(all_bad_channels))
    logger.info(f"Updated raw.info['bads']: +{new_bads} ⇒ all bads: {raw.info['bads']}")

    # Helper: remove prior BAD_GlobalEpoch annotations safely
    def _remove_bad_epoch_anns(raw_obj):
        if raw_obj.annotations is None or len(raw_obj.annotations) == 0:
            return
        keep_mask = [desc != "BAD_GlobalEpoch" for desc in raw_obj.annotations.description]
        if all(keep_mask):
            return
        kept = raw_obj.annotations[keep_mask]
        raw_obj.set_annotations(kept)

    # Remove old BAD_GlobalEpochs from ORIGINAL raw
    _remove_bad_epoch_anns(raw)

    # Add NEW BAD_GlobalEpoch annotations using sample indices (aligned to ORIGINAL raw)
    if len(all_global_bad_epochs) > 0:
        # Determine a stable orig_time for annotations
        base_orig = raw.annotations.orig_time if (raw.annotations is not None and raw.annotations.orig_time is not None) \
                    else raw.info.get('meas_date', None)

        sfreq = float(raw.info['sfreq'])
        onsets = [int(samp) / sfreq for samp in sorted(set(all_global_bad_epochs))]
        durs = [duration] * len(onsets)
        descs = ["BAD_GlobalEpoch"] * len(onsets)

        new_anns = Annotations(onset=onsets, duration=durs, description=descs, orig_time=base_orig)
        if raw.annotations is not None and len(raw.annotations) > 0:
            combined = mne.Annotations(
                onset=list(raw.annotations.onset) + list(new_anns.onset),
                duration=list(raw.annotations.duration) + list(new_anns.duration),
                description=list(raw.annotations.description) + list(new_anns.description),
                orig_time=raw.annotations.orig_time
            )
            raw.set_annotations(combined)
        else:
            raw.set_annotations(new_anns)
        logger.info(f"Added {len(new_anns)} BAD_GlobalEpoch annotations to ORIGINAL raw.")

    # ---- End-of-run SUMMARY ----
    runtime_min = round((time.perf_counter() - t0) / 60.0, 2)
    total_autodetect_bad = sum(v for v in autodetect_bad_counts.values() if isinstance(v, int))
    total_global_bad_epochs = len(all_global_bad_epochs)

    # Compact, grep-friendly line:
    logger.info(
        "=== AutoReject SUMMARY === "
        f"runtime={runtime_min:.2f} min; "
        f"params(cv={ar_cv}, method={ar_method}, n_interp={ar_n_interpolate}); "
        f"autodetect_bad_channels_total={total_autodetect_bad}; "
        f"global_bad_epochs_union={total_global_bad_epochs}"
    )
    # Optional per-type line for quick comparisons:
    per_type_bits = []
    for typ in which_types:
        if typ in autodetect_bad_counts:
            per_type_bits.append(
                f"{typ}: autodetect_bad={autodetect_bad_counts[typ]}, "
                f"global_epochs={global_bad_epoch_counts.get(typ, 0)}"
            )
    if per_type_bits:
        logger.info("[AutoReject] Per-type: " + " | ".join(per_type_bits))

    # Cleanup filtered copy and return
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

def apply_final_filter_and_cleanup(raw: mne.io.Raw, cfg: dict) -> mne.io.Raw:
    """
    Applies optional highpass/lowpass filtering, resampling, and
    drops specified channel types or name patterns before saving.

    Controlled entirely by the 'final_filter' section of the YAML config.
    """
    final_cfg = cfg.get("final_filter", {})
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
# -------------------------------------------------------------------------
# New: parameter handling utilities for layered config management
# -------------------------------------------------------------------------
import os
import copy
from ruamel.yaml import YAML

# Hard-wired base defaults (expand as needed)
_BASE_DEFAULTS = {
    "paths": {
        "bids_root": "./bids",
        "derivatives_root": "./derivatives",
        "output_subdir": "preproc",
    },
    "pipeline_name": "meg_pipeline",
    "line_freq": 60,
    "eeg_handling": "average",
    "maxwell_filter": {
        "apply": False,
        "calibration_file": None,
        "crosstalk_file": None,
    },
    "autoreject": {
        "apply": False,
        "n_interpolates": [1, 4, 32],
        "consensus": [0.5, 0.7, 0.9],
    },
}

def _load_yaml(path):
    yaml = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. Lists are replaced, not concatenated."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

def build_effective_config(user_yaml_path, lab_defaults_path=None, fif_path=None):
    """
    Merge base defaults, optional lab defaults, and user YAML.
    Precedence for selecting the lab defaults path:
      1) explicit function arg (lab_defaults_path)
      2) environment variable LAB_DEFAULTS_YAML
      3) 'lab_defaults' field inside the user YAML

    Merge precedence for config keys: user > lab > base.
    """
    cfg = copy.deepcopy(_BASE_DEFAULTS)

    # --- Step 1: (light) read of user YAML to discover optional lab_defaults ---
    user_cfg_light = {}
    if user_yaml_path and os.path.exists(user_yaml_path):
        try:
            user_cfg_light = _load_yaml(user_yaml_path) or {}
            if not isinstance(user_cfg_light, dict):
                raise ValueError(f"Top-level of user YAML must be a mapping: {user_yaml_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read user YAML '{user_yaml_path}': {e}") from e

    # Resolve lab defaults path with clear precedence:
    # explicit arg > env var > user YAML field
    lab_path_resolved = (
        lab_defaults_path
        if lab_defaults_path
        else os.getenv("LAB_DEFAULTS_YAML")
        or user_cfg_light.get("lab_defaults")
    )

    # --- Step 2: merge lab defaults (if any) into base ---
    if lab_path_resolved:
        if not os.path.exists(lab_path_resolved):
            raise FileNotFoundError(f"Lab defaults YAML not found: {lab_path_resolved}")
        lab_cfg = _load_yaml(lab_path_resolved) or {}
        if not isinstance(lab_cfg, dict):
            raise ValueError(f"Top-level of lab defaults YAML must be a mapping: {lab_path_resolved}")
        _deep_merge(cfg, lab_cfg)

    # --- Step 3: merge full user config (after removing 'lab_defaults' key, if present) ---
    if user_cfg_light:
        user_cfg_full = dict(user_cfg_light)  # shallow copy
        user_cfg_full.pop("lab_defaults", None)  # don't leak this into effective config
        _deep_merge(cfg, user_cfg_full)

    # --- Step 4: optional FIF auto-detection (future hook) ---
    if fif_path:
        # TODO: use mne to detect has_eeg/meg/eog/ecg and set modality flags accordingly
        pass

    return cfg

def get_param(cfg, key, default=None, required=False, dtype=None, choices=None):
    """
    Safe getter for config values with validation.
    """
    if key not in cfg:
        if required:
            raise ValueError(f"Missing required parameter: {key}")
        return default

    val = cfg[key]

    if dtype and not isinstance(val, dtype):
        raise TypeError(f"Parameter '{key}' must be {dtype}, got {type(val)}")

    if choices and val not in choices:
        raise ValueError(f"Parameter '{key}' must be one of {choices}, got {val}")

    return val
