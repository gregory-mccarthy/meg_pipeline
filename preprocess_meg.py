#!/usr/bin/env python3
# ==========================================================
# MNE-PYTHON PREPROCESSING PIPELINE — VERSION 1.0
# by Gregory McCarthy and ChatGPT — 2025-07-01
# ==========================================================
"""
Performs modular MEG/EEG preprocessing using MNE-Python:

  - Loads BIDS MEG data (.fif)
  - EEG setup (digitized vs montage)
  - Metadata repairs (YAML or expert_patch)
  - Bad channel detection (auto + interactive)
  - Maxwell filter (tSSS)
  - ICA for EEG and MEG (interactive)
  - Event decoding (bitwise STI101)
  - Saves final data and full YAML log

Usage:
    python preprocessing_pipeline_v1_0.py config.yaml
"""

# ========== IMPORTS ==========
import os, sys, platform, socket, logging, getpass, gc
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_bids import BIDSPath
from scipy.signal import find_peaks
import json

from bids_io_utils import fetch_bids_data_and_sidecars, push_bids_derivatives_rsync, detect_environment
from bids_io_utils import write_bids_robust, read_raw_bids_robust
from bids_io_utils import get_all_bids_split_files, get_bids_headpos_path
from glob import glob

PIPELINE_VERSION = "1.0"

# ========== YAML SUPPORT ==========
try:
    from ruamel.yaml import YAML
    _yaml_mode = 'ruamel'
except ImportError:
    import yaml
    _yaml_mode = 'pyyaml'

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
    import numpy as np
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

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")
plt.ion()

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

def handle_bad_channels(raw: mne.io.Raw, config: dict, logger: logging.Logger) -> list:

    auto_keys_present = all(k in config for k in [
        "auto_bad_flat_thresh_meg",
        "auto_bad_flat_thresh_eeg",
        "auto_bad_noisy_thresh"
    ])

    use_interactive = config.get("interactive_bad_channels", True)

    picks = mne.pick_types(
        raw.info,
        meg=True,
        eeg=True,
        eog=True,
        ecg=False,
        stim=False,
        exclude=[]
    )

    if not auto_keys_present and not use_interactive:
        logger.info("No thresholds defined and interactive review disabled. Skipping bad channel marking.")
        return []

    logger.info("Creating filtered copy for bad channel detection and/or interactive review...")
    raw_filt = raw.copy()
    raw_filt.filter(None, 60.0, picks=picks, fir_design='firwin', verbose='error')

    if auto_keys_present:
        flat_thresh_meg = float(config["auto_bad_flat_thresh_meg"])
        flat_thresh_eeg = float(config["auto_bad_flat_thresh_eeg"])
        noisy_thresh = float(config["auto_bad_noisy_thresh"])

        flat, noisy = [], []

        for typ, label, flat_thresh in [
            ("eeg", "EEG", flat_thresh_eeg),
            ("mag", "Magnetometers", flat_thresh_meg),
            ("grad", "Gradiometers", flat_thresh_meg),
        ]:
            picks_type = mne.pick_types(
                raw.info,
                meg=(typ if typ != "eeg" else False),
                eeg=(typ == "eeg"),
                exclude=[]
            )
            if len(picks_type) == 0:
                continue
            data = raw_filt.get_data(picks=picks_type)
            ch_names = [raw.ch_names[i] for i in picks_type]
            stdev = np.std(data, axis=1)

            flat_chs = [ch for ch, s in zip(ch_names, stdev) if s < flat_thresh]
            med = np.median([s for s in stdev if s >= flat_thresh])
            noisy_chs = [ch for ch, s in zip(ch_names, stdev)
                         if (s > noisy_thresh * med and ch not in flat_chs)]

            flat.extend(flat_chs)
            noisy.extend(noisy_chs)

            logger.info(f"{label}: {len(flat_chs)} flat, {len(noisy_chs)} noisy channels")

        auto_bads = sorted(set(flat + noisy))
        raw_filt.info['bads'] = auto_bads
        logger.info(f"Auto-detected bad channels: {auto_bads}")

    if use_interactive:
        logger.info("Launching interactive viewer for bad channel review...")
        raw_filt.plot(n_channels=16, block=True)
        logger.info(f"Final bad channels after interactive review: {raw_filt.info['bads']}")

    raw.info['bads'] = list(raw_filt.info['bads'])
    del raw_filt
    gc.collect()
    return raw.info['bads']

def qc_meg_raw(raw: mne.io.Raw, hf_band: tuple = (250, 400)):
    data = raw.get_data()
    rms = np.sqrt((data ** 2).mean(axis=1))
    for typ, label in [("mag", "Magnetometers"), ("grad", "Gradiometers"), ("eeg", "EEG")]:
        picks = mne.pick_types(raw.info, meg=typ if "meg" in typ else False, eeg=(typ == 'eeg'), exclude='bads')
        if len(picks) == 0:
            continue
        vals = rms[picks]
        logger.info(f"{label}: {vals.mean():.2e} ± {vals.std():.2e} (n={len(vals)})")
        plt.hist(vals, bins=40, alpha=0.7)
        plt.title(f"RMS {label}")
        plt.pause(0.1)
        if "meg" in typ:
            psd = raw.compute_psd(picks=picks, fmin=hf_band[0], fmax=hf_band[1])
            hf_power = psd.get_data().mean(axis=1)
            logger.info(f"{label} HF noise: {hf_power.mean():.2e} ± {hf_power.std():.2e}")

def plot_psd_and_peaks(raw: mne.io.Raw, title: str = "", fmax: Optional[float] = None, n_peaks: int = 10):
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
    plt.pause(0.1)

def run_ica(raw: mne.io.Raw, config: dict, output_path: Path, modality: str) -> tuple:
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

    raw_ica = raw.copy()
    l_freq = config.get('highpass', None)
    h_freq = config.get('lowpass', None)
    if l_freq is not None or h_freq is not None:
        raw_ica.filter(l_freq, h_freq, picks=picks, fir_design='firwin', verbose='error')
        logger.info(f"Filtered {modality.upper()} for ICA: highpass={l_freq}, lowpass={h_freq}")

    hz = config.get('resample_hz', None)
    if hz:
        raw_ica.resample(hz, npad="auto")
        logger.info(f"Resampled {modality.upper()} for ICA to {hz} Hz")

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

    ica = ICA(n_components=min(20, len(picks)), method='fastica', random_state=97)
    ica.fit(raw_ica, picks=picks)
    interactive = config.get("interactive", True)
    if interactive:
        ica.plot_components(inst=raw_ica)
        ica.plot_sources(raw_ica)
        input(f"Review {modality.upper()} ICA components. Press Enter to continue...")
    else:
        logger.info(f"Skipping interactive ICA viewer for {modality.upper()}.")
    exclude = list(ica.exclude)

    raw_clean = ica.apply(raw.copy())
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

def run_pipeline(yaml_path: str):
    """
    Orchestrate full preprocessing pipeline using config file.
    """
    import os
    import sys

    log_section("1. Load config and check paths and critical files")
    # 1.1 Check that the YAML config file exists
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}")
        sys.exit(1)

    # 1.2 Load YAML config
    p = load_yaml(yaml_path)

    # 1.3 Set repo root as the directory containing this script
    repo_root = Path(__file__).resolve().parent
    logger.info(f"Repo root set to: {repo_root}")

    # 1.4 Check all critical files (fail fast if missing)
    try:
        checked_paths = check_critical_files_exist(p, repo_root)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    # 1.5 (Optional) Print/log resolved file paths for debug
    logger.info(f"All critical files found:")
    for key, path in checked_paths.items():
        logger.info(f"  {key}: {path}")

    # ------ MAIN PIPELINE MODULES BEGIN HERE ------
    runtime_info = get_runtime_info()

    subject = p["subject"]
    session = p.get("session", None)
    task = p.get("task", None)
    run = p.get("run", None)
    bids_root = p["bids_root"]
    hpc_host = p.get("hpc_host")
    hpc_user = p.get("hpc_user")

    # Detect execution environment
    ENV = detect_environment(hpc_hostname_tag="milgram")

    # Decide where data lives and whether to fetch
    if ENV == "hpc" or bids_root.startswith("/Users"):
        # Data is local to execution environment; just use as-is
        local_bids_root = bids_root
    else:
        # Running locally, data lives on HPC: fetch/copy/cached to temp
        temp_dir = p.get("temp_dir", "./temp")
        # Compute BIDS MEG dir and base_stem
        base_stem = f"sub-{subject}"
        if session: base_stem += f"_ses-{session}"
        if task: base_stem += f"_task-{task}"
        if run: base_stem += f"_run-{run}"
        # base_stem += "_meg"
        remote_meg_dir = os.path.join(bids_root, f"sub-{subject}", f"ses-{session}" if session else "", "meg")
        local_bids_root = os.path.join(temp_dir, "bids")
        local_meg_dir = os.path.join(local_bids_root, f"sub-{subject}", f"ses-{session}" if session else "", "meg")
        fetch_bids_data_and_sidecars(hpc_host, hpc_user, remote_meg_dir, base_stem, local_meg_dir)

    # Construct canonical BIDSPath for input (now points to the local cache or as before)
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        root=local_bids_root  # ***THIS IS THE ONLY CHANGE***
    )

    # Construct correct BIDS-compliant derivative path and filename
    bids_path_deriv = bids_path.copy().update(
        root=Path(local_bids_root) / "derivatives" / "preprocessing",
        suffix="meg",
        description="preproc",
        extension=".fif"
    )
    bids_path_ica_eeg = bids_path_deriv.copy().update(description="preprocICAeeg")
    bids_path_ica_meg = bids_path_deriv.copy().update(description="preprocICAmeg")

    log_section("2. Load Raw Data")
    raw = read_raw_bids_robust(bids_path)
    raw_fif_path = raw.filenames[0]  # Use the actual file loaded by MNE

    # Build the MEG directory and expected .pos filename from BIDS fields
    meg_dir = os.path.dirname(raw_fif_path)
    subject = bids_path.subject
    session = bids_path.session
    task = bids_path.task
    run = bids_path.run

    # Use the BIDS convention to find the headpos file
    expected_pos_file = get_bids_headpos_path(subject, session, task, run, meg_dir)
    if os.path.exists(expected_pos_file):
        pos_file_requested = expected_pos_file
        print(f"Using existing head position file: {pos_file_requested}")
    else:
        pos_file_requested = None
        print("No matching headpos.pos file found. Will compute if needed.")

    log_section("3. Compute or Load Head Position")
    new_headpos_computed = False

    head_movement_cfg = p.get("head_movement", {})
    movement_enabled = head_movement_cfg.get("enabled", False)

    head_pos_array = None
    head_pos_path = None
    head_movement_log = {
        "enabled": movement_enabled,
        "method": None,
        "file_used": None
    }

    if movement_enabled:
        head_pos_path = get_head_pos_for_maxwell(
            raw,
            pos_file=pos_file_requested,
            compute_if_missing=True,
            logger=logger
        )
        if head_pos_path and os.path.exists(head_pos_path):
            try:
                head_pos_array = mne.chpi.read_head_pos(head_pos_path)
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
        # If you passed None, then you wanted to compute if missing
        if head_pos_path and os.path.exists(head_pos_path):
            new_headpos_computed = True

    log_section("4. EEG Channel Setup")
    eeg_cfg = p.get('eeg_handling', {})

    # Get the already-resolved montage path from checked_paths
    montage_path = checked_paths["montage"]

    eeg_status = prepare_eeg_channels(raw, str(montage_path), logger)

    log_section("5. Metadata Repair")
    metadata_fixes = p.get('metadata_fixes', {})
    metadata_log = apply_metadata_repairs(raw, metadata_fixes)

    log_section("6. PSD & RMS Diagnostics (Pre-filtering)")
    qc_meg_raw(raw)
    plot_psd_and_peaks(raw, "Raw PSD Before Notch", 400)

    log_section("7. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    raw.notch_filter(notch_freqs, picks=picks, method='fir', filter_length='auto')
    plot_psd_and_peaks(raw, "After Notch", 400)

    log_section("8. Bad Channel Detection")
    final_bads = handle_bad_channels(raw, p, logger)

    log_section("9. Maxwell Filter (tSSS)")

    raw = apply_maxwell_filter(
        raw,
        head_pos=head_pos_array,
        destination=None,
        cal=str(checked_paths["calibration_file"]),
        crosstalk=str(checked_paths["cross_talk_file"])
    )

    plot_psd_and_peaks(raw, "After Maxwell", 400)

    log_section("10. ICA: EEG")
    ica_eeg_cfg = p["ica_preprocessing"]["eeg"]
    raw, eeg_exclude = run_ica(raw, ica_eeg_cfg, bids_path_ica_eeg.fpath, modality="eeg")

    log_section("11. ICA: MEG")
    ica_meg_cfg = p["ica_preprocessing"]["meg"]
    raw, meg_exclude = run_ica(raw, ica_meg_cfg, bids_path_ica_meg.fpath, modality="meg")

    plot_psd_and_peaks(raw, "After ICA", 400)

    log_section("12. Event Detection")
    events = bitwise_events(raw)
    event_counts = {}
    if events.size:
        annots = mne.annotations_from_events(events, sfreq=raw.info['sfreq'])
        raw.set_annotations(annots)
        codes, counts = np.unique(events[:, 2], return_counts=True)
        event_counts = {int(c): int(n) for c, n in zip(codes, counts)}
        for code, n in event_counts.items():
            logger.info(f"Event {code}: {n} occurrences")
    else:
        logger.warning("No events detected")

    log_section("13. Optional Final Filter, Downsample, and Cleanup")
    raw = apply_final_filter_and_cleanup(raw, p)

    log_section("14. Save Final Data")
    out_fif = bids_path_deriv.fpath
    written_files = write_bids_robust(raw, bids_path_deriv, overwrite=True, verbose=True)
    logger.info(f"Saved preprocessed data to: {out_fif}")

    log_section("15. Write YAML and JSON Log")

    # Final filter summary (if it was applied)
    final_filter_cfg = p.get("final_filter", {})

    # Include drop list for log output (default to empty if not defined)
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
    # Duration of final processed data
    recording_duration_sec = raw.times[-1]

    # ICA durations, if available
    ica_durations = {
        "eeg_duration_sec": raw.times[-1] if 'eeg_exclude' in locals() else None,
        "meg_duration_sec": raw.times[-1] if 'meg_exclude' in locals() else None
    }

    # Gather all written output files (including splits)
    main_output_files = get_all_bids_split_files(out_fif)

    # ICA output FIFs and their splits
    ica_output_files = []
    for ica_fif in [bids_path_ica_eeg.fpath, bids_path_ica_meg.fpath]:
        ica_output_files += get_all_bids_split_files(ica_fif)
    ica_output_files = list(dict.fromkeys(ica_output_files))  # In case both ICA point to same file

    yaml_log = {
        "input_config": yaml_path,
        "bids_basefile": str(out_fif),
        "main_output_files": main_output_files,
        "ica_output_files": ica_output_files,
        "output_file": str(out_fif),
        "runtime_info": runtime_info,
        "recording_duration_sec": float(recording_duration_sec),
        "final_filter": final_filter_log,
        "ica_duration": ica_durations,
        "bad_channels": final_bads,
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

    save_yaml(out_log_yaml, make_serializable(yaml_log))
    logger.info(f"YAML log written to: {out_log_yaml}")

    with open(out_log_json, "w") as f:
        json.dump(make_serializable(yaml_log), f, indent=2)
    logger.info(f"JSON log written to: {out_log_json}")

    ENV = detect_environment(hpc_hostname_tag="milgram")

    main_output_files = write_bids_robust(raw, out_fif, overwrite=True, verbose=True)

    # --- Gather ICA outputs ---
    ica_output_files = []
    for ica_fif in [bids_path_ica_eeg.fpath, bids_path_ica_meg.fpath]:
        ica_output_files += get_all_bids_split_files(ica_fif)
    ica_output_files = list(dict.fromkeys(ica_output_files))  # Deduplicate

    # --- Gather all outputs for logging/upload ---
    outputs_to_push = [str(out_log_yaml), str(out_log_json)] + main_output_files + ica_output_files
    outputs_to_push = list(dict.fromkeys(outputs_to_push))  # Deduplicate

    # Gather all outputs: logs, main FIF and splits, ICA FIFs and splits
    outputs_to_push = [str(out_log_yaml), str(out_log_json)] + main_output_files + ica_output_files
    outputs_to_push = list(dict.fromkeys(outputs_to_push))  # Remove duplicates, preserve order

    if is_hybrid_workflow(p["bids_root"], out_fif):
        print("\n[User action required]")
        print(
            "Processing complete. Press ENTER when ready to upload all derivatives to the HPC (be ready for DUO prompt).")
        input()
        push_bids_derivatives_rsync(
            local_bids_root="./temp/BIDS",  # Your local BIDS cache root
            remote_bids_root=p["bids_root"],  # Remote BIDS root on HPC
            hpc_host=p["hpc_host"],
            hpc_user=p["hpc_user"],
            verbose=True
        )
        if new_headpos_computed:
            print(
                "\nA new head position .pos file was computed. Press ENTER to upload it to the HPC raw folder (be ready for DUO prompt).")
            input()
            # Build the path relative to local BIDS root
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python preprocessing_pipeline_v1_0.py config.yaml")
        sys.exit(1)
    run_pipeline(sys.argv[1])
