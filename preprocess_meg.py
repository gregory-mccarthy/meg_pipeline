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

# ========== Function to replace MNE-BIDS raw data read ==========
def read_raw_bids_robust(bids_path, **kwargs):
    """
    Drop-in replacement for mne_bids.read_raw_bids for MEG FIF files.
    Accepts a BIDSPath and loads data with plain MNE, bypassing split file ambiguity.
    """
    if isinstance(bids_path, dict):
        subject = bids_path["subject"]
        session = bids_path.get("session", None)
        task = bids_path.get("task", None)
        run = bids_path.get("run", None)
        datatype = bids_path.get("datatype", "meg")
        root = bids_path["bids_root"]
    else:
        subject = bids_path.subject
        session = bids_path.session
        task = bids_path.task
        run = bids_path.run
        datatype = bids_path.datatype
        root = bids_path.root

    fname = f"sub-{subject}"
    if session:
        fname += f"_ses-{session}"
    if task:
        fname += f"_task-{task}"
    if run:
        fname += f"_run-{run}"
    fname += f"_{datatype}.fif"

    fif_file = os.path.join(
        root, f"sub-{subject}",
        f"ses-{session}" if session else "",
        datatype,
        fname
    )
    fif_file = fif_file.replace("//", "/")

    if not os.path.exists(fif_file):
        raise FileNotFoundError(f"Base MEG FIF file not found: {fif_file}")

    raw = mne.io.read_raw_fif(fif_file, preload=True, **kwargs)
    return raw


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
    Includes all diagnostics, ICA, Maxwell, events, and full YAML logging.
    """
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}")
        sys.exit(1)

    p = load_yaml(yaml_path)
    runtime_info = get_runtime_info()

    subject = p["subject"]
    session = p.get("session", None)
    task = p.get("task", None)
    run = p.get("run", None)
    bids_root = p["bids_root"]

    # Construct canonical BIDSPath for input
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        root=bids_root
    )

    # Construct correct BIDS-compliant derivative path and filename
    bids_path_deriv = bids_path.copy().update(
        root=Path(bids_root) / "derivatives" / "preprocessing",
        suffix="meg",
        description="preproc",
        extension=".fif"
    )
    bids_path_ica_eeg = bids_path_deriv.copy().update(description="preprocICAeeg")
    bids_path_ica_meg = bids_path_deriv.copy().update(description="preprocICAmeg")

    log_section("1. Load Raw Data")
    raw = read_raw_bids_robust(bids_path)

    log_section("2. Compute or Load Head Position")

    head_movement_cfg = p.get("head_movement", {})
    movement_enabled = head_movement_cfg.get("enabled", False)
    pos_file_requested = head_movement_cfg.get("pos_file", "").strip() or None

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

    log_section("3. EEG Channel Setup")
    eeg_cfg = p.get('eeg_handling', {})
    montage_path = eeg_cfg.get("montage")
    eeg_status = prepare_eeg_channels(raw, montage_path, logger)

    log_section("4. Metadata Repair")
    metadata_fixes = p.get('metadata_fixes', {})
    metadata_log = apply_metadata_repairs(raw, metadata_fixes)

    log_section("5. PSD & RMS Diagnostics (Pre-filtering)")
    qc_meg_raw(raw)
    plot_psd_and_peaks(raw, "Raw PSD Before Notch", 400)

    log_section("6. Notch Filter")
    line_freq = float(p.get("line_freq", 60.0))
    notch_freqs = [line_freq * i for i in range(1, 5)]
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    raw.notch_filter(notch_freqs, picks=picks, method='fir', filter_length='auto')
    plot_psd_and_peaks(raw, "After Notch", 400)

    log_section("7. Bad Channel Detection")
    final_bads = handle_bad_channels(raw, p, logger)

    log_section("8. Maxwell Filter (tSSS)")
    raw = apply_maxwell_filter(
        raw,
        head_pos=head_pos_array,
        destination=None,
        cal=p.get("calibration_fname"),
        crosstalk=p.get("cross_talk_fname")
    )
    plot_psd_and_peaks(raw, "After Maxwell", 400)

    log_section("9. ICA: EEG")
    ica_eeg_cfg = p["ica_preprocessing"]["eeg"]
    raw, eeg_exclude = run_ica(raw, ica_eeg_cfg, bids_path_ica_eeg.fpath, modality="eeg")

    log_section("10. ICA: MEG")
    ica_meg_cfg = p["ica_preprocessing"]["meg"]
    raw, meg_exclude = run_ica(raw, ica_meg_cfg, bids_path_ica_meg.fpath, modality="meg")

    plot_psd_and_peaks(raw, "After ICA", 400)

    log_section("11. Event Detection")
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

    log_section("12. Optional Final Filter, Downsample, and Cleanup")
    raw = apply_final_filter_and_cleanup(raw, p)

    log_section("13. Save Final Data")
    out_fif = bids_path_deriv.fpath
    raw.save(str(out_fif), overwrite=True)
    logger.info(f"Saved preprocessed data to: {out_fif}")

    log_section("14. Write YAML and JSON Log")

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

    yaml_log = {
        "input_config": yaml_path,
        "bids_path": str(bids_path),
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python preprocessing_pipeline_v1_0.py config.yaml")
        sys.exit(1)
    run_pipeline(sys.argv[1])
