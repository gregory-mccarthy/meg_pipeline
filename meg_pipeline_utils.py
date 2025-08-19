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

# ---------- FIXED CHECKPOINTED AUTOREJECT WRAPPER ----------
def run_autoreject_with_checkpoint(raw, cfg, bids_path, logger):
    """
    Split autoreject into Stage 1 (non-interactive) and Stage 2 (interactive review only).

    DESIGN:
    - Stage 1: Non-interactive AutoReject, writes checkpoint (parproc + manifest), returns
    - Stage 2: Load checkpoint, run *interactive* review only (no recompute), update checkpoint, return

    Returns:
        ('continue', raw_like) on success,
        ('exit', raw_like) if configured to stop after Stage 1 (exit_after_checkpoint=True).
    """
    import hashlib
    from pathlib import Path
    import datetime
    import mne
    from mne.annotations import Annotations
    import numpy as np

    # ---- Resolve checkpoint config with safe defaults
    chk_cfg = cfg.get("checkpoint", {})
    enabled = chk_cfg.get("enabled", True)
    exit_after = chk_cfg.get("exit_after_checkpoint", False)

    derivatives_root = chk_cfg.get("derivatives_root", "derivatives")
    pipeline_name = chk_cfg.get("pipeline_name", "preprocessing")

    atomic_writes = chk_cfg.get("atomic_writes", True)
    allow_yaml_diff = chk_cfg.get("allow_resume_with_different_yaml", False)
    validate_integrity = chk_cfg.get("validate_checkpoint_integrity", True)
    backup_before_overwrite = chk_cfg.get("backup_before_overwrite", False)

    # Get interactive setting from main config
    interactive_bad_channels = cfg.get("interactive_bad_channels", True)

    if not enabled:
        logger.info("[checkpoint] Disabled in YAML; falling back to original interactive AR path.")
        # When checkpointing is disabled, run the old way
        ar_cfg = cfg.get("autoreject", {})
        ar_types = ar_cfg.get("which_types", ['eeg', 'mag', 'grad'])
        fset = ar_cfg.get("filter", {'highpass': 1.0, 'lowpass': 40.0, 'resample_hz': None})
        eset = ar_cfg.get("epoch", {'duration': 2.0, 'tmin': 0.0, 'tmax': 2.0})
        consensus_thresh = ar_cfg.get("consensus_thresh", 0.3)
        global_epoch_thresh = ar_cfg.get("global_epoch_thresh", 0.3)
        bads_dict = find_bad_channels_autoreject_by_type_improved(
            raw,
            which_types=ar_types,
            filter_settings=fset,
            epoch_settings=eset,
            consensus_thresh=consensus_thresh,
            global_epoch_thresh=global_epoch_thresh,
            interactive=interactive_bad_channels,
            logger=logger
        )
        logger.info(f"[checkpoint] (disabled) AR result: {bads_dict}; bads={raw.info.get('bads', [])}")
        return ('continue', raw)

    # ---- Build paths (respect current bids_root)
    bids_root = Path(cfg.get("bids_root", bids_path.root))
    deriv_root = bids_root / derivatives_root / pipeline_name

    # Build proper subject/session directory structure
    subject_dir = f"sub-{bids_path.subject}"
    if bids_path.session:
        session_dir = f"ses-{bids_path.session}"
        parproc_dir = deriv_root / subject_dir / session_dir / "meg"
    else:
        parproc_dir = deriv_root / subject_dir / "meg"
    parproc_dir.mkdir(parents=True, exist_ok=True)

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

    # ---- Helpers
    def _atomic_save_fif(raw_obj, out_path: Path):
        """Write FIF atomically by using temporary file with .fif extension (MNE warns about naming; harmless)."""
        if not atomic_writes:
            raw_obj.save(str(out_path), overwrite=True)
            return

        if backup_before_overwrite and out_path.exists():
            backup_path = out_path.with_suffix(out_path.suffix + ".backup")
            import shutil
            shutil.copy2(out_path, backup_path)
            logger.info(f"[checkpoint] Created backup: {backup_path}")

        tmp = out_path.with_name(out_path.stem + "_tmp.fif")
        try:
            raw_obj.save(str(tmp), overwrite=True)
            tmp.replace(out_path)
            logger.info(f"[checkpoint] Saved to: {out_path}")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def _yaml_sha256(yaml_text: str) -> str:
        return hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()

    def _build_manifest(stage, extra=None, current_yaml_hash=None, prior_yaml_hash=None):
        """Build manifest dictionary with metadata and (optionally) both YAML hashes."""
        yaml_text = None
        try:
            yaml_path_local = cfg.get("_cfg_path", "")
            if yaml_path_local and Path(yaml_path_local).exists():
                yaml_text = Path(yaml_path_local).read_text()
        except Exception as e:
            logger.warning(f"Could not read YAML for manifest: {e}")

        man = {
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "pipeline_version": "1.2",
            "checkpoint_version": cfg.get("_checkpoint_version", "1.0"),
            "exit_after_checkpoint": exit_after,
            "inputs": {
                "bids": {
                    "subject": bids_path.subject,
                    "session": bids_path.session,
                    "task": bids_path.task,
                    "run": bids_path.run,
                },
                "bids_root": str(bids_root),
                "yaml_path": cfg.get("_cfg_path", None),
                "yaml_sha256": _yaml_sha256(yaml_text) if yaml_text else None,
                # audit fields (optional)
                "yaml_sha256_current": current_yaml_hash,
                "yaml_sha256_prior": prior_yaml_hash,
            },
            "artifacts": {
                "parproc_raw_fif": str(parproc_fif),
                "manifest_path": str(manifest_path),
            },
            "params": {
                "autoreject": cfg.get("autoreject", {}),
                "checkpoint": chk_cfg,
                "interactive_bad_channels": interactive_bad_channels,
            },
            "environment": get_runtime_info(),
        }
        if extra:
            man.update(extra)
        return man

    def _validate_checkpoint_integrity(raw_obj, manifest_data):
        try:
            if len(raw_obj.ch_names) == 0:
                raise ValueError("No channels in checkpoint data")
            if raw_obj.times[-1] <= 0:
                raise ValueError("Invalid recording duration")
            bads = raw_obj.info.get('bads', [])
            invalid_bads = [ch for ch in bads if ch not in raw_obj.ch_names]
            if invalid_bads:
                raise ValueError(f"Invalid bad channels: {invalid_bads}")
            logger.info("[checkpoint] ✅ Checkpoint integrity validation passed")
            return True
        except Exception as e:
            logger.error(f"[checkpoint] ❌ Checkpoint integrity validation failed: {e}")
            return False

    def _run_interactive_review_only(raw_obj, ar_cfg, logger):
        """Run ONLY the interactive review; no AR recompute."""
        logger.info("[checkpoint] Opening interactive review (no re-computation)")
        fset = ar_cfg.get("filter", {'highpass': 1.0, 'lowpass': 40.0, 'resample_hz': None})
        l_freq = float(fset.get('highpass', 1.0))
        h_freq = float(fset.get('lowpass', 40.0))
        resample_hz = fset.get('resample_hz')

        logger.info(f"Creating filtered copy for interactive viewing: {l_freq}-{h_freq} Hz")
        raw_filt = raw_obj.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose='ERROR')
        if resample_hz:
            raw_filt.resample(float(resample_hz), npad="auto")

        raw_filt.info['bads'] = list(raw_obj.info.get('bads', []))
        if raw_obj.annotations is not None:
            raw_filt.set_annotations(raw_obj.annotations)

        original_bads = set(raw_obj.info.get('bads', []))
        original_annots_count = len(raw_obj.annotations) if raw_obj.annotations else 0

        logger.info("🎮 Opening interactive review window...")
        logger.info(f"   Current bad channels: {len(original_bads)}")
        logger.info(f"   Current annotations: {original_annots_count}")

        try:
            n_channels = min(32, len(raw_filt.ch_names))
            raw_filt.plot(
                n_channels=n_channels,
                duration=30.0,
                scalings='auto',
                show=True,
                block=True,
                title=f"Interactive Review - {len(raw_filt.info['bads'])} bad channels"
            )
            response = input("Press Enter to keep changes (or 'abort' to cancel): ").strip().lower()
            if response == 'abort':
                logger.info("❌ User aborted interactive review - keeping original")
                del raw_filt
                return raw_obj
        except Exception as e:
            logger.error(f"Interactive plotting failed: {e}")
            del raw_filt
            return raw_obj

        # Transfer changes back
        reviewed_bads = set(raw_filt.info.get('bads', []))
        added_bads = reviewed_bads - original_bads
        removed_bads = original_bads - reviewed_bads
        if added_bads:
            logger.info(f"✅ User added bad channels: {sorted(added_bads)}")
        if removed_bads:
            logger.info(f"🔄 User rescued channels: {sorted(removed_bads)}")

        raw_obj.info['bads'] = sorted(reviewed_bads)

        if raw_filt.annotations is not None and len(raw_filt.annotations) > 0:
            raw_obj.set_annotations(raw_filt.annotations)
            logger.info(f"✅ Updated annotations: {len(raw_filt.annotations)} total")

        del raw_filt
        logger.info("🎉 Interactive review completed!")
        return raw_obj

    # ---- Detect existing checkpoint
    checkpoint_exists = parproc_fif.exists() and manifest_path.exists()

    # ========================================================================
    # STAGE 1: Non-interactive AutoReject and checkpoint (when NO checkpoint exists)
    # ========================================================================
    if not checkpoint_exists:
        logger.info("[checkpoint] Stage 1: non-interactive AutoReject")

        # Get AutoReject parameters
        ar_cfg = cfg.get("autoreject", {})
        ar_types = ar_cfg.get("which_types", ['eeg', 'mag', 'grad'])
        fset = ar_cfg.get("filter", {'highpass': 1.0, 'lowpass': 40.0, 'resample_hz': None})
        eset = ar_cfg.get("epoch", {'duration': 2.0, 'tmin': 0.0, 'tmax': 2.0})
        consensus_thresh = ar_cfg.get("consensus_thresh", 0.3)
        global_epoch_thresh = ar_cfg.get("global_epoch_thresh", 0.3)

        # Run non-interactive AR
        bads_dict = find_bad_channels_autoreject_by_type_improved(
            raw,
            which_types=ar_types,
            filter_settings=fset,
            epoch_settings=eset,
            consensus_thresh=consensus_thresh,
            global_epoch_thresh=global_epoch_thresh,
            interactive=False,
            logger=logger
        )
        logger.info(f"[checkpoint] Stage 1 AutoReject complete")
        logger.info(f"[checkpoint] Detected bad channels: {raw.info.get('bads', [])}")

        # Save checkpoint
        _atomic_save_fif(raw, parproc_fif)
        man = _build_manifest(stage="stage1_complete", extra={
            "results": {
                "bads_detected": list(raw.info.get('bads', [])),
                "n_bad_channels": len(raw.info.get('bads', [])),
                "n_annotations": int(len(raw.annotations) if raw.annotations is not None else 0),
                "autoreject_details": bads_dict,
            }
        })
        save_yaml(str(manifest_path), make_serializable(man))
        logger.info(f"[checkpoint] ✅ checkpoint saved: {parproc_fif}")

        # Return control to main
        if exit_after:
            logger.info("[checkpoint] Two-stage mode: Exiting after Stage 1")
            return ('exit', raw)
        else:
            logger.info("[checkpoint] Continuous mode: Stage 1 complete")
            return ('continue', raw)

    # ========================================================================
    # STAGE 2: Load checkpoint and do interactive review (when checkpoint EXISTS)
    # ========================================================================
    logger.info("[checkpoint] Stage 2: interactive review from checkpoint")

    # Load prior manifest for validation
    try:
        prior_manifest = load_yaml(str(manifest_path))
    except Exception as e:
        raise RuntimeError(f"Cannot read checkpoint manifest: {e}")

    # YAML consistency check — warn by default (hybrid-friendly)
    want = prior_manifest.get("inputs", {}).get("yaml_sha256")
    have = None
    yaml_path_local = cfg.get("_cfg_path", "")
    if yaml_path_local and Path(yaml_path_local).exists():
        try:
            have = _yaml_sha256(Path(yaml_path_local).read_text())
        except Exception as e:
            logger.warning(f"[checkpoint] Cannot read current YAML for validation: {e}")
            have = want
    yaml_mismatch = bool(want and have and want != have)

    if yaml_mismatch:
        if allow_yaml_diff:
            logger.warning(
                "[checkpoint] ⚠️ YAML has changed since checkpoint creation. "
                "Proceeding with Stage 2 using the CURRENT YAML. "
                "The updated manifest will record both hashes."
            )
        else:
            logger.warning(
                "[checkpoint] ⚠️ YAML has changed since checkpoint creation. "
                "Proceeding with Stage 2 using the CURRENT YAML. "
                "To silence this warning, set 'checkpoint.allow_resume_with_different_yaml: true'."
            )
    else:
        logger.info("[checkpoint] YAML matches the checkpoint YAML.")

    # Load checkpoint data
    raw_parproc = mne.io.read_raw_fif(str(parproc_fif), preload=True, verbose="error")
    if validate_integrity and not _validate_checkpoint_integrity(raw_parproc, prior_manifest.get("results", {})):
        raise RuntimeError("Checkpoint integrity validation failed")

    # Interactive review
    if interactive_bad_channels:
        ar_cfg = cfg.get("autoreject", {})
        raw_parproc = _run_interactive_review_only(raw_parproc, ar_cfg, logger)
        _atomic_save_fif(raw_parproc, parproc_fif)

        # Update manifest, recording both hashes if available
        man2 = _build_manifest(
            "stage2_reviewed",
            extra={
                "results": {
                    "bads_after_review": list(raw_parproc.info.get('bads', [])),
                    "n_bad_channels_reviewed": len(raw_parproc.info.get('bads', [])),
                    "n_annotations": int(len(raw_parproc.annotations) if raw_parproc.annotations is not None else 0),
                }
            },
            current_yaml_hash=have,
            prior_yaml_hash=want
        )
        save_yaml(str(manifest_path), make_serializable(man2))
        logger.info("[checkpoint] ✅ checkpoint updated after interactive review (manifest includes YAML audit)")

        # IMPORTANT: return to caller
        return ('continue', raw_parproc)
    else:
        logger.info("[checkpoint] Interactive review disabled.")
        return ('continue', raw_parproc)

    # ---- Safety guard (should never hit)
    # If we ever get here, something fell through unexpectedly
    logger.error("[checkpoint] Internal error: unexpected fallthrough in run_autoreject_with_checkpoint")
    return ('continue', raw)

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


# ---------- IMPROVED AUTOREJECT FUNCTION ----------
def find_bad_channels_autoreject_by_type_improved(
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
    IMPROVED: Detect bad channels (RANSAC + AutoReject) and globally bad epochs.

    Key improvements:
    - Better memory management (process types separately if configured)
    - More robust annotation handling using sample indices
    - Improved error handling and validation
    - Better interactive workflow with progress saving
    - Enhanced logging and debugging information
    """
    import numpy as np
    from collections import Counter
    import logging
    import mne
    from mne.annotations import Annotations
    import time

    if logger is None:
        logger = logging.getLogger("pipeline")

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

    detected_bads = {}
    all_bad_channels = set()
    all_global_bad_epochs = set()  # Use sample indices for precision

    # Create filtered copy - improved memory management
    logger.info(f"Creating filtered copy for bad detection: {l_freq}-{h_freq} Hz"
                + (f", resample→{resample_hz} Hz" if resample_hz else ""))

    try:
        raw_filt_all = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose='ERROR')
        if resample_hz:
            original_sfreq = raw_filt_all.info['sfreq']
            raw_filt_all.resample(resample_hz, npad="auto")
            logger.info(f"Resampled from {original_sfreq:.1f} Hz to {resample_hz:.1f} Hz")
    except Exception as e:
        logger.error(f"Failed to create filtered copy: {e}")
        raise e

    # Helper: pick channels per type
    def _picks_for(typ):
        if typ == 'eeg':
            return mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        if typ == 'mag':
            return mne.pick_types(raw.info, meg='mag', eeg=False, exclude='bads')
        if typ == 'grad':
            return mne.pick_types(raw.info, meg='grad', eeg=False, exclude='bads')
        return np.array([], dtype=int)

    # Run detection per type
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
            f"Analyzing {len(ch_names)} {typ.upper()} channels: {ch_names[:5]}{'...' if len(ch_names) > 5 else ''}")

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

        # RANSAC (EEG only, deterministic)
        bads_ransac = []
        if typ == 'eeg':
            try:
                from autoreject import Ransac
                logger.info("Running RANSAC for EEG channels...")
                ransac = Ransac(n_jobs=1, random_state=42)
                ransac.fit(epochs)
                bads_ransac = list(ransac.bad_chs_)
                logger.info(f"RANSAC identified {len(bads_ransac)} bad EEG channels: {bads_ransac}")
            except Exception as e:
                logger.warning(f"RANSAC failed for EEG: {e}")
                bads_ransac = []

        # Local AutoReject on remaining "good" channels
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
                logger.info(f"Running AutoReject on {len(picks_good)} good {typ.upper()} channels...")

                epochs_good = epochs.copy().pick_channels(picks_good)
                # Get AutoReject configuration parameters
                ar_cfg = cfg.get("autoreject", {})
                ar = AutoReject(
                    n_interpolate=ar_cfg.get("n_interpolate", [0]),
                    cv=ar_cfg.get("cv_folds", 5),
                    thresh_method=ar_cfg.get("thresh_method", "random_search"),
                    consensus=None,
                    n_jobs=1,
                    random_state=42
                )
                ar.fit(epochs_good)
                reject_log = ar.get_reject_log(epochs_good)

                # Per-channel bads from reject_log
                bad_labels = []
                if hasattr(reject_log, "labels") and isinstance(reject_log.labels, (list, np.ndarray)):
                    for labs in reject_log.labels:
                        for lbl in labs:
                            if isinstance(lbl, str):
                                bad_labels.append(lbl)

                # Count and threshold by consensus
                n_epochs = len(reject_log.labels) if hasattr(reject_log, "labels") else len(epochs_good)
                bads_ar = set()
                if n_epochs > 0 and len(bad_labels) > 0:
                    counts = Counter(bad_labels)
                    bads_ar = {ch for ch, ct in counts.items() if (ct / n_epochs) > consensus_thresh}

                logger.info(
                    f"AutoReject consensus>{consensus_thresh:.2f} → {len(bads_ar)} {typ.upper()} bads: {sorted(bads_ar)}")

                bads_final = list(set(bads_ransac) | bads_ar)

                # Global bad epochs - IMPROVED: use sample indices for precision
                global_bad_epochs = []
                if hasattr(reject_log, "labels"):
                    rl = np.array([[bool(x) for x in row] for row in reject_log.labels], dtype=bool)
                    if rl.ndim == 2 and rl.shape[0] == n_epochs:
                        frac_bad_per_epoch = rl.mean(axis=1)
                        bad_epoch_indices = np.where(frac_bad_per_epoch > global_epoch_thresh)[0]

                        # Convert epoch indices to sample indices in ORIGINAL raw
                        for epoch_idx in bad_epoch_indices:
                            # Convert back to original raw sample indices
                            sample_start = int(events[epoch_idx, 0])  # Sample index in filtered raw
                            # Scale back to original sampling rate if needed
                            if resample_hz and raw.info['sfreq'] != resample_hz:
                                scale_factor = raw.info['sfreq'] / resample_hz
                                sample_start = int(sample_start * scale_factor)
                            global_bad_epochs.append(sample_start)

                logger.info(f"Global bad epochs for {typ.upper()}: {len(global_bad_epochs)} "
                            f"(threshold={global_epoch_thresh:.2f})")
                detected_bads[f"{typ}_global_bad_epochs"] = global_bad_epochs
                all_global_bad_epochs.update(global_bad_epochs)

            except Exception as e:
                logger.warning(f"AutoReject failed for {typ.upper()}: {e}")
                bads_final = bads_ransac
                detected_bads[f"{typ}_global_bad_epochs"] = []

        # Collect type-level bads
        bads_final = [ch for ch in bads_final if isinstance(ch, str)]
        detected_bads[typ] = bads_final
        all_bad_channels.update(bads_final)

    # Transfer results to ORIGINAL raw - IMPROVED annotation handling
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

    # Add NEW BAD_GlobalEpoch annotations using sample indices
    if len(all_global_bad_epochs) > 0:
        # Determine stable orig_time for annotations
        if raw.annotations is not None and raw.annotations.orig_time is not None:
            base_orig = raw.annotations.orig_time
        else:
            base_orig = raw.info.get('meas_date', None)

        # Convert sample indices to time (more precise than duration math)
        sfreq = raw.info['sfreq']
        onsets = []
        for sample_idx in sorted(set(all_global_bad_epochs)):
            time_sec = sample_idx / sfreq
            onsets.append(time_sec)

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

    # INTERACTIVE REVIEW with improved workflow
    if interactive:
        logger.info("🎮 Starting interactive review on filtered copy...")

        # Mirror bads and annotations to viewer copy
        raw_filt_all.info['bads'] = list(raw.info['bads'])

        # Show current BAD_GlobalEpochs in viewer
        if raw.annotations is not None and len(raw.annotations) > 0:
            # Scale annotations to filtered copy timing
            viewer_anns = []
            filt_sfreq = raw_filt_all.info['sfreq']
            orig_sfreq = raw.info['sfreq']
            scale_factor = filt_sfreq / orig_sfreq if orig_sfreq != filt_sfreq else 1.0

            for ann in raw.annotations:
                if ann['description'] == "BAD_GlobalEpoch":
                    scaled_onset = ann['onset'] * scale_factor
                    scaled_duration = ann['duration'] * scale_factor
                    viewer_anns.append({
                        'onset': scaled_onset,
                        'duration': scaled_duration,
                        'description': 'BAD_GlobalEpoch'
                    })

            if viewer_anns:
                if raw_filt_all.annotations is None:
                    raw_filt_all.set_annotations(Annotations(onset=[], duration=[], description=[]))

                view_orig = raw.annotations.orig_time if raw.annotations is not None else raw.info.get('meas_date',
                                                                                                       None)
                anns_view = Annotations(
                    onset=[a['onset'] for a in viewer_anns],
                    duration=[a['duration'] for a in viewer_anns],
                    description=[a['description'] for a in viewer_anns],
                    orig_time=view_orig
                )
                raw_filt_all.set_annotations(raw_filt_all.annotations + anns_view)

        # Enhanced interactive plotting with better error handling
        try:
            logger.info("🖥️  Opening interactive plot window...")
            logger.info("📋 Instructions:")
            logger.info("   - Click channels to mark/unmark as bad")
            logger.info("   - Click time segments to mark/unmark BAD_GlobalEpoch annotations")
            logger.info("   - Use scroll wheel to zoom in/out")
            logger.info("   - Close the plot window when done")

            # Show plot with reasonable defaults
            n_channels = min(32, len(raw_filt_all.ch_names))
            raw_filt_all.plot(
                n_channels=n_channels,
                duration=30.0,  # Show 30 seconds at a time
                scalings='auto',
                show=True,
                block=True,
                title=f"Interactive Review - {len(raw_filt_all.info['bads'])} bad channels"
            )

            # Wait for user confirmation
            print("\n" + "=" * 60)
            print("🎯 INTERACTIVE REVIEW COMPLETE")
            print("   - Close the plot window if you haven't already")
            print("   - Your changes will be saved to the checkpoint")
            print("=" * 60)
            response = input("Press Enter to continue with your changes (or 'abort' to cancel): ").strip().lower()

            if response == 'abort':
                logger.info("❌ User aborted interactive review - keeping original results")
                del raw_filt_all
                return detected_bads

        except Exception as e:
            logger.error(f"Interactive plotting failed: {e}")
            logger.info("⚠️  Continuing with automated results only")
            del raw_filt_all
            return detected_bads

        # Transfer user-reviewed bad channels back to ORIGINAL raw
        final_bads = list(raw_filt_all.info.get('bads', []))
        original_bads = set(raw.info.get('bads', []))
        reviewed_bads = set(final_bads)

        added_bads = reviewed_bads - original_bads
        removed_bads = original_bads - reviewed_bads

        if added_bads:
            logger.info(f"✅ User added bad channels: {sorted(added_bads)}")
        if removed_bads:
            logger.info(f"🔄 User rescued channels: {sorted(removed_bads)}")

        raw.info['bads'] = sorted(final_bads)

        # Transfer user-reviewed BAD_GlobalEpochs back to ORIGINAL raw
        _remove_bad_epoch_anns(raw)

        if raw_filt_all.annotations is not None and len(raw_filt_all.annotations) > 0:
            gb_annots = [ann for ann in raw_filt_all.annotations if ann['description'] == "BAD_GlobalEpoch"]

            if gb_annots:
                # Convert back to original timeline
                orig_sfreq = raw.info['sfreq']
                filt_sfreq = raw_filt_all.info['sfreq']
                scale_factor = orig_sfreq / filt_sfreq if filt_sfreq != orig_sfreq else 1.0

                final_onsets = []
                final_durations = []
                for ann in gb_annots:
                    scaled_onset = ann['onset'] * scale_factor
                    scaled_duration = ann['duration'] * scale_factor
                    final_onsets.append(scaled_onset)
                    final_durations.append(scaled_duration)

                if final_onsets:
                    base_orig = raw.annotations.orig_time if raw.annotations is not None else raw.info.get('meas_date',
                                                                                                           None)
                    anns_final = Annotations(
                        onset=final_onsets,
                        duration=final_durations,
                        description=["BAD_GlobalEpoch"] * len(final_onsets),
                        orig_time=base_orig
                    )

                    if raw.annotations is not None and len(raw.annotations) > 0:
                        combined = mne.Annotations(
                            onset=list(raw.annotations.onset) + list(anns_final.onset),
                            duration=list(raw.annotations.duration) + list(anns_final.duration),
                            description=list(raw.annotations.description) + list(anns_final.description),
                            orig_time=raw.annotations.orig_time
                        )
                        raw.set_annotations(combined)
                    else:
                        raw.set_annotations(anns_final)

                    logger.info(f"✅ Committed {len(anns_final)} user-reviewed BAD_GlobalEpoch annotations")
                else:
                    logger.info("🗑️  User removed all BAD_GlobalEpoch annotations")
            else:
                logger.info("🗑️  No BAD_GlobalEpoch annotations after user review")

        # Update detected_bads with final results for caller
        final_bad_epochs = []
        if raw.annotations is not None:
            sfreq = raw.info['sfreq']
            for ann in raw.annotations:
                if ann['description'] == "BAD_GlobalEpoch":
                    sample_idx = int(ann['onset'] * sfreq)
                    final_bad_epochs.append(sample_idx)

        for typ in which_types:
            detected_bads[f"{typ}_global_bad_epochs"] = sorted(set(final_bad_epochs))

        logger.info("🎉 Interactive review completed successfully!")

    # Cleanup filtered copy
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
