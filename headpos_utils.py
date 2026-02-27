"""
headpos_utils.py

Head-position utilities for MEGIN / Neuromag data.

This module centralizes all head-position logic:

- Reading .pos files safely
- Computing head_pos from cHPI
- Aligning head_pos times to (possibly cropped) Raw objects
- Computing movement statistics and destination poses
- Plotting head movement
- Handling a rich YAML configuration block, e.g.:

    head_position_processing:
      # Source of head position data
      source: "compute"      # "compute" | "file" | "coordinates" | "run"

      # If source="file": path to .pos file
      file_path: "/path/to/head_position.pos"

      # If source="coordinates": direct [x, y, z] in meters (static pose)
      coordinates: [-0.075, 0.012, 0.030]

      # If source="run": specify different run to use
      reference_run: 1       # use run-01's head position for this run

      # Destination for Maxwell filter (static reference pose)
      destination: "median"  # "median" | "mean" | "first" | "last"
                             # | "none" | "coordinates" | "reference"

      # If destination="coordinates": specific target position (HEAD, meters)
      destination_coordinates: [-0.075, 0.012, 0.030]

      # If destination="reference": use position from reference run
      destination_reference_run: 1

      # Movement compensation (dynamic head_pos in tSSS)
      movement_compensation: true   # true = use head_pos array for tSSS

      # Advanced options
      force_recompute: false        # recompute even if .pos exists (for compute source)
      write_subset: true            # save windowed .pos file for this run

Functions here are independent of your main preprocessing script.
You can gradually wire them in to simplify and de-clutter your pipeline.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Sequence, Callable, Dict, Any

from pathlib import Path

import numpy as np
import mne


# ---------------------------------------------------------------------------
# Basic I/O and naming for .pos files
# ---------------------------------------------------------------------------

def get_bids_headpos_path(
    subject: str,
    session: Optional[str],
    task: Optional[str],
    run: Optional[str],
    meg_dir: str,
) -> str:
    """Return the BIDS-style *_headpos.pos path for a given subject/session/task/run."""
    fname = f"sub-{subject}"
    if session:
        fname += f"_ses-{session}"
    if task:
        fname += f"_task-{task}"
    if run:
        fname += f"_run-{run}"
    fname += "_headpos.pos"
    return os.path.join(meg_dir, fname)


def read_head_pos_safe(pos_path: Optional[str], logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """Read a MEGIN .pos file safely.

    Parameters
    ----------
    pos_path : str or None
        Path to the .pos file, or None if no file is available.
    logger : logging.Logger | None
        Optional logger for info/warning messages.

    Returns
    -------
    head_pos : np.ndarray or None
        Array of shape (N, 10) (or at least (N, 1) for time), or None if not found.
    """
    def _log_info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    def _log_warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)

    if pos_path is None:
        _log_info("read_head_pos_safe: no .pos path provided → returning None.")
        return None
    if not os.path.exists(pos_path):
        _log_warn(f"read_head_pos_safe: .pos file does not exist: {pos_path}")
        return None

    try:
        hp = mne.chpi.read_head_pos(pos_path)
        if hp.ndim != 2 or hp.shape[1] < 1:
            _log_warn(
                f"read_head_pos_safe: unexpected head_pos shape {hp.shape}; "
                "expected (N, >=1). Returning None."
            )
            return None
        _log_info(
            f"read_head_pos_safe: loaded head_pos from {pos_path} with "
            f"{hp.shape[0]} samples, time range [{hp[0,0]:.3f}, {hp[-1,0]:.3f}] s."
        )
        return hp
    except Exception as e:
        _log_warn(f"read_head_pos_safe: failed to read head_pos from {pos_path}: {e}")
        return None


def compute_head_pos_from_raw(
    raw: mne.io.Raw,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Compute head position from cHPI signals in a Raw object.

    This wraps mne.chpi.* calls with simple logging and error handling.
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    try:
        _log("Computing head position from continuous HPI (cHPI)...")
        # Use the high-level convenience if available:
        try:
            head_pos = mne.chpi.compute_head_pos(raw)
        except TypeError:
            # Older MNE versions require info + chpi_locs:
            chpi_amps = mne.chpi.compute_chpi_amplitudes(raw)
            chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amps)
            head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)

        if head_pos.shape[0] == 0:
            _log("cHPI present but head_pos is empty. Skipping head movement correction.")
            return None
        return head_pos
    except Exception as e:
        if logger:
            logger.error(f"Failed to compute head position from cHPI: {e}")
        return None


def default_headpos_path_for_raw(raw: mne.io.Raw) -> str:
    """Derive a default *_headpos.pos path from the Raw's first filename."""
    raw_fname = raw.filenames[0] if hasattr(raw, "filenames") and raw.filenames else None
    if not raw_fname:
        raise ValueError("Cannot determine raw filename to derive output headpos path.")
    raw_base = os.path.splitext(os.path.basename(raw_fname))[0]
    if raw_base.endswith("_meg"):
        raw_base = raw_base[:-4]
    return os.path.join(os.path.dirname(raw_fname), f"{raw_base}_headpos.pos")


def write_head_pos_for_raw(
    raw: mne.io.Raw,
    head_pos: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Write head_pos array to a default *_headpos.pos path derived from Raw."""
    try:
        pos_path = default_headpos_path_for_raw(raw)
        mne.chpi.write_head_pos(pos_path, head_pos)
        if logger:
            logger.info(f"Computed head position saved to: {pos_path}")
        return pos_path
    except Exception as e:
        if logger:
            logger.error(f"Failed to write computed .pos file: {e}")
        return None


def get_head_pos_for_maxwell(
    raw: mne.io.Raw,
    pos_file: Optional[str] = None,
    compute_if_missing: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Backward-compatible wrapper to get a .pos path for Maxwell.

    - If pos_file is provided and exists, return it.
    - Else, if compute_if_missing=True, compute head_pos and write a .pos.
    - Else, return None.
    """
    if pos_file and os.path.isfile(pos_file):
        if logger:
            logger.info(f"Using user-specified head position file: {pos_file}")
        return pos_file

    if not compute_if_missing:
        if logger:
            logger.warning("No head pos file specified, and compute_if_missing is False.")
        return None

    head_pos = compute_head_pos_from_raw(raw, logger=logger)
    if head_pos is None:
        return None
    return write_head_pos_for_raw(raw, head_pos, logger=logger)


# ---------------------------------------------------------------------------
# Time-base alignment: the main workhorse
# ---------------------------------------------------------------------------

def align_headpos_to_cropped_raw(
    raw: mne.io.BaseRaw,
    head_pos: Optional[np.ndarray],
    mode: str = "absolute",
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Align a MEGIN head-position (.pos) array to a (possibly cropped) Raw.

    This function handles the MEGIN/MNE split between:
        - raw.times (relative, 0 → duration), and
        - raw.first_time (absolute, e.g. 643.5 s)

    It also handles the typical .pos time axis cases:
        - session-absolute times (e.g. 6.0 → 3652.88 s), or
        - run-relative times (e.g. 0.0 → 600.0 s) if they were post-processed.

    The goal is to ensure that head_pos[:, 0] is consistent with the Raw
    segment you will pass to mne.preprocessing.maxwell_filter().

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object (full or cropped).
    head_pos : np.ndarray or None
        The original head-pos array. If None, returns None.
    mode : {"absolute", "relative"}, default "absolute"
        "absolute":
            Return times in the absolute session frame that MNE uses internally
            (i.e., consistent with raw.first_time). This is what you should pass
            directly to maxwell_filter(head_pos=...).
        "relative":
            Return times re-based to the current Raw segment (0 → duration).
            Useful if you want to reason about movement in the cropped run’s
            own time frame.
    logger : logging.Logger | None
        If provided, logs details via logger.info().

    Returns
    -------
    head_pos_aligned : np.ndarray or None
        A copy of the head_pos array, aligned and clipped to this Raw window.
        If no samples overlap the Raw window, returns None.
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if head_pos is None:
        _log("align_headpos_to_cropped_raw: head_pos is None → returning None.")
        return None

    hp = np.array(head_pos, dtype=float, copy=True)
    if hp.ndim != 2 or hp.shape[1] < 1:
        raise ValueError(f"head_pos must be (N, >=1), got shape {hp.shape}.")

    # RAW timing info
    abs_start = float(raw.first_time)          # absolute session start of this Raw
    duration = float(raw.times[-1])            # relative duration
    abs_end = abs_start + duration             # absolute end
    hp_times = hp[:, 0]
    hp_min = float(hp_times.min())
    hp_max = float(hp_times.max())

    _log(
        f"align_headpos_to_cropped_raw: "
        f"raw.abs_window = [{abs_start:.3f}, {abs_end:.3f}] s; "
        f"hp.range = [{hp_min:.3f}, {hp_max:.3f}] s."
    )

    # Heuristic: decide whether hp is relative 0..T or already absolute
    margin = 1.0  # seconds
    hp_is_relative = (
        hp_min >= -margin and
        hp_max <= duration + margin and
        # also ensure it's not obviously extending far beyond this raw's abs window length
        hp_max <= (abs_end - abs_start) + margin
    )

    if hp_is_relative:
        # Relative times → shift into absolute session coordinates
        hp_abs = hp.copy()
        hp_abs[:, 0] += abs_start
        _log(
            "align_headpos_to_cropped_raw: detected RELATIVE hp; "
            f"shifted times by +{abs_start:.3f}s → "
            f"range [{hp_abs[:,0].min():.3f}, {hp_abs[:,0].max():.3f}] s."
        )
    else:
        hp_abs = hp
        _log(
            "align_headpos_to_cropped_raw: detected ABSOLUTE hp; "
            f"range [{hp_abs[:,0].min():.3f}, {hp_abs[:,0].max():.3f}] s remains."
        )

    # Clip to the current Raw’s absolute window
    eps = 1e-6
    mask = (hp_abs[:, 0] >= abs_start - eps) & (hp_abs[:, 0] <= abs_end + eps)
    n_before = len(hp_abs)
    n_after = int(mask.sum())
    if n_after == 0:
        _log(
            "align_headpos_to_cropped_raw: no head-pos samples overlap the raw window "
            f"[{abs_start:.3f}, {abs_end:.3f}] s → returning None."
        )
        return None

    if n_after < n_before:
        _log(
            f"align_headpos_to_cropped_raw: clipped {n_before - n_after} samples "
            f"outside [{abs_start:.3f}, {abs_end:.3f}] s."
        )

    hp_clip = hp_abs[mask].copy()
    # enforce monotonic time
    if not np.all(np.diff(hp_clip[:, 0]) >= 0):
        hp_clip = hp_clip[np.argsort(hp_clip[:, 0])]

    _log(
        "align_headpos_to_cropped_raw: final ABS-aligned hp range = "
        f"[{hp_clip[0,0]:.3f}, {hp_clip[-1,0]:.3f}] s (n={len(hp_clip)})."
    )

    if mode.lower() == "absolute":
        return hp_clip

    if mode.lower() == "relative":
        hp_rel = hp_clip.copy()
        hp_rel[:, 0] -= abs_start
        _log(
            "align_headpos_to_cropped_raw: converted to RELATIVE times; "
            f"range = [{hp_rel[0,0]:.3f}, {hp_rel[-1,0]:.3f}] s."
        )
        return hp_rel

    raise ValueError(f"mode must be 'absolute' or 'relative', got '{mode}'.")


# ---------------------------------------------------------------------------
# Movement statistics & plotting
# ---------------------------------------------------------------------------

def compute_head_movement_stats(head_pos: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Compute movement statistics from an aligned head-pos array.

    Parameters
    ----------
    head_pos : np.ndarray or None
        Head-pos array after alignment/clipping (N, 10). If None or empty,
        returns None.

    Returns
    -------
    stats : dict or None
        Nested dict with translation and rotation stats, suitable for logging
        or putting into a YAML manifest.
    """
    if head_pos is None or len(head_pos) == 0:
        return None

    hp = np.asarray(head_pos, float)
    if hp.shape[1] < 7:
        raise ValueError(f"head_pos must have at least 7 columns for movement stats; got {hp.shape}.")

    quats = hp[:, 1:4]  # Quaternions
    translations_mm = hp[:, 4:7] * 1000.0  # Translations (meters -> mm)

    # 1. Calculate relative translations (from the starting position of this run)
    rel_translations_mm = translations_mm - translations_mm[0]

    # Calculate max relative displacement (magnitude of the relative translation vector)
    max_disp = float(np.linalg.norm(rel_translations_mm, axis=1).max())

    # Calculate step-by-step cumulative displacement
    if len(translations_mm) > 1:
        step_disp = np.linalg.norm(np.diff(translations_mm, axis=0), axis=1)
        total_disp = float(step_disp.sum())
    else:
        total_disp = 0.0

    # 2. Convert quaternions to approximate rotation angles (degrees) for individual axes
    rotations_deg_axes = quats * 2.0 * (180.0 / np.pi)

    # Calculate relative rotations (from the starting orientation of this run)
    rel_rotations_deg_axes = rotations_deg_axes - rotations_deg_axes[0]

    # Calculate max relative rotation (magnitude of the relative rotation vector)
    max_rot = float(np.linalg.norm(rel_rotations_deg_axes, axis=1).max())

    # Calculate step-by-step cumulative rotation
    if len(rotations_deg_axes) > 1:
        step_rot = np.linalg.norm(np.diff(rotations_deg_axes, axis=0), axis=1)
        total_rot = float(step_rot.sum())
    else:
        total_rot = 0.0

    stats = {
        "n_timepoints": int(len(hp)),
        "time_range_sec": [float(hp[0, 0]), float(hp[-1, 0])],
        "translation_stats_mm": {
            "mean": [float(x) for x in translations_mm.mean(axis=0)],
            "std": [float(x) for x in translations_mm.std(axis=0)],
            "max_displacement": max_disp,  # Now relative to start
            "total_movement": total_disp,
        },
        "rotation_stats_deg": {
            "mean": [float(x) for x in rotations_deg_axes.mean(axis=0)],
            "std": [float(x) for x in rotations_deg_axes.std(axis=0)],
            "max_rotation": max_rot,  # Now relative to start
            "total_rotation": total_rot,
        },
    }
    return stats


def plot_head_movement(
        head_pos_array: Optional[np.ndarray],
        plots_dir: str,
        fname: str = "head_movement_over_time.png",
        logger: Optional[logging.Logger] = None,
) -> None:
    """Plot relative translation (mm) and rotation (deg) over time from a head_pos array."""
    import matplotlib.pyplot as plt

    def _log_warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)

    def _log_info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if head_pos_array is None or head_pos_array.shape[0] == 0:
        _log_warn("No head position data found to plot head movement.")
        return

    hp = np.asarray(head_pos_array, float)
    t = hp[:, 0]

    # Calculate relative rotation (zeroed at start)
    rot = hp[:, 1:4] * 2.0 * (180.0 / np.pi)  # Quaternions approx to degrees
    rot = rot - rot[0]

    # Calculate relative translation (zeroed at start)
    translations = hp[:, 4:7] * 1000.0  # Translations (meters -> mm)
    translations = translations - translations[0]
    x, y, z = translations.T

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, x, label="X")
    axes[0].plot(t, y, label="Y")
    axes[0].plot(t, z, label="Z")
    axes[0].set_ylabel("Relative Translation (mm)")
    axes[0].legend()
    axes[0].set_title("Head Translation Relative to Start")

    axes[1].plot(t, rot[:, 0], label="X rot")
    axes[1].plot(t, rot[:, 1], label="Y rot")
    axes[1].plot(t, rot[:, 2], label="Z rot")
    axes[1].set_ylabel("Relative Rotation (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].set_title("Head Rotation Relative to Start")

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    _log_info(f"Head movement plot saved to {out_path}")

# ---------------------------------------------------------------------------
# Destination pose (static reference for movement comp)
# ---------------------------------------------------------------------------

def compute_destination_from_pos(
        head_pos: Optional[np.ndarray],
        strategy: str = "median",
        logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Compute a static destination pose from head-pos data.

    Parameters
    ----------
    head_pos : np.ndarray or None
        Aligned head-pos array in ABS or REL frame. Only the translations
        (columns 4:7) are used.
    strategy : {"median", "mean", "first", "last"}, default "median"
        How to pick the destination:
            - "median": robust median translation over time.
            - "mean":   mean translation over time.
            - "first":  translation at the first time point.
            - "last":   translation at the last time point.
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    dest : np.ndarray or None, shape (3,)
        A 3-vector in meters (HEAD coords) or None if head_pos is None/empty
        or strategy is unrecognized.
    """

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if head_pos is None or len(head_pos) == 0:
        _log("compute_destination_from_pos: no head_pos available -> destination=None.")
        return None

    hp = np.asarray(head_pos, float)
    # FIX: Ensure we have at least 7 columns to access the translations (4:7)
    if hp.shape[1] < 7:
        _log(
            f"compute_destination_from_pos: head_pos has shape {hp.shape}; "
            "need at least columns 4:7 for translations → destination=None."
        )
        return None

    strategy = strategy.lower()

    # FIX: Corrected all indices from 1:4 (quaternions) to 4:7 (translations)
    if strategy == "median":
        dest = np.median(hp[:, 4:7], axis=0)
    elif strategy == "mean":
        dest = hp[:, 4:7].mean(axis=0)
    elif strategy == "first":
        dest = hp[0, 4:7]
    elif strategy == "last":
        dest = hp[-1, 4:7]
    else:
        _log(f"compute_destination_from_pos: unknown strategy '{strategy}' → destination=None.")
        return None

    _log(f"compute_destination_from_pos: strategy={strategy}, destination={dest} (m).")
    return dest


# ---------------------------------------------------------------------------
# Config-driven preparation: integrates YAML semantics
# ---------------------------------------------------------------------------

def prepare_headpos_from_config(
    raw: mne.io.BaseRaw,
    cfg: Dict[str, Any],
    *,
    default_pos_path: Optional[str] = None,
    # optional callback to compute head_pos from Raw (e.g., mne.chpi.compute_head_pos)
    compute_headpos_fn: Optional[Callable[[mne.io.BaseRaw], np.ndarray]] = None,
    # optional callback: given a reference_run, return a head_pos array (absolute time)
    load_reference_headpos_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    # optional callback: save subset .pos for this run and return its path
    save_subset_fn: Optional[Callable[[np.ndarray], str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    High-level head-position handler driven by the 'head_position_processing' YAML block.

    Supports:

        head_position_processing:
          source: "compute" | "file" | "coordinates" | "run"

          file_path: "/path/to/head_position.pos"

          coordinates: [-0.075, 0.012, 0.030]

          reference_run: 1  # for source="run"

          destination: "median" | "mean" | "first" | "last"
                       | "none" | "coordinates" | "reference"

          destination_coordinates: [-0.075, 0.012, 0.030]

          destination_reference_run: 1

          movement_compensation: true | false

          force_recompute: false

          write_subset: true

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object for this run (may be cropped).
    cfg : dict
        The 'head_position_processing' configuration dict.
    default_pos_path : str or None
        Optional default .pos path for this run (e.g., discovered from BIDS).
    compute_headpos_fn : callable or None
        Function to compute head_pos from Raw: hp = compute_headpos_fn(raw).
        If None, uses compute_head_pos_from_raw().
    load_reference_headpos_fn : callable or None
        Function that, given an integer reference_run, returns a head_pos array
        for that run (absolute session time) or None.
    save_subset_fn : callable or None
        Function that, given an aligned head_pos array for this run, saves a
        window-specific .pos file and returns its path (str).
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    result : dict with keys:
        - head_pos_abs : np.ndarray or None
            Aligned head_pos in absolute time (for maxwell_filter).
        - head_pos_rel : np.ndarray or None
            Aligned head_pos in relative time (0 → duration), or None.
        - destination : np.ndarray or None
            3-vector (meters, HEAD coords) for static destination, or None.
        - subset_path : str or None
            Path to a subset .pos file if write_subset and save_subset_fn were used.
        - movement_enabled : bool
            Whether movement compensation (head_pos) should be used for tSSS.
        - source_used : str
            Which source branch was selected ("compute", "file", "coordinates", "run").
        - destination_mode : str
            Effective destination mode used.
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    source = str(cfg.get("source", "compute")).lower()
    movement_comp = bool(cfg.get("movement_compensation", True))
    force_recompute = bool(cfg.get("force_recompute", False))
    write_subset = bool(cfg.get("write_subset", False))

    # -------- HEAD POS: choose source --------
    hp_raw: Optional[np.ndarray] = None
    source_used = source

    if source == "compute":
        # Compute from this Raw, optionally ignoring existing Pos file.
        if compute_headpos_fn is None:
            # Use our wrapper which handles different MNE versions.
            compute_headpos_fn = compute_head_pos_from_raw
        _log(f"[headpos] source='compute' (force_recompute={force_recompute}).")
        hp_raw = compute_headpos_fn(raw)

    elif source == "file":
        pos_path = cfg.get("file_path") or default_pos_path
        _log(f"[headpos] source='file', path={pos_path!r}.")
        hp_raw = read_head_pos_safe(pos_path, logger=logger)

    elif source == "coordinates":
        # No dynamic head_pos; treat the coordinates as a static pose only.
        _log("[headpos] source='coordinates' → no dynamic head_pos (movement_compensation will be disabled).")
        hp_raw = None
        movement_comp = False  # cannot do dynamic movement without a time series

    elif source == "run":
        ref_run = int(cfg.get("reference_run", 1))
        _log(f"[headpos] source='run', reference_run={ref_run}.")
        if load_reference_headpos_fn is None:
            _log("[headpos] WARNING: load_reference_headpos_fn is None; cannot load reference run. head_pos=None.")
            hp_raw = None
        else:
            hp_raw = load_reference_headpos_fn(ref_run)
            if hp_raw is None:
                _log("[headpos] WARNING: reference head_pos is None; no movement comp possible for this run.")
    else:
        _log(f"[headpos] WARNING: unknown source='{source}'; treating as no head_pos.")
        hp_raw = None
        source_used = "unknown"

    # -------- ALIGN TO THIS RAW --------
    hp_abs = align_headpos_to_cropped_raw(raw, hp_raw, mode="absolute", logger=logger)
    hp_rel = align_headpos_to_cropped_raw(raw, hp_raw, mode="relative", logger=logger) if hp_abs is not None else None

    subset_path = None
    if write_subset and hp_abs is not None and save_subset_fn is not None:
        try:
            subset_path = save_subset_fn(hp_abs)
            _log(f"[headpos] write_subset=True → wrote subset .pos to {subset_path}")
        except Exception as e:
            _log(f"[headpos] WARNING: write_subset failed: {e}")

    # The head_pos series we actually want to feed to tSSS
    head_pos_for_tsss = hp_abs if movement_comp and hp_abs is not None else None
    if not movement_comp:
        _log("[headpos] movement_compensation = False → head_pos_for_tsss=None.")

    # -------- DESTINATION POSE --------
    dest_cfg = cfg.get("destination", "median")
    dest_mode = str(dest_cfg).lower()
    destination: Optional[np.ndarray] = None

    if dest_mode in ("median", "mean", "first", "last"):
        destination = compute_destination_from_pos(hp_abs, strategy=dest_mode, logger=logger)

    elif dest_mode == "coordinates":
        coords = cfg.get("destination_coordinates", None)
        if coords is None:
            _log("[headpos] destination='coordinates' but no destination_coordinates set → destination=None.")
            destination = None
        else:
            destination = np.array(coords, float).reshape(3)
            _log(f"[headpos] destination='coordinates' → {destination} (m).")

    elif dest_mode == "reference":
        ref_run = int(cfg.get("destination_reference_run", cfg.get("reference_run", 1)))
        _log(f"[headpos] destination='reference' from run {ref_run}.")
        if load_reference_headpos_fn is None:
            _log("[headpos] WARNING: load_reference_headpos_fn is None; cannot load reference destination.")
            destination = None
        else:
            hp_ref = load_reference_headpos_fn(ref_run)
            if hp_ref is None or len(hp_ref) == 0:
                _log("[headpos] WARNING: reference head_pos is None/empty; destination=None.")
                destination = None
            else:
                destination = compute_destination_from_pos(hp_ref, strategy="median", logger=logger)

    elif dest_mode in ("none", "null"):
        _log("[headpos] destination='none' → no static destination pose.")
        destination = None

    else:
        _log(f"[headpos] WARNING: unknown destination='{dest_mode}' → destination=None.")
        destination = None

    result = {
        "head_pos_abs": head_pos_for_tsss,  # alignment done, absolute time base
        "head_pos_rel": hp_rel,             # relative (0 → duration) if useful
        "destination": destination,         # static destination pose (HEAD meters) or None
        "subset_path": subset_path,         # path to subset .pos if written
        "movement_enabled": bool(head_pos_for_tsss is not None),
        "source_used": source_used,
        "destination_mode": dest_mode,
    }
    return result
