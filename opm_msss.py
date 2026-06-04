#!/usr/bin/env python3
"""Experimental mSSS backend for Cerca/QuSpin OPM data.

This module implements a Python-first multi-origin SSS (mSSS) workflow for
triaxial OPM MEG data by combining two single-origin SSS internal bases using
SVD, mirroring the high-level algorithm in McPherson et al. (2025) and their
reference MATLAB repository.

Design goals
------------
1. Keep the production pipeline in Python/MNE-Python.
2. Avoid a full immediate port of the repository's low-level MATLAB basis code.
3. Allow optional MATLAB round-trip validation against the authors' repo.

Important caveat
----------------
The basis engine used here is MNE-Python's public ``compute_maxwell_basis``
function. That makes this implementation convenient and reproducible inside a
Python pipeline, but it should still be treated as experimental for non-
Neuromag systems.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.preprocessing import compute_maxwell_basis
from mne.transforms import apply_trans, invert_transform
from scipy.io import loadmat, savemat


POINT_MAGNETOMETER_COIL = int(FIFF.FIFFV_COIL_POINT_MAGNETOMETER)
COIL_NONE = int(FIFF.FIFFV_COIL_NONE)


@dataclass
class MSSSConfig:
    """Configuration for the experimental Python mSSS backend.

    Parameters
    ----------
    center1, center2
        The two interior expansion centers.
    centers_frame
        Coordinate frame for ``center1`` and ``center2``. Use ``'meg'`` when the
        centers are already in device/MEG coordinates. Use ``'head'`` when the
        centers came from MRI/head-space optimization.
    int_order, ext_order
        SSS internal/external harmonic orders.
    threshold
        Singular-value ratio threshold used after concatenating the two internal
        bases. The paper/repository default is 0.005.
    mag_scale
        Magnetometer scale used by MNE when building the SSS basis.
    regularize
        Passed to ``mne.preprocessing.compute_maxwell_basis``. ``None`` is the
        closest match to the repository's direct pseudoinverse workflow.
    bad_condition
        How to handle ill-conditioned SSS bases in MNE.
    ignore_ref
        Ignore reference MEG channels. This should stay ``True`` for Cerca OPM.
    patch_point_magnetometers
        Patch MEG channels with missing coil definitions to point magnetometers.
    out_origin_meg
        External basis origin in MEG/device coordinates. The paper repository
        uses the device origin, i.e. (0, 0, 0).
    pinv_rcond
        Optional rcond passed to ``numpy.linalg.pinv``.
    """

    center1: Sequence[float]
    center2: Sequence[float]
    centers_frame: str = "meg"
    int_order: int = 8
    ext_order: int = 3
    threshold: float = 0.005
    mag_scale: float = 100.0
    regularize: Optional[str] = None
    bad_condition: str = "warning"
    ignore_ref: bool = True
    patch_point_magnetometers: bool = True
    out_origin_meg: Sequence[float] = (0.0, 0.0, 0.0)
    pinv_rcond: Optional[float] = None

    def validate(self) -> None:
        """Validate configuration values."""
        self.center1 = _as_xyz(self.center1, "center1")
        self.center2 = _as_xyz(self.center2, "center2")
        self.out_origin_meg = _as_xyz(self.out_origin_meg, "out_origin_meg")
        if self.centers_frame not in {"meg", "head"}:
            raise ValueError("centers_frame must be 'meg' or 'head'.")
        if self.int_order < 1:
            raise ValueError("int_order must be >= 1.")
        if self.ext_order < 0:
            raise ValueError("ext_order must be >= 0.")
        if not (0.0 < float(self.threshold) <= 1.0):
            raise ValueError("threshold must be in (0, 1].")
        if self.regularize not in {None, "in"}:
            raise ValueError("regularize must be None or 'in'.")
        if self.bad_condition not in {"error", "warning", "info", "ignore"}:
            raise ValueError(
                "bad_condition must be one of 'error', 'warning', 'info', 'ignore'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary dictionary."""
        self.validate()
        out = asdict(self)
        for key in ("center1", "center2", "out_origin_meg"):
            out[key] = [float(x) for x in out[key]]
        return out


@dataclass
class MSSSResult:
    """Summary of the computed mSSS basis and reconstruction."""

    good_picks: List[int]
    bad_names: List[str]
    good_names: List[str]
    centers_meg: np.ndarray
    out_origin_meg: np.ndarray
    s_in: np.ndarray
    s_out: np.ndarray
    s_full: np.ndarray
    p_full: np.ndarray
    singular_values: np.ndarray
    keep_mask: np.ndarray
    n_in1: int
    n_in2: int
    n_in_kept: int
    config: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        """Return a compact JSON-safe summary."""
        return {
            "good_picks": self.good_picks,
            "bad_names": self.bad_names,
            "good_names": self.good_names,
            "centers_meg": self.centers_meg.tolist(),
            "out_origin_meg": self.out_origin_meg.tolist(),
            "singular_values": self.singular_values.tolist(),
            "keep_mask": self.keep_mask.tolist(),
            "n_in1": self.n_in1,
            "n_in2": self.n_in2,
            "n_in_kept": self.n_in_kept,
            "config": self.config,
        }


@dataclass
class CenterSuggestion:
    """Heuristic two-center suggestion for mSSS origin placement."""

    centers: np.ndarray
    centers_frame: str
    centers_meg: np.ndarray
    centers_head: np.ndarray
    n_unique_sensor_positions: int
    heuristic: Dict[str, float]

    def summary(self) -> Dict[str, Any]:
        return {
            "centers_frame": self.centers_frame,
            "centers": self.centers.tolist(),
            "centers_meg": self.centers_meg.tolist(),
            "centers_head": self.centers_head.tolist(),
            "n_unique_sensor_positions": int(self.n_unique_sensor_positions),
            "heuristic": {k: float(v) for k, v in self.heuristic.items()},
        }


def _as_xyz(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 elements, got shape {arr.shape}.")
    return arr


def patch_cerca_opm_coil_types(raw: mne.io.BaseRaw, *, replace_all: bool = False) -> int:
    """Patch missing Cerca OPM coil types to point magnetometers.

    Parameters
    ----------
    raw
        Raw object to patch in-place.
    replace_all
        If ``False`` (default), only channels with ``coil_type == FIFFV_COIL_NONE``
        are patched. If ``True``, all MEG channels are forced to point
        magnetometers.

    Returns
    -------
    int
        Number of channels patched.
    """
        
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    patched = 0
    for pick in meg_picks:
        coil_type = int(raw.info["chs"][pick].get("coil_type", COIL_NONE))
        if replace_all or coil_type == COIL_NONE:
            raw.info["chs"][pick]["coil_type"] = POINT_MAGNETOMETER_COIL
            patched += 1
    return patched


def summarize_opm_geometry(raw: mne.io.BaseRaw, *, decimals: int = 6) -> Dict[str, Any]:
    """Summarize OPM geometry quality from ``info['chs'][*]['loc']``.

    This is intended as a first-pass intake check for Cerca FIF exports.
    """
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    positions = []
    sensing = []
    missing = []
    nonunit = []
    for pick in meg_picks:
        ch = raw.info["chs"][pick]
        loc = np.asarray(ch["loc"][:12], dtype=float)
        pos = loc[:3]
        ez = loc[9:12]
        positions.append(pos)
        sensing.append(ez)
        if not np.isfinite(loc).all() or np.allclose(pos, 0.0) or np.allclose(loc[3:12], 0.0):
            missing.append(raw.ch_names[pick])
        norm_ez = np.linalg.norm(ez)
        if norm_ez and not np.isclose(norm_ez, 1.0, atol=1e-2):
            nonunit.append(raw.ch_names[pick])
    positions_arr = np.round(np.array(positions), decimals=decimals)
    if len(positions_arr):
        _, counts = np.unique(positions_arr, axis=0, return_counts=True)
        uniq_counts, uniq_freqs = np.unique(counts, return_counts=True)
        coloc_hist = {int(c): int(n) for c, n in zip(uniq_counts, uniq_freqs)}
    else:
        coloc_hist = {}
    return {
        "n_meg": int(len(meg_picks)),
        "n_missing_geometry": int(len(missing)),
        "missing_geometry_channels": missing,
        "n_nonunit_sensing": int(len(nonunit)),
        "nonunit_sensing_channels": nonunit,
        "co_located_position_histogram": coloc_hist,
    }


def extract_sensor_geometry(
    raw: mne.io.BaseRaw,
    *,
    picks: Optional[Sequence[int]] = None,
    frame: str = "meg",
    z_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract sensor positions and local axes from an MNE Raw object.

    Parameters
    ----------
    raw
        Input raw object.
    picks
        Channel picks. Defaults to all non-reference MEG channels.
    frame
        ``'meg'`` or ``'head'``.
    z_offset
        Optional shift of sensor positions along each channel's sensing axis.

    Returns
    -------
    R, EX, EY, EZ : ndarray
        Arrays with shape ``(3, n_channels)``.
    """
    if frame not in {"meg", "head"}:
        raise ValueError("frame must be 'meg' or 'head'.")
    if picks is None:
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    picks = list(map(int, picks))

    n_chan = len(picks)
    R = np.zeros((3, n_chan), dtype=float)
    EX = np.zeros((3, n_chan), dtype=float)
    EY = np.zeros((3, n_chan), dtype=float)
    EZ = np.zeros((3, n_chan), dtype=float)

    transform = None
    if frame == "head":
        if raw.info.get("dev_head_t") is None:
            raise ValueError(
                "Cannot extract head-frame geometry because raw.info['dev_head_t'] is missing."
            )
        transform = raw.info["dev_head_t"]["trans"]

    for out_idx, pick in enumerate(picks):
        loc = np.asarray(raw.info["chs"][pick]["loc"][:12], dtype=float)
        pos = loc[:3]
        ex = loc[3:6]
        ey = loc[6:9]
        ez = loc[9:12]
        if transform is not None:
            pos_h = np.ones(4, dtype=float)
            pos_h[:3] = pos
            pos = (transform @ pos_h)[:3]
            rot = transform[:3, :3]
            ex = rot @ ex
            ey = rot @ ey
            ez = rot @ ez
        pos = pos + float(z_offset) * ez
        R[:, out_idx] = pos
        EX[:, out_idx] = ex
        EY[:, out_idx] = ey
        EZ[:, out_idx] = ez
    return R, EX, EY, EZ


def centers_to_meg(raw_info: Mapping[str, Any], centers: np.ndarray, centers_frame: str) -> np.ndarray:
    """Convert centers from head frame to MEG frame when needed."""
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    if centers.shape[1] != 3:
        raise ValueError(f"centers must have shape (n, 3), got {centers.shape}.")
    if centers_frame == "meg":
        return centers.copy()
    if centers_frame != "head":
        raise ValueError("centers_frame must be 'meg' or 'head'.")
    dev_head_t = raw_info.get("dev_head_t")
    if dev_head_t is None:
        raise ValueError(
            "centers are in head coordinates, but raw.info['dev_head_t'] is missing."
        )
    head_to_meg = invert_transform(dev_head_t)
    return apply_trans(head_to_meg, centers)


def _centers_meg_to_head(raw_info: Mapping[str, Any], centers_meg: np.ndarray) -> np.ndarray:
    centers_meg = np.atleast_2d(np.asarray(centers_meg, dtype=float))
    if centers_meg.shape[1] != 3:
        raise ValueError(f"centers_meg must have shape (n, 3), got {centers_meg.shape}.")
    dev_head_t = raw_info.get("dev_head_t")
    if dev_head_t is None:
        raise ValueError("raw.info['dev_head_t'] is required to convert MEG centers to head coordinates.")
    return apply_trans(dev_head_t, centers_meg)



def suggest_two_msss_centers(
    raw: mne.io.BaseRaw,
    *,
    frame: str = "head",
    ap_fraction: float = 0.22,
    depth_fraction: float = 0.25,
    min_separation: float = 0.02,
    max_separation: float = 0.06,
    min_depth: float = 0.02,
    max_depth: float = 0.045,
) -> CenterSuggestion:
    """Suggest two interior mSSS origins from the OPM sensor cloud.

    The heuristic places two origins along the anterior-posterior axis of the
    unique physical sensor locations and shifts them inward from the mean sensor
    height to avoid placing the origins too close to the helmet.
    """
    if frame not in {"head", "meg"}:
        raise ValueError("frame must be 'head' or 'meg'.")

    R, _, _, _ = extract_sensor_geometry(raw, frame=frame)
    pos = R.T
    pos_unique = np.unique(np.round(pos, decimals=6), axis=0)
    if pos_unique.shape[0] < 2:
        raise ValueError("Need at least two unique sensor positions to suggest mSSS centers.")

    mins = pos_unique.min(axis=0)
    maxs = pos_unique.max(axis=0)
    spans = maxs - mins
    center = pos_unique.mean(axis=0)

    ap_sep = float(np.clip(ap_fraction * spans[1], min_separation, max_separation))
    depth = float(np.clip(depth_fraction * spans[2], min_depth, max_depth))

    center1 = center.copy()
    center2 = center.copy()
    center1[1] -= ap_sep / 2.0
    center2[1] += ap_sep / 2.0
    center1[2] -= depth
    center2[2] -= depth

    centers = np.vstack([center1, center2])
    if frame == "head":
        centers_head = centers.copy()
        centers_meg = centers_to_meg(raw.info, centers, "head")
    else:
        centers_meg = centers.copy()
        centers_head = _centers_meg_to_head(raw.info, centers)

    return CenterSuggestion(
        centers=centers,
        centers_frame=frame,
        centers_meg=centers_meg,
        centers_head=centers_head,
        n_unique_sensor_positions=int(pos_unique.shape[0]),
        heuristic={
            "x_center": float(center[0]),
            "y_center": float(center[1]),
            "z_center": float(center[2]),
            "x_span": float(spans[0]),
            "y_span": float(spans[1]),
            "z_span": float(spans[2]),
            "ap_separation": float(ap_sep),
            "depth": float(depth),
        },
    )


def _column_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a basis containing zero-norm columns.")
    return matrix / norms[np.newaxis, :]


def _pick_good_meg_channels(raw: mne.io.BaseRaw) -> Tuple[List[int], List[str], List[str]]:
    meg_picks = list(map(int, mne.pick_types(raw.info, meg=True, ref_meg=False)))
    bads = set(raw.info.get("bads", []))
    good_picks = [pick for pick in meg_picks if raw.ch_names[pick] not in bads]
    good_names = [raw.ch_names[pick] for pick in good_picks]
    bad_names = [name for name in raw.info.get("bads", []) if name in {raw.ch_names[p] for p in meg_picks}]
    if not good_picks:
        raise RuntimeError("No good MEG channels remain after excluding raw.info['bads'].")
    return good_picks, good_names, bad_names


def _compute_normalized_sss_basis(
    info: mne.Info,
    *,
    origin_meg: Sequence[float],
    int_order: int,
    ext_order: int,
    regularize: Optional[str],
    bad_condition: str,
    ignore_ref: bool,
    mag_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized internal/external SSS bases using MNE's public API."""
    S, _pS, _reg_moments, n_use_in = compute_maxwell_basis(
        info,
        origin=np.asarray(origin_meg, dtype=float),
        int_order=int(int_order),
        ext_order=int(ext_order),
        calibration=None,
        coord_frame="meg",
        regularize=regularize,
        ignore_ref=ignore_ref,
        bad_condition=bad_condition,
        mag_scale=float(mag_scale),
    )
    s_in = _column_normalize(np.asarray(S[:, :n_use_in], dtype=float))
    s_out = _column_normalize(np.asarray(S[:, n_use_in:], dtype=float))
    return s_in, s_out


def compute_python_msss_basis(raw: mne.io.BaseRaw, config: MSSSConfig) -> MSSSResult:
    """Compute the experimental Python mSSS basis for a Raw object.

    Notes
    -----
    The algorithm is:

    1. Compute two single-origin internal SSS bases.
    2. Column-normalize them.
    3. Concatenate and SVD the internal bases.
    4. Keep columns with ``sigma_i / sigma_1 >= threshold``.
    5. Combine with a single external basis at the device origin.
    6. Use a pseudoinverse to reconstruct the internal signal.
    """
    config.validate()

    # Work on a copy of info only; the raw data are handled separately.
    info = raw.info.copy()
    if config.patch_point_magnetometers:
        temp_raw = raw.copy()
        patch_cerca_opm_coil_types(temp_raw)
        info = temp_raw.info.copy()

    good_picks, good_names, bad_names = _pick_good_meg_channels(raw if not config.patch_point_magnetometers else temp_raw)
    info_good = mne.pick_info(info, good_picks, copy=True)

    centers_meg = centers_to_meg(
        info,
        np.vstack([_as_xyz(config.center1, "center1"), _as_xyz(config.center2, "center2")]),
        config.centers_frame,
    )
    out_origin_meg = _as_xyz(config.out_origin_meg, "out_origin_meg")

    s_in1, _ = _compute_normalized_sss_basis(
        info_good,
        origin_meg=centers_meg[0],
        int_order=config.int_order,
        ext_order=config.ext_order,
        regularize=config.regularize,
        bad_condition=config.bad_condition,
        ignore_ref=config.ignore_ref,
        mag_scale=config.mag_scale,
    )
    s_in2, _ = _compute_normalized_sss_basis(
        info_good,
        origin_meg=centers_meg[1],
        int_order=config.int_order,
        ext_order=config.ext_order,
        regularize=config.regularize,
        bad_condition=config.bad_condition,
        ignore_ref=config.ignore_ref,
        mag_scale=config.mag_scale,
    )
    _, s_out = _compute_normalized_sss_basis(
        info_good,
        origin_meg=out_origin_meg,
        int_order=config.int_order,
        ext_order=config.ext_order,
        regularize=config.regularize,
        bad_condition=config.bad_condition,
        ignore_ref=config.ignore_ref,
        mag_scale=config.mag_scale,
    )

    s_cat = np.concatenate([s_in1, s_in2], axis=1)
    u, singular_values, _vh = np.linalg.svd(s_cat, full_matrices=False)
    if singular_values.size == 0:
        raise RuntimeError("SVD of the concatenated internal basis returned no singular values.")
    keep_mask = singular_values / singular_values[0] >= float(config.threshold)
    if not np.any(keep_mask):
        raise RuntimeError(
            "No singular vectors survived the threshold. Lower the threshold or inspect geometry."
        )
    s_in = u[:, keep_mask]
    s_full = np.concatenate([s_in, s_out], axis=1)
    if config.pinv_rcond is None:
        p_full = np.linalg.pinv(s_full)
    else:
        p_full = np.linalg.pinv(s_full, rcond=float(config.pinv_rcond))

    return MSSSResult(
        good_picks=good_picks,
        bad_names=bad_names,
        good_names=good_names,
        centers_meg=centers_meg,
        out_origin_meg=out_origin_meg,
        s_in=s_in,
        s_out=s_out,
        s_full=s_full,
        p_full=p_full,
        singular_values=singular_values,
        keep_mask=keep_mask,
        n_in1=s_in1.shape[1],
        n_in2=s_in2.shape[1],
        n_in_kept=int(s_in.shape[1]),
        config=config.to_dict(),
    )


def apply_python_msss(
    raw: mne.io.BaseRaw,
    config: MSSSConfig,
    *,
    copy: bool = True,
    annotate_description: bool = True,
) -> Tuple[mne.io.BaseRaw, MSSSResult]:
    """Apply the experimental Python mSSS transform to Raw data.

    Parameters
    ----------
    raw
        Input raw object.
    config
        mSSS configuration.
    copy
        If ``True`` (default), operate on a copy.
    annotate_description
        If ``True``, append a short provenance note to ``raw.info['description']``.
    """
    config.validate()
    inst = raw.copy() if copy else raw
    inst.load_data()
    if config.patch_point_magnetometers:
        patch_cerca_opm_coil_types(inst)
    result = compute_python_msss_basis(inst, config)

    data_good = inst.get_data(picks=result.good_picks)
    moments = result.p_full @ data_good
    data_clean_good = np.real(result.s_in @ moments[: result.n_in_kept, :])
    inst._data[np.array(result.good_picks)] = data_clean_good

    if annotate_description:
        desc = inst.info.get("description") or ""
        tag = (
            f"python-mSSS(int={config.int_order}, ext={config.ext_order}, "
            f"thresh={config.threshold}, n_keep={result.n_in_kept})"
        )
        inst.info["description"] = f"{desc} | {tag}" if desc else tag

    return inst, result


def write_matlab_bridge_payload(
    raw: mne.io.BaseRaw,
    output_mat: Path | str,
    config: MSSSConfig,
    *,
    include_bad_meg: bool = False,
    write_manifest: bool = True,
) -> Dict[str, Any]:
    """Write a MATLAB bridge payload for the reference repository.

    The exported geometry is already extracted from MNE, so the MATLAB side only
    needs to call ``multi_sss.m`` and reconstruct the data.
    """
    config.validate()
    inst = raw.copy()
    inst.load_data()
    if config.patch_point_magnetometers:
        patch_cerca_opm_coil_types(inst)

    meg_picks = list(map(int, mne.pick_types(inst.info, meg=True, ref_meg=False)))
    if include_bad_meg:
        export_picks = meg_picks
    else:
        export_picks = [pick for pick in meg_picks if inst.ch_names[pick] not in set(inst.info.get("bads", []))]
    if not export_picks:
        raise RuntimeError("No MEG channels selected for MATLAB export.")

    centers_meg = centers_to_meg(
        inst.info,
        np.vstack([_as_xyz(config.center1, "center1"), _as_xyz(config.center2, "center2")]),
        config.centers_frame,
    )
    R, EX, EY, EZ = extract_sensor_geometry(inst, picks=export_picks, frame="meg")
    ch_types = np.ones((len(export_picks), 1), dtype=float)  # Cerca OPM: all mags
    payload = {
        "data": inst.get_data(picks=export_picks),
        "R": R,
        "EX": EX,
        "EY": EY,
        "EZ": EZ,
        "center1": centers_meg[0].reshape(3, 1),
        "center2": centers_meg[1].reshape(3, 1),
        "Lin": np.array([[int(config.int_order)]], dtype=float),
        "Lout": np.array([[int(config.ext_order)]], dtype=float),
        "thresh": np.array([[float(config.threshold)]], dtype=float),
        "ch_types": ch_types,
        "sfreq": np.array([[float(inst.info["sfreq"])]], dtype=float),
        "times": np.asarray(inst.times, dtype=float).reshape(1, -1),
        "out_origin_meg": _as_xyz(config.out_origin_meg, "out_origin_meg").reshape(3, 1),
    }
    output_mat = Path(output_mat)
    output_mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_mat), payload, do_compression=True)

    manifest = {
        "mat_file": str(output_mat),
        "channel_indices": export_picks,
        "channel_names": [inst.ch_names[p] for p in export_picks],
        "bad_names": [name for name in inst.info.get("bads", []) if name in {inst.ch_names[p] for p in meg_picks}],
        "config": config.to_dict(),
    }
    if write_manifest:
        manifest_path = output_mat.with_suffix(output_mat.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest["manifest_file"] = str(manifest_path)
    return manifest


def read_matlab_bridge_result(
    raw: mne.io.BaseRaw,
    result_mat: Path | str,
    manifest_json: Path | str,
    *,
    copy: bool = True,
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Load MATLAB mSSS output and insert it back into an MNE Raw object."""
    inst = raw.copy() if copy else raw
    inst.load_data()
    result_mat = Path(result_mat)
    manifest_json = Path(manifest_json)
    result = loadmat(str(result_mat), squeeze_me=True)
    manifest = json.loads(manifest_json.read_text())
    channel_indices = [int(x) for x in manifest["channel_indices"]]

    if "data_rec_msss" not in result:
        raise KeyError(f"MATLAB result file {result_mat} does not contain 'data_rec_msss'.")
    data_rec = np.asarray(result["data_rec_msss"], dtype=float)
    if data_rec.ndim == 1:
        data_rec = data_rec[:, np.newaxis]
    if data_rec.shape[0] != len(channel_indices):
        raise ValueError(
            f"Result shape mismatch: got {data_rec.shape[0]} channels but manifest has {len(channel_indices)}."
        )
    inst._data[np.array(channel_indices)] = data_rec

    desc = inst.info.get("description") or ""
    tag = "matlab-repo-mSSS"
    inst.info["description"] = f"{desc} | {tag}" if desc else tag

    return inst, {
        "matlab_result_keys": sorted(result.keys()),
        "manifest": manifest,
    }


def matlab_batch_command(repo_root: Path | str, wrapper_m: Path | str, input_mat: Path | str, output_mat: Path | str) -> str:
    """Build a non-interactive MATLAB command string for the wrapper."""
    repo_root = Path(repo_root)
    wrapper_m = Path(wrapper_m)
    input_mat = Path(input_mat)
    output_mat = Path(output_mat)

    def _mq(path: Path) -> str:
        # MATLAB single-quoted string escaping.
        return "'" + str(path).replace("'", "''") + "'"

    wrapper_dir = wrapper_m.parent
    wrapper_name = wrapper_m.stem
    return (
        f"addpath({_mq(repo_root)});"
        f"addpath(fullfile({_mq(repo_root)}, 'SSS_function'));"
        f"addpath({_mq(wrapper_dir)});"
        f"{wrapper_name}({_mq(input_mat)}, {_mq(output_mat)});"
    )


def run_matlab_bridge(
    repo_root: Path | str,
    wrapper_m: Path | str,
    input_mat: Path | str,
    output_mat: Path | str,
    *,
    matlab_executable: str = "matlab",
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run the reference MATLAB repository non-interactively."""
    cmd = [matlab_executable, "-batch", matlab_batch_command(repo_root, wrapper_m, input_mat, output_mat)]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def auto_detect_dead_channels(raw: mne.io.BaseRaw, *, threshold: float = 1e-14) -> List[str]:
    """Simple dead-channel detector for OPM raw data."""
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    data = raw.get_data(picks=meg_picks)
    stds = np.std(data, axis=1)
    return [raw.ch_names[pick] for pick, std in zip(meg_picks, stds) if std < threshold]


def apply_hfc(
    raw: mne.io.BaseRaw,
    *,
    order: int = 2,
    copy: bool = True,
) -> mne.io.BaseRaw:
    """Convenience wrapper for HFC with the same coil patching logic."""
    if not hasattr(mne.preprocessing, "compute_proj_hfc"):
        raise RuntimeError(
            "This MNE version does not provide compute_proj_hfc(). "
            "Upgrade MNE-Python to use the HFC branch."
        )
    inst = raw.copy() if copy else raw
    inst.load_data()
    patch_cerca_opm_coil_types(inst)
    projs = mne.preprocessing.compute_proj_hfc(inst.info, order=order, picks="meg", exclude="bads")
    inst.add_proj(projs)
    inst.apply_proj()
    desc = inst.info.get("description") or ""
    tag = f"HFC(order={order})"
    inst.info["description"] = f"{desc} | {tag}" if desc else tag
    return inst


__all__ = [
    "MSSSConfig",
    "MSSSResult",
    "POINT_MAGNETOMETER_COIL",
    "apply_hfc",
    "apply_python_msss",
    "auto_detect_dead_channels",
    "compute_python_msss_basis",
    "centers_to_meg",
    "extract_sensor_geometry",
    "matlab_batch_command",
    "patch_cerca_opm_coil_types",
    "read_matlab_bridge_result",
    "run_matlab_bridge",
    "suggest_two_msss_centers",
    "summarize_opm_geometry",
    "write_matlab_bridge_payload",
]
