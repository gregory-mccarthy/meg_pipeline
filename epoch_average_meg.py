# epoch_average_meg.py
"""
Epoch averaging for MEGIN/BIDS with expression- and bit-logic bins.
Reads ONLY preprocessed derivatives (_desc-<desc>_meg.fif).
Flexible events source:
  - stim (STI101)
  - annotations (codes via regex)
  - annot_timing_stim_codes (annotation timing + STI codes)
  - auto (prefer annotations if present)

Annotation timing base:
  events.base: "rel" | "abs" | "auto"
    rel  -> use annotation onsets as-is (relative seconds)
    abs  -> subtract raw.first_time if ann.orig_time is present (absolute->relative)
    auto -> choose whichever aligns better to STI times

Condition selection supports:
  - List of numeric codes
  - Bit masks / logic (e.g., val & 0x0F == 7)
  - Python expressions via "expr": evaluated against a symbol table with "val"
    (the event value) and optional "bits" mapping or named flags supplied in YAML.

Global and per-condition epoch windows:
  - tmin/tmax/baseline global
  - epoch_windows: {COND_NAME: {tmin,tmax,baseline}}
  - artifact_suppression.window: [start_ms, end_ms] -> annotate around events

Filter:
  - Optional bandpass prior to epoching (applied on raw copy used for epoching)

Reject:
  - Optional reject dict; we auto-prune keys not present in the data (e.g., EEG
    thresholds omitted if no EEG channels).
  - Optional reject_by_annotation bool (default True) to drop epochs overlapping with BAD annotations.

Composites:
  - Arithmetic on Evokeds after averaging; e.g., "odd - even".

Outputs:
  - Evokeds -> derivatives/avg by default (configurable)
  - Epochs  -> derivatives/epochs
  - Both names derive from canonical BIDS stem: sub-XXX[_ses-YY][_task-ZZ][_run-WW]
"""

from __future__ import annotations
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mne
from bids_io_utils import (
    norm_subject, norm_session, norm_run,
    build_bids_stem, make_derivative_path,
)

import numpy as np


# -------------------------- Console logger --------------------------
class Logger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    def log(self, *a, **k):
        if self.verbose:
            msg = " ".join(str(x) for x in a if x is not None)
            if k:
                try:
                    j = json.dumps(k, ensure_ascii=False)
                except Exception:
                    j = str(k)
                msg = f"{msg} {j}".rstrip()
            print(msg, flush=True)


# ---------------------- YAML loader (robust) ----------------------
def _read_yaml_text(p: Path) -> str:
    text = p.read_text(encoding="utf-8")
    # If pasted from a fenced block, strip fences
    if text.lstrip().startswith("```"):
        lines = text.splitlines()
        # drop the first fence line
        lines = lines[1:]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("```"):
                lines = lines[:i]
                break
        text = "\n".join(lines)
    text = text.replace("\t", "  ")
    stripped = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if stripped and not stripped[0].lstrip().startswith("---"):
        text = "---\n" + text
    return text

def _make_yaml_loader():
    try:
        from ruamel.yaml import YAML
        y = YAML(typ="safe")
        y.default_flow_style = False
        y.indent(mapping=2, sequence=2, offset=2)
        return lambda s: y.load(s)
    except Exception:
        try:
            import yaml
            return lambda s: yaml.safe_load(s)
        except Exception as e:
            raise RuntimeError("No YAML parser available (ruamel.yaml or PyYAML required).") from e

_yaml_load = _make_yaml_loader()


# ------------------------ Utilities ------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def parse_int_auto(x: Any) -> Optional[int]:
    if x in (None, "", "auto"): return None
    try: return int(x)
    except Exception: return None

def _zp2(x: Optional[Union[str,int]]) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    return f"{int(s):02d}" if s.isdigit() else s

def _build_derivative_fif_path(bids_root: Path, subject: str, session: str | None,
                               task: str | None, run: str | None,
                               derivatives_subdir: str, derivatives_desc: str,
                               datatype: str = "meg") -> Path:
    # Use canonical normalization and path building from bids_io_utils
    s  = norm_subject(subject)
    se = norm_session(session)
    r  = norm_run(run)
    stem = build_bids_stem(s, se, task, r)
    pipeline = derivatives_subdir.replace("derivatives/", "").strip("/")
    return make_derivative_path(
        bids_root, pipeline, s, se,
        ["meg", f"{stem}_desc-{derivatives_desc}_{datatype}.fif"]
    )

def output_dir_for_subdir(bids_root: Path, ent: Dict[str, Any], subdir: str) -> Path:
    """
    Create/return "<BIDS>/derivatives/<pipeline>/sub-XXX[/ses-YY]/" using bids_io_utils.
    """
    s  = norm_subject(ent.get("subject"))
    se = norm_session(ent.get("session"))
    pipeline = subdir.replace("derivatives/", "").strip("/")
    # make_derivative_path(..., ["."]) gives "<.../sub-XXX[/ses-YY]/.>" — take parent
    path = make_derivative_path(bids_root, pipeline, s, se, ["."]).parent
    path.mkdir(parents=True, exist_ok=True)
    return path

def output_basename(ent: Dict[str, Any]) -> str:
    """
    Build canonical "sub-XXX[_ses-YY][_task-ZZ][_run-WW]" stem via bids_io_utils.
    """
    s  = norm_subject(ent.get("subject"))
    se = norm_session(ent.get("session"))
    task = ent.get("task")
    r  = norm_run(ent.get("run"))
    return build_bids_stem(s, se, task, r)

def _raise(msg: str): raise RuntimeError(msg)


# ------------------------ Event selection ------------------------
def _select_by_expr(values: np.ndarray, expr: str, bits: Optional[Dict[str,int]] = None) -> np.ndarray:
    """
    values: 1-D array of event 'value' (codes).
    expr: python expression string, evaluated against each 'val'.
          Allowed names:
            - val : current integer event code
            - bits: provided mapping (name -> mask)
    Returns boolean mask same length as values.
    """
    # Sanitize / safe eval: permit Name, Load, BinOp, BoolOp, Compare, Num, BitAnd/Or/Xor, Invert, Mod, etc.
    # Build a code object that evaluates to True/False per 'val'.
    allowed_nodes = (
        ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.BitOr, ast.BitAnd,
        ast.BitXor, ast.Invert, ast.Mod, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or,
        ast.Compare, ast.Name, ast.Load, ast.Constant, ast.Num, ast.BitOr, ast.USub, ast.UAdd, ast.Paren
        if hasattr(ast, "Paren") else ast.UnaryOp
    )
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Disallowed AST node in expr: {type(n).__name__}")
        if isinstance(n, ast.Name) and n.id not in {"val", "bits"}:
            raise ValueError(f"Unknown name in expr: {n.id}")
    code = compile(tree, "<expr>", "eval")
    out = np.zeros_like(values, dtype=bool)
    local_bits = bits or {}
    for i, v in enumerate(values):
        try:
            out[i] = bool(eval(code, {"__builtins__": {}}, {"val": int(v), "bits": local_bits}))
        except Exception:
            out[i] = False
    return out


def annotate_artifacts_around_events(raw: mne.io.BaseRaw, events: np.ndarray,
                                     window: Tuple[float, float], desc: str = "ARTIFACT") -> None:
    on, off = float(window[0]), float(window[1])
    if on >= off: raise ValueError("artifact_suppression.window must be (start<end)")
    sfreq = float(raw.info["sfreq"]); on_samp = int(round(on * sfreq)); off_samp = int(round(off * sfreq))
    dur = (off_samp - on_samp) / sfreq
    existing = raw.annotations if raw.annotations is not None else mne.Annotations([], [], [])
    onset_times = (events[:, 0] + on_samp) / sfreq
    new = mne.Annotations(onset_times, [dur] * len(onset_times), [desc] * len(onset_times))
    raw.set_annotations(existing + new)



def blank_epochs_time_window(epochs: mne.Epochs, window_s: Tuple[float, float], mode: str = "zero") -> None:
    """Blank a time window within each epoch (relative to event time 0).

    window_s: (tmin_s, tmax_s) in seconds.
    mode:
      - "zero": set data to 0 in that window
    """
    t0, t1 = float(window_s[0]), float(window_s[1])
    if t0 >= t1:
        raise ValueError("artifact_suppression.window must be (start<end)")
    if epochs._data is None:
        raise RuntimeError("Epochs must be preloaded to blank data (use preload=True).")

    mask = (epochs.times >= t0) & (epochs.times <= t1)
    if not mask.any():
        return

    if mode == "zero":
        epochs._data[:, :, mask] = 0.0
    else:
        raise ValueError(f"Unknown artifact_suppression.mode: {mode!r} (use 'zero')")


def _events_from_stim(raw: mne.io.BaseRaw, stim_channel: str) -> Tuple[np.ndarray, np.ndarray]:
    # Added uint_cast=True to bypass the MEGIN STI016 sign-bit bug
    events = mne.find_events(
        raw,
        stim_channel=stim_channel,
        shortest_event=1,
        initial_event=False,
        uint_cast=True,
        verbose=False
    )
    values = events[:, 2].astype(int)

    # Print the exact integers MNE extracted to STDOUT for easy debugging
    print(f"Unique trigger values extracted from {stim_channel}: {set(values)}")

    return events, values


def _events_from_annotations_by_regex(raw: mne.io.BaseRaw, regex: str, base_choice: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract events from annotations whose description matches regex capturing group 'code'.
    base_choice:
      "rel" -> treat annotation onset as relative seconds (use as-is)
      "abs" -> if orig_time present, convert absolute->relative by subtracting raw.first_time
    """
    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0,), dtype=int)
    pat = re.compile(regex)
    rows = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        m = pat.search(desc)
        if not m: continue
        # allow either named group (?P<code>...) or fallback to first group
        if "code" in m.groupdict():
            code_str = m.group("code")
        elif m.groups():
            code_str = m.group(1)
        else:
            continue
        try:
            code = int(code_str)
        except Exception:
            # support hex like 0x0F
            code = int(code_str, 0) if isinstance(code_str, str) else None
        if code is None: continue
        rows.append((onset, int(code)))
    if not rows:
        return np.empty((0, 3), dtype=int), np.empty((0,), dtype=int)
    rel_onsets = np.array([r[0] for r in rows], dtype=float)
    codes = np.array([r[1] for r in rows], dtype=int)

    if base_choice == "auto":
        base_choice = "abs" if raw.annotations.orig_time is not None else "rel"
    if base_choice == "abs" and raw.annotations.orig_time is not None:
        # Convert absolute annotation onset to relative by subtracting raw.first_time
        rel_onsets = rel_onsets - float(raw.first_samp) / float(raw.info["sfreq"])

    # Now convert to MNE 'events' (sample, 0, code)
    sfreq = float(raw.info["sfreq"])
    samp = np.round(rel_onsets * sfreq).astype(int)
    events = np.column_stack([samp, np.zeros_like(samp, dtype=int), codes])
    return events, codes


def _nearest_dt_samples(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # For each a, return index of b with nearest sample index
    # (Assumes both arrays are 1-D int sample indices)
    idx = np.empty(len(a), dtype=int)
    j = 0
    for i in range(len(a)):
        # advance j while b[j] is less than a[i] (single pass)
        while j + 1 < len(b) and abs(b[j+1] - a[i]) <= abs(b[j] - a[i]):
            j += 1
        idx[i] = j
    return idx


def _match_events_by_time(events_a: np.ndarray, events_b: np.ndarray, tol_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy nearest-neighbor match in sample time with tolerance; return paired indices into A and B.
    """
    if len(events_a) == 0 or len(events_b) == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    sa = events_a[:, 0].astype(int); sb = events_b[:, 0].astype(int)
    ia = np.arange(len(sa))
    ib = _nearest_dt_samples(sa, sb)
    ok = np.abs(sb[ib] - sa) <= tol_samples
    return ia[ok], ib[ok]


# ------------------------ Epoch creation ------------------------
def create_epochs_for_condition(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    values: np.ndarray,
    cond_name: str,
    cond_spec: Any,
    global_tmin: float,
    global_tmax: float,
    global_baseline: Optional[Tuple[Optional[float], Optional[float]]],
    epoch_windows: Optional[Dict[str, Dict[str, Any]]],
    reject: Optional[Dict[str, float]],
    reject_by_annotation: bool, # NEW
    stim_channel: Optional[str],
    artifact_suppression_window: Optional[Tuple[float, float]],
    artifact_suppression_mode: str,
    logger: Logger,
) -> Tuple[Optional[mne.Epochs], Optional[mne.Evoked]]:

    # Determine selection by cond_spec (list of ints OR dict with "expr")
    if isinstance(cond_spec, dict) and "expr" in cond_spec:
        expr = str(cond_spec["expr"])
        bits = cond_spec.get("bits") or {}
        if not isinstance(bits, dict):
            raise ValueError(f"Condition '{cond_name}' has non-dict 'bits'.")
        mask = _select_by_expr(values, expr, bits)
    else:
        # assume sequence of integers (codes)
        try:
            codes = np.array(list(cond_spec), dtype=int)
        except Exception:
            raise ValueError(f"Condition '{cond_name}' must be a list of ints or dict with 'expr'.")
        mask = np.isin(values, codes)

    sel_events = events[mask]
    if len(sel_events) == 0:
        logger.log(f"No events for condition '{cond_name}', skipping.")
        return None, None

    # Per-condition windows override global
    tmin = global_tmin; tmax = global_tmax; baseline = global_baseline
    if epoch_windows and cond_name in epoch_windows:
        ew = epoch_windows[cond_name]
        tmin = float(ew.get("tmin", tmin))
        tmax = float(ew.get("tmax", tmax))
        base = ew.get("baseline", baseline)
        if base is None:
            baseline = None
        else:
            baseline = (None if base[0] is None else float(base[0]),
                        None if base[1] is None else float(base[1]))

    # Build reject dict pruned to present channel types
    rej = None
    if reject:
        rej = {}
        ch_types = mne.channels.make_ch_type_mapping(raw.info)["kind"]
        present = set(ch_types)
        for k, v in reject.items():
            if k in present:
                rej[k] = float(v)

    # Make epochs
    picks = None  # let MNE decide based on channels present
    epochs = mne.Epochs(
        raw, sel_events, event_id=None, tmin=tmin, tmax=tmax,
        baseline=baseline, preload=True, picks=picks, reject=rej,
        detrend=None, reject_by_annotation=reject_by_annotation, verbose=False # UPDATED
    )

    if artifact_suppression_window is not None:
        blank_epochs_time_window(epochs, artifact_suppression_window, mode=artifact_suppression_mode)

    evoked = epochs.average()
    logger.log("Created condition", name=cond_name, n_epochs=len(epochs), tmin=tmin, tmax=tmax, baseline=str(baseline))
    return epochs, evoked


# ------------------------ One subject/session/run ------------------------
def run_one(bids_root: Path, ent: Dict[str, str], config: Dict[str, Any], logger: Logger) -> None:
    # Require derivatives keys
    deriv_sub = config.get("derivatives_subdir"); deriv_desc = config.get("derivatives_desc")
    if not deriv_sub or not deriv_desc:
        raise RuntimeError("Provide 'derivatives_subdir' and 'derivatives_desc' in YAML.")
    fif_path = _build_derivative_fif_path(bids_root, str(ent.get("subject")), ent.get("session"),
                                          ent.get("task"), ent.get("run"), deriv_sub, deriv_desc, "meg")
    if not fif_path.exists(): raise FileNotFoundError(f"Expected derivative not found:\n  {fif_path}")
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    logger.log("Loaded preprocessed derivative", file=str(fif_path))

    # Events config
    ev_cfg = config.get("events", {}) or {}
    source   = str(ev_cfg.get("source", "auto")).lower()
    base     = str(ev_cfg.get("base",   "auto")).lower()   # NEW
    stim_ch  = str(ev_cfg.get("stim_channel", config.get("stim_channel", "STI101")))
    regex    = ev_cfg.get("regex", r"EVT:(?P<code>\d+)")
    tol_sec  = float(ev_cfg.get("tolerance_sec", 0.005))
    verify   = bool(ev_cfg.get("verify_against_stim", False))
    align_n  = parse_int_auto(ev_cfg.get("alignment_correction_samples"))
    offset_ms = float(ev_cfg.get("offset_ms", 0.0) or 0.0)
    offset_samples = int(round(offset_ms * 1e-3 * float(raw.info["sfreq"])))

    # Candidate events from stim
    try:
        events_stim, codes_stim = _events_from_stim(raw, stim_ch)
    except Exception:
        events_stim = np.empty((0, 3), dtype=int); codes_stim = np.empty((0,), dtype=int)

    # Candidate events from annotations (regex)
    def build_ann_for(base_choice: str):
        # If user asks for "auto", we fill later
        return _events_from_annotations_by_regex(raw, regex, base_choice)

    events_ann = codes_ann = None; matched_total = 0

    if source in ("auto", "annotations", "annot_timing_stim_codes"):
        # If base is 'auto', defer the choice until we compare to stim
        if base == "auto":
            # Compute both, measure timing match to stim, choose better
            ea_rel, ca_rel = _events_from_annotations_by_regex(raw, regex, "rel")
            ea_abs, ca_abs = _events_from_annotations_by_regex(raw, regex, "abs")
            # default tol_samples from config
            tol_samp = int(round(float(tol_sec) * float(raw.info["sfreq"])))
            ia_rel, ib_rel = _match_events_by_time(ea_rel, events_stim, tol_samp)
            ia_abs, ib_abs = _match_events_by_time(ea_abs, events_stim, tol_samp)
            if len(ia_abs) > len(ia_rel):
                events_ann, codes_ann, base_choice = ea_abs, ca_abs, "abs"
            else:
                events_ann, codes_ann, base_choice = ea_rel, ca_rel, "rel"
        else:
            events_ann, codes_ann = _events_from_annotations_by_regex(raw, regex, base)
            base_choice = base

        if verify and len(events_stim) and len(events_ann):
            tol_samp = int(round(float(tol_sec) * float(raw.info["sfreq"])))
            ia, ib = _match_events_by_time(events_ann, events_stim, tol_samp)
            matched_total = len(ia)
            logger.log("Verify annot vs stim", matched=matched_total, annot=len(events_ann), stim=len(events_stim), base=base_choice)

        if source == "annot_timing_stim_codes" and len(events_stim):
            # Borrow codes from stim by nearest time
            tol_samp = int(round(float(tol_sec) * float(raw.info["sfreq"])))
            ia, ib = _match_events_by_time(events_ann, events_stim, tol_samp)
            codes_ann = codes_stim[ib]
            events_ann = np.column_stack([events_ann[:, 0], np.zeros(len(events_ann), dtype=int), codes_ann])

        if events_ann is not None and align_n:
            events_ann = events_ann.copy(); events_ann[:, 0] += int(align_n)
            logger.log("Applied alignment correction", samples=align_n)

    # --- Parse optional timing corrections from config ---
    sfreq = float(raw.info["sfreq"])

    # Global ms-based offset applied AFTER final event selection (restores older behavior)
    offset_ms = float(ev_cfg.get("offset_ms", 0.0) or 0.0)
    offset_samples = int(round(offset_ms * 1e-3 * sfreq)) if offset_ms else 0

    # Optional sample-based correction applied to ANNOTATION times BEFORE code grafting/matching
    try:
        align_n = ev_cfg.get("alignment_correction_samples", None)
        align_n = int(align_n) if align_n not in (None, "", "auto") else None
    except Exception:
        align_n = None

    # If we need an alignment correction for annotation-timed paths, make a shifted copy
    events_ann_for_match = events_ann
    if (source in ("annotations", "annot_timing_stim_codes")) and (events_ann is not None) and (
            align_n not in (None, 0)):
        events_ann_for_match = events_ann.copy()
        events_ann_for_match[:, 0] += int(align_n)
        logger.log(
            "Applied annotation-time alignment correction",
            samples=int(align_n),
            seconds=float(align_n) / sfreq
        )

    # --- Select events & values depending on source ---
    if source == "stim" or (source == "auto" and events_ann is None):
        events, values = events_stim, codes_stim

    elif source == "annotations":
        if events_ann is None or len(events_ann) == 0:
            raise RuntimeError("No annotation-derived events available.")
        # Use (possibly) shifted annotation timing
        events, values = events_ann_for_match, codes_ann
        if verify:
            ia, ib = _match_events_by_time(events_ann_for_match, events_stim, tol_samp)
            logger.log(
                "Verify annotations vs stim",
                matched=len(ia), annot=len(events_ann_for_match), stim=len(events_stim)
            )

    elif source == "annot_timing_stim_codes":
        if events_ann is None or len(events_ann) == 0:
            raise RuntimeError("No annotation-derived events available for annot_timing_stim_codes.")
        # Match annotation-timed events (possibly shifted) to stim to borrow codes
        ia, ib = _match_events_by_time(events_ann_for_match, events_stim, tol_samp)
        events = events_ann_for_match.copy()
        values = codes_stim[ib]
        events[:, 2] = values
        if verify:
            logger.log(
                "Verify annot-timing+stim-codes",
                matched=len(ia), annot=len(events_ann_for_match), stim=len(events_stim)
            )

    else:
        # source == "auto" but annotations unavailable; already handled -> stim
        events, values = events_stim, codes_stim

    # --- Apply a GLOBAL offset in milliseconds to the FINAL selected events (restored behavior) ---
    if offset_samples:
        events = events.copy()
        events[:, 0] += int(offset_samples)
        logger.log(
            "Applied global event offset",
            offset_ms=float(offset_ms),
            offset_samples=int(offset_samples),
            seconds=float(offset_samples) / sfreq,
            direction=("earlier" if offset_ms < 0 else "later")
        )

    logger.log("Events summary", n_events=len(events), unique_vals=len(np.unique(values)))

    # Optional filter
    filt_cfg = config.get("filter") or {}
    if filt_cfg:
        l = filt_cfg.get("l_freq", None)
        h = filt_cfg.get("h_freq", None)
        l = None if l in (None, "", False) else float(l)
        h = None if h in (None, "", False) else float(h)
        if l is not None or h is not None:
            raw_filt = raw.copy().filter(l_freq=l, h_freq=h, picks=None, verbose=False)
        else:
            raw_filt = raw.copy()
    else:
        raw_filt = raw.copy()

    # Global epoch window
    tmin = float(config.get("tmin", -0.2))
    tmax = float(config.get("tmax", 0.5))
    base = config.get("baseline", (None, 0.0))
    if base is None:
        baseline = None
    else:
        baseline = (None if base[0] is None else float(base[0]),
                    None if base[1] is None else float(base[1]))

    # Artifact suppression window (ms) -> seconds
    as_win = None
    as_cfg = config.get("artifact_suppression") or {}
    if "window" in as_cfg and as_cfg["window"] is not None:
        w = as_cfg["window"]
        if isinstance(w, (list, tuple)) and len(w) == 2:
            as_win = (float(w[0]) / 1000.0, float(w[1]) / 1000.0)
        else:
            raise ValueError("artifact_suppression.window must be [start_ms, end_ms]")


    as_mode = "zero"
    if isinstance(as_cfg, dict) and as_cfg.get("mode") is not None:
        as_mode = str(as_cfg["mode"]).strip().lower()
    # Conditions
    conds = config.get("conditions") or {}
    if not isinstance(conds, dict) or not conds:
        raise RuntimeError("Provide non-empty 'conditions' mapping in YAML.")

    # Per-condition windows override
    epoch_windows = config.get("epoch_windows") or {}

    # Optional reject dict
    use_reject = bool(config.get("use_reject", False))
    reject = config.get("reject") if use_reject else None
    if reject is not None and not isinstance(reject, dict):
        reject = None

    # NEW: Extract annotation rejection flag
    reject_by_ann = bool(config.get("reject_by_annotation", True))

    # Stim channel override at condition-level
    stim_override = config.get("stim_channel", None)

    # Build epochs & evokeds per condition
    epochs_dict: Dict[str, mne.Epochs] = {}
    evokeds: Dict[str, mne.Evoked] = {}

    for name, spec in conds.items():
        ep, ev = create_epochs_for_condition(
            raw=raw_filt,
            events=events,
            values=values,
            cond_name=name,
            cond_spec=spec,
            global_tmin=tmin, global_tmax=tmax, global_baseline=baseline,
            epoch_windows=epoch_windows,
            reject=reject,
            reject_by_annotation=reject_by_ann, # NEW
            stim_channel=stim_override,
            artifact_suppression_window=as_win,
            artifact_suppression_mode=as_mode,
            logger=logger,
        )
        if ep is not None: epochs_dict[name] = ep
        if ev is not None: evokeds[name] = ev

    # Composites
    comps = config.get("composites") or {}
    if isinstance(comps, dict):
        def _safe_eval_evoked_expr(expr: str, ev_map: Dict[str, mne.Evoked]) -> mne.Evoked:
            # permit names/sum/diff/scale
            # parse expr AST and allow Name, BinOp (+,-), Mult by constant
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
                ast.Name, ast.Load, ast.Constant, ast.Num, ast.UnaryOp, ast.USub, ast.UAdd
            )
            tree = ast.parse(expr, mode="eval")
            for n in ast.walk(tree):
                if not isinstance(n, allowed_nodes):
                    raise ValueError(f"Disallowed node in composite expr: {type(n).__name__}")
                if isinstance(n, ast.Name) and n.id not in ev_map:
                    raise ValueError(f"Unknown evoked name in composite expr: {n.id}")
            code = compile(tree, "<evoked-expr>", "eval")
            # Create a thin wrapper so that Name resolves to ev_map[name]
            class _NS(dict):
                def __getitem__(self, k): return ev_map[k]
            ns = _NS()
            out = eval(code, {"__builtins__": {}}, ns)
            if not isinstance(out, mne.Evoked):
                raise ValueError("Composite expression did not yield an Evoked.")
            return out

        for cname, cexpr in comps.items():
            try:
                ev = _safe_eval_evoked_expr(str(cexpr), evokeds)
                evokeds[cname] = ev
                logger.log("Composite created", name=cname, expr=str(cexpr))
            except Exception as e:
                logger.log("Composite failed", name=cname, error=str(e))

    # ---------------- Outputs ----------------
    out_cfg = config.get("outputs") or {}
    save_evokeds = out_cfg.get("save_evokeds", True)
    save_epochs = out_cfg.get("save_epochs", True)
    evk_subdir   = out_cfg.get("evokeds_subdir", "derivatives/avg")
    epo_subdir   = out_cfg.get("epochs_subdir",  "derivatives/epochs")
    base_name = output_basename(ent)

    if save_evokeds and evokeds:
        out_dir_evk = output_dir_for_subdir(bids_root, ent, evk_subdir)
        # Existing per-condition evoked writes
        for name, ev in evokeds.items():
            fn = out_dir_evk / f"{base_name}_desc-{name}_ave.fif"
            ev.save(fn, overwrite=True)
            logger.log("Saved", file=str(fn))

        # Combined-evoked write
        if out_cfg.get("combined_evokeds", False) and evokeds:
            out_dir_evk = output_dir_for_subdir(bids_root, ent, evk_subdir)
            cond_order = list((config.get("conditions") or {}).keys())
            extras = [k for k in evokeds.keys() if k not in cond_order]
            ordered = cond_order + extras
            for name in ordered:
                try:
                    evokeds[name].comment = name
                except Exception:
                    pass
            comb_desc = str(out_cfg.get("combined_desc", "all"))
            comb_fn = out_dir_evk / f"{base_name}_desc-{comb_desc}_ave.fif"
            mne.write_evokeds(str(comb_fn), [evokeds[name] for name in ordered], overwrite=True)
            logger.log("Saved combined evokeds", file=str(comb_fn), n=len(ordered))

    if save_epochs and epochs_dict:
        out_dir_epo = output_dir_for_subdir(bids_root, ent, epo_subdir)
        for name, ep in epochs_dict.items():
            fn = out_dir_epo / f"{base_name}_desc-{name}_epo.fif"
            ep.save(fn, overwrite=True); logger.log("Saved", file=str(fn))

    # ---------------- Run-log (JSON + YAML) written to both output dirs ----------------
    try:
        from ruamel.yaml import YAML as _R_YAML
        _R_YAML_WRITER = _R_YAML()
        _R_YAML_WRITER.default_flow_style = False
        _R_YAML_WRITER.indent(mapping=2, sequence=2, offset=2)
    except Exception:
        _R_YAML_WRITER = None

    # Build structured run log
    first_event_time_sec = float(events[0, 0] / raw.info["sfreq"]) if len(events) else None
    run_log = {
        "script_version": "phase10-2025-10-23",
        "input_derivative": str(fif_path),
        "n_events_total": int(len(events)),
        "unique_event_values": sorted([int(x) for x in np.unique(values).tolist()]),
        "first_event_time_sec": first_event_time_sec,
        "filter": config.get("filter"),
        "global_epoch_window": {"tmin": tmin, "tmax": tmax, "baseline": baseline},
        "artifact_suppression": config.get("artifact_suppression"),
        "conditions": list(conds.keys()),
        "composites": comps,
        "outputs": {
            "evokeds_dir": str(out_cfg.get("evokeds_subdir", "derivatives/avg")),
            "epochs_dir": str(out_cfg.get("epochs_subdir", "derivatives/epochs")),
            "basename": base_name,
        },
    }

    for subdir in (evk_subdir, epo_subdir):
        try:
            od = output_dir_for_subdir(bids_root, ent, subdir)
            ensure_dir(od / "logs")
            (od / "logs" / f"{base_name}_run_log.json").write_text(json.dumps(run_log, indent=2))
            if _R_YAML_WRITER is not None:
                with (od / "logs" / f"{base_name}_run_log.yaml").open("w", encoding="utf-8") as f:
                    _R_YAML_WRITER.dump(run_log, f)
            else:
                (od / "logs" / f"{base_name}_run_log.yaml").write_text(json.dumps(run_log, indent=2))
        except Exception as e:
            logger.log("Failed to write run logs", subdir=subdir, error=str(e))


# ------------------------ CLI ------------------------
def load_config(path: Path) -> Dict[str, Any]: return _yaml_load(path)

def main():
    ap = argparse.ArgumentParser(description="Epoch averaging with expressions/bit-logic (MEGIN/BIDS derivatives-only).")
    ap.add_argument("config", help="Path to YAML configuration.")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output.")
    args = ap.parse_args()
    cfg = load_config(Path(args.config)); logger = Logger(verbose=not args.quiet)
    bids_root = Path(cfg["bids_root"]).expanduser().resolve()
    for ent in cfg["inputs"]:
        run_one(bids_root, ent, cfg, logger)

if __name__ == "__main__":
    main()