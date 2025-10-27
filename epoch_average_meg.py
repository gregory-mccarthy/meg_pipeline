# epoch_average_MEG_BIDS_phase8b.py
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

Other events knobs:
  events.tolerance_sec: float (e.g., 0.002)
  events.alignment_correction_samples: int (optional nudge after conversion)
  events.regex: e.g., "TRIG/(\\d+)"
  events.verify_against_stim: true/false

Saves BOTH Evokeds (_ave.fif) and Epochs (_epo.fif).
"""

from __future__ import annotations
import argparse
import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import mne

# ----------------------------- YAML loader (robust) ---------------------------
def _read_yaml_text(path: Path) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
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
        from ruamel.yaml import YAML  # type: ignore
        _yaml = YAML(typ="safe")
        def _yaml_load(path: Path):
            return _yaml.load(_read_yaml_text(path))
        return _yaml_load
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise SystemExit("Neither ruamel.yaml nor PyYAML available (pip install ruamel.yaml)") from e
        def _yaml_load(path: Path):
            return yaml.safe_load(_read_yaml_text(path))
        return _yaml_load

_yaml_load = _make_yaml_loader()

# ----------------------------- Utilities & Logging ----------------------------
class Logger:
    def __init__(self, verbose: bool = True): self.verbose = verbose
    def log(self, *args, **kwargs):
        if not self.verbose: return
        parts = [str(a) for a in args]
        if kwargs: parts.append(" | " + ", ".join(f"{k}={v}" for k, v in kwargs.items()))
        print("[avg]", *parts)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def parse_int_auto(val: Any) -> int:
    if isinstance(val, int): return val
    if isinstance(val, str): return int(val, 0)  # 0x.. supported
    raise TypeError(f"Expected int or str, got {type(val)}")

# ----------------------------- Derivatives path helpers -----------------------
def _zp2(x: str | int | None) -> str | None:
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    return f"{int(s):02d}" if s.isdigit() else s

def _build_derivative_fif_path(bids_root: Path, subject: str, session: str | None,
                               task: str | None, run: str | None,
                               derivatives_subdir: str, derivatives_desc: str,
                               datatype: str = "meg") -> Path:
    sub = _zp2(subject); ses = _zp2(session); rn = _zp2(run)
    sub_tag = f"sub-{sub}"; ses_tag = f"ses-{ses}" if ses else None
    task_tag = f"_task-{task}" if task else ""; run_tag = f"_run-{rn}" if rn else ""
    desc_tag = f"_desc-{derivatives_desc}" if derivatives_desc else ""
    fname = f"{sub_tag}{('_' + ses_tag) if ses_tag else ''}{task_tag}{run_tag}{desc_tag}_{datatype}.fif"
    parts = [bids_root, derivatives_subdir, sub_tag];
    if ses_tag: parts.append(ses_tag)
    parts.append(datatype)
    return Path(*parts) / fname

def output_dir_for_subdir(bids_root: Path, ent: Dict[str, str], subdir: str) -> Path:
    subject = _zp2(ent.get("subject")); session = _zp2(ent.get("session"))
    parts = [bids_root, subdir]
    if subject: parts.append(f"sub-{subject}")
    if session: parts.append(f"ses-{session}")
    return ensure_dir(Path(*parts))

def output_basename(ent: Dict[str, str]) -> str:
    subject = _zp2(ent.get("subject")); session = _zp2(ent.get("session"))
    task = ent.get("task"); run = _zp2(ent.get("run"))
    bits = []
    if subject: bits.append(f"sub-{subject}")
    if session: bits.append(f"ses-{session}")
    if task:    bits.append(f"task-{task}")
    if run:     bits.append(f"run-{run}")
    return "_".join(bits) if bits else "avg"

# ----------------------------- Legacy bit-logic selection ---------------------
def _bitmask_from_bits(bits: List[int]) -> int:
    m = 0
    for b in bits:
        if not (1 <= b <= 32): raise ValueError(f"Bit index {b} out of range 1..32")
        m |= (1 << (b - 1))
    return m

def _select_by_spec(values: np.ndarray, spec: Any) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64); sel = np.ones(v.shape, bool)
    if isinstance(spec, (int, str)): return v == parse_int_auto(spec)
    if isinstance(spec, (list, tuple)):
        codes = np.array([parse_int_auto(x) for x in spec], dtype=np.int64)
        return np.isin(v, codes)
    if not isinstance(spec, dict): raise TypeError(f"Unsupported condition spec: {type(spec)}")
    if 'codes' in spec:
        codes = np.array([parse_int_auto(x) for x in spec['codes']], dtype=np.int64)
        sel &= np.isin(v, codes)
    if 'mask' in spec:
        mask_int = parse_int_auto(spec['mask'])
        mode = str(spec.get('mask_mode', 'any')).lower()
        sel &= ((v & mask_int) == mask_int) if mode == 'all' else ((v & mask_int) != 0)
    if 'all_bits' in spec:
        m = _bitmask_from_bits(list(spec['all_bits'])); sel &= (v & m) == m
    if 'any_bits' in spec:
        m = _bitmask_from_bits(list(spec['any_bits'])); sel &= (v & m) != 0
    if 'not_bits' in spec:
        m = _bitmask_from_bits(list(spec['not_bits'])); sel &= (v & m) == 0
    if 'field' in spec and 'equals' in spec:
        fb = spec['field']; start_bit = int(fb['start_bit']); width = int(fb['width'])
        if start_bit < 1 or width < 1: raise ValueError("field.start_bit and field.width must be >= 1")
        shift = start_bit - 1; mask = ((1 << width) - 1) << shift
        sel &= (((v & mask) >> shift) == int(spec['equals']))
    return sel

# ----------------------------- Expression-based selection ---------------------
class _ExprEval(ast.NodeVisitor):
    def __init__(self, code_array: np.ndarray): self.code = code_array
    def _bit(self, n):        return (self.code & (1 << (int(n)-1))) != 0
    def _field(self, s, w):   return (self.code & (((1<<int(w))-1) << (int(s)-1))) >> (int(s)-1)
    def _anymask(self, m):    return (self.code & int(m)) != 0
    def _allmask(self, m):    return (self.code & int(m)) == int(m)
    def eval(self, expr: str): return self.visit(ast.parse(expr, mode="eval").body)
    def visit_Name(self, node):      return self.code if node.id == "code" else _raise(f"Unknown name: {node.id}")
    def visit_Constant(self, node):  return node.value if isinstance(node.value, (int,float,bool)) else _raise("Only numeric/bool constants")
    def visit_UnaryOp(self, node):
        x = self.visit(node.operand)
        if   isinstance(node.op, ast.Invert): return ~x
        elif isinstance(node.op, ast.UAdd):   return +x
        elif isinstance(node.op, ast.USub):   return -x
        else: _raise("Unary op not allowed")
    def visit_BinOp(self, node):
        L, R, op = self.visit(node.left), self.visit(node.right), node.op
        if   isinstance(op, ast.BitAnd):   return L & R
        elif isinstance(op, ast.BitOr):    return L | R
        elif isinstance(op, ast.BitXor):   return L ^ R
        elif isinstance(op, ast.LShift):   return L << R
        elif isinstance(op, ast.RShift):   return L >> R
        elif isinstance(op, ast.Add):      return L + R
        elif isinstance(op, ast.Sub):      return L - R
        elif isinstance(op, ast.Mult):     return L * R
        elif isinstance(op, ast.Div):      return L / R
        elif isinstance(op, ast.FloorDiv): return L // R
        elif isinstance(op, ast.Mod):      return L % R
        else: _raise("Binary op not allowed")
    def visit_Compare(self, node):
        cur = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            rhs = self.visit(comp)
            if   isinstance(op, ast.Eq):    cur = (cur == rhs)
            elif isinstance(op, ast.NotEq): cur = (cur != rhs)
            elif isinstance(op, ast.Lt):    cur = (cur <  rhs)
            elif isinstance(op, ast.LtE):   cur = (cur <= rhs)
            elif isinstance(op, ast.Gt):    cur = (cur >  rhs)
            elif isinstance(op, ast.GtE):   cur = (cur >= rhs)
            else: _raise("Cmp op not allowed")
        return cur
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name): _raise("Only simple function names allowed")
        fname = node.func.id; args = [self.visit(a) for a in node.args]
        if   fname == "bit":     return self._bit(*args)
        elif fname == "field":   return self._field(*args)
        elif fname == "anymask": return self._anymask(*args)
        elif fname == "allmask": return self._allmask(*args)
        else: _raise(f"Function not allowed: {fname}")
    def generic_visit(self, node): _raise(f"Unsupported syntax: {type(node).__name__}")

def _raise(msg): raise ValueError(msg)

def _select_by_expr(values: np.ndarray, expr: str) -> np.ndarray:
    evaluator = _ExprEval(np.asarray(values, dtype=np.int64))
    out = evaluator.eval(expr)
    return np.asarray(out, dtype=bool)

# ----------------------------- Artifact suppression ---------------------------
def annotate_artifacts_around_events(raw: mne.io.BaseRaw, events: np.ndarray,
                                     window: Tuple[float, float], desc: str = "ARTIFACT") -> None:
    on, off = float(window[0]), float(window[1])
    if on >= off: raise ValueError("artifact_suppression.window must be (start<end)")
    sfreq = float(raw.info["sfreq"]); on_samp = int(round(on * sfreq)); off_samp = int(round(off * sfreq))
    dur = (off_samp - on_samp) / sfreq
    existing = raw.annotations
    rel_onsets = (events[:, 0] + on_samp) / sfreq
    if existing is not None and existing.orig_time is not None:
        abs_onsets = raw.first_time + rel_onsets
        ann = mne.Annotations(onset=abs_onsets, duration=np.full_like(abs_onsets, dur, float),
                              description=[desc]*len(events), orig_time=existing.orig_time)
        raw.set_annotations(existing + ann)
    else:
        ann = mne.Annotations(onset=rel_onsets, duration=np.full_like(rel_onsets, dur, float),
                              description=[desc]*len(events), orig_time=None)
        raw.set_annotations(ann if existing is None else (existing + ann))

# ----------------------------- Composites safe eval ---------------------------
class _AstExprEvaluator(ast.NodeVisitor):
    def __init__(self, names: Dict[str, mne.Evoked]): self.names = names
    def eval(self, expr: str) -> mne.Evoked: return self.visit(ast.parse(expr, mode="eval").body)
    def visit_BinOp(self, node):
        L, R = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add): return L + R
        if isinstance(node.op, ast.Sub): return L - R
        if isinstance(node.op, ast.Mult):
            if isinstance(L, (int, float)) and isinstance(R, mne.Evoked):
                ev = R.copy(); ev.data *= float(L); return ev
            if isinstance(R, (int, float)) and isinstance(L, mne.Evoked):
                ev = L.copy(); ev.data *= float(R); return ev
            _raise("Only scalar*Evoked or Evoked*scalar allowed for '*'")
        _raise("Operator not allowed")
    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd): return val
        if isinstance(node.op, ast.USub):
            if isinstance(val, (int, float)): return -val
            if isinstance(val, mne.Evoked): ev = val.copy(); ev.data *= -1.0; return ev
        _raise("Unary operator not allowed")
    def visit_Name(self, node):
        if node.id not in self.names: _raise(f"Unknown condition in composite: {node.id}")
        return self.names[node.id]
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)): return node.value
        _raise("Only numeric constants allowed")
    def generic_visit(self, node): _raise(f"Unsupported construct: {type(node).__name__}")

# ----------------------------- Events helpers --------------------------------
def _events_from_stim(raw: mne.io.BaseRaw, stim_channel: str):
    ev = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, initial_event=False)
    return ev, ev[:, 2].copy()

def _events_from_annotations_by_regex(raw: mne.io.BaseRaw, regex: str, base_mode: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    patt = re.compile(regex); ann = raw.annotations
    if ann is None or len(ann) == 0: return None, None, 0
    sf = float(raw.info["sfreq"])
    use_abs = (base_mode == "abs") and (ann.orig_time is not None)
    base = raw.first_time if use_abs else 0.0
    ev = []; matched = 0
    for onset, desc in zip(ann.onset, ann.description):
        m = patt.search(desc)
        if not m: continue
        matched += 1
        code = int(m.group(1), 0)
        samp = int(round((onset - base) * sf))
        ev.append((samp, 0, code))
    if matched == 0 or not ev: return None, None, matched
    arr = np.asarray(ev, int); return arr, arr[:, 2].copy(), matched

def _nearest_dt_samples(ev_a: np.ndarray, ev_b: np.ndarray) -> Optional[np.ndarray]:
    if ev_a is None or ev_b is None or len(ev_a)==0 or len(ev_b)==0: return None
    sb = ev_b[:, 0]; dts = []; j = 0
    for sa in ev_a[:, 0]:
        while j + 1 < len(sb) and sb[j + 1] <= sa: j += 1
        candidates = [j] + ([j + 1] if j + 1 < len(sb) else [])
        best = min(candidates, key=lambda k: abs(sb[k] - sa))
        dts.append(sa - sb[best])
    return np.asarray(dts, int)

def _match_events_by_time(ea: np.ndarray, eb: np.ndarray, tol_samples: int) -> tuple[np.ndarray, np.ndarray]:
    ia = ib = 0; pa = []; pb = []
    while ia < len(ea) and ib < len(eb):
        sa, sb = ea[ia, 0], eb[ib, 0]
        if abs(sa - sb) <= tol_samples: pa.append(ia); pb.append(ib); ia += 1; ib += 1
        elif sa < sb: ia += 1
        else: ib += 1
    return np.asarray(pa, int), np.asarray(pb, int)

# ----------------------------- Core processing --------------------------------
def create_epochs_for_condition(raw: mne.io.BaseRaw, base_events: np.ndarray, values: np.ndarray, spec: Any,
                                tmin: float, tmax: float, baseline: Tuple[float | None, float | None] | List[float | None] | None,
                                reject: Dict[str, float] | None, logger: Logger, event_id_name: str) -> mne.Epochs:
    sel = _select_by_expr(values, spec['expr']) if isinstance(spec, dict) and 'expr' in spec else _select_by_spec(values, spec)
    cond_events = base_events[sel]
    if cond_events.size == 0: raise RuntimeError("No events after selection")
    tmp = cond_events.copy(); tmp[:, 2] = 1; event_id = {event_id_name: 1}
    logger.log("Creating epochs", condition=event_id_name, n_events=len(tmp), tmin=tmin, tmax=tmax, baseline=baseline)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True, ecg=True, exclude='bads')
    return mne.Epochs(raw, tmp, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline,
                      reject=reject, reject_by_annotation=True, preload=True, picks=picks)

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
    regex    = ev_cfg.get("regex", r"code=(\d+)")
    verify   = bool(ev_cfg.get("verify_against_stim", False))
    tol_sec  = float(ev_cfg.get("tolerance_sec", 0.002))
    tol_samp = int(round(tol_sec * float(raw.info["sfreq"])))
    align_n  = int(ev_cfg.get("alignment_correction_samples", 0))
    stim_ch  = config.get("stim_channel", "STI101")
    offset_ms = float(ev_cfg.get("offset_ms", 0.0))
    offset_samples = int(round(offset_ms/1000.0 * float(raw.info["sfreq"])))

    logger.log("Using stim channel", stim_channel=stim_ch, events_source=source, base=base,
               tolerance_sec=tol_sec, align_correction_samples=align_n)

    # Always compute STI (for verify and/or code sourcing)
    events_stim, codes_stim = _events_from_stim(raw, stim_ch)
    logger.log("Found events on stim", n_events=len(events_stim))

    # Build annotations under both base hypotheses if needed
    def build_ann_for(base_choice: str):
        # If user asks for "auto", we fill later
        return _events_from_annotations_by_regex(raw, regex, base_choice)

    events_ann = codes_ann = None; matched_total = 0

    if source in ("auto", "annotations", "annot_timing_stim_codes"):
        # If base is auto, pick tighter |median Δsamples| vs stim
        if base == "auto":
            ev_abs, cd_abs, m_abs = build_ann_for("abs")
            ev_rel, cd_rel, m_rel = build_ann_for("rel")
            score_abs = score_rel = np.inf
            dt_abs = _nearest_dt_samples(ev_abs, events_stim) if ev_abs is not None else None
            dt_rel = _nearest_dt_samples(ev_rel, events_stim) if ev_rel is not None else None
            if dt_abs is not None and len(dt_abs): score_abs = abs(np.median(dt_abs))
            if dt_rel is not None and len(dt_rel): score_rel = abs(np.median(dt_rel))
            choose_abs = score_abs <= score_rel
            events_ann, codes_ann, matched_total = (ev_abs, cd_abs, m_abs) if choose_abs else (ev_rel, cd_rel, m_rel)
            logger.log("Auto base choice", use=("abs" if choose_abs else "rel"),
                       score_abs=score_abs, score_rel=score_rel)
        else:
            events_ann, codes_ann, matched_total = build_ann_for(base)
            logger.log("Built events from annotations", n_events=0 if events_ann is None else len(events_ann), base=base)

        if events_ann is not None and align_n:
            events_ann = events_ann.copy(); events_ann[:, 0] += int(align_n)
            logger.log("Applied alignment correction", samples=align_n)

    # Select events & values depending on source
    if source == "stim" or (source == "auto" and events_ann is None):
        events, values = events_stim, codes_stim
    elif source == "annotations":
        if events_ann is None or len(events_ann) == 0:
            raise RuntimeError("No annotation-derived events available.")
        events, values = events_ann, codes_ann
        if verify:
            # Compare to stim (counts/timing/codes)
            ia, ib = _match_events_by_time(events_ann, events_stim, tol_samp)
            logger.log("Verify annotations vs stim", matched=len(ia), annot=len(events_ann), stim=len(events_stim))
    elif source == "annot_timing_stim_codes":
        if events_ann is None or len(events_ann) == 0:
            raise RuntimeError("No annotation-derived events available for annot_timing_stim_codes.")
        # Match annotation-timed events to stim to borrow codes
        ia, ib = _match_events_by_time(events_ann, events_stim, tol_samp)
        if len(ia) == 0:
            raise RuntimeError("Could not match annotations to stim within tolerance; increase events.tolerance_sec or check base.")
        matched_ann_samples = events_ann[ia, 0]
        matched_codes_from_stim = codes_stim[ib]
        events = np.column_stack([matched_ann_samples, np.zeros_like(matched_ann_samples), matched_codes_from_stim])
        values = events[:, 2].copy()
        logger.log("Using annotation-timed events with stim codes", matched=len(ia), tol_samples=tol_samp)
    else:
        raise ValueError(f"Unknown events.source: {source}")


    # Apply global offset (ms) if requested
    if offset_samples != 0:
        events = events.copy(); events[:, 0] += int(offset_samples)
        direction = "earlier" if offset_ms < 0 else "later"
        logger.log("Applied global event offset", offset_ms=offset_ms, offset_samples=int(offset_samples), direction=direction,
                   first_event_sec=(events[0,0]/float(raw.info["sfreq"])) )
    else:
        logger.log("No global event offset applied")

    logger.log("Using events", n_events=len(events))

    # Optional artifact suppression
    if "artifact_suppression" in config:
        w = config["artifact_suppression"].get("window", None)
        if w and isinstance(w, (list, tuple)) and len(w) == 2 and w[0] is not None and w[1] is not None:
            annotate_artifacts_around_events(raw, events, (w[0], w[1]))
            logger.log("Annotated artifact window around events", window=w)

    # Optional filter
    if "filter" in config:
        fcfg = config["filter"]; l_freq = fcfg.get("l_freq", None); h_freq = fcfg.get("h_freq", None)
        logger.log("Applying filter", l_freq=l_freq, h_freq=h_freq)
        raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)

    # Reject dict
    reject = None
    if config.get("use_reject", False) and "reject" in config:
        present = Counter(raw.get_channel_types()); rcopy = dict(config["reject"])
        for k in list(rcopy):
            if k not in present:
                logger.log("Dropping reject rule (channel type not present)", ch_type=k)
                rcopy.pop(k, None)
        reject = rcopy if rcopy else None

    # Defaults
    default_tmin = float(config.get("tmin", -0.2)); default_tmax = float(config.get("tmax", 0.5))
    default_baseline = config.get("baseline", [None, 0.0])

    # Build conditions
    epochs_dict: Dict[str, mne.Epochs] = {}; evokeds: Dict[str, mne.Evoked] = {}
    cond_win = config.get("epoch_windows", {})

    for cond_name, spec in config.get("conditions", {}).items():
        ew = cond_win.get(cond_name, {})
        tmin = float(ew.get("tmin", default_tmin)); tmax = float(ew.get("tmax", default_tmax))
        baseline = ew.get("baseline", default_baseline)

        use_events = events; use_values = values
        if isinstance(spec, dict) and "stim" in spec and spec["stim"] != stim_ch and source != "annotations":
            alt = mne.find_events(raw, stim_channel=spec["stim"], shortest_event=1, initial_event=False)
            logger.log("Per-condition stim channel", condition=cond_name, stim=spec["stim"], n_events=len(alt))
            use_events = alt; use_values = alt[:, 2].copy()

        try:
            ep = create_epochs_for_condition(raw, use_events, use_values, spec, tmin, tmax, baseline, reject, logger, cond_name)
        except RuntimeError:
            logger.log("No events for condition; skipping", condition=cond_name)
            continue

        epochs_dict[cond_name] = ep; evokeds[cond_name] = ep.average()
        logger.log("Averaged condition", condition=cond_name, n_epochs=len(ep))

    # Composites
    composites = config.get("composites", {})
    if composites and evokeds:
        evaluator = _AstExprEvaluator(evokeds)
        for name, expr in composites.items():
            try:
                ev = evaluator.eval(expr); evokeds[name] = ev
            except Exception as e:
                logger.log("Failed composite", composite=name, expr=expr, error=str(e)); continue
            logger.log("Built composite", composite=name, expr=expr)

    # Outputs
    out_cfg = config.get("outputs", {})
    save_evokeds = out_cfg.get("save_evokeds", True); save_epochs = out_cfg.get("save_epochs", True)
    evk_subdir   = out_cfg.get("evokeds_subdir", "derivatives/avg")
    epo_subdir   = out_cfg.get("epochs_subdir",  "derivatives/epochs")
    base_name = output_basename(ent)

    if save_evokeds and evokeds:
        out_dir_evk = output_dir_for_subdir(bids_root, ent, evk_subdir)
        for name, ev in evokeds.items():
            fn = out_dir_evk / f"{base_name}_desc-{name}_ave.fif"
            ev.save(fn, overwrite=True); logger.log("Saved", file=str(fn))

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
        "timestamp_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provenance": {
            "config": config,
            "raw_input_file": str(fif_path),
            "bids": {"subject": ent.get("subject"), "session": ent.get("session"),
                     "task": ent.get("task"), "run": ent.get("run")},
        },
        "mne_info": {
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(raw.info["nchan"]),
            "stim_channel": config.get("stim_channel", "STI101"),
        },
        "events": {
            "source": str((config.get("events", {}) or {}).get("source", "auto")),
            "regex":  str((config.get("events", {}) or {}).get("regex", r"TRIG/(\\d+)")),
            "offset_ms": float((config.get("events", {}) or {}).get("offset_ms", 0.0)),
            "first_event_time_sec": first_event_time_sec,
            "n_events_total": int(len(events)),
        },
        "artifact_rejection": {
            "annotation_reject_applied": True,
            "local_reject_applied": bool(config.get("use_reject", False) and config.get("reject")),
            "local_reject_params": config.get("reject") if config.get("use_reject", False) else None,
        },
        "conditions": {},
        "outputs": {"files": []},
    }

    # Per-condition tallies using stored Epochs
    if epochs_dict:
        for _cname, _ep in epochs_dict.items():
            _total = int(len(_ep.drop_log))
            _kept = int(len(_ep))
            _rej = int(_total - _kept)
            # Tally reasons
            _counts = {}
            for _reasons in _ep.drop_log:
                if not _reasons: continue
                for _r in _reasons:
                    _counts[_r] = _counts.get(_r, 0) + 1
            run_log["conditions"][_cname] = {
                "n_events_selected": _total,
                "n_epochs_kept": _kept,
                "n_epochs_rejected": _rej,
                "drop_reason_counts": _counts,
            }

    # Record output file paths we just saved
    # (out_dir_evk/out_dir_epo/base_name/condition names exist in this scope)
    try:
        if save_evokeds and evokeds:
            for _name in evokeds.keys():
                _p = (out_dir_evk / f"{base_name}_desc-{_name}_ave.fif")
                run_log["outputs"]["files"].append({"type": "evoked", "condition": _name, "path": str(_p)})
        if save_epochs and epochs_dict:
            for _name in epochs_dict.keys():
                _p = (out_dir_epo / f"{base_name}_desc-{_name}_epo.fif")
                run_log["outputs"]["files"].append({"type": "epochs", "condition": _name, "path": str(_p)})
    except Exception as _e:
        pass

    # Write logs to BOTH evokeds and epochs directories
    _log_stem = f"{base_name}_desc-runlog_phase10-2025-10-23"
    try:
        import json as _json
        # evoked dir
        _json_path1 = (out_dir_evk / f"{_log_stem}.json")
        with open(_json_path1, "w") as _f: _json.dump(run_log, _f, indent=2)
        # epochs dir
        _json_path2 = (out_dir_epo / f"{_log_stem}.json")
        with open(_json_path2, "w") as _f: _json.dump(run_log, _f, indent=2)
        logger.log("Wrote run logs (JSON)", evoked=str(_json_path1), epochs=str(_json_path2))
    except Exception as _e:
        logger.log("Failed to write JSON run logs", error=str(_e))

    if _R_YAML_WRITER is not None:
        try:
            _yml1 = (out_dir_evk / f"{_log_stem}.yml")
            with open(_yml1, "w") as _f: _R_YAML_WRITER.dump(run_log, _f)
            _yml2 = (out_dir_epo / f"{_log_stem}.yml")
            with open(_yml2, "w") as _f: _R_YAML_WRITER.dump(run_log, _f)
            logger.log("Wrote run logs (YAML)", evoked=str(_yml1), epochs=str(_yml2))
        except Exception as _e:
            logger.log("Failed to write YAML run logs", error=str(_e))
    else:
        logger.log("YAML run log skipped (ruamel.yaml not installed)")

# ----------------------------------- CLI --------------------------------------
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
