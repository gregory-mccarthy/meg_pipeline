# bids_io_utils.py
# Robust BIDS I/O helpers for Elekta/MEGIN FIF with optional run entity
# and dual split conventions (BIDS split-* and legacy MEGIN -1, -2 files).
#
# Additions in this revision:
# - Fault-tolerant numeric matching for sub/ses/run/split (1 == 01 == 001)
# - Directory probing to use what's on disk, with canonical fallback
# - Derivatives helpers that create canonical dirs when we create them
# - Subject normalization set to 3 digits for emitted names (sub-001)

from __future__ import annotations

import re
import socket
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import mne

# =============================================================================
# Basic utilities and normalization
# =============================================================================

def _none_like(x) -> bool:
    return x in (None, "", False, "None", "null", "NULL")

def _strip_entity_prefix(s: str) -> str:
    for p in ("sub-", "ses-", "run-", "split-"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _canon_numeric(val, width: int) -> Optional[str]:
    if _none_like(val):
        return None
    s = str(val).strip()
    s = _strip_entity_prefix(s)
    return s.zfill(width) if s.isdigit() else s

# --- Canonicalizers (tolerant on input) --------------------------------------
# NOTE: subject width set to 3 per your request (sub-001).
def norm_subject(val) -> Optional[str]:
    return _canon_numeric(val, 3)

def norm_session(val) -> Optional[str]:
    return _canon_numeric(val, 2)

def norm_run(val) -> Optional[str]:
    return _canon_numeric(val, 2)

def norm_split(val) -> Optional[str]:
    return _canon_numeric(val, 2)

def _ent(stem: str, key: str, val: Optional[str]) -> str:
    return f"{stem}_{key}-{val}" if not _none_like(val) else stem

# ####################################################
# Time window parsing helpers (additive; safe)
# ####################################################

def _none_like(val) -> bool:
    return val is None or (isinstance(val, str) and val.strip().lower() in {"", "none", "null"})

def _parse_hms_or_float(val):
    """Return seconds (float) from 'hh:mm:ss' | 'mm:ss' | float/int | None."""
    if _none_like(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        else:
            return float(s)
    except Exception as e:
        raise ValueError(f"Unrecognized time format '{val}': {e}")

def parse_time_window(cfg: dict):
    """Read optional time window from YAML-like dict.
    Returns (tmin, tmax) in seconds or None.
    """
    tw = (cfg or {}).get("time_window")
    if not tw:
        return None
    tmin = _parse_hms_or_float(tw.get("start"))
    tmax = _parse_hms_or_float(tw.get("end"))
    if tmin is None and tmax is None:
        return None
    if (tmin is not None) and (tmax is not None) and (tmax <= tmin):
        raise ValueError(f"time_window end ({tmax}) must be > start ({tmin}).")
    return (tmin, tmax)

# =============================================================================
# Fault-tolerant helpers: numeric equivalence and on-disk probing
# =============================================================================

def _int_if_numeric(s: Optional[str]) -> Optional[int]:
    if _none_like(s):
        return None
    try:
        return int(_strip_entity_prefix(str(s)).strip())
    except Exception:
        return None

def _numeric_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Treat '1', '01', '001' as equal numbers; exact compare otherwise."""
    ia, ib = _int_if_numeric(a), _int_if_numeric(b)
    if ia is not None and ib is not None:
        return ia == ib
    return (a or "") == (b or "")

def _variants_numeric(val: Optional[str], widths=(1, 2, 3)) -> List[str]:
    """Return plausible numeric string variants plus the original."""
    out: List[str] = []
    if _none_like(val):
        return out
    raw = str(val).strip()
    if raw and raw not in out:
        out.append(raw)
    core = _strip_entity_prefix(raw)
    if core != raw and core not in out:
        out.append(core)
    if core.isdigit():
        n = int(core)
        for w in widths:
            s = (str(n).zfill(w) if w > 1 else str(n))
            if s not in out:
                out.append(s)
    return out

def _find_existing_meg_dir(bids_root: Path, subject: str, session: Optional[str]) -> Optional[Path]:
    """Search the filesystem for an existing MEG dir for the requested subject/session,
    tolerating sub-1/sub-01/sub-001 and ses-1/ses-01. Returns Path or None."""
    subj_can = norm_subject(subject)
    ses_can  = norm_session(session)

    # Try exact canonical first (fast path)
    exact = bids_root / f"sub-{subj_can}" / ("meg" if _none_like(ses_can) else f"ses-{ses_can}/meg")
    if exact.exists():
        return exact

    # Scan sub-* dirs and ses-* children with numeric equality
    for sub_dir in sorted(p for p in bids_root.glob("sub-*") if p.is_dir()):
        sub_lab = sub_dir.name.split("-", 1)[-1]
        if not _numeric_equal(sub_lab, subject):
            continue
        if _none_like(session):
            cand = sub_dir / "meg"
            if cand.exists():
                return cand
        else:
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                ses_lab = ses_dir.name.split("-", 1)[-1]
                if _numeric_equal(ses_lab, session):
                    cand = ses_dir / "meg"
                    if cand.exists():
                        return cand
    return None

def _filter_candidates_by_task_run(files: List[Path], task: Optional[str], run: Optional[str]) -> List[Path]:
    """Parse filenames and keep only those matching task exactly and run numerically."""
    keep: List[Path] = []
    for p in files:
        try:
            info = parse_meg_fname(p.name)
        except ValueError:
            continue
        if not _none_like(task) and info.get("task") != task:
            continue
        if not _none_like(run) and not _numeric_equal(info.get("run"), run):
            continue
        keep.append(p)
    return keep

# =============================================================================
# Stems and directory resolution
# =============================================================================

def build_bids_stem(subject: str,
                    session: Optional[str] = None,
                    task: Optional[str] = None,
                    run: Optional[str] = None) -> str:
    """
    Return canonical BIDS stem 'sub-XXX[_ses-YY][_task-ZZ][_run-WW]'.
    """
    s  = norm_subject(subject)
    se = norm_session(session)
    r  = norm_run(run)
    stem = f"sub-{s}"
    stem = _ent(stem, "ses", se)
    stem = _ent(stem, "task", task if not _none_like(task) else None)
    stem = _ent(stem, "run", r)
    return stem

def resolve_meg_dir(bids_root: Union[str, Path],
                    subject: str,
                    session: Optional[str] = None) -> Path:
    """
    Return the MEG directory path for the subject/session:
      <root>/sub-<subject>/[ses-<session>/]meg

    Fault-tolerant: prefer an existing on-disk directory that matches numerically
    (e.g., sub-1 == sub-01 == sub-001; ses-1 == ses-01). If none exists, return
    the canonical path (so writers emit a consistent form).
    """
    bids_root = Path(bids_root)
    found = _find_existing_meg_dir(bids_root, subject, session)
    if found is not None:
        return found

    s  = norm_subject(subject)
    se = norm_session(session)
    parts = [bids_root, f"sub-{s}"]
    if not _none_like(se):
        parts.append(f"ses-{se}")
    parts.append("meg")
    return Path(*parts)

# =============================================================================
# Derivatives directory helpers (canonical on create)
# =============================================================================

def _find_existing_deriv_sub_ses(deriv_root: Path,
                                 subject: str,
                                 session: Optional[str]) -> Optional[Path]:
    """
    Search deriv_root for an existing sub-*/[ses-*/] that numerically equals the
    requested subject/session. Return the first match or None.
    """
    for sub_dir in sorted(p for p in deriv_root.glob("sub-*") if p.is_dir()):
        sub_lab = sub_dir.name.split("-", 1)[-1]
        if not _numeric_equal(sub_lab, subject):
            continue
        if _none_like(session):
            return sub_dir
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses_lab = ses_dir.name.split("-", 1)[-1]
            if _numeric_equal(ses_lab, session):
                return ses_dir
    return None

def ensure_derivatives_dir(deriv_root: Union[str, Path],
                           subject: str,
                           session: Optional[str] = None) -> Path:
    """
    Ensure a derivatives subject[/session] directory exists, enforcing canonical
    names *when we are creating* a new path.

    Behavior:
      - If a numerically equivalent sub/ses directory already exists (e.g., sub-1),
        return that existing directory unchanged.
      - Otherwise, create and return the canonical path:
            deriv_root/sub-<norm_subject>/[ses-<norm_session>]
    """
    deriv_root = Path(deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)

    found = _find_existing_deriv_sub_ses(deriv_root, subject, session)
    if found is not None:
        found.mkdir(parents=True, exist_ok=True)
        return found

    s  = norm_subject(subject)   # 3-digit
    se = norm_session(session)   # 2-digit
    out = deriv_root / f"sub-{s}"
    if not _none_like(se):
        out = out / f"ses-{se}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def make_derivative_path(bids_root: Union[str, Path],
                         pipeline_name: Optional[str],
                         subject: str,
                         session: Optional[str],
                         relative_parts: List[str]) -> Path:
    """
    Returns a full Path under derivatives, ensuring canonical dirs iff created.

    Example:
        stem = build_bids_stem(subject, session, task, run)
        svg  = make_derivative_path(bids_root, "meg-preproc-v1", subject, session,
                                    ["figures", f"{stem}_ica_qc.svg"])
    """
    root = Path(bids_root) / "derivatives"
    if pipeline_name:
        root = root / pipeline_name
    sub_ses_dir = ensure_derivatives_dir(root, subject, session)
    out = sub_ses_dir
    for p in relative_parts:
        out = out / p
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

# =============================================================================
# Split enumeration (BIDS split + legacy MEGIN)
# =============================================================================

_SPLIT01 = "_split-01_meg.fif"
_SPLITALL = "_split-*_meg.fif"
_SPLIT_RE = r"_split-(\d+)_meg\.fif$"
_MEGBAS = "_meg.fif"
_MEGNUM = r"_meg-(\d+)\.fif$"

def list_all_split_files(first_file: Union[str, Path]) -> List[Path]:
    """
    Given the 'first' file for a run, list all FIF files that belong to it
    across both split conventions, in natural order.
    """
    first_file = Path(first_file).resolve()
    parent = first_file.parent
    name = first_file.name

    if name.endswith(_SPLIT01) or re.search(_SPLIT_RE, name):
        base = re.sub(_SPLIT_RE, "_split-", name)
        base = base[:-len("split-")] if base.endswith("split-") else base[:-len("_split-")]
        files = sorted(parent.glob(f"{base}{_SPLITALL}"))
        return [f.resolve() for f in files]

    if name.endswith(_MEGBAS):
        base = name[:-len(_MEGBAS)]
        files = [first_file] + sorted(parent.glob(f"{base}_meg-*.fif"))
        return [f.resolve() for f in files]

    if re.search(_MEGNUM, name):
        base = re.sub(_MEGNUM, "", name)
        candidates = []
        single = parent / f"{base}{_MEGBAS}"
        if single.exists():
            candidates.append(single)
        candidates.extend(sorted(parent.glob(f"{base}_meg-*.fif")))
        return [f.resolve() for f in candidates]

    return [first_file]

# =============================================================================
# Run discovery
# =============================================================================

def discover_runs(bids_root: Union[str, Path],
                  subject: str,
                  session: Optional[str] = None,
                  task: Optional[str] = None) -> List[str]:
    """
    Discover available 'run-XX' entities for sub/ses/task by scanning filenames.
    Returns a sorted list of run strings (e.g., ['01', '02']). If there is a
    single-run layout with no run entity, returns [].
    """
    meg_dir = resolve_meg_dir(bids_root, subject, session)
    if not meg_dir.exists():
        return []
    stem_norun = build_bids_stem(subject, session, task, run=None)
    runs = set()
    for p in meg_dir.glob(f"{stem_norun}_run-*_meg*.fif"):
        m = re.search(r"_run-([A-Za-z0-9]+)_meg", p.name)
        if m:
            runs.add(norm_run(m.group(1)) or m.group(1))
    out = sorted(runs)
    if any(len(r)==1 for r in runs) and any(len(r)==2 for r in runs):
        print(f"[WARN] Mixed run padding detected in {meg_dir}; canonical: {out}")
    return out

# =============================================================================
# First-file selection (fault tolerant)
# =============================================================================

def find_first_file_for_run(bids_root: Union[str, Path],
                            subject: str,
                            session: Optional[str] = None,
                            task: Optional[str] = None,
                            run: Optional[str] = None) -> Tuple[Path, str, List[Path]]:
    """
    Locate the appropriate 'first' file to open for a given run.
    Returns (first_file, style, all_candidates), style ∈ {'bids-split','megin-split','single'}.
    Fault-tolerant to padding and split styles.
    """
    run_c = norm_run(run)
    bids_root = Path(bids_root)

    meg_dir = resolve_meg_dir(bids_root, subject, session)
    if not meg_dir.exists():
        subj_dir = meg_dir.parent if meg_dir.name == "meg" else meg_dir
        listing = sorted(p.name for p in subj_dir.glob("*")) if subj_dir.exists() else []
        raise FileNotFoundError(
            f"MEG directory not found: {meg_dir}\n"
            f"Subject directory contents:\n  " + ("\n  ".join(listing) if listing else "(none)")
        )

    stem = build_bids_stem(norm_subject(subject), norm_session(session), task, run_c)
    candidates: List[Path] = []

    # Fast-path patterns
    for pat in (f"{stem}_split-01_meg.fif", f"{stem}_meg.fif", f"{stem}_meg-1.fif"):
        candidates.extend(sorted(meg_dir.glob(pat)))

    # Flexible scan if needed: parse and filter by task + numeric run
    if not candidates:
        all_meg = sorted(meg_dir.glob("*.fif"))
        candidates = _filter_candidates_by_task_run(all_meg, task=task, run=run)

    # Deduplicate preserving order
    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        all_names = sorted(p.name for p in meg_dir.glob("*.fif"))
        msg = [f"No MEG FIF matched in {meg_dir}",
               f"Requested task={task!r} run={run!r} (numeric match allowed)",
               "Available FIF files:"]
        msg += [f"  - {n}" for n in all_names] if all_names else ["  (none)"]
        raise FileNotFoundError("\n".join(msg))

    # If run omitted and multiple distinct runs exist, ask for specificity
    if _none_like(run):
        found_runs = set()
        for p in candidates:
            try:
                info = parse_meg_fname(p.name)
                if info.get("run"):
                    found_runs.add(norm_run(info["run"]) or info["run"])
            except ValueError:
                continue
        if len(found_runs) > 1:
            raise ValueError(
                "Ambiguous: run was omitted but multiple runs were found: "
                f"{sorted(found_runs)}. Specify e.g., run: '01'."
            )

    # Rank first file: split-01 > meg-1 > single
    def _rank_first_file(p: Path) -> int:
        name = p.name
        if name.endswith("_split-01_meg.fif"):
            return 0
        m = re.search(r"_meg-(\d+)\.fif$", name)
        if m:
            n = int(m.group(1))
            return 1 + min(n, 9998)
        if name.endswith("_meg.fif"):
            return 10000
        return 20000

    candidates.sort(key=_rank_first_file)
    first = candidates[0]
    nm = first.name
    if nm.endswith("_split-01_meg.fif") or re.search(r"_split-\d+_meg\.fif$", nm):
        style = "bids-split"
    elif re.search(r"_meg-\d+\.fif$", nm):
        style = "megin-split"
    else:
        style = "single"
    return first.resolve(), style, [c.resolve() for c in candidates]

# =============================================================================
# Reading & writing
# =============================================================================

def read_raw_bids_smart(
    bids_root_or_path: Union[str, Path, "mne_bids.BIDSPath", Dict[str, str]],
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    **read_kwargs,
):
    """
    Smart BIDS reader:
      - Accepts either a BIDS root + entities or a BIDSPath/dict with entities.
      - Run can be None/'' for single-run layouts.
      - Auto-detects split style and opens the correct first file; MNE stitches the rest.

    Returns:
      raw : mne.io.Raw
      info: dict(style, first_file, all_matches)
    """
    bids_root: Optional[Path] = None
    if hasattr(bids_root_or_path, "root"):
        bp = bids_root_or_path
        bids_root = Path(bp.root)
        subject = getattr(bp, "subject", subject)
        session = getattr(bp, "session", session)
        task = getattr(bp, "task", task)
        run = getattr(bp, "run", run)
    elif isinstance(bids_root_or_path, dict):
        d = bids_root_or_path
        bids_root = Path(d.get("root") or d.get("bids_root") or ".")
        subject = d.get("subject", subject)
        session = d.get("session", session)
        task = d.get("task", task)
        run = d.get("run", run)
    else:
        bids_root = Path(bids_root_or_path)

    # Canonicalize (we still do tolerant probing later)
    subject = norm_subject(subject)
    session = norm_session(session)
    run = norm_run(run)

    first, style, candidates = find_first_file_for_run(
        bids_root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
    )

    raw = mne.io.read_raw_fif(first, **read_kwargs)
    info = dict(style=style, first_file=str(first), all_matches=[str(c) for c in candidates])
    return raw, info

def _infer_entities_from_raw(raw) -> Dict[str, Optional[str]]:
    """
    Try to infer subject/session/task/run from the Raw's original filename.
    Uses raw.filenames[0] (MNE) if available; falls back to raw.fids/'meas_file' if present.
    Returns dict with keys: subject, session, task, run (each may be None).
    """
    cand = None
    # MNE Raw usually has .filenames (list or tuple)
    fn = getattr(raw, "filenames", None)
    if fn:
        try:
            cand = Path(fn[0]).name
        except Exception:
            pass
    # Some objects store a single filename
    if cand is None:
        fn = getattr(raw, "filename", None)
        if fn:
            cand = Path(fn).name
    # Very defensive: MNE sometimes stashes metadata in info; keep as last resort
    if cand is None:
        meas = raw.info.get("meas_file") if hasattr(raw, "info") else None
        if meas:
            try:
                cand = Path(meas).name
            except Exception:
                pass

    entities = dict(subject=None, session=None, task=None, run=None)
    if cand:
        try:
            info = parse_meg_fname(cand)
            entities.update({
                "subject": info.get("subject"),
                "session": info.get("session"),
                "task":    info.get("task"),
                "run":     info.get("run"),
            })
        except Exception:
            # Not a fatal error; just means we couldn’t infer
            pass
    return entities


def write_bids_robust(raw: mne.io.BaseRaw,
                      bids_root: Union[str, Path],
                      subject: Optional[str] = None,
                      session: Optional[str] = None,
                      task: Optional[str] = None,
                      run: Optional[str] = None,
                      overwrite: bool = False,
                      **kwargs) -> Path:
    """
    Write raw to BIDS-like MEG path, supporting single file or split files.
    Emits canonical stems; does not enforce full BIDS sidecars.

    Entities can be omitted; if so we attempt to infer them from raw.filenames[0]
    (or similar) using parse_meg_fname(). Any explicitly provided entity wins.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    # 1) Infer missing entities from the raw’s original filename, if possible
    if subject is None or session is None or task is None or run is None:
        inferred = _infer_entities_from_raw(raw)
        subject = subject if subject is not None else inferred.get("subject")
        session = session if session is not None else inferred.get("session")
        task    = task    if task    is not None else inferred.get("task")
        run     = run     if run     is not None else inferred.get("run")

    # 2) Normalize entities (tolerant input, canonical output)
    s  = norm_subject(subject)
    se = norm_session(session)
    r  = norm_run(run)
    # task is free text; leave as-is except None-like
    t  = None if _none_like(task) else str(task)

    # 3) Resolve MEG directory (fault-tolerant to on-disk padding); create if needed
    meg_dir = resolve_meg_dir(bids_root, s, se)
    meg_dir.mkdir(parents=True, exist_ok=True)

    # 4) Build canonical stem and output path
    stem = build_bids_stem(s, se, t, r)
    out = meg_dir / f"{stem}_meg.fif"

    # 5) Write
    if out.exists() and not overwrite:
        raise FileExistsError(f"File exists: {out}")
    raw.save(str(out), overwrite=overwrite, **kwargs)
    return out

# =============================================================================
# Environment & transfer helpers
# =============================================================================

def detect_environment(hpc_hostname_tag: str = "milgram") -> str:
    """Crude environment detection based on hostname to pick transfer defaults."""
    try:
        host = socket.gethostname().lower()
    except Exception:
        return "unknown"
    if hpc_hostname_tag in host:
        return "hpc"
    return "local"

def get_bids_headpos_path(subject: str,
                          session: Optional[str],
                          task: str,
                          run: Optional[str],
                          bids_root: Union[str, Path]) -> Path:
    """Return the expected path for MaxFilter head position text file alongside FIF."""
    s  = norm_subject(subject)
    se = norm_session(session)
    r  = norm_run(run)
    stem = build_bids_stem(s, se, task, r)
    meg_dir = resolve_meg_dir(bids_root, s, se)
    return meg_dir / f"{stem}_headpos.txt"

def _rsync_cmd(base_args: Iterable[str]) -> List[str]:
    return ["rsync", "-av", "--partial", "--progress", *base_args]

def fetch_bids_data_and_sidecars(hpc_host: str, hpc_user: str,
                                 remote_bids_root: Union[str, Path],
                                 local_bids_root: Union[str, Path],
                                 subject: str,
                                 session: Optional[str],
                                 task: Optional[str],
                                 run: Optional[str]) -> None:
    """Minimal fetch using rsync include rules; tolerant of run omission."""
    s  = norm_subject(subject)
    se = norm_session(session)
    r  = norm_run(run)

    remote = str(Path(remote_bids_root).as_posix()).rstrip("/")
    local = Path(local_bids_root).expanduser().resolve()

    dpre = f"sub-{s}" + (f"/ses-{se}" if not _none_like(se) else "")
    fpre = f"sub-{s}" + (f"_ses-{se}" if not _none_like(se) else "")

    run_pat = f"run-{r}" if not _none_like(r) else "run-*"
    task_pat = f"task-{task}" if not _none_like(task) else "task-*"

    includes = [
        f"{dpre}/",
        f"{dpre}/meg/",
        f"{dpre}/meg/{fpre}_{task_pat}_{run_pat}_split-*_meg.fif",
        f"{dpre}/meg/{fpre}_{task_pat}_{run_pat}_meg-*.fif",
        f"{dpre}/meg/{fpre}_{task_pat}_{run_pat}_meg.fif",
        f"{dpre}/meg/{fpre}_{task_pat}_split-*_meg.fif",
        f"{dpre}/meg/{fpre}_{task_pat}_meg-*.fif",
        f"{dpre}/meg/{fpre}_{task_pat}_meg.fif",
        f"{dpre}/meg/{fpre}_split-*_meg.fif",
        f"{dpre}/meg/{fpre}_meg-*.fif",
        f"{dpre}/meg/{fpre}_meg.fif",
    ]

    cmd = _rsync_cmd([*(arg for inc in includes for arg in ("--include", inc)),
                      "--exclude", "*",
                      f"{hpc_user}@{hpc_host}:{remote}/{dpre}/meg/",
                      str(local / dpre / "meg")])
    subprocess.run(cmd, check=True)

def push_bids_derivatives_rsync(local_bids_root: Union[str, Path],
                                remote_bids_root: Union[str, Path],
                                hpc_host: str,
                                hpc_user: str) -> None:
    """Push local derivatives to remote using rsync. Caller chooses layout."""
    local = Path(local_bids_root).expanduser().resolve()
    remote = str(Path(remote_bids_root).as_posix()).rstrip("/")
    cmd = _rsync_cmd([str(local) + "/", f"{hpc_user}@{hpc_host}:{remote}"])
    subprocess.run(cmd, check=True)

# =============================================================================
# Tolerant filename parse & symlink repair
# =============================================================================

BIDS_FIF_RE = re.compile(
    r"^sub-(?P<sub>[A-Za-z0-9]+)"
    r"(?:_ses-(?P<ses>[A-Za-z0-9]+))?"
    r"(?:_task-(?P<task>[A-Za-z0-9]+))?"
    r"(?:_run-(?P<run>[A-Za-z0-9]+))?"
    r"_(?:(?P<bids_split>split)-(?P<split>[A-Za-z0-9]+)_)?meg(?:-(?P<legacy_part>\d+))?\.fif$"
)

def parse_meg_fname(name: str) -> Dict[str, Optional[str]]:
    m = BIDS_FIF_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized MEG filename: {name}")
    d = m.groupdict()
    sub = norm_subject(d["sub"])
    ses = norm_session(d["ses"])
    task = d["task"]
    run = norm_run(d["run"])
    if d["bids_split"]:
        style = "bids-split"
        split = norm_split(d["split"])
    elif d["legacy_part"]:
        style = "megin-split"
        split = norm_split(d["legacy_part"])
    else:
        style = "single"
        split = None
    return dict(subject=sub, session=ses, task=task, run=run, split=split, style=style)

def ensure_canonical_symlinks(meg_dir: Union[str, Path]) -> int:
    """
    For each MEG FIF in `meg_dir`, create a canonical BIDS-named symlink
    alongside the file if the on-disk name is non-canonical.
    Returns count of symlinks created.
    """
    meg_dir = Path(meg_dir)
    n = 0
    for p in meg_dir.glob("*.fif"):
        try:
            info = parse_meg_fname(p.name)
        except ValueError:
            continue
        stem = build_bids_stem(info["subject"], info["session"], info["task"], info["run"])
        if info["style"] == "bids-split" and info["split"]:
            cname = f"{stem}_split-{info['split']}_meg.fif"
        elif info["style"] == "megin-split" and info["split"]:
            cname = f"{stem}_meg-{info['split']}.fif"
        else:
            cname = f"{stem}_meg.fif"
        target = p.parent / cname
        if cname != p.name and not target.exists():
            try:
                target.symlink_to(p.name)
                n += 1
            except OSError:
                pass
    return n

# =============================================================================
# Convenience wrappers
# =============================================================================

def get_all_bids_split_files(base_fif: Union[str, Path]) -> List[str]:
    """
    Return a list of all FIFs in a split set given any member of the set.
    """
    return [str(p) for p in list_all_split_files(base_fif)]

def read_raw_bids_robust(bids_path_or_dict, **kwargs):
    """
    Convenience wrapper around read_raw_bids_smart returning just Raw.
    """
    raw, _ = read_raw_bids_smart(bids_path_or_dict, **kwargs)
    return raw
