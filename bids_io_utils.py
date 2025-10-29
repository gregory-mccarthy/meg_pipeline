# bids_io_utils.py
# Robust BIDS I/O helpers for Elekta/MEGIN FIF with optional run entity
# and dual split conventions (BIDS split-* and legacy MEGIN -1, -2 files).

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple, Union

import mne

# ------------------------------- Normalization --------------------------------

def _none_like(x) -> bool:
    return x in (None, "", False, "None", "null", "NULL")

def norm_run(val) -> Optional[str]:
    """
    Normalize a run value:
      - None / '' / falsey → None (no run entity in filename)
      - Numeric strings → zero-pad to 2 (e.g., '2' -> '02')
      - Non-numeric strings left as-is
    """
    if _none_like(val):
        return None
    s = str(val)
    return s.zfill(2) if s.isdigit() else s

def _ent(s: str, key: str, val: Optional[str]) -> str:
    """Append a BIDS entity if val is not None."""
    return f"{s}_{key}-{val}" if not _none_like(val) else s

# -------------------------------- Path builders -------------------------------

def build_bids_stem(subject: str,
                    session: Optional[str] = None,
                    task: Optional[str] = None,
                    run: Optional[str] = None) -> str:
    """
    Build the filename stem up to (but not including) suffix/extension, e.g.:
      sub-001_ses-01_task-fairy_run-02
    Run is optional; if None, it is omitted entirely.
    """
    run = norm_run(run)
    stem = f"sub-{subject}"
    stem = _ent(stem, "ses", session)
    stem = _ent(stem, "task", task)
    stem = _ent(stem, "run", run)
    return stem

def resolve_meg_dir(bids_root: Union[str, Path],
                    subject: str,
                    session: Optional[str] = None) -> Path:
    """
    Return the MEG directory path for the subject/session:
      <root>/sub-<subject>/[ses-<session>/]meg
    """
    bids_root = Path(bids_root)
    parts = [bids_root, f"sub-{subject}"]
    if not _none_like(session):
        parts.append(f"ses-{session}")
    parts.append("meg")
    return Path(*parts)

# ------------------------------ Split enumeration -----------------------------

_SPLIT01 = "_split-01_meg.fif"
_SPLITALL = "_split-*_meg.fif"
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

    if name.endswith(_SPLIT01):
        base = name[:-len(_SPLIT01)]  # remove "_split-01_meg.fif"
        files = sorted(parent.glob(f"{base}{_SPLITALL}"))
        return [f.resolve() for f in files]

    if name.endswith(_MEGBAS):
        base = name[:-len(_MEGBAS)]  # remove "_meg.fif"
        files = [first_file] + sorted(parent.glob(f"{base}_meg-*.fif"))
        return [f.resolve() for f in files]

    if re.search(_MEGNUM, name):
        # e.g., sub-XXX_..._meg-1.fif → include base if it exists, then all _meg-*.fif
        base = re.sub(_MEGNUM, "", name)
        candidates = []
        base_path = parent / f"{base}{_MEGBAS}"
        if base_path.exists():
            candidates.append(base_path)
        candidates.extend(sorted(parent.glob(f"{base}_meg-*.fif")))
        return [f.resolve() for f in candidates]

    # Fallback: not recognized as a split starter; return itself
    return [first_file]

# --------------------------------- Discovery ----------------------------------

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
    # Match files that have run-XX
    for p in meg_dir.glob(f"{stem_norun}_run-*_meg*.fif"):
        m = re.search(r"_run-([A-Za-z0-9]+)_meg", p.name)
        if m:
            runs.add(m.group(1))
    return sorted(runs)

def _rank_first_file(p: Path) -> int:
    # Prefer BIDS split-01, then base _meg.fif, then numbered -1
    s = p.name
    if s.endswith(_SPLIT01):
        return 0
    if s.endswith(_MEGBAS):
        return 1
    if re.search(_MEGNUM, s):
        return 2
    return 9

def find_first_file_for_run(bids_root: Union[str, Path],
                            subject: str,
                            session: Optional[str] = None,
                            task: Optional[str] = None,
                            run: Optional[str] = None) -> Tuple[Path, str, List[Path]]:
    """
    Locate the appropriate 'first' file to open for a given run, allowing:
      - run omitted (None/'') for single-run layouts
      - both split conventions
    Returns (first_file, style, all_candidates), where style ∈ {'bids-split','megin-split','single'}.
    Raises FileNotFoundError if no candidate found.
    """
    run = norm_run(run)
    meg_dir = resolve_meg_dir(bids_root, subject, session)
    if not meg_dir.exists():
        raise FileNotFoundError(f"MEG directory not found: {meg_dir}")

    stem = build_bids_stem(subject, session, task, run)
    candidates: List[Path] = []

    # Direct candidates (with given run, or no run if run=None)
    patterns = [
        f"{stem}{_SPLIT01}",
        f"{stem}{_MEGBAS}",
        f"{stem}_meg-1.fif",
    ]
    for pat in patterns:
        candidates.extend(sorted(meg_dir.glob(pat)))

    # If run omitted, also try discovering with/without run entities
    if _none_like(run):
        # We already searched without run above, but also search the with-run variants
        # in case the user omitted run but files have it (we'll handle ambiguity later).
        stem_norun = build_bids_stem(subject, session, task, run=None)
        for p in meg_dir.glob(f"{stem_norun}_run-*_meg.fif"):
            candidates.append(p)
        for p in meg_dir.glob(f"{stem_norun}_run-*_split-01_meg.fif"):
            candidates.append(p)
        for p in meg_dir.glob(f"{stem_norun}_run-*_meg-1.fif"):
            candidates.append(p)

    # Deduplicate while preserving order
    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        # Help the user by listing what's in the folder
        listing = sorted(str(p.name) for p in meg_dir.glob("*.fif"))
        raise FileNotFoundError(
            f"No MEG FIF file found for sub={subject!r} ses={session!r} task={task!r} run={run!r} in {meg_dir}\n"
            f"Available FIF files:\n  " + "\n  ".join(listing)
        )

    # If run is omitted and multiple distinct run candidates exist, be explicit
    if _none_like(run):
        # Extract distinct run labels present among candidates
        found_runs = set()
        for p in candidates:
            m = re.search(r"_run-([A-Za-z0-9]+)_meg", p.name)
            if m:
                found_runs.add(m.group(1))
        if len(found_runs) > 1:
            raise ValueError(
                "Ambiguous: run was omitted but multiple runs were found: "
                f"{sorted(found_runs)}. Specify a concrete run in YAML (e.g., run: '01')."
            )

    candidates.sort(key=_rank_first_file)
    first = candidates[0]
    name = first.name
    if name.endswith(_SPLIT01):
        style = "bids-split"
    elif re.search(_MEGNUM, name):
        style = "megin-split"
    else:
        style = "single"

    return first.resolve(), style, [c.resolve() for c in candidates]

# ------------------------------------ Read ------------------------------------

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
      - Accepts either a BIDS root + entities (subject/session/task/run)
        OR an mne_bids.BIDSPath / dict with those fields.
      - Treats run as optional (None/'' means no run entity).
      - Auto-detects split style and opens the correct first file; MNE stitches the rest.

    Returns:
      raw : mne.io.Raw
      info : dict(style, first_file, all_matches)

    Example:
      raw, info = read_raw_bids_smart(bids_root, subject="001", session="01", task="fairy", run=None)
    """
    # Unpack various input styles
    bids_root: Optional[Path] = None
    if hasattr(bids_root_or_path, "root"):  # likely a BIDSPath
        bp = bids_root_or_path
        # mne-bids BIDSPath always has .root; entities may be None
        bids_root = Path(bp.root)
        subject = getattr(bp, "subject", subject)
        session = getattr(bp, "session", session)
        task = getattr(bp, "task", task)
        run = getattr(bp, "run", run)
    elif isinstance(bids_root_or_path, dict):
        d = bids_root_or_path
        bids_root = Path(d.get("bids_root") or d.get("root") or d.get("bidsroot"))
        subject = d.get("subject", subject)
        session = d.get("session", session)
        task = d.get("task", task)
        run = d.get("run", run)
    else:
        bids_root = Path(bids_root_or_path)

    run = norm_run(run)

    if "preload" not in read_kwargs:
        read_kwargs["preload"] = True

    first_file, style, matches = find_first_file_for_run(
        bids_root=bids_root, subject=subject, session=session, task=task, run=run
    )
    print(f"[bids_io_utils] Opening ({style}) → {first_file}")
    raw = mne.io.read_raw_fif(first_file, **read_kwargs)
    return raw, {"style": style, "first_file": first_file, "all_matches": matches}

# ------------------------------------ Write -----------------------------------

def write_bids_robust(raw: mne.io.BaseRaw,
                      out_path: Union[str, Path, Dict, "mne_bids.BIDSPath"],
                      overwrite: bool = True,
                      verbose: bool = True) -> List[str]:
    """
    Save an MNE Raw to a FIF file path (BIDSPath, dict with 'fpath', Path, or str).
    Returns a list of all written split files (absolute paths), covering both split styles.
    """
    # Resolve destination filename
    if hasattr(out_path, "fpath"):  # BIDSPath
        fif_file = str(out_path.fpath)
    elif isinstance(out_path, dict) and "fpath" in out_path:
        fif_file = str(out_path["fpath"])
    else:
        fif_file = str(out_path)
    fif_path = Path(fif_file).expanduser().resolve()

    # Ensure parent exists
    fif_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[bids_io_utils] Writing: {fif_path}")

    # Save; MNE will split if needed based on file size
    raw.save(str(fif_path), overwrite=overwrite)

    # Enumerate all split parts robustly
    files = list_all_split_files(fif_path)
    if verbose:
        print(f"[bids_io_utils] All files written ({len(files)}):")
        for f in files:
            print(f"  {f}")
    return [str(f) for f in files]

# ------------------------------ Environment utils -----------------------------

def detect_environment(hpc_hostname_tag: str = "milgram") -> str:
    """
    Detect execution environment: returns 'hpc', 'darwin', 'linux', 'windows', or 'unknown'.
    """
    sysname = os.uname().sysname.lower() if hasattr(os, "uname") else ""
    hostname = socket.gethostname().lower()
    if "darwin" in sysname:
        return "darwin"
    if "windows" in sysname:
        return "windows"
    if hpc_hostname_tag.lower() in hostname:
        return "hpc"
    if "linux" in sysname:
        return "linux"
    return "unknown"

# ------------------------------ Convenience paths -----------------------------

def get_bids_headpos_path(subject: str,
                          session: Optional[str],
                          task: Optional[str],
                          run: Optional[str],
                          meg_dir: Union[str, Path]) -> str:
    """
    Return the expected BIDS .pos path for sub/ses/task/run (run optional).
    """
    run = norm_run(run)
    fname = build_bids_stem(subject, session, task, run) + "_headpos.pos"
    return str(Path(meg_dir) / fname)

# ------------------------------ Rsync helpers ---------------------------------

def fetch_bids_data_and_sidecars(hpc_host: str, hpc_user: str,
                                 remote_meg_dir: str,
                                 base_stem: str,
                                 local_meg_dir: str,
                                 verbose: bool = True) -> List[str]:
    """
    Mirror a remote MEG dir (all splits + sidecars for a given base stem) into local_meg_dir via one rsync.
    Returns a list of fetched files (absolute paths).
    """
    Path(local_meg_dir).mkdir(parents=True, exist_ok=True)
    remote_pattern = f"{remote_meg_dir}/{base_stem}_*.*"
    remote_spec = f"{hpc_user}@{hpc_host}:{remote_pattern}"
    if verbose:
        print(f"[bids_io_utils] Rsync fetch: {remote_spec} -> {local_meg_dir}")
    try:
        subprocess.run(["rsync", "-avz", "--update", remote_spec, local_meg_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[bids_io_utils][warning] rsync failed: {e}")

    fetched = glob(os.path.join(local_meg_dir, f"{base_stem}*.*"))
    # Deduplicate, preserve order
    seen = set()
    out = []
    for f in fetched:
        if f not in seen:
            seen.add(f)
            out.append(os.path.abspath(f))
    return out

def push_bids_derivatives_rsync(local_bids_root: Union[str, Path],
                                remote_bids_root: Union[str, Path],
                                hpc_host: str,
                                hpc_user: str,
                                verbose: bool = True) -> None:
    """
    Rsync everything under <local>/derivatives to <remote>/derivatives in one command.
    """
    local_deriv = Path(local_bids_root).expanduser().resolve() / "derivatives"
    remote_deriv_dir = Path(remote_bids_root) / "derivatives"
    remote_spec = f"{hpc_user}@{hpc_host}:{remote_deriv_dir}"
    if verbose:
        print(f"[bids_io_utils] Rsync upload: {local_deriv}/ -> {remote_spec}/")
    subprocess.run([
        "rsync", "-avz", "--update",
        f"{str(local_deriv)}/",  # trailing slash = contents only
        remote_spec
    ], check=True)
    if verbose:
        print("[bids_io_utils] Upload complete.")

# ---------------------------- Backward-compatible API -------------------------

def get_all_bids_split_files(base_fif: Union[str, Path]) -> List[str]:
    """
    Back-compat helper: list all split FIFs given a base file.
    Uses the robust list_all_split_files implementation.
    """
    files = list_all_split_files(base_fif)
    return [str(f) for f in files]

def read_raw_bids_robust(bids_path_or_dict, **kwargs):
    """
    Back-compat shim: previously your code passed a BIDSPath/dict and assumed a single path.
    This now routes through read_raw_bids_smart with optional run handling and split auto-detect.
    """
    # Try to detect a BIDSPath-like object
    if hasattr(bids_path_or_dict, "root") or isinstance(bids_path_or_dict, dict):
        raw, info = read_raw_bids_smart(bids_path_or_dict, **kwargs)
        return raw
    # Otherwise assume it was a direct FIF file path
    fif = Path(bids_path_or_dict).expanduser().resolve()
    if not fif.exists():
        raise FileNotFoundError(f"Base FIF not found: {fif}")
    if "preload" not in kwargs:
        kwargs["preload"] = True
    return mne.io.read_raw_fif(str(fif), **kwargs)
