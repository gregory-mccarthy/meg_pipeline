#!/usr/bin/env python3
"""
transfer_manager.py

High-level, TFA-aware transfer utilities for BIDS MEG datasets.

Goals:
- Minimize SSH/TFA prompts by performing large, batched rsync operations
  (ideally ONE rsync per fetch/push).
- Support both BIDS split style (..._split-01_meg.fif, _split-02_...) and
  legacy MEGIN style (..._meg.fif, ..._meg-1.fif, ..._meg-2.fif).
- Support optional run entity (run can be None / null / '' in YAML).
- Zero shell globbing on the remote host beyond rsync itself (no extra ssh).
- Optional SSH multiplexing (ControlMaster) to reuse an authenticated session.

Typical use:
    from transfer_manager import (
        fetch_meg_run, fetch_meg_subject_session, push_derivatives
    )

    # Fetch a single run in 1 rsync:
    fetch_meg_run(
        remote_root="/gpfs/milgram/scratch/mccarthy/gm33/BIDS/epi",
        local_root="~/BIDS/epi",
        subject="001", session="01", task="fairy", run=None,   # run: None → no run entity
        host="hpc.yale.edu", user="gm33",
        use_multiplex=True, verbose=True
    )

    # Fetch all runs for sub+ses in 1 rsync:
    fetch_meg_subject_session(
        remote_root="/gpfs/milgram/scratch/mccarthy/gm33/BIDS/epi",
        local_root="~/BIDS/epi",
        subject="001", session="01", task=None,
        host="hpc.yale.edu", user="gm33",
        use_multiplex=True, verbose=True
    )

    # Push entire derivatives tree in 1 rsync:
    push_derivatives(
        local_root="~/BIDS/epi",
        remote_root="/gpfs/milgram/scratch/mccarthy/gm33/BIDS/epi",
        host="hpc.yale.edu", user="gm33",
        use_multiplex=True, verbose=True
    )
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------------------
# Small, local helpers (no external deps). We intentionally do not import bids_io_utils
# to keep this module standalone for transfers.
# --------------------------------------------------------------------------------------

def _none_like(x) -> bool:
    return x in (None, "", False, "None", "null", "NULL")

def _norm_run(val) -> Optional[str]:
    """Normalize run to a zero-padded 2-digit string or None."""
    if _none_like(val):
        return None
    s = str(val)
    return s.zfill(2) if s.isdigit() else s

def _ent(stem: str, key: str, val: Optional[str]) -> str:
    return f"{stem}_{key}-{val}" if not _none_like(val) else stem

def build_bids_stem(subject: str,
                    session: Optional[str] = None,
                    task: Optional[str] = None,
                    run: Optional[str] = None) -> str:
    """sub-001[_ses-01][_task-fairy][_run-02]"""
    run = _norm_run(run)
    stem = f"sub-{subject}"
    stem = _ent(stem, "ses", session)
    stem = _ent(stem, "task", task)
    stem = _ent(stem, "run", run)
    return stem

def meg_dir(root: Union[str, Path], subject: str, session: Optional[str] = None) -> Path:
    """<root>/sub-<subject>/[ses-<session>/]meg"""
    root = Path(root).expanduser().resolve()
    parts = [root, f"sub-{subject}"]
    if not _none_like(session):
        parts.append(f"ses-{session}")
    parts.append("meg")
    return Path(*parts)

# --------------------------------------------------------------------------------------
# SSH / rsync plumbing with optional ControlMaster (SSH multiplexing).
# --------------------------------------------------------------------------------------

def _ssh_cmd_base(use_multiplex: bool) -> List[str]:
    """
    Return an SSH command prefix for rsync's -e option.
    We return a list suitable for: rsync ... -e 'ssh ...'
    """
    base = ["ssh"]
    if use_multiplex:
        # Reuse an authenticated control connection if permitted by IT policy.
        # First rsync will do the TFA; subsequent rsyncs reuse the control socket.
        base += [
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=300s",
            "-o", "ControlPath=~/.ssh/cm-%r@%h:%p"
        ]
    return base

def _rsync_common_args(verbose: bool, preserve_times: bool = True) -> List[str]:
    # -a: archive; -z: compress; --update: skip newer on receiver
    args = ["rsync", "-avz", "--update"]
    if preserve_times:
        args += ["--times"]
    if verbose:
        args += ["--progress"]
    return args

def _run_subprocess(cmd: Sequence[str], dry_run: bool = False, verbose: bool = True) -> None:
    if verbose:
        print("[transfer_manager] Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)

# --------------------------------------------------------------------------------------
# INCLUDE/EXCLUDE patterns to pull exactly what we need in ONE rsync.
# --------------------------------------------------------------------------------------

# Core sidecars expected in /meg for each run
_MEG_SIDECARS = [
    "_meg.json",
    "_channels.tsv",
    "_coordsystem.json",
    "_events.tsv",         # optional but common
    "_scans.tsv",          # often up at session level, but include in case it’s placed here
    "_headpos.pos",        # your center convention
    "_headshape.*",        # if present
]

def _include_patterns_for_run(stem: str) -> List[str]:
    """
    Build rsync --include patterns for a single run (or single-run w/o run entity).
    We include both split styles and common sidecars.
    """
    pats: List[str] = []

    # FIF data — both split styles
    pats.append(f"{stem}_split-*_meg.fif")  # BIDS split
    pats.append(f"{stem}_meg.fif")          # single or base legacy
    pats.append(f"{stem}_meg-*.fif")        # legacy numbered parts

    # Sidecars in MEG folder for this stem
    for suffix in _MEG_SIDECARS:
        pats.append(f"{stem}{suffix}")

    return pats

def _include_patterns_for_subject_session(subject: str,
                                          session: Optional[str],
                                          task: Optional[str]) -> List[str]:
    """
    Build patterns that match ALL runs (and also the single-run case)
    for a given subject/session[/task], in one rsync.
    """
    pats: List[str] = []

    # Base stems:
    stem_norun = build_bids_stem(subject, session, task, run=None)        # no run entity
    stem_anyrun = f"{stem_norun}_run-*"                                   # any run

    # For "no run" (single-run datasets)
    pats += _include_patterns_for_run(stem_norun)

    # For "any run"
    pats += _include_patterns_for_run(stem_anyrun)

    return pats

# --------------------------------------------------------------------------------------
# Fetch functions (ONE rsync each)
# --------------------------------------------------------------------------------------

def fetch_meg_run(remote_root: Union[str, Path],
                  local_root: Union[str, Path],
                  subject: str,
                  session: Optional[str],
                  task: Optional[str],
                  run: Optional[str],
                  host: str,
                  user: str,
                  use_multiplex: bool = True,
                  delete_extraneous: bool = False,
                  dry_run: bool = False,
                  verbose: bool = True) -> Path:
    """
    Fetch exactly one run (or a single-run dataset with run=None) in ONE rsync call.
    Downloads both FIF data and sidecars into local BIDS tree.

    Returns the local meg dir path.
    """
    run = _norm_run(run)
    remote_meg = meg_dir(remote_root, subject, session)
    local_meg = meg_dir(local_root, subject, session)
    local_meg.mkdir(parents=True, exist_ok=True)

    stem = build_bids_stem(subject, session, task, run)
    include = _include_patterns_for_run(stem)

    # Build rsync command
    cmd = _rsync_common_args(verbose=verbose)
    if delete_extraneous:
        cmd += ["--delete"]

    # Include patterns first, then a terminal exclude-all to limit transfer
    for pat in include:
        cmd += ["--include", pat]
    cmd += ["--exclude", "*"]

    # SSH transport (single connection)
    ssh_cmd = _ssh_cmd_base(use_multiplex=use_multiplex)
    cmd += ["-e", " ".join(ssh_cmd)]

    # Source and destination
    src = f"{user}@{host}:{str(remote_meg)}/"   # trailing slash = dir contents
    dst = f"{str(local_meg)}/"
    cmd += [src, dst]

    _run_subprocess(cmd, dry_run=dry_run, verbose=verbose)
    return local_meg

def fetch_meg_subject_session(remote_root: Union[str, Path],
                              local_root: Union[str, Path],
                              subject: str,
                              session: Optional[str],
                              task: Optional[str],
                              host: str,
                              user: str,
                              use_multiplex: bool = True,
                              delete_extraneous: bool = False,
                              dry_run: bool = False,
                              verbose: bool = True) -> Path:
    """
    Fetch ALL runs (and the single-run case) for subject/session[/task] in ONE rsync.
    If task is None, we fetch all tasks for this sub/ses.

    Returns the local meg dir path.
    """
    remote_meg = meg_dir(remote_root, subject, session)
    local_meg = meg_dir(local_root, subject, session)
    local_meg.mkdir(parents=True, exist_ok=True)

    # If task is given, limit to that task; otherwise include all tasks for sub/ses.
    if _none_like(task):
        # All tasks: pattern uses 'task-*' stems
        stem_task_any = build_bids_stem(subject, session, task="*")
        include = _include_patterns_for_subject_session(subject, session, task="*")
        # We also include subjects' session-level scans file if it happens to live here
        include += [f"{stem_task_any}_scans.tsv"]
    else:
        include = _include_patterns_for_subject_session(subject, session, task)

    cmd = _rsync_common_args(verbose=verbose)
    if delete_extraneous:
        cmd += ["--delete"]

    for pat in include:
        cmd += ["--include", pat]
    cmd += ["--exclude", "*"]

    ssh_cmd = _ssh_cmd_base(use_multiplex=use_multiplex)
    cmd += ["-e", " ".join(ssh_cmd)]

    src = f"{user}@{host}:{str(remote_meg)}/"
    dst = f"{str(local_meg)}/"
    cmd += [src, dst]

    _run_subprocess(cmd, dry_run=dry_run, verbose=verbose)
    return local_meg

# --------------------------------------------------------------------------------------
# Push derivatives (ONE rsync)
# --------------------------------------------------------------------------------------

def push_derivatives(local_root: Union[str, Path],
                     remote_root: Union[str, Path],
                     host: str,
                     user: str,
                     use_multiplex: bool = True,
                     delete_extraneous: bool = False,
                     dry_run: bool = False,
                     verbose: bool = True) -> Path:
    """
    Push the entire derivatives tree from local_root → remote_root in ONE rsync.
    """
    local_deriv = Path(local_root).expanduser().resolve() / "derivatives"
    remote_deriv = Path(remote_root).expanduser().resolve() / "derivatives"

    if verbose:
        print(f"[transfer_manager] Pushing derivatives: {local_deriv} → {user}@{host}:{remote_deriv}")

    cmd = _rsync_common_args(verbose=verbose)
    if delete_extraneous:
        cmd += ["--delete"]

    ssh_cmd = _ssh_cmd_base(use_multiplex=use_multiplex)
    cmd += ["-e", " ".join(ssh_cmd)]

    src = f"{str(local_deriv)}/"                    # send contents only
    dst = f"{user}@{host}:{str(remote_deriv)}"      # dst without trailing slash to create if needed
    cmd += [src, dst]

    _run_subprocess(cmd, dry_run=dry_run, verbose=verbose)
    return remote_deriv

# --------------------------------------------------------------------------------------
# Optional: quick subject/session existence check (NO extra SSH—uses rsync dry-run).
# --------------------------------------------------------------------------------------

def remote_meg_exists(remote_root: Union[str, Path],
                      subject: str,
                      session: Optional[str],
                      host: str,
                      user: str,
                      use_multiplex: bool = True,
                      verbose: bool = True) -> bool:
    """
    Cheap existence probe using rsync --list-only (still one ssh, but avoids full copy).
    If your IT policy prompts TFA for any SSH, call this only if necessary.
    """
    remote_meg_path = meg_dir(remote_root, subject, session)

    cmd = ["rsync", "--list-only", "-e", " ".join(_ssh_cmd_base(use_multiplex))]
    src = f"{user}@{host}:{str(remote_meg_path)}/"
    cmd.append(src)

    try:
        _run_subprocess(cmd, dry_run=False, verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False
