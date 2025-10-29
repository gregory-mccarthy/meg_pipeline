#!/usr/bin/env python3
"""
transfer_manager_v3.py

HybridTransferManager (v3): class-based API that keeps the original v1 surface
while incorporating v2 improvements and recent fixes:

- Single-rsync philosophy to minimize Duo/TFA prompts.
- Robust include patterns for both BIDS split files (*_split-*_meg.fif)
  and legacy MEGIN parts (*_meg.fif, *_meg-*.fif).
- Optional SSH multiplexing (ControlMaster) to reuse one authenticated session.
- Functional parity with prior versions:
    - fetch_all_bids_data(...) -> (local_bids_root:str, checkpoint_exists:bool)
    - push_results(...): push derivatives (and optionally raw sidecar updates)
- Preflight probe via remote_meg_exists(...)
- Dry-run and verbose flags for safety and logging.

Fixes relative to earlier drafts:
- Uses legacy-safe rsync flags (--stats --progress -v) instead of --info=...
- SIDECARES ARE TASK/RUN-SCOPED so we do not pull other tasks.
- Correct filename stem (sub-XXX[_ses-YYY] with underscores, not slashes).
- Defensive guards to avoid None / empty include lists.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def _expand(p: os.PathLike | str) -> Path:
    return Path(p).expanduser().resolve()


@dataclass
class SSHOptions:
    host: str
    user: str
    use_multiplex: bool = True
    control_dir: Optional[Path] = None
    port: Optional[int] = None  # allow non-standard ports if needed

    def control_path(self) -> Optional[Path]:
        if not self.use_multiplex:
            return None
        base = self.control_dir or _expand("~/.ssh")
        base.mkdir(parents=True, exist_ok=True)
        # OpenSSH requires a literal path; %r,%h,%p get expanded by ssh itself.
        return base / "cm-%r@%h:%p"

    def ssh_base_args(self) -> List[str]:
        args = ["ssh"]
        if self.port:
            args += ["-p", str(self.port)]
        if self.use_multiplex:
            cp = str(self.control_path())
            args += [
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={cp}",
                "-o", "ControlPersist=600",
            ]
        return args

    def rsync_ssh(self) -> str:
        # Build the -e "ssh ..." payload for rsync
        return " ".join(shlex.quote(x) for x in self.ssh_base_args())

    def open_master(self) -> None:
        if not self.use_multiplex:
            return
        args = self.ssh_base_args() + ["-MNf", f"{self.user}@{self.host}"]
        # Ignore errors if already open
        try:
            subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass

    def close_master(self) -> None:
        if not self.use_multiplex:
            return
        args = self.ssh_base_args() + ["-O", "exit", f"{self.user}@{self.host}"]
        subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class HybridTransferManager:
    """
    Class wrapper matching the original v1 surface while using improved v2 logic.

    Typical use:
        tm = HybridTransferManager(hpc_host, hpc_user, local_temp_dir)
        local_root, ckpt = tm.fetch_all_bids_data(sub, ses, task, run, remote_bids_root)
        rc = tm.push_results(local_root, remote_bids_root, sub, ses, push_raw_sidecars=False)
    """

    def __init__(
        self,
        host: str,
        user: str,
        local_temp_dir: str | os.PathLike,
        *,
        use_multiplex: bool = True,
        control_dir: Optional[str | os.PathLike] = None,
        dry_run: bool = False,
        verbose: bool = True,
        delete: bool = False,
        fetch_derivatives: bool = False,
    ) -> None:
        self.ssh = SSHOptions(
            host=host,
            user=user,
            use_multiplex=use_multiplex,
            control_dir=_expand(control_dir) if control_dir else None,
        )
        self.local_root = _expand(local_temp_dir)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.verbose = verbose
        self.delete = delete
        self.fetch_derivatives = fetch_derivatives

    # ------------------------- Public API -------------------------

    def fetch_all_bids_data(
            self,
            subject: str,
            session: Optional[str],
            task: Optional[str],
            run: Optional[str],
            remote_bids_root: str | os.PathLike,
    ) -> Tuple[str, bool]:
        """
        Robust fetch: rsync directly from the remote MEG directory into the local MEG directory,
        including only the run stem and the (session) coordsystem file.
        """
        self.ssh.open_master()
        remote_root = _expand(remote_bids_root)

        # Directory vs filename stems (slash vs underscore)
        base_dir = f"sub-{subject}" + (f"/ses-{session}" if session and str(session).strip() else "")
        base_file = f"sub-{subject}" + (f"_ses-{session}" if session and str(session).strip() else "")
        remote_meg_dir = remote_root / base_dir / "meg"
        local_meg_dir = self.local_root / base_dir / "meg"
        local_meg_dir.mkdir(parents=True, exist_ok=True)

        if not task or not str(task).strip():
            raise ValueError("Explicit task required, e.g., task='rest'.")
        if not run or not str(run).strip():
            raise ValueError("Explicit run required, e.g., run='01'.")

        stem = f"{base_file}_task-{task}_run-{run}"

        # Minimal include set at the MEG-dir root
        includes: List[str] = [
            f"{stem}*",  # everything for this run (FIF, parts, splits, sidecars)
            f"{base_file}_coordsystem.json",  # session coordsystem (if present)
        ]

        # Build rsync command: from the MEG dir to the local MEG dir
        args = [
            "rsync", "-a", "--partial", "--prune-empty-dirs",
            "--stats", "--progress", "-v",
            "-e", self.ssh.rsync_ssh(),
        ]
        for pat in includes:
            args += ["--include", pat]
        args += ["--exclude", "*"]
        args += [
            f"{self.ssh.user}@{self.ssh.host}:{str(remote_meg_dir)}/",
            str(local_meg_dir) + "/",
        ]

        # Always show the exact command we run
        print("\n================ RSYNC COMMAND ====================")
        print(" ".join(shlex.quote(a) for a in args))
        print("===================================================\n")

        cp = subprocess.run(args, check=False)
        if cp.returncode != 0 and not self.dry_run:
            raise RuntimeError(f"Command failed (exit {cp.returncode}): {' '.join(args)}")

        # Verify we actually have the MEG directory and at least one FIF now
        if not local_meg_dir.exists():
            raise FileNotFoundError(f"Fetch finished but MEG directory is missing locally: {local_meg_dir}")
        fif_matches = list(local_meg_dir.glob(f"{stem}_meg*.fif")) + list(
            local_meg_dir.glob(f"{stem}_split-*_meg*.fif"))
        if not fif_matches:
            # list what we *do* have to aid debugging
            present = "\n".join(f"  - {p.name}" for p in sorted(local_meg_dir.glob("*")))
            raise FileNotFoundError(
                f"No FIF files copied for stem '{stem}' into {local_meg_dir}\n"
                f"Present files:\n{present if present else '  (none)'}"
            )

        ckpt = self._detect_checkpoint(self.local_root, subject, session)
        return (str(self.local_root), ckpt)

    def push_results(
        self,
        local_bids_root: str | os.PathLike,
        remote_bids_root: str | os.PathLike,
        subject: str,
        session: Optional[str],
        *,
        push_raw_sidecars: bool = False,
    ) -> int:
        """
        Push derivatives for a subject[/session] back to remote using a single rsync call.
        If push_raw_sidecars=True, also include commonly edited raw sidecars.
        Returns 0 on success.
        """
        self.ssh.open_master()
        local_root = _expand(local_bids_root)
        remote_root = _expand(remote_bids_root)

        includes = self._build_derivatives_includes(subject, session)
        if push_raw_sidecars:
            includes += self._build_includes(subject, session, task=None, run=None, raw_sidecars=True, fif_files=False)

        if not includes:
            raise ValueError("No include patterns were generated for push; this should not happen.")

        self._rsync_push(local_root, remote_root, includes)
        return 0

    def remote_meg_exists(
        self,
        remote_bids_root: str | os.PathLike,
        subject: str,
        session: Optional[str] = None,
    ) -> bool:
        """
        Lightweight probe to check if sub[/ses] MEG files exist remotely.
        Uses rsync --list-only with include patterns.
        """
        remote_root = _expand(remote_bids_root)
        incl = self._build_includes(subject, session, task=None, run=None, raw_sidecars=False, fif_files=True)
        return self._rsync_probe(remote_root, incl)

    # ------------------------- Internal helpers -------------------------

    def _rsync_common_args(self) -> List[str]:
        args = [
            "rsync",
            "-a",                    # archive preserves perms/times/links
            "--partial",
            "--prune-empty-dirs",
        ]
        if self.verbose:
            # Compatible with older rsync (no --info=stats2,progress2)
            args += ["--stats", "--progress", "-v"]
        if self.dry_run:
            args += ["--dry-run"]
        if self.delete:
            args += ["--delete"]
        return args

    def _rsync_pull(self, remote_root: Path, local_root: Path, includes: List[str]) -> None:
        if not includes:
            raise ValueError("rsync pull called with empty or None include set.")
        args = self._rsync_common_args()
        # Exclude everything by default, then include our whitelist
        for pat in includes:
            args += ["--include", pat]
        args += ["--exclude", "*"]

        remote = f"{self.ssh.user}@{self.ssh.host}:{str(remote_root)}/"
        args += ["-e", self.ssh.rsync_ssh()]
        args += [remote, str(local_root) + "/"]

        self._run(args)

    def _rsync_push(self, local_root: Path, remote_root: Path, includes: List[str]) -> None:
        if not includes:
            raise ValueError("rsync push called with empty or None include set.")
        args = self._rsync_common_args()
        for pat in includes:
            args += ["--include", pat]
        args += ["--exclude", "*"]

        src = str(local_root) + "/"
        dest = f"{self.ssh.user}@{self.ssh.host}:{str(remote_root)}/"

        args += ["-e", self.ssh.rsync_ssh()]
        args += [src, dest]

        self._run(args)

    def _rsync_probe(self, remote_root: Path, includes: List[str]) -> bool:
        # Use --list-only to avoid copying anything.
        args = ["rsync", "-a", "--list-only"]
        for pat in includes:
            args += ["--include", pat]
        args += ["--exclude", "*"]
        remote = f"{self.ssh.user}@{self.ssh.host}:{str(remote_root)}/"
        args += ["-e", self.ssh.rsync_ssh(), remote, "/dev/null"]  # dummy sink

        try:
            cp = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # If any file is listed, stdout will have lines.
            return bool(cp.stdout.strip())
        except Exception:
            return False

    def _run(self, args: List[str]) -> None:
        # Always print the full command being executed
        print("\n================ RSYNC COMMAND ====================")
        print(" ".join(shlex.quote(a) for a in args))
        print("===================================================\n")

        # actually execute it
        cp = subprocess.run(args, check=False)
        if cp.returncode != 0 and not self.dry_run:
            raise RuntimeError(f"Command failed (exit {cp.returncode}): {' '.join(args)}")

    # ------------------------- Pattern builders -------------------------

    @staticmethod
    def _entity_prefix_dir(subject: str, session: Optional[str]) -> str:
        """Directory prefix: 'sub-XXX[/ses-YYY]' (with slash)."""
        if session and str(session).strip():
            return f"sub-{subject}/ses-{session}"
        return f"sub-{subject}"

    @staticmethod
    def _entity_prefix_file(subject: str, session: Optional[str]) -> str:
        """Filename prefix: 'sub-XXX[_ses-YYY]' (with underscore)."""
        if session and str(session).strip():
            return f"sub-{subject}_ses-{session}"
        return f"sub-{subject}"

    def _build_includes(
        self,
        subject: str,
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
        *,
        raw_sidecars: bool,
        fif_files: bool = True,
    ) -> List[str]:
        """
        Build include patterns for raw MEG (and optional sidecars).

        Supports legacy FIF parts and BIDS split style, and constrains sidecars
        to the requested task/run so we don't pull other tasks.
        """
        base_dir = self._entity_prefix_dir(subject, session)    # e.g., 'sub-011/ses-01'
        base_fname = self._entity_prefix_file(subject, session) # e.g., 'sub-011_ses-01'
        meg_dir = f"{base_dir}/meg"

        patterns: List[str] = [
            base_dir,
            f"{base_dir}/",
            meg_dir,
            f"{meg_dir}/",
        ]

        # FIF patterns (supporting both legacy and BIDS split style)
        if fif_files:
            task_pat = f"task-{task}" if (task and str(task).strip()) else "task-*"
            run_pat  = f"run-{run}"   if (run  and str(run).strip())  else "run-*"
            stem = f"{base_fname}_{task_pat}_{run_pat}"
            patterns += [
                f"{meg_dir}/{stem}_meg.fif",
                f"{meg_dir}/{stem}_meg-*.fif",
                f"{meg_dir}/{stem}_split-*_meg.fif",
            ]

        # Sidecars commonly needed for MEG processing (TASK/RUN scoped)
        if raw_sidecars:
            task_pat = f"task-{task}" if (task and str(task).strip()) else "task-*"
            run_pat  = f"run-{run}"   if (run  and str(run).strip())  else "run-*"
            stem = f"{base_fname}_{task_pat}_{run_pat}"

            # Per-recording sidecars
            patterns += [
                f"{meg_dir}/{stem}_channels.tsv",
                f"{meg_dir}/{stem}_events.tsv",
                f"{meg_dir}/{stem}_meg.json",
                f"{meg_dir}/{stem}_headshape.*",
                f"{meg_dir}/{stem}_headpos.*",
                f"{meg_dir}/{stem}*.cal",
                f"{meg_dir}/{stem}*.dat",
            ]

            # Session-level coordsystem, narrowly scoped to THIS session
            patterns += [f"{meg_dir}/{base_fname}_coordsystem.json"]
            if session and str(session).strip():
                patterns += [f"{meg_dir}/{base_fname}_ses-{session}_coordsystem.json"]

        return patterns

    def _build_derivatives_includes(self, subject: str, session: Optional[str]) -> List[str]:
        base_dir = self._entity_prefix_dir(subject, session)    # 'sub-XXX[/ses-YYY]'
        deriv = "derivatives"
        # Allow traversal into derivatives tree, but limit to sub[/ses] subtree
        return [
            deriv,
            f"{deriv}/",
            f"{deriv}/**/",
            f"{deriv}/**/{base_dir}/",
            f"{deriv}/**/{base_dir}/**",
        ]

    # ------------------------- Checkpoint heuristic -------------------------

    def _detect_checkpoint(self, local_root: Path, subject: str, session: Optional[str]) -> bool:
        """
        Heuristic: look for common checkpoint markers under derivatives for this subject/session.
        Adapt this to your pipeline's real checkpoint convention if needed.
        """
        base_dir = self._entity_prefix_dir(subject, session)
        deriv = local_root / "derivatives"
        if not deriv.exists():
            return False

        # Common extensions or marker names (customize to your convention)
        patterns = [
            f"**/{base_dir}/**/*.ckpt",
            f"**/{base_dir}/**/*.checkpoint",
            f"**/{base_dir}/**/.checkpoint",
            f"**/{base_dir}/**/CHECKPOINT*",
            f"**/{base_dir}/**/*.stage-*",
        ]
        try:
            for pat in patterns:
                if any(deriv.glob(pat)):
                    return True
        except Exception:
            pass
        return False
