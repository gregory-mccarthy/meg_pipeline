#!/usr/bin/env python3
"""
Compact Hybrid Data Transfer Manager
Minimizes TFA authentications by batching all transfers
"""

import os
import subprocess
import shlex
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("transfer")


class HybridTransferManager:
    """
    Handles all data transfers between HPC and local temp with minimal authentications.
    Strategy: Always fetch both raw AND checkpoint files in single transfer.
    """

    def __init__(self, hpc_host: str, hpc_user: str, local_temp_dir: str):
        self.hpc_host = hpc_host
        self.hpc_user = hpc_user
        self.local_temp_dir = Path(local_temp_dir)
        self.remote_prefix = f"{hpc_user}@{hpc_host}"

    def fetch_all_bids_data(self, subject: str, session: Optional[str],
                            task: Optional[str], run: Optional[str],
                            remote_bids_root: str) -> Tuple[str, bool]:
        """
        Fetch ALL potentially needed files in single rsync:
        - Raw data + sidecars
        - Checkpoint files (if they exist)

        Returns:
            (local_bids_root, checkpoint_exists)
        """

        # Build BIDS path components
        rel_dir = f"sub-{subject}"
        if session:
            rel_dir += f"/ses-{session}"

        base_name = f"sub-{subject}"
        if session: base_name += f"_ses-{session}"
        if task:    base_name += f"_task-{task}"
        if run:     base_name += f"_run-{run}"

        # Setup local directory structure
        local_bids_root = self.local_temp_dir / "bids"
        local_meg_dir = local_bids_root / rel_dir / "meg"
        local_deriv_dir = local_bids_root / "derivatives" / "preprocessing" / rel_dir / "meg"

        for dir_path in [local_meg_dir, local_deriv_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Build include patterns for single rsync command
        # Need to include directory structure for rsync traversal
        includes = []

        # 1. Include directory structure for raw data
        includes.extend([
            f"sub-{subject}/",
            f"sub-{subject}/ses-{session}/" if session else f"sub-{subject}/",
            f"sub-{subject}/ses-{session}/meg/" if session else f"sub-{subject}/meg/",
        ])

        # 2. Include directory structure for derivatives
        includes.extend([
            "derivatives/",
            "derivatives/preprocessing/",
            f"derivatives/preprocessing/sub-{subject}/",
            f"derivatives/preprocessing/sub-{subject}/ses-{session}/" if session else f"derivatives/preprocessing/sub-{subject}/",
            f"derivatives/preprocessing/sub-{subject}/ses-{session}/meg/" if session else f"derivatives/preprocessing/sub-{subject}/meg/",
        ])

        # 3. Raw data file patterns
        includes.extend([
            f"sub-{subject}/ses-{session}/meg/{base_name}_meg.fif*" if session else f"sub-{subject}/meg/{base_name}_meg.fif*",
            f"sub-{subject}/ses-{session}/meg/{base_name}_meg.json" if session else f"sub-{subject}/meg/{base_name}_meg.json",
            f"sub-{subject}/ses-{session}/meg/{base_name}_headpos.pos" if session else f"sub-{subject}/meg/{base_name}_headpos.pos",
            f"sub-{subject}/ses-{session}/meg/{base_name}_channels.tsv" if session else f"sub-{subject}/meg/{base_name}_channels.tsv",
            f"sub-{subject}/ses-{session}/meg/{base_name}_events.tsv" if session else f"sub-{subject}/meg/{base_name}_events.tsv",
        ])

        # 4. Checkpoint file patterns
        deriv_path = f"derivatives/preprocessing/sub-{subject}/ses-{session}/meg/" if session else f"derivatives/preprocessing/sub-{subject}/meg/"
        includes.extend([
            f"{deriv_path}{base_name}_desc-parproc_meg.fif*",
            f"{deriv_path}{base_name}_desc-parproc_meg_manifest.yaml",
        ])

        # Execute single rsync with all patterns
        remote_root = f"{self.remote_prefix}:{remote_bids_root}/"
        local_root = str(local_bids_root) + "/"

        exit_code = self._rsync_with_includes(remote_root, local_root, includes)

        if exit_code != 0:
            logger.warning(f"Rsync completed with exit code {exit_code}")

        # Check if checkpoint files were actually transferred
        checkpoint_fif = local_deriv_dir / f"{base_name}_desc-parproc_meg.fif"
        checkpoint_manifest = local_deriv_dir / f"{base_name}_desc-parproc_meg_manifest.yaml"
        checkpoint_exists = checkpoint_fif.exists() and checkpoint_manifest.exists()

        logger.info(f"Fetched data to {local_bids_root}")
        logger.info(f"Checkpoint available: {checkpoint_exists}")

        return str(local_bids_root), checkpoint_exists

    def push_results(self, local_bids_root: str, remote_bids_root: str,
                     subject: str, session: Optional[str]) -> int:
        """
        Push ALL changes back to HPC in single transfer:
        - New derivatives (preprocessing results)
        - New sidecar files (headpos.pos, etc.) in raw data directories
        - Any other files created during processing

        Uses complex rsync with includes to sync entire BIDS structure efficiently.
        """
        local_bids_path = Path(local_bids_root)
        if not local_bids_path.exists():
            logger.warning(f"Local BIDS root doesn't exist: {local_bids_root}")
            return 1

        # Build directory patterns for this subject
        rel_dir = f"sub-{subject}"
        if session:
            rel_dir += f"/ses-{session}"

        # FIXED: Build includes to only sync THIS subject's data, even if temp/BIDS has other subjects
        includes = []

        # 1. Subject-specific directory structure only
        includes.extend([
            f"sub-{subject}/",
            f"sub-{subject}/ses-{session}/" if session else "",
            f"sub-{subject}/ses-{session}/meg/" if session else f"sub-{subject}/meg/",
        ])

        # 2. Subject-specific derivatives structure only
        includes.extend([
            "derivatives/",
            "derivatives/preprocessing/",
            f"derivatives/preprocessing/sub-{subject}/",
            f"derivatives/preprocessing/sub-{subject}/ses-{session}/" if session else "",
            f"derivatives/preprocessing/sub-{subject}/ses-{session}/meg/" if session else f"derivatives/preprocessing/sub-{subject}/meg/",
        ])

        # 3. ONLY this subject's raw data files (not other subjects)
        raw_pattern = f"sub-{subject}/ses-{session}/meg/**" if session else f"sub-{subject}/meg/**"
        includes.append(raw_pattern)

        # 4. ONLY this subject's derivatives (not other subjects)
        deriv_pattern = f"derivatives/preprocessing/sub-{subject}/**"
        includes.append(deriv_pattern)

        # Remove empty strings from includes
        includes = [inc for inc in includes if inc]

        # Create remote BIDS root and execute rsync with includes
        remote_target = f"{self.remote_prefix}:{remote_bids_root}/"

        cmd = [
            "rsync", "-avz", "--progress", "--partial",
            "--rsync-path", f"mkdir -p {remote_bids_root} && rsync"
        ]

        # Add all include patterns
        for pattern in includes:
            cmd.extend(["--include", pattern])

        # Exclude everything else
        cmd.extend(["--exclude", "*"])

        # Add source and destination
        cmd.extend([str(local_bids_path) + "/", remote_target])

        logger.info(f"Pushing all BIDS changes: {' '.join(shlex.quote(arg) for arg in cmd)}")
        return subprocess.call(cmd)

    def _rsync_with_includes(self, src: str, dst: str, includes: List[str]) -> int:
        """
        Execute rsync with multiple include patterns.
        """
        cmd = ["rsync", "-avz", "--progress", "--partial"]

        # Add all include patterns
        for pattern in includes:
            cmd.extend(["--include", pattern])

        # Exclude everything else
        cmd.extend(["--exclude", "*"])

        # Add source and destination
        cmd.extend([src, dst])

        logger.info(f"Transfer command: {' '.join(shlex.quote(arg) for arg in cmd)}")
        return subprocess.call(cmd)


def create_transfer_manager_from_config(config: dict) -> Optional[HybridTransferManager]:
    """
    Factory function to create transfer manager from pipeline config.
    """
    hpc_host = config.get("hpc_host")
    hpc_user = config.get("hpc_user")
    temp_dir = config.get("temp_dir", "./temp")

    if not hpc_host or not hpc_user:
        return None

    return HybridTransferManager(hpc_host, hpc_user, temp_dir)


# Integration example for your pipeline
def replace_fetch_bids_data_and_sidecars(hpc_host: str, hpc_user: str,
                                         remote_meg_dir: str, base_stem: str,
                                         local_meg_dir: str):
    """
    Drop-in replacement for your existing fetch function.
    """
    # Extract BIDS components from base_stem
    parts = base_stem.split('_')
    subject = parts[0].replace('sub-', '')
    session = task = run = None

    for part in parts[1:]:
        if part.startswith('ses-'):
            session = part.replace('ses-', '')
        elif part.startswith('task-'):
            task = part.replace('task-', '')
        elif part.startswith('run-'):
            run = part.replace('run-', '')

    # Derive remote_bids_root from remote_meg_dir
    # remote_meg_dir format: /path/to/bids/sub-X/ses-Y/meg
    remote_bids_root = str(Path(remote_meg_dir).parents[2])
    temp_dir = str(Path(local_meg_dir).parents[2])

    # Use new transfer manager
    transfer_mgr = HybridTransferManager(hpc_host, hpc_user, temp_dir)
    local_bids_root, checkpoint_exists = transfer_mgr.fetch_all_bids_data(
        subject, session, task, run, remote_bids_root
    )

    return local_bids_root, checkpoint_exists
