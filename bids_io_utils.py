import os
import subprocess
import socket
from typing import List
import mne
from glob import glob
from pathlib import Path

from glob import glob
import os

def write_bids_robust(raw, out_path, overwrite=True, verbose=True):
    """
    Save MNE Raw to a file (BIDSPath, dict, Path, or str). Returns all written splits.
    """
    import os
    from pathlib import Path

    # Accept BIDSPath, dict, Path, or str
    if hasattr(out_path, "fpath"):  # BIDSPath
        fif_file = str(out_path.fpath)
    elif isinstance(out_path, dict) and "fpath" in out_path:
        fif_file = out_path["fpath"]
    else:
        fif_file = str(out_path)

    # Make sure parent directory exists
    os.makedirs(os.path.dirname(fif_file), exist_ok=True)
    if verbose:
        print(f"[robust_BIDS_write] Writing: {fif_file}")

    # Save the raw object; MNE will split files if needed
    raw.save(fif_file, overwrite=overwrite)

    # Glob for all possible split files matching this derivative stem
    base = Path(fif_file)
    parent = base.parent
    stem = base.stem
    split_files = sorted([str(f) for f in parent.glob(f"{stem}*.fif")])
    if fif_file not in split_files:
        split_files.append(fif_file)
    split_files = list(dict.fromkeys(split_files))

    if verbose:
        print(f"[robust_BIDS_write] All files written: {split_files}")
    return split_files


def read_raw_bids_robust(bids_path, **kwargs):
    """
    Drop-in replacement for mne_bids.read_raw_bids for MEG FIF files.
    Accepts a BIDSPath and loads data with plain MNE, bypassing split file ambiguity.
    """
    if isinstance(bids_path, dict):
        subject = bids_path["subject"]
        session = bids_path.get("session", None)
        task = bids_path.get("task", None)
        run = bids_path.get("run", None)
        datatype = bids_path.get("datatype", "meg")
        root = bids_path["bids_root"]
    else:
        subject = bids_path.subject
        session = bids_path.session
        task = bids_path.task
        run = bids_path.run
        datatype = bids_path.datatype
        root = bids_path.root

    fname = f"sub-{subject}"
    if session:
        fname += f"_ses-{session}"
    if task:
        fname += f"_task-{task}"
    if run:
        fname += f"_run-{run}"
    fname += f"_{datatype}.fif"

    fif_file = os.path.join(
        root, f"sub-{subject}",
        f"ses-{session}" if session else "",
        datatype,
        fname
    )
    fif_file = fif_file.replace("//", "/")

    if not os.path.exists(fif_file):
        raise FileNotFoundError(f"Base MEG FIF file not found: {fif_file}")

    # Only set preload if not already passed by the user
    if 'preload' not in kwargs:
        kwargs['preload'] = True

    raw = mne.io.read_raw_fif(fif_file, **kwargs)
    return raw

def detect_environment(hpc_hostname_tag="milgram"):
    """
    Detect execution environment: returns 'hpc', 'darwin', 'linux', or 'windows'.
    """
    sysname = os.uname().sysname.lower() if hasattr(os, "uname") else ""
    hostname = socket.gethostname().lower()
    if "darwin" in sysname:
        return "darwin"
    elif "windows" in sysname:
        return "windows"
    elif hpc_hostname_tag.lower() in hostname:
        return "hpc"
    elif "linux" in sysname:
        return "linux"
    else:
        return "unknown"

import os

def get_bids_headpos_path(subject, session, task, run, meg_dir):
    """
    Returns the full path to the expected BIDS .pos file for the given subject/session/task/run.
    """
    fname = f"sub-{subject}"
    if session:
        fname += f"_ses-{session}"
    if task:
        fname += f"_task-{task}"
    if run:
        fname += f"_run-{run}"
    fname += "_headpos.pos"
    return os.path.join(meg_dir, fname)

def fetch_bids_data_and_sidecars(hpc_host, hpc_user, remote_meg_dir, base_stem, local_meg_dir, verbose=True):
    """
    Mirrors the cluster BIDS MEG dir (with all splits and sidecars) into local_meg_dir using a single rsync call.
    Only one DUO authentication per fetch.
    Returns a list of all fetched files (absolute paths).
    """
    import os
    import subprocess
    from glob import glob

    os.makedirs(local_meg_dir, exist_ok=True)
    remote_pattern = f"{remote_meg_dir}/{base_stem}_*.*"
    remote_spec = f"{hpc_user}@{hpc_host}:{remote_pattern}"
    if verbose:
        print(f"[bids_io_utils] Rsync fetching (all splits and sidecars): {remote_spec} -> {local_meg_dir}")
    try:
        subprocess.run(
            ["rsync", "-avz", "--update", remote_spec, local_meg_dir],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[bids_io_utils][warning] Rsync failed: {e}")

    # Gather all matching files now present locally (under local_meg_dir)
    fetched_files = glob(os.path.join(local_meg_dir, f"{base_stem}*.*"))
    fetched_files = list(dict.fromkeys(fetched_files))  # Deduplicate, preserve order
    return fetched_files

def push_bids_derivatives_rsync(local_bids_root, remote_bids_root, hpc_host, hpc_user, verbose=True):
    """
    Rsyncs everything under local_bids_root/derivatives to remote_bids_root/derivatives in one command.
    Only one DUO authentication required.
    """
    import subprocess
    import os

    local_deriv_dir = os.path.join(os.path.abspath(local_bids_root), "derivatives")
    remote_deriv_dir = os.path.join(remote_bids_root, "derivatives")
    remote_spec = f"{hpc_user}@{hpc_host}:{remote_deriv_dir}"

    if verbose:
        print(f"[bids_io_utils] Rsync uploading: {local_deriv_dir}/ -> {remote_spec}/")
    subprocess.run([
        "rsync", "-avz", "--update",
        f"{local_deriv_dir}/",  # trailing slash = contents only
        remote_spec
    ], check=True)
    if verbose:
        print("[info] Upload complete.")

def get_all_bids_split_files(base_fif):
    """
    Returns a deduplicated list of all FIF files (base and splits) matching the base_fif.
    """
    base_path = Path(base_fif)
    parent = base_path.parent
    stem = base_path.stem
    all_files = [str(f.resolve()) for f in parent.glob(f"{stem}*.fif")]
    # Always include the base file first
    if str(base_path.resolve()) not in all_files:
        all_files.insert(0, str(base_path.resolve()))
    # Remove duplicates, preserve order
    all_files = list(dict.fromkeys(all_files))
    return all_files

