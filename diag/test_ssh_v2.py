import os
import sys
import yaml
import platform
import socket
import subprocess

from pathlib import Path
from bids_io_utils import fetch_bids_fif_files, push_bids_output

def get_execution_environment(hpc_hostname_tag="milgram"):
    sysname = platform.system().lower()
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

# Robust reader
import mne
def read_raw_bids_robust(bids_path, **kwargs):
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
    if session: fname += f"_ses-{session}"
    if task: fname += f"_task-{task}"
    if run: fname += f"_run-{run}"
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
    raw = mne.io.read_raw_fif(fif_file, preload=True, **kwargs)
    return raw

# ---- Safe Wrappers ----

def safe_remote_ls(*args, **kwargs):
    try:
        return kwargs['fn'](*args)
    except subprocess.CalledProcessError as e:
        print("[error] Failed to list remote files via SSH.")
        print("Action: Make sure you are on Yale's VPN, your DUO device is ready, and your cluster login is correct.")
        print(f"Details: {e}")
        sys.exit(1)

def safe_fetch(*args, **kwargs):
    try:
        return fetch_bids_fif_files(*args, **kwargs)
    except subprocess.CalledProcessError as e:
        print("[error] Failed to fetch file from HPC using SCP.")
        print("Action: Ensure you're on VPN, respond to DUO prompt, check remote path spelling, and that you have read permissions.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[error] Unexpected error during fetch: {e}")
        sys.exit(1)

def safe_push(*args, **kwargs):
    try:
        return push_bids_output(*args, **kwargs)
    except subprocess.CalledProcessError as e:
        print("[error] Failed to upload output to HPC using SCP.")
        print("Action: Is your VPN on? Did you respond to DUO? Does the remote output directory exist? Do you have write permissions?")
        print("Tip: If not, log into the cluster and run `mkdir -p /desired/dir` first.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[error] Unexpected error during upload: {e}")
        sys.exit(1)

# ---- Main script ----

def print_banner():
    print("="*70)
    print("Yale BIDS-Aware MEG Data Fetch/Process/Return Demo")
    print("--------------------------------------------------")
    print("  - Fetches BIDS files from Yale cluster to your local temp cache (Mac/Linux).")
    print("  - Runs robust MEG file reader (split file support).")
    print("  - Writes output and pushes it back to cluster (will prompt for DUO).")
    print("Requirements:")
    print("  - You MUST be connected to Yale's VPN.")
    print("  - Your DUO device must be ready for 2FA prompts (for each network operation).")
    print("  - The remote output directory must exist.")
    print("="*70)

def main():
    print_banner()
    if len(sys.argv) < 2:
        print("Usage: python test_bids_aware_io.py config.yaml")
        sys.exit(1)
    config = yaml.safe_load(open(sys.argv[1], 'r'))
    subject = config['subject']
    session = config.get('session', None)
    task = config.get('task', None)
    run = config.get('run', None)
    bids_root_remote = config['bids_root_remote']
    hpc_host = config['hpc_host']
    hpc_user = config['hpc_user']
    temp_dir = config.get('temp_dir', './temp')
    output_deriv_rel = config.get('output_deriv_rel', 'derivatives/preprocessing')
    ENV = get_execution_environment(hpc_hostname_tag="milgram")
    print(f"[info] Detected environment: {ENV}")

    bids_subdir = f"sub-{subject}"
    bids_sesdir = f"ses-{session}" if session else ""
    meg_dir_rel = os.path.join(bids_subdir, bids_sesdir, "meg")
    base_stem = f"sub-{subject}"
    if session: base_stem += f"_ses-{session}"
    if task: base_stem += f"_task-{task}"
    if run: base_stem += f"_run-{run}"
    base_stem += "_meg"
    remote_meg_dir = os.path.join(bids_root_remote, meg_dir_rel)
    local_bids_root = os.path.join(temp_dir, "bids")
    local_meg_dir = os.path.join(local_bids_root, meg_dir_rel)
    os.makedirs(local_meg_dir, exist_ok=True)

    if ENV == "hpc":
        print("[info] Running on HPC: using remote BIDS files in place.")
        input_bids_root = bids_root_remote
    elif ENV == "darwin":
        print("[info] Running on Mac: fetching required BIDS FIF files to local temp cache...")
        safe_fetch(hpc_host, hpc_user, remote_meg_dir, base_stem, local_meg_dir)
        input_bids_root = local_bids_root
    elif ENV == "linux":
        print("[info] Running on local Linux: fetching required BIDS FIF files to local temp cache...")
        safe_fetch(hpc_host, hpc_user, remote_meg_dir, base_stem, local_meg_dir)
        input_bids_root = local_bids_root
    elif ENV == "windows":
        print("[info] Running on Windows: not supported yet. Exiting.")
        sys.exit(1)
    else:
        print(f"[error] Unknown environment: {ENV}. Exiting.")
        sys.exit(1)

    print("[info] Reading raw data using robust BIDS-aware function...")
    bids_path = dict(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype='meg',
        bids_root=input_bids_root
    )
    try:
        raw = read_raw_bids_robust(bids_path)
    except Exception as e:
        print(f"[error] Failed to load MEG FIF file(s): {e}")
        print("Check that your temp/ folder contains all required split FIFs.")
        sys.exit(1)
    print(f"[info] Loaded raw object: {raw}")

    # Output
    local_deriv_dir = os.path.join(input_bids_root, output_deriv_rel, meg_dir_rel)
    os.makedirs(local_deriv_dir, exist_ok=True)
    local_output_file = os.path.join(local_deriv_dir, f"{base_stem}_dummyproc.fif")
    print(f"[info] Writing dummy output file: {local_output_file}")
    try:
        raw.save(local_output_file, overwrite=True)
    except Exception as e:
        print(f"[error] Failed to save output file: {e}")
        print("Check disk space and permissions in your temp/ folder.")
        sys.exit(1)

    if ENV in ("darwin", "linux"):
        remote_deriv_dir = os.path.join(bids_root_remote, output_deriv_rel, meg_dir_rel)
        print(f"[info] Pushing output files back to: {remote_deriv_dir}")
        safe_push([local_output_file], remote_deriv_dir, hpc_host, hpc_user)

    print("[info] DONE. All steps completed successfully.")

if __name__ == "__main__":
    main()
    