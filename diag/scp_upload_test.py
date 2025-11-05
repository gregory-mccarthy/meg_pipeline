import subprocess
import os

# --- HARDCODED PATHS/VALUES FROM YOUR TEST SCRIPT ---
local_file = './temp/bids/derivatives/preprocessing/sub-005/ses-1/meg/sub-005_ses-1_task-mns_meg_dummyproc.fif'
hpc_user = 'gm33'
hpc_host = 'transfer-milgram.ycrc.yale.edu'
remote_dir = '/gpfs/milgram/scratch/mccarthy/gm33/BIDS/sep/derivatives/preprocessing/sub-005/ses-1/meg'

remote_spec = f"{hpc_user}@{hpc_host}:{remote_dir}/"

if not os.path.exists(local_file):
    print(f"[error] Local file does not exist: {local_file}")
    exit(1)

print(f"[info] Uploading {local_file} to {remote_spec}")
print("[info] You will be prompted for DUO. Approve as needed.")

try:
    # Call scp as a subprocess, using your terminal for input/output
    subprocess.run(
        ["scp", local_file, remote_spec],
        check=True
    )
    print("[success] File uploaded successfully.")
except subprocess.CalledProcessError as e:
    print(f"[error] SCP upload failed: {e}")
    print("Common causes: VPN not active, DUO not approved, remote directory missing, or network issue.")
except Exception as e:
    print(f"[error] Unexpected error: {e}")
