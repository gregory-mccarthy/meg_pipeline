import os
import time
import subprocess

def scp_copy_fif(hpc_host, hpc_user, hpc_path, local_temp='temp'):
    # Prepare local destination directory
    os.makedirs(local_temp, exist_ok=True)
    local_path = os.path.join(local_temp, os.path.basename(hpc_path))

    print(f"Transferring file from {hpc_user}@{hpc_host}:{hpc_path} to {local_path}")
    t0 = time.time()
    try:
        remote = f"{hpc_user}@{hpc_host}:{hpc_path}"
        cmd = ["scp", remote, local_path]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        t1 = time.time()
        elapsed = t1 - t0
        print(f"Transfer complete. Elapsed time: {elapsed:.2f} seconds.")
        return elapsed
    except subprocess.CalledProcessError as e:
        print("Error during transfer:", e)
        return None

if __name__ == '__main__':
    # Example usage (replace values as needed)
    hpc_host = 'transfer-milgram.ycrc.yale.edu'  # e.g. login node
    hpc_user = 'gm33'
    hpc_path = '/gpfs/milgram/scratch/mccarthy/gm33/BIDS/sep/sub-005/ses-1/meg/sub-005_ses-1_task-mns_meg.fif'
    elapsed_time = scp_copy_fif(hpc_host, hpc_user, hpc_path)
