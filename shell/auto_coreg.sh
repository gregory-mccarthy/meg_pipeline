# 1. Setup FreeSurfer
export FREESURFER_HOME=/Applications/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# 2. Unset FreeSurfer's python paths
unset PYTHONPATH

# 3. Setup Subjects Dir
export SUBJECTS_DIR=/Users/gm33/data/benchmark_squid/derivatives/freesurfer

# 4. Navigate to your project
cd ~/meg_project/

# 5. THE FIX: Run the script using the ABSOLUTE path to your Conda Python
/Users/gm33/miniforge3/envs/mne-qt6/bin/python auto_coreg.py
