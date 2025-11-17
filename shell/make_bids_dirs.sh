#!/bin/bash
# Usage: ./make_bids_full_structure.sh /path/to/project-root

PROJECT_ROOT=$1

if [ -z "$PROJECT_ROOT" ]; then
  echo "Usage: $0 /path/to/project-root"
  exit 1
fi

# --- RAW BIDS STRUCTURE ---
mkdir -p "${PROJECT_ROOT}/sub-01/ses-01/meg"
touch "${PROJECT_ROOT}/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_meg.fif"
touch "${PROJECT_ROOT}/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_eeg.edf"
touch "${PROJECT_ROOT}/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_channels.tsv"
touch "${PROJECT_ROOT}/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_events.tsv"
touch "${PROJECT_ROOT}/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_meg.json"

# --- EMPTY ROOM (BIDS style) ---
mkdir -p "${PROJECT_ROOT}/sub-emptyroom/ses-20240618/meg"
touch "${PROJECT_ROOT}/sub-emptyroom/ses-20240618/meg/sub-emptyroom_ses-20240618_task-noise_meg.fif"

# --- CALIBRATION FILES ---
mkdir -p "${PROJECT_ROOT}/calibration"
touch "${PROJECT_ROOT}/calibration/sss_cal_factory_20230619.dat"
touch "${PROJECT_ROOT}/calibration/ct_sparse_triux2.fif"
touch "${PROJECT_ROOT}/calibration/README.md"

# --- MONTAGES (custom caps) ---
mkdir -p "${PROJECT_ROOT}/montages"
touch "${PROJECT_ROOT}/montages/EasyCap_53.sfp"
touch "${PROJECT_ROOT}/montages/custom64.elc"
touch "${PROJECT_ROOT}/montages/README.md"

# --- CONFIG (YAMLs etc.) ---
mkdir -p "${PROJECT_ROOT}/config"
touch "${PROJECT_ROOT}/config/preproc_params_sub-01_run01.yaml"
touch "${PROJECT_ROOT}/config/epoching_params.yaml"

# --- CODE ---
mkdir -p "${PROJECT_ROOT}/code"
touch "${PROJECT_ROOT}/code/preprocess_meg_eeg.py"
touch "${PROJECT_ROOT}/code/run_pipeline.sh"
touch "${PROJECT_ROOT}/code/requirements.txt"

# --- LOGS ---
mkdir -p "${PROJECT_ROOT}/logs"
touch "${PROJECT_ROOT}/logs/sub-01_ses-01_run-01_preproc-log.yaml"
touch "${PROJECT_ROOT}/logs/batch_log_20240618.txt"

# --- DERIVATIVES ---
mkdir -p "${PROJECT_ROOT}/derivatives/preprocessing/sub-01/ses-01/meg"
touch "${PROJECT_ROOT}/derivatives/preprocessing/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_desc-preproc_meg-epo.fif"
touch "${PROJECT_ROOT}/derivatives/preprocessing/sub-01/ses-01/meg/sub-01_ses-01_task-mytask_run-01_preproc-log.yaml"
mkdir -p "${PROJECT_ROOT}/derivatives/preprocessing/sub-01/meg"
touch "${PROJECT_ROOT}/derivatives/preprocessing/sub-01/meg/sub-01_task-mytask_allsessions_desc-preproc_meg-epo.fif"
mkdir -p "${PROJECT_ROOT}/derivatives/group"
touch "${PROJECT_ROOT}/derivatives/group/group_task-mytask_desc-evoked_meg-ave.fif"
touch "${PROJECT_ROOT}/derivatives/group/group_task-mytask_preproc-log.yaml"

# --- PROJECT README ---
touch "${PROJECT_ROOT}/README.md"

echo "BIDS best-practices directory structure created in: $PROJECT_ROOT"