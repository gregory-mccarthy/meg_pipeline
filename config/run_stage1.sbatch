#!/bin/bash
#SBATCH --job-name=meg_stage1
#SBATCH --output=logs/meg_stage1_%j.out
#SBATCH --error=logs/meg_stage1_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=day
#SBATCH --account=mccarthy
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs

# Headless-safe matplotlib (Stage 1 is non-interactive)
export MPLBACKEND=Agg

# Make NumPy/BLAS respect SLURM CPU request
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Use your exact Python
PY=/home/gm33/.conda/envs/mne/bin/python

# Sanity checks in logs
$PY -c "import sys; print('Python:', sys.executable); print('Version:', sys.version)"
$PY -c "import mne; print('MNE:', mne.__version__)"

# Stage 1 only (writes checkpoint then exits)
srun $PY preprocess_meg.py config/preproc_params_epi_SLURM.yml --force-stage stage1