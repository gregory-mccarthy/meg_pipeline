#!/bin/bash
#SBATCH --job-name=meg_stage1
#SBATCH --output=logs/meg_stage1_%j.out
#SBATCH --error=logs/meg_stage1_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --account=mccarthy
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# If your site uses modules for conda:
module load miniconda 2>/dev/null || true

# Activate your environment
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mne

# Headless-safe matplotlib (Stage 1 is non-interactive anyway)
export MPLBACKEND=Agg

# Make NumPy/BLAS respect SLURM CPU request
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run STAGE 1 only (write checkpoint + exit)
srun python preprocess_meg.py config/preprroc_params_epi_SLURM.yaml --force-stage stage1