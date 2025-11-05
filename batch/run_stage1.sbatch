#!/usr/bin/env bash
#SBATCH --job-name=meg_stage1
#SBATCH --output=logs/meg_stage1_%j.out
#SBATCH --error=logs/meg_stage1_%j.err
#SBATCH --account=mccarthy
#SBATCH --partition=day          # or debug/general as appropriate
#SBATCH --qos=normal
#SBATCH --time=06:00:00          # give yourself cushion; adjust after 1 run
#SBATCH --cpus-per-task=1        # Stage-1 is effectively single-threaded
#SBATCH --mem=144G               # you observed ~128G; add headroom

set -euo pipefail
mkdir -p logs

# Headless-safe matplotlib
export MPLBACKEND=Agg

# Keep BLAS threads sane (too many can slow Python pipelines)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Fast local temp/cache
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export MNE_CACHE_DIR="$TMPDIR/mne_cache"
mkdir -p "$MNE_CACHE_DIR"

# Your exact Python
PY=/home/gm33/.conda/envs/mne/bin/python

# Repo & YAML – use the new Stage-1 SLURM YAML you just created
REPO=/home/gm33/projects/meg_pipeline
YAML=/home/gm33/projects/work/preproc_params_SLURM_sub011_rest.yml   # <- update if different

# Sanity log
$PY - <<'PY'
import sys, mne, os
print("Python:", sys.executable)
print("Version:", sys.version.split()[0])
print("MNE:", mne.__version__)
print("OMP/MKL/OPENBLAS:", os.getenv("OMP_NUM_THREADS"), os.getenv("MKL_NUM_THREADS"), os.getenv("OPENBLAS_NUM_THREADS"))
PY

cd "$REPO"

echo "[INFO] Host: $(hostname)"
echo "[INFO] YAML: $YAML"
echo "[INFO] CPUs: ${SLURM_CPUS_PER_TASK}  MEM: ${SLURM_MEM_PER_NODE:---mem=144G}"

# Stage 1 only (writes checkpoint then exits)
# Wrap with time to log peak RSS
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%M KB' \
srun $PY preprocess_meg.py "$YAML" --force-stage stage1

echo "[INFO] sacct summary:"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ReqMem,ReqCPUS
