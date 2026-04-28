#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail

# Require FreeSurfer to be set up in the environment
: "${FREESURFER_HOME:?Please export FREESURFER_HOME first}"

set +eu
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
set -eu

# ---- edit these if needed ----
BIDS_ROOT="/Users/gm33/Desktop/ds004837"
SESSION="ses-0001"
export SUBJECTS_DIR="${BIDS_ROOT}/derivatives/freesurfer"

# FreeSurfer threading per subject
OMP_THREADS="${OMP_THREADS:-4}"
# ------------------------------

mkdir -p "${SUBJECTS_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 2218A [2219A 2220B ...]"
  echo "       accepts either 2218A or sub-2218A"
  exit 1
fi

for code in "$@"; do
  sub="sub-${code#sub-}"
  t1="${BIDS_ROOT}/${sub}/${SESSION}/anat/${sub}_${SESSION}_T1w.nii.gz"
  subjdir="${SUBJECTS_DIR}/${sub}"

  echo
  echo "========================================"
  echo "Subject : ${sub}"
  echo "Input   : ${t1}"
  echo "Output  : ${subjdir}"
  echo "========================================"

  if [[ ! -f "${t1}" ]]; then
    echo "ERROR: missing T1: ${t1}" >&2
    continue
  fi

  if [[ -f "${subjdir}/mri/orig/001.mgz" ]]; then
    echo "Existing FreeSurfer subject found -> resuming without -i"
    recon-all \
      -sd "${SUBJECTS_DIR}" \
      -s "${sub}" \
      -all \
      -parallel \
      -openmp "${OMP_THREADS}"
  else
    echo "No existing FreeSurfer subject found -> starting new recon"
    recon-all \
      -sd "${SUBJECTS_DIR}" \
      -s "${sub}" \
      -i "${t1}" \
      -all \
      -parallel \
      -openmp "${OMP_THREADS}"
  fi
done