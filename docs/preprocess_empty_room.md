# preprocess_empty_room.py — Empty Room Noise Covariance Pipeline

Preprocesses an MEG empty room recording and computes a noise covariance matrix for use in inverse modeling (e.g. MNE, dSPM, LCMV beamformer). This script is designed to be driven by the same YAML configuration file used by `preprocess_meg.py`, so that identical Maxwell filter and notch filter parameters are applied to both the task data and the empty room recording. Steps that are only meaningful for recordings with a subject in the scanner — head movement compensation, ICA, and AutoReject — are skipped.

## Purpose

Source localization methods require a noise covariance matrix that characterizes the sensor noise floor. The standard approach is to record a few minutes of data with no subject in the scanner (the "empty room"), then preprocess it with the same filtering chain applied to the task data so that the noise estimate reflects the same signal conditioning. This script automates that workflow.

## Quick Start

```bash
# Automatic discovery — finds task-noise or task-emptyroom in the subject's MEG directory
python preprocess_empty_room.py config.yaml

# Explicit override — use a specific empty room file
python preprocess_empty_room.py config.yaml --empty-room-file /data/sub-001/ses-01/meg/sub-001_ses-01_task-noise_run-02_meg.fif
```

Pass the same YAML you use with `preprocess_meg.py`. By default, the script ignores the `task` and `run` fields in the YAML and instead scans the subject's MEG directory for a file with `task-noise` or `task-emptyroom` in its filename. If the session contains multiple noise recordings and you need to select one, use `--empty-room-file` to point to it directly.

## How Discovery Works

Given the `subject`, `session`, and `bids_root` from the YAML, the script resolves the subject's MEG directory (e.g. `<bids_root>/sub-001/ses-01/meg/`) and parses every `.fif` filename in it. It looks for files whose BIDS `task` entity is either `noise` or `emptyroom`, in that priority order. If neither is found, the script exits with an error listing the task labels that are present on disk so you can diagnose the problem.

The `task` and `run` fields from the YAML are ignored entirely — `task` because it refers to the active experiment (e.g. `mns`), and `run` because the task data's run number does not apply to the empty room recording.

When `--empty-room-file` is provided, the entire discovery step is skipped. The script loads the specified file directly and parses its filename to extract the task label for output naming. If the filename doesn't follow BIDS conventions, the task defaults to `noise`.

## Pipeline Steps

1. **Load configuration** — reads the YAML, resolves BIDS paths, and locates the Maxwell filter calibration and cross-talk files.
2. **Load empty room data** — reads the `.fif` file identified by the BIDS path. Optionally crops to a time window if one is specified in the YAML.
3. **Maxwell filter** — applies Signal Space Separation in empty-room mode: `coord_frame='meg'`, origin at `(0, 0, 0)`, no head position file, no destination transform. The `st_duration` parameter is read from the YAML and capped to the recording length if necessary.
4. **Notch filter** — removes power line harmonics (default 60 Hz and harmonics up to the 4th) using an FIR filter on MEG channels only.
5. **Compute noise covariance** — estimates the covariance matrix from the full preprocessed recording using both shrinkage and empirical methods, which ensures the matrix is well-conditioned even when the number of samples is modest.
6. **Save outputs** — writes the preprocessed data, the covariance matrix, and QC plots.
7. **Write logs** — saves processing parameters and summary statistics in both YAML and JSON format.

## Outputs

All outputs are written under `<bids_root>/derivatives/<pipeline_name>/`. The filenames follow BIDS derivative conventions, derived from the subject, session, task, and run fields in the YAML. For a typical configuration with `subject: 01`, `session: meg`, `task: noise`:

| Output | Filename pattern | Description |
|---|---|---|
| Preprocessed data | `*_desc-preproc_meg.fif` | Maxwell-filtered and notch-filtered empty room recording |
| Noise covariance | `*_desc-preproc_cov.fif` | Covariance matrix for inverse modeling |
| QC plots | `plots/` directory alongside the outputs | PSD plots at each stage, covariance matrix visualization |
| Processing log | `*_desc-preproc_log.yaml` and `*_desc-preproc_log.json` | Full parameter record and summary statistics |

## Command-Line Parameters

### Positional

```
config            Path to the YAML configuration file.
                  This is the same YAML used by preprocess_meg.py.
                  The script reads subject, session, bids_root,
                  Maxwell filter calibration/cross-talk paths,
                  line frequency, and optional time window from
                  this file. The 'task' and 'run' fields are
                  ignored — the empty room file is discovered
                  automatically.
```

### Optional

```
--empty-room-file PATH
                  Path to a specific empty room .fif file. When
                  provided, the script loads this file directly
                  and skips automatic discovery. Use this when the
                  session contains multiple noise recordings (e.g.
                  a bad first recording that was re-collected) and
                  you need to select one explicitly.
```

There are no other command-line flags. All processing parameters come from the YAML.

## What Is Skipped Compared to preprocess_meg.py

The following steps from the full task pipeline are not applicable to empty room data and are silently ignored, even if configured in the YAML:

- **Head movement compensation** — no subject is present, so there is no head position to track. The Maxwell filter runs with `head_pos=None` and `destination=None`.
- **ICA** — there are no physiological artifacts (heartbeat, blinks) to remove.
- **AutoReject** — there are no movement artifacts or trial structure to evaluate.

## Relationship to preprocess_meg.py

The intended workflow is:

1. Preprocess your task data with `preprocess_meg.py` using a YAML config.
2. Run `preprocess_empty_room.py` with the **same YAML** — no edits needed. The script automatically locates the empty room recording (`task-noise` or `task-emptyroom`) in the subject's MEG directory and applies identical Maxwell and notch filtering.
3. Use the resulting `*_cov.fif` as the noise covariance in your inverse modeling pipeline.

## Dependencies

- Python 3
- MNE-Python
- NumPy
- `meg_pipeline_utils` (project-internal utility module)
- `bids_io_utils` (project-internal BIDS I/O module)
