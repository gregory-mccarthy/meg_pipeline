# detect_breaks_utility.py — Rest-Break Detection for BIDS Events

Optional utility that scans the stimulus channel of a `.fif` file for long gaps between events, interprets them as rest breaks or pre/post-experiment dead time, and writes `BAD_break` entries into a BIDS-compliant `_events.tsv` file.

## Quick Start

```bash
python detect_breaks_utility.py sub-01_task-mytask_meg.fif
```

This reads the stimulus channel, identifies any gaps longer than 15 seconds, and writes (or updates) an `_events.tsv` alongside the input file.

## What It Detects

The script identifies three categories of break:

- **Pre-experiment dead time** — the gap between the start of the recording and the first stimulus event.
- **Inter-stimulus breaks** — gaps between consecutive stimulus events that exceed the minimum break duration (e.g. rest periods between experimental blocks).
- **Post-experiment dead time** — the gap between the last stimulus event and the end of the recording.

All three are labelled `BAD_break` in the output, which causes downstream MNE routines (e.g. epoching, averaging) to exclude those segments.

## Output

The script derives the output filename from the input by replacing `_meg.fif` (or `.fif`) with `_events.tsv`. For example:

| Input | Output |
|---|---|
| `sub-01_task-mytask_meg.fif` | `sub-01_task-mytask_events.tsv` |
| `recording_raw.fif` | `recording_raw_events.tsv` |

The TSV has four columns:

| Column | Description |
|---|---|
| `onset` | Start time in seconds relative to the beginning of the recording |
| `duration` | Length of the break in seconds (after padding is applied) |
| `trial_type` | `BAD_break` for detected breaks; `stimulus` for trigger events |
| `value` | Trigger code for stimulus rows; `n/a` for breaks |

If the TSV already exists, existing `BAD_break` rows are removed and replaced with freshly detected ones (all other rows are preserved). If no TSV exists, one is created with both stimulus events and break annotations.

## Padding

Each detected break is trimmed inward by `--pad_sec` seconds on both sides. This avoids clipping the tail end of a stimulus response or the onset of the next block. For example, a 20-second gap with 2-second padding becomes a 16-second `BAD_break` starting 2 seconds after the last pre-gap event.

The post-experiment break includes an additional 0.5-second overshoot on the duration to ensure the full tail of the recording is covered.

## Command-Line Parameters

### Positional

```
raw_path          Path to the input .fif file (absolute or relative).
```

### Optional

```
--stim_channel TEXT
                  Name of the stimulus channel to extract events from.
                  Default: STI101

--min_break_duration_sec FLOAT
                  Minimum gap length (in seconds) between consecutive
                  stimulus events for a gap to be classified as a break.
                  Shorter gaps are assumed to be normal inter-trial
                  intervals and are ignored. Default: 15.0

--pad_sec FLOAT   Seconds to trim from each side of a detected break,
                  so that the BAD_break region does not encroach on
                  legitimate task data. Default: 2.0
```

## Dependencies

- Python 3
- MNE-Python
- NumPy, Pandas
