
# `annotate_data.py`

User guide for interactive annotation of raw or preprocessed FIF files, with optional automatic break detection and BIDS-compliant bad channel export.

## What this program does

`annotate_data.py` opens an MNE browser on a FIF file so a reviewer can:

* add or edit time-segment annotations;
* mark channels as bad by clicking their names;
* optionally inspect a display-filtered copy of the data;
* optionally auto-generate `BAD_*` annotations for long breaks between experimental runs;
* save the reviewed annotation state and bad channel flags to sidecar TSVs without modifying the original FIF file.

This utility is intended for two common places in a processing pipeline:

* before preprocessing, to mark trim regions at the beginning or end of a run, mark breaks that should be excluded from ICA or automated artifact rejection, or tag obvious bad spans/channels;
* after preprocessing, to add after-the-fact annotations such as epileptic spikes, suspicious intervals, or other reviewer-defined events on cleaned data.

The program is deliberately conservative about the data file itself:

* the FIF file is read and displayed;
* the browser works on an in-memory copy;
* the original FIF file is left untouched.

## What this version adds

This version includes all of the earlier annotation-preservation behavior and adds integrated break detection, as well as an automated BIDS-compliant bad channel hand-off.

### Existing annotation behavior

* existing annotations embedded in the FIF are preserved by default;
* existing annotation sidecars are preserved by default;
* `--clear-current-annotations` starts from a blank annotation set;
* the final event sidecar is always written, even if the final annotation set is empty.

### New break-detection behavior

* `--auto-detect-breaks` detects long event-free intervals and turns them into editable `BAD_*` annotations before the browser opens;
* the dead time before the first event and after the last event can also be marked automatically;
* automatic break detection can use either a stimulus channel or preserved event rows already present in a TSV sidecar;
* if the sidecar is an existing `_events.tsv` with experimental events, those event rows are preserved when the file is rewritten.

### New BIDS channel export

* bad channels marked during the interactive review are now automatically exported to a `_channels.tsv` sidecar.
* if an existing `_channels.tsv` is found, the program safely updates only the `status` column without overwriting other metadata.
* if one does not exist, a new BIDS-compliant `_channels.tsv` is generated automatically.

### New batch mode

* `--no-browser` lets you update or generate the sidecars without opening the MNE browser.

This makes the tool usable both as an interactive reviewer and as a pipeline utility.

## Core idea

The program manages three different kinds of information that live in TSV sidecars:

* **event rows**, such as stimulus events already present in a BIDS-style `_events.tsv`;
* **annotation rows**, such as `BAD_break`, `BAD_motion`, `spike`, or reviewer-defined artifact spans (also stored in the event sidecar);
* **channel status flags**, mapping individual channels as `good` or `bad` (stored in the `_channels.tsv` sidecar).

When the event sidecar already contains real event rows, the program preserves them and only updates the annotation rows. When updating channels, it preserves existing channel metadata and only updates the status flag.

That means you can safely use the tool on files that already have experimental events and BIDS metadata on disk.

## Main workflow

The program has five stages.

### 1. Load the FIF file

The input must be a `.fif` or `.fif.gz` file.

The file is read with preload enabled so the browser is responsive and optional display filtering can be applied.

### 2. Build the starting annotation set

By default, the browser session begins with all currently known annotations preserved.

That means the program will keep:

* annotations already embedded in the FIF file;
* annotation rows already present in the chosen sidecar TSV.

If you use:

```
--clear-current-annotations

```

the program ignores both of those annotation sources for the current session and starts from a blank annotation set.

Important: when a sidecar also contains **true event rows**, those are still preserved on disk when the sidecar is rewritten. `--clear-current-annotations` clears the annotation set for the review session. It does **not** erase non-annotation event rows.

### 3. Optionally auto-detect breaks and trims

If you use:

```
--auto-detect-breaks

```

the program looks for long intervals with no events and adds `BAD_*` annotations for them before the browser opens.

This is useful for acquisitions that contain:

* several experimental runs within a single recording;
* long breaks between runs;
* dead time before the first stimulus;
* dead time after the last stimulus.

The added annotations become normal MNE annotations in the browser, so you can inspect them, delete them, rename them, or adjust them manually.

### 4. Open the MNE browser

Unless `--no-browser` is used, the program opens the MNE browser so you can:

* press `a` to enter annotation mode;
* click and drag to create time-segment annotations;
* click a channel name to toggle it in or out of the bad-channel list;
* close the window to finish the session.

### 5. Write the sidecar TSVs

When the session ends, the program writes the reviewed state to two distinct sidecar TSVs:

1. **The Event/Annotation Sidecar:** Updates the time segments. If the sidecar already contained true event rows, those rows are preserved and the annotation rows are updated.
2. **The Channels Sidecar:** Updates the `status` column for the BIDS `_channels.tsv` to formally register which channels were marked bad during the session.

## Sidecar naming rules

You can set the event/annotation sidecar path explicitly with:

```
--annotation-file /path/to/your_sidecar.tsv

```

If you do not specify `--annotation-file`, the program chooses the paths automatically based on the input file.

### BIDS-like FIF names

If the input name ends with one of these entity suffixes before `.fif` or `.fif.gz`:

* `_meg`
* `_eeg`
* `_ieeg`
* `_nirs`
* `_beh`

then the default sidecars are the matching BIDS-style `*_events.tsv` and `*_channels.tsv`.

Example:

* input: `sub-01_task-rest_meg.fif`
* default events sidecar: `sub-01_task-rest_events.tsv`
* default channels sidecar: `sub-01_task-rest_channels.tsv`

### Generic FIF names

If the input does not look like a BIDS raw file, the default sidecars are:

`<stem>_annotations.tsv`
`<stem>_channels.tsv`

Example:

* input: `subject01_rest_preproc.fif`
* default events sidecar: `subject01_rest_preproc_annotations.tsv`
* default channels sidecar: `subject01_rest_preproc_channels.tsv`

### Alternate sidecars

If a default events sidecar exists, it is used for the session.

If another plausible events sidecar also exists beside it, that alternate file is reported but not written during the session.

This matters most for preprocessed data where you may have both:

* an annotations sidecar for manual marks;
* an events sidecar containing stimulus events.

The program can still use preserved event rows from the alternate sidecar for break detection when needed.

## How sidecar rows are interpreted

When an event sidecar exists, the program separates its rows into two groups.

### Preserved event rows

Rows are treated as true event rows and preserved when they look like ordinary experimental events, for example rows with:

* numeric `value` codes;
* or, in some sidecar layouts, numeric `sample` values with no annotation description.

These rows are **not** loaded into the MNE browser as editable annotations.

### Annotation rows

Rows are treated as annotations when they look like reviewer marks or bad segments, for example:

* rows with descriptions such as `BAD_break`, `BAD_motion`, `spike`, or other manual labels;
* rows without a numeric event code;
* rows created by this annotator in earlier sessions.

These rows are loaded into the browser as MNE annotations.

## Automatic break detection

This is the main integrated feature from the separate break utility.

### What it detects

With `--auto-detect-breaks`, the program examines event onsets and marks three kinds of event-free spans when they are long enough:

* **pre-experiment dead time**: from file start to just before the first event;
* **inter-run breaks**: long gaps between one event and the next;
* **post-experiment dead time**: from just after the last event to the end of the file.

These become normal annotation rows with descriptions starting with `BAD_` by default.

### Why this is useful

Breaks often contain more movement, talking, repositioning, or environmental artifact than the actual task periods.

Marking them as bad means they can be excluded from downstream steps that respect MNE annotations, such as artifact-rejection workflows that reject `BAD_*` spans.

### Source of event onsets

Break detection can get event onsets from three places.

#### `--break-source auto`  (default)

Try the stimulus channel first. If that does not work, fall back to preserved event rows from existing sidecar TSVs.

#### `--break-source stim`

Use only the stimulus channel.

#### `--break-source sidecar`

Use only preserved event rows from existing sidecar TSVs.

This is especially useful for preprocessed files or workflows where the event timing already lives in an `_events.tsv`.

### Stimulus channel settings

Use these options when break detection reads the recording itself:

```
--break-stim-channel STI101
--break-event-min-duration 0.002

```

The defaults match the behavior of the original standalone break utility.

### Gap threshold and padding

Use these options to control what gets marked:

```
--break-min-gap-sec 15.0
--break-pad-sec 2.0

```

Interpretation:

* a gap must be longer than `--break-min-gap-sec` to be marked;
* `--break-pad-sec` is trimmed off the edges of the break so the annotation does not swallow the task periods around it;
* for the pre-first and post-last trim regions, the padding is removed only on the side adjacent to the first or last event.

### Break labels

By default, inter-run breaks use:

`BAD_break`

You can change that with:

```
--break-description BAD_break

```

If you want the pre-first and post-last trim regions to use a different label, set:

```
--break-edge-description BAD_trim

```

If `--break-edge-description` is not set, the edge trims use the same label as the inter-run breaks.

### Replacing older auto-break annotations

When `--auto-detect-breaks` is used, the program will, by default, remove existing annotation rows whose description matches the requested break labels and then regenerate them.

This mirrors the behavior of the original standalone utility, which replaced earlier `BAD_break` rows.

If you want to keep any existing rows with those descriptions, use:

```
--keep-existing-break-annotations

```

## Browser mode and batch mode

### Interactive review mode

This is the default.

Example:

```
python annotate_data_updated.py sub-01_task-rest_meg.fif --auto-detect-breaks

```

The browser opens with the current annotations already loaded, including any newly detected breaks.

### Batch mode

Use:

```
--no-browser

```

to skip the browser and just write the resulting sidecar.

This is useful for:

* batch generation of `BAD_break` spans;
* updating sidecars in a headless pipeline;
* clearing annotations while preserving event rows;
* repairing or shifting annotation sidecars without doing a manual review immediately.

Example:

```
python annotate_data_updated.py sub-01_task-rest_meg.fif \
    --auto-detect-breaks \
    --no-browser

```

## Existing annotations and `--clear-current-annotations`

This option controls the starting annotation set in the browser.

Without it, the program preserves:

* embedded FIF annotations;
* annotation rows from the chosen sidecar.

With it, the program starts from a blank annotation set.

Example:

```
python annotate_data.py my_run_preproc.fif --clear-current-annotations

```

If you combine it with automatic break detection:

```
python annotate_data.py my_run_preproc.fif \
    --clear-current-annotations \
    --auto-detect-breaks

```

you will start from a blank annotation set and then add only the newly detected break/trim annotations.

## Optional display filtering

The browser can show a filtered copy of the data.

This affects only what you see during review. It does not alter the source FIF and it does not alter the saved annotation times.

Use:

```
--hpf 1.0

```

for a high-pass display filter, or:

```
--lpf 50.0

```

for a low-pass display filter, or both together:

```
python annotate_data_updated.py my_run.fif --hpf 1.0 --lpf 40.0

```

## Browser scaling

The script computes robust display scalings separately for MAG, GRAD, EEG, EOG, and ECG channels when those channel types are present.

The goal is not to change the data. It is to make mixed-modality browsing easier by preventing one channel type from dominating the display.

Relevant options are:

* `--scale-window-sec`
* `--scale-abs-quantile`
* `--scale-channel-quantile`
* `--scale-mult`

In most cases, the defaults are reasonable.

## Output sidecar formats

### The Events/Annotation Sidecar

The final events sidecar always contains annotation rows written with these columns:

* `onset`
* `duration`
* `description`
* `trial_type`

For annotation rows, `trial_type` is set equal to `description`.

If the sidecar also contains preserved event rows from an earlier `_events.tsv`, those rows remain in the file and keep their original event columns such as:

* `value`
* `sample`
* or any other event metadata already present.

### The Channels Sidecar

The `_channels.tsv` sidecar tracks which channels are currently active and which should be ignored during processing.

If the script creates a new `_channels.tsv`, it will include these base columns:

* `name`
* `type`
* `status` (populated with `good` or `bad`)

If the script updates an *existing* `_channels.tsv`, it will leave all pre-existing columns intact (e.g., coordinates, units, filtering parameters) and safely update only the `status` column for the specific channels reviewed.

### Empty annotation result

If no annotation rows remain at the end of the session, the program still writes the event sidecar.

Two cases are possible:

* if the sidecar also had preserved event rows, the file is saved with those event rows and no annotation rows;
* if there were no preserved event rows, the program writes a header-only annotation table.

This behavior prevents stale annotations from an earlier run from remaining on disk.

## Existing sidecar loading rules

When the program loads an existing events sidecar, annotation rows can come from either:

* a `description` column;
* or a `trial_type` column.

The option:

```
--shift-annotations X

```

adds `X` seconds to the annotation rows loaded from the sidecar before the browser opens.

This is intended only for correcting older sidecars that were known to be offset.

It does not shift:

* embedded FIF annotations;
* preserved event rows.

## Typical workflows

### 1. Review a raw BIDS run and mark bad spans manually

```
python annotate_data.py sub-01_task-rest_meg.fif

```

### 2. Mark breaks automatically, then inspect them manually

```
python annotate_data.py sub-01_task-rest_meg.fif \
    --auto-detect-breaks

```

### 3. Mark breaks automatically without opening a browser

```
python annotate_data.py sub-01_task-rest_meg.fif \
    --auto-detect-breaks \
    --no-browser

```

### 4. Use an existing events sidecar to mark breaks on preprocessed data

```
python annotate_data.py sub-01_task-rest_preproc.fif \
    --auto-detect-breaks \
    --break-source sidecar

```

This is a common pattern when the preprocessed file does not have a useful stimulus channel but an `_events.tsv` already exists.

### 5. Add after-the-fact annotations to preprocessed data

```
python annotate_data_updated.py sub-01_task-rest_preproc.fif \
    --hpf 1.0 --lpf 70.0

```

### 6. Start from scratch, then add only auto-detected breaks

```
python annotate_data.py sub-01_task-rest_preproc.fif \
    --clear-current-annotations \
    --auto-detect-breaks

```

### 7. Use a pipeline-specific sidecar path

```
python annotate_data.py my_run.fif \
    --annotation-file review/manual_annotations.tsv

```

### 8. Repair an older annotation sidecar with a known onset offset

```
python annotate_data.py my_run.fif \
    --shift-annotations -0.500

```

## Bad-channel marking

Clicking channel names in the browser to mark them as bad works exactly as expected (turning the channel trace gray).

Channels marked this way are stored in the browser session’s `bads` list and exported directly to the `_channels.tsv` sidecar when the window closes.

Important: this utility still does **not** write bad channels back into the FIF file header on disk.

The original non-destructive design is fully preserved:

* annotation time segments go to the `_events.tsv` sidecar;
* bad channel flags go to the `status` column of the `_channels.tsv` sidecar.
* the raw `.fif` is left completely unmodified.

## Command-line reference

### Required argument

`file`
: path to the input `.fif` or `.fif.gz` file

### Sidecar control

`--annotation-file`
: explicit path to the events sidecar TSV

`--clear-current-annotations`
: ignore existing annotations for the current session and start from a blank annotation set

`--shift-annotations`
: apply a fixed shift, in seconds, to annotation rows loaded from the sidecar

### Automatic break detection

`--auto-detect-breaks`
: detect long event-free gaps and add BAD-style annotations before review

`--break-source {auto,stim,sidecar}`
: choose whether event onsets come from the stimulus channel, sidecar event rows, or automatic fallback behavior

`--break-stim-channel`
: stimulus channel used for stim-based break detection

`--break-event-min-duration`
: `min_duration` passed to `mne.find_events`

`--break-min-gap-sec`
: minimum event-free interval required to mark a break

`--break-pad-sec`
: padding removed from break edges

`--break-description`
: label used for inter-run breaks

`--break-edge-description`
: label used for pre-first and post-last trim spans

`--keep-existing-break-annotations`
: do not remove older rows whose description already matches the break labels

### Browser display

`--hpf`
: high-pass cutoff for the browser copy only

`--lpf`
: low-pass cutoff for the browser copy only

### Browser scaling

`--scale-window-sec`
: window length used to estimate robust scaling

`--scale-abs-quantile`
: per-channel absolute-value quantile used in scaling estimation

`--scale-channel-quantile`
: across-channel quantile used to choose the displayed scale for a channel type

`--scale-mult`
: multiplier applied to the chosen scale

### Batch mode

`--no-browser`
: skip opening the MNE browser

## Practical notes

### The original FIF is untouched

This program is a review and sidecar-writing utility. It does not rewrite the original FIF file.

### Event rows are preserved

If the chosen sidecar already contains real event rows, they are preserved when the sidecar is rewritten.

That is the main reason this integrated version is safer than a naive “annotation-only” TSV writer.

### Auto-detected breaks are editable

The breaks created by `--auto-detect-breaks` are only a starting point. They appear in the browser as ordinary annotations, so they can be adjusted or deleted.

### Annotation timing is relative to the file you opened

The saved `onset` values are relative to the beginning of the loaded FIF file. This is important for cropped or preprocessed datasets.

## Troubleshooting

### “Why do I still see old annotations?”

Because the default behavior is to preserve existing annotations.

Use:

```
--clear-current-annotations

```

when you want a blank starting point.

### “Why were my event rows kept even though I cleared annotations?”

Because `--clear-current-annotations` only clears the annotation set for the review session.

True event rows in an events sidecar are preserved.

### “Why didn’t break detection find anything?”

Possible reasons:

* the recording did not contain usable stimulus events on the requested channel;
* the sidecar did not contain preserved event rows;
* the gaps were shorter than `--break-min-gap-sec`;
* the padding removed the entire gap.

Try:

* `--break-source stim`
* `--break-source sidecar`
* lowering `--break-min-gap-sec`
* lowering `--break-pad-sec`

### “Why is my bad-channel list not saved into the FIF?”

Because the tool fundamentally does not modify the original `.fif` file. Instead, bad channels are stored compliantly in the `status` column of the generated `_channels.tsv` sidecar, which can be natively ingested by downstream BIDS parsers and headless pipelines.

### “Why are my display filters not changing the saved timings?”

Because filtering is applied only to the browser copy.

The saved annotation times always refer to the original file time base.

## Summary

`annotate_data.py` is now both an interactive annotation tool and a practical BIDS pipeline utility. It preserves existing annotations by default, can start from a blank annotation set when requested, can preserve event rows already present in an `_events.tsv`, can auto-detect and mark long breaks between task periods, safely exports bad channels to a `_channels.tsv`, and keeps the original FIF file untouched throughout the review process.