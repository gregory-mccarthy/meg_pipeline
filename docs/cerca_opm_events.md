# `cerca_opm_events.py`

User guide for extracting BIDS-style `_events.tsv` files from Cerca OPM analog trigger lines

## What this script does

`cerca_opm_events.py` extracts event onsets from analog trigger bit lines and writes a BIDS-style `_events.tsv` file in the same folder as the raw data file.

It is designed for Cerca OPM recordings, and for any other MNE-readable raw file, where trigger information is stored as separate analog channels rather than as a single packed digital channel such as `STI101`.

The script is built to handle the main failure modes that make naive trigger extraction unreliable:

- a TTL line may show noise or a brief threshold crossing that is not a real event;
- different trigger bits that belong to the same code may be captured on adjacent digitized samples instead of all at once;
- some trigger channels may never truly change state and should not be allowed to generate false events;
- one-sample blips may need to be rejected.

The script treats any activity on any trigger line as a potential onset, confirms the event code over the onset sample plus one or more look-ahead samples, and stores the onset at the first active sample. Pulse duration is used only for filtering and diagnostics. The written BIDS `duration` is always `0`.

## Main idea

Suppose your trigger bits are stored on 8 analog channels.

The script:

1. thresholds each analog line into 0 or 1;
2. packs those bit values into a composite integer on every sample;
3. detects any region where at least one bit is active;
4. marks the onset at the first active sample in that region;
5. confirms the final event value using the onset sample and the next sample by default;
6. optionally opens an MNE browser so the trigger lines and estimated events can be reviewed;
7. writes a final `_events.tsv` and a diagnostic table.

By default, the packed value is computed in least-significant-bit to most-significant-bit order:

- first trigger channel = bit 0 = value 1
- second trigger channel = bit 1 = value 2
- third trigger channel = bit 2 = value 4
- ...
- eighth trigger channel = bit 7 = value 128

So the integer value is:

`value = bit0*1 + bit1*2 + bit2*4 + ...`

## Requirements

You need Python 3 plus the packages used by the script:

- `mne`
- `numpy`

A typical installation is:

    pip install mne numpy

The script uses MNE’s generic raw-file reader, so the input file must be in a format that `mne.io.read_raw()` can open.

## Quick start

Basic use with default trigger naming:

    python cerca_opm_events.py sub-01_task-oddball_meg.fif

Explicit channel order:

    python cerca_opm_events.py raw.fif \
        --channels Trigger1[Z],Trigger2[Z],Trigger3[Z],Trigger4[Z],Trigger5[Z],Trigger6[Z],Trigger7[Z],Trigger8[Z]

Diagnostic review with the MNE browser:

    python cerca_opm_events.py raw.fif --diagnostic --save-composite-channel

More conservative decoding that requires bits to persist:

    python cerca_opm_events.py raw.fif \
        --value-strategy persist \
        --min-bit-high-samples 2

Apply a codebook to map numeric values to `trial_type` and additional columns:

    python cerca_opm_events.py raw.fif --codebook trigger_codebook.tsv

Delete known bad codes everywhere:

    python cerca_opm_events.py raw.fif --drop-values 255,0x7F

Delete specific rows from the diagnostic table:

    python cerca_opm_events.py raw.fif --drop-rows 2,5,17

## Input assumptions

### Trigger channels

By default, the script looks for channel names matching:

`^Trigger(\d+)\b`

That means it expects channels such as:

- `Trigger1`
- `Trigger2`
- `Trigger3`
- ...
- `Trigger8`

Names like `Trigger1[Z]` also work with the default pattern.

If your channels have different names, use either:

- `--channels` to provide the exact channel list in LSB-to-MSB order, or
- `--trigger-regex` to define a custom naming pattern.

Example:

    python cerca_opm_events.py raw.fif --trigger-regex '^Trig_(\d+)$'

### Bit order

When you use `--channels`, the order must be:

least significant bit to most significant bit

That means the first channel in the list is value 1, the second is value 2, the third is value 4, and so on.

### Number of bits

The default is 8 bits. To change that:

    python cerca_opm_events.py raw.fif --nbits 4

If some bits are missing and should be treated as always zero:

    python cerca_opm_events.py raw.fif --allow-missing

## How detection works

### 1. Thresholding

Each analog trigger channel is converted to a binary 0/1 line.

You can either:

- let the script estimate a threshold automatically, or
- provide a fixed threshold with `--threshold`.

Automatic thresholding uses a midpoint between estimated low and high levels. The threshold estimation method is controlled by `--threshold-mode`:

- `minmax`  
  uses the channel minimum and maximum;
- `extrema-median`  
  uses robust medians of the extreme tails and can be more stable if there are outliers.

### 2. Active polarity

By default, the script assumes a standard positive TTL signal:

`--active-when high`

If your TTL is inverted, use:

`--active-when low`

If you want the script to infer the likely active direction from the data:

`--active-when auto`

### 3. Inactive-channel suppression

When automatic thresholding is used, channels that do not show a convincing two-level separation are treated as inactive and forced to zero. This helps avoid false events on lines that never actually changed state.

The separation test is controlled by:

`--min-level-snr`

The default is `10.0`.

### 4. Candidate event regions

A candidate event begins whenever any trigger line becomes active. The script forms active regions from all samples where at least one bit is high.

Nearby activity can be merged into one pulse by allowing a short zero gap:

`--merge-gap-samples`

### 5. Minimum pulse length

Very short pulses can be rejected:

`--min-pulse-samples`

The default is `2`, which protects against one-sample blips.

If your true triggers are only one sample long, reduce this to `1`, but do so carefully.

### 6. Final event value

The onset is always the first active sample in the pulse.

The final numeric event code is then determined from the onset window, which includes:

- the onset sample, and
- `--lookahead-samples` additional samples

By default:

`--lookahead-samples 1`

That means the script checks the onset sample and the next sample, which is often the right choice when some TTL lines are captured one digitized frame later than others.

The final value can be chosen in three ways:

#### `--value-strategy or`  (default)

Bitwise OR across the non-zero packed words in the onset window.

This is usually the best default for staggered capture across adjacent samples.

#### `--value-strategy mode`

Uses the most frequent non-zero packed word across the full pulse.

This can be useful if the pulse stabilizes after onset and you want the most common value during the pulse.

#### `--value-strategy persist`

A bit is kept only if it is high in at least `--min-bit-high-samples` samples within the onset window.

Example:

    python cerca_opm_events.py raw.fif \
        --value-strategy persist \
        --min-bit-high-samples 2

This is useful when you want to ignore transient one-sample bits.

## Output files

By default, the script writes files in the same folder as the raw file.

### Final events file

The main output is:

`*_events.tsv`

If the input stem ends with `_meg`, `_eeg`, `_ieeg`, `_nirs`, or `_beh`, that suffix is replaced with `_events.tsv`.

Example:

- input: `sub-01_task-oddball_meg.fif`
- output: `sub-01_task-oddball_events.tsv`

If the input stem does not end with one of those suffixes, `_events.tsv` is appended to the stem.

You can override the output path with:

`--output /path/to/file_events.tsv`

### Diagnostic file

The script also writes a diagnostic table:

`*_events_diagnostics.tsv`

This file contains every detected candidate event, including events later dropped by the user.

### Columns in the final `_events.tsv`

The final file always contains:

- `onset`  
  seconds from the start of the file
- `duration`  
  always `0`
- `sample`  
  onset sample index, 0-based, relative to the start of the raw file
- `value`  
  packed integer trigger code
- `trial_type`  
  from the codebook if available, otherwise `trigger_<value>`

If a codebook contains extra columns, those columns are also copied into the final `_events.tsv`.

## Diagnostic review mode

Run with:

    python cerca_opm_events.py raw.fif --diagnostic

This does three things:

1. prints summary statistics and a diagnostic table in the terminal;
2. writes the diagnostic TSV file;
3. opens an MNE raw browser unless `--no-browser` is used.

### What you see in the browser

The browser shows the trigger channels. Estimated events are displayed as event markers, which appear as vertical lines.

If you add:

`--save-composite-channel`

the browser also includes a synthetic `COMPOSITE_TRIGGER` channel so you can inspect the packed integer word alongside the analog trigger lines.

### How to reject events in the browser

In the browser:

1. press `a` to enter annotation mode;
2. draw an annotation over a bad event;
3. make the annotation label start with `BAD`, for example `BAD_drop`;
4. close the browser window.

Any detected event whose onset falls inside a `BAD...` annotation is dropped.

The annotation does not need to cover the full pulse. It only needs to cover the event onset.

### Terminal-based review

If the script is running in an interactive terminal, diagnostic mode also offers a prompt after the browser closes.

You can then drop additional events:

- by row number from the diagnostic table, or
- by trigger value everywhere in the file

If the script is not running interactively, that prompt is skipped automatically.

### Browser troubleshooting

If the browser backend needs to be forced:

    python cerca_opm_events.py raw.fif --diagnostic --browser-backend matplotlib

or:

    python cerca_opm_events.py raw.fif --diagnostic --browser-backend qt

If you want diagnostic output but no GUI:

    python cerca_opm_events.py raw.fif --diagnostic --no-browser

If you need the script to wait until the browser window is closed on your platform:

    python cerca_opm_events.py raw.fif --diagnostic --browser-block

## Codebook support

A codebook is optional. It can be TSV or CSV.

The file must contain a column named:

`value`

Any additional columns are copied into the final `_events.tsv` for matching trigger values.

A simple codebook might look like this:

    value	trial_type	condition
    1	fixation	baseline
    5	face	famous
    13	target	oddball

Use it like this:

    python cerca_opm_events.py raw.fif --codebook trigger_codebook.tsv

Behavior:

- if a trigger value is found in the codebook, its metadata are copied into the output row;
- if a trigger value is not found, `trial_type` becomes `trigger_<value>`;
- extra codebook columns are left blank for unmatched values.

## Diagnostic table columns

The diagnostic TSV contains the following columns:

- `row`  
  1-based row number for review and dropping
- `keep`  
  `1` if the event is currently kept, `0` if dropped
- `drop_reason`  
  why the event was dropped
- `onset`  
  onset time in seconds
- `sample`  
  onset sample, 0-based
- `value`  
  final packed integer value
- `pulse_samples`  
  pulse length in samples
- `pulse_ms`  
  pulse length in milliseconds
- `onset_word`  
  packed value at the first active sample only
- `confirm_or_word`  
  OR-combined value across the onset window
- `mode_word`  
  most frequent non-zero packed value across the pulse
- `full_or_word`  
  OR-combined value across the full pulse
- `n_unique_nonzero_words`  
  number of distinct non-zero packed values seen during the pulse
- `flags`  
  warning flags about possible instability or partial capture

Rows are intentionally not renumbered after dropping, so diagnostic row numbers remain stable during review.

## Meaning of common flags

### `partial_or_staggered_onset`

The first active sample did not contain the full final value, but the look-ahead window recovered the missing bits.

This is often expected when different TTL lines are captured on adjacent samples.

### `unstable_pulse`

More than one non-zero packed value appeared during the pulse.

This does not necessarily mean the event is wrong, but it is a good reason to inspect it.

### `mode_mismatch`

The most common non-zero value during the pulse did not match the final selected value.

This is a sign that the pulse changed over time in a way worth reviewing.

### `late_or_transient_bits`

The OR across the full pulse did not match the final selected value.

This usually means extra bits appeared later in the pulse or only transiently.

### `weak_bits=...`

One or more bits in the final selected value appeared only once across the full pulse.

Those bits may be suspicious.

Important: the bit numbers in `weak_bits` are zero-based internal bit indices. Bit `0` means the first channel in the LSB-to-MSB order.

## Command-line reference

### Required argument

`raw_file`  
Path to the raw file to read.

### Output control

`--output`  
Path of the final `_events.tsv`. Default: same folder as input.

### Trigger channel selection

`--channels`  
Comma-separated trigger channels in LSB-to-MSB order.

`--trigger-regex`  
Regex used to detect trigger channels when `--channels` is not supplied. Must contain exactly one capture group for the 1-based bit number. Default: `^Trigger(\d+)\b`

`--nbits`  
Number of trigger bits. Default: `8`

`--allow-missing`  
Treat missing bit positions as always zero.

### Thresholding and polarity

`--threshold`  
Use the same fixed threshold on every trigger channel.

`--threshold-mode`  
Automatic threshold mode: `minmax` or `extrema-median`

`--min-level-snr`  
Minimum robust low/high separation needed before an automatically-thresholded channel is treated as active. Default: `10.0`

`--active-when`  
Choose `high`, `low`, or `auto`. Default: `high`

### Event confirmation

`--lookahead-samples`  
Number of additional samples after onset used for confirmation. Default: `1`

`--value-strategy`  
How to choose the final value: `or`, `mode`, or `persist`

`--min-bit-high-samples`  
Used with `persist`. A bit must be high in at least this many samples within the confirmation window. Default: `1`

`--min-pulse-samples`  
Minimum pulse length to keep. Default: `2`

`--merge-gap-samples`  
Merge pulses separated by this many zero samples or fewer. Default: `0`

### Automatic dropping

`--drop-values`  
Comma-separated trigger values to remove after detection. Decimal and hex are both accepted, for example `255,0x7F`

`--drop-rows`  
Comma-separated diagnostic row numbers to remove after detection

### Metadata

`--codebook`  
Optional TSV or CSV file with a required `value` column

### Diagnostics and review

`--diagnostic`  
Enable diagnostic summary, diagnostic file writing, and review workflow

`--no-browser`  
In diagnostic mode, skip opening the MNE browser

`--browser-backend`  
Browser backend to request: `auto`, `qt`, or `matplotlib`

`--browser-block`  
Ask the browser to block until it closes

`--save-composite-channel`  
Add a synthetic `COMPOSITE_TRIGGER` channel to the browser

### Other

`--verbose`  
Present in the command-line interface, but in this version it does not materially change the printed output

## Common workflows

### Standard extraction

Use this when your channels are named `Trigger1` through `Trigger8` and the default onset-plus-next-sample confirmation is appropriate.

    python cerca_opm_events.py sub-01_task-oddball_meg.fif

### Explicit channel order

Use this when the trigger channels are named differently or do not sort naturally.

    python cerca_opm_events.py raw.fif \
        --channels Trigger1[Z],Trigger2[Z],Trigger3[Z],Trigger4[Z],Trigger5[Z],Trigger6[Z],Trigger7[Z],Trigger8[Z]

### Inverted TTL lines

Use this when the active TTL state is low rather than high.

    python cerca_opm_events.py raw.fif --active-when low

### More robust outlier handling

Use this if min/max values are distorted by outliers.

    python cerca_opm_events.py raw.fif --threshold-mode extrema-median

### Require persistence

Use this when transient bits are a problem.

    python cerca_opm_events.py raw.fif \
        --value-strategy persist \
        --min-bit-high-samples 2

### Review suspicious datasets interactively

    python cerca_opm_events.py raw.fif --diagnostic --save-composite-channel

### Headless diagnostic run

    python cerca_opm_events.py raw.fif --diagnostic --no-browser

## Troubleshooting

### “No trigger channels were found”

Your channel names do not match the default regex.

Fix by supplying either:

- `--channels`
- or `--trigger-regex`

### Too many false events

Try one or more of the following:

- increase `--min-pulse-samples`
- use `--value-strategy persist --min-bit-high-samples 2`
- increase `--min-level-snr`
- inspect the channels in `--diagnostic` mode
- use a carefully chosen fixed `--threshold` if you know the analog levels well

### Expected bits are missing from some events

Try:

- increasing `--lookahead-samples`
- keeping `--value-strategy or`
- checking whether the TTL polarity should be `low` rather than `high`

### The browser does not open correctly

Try:

- `--browser-backend matplotlib`
- `--browser-backend qt`
- `--no-browser` if you only need non-GUI diagnostics

### The file is large

The script loads the raw file into memory. Make sure enough RAM is available for the full recording.

## Notes and limitations

- The script is designed for event onset timing, not pulse-duration measurement.
- The final BIDS `duration` is always written as `0`.
- The `sample` column is 0-based relative to the start of the loaded raw file.
- The browser shows estimated triggers as event markers, not as per-event annotations.
- User-drawn `BAD...` annotations are used only to reject detected events whose onsets fall inside those spans.

## Recommended default practice

For most Cerca OPM recordings, a good starting command is:

    python cerca_opm_events.py your_file.fif --diagnostic --save-composite-channel

Then:

1. inspect the trigger lines and event markers;
2. add `BAD...` annotations to any clearly incorrect events;
3. optionally drop rows or values in the terminal prompt;
4. keep the generated `_events.tsv` and `_events_diagnostics.tsv` together for provenance.

## Summary

`cerca_opm_events.py` provides a reliable way to turn separate analog trigger lines into packed event codes, while guarding against staggered bit capture, inactive channels, and short noise pulses. It is suitable for writing BIDS-style `_events.tsv` files from Cerca OPM data and for reviewing the results interactively before finalizing the output.