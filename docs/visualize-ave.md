# visualize_ave.py — User Documentation

## Purpose

`visualize_ave.py` is an interactive command-line tool for exploring evoked MEG/EEG data stored in MNE-Python's `-ave.fif` format. It is designed for researchers working in BIDS-organised datasets who want to quickly inspect averaged waveforms, topographic maps, global field power, and sensor-level activity across one or more experimental conditions — without writing any analysis code.

The program loads a single `-ave.fif` file containing one or more averaged conditions, then presents a menu of visualisation modes. You work interactively: choose a view, adjust what you see (axes, conditions, sensors), and return to the menu when done. All plots are non-blocking, meaning the terminal remains active while plot windows are open.

---

## Requirements

The following Python packages must be installed:

- `mne` — MEG/EEG analysis library (core dependency)
- `numpy` — numerical arrays
- `matplotlib` — plotting
- `ruamel.yaml` or `PyYAML` — YAML config file support (only needed if you launch with a config file rather than a `.fif` path directly)

Install with pip if needed:

```
pip install mne numpy matplotlib ruamel.yaml
```

The script also prints the active Matplotlib backend at startup. For interactive plots to display correctly, you generally need a GUI-capable backend such as `TkAgg`, `Qt5Agg`, or `MacOSX`. If plots do not appear, set the backend explicitly before running:

```
MPLBACKEND=TkAgg python visualize_ave.py my_data-ave.fif
```

---

## Running the Program

```
python visualize_ave.py <input>
```

`<input>` is either:

**A path to an `-ave.fif` file directly:**

```
python visualize_ave.py /path/to/sub-01_task-oddball_ave-ave.fif
```

**A YAML config file** (if your data lives inside a BIDS derivatives tree):

```
python visualize_ave.py config.yaml
```

When a YAML config is used, the program constructs the expected file path automatically from the BIDS structure:

```
<bids_root>/derivatives/preprocessing/sub-<subject>/ses-<session>/meg/
    sub-<subject>_ses-<session>_task-<task>_ave-ave.fif
```

A minimal YAML config looks like this:

```yaml
bids_root: /data/myproject
subject: "01"
session: "01"
task: oddball
```

The `session` and `task` fields are optional; omit them if your filenames do not include those BIDS entities.

---

## Startup Output

After loading, the program prints a summary of every condition in the file:

```
Loaded 3 evoked conditions:
  (1) standard (nave=412)
  (2) deviant  (nave=88)
  (3) button   (nave=54)
```

`nave` is the number of epochs that were averaged together. Keep these condition numbers in mind — you will use them throughout the menus.

---

## Main Menu

```
Menu:
  (1) Plot Butterfly/Topomap
  (2) Plot Waveforms
  (3) Overlay Sensors
  (4) Interactive Sensor Layout
  (5) GFP Browser
  (6) Sensor Layout Reference
  (7) Regional Grid Plot (Stable Matplotlib)
  (b) Apply baseline correction
  (f) Apply low-pass filter
  (c) Close all open plots
  (q) Quit
```

Type the letter or number and press Enter. Each option is described in detail below.

---

## Global Commands

At any prompt throughout the program, two commands always work regardless of context:

- **`c`** — Close all currently open plot windows and continue. Use this to declutter your screen without leaving the current menu or browser.
- **`q`** — Quit the current sub-menu or browser and return to the main menu (or exit the program if typed at the main menu).

---

## Condition Selection Syntax

Several views ask you to select conditions. There are two modes:

### Overlay mode

Type one or more condition numbers separated by commas. Each condition is plotted as a separate coloured line on the same axes.

```
Enter condition numbers: 1,2
```

This overlays conditions 1 and 2.

### Difference mode

Type a positive condition number and a negative condition number: `A,-B`. The program computes condition A minus condition B and plots the result as a single waveform labelled `"A label - B label"`.

```
Enter condition numbers: 1,-2
```

This plots condition 1 minus condition 2. Exactly two values are required in this format; mixed selections such as `1,2,-3` are not supported.

### Overlay all

In views that support it, typing `all` selects every condition at once.

---

## Option 1 — Butterfly / Topomap

Produces two complementary views for a single condition:

- A **butterfly plot** — all channels of each type drawn on the same time axis, with spatial colouring. This gives an immediate sense of the overall response shape and any outlier channels.
- One or more **topomap panels** — scalp maps of the field distribution at user-specified time points.

### Steps

1. You are first asked for the time range and step size for the topomaps (in milliseconds):

   ```
   Enter topomap start,end,step in ms (e.g. 10,100,10)
     within [-200 ms, 600 ms].
     [default: -200,600,100]
     Press Enter for defaults, 'c' to close plots, 'q' to quit:
   ```

   Press Enter to accept the suggested defaults, or type three comma-separated values. The step determines how many topomap panels are produced; for example, `0,300,50` produces panels at 0, 50, 100, 150, 200, 250, and 300 ms.

2. You are then shown the available conditions and asked to select one by number. A butterfly plot and topomap figure(s) appear. Separate topomap figures are created for each channel type present (magnetometers, gradiometers, EEG).

3. From within this sub-menu you can:
   - Type another condition number to plot a different condition.
   - Type `r` or `reset` to re-enter the topomap time range.
   - Type `c` to close all plots.
   - Type `q` to return to the main menu.

---

## Option 2 — Waveform Browser

Displays the time-series waveform for one sensor at a time, advancing through the sensor list interactively. This is the core exploration tool for examining individual channel responses.

### Steps

1. Select conditions to display (overlay or difference syntax, see above).

2. A plot opens for the first sensor. The window title and plot title both show the current sensor name and channel type. Horizontal and vertical reference lines mark zero amplitude and stimulus onset.

3. Navigate with the following commands at the `Choice:` prompt:

   | Key | Action |
   |-----|--------|
   | Enter (or `n`) | Advance to the next sensor |
   | `p` | Go back to the previous sensor |
   | Number (e.g. `47`) | Jump to sensor number 47 in the channel list |
   | Sensor name (e.g. `MEG0111`) | Jump directly to that sensor by name |
   | Last 3 digits (e.g. `111`) | Jump to the MEG sensor whose name ends in those digits |
   | `s` | Re-scale the Y axis for the current channel type |
   | `a` | Set both X and Y axis limits manually |
   | `d` | Change the selected conditions without leaving the browser |
   | `c` | Close all plots |
   | `q` | Return to the main menu |

### Axis scaling

**`s` (scale)** prompts for new Y-axis limits for the current channel type only:

```
Y axis for MAG in fT (min,max) [default: -200,200]:
```

Press Enter to keep the current value. The new scale applies to all subsequent magnetometer plots until changed again. Gradiometer and EEG scales are independent.

**`a` (axis)** prompts for both X and Y limits together:

```
Enter X limits in ms (start,end) [default -200,600]:
Enter Y limits in fT (min,max) [default -156.3,204.1]:
```

Press Enter for either prompt to keep the current value.

### Default units and scale

| Channel type | Unit | Typical range |
|---|---|---|
| Magnetometers | fT | −200 to 200 |
| Gradiometers | fT/cm | −100 to 100 |
| EEG | μV | −20 to 20 |

Non-MEG/EEG channels (e.g. stimulus channels) are silently skipped.

---

## Option 3 — Overlay Sensors

Plots multiple sensors from a single condition on the same axes. Useful for comparing a small set of specific channels, or for checking whether two nearby sensors agree.

### Steps

1. Select a single condition by number.

2. Enter the channels to overlay, separated by commas. You may use any of the following formats:

   ```
   MEG0111, MEG0121, MEG0131
   ```
   ```
   0111, 0121, 0131
   ```
   (Last-3-digit shorthand for MEG sensors)
   ```
   1, 5, 12
   ```
   (1-based index into the full channel list)

   Mixed formats in a single entry are accepted.

3. A single plot opens with all selected channels drawn together. Each line is labelled with its sensor name and type.

4. From the `Options:` prompt:

   | Key | Action |
   |-----|--------|
   | Any channel input | Replace the current set of sensors |
   | `s` | Re-scale the Y axis |
   | `d` | Switch to a different condition |
   | `c` | Close all plots |
   | `q` | Return to the main menu |

Note: if you select sensors of mixed types (e.g. magnetometers and gradiometers together), the Y axis uses the broadest range across all types. For a cleaner view, stick to one channel type at a time.

---

## Option 4 — Interactive Sensor Layout

Renders a scalp-layout plot in which every sensor position appears as a small waveform thumbnail. Clicking on a thumbnail in the MNE figure opens a larger version of that sensor's waveform (this is MNE's built-in interactivity, not part of this script).

### Steps

1. Select one or more conditions using the standard syntax (overlay, difference, or `all`).

2. If a single condition is selected, `plot_topo` is called for that condition alone. If multiple conditions are selected, all waveforms are overlaid within each thumbnail using MNE's `plot_evoked_topo`.

This view is particularly useful for locating the scalp distribution of a response at a glance and for identifying suspicious channels. Note that MNE's interactive topo plot can sometimes be slow to render for very dense sensor arrays.

---

## Option 5 — GFP Browser

Plots the **Global Field Power** (GFP) — the standard deviation across all sensors at each time point — as a function of time. GFP peaks correspond to moments of maximum neural synchrony across the scalp and provide a condition-level summary that does not depend on sensor choice.

### Steps

1. Select one or more conditions by number (comma-separated). Unlike the waveform browser, difference mode is not available here.

2. GFP is computed and displayed. The browser cycles through different channel-type views:

   - **all** — GFP computed across all channel types together (mixed units)
   - **mag** — magnetometers only (fT)
   - **grad** — gradiometers only (fT/cm)
   - **eeg** — EEG only (μV)

   Only channel types present in your data appear.

3. Navigation:

   | Key | Action |
   |-----|--------|
   | Enter (or `n`) | Next channel type |
   | `p` | Previous channel type |
   | `s` | Re-scale Y axis for the current type |
   | `a` | Set both X and Y axis limits |
   | `d` | Change selected conditions |
   | `c` | Close all plots |
   | `q` | Return to the main menu |

---

## Option 6 — Sensor Layout Reference

Displays a static diagram of sensor positions on the scalp, labelled with sensor names. This is a reference tool — use it when you want to identify which physical location corresponds to a sensor name before navigating to it in another browser.

### Steps

1. Choose which channel type to display:

   ```
   Display options: (1) All (2) Mag (3) Grad (4) EEG (q) Quit
   ```

2. A labelled sensor map opens. After viewing, press Enter in the terminal to return to the main menu.

The program also prints a sensor naming guide:

```
SENSOR NAMING CONVENTION:
  MEG####:
    - First digit (0-2): Region
      0 = Frontal/Central
      1 = Left hemisphere
      2 = Right hemisphere
    - Last digit:
      1 = Magnetometer
      2,3 = Gradiometer pair
```

So `MEG0111` is a frontal/central magnetometer; `MEG2132` is a right-hemisphere latitudinal gradiometer; `MEG1423` is a left-hemisphere longitudinal gradiometer, and so on.

**Tip:** In the Waveform Browser and Overlay Sensors view, you can jump to any MEG sensor by typing only its last three digits (e.g. `111` for `MEG0111`). Use the layout reference to find the digits for the region you care about.

---

## Option 7 — Regional Grid Plot

Plots all sensors in a given anatomical region as a grid of individual waveforms, each sharing the same X and Y axes. This is useful for seeing regional response consistency and for identifying outlier sensors within a region.

### Steps

1. Select a region:

   ```
   (1) Left-frontal
   (2) Right-frontal
   (3) Left-parietal
   (4) Right-parietal
   (5) Left-temporal
   (6) Right-temporal
   (7) Left-occipital
   (8) Right-occipital
   (9) Vertex
   ```

2. Select a sensor type within that region:

   ```
   (1) Magnetometers (ends in '1')
   (2) Gradiometers - Latitudinal (ends in '2')
   (3) Gradiometers - Longitudinal (ends in '3')
   ```

3. Select conditions to overlay or difference using the standard syntax.

4. You are prompted for X (time) and Y (amplitude) axis limits. Defaults are the full epoch and the global min/max across all selected conditions with a 5% margin.

5. A grid figure opens with one panel per sensor in the region. Conditions are colour-coded (blue, red, green, ...). The legend appears in the first panel only. Horizontal and vertical zero lines appear in each panel.

This view uses plain Matplotlib and is the most stable across environments (hence "Stable Matplotlib" in the menu label).

---

## Option b — Apply Baseline Correction

Applies a mean-baseline subtraction to all conditions in memory. The mean amplitude during the specified pre-stimulus window is subtracted from every time point.

```
Enter baseline period in ms (start,end) [e.g., -200,0]:
```

A typical entry for a 200 ms pre-stimulus baseline is `-200,0`. The correction is applied to all conditions simultaneously and persists for all subsequent plots in the current session. This operation modifies the data in memory; it cannot be undone without reloading the file.

---

## Option f — Apply Low-Pass Filter

Applies a zero-phase low-pass filter to all conditions in memory. You are prompted for the cutoff frequency in Hz:

```
Enter low-pass cutoff in Hz (e.g., 40):
```

Enter a single number such as `40` to attenuate frequencies above 40 Hz. As with baseline correction, this modifies all conditions in memory and cannot be undone in the current session. Apply filtering before baseline correction if both are needed.

---

## Tips for a New User

**Start here:** Run option 6 (Sensor Layout Reference) first to orient yourself to the sensor names and regions relevant to your experiment.

**Quick overview of a condition:** Run option 1 (Butterfly/Topomap) to get a high-level view of the evoked response — you will see the overall waveform shape and scalp distribution in one step.

**Investigate a specific component:** Use option 7 (Regional Grid Plot) to look at all sensors in a region at once, then use option 2 (Waveform Browser) to drill into individual sensors.

**Compare conditions:** In any browser (options 2, 5, 7), enter both condition numbers (e.g. `1,2`) to see them overlaid, or use `1,-2` to plot their difference directly.

**Preprocessing on the fly:** If your file has not yet been baseline-corrected or high-frequency noise is visible, apply option `b` and/or option `f` before exploring waveforms.

**Managing windows:** Multiple plot windows accumulate as you work. Type `c` at any prompt to close them all and start fresh.