# visualize_evoked_BIDS.py
import sys
from pathlib import Path
import subprocess
import yaml
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mne

DEFAULT_YLIMS = {
    'mag': (-200, 200),     # fT
    'grad': (-100, 100),    # fT/cm
    'eeg': (-20, 20),       # μV
}
UNITS = {'mag': 'fT', 'grad': 'fT/cm', 'eeg': 'μV'}
SCALE = {'mag': 1e15, 'grad': 1e13, 'eeg': 1e6}

def get_channel_type(info, idx):
    return mne.channel_type(info, idx)

def prompt_for_channels(sensor_names):
    """
    Accept full names (any type), last 3 digits for MEG, or 1-based index. Returns list of indices.
    """
    sensor_lookup = {name.lower(): i for i, name in enumerate(sensor_names)}
    suffix_lookup = {name[-3:]: i for i, name in enumerate(sensor_names) if name.startswith('MEG')}
    while True:
        sensors_in = input("Type channel names, EEG names, or last three digits (comma-separated), or 'quit': ").strip()
        if sensors_in.lower() == 'quit':
            return None
        requested = [s.strip() for s in sensors_in.split(',')]
        indices = []
        for req in requested:
            if req.isdigit():
                n = int(req)
                if 1 <= n <= len(sensor_names):
                    indices.append(n - 1)
                elif req in suffix_lookup:
                    indices.append(suffix_lookup[req])
            elif req.lower() in sensor_lookup:
                indices.append(sensor_lookup[req.lower()])
            elif req in suffix_lookup:
                indices.append(suffix_lookup[req])
        if indices:
            return indices
        print("No valid channels entered. Try again.")

def prompt_for_ylims(channel_types, current_ylims):
    """
    Prompt for Y-limits per channel type. Returns updated dict.
    """
    ylims = current_ylims.copy()
    for ch_type in sorted(channel_types):
        default = ylims[ch_type]
        units = UNITS[ch_type]
        entry = input(
            f"Y axis for {ch_type.upper()} in {units} (min,max) [default: {default[0]},{default[1]}]: "
        ).strip()
        if entry:
            try:
                ymin, ymax = [float(x) for x in entry.split(',')]
                ylims[ch_type] = (ymin, ymax)
            except Exception:
                print("Invalid input, using default.")
    return ylims

def focus_terminal():
    """Return focus to Terminal.app on macOS, no-op elsewhere."""
    if sys.platform == "darwin":
        script = 'tell application "Terminal" to activate'
        try:
            subprocess.run(['osascript', '-e', script], check=True)
        except Exception as e:
            print(f"Warning: Could not bring Terminal to front: {e}")

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_evoked_fname(config):
    bids_root = config['bids_root']
    subject   = config['subject']
    session   = config.get('session', None)
    task      = config.get('task', None)
    deriv_root = Path(bids_root) / "derivatives" / "preprocessing"
    base = f"sub-{subject}"
    if session:
        base += f"_ses-{session}"
    if task:
        base += f"_task-{task}"
    evoked_fname = deriv_root / f"sub-{subject}/ses-{session}/meg/{base}_ave-ave.fif"
    return evoked_fname

def round_ms(times):
    return np.round(np.array(times) * 1000).astype(int) / 1000

def get_channel_indices(ev, ch_type):
    """Return indices and names of all channels of the specified type."""
    return [(i, ch) for i, (ch, typ) in enumerate(zip(ev.ch_names, ev.get_channel_types())) if typ == ch_type]

def get_data_minmax(evokeds, ch_type):
    """Return global min/max for the given ch_type across all sensors and all evokeds."""
    vals = []
    for ev in evokeds:
        picks = mne.pick_types(ev.info, meg=ch_type if ch_type in ('mag', 'grad') else False,
                               eeg=(ch_type == 'eeg'), exclude=[])
        if picks.size == 0:
            continue
        vals.append(ev.data[picks, :])
    if not vals:
        return None, None
    all_vals = np.concatenate(vals, axis=0)
    return np.nanmin(all_vals), np.nanmax(all_vals)

def get_signal_minmax_in_window(ev, ch_indices, t_min=0.01):
    """Return min/max across given channels and all time points with t >= t_min."""
    t = ev.times
    idx = np.where(t >= t_min)[0]
    if not len(idx):
        idx = np.arange(len(t))  # fallback to all times
    data = ev.data[ch_indices, :][:, idx]
    return np.nanmin(data), np.nanmax(data)

def get_default_axis_limits(evokeds, ch_type, ch_indices, cond_indices):
    """
    Returns default xlim (entire epoch) and ylim (min/max of all selected signals in window t >= 0.01s).
    ch_indices: list of channel indices (for multi-sensor: all, for single-sensor: [idx])
    cond_indices: list of condition indices to plot
    """
    # Default X-axis: full epoch
    xlim = [evokeds[0].times[0], evokeds[0].times[-1]]

    # Default Y-axis: min/max across conditions/channels in window t >= 0.01s
    min_val, max_val = None, None
    for cond_i in cond_indices:
        ev = evokeds[cond_i]
        mn, mx = get_signal_minmax_in_window(ev, ch_indices)
        min_val = mn if min_val is None else min(min_val, mn)
        max_val = mx if max_val is None else max(max_val, mx)
    # Pad the Y limits slightly for visual margin
    pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0
    ylim = [min_val - pad, max_val + pad]
    return xlim, ylim

def prompt_for_conditions(evokeds):
    while True:
        cond_str = input("Enter condition numbers to plot (comma-separated, e.g., 1,2) or 'quit': ").strip()
        if cond_str.lower() == 'quit':
            return None
        try:
            cond_indices = [int(c)-1 for c in cond_str.split(',') if c]
            if all(0 <= i < len(evokeds) for i in cond_indices):
                return cond_indices
        except Exception:
            pass
        print("Invalid selection. Please enter valid condition numbers.")

def prompt_for_sensor_index(sensor_names):
    while True:
        user = input(f"Type sensor name/number or 'quit' to exit: ").strip()
        if user.lower() == 'quit':
            return None
        if user.isdigit() and 1 <= int(user) <= len(sensor_names):
            return int(user) - 1
        for i, name in enumerate(sensor_names):
            if user.lower() == name.lower():
                return i
        print(f"Unknown sensor '{user}'. Try again.")

def prompt_for_axis_limits(default_xlim, default_ylim, units):
    """Prompt for axis limits, using ms-precision for X and correct physical units for Y."""
    # Helper for rounding to the nearest ms
    def round_ms(val):
        return round(val * 1000) / 1000

    rounded_xlim = [round_ms(default_xlim[0]), round_ms(default_xlim[1])]
    rounded_ylim = [round(default_ylim[0], 2), round(default_ylim[1], 2)]

    while True:
        try:
            xlim_in = input(
                f"Enter X axis limits as start,end in seconds (default {rounded_xlim[0]},{rounded_xlim[1]}): "
            ).strip()
            if xlim_in:
                xstart, xend = [float(x) for x in xlim_in.split(',')]
                xlim = [round_ms(xstart), round_ms(xend)]
            else:
                xlim = rounded_xlim

            ylim_in = input(
                f"Enter Y axis limits in {units} as min,max (default {rounded_ylim[0]},{rounded_ylim[1]}): "
            ).strip()
            if ylim_in:
                ymin, ymax = [float(y) for y in ylim_in.split(',')]
                ylim = [ymin, ymax]
            else:
                ylim = rounded_ylim

            return xlim, ylim
        except Exception as e:
            print(f"Invalid input: {e}. Please use start,end for X (s, ms precision) and Y ({units})")

def plot_waveforms(evokeds, ch_type, ch_idx, cond_indices, xlim, ylim, colors, sensor_names, times):
    """Plot waveforms from multiple evokeds for the same channel."""
    plt.figure(figsize=(7, 5))
    for cond_i, color in zip(cond_indices, colors):
        ev = evokeds[cond_i]
        picks = get_channel_indices(ev, ch_type)
        if not picks:
            continue
        idx, ch_name = picks[ch_idx]
        plt.plot(times, ev.data[idx, :], color=color, lw=2, label=f"{ev.comment}")
    plt.xlabel('Time (s)')
    plt.ylabel(f'Amplitude [{ch_type}]')
    plt.title(f"{sensor_names[ch_idx]} ({ch_type})")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.show(block=False)
    focus_terminal()  # <-- Added

def get_channel_types(evokeds):
    ch_types = set()
    for ev in evokeds:
        ch_types.update(ev.get_channel_types())
    return [t for t in ('mag', 'grad', 'eeg') if t in ch_types]

def get_data_minmax(evokeds, ch_type, scale):
    vals = []
    for ev in evokeds:
        picks = get_channel_indices(ev, ch_type)
        idxs = [i for i, _ in picks]
        if not idxs:
            continue
        vals.append(np.concatenate([ev.data[i, :] for i in idxs]))
    if not vals:
        return None, None
    all_vals = np.concatenate(vals)
    return np.nanmin(all_vals) * scale, np.nanmax(all_vals) * scale

def get_sensor_names(ev, ch_type):
    picks = get_channel_indices(ev, ch_type)
    return [name for idx, name in picks]

def prompt_for_channel_type(ch_types):
    while True:
        ch_type = input(f"Select channel type {ch_types}: ").strip().lower()
        if ch_type in ch_types:
            return ch_type
        print(f"Please type one of {ch_types}.")

def prompt_for_conditions(evokeds):
    while True:
        cond_str = input("Enter condition numbers to plot (comma-separated, e.g., 1,2): ").strip()
        try:
            cond_indices = [int(c)-1 for c in cond_str.split(',') if c]
            if all(0 <= i < len(evokeds) for i in cond_indices):
                return cond_indices
        except Exception:
            pass
        print("Invalid selection. Please enter valid condition numbers.")

def plot_single_sensor_multi_condition(evokeds, ch_type, idx, cond_indices, xlim, ylim, colors, sensor_names, times, scale, units):
    plt.figure(figsize=(7, 5))
    for cond_i, color in zip(cond_indices, colors):
        ev = evokeds[cond_i]
        picks = get_channel_indices(ev, ch_type)
        if not picks:
            continue
        ch_i, ch_name = picks[idx]
        plt.plot(ev.times, ev.data[ch_i, :] * scale, color=color, lw=2, label=f"{ev.comment}")
    plt.xlabel('Time (s)')
    plt.ylabel(f'Amplitude [{units}]')
    plt.title(f"{sensor_names[idx]} ({ch_type})")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.show(block=False)
    focus_terminal()  # <-- Added

def navigate_sensors(sensor_names, idx, user_input):
    """
    Accepts l/r/u/d, number (1-based), full name, or last 3 digits.
    """
    sensor_lookup = {name.lower(): i for i, name in enumerate(sensor_names)}
    suffix_lookup = {name[-3:]: i for i, name in enumerate(sensor_names) if name.startswith('MEG')}
    nav = user_input.strip().lower()
    if nav in ('exit', 'quit'):
        return None
    elif nav in ('right', 'r', ''):
        return (idx + 1) % len(sensor_names)
    elif nav in ('left', 'l'):
        return (idx - 1) % len(sensor_names)
    elif nav in ('up', 'u'):
        return (idx - 1) % len(sensor_names)
    elif nav in ('down', 'd'):
        return (idx + 1) % len(sensor_names)
    elif nav.isdigit():
        n = int(nav)
        if 1 <= n <= len(sensor_names):
            return n - 1  # 1-based index
        # Also try as last 3 digits
        if nav in suffix_lookup:
            return suffix_lookup[nav]
    elif nav in sensor_lookup:
        return sensor_lookup[nav]
    else:
        # Try last 3 digits with leading zeros
        nav3 = nav.zfill(3)
        if nav3 in suffix_lookup:
            return suffix_lookup[nav3]
        # Try partial matches (optional)
        for i, name in enumerate(sensor_names):
            if nav in name.lower():
                return i
        print("Use l/r/u/d, a number, a sensor name, last three digits, or 'exit' to quit.")
        return idx

def plot_magnetometer_matrix(info, n_rows=8, n_cols=16, label_style='short'):
    """
    Display magnetometer sensors in a clean 2D matrix:
    - Columns = Anterior–Posterior (A–P)
    - Rows   = Left–Right (L–R)
    - label_style: 'short' = last 3 digits, 'index' = 1-based, 'full' = full name
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Pick magnetometers
    picks = mne.pick_types(info, meg='mag')
    coords = np.array([info['chs'][i]['loc'][:2] for i in picks])
    sensor_names = [info['ch_names'][i] for i in picks]

    def short_label(name):
        return name[-3:] if len(name) >= 3 else name

    def label_for(i, name):
        if label_style == 'short':
            return short_label(name)
        elif label_style == 'index':
            return str(i+1)
        else:
            return name

    x = coords[:, 0]
    y = coords[:, 1]
    x_bins = np.linspace(x.min(), x.max(), n_cols+1)
    y_bins = np.linspace(y.min(), y.max(), n_rows+1)
    grid = np.full((n_rows, n_cols), '', dtype=object)

    for i, (xi, yi) in enumerate(zip(x, y)):
        col = np.digitize(xi, x_bins) - 1  # L–R
        row = np.digitize(yi, y_bins) - 1  # A–P
        row = np.clip(row, 0, n_rows-1)
        col = np.clip(col, 0, n_cols-1)
        lbl = label_for(i, sensor_names[i])
        if grid[row, col] == '':
            grid[row, col] = lbl
        else:
            grid[row, col] += f",{lbl}"

    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    ax.set_xlim(-0.5, n_cols-0.5)
    ax.set_ylim(-0.5, n_rows-0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # No axis labels for a clean grid

    for row in range(n_rows):
        for col in range(n_cols):
            txt = grid[row, col]
            if txt != '':
                ax.text(col, row, txt, ha='center', va='center', fontsize=10, color='red')

    ax.set_title("Magnetometer Matrix (A–P columns × L–R rows)")
    ax.invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def waveform_browser(evokeds):
    import matplotlib.pyplot as plt

    sensor_names = evokeds[0].info['ch_names']
    n_sensors = len(sensor_names)
    info = evokeds[0].info
    print(f"{n_sensors} channels available. (mag, grad, eeg, etc)")

    print("Available conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment}")

    # Prompt for initial channel to browse
    indices = prompt_for_channels(sensor_names)
    if not indices:
        print("Exiting waveform browser.")
        return
    idx = indices[0]

    # Prompt for initial conditions ONCE
    cond_indices = None
    while cond_indices is None:
        cond_in = input("Enter condition numbers to overlay (comma-separated, e.g., 1,2): ").strip()
        try:
            cond_indices = [int(c)-1 for c in cond_in.split(',') if c]
            if not all(0 <= i < len(evokeds) for i in cond_indices):
                print("Invalid selection."); cond_indices = None
        except Exception:
            print("Invalid input."); cond_indices = None

    # Y-limits (per type) setup
    current_ylims = DEFAULT_YLIMS.copy()

    def get_type_indices(current_type):
        """Get all indices of channels of the same type as current_type."""
        return [i for i, name in enumerate(sensor_names) if get_channel_type(info, i) == current_type]

    current_type = get_channel_type(info, idx)
    type_indices = get_type_indices(current_type)
    type_pos = type_indices.index(idx)

    while True:
        # Plot
        ch_type = get_channel_type(info, idx)
        plt.figure(figsize=(9, 5))
        colors = plt.get_cmap('tab10').colors
        for color, cond_idx in zip(colors, cond_indices):
            ev = evokeds[cond_idx]
            plt.plot(
                ev.times, ev.data[idx, :] * SCALE[ch_type],
                color=color, lw=2, label=f"{sensor_names[idx]} ({ch_type}) - {ev.comment}"
            )
        plt.xlabel('Time (s)')
        plt.ylabel(f"Amplitude [{UNITS[ch_type]}]")
        plt.legend()
        y_min, y_max = current_ylims[ch_type]
        plt.ylim((y_min, y_max))
        plt.grid(True)
        plt.show(block=False)

        print(f"\nCurrently plotting channel {idx+1}/{len(sensor_names)}: {sensor_names[idx]} ({ch_type})")
        print("Type l/r/u/d (prev/next within this type), a number, full channel name, last three digits,")
        print("'scale' to adjust y-limits, 'cond' to change overlayed conditions,")
        print("'close' to close plots, or 'quit': ")
        nav = input().strip().lower()

        if nav in ('quit', 'exit'):
            break
        elif nav in ('close', 'closeall', 'c'):
            plt.close('all')
            print("All plot windows closed.")
            continue
        elif nav == 'scale':
            current_ylims = prompt_for_ylims([ch_type], current_ylims)
            continue
        elif nav == 'cond':
            # Prompt for new conditions
            while True:
                cond_in = input("Enter new condition numbers to overlay (comma-separated, e.g., 1,2): ").strip()
                try:
                    new_cond_indices = [int(c)-1 for c in cond_in.split(',') if c]
                    if all(0 <= i < len(evokeds) for i in new_cond_indices):
                        cond_indices = new_cond_indices
                        break
                except Exception:
                    pass
                print("Invalid input. Please enter valid condition numbers.")
            continue
        else:
            # Navigation within type
            next_idx = None
            # Navigation logic
            if nav in ('right', 'r', ''):
                type_pos = (type_pos + 1) % len(type_indices)
                next_idx = type_indices[type_pos]
            elif nav in ('left', 'l'):
                type_pos = (type_pos - 1) % len(type_indices)
                next_idx = type_indices[type_pos]
            elif nav.isdigit():
                # Check if this digit matches a position in type_indices
                n = int(nav)
                if 1 <= n <= len(type_indices):
                    type_pos = n - 1
                    next_idx = type_indices[type_pos]
                else:
                    # Also try as last 3 digits
                    nav3 = nav.zfill(3)
                    for i, idx_i in enumerate(type_indices):
                        if sensor_names[idx_i][-3:] == nav3:
                            type_pos = i
                            next_idx = type_indices[type_pos]
                            break
            elif nav in sensor_names:
                target_idx = sensor_names.index(nav)
                if get_channel_type(info, target_idx) == current_type:
                    type_pos = type_indices.index(target_idx)
                    next_idx = target_idx
                else:
                    # Switch type: start new navigation list
                    idx = target_idx
                    current_type = get_channel_type(info, idx)
                    type_indices = get_type_indices(current_type)
                    type_pos = type_indices.index(idx)
                    continue
            elif len(nav) == 3 and any(sensor_names[i][-3:] == nav for i in type_indices):
                for i, idx_i in enumerate(type_indices):
                    if sensor_names[idx_i][-3:] == nav:
                        type_pos = i
                        next_idx = type_indices[type_pos]
                        break
            if next_idx is not None:
                idx = next_idx
            else:
                print(f"Use l/r/u/d, a number (within this type), sensor name, or last 3 digits within {ch_type.upper()} channels.")

    print("Waveform browsing complete.")

def multi_sensor_overlay(evokeds):
    sensor_names = evokeds[0].info['ch_names']
    n_sensors = len(sensor_names)
    info = evokeds[0].info
    print(f"{n_sensors} channels available. (mag, grad, eeg, etc)")
    print("Available conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment}")

    # Prompt ONCE for condition
    cond_idx = None
    while cond_idx is None:
        cond_in = input("Select a condition number for overlay: ").strip()
        try:
            idx = int(cond_in) - 1
            if 0 <= idx < len(evokeds):
                cond_idx = idx
        except Exception:
            pass
        if cond_idx is None:
            print("Please enter a valid condition number.")

    # Prompt ONCE for sensors to overlay
    indices = prompt_for_channels(sensor_names)
    if not indices:
        print("Exiting overlay.")
        return

    # Default y-limits per type
    current_ylims = DEFAULT_YLIMS.copy()

    while True:
        ch_types = set(get_channel_type(info, i) for i in indices)
        plt.figure(figsize=(9, 5))
        colors = plt.get_cmap('tab10').colors
        for idx, color in zip(indices, colors):
            ch_type = get_channel_type(info, idx)
            plt.plot(
                evokeds[cond_idx].times, evokeds[cond_idx].data[idx, :] * SCALE[ch_type],
                color=color, lw=2, label=f"{sensor_names[idx]} ({ch_type})"
            )
        plt.xlabel('Time (s)')
        plt.ylabel("Amplitude")
        plt.legend()
        # Set y-limits to encompass all selected types
        y_min = min(current_ylims[t][0] for t in ch_types)
        y_max = max(current_ylims[t][1] for t in ch_types)
        plt.ylim((y_min, y_max))
        plt.grid(True)
        plt.show(block=False)

        print("\nCurrently overlaying sensors: " +
              ', '.join([sensor_names[i] for i in indices]))
        print("Type new channel names, EEG names, last three digits (comma-separated),")
        print("'scale' to adjust y-limits, 'cond' to change condition,")
        print("'close' to close plots, or 'quit': ")
        nav = input().strip().lower()

        if nav == 'quit':
            break
        elif nav == 'scale':
            current_ylims = prompt_for_ylims(ch_types, current_ylims)
            continue
        elif nav == 'cond':
            # Prompt for new condition
            while True:
                cond_in = input("Select a new condition number for overlay: ").strip()
                try:
                    idx = int(cond_in) - 1
                    if 0 <= idx < len(evokeds):
                        cond_idx = idx
                        break
                except Exception:
                    pass
                print("Please enter a valid condition number.")
            continue
        elif nav in ('close', 'closeall', 'c'):
            plt.close('all')
            print("All plot windows closed.")
            continue
        elif nav:
            # Update sensor set
            new_indices = prompt_for_channels(sensor_names)
            if new_indices:
                indices = new_indices
            else:
                print("No valid sensors specified. Overlay unchanged.")

    print("Overlay plot complete. Window will stay open until you close it.\n")


def get_topomap_params(ev):
    tmin_ev, tmax_ev = ev.times[0], ev.times[-1]
    duration = tmax_ev - tmin_ev

    # Smart default increment (in seconds)
    if duration < 0.100:
        dt = 0.01  # 10 ms
    elif duration < 0.200:
        dt = 0.02  # 20 ms
    elif duration < 0.400:
        dt = 0.05  # 50 ms
    else:
        dt = 0.1   # 100 ms

    # Generate defaults
    default_start = round(tmin_ev, 3)
    default_end = round(tmax_ev, 3)
    default_dt = dt

    # Calculate default number of plots (just for info)
    n_default = int((default_end - default_start) / default_dt) + 1

    prompt = (
        f"Enter topomap start,end,increment in seconds (e.g. 0.01,0.10,0.05)\n"
        f"  within [{default_start:.3f}, {default_end:.3f}].\n"
        f"  [default: {default_start:.3f},{default_end:.3f},{default_dt:.3f} → {n_default} topomaps]\n"
        f"  Press Enter to accept defaults, or type 'quit' to exit: "
    )
    while True:
        inp = input(prompt).strip()
        if not inp:
            return default_start, default_end, default_dt
        if inp.lower() in ('quit', 'exit'):
            return None
        try:
            t_start, t_end, t_step = [float(x) for x in inp.split(',')]
            if t_step <= 0:
                print("Increment must be positive and nonzero."); continue
            if t_start >= t_end:
                print("Start time must be less than end time."); continue
            if t_start < tmin_ev or t_end > tmax_ev:
                print(f"Start/end times must be within the epoch [{tmin_ev:.3f}, {tmax_ev:.3f}]."); continue
            return t_start, t_end, t_step
        except Exception:
            print("Please enter valid numbers in the format: 0.01,0.10,0.05 or hit Enter for defaults.")

def available_types(ev):
    all_types = set(ev.get_channel_types())
    return [t for t in ('mag', 'grad', 'eeg') if t in all_types]

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_evoked_BIDS.py <config.yaml | evoked-ave-file.fif>")
        sys.exit(1)

    arg = Path(sys.argv[1])
    if arg.suffix in ('.yaml', '.yml'):
        # YAML mode
        with open(arg, 'r') as f:
            config = yaml.safe_load(f)
        try:
            evoked_fname = build_evoked_fname(config)
        except Exception as e:
            print(f"Could not build evoked filename from YAML: {e}")
            sys.exit(1)
    else:
        # Direct file path mode
        evoked_fname = arg

    print(f"Loading evoked file: {evoked_fname}")
    if not evoked_fname.exists():
        print(f"ERROR: File does not exist: {evoked_fname}")
        sys.exit(1)

    evokeds = mne.read_evokeds(str(evoked_fname))
    print(f"\nLoaded {len(evokeds)} evoked conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment} (nave={ev.nave})")


    # Main menu loop
    while True:
        print("\nMenu:")
        print("  (1) Plot Butterfly/Topomap")
        print("  (2) Plot Waveforms (single sensor, overlay conditions)")
        print("  (3) Overlay Sensors (single condition, multiple sensors)")
        print("  (c) Close all open plots")
        print("  (q) Quit")
        choice = input("Select option (1/2/3/c/q): ").strip().lower()

        if choice == '1':
            # Topomap/Butterfly loop
            params = get_topomap_params(evokeds[0])
            if params is None:
                continue
            t_start, t_end, t_step = params
            times = round_ms(np.arange(t_start, t_end + t_step / 2, t_step))
            print(f"Topomap times (rounded to ms): {times}")

            while True:
                print("\nSelect a condition to plot by number, 'reset' to change topomap times, or 'quit' to exit:")
                for i, ev in enumerate(evokeds, 1):
                    print(f"  ({i}) {ev.comment}")
                inp = input("Condition #: ").strip()
                if inp.lower() in ('quit', 'exit'):
                    break
                if inp.lower() == 'reset':
                    params = get_topomap_params(evokeds[0])
                    if params is None:
                        break
                    t_start, t_end, t_step = params
                    times = round_ms(np.arange(t_start, t_end + t_step / 2, t_step))
                    print(f"Topomap times (rounded to ms): {times}")
                    continue
                try:
                    selection = int(inp)
                    if not (1 <= selection <= len(evokeds)):
                        print("Invalid selection."); continue
                except Exception:
                    print("Please enter a valid condition number, 'reset', or 'quit'.")
                    continue
                ev = evokeds[selection - 1]
                # Butterfly plot (persistent, non-blocking)
                print(f"Plotting butterfly for: {ev.comment}")
                ev.plot(spatial_colors=True, titles=f"Evoked: {ev.comment}", show=False, window_title=f"Butterfly: {ev.comment}")
                # Topomaps for all available channel types (mag, grad, eeg)
                ch_types = available_types(ev)
                for ch_type in ch_types:
                    print(f"Plotting topomap ({ch_type}) for: {ev.comment} at times (s): {times}")
                    fig = ev.plot_topomap(times=times, ch_type=ch_type, show=False, time_unit='s', colorbar=True)
                    fig.suptitle(f"Topomap: {ev.comment} [{ch_type}]", fontsize=14)
                plt.show(block=False)
                focus_terminal()  # <-- Added here
        elif choice == '2':
            waveform_browser(evokeds)
            # waveform_browser already calls focus_terminal() internally
        elif choice == '3':
            multi_sensor_overlay(evokeds)
            # multi_sensor_overlay already calls focus_terminal() internally
        elif choice in ('c', 'close', 'closeall'):
            plt.close('all')
            print("All plot windows closed.")
        elif choice in ('4', 'q', 'quit', 'exit'):
            print("Exiting program.")
            break
        else:
            print("Invalid selection. Please type 1, 2, 3, or 'q' to quit.")

    print("Visualization complete.")

if __name__ == '__main__':
    main()
    