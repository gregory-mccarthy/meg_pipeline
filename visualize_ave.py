# visualize_evoked_BIDS.py
import sys
from pathlib import Path
import subprocess
# Try to import ruamel.yaml first, fallback to PyYAML
try:
    from ruamel.yaml import YAML
    yaml_handler = YAML(typ='safe', pure=True)
    def yaml_safe_load(stream):
        return yaml_handler.load(stream)
except ImportError:
    import yaml
    def yaml_safe_load(stream):
        return yaml.safe_load(stream)

import numpy as np
import matplotlib

import matplotlib
import matplotlib.pyplot as plt
print("[viz] Matplotlib backend:", matplotlib.get_backend())  # should print 'MacOSX' on your machine

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
        config = yaml_safe_load(f)
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

def waveform_browser(evokeds):
    """Interactive single-sensor waveform browser with condition overlay."""
    print("\n=== Waveform Browser (single sensor, overlay conditions) ===")

    # Display available conditions
    print("Available conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment}")

    # Prompt for conditions
    cond_indices = prompt_for_conditions(evokeds)
    if cond_indices is None:
        return

    # Channel types to consider
    available_ch_types = available_types(evokeds[0])
    if not available_ch_types:
        print("No plottable channel types found.")
        return

    sensor_names = evokeds[0].ch_names

    # Starting sensor index
    sensor_idx = 0

    # Colors for conditions
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Y-limits (per type) setup - using DEFAULT_YLIMS
    current_ylims = DEFAULT_YLIMS.copy()

    while True:
        # Plot the selected sensor for all conditions
        sensor_name = sensor_names[sensor_idx]
        ch_type = get_channel_type(evokeds[0].info, sensor_idx)
        
        # Skip if not a plottable channel type
        if ch_type not in ('mag', 'grad', 'eeg'):
            sensor_idx = (sensor_idx + 1) % len(sensor_names)
            continue
            
        units = UNITS.get(ch_type, 'AU')
        scale = SCALE.get(ch_type, 1.0)

        print(f"\nSensor: {sensor_name} ({ch_type})")

        # Get X-axis limits (full epoch by default)
        xlim = [evokeds[0].times[0], evokeds[0].times[-1]]
        
        # Use the DEFAULT_YLIMS or current_ylims for this channel type
        ylim = current_ylims[ch_type]

        # Plot the waveforms
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cond_i in enumerate(cond_indices):
            ev = evokeds[cond_i]
            times = ev.times
            signal = ev.data[sensor_idx, :] * scale
            label = ev.comment
            color = colors[i % len(colors)]
            ax.plot(times, signal, label=label, color=color, linewidth=1.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'Amplitude ({units})', fontsize=12)
        ax.set_title(f'Sensor: {sensor_name} ({ch_type})', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block=False)

        # Bring terminal to foreground
        focus_terminal()

        # Navigation prompt
        print("\nOptions:")
        print("  Press 'Enter' for next sensor")
        print(f"  Type a number 1-{len(sensor_names)} for specific sensor index")
        print("  Type 'p' for previous sensor")
        print("  Type sensor name (e.g., 'MEG0111', 'STI101') to jump to it")
        print("  Type 'scale' to adjust y-limits")
        print("  Type 'axis' to set custom axis limits")
        print("  Type 'cond' to select different conditions")
        print("  Type 'close' to close plots")
        print("  Type 'quit' to exit browser")
        nav = input("Your choice: ").strip().lower()

        if nav == 'quit':
            break
        elif nav == '' or nav == 'n':
            sensor_idx = (sensor_idx + 1) % len(sensor_names)
        elif nav == 'p':
            sensor_idx = (sensor_idx - 1) % len(sensor_names)
        elif nav.isdigit():
            idx = int(nav) - 1
            if 0 <= idx < len(sensor_names):
                sensor_idx = idx
                # Check if this is a plottable channel
                new_ch_type = get_channel_type(evokeds[0].info, sensor_idx)
                if new_ch_type not in ('mag', 'grad', 'eeg'):
                    print(f"Warning: {sensor_names[sensor_idx]} is type '{new_ch_type}' which cannot be plotted.")
                    print("Only mag, grad, and eeg channels can be displayed.")
            else:
                print(f"Invalid index. Must be 1-{len(sensor_names)}.")
        elif nav == 'scale':
            # Adjust Y-limits for the current channel type
            current_ylims = prompt_for_ylims([ch_type], current_ylims)
        elif nav == 'axis':
            xlim, ylim = prompt_for_axis_limits(xlim, ylim, units)
            # Update the current_ylims to remember this change
            current_ylims[ch_type] = ylim
        elif nav == 'cond':
            new_conds = prompt_for_conditions(evokeds)
            if new_conds:
                cond_indices = new_conds
        elif nav in ('close', 'closeall', 'c'):
            plt.close('all')
            print("All plot windows closed.")
        else:
            # Try to find sensor by name
            found = False
            for i, name in enumerate(sensor_names):
                if nav.upper() == name.upper():
                    sensor_idx = i
                    found = True
                    # Check if this is a plottable channel
                    new_ch_type = get_channel_type(evokeds[0].info, sensor_idx)
                    if new_ch_type not in ('mag', 'grad', 'eeg'):
                        print(f"Warning: {sensor_names[sensor_idx]} is type '{new_ch_type}' which cannot be plotted.")
                        print("Only mag, grad, and eeg channels can be displayed.")
                    break
            if not found:
                print(f"Unknown command or sensor: '{nav}'")

    print("Waveform browser complete.\n")


def multi_sensor_overlay(evokeds):
    """Interactive multi-sensor overlay on a single axis."""
    print("\n=== Multi-Sensor Overlay (multiple sensors, single condition) ===")

    # Display available conditions
    print("Available conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment}")

    # Select condition
    while True:
        cond_in = input("Select a condition number or 'quit': ").strip()
        if cond_in.lower() == 'quit':
            return
        try:
            idx = int(cond_in) - 1
            if 0 <= idx < len(evokeds):
                cond_idx = idx
                break
        except Exception:
            pass
        print("Please enter a valid condition number.")

    ev = evokeds[cond_idx]
    sensor_names = ev.ch_names
    print(f"\nCondition: {ev.comment}")
    print(f"Total sensors: {len(sensor_names)}")

    # Prompt for initial set of sensors to overlay
    indices = prompt_for_channels(sensor_names)
    if not indices:
        return

    # Initial Y-limits by channel type
    current_ylims = DEFAULT_YLIMS.copy()

    while True:
        # Determine channel types for selected sensors
        ch_types = set()
        for idx in indices:
            ch_types.add(get_channel_type(ev.info, idx))

        # Filter to only plottable types
        ch_types = [t for t in ch_types if t in ('mag', 'grad', 'eeg')]
        if not ch_types:
            print("No plottable channel types selected.")
            indices = prompt_for_channels(sensor_names)
            if not indices:
                break
            continue

        # Plot overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in indices:
            sensor_name = sensor_names[idx]
            ch_type = get_channel_type(ev.info, idx)
            if ch_type not in ('mag', 'grad', 'eeg'):
                continue
            units = UNITS[ch_type]
            scale = SCALE[ch_type]
            signal = ev.data[idx, :] * scale
            plt.plot(ev.times, signal, label=f"{sensor_name} ({ch_type})")

        plt.title(f"Multi-Sensor Overlay: {ev.comment}")
        plt.xlabel('Time (s)')
        plt.ylabel("Amplitude")
        plt.legend()
        # Set y-limits to encompass all selected types
        y_min = min(current_ylims[t][0] for t in ch_types)
        y_max = max(current_ylims[t][1] for t in ch_types)
        plt.ylim((y_min, y_max))
        plt.grid(True)
        plt.show(block=False)

        # Bring terminal to foreground
        focus_terminal()

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

def compute_gfp(evoked, ch_type='all'):
    """
    Compute Global Field Power (GFP) for specified channel type.
    GFP is the standard deviation across all sensors at each time point.
    
    Parameters:
    - evoked: MNE Evoked object
    - ch_type: 'all', 'mag', 'grad', or 'eeg'
    
    Returns:
    - times: time points
    - gfp: GFP values
    """
    import numpy as np
    
    if ch_type == 'all':
        # Use all channels except stim, misc, etc.
        picks = mne.pick_types(evoked.info, meg=True, eeg=True, exclude='bads')
    elif ch_type in ['mag', 'grad']:
        picks = mne.pick_types(evoked.info, meg=ch_type, eeg=False, exclude='bads')
    elif ch_type == 'eeg':
        picks = mne.pick_types(evoked.info, meg=False, eeg=True, exclude='bads')
    else:
        raise ValueError(f"Unknown channel type: {ch_type}")
    
    if len(picks) == 0:
        return evoked.times, np.zeros_like(evoked.times)
    
    # Get data for selected channels and scale appropriately
    data = evoked.data[picks, :]
    
    # Apply scaling to get proper units
    if ch_type in SCALE:
        data = data * SCALE[ch_type]
    elif ch_type == 'all':
        # For 'all', we need to scale different channel types appropriately
        scaled_data = []
        for pick in picks:
            ch_type_single = get_channel_type(evoked.info, pick)
            if ch_type_single in SCALE:
                scaled_data.append(evoked.data[pick, :] * SCALE[ch_type_single])
            else:
                scaled_data.append(evoked.data[pick, :])
        data = np.array(scaled_data)
    
    # Compute GFP (standard deviation across channels at each time point)
    gfp = np.std(data, axis=0)
    
    return evoked.times, gfp

def gfp_browser(evokeds):
    """Browse Global Field Power (GFP) for different channel types and conditions."""
    print("\n=== Global Field Power (GFP) Browser ===")
    print("GFP shows the overall field strength across all sensors of a given type.")
    
    # Display available conditions
    print("\nAvailable conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment}")
    
    # Prompt for conditions to overlay
    cond_indices = prompt_for_conditions(evokeds)
    if cond_indices is None:
        return
    
    # Determine available channel types
    available_ch_types = available_types(evokeds[0])
    if not available_ch_types:
        print("No plottable channel types found.")
        return
    
    # Add 'all' as an option if we have any channels
    gfp_types = ['all'] + available_ch_types
    current_type_idx = 0
    
    # Colors for conditions
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Set up Y-limits for GFP (these are reasonable defaults for GFP)
    gfp_ylims = {
        'all': (0, 100),    # Mixed units, approximate
        'mag': (0, 200),    # fT
        'grad': (0, 100),   # fT/cm  
        'eeg': (0, 20)      # μV
    }
    current_ylims = gfp_ylims.copy()
    
    while True:
        # Current GFP type
        gfp_type = gfp_types[current_type_idx]
        
        # Determine units for display
        if gfp_type == 'all':
            units = 'mixed'
        else:
            units = UNITS.get(gfp_type, 'AU')
        
        print(f"\nPlotting GFP for: {gfp_type} channels")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot GFP for each selected condition
        for i, cond_idx in enumerate(cond_indices):
            ev = evokeds[cond_idx]
            times, gfp_values = compute_gfp(ev, gfp_type)
            
            label = f"{ev.comment}"
            color = colors[i % len(colors)]
            ax.plot(times, gfp_values, label=label, color=color, linewidth=2)
        
        # Set plot properties
        xlim = [evokeds[0].times[0], evokeds[0].times[-1]]
        ylim = current_ylims[gfp_type]
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'GFP ({units})', fontsize=12)
        ax.set_title(f'Global Field Power - {gfp_type} channels', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block=False)
        
        # Bring terminal to foreground
        focus_terminal()
        
        # Navigation prompt
        print(f"\nCurrently showing: {gfp_type} (option {current_type_idx + 1}/{len(gfp_types)})")
        print("Options:")
        print("  Press 'Enter' for next channel type")
        print("  Type 'p' for previous channel type")
        print(f"  Type 1-{len(gfp_types)} to jump to specific type: {', '.join([f'{i+1}={t}' for i, t in enumerate(gfp_types)])}")
        print("  Type 'scale' to adjust y-limits")
        print("  Type 'axis' to set custom axis limits")
        print("  Type 'cond' to change conditions")
        print("  Type 'close' to close plots")
        print("  Type 'quit' to exit GFP browser")
        
        nav = input("Your choice: ").strip().lower()
        
        if nav == 'quit':
            break
        elif nav == '' or nav == 'n':
            current_type_idx = (current_type_idx + 1) % len(gfp_types)
        elif nav == 'p':
            current_type_idx = (current_type_idx - 1) % len(gfp_types)
        elif nav.isdigit():
            idx = int(nav) - 1
            if 0 <= idx < len(gfp_types):
                current_type_idx = idx
            else:
                print(f"Invalid index. Must be 1-{len(gfp_types)}.")
        elif nav == 'scale':
            # Adjust Y-limits for current GFP type
            print(f"Adjusting Y-limits for {gfp_type} GFP")
            default = current_ylims[gfp_type]
            entry = input(f"Enter Y-axis limits (min,max) [current: {default[0]},{default[1]}]: ").strip()
            if entry:
                try:
                    ymin, ymax = [float(x) for x in entry.split(',')]
                    current_ylims[gfp_type] = (ymin, ymax)
                except Exception:
                    print("Invalid input, keeping current limits.")
        elif nav == 'axis':
            xlim_new, ylim_new = prompt_for_axis_limits(xlim, ylim, units)
            xlim = xlim_new
            current_ylims[gfp_type] = ylim_new
        elif nav == 'cond':
            new_conds = prompt_for_conditions(evokeds)
            if new_conds:
                cond_indices = new_conds
        elif nav in ('close', 'closeall', 'c'):
            plt.close('all')
            print("All plot windows closed.")
        else:
            print(f"Unknown command: '{nav}'")
    
    print("GFP browser complete.\n")

def show_sensor_layout(evokeds):
    """Show sensor layout reference map with sensor names."""
    print("\n=== Sensor Layout Reference ===")
    
    # Use the first evoked object's info for sensor positions
    info = evokeds[0].info
    
    # Determine what types of sensors we have
    available_ch_types = available_types(evokeds[0])
    
    print("This will display sensor positions with labels.")
    print(f"Available sensor types: {', '.join(available_ch_types)}")
    
    # Options for what to display
    print("\nDisplay options:")
    print("  (1) All sensors")
    print("  (2) Magnetometers only") 
    print("  (3) Gradiometers only")
    print("  (4) EEG only (if available)")
    print("  (q) Cancel")
    
    choice = input("Select option: ").strip().lower()
    
    if choice == 'q':
        return
    
    # Determine which sensors to show
    if choice == '2' and 'mag' in available_ch_types:
        picks = mne.pick_types(info, meg='mag', exclude=[])
        title = "Magnetometer Layout"
    elif choice == '3' and 'grad' in available_ch_types:
        picks = mne.pick_types(info, meg='grad', exclude=[])
        title = "Gradiometer Layout"
    elif choice == '4' and 'eeg' in available_ch_types:
        picks = mne.pick_types(info, meg=False, eeg=True, exclude=[])
        title = "EEG Layout"
    else:
        # Default to all sensors
        picks = mne.pick_types(info, meg=True, eeg=True, exclude=[])
        title = "All Sensors Layout"
    
    if len(picks) == 0:
        print("No sensors of that type found.")
        return
    
    print(f"\nShowing layout for {len(picks)} sensors...")
    print("TIP: You can use the last 3 digits shown here when navigating in other modes.")
    
    # Create the sensor layout plot
    fig = mne.viz.plot_sensors(info, ch_type='all', picks=picks, 
                                show_names=True, show=False, 
                                title=title, block=False)
    
    # Make the plot a bit larger for better readability
    if fig:
        fig.set_size_inches(12, 10)
    
    plt.show(block=False)
    
    # Add a simple legend/guide
    print("\n" + "="*60)
    print("SENSOR NAMING CONVENTION:")
    print("  MEG####:")
    print("    - First digit (0-2): Region")
    print("      0 = Frontal/Central")
    print("      1 = Left hemisphere") 
    print("      2 = Right hemisphere")
    print("    - Last digit:")
    print("      1 = Magnetometer")
    print("      2,3 = Gradiometer pair")
    print("\nNAVIGATION TIPS:")
    print("  - In browsers, type the last 3 digits to jump to a sensor")
    print("  - Or type the full name (e.g., MEG0111)")
    print("  - Sensors are roughly organized in a grid pattern")
    print("="*60)
    
    print("\nPress Enter to continue...")
    input()
    
    focus_terminal()

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_evoked_BIDS.py <config.yaml | evoked-ave-file.fif>")
        sys.exit(1)

    arg = Path(sys.argv[1])
    if arg.suffix in ('.yaml', '.yml'):
        # YAML mode
        with open(arg, 'r') as f:
            config = yaml_safe_load(f)
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
        print("  (4) Interactive Sensor Layout (click sensors for waveforms)")
        print("  (5) Global Field Power (GFP) browser")
        print("  (6) Show Sensor Layout Reference")
        print("  (c) Close all open plots")
        print("  (q) Quit")
        choice = input("Select option (1/2/3/4/5/6/c/q): ").strip().lower()

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
        elif choice == '4':
            # Interactive sensor layout with plot_topo
            print("\n=== Interactive Sensor Layout ===")
            print("Available conditions:")
            for i, ev in enumerate(evokeds, 1):
                print(f"  ({i}) {ev.comment}")
            
            # Select conditions to display
            print("\nYou can plot multiple conditions overlaid on the same sensors.")
            cond_str = input("Enter condition numbers (comma-separated, or 'all' for all conditions): ").strip()
            
            if cond_str.lower() == 'all':
                selected_evokeds = evokeds
                print(f"Plotting all {len(evokeds)} conditions")
            else:
                try:
                    indices = [int(x.strip())-1 for x in cond_str.split(',') if x.strip()]
                    if all(0 <= i < len(evokeds) for i in indices):
                        selected_evokeds = [evokeds[i] for i in indices]
                        print(f"Plotting conditions: {', '.join([ev.comment for ev in selected_evokeds])}")
                    else:
                        print("Invalid selection, using first condition only.")
                        selected_evokeds = [evokeds[0]]
                except Exception:
                    print("Invalid input, using first condition only.")
                    selected_evokeds = [evokeds[0]]
            
            # Create the interactive plot
            print("\nCreating interactive sensor layout plot...")
            print("Click on any sensor to see its detailed waveform in a popup window.")
            print("Note: The plot may take a moment to appear.")
            print("This will show all channel types (mag, grad, eeg) present in the data.")
            
            # Plot first condition and overlay others if selected
            if len(selected_evokeds) == 1:
                # Single condition - simple plot
                fig = selected_evokeds[0].plot_topo(
                    show=False,
                    title=f"Interactive Layout: {selected_evokeds[0].comment}"
                )
            else:
                # Multiple conditions - plot them using merge
                from mne import combine_evoked
                
                # Create a list of evokeds with weights for combining
                all_evokeds = []
                for ev in selected_evokeds:
                    all_evokeds.append(ev)
                
                # Plot first one
                fig = selected_evokeds[0].plot_topo(
                    show=False,
                    title=f"Interactive Layout: {len(selected_evokeds)} conditions"
                )
                
                print(f"Note: plot_topo shows the first condition ({selected_evokeds[0].comment}).")
                print("For overlaid conditions, use option 2 or 3 from the main menu.")
            
            plt.show(block=False)
            focus_terminal()
            
            print("\nInteractive plot displayed. Click on sensors to view details.")
            print("Keep this plot open and return here to select other menu options.")
            
        elif choice == '5':
            gfp_browser(evokeds)
            # gfp_browser already calls focus_terminal() internally
        
        elif choice == '6':
            show_sensor_layout(evokeds)
            # show_sensor_layout already calls focus_terminal() internally
            
        elif choice in ('c', 'close', 'closeall'):
            plt.close('all')
            print("All plot windows closed.")
        elif choice in ('q', 'quit', 'exit'):
            print("Exiting program.")
            break
        else:
            print("Invalid selection. Please type 1, 2, 3, 4, 5, 6, or 'q' to quit.")

    print("Visualization complete.")

if __name__ == '__main__':
    main()
