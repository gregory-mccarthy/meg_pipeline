# visualize_evoked_BIDS_merged.py
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
import matplotlib.pyplot as plt

print("[viz] Matplotlib backend:", matplotlib.get_backend())

import mne

DEFAULT_YLIMS = {
    'mag': (-200, 200),  # fT
    'grad': (-100, 100),  # fT/cm
    'eeg': (-20, 20),  # μV
}
UNITS = {'mag': 'fT', 'grad': 'fT/cm', 'eeg': 'μV'}
SCALE = {'mag': 1e15, 'grad': 1e13, 'eeg': 1e6}


def get_channel_type(info, idx):
    return mne.channel_type(info, idx)


def handle_global_commands(user_input):
    """Intercept global commands like 'c' to close plots."""
    if user_input.lower() == 'c':
        plt.close('all')
        print("All plot windows closed.")
        return True
    return False


# --- Restored Utility Functions ---
def round_ms(times):
    return np.round(np.array(times) * 1000).astype(int) / 1000


def get_channel_indices(ev, ch_type):
    return [(i, ch) for i, (ch, typ) in enumerate(zip(ev.ch_names, ev.get_channel_types())) if typ == ch_type]


def get_data_minmax(evokeds, ch_type):
    vals = []
    for ev in evokeds:
        picks = mne.pick_types(ev.info, meg=ch_type if ch_type in ('mag', 'grad') else False,
                               eeg=(ch_type == 'eeg'), exclude=[])
        if picks.size == 0: continue
        vals.append(ev.data[picks, :])
    if not vals: return None, None
    all_vals = np.concatenate(vals, axis=0)
    return np.nanmin(all_vals), np.nanmax(all_vals)


def get_signal_minmax_in_window(ev, ch_indices, t_min=0.01):
    t = ev.times
    idx = np.where(t >= t_min)[0]
    if not len(idx): idx = np.arange(len(t))
    data = ev.data[ch_indices, :][:, idx]
    return np.nanmin(data), np.nanmax(data)


def get_default_axis_limits(evokeds, ch_type, ch_indices, cond_indices):
    xlim = [evokeds[0].times[0], evokeds[0].times[-1]]
    min_val, max_val = None, None
    for cond_i in cond_indices:
        ev = evokeds[cond_i]
        mn, mx = get_signal_minmax_in_window(ev, ch_indices)
        min_val = mn if min_val is None else min(min_val, mn)
        max_val = mx if max_val is None else max(max_val, mx)
    pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0
    ylim = [min_val - pad, max_val + pad]
    return xlim, ylim


def prompt_for_sensor_index(sensor_names):
    while True:
        user = input(f"Type sensor name/number, 'c' to close plots, or 'q' to quit: ").strip()
        if handle_global_commands(user): continue
        if user.lower() == 'q': return None
        if user.isdigit() and 1 <= int(user) <= len(sensor_names):
            return int(user) - 1
        for i, name in enumerate(sensor_names):
            if user.lower() == name.lower():
                return i
        print(f"Unknown sensor '{user}'. Try again.")


def prompt_for_channels(sensor_names):
    sensor_lookup = {name.lower(): i for i, name in enumerate(sensor_names)}
    suffix_lookup = {name[-3:]: i for i, name in enumerate(sensor_names) if name.startswith('MEG')}
    while True:
        sensors_in = input("Type channels (comma-separated), 'c' to close plots, or 'q' to quit: ").strip()
        if handle_global_commands(sensors_in): continue
        if sensors_in.lower() == 'q': return None

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
        if indices: return indices
        print("No valid channels entered. Try again.")


def prompt_for_ylims(channel_types, current_ylims):
    ylims = current_ylims.copy()
    for ch_type in sorted(channel_types):
        default = ylims[ch_type]
        units = UNITS[ch_type]
        entry = input(
            f"Y axis for {ch_type.upper()} in {units} (min,max) [default: {default[0]},{default[1]}]: "
        ).strip()
        if handle_global_commands(entry): continue
        if entry:
            try:
                ymin, ymax = [float(x) for x in entry.split(',')]
                ylims[ch_type] = (ymin, ymax)
            except ValueError:
                print("Invalid input, using default.")
    return ylims


def focus_terminal():
    if sys.platform == "darwin":
        script = 'tell application "System Events" to set frontmost of process "Terminal" to true'
        try:
            subprocess.run(['osascript', '-e', script], check=True)
        except Exception:
            pass


def build_evoked_fname(config):
    bids_root = config['bids_root']
    subject = config['subject']
    session = config.get('session', None)
    task = config.get('task', None)
    deriv_root = Path(bids_root) / "derivatives" / "preprocessing"
    base = f"sub-{subject}"
    if session: base += f"_ses-{session}"
    if task: base += f"_task-{task}"
    return deriv_root / f"sub-{subject}/ses-{session}/meg/{base}_ave-ave.fif"


def prompt_for_conditions(evokeds):
    while True:
        cond_str = input("Enter condition numbers (e.g., 1,2), 'c' to close plots, or 'q' to quit: ").strip()
        if handle_global_commands(cond_str): continue
        if cond_str.lower() == 'q': return None
        try:
            cond_indices = [int(c) - 1 for c in cond_str.split(',') if c]
            if all(0 <= i < len(evokeds) for i in cond_indices):
                return cond_indices
        except ValueError:
            pass
        print("Invalid selection. Please enter valid condition numbers.")


def parse_waveform_condition_selection(cond_str, n_conditions, allow_all=False):
    cond_str = cond_str.strip()

    if allow_all and cond_str.lower() == 'all':
        return {'mode': 'overlay', 'indices': list(range(n_conditions))}, None

    tokens = [tok.strip() for tok in cond_str.split(',') if tok.strip()]
    if not tokens:
        return None, "Invalid selection. Please enter valid condition numbers."

    try:
        values = [int(tok) for tok in tokens]
    except ValueError:
        return None, "Invalid selection. Use comma-separated integers such as 1,2 or 1,-2."

    if any(val == 0 for val in values):
        return None, "Condition numbers start at 1. Zero is not valid."

    has_negative = any(val < 0 for val in values)
    if has_negative:
        if len(values) != 2 or values[0] <= 0 or values[1] >= 0:
            return None, (
                "Difference mode supports exactly two entries in the form 1,-2. "
                "Mixed selections like 1,2,-3 are not supported."
            )

        pos_idx = values[0] - 1
        neg_idx = abs(values[1]) - 1
        if not (0 <= pos_idx < n_conditions and 0 <= neg_idx < n_conditions):
            return None, "Invalid selection. Please enter valid condition numbers."

        return {'mode': 'difference', 'positive': pos_idx, 'negative': neg_idx}, None

    indices = [val - 1 for val in values]
    if not all(0 <= idx < n_conditions for idx in indices):
        return None, "Invalid selection. Please enter valid condition numbers."

    return {'mode': 'overlay', 'indices': indices}, None


def build_difference_evoked(evokeds, positive_idx, negative_idx):
    diff_evoked = mne.combine_evoked([evokeds[positive_idx], evokeds[negative_idx]], weights=[1, -1])
    diff_evoked.comment = f"{evokeds[positive_idx].comment} - {evokeds[negative_idx].comment}"
    return diff_evoked


def resolve_waveform_condition_selection(evokeds, selection):
    if selection['mode'] == 'overlay':
        return [evokeds[idx] for idx in selection['indices']]

    return [build_difference_evoked(evokeds, selection['positive'], selection['negative'])]


def prompt_for_waveform_conditions(evokeds, allow_all=False):
    prompt = "Enter condition numbers (e.g., 1,2 or 1,-2)"
    if allow_all:
        prompt += ", or 'all'"
    prompt += ", 'c' to close plots, or 'q' to quit: "

    while True:
        cond_str = input(prompt).strip()
        if handle_global_commands(cond_str):
            continue
        if cond_str.lower() == 'q':
            return None

        selection, error = parse_waveform_condition_selection(cond_str, len(evokeds), allow_all=allow_all)
        if selection is not None:
            return resolve_waveform_condition_selection(evokeds, selection)

        print(error)


def prompt_for_axis_limits(default_xlim_ms, default_ylim, units):
    rounded_xlim = [int(default_xlim_ms[0]), int(default_xlim_ms[1])]
    rounded_ylim = [round(default_ylim[0], 2), round(default_ylim[1], 2)]

    while True:
        xlim_in = input(f"Enter X limits in ms (start,end) [default {rounded_xlim[0]},{rounded_xlim[1]}]: ").strip()
        if handle_global_commands(xlim_in): continue

        if xlim_in:
            parts = xlim_in.split(',')
            if len(parts) != 2:
                print("Invalid input. Provide exactly two values (start,end).")
                continue
            try:
                xstart, xend = [float(x) for x in parts]
                xlim = [xstart, xend]
            except ValueError:
                print("Invalid input. Use numbers only.")
                continue
        else:
            xlim = rounded_xlim

        ylim_in = input(f"Enter Y limits in {units} (min,max) [default {rounded_ylim[0]},{rounded_ylim[1]}]: ").strip()
        if ylim_in:
            parts = ylim_in.split(',')
            if len(parts) != 2:
                print("Invalid input. Provide exactly two values (min,max).")
                continue
            try:
                ymin, ymax = [float(y) for y in parts]
                ylim = [ymin, ymax]
            except ValueError:
                print("Invalid input. Use numbers only.")
                continue
        else:
            ylim = rounded_ylim

        return xlim, ylim


def apply_lowpass_filter(evokeds):
    print("\n=== Apply Low-Pass Filter ===")
    freq_in = input("Enter low-pass cutoff in Hz (e.g., 40), 'c' to close plots, or 'q' to cancel: ").strip()
    if handle_global_commands(freq_in): return
    if freq_in.lower() == 'q': return
    try:
        h_freq = float(freq_in)
        for ev in evokeds:
            ev.filter(l_freq=None, h_freq=h_freq)
        print(f"Successfully applied {h_freq} Hz low-pass filter to all conditions.")
    except ValueError:
        print("Invalid frequency entered. Filter application cancelled.")


def apply_baseline_correction(evokeds):
    print("\n=== Apply Baseline Correction ===")
    while True:
        base_in = input(
            "Enter baseline period in ms (start,end) [e.g., -200,0], 'c' to close plots, or 'q' to cancel: ").strip()
        if handle_global_commands(base_in): return
        if base_in.lower() == 'q': return

        parts = base_in.split(',')
        if len(parts) != 2:
            print("Invalid input. Please provide exactly two values separated by a comma (e.g., -200,0).")
            continue

        try:
            start_ms, end_ms = [float(x) for x in parts]
            baseline = (start_ms / 1000.0, end_ms / 1000.0)
            for ev in evokeds:
                ev.apply_baseline(baseline)
            print(f"Successfully applied baseline correction ({start_ms} ms to {end_ms} ms) to all conditions.")
            break
        except ValueError:
            print("Invalid format. Please use numbers only.")


def waveform_browser(evokeds):
    print("\n=== Waveform Browser ===")
    for i, ev in enumerate(evokeds, 1): print(f"  ({i}) {ev.comment}")

    selected_evokeds = prompt_for_waveform_conditions(evokeds)
    if selected_evokeds is None:
        return

    sensor_names = evokeds[0].ch_names
    sensor_idx = 0
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    current_ylims = DEFAULT_YLIMS.copy()

    # MOVED OUTSIDE: Fix for X-axis not updating
    xlim_ms = [evokeds[0].times[0] * 1000, evokeds[0].times[-1] * 1000]

    while True:
        sensor_name = sensor_names[sensor_idx]
        ch_type = get_channel_type(evokeds[0].info, sensor_idx)

        if ch_type not in ('mag', 'grad', 'eeg'):
            sensor_idx = (sensor_idx + 1) % len(sensor_names)
            continue

        units = UNITS.get(ch_type, 'AU')
        scale = SCALE.get(ch_type, 1.0)
        ylim = current_ylims[ch_type]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, ev in enumerate(selected_evokeds):
            times_ms = ev.times * 1000
            signal = ev.data[sensor_idx, :] * scale
            ax.plot(times_ms, signal, label=ev.comment, color=colors[i % len(colors)], linewidth=1.5)

        ax.set_xlim(xlim_ms)
        ax.set_ylim(ylim)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(f'Amplitude ({units})', fontsize=12)
        ax.set_title(f'Sensor: {sensor_name} ({ch_type})', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block=False)

        focus_terminal()

        # INNER LOOP: Handles 'c' closing without redrawing
        need_redraw = False
        while not need_redraw:
            print(f"\n[Sensor: {sensor_name}] Options:")
            print("  [Enter]=Next  | [p]=Prev | [Number/Name]=Jump | [s]=Scale Y-axis")
            print("  [a]=Axis X/Y  | [d]=Change Conds | [c]=Close plots | [q]=Quit")
            nav = input("Choice: ").strip().lower()

            if handle_global_commands(nav):
                continue

            if nav == 'q':
                return
            elif nav == '' or nav == 'n':
                sensor_idx = (sensor_idx + 1) % len(sensor_names)
                need_redraw = True
            elif nav == 'p':
                sensor_idx = (sensor_idx - 1) % len(sensor_names)
                need_redraw = True
            elif nav.isdigit():
                idx = int(nav) - 1
                if 0 <= idx < len(sensor_names):
                    sensor_idx = idx
                    need_redraw = True
            elif nav in ['scale', 's']:
                current_ylims = prompt_for_ylims([ch_type], current_ylims)
                need_redraw = True
            elif nav in ['axis', 'a']:
                xlim_ms, ylim = prompt_for_axis_limits(xlim_ms, ylim, units)
                current_ylims[ch_type] = ylim
                need_redraw = True
            elif nav in ['cond', 'd']:
                new_selection = prompt_for_waveform_conditions(evokeds)
                if new_selection is not None:
                    selected_evokeds = new_selection
                    need_redraw = True
            else:
                found = False
                for i, name in enumerate(sensor_names):
                    if nav.upper() == name.upper():
                        sensor_idx = i
                        found = True
                        need_redraw = True
                        break
                if not found:
                    print(f"Unknown command or sensor: '{nav}'")


def multi_sensor_overlay(evokeds):
    print("\n=== Multi-Sensor Overlay ===")
    for i, ev in enumerate(evokeds, 1): print(f"  ({i}) {ev.comment}")

    while True:
        cond_in = input("Select a condition number, 'c' to close plots, or 'q' to quit: ").strip()
        if handle_global_commands(cond_in): continue
        if cond_in.lower() == 'q': return
        try:
            idx = int(cond_in) - 1
            if 0 <= idx < len(evokeds):
                cond_idx = idx
                break
        except ValueError:
            pass

    ev = evokeds[cond_idx]
    sensor_names = ev.ch_names
    indices = prompt_for_channels(sensor_names)
    if not indices: return

    current_ylims = DEFAULT_YLIMS.copy()

    while True:
        ch_types = [t for t in {get_channel_type(ev.info, idx) for idx in indices} if t in ('mag', 'grad', 'eeg')]
        if not ch_types:
            indices = prompt_for_channels(sensor_names)
            if not indices: break
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in indices:
            sensor_name = sensor_names[idx]
            ch_type = get_channel_type(ev.info, idx)
            if ch_type not in ('mag', 'grad', 'eeg'): continue
            signal = ev.data[idx, :] * SCALE[ch_type]
            plt.plot(ev.times * 1000, signal, label=f"{sensor_name} ({ch_type})")

        plt.title(f"Multi-Sensor Overlay: {ev.comment}")
        plt.xlabel('Time (ms)')
        plt.ylabel("Amplitude")
        plt.legend()
        plt.ylim((min(current_ylims[t][0] for t in ch_types), max(current_ylims[t][1] for t in ch_types)))
        plt.grid(True)
        plt.show(block=False)
        focus_terminal()

        # INNER LOOP: Handles 'c' closing without redrawing
        need_redraw = False
        while not need_redraw:
            print("\nOptions: Type new channels, [s]=scale, [d]=cond, [c]=close plots, [q]=quit")
            nav = input("Choice: ").strip().lower()

            if handle_global_commands(nav):
                continue

            if nav == 'q':
                return
            elif nav in ['scale', 's']:
                current_ylims = prompt_for_ylims(ch_types, current_ylims)
                need_redraw = True
            elif nav in ['cond', 'd']:
                cond_in = input("New condition number: ").strip()
                try:
                    if 0 <= int(cond_in) - 1 < len(evokeds):
                        cond_idx = int(cond_in) - 1
                        need_redraw = True
                except ValueError:
                    pass
            elif nav:
                new_indices = prompt_for_channels(sensor_names)
                if new_indices:
                    indices = new_indices
                    need_redraw = True


def get_topomap_params(ev):
    tmin_ms, tmax_ms = ev.times[0] * 1000, ev.times[-1] * 1000
    duration = tmax_ms - tmin_ms

    if duration < 100:
        dt = 10
    elif duration < 200:
        dt = 20
    elif duration < 400:
        dt = 50
    else:
        dt = 100

    default_start = int(tmin_ms)
    default_end = int(tmax_ms)

    prompt = (
        f"Enter topomap start,end,step in ms (e.g. 10,100,10)\n"
        f"  within [{default_start} ms, {default_end} ms].\n"
        f"  [default: {default_start},{default_end},{dt}]\n"
        f"  Press Enter for defaults, 'c' to close plots, 'q' to quit: "
    )
    while True:
        inp = input(prompt).strip()
        if handle_global_commands(inp): continue
        if not inp: return default_start / 1000.0, default_end / 1000.0, dt / 1000.0
        if inp.lower() == 'q': return None

        parts = inp.split(',')
        if len(parts) != 3:
            print("Invalid input. Please provide exactly three values separated by commas (e.g., 10,100,10).")
            continue

        try:
            t_start, t_end, t_step = [float(x) for x in parts]
            if t_step <= 0 or t_start >= t_end:
                print("Invalid range or step. Start must be less than end, and step must be positive.")
                continue
            return t_start / 1000.0, t_end / 1000.0, t_step / 1000.0
        except ValueError:
            print("Invalid format. Please use numbers only (e.g., 10,100,10).")


def available_types(ev):
    return [t for t in ('mag', 'grad', 'eeg') if t in set(ev.get_channel_types())]


def compute_gfp(evoked, ch_type='all'):
    picks = mne.pick_types(evoked.info, meg=(ch_type if ch_type in ['mag', 'grad'] else True),
                           eeg=(ch_type in ['eeg', 'all']), exclude='bads')
    if len(picks) == 0: return evoked.times * 1000, np.zeros_like(evoked.times)

    data = evoked.data[picks, :]
    if ch_type in SCALE:
        data = data * SCALE[ch_type]
    elif ch_type == 'all':
        data = np.array([evoked.data[p, :] * SCALE.get(get_channel_type(evoked.info, p), 1) for p in picks])

    return evoked.times * 1000, np.std(data, axis=0)


def gfp_browser(evokeds):
    print("\n=== GFP Browser ===")
    for i, ev in enumerate(evokeds, 1): print(f"  ({i}) {ev.comment}")

    cond_indices = prompt_for_conditions(evokeds)
    if cond_indices is None: return

    gfp_types = ['all'] + available_types(evokeds[0])
    current_type_idx = 0
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    current_ylims = {'all': (0, 100), 'mag': (0, 200), 'grad': (0, 100), 'eeg': (0, 20)}

    # MOVED OUTSIDE: Fix for X-axis not updating
    xlim_ms = [evokeds[0].times[0] * 1000, evokeds[0].times[-1] * 1000]

    while True:
        gfp_type = gfp_types[current_type_idx]
        units = 'mixed' if gfp_type == 'all' else UNITS.get(gfp_type, 'AU')

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cond_idx in enumerate(cond_indices):
            times_ms, gfp_values = compute_gfp(evokeds[cond_idx], gfp_type)
            ax.plot(times_ms, gfp_values, label=evokeds[cond_idx].comment, color=colors[i % len(colors)], linewidth=2)

        ax.set_xlim(xlim_ms)
        ax.set_ylim(current_ylims[gfp_type])
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(f'GFP ({units})', fontsize=12)
        ax.set_title(f'Global Field Power - {gfp_type}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
        focus_terminal()

        # INNER LOOP: Handles 'c' closing without redrawing
        need_redraw = False
        while not need_redraw:
            print(f"\n[{gfp_type}] Options: [Enter]=Next | [p]=Prev | [s]=Scale Y-axis")
            print("  [a]=Axis X/Y | [d]=Change Conds | [c]=Close plots | [q]=Quit")
            nav = input("Choice: ").strip().lower()

            if handle_global_commands(nav):
                continue

            if nav == 'q':
                return
            elif nav == '' or nav == 'n':
                current_type_idx = (current_type_idx + 1) % len(gfp_types)
                need_redraw = True
            elif nav == 'p':
                current_type_idx = (current_type_idx - 1) % len(gfp_types)
                need_redraw = True
            elif nav in ['scale', 's']:
                entry = input(f"Y limits (min,max) [current: {current_ylims[gfp_type]}]: ").strip()
                if entry and not handle_global_commands(entry):
                    try:
                        current_ylims[gfp_type] = tuple(float(x) for x in entry.split(','))
                        need_redraw = True
                    except ValueError:
                        pass
            elif nav in ['axis', 'a']:
                xlim_ms, ylim_new = prompt_for_axis_limits(xlim_ms, current_ylims[gfp_type], units)
                current_ylims[gfp_type] = ylim_new
                need_redraw = True
            elif nav in ['cond', 'd']:
                new_conds = prompt_for_conditions(evokeds)
                if new_conds:
                    cond_indices = new_conds
                    need_redraw = True


def show_sensor_layout(evokeds):
    info = evokeds[0].info
    avail = available_types(evokeds[0])

    print("\nDisplay options: (1) All (2) Mag (3) Grad (4) EEG (q) Quit")
    choice = input("Select: ").strip().lower()
    if choice == 'q' or handle_global_commands(choice): return

    # Default to all
    ch_type_arg = 'all'
    title = "All Sensors Layout"

    # Update based on selection
    if choice == '2' and 'mag' in avail:
        ch_type_arg, title = 'mag', "Magnetometers"
    elif choice == '3' and 'grad' in avail:
        ch_type_arg, title = 'grad', "Gradiometers"
    elif choice == '4' and 'eeg' in avail:
        ch_type_arg, title = 'eeg', "EEG"

    # Plot using the ch_type argument instead of picks
    fig = mne.viz.plot_sensors(info, ch_type=ch_type_arg, show_names=True, show=False, title=title)
    if fig: fig.set_size_inches(12, 10)
    plt.show(block=False)

    # Restored Sensor Naming Guide
    print("\n" + "=" * 60)
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
    print("=" * 60)

    focus_terminal()
    input("\nPress Enter to return...")


def regional_grid_plot(evokeds):
    print("\n=== Regional Grid Plot ===")
    regions = [
        'Left-frontal', 'Right-frontal', 'Left-parietal', 'Right-parietal',
        'Left-temporal', 'Right-temporal', 'Left-occipital', 'Right-occipital', 'Vertex'
    ]

    for i, r in enumerate(regions, 1):
        print(f"  ({i}) {r}")

    reg_in = input("\nSelect region number, 'c' to close plots, or 'q' to quit: ").strip()
    if handle_global_commands(reg_in): return
    if reg_in.lower() == 'q': return

    try:
        region_name = regions[int(reg_in) - 1]
    except (ValueError, IndexError):
        print("Invalid region selection.")
        return

    print("\nSelect sensor type:")
    print("  (1) Magnetometers (ends in '1')")
    print("  (2) Gradiometers - Latitudinal (ends in '2')")
    print("  (3) Gradiometers - Longitudinal (ends in '3')")

    type_in = input("Select type (1/2/3), 'c' to close plots, or 'q' to quit: ").strip()
    if handle_global_commands(type_in): return
    if type_in.lower() == 'q': return
    if type_in not in ['1', '2', '3']:
        print("Invalid type.")
        return

    suffix = type_in

    # Get all sensors for the selected region based on the info object
    sel = mne.read_vectorview_selection(region_name, info=evokeds[0].info)

    # Filter by the last digit to isolate mag vs gradx vs grady
    ch_names = [ch for ch in sel if ch.endswith(suffix)]

    if not ch_names:
        print(f"No sensors found for {region_name} ending in '{suffix}'.")
        return

    # Select conditions to overlay, or build a single difference waveform via 1,-2
    selected_evokeds = prompt_for_waveform_conditions(evokeds)
    if selected_evokeds is None:
        return

    # Setup correct units and scale
    if suffix == '1':
        scale = SCALE['mag']
        units = UNITS['mag']
        sensor_label = "Magnetometers"
    else:
        scale = SCALE['grad']
        units = UNITS['grad']
        sensor_label = f"Gradiometers (Type {suffix})"

    # Isolate valid channel indices present in the data
    ch_indices = [evokeds[0].ch_names.index(ch) for ch in ch_names if ch in evokeds[0].ch_names]

    # Determine default time limits in ms
    tmin_ms, tmax_ms = evokeds[0].times[0] * 1000, evokeds[0].times[-1] * 1000

    # Calculate global min/max for regional autoscaling across all selected conditions
    global_min, global_max = float('inf'), float('-inf')
    for ev in selected_evokeds:
        data = ev.data[ch_indices, :] * scale
        global_min = min(global_min, np.min(data))
        global_max = max(global_max, np.max(data))

    # Add a 5% margin for visual padding
    margin = (global_max - global_min) * 0.05 if global_max > global_min else 1.0
    default_ylim = (global_min - margin, global_max + margin)

    # Prompt user for axis limits (defaults to full epoch and global min/max)
    xlim, ylim = prompt_for_axis_limits([tmin_ms, tmax_ms], default_ylim, units)

    # Calculate an optimal grid layout
    n_sensors = len(ch_indices)
    ncols = int(np.ceil(np.sqrt(n_sensors)))
    nrows = int(np.ceil(n_sensors / ncols))

    # Create the grid sharing X and Y axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten() if n_sensors > 1 else [axes]

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

    # Plot data into the grid
    for i, ch_idx in enumerate(ch_indices):
        ax = axes[i]
        ch_name = evokeds[0].ch_names[ch_idx]

        for j, ev in enumerate(selected_evokeds):
            times_ms = ev.times * 1000
            signal = ev.data[ch_idx, :] * scale
            ax.plot(times_ms, signal, label=ev.comment, color=colors[j % len(colors)], linewidth=1.5)

        ax.set_title(ch_name, fontsize=10, pad=3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

        # Only add the legend to the first plot to avoid cluttering the grid
        if i == 0:
            ax.legend(fontsize=8, loc='best')

    # Hide any unused subplots
    for i in range(len(ch_indices), len(axes)):
        axes[i].set_visible(False)

    # Add overarching figure labels
    fig.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, f'Amplitude ({units})', va='center', rotation='vertical', fontsize=12)
    fig.suptitle(f"{region_name} - {sensor_label}", fontsize=14, y=0.97)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show(block=False)
    focus_terminal()
    print(f"\nSuccessfully plotted {len(ch_indices)} {sensor_label.lower()} in the {region_name} region.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_evoked_BIDS_merged.py <config.yaml | evoked-ave-file.fif>")
        sys.exit(1)

    arg = Path(sys.argv[1])
    if arg.suffix in ('.yaml', '.yml'):
        with open(arg, 'r') as f:
            config = yaml_safe_load(f)
        evoked_fname = build_evoked_fname(config)
    else:
        evoked_fname = arg

    evokeds = mne.read_evokeds(str(evoked_fname))

    # Restored nave printout
    print(f"\nLoaded {len(evokeds)} evoked conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment} (nave={ev.nave})")

    while True:
        print("\nMenu:")
        print("  (1) Plot Butterfly/Topomap")
        print("  (2) Plot Waveforms")
        print("  (3) Overlay Sensors")
        print("  (4) Interactive Sensor Layout")
        print("  (5) GFP Browser")
        print("  (6) Sensor Layout Reference")
        print("  (7) Regional Grid Plot (Stable Matplotlib)")
        print("  (b) Apply baseline correction")
        print("  (f) Apply low-pass filter")
        print("  (c) Close all open plots")
        print("  (q) Quit")
        choice = input("Select: ").strip().lower()

        if handle_global_commands(choice): continue
        if choice == 'q':
            break
        elif choice == '1':
            params = get_topomap_params(evokeds[0])
            if not params: continue
            t_start, t_end, t_step = params
            times = np.arange(t_start, t_end + t_step / 2, t_step)

            while True:
                print("\nAvailable conditions:")
                for i, ev in enumerate(evokeds, 1):
                    print(f"  ({i}) {ev.comment}")

                inp = input("\nCondition #, 'r' (reset times), 'c' (close plots), 'q' (quit): ").strip()
                if handle_global_commands(inp): continue
                if inp.lower() == 'q': break
                if inp.lower() in ['reset', 'r']:
                    params = get_topomap_params(evokeds[0])
                    if params: times = np.arange(params[0], params[1] + params[2] / 2, params[2])
                    continue
                try:
                    cond_idx = int(inp) - 1
                    if not (0 <= cond_idx < len(evokeds)):
                        print("Invalid selection. Condition number out of range.")
                        continue
                    ev = evokeds[cond_idx]
                    # Restored descriptive titles
                    ev.plot(spatial_colors=True, titles=f"Evoked: {ev.comment}",
                            window_title=f"Butterfly: {ev.comment}", show=False, time_unit='ms')
                    for ch_type in available_types(ev):
                        fig = ev.plot_topomap(times=times, ch_type=ch_type, show=False, time_unit='ms', colorbar=True)
                        if fig: fig.suptitle(f"Topomap: {ev.comment} [{ch_type}]", fontsize=14)
                    plt.show(block=False)
                    focus_terminal()
                except ValueError:
                    print("Invalid input. Please enter a number, 'r', 'c', or 'q'.")
        elif choice == '2':
            waveform_browser(evokeds)
        elif choice == '3':
            multi_sensor_overlay(evokeds)
        elif choice == '4':
            print("\n=== Interactive Sensor Layout ===")
            print("Available conditions:")
            for i, ev in enumerate(evokeds, 1):
                print(f"  ({i}) {ev.comment}")

            selected = prompt_for_waveform_conditions(evokeds, allow_all=True)
            if selected is None:
                continue

            if len(selected) == 1:
                selected[0].plot_topo(show=False, title=f"Interactive Layout: {selected[0].comment}")
            else:
                try:
                    mne.viz.plot_evoked_topo(
                        selected,
                        show=False,
                        legend=True,
                        title=f"Interactive Layout: {len(selected)} conditions",
                    )
                    print("Overlaying conditions in the interactive layout:")
                    for ev in selected:
                        print(f"  - {ev.comment}")
                except Exception as exc:
                    print("Could not overlay conditions in the interactive layout; showing the first selection instead.")
                    print(f"Reason: {exc}")
                    selected[0].plot_topo(show=False, title=f"Interactive Layout: {selected[0].comment}")

            plt.show(block=False)
            focus_terminal()
        elif choice == '5':
            gfp_browser(evokeds)
        elif choice == '6':
            show_sensor_layout(evokeds)
        elif choice == '7':
            regional_grid_plot(evokeds)
        elif choice == 'b':
            apply_baseline_correction(evokeds)
        elif choice == 'f':
            apply_lowpass_filter(evokeds)


if __name__ == '__main__':
    main()