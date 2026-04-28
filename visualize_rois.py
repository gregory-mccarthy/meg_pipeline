# visualize_rois.py
import sys
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne

print("[viz] Matplotlib backend:", matplotlib.get_backend())


def handle_global_commands(user_input):
    if user_input.lower() == 'c':
        plt.close('all')
        print("All plot windows closed.")
        return True
    return False


def focus_terminal():
    if sys.platform == "darwin":
        script = 'tell application "System Events" to set frontmost of process "Terminal" to true'
        try:
            subprocess.run(['osascript', '-e', script], check=True)
        except Exception:
            pass


def prompt_for_conditions(evokeds, allow_difference=False, allow_all=True, allow_multi_overlay=True):
    print("\nAvailable conditions:")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment}")
    if allow_all:
        print("  (all) All conditions")
    if allow_difference:
        print("  Difference mode: enter A,-B to plot condition A minus condition B (e.g., 1,-2 or 2,-1)")

    while True:
        if allow_difference and allow_all and allow_multi_overlay:
            prompt = (
                "Enter condition numbers (comma-separated), 'all' for all conditions, "
                "or A,-B for a difference wave; 'c' to close plots, or 'q' to quit: "
            )
        elif allow_difference and allow_multi_overlay:
            prompt = (
                "Enter condition numbers (comma-separated) or A,-B for a difference wave; "
                "'c' to close plots, or 'q' to quit: "
            )
        elif allow_difference and allow_all:
            prompt = (
                "Enter ONE condition number, 'all' for all conditions, or A,-B for a difference wave; "
                "'c' to close plots, or 'q' to quit: "
            )
        elif allow_difference:
            prompt = (
                "Enter ONE condition number or A,-B for a difference wave; "
                "'c' to close plots, or 'q' to quit: "
            )
        elif allow_all and allow_multi_overlay:
            prompt = (
                "Enter condition numbers (comma-separated), 'all' for all conditions, "
                "'c' to close plots, or 'q' to quit: "
            )
        elif allow_all:
            prompt = "Enter ONE condition number, 'all' for all conditions, 'c' to close plots, or 'q' to quit: "
        elif allow_multi_overlay:
            prompt = "Enter condition numbers (comma-separated), 'c' to close plots, or 'q' to quit: "
        else:
            prompt = "Enter ONE condition number, 'c' to close plots, or 'q' to quit: "

        cond_str = input(prompt).strip()
        if handle_global_commands(cond_str):
            continue
        if cond_str.lower() == 'q':
            return None
        if allow_all and cond_str.lower() == 'all':
            return {'mode': 'overlay', 'indices': list(range(len(evokeds)))}

        tokens = [token.strip() for token in cond_str.split(',') if token.strip()]
        if not tokens:
            print("Invalid selection. Please enter at least one condition number.")
            continue

        try:
            raw_values = [int(token) for token in tokens]
        except ValueError:
            raw_values = None

        if raw_values is not None:
            if allow_difference and len(raw_values) == 2 and raw_values[0] > 0 and raw_values[1] < 0:
                first_idx = raw_values[0] - 1
                second_idx = abs(raw_values[1]) - 1
                if 0 <= first_idx < len(evokeds) and 0 <= second_idx < len(evokeds):
                    return {'mode': 'difference', 'indices': [first_idx, second_idx]}

            if all(value > 0 for value in raw_values):
                if allow_multi_overlay or len(raw_values) == 1:
                    cond_indices = [value - 1 for value in raw_values]
                    if all(0 <= idx < len(evokeds) for idx in cond_indices):
                        return {'mode': 'overlay', 'indices': cond_indices}

        if allow_difference:
            if allow_multi_overlay:
                print(
                    "Invalid selection. Use positive condition numbers for overlays (e.g., 1,2,3), "
                    "or exactly two entries in the form A,-B for a difference wave (e.g., 1,-2 or 2,-1). "
                    "Mixed selections such as 1,2,-3 are not supported."
                )
            else:
                print(
                    "Invalid selection. Use ONE positive condition number, or exactly two entries in the form "
                    "A,-B for a difference wave (e.g., 1,-2 or 2,-1)."
                )
        elif allow_all:
            if allow_multi_overlay:
                print("Invalid selection. Please choose from the available condition numbers, or enter 'all'.")
            else:
                print("Invalid selection. Please choose ONE available condition number, or enter 'all'.")
        else:
            if allow_multi_overlay:
                print("Invalid selection. Please choose from the available condition numbers.")
            else:
                print("Invalid selection. Please choose ONE available condition number.")


def build_plot_evokeds(evokeds, condition_selection):
    if condition_selection['mode'] == 'difference':
        first_idx, second_idx = condition_selection['indices']
        diff_evoked = mne.combine_evoked([evokeds[first_idx], evokeds[second_idx]], weights=[1, -1])
        diff_evoked.comment = f"{evokeds[first_idx].comment} - {evokeds[second_idx].comment}"
        return [diff_evoked]

    return [evokeds[idx] for idx in condition_selection['indices']]


def compute_global_ylim(plot_evokeds, ch_indices=None):
    global_min = float('inf')
    global_max = float('-inf')
    indexer = slice(None) if ch_indices is None else np.atleast_1d(ch_indices)

    for ev in plot_evokeds:
        data = ev.data[indexer, :]
        global_min = min(global_min, np.min(data))
        global_max = max(global_max, np.max(data))

    padding = (global_max - global_min) * 0.05 if global_max > global_min else 1.0
    return [global_min - padding, global_max + padding]


def prompt_for_parcels(ch_names):
    while True:
        search_str = input("Enter parcel name/substring (e.g., 'fusiform' or 'rh'), 'c' close, 'q' quit: ").strip()
        if handle_global_commands(search_str):
            continue
        if search_str.lower() == 'q':
            return None

        matches = [i for i, name in enumerate(ch_names) if search_str.lower() in name.lower()]

        if matches:
            print(f"  -> Found {len(matches)} matching parcels.")
            return matches
        else:
            print("  -> No matching parcels found. Try again.")


def prompt_for_axis_limits(default_xlim_ms, default_ylim):
    rounded_xlim = [int(default_xlim_ms[0]), int(default_xlim_ms[1])]
    rounded_ylim = [round(default_ylim[0], 2), round(default_ylim[1], 2)]

    while True:
        xlim_in = input(f"Enter X limits in ms (start,end) [default {rounded_xlim[0]},{rounded_xlim[1]}]: ").strip()
        if handle_global_commands(xlim_in):
            continue

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

        ylim_in = input(f"Enter Y limits (min,max) [default {rounded_ylim[0]},{rounded_ylim[1]}]: ").strip()
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
    if handle_global_commands(freq_in):
        return
    if freq_in.lower() == 'q':
        return
    try:
        h_freq = float(freq_in)
        for ev in evokeds:
            # We use FIR design for standard filtering in MNE
            ev.filter(l_freq=None, h_freq=h_freq, fir_design='firwin')
        print(f"Successfully applied {h_freq} Hz low-pass filter to all conditions.")
    except ValueError:
        print("Invalid frequency entered. Filter application cancelled.")


def apply_baseline_correction(evokeds):
    print("\n=== Apply Baseline Correction ===")
    while True:
        base_in = input(
            "Enter baseline period in ms (start,end) [e.g., -200,0], 'c' to close plots, or 'q' to cancel: ").strip()
        if handle_global_commands(base_in):
            return
        if base_in.lower() == 'q':
            return

        parts = base_in.split(',')
        if len(parts) != 2:
            print("Invalid input. Please provide exactly two values separated by a comma (e.g., -200,0).")
            continue

        try:
            start_ms, end_ms = [float(x) for x in parts]
            # MNE requires baseline in seconds, so we divide the ms inputs by 1000
            baseline = (start_ms / 1000.0, end_ms / 1000.0)
            for ev in evokeds:
                ev.apply_baseline(baseline)
            print(f"Successfully applied baseline correction ({start_ms} ms to {end_ms} ms) to all conditions.")
            break
        except ValueError:
            print("Invalid format. Please use numbers only.")


def waveform_browser(evokeds):
    print("\n=== Parcel Waveform Browser ===")
    condition_selection = prompt_for_conditions(evokeds, allow_difference=True)
    if condition_selection is None:
        return

    plot_evokeds = build_plot_evokeds(evokeds, condition_selection)
    ch_names = evokeds[0].ch_names
    ch_idx = 0
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

    xlim_ms = [evokeds[0].times[0] * 1000, evokeds[0].times[-1] * 1000]

    print("  -> Calculating global amplitude scale across all parcels...")
    current_ylim = compute_global_ylim(plot_evokeds)
    print(f"  -> Global scale set to: [{current_ylim[0]:.2f}, {current_ylim[1]:.2f}]")

    while True:
        parcel_name = ch_names[ch_idx]

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, ev in enumerate(plot_evokeds):
            times_ms = ev.times * 1000
            signal = ev.data[ch_idx, :]
            ax.plot(times_ms, signal, label=ev.comment, color=colors[i % len(colors)], linewidth=2.0)

        ax.set_ylim(current_ylim)
        ax.set_xlim(xlim_ms)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Amplitude (dSPM Pseudo-F)', fontsize=12)
        ax.set_title(f'Anatomical ROI: {parcel_name}', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show(block=False)
        focus_terminal()

        need_redraw = False
        while not need_redraw:
            print(f"\n[Parcel: {parcel_name}] Options:")
            print("  [Enter]=Next | [p]=Prev | [Search text]=Jump to parcel")
            print("  [a]=Axis X/Y | [d]=Change Conds | [c]=Close plots | [q]=Quit")
            nav = input("Choice: ").strip().lower()

            if handle_global_commands(nav):
                continue
            if nav == 'q':
                return

            elif nav == '' or nav == 'n':
                ch_idx = (ch_idx + 1) % len(ch_names)
                need_redraw = True
            elif nav == 'p':
                ch_idx = (ch_idx - 1) % len(ch_names)
                need_redraw = True
            elif nav in ['axis', 'a', 'scale', 's']:
                xlim_ms, current_ylim = prompt_for_axis_limits(xlim_ms, current_ylim)
                need_redraw = True
            elif nav in ['cond', 'd']:
                new_selection = prompt_for_conditions(evokeds, allow_difference=True)
                if new_selection is not None:
                    condition_selection = new_selection
                    plot_evokeds = build_plot_evokeds(evokeds, condition_selection)

                    print("  -> Recalculating global amplitude scale across all parcels...")
                    current_ylim = compute_global_ylim(plot_evokeds)
                    print(f"  -> Global scale updated to: [{current_ylim[0]:.2f}, {current_ylim[1]:.2f}]")
                    need_redraw = True
            else:
                matches = [i for i, name in enumerate(ch_names) if nav in name.lower()]
                if matches:
                    ch_idx = matches[0]
                    need_redraw = True
                else:
                    print(f"Unknown command or parcel: '{nav}'")


def multi_parcel_overlay(evokeds):
    print("\n=== Multi-Parcel Overlay ===")

    condition_selection = prompt_for_conditions(
        evokeds,
        allow_difference=True,
        allow_all=False,
        allow_multi_overlay=False,
    )
    if condition_selection is None:
        return

    ev = build_plot_evokeds(evokeds, condition_selection)[0]
    ch_names = ev.ch_names
    indices = prompt_for_parcels(ch_names)
    if not indices:
        return

    xlim_ms = [ev.times[0] * 1000, ev.times[-1] * 1000]
    current_ylim = compute_global_ylim([ev], indices)

    while True:
        fig, ax = plt.subplots(figsize=(12, 7))

        for idx in indices:
            signal = ev.data[idx, :]
            ax.plot(ev.times * 1000, signal, label=ch_names[idx], linewidth=1.5)

        ax.set_title(f"Overlay: {ev.comment}", fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel("Amplitude (dSPM Pseudo-F)", fontsize=12)
        ax.set_xlim(xlim_ms)
        ax.set_ylim(current_ylim)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        if len(indices) > 10:
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
            plt.subplots_adjust(right=0.7)
        else:
            ax.legend()

        plt.show(block=False)
        focus_terminal()

        need_redraw = False
        while not need_redraw:
            print("\nOptions: [Type search text to add/change parcels] | [a]=Axis X/Y | [d]=Change Cond | [c]=close | [q]=quit")
            nav = input("Choice: ").strip().lower()

            if handle_global_commands(nav):
                continue
            if nav == 'q':
                return
            if nav in ['axis', 'a', 'scale', 's']:
                xlim_ms, current_ylim = prompt_for_axis_limits(xlim_ms, current_ylim)
                need_redraw = True
            elif nav in ['cond', 'd']:
                new_selection = prompt_for_conditions(
                    evokeds,
                    allow_difference=True,
                    allow_all=False,
                    allow_multi_overlay=False,
                )
                if new_selection is not None:
                    condition_selection = new_selection
                    ev = build_plot_evokeds(evokeds, condition_selection)[0]
                    xlim_ms = [ev.times[0] * 1000, ev.times[-1] * 1000]
                    current_ylim = compute_global_ylim([ev], indices)
                    need_redraw = True
            elif nav:
                new_indices = prompt_for_parcels(ch_names)
                if new_indices:
                    indices = new_indices
                    current_ylim = compute_global_ylim([ev], indices)
                    need_redraw = True


def lobe_grid_plot(evokeds):
    aparc_lobe_mapping = {
        'Occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
        'Parietal': ['inferiorparietal', 'postcentral', 'precuneus', 'superiorparietal', 'supramarginal'],
        'Temporal': ['bankssts', 'entorhinal', 'fusiform', 'inferiortemporal', 'middletemporal', 'parahippocampal',
                     'superiortemporal', 'temporalpole', 'transversetemporal'],
        'Frontal': ['caudalmiddlefrontal', 'frontalpole', 'lateralorbitofrontal', 'medialorbitofrontal', 'paracentral',
                    'parsopercularis', 'parsorbitalis', 'parstriangularis', 'precentral', 'rostralmiddlefrontal',
                    'superiorfrontal'],
        'Cingulate': ['caudalanteriorcingulate', 'isthmuscingulate', 'posteriorcingulate', 'rostralanteriorcingulate'],
        'Insula': ['insula']
    }
    lobes = list(aparc_lobe_mapping.keys())

    # Outer loop to allow re-selecting the lobe
    while True:
        print("\n=== Anatomical Lobe Grid Plot ===")
        for i, lobe in enumerate(lobes, 1):
            print(f"  ({i}) {lobe}")

        lobe_in = input("\nSelect lobe number, 'c' to close plots, or 'q' to return to menu: ").strip()
        if handle_global_commands(lobe_in):
            continue
        if lobe_in.lower() == 'q':
            return

        try:
            selected_lobe = lobes[int(lobe_in) - 1]
        except (ValueError, IndexError):
            print("Invalid lobe selection.")
            continue

        print("\nSelect Hemisphere:")
        print("  (1) Both (Left & Right)")
        print("  (2) Left (-lh only)")
        print("  (3) Right (-rh only)")

        hemi_in = input("Select (1/2/3), 'c' close, 'q' return to menu: ").strip()
        if handle_global_commands(hemi_in):
            continue
        if hemi_in.lower() == 'q':
            return

        hemi_suffixes = {'1': ['-lh', '-rh'], '2': ['-lh'], '3': ['-rh']}.get(hemi_in)
        if not hemi_suffixes:
            print("Invalid hemisphere selection.")
            continue

        condition_selection = prompt_for_conditions(evokeds, allow_difference=True)
        if condition_selection is None:
            continue
        plot_evokeds = build_plot_evokeds(evokeds, condition_selection)

        ch_names = evokeds[0].ch_names
        base_parcels = aparc_lobe_mapping[selected_lobe]

        target_indices = []
        for i, ch_name in enumerate(ch_names):
            if len(ch_name) > 3 and ch_name[-3:] in ['-lh', '-rh']:
                base_name = ch_name[:-3]
                suffix = ch_name[-3:]
            else:
                base_name = ch_name
                suffix = ''

            if base_name in base_parcels and (suffix in hemi_suffixes or not suffix):
                target_indices.append(i)

        if not target_indices:
            print(f"No parcels found for the {selected_lobe} lobe matching your criteria.")
            continue

        current_ylim = compute_global_ylim(plot_evokeds, target_indices)
        xlim_ms = [evokeds[0].times[0] * 1000, evokeds[0].times[-1] * 1000]

        # Inner loop for plotting and tweaking axes/conditions for the currently selected lobe
        while True:
            n_plots = len(target_indices)
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(14, 10), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).flatten()

            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

            for i, ch_idx in enumerate(target_indices):
                ax = axes[i]
                ch_name = ch_names[ch_idx]

                for j, ev in enumerate(plot_evokeds):
                    times_ms = ev.times * 1000
                    signal = ev.data[ch_idx, :]
                    ax.plot(times_ms, signal, label=ev.comment, color=colors[j % len(colors)], linewidth=1.5)

                ax.set_title(ch_name, fontsize=10, pad=3)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax.set_xlim(xlim_ms)
                ax.set_ylim(current_ylim)
                ax.grid(True, alpha=0.3)

                if i == 0:
                    ax.legend(fontsize=8, loc='best')

            for i in range(len(target_indices), len(axes)):
                axes[i].set_visible(False)

            fig.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=12)
            fig.text(0.04, 0.5, 'Amplitude (dSPM Pseudo-F)', va='center', rotation='vertical', fontsize=12)

            hemi_label = "Both Hemispheres" if hemi_in == '1' else (
                "Left Hemisphere" if hemi_in == '2' else "Right Hemisphere")
            fig.suptitle(f"{selected_lobe} Lobe ROI Waveforms ({hemi_label})", fontsize=14, y=0.97)

            plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
            plt.show(block=False)
            focus_terminal()

            need_redraw = False
            change_lobe = False
            while not need_redraw:
                print(
                    f"\n[{selected_lobe} Lobe] Options: [l]=Change Lobe | [a]=Axis X/Y | [d]=Change Conds | [c]=Close plots | [q]=Quit to Menu")
                nav = input("Choice: ").strip().lower()

                if handle_global_commands(nav):
                    continue
                if nav == 'q':
                    return

                if nav == 'l':
                    change_lobe = True
                    break
                elif nav in ['axis', 'a', 'scale', 's']:
                    xlim_ms, current_ylim = prompt_for_axis_limits(xlim_ms, current_ylim)
                    need_redraw = True
                elif nav in ['cond', 'd']:
                    new_selection = prompt_for_conditions(evokeds, allow_difference=True)
                    if new_selection is not None:
                        condition_selection = new_selection
                        plot_evokeds = build_plot_evokeds(evokeds, condition_selection)

                        print("  -> Recalculating amplitude scale for selected lobe/parcels...")
                        current_ylim = compute_global_ylim(plot_evokeds, target_indices)
                        print(f"  -> Scale updated to: [{current_ylim[0]:.2f}, {current_ylim[1]:.2f}]")
                        need_redraw = True

            if change_lobe:
                break  # Breaks out of the redraw loop and returns to the outer lobe selection


def plot_heatmap(evokeds):
    print("\n=== Parcel Heatmap Image ===")
    condition_selection = prompt_for_conditions(evokeds, allow_difference=True)
    if condition_selection is None:
        return

    plot_evokeds = build_plot_evokeds(evokeds, condition_selection)

    for ev in plot_evokeds:
        ev.plot_image(titles=f"Heatmap: {ev.comment}", show=False, time_unit='ms', cmap='inferno')

    plt.show(block=False)
    focus_terminal()


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_rois.py <path_to_rois-ave.fif>")
        sys.exit(1)

    evoked_fname = sys.argv[1]

    try:
        evokeds = mne.read_evokeds(evoked_fname, verbose=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"\nLoaded {len(evokeds)} evoked conditions (Virtual Sensors):")
    for i, ev in enumerate(evokeds, 1):
        print(f"  ({i}) {ev.comment} (nave={ev.nave})")

    while True:
        print("\nVirtual Sensor Menu:")
        print("  (1) Browse Individual Parcels")
        print("  (2) Overlay Multiple Parcels")
        print("  (3) Plot Parcel Heatmap (All Regions)")
        print("  (4) Lobe-based Regional Grid Plot")
        print("  (b) Apply baseline correction")
        print("  (f) Apply low-pass filter")
        print("  (c) Close all open plots")
        print("  (q) Quit")

        choice = input("Select: ").strip().lower()

        if handle_global_commands(choice):
            continue
        if choice == 'q':
            break

        elif choice == '1':
            waveform_browser(evokeds)
        elif choice == '2':
            multi_parcel_overlay(evokeds)
        elif choice == '3':
            plot_heatmap(evokeds)
        elif choice == '4':
            lobe_grid_plot(evokeds)
        elif choice == 'b':
            apply_baseline_correction(evokeds)
        elif choice == 'f':
            apply_lowpass_filter(evokeds)


if __name__ == '__main__':
    main()
