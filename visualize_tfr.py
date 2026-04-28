# visualize_tfr.py
import sys
import warnings
from pathlib import Path
import mne
import matplotlib.pyplot as plt

# Suppress harmless Matplotlib masking warnings and MNE baseline chatter
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib.cbook")
mne.set_log_level('WARNING')


def handle_global_commands(user_input):
    if user_input.lower() in ('c', 'close'):
        plt.close('all')
        print("All plot windows closed.")
        return True
    return False


def select_channel_type(tfr):
    avail = [t for t in ('mag', 'grad', 'eeg') if t in set(tfr.get_channel_types())]
    if not avail:
        print("No valid MEG/EEG channels found.")
        return None

    print("\nSelect channel type to plot:")
    for i, t in enumerate(avail, 1):
        print(f"  ({i}) {t.upper()}")

    while True:
        choice = input("Choice (or 'q' to cancel): ").strip().lower()
        if choice == 'q': return None
        if choice.isdigit() and 1 <= int(choice) <= len(avail):
            return avail[int(choice) - 1]
        print("Invalid selection.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_tfr.py <path_to_tfr_file.h5>")
        sys.exit(1)

    tfr_file = Path(sys.argv[1])
    if not tfr_file.exists():
        print(f"File not found: {tfr_file}")
        sys.exit(1)

    print(f"Loading TFR data from {tfr_file.name}...")

    # Robust loading: handles both lists and single TFR objects
    loaded_data = mne.time_frequency.read_tfrs(str(tfr_file))
    tfr = loaded_data[0] if isinstance(loaded_data, list) else loaded_data

    # Extract clean condition name from filename (e.g., 'faces_target')
    try:
        cond_name = tfr_file.name.split("desc-")[1].replace("_tfr.h5", "")
    except IndexError:
        cond_name = tfr_file.stem

    print(f"\nLoaded Condition: {cond_name}")
    print(f"Frequencies: {tfr.freqs[0]:.1f} Hz to {tfr.freqs[-1]:.1f} Hz")
    print(f"Time window: {tfr.times[0] * 1000:.1f} ms to {tfr.times[-1] * 1000:.1f} ms")

    while True:
        print("\n=== TFR Visualization Menu ===")
        print("  (1) Interactive Topo (Click sensors to expand)")
        print("  (2) Joint Plot (Summary heatmap + topomaps)")
        print("  (3) Plot Average over Channel Type")
        print("  (c) Close all plots")
        print("  (q) Quit")

        choice = input("Select: ").strip().lower()

        if handle_global_commands(choice): continue
        if choice == 'q': break

        if choice in ('1', '2', '3'):
            ch_type = select_channel_type(tfr)
            if not ch_type: continue

            # Diverging colormap for logratio dB data
            cmap = 'RdBu_r'

            try:
                if choice == '1':
                    print("Generating Interactive Topo... (Click on a mini-plot to expand)")
                    fig = tfr.plot_topo(picks=ch_type, title=f"Topo: {cond_name} ({ch_type.upper()})",
                                        cmap=cmap, baseline=None, show=False)
                    plt.show(block=False)

                elif choice == '2':
                    print("\n--- Joint Plot Topomaps ---")
                    print("By default, MNE plots a single topomap at the absolute maximum peak.")
                    print("To plot multiple, enter pairs of Time(s),Freq(Hz) separated by semicolons.")
                    print("Example: 0.1,10 ; 0.35,4 ; 0.4,25")
                    tf_input = input("Enter points (or press Enter for default max peak): ").strip()

                    timefreqs = None
                    if tf_input:
                        try:
                            timefreqs = []
                            for pair in tf_input.split(';'):
                                t_str, f_str = pair.split(',')
                                timefreqs.append((float(t_str.strip()), float(f_str.strip())))
                        except Exception:
                            print("Invalid format. Falling back to default maximum peak.")
                            timefreqs = None

                    print("Generating Joint Plot...")
                    fig = tfr.plot_joint(picks=ch_type, title=f"Joint Plot: {cond_name} ({ch_type.upper()})",
                                         timefreqs=timefreqs, baseline=None, show=False)
                    plt.show(block=False)

                elif choice == '3':
                    print("Generating Average Plot...")
                    picks = mne.pick_types(tfr.info, meg=ch_type if ch_type in ('mag', 'grad') else False,
                                           eeg=(ch_type == 'eeg'))

                    fig = tfr.plot(picks=picks, combine='mean', title=f"Average Power: {cond_name} ({ch_type.upper()})",
                                   cmap=cmap, baseline=None, show=False)

                    if isinstance(fig, list):
                        for f in fig: f.show()
                    else:
                        plt.show(block=False)

            except Exception as e:
                print(f"An error occurred while plotting: {e}")


if __name__ == "__main__":
    main()