import sys
import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick summary and visualization tool for MNE FIF files."
    )
    parser.add_argument(
        "file", type=str, help="Path to raw .fif file"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary only, no plots"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Disable interactive raw browser"
    )
    parser.add_argument(
        "--ch-types", type=str, default="mag,grad,eeg",
        help="Comma-separated list of channel types to plot PSD for (default: mag,grad,eeg)"
    )
    parser.add_argument(
        "--save-psd", action="store_true", help="Save PSD plots to disk"
    )
    parser.add_argument(
        "-f", "--lpf", type=float, default=None,
        help="Low-pass filter cutoff (Hz). If specified, applies filter before launching browser"
    )
    return parser.parse_args()

def print_metadata(raw, fname):
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    duration_sec = (n_samples - 1) / sfreq
    ch_types = ['mag', 'grad', 'eeg', 'stim', 'eog', 'ecg']
    print("\n===== METADATA =====")
    print(f"File: {fname}")
    print(f"Sampling frequency: {sfreq:.2f} Hz")
    print(f"Number of samples: {n_samples}")
    print(f"Duration: {duration_sec:.2f} s")
    print(f"Number of channels: {len(raw.ch_names)}")
    for typ in ch_types:
        picks = mne.pick_types(raw.info, meg=(typ if typ in ['mag', 'grad'] else False),
                              eeg=(typ == 'eeg'), stim=(typ == 'stim'),
                              eog=(typ == 'eog'), ecg=(typ == 'ecg'),
                              exclude='bads')
        print(f"  {typ.upper():<5}: {len(picks)} channels")
    bads = raw.info.get('bads', [])
    if bads:
        print(f"Bads: {', '.join(bads)}")
    else:
        print("Bads: []")
    print(f"File start sample (raw.first_samp): {raw.first_samp}")
    print(f"File end sample (raw.last_samp): {getattr(raw, 'last_samp', 'n/a')}")

def plot_psd_group(raw, picks, label, save_psd=False):
    if len(picks) == 0:
        print(f"No {label} channels to plot.")
        return
    print(f"\nCalculating PSD for {label} ({len(picks)} channels)...")
    sfreq = raw.info['sfreq']
    # Full spectrum
    psd = raw.compute_psd(picks=picks, method="welch", fmax=sfreq/2)
    fig_full = psd.plot(show=False)
    fig_full.canvas.manager.set_window_title(f"{label.upper()} PSD: 0–Nyquist Hz (unfiltered)")
    plt.title(f"{label.upper()} PSD: 0–Nyquist Hz (unfiltered)")
    plt.tight_layout()
    if save_psd:
        fig_full.savefig(f"{label}_psd_full_unfiltered.png")
        print(f"Saved: {label}_psd_full_unfiltered.png")
    # 0–50 Hz
    psd_50 = raw.compute_psd(picks=picks, method="welch", fmax=50.0)
    fig_50 = psd_50.plot(show=False)
    fig_50.canvas.manager.set_window_title(f"{label.upper()} PSD: 0–50 Hz (unfiltered)")
    plt.title(f"{label.upper()} PSD: 0–50 Hz (unfiltered)")
    plt.tight_layout()
    if save_psd:
        fig_50.savefig(f"{label}_psd_0_50Hz_unfiltered.png")
        print(f"Saved: {label}_psd_0_50Hz_unfiltered.png")

def main():
    args = parse_args()
    fname = args.file
    if not os.path.exists(fname):
        print(f"ERROR: File '{fname}' does not exist.")
        sys.exit(1)
    if not fname.lower().endswith('.fif'):
        print("WARNING: This script is designed for FIF files. Other formats not supported.")
        # Optionally add more file formats here.

    try:
        print(f"\n=== Loading file: {fname} ===")
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=True)
    except Exception as e:
        print(f"ERROR reading {fname}: {e}")
        sys.exit(1)

    print_metadata(raw, fname)

    if args.summary:
        print("Summary only (--summary): Skipping plots.")
        return

    plt.ion()

    # === PSD on UNFILTERED data ===
    ch_types = [t.strip() for t in args.ch_types.split(',') if t.strip()]
    valid_types = ['mag', 'grad', 'eeg', 'stim', 'eog', 'ecg']
    for typ in ch_types:
        if typ not in valid_types:
            print(f"WARNING: Unknown channel type '{typ}'. Skipping.")
            continue
        picks = mne.pick_types(raw.info, meg=(typ if typ in ['mag', 'grad'] else False),
                              eeg=(typ == 'eeg'), stim=(typ == 'stim'),
                              eog=(typ == 'eog'), ecg=(typ == 'ecg'),
                              exclude='bads')
        plot_psd_group(raw, picks, typ, save_psd=args.save_psd)

    # Draw all open figures (non-blocking)
    plt.show(block=False)
    time.sleep(0.5)

    # === NOW apply LPF if requested, only for browser ===
    if args.lpf is not None:
        print(f"\nApplying low-pass filter at {args.lpf:.1f} Hz for browser display...")
        raw.load_data()
        raw.filter(None, args.lpf, fir_design='firwin', verbose=True)
    else:
        print("\nNo filtering applied to browser.")

    if not args.no_browser:
        print("\nLaunching interactive raw browser. Close this window to end the program and all figures.")
        raw.plot(title="Interactive Raw Browser" +
                 (f" (LPF: {args.lpf} Hz)" if args.lpf else ""))
    else:
        print("--no-browser specified: Skipping interactive browser.")

    input("Press Enter after closing all figures to exit completely...")
    plt.close('all')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        plt.close('all')
        sys.exit(0)
