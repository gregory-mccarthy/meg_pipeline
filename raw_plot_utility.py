import sys
import os
import argparse
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick summary and visualization tool for MNE FIF files (raw or processed)."
    )
    parser.add_argument("file", type=str, help="Path to raw/processed .fif file")

    parser.add_argument("--summary", action="store_true", help="Print summary only, no plots")
    parser.add_argument("--no-browser", action="store_true", help="Disable interactive raw browser")

    # --- NEW: BIDS Events TSV flag ---
    parser.add_argument(
        "--read-events-tsv",
        action="store_true",
        help="Look for and apply a companion _events.tsv file to annotate the raw data (shades BAD_breaks)."
    )

    parser.add_argument(
        "--ch-types",
        type=str,
        default="mag,grad,eeg",
        help="Comma-separated list of channel types to plot PSD for (default: mag,grad,eeg)",
    )
    parser.add_argument("--save-psd", action="store_true", help="Save PSD plots to disk")

    # Filtering for browser display
    parser.add_argument("--hpf", type=float, default=None, help="High-pass cutoff (Hz) for browser display")
    parser.add_argument("-f", "--lpf", type=float, default=None, help="Low-pass cutoff (Hz) for browser display")

    # Robust scaling controls for the browser
    parser.add_argument(
        "--scale-window-sec",
        type=float,
        default=60.0,
        help="Seconds of data (from start of file) used to estimate robust plot scalings (default: 60)",
    )
    parser.add_argument(
        "--scale-abs-quantile",
        type=float,
        default=0.99,
        help="Quantile of |signal| per channel used as a magnitude estimate (default: 0.99)",
    )
    parser.add_argument(
        "--scale-channel-quantile",
        type=float,
        default=0.80,
        help=(
            "Across-channel quantile of per-channel magnitude used as the display scaling (default: 0.80). "
            "Lower values ignore more outliers."
        ),
    )
    parser.add_argument(
        "--scale-mult",
        type=float,
        default=1.2,
        help="Multiplier applied to the robust scaling estimate (default: 1.2). Increase if traces look clipped.",
    )

    return parser.parse_args()


def _picks_for_type(info, typ: str):
    """Return picks for a simple channel type string."""
    typ = typ.lower().strip()
    return mne.pick_types(
        info,
        meg=(typ if typ in ["mag", "grad"] else False),
        eeg=(typ == "eeg"),
        stim=(typ == "stim"),
        eog=(typ == "eog"),
        ecg=(typ == "ecg"),
        misc=(typ == "misc"),
        exclude="bads",
    )


def print_metadata(raw, fname):
    sfreq = float(raw.info["sfreq"])
    n_samples = int(raw.n_times)
    duration_sec = (n_samples - 1) / sfreq if n_samples > 1 else 0.0
    ch_types = ["mag", "grad", "eeg", "stim", "eog", "ecg", "misc"]

    print("\n===== METADATA =====")
    print(f"File: {fname}")
    print(f"Sampling frequency: {sfreq:.2f} Hz")
    print(f"Number of samples: {n_samples}")
    print(f"Duration: {duration_sec:.2f} s")
    print(f"Number of channels: {len(raw.ch_names)}")

    for typ in ch_types:
        picks = _picks_for_type(raw.info, typ)
        if len(picks) > 0:
            print(f"  {typ.upper():<5}: {len(picks)} channels")

    bads = raw.info.get("bads", []) or []
    print(f"Bads: {bads}")

    print(f"File start sample (raw.first_samp): {raw.first_samp}")
    print(f"File end sample (raw.last_samp): {getattr(raw, 'last_samp', 'n/a')}")


def apply_bids_annotations(raw, fif_path):
    """
    Looks for a companion _events.tsv file. If found, converts the rows
    into MNE Annotations and attaches them to the raw object.
    """
    base_dir = os.path.dirname(fif_path)
    base_name = os.path.basename(fif_path)

    if base_name.endswith('_meg.fif'):
        tsv_name = base_name.replace('_meg.fif', '_events.tsv')
    else:
        tsv_name = base_name.replace('.fif', '_events.tsv')

    tsv_path = os.path.join(base_dir, tsv_name)

    if not os.path.exists(tsv_path):
        print(f"\nWARNING: --read-events-tsv was specified, but no TSV was found at:\n  {tsv_path}")
        print("Continuing without annotations...")
        return raw

    print(f"\nFound companion TSV. Reading events from:\n  {tsv_path}")
    try:
        df = pd.read_csv(tsv_path, sep='\t')

        # Verify it has the required BIDS columns
        if all(col in df.columns for col in ['onset', 'duration', 'trial_type']):
            # MNE expects these as pure numpy arrays/lists
            onsets = df['onset'].values
            durations = df['duration'].values
            descriptions = df['trial_type'].values

            # Create and set the annotations
            annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
            raw.set_annotations(annotations)

            # Count the BAD breaks specifically for user feedback
            bad_count = sum('BAD' in desc.upper() for desc in descriptions)
            print(f"Successfully applied {len(onsets)} annotations ({bad_count} marked as BAD).")
        else:
            print(
                "WARNING: TSV is missing required BIDS columns ('onset', 'duration', 'trial_type'). Skipping annotations.")

    except Exception as e:
        print(f"ERROR: Failed to read or apply annotations from TSV: {e}")

    return raw


def plot_psd_group(raw, picks, label, save_psd=False):
    if len(picks) == 0:
        print(f"No {label} channels to plot.")
        return

    print(f"\nCalculating PSD for {label} ({len(picks)} channels)...")
    sfreq = float(raw.info["sfreq"])

    # Full spectrum (0–Nyquist)
    psd = raw.compute_psd(picks=picks, method="welch", fmax=sfreq / 2)
    fig_full = psd.plot(show=False)
    try:
        fig_full.canvas.manager.set_window_title(f"{label.upper()} PSD: 0–Nyquist Hz (unfiltered)")
    except Exception:
        pass
    plt.title(f"{label.upper()} PSD: 0–Nyquist Hz (unfiltered)")
    plt.tight_layout()
    if save_psd:
        out = f"{label}_psd_full_unfiltered.png"
        fig_full.savefig(out)
        print(f"Saved: {out}")

    # 0–50 Hz
    psd_50 = raw.compute_psd(picks=picks, method="welch", fmax=50.0)
    fig_50 = psd_50.plot(show=False)
    try:
        fig_50.canvas.manager.set_window_title(f"{label.upper()} PSD: 0–50 Hz (unfiltered)")
    except Exception:
        pass
    plt.title(f"{label.upper()} PSD: 0–50 Hz (unfiltered)")
    plt.tight_layout()
    if save_psd:
        out = f"{label}_psd_0_50Hz_unfiltered.png"
        fig_50.savefig(out)
        print(f"Saved: {out}")


def _robust_scaling_for_type(raw, picks, window_sec, abs_q, ch_q, mult):
    """Compute a robust display scaling for a channel set, downweighting outliers."""
    if len(picks) == 0:
        return None

    sfreq = float(raw.info["sfreq"])
    n = int(min(raw.n_times, max(1, round(window_sec * sfreq))))

    data = raw.get_data(picks=picks, start=0, stop=n)  # (n_ch, n_t)
    if data.size == 0:
        return None

    # Per-channel magnitude estimate: quantile of absolute value
    per_ch = np.quantile(np.abs(data), abs_q, axis=1)

    # Robust across channels: ignore top outliers via across-channel quantile
    scale = float(np.quantile(per_ch, ch_q) * mult)

    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale


def compute_browser_scalings(raw_browser, args):
    """Compute scalings dict for raw.plot that is robust to outlier channels."""
    scalings = {}
    for typ in ["mag", "grad", "eeg", "eog", "ecg", "stim", "misc"]:
        picks = _picks_for_type(raw_browser.info, typ)
        scale = _robust_scaling_for_type(
            raw_browser,
            picks,
            window_sec=args.scale_window_sec,
            abs_q=args.scale_abs_quantile,
            ch_q=args.scale_channel_quantile,
            mult=args.scale_mult,
        )
        if scale is not None:
            scalings[typ] = scale
    return scalings


def main():
    args = parse_args()
    fname = args.file

    if not os.path.exists(fname):
        print(f"ERROR: File '{fname}' does not exist.")
        sys.exit(1)

    if not fname.lower().endswith(".fif"):
        print("WARNING: This script is designed for FIF files. Other formats may not be supported.")

    try:
        print(f"\n=== Loading file: {fname} ===")
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=True)
    except Exception as e:
        print(f"ERROR reading {fname}: {e}")
        sys.exit(1)

    # --- NEW: Check for and apply BIDS Annotations ---
    if args.read_events_tsv:
        raw = apply_bids_annotations(raw, fname)

    print_metadata(raw, fname)

    if args.summary:
        print("Summary only (--summary): Skipping plots.")
        return

    plt.ion()

    # === PSD on UNFILTERED data ===
    ch_types = [t.strip() for t in args.ch_types.split(",") if t.strip()]
    valid_types = ["mag", "grad", "eeg", "stim", "eog", "ecg", "misc"]
    for typ in ch_types:
        if typ not in valid_types:
            print(f"WARNING: Unknown channel type '{typ}'. Skipping.")
            continue
        picks = _picks_for_type(raw.info, typ)
        plot_psd_group(raw, picks, typ, save_psd=args.save_psd)

    plt.show(block=False)
    time.sleep(0.3)

    # === Prepare a filtered COPY for the interactive browser ===
    if (args.hpf is not None) or (args.lpf is not None):
        print(
            "\nPreparing filtered copy for browser display: "
            f"HPF={args.hpf if args.hpf is not None else 'None'} Hz, "
            f"LPF={args.lpf if args.lpf is not None else 'None'} Hz"
        )
        raw_browser = raw.copy().load_data()
        raw_browser.filter(l_freq=args.hpf, h_freq=args.lpf, fir_design="firwin", verbose=True)
    else:
        raw_browser = raw.copy().load_data()
        print("\nNo filtering requested; loaded data for robust browser scaling.")

    scalings = compute_browser_scalings(raw_browser, args)
    if scalings:
        print("\nRobust browser scalings (units of each channel type):")
        for k in sorted(scalings.keys()):
            print(f"  {k:<5}: {scalings[k]:.3g}")
    else:
        print("\nRobust browser scalings: (none computed; using MNE defaults)")

    if not args.no_browser:
        title_bits = ["Interactive Raw Browser"]
        if args.hpf is not None or args.lpf is not None:
            title_bits.append(f"(HPF={args.hpf}, LPF={args.lpf})")
        print("\nLaunching interactive raw browser. Close this window to end the program and all figures.")
        raw_browser.plot(
            title=" ".join(title_bits),
            remove_dc=True,
            scalings=scalings if scalings else None,
        )
    else:
        print("--no-browser specified: Skipping interactive browser.")

    input("Press Enter after closing all figures to exit completely...")
    plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        plt.close("all")
        sys.exit(0)