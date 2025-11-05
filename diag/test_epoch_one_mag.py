#!/usr/bin/env python3
# test_epoch_one_mag.py
#
# Plot a single MAG channel's evoked time course from the saved epochs.
# - Auto-selects the MAG with the largest |peak| in a window (default 80–150 ms), or use --channel.
# - Optionally overlays:
#     * the same channel re-epoched from the preproc Raw (should match exactly)
#     * the mean STI101 edge (scaled) for timing reference.
#
# Usage (paths default to your files):
#   python test_epoch_one_mag.py
#   python test_epoch_one_mag.py --channel MEG0121  # choose specific sensor
#
# Gregory’s current files are set as defaults below.

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot a single MAG channel evoked and report which one was used.")
    ap.add_argument(
        "--epochs",
        default="/Users/gm33/data/fairy/derivatives/epochs/sub-01/ses-01/sub-01_ses-01_task-faces_run-01_desc-faces_non_target_epo.fif",
        help="Path to *_desc-<cond>_epo.fif.",
    )
    ap.add_argument(
        "--raw",
        default="/Users/gm33/data/fairy/derivatives/preprocessing/sub-01/ses-01/meg/sub-01_ses-01_task-faces_run-01_desc-preproc_meg.fif",
        help="Path to preprocessed Raw FIF.",
    )
    ap.add_argument("--stim-channel", default="STI101", help="Stim channel name (default: STI101).")
    ap.add_argument("--channel", default=None, help="Specific MAG channel to plot (e.g., MEG0121).")
    ap.add_argument("--auto-window", nargs=2, type=float, default=[0.08, 0.15],
                    help="Seconds window for auto-selection peak search (default: 0.08 0.15).")
    ap.add_argument("--overlay-raw", action="store_true",
                    help="Also overlay the same channel re-epoched from the preproc Raw.")
    ap.add_argument("--overlay-sti", action="store_true",
                    help="Also overlay the mean STI101 edge (scaled).")
    args = ap.parse_args()

    epochs_path = Path(args.epochs).expanduser()
    raw_path = Path(args.raw).expanduser()
    if not epochs_path.exists():
        raise FileNotFoundError(epochs_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    # --- Load saved epochs and compute MAG evoked ---
    ep = mne.read_epochs(str(epochs_path), preload=True, verbose="ERROR")
    ev_mag = ep.average(picks="mag")
    t = ev_mag.times
    data_mag = ev_mag.data  # shape (n_mag, n_times)
    mag_ch_names = [ev_mag.ch_names[i] for i in range(data_mag.shape[0])]

    # --- Choose channel (auto or user-specified) ---
    if args.channel is not None:
        if args.channel not in mag_ch_names:
            raise ValueError(f"Requested channel {args.channel!r} not found among MAG channels.")
        idx = mag_ch_names.index(args.channel)
        chosen = args.channel
    else:
        # Auto-pick: max |peak| in [auto-window]
        w0, w1 = float(args.auto_window[0]), float(args.auto_window[1])
        i0 = int(np.searchsorted(t, w0))
        i1 = int(np.searchsorted(t, w1))
        if i1 <= i0:
            raise ValueError("Bad --auto-window; must have start < end and within epoch time range.")
        peaks = np.max(np.abs(data_mag[:, i0:i1]), axis=1)
        idx = int(np.argmax(peaks))
        chosen = mag_ch_names[idx]

    y_ep = data_mag[idx, :]

    print(f"[info] Chosen MAG channel: {chosen}")
    print(f"[info] Epochs file: {epochs_path.name}")
    print(f"[info] tmin={ep.tmin*1e3:.1f} ms, tmax={ep.tmax*1e3:.1f} ms, baseline={ep.baseline}")
    print(f"[info] n_epochs={len(ep)}")

    # --- Optional overlays from RAW and STI ---
    y_raw = None
    sti_trace = None
    if args.overlay_raw or args.overlay_sti:
        raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
        sf = float(raw.info["sfreq"])

    if args.overlay_raw:
        # Re-epoch RAW for the same MAG channel
        picks_one_mag = mne.pick_channels(raw.ch_names, include=[chosen])
        ep_raw = mne.Epochs(
            raw, ep.events, event_id={"x": 1},
            tmin=ep.tmin, tmax=ep.tmax, baseline=ep.baseline,
            picks=picks_one_mag, preload=True, reject_by_annotation=True, verbose="ERROR"
        )
        ep_raw.events[:, 2] = 1
        ev_raw = ep_raw.average()
        y_raw = ev_raw.data[0, :]  # single channel

    if args.overlay_sti:
        stim_ch = args.stim_channel
        if stim_ch not in raw.ch_names:
            raise RuntimeError(f"Stim channel {stim_ch!r} not found in Raw.")
        picks_sti = mne.pick_channels(raw.ch_names, include=[stim_ch])
        sti_ep = mne.Epochs(
            raw, ep.events, event_id={"x": 1},
            tmin=ep.tmin, tmax=ep.tmax, baseline=None,
            picks=picks_sti, preload=True, reject_by_annotation=False, verbose="ERROR"
        )
        sti_ep.events[:, 2] = 1
        sti_ev = sti_ep.average(picks="stim")
        sti_trace = sti_ev.data[0, :]
        # center and scale STI for display
        sti_trace = sti_trace - np.mean(sti_trace[:max(1, int(0.02 * sf))])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.axvline(0, color="k", lw=1, alpha=0.8)
    ax.plot(t, y_ep, label=f"{chosen} (saved epochs)", color="C0", alpha=0.95)
    if y_raw is not None:
        ax.plot(t, y_raw, label=f"{chosen} (re-epoched raw)", color="C1", ls="--", alpha=0.85)
    if sti_trace is not None:
        scale = (np.nanmax(np.abs(y_ep)) or 1.0) / (np.nanmax(np.abs(sti_trace)) or 1.0)
        ax.plot(t, sti_trace * scale, label=f"{args.stim_channel} (scaled)", color="C2", alpha=0.7)

    ax.set_xlim(ep.tmin, ep.tmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ttl = f"Single MAG evoked: {chosen}"
    ax.set_title(ttl)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()