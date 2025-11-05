#!/usr/bin/env python3
# test_epoch_timing_single_mag.py
#
# Overlay ONE magnetometer channel from three sources, aligned to the same events:
#   (A) Saved evoked (-ave.fif)   OR   evoked computed from the saved epochs (-epo.fif)
#   (B) Evoked re-computed by re-epoching the preprocessed Raw at the same events
#   (C) Mean STI101 edge from the Raw (scaled)
#
# Goal: verify that the visualization program's time axis is correct by checking
# that (A), (B), and (C) align at 0 s for a single channel.

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt


def rms(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def load_evoked_mag_from_epochs_or_ave(epochs_path: Path, evoked_path: Path | None,
                                       channel: str | None, auto_window=(0.08, 0.15)):
    """
    Return (times, data_1ch, chosen_channel_name, events_used, baseline_tuple).
    If evoked_path is given, read that evoked; else load epochs and average.
    """
    if evoked_path is not None:
        ev = mne.read_evokeds(str(evoked_path), verbose="ERROR")
        # If multiple Evokeds in file, take the first by default
        if isinstance(ev, list):
            ev = ev[0]
        # pick only MAGs
        ev_mag = ev.copy().pick("mag")
        t = ev_mag.times
        ch_names = ev_mag.ch_names
        data = ev_mag.data  # (n_mag, n_times)
        # choose channel
        if channel is not None:
            if channel not in ch_names:
                raise ValueError(f"Channel {channel!r} not found in evoked MAGs.")
            idx = ch_names.index(channel)
        else:
            w0, w1 = auto_window
            i0 = int(np.searchsorted(t, w0))
            i1 = int(np.searchsorted(t, w1))
            peaks = np.max(np.abs(data[:, i0:i1]), axis=1)
            idx = int(np.argmax(peaks))
        y = data[idx, :]
        chosen = ch_names[idx]
        # we don't have events in -ave.fif; load epochs header to get events & baseline
        ep = mne.read_epochs(str(epochs_path), preload=False, verbose="ERROR")
        events_used = ep.events.copy()
        baseline = ep.baseline
        return t, y, chosen, events_used, baseline
    else:
        # use epochs: average MAGs then pick a channel
        ep = mne.read_epochs(str(epochs_path), preload=True, verbose="ERROR")
        ev_mag = ep.average(picks="mag")
        t = ev_mag.times
        ch_names = ev_mag.ch_names
        data = ev_mag.data
        if channel is not None:
            if channel not in ch_names:
                raise ValueError(f"Channel {channel!r} not found in epochs' MAGs.")
            idx = ch_names.index(channel)
        else:
            w0, w1 = auto_window
            i0 = int(np.searchsorted(t, w0))
            i1 = int(np.searchsorted(t, w1))
            peaks = np.max(np.abs(data[:, i0:i1]), axis=1)
            idx = int(np.argmax(peaks))
        y = data[idx, :]
        chosen = ch_names[idx]
        events_used = ep.events.copy()
        baseline = ep.baseline
        return t, y, chosen, events_used, baseline


def main():
    ap = argparse.ArgumentParser(
        description="Overlay one MAG channel from -ave/-epo, Raw re-epoch, and mean STI edge."
    )
    ap.add_argument(
        "--epochs",
        default="/Users/gm33/data/fairy/derivatives/epochs/sub-01/ses-01/sub-01_ses-01_task-faces_run-01_desc-faces_non_target_epo.fif",
        help="Path to *_desc-<cond>_epo.fif (for events & optional averaging).",
    )
    ap.add_argument(
        "--evoked",
        default="/Users/gm33/data/fairy/derivatives/avg/sub-01/ses-01/sub-01_ses-01_task-faces_run-01_desc-faces_non_target_ave.fif",
        help="Path to *_desc-<cond>_ave.fif. If missing, evoked will be computed from epochs.",
    )
    ap.add_argument(
        "--raw",
        default="/Users/gm33/data/fairy/derivatives/preprocessing/sub-01/ses-01/meg/sub-01_ses-01_task-faces_run-01_desc-preproc_meg.fif",
        help="Path to preprocessed Raw FIF.",
    )
    ap.add_argument("--stim-channel", default="STI101", help="Stim channel (default: STI101)")
    ap.add_argument("--channel", default=None, help="Specific MAG channel to plot (e.g., MEG2431).")
    ap.add_argument("--auto-window", nargs=2, type=float, default=[0.08, 0.15],
                    help="Seconds window for auto channel selection peak search.")
    args = ap.parse_args()

    epochs_path = Path(args.epochs).expanduser()
    evoked_path = Path(args.evoked).expanduser() if args.evoked else None
    raw_path = Path(args.raw).expanduser()

    if not epochs_path.exists():
        raise FileNotFoundError(epochs_path)
    if evoked_path is not None and not evoked_path.exists():
        print(f"[warn] Evoked file not found: {evoked_path} — will compute evoked from epochs instead.")
        evoked_path = None
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    # (A) Get single-channel evoked from ave (preferred) or epochs
    t, y_from_ave, chosen, events_used, baseline = load_evoked_mag_from_epochs_or_ave(
        epochs_path, evoked_path, args.channel, tuple(args.auto_window)
    )
    print(f"[info] Channel used: {chosen}")
    print(f"[info] Baseline in saved epochs: {baseline}")

    # (B) Re-epoch Raw for the same channel and average
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    print(f"[info] Raw sfreq={sf:.1f} Hz | first_samp={raw.first_samp} | first_time={raw.first_time:.3f} s")

    # MEG single-channel epochs
    picks_one_mag = mne.pick_channels(raw.ch_names, include=[chosen])
    ep_raw = mne.Epochs(
        raw, events_used, event_id={"x": 1},
        tmin=t[0], tmax=t[-1], baseline=baseline,
        picks=picks_one_mag, preload=True, reject_by_annotation=True, verbose="ERROR"
    )
    ep_raw.events[:, 2] = 1
    ev_raw = ep_raw.average()
    y_from_raw = ev_raw.data[0, :]

    # (C) STI epochs & mean edge
    stim_ch = args.stim_channel
    if stim_ch not in raw.ch_names:
        raise RuntimeError(f"Stim channel {stim_ch!r} not in Raw.")
    picks_sti = mne.pick_channels(raw.ch_names, include=[stim_ch])
    sti_ep = mne.Epochs(
        raw, events_used, event_id={"x": 1},
        tmin=t[0], tmax=t[-1], baseline=None,
        picks=picks_sti, preload=True, reject_by_annotation=False, verbose="ERROR"
    )
    sti_ep.events[:, 2] = 1
    sti_ev = sti_ep.average(picks="stim")
    sti_trace = sti_ev.data[0, :]
    # center STI for display
    sti_trace = sti_trace - np.mean(sti_trace[:max(1, int(0.02 * sf))])

    # Numeric agreement between (A) and (B)
    diff = rms(y_from_ave, y_from_raw)
    print(f"[check] RMS difference (saved evoked vs raw re-epoch) for {chosen} = {diff:.6g} (channel units)")

    # Scale STI to overlay nicely
    peak = np.nanmax(np.abs(np.r_[y_from_ave, y_from_raw])) or 1.0
    sti_scale = peak / (np.nanmax(np.abs(sti_trace)) or 1.0)

    # Plot overlay
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.axvline(0, color="k", lw=1, alpha=0.8)
    ax.plot(t, y_from_ave, label=f"{chosen} (from -ave/-epo)", color="C0", alpha=0.95)
    ax.plot(t, y_from_raw, label=f"{chosen} (re-epoched raw)", color="C1", ls="--", alpha=0.85)
    ax.plot(t, sti_trace * sti_scale, label=f"{stim_ch} mean (scaled)", color="C2", alpha=0.7)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"Single MAG overlay: {chosen}  (ave/epo vs raw vs STI)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
