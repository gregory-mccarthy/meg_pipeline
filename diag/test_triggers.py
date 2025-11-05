#!/usr/bin/env python3
"""
test_triggers.py — Quick trigger diagnostics using your YAML.

Usage:
  ./test_triggers.py /path/to/config.yaml [--plot 5]

- Loads your YAML (expects 'triggers' block and BIDS identifiers)
- Reads the raw via MNE (BIDS if identifiers present, else raw FIF)
- Applies mask, settle, and refractory
- Prints event ID counts and timing stats
"""

from __future__ import annotations
import argparse, math, re, sys
from pathlib import Path
from ruamel.yaml import YAML

import numpy as np
import mne

def load_yaml(p):
    y = YAML(typ="safe")
    with open(p, "r") as f:
        return y.load(f)

def parse_mask(mask_str: str | int | None) -> int | None:
    if mask_str is None:
        return None
    if isinstance(mask_str, int):
        return mask_str
    s = str(mask_str).strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    return int(s)

def read_raw_from_yaml(cfg):
    # Prefer BIDS if keys are present, otherwise fall back to a raw .fif path (not provided here).
    bids_root = cfg.get("bids_root")
    subj = cfg.get("subject")
    ses = cfg.get("session")
    task = cfg.get("task")
    run  = cfg.get("run")
    if bids_root and subj and ses and task:
        bp = mne.BIDSPath(
            root=Path(bids_root),
            subject=str(subj),
            session=str(ses),
            task=str(task),
            run=None if (run in (None, "", "null")) else str(run),
            datatype="meg",
        )
        raw = mne.read_raw_bids(bp, verbose="WARNING")
    else:
        raise SystemExit("YAML must specify bids_root, subject, session, task for this tester.")
    return raw

def find_events_with_params(raw, stim_ch, mask, settle_ms, refractory_ms):
    sfreq = raw.info["sfreq"]
    # mne.find_events supports mask and mask_type='and' to zero out unwanted bits
    events = mne.find_events(
        raw,
        stim_channel=stim_ch,
        consecutive="increasing",
        mask=mask if mask is not None else None,
        mask_type="and" if mask is not None else None,
        shortest_event=1,  # in samples
        uint_cast=True,
        verbose="ERROR",
    )

    # Debounce / settle: drop events whose code changes within settle_ms after onset
    if settle_ms and settle_ms > 0 and len(events) > 0:
        win = int(round((settle_ms / 1000.0) * sfreq))
        data = raw.get_data(picks=stim_ch, reject_by_annotation="omit")[0]
        keep = []
        for i, (samp, eid, val) in enumerate(events):
            end = min(samp + win, data.shape[-1] - 1)
            # stabilize by taking the mode of codes in the window; if mode != first code, drop
            seg = data[samp:end+1].astype(int)
            if seg.size == 0:
                keep.append(True)
                continue
            mode = np.bincount(seg & (mask if mask is not None else (2**16 - 1))).argmax()
            first = (seg[0] & (mask if mask is not None else (2**16 - 1)))
            keep.append(int(mode) == int(first))
        events = events[np.array(keep, dtype=bool)]

    # Refractory: remove events that occur within refractory_ms of previous
    if refractory_ms and refractory_ms > 0 and len(events) > 1:
        min_gap = (refractory_ms / 1000.0) * sfreq
        pruned = [events[0]]
        last = events[0][0]
        for k in range(1, len(events)):
            if (events[k][0] - last) >= min_gap:
                pruned.append(events[k])
                last = events[k][0]
        events = np.array(pruned)

    return events

def summarize_events(raw, events):
    sfreq = raw.info["sfreq"]
    if events.size == 0:
        print("No events found.")
        return
    codes, counts = np.unique(events[:, 2], return_counts=True)
    print("\nEvent IDs (value : count):")
    for c, n in zip(codes, counts):
        print(f"  {c:4d} : {n}")

    times = events[:, 0] / sfreq
    if len(times) > 1:
        isi = np.diff(times)
        print("\nTiming:")
        print(f"  First event @ {times[0]:.3f}s, last @ {times[-1]:.3f}s, N={len(times)}")
        print(f"  ISI mean={isi.mean():.4f}s, median={np.median(isi):.4f}s, min={isi.min():.4f}s, max={isi.max():.4f}s")

def check_stuck_bits(raw, stim_ch, mask):
    """Estimate stuck-high/low fractions for low 16 bits of stim channel."""
    data = raw.get_data(picks=stim_ch, reject_by_annotation="omit")[0].astype(np.uint16)
    if mask is not None:
        data = data & mask
    total = data.size
    print("\nBit usage (low 16 bits):")
    for b in range(16):
        ones = ((data >> b) & 1).sum()
        frac = ones / total
        if frac < 0.02 or frac > 0.98:
            flag = " (stuck?)"
        else:
            flag = ""
        print(f"  bit {b:2d}: {frac:6.3f}{flag}")

def maybe_plot(raw, stim_ch, events, n=5):
    import matplotlib.pyplot as plt
    sfreq = raw.info["sfreq"]
    picks = mne.pick_channels(raw.ch_names, include=[stim_ch])
    if len(picks) == 0:
        print(f"Cannot plot; stim channel {stim_ch} not found.")
        return
    for i, ev in enumerate(events[:n]):
        t0 = max(0, int(ev[0] - 0.25 * sfreq))
        t1 = min(raw.n_times - 1, int(ev[0] + 0.75 * sfreq))
        seg, times = raw.get_data(picks=picks, start=t0, stop=t1, return_times=True)
        plt.figure()
        plt.plot(times - times[0], seg[0])
        plt.title(f"{stim_ch} around event {i+1} (code {ev[2]})")
        plt.xlabel("Time (s)")
        plt.ylabel("Stim value")
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", help="YAML config file with BIDS identifiers and 'triggers' block.")
    ap.add_argument("--plot", type=int, default=0, help="Plot N example windows around events.")
    args = ap.parse_args()

    cfg = load_yaml(args.yaml)

    trig = cfg.get("triggers", {}) or {}
    stim_ch = trig.get("stim_channel", "STI101")
    mask = parse_mask(trig.get("mask"))
    settle_ms = float(trig.get("max_settle_ms", 0.0) or 0.0)
    refractory_ms = float(trig.get("refractory_ms", 0.0) or 0.0)

    print(f"Stim channel : {stim_ch}")
    print(f"Mask         : {hex(mask) if mask is not None else 'None'}")
    print(f"Settle (ms)  : {settle_ms}")
    print(f"Refractory(ms): {refractory_ms}")

    raw = read_raw_from_yaml(cfg)
    print(f"Loaded raw: {raw}, sfreq={raw.info['sfreq']:.2f} Hz, duration={raw.n_times/raw.info['sfreq']:.1f}s")

    # Stuck-bit scan (helps choose masks)
    check_stuck_bits(raw, stim_ch, mask)

    # Find events with parameters
    events = find_events_with_params(raw, stim_ch, mask, settle_ms, refractory_ms)
    print(f"\nDetected {len(events)} events.")
    summarize_events(raw, events)

    if args.plot and len(events) > 0:
        maybe_plot(raw, stim_ch, events, n=args.plot)

if __name__ == "__main__":
    main()
