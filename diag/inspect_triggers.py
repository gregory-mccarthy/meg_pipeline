#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_triggers.py
-------------------
Side-by-side trigger diagnostics for MEG STI channels.

Features:
- Loads a FIF raw file (unfiltered)
- Extracts STI101 and STI001..STI016 (if present; otherwise decodes bits from STI101)
- Runs two detectors:
  A) legacy_bitwise_events_like_pipeline: per-bit runs -> powers-of-two events
  B) events_from_sti101_early_onset_final_code: early-onset timing, final composite labeling
- Prints event histograms for both methods
- For the first N merged events, prints sample-by-sample rows: sample, STI101, bit01..bit16
- Reports first occurrence and first "stable" occurrence (>= K consecutive samples) of the final code
- No filtering is applied to stim channels (or anything else)

Usage:
  python inspect_triggers.py /path/to/raw.fif --window-ms 10 --max-settle-ms 6 --refractory-ms 20 --n-events 10 --stable-k 2 --mask 0x00FF
"""

import argparse
from collections import Counter
from typing import List, Tuple

import numpy as np
import mne


# =========================
# Detector B: merged events
# =========================
def events_from_sti101_early_onset_final_code(
    raw: mne.io.BaseRaw,
    *,
    stim_channel: str = "STI101",
    mask: int = 0xFFFF,
    max_settle_ms: float = 6.0,
    refractory_ms: float = 20.0
) -> np.ndarray:
    """
    Return MNE-style events (n,3): [sample_index, 0, code].

    - Onset is the FIRST sample where STI101 goes 0 -> nonzero.
    - Code is the MAX nonzero value within max_settle_ms after onset
      (captures the final composite code without delaying the timestamp).
    - A global refractory prevents counting late-arriving bits as separate events.
    """
    sf = float(raw.info["sfreq"])
    settle = int(round(max_settle_ms * 1e-3 * sf))
    refr = int(round(refractory_ms * 1e-3 * sf))

    picks = mne.pick_channels(raw.ch_names, [stim_channel])
    if len(picks) == 0:
        return np.empty((0, 3), dtype=int)

    stim = raw.get_data(picks=picks, start=0, stop=None)[0].astype(np.uint16)
    stim &= np.uint16(mask)

    prev = np.empty_like(stim)
    prev[0] = 0
    prev[1:] = stim[:-1]
    rising_idx = np.flatnonzero((prev == 0) & (stim > 0))

    events = []
    last_onset = -refr - 1
    N = stim.size

    for idx in rising_idx:
        if idx - last_onset < refr:
            continue
        end = min(idx + settle, N)
        window = stim[idx:end]
        nz = window[window > 0]
        if nz.size == 0:
            continue
        code = int(nz.max())
        events.append([int(idx), 0, code])
        last_onset = int(idx)

    events = np.asarray(events, dtype=int)
    if events.size:
        events = events[np.argsort(events[:, 0])]
    return events


# =========================================
# Detector A: legacy per-bit (powers-of-two)
# =========================================
def legacy_bitwise_events_like_pipeline(
    raw: mne.io.BaseRaw,
    *,
    stim_channel: str = "STI101",
    mask: int = 0xFFFF,
    min_high: int = 2,  # samples bit must remain high to count
    min_off: int = 5    # samples between two events on the SAME bit
) -> np.ndarray:
    """
    Approximate the per-bit run-length detector that emits one event per raised bit.
    This mirrors the behavior that produces powers-of-two histograms.
    """
    picks = mne.pick_channels(raw.ch_names, [stim_channel])
    if len(picks) == 0:
        return np.empty((0, 3), dtype=int)

    stim = raw.get_data(picks=picks)[0].astype(np.uint16) & np.uint16(mask)

    events = []
    for b in range(16):
        code = 1 << b
        if (mask & code) == 0:
            continue
        vec = (stim & code) != 0
        idx = np.flatnonzero(vec)
        if idx.size == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1) + 1
        runs = np.split(idx, splits)
        last_rise = -min_off - 1
        for r in runs:
            if len(r) >= min_high and (r[0] - last_rise) >= min_off:
                events.append([int(r[0]), 0, int(code)])
                last_rise = r[0]

    events = np.asarray(events, dtype=int)
    if events.size:
        events = events[np.argsort(events[:, 0])]
    return events


# ========================
# Stim arrays and bit view
# ========================
def get_sti_arrays(
    raw: mne.io.BaseRaw,
    mask: int = 0xFFFF
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      stim101: (n_samples,) uint16 masked composite
      bits_matrix: (16, n_samples) bool array for bits 1..16 (index 0 = bit1)
      names_used: list of 16 strings for each bit source
                  (STI00x if present, else 'STI00x(decoded)')
    """
    pick101 = mne.pick_channels(raw.ch_names, ["STI101"])
    if len(pick101) == 0:
        raise RuntimeError("STI101 channel not found in raw.")
    stim101 = raw.get_data(picks=pick101)[0].astype(np.uint16) & np.uint16(mask)

    sti_names = [f"STI{str(i).zfill(3)}" for i in range(1, 17)]
    have = [nm for nm in sti_names if nm in raw.ch_names]

    bits_matrix = np.zeros((16, stim101.size), dtype=bool)
    names_used: List[str] = []

    if len(have) > 0:
        # Use available physical bit lines; decode missing ones from STI101
        for i, ch in enumerate(sti_names):  # i = 0..15
            if ch in raw.ch_names:
                arr = raw.get_data(picks=[ch])[0]
                # Treat > 50% of channel max as "high" (robust to float TTLs)
                thr = 0.5 * float(arr.max() or 1.0)
                bits_matrix[i] = arr.astype(np.float64) > thr
                names_used.append(ch)
            else:
                bits_matrix[i] = (stim101 & (1 << i)) != 0
                names_used.append(f"{ch}(decoded)")
    else:
        # No STI001..016 -> decode all from composite
        for i in range(16):
            bits_matrix[i] = (stim101 & (1 << i)) != 0
        names_used = [f"STI{str(i).zfill(3)}(decoded)" for i in range(1, 17)]

    return stim101, bits_matrix, names_used


# ===========================
# Per-event windowed printing
# ===========================
def print_event_window(
    ev_index: int,
    onset_sample: int,
    final_code: int,
    stim101: np.ndarray,
    bits_matrix: np.ndarray,
    sfreq: float,
    window_ms: float,
    stable_k: int
) -> None:
    """
    Print a table of samples starting at onset for window_ms duration.

    Columns: sample_index, STI101, b01..b16
    Also reports the first occurrence of the final code and the first
    'stable' occurrence (>= stable_k consecutive samples).
    """
    win_len = int(round(window_ms * 1e-3 * sfreq))
    N = stim101.size
    start = int(onset_sample)
    end = min(start + win_len, N)

    window_vals = stim101[start:end]
    first_final_rel = np.flatnonzero(window_vals == final_code)
    first_final_idx = int(first_final_rel[0]) if first_final_rel.size else None

    # find first position with stable_k consecutive final_code samples
    first_stable_idx = None
    if stable_k <= 1:
        first_stable_idx = first_final_idx
    else:
        run = 0
        for i, v in enumerate(window_vals == final_code):
            run = run + 1 if v else 0
            if run >= stable_k:
                first_stable_idx = i - stable_k + 1
                break

    print(f"\n=== Event {ev_index + 1} ===")
    print(f"Onset sample: {start}  (t = {start/sfreq*1e3:.3f} ms)")
    print(f"Final code: {final_code} (0x{final_code:04X})")
    if first_final_idx is not None:
        print(f" First occurrence of final code: sample {start + first_final_idx} "
              f"(+{first_final_idx} samples, {first_final_idx/sfreq*1e3:.3f} ms after onset)")
    else:
        print(" Final code NOT seen within window.")
    if first_stable_idx is not None:
        print(f" First stable (≥{stable_k}) final code: sample {start + first_stable_idx} "
              f"(+{first_stable_idx} samples, {first_stable_idx/sfreq*1e3:.3f} ms after onset)")
    else:
        print(f" Stable final code (≥{stable_k}) NOT reached within window.")

    # Header
    bit_headers = [f"b{b+1:02d}" for b in range(16)]
    header = ["sample", "STI101"] + bit_headers
    print("\t".join(header))

    # Rows
    for s in range(start, end):
        row = [str(s), str(int(stim101[s]))]
        for b in range(16):
            row.append("1" if bits_matrix[b, s] else "0")
        print("\t".join(row))


# =====
# Main
# =====
def main():
    ap = argparse.ArgumentParser(description="Inspect MEG trigger settling on STI channels.")
    ap.add_argument("fif", help="Path to raw FIF file (e.g., *_raw.fif)")
    ap.add_argument("--mask", type=lambda x: int(x, 0), default=0xFFFF,
                    help="Bitmask for STI101 (e.g., 0x00FF to restrict to low 8 bits; accepts 0x.. or int).")
    ap.add_argument("--window-ms", type=float, default=10.0,
                    help="Per-event print window length (ms). Default 10.")
    ap.add_argument("--max-settle-ms", type=float, default=6.0,
                    help="Lookahead window to capture final composite code (ms). Default 6.")
    ap.add_argument("--refractory-ms", type=float, default=20.0,
                    help="Global refractory between events (ms). Default 20.")
    ap.add_argument("--n-events", type=int, default=10,
                    help="How many merged events to print. Default 10.")
    ap.add_argument("--stable-k", type=int, default=2,
                    help="Consecutive samples required to call final code 'stable'. Default 2.")
    args = ap.parse_args()

    print(f"Loading: {args.fif}")
    raw = mne.io.read_raw_fif(args.fif, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    print(f"Sampling frequency: {sfreq:.3f} Hz")
    print(f"Channels: {len(raw.ch_names)}")
    print("NOTE: No filtering is applied; stim channels are read exactly as stored.\n")

    # Extract composite and bits
    stim101, bits_matrix, bit_names = get_sti_arrays(raw, mask=args.mask)
    print("Bit channels used for display:")
    for i, nm in enumerate(bit_names):
        print(f"  bit {i+1:02d}: {nm}")

    # A) Legacy per-bit (reproduce powers-of-two histograms)
    legacy = legacy_bitwise_events_like_pipeline(
        raw, stim_channel="STI101", mask=args.mask
    )
    legacy_counts = Counter(map(int, legacy[:, 2])) if legacy.size else {}
    print("\n[Legacy per-bit] total events:", int(legacy.shape[0]))
    print("[Legacy per-bit] counts:", dict(sorted(legacy_counts.items())))

    # B) Settled composite (one per stimulus)
    merged = events_from_sti101_early_onset_final_code(
        raw,
        stim_channel="STI101",
        mask=args.mask,
        max_settle_ms=args.max_settle_ms,
        refractory_ms=args.refractory_ms,
    )
    merged_counts = Counter(map(int, merged[:, 2])) if merged.size else {}
    print("\n[Settled composite] total events:", int(merged.shape[0]))
    print("[Settled composite] counts:", dict(sorted(merged_counts.items())))

    # Optional diagnostic: how many legacy hits fall in the initial settle/refr window per merged event
    if legacy.size and merged.size:
        sf = raw.info["sfreq"]
        settle = int(round(args.max_settle_ms * 1e-3 * sf))
        refr = int(round(args.refractory_ms * 1e-3 * sf))
        multi_hits = 0
        for s, _, _ in merged:
            lo = s
            hi = s + max(settle, refr)
            k = ((legacy[:, 0] >= lo) & (legacy[:, 0] <= hi)).sum()
            if k > 1:
                multi_hits += 1
        print(f"\nMerged events with >1 legacy hit inside settle/refr window: {multi_hits} / {len(merged)}")

    # Print first N merged events' raw windows
    if merged.size == 0:
        print("\nNo merged events to display.")
        return

    n_show = min(args.n_events, merged.shape[0])
    print(f"\nPrinting sample-by-sample windows for the first {n_show} merged events "
          f"(window {args.window_ms} ms from merged onset):")
    for i in range(n_show):
        onset, _, code = merged[i]
        print_event_window(
            ev_index=i,
            onset_sample=int(onset),
            final_code=int(code),
            stim101=stim101,
            bits_matrix=bits_matrix,
            sfreq=sfreq,
            window_ms=args.window_ms,
            stable_k=args.stable_k,
        )

    # Quick IEI summary
    if merged.shape[0] > 1:
        iei_ms = np.diff(merged[:, 0]) / sfreq * 1e3
        show = iei_ms[:100]
        print("\n[Settled composite] Inter-event interval (ms) summary (first 100):")
        if show.size:
            print(f" count={len(show)}, mean={show.mean():.2f}, median={np.median(show):.2f}, "
                  f"min={show.min():.2f}, max={show.max():.2f}")

if __name__ == "__main__":
    main()