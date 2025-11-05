#!/usr/bin/env python3
"""
Interactive viewer + Trigger diagnostics & comparison for BIDS-style MEGIN split FIF.

Features
--------
• Reads first split; MNE auto-collects subsequent parts (_split-02, _split-03, …)
• Interactive Raw browser (Qt or Matplotlib backend), optional PSD & sensor plots
• Stim pretest: estimates pulse width distribution & auto-suggests params
• Robust event detection with configurable and fallback strategies
• Compares STI events vs Annotations by code & nearest time within ± tolerance
• Optional CSV export of the full comparison table
• NEW: For the first N pulses on the stim channel, dump the value sequence inside
  each non-zero run to detect bit "stair-steps" during rise/fall.

Examples
--------
# Full interactive + comparison, with pretest-informed fallback and CSV:
python view_megin_splits.py "/path/sub-..._split-01_meg.fif" \
  --preload --duration 20 --n-chans 80 --compare --stim STI101 --tol-ms 5 \
  --retry-permissive --csv ~/Desktop/trigger_check.csv

# Just browse, no comparison:
python view_megin_splits.py "/path/sub-..._split-01_meg.fif" --no_psd --no_sensors

# Dump first 10 pulses as change-points (default) and also allow permissive retry:
python view_megin_splits.py "/path/sub-..._split-01_meg.fif" \
  --compare --stim STI101 --dump-first-n 10 --retry-permissive
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import mne

# ----------------------------- Annotation parsing -----------------------------

CODE_PATTERNS = [
    re.compile(r".*?(?P<code>\d+)$", re.IGNORECASE),             # .../123  or plain "123"
    re.compile(r".*?(stimulus|trigger|trig|event)\s*/?\s*(?P<code>\d+)", re.IGNORECASE),
]

def parse_code(desc: str) -> Optional[int]:
    """Extract a numeric code from an annotation description, if present."""
    for pat in CODE_PATTERNS:
        m = pat.match(desc.strip())
        if m:
            try:
                return int(m.group("code"))
            except Exception:
                continue
    return None


# ------------------------------- Stim pretesting ------------------------------

def _run_lengths_and_bounds(binary_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return run lengths and (start,end) indices of consecutive True runs in a boolean vector."""
    if binary_vec.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    edges = np.diff(np.r_[False, binary_vec, False].astype(np.int32))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = (ends - starts).astype(int)
    return lengths, starts.astype(int), ends.astype(int)  # ends are exclusive

def stim_pretest(raw: mne.io.BaseRaw, stim_name: str) -> Dict[str, Any]:
    """
    Inspect the stim channel to estimate pulse widths (samples & ms) and suggest
    mne.find_events shortest_event (samples) and min_duration (seconds).
    Also returns the stim array and nonzero run boundaries for optional dumps.
    """
    info: Dict[str, Any] = {}
    sfreq = float(raw.info["sfreq"])
    info["sfreq"] = sfreq
    info["stim_name_present"] = stim_name in raw.ch_names

    if not info["stim_name_present"]:
        return info  # minimal info

    picks = mne.pick_channels(raw.ch_names, include=[stim_name])
    if len(picks) != 1:
        info["error"] = f"Stim channel '{stim_name}' not uniquely found."
        return info

    stim = raw.get_data(picks=picks, reject_by_annotation="omit", verbose=False)[0]
    stim_i = stim.astype(np.int64, copy=False)
    nonzero = stim_i != 0

    run_lengths_samp, starts, ends = _run_lengths_and_bounds(nonzero)
    info["n_pulses"] = int((run_lengths_samp > 0).sum())
    info["run_lengths_samp"] = run_lengths_samp
    info["run_lengths_ms"] = run_lengths_samp / sfreq * 1000.0
    info["starts"] = starts
    info["ends"] = ends
    info["stim_i"] = stim_i  # keep the integer stim trace for dumps

    if info["n_pulses"] > 0:
        rl = run_lengths_samp
        info["min_len_samp"] = int(rl.min())
        info["max_len_samp"] = int(rl.max())
        info["median_len_samp"] = float(np.median(rl))
        info["p05_len_samp"] = float(np.percentile(rl, 5))
        info["p25_len_samp"] = float(np.percentile(rl, 25))
        info["p75_len_samp"] = float(np.percentile(rl, 75))

        # Heuristic suggestions:
        shortest_event_samp_suggest = max(1, int(np.floor(info["p05_len_samp"])))
        min_duration_s_suggest = max(0.0, (shortest_event_samp_suggest - 1) / sfreq)

        info["shortest_event_samp_suggest"] = int(shortest_event_samp_suggest)
        info["min_duration_s_suggest"] = float(min_duration_s_suggest)
    else:
        info["min_len_samp"] = info["max_len_samp"] = info["median_len_samp"] = None
        info["p05_len_samp"] = info["p25_len_samp"] = info["p75_len_samp"] = None
        info["shortest_event_samp_suggest"] = 1
        info["min_duration_s_suggest"] = 0.0

    return info


# ---------------------------- Matching helper funcs ---------------------------

def nearest_match_by_code(
    event_time: float,
    code: int,
    ann_df: pd.DataFrame,
    tol_s: float,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Find the nearest annotation with the same code within tol_s seconds.
    Returns (ann_idx, ann_onset, delta_s) or (None, None, None) if none within tol.
    """
    if ann_df.empty:
        return None, None, None
    pool = ann_df[ann_df["ann_code"] == code]
    if pool.empty:
        return None, None, None
    idx = int(np.argmin(np.abs(pool["ann_onset"].to_numpy() - event_time)))
    ann_row = pool.iloc[idx]
    delta = float(ann_row["ann_onset"] - event_time)
    if abs(delta) <= tol_s:
        return int(ann_row.name), float(ann_row["ann_onset"]), delta
    return None, None, None


# ------------------------------- Dump utilities -------------------------------

def dump_pulse_sequence(
    stim_i: np.ndarray,
    start: int,
    end: int,
    sfreq: float,
    pulse_idx: int,
    mode: str = "changes",
    max_samples: int = 2000,
) -> None:
    """
    Print the value sequence for one non-zero run [start:end), either:
      • 'changes' : only when the value changes inside the pulse
      • 'full'    : every sample (capped by max_samples)
    """
    dur_samp = end - start
    onset_t = start / sfreq
    seg = stim_i[start:end]

    print(f"\n--- Pulse {pulse_idx} ---")
    print(f"Samples [{start}:{end})  len={dur_samp}  onset_s={onset_t:.6f}")
    if seg.size == 0:
        print("  [empty segment]")
        return

    first_val = int(seg[0])
    print(f"  first_value={first_val}")

    if mode == "full":
        cap = min(len(seg), max_samples)
        print(f"  values (first {cap} samples):")
        # chunk printing for readability
        chunk = 32
        for i in range(0, cap, chunk):
            j = min(i + chunk, cap)
            vals = " ".join(str(int(v)) for v in seg[i:j])
            print(f"    +{i:5d} → +{j-1:5d}: {vals}")
        if len(seg) > cap:
            print(f"    ... ({len(seg)-cap} more samples not shown; increase --dump-max-samples)")
    else:  # 'changes'
        print("  value changes within pulse:")
        prev = int(seg[0])
        print(f"    t=+0.000000s (+0 samp): {prev}")
        for k in range(1, len(seg)):
            v = int(seg[k])
            if v != prev:
                t_rel = k / sfreq
                print(f"    t=+{t_rel:.6f}s (+{k} samp): {v}")
                prev = v
        print("    end (returns to 0 after pulse)")

# ------------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer + Stim pretest + STI vs Annotation comparison + pulse dumps."
    )
    parser.add_argument("first_split", help="Path to first split, e.g., /path/sub-..._split-01_meg.fif")

    # I/O and reading options
    parser.add_argument("--on-missing", default="raise", choices=("raise", "warn", "ignore"),
                        help="Behavior if a later split is missing (default: raise).")
    parser.add_argument("--preload", action="store_true",
                        help="Preload raw into RAM (useful for filtering/resampling).")

    # Viewer options
    parser.add_argument("--backend", default="qt", choices=("qt", "matplotlib"),
                        help="Browser backend (default: qt).")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Initial browser window duration in seconds (default: 10).")
    parser.add_argument("--n-chans", type=int, default=50,
                        help="Initial number of channels per view (default: 50).")
    parser.add_argument("--no_psd", action="store_true", help="Skip PSD plot.")
    parser.add_argument("--no_sensors", action="store_true", help="Skip sensor layout.")

    # Comparison / event-finding options
    parser.add_argument("--compare", action="store_true",
                        help="Compare STI events against annotations by code/time.")
    parser.add_argument("--stim", default="STI101",
                        help="Stim channel name (default: STI101; set '' to skip).")
    parser.add_argument("--consecutive", default="increasing",
                        choices=("increasing", "increasing_including_zeros", "False"),
                        help="mne.find_events 'consecutive' behavior (default: increasing).")
    parser.add_argument("--min-duration", type=float, default=0.0,
                        help="mne.find_events min_duration in seconds (default: 0).")
    parser.add_argument("--shortest-event-samp", type=int, default=1,
                        help="mne.find_events shortest_event in samples (default: 1).")
    parser.add_argument("--tol-ms", type=float, default=5.0,
                        help="Time tolerance for event↔annotation matching (default: 5 ms).")
    parser.add_argument("--retry-permissive", action="store_true",
                        help="If find_events fails, retry with permissive settings.")

    # CSV export
    parser.add_argument("--csv", default=None,
                        help="Optional path to save the comparison table as CSV.")

    # NEW: pulse dump controls
    parser.add_argument("--dump-first-n", type=int, default=10,
                        help="Dump the first N non-zero pulses on the stim channel (default: 10; 0 = disable).")
    parser.add_argument("--dump-mode", default="changes", choices=("changes", "full"),
                        help="Pulse dump mode: 'changes' (value change points) or 'full' (every sample).")
    parser.add_argument("--dump-max-samples", type=int, default=2000,
                        help="Limit for samples printed per pulse in 'full' mode (default: 2000).")

    args = parser.parse_args()

    # Backend preference
    try:
        if args.backend == "qt":
            mne.viz.set_browser_backend("qt")
        else:
            mne.viz.set_browser_backend("matplotlib")
    except Exception as e:
        print(f"[WARN] Could not set requested backend ({args.backend}): {e}")

    # --- Read raw (MNE auto-collects split parts) ---
    fpath = Path(args.first_split).expanduser().resolve()
    print(f"Reading first split: {fpath}")
    raw = mne.io.read_raw_fif(
        fpath,
        preload=args.preload,
        on_split_missing=args.on_missing,
        verbose="info",
    )

    sfreq = float(raw.info["sfreq"])

    # --- High-level summary ---
    print("\n=== Summary ===")
    print(f"Files combined : {len(raw.filenames)}")
    print(f"Sampling rate  : {sfreq:.3f} Hz")
    print(f"Channels       : {len(raw.ch_names)}")
    print(f"Duration (s)   : {raw.times[-1]:.3f}")
    print(f"Preloaded      : {raw.preload}")

    # --- Interactive Raw browser ---
    print("\nLaunching Raw browser… (press 'H' in the window for help)")
    raw.plot(
        duration=args.duration,
        n_channels=args.n_chans,
        block=False,
        scalings="auto",
        clipping=None,
        show=True,
    )

    # Optional PSD
    if not args.no_psd:
        print("Opening PSD… (tip: prefer raw.compute_psd().plot() in new code)")
        raw.plot_psd(average=False, show=True)

    # Optional sensor layout
    if not args.no_sensors:
        print("Opening sensor layout…")
        mne.viz.plot_sensors(raw.info, kind="topomap", show=True)

    # --- Pretest on stim channel (also provides stim array & run bounds) ---
    pre = stim_pretest(raw, args.stim) if args.stim else {"stim_name_present": False}
    print("\n=== Stim pretest ===")
    print(f"Stim channel present : {pre.get('stim_name_present', False)}")
    if pre.get("stim_name_present", False):
        print(f"Sampling rate        : {pre['sfreq']:.3f} Hz")
        print(f"Pulses (nonzero runs): {pre.get('n_pulses', 0)}")
        if pre.get("n_pulses", 0) > 0:
            print(f"Run length (samples) : min={pre['min_len_samp']}  "
                  f"p05={pre['p05_len_samp']:.1f}  p25={pre['p25_len_samp']:.1f}  "
                  f"median={pre['median_len_samp']:.1f}  p75={pre['p75_len_samp']:.1f}  "
                  f"max={pre['max_len_samp']}")
            print(f"Suggested shortest_event (samples): {pre['shortest_event_samp_suggest']}")
            print(f"Suggested min_duration (seconds) : {pre['min_duration_s_suggest']:.6f}")
        else:
            print("No nonzero pulses detected on stim; event finding may yield zero events.")
    else:
        if args.stim:
            print(f"[WARN] Stim channel '{args.stim}' not in raw; skipping stim pretest.")

    # --- NEW: Dump first N pulse value sequences to catch bit-staircases ---
    if pre.get("stim_name_present", False) and pre.get("n_pulses", 0) > 0 and args.dump_first_n > 0:
        print(f"\n=== Pulse dumps (first {min(args.dump_first_n, pre['n_pulses'])} pulses; mode={args.dump_mode}) ===")
        stim_i = pre["stim_i"]
        starts: np.ndarray = pre["starts"]
        ends:   np.ndarray = pre["ends"]
        n = min(args.dump_first_n, len(starts))
        for p in range(n):
            dump_pulse_sequence(
                stim_i=stim_i,
                start=int(starts[p]),
                end=int(ends[p]),
                sfreq=sfreq,
                pulse_idx=p,
                mode=args.dump_mode,
                max_samples=args.dump_max_samples,
            )

    # --- Optional comparison: STI events vs Annotations ---
    if args.compare:
        tol_s = args.tol_ms / 1000.0

        # Collect annotations
        anns = raw.annotations or mne.Annotations(onset=[], duration=[], description=[])
        ann_records = []
        for k, (onset, dur, desc) in enumerate(zip(anns.onset, anns.duration, anns.description)):
            code = parse_code(desc)
            ann_records.append({
                "ann_idx": k,
                "ann_onset": float(onset),    # seconds from raw start in MNE time base
                "ann_duration": float(dur),
                "ann_desc": str(desc),
                "ann_code": code,
            })
        ann_df = pd.DataFrame(ann_records).set_index("ann_idx", drop=True) if ann_records else \
                 pd.DataFrame(columns=["ann_onset","ann_duration","ann_desc","ann_code"])
        has_codes = ann_df["ann_code"].notna().any() if not ann_df.empty else False

        print("\n=== Annotations ===")
        print(f"Total annotations : {len(ann_df)}")
        print(f"With numeric codes: {int(has_codes)}")
        if not ann_df.empty:
            with pd.option_context("display.max_colwidth", 80):
                print(ann_df.head(min(5, len(ann_df)))[["ann_onset", "ann_desc", "ann_code"]])

        # ---------------- Event finding with robust fallbacks ----------------
        events = np.empty((0, 3), dtype=int)
        if args.stim and args.stim in raw.ch_names:
            consecutive = False if args.consecutive == "False" else args.consecutive
            print(f"\nFinding events on stim channel: {args.stim}")
            try:
                events = mne.find_events(
                    raw,
                    stim_channel=args.stim,
                    consecutive=consecutive,
                    min_duration=args.min_duration,             # seconds
                    shortest_event=args.shortest_event_samp,    # samples
                    verbose=True,
                )
            except ValueError as e:
                print(f"[WARN] find_events failed with your settings: {e}")
                if args.retry_permissive:
                    print("[INFO] Retrying with permissive settings: shortest_event=1 sample, min_duration=0 s …")
                    try:
                        events = mne.find_events(
                            raw,
                            stim_channel=args.stim,
                            consecutive=consecutive,
                            min_duration=0.0,
                            shortest_event=1,
                            verbose=True,
                        )
                    except ValueError as e2:
                        print(f"[WARN] Permissive retry also failed: {e2}")
                        # If pretest suggests something different, try that too
                        if pre.get("stim_name_present", False) and pre.get("n_pulses", 0) > 0:
                            se_sugg = int(pre["shortest_event_samp_suggest"])
                            md_sugg = float(pre["min_duration_s_suggest"])
                            print(f"[INFO] Trying pretest suggestion: shortest_event={se_sugg}, min_duration={md_sugg:.6f} …")
                            try:
                                events = mne.find_events(
                                    raw,
                                    stim_channel=args.stim,
                                    consecutive=consecutive,
                                    min_duration=md_sugg,
                                    shortest_event=se_sugg,
                                    verbose=True,
                                )
                            except ValueError as e3:
                                print(f"[ERROR] Pretest-suggested retry failed: {e3}")
                        else:
                            print("[HINT] No usable pretest suggestion; consider adjusting "
                                  "--shortest-event-samp and/or --min-duration manually.")
                else:
                    print("[HINT] Add --retry-permissive to auto-retry, or adjust "
                          "--shortest-event-samp / --min-duration.")
        else:
            if not args.stim:
                print("\n[WARN] No stim channel requested (--stim ''), skipping event detection.")
            else:
                print(f"\n[WARN] Stim channel '{args.stim}' not in raw. Skipping event detection.")

        # ---------------- Build comparison table ----------------
        rows = []
        for i, (samp, _prev, code) in enumerate(events):
            t = samp / sfreq
            ann_idx, ann_onset, delta = (None, None, None)
            if has_codes and code is not None:
                ann_idx, ann_onset, delta = nearest_match_by_code(t, code, ann_df, tol_s)
            rows.append({
                "event_i": i,
                "event_code": int(code),
                "event_sample": int(samp),
                "event_time_s": float(t),
                "event_time_ms": float(t * 1000.0),
                "match_ann_idx": ann_idx,
                "match_ann_onset_s": ann_onset,
                "delta_s": delta,
                "delta_ms": (delta * 1000.0) if delta is not None else None,
                "delta_samples": (delta * sfreq) if delta is not None else None,
            })

        df = pd.DataFrame(rows)

        # Attach annotation metadata for matched rows (description & duration)
        if not ann_df.empty and not df.empty:
            df = df.merge(
                ann_df[["ann_onset", "ann_duration", "ann_desc", "ann_code"]],
                how="left",
                left_on="match_ann_idx",
                right_index=True,
                suffixes=("", "_from_ann"),
            ).rename(columns={"ann_onset": "match_ann_onset_s"})

        # Print readable listing + summaries
        if df.empty:
            print("\nNo STI events found; nothing to compare.")
        else:
            unique_ids = np.unique(events[:, 2]) if events.size else np.array([])
            print("\n=== Event detection summary ===")
            print(f"Events found          : {len(df)}")
            print(f"Unique event IDs      : {unique_ids.tolist()}")
            if pre.get("n_pulses", 0):
                rl_ms = pre["run_lengths_ms"]
                print("Pulse width stats (ms): "
                      f"min={rl_ms.min():.3f}  p05={np.percentile(rl_ms,5):.3f}  "
                      f"median={np.median(rl_ms):.3f}  p95={np.percentile(rl_ms,95):.3f}  "
                      f"max={rl_ms.max():.3f}")

            print("\n=== Event ↔ Annotation comparison (±{:.1f} ms) ===".format(args.tol_ms))
            preview_cols = [
                "event_i", "event_code",
                "event_sample", "event_time_s",
                "match_ann_idx", "match_ann_onset_s",
                "delta_ms", "delta_samples", "ann_desc"
            ]
            for c in preview_cols:  # ensure all columns exist
                if c not in df.columns:
                    df[c] = np.nan

            with pd.option_context("display.max_rows", 40, "display.max_colwidth", 70, "display.width", 140):
                print(df[preview_cols].to_string(index=False))

            # Summaries
            n_total = len(df)
            n_matched = int(df["match_ann_idx"].notna().sum())
            n_unmatched = n_total - n_matched
            n_outside_tol = int(df["delta_ms"].abs().gt(args.tol_ms).sum()) if n_matched else 0
            print("\n--- Comparison summary ---")
            print(f"Events total           : {n_total}")
            print(f"Matched to annotations : {n_matched}")
            print(f"Unmatched              : {n_unmatched}")
            if n_matched and df["delta_ms"].notna().any():
                print(f"Outside tolerance      : {n_outside_tol}")
                print(f"Median Δt (ms)         : {df['delta_ms'].dropna().median():.3f}")
                print(f"Mean  Δt (ms)          : {df['delta_ms'].dropna().mean():.3f}")

            # Optional CSV
            if args.csv:
                out = Path(args.csv).expanduser().resolve()
                df.to_csv(out, index=False)
                print(f"\nSaved full table to: {out}")

    # Keep figures open until user closes
    mne.viz.utils.plt_show(block=True)


if __name__ == "__main__":
    main()
