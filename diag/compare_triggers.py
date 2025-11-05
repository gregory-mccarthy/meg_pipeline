#!/usr/bin/env python3
# compare_triggers_detailed.py
"""
Print a per-event CSV comparison of STI101 vs Annotation triggers.

For each matched pair (nearest in time within tolerance), prints:
idx, stim_idx, stim_sample, stim_time_s, stim_code,
     annot_idx, annot_sample, annot_time_s, annot_code,
     dt_samples(annot-stim), dt_ms(annot-stim)

Options:
  --base auto|abs|rel   : how to convert annotation onsets to samples
                          abs = subtract raw.first_time (when ann.orig_time is set)
                          rel = don't subtract (annotations already relative)
                          auto (default) = pick whichever gives smaller |median Δsamples|
  --include-unmatched   : also print rows for unmatched events (with blanks for the other side)
  --tol-sec             : time-match tolerance in seconds (default 0.002)
  --regex               : regex to extract an integer code from annotation descriptions
                          (default 'TRIG/(\\d+)')
"""

from __future__ import annotations
import argparse
import re
from typing import List, Tuple, Optional

import numpy as np
import mne


def events_from_stim(raw: mne.io.BaseRaw, stim_channel: str) -> Tuple[np.ndarray, np.ndarray]:
    events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, initial_event=False)
    codes = events[:, 2].copy()
    return events, codes


def events_from_annotations_by_regex(
    raw: mne.io.BaseRaw,
    regex: str,
    *,
    align_base: str = "abs",  # "abs" or "rel"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Build events from annotations using regex with one capture group for an integer.
    align_base='abs' subtracts raw.first_time if ann.orig_time is not None.
    align_base='rel' does not subtract anything.
    Returns (events, codes, matched_annotation_count).
    """
    patt = re.compile(regex)
    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return None, None, 0

    sf = float(raw.info["sfreq"])
    use_abs = align_base == "abs" and ann.orig_time is not None
    base = raw.first_time if use_abs else 0.0

    ev: List[Tuple[int, int, int]] = []
    matched = 0
    for onset, desc in zip(ann.onset, ann.description):
        m = patt.search(desc)
        if not m:
            continue
        matched += 1
        code = int(m.group(1), 0)  # allow hex/dec
        samp = int(round((onset - base) * sf))
        ev.append((samp, 0, code))

    if matched == 0 or not ev:
        return None, None, matched

    events = np.asarray(ev, dtype=int)
    codes = events[:, 2].copy()
    return events, codes, matched


def nearest_dt_samples(ev_a: np.ndarray, ev_b: np.ndarray) -> np.ndarray:
    """
    For each event in ev_a, find the nearest event in ev_b (by sample index).
    Return an array of Δsamples = sa - sb, same length as ev_a.
    """
    sb = ev_b[:, 0]
    dts = []
    j = 0
    for sa in ev_a[:, 0]:
        while j + 1 < len(sb) and sb[j + 1] <= sa:
            j += 1
        candidates = [j] + ([j + 1] if j + 1 < len(sb) else [])
        best = min(candidates, key=lambda k: abs(sb[k] - sa))
        dts.append(sa - sb[best])
    return np.asarray(dts, dtype=int)


def match_events_by_time(
    ea: np.ndarray, eb: np.ndarray, tol_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy, order-preserving nearest-neighbor pairing of events by sample index.
    Returns arrays of indices (ia, ib) for matched pairs within +/- tol_samples.
    """
    ia = ib = 0
    pairs_a, pairs_b = [], []
    while ia < len(ea) and ib < len(eb):
        sa = ea[ia, 0]; sb = eb[ib, 0]
        if abs(sa - sb) <= tol_samples:
            pairs_a.append(ia); pairs_b.append(ib)
            ia += 1; ib += 1
        elif sa < sb:
            ia += 1
        else:
            ib += 1
    return np.asarray(pairs_a, int), np.asarray(pairs_b, int)


def main():
    ap = argparse.ArgumentParser(description="Detailed comparison of STI101 and Annotation triggers (per-event CSV).")
    ap.add_argument("fif", help="Path to preprocessed FIF (e.g., *_desc-preproc_meg.fif)")
    ap.add_argument("--stim-channel", default="STI101", help="Stim channel name (default: STI101)")
    ap.add_argument("--regex", default=r"TRIG/(\d+)",
                    help=r"Regex with one capture group for integer code in annotation description (default: 'TRIG/(\d+)')")
    ap.add_argument("--tol-sec", type=float, default=0.002, help="Time matching tolerance in seconds (default: 0.002)")
    ap.add_argument("--base", choices=["auto", "abs", "rel"], default="auto",
                    help="Annotation time base: abs=subtract first_time, rel=dont subtract, auto=pick better (default: auto)")
    ap.add_argument("--include-unmatched", action="store_true",
                    help="Print rows for unmatched events too (with blanks for the missing side)")
    args = ap.parse_args()

    raw = mne.io.read_raw_fif(args.fif, preload=False, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    tol_samples = int(round(args.tol_sec * sf))

    # --- Build stim events ---
    events_stim, codes_stim = events_from_stim(raw, args.stim_channel)

    # --- Build annotation events (abs/rel or auto choose) ---
    ev_abs, cd_abs, match_abs = events_from_annotations_by_regex(raw, args.regex, align_base="abs")
    ev_rel, cd_rel, match_rel = events_from_annotations_by_regex(raw, args.regex, align_base="rel")

    # Decide which annotation base to use
    use_mode = args.base
    if use_mode == "auto":
        # Prefer whichever yields smaller |median Δsamples| vs stim (when both exist)
        score_abs = score_rel = np.inf
        if ev_abs is not None and len(ev_abs) and len(events_stim):
            dt_abs = nearest_dt_samples(ev_abs, events_stim)
            score_abs = abs(np.median(dt_abs))
        if ev_rel is not None and len(ev_rel) and len(events_stim):
            dt_rel = nearest_dt_samples(ev_rel, events_stim)
            score_rel = abs(np.median(dt_rel))
        use_mode = "abs" if score_abs <= score_rel else "rel"

    if use_mode == "abs":
        events_ann, codes_ann, matched_total = ev_abs, cd_abs, match_abs
    else:
        events_ann, codes_ann, matched_total = ev_rel, cd_rel, match_rel

    # Header summary to stderr
    print(f"# sfreq={sf:.3f}Hz  meas_date={raw.info.get('meas_date')}  first_samp={raw.first_samp}  first_time={raw.first_time}")
    print(f"# annotations={0 if raw.annotations is None else len(raw.annotations)}  ann.orig_time={None if raw.annotations is None else raw.annotations.orig_time}")
    print(f"# stim_events={len(events_stim)}  annot_events={0 if events_ann is None else len(events_ann)}  annot_regex_matches={matched_total}")
    print(f"# base={use_mode}  tol_sec={args.tol_sec}  tol_samples={tol_samples}")
    print("# idx,stim_idx,stim_sample,stim_time_s,stim_code,annot_idx,annot_sample,annot_time_s,annot_code,dt_samples,dt_ms")

    if events_ann is None or len(events_ann) == 0:
        # Nothing to compare; still print STI lines if requested
        if args.include_unmatched:
            for ib in range(len(events_stim)):
                sb = events_stim[ib, 0]
                print(f"{ib},{ib},{sb},{(sb/raw.info['sfreq']):.6f},{codes_stim[ib]},,,,"
                      f",")  # no annot
        return

    # --- Match pairs within tolerance ---
    ia, ib = match_events_by_time(events_ann, events_stim, tol_samples)

    # Print matched pairs
    for k, (i_a, i_b) in enumerate(zip(ia, ib)):
        sa = int(events_ann[i_a, 0]); sb = int(events_stim[i_b, 0])
        ta = sa / sf; tb = sb / sf
        ca = int(codes_ann[i_a]); cb = int(codes_stim[i_b])
        dt_samp = sa - sb
        dt_ms = 1000.0 * (ta - tb)
        print(f"{k},{i_b},{sb},{tb:.6f},{cb},{i_a},{sa},{ta:.6f},{ca},{dt_samp},{dt_ms:.3f}")

    if not args.include_unmatched:
        return

    # --- Print unmatched on each side (with blanks for missing columns) ---
    matched_a = set(ia.tolist())
    matched_b = set(ib.tolist())

    # Unmatched annotations
    um_a = [i for i in range(len(events_ann)) if i not in matched_a]
    for i_a in um_a:
        sa = int(events_ann[i_a, 0]); ta = sa / sf; ca = int(codes_ann[i_a])
        print(f",,,,"  # no stim
              f"{i_a},{sa},{ta:.6f},{ca},,")

    # Unmatched stim
    um_b = [i for i in range(len(events_stim)) if i not in matched_b]
    for i_b in um_b:
        sb = int(events_stim[i_b, 0]); tb = sb / sf; cb = int(codes_stim[i_b])
        print(f"{i_b},{i_b},{sb},{tb:.6f},{cb},,,,,")  # no annot


if __name__ == "__main__":
    main()