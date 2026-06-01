#!/usr/bin/env python3
"""
MEG/EEG FIF annotation utility.

Reads a raw or preprocessed FIF file, optionally applies display-only filtering
to EEG/MEG channels in a browser-only copy, and opens the MNE interactive
browser for manual time-segment annotation and bad channel marking.

The original FIF file is never modified. Time annotations are written to a TSV
sidecar, and bad channels are exported to a BIDS-compliant _channels.tsv.

Key behaviors:
- Existing annotations already stored in the FIF are preserved by default.
- Existing annotation sidecars are also preserved by default.
- `--clear-current-annotations` starts the session with a blank annotation set.
- `--auto-detect-breaks` can add BAD-style break/trim annotations before review
  by looking for long event-free gaps between stimuli.
- Existing non-annotation event rows already present in a *_events.tsv sidecar
  are preserved when the sidecar is rewritten.
- Display filtering, when requested, is applied only to EEG/MEG channels in the
  browser copy. Stimulus and auxiliary channels are left untouched.
- The final annotation sidecar is always written, even if it is empty, so stale
  annotations are not left behind after a clearing/review session.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd

import matplotlib

try:  # Prefer Qt when available, but do not hard-fail if it is missing.
    matplotlib.use("QtAgg")
except Exception:
    pass


ANNOTATION_COLS = ["onset", "duration", "description", "trial_type"]
BIDS_ENTITY_SUFFIXES = ("_meg", "_eeg", "_ieeg", "_nirs", "_beh")
BADLIKE_PREFIXES = ("bad", "edge")


def parse_args():
    parser = argparse.ArgumentParser(description="MNE annotator for raw or preprocessed FIF data.")
    parser.add_argument("file", type=str, help="Path to raw or processed .fif/.fif.gz file")

    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help=(
            "Path to the annotation sidecar TSV. Default: for BIDS-like FIF names ending in "
            "_meg/_eeg/_ieeg/_nirs/_beh, use the matching *_events.tsv; otherwise use "
            "<stem>_annotations.tsv."
        ),
    )
    parser.add_argument(
        "--clear-current-annotations",
        action="store_true",
        help=(
            "Start the review session with no annotations. By default, annotations already "
            "stored in the FIF file and in the annotation sidecar are preserved."
        ),
    )

    # Bug-fix shift for previously written sidecars.
    parser.add_argument(
        "--shift-annotations",
        type=float,
        default=0.0,
        help="Shift loaded sidecar annotations by X seconds to correct previous offsets.",
    )

    # Break/trim auto-detection
    parser.add_argument(
        "--auto-detect-breaks",
        action="store_true",
        help=(
            "Detect long event-free intervals and add BAD-style break annotations before the "
            "browser opens. This includes the dead time before the first event and after the "
            "last event when those intervals exceed the minimum gap."
        ),
    )
    parser.add_argument(
        "--break-source",
        choices=["auto", "stim", "sidecar"],
        default="auto",
        help=(
            "Where event onsets should come from when --auto-detect-breaks is enabled. "
            "'auto' tries the stimulus channel first, then preserved event rows from existing "
            "sidecar TSVs."
        ),
    )
    parser.add_argument(
        "--break-stim-channel",
        type=str,
        default="STI101",
        help="Stimulus channel used for stim-based break detection (default: STI101).",
    )
    parser.add_argument(
        "--break-event-min-duration",
        type=float,
        default=0.002,
        help="min_duration passed to mne.find_events when using stim-based break detection.",
    )
    parser.add_argument(
        "--break-min-gap-sec",
        type=float,
        default=15.0,
        help="Minimum event-free interval, in seconds, to mark as a break (default: 15.0).",
    )
    parser.add_argument(
        "--break-pad-sec",
        type=float,
        default=2.0,
        help=(
            "Padding removed from both ends of each detected gap. For pre/post dead time, the "
            "padding is removed from the side adjacent to the first/last event (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--break-description",
        type=str,
        default="BAD_break",
        help="Description used for inter-run break annotations (default: BAD_break).",
    )
    parser.add_argument(
        "--break-edge-description",
        type=str,
        default=None,
        help=(
            "Description used for pre-first and post-last trim annotations. Default: same as "
            "--break-description."
        ),
    )
    parser.add_argument(
        "--keep-existing-break-annotations",
        action="store_true",
        help=(
            "When auto-detecting breaks, keep any existing annotations whose description matches "
            "the break labels instead of replacing them."
        ),
    )

    # Display Filtering (browser only)
    parser.add_argument(
        "--hpf",
        type=float,
        default=None,
        help="High-pass cutoff for the browser copy (EEG/MEG channels only; Hz)",
    )
    parser.add_argument(
        "-f",
        "--lpf",
        type=float,
        default=None,
        help="Low-pass cutoff for the browser copy (EEG/MEG channels only; Hz)",
    )

    # Robust Scaling Parameters
    parser.add_argument("--scale-window-sec", type=float, default=60.0)
    parser.add_argument("--scale-abs-quantile", type=float, default=0.99)
    parser.add_argument("--scale-channel-quantile", type=float, default=0.80)
    parser.add_argument("--scale-mult", type=float, default=1.2)

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help=(
            "Do not open the MNE browser. Useful for batch sidecar rewriting or automatic break "
            "annotation generation."
        ),
    )

    return parser.parse_args()


def _strip_fif_extension(name: str) -> str:
    for ext in (".fif.gz", ".fif"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _default_annotation_sidecar(fname: str) -> str:
    path = Path(fname)
    stem = _strip_fif_extension(path.name)
    for suffix in BIDS_ENTITY_SUFFIXES:
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            return str(path.with_name(f"{base}_events.tsv"))
    return str(path.with_name(f"{stem}_annotations.tsv"))


def _channels_sidecar_path(fname: str) -> str:
    path = Path(fname)
    stem = _strip_fif_extension(path.name)
    for suffix in BIDS_ENTITY_SUFFIXES:
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            return str(path.with_name(f"{base}_channels.tsv"))
    return str(path.with_name(f"{stem}_channels.tsv"))


def _alternate_annotation_sidecars(fname: str, primary: str) -> list[str]:
    path = Path(fname)
    stem = _strip_fif_extension(path.name)
    candidates = [
        str(path.with_name(f"{stem}_annotations.tsv")),
        str(path.with_name(f"{stem}_events.tsv")),
    ]
    seen = {primary}
    out = []
    for cand in candidates:
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out


def _choose_sidecar_for_session(fname: str, explicit_path: str | None) -> tuple[str, list[str]]:
    if explicit_path is not None:
        return explicit_path, []

    primary = _default_annotation_sidecar(fname)
    ordered_candidates = [primary] + _alternate_annotation_sidecars(fname, primary)
    existing = [cand for cand in ordered_candidates if os.path.exists(cand)]
    if existing:
        chosen = existing[0]
        ignored = existing[1:]
    else:
        chosen = primary
        ignored = []
    return chosen, ignored


def _empty_annotation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["onset", "duration", "description"])


def _annotation_df_from_mne(annotations: mne.Annotations, raw_like) -> pd.DataFrame:
    if annotations is None or len(annotations) == 0:
        return _empty_annotation_df()

    onsets = np.asarray(annotations.onset, dtype=float) - float(raw_like.first_time)
    onsets[np.isclose(onsets, 0.0, atol=1e-12)] = 0.0

    df = pd.DataFrame(
        {
            "onset": onsets.astype(float),
            "duration": np.asarray(annotations.duration, dtype=float),
            "description": np.asarray(annotations.description, dtype=str),
        }
    )
    return df


def _read_sidecar_table(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        # Preserve whatever headers are present, but still require onset/duration if rows exist.
        return df.copy()

    required = {"onset", "duration"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    return df


def _sidecar_description_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)

    desc = pd.Series("", index=df.index, dtype=object)
    if "description" in df.columns:
        desc = df["description"].fillna("").astype(str)
    if "trial_type" in df.columns:
        tt = df["trial_type"].fillna("").astype(str)
        desc = desc.where(desc.astype(str).str.strip() != "", tt)
    return desc.fillna("").astype(str)


def _preserved_event_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    desc = _sidecar_description_series(df)
    desc_lower = desc.str.strip().str.lower()
    badlike = desc_lower.str.startswith(BADLIKE_PREFIXES)

    numeric_value = pd.Series(False, index=df.index, dtype=bool)
    if "value" in df.columns:
        numeric_value = pd.to_numeric(df["value"], errors="coerce").notna()

    numeric_sample = pd.Series(False, index=df.index, dtype=bool)
    if "sample" in df.columns:
        numeric_sample = pd.to_numeric(df["sample"], errors="coerce").notna()

    trial_type_present = "trial_type" in df.columns
    if trial_type_present:
        desc_blank = desc.str.strip() == ""
        sample_based_event = numeric_sample & desc_blank
    else:
        sample_based_event = pd.Series(False, index=df.index, dtype=bool)

    preserve_mask = (~badlike) & (numeric_value | sample_based_event)
    return preserve_mask.astype(bool)


def _split_sidecar_table(
    df: pd.DataFrame, *, shift_seconds: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return _empty_annotation_df(), df.copy()

    work = df.copy()
    mask = _preserved_event_mask(work)
    desc = _sidecar_description_series(work)

    ann_rows = work.loc[~mask].copy()
    ann_df = pd.DataFrame(
        {
            "onset": pd.to_numeric(ann_rows.get("onset", pd.Series(dtype=float)), errors="coerce").astype(float)
            + float(shift_seconds),
            "duration": pd.to_numeric(ann_rows.get("duration", pd.Series(dtype=float)), errors="coerce")
            .fillna(0.0)
            .astype(float),
            "description": desc.loc[ann_rows.index].fillna("").astype(str),
        }
    )
    ann_df = ann_df[np.isfinite(ann_df["onset"]) & np.isfinite(ann_df["duration"])].reset_index(drop=True)

    preserved_events = work.loc[mask].copy().reset_index(drop=True)
    return ann_df, preserved_events


def _deduplicate_annotation_df(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return _empty_annotation_df(), 0

    work = df.copy()
    work["onset"] = pd.to_numeric(work["onset"], errors="coerce").astype(float)
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce").fillna(0.0).astype(float)
    work["description"] = work["description"].fillna("").astype(str)

    # Use a modest rounding tolerance so annotations that came from the same
    # sample boundaries are merged even if MNE or TSV I/O introduced tiny
    # floating-point differences.
    work["_onset_key"] = np.round(work["onset"].to_numpy(dtype=float), 6)
    work["_duration_key"] = np.round(work["duration"].to_numpy(dtype=float), 6)
    mask = ~work[["_onset_key", "_duration_key", "description"]].duplicated(keep="first")
    dropped = int((~mask).sum())

    deduped = (
        work.loc[mask, ["onset", "duration", "description"]]
        .sort_values(["onset", "duration", "description"], kind="stable")
        .reset_index(drop=True)
    )
    return deduped, dropped


def _remove_annotations_by_description(
    df: pd.DataFrame, descriptions: Iterable[str]
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return _empty_annotation_df(), 0

    desc_set = {str(item) for item in descriptions if str(item) != ""}
    if not desc_set:
        return df.copy(), 0

    work = df.copy()
    mask = ~work["description"].fillna("").astype(str).isin(desc_set)
    removed = int((~mask).sum())
    out = work.loc[mask].reset_index(drop=True)
    return out, removed


def _set_annotations_from_df(raw_obj, df: pd.DataFrame) -> None:
    if df.empty:
        raw_obj.set_annotations(mne.Annotations(onset=[], duration=[], description=[], orig_time=None))
        return

    ann = mne.Annotations(
        onset=df["onset"].to_numpy(dtype=float),
        duration=df["duration"].to_numpy(dtype=float),
        description=df["description"].astype(str).tolist(),
        orig_time=None,
    )
    raw_obj.set_annotations(ann)


def _insert_missing_column(order: list[str], column: str, *, after: str | None = None) -> list[str]:
    if column in order:
        return order
    if after is None or after not in order:
        order.append(column)
    else:
        idx = order.index(after) + 1
        order.insert(idx, column)
    return order


def _restore_integer_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    skip = {"onset", "duration"}
    for col in out.columns:
        if col in skip:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any() and numeric.notna().all():
            rounded = np.round(numeric.to_numpy(dtype=float))
            if np.allclose(numeric.to_numpy(dtype=float), rounded, atol=1e-12, rtol=0.0):
                out[col] = pd.Series(rounded.astype(np.int64), index=out.index, dtype=object)
    return out


def _combine_preserved_events_and_annotations(
    preserved_events_df: pd.DataFrame | None, annotation_df: pd.DataFrame
) -> pd.DataFrame:
    ann_rows = annotation_df.copy()
    if ann_rows.empty:
        ann_rows = pd.DataFrame(columns=ANNOTATION_COLS)
    else:
        ann_rows["onset"] = pd.to_numeric(ann_rows["onset"], errors="coerce").astype(float)
        ann_rows["duration"] = pd.to_numeric(ann_rows["duration"], errors="coerce").fillna(0.0).astype(float)
        ann_rows["description"] = ann_rows["description"].fillna("").astype(str)
        ann_rows["trial_type"] = ann_rows["description"]
        ann_rows = ann_rows[ANNOTATION_COLS]

    if preserved_events_df is None or preserved_events_df.empty:
        combined = ann_rows.copy()
        if combined.empty:
            return pd.DataFrame(columns=ANNOTATION_COLS)
        combined = combined.sort_values(["onset", "duration", "description"], kind="stable").reset_index(drop=True)
        return combined

    base = _restore_integer_like_columns(preserved_events_df.copy().reset_index(drop=True))

    # Preserve the original event columns and append annotation-friendly columns
    # when they are absent.
    columns = list(base.columns)
    columns = _insert_missing_column(columns, "description", after="duration")
    columns = _insert_missing_column(columns, "trial_type", after="description")

    base_aligned = base.copy()
    if "description" not in base_aligned.columns:
        base_aligned["description"] = pd.NA
    if "trial_type" not in base_aligned.columns:
        base_aligned["trial_type"] = pd.NA

    ann_aligned = ann_rows.copy()
    for col in columns:
        if col not in ann_aligned.columns:
            ann_aligned[col] = pd.NA
    ann_aligned = ann_aligned[columns]
    base_aligned = base_aligned.reindex(columns=columns)

    combined = pd.concat([base_aligned, ann_aligned], axis=0, ignore_index=True, sort=False)

    # Keep annotation rows easy to read and preserve event ordering by onset.
    if "onset" in combined.columns:
        combined["onset"] = pd.to_numeric(combined["onset"], errors="coerce").astype(float).round(6)
    if "duration" in combined.columns:
        combined["duration"] = pd.to_numeric(combined["duration"], errors="coerce").fillna(0.0).astype(float).round(6)

    sort_cols = [col for col in ("onset", "duration", "description", "trial_type") if col in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return combined


def _write_sidecar(tsv_path: str, preserved_events_df: pd.DataFrame | None, annotation_df: pd.DataFrame) -> None:
    out = _combine_preserved_events_and_annotations(preserved_events_df, annotation_df)
    if out.empty:
        out = pd.DataFrame(columns=ANNOTATION_COLS)
        out.to_csv(tsv_path, sep="\t", index=False)
        return
    out.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")


def _picks_for_type(info, typ: str):
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


def _robust_scaling_for_type(raw, picks, window_sec, abs_q, ch_q, mult):
    if len(picks) == 0:
        return None
    sfreq = float(raw.info["sfreq"])
    n = int(min(raw.n_times, max(1, round(window_sec * sfreq))))
    data = raw.get_data(picks=picks, start=0, stop=n)
    if data.size == 0:
        return None
    per_ch = np.quantile(np.abs(data), abs_q, axis=1)
    scale = float(np.quantile(per_ch, ch_q) * mult)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale


def compute_browser_scalings(raw_browser, args):
    scalings = {}
    for typ in ["mag", "grad", "eeg", "eog", "ecg"]:
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


def _browser_filter_picks(info):
    """Return channel indices for browser-only EEG/MEG filtering."""
    return mne.pick_types(info, meg=True, eeg=True, ref_meg=False, exclude=[])


def _count_browser_filter_channel_types(raw_like, picks) -> tuple[int, int]:
    if len(picks) == 0:
        return 0, 0
    channel_types = raw_like.get_channel_types(picks=picks)
    meg_count = sum(ch_type in {"mag", "grad"} for ch_type in channel_types)
    eeg_count = sum(ch_type == "eeg" for ch_type in channel_types)
    return meg_count, eeg_count


def _event_onsets_from_stim(raw, stim_channel: str, min_duration: float) -> tuple[np.ndarray | None, str]:
    try:
        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=min_duration, verbose=False)
    except Exception as exc:
        return None, f"Stim-based break detection failed on channel {stim_channel!r}: {exc}"

    if events is None or len(events) == 0:
        return None, f"No events were found on stimulus channel {stim_channel!r}."

    sfreq = float(raw.info["sfreq"])
    onsets = (events[:, 0].astype(float) - float(raw.first_samp)) / sfreq
    onsets = np.asarray(onsets, dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0:
        return None, f"Stimulus channel {stim_channel!r} produced no usable event onsets."
    onsets = np.unique(np.round(onsets, 9))
    return onsets, f"Using {len(onsets)} event onset(s) from stimulus channel {stim_channel!r}."


def _event_onsets_from_preserved_rows(preserved_rows: pd.DataFrame, sfreq: float) -> tuple[np.ndarray | None, str]:
    if preserved_rows is None or preserved_rows.empty:
        return None, "No preserved event rows were available in sidecar TSVs."

    if "onset" not in preserved_rows.columns:
        return None, "Preserved sidecar event rows do not have an 'onset' column."

    onset = pd.to_numeric(preserved_rows["onset"], errors="coerce")
    if "sample" in preserved_rows.columns:
        sample = pd.to_numeric(preserved_rows["sample"], errors="coerce")
        use_sample = sample.notna()
        onset = onset.astype(float)
        onset.loc[use_sample] = sample.loc[use_sample].astype(float) / float(sfreq)

    onsets = onset.to_numpy(dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0:
        return None, "Preserved sidecar event rows did not contain any usable event onsets."

    onsets = np.unique(np.round(onsets, 9))
    return onsets, f"Using {len(onsets)} event onset(s) from preserved sidecar event rows."


def _load_preserved_event_rows(paths: list[str]) -> tuple[pd.DataFrame, list[str]]:
    pieces: list[pd.DataFrame] = []
    notes: list[str] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            full_df = _read_sidecar_table(path)
            _, preserved = _split_sidecar_table(full_df, shift_seconds=0.0)
            if preserved is not None and not preserved.empty:
                piece = preserved.copy()
                piece["__source_sidecar"] = path
                pieces.append(piece)
                notes.append(f"Found {len(piece)} preserved event row(s) in {path}")
        except Exception as exc:
            notes.append(f"Could not read sidecar event rows from {path}: {exc}")
    if pieces:
        merged = pd.concat(pieces, axis=0, ignore_index=True, sort=False)
        return merged, notes
    return pd.DataFrame(), notes


def _detect_break_annotations_from_onsets(
    event_onsets_sec: np.ndarray,
    recording_duration_sec: float,
    *,
    min_gap_sec: float,
    pad_sec: float,
    break_description: str,
    edge_description: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float | str]]]:
    edge_description = break_description if edge_description is None else edge_description

    rows: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    if event_onsets_sec is None or len(event_onsets_sec) == 0:
        return _empty_annotation_df(), summaries

    onsets = np.asarray(event_onsets_sec, dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0:
        return _empty_annotation_df(), summaries

    onsets = np.sort(np.unique(np.clip(onsets, 0.0, float(recording_duration_sec))))
    recording_duration_sec = float(max(0.0, recording_duration_sec))
    pad_sec = float(max(0.0, pad_sec))
    min_gap_sec = float(max(0.0, min_gap_sec))

    def _append(label: str, kind: str, onset_sec: float, end_sec: float) -> None:
        onset_sec = max(0.0, float(onset_sec))
        end_sec = min(recording_duration_sec, float(end_sec))
        duration_sec = end_sec - onset_sec
        if duration_sec <= 0:
            return
        rows.append(
            {
                "onset": onset_sec,
                "duration": duration_sec,
                "description": label,
            }
        )
        summaries.append(
            {
                "kind": kind,
                "onset": onset_sec,
                "duration": duration_sec,
                "description": label,
            }
        )

    first_event = float(onsets[0])
    if first_event > min_gap_sec:
        _append(edge_description, "pre_experiment", 0.0, first_event - pad_sec)

    for prev, curr in zip(onsets[:-1], onsets[1:]):
        gap = float(curr - prev)
        if gap > min_gap_sec:
            _append(break_description, "inter_run", prev + pad_sec, curr - pad_sec)

    last_event = float(onsets[-1])
    tail_gap = float(recording_duration_sec - last_event)
    if tail_gap > min_gap_sec:
        _append(edge_description, "post_experiment", last_event + pad_sec, recording_duration_sec)

    if not rows:
        return _empty_annotation_df(), summaries

    df = pd.DataFrame(rows)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce").astype(float)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").astype(float)
    df["description"] = df["description"].fillna("").astype(str)
    return df, summaries


def _auto_detect_breaks(
    raw,
    args,
    *,
    chosen_sidecar_path: str,
    ignored_sidecars: list[str],
    preserved_events_from_chosen: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    messages: list[str] = []
    event_onsets = None

    sidecar_search_paths = [path for path in ignored_sidecars if os.path.exists(path)]
    if preserved_events_from_chosen is None or preserved_events_from_chosen.empty:
        if chosen_sidecar_path and os.path.exists(chosen_sidecar_path):
            sidecar_search_paths = [chosen_sidecar_path] + sidecar_search_paths

    if args.break_source in ("auto", "stim"):
        event_onsets, msg = _event_onsets_from_stim(
            raw, stim_channel=args.break_stim_channel, min_duration=args.break_event_min_duration
        )
        messages.append(msg)

    if event_onsets is None and args.break_source in ("auto", "sidecar"):
        preserved_pool = preserved_events_from_chosen.copy() if preserved_events_from_chosen is not None else pd.DataFrame()
        extra_preserved, notes = _load_preserved_event_rows(sidecar_search_paths)
        messages.extend(notes)
        if extra_preserved is not None and not extra_preserved.empty:
            if preserved_pool.empty:
                preserved_pool = extra_preserved.copy()
            else:
                preserved_pool = pd.concat([preserved_pool, extra_preserved], axis=0, ignore_index=True, sort=False)
        event_onsets, msg = _event_onsets_from_preserved_rows(preserved_pool, sfreq=float(raw.info["sfreq"]))
        messages.append(msg)

    if event_onsets is None:
        messages.append("Automatic break detection did not find any usable event onsets.")
        return _empty_annotation_df(), messages

    recording_duration_sec = float(raw.n_times) / float(raw.info["sfreq"])
    break_df, summaries = _detect_break_annotations_from_onsets(
        event_onsets,
        recording_duration_sec,
        min_gap_sec=args.break_min_gap_sec,
        pad_sec=args.break_pad_sec,
        break_description=args.break_description,
        edge_description=args.break_edge_description,
    )

    if break_df.empty:
        messages.append(
            "Automatic break detection found event onsets, but no gap exceeded the requested minimum duration."
        )
    else:
        messages.append(f"Automatic break detection produced {len(break_df)} BAD-style annotation segment(s).")
        for idx, item in enumerate(summaries, start=1):
            messages.append(
                f"  {idx:>2}. {item['kind']}: onset={item['onset']:.3f}s "
                f"duration={item['duration']:.3f}s desc={item['description']}"
            )
    return break_df, messages


def main():
    args = parse_args()
    fname = args.file

    if not os.path.exists(fname):
        print(f"ERROR: File '{fname}' does not exist.")
        sys.exit(1)

    annotation_file, ignored_sidecars = _choose_sidecar_for_session(fname, args.annotation_file)

    print(f"\n=== Loading file: {fname} ===")
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=True)

    print(f"Annotation sidecar for this session:\n  {annotation_file}")
    if ignored_sidecars:
        print("Additional sidecar candidate(s) were found but will not be written in this session:")
        for extra in ignored_sidecars:
            print(f"  - {extra}")

    preserved_event_rows_from_chosen = pd.DataFrame()
    sidecar_annotation_df = _empty_annotation_df()
    existing_sidecar_loaded = False

    if os.path.exists(annotation_file):
        print(f"Found existing sidecar. Inspecting:\n  {annotation_file}")
        try:
            sidecar_full_df = _read_sidecar_table(annotation_file)
            sidecar_annotation_df, preserved_event_rows_from_chosen = _split_sidecar_table(
                sidecar_full_df, shift_seconds=args.shift_annotations
            )
            existing_sidecar_loaded = True

            if not preserved_event_rows_from_chosen.empty:
                print(
                    f"Preserving {len(preserved_event_rows_from_chosen)} non-annotation event row(s) already "
                    "present in the sidecar."
                )
            if not sidecar_annotation_df.empty:
                print(f"Loaded {len(sidecar_annotation_df)} annotation row(s) from the sidecar.")
            elif sidecar_full_df.empty:
                print("The existing sidecar is empty (header only).")
            else:
                print("The existing sidecar did not contain any annotation rows to preload into the browser.")
            if args.shift_annotations != 0 and not sidecar_annotation_df.empty:
                print(f"Applied a temporal shift of {args.shift_annotations} second(s) to sidecar annotations.")
        except Exception as exc:
            print(f"ERROR reading sidecar TSV: {exc}")
    else:
        if args.shift_annotations != 0:
            print("No sidecar exists yet, so --shift-annotations had nothing to shift.")
        print(f"No existing sidecar found. One will be written to:\n  {annotation_file}")

    if args.clear_current_annotations:
        embedded_count = len(raw.annotations)
        if embedded_count:
            print(f"\nClearing {embedded_count} annotation(s) already present in the FIF before review.")
        else:
            print("\nStarting with no embedded FIF annotations.")
        if existing_sidecar_loaded and not sidecar_annotation_df.empty:
            print(f"Ignoring {len(sidecar_annotation_df)} existing sidecar annotation row(s) for this session.")
        session_annotations = _empty_annotation_df()
    else:
        merged_parts = []

        fif_df = _annotation_df_from_mne(raw.annotations, raw)
        if not fif_df.empty:
            print(f"\nPreserving {len(fif_df)} annotation(s) already present in the FIF file.")
            merged_parts.append(fif_df)
        else:
            print("\nNo annotations were embedded in the FIF file.")

        if existing_sidecar_loaded and not sidecar_annotation_df.empty:
            merged_parts.append(sidecar_annotation_df)
        elif not existing_sidecar_loaded:
            print("No existing annotation sidecar rows were available to preload.")

        if merged_parts:
            session_annotations = pd.concat(merged_parts, axis=0, ignore_index=True)
            session_annotations, duplicate_count = _deduplicate_annotation_df(session_annotations)
            print(f"Starting review with {len(session_annotations)} annotation(s).")
            if duplicate_count:
                print(
                    f"Removed {duplicate_count} duplicate annotation row(s) while merging FIF and sidecar annotations."
                )
        else:
            session_annotations = _empty_annotation_df()
            print("Starting review with an empty annotation set.")

    if args.auto_detect_breaks:
        print("\n--- AUTOMATIC BREAK DETECTION ---")
        edge_desc = args.break_edge_description or args.break_description
        if not args.keep_existing_break_annotations:
            session_annotations, removed_count = _remove_annotations_by_description(
                session_annotations,
                descriptions=[args.break_description, edge_desc],
            )
            if removed_count:
                print(
                    f"Removed {removed_count} existing annotation row(s) whose description matched the "
                    "auto-break label(s) before regenerating them."
                )

        auto_break_df, break_messages = _auto_detect_breaks(
            raw,
            args,
            chosen_sidecar_path=annotation_file,
            ignored_sidecars=ignored_sidecars,
            preserved_events_from_chosen=preserved_event_rows_from_chosen,
        )
        for line in break_messages:
            print(line)

        if not auto_break_df.empty:
            if session_annotations.empty:
                session_annotations = auto_break_df.copy().reset_index(drop=True)
            else:
                session_annotations = pd.concat([session_annotations, auto_break_df], axis=0, ignore_index=True)
            session_annotations, duplicate_count = _deduplicate_annotation_df(session_annotations)
            if duplicate_count:
                print(f"Removed {duplicate_count} duplicate annotation row(s) after adding auto-detected breaks.")

    _set_annotations_from_df(raw, session_annotations)
    # Prepare browser-only display copy. The original FIF on disk is never changed.
    raw_browser = raw.copy()
    if (args.hpf is not None) or (args.lpf is not None):
        filter_picks = _browser_filter_picks(raw_browser.info)
        if len(filter_picks) == 0:
            print(
                "\nDisplay filtering was requested, but no EEG/MEG channels were found. "
                "The browser copy will be shown without filtering."
            )
        else:
            meg_count, eeg_count = _count_browser_filter_channel_types(raw_browser, filter_picks)
            print(
                "\nFiltering EEG/MEG channels in the browser copy only: "
                f"HPF={args.hpf} Hz, LPF={args.lpf} Hz "
                f"(MEG={meg_count}, EEG={eeg_count}; stim/aux channels left unfiltered)"
            )
            raw_browser.filter(
                l_freq=args.hpf,
                h_freq=args.lpf,
                picks=filter_picks,
                fir_design="firwin",
                verbose=False,
            )
    else:
        print("\nNo display filtering requested.")

    scalings = compute_browser_scalings(raw_browser, args)
    if scalings:
        print("Applied robust browser scalings to balance MAG, GRAD, and EEG channel amplitudes.")

    if args.no_browser:
        print("\n--no-browser was requested, so the MNE browser will not be opened.")
    else:
        print("\n--- INTERACTIVE ANNOTATION MODE ---")
        print("  1. Press 'a' to enter time-segment annotation mode.")
        print("  2. Click and drag to mark segments.")
        print("  3. Click a channel name to mark it as bad (it will turn gray).")
        print("  4. Close the browser window to save the sidecar and exit.\n")

        raw_browser.plot(
            title="Annotator",
            scalings=scalings if scalings else None,
            block=True,
            remove_dc=False,
        )

    final_df = _annotation_df_from_mne(raw_browser.annotations, raw_browser)
    final_df, duplicate_count = _deduplicate_annotation_df(final_df)
    if duplicate_count:
        print(f"Removed {duplicate_count} duplicate annotation row(s) before writing the sidecar.")

    _write_sidecar(annotation_file, preserved_event_rows_from_chosen, final_df)
    if final_df.empty:
        if preserved_event_rows_from_chosen is not None and not preserved_event_rows_from_chosen.empty:
            print("\nSaved the sidecar with preserved event rows and no annotation rows.")
        else:
            print("\nSaved an empty annotation sidecar (header only).")
    else:
        print(f"\nSUCCESS: Saved {len(final_df)} annotation segment(s) to:")
    print(f"  -> {annotation_file}")

    # ==========================================
    # BAD CHANNEL EXPORT (_channels.tsv)
    # ==========================================
    bads = raw_browser.info.get("bads", [])
    channels_file = _channels_sidecar_path(fname)

    if args.no_browser:
        print("\nNo bad channels were marked because the browser was not opened.")
    else:
        # Check if we are updating an existing BIDS file or creating a new one
        if os.path.exists(channels_file):
            try:
                ch_df = pd.read_csv(channels_file, sep='\t')
                if 'name' in ch_df.columns:
                    # Safely update only the status column based on current bads
                    ch_df['status'] = ch_df['name'].apply(lambda x: 'bad' if x in bads else 'good')
                    ch_df.to_csv(channels_file, sep='\t', index=False, na_rep='n/a')
                    print(f"\nSUCCESS: Updated existing channel sidecar.")
                    print(f"  -> {channels_file}")
                else:
                    print(f"\nWARNING: Existing {channels_file} lacks a 'name' column. Could not update.")
            except Exception as e:
                print(f"\nERROR updating {channels_file}: {e}")
        else:
            # Create a brand new _channels.tsv from scratch
            ch_data = {
                'name': raw_browser.ch_names,
                'type': raw_browser.get_channel_types(),
                'status': ['bad' if ch in bads else 'good' for ch in raw_browser.ch_names]
            }
            pd.DataFrame(ch_data).to_csv(channels_file, sep='\t', index=False)
            print(f"\nSUCCESS: Created new channel sidecar.")
            print(f"  -> {channels_file}")

        # Print summary
        if bads:
            print(f"  -> Marked {len(bads)} channel(s) as bad: {', '.join(bads)}")
        else:
            print("  -> No bad channels were marked during this session.")


if __name__ == "__main__":
    main()