#!/usr/bin/env python3
"""
Extract BIDS-style _events.tsv files from analog trigger bit lines.

This script is meant for Cerca OPM recordings (and other MNE-readable raw
files) where trigger bits are recorded as separate analog channels instead of a
single packed STI channel.

Key behavior
------------
1. Threshold each analog bit line into 0/1 samples.
2. Treat any 0 -> non-zero activity as a potential event onset.
3. Confirm the event code using the onset sample plus one or more look-ahead
   samples, so a trigger is not missed when different TTL lines are captured on
   adjacent digitized frames.
4. Record the event onset at the *first* active sample. The pulse duration is
   only used for filtering and diagnostics; the written BIDS duration is 0.
5. Optionally open the MNE raw browser for diagnostic review, overlay event
   markers, and allow the user to mark BAD spans to reject events.

Examples
--------
Basic use with default Trigger1..Trigger8 channel naming::

    python cerca_opm_events.py sub-01_task-oddball_meg.fif

More explicit channel selection::

    python cerca_opm_events.py raw.fif \
        --channels Trigger1[Z],Trigger2[Z],Trigger3[Z],Trigger4[Z],\\
                   Trigger5[Z],Trigger6[Z],Trigger7[Z],Trigger8[Z]

Diagnostic review with browser and interactive deletion::

    python cerca_opm_events.py raw.fif --diagnostic

Using a codebook to map integer values to trial_type and extra columns::

    python cerca_opm_events.py raw.fif --codebook trigger_codebook.tsv

The codebook must contain a ``value`` column. Any additional columns will be
copied into the final ``_events.tsv`` for matching trigger values.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "MNE-Python is required. Install it with: pip install mne"
    ) from exc


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class ChannelThreshold:
    name: str
    bit_index: int
    threshold: float
    polarity: str
    data_min: float
    data_max: float
    n_high: int
    pct_high: float


@dataclass
class EventCandidate:
    row: int
    onset_sample: int          # sample relative to start of file (0-based)
    offset_sample: int         # half-open interval [onset, offset)
    onset_time: float          # seconds relative to start of file
    pulse_samples: int
    pulse_ms: float
    value: int
    onset_word: int
    confirm_or_word: int
    mode_word: int
    full_or_word: int
    n_unique_nonzero_words: int
    flags: List[str] = field(default_factory=list)
    keep: bool = True
    drop_reason: str = ""


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract BIDS _events.tsv files from analog trigger bit lines in "
            "Cerca OPM / other MNE-readable raw data."
        )
    )
    parser.add_argument(
        "raw_file",
        help="Path to an MNE-readable raw file (for example .fif)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path of the final _events.tsv file. Default: same folder as input.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated bit-line channels in LSB->MSB order. "
            "If supplied, overrides --trigger-regex."
        ),
    )
    parser.add_argument(
        "--trigger-regex",
        default=r"^Trigger(\d+)\b",
        help=(
            "Regex used to find trigger channels when --channels is not given. "
            "Must contain exactly one capture group for the 1-based bit number. "
            "Default: ^Trigger(\\d+)\\b"
        ),
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=8,
        help="Number of trigger bit lines. Default: 8.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow missing bit channels. Missing bits are treated as always 0. "
            "By default all bit positions 1..nbits must be present."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Use a fixed threshold for every bit channel. By default a midpoint "
            "between channel min and max is used."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("minmax", "extrema-median"),
        default="minmax",
        help=(
            "Automatic threshold estimation method when --threshold is not given. "
            "Default: minmax."
        ),
    )
    parser.add_argument(
        "--min-level-snr",
        type=float,
        default=10.0,
        help=(
            "When automatic thresholding is used, a channel is treated as "
            "inactive (always zero) unless the separation between its estimated "
            "low and high levels is at least this many robust-noise units "
            "(MAD-based). This helps avoid false events on trigger lines that "
            "never changed state. Default: 10.0."
        ),
    )
    parser.add_argument(
        "--active-when",
        choices=("auto", "high", "low"),
        default="high",
        help=(
            "Whether an active TTL state is above threshold, below threshold, or "
            "automatically inferred from the direction of the larger excursion. "
            "Default: high (appropriate for standard TTL)."
        ),
    )
    parser.add_argument(
        "--lookahead-samples",
        type=int,
        default=1,
        help=(
            "Number of additional samples after onset to include when confirming "
            "the code. Default: 1 (onset sample + next sample)."
        ),
    )
    parser.add_argument(
        "--value-strategy",
        choices=("or", "mode", "persist"),
        default="or",
        help=(
            "How to choose the final event value within the confirmation window. "
            "'or' = bitwise OR across onset+lookahead samples; 'mode' = most "
            "frequent non-zero composite word during the full pulse; 'persist' = "
            "set a bit only if it is high in at least --min-bit-high-samples of "
            "the onset+lookahead window. Default: or."
        ),
    )
    parser.add_argument(
        "--min-bit-high-samples",
        type=int,
        default=1,
        help=(
            "For --value-strategy persist, a bit is considered set only if it is "
            "high in at least this many samples within the confirmation window. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--min-pulse-samples",
        type=int,
        default=2,
        help=(
            "Minimum length of a non-zero pulse to keep as a candidate event. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--merge-gap-samples",
        type=int,
        default=0,
        help=(
            "Merge activity separated by <= this many all-zero samples into one "
            "pulse. Default: 0."
        ),
    )
    parser.add_argument(
        "--drop-values",
        default="",
        help=(
            "Comma-separated trigger integer values to delete after detection. "
            "Example: 255,0x7F"
        ),
    )
    parser.add_argument(
        "--drop-rows",
        default="",
        help=(
            "Comma-separated 1-based row numbers from the diagnostic table to "
            "delete after detection. Example: 2,5,17"
        ),
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help=(
            "Optional TSV/CSV file with at least a 'value' column. Additional "
            "columns will be copied into the final _events.tsv."
        ),
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Open the MNE raw browser with trigger lines and event markers, write "
            "a diagnostic table, and allow interactive review."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="In diagnostic mode, skip opening the MNE raw browser.",
    )
    parser.add_argument(
        "--browser-backend",
        choices=("auto", "qt", "matplotlib"),
        default="auto",
        help=(
            "Browser backend to request for MNE raw.plot(). Default: auto."
        ),
    )
    parser.add_argument(
        "--browser-block",
        action="store_true",
        help=(
            "Pass block=True to the browser so the script waits until the plot "
            "window is closed. Usually not needed outside notebooks, but can be "
            "useful on some systems."
        ),
    )
    parser.add_argument(
        "--save-composite-channel",
        action="store_true",
        help=(
            "Include a synthetic COMPOSITE_TRIGGER channel in the diagnostic "
            "browser so the packed integer word can be inspected alongside the "
            "analog bit lines."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostic detail.",
    )
    return parser


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part, 0))
    return out


def popcount(value: int) -> int:
    return int(value).bit_count()


def default_events_path(raw_path: Path) -> Path:
    stem = raw_path.stem
    for suffix in ("_meg", "_eeg", "_ieeg", "_nirs", "_beh"):
        if stem.endswith(suffix):
            return raw_path.with_name(stem[: -len(suffix)] + "_events.tsv")
    return raw_path.with_name(stem + "_events.tsv")


def diagnostic_table_path(events_path: Path) -> Path:
    stem = events_path.stem
    if "_events" in stem:
        diag_stem = stem.replace("_events", "_events_diagnostics", 1)
    else:
        diag_stem = stem + "_diagnostics"
    return events_path.with_name(diag_stem + ".tsv")


def choose_polarity(data: np.ndarray, active_when: str) -> str:
    if active_when in {"high", "low"}:
        return active_when
    median = float(np.nanmedian(data))
    pos_exc = float(np.nanmax(data) - median)
    neg_exc = float(median - np.nanmin(data))
    return "high" if pos_exc >= neg_exc else "low"


def robust_mad_sigma(data: np.ndarray) -> float:
    data = np.asarray(data, dtype=float)
    median = float(np.nanmedian(data))
    mad = float(np.nanmedian(np.abs(data - median)))
    return 1.4826 * mad


def estimate_levels(data: np.ndarray, mode: str = "minmax") -> Tuple[float, float]:
    data = np.asarray(data, dtype=float)
    if mode == "minmax":
        lo = float(np.nanmin(data))
        hi = float(np.nanmax(data))
        return lo, hi
    if mode == "extrema-median":
        n = len(data)
        k = max(5, min(n // 10, int(np.ceil(0.001 * n))))
        # If data are very short, fall back cleanly.
        if k <= 0 or k * 2 > n:
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))
            return lo, hi
        sort = np.sort(data)
        lo = float(np.nanmedian(sort[:k]))
        hi = float(np.nanmedian(sort[-k:]))
        return lo, hi
    raise ValueError(f"Unknown threshold mode: {mode}")


def estimate_threshold(data: np.ndarray, mode: str = "minmax") -> float:
    lo, hi = estimate_levels(data, mode)
    return (lo + hi) / 2.0


def reduce_bitwise_or(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return 0
    return int(np.bitwise_or.reduce(values))


def modal_nonzero_word(words: np.ndarray) -> Tuple[int, int]:
    words = np.asarray(words, dtype=np.int64)
    words = words[words != 0]
    if words.size == 0:
        return 0, 0
    counts = Counter(int(v) for v in words)
    best_val = 0
    best_key = (-1, -1, -1)
    for value, count in counts.items():
        key = (count, popcount(value), value)
        if key > best_key:
            best_key = key
            best_val = value
    return best_val, counts[best_val]


def values_to_string(values: Sequence[int]) -> str:
    return ",".join(str(v) for v in values)


# -----------------------------------------------------------------------------
# Reading channels and binarizing them
# -----------------------------------------------------------------------------


def resolve_bit_channels(
    raw: mne.io.BaseRaw,
    channels_arg: str | None,
    trigger_regex: str,
    nbits: int,
    allow_missing: bool,
) -> List[str | None]:
    """Return a list of channel names in LSB->MSB order."""
    if channels_arg:
        names = [name.strip() for name in channels_arg.split(",") if name.strip()]
        if len(names) != nbits:
            raise SystemExit(
                f"--channels specified {len(names)} names but --nbits={nbits}. "
                "Provide exactly one channel per bit in LSB->MSB order."
            )
        missing = [name for name in names if name not in raw.ch_names]
        if missing:
            raise SystemExit(
                "The following --channels were not found in the raw file:\n  "
                + "\n  ".join(missing)
            )
        return names

    try:
        pattern = re.compile(trigger_regex)
    except re.error as exc:
        raise SystemExit(f"Invalid --trigger-regex: {exc}") from exc

    bit_map: Dict[int, str] = {}
    for ch_name in raw.ch_names:
        match = pattern.search(ch_name)
        if not match:
            continue
        if len(match.groups()) != 1:
            raise SystemExit(
                "--trigger-regex must contain exactly one capture group for the bit number."
            )
        bit_number = int(match.group(1))
        if 1 <= bit_number <= nbits:
            bit_map[bit_number] = ch_name

    if not bit_map:
        raise SystemExit(
            "No trigger channels were found.\n"
            f"Regex: {trigger_regex}\n"
            "Use --channels to specify them manually."
        )

    missing_bits = [bit for bit in range(1, nbits + 1) if bit not in bit_map]
    if missing_bits and not allow_missing:
        available = "\n  ".join(raw.ch_names)
        raise SystemExit(
            "Missing trigger channels for bit positions: "
            + ", ".join(str(b) for b in missing_bits)
            + "\nUse --allow-missing to treat them as always zero, or use --channels.\n"
            + "Available channels:\n  "
            + available
        )

    ordered = [bit_map.get(bit) for bit in range(1, nbits + 1)]
    return ordered


def binarize_bit_channels(
    raw: mne.io.BaseRaw,
    ordered_channels: Sequence[str | None],
    threshold: float | None,
    threshold_mode: str,
    active_when: str,
    min_level_snr: float,
) -> Tuple[np.ndarray, List[ChannelThreshold]]:
    """Return bit matrix of shape (nbits, n_samples) with values 0/1."""
    nbits = len(ordered_channels)
    n_samples = raw.n_times
    bit_matrix = np.zeros((nbits, n_samples), dtype=np.uint8)
    channel_info: List[ChannelThreshold] = []

    for bit_index, ch_name in enumerate(ordered_channels):
        if ch_name is None:
            channel_info.append(
                ChannelThreshold(
                    name=f"<missing bit {bit_index}>",
                    bit_index=bit_index,
                    threshold=float("nan"),
                    polarity="missing",
                    data_min=float("nan"),
                    data_max=float("nan"),
                    n_high=0,
                    pct_high=0.0,
                )
            )
            continue

        data, _ = raw[ch_name, :]
        data = np.asarray(data[0], dtype=float)
        lo, hi = estimate_levels(data, threshold_mode)
        thr = float(threshold) if threshold is not None else (lo + hi) / 2.0
        noise_sigma = robust_mad_sigma(data)
        polarity = choose_polarity(data, active_when)

        inactive = False
        if threshold is None:
            separation = abs(hi - lo)
            if noise_sigma == 0.0:
                inactive = separation == 0.0
            else:
                inactive = separation < (min_level_snr * noise_sigma)

        if inactive:
            bits = np.zeros_like(data, dtype=np.uint8)
            polarity = "inactive"
        elif polarity == "high":
            bits = (data > thr).astype(np.uint8)
        else:
            bits = (data < thr).astype(np.uint8)

        bit_matrix[bit_index] = bits
        n_high = int(bits.sum())
        pct_high = 100.0 * n_high / len(bits) if len(bits) else 0.0
        channel_info.append(
            ChannelThreshold(
                name=ch_name,
                bit_index=bit_index,
                threshold=thr,
                polarity=polarity,
                data_min=float(np.nanmin(data)),
                data_max=float(np.nanmax(data)),
                n_high=n_high,
                pct_high=pct_high,
            )
        )

    return bit_matrix, channel_info


def compose_word(bit_matrix: np.ndarray) -> np.ndarray:
    word = np.zeros(bit_matrix.shape[1], dtype=np.int64)
    for bit_index in range(bit_matrix.shape[0]):
        word += bit_matrix[bit_index].astype(np.int64) * (1 << bit_index)
    return word


# -----------------------------------------------------------------------------
# Event detection
# -----------------------------------------------------------------------------


def active_regions(any_high: np.ndarray, max_gap_samples: int = 0) -> List[Tuple[int, int]]:
    """Find contiguous or near-contiguous active regions [start, stop)."""
    active_idx = np.flatnonzero(any_high)
    if active_idx.size == 0:
        return []
    regions: List[Tuple[int, int]] = []
    start = int(active_idx[0])
    prev = int(active_idx[0])
    for idx in active_idx[1:]:
        idx = int(idx)
        gap = idx - prev - 1
        if gap > max_gap_samples:
            regions.append((start, prev + 1))
            start = idx
        prev = idx
    regions.append((start, prev + 1))
    return regions


def persist_word_from_window(
    bit_matrix: np.ndarray,
    start: int,
    stop: int,
    min_bit_high_samples: int,
) -> int:
    if start >= stop:
        return 0
    window = bit_matrix[:, start:stop]
    counts = window.sum(axis=1)
    out = 0
    for bit_index, count in enumerate(counts):
        if int(count) >= min_bit_high_samples:
            out |= 1 << bit_index
    return int(out)


def build_event_candidates(
    bit_matrix: np.ndarray,
    word: np.ndarray,
    sfreq: float,
    lookahead_samples: int,
    min_pulse_samples: int,
    merge_gap_samples: int,
    value_strategy: str,
    min_bit_high_samples: int,
) -> Tuple[List[EventCandidate], Dict[str, int]]:
    any_high = np.any(bit_matrix > 0, axis=0)
    regions = active_regions(any_high, max_gap_samples=merge_gap_samples)

    stats: Dict[str, int] = defaultdict(int)
    events: List[EventCandidate] = []

    for row_index, (start, stop) in enumerate(regions, start=1):
        pulse_len = stop - start
        if pulse_len < min_pulse_samples:
            stats["skipped_short_pulse"] += 1
            continue

        region_words = word[start:stop]
        region_nonzero = region_words[region_words != 0]
        if region_nonzero.size == 0:
            stats["skipped_zero_region"] += 1
            continue

        confirm_stop = min(stop, start + lookahead_samples + 1)
        confirm_words = word[start:confirm_stop]
        confirm_nonzero = confirm_words[confirm_words != 0]
        onset_word = int(word[start])
        confirm_or_word = reduce_bitwise_or(confirm_nonzero)
        mode_word, mode_count = modal_nonzero_word(region_nonzero)
        full_or_word = reduce_bitwise_or(region_nonzero)
        persist_word = persist_word_from_window(
            bit_matrix, start, confirm_stop, min_bit_high_samples=min_bit_high_samples
        )

        if value_strategy == "or":
            value = confirm_or_word
        elif value_strategy == "mode":
            value = mode_word
        elif value_strategy == "persist":
            value = persist_word
        else:  # pragma: no cover
            raise ValueError(f"Unexpected value strategy: {value_strategy}")

        if value == 0:
            stats["skipped_zero_value"] += 1
            continue

        counts = Counter(int(v) for v in region_nonzero)
        n_unique = len(counts)
        flags: List[str] = []
        if onset_word != value:
            flags.append("partial_or_staggered_onset")
        if n_unique > 1:
            flags.append("unstable_pulse")
        if mode_word != 0 and mode_word != value:
            flags.append("mode_mismatch")
        if full_or_word != value:
            flags.append("late_or_transient_bits")

        # Flag final bits that only appeared once in the whole pulse.
        bit_counts = bit_matrix[:, start:stop].sum(axis=1)
        weak_bits = [str(bit) for bit in range(bit_matrix.shape[0]) if (value & (1 << bit)) and bit_counts[bit] <= 1]
        if weak_bits:
            flags.append("weak_bits=" + ",".join(weak_bits))

        events.append(
            EventCandidate(
                row=len(events) + 1,
                onset_sample=start,
                offset_sample=stop,
                onset_time=start / sfreq,
                pulse_samples=pulse_len,
                pulse_ms=1000.0 * pulse_len / sfreq,
                value=int(value),
                onset_word=onset_word,
                confirm_or_word=confirm_or_word,
                mode_word=mode_word,
                full_or_word=full_or_word,
                n_unique_nonzero_words=n_unique,
                flags=flags,
            )
        )
        stats["kept_candidates"] += 1

    stats["n_regions"] = len(regions)
    stats["n_events"] = len(events)
    return events, stats


# -----------------------------------------------------------------------------
# Codebook support and BIDS writing
# -----------------------------------------------------------------------------


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    if path.suffix.lower() == ".tsv":
        return "\t"
    sample = path.read_text(encoding="utf-8")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t"


def load_codebook(path: Path | None) -> Tuple[Dict[int, Dict[str, str]], List[str]]:
    if path is None:
        return {}, []
    delimiter = sniff_delimiter(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or "value" not in reader.fieldnames:
            raise SystemExit(
                f"Codebook must contain a 'value' column: {path}"
            )
        fieldnames = [name for name in reader.fieldnames if name != "value"]
        mapping: Dict[int, Dict[str, str]] = {}
        for row in reader:
            value_text = (row.get("value") or "").strip()
            if not value_text:
                continue
            try:
                value = int(value_text, 0)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid codebook value '{value_text}' in {path}"
                ) from exc
            extra = {k: (row.get(k) or "") for k in fieldnames}
            mapping[value] = extra
    return mapping, fieldnames


def event_to_output_row(
    event: EventCandidate,
    codebook: Dict[int, Dict[str, str]],
    codebook_columns: Sequence[str],
) -> Dict[str, str]:
    row: Dict[str, str] = {
        "onset": f"{event.onset_time:.6f}",
        "duration": "0",
        "sample": str(event.onset_sample),
        "value": str(event.value),
    }

    extra = codebook.get(event.value, {})
    trial_type = extra.get("trial_type") or f"trigger_{event.value}"
    row["trial_type"] = trial_type

    for column in codebook_columns:
        if column == "trial_type":
            continue
        row[column] = extra.get(column, "")
    return row


def write_events_tsv(
    path: Path,
    events: Sequence[EventCandidate],
    codebook: Dict[int, Dict[str, str]],
    codebook_columns: Sequence[str],
) -> None:
    fieldnames = ["onset", "duration", "sample", "value", "trial_type"]
    fieldnames.extend([col for col in codebook_columns if col != "trial_type"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for event in events:
            if not event.keep:
                continue
            writer.writerow(event_to_output_row(event, codebook, codebook_columns))


def write_diagnostic_tsv(path: Path, events: Sequence[EventCandidate]) -> None:
    fieldnames = [
        "row",
        "keep",
        "drop_reason",
        "onset",
        "sample",
        "value",
        "pulse_samples",
        "pulse_ms",
        "onset_word",
        "confirm_or_word",
        "mode_word",
        "full_or_word",
        "n_unique_nonzero_words",
        "flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "row": event.row,
                    "keep": "1" if event.keep else "0",
                    "drop_reason": event.drop_reason,
                    "onset": f"{event.onset_time:.6f}",
                    "sample": event.onset_sample,
                    "value": event.value,
                    "pulse_samples": event.pulse_samples,
                    "pulse_ms": f"{event.pulse_ms:.3f}",
                    "onset_word": event.onset_word,
                    "confirm_or_word": event.confirm_or_word,
                    "mode_word": event.mode_word,
                    "full_or_word": event.full_or_word,
                    "n_unique_nonzero_words": event.n_unique_nonzero_words,
                    "flags": ";".join(event.flags),
                }
            )


# -----------------------------------------------------------------------------
# Diagnostics and review
# -----------------------------------------------------------------------------


def print_channel_summary(info: Sequence[ChannelThreshold]) -> None:
    print("\nTrigger channel summary")
    print("-----------------------")
    print(
        "bit  channel                     polarity   threshold     range                  high_samples"
    )
    for row in info:
        if row.polarity == "missing":
            print(f"{row.bit_index:>3}  <missing>                    missing    n/a         n/a                    0 (0.0%)")
            continue
        range_txt = f"[{row.data_min:.5g}, {row.data_max:.5g}]"
        print(
            f"{row.bit_index:>3}  {row.name:25.25s}  {row.polarity:8s}  {row.threshold:10.5g}  "
            f"{range_txt:22s}  {row.n_high:8d} ({row.pct_high:5.2f}%)"
        )


def print_event_summary(events: Sequence[EventCandidate], stats: Dict[str, int]) -> None:
    print("\nEvent extraction summary")
    print("------------------------")
    for key in sorted(stats):
        print(f"{key:>22s}: {stats[key]}")

    kept = [event for event in events if event.keep]
    if not kept:
        print("\nNo events kept.")
        return

    pulse_ms = np.array([event.pulse_ms for event in kept], dtype=float)
    print("\nPulse length statistics (kept events)")
    print("-------------------------------------")
    print(f"count: {len(pulse_ms)}")
    print(f"min  : {pulse_ms.min():.3f} ms")
    print(f"median: {np.median(pulse_ms):.3f} ms")
    print(f"mean : {pulse_ms.mean():.3f} ms")
    print(f"max  : {pulse_ms.max():.3f} ms")

    counts = Counter(event.value for event in kept)
    print("\nCounts by trigger value")
    print("-----------------------")
    print("value  count")
    for value, count in sorted(counts.items()):
        print(f"{value:5d}  {count:5d}")

    flagged = [event for event in kept if event.flags]
    if flagged:
        print("\nFlagged events")
        print("--------------")
        print("row  onset(s)   value  flags")
        for event in flagged[:50]:
            print(
                f"{event.row:3d}  {event.onset_time:8.3f}  {event.value:5d}  {';'.join(event.flags)}"
            )
        if len(flagged) > 50:
            print(f"... {len(flagged) - 50} additional flagged events not shown")


def print_event_table(events: Sequence[EventCandidate], max_rows: int = 40) -> None:
    print("\nDiagnostic event table")
    print("----------------------")
    print(
        "row  keep  onset(s)   sample   value   pulse(ms)  onset  ORwin  mode  fullOR  flags"
    )
    for event in events[:max_rows]:
        flags = ";".join(event.flags)
        if len(flags) > 30:
            flags = flags[:27] + "..."
        print(
            f"{event.row:3d}  {int(event.keep):4d}  {event.onset_time:8.3f}  {event.onset_sample:7d}  "
            f"{event.value:5d}  {event.pulse_ms:9.3f}  {event.onset_word:5d}  {event.confirm_or_word:5d}  "
            f"{event.mode_word:4d}  {event.full_or_word:6d}  {flags}"
        )
    if len(events) > max_rows:
        print(f"... {len(events) - max_rows} additional rows not shown")


def mark_drops_by_values(events: Sequence[EventCandidate], values: Iterable[int], reason_prefix: str) -> None:
    value_set = set(int(v) for v in values)
    for event in events:
        if event.value in value_set:
            event.keep = False
            event.drop_reason = (event.drop_reason + ";" if event.drop_reason else "") + f"{reason_prefix}{event.value}"


def mark_drops_by_rows(events: Sequence[EventCandidate], rows: Iterable[int], reason_prefix: str) -> None:
    row_set = set(int(r) for r in rows)
    for event in events:
        if event.row in row_set:
            event.keep = False
            event.drop_reason = (event.drop_reason + ";" if event.drop_reason else "") + f"{reason_prefix}{event.row}"


def drop_events_overlapping_bad_annotations(
    events: Sequence[EventCandidate],
    annotations: mne.Annotations,
) -> int:
    if len(annotations) == 0:
        return 0
    dropped = 0
    for annot in annotations:
        desc = str(annot["description"])
        if not desc.lower().startswith("bad"):
            continue
        start = float(annot["onset"])
        end = start + float(annot["duration"])
        for event in events:
            if not event.keep:
                continue
            if start <= event.onset_time <= end:
                event.keep = False
                event.drop_reason = (
                    event.drop_reason + ";" if event.drop_reason else ""
                ) + f"BAD_annotation:{desc}"
                dropped += 1
    return dropped


def open_browser_for_review(
    raw: mne.io.BaseRaw,
    trigger_channels: Sequence[str | None],
    word: np.ndarray,
    events: Sequence[EventCandidate],
    browser_backend: str,
    block: bool,
    save_composite_channel: bool,
) -> mne.io.BaseRaw:
    """Open MNE browser on a copy of raw and return the reviewed copy."""
    ch_names = [name for name in trigger_channels if name is not None]
    diag_raw = raw.copy().pick(ch_names)

    if save_composite_channel:
        composite = word.astype(float)[np.newaxis, :]
        info = mne.create_info(["COMPOSITE_TRIGGER"], sfreq=raw.info["sfreq"], ch_types=["stim"])
        comp_raw = mne.io.RawArray(composite, info, verbose=False)
        diag_raw.add_channels([comp_raw], force_update_info=True)

    # Add event markers for plotting. MNE events expect sample numbers in the raw
    # coordinate system, which includes first_samp when present.
    if events:
        plot_events = np.array(
            [[event.onset_sample + raw.first_samp, 0, event.value] for event in events if event.keep],
            dtype=int,
        )
    else:
        plot_events = None

    print(
        "\nOpening diagnostic browser.\n"
        "Tips:\n"
        "  - Event markers are shown as vertical lines.\n"
        "  - Press 'a' to enter annotation mode.\n"
        "  - Create annotations starting with BAD (for example BAD_drop) over\n"
        "    any events you want removed, then close the browser.\n"
        "  - Trigger events whose onsets fall inside BAD annotations will be dropped."
    )

    # Use a context manager only when a specific backend was requested.
    if browser_backend == "auto":
        diag_raw.plot(
            events=plot_events,
            n_channels=min(max(len(diag_raw.ch_names), 1), 20),
            title="Trigger diagnostics",
            duration=10.0,
            show=True,
            block=block,
            remove_dc=False,
        )
    else:
        with mne.viz.use_browser_backend(browser_backend):
            diag_raw.plot(
                events=plot_events,
                n_channels=min(max(len(diag_raw.ch_names), 1), 20),
                title="Trigger diagnostics",
                duration=10.0,
                show=True,
                block=block,
                remove_dc=False,
            )
    return diag_raw


def interactive_drop_prompt(events: Sequence[EventCandidate]) -> None:
    if not sys.stdin.isatty():
        return

    print(
        "\nInteractive review\n"
        "------------------\n"
        "You may now drop additional events by row number or by trigger value.\n"
        "Press Enter to keep everything currently marked as keep."
    )

    row_text = input("Rows to drop (comma-separated, based on the diagnostic table): ").strip()
    if row_text:
        rows = parse_int_list(row_text)
        mark_drops_by_rows(events, rows, reason_prefix="interactive_row:")

    value_text = input("Trigger values to drop everywhere (comma-separated): ").strip()
    if value_text:
        values = parse_int_list(value_text)
        mark_drops_by_values(events, values, reason_prefix="interactive_value:")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_path = Path(args.raw_file).expanduser().resolve()
    if not raw_path.exists():
        raise SystemExit(f"Input file not found: {raw_path}")

    events_path = Path(args.output).expanduser().resolve() if args.output else default_events_path(raw_path)
    diag_path = diagnostic_table_path(events_path)

    print(f"Loading raw file: {raw_path}")
    try:
        raw = mne.io.read_raw(raw_path, preload=True, verbose=False)
    except Exception as exc:
        raise SystemExit(f"Could not read raw file with mne.io.read_raw(): {exc}") from exc

    print(
        f"Loaded {raw.n_times} samples at {raw.info['sfreq']:.6g} Hz "
        f"({raw.n_times / raw.info['sfreq']:.3f} s)"
    )

    ordered_channels = resolve_bit_channels(
        raw=raw,
        channels_arg=args.channels,
        trigger_regex=args.trigger_regex,
        nbits=args.nbits,
        allow_missing=args.allow_missing,
    )

    print("\nBit channel order (LSB -> MSB)")
    print("-----------------------------")
    for bit_index, ch_name in enumerate(ordered_channels):
        print(f"bit {bit_index}: {ch_name if ch_name is not None else '<missing>'}")

    bit_matrix, channel_info = binarize_bit_channels(
        raw=raw,
        ordered_channels=ordered_channels,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        active_when=args.active_when,
        min_level_snr=float(args.min_level_snr),
    )
    word = compose_word(bit_matrix)

    print_channel_summary(channel_info)

    events, stats = build_event_candidates(
        bit_matrix=bit_matrix,
        word=word,
        sfreq=float(raw.info["sfreq"]),
        lookahead_samples=int(args.lookahead_samples),
        min_pulse_samples=int(args.min_pulse_samples),
        merge_gap_samples=int(args.merge_gap_samples),
        value_strategy=args.value_strategy,
        min_bit_high_samples=int(args.min_bit_high_samples),
    )

    if not events:
        print("\nNo events were detected.")
        write_diagnostic_tsv(diag_path, events)
        print(f"Wrote empty diagnostic table: {diag_path}")
        write_events_tsv(events_path, events, codebook={}, codebook_columns=[])
        print(f"Wrote empty events file: {events_path}")
        return

    # Apply CLI deletions first.
    drop_values = parse_int_list(args.drop_values)
    drop_rows = parse_int_list(args.drop_rows)
    if drop_values:
        mark_drops_by_values(events, drop_values, reason_prefix="cli_value:")
    if drop_rows:
        mark_drops_by_rows(events, drop_rows, reason_prefix="cli_row:")

    print_event_summary(events, stats)
    print_event_table(events)

    # Diagnostic review.
    if args.diagnostic:
        reviewed_raw = raw
        if not args.no_browser:
            reviewed_raw = open_browser_for_review(
                raw=raw,
                trigger_channels=ordered_channels,
                word=word,
                events=events,
                browser_backend=args.browser_backend,
                block=args.browser_block,
                save_composite_channel=args.save_composite_channel,
            )
            dropped_by_bad = drop_events_overlapping_bad_annotations(events, reviewed_raw.annotations)
            if dropped_by_bad:
                print(f"\nDropped {dropped_by_bad} events due to BAD annotations from the browser.")
        interactive_drop_prompt(events)

    # Re-numbering the row column is intentionally skipped so diagnostic row
    # numbers remain stable after review.
    codebook_path = Path(args.codebook).expanduser().resolve() if args.codebook else None
    codebook, codebook_columns = load_codebook(codebook_path)

    write_diagnostic_tsv(diag_path, events)
    kept_events = [event for event in events if event.keep]
    write_events_tsv(events_path, kept_events, codebook=codebook, codebook_columns=codebook_columns)

    print(f"\nWrote diagnostic table: {diag_path}")
    print(f"Wrote final events file: {events_path}")
    print(f"Kept {len(kept_events)} / {len(events)} detected events")

    # Small preview of final file.
    print("\nFinal _events.tsv preview")
    print("-------------------------")
    with events_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            print(line.rstrip())
            if line_no >= 15:
                remaining = sum(1 for _ in f)
                if remaining:
                    print("...")
                break


if __name__ == "__main__":
    main()
