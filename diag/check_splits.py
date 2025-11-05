#!/usr/bin/env python3
"""
check_splits.py — Diagnose MEG FIF split chains.

Usage:
  python check_splits.py /path/to/sub-003_ses-01_task-test_split-01_meg.fif
  python check_splits.py /path/to/sub-003_task-test_meg.fif
  python check_splits.py --mne /path/to/file.fif

What it does:
  - Detects split *style* from the filename:
      BIDS-style: ..._split-01_meg.fif, ..._split-02_meg.fif, ...
      MEGIN-style: ..._meg.fif, ..._meg-1.fif, ..._meg-2.fif, ...
  - Scans the directory to list the contiguous chain it *expects* to exist.
  - (Optional) With --mne, tries mne.io.read_raw_fif() and prints raw.filenames
    as resolved by internal FIFF "next-file" pointers.

Exit codes:
  0 = OK (chain found and contiguous)
  1 = Nonexistent input file
  2 = Split style unrecognized
  3 = Gaps or missing expected files
  4 = MNE read failed (only when --mne is used)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

MAX_PARTS = 128  # safety cap

BIDS_RE   = re.compile(r"^(?P<prefix>.*)_split-(?P<idx>\d{2})_meg\.fif$")
MEGIN1_RE = re.compile(r"^(?P<prefix>.*)_meg-(?P<idx>\d+)\.fif$")
MEGIN0_RE = re.compile(r"^(?P<prefix>.*)_meg\.fif$")


def detect_style(fname: str) -> Tuple[str, str, int]:
    """
    Return (style, prefix, start_idx)

    style in {"bids", "megin", "megin_base"}:
      - "bids":  prefix + _split-%02d_meg.fif, start at given idx (usually 1)
      - "megin": prefix + _meg-%d.fif, start at given idx (usually 1)
      - "megin_base": prefix + _meg.fif (first), then _meg-1.fif, -2.fif, ...
    """
    m = BIDS_RE.match(fname)
    if m:
        return "bids", m.group("prefix"), int(m.group("idx"))

    m = MEGIN1_RE.match(fname)
    if m:
        return "megin", m.group("prefix"), int(m.group("idx"))

    m = MEGIN0_RE.match(fname)
    if m:
        return "megin_base", m.group("prefix"), 0

    return "", "", -1


def expected_chain(style: str, prefix: str, start_idx: int, directory: Path) -> List[Path]:
    """
    Build the expected contiguous file list *that exist on disk* (no gaps).
    """
    found: List[Path] = []

    if style == "bids":
        # _split-XX from start_idx upward while files exist and contiguous
        idx = start_idx
        while idx <= MAX_PARTS:
            p = directory / f"{prefix}_split-{idx:02d}_meg.fif"
            if not p.exists():
                break
            found.append(p)
            idx += 1

    elif style == "megin":
        # _meg-<idx>, _meg-<idx+1>, ...
        idx = start_idx
        while idx <= MAX_PARTS:
            p = directory / f"{prefix}_meg-{idx}.fif"
            if not p.exists():
                break
            found.append(p)
            idx += 1

        # If user provided _meg-1.fif as input, check if there is also a base _meg.fif before it
        base = directory / f"{prefix}_meg.fif"
        if base.exists():
            found.insert(0, base)

    elif style == "megin_base":
        # start with base _meg.fif then _meg-1.fif, _meg-2.fif, ...
        base = directory / f"{prefix}_meg.fif"
        if base.exists():
            found.append(base)
        idx = 1
        while idx <= MAX_PARTS:
            p = directory / f"{prefix}_meg-{idx}.fif"
            if not p.exists():
                break
            found.append(p)
            idx += 1

    return found


def contiguous_ok(style: str, prefix: str, files: List[Path]) -> Tuple[bool, Optional[str]]:
    """
    Verify there are no gaps between expected parts for the detected style.
    Returns (ok, msg_if_problem).
    """
    if not files:
        return False, "No parts found."

    # For BIDS, check split numbers are 01..N with no gaps.
    if style == "bids":
        nums = []
        for p in files:
            m = BIDS_RE.match(p.name)
            if not m:
                return False, f"Unexpected filename pattern: {p.name}"
            nums.append(int(m.group("idx")))
        nums_sorted = sorted(nums)
        for i, n in enumerate(nums_sorted, start=nums_sorted[0]):
            if n != i:
                return False, f"Gap in BIDS splits (expected split-{i:02d}, found split-{n:02d})."
        return True, None

    # For MEGIN, if base present it should be first, followed by -1, -2, ...
    if style in {"megin", "megin_base"}:
        base_first = files[0].name.endswith("_meg.fif")
        # build expected sequence of indices after base (if base present)
        start_idx = 1 if base_first else int(MEGIN1_RE.match(files[0].name).group("idx"))
        idxs = []
        for p in files:
            if p.name.endswith("_meg.fif"):
                continue
            m = MEGIN1_RE.match(p.name)
            if not m:
                return False, f"Unexpected filename pattern: {p.name}"
            idxs.append(int(m.group("idx")))
        if idxs:
            idxs_sorted = sorted(idxs)
            for i, n in enumerate(idxs_sorted, start=start_idx):
                if n != i:
                    return False, f"Gap in MEGIN splits (expected -{i}, found -{n})."
        return True, None

    return False, "Unrecognized style during contiguity check."


def main():
    ap = argparse.ArgumentParser(description="Diagnose MEG FIF split chains.")
    ap.add_argument("path", help="Path to a FIF file (any part of the chain).")
    ap.add_argument("--mne", action="store_true",
                    help="Also attempt MNE read and print raw.filenames resolved by FIFF pointers.")
    args = ap.parse_args()

    in_path = Path(args.path).expanduser().resolve()
    if not in_path.exists():
        print(f"[ERROR] File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    style, prefix, start_idx = detect_style(in_path.name)
    if not style:
        print(f"[ERROR] Could not infer split style from filename: {in_path.name}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Detected style: {style}")
    print(f"[INFO] Directory: {in_path.parent}")
    print(f"[INFO] Prefix:    {prefix}")
    if style in {"bids", "megin"}:
        print(f"[INFO] Start idx: {start_idx}")

    files = expected_chain(style, prefix, start_idx, in_path.parent)
    if not files:
        print("[ERROR] No expected parts found on disk.", file=sys.stderr)
        sys.exit(3)

    print("\n[FOUND] Contiguous parts on disk:")
    for p in files:
        print(f"  - {p.name}")

    ok, msg = contiguous_ok(style, prefix, files)
    if not ok:
        print(f"\n[WARNING] Chain continuity problem: {msg}", file=sys.stderr)
        sys.exit(3)

    print("\n[OK] Chain appears contiguous on disk.")

    if args.mne:
        try:
            import mne
            print("\n[MNE] Attempting to read with MNE (no preload)...")
            raw = mne.io.read_raw_fif(str(in_path), preload=False, verbose="ERROR")
            print("[MNE] raw.filenames (resolved by FIFF pointers):")
            for f in raw.filenames:
                print(f"  - {Path(f).name}")
            print("[MNE] Success.")
        except Exception as e:
            print(f"\n[ERROR] MNE read failed: {e}", file=sys.stderr)
            # Often the error includes the missing filename — useful for creating symlinks.
            sys.exit(4)


if __name__ == "__main__":
    main()
