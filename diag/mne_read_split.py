#!/usr/bin/env python

import mne
from pathlib import Path

# Path to the first split file (the rest will be detected automatically)
fpath = Path("/Users/gm33/Desktop/sub-test_task-test_split-01_meg.fif")

# --- Read the raw file ---
print(f"Reading: {fpath}")
raw = mne.io.read_raw_fif(fpath, preload=False, verbose=True)

# --- Print some information ---
print("\n=== Summary ===")
print(f"Files combined: {raw.filenames}")
print(f"Duration: {raw.times[-1]:.1f} sec")
print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")

# --- Optionally check that the data loads correctly ---
raw.load_data()  # actually load into memory
print("\nData loaded successfully.")
print(f"Data shape: {raw.get_data().shape}")
