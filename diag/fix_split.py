import mne
import os

# ---- List ALL split input files in correct order ----
input_files = [
    "/Users/gm33/Desktop/temp/sub-010MPA_ses-01_task-rest_run-1_meg.fif",
    "/Users/gm33/Desktop/temp/sub-010MPA_ses-01_task-rest_run-1_split-1_meg.fif",
    "/Users/gm33/Desktop/temp/sub-010MPA_ses-01_task-rest_run-1_split-2_meg.fif",
    "/Users/gm33/Desktop/temp/sub-010MPA_ses-01_task-rest_run-1_split-3_meg.fif"
]

# ---- Output BIDS-compliant base filename (no manual renaming of splits!) ----
out_file = "/Users/gm33/Desktop/temp/sub-010MPA/ses-01/meg/sub-010MPA_ses-01_task-rest_run-01_meg.fif"

# Make output directory if it doesn't exist
os.makedirs(os.path.dirname(out_file), exist_ok=True)

# Read all input splits as a single Raw object
raw = mne.io.read_raw_fif(input_files, preload=True)

# Remove only annotations with 'boundary' (case-insensitive) in description
if raw.annotations is not None and len(raw.annotations) > 0:
    keep = [not ("boundary" in desc.lower()) for desc in raw.annotations.description]
    cleaned_annot = raw.annotations[keep]
    raw.set_annotations(cleaned_annot)

# Save as new base (MNE will split at 2GB as needed, do not set split_size > 2GB)
raw.save(out_file, overwrite=True)  # Let MNE do default splitting

print(f"Saved file to {out_file} (with proper split files if needed).")