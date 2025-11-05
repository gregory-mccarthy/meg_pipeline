import os
from mne_bids import BIDSPath
from bids_io_utils import read_raw_bids_robust, write_bids_robust

# --- Edit these lines for your test data ---
bids_root = "/Users/gm33/data/epi"
subject = "010MPA"
session = "01"
run = "01"
task = "rest"
datatype = "meg"

# --- Set up BIDSPath for reading ---
bids_path = BIDSPath(
    subject=subject,
    session=session,
    run=run,
    task=task,
    datatype=datatype,
    root=bids_root,
)

print("Reading raw split files (do not call .fpath)...")
raw = read_raw_bids_robust(bids_path, preload=True)
print(f"Read raw object: {raw}")

# --- Set up BIDSPath for derivatives output ---
bids_path_deriv = bids_path.copy()
bids_path_deriv.update(
    root=os.path.join(bids_root, "derivatives", "preprocessing"),
    suffix="meg",
    description="preproc",
    extension=".fif"
)
out_fif = bids_path_deriv.fpath  # Only use for output, not input split discovery

print(f"Channels: {len(raw.ch_names)}")
print(f"Samples: {raw.n_times}")
print(f"Duration (sec): {raw.times[-1]:.2f}")
print(f"Sampling rate: {raw.info['sfreq']}")

print(f"Saving to derivatives path: {out_fif}")
written_files = write_bids_robust(raw, out_fif, overwrite=True, verbose=True)

print("\nOutput files written:")
for f in written_files:
    print(f"{f}    ({os.path.getsize(f)/1e6:.2f} MB)")

print("\n[Done!]")
