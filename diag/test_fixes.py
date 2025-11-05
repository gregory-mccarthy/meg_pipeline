import mne
import yaml
import os
from datetime import datetime, timezone
import pprint

# --- Path setup ---
infile = '/Users/gm33/data/fairy/sub-008MPA/ses-01/meg/sub-008MPA_ses-01_task-fairy_run-01_meg.fif'
outfile = os.path.expanduser('~/Desktop/temp_fix.fif')
yaml_file = 'test_fixes.yml'  # Path to your YAML fixes

def print_metadata(raw, label="", max_channels=16):
    print(f"\n===== {label} METADATA =====")
    print(f"File: {getattr(raw, 'filenames', [None])[0]}")
    ch_names = raw.info['ch_names']
    ch_types = [raw.get_channel_types(picks=[name])[0] for name in ch_names]
    n_show = min(max_channels, len(ch_names))

    print("\nChannels (first {} of {}):".format(n_show, len(ch_names)))
    print("  {:<10} {:<8}".format("Name", "Type"))
    print("  " + "-" * 19)
    for i in range(n_show):
        print("  {:<10} {:<8}".format(ch_names[i], ch_types[i]))
    if len(ch_names) > n_show:
        print(f"  ... ({len(ch_names) - n_show} more channels)")

    print("\nBad channels:  {}".format(", ".join(raw.info['bads']) or "None"))
    subj_info = raw.info.get('subject_info', None)
    print("\nSubject info:")
    pprint.pprint(subj_info, compact=True, indent=4)

    meas_date = raw.info.get('meas_date', None)
    print("\nMeasurement date: {}".format(meas_date if meas_date else "None"))
    device_info = raw.info.get('device_info', None)
    print("\nDevice info:")
    pprint.pprint(device_info, compact=True, indent=4)

    montage = raw.get_montage()
    print("\nMontage: {}".format(
        list(montage.get_positions()['ch_pos'].keys()) if montage else "None"
    ))
    print("=" * 38 + "\n")

def repair_metadata(raw, fixes):
    mf = fixes.get('metadata_fixes', {})

    # Fix channels: rename and set type
    if 'fix_channels' in mf:
        for old, changes in mf['fix_channels'].items():
            if old not in raw.info['ch_names']:
                print(f"WARNING: Channel {old} not found in raw data, skipping.")
                continue
            # Rename
            if 'name' in changes:
                raw.rename_channels({old: changes['name']})
            # Set type (using new name!)
            if 'type' in changes and 'name' in changes:
                raw.set_channel_types({changes['name']: changes['type']})
            elif 'type' in changes:
                raw.set_channel_types({old: changes['type']})

    # Mark bads
    if 'set_bads' in mf:
        raw.info['bads'] = mf['set_bads']

    # Subject info
    if 'set_subject_info' in mf:
        raw.info['subject_info'] = mf['set_subject_info']

    # Measurement date
    if 'set_meas_date' in mf:
        date_str = mf['set_meas_date']
        if date_str:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            raw.set_meas_date(dt)

    # Montage (for EEG)
    if 'set_montage' in mf:
        montage = mne.channels.make_standard_montage(mf['set_montage'])
        raw.set_montage(montage)

    # Remove device info
    if mf.get('remove_device_info', False):
        raw.info['device_info'] = None

    return raw

# --- MAIN WORKFLOW ---

# Load YAML
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

# Load raw data
raw = mne.io.read_raw_fif(infile, preload=True)
print_metadata(raw, label="Before Fixes")

# Apply metadata repairs
raw = repair_metadata(raw, config)
print_metadata(raw, label="After Fixes")

# Save fixed file
raw.save(outfile, overwrite=True)
print(f"Saved fixed file to: {outfile}")

# Reload and show metadata to verify
raw2 = mne.io.read_raw_fif(outfile, preload=False)
print_metadata(raw2, label="Reloaded Fixed File")