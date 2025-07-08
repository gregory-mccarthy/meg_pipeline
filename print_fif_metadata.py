import sys
import mne
from mne.io import read_raw_fif
from collections import Counter
import datetime
import numpy as np

def fmt_date(x):
    """Format a datetime or numpy datetime64 for readability."""
    if x is None:
        return 'n/a'
    try:
        if hasattr(x, 'strftime'):
            return x.strftime('%Y-%m-%d %H:%M:%S')
        elif hasattr(x, 'astype'):
            return str(x.astype('M8[s]'))
        else:
            return str(x)
    except Exception:
        return str(x)

def shortval(val):
    """Summarize value without printing giant objects."""
    if isinstance(val, str):
        return val if len(val) < 40 else val[:37] + "..."
    if isinstance(val, (int, float, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return f"[{len(val)} items]"
    if isinstance(val, dict):
        return f"{{{len(val)} keys}}"
    if isinstance(val, np.ndarray):
        return f"array shape {val.shape} dtype {val.dtype}"
    return f"{type(val).__name__}"

def print_raw_info(raw):
    print("="*60)
    print(f"File type: Raw")
    print(f"File name: {raw.filenames[0] if raw.filenames else 'n/a'}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {raw.info['nchan']}")
    ct_counts = Counter(raw.get_channel_types())
    print(f"Channel types: {dict(ct_counts)}")
    print("First 10 channel names:", ', '.join(raw.ch_names[:10]))
    print(f"Duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")
    print(f"Time range: {raw.times[0]:.3f} ... {raw.times[-1]:.3f} s")
    print()

    # Dates and timing
    meas_date = raw.info.get('meas_date', None)
    print(f"Measurement date: {fmt_date(meas_date)}")

    file_id = raw.info.get('file_id', None)
    if file_id and 'secs' in file_id:
        dt = datetime.datetime.utcfromtimestamp(file_id['secs'])
        print(f"Creation date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Creation date: n/a")

    print(f"Description: {raw.info.get('description', 'n/a')}")
    print(f"Experimenter/Operator: {raw.info.get('experimenter', 'n/a')}")
    print(f"Project name: {raw.info.get('proj_name', 'n/a')}")
    print(f"Lab/Institution: {raw.info.get('institution', 'n/a')}")
    print(f"Acquisition system: {shortval(raw.info.get('acq_pars', 'n/a'))}")
    print(f"Device info: {shortval(raw.info.get('device_info', 'n/a'))}")
    print(f"Line freq (Hz): {raw.info.get('line_freq', 'n/a')}")
    print()

    # Subject info (detailed if present)
    subj = raw.info.get('subject_info', {})
    if subj:
        print("Subject info:")
        for k, v in subj.items():
            print(f"  {k}: {v}")
    else:
        print("Subject info: n/a")
    print()

    # Bad channels (by name)
    bads = raw.info.get('bads', [])
    print(f"Bad channels: {bads if bads else 'None'}")

    # Summarize extra fields without giant dumps or ambiguous truth value
    extra_keys = [
        'proj_id', 'highpass', 'lowpass', 'dig', 'xplotter_layout',
        'proc_history', 'hpi_results', 'hpi_meas', 'helium_info', 'gantry_angle',
        'experimenter', 'meas_id', 'file_id'
    ]
    print("\nExtra info fields (summary):")
    for key in extra_keys:
        value = raw.info.get(key, None)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                continue
        elif value is None or value == '' or value == [] or value == {}:
            continue
        print(f"  {key}: {shortval(value)}")
    print()

    # Annotations and events summary
    if hasattr(raw, 'annotations') and raw.annotations is not None and len(raw.annotations) > 0:
        print(f"Annotations: {len(raw.annotations)} event(s)")
        print("  First 5:", [(a['onset'], a['description']) for a in raw.annotations[:5]])
    else:
        print("Annotations: None")
    try:
        events = mne.find_events(raw, shortest_event=1, verbose=False)
        print(f"Events detected in stim channels: {events.shape[0]}")
        if events.shape[0] > 0:
            # Tabular event count summary
            event_codes = events[:, 2]
            # Exclude zero (background)
            event_counter = Counter(e for e in event_codes if e != 0)
            if event_counter:
                print("\nEvent code summary (excluding zero):")
                print("  Code      Count")
                print("  -----    ------")
                for code, count in sorted(event_counter.items()):
                    print(f"  {code:<8} {count}")
            else:
                print("  No nonzero event codes found.")
            print("\n  First 5 events:", events[:5])
        else:
            print("  No events found.")
    except Exception:
        print("Events: (not detected or no stim channel)")

    # Only print keys, not contents
    print("\nNon-empty Raw.info keys present:")
    info_keys = []
    for k in raw.info.keys():
        v = raw.info[k]
        if isinstance(v, np.ndarray):
            if v.size == 0:
                continue
        elif v is None or v == '' or v == [] or v == {}:
            continue
        info_keys.append(k)
    print(sorted(info_keys))
    print("="*60)

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_fif_metadata.py path_to_file.fif")
        sys.exit(1)
    fif_file = sys.argv[1]
    try:
        raw = read_raw_fif(fif_file, allow_maxshield=True, preload=False, verbose='ERROR')
        print_raw_info(raw)
        return
    except Exception as e:
        print(f"read_raw_fif failed: {e}")

    print("Could not determine FIF file type or failed to read file.")

if __name__ == "__main__":
    main()