# =====================================================================
# Epoch_Average_MEG_BIDS_Phase4.py
# Multi-input, multi-run/session, variable epoch window, generalized composites,
# per-condition baseline, and BIDS-compliant output.
# =====================================================================

import yaml
import json
import sys
import mne
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import warnings
from mne_bids import BIDSPath
import datetime
import re

# =====================================================================
# Helper Functions
# =====================================================================

def to_native(obj):
    """Convert numpy/scalar types to Python built-ins for YAML logging."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class YAMLLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.data = {'steps': []}
        self.fatal = False

    def log(self, message, add_timestamp=False, fatal=False, **kwargs):
        entry = {'message': message}
        if add_timestamp:
            entry['timestamp'] = datetime.datetime.now().isoformat()
        if fatal:
            entry['FATAL'] = True
            self.fatal = True
        entry.update({k: to_native(v) for k, v in kwargs.items()})
        self.data['steps'].append(entry)

    def finalize(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.data, f, indent=2)

# =====================================================================
# Channel Harmonization (intersection, drop all bads)
# =====================================================================

def harmonize_channels(raw_list, logger):
    all_ch_names = [set(raw.info['ch_names']) for raw in raw_list]
    shared_chs = set.intersection(*all_ch_names)
    all_bad_chs = set()
    bads_per_run = defaultdict(list)
    for i, raw in enumerate(raw_list):
        for bad in raw.info.get('bads', []):
            all_bad_chs.add(bad)
            bads_per_run[bad].append(i+1)
    final_chs = sorted(shared_chs - all_bad_chs)
    dropped_missing = [sorted(names - shared_chs) for names in all_ch_names]
    logger.log("Channels present in all runs", channels=sorted(shared_chs))
    logger.log("Bad channels (excluded from all runs)", bad_channels=sorted(all_bad_chs), bads_per_run=to_native(bads_per_run))
    for idx, dropped in enumerate(dropped_missing):
        if dropped:
            logger.log(f"Channels missing in run {idx+1}", channels=dropped)
    logger.log("Channels included in concatenated analysis", final_channels=final_chs)
    raw_list_subset = []
    for i, raw in enumerate(raw_list):
        raw_sel = raw.copy()
        raw_sel.pick(final_chs)
        raw_sel.info['bads'] = []
        raw_list_subset.append(raw_sel)
    return raw_list_subset, final_chs

# =====================================================================
# Artifact Suppression (interpolation)
# =====================================================================

def suppress_artifact(raw, events, config, logger, max_log_events=1):
    suppress = config.get("artifact_suppression", None)
    if suppress is None:
        return raw
    tmin, tmax = suppress["window"]
    channel_types = suppress.get("channel_types", ["mag", "grad"])
    picks = mne.pick_types(
        raw.info,
        meg='mag' in channel_types or 'grad' in channel_types,
        eeg='eeg' in channel_types,
        ref_meg=False
    )
    sfreq = raw.info["sfreq"]
    n_start = int(round(tmin * sfreq))
    n_end = int(round(tmax * sfreq))
    n_events = 0
    data = raw._data
    n_times = data.shape[1]
    for i, (onset, _, _) in enumerate(events):
        start = onset + n_start
        end = onset + n_end
        start = max(start, 1)
        end = min(end, n_times-1)
        n_pts = end - start
        if n_pts <= 0:
            continue
        before = data[picks, start - 1][:, np.newaxis]
        after = data[picks, end][:, np.newaxis]
        interp = before + (after - before) * np.linspace(0, 1, n_pts)
        data[picks, start:end] = interp
        n_events += 1
        if i < max_log_events:
            logger.log(
                "Artifact interpolated for event",
                event_onset_sample=int(onset),
                sample_range=[int(start), int(end)],
                picks=int(len(picks))
            )
    logger.log(
        "Interpolated artifact window",
        window=[tmin, tmax],
        picks=int(len(picks)),
        n_events=int(n_events)
    )
    return raw

# =====================================================================
# Per-condition epoch extraction with per-condition baseline
# =====================================================================

def extract_epochs_variable(raw, config, logger):
    """
    Extract epochs for each basic condition, using per-condition epoch windows
    and baselines if provided.
    Returns: dict of condition_name -> epochs object.
    """
    events, annot_event_id = mne.events_from_annotations(raw)
    logger.log("Extracted events from annotations", num_events=len(events))
    raw = suppress_artifact(raw, events, config, logger)
    if 'filter' in config:
        filt = config['filter']
        l_freq = filt.get('l_freq', None)
        h_freq = filt.get('h_freq', None)
        logger.log("Applying bandpass filter", l_freq=l_freq, h_freq=h_freq)
        raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    # Prepare event id mapping
    simple_conditions = {}
    for name, code in config['conditions'].items():
        if isinstance(code, int):
            code = str(code)
        if code not in annot_event_id:
            raise ValueError(f"Annotation code '{code}' not found in the dataset.")
        simple_conditions[name] = annot_event_id[code]
    reject = config['reject'] if config.get('use_reject', False) else None
    if reject is not None:
        present_types = Counter(raw.get_channel_types())
        for key in list(reject):
            if key not in present_types:
                logger.log(f"Reject rule for '{key}' skipped (channel type not present)")
                del reject[key]
        for key, val in reject.items():
            if isinstance(val, str):
                try:
                    reject[key] = float(val)
                except ValueError:
                    raise TypeError(f"Reject threshold for '{key}' must be a number, got '{val}' (str)")
            elif not isinstance(val, (int, float)):
                raise TypeError(f"Reject threshold for '{key}' must be numeric, got {val} ({type(val)})")
    epoch_windows = config.get('epoch_windows', {})
    default_tmin = config['tmin']
    default_tmax = config['tmax']
    default_baseline = tuple(config['baseline'])
    epochs_dict = {}
    drop_logs = {}
    for cond, event_id in simple_conditions.items():
        ew = epoch_windows.get(cond, {})
        tmin = ew.get('tmin', default_tmin)
        tmax = ew.get('tmax', default_tmax)
        baseline = tuple(ew.get('baseline', default_baseline))
        ep = mne.Epochs(
            raw, events, event_id={cond: event_id},
            tmin=tmin, tmax=tmax, baseline=baseline,
            reject=reject, reject_by_annotation=True, preload=True
        )
        drop_logs[cond] = to_native(ep.drop_log_stats())
        logger.log(
            "Created epochs",
            condition=cond,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            n_epochs=len(ep),
            drop_stats=drop_logs[cond]
        )
        epochs_dict[cond] = ep
    return epochs_dict

# =====================================================================
# Composite averaging: generalized expressions, user labels
# =====================================================================

def parse_composite_expr(expr, available):
    """
    Parse an expression like "A + 0.5*B - C" into a list of (name, weight).
    Supports arbitrary whitespace, coefficients (floats or ints), plus/minus.
    """
    # Remove all whitespace for simpler parsing
    expr = expr.replace(" ", "")
    # Acceptable basic term: [+/-][coefficient*]Condition
    tokens = re.findall(r'([+-]?)(\d*\.?\d*)\*?([A-Za-z0-9_]+)', expr)
    if not tokens:
        raise ValueError(f"Could not parse composite expression: {expr}")
    weights = defaultdict(float)
    for sign, coeff, name in tokens:
        sign_val = -1 if sign == '-' else 1
        if coeff == '':
            coeff_val = 1.0
        else:
            coeff_val = float(coeff)
        if name not in available:
            raise ValueError(f"Condition '{name}' in composite not found among: {list(available.keys())}")
        weights[name] += sign_val * coeff_val
    return [(k, v) for k, v in weights.items() if v != 0]

def compute_and_save_evokeds_general(epochs_dict, composites_yaml, logger, evoked_fname):
    """
    Average all basic conditions, and compute composites from YAML, handling any expression.
    All composite outputs get user-defined labels.
    All composites are cropped to the common overlapping time window before combining.
    """
    evokeds = {}
    # First, average basics
    for cond, epochs in epochs_dict.items():
        evokeds[cond] = epochs[cond].average()
        logger.log("Averaged condition", condition=cond, n_epochs=len(epochs[cond]))
    # Now, composites
    for label, cinfo in composites_yaml.items():
        expr = cinfo['expr']
        try:
            terms = parse_composite_expr(expr, evokeds)
        except Exception as e:
            logger.log(f"FATAL: Error parsing composite {label}: {e}", fatal=True, expr=expr)
            continue
        # Determine cropping (overlap)
        tmins = [evokeds[name].times[0] for name, _ in terms]
        tmaxs = [evokeds[name].times[-1] for name, _ in terms]
        crop_min = max(tmins)
        crop_max = min(tmaxs)
        if crop_max <= crop_min:
            logger.log(
                f"FATAL: No overlapping window for composite {label}",
                fatal=True,
                components=[name for name, _ in terms],
                tmin_list=tmins,
                tmax_list=tmaxs,
                expr=expr
            )
            continue
        # Crop, store, and use combine_evoked for weighted sum
        evks_cropped = []
        weights = []
        for name, weight in terms:
            evk = evokeds[name].copy().crop(crop_min, crop_max)
            evks_cropped.append(evk)
            weights.append(weight)
        if evks_cropped:
            pooled = mne.combine_evoked(evks_cropped, weights=weights)
        else:
            pooled = None
        if pooled is not None:
            evokeds[label] = pooled
            logger.log(
                "Computed composite condition",
                label=label,
                expr=expr,
                components=terms,
                cropped_tmin=crop_min,
                cropped_tmax=crop_max
            )
    mne.write_evokeds(str(evoked_fname), list(evokeds.values()), overwrite=True)
    logger.log("Saved evoked averages file", path=str(evoked_fname))
    return evokeds

# =====================================================================
# Output path and naming logic (unchanged)
# =====================================================================

def make_output_base(inputs, outdir, task):
    subject = inputs[0]['subject']
    all_sessions = {inp.get('session') for inp in inputs if 'session' in inp}
    all_runs = {inp.get('run') for inp in inputs if 'run' in inp}
    fname = f"sub-{subject}"
    if len(all_sessions) == 1 and None not in all_sessions:
        fname += f"_ses-{list(all_sessions)[0]}"
    fname += f"_task-{task}"
    if len(all_runs) == 1 and None not in all_runs:
        fname += f"_run-{list(all_runs)[0]}"
    if len(all_runs) > 1:
        fname += "_allruns"
    if len(all_sessions) > 1:
        fname += "_allsessions"
    base = outdir / f"{fname}_desc-preproc_meg"
    return base

# =====================================================================
# Main workflow (Phase 4)
# =====================================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python Epoch_Average_MEG_BIDS_Phase4.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]
    config = load_config(config_path)
    bids_root = Path(config['bids_root'])
    inputs = config['inputs']
    if not isinstance(inputs, list) or len(inputs) == 0:
        print("YAML 'inputs' list is missing or empty.")
        sys.exit(1)
    deriv_root = bids_root / "derivatives" / "preprocessing"
    subject = inputs[0]['subject']
    all_sessions = {inp.get('session') for inp in inputs if 'session' in inp}
    session = list(all_sessions)[0] if len(all_sessions) == 1 and None not in all_sessions else None
    task = inputs[0].get('task', None)
    if session:
        outdir = deriv_root / f"sub-{subject}" / f"ses-{session}" / "meg"
    else:
        outdir = deriv_root / f"sub-{subject}" / "meg"
    outdir.mkdir(parents=True, exist_ok=True)
    base = make_output_base(inputs, outdir, task)
    epochs_fname = base.with_name(base.name + "-epo.fif")
    evoked_fname = base.with_name(base.name + "-ave.fif")
    log_fname = base.with_name(base.name + "-log.json")
    logger = YAMLLogger(log_fname)
    logger.log("Starting processing", subject=subject, task=task, add_timestamp=True)
    # --------------------------
    # Load input files
    # --------------------------
    raw_list = []
    load_errors = []
    for inp in inputs:
        preproc_bids_path = BIDSPath(
            subject=inp['subject'],
            session=inp.get('session', None),
            task=inp.get('task', None),
            run=inp.get('run', None),
            root=deriv_root,
            datatype='meg',
            suffix='meg',
            description='preproc',
            extension='.fif'
        )
        try:
            raw = mne.io.read_raw_fif(preproc_bids_path.fpath, preload=True)
            raw_list.append(raw)
            logger.log("Loaded input file", path=str(preproc_bids_path.fpath))
        except Exception as e:
            err_msg = f"Could not load {preproc_bids_path.fpath}: {e}"
            logger.log(err_msg, fatal=True, input=inp)
            load_errors.append(err_msg)
    if load_errors:
        logger.log("FATAL: Some input files could not be loaded. Exiting.", fatal=True, errors=load_errors)
        logger.finalize()
        print(f"Errors loading files. See log: {log_fname}")
        sys.exit(1)
    # --------------------------
    # Channel harmonization/validation
    # --------------------------
    if len(raw_list) > 1:
        # Sampling rates, subject, and task check
        ref = raw_list[0]
        issues = []
        subjects = {i['subject'] for i in inputs}
        if len(subjects) != 1:
            issues.append(f"Subjects do not match: {subjects}")
        tasks = {i.get('task') for i in inputs}
        if len(tasks) != 1:
            issues.append(f"Tasks do not match: {tasks}")
        sfreqs = {round(r.info['sfreq'], 5) for r in raw_list}
        if len(sfreqs) != 1:
            issues.append(f"Sampling rates differ: {sfreqs}")
        if issues:
            logger.log("FATAL: Input compatibility check failed", fatal=True, issues=issues)
            logger.finalize()
            print(f"Incompatibility detected across inputs. See log: {log_fname}")
            sys.exit(1)
        raw_list, final_chs = harmonize_channels(raw_list, logger)
        raw = mne.concatenate_raws(raw_list)
        logger.log("Concatenated all input files into single Raw object.", n_files=len(raw_list))
    else:
        raw_list, final_chs = harmonize_channels(raw_list, logger)
        raw = raw_list[0]
    # --------------------------
    # Per-condition epoch extraction (with per-condition baseline)
    # --------------------------
    epochs_dict = extract_epochs_variable(raw, config, logger)

    # Check for per-condition baseline differences and log a warning if present
    baselines = [ep.baseline for ep in epochs_dict.values()]
    if len(set(baselines)) > 1:
        logger.log("WARNING: Baseline intervals differ across conditions.", baselines=baselines)

    # ===============================
    # Save epochs: merge if tmin/tmax match, else separate files
    # ===============================
    if epochs_dict:
        # Gather (tmin, tmax) for all conditions
        epoch_windows = [(ep.tmin, ep.tmax) for ep in epochs_dict.values()]
        all_same_window = len(set(epoch_windows)) == 1
        if all_same_window:
            # Merge all conditions (mixed events in one file)
            all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))
            all_epochs.save(str(epochs_fname), overwrite=True)
            logger.log("Saved single mixed-condition epochs file", path=str(epochs_fname))
        else:
            # Save each separately
            for cond, ep in epochs_dict.items():
                ep_fname = base.with_name(base.name + f"_{cond}-epo.fif")
                ep.save(str(ep_fname), overwrite=True)
                logger.log(f"Saved epochs file for {cond}", path=str(ep_fname))
    # --------------------------
    # Compute and save averages (including generalized composites)
    # --------------------------
    composites_yaml = config.get('composites', {})
    compute_and_save_evokeds_general(epochs_dict, composites_yaml, logger, evoked_fname)
    logger.log("Processing complete.", add_timestamp=True)
    logger.finalize()
    print(f"Processing complete. Log written to: {log_fname}")

if __name__ == '__main__':
    main()