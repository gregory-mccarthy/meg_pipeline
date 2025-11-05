import mne
from autoreject import AutoReject, Ransac
import numpy as np

# ----- User options -----
epoch_duration = 1.5  # Epoch length in seconds (shorter for resting state)
n_jobs = 1  # Set -1 for all cores
meg_rejection_threshold = 0.5  # Fraction of channels that must be bad (more conservative)


def load_and_preprocess_data(file_path, crop_fraction=0.25):
    print("Loading raw data...")
    raw = mne.io.read_raw_fif(file_path, preload=True)
    duration_sec = raw.times[-1]
    raw.crop(tmin=0, tmax=duration_sec * crop_fraction)
    print("Applying filters...")
    raw.filter(l_freq=1.0, h_freq=55.0, fir_design='firwin', verbose=False)
    if raw.info['lowpass'] > 60:
        raw.notch_filter([60], verbose=False)
    return raw


def print_channel_diagnostics(raw):
    print("\n======= CHANNEL DIAGNOSTICS =======")
    print(f"Total channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']:.1f} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")

    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)
    mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False)
    grad_picks = mne.pick_types(raw.info, meg='grad', eeg=False)
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)

    print(f"\nMEG channels: {len(meg_picks)} (mag: {len(mag_picks)}, grad: {len(grad_picks)})")
    if len(meg_picks) > 0:
        print(f"MEG names (first 5): {[raw.ch_names[i] for i in meg_picks[:5]]}")
    print(f"EEG channels: {len(eeg_picks)}")
    if len(eeg_picks) > 0:
        print(f"EEG names: {[raw.ch_names[i] for i in eeg_picks]}")
    print(f"Bad channels (pre-processing): {raw.info['bads']}")
    print("=====================================\n")


def find_bad_channels_ransac(epochs_eeg):
    print(f"Running RANSAC to find bad EEG channels...")
    try:
        ransac = Ransac(verbose=True, n_jobs=n_jobs, random_state=42)
        ransac.fit(epochs_eeg)
        print(f"\nRANSAC Results:")
        print(f"Bad EEG channels found: {ransac.bad_chs_}")
        return ransac
    except Exception as e:
        print(f"ERROR: RANSAC failed with error: {e}")
        return None


def run_autoreject_analysis(epochs_eeg):
    if len(epochs_eeg) < 10 or len(epochs_eeg.ch_names) < 4:
        print("WARNING: Too few epochs or EEG channels for reliable autoreject.")
        return None, None
    print(f"Running autoreject on {len(epochs_eeg)} epochs with {len(epochs_eeg.ch_names)} EEG channels...")
    ar_eeg = AutoReject(
        n_interpolate=[0],
        consensus=[0.5],
        n_jobs=n_jobs,
        verbose=True,
        random_state=42
    )
    try:
        ar_eeg.fit(epochs_eeg)
        print("\nAutoreject EEG Results:")
        print(f"Rejection thresholds (µV): {ar_eeg.threshes_}")
        reject_log = ar_eeg.get_reject_log(epochs_eeg)
        return ar_eeg, reject_log
    except Exception as e:
        print(f"ERROR: Autoreject failed with error: {e}")
        return None, None


def run_meg_epoch_rejection(epochs):
    """
    Identify globally bad epochs in MEG data using autoreject.
    Returns annotations for bad epochs that can be added to raw data.
    """
    print("\n======= MEG EPOCH REJECTION =======")

    # Check if we have MEG data
    meg_picks = mne.pick_types(epochs.info, meg=True, eeg=False)
    if len(meg_picks) == 0:
        print("No MEG channels found - skipping MEG epoch rejection")
        return None

    # Create separate epochs for mag and grad
    epochs_mag = epochs.copy().pick_types(meg='mag', eeg=False)
    epochs_grad = epochs.copy().pick_types(meg='grad', eeg=False)

    bad_epoch_indices = set()

    # Process magnetometers
    if len(epochs_mag.ch_names) > 0:
        print(f"\nProcessing {len(epochs_mag.ch_names)} magnetometer channels...")
        ar_mag, reject_log_mag = run_autoreject_meg(epochs_mag, "magnetometer")
        if reject_log_mag is not None:
            # Find epochs that are globally bad (rejected in many channels)
            bad_epochs_mag = find_globally_bad_epochs(reject_log_mag, epochs_mag, threshold=meg_rejection_threshold)
            bad_epoch_indices.update(bad_epochs_mag)
            print(f"Magnetometer bad epochs: {len(bad_epochs_mag)}")

    # Process gradiometers
    if len(epochs_grad.ch_names) > 0:
        print(f"\nProcessing {len(epochs_grad.ch_names)} gradiometer channels...")
        ar_grad, reject_log_grad = run_autoreject_meg(epochs_grad, "gradiometer")
        if reject_log_grad is not None:
            # Find epochs that are globally bad
            bad_epochs_grad = find_globally_bad_epochs(reject_log_grad, epochs_grad, threshold=meg_rejection_threshold)
            bad_epoch_indices.update(bad_epochs_grad)
            print(f"Gradiometer bad epochs: {len(bad_epochs_grad)}")

    # Convert epoch indices to time annotations
    if bad_epoch_indices:
        bad_epoch_list = sorted(list(bad_epoch_indices))
        print(f"\nTotal unique bad epochs identified: {len(bad_epoch_list)}")
        print(f"Bad epoch indices: {bad_epoch_list[:10]}{'...' if len(bad_epoch_list) > 10 else ''}")

        # Create annotations for bad epochs
        annotations = create_bad_epoch_annotations(epochs, bad_epoch_list)
        print("Created annotations for bad MEG epochs")
        print("=====================================\n")
        return annotations, bad_epoch_list
    else:
        print("No globally bad MEG epochs identified")
        print("=====================================\n")
        return None, []


def run_autoreject_meg(epochs_meg, sensor_type):
    """Run autoreject on MEG data (mag or grad) with resting-state appropriate parameters"""
    if len(epochs_meg) < 10:
        print(f"WARNING: Too few epochs for reliable {sensor_type} autoreject.")
        return None, None

    print(f"Running autoreject on {len(epochs_meg)} epochs with {len(epochs_meg.ch_names)} {sensor_type} channels...")

    # More conservative parameters for resting state MEG
    # Focus on obvious artifacts rather than physiological variations
    ar_meg = AutoReject(
        n_interpolate=[1, 4],  # Reduced interpolation options
        consensus=[0.8, 0.9],  # Higher consensus (more conservative)
        thresh_method='bayesian_optimization',  # Better threshold estimation
        cv=3,  # Reduced cross-validation folds for speed
        n_jobs=n_jobs,
        verbose=True,
        random_state=42
    )

    try:
        ar_meg.fit(epochs_meg)
        unit = 'fT' if sensor_type == 'magnetometer' else 'fT/cm'
        print(f"\nAutoreject {sensor_type} Results:")
        print(f"Rejection thresholds ({unit}): {ar_meg.threshes_}")
        reject_log = ar_meg.get_reject_log(epochs_meg)
        return ar_meg, reject_log
    except Exception as e:
        print(f"ERROR: Autoreject {sensor_type} failed with error: {e}")
        return None, None


def find_globally_bad_epochs(reject_log, epochs, threshold=0.3):
    """
    Identify epochs that are bad in a high proportion of channels.
    threshold: fraction of channels that must be bad for epoch to be considered globally bad
    """
    if reject_log is None:
        return []

    n_channels = len(epochs.ch_names)
    bad_channel_counts = reject_log.labels.sum(axis=1)  # Count bad channels per epoch
    bad_channel_fraction = bad_channel_counts / n_channels

    # Find epochs where more than threshold fraction of channels are bad
    globally_bad = np.where(bad_channel_fraction > threshold)[0]

    print(f"Epochs with >{threshold * 100:.0f}% bad channels: {len(globally_bad)}")
    if len(globally_bad) > 0:
        print(f"Bad channel fractions for worst epochs: {bad_channel_fraction[globally_bad][:5]}")

    return globally_bad.tolist()


def create_bad_epoch_annotations(epochs, bad_epoch_indices):
    """Create MNE annotations for bad epochs"""
    onset_times = []
    durations = []
    descriptions = []

    for idx in bad_epoch_indices:
        if idx < len(epochs):
            onset_time = epochs.times[0] + idx * epoch_duration  # Time relative to start of recording
            onset_times.append(onset_time)
            durations.append(epoch_duration)
            descriptions.append('BAD_MEG_epoch')

    if onset_times:
        annotations = mne.Annotations(
            onset=onset_times,
            duration=durations,
            description=descriptions,
            orig_time=None
        )
        return annotations
    return None


def analyze_rejection_statistics(reject_log, epochs_eeg):
    if reject_log is None:
        return
    print(f"\n======= EEG REJECTION STATISTICS =======")
    print(f"Rejection log shape: {reject_log.labels.shape}")
    print(f"(epochs x channels): ({len(epochs_eeg)} x {len(epochs_eeg.ch_names)})")
    bad_counts = reject_log.labels.sum(axis=0)
    print(f"\nPer-channel rejection rates:")
    for ch_name, n_bad in zip(epochs_eeg.ch_names, bad_counts):
        if n_bad > 0:
            pct = 100.0 * n_bad / len(epochs_eeg)
            print(f"  {ch_name}: {n_bad}/{len(epochs_eeg)} epochs ({pct:.1f}%)")
    epochs_with_bads = (reject_log.labels.sum(axis=1) > 0).sum()
    print(f"\nPer-epoch statistics:")
    print(
        f"  Epochs with any bad channels: {epochs_with_bads}/{len(epochs_eeg)} ({100.0 * epochs_with_bads / len(epochs_eeg):.1f}%)")
    bad_ch_per_epoch = reject_log.labels.sum(axis=1)
    worst_epochs = np.argsort(bad_ch_per_epoch)[-5:][::-1]
    print(f"  Worst 5 epochs (# bad channels): {bad_ch_per_epoch[worst_epochs]}")
    print("==========================================\n")


def visualize_results(raw, epochs_eeg, ransac, bad_meg_epochs=None):
    if ransac is None:
        print("Skipping EEG visualization - no RANSAC results")
    else:
        print(f"Visualization: Bad EEG channels marked in red")
        print(f"RANSAC bad channels: {getattr(ransac, 'bad_chs_', [])}")
        print(f"Combined bads: {raw.info['bads']}")

    if bad_meg_epochs:
        print(f"Bad MEG epochs: {len(bad_meg_epochs)} epochs will be annotated")

    print("\nOpening raw data browser...")
    raw.plot(block=False, title="Raw data with bad channels marked and MEG epoch annotations")

    if ransac is not None and len(epochs_eeg.ch_names) > 0:
        print("Opening EEG epochs browser...")
        epochs_eeg.plot(block=False, title="EEG epochs (global bads grayed)")

    # Note: MEG epoch annotations will be visible in the raw data browser
    print("MEG epoch rejections are shown as annotations in the raw data browser")
    print("Look for 'BAD_MEG_epoch' annotations (typically shown in red)")
    input("Press Enter to continue...")


def main():
    file_path = "/Users/gm33/data/epi/sub-010MPA/ses-01/meg/sub-010MPA_ses-01_task-rest_run-01_meg.fif"
    try:
        raw = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    print_channel_diagnostics(raw)

    # ---- FIND GLOBAL BAD EEG CHANNELS BEFORE EPOCHING ----
    tmp_epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
    tmp_epochs_eeg = tmp_epochs.copy().pick_types(eeg=True, meg=False)
    ransac = find_bad_channels_ransac(tmp_epochs_eeg)
    if ransac is not None:
        raw.info['bads'] = list(set(raw.info['bads'] + ransac.bad_chs_))
    del tmp_epochs, tmp_epochs_eeg

    # Now create epochs (they'll inherit bads)
    print(f"Creating {epoch_duration}s epochs (with global bads marked)...")
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
    print(f"Created {len(epochs)} epochs")

    # Process EEG for channel-level rejection (your existing workflow)
    epochs_eeg = epochs.copy().pick_types(eeg=True, meg=False)
    if len(epochs_eeg.ch_names) > 0:
        print(f"EEG epochs shape: {epochs_eeg._data.shape}")
        epochs_eeg_clean = epochs_eeg
        ar_eeg, reject_log = run_autoreject_analysis(epochs_eeg_clean)
        if reject_log is not None:
            analyze_rejection_statistics(reject_log, epochs_eeg_clean)
    else:
        print("No EEG channels found in data!")
        epochs_eeg = None
        ransac = None

    # NEW: Process MEG for epoch-level rejection
    meg_annotations, bad_meg_epochs = run_meg_epoch_rejection(epochs)

    # Add MEG epoch annotations to raw data
    if meg_annotations is not None:
        if raw.annotations is None:
            raw.set_annotations(meg_annotations)
        else:
            raw.set_annotations(raw.annotations + meg_annotations)
        print(f"Added {len(bad_meg_epochs)} bad MEG epoch annotations to raw data")

    # Visualization
    visualize_results(raw, epochs_eeg if epochs_eeg is not None else mne.EvokedArray(np.array([[]]), mne.create_info([],
                                                                                                                     raw.info[
                                                                                                                         'sfreq'])),
                      ransac, bad_meg_epochs)

    print("Analysis complete. Cleaning up...")
    del epochs


if __name__ == "__main__":
    main()
