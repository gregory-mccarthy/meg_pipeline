import mne
from mne.preprocessing import ICA
import matplotlib
#matplotlib.use('QtAgg')

raw = mne.io.read_raw_fif("/Users/gm33/data/epi/sub-010MPA/ses-01/meg/sub-010MPA_ses-01_task-stimulation_run-01_meg.fif", preload=True)
picks = mne.pick_types(raw.info, meg=True, eeg=False, exclude='bads')  # for MEG; change for EEG

# --- Add interactive bad channel review here ---
print("Opening interactive raw.plot() for bad channel review...")
raw.plot()  # Close this window before continuing
input("Review plot, then close the window and press Enter...")

ica = ICA(n_components=20, method='fastica', random_state=97)
ica.fit(raw, picks=picks)

print("Opening ICA component topo plot...")
ica.plot_components(inst=raw)         # Close this window before continuing

print("Opening ICA time course plot...")
ica.plot_sources(raw)                 # Close this window before continuing

input("Press Enter after closing all windows.")

print("Done plotting both.")
