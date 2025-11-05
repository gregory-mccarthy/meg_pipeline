import numpy as np
import matplotlib.pyplot as plt
import mne

FIF_PATH = "/Users/gm33/data/sep/derivatives/preprocessing/sub-005/ses-1/meg/sub-005_ses-1_task-mns_allruns_desc-preproc_meg-ave.fif"

def sensor_grid_demo(sensor_names, coords, n_rows=8, n_cols=16, label_style='short'):
    """
    Display sensors in a 2D grid: columns = A–P, rows = L–R.
    No axis tick labels so grid is clean.
    """
    def short_label(name):
        return name[-3:] if len(name) >= 3 else name

    def label_for(i, name):
        if label_style == 'short':
            return short_label(name)
        elif label_style == 'index':
            return str(i+1)
        else:
            return name

    x = coords[:, 0]
    y = coords[:, 1]
    x_bins = np.linspace(x.min(), x.max(), n_cols+1)
    y_bins = np.linspace(y.min(), y.max(), n_rows+1)
    grid = np.full((n_rows, n_cols), '', dtype=object)

    for i, (xi, yi) in enumerate(zip(x, y)):
        col = np.digitize(xi, x_bins) - 1  # L-R
        row = np.digitize(yi, y_bins) - 1  # A-P
        row = np.clip(row, 0, n_rows-1)
        col = np.clip(col, 0, n_cols-1)
        lbl = label_for(i, sensor_names[i])
        if grid[row, col] == '':
            grid[row, col] = lbl
        else:
            grid[row, col] += f",{lbl}"

    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    ax.set_xlim(-0.5, n_cols-0.5)
    ax.set_ylim(-0.5, n_rows-0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # No tick labels for a clean grid

    for row in range(n_rows):
        for col in range(n_cols):
            txt = grid[row, col]
            if txt != '':
                ax.text(col, row, txt, ha='center', va='center', fontsize=10, color='red')

    ax.set_title("Sensor Matrix (A–P columns × L–R rows)")
    ax.invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evokeds = mne.read_evokeds(FIF_PATH)
    evoked = evokeds[0]
    ch_type = 'mag'  # Change as needed ('mag', 'grad', 'eeg')
    picks = mne.pick_types(evoked.info, meg=ch_type)
    coords = np.array([evoked.info['chs'][i]['loc'][:2] for i in picks])
    sensor_names = [evoked.info['ch_names'][i] for i in picks]

    # 8 rows × 16 columns, short sensor labels
    sensor_grid_demo(sensor_names, coords, n_rows=8, n_cols=16, label_style='short')