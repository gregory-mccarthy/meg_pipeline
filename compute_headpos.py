#!/usr/bin/env python3
"""
Standalone Head Motion Measurement Utility

- Input: Path to MEG .fif file
- Output: _headpos.pos, _headpos_summary.json, _headpos_plot.png in the same directory

Usage:
    python measure_head_motion.py /path/to/datafile.fif
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure headless operation
import matplotlib.pyplot as plt
import mne

def compute_head_motion(fif_path):
    # Load raw MEG data
    print(f"Loading raw data: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose='error')

    # Compute head position (cHPI)
    print("Computing head position from continuous HPI ...")
    chpi_amps = mne.chpi.compute_chpi_amplitudes(raw)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amps)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)

    if head_pos.shape[0] == 0:
        raise RuntimeError("cHPI present but head_pos is empty. No head movement data computed.")

    # Save .pos file
    base = os.path.splitext(os.path.basename(fif_path))[0]
    out_dir = os.path.dirname(fif_path)
    pos_path = os.path.join(out_dir, f"{base}_headpos.pos")
    mne.chpi.write_head_pos(pos_path, head_pos)
    print(f"Wrote head position file: {pos_path}")

    # Calculate translation and rotation
    times = head_pos[:, 0]
    translations = head_pos[:, 1:4]  # X, Y, Z in meters
    rotations = head_pos[:, 4:7]     # Rotations (radians)

    # Displacement relative to first frame
    ref_pos = translations[0]
    disp = np.linalg.norm(translations - ref_pos, axis=1)
    rot_deg = np.degrees(rotations)  # For plot readability

    # Compute summary statistics
    summary = {
        "fif_file": fif_path,
        "pos_file": pos_path,
        "n_samples": len(times),
        "duration_sec": float(times[-1] - times[0]),
        "max_translation_mm": float(np.max(disp) * 1000),
        "mean_translation_mm": float(np.mean(disp) * 1000),
        "max_rotation_deg": float(np.max(np.abs(rot_deg))),
        "mean_rotation_deg": float(np.mean(np.abs(rot_deg))),
        "translation_axis_max_mm": [float(np.max(np.abs(translations[:,i] - ref_pos[i]))*1000) for i in range(3)],
        "translation_axis_labels": ["X", "Y", "Z"],
        "rotation_axis_labels": ["Yaw", "Pitch", "Roll"],
    }

    # Save summary JSON
    summary_path = os.path.join(out_dir, f"{base}_headpos_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary JSON: {summary_path}")

    # Plot and save
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times, (translations - ref_pos) * 1000)
    axes[0].set_ylabel('Displacement (mm)')
    axes[0].set_title('Head Translation (relative to start)')
    axes[0].legend(['X', 'Y', 'Z'])
    axes[0].grid(True)

    axes[1].plot(times, rot_deg)
    axes[1].set_ylabel('Rotation (deg)')
    axes[1].set_title('Head Rotation (Yaw, Pitch, Roll)')
    axes[1].legend(['Yaw', 'Pitch', 'Roll'])
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{base}_headpos_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote head motion plot: {plot_path}")

    return pos_path, summary_path, plot_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python measure_head_motion.py /path/to/datafile.fif")
        sys.exit(1)
    fif_path = sys.argv[1]
    if not os.path.isfile(fif_path):
        print(f"File not found: {fif_path}")
        sys.exit(1)
    try:
        compute_head_motion(fif_path)
        print("Done.")
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
