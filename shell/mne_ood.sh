#!/usr/bin/env bash
# ------------------------------------------------------------------
# MNE OOD startup script: activates env + sets Qt6 paths
# ------------------------------------------------------------------

# --- initialize conda ---
# try common install locations; ignore if not found
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
else
    echo "⚠️ Could not find conda.sh — edit this script to point to your conda installation."
    exit 1
fi

# --- activate env ---
conda activate mne || { echo "❌ Failed to activate conda env 'mne'"; exit 1; }

# --- Qt / Matplotlib settings for OOD ---
export QT_QPA_PLATFORM=xcb
export MPLBACKEND=QtAgg
export QT_API=pyqt6

# --- locate PyQt6 plugin/lib paths (pip install) ---
PYQT6_BASE=$(python - <<'PY'
import PyQt6, os
print(os.path.dirname(PyQt6.__file__))
PY
)
export QT_PLUGIN_PATH="$PYQT6_BASE/Qt6/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT6_BASE/Qt6/plugins/platforms"
export LD_LIBRARY_PATH="$PYQT6_BASE/Qt6/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

echo "✅ MNE OOD environment ready."
echo "You can now run:  python visualize_ave.py"