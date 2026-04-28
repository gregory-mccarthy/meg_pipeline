#!/bin/bash

# ------------------------------------------------------------
# Sync selected pipeline scripts from meg_project -> meg_pipeline
# ------------------------------------------------------------

SRC="/Users/gm33/Dropbox/meg_project"
DST="/Users/gm33/meg_pipeline"

FILES=(
bids_io_utils.py
compute_headpos.py
epoch_average_meg.py
headpos_utils.py
meg_pipeline_utils.py
preprocess_meg.py
print_fif_metadata.py
raw_plot_utility.py
transfer_manager.py
visualize_ave.py
)

echo "Syncing files from $SRC to $DST"
echo "--------------------------------"

for file in "${FILES[@]}"; do
    rsync -av --update "$SRC/$file" "$DST/"
done

echo "--------------------------------"
echo "Sync complete."