#!/bin/bash
conda env create -f conda-OOD-environment.yml
conda activate mne
conda install -c conda-forge xcb-util-cursor  # Your critical missing piece
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

