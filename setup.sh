#!/bin/bash

# VENV para picam
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install packages
pip install numpy
pip install pandas
pip install openpyxl
pip install matplotlib
pip install opencv-python
pip install opencv-contrib-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'git+https://github.com/facebookresearch/sam2.git'
pip install pyside6
pip install scikit-image

# Download checkpoints
mkdir -p ../checkpoints/
wget -nc -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Open GUI
python3 gui_pyside.py