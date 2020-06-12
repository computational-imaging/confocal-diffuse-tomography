#!/bin/bash

# Script to set up python environment to run the confocal diffuse tomography code
# note that this requires a system installaton of anaconda or miniconda
# if you don't have this, follow the instructions here for miniconda
# https://docs.conda.io/en/latest/miniconda.html

# create a new anaconda environment for this project
conda create -n cdt python=3.6
eval "$(conda shell.bash hook)"  # this is just required to activate an anaconda env in a bash script
conda activate cdt
pip install -r requirements.txt

# run the reconstructions on captured data and display each for 5 sec.
python cdt.py --scene letter_s  --pause 2  # reconstruct scene from Fig. 2
python cdt.py --scene mannequin --pause 2  # reconstruct scene from Fig. 3
python cdt.py --scene letters_ut --pause 2  # reconstruct scene from Fig. 3
python cdt.py --scene letter_t --pause 2  # reconstruct scene from Fig. 3
python cdt.py --scene cones --pause 2  # reconstruct scene from Fig. 3

# optionally remove environment when finished with code
# conda deactivate
# conda env remove -n cdt
