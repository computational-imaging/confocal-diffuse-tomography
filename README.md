# README

This repository contains the code to reproduce the results presented in "Three-dimensional imaging through scattering media based on confocal diffuse
tomography" by David B. Lindell and Gordon Wetzstein.

Code prerequisites: Linux-based operating system, installation of Anaconda or
Miniconda. Developed and tested with Python 3.6 on Arch Linux kernel 5.6.8-arch1-1
with an Intel Core i7-9750H CPU.

The code is developed in Pytorch and supports both CPU and GPU execution. Most
NVIDIA GPUs should be automatically detected and used.

# List of contents 
* `cdt.py` - main program to run the reconstructions and reproduce results of main
paper.
* `data/cones.mat` - data file for Fig. 3
* `data/letter_s.mat` - data file for Fig. 2
* `data/letters_ut.mat` - data file for Fig. 3
* `data/letter_t.mat` - data file for Fig. 3
* `data/mannequin.mat` - data file for Fig. 3
* `README.txt` - this file
* `requirements.txt` - list of requisite Python packages
* setup.bash - example script to install a conda environment and required packages
and run the code 

# Instructions
To run the demo code, follow the instructions in the setup.bash
script. This script assumes that you have Anaconda or Miniconda installed. If
not, follow the provided link for instructions on how to do this. The script
sets up a new Python 3.6 environment, installs the required Python packages
(listed in `requirements.txt`), and runs the reconstruction code.

The install time of the setup.bash script is less than 5 minutes on the tested
configuration.  Runtime of the demo program `cdt.py` is less than one minute on
the tested configuration and outputs. 

The expected output of the `cdt.py` program is a figure showing maximum intensity
projections of the selected 3D measurement volume and reconstruction.

Execute `python cdt.py --help` for a list of commandline options for running 
the demo.

[Anaconda installation instructions](https://docs.anaconda.com/anaconda/install/)

Please direct questions to lindell@stanford.edu and
gordon.wetzstein@stanford.edu.
