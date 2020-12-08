# README

This repository contains the code to reproduce the results presented in "Three-dimensional imaging through scattering media based on confocal diffuse tomography" by David B. Lindell and Gordon Wetzstein.

[Project Webpage](http://www.computationalimaging.org/publications/confocal-diffuse-tomography/)

Code prerequisites: Linux-based operating system, installation of Anaconda or Miniconda. Developed and tested with Python 3.6 on Arch Linux kernel 5.6.8-arch1-1 with an Intel Core i7-9750H CPU.

The code is developed in Pytorch and supports both CPU and GPU execution. Most NVIDIA GPUs should be automatically detected and used.

# List of contents 
* `main.py` - wrapper program called to generate results
* `cdt_reconstruction.py` - contains processing code to run the reconstructions and reproduce results of main
paper.
* `data/cones.mat` - data file for Fig. 3
* `data/letter_s.mat` - data file for Fig. 2
* `data/letters_ut.mat` - data file for Fig. 3
* `data/letter_t.mat` - data file for Fig. 3
* `data/letter_u*.mat` - data files shown in Supplementary Movie 2 for an object at varying distances behind the media
* `data/mannequin.mat` - data file for Fig. 3
* `data/resolution*.mat` - data files for Supplementary Fig. 17
* `README.txt` - this file
* `requirements.txt` - list of requisite Python packages
* `setup.bash` - example script to install a conda environment and required packages
and run the code 
* `utils.py` - contains helper functions for the processing 

# Instructions
To run the demo code, follow the instructions in the setup.bash script. This script assumes that you have Anaconda or Miniconda installed. If not, follow the provided link for instructions on how to do this. The script sets up a new Python 3.6 environment, installs the required Python packages (listed in `requirements.txt`), and runs the reconstruction code.

The install time of the setup.bash script is less than 5 minutes on the tested configuration.  Runtime of the demo program `main.py` is less than one minute on the tested configuration and outputs. 

The expected output of the `main.py` program is a figure showing maximum intensity projections of the selected 3D measurement volume and reconstruction.

Execute `python main.py --help` for a list of commandline options for running the demo.

[Anaconda installation instructions](https://docs.anaconda.com/anaconda/install/)

# Calibration

Code to estimate the parameters of the scattering medium is included in the `calibration` folder. The data capture and optimization procedure are described in the Methods and Supplementary Information of the paper. We jointly optimize the parameters over captured time-resolved transmittance profiles for varying thicknesses of the scattering material (from 1 to 8 inches). Since the optimization is non-linear, we vary the initialized values of the reduced scattering coefficient and refractive index and select the values that best fit the data. Analyzing the distribution of optimized values is also useful for understanding the variance of the resulting estimate. 

* `sweep_n.py` - script to run the optimizations over all initializations
* `optimize.py` - main optimization script
* `diffusion_model.py` - wrapper class that contains the diffusion model and loads and processes captured measurements

Please direct questions to lindell@stanford.edu and
gordon.wetzstein@stanford.edu.
