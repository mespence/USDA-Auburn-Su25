# USDA-Auburn

This repository contains our machine learning, GUI, data cleaning, and experimentation code used during this project. The code is generally organized in the same way, with a folder for each part of the project. For setup, you are likely going to want to clone this repository and install the python packages in ```GUI/requirements.py```.

## GUI

This folder contains all the code needed to run SCIDO, which can be done by running ```python main.py```. Code and weights for the models implemented in SCIDO is also contained in this folder.

## ML

This folder contains code implementing each of our models along with scripts for training and evaluating them.

## Data Cleaning

This folder contains scripts for taking in the dataset from Cooper et al. 2024's waveform library and turning them into densely-labeled CSVs.

## Exploration

This folder contains various notebooks and scripts used as we were testing out code, making figures, and prototyping. It is largely undocumented but is included for completeness.
