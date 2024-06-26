#!/usr/bin/bash

# Download the QM7 dataset
shred -u qm7.dat
wget http://quantum-machine.org/data/qm7.mat

# Download auxiliary scripts to the current env
shred -u molecule_visualizer.py
wget https://github.com/rhyan10/CECAM_unsupervised_learning_best_practices/blob/main/utils/view_molecule.py -O view_molecule.py

# Install and update required packages
pip install pacmap -U
pip install py3Dmol -U
pip install scikit-learn -U
pip install pandas -U
pip install seaborn -U