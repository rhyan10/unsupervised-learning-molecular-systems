# Unsupervised Learning for Molecular Systems
In this repository, we have designed and collected various Python notebooks to illustrate some of the concepts and best practices for applying unsupervised learning methods to molecular data.

## Contents

### Dependencies
To run these notebooks, you will need to use a kernel with the following libraries installed:
* RDKit
* Pandas
* ipykernel

You can create a new environment in the terminal via the following sequence of commands:
```
conda create -n myenv python=3.9
conda activate myenv
pip install pandas ipykernel rdkit-pypi
```

Then, update the Python kernel you are using to run each notebook. Go to `Kernel` > `Change kernel` > `Python (myenv)`.

Additional installation instructions for specific libraries are given within each notebook.

### Data sets
There are three main data sets used in the Python notebooks presented here.

#### ZINC-250k
ZINC-250k is a subset of ZINC, a free database of commercially-available compounds for virtual screening. The ZINC-250k subset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/basu369victor/zinc250k); each compound contains values for the partition coefficient (logP), quantitative estimate of drug-likeness (QED), and synthetic accessibility score (SAS). 1K compounds from ZINC-250k are provided in the [data](./data/) directory.

[This subset](./data/zinc-250k-sample.csv) is used to explore the various ways we can represent molecules computationally.

#### QM7
The QM7 data set is also used for some of the walk-through examples using molecular structural information. A copy of the data set is provided in [data/qm7.xyz](./data/qm7.xyz), and has already been preprocessed.

It includes SMILES strings, 3D coordinates, and various quantum properties.

The original data is available on [this page](http://quantum-machine.org/datasets/).

#### MD17
One notebook applies clustering to molecular dynamics data and uses a 10k frame subset of the aspirin trajectory in MD17. The full trajectory for aspirin, as well as other MD17 trajectories, can be downloaded from [www.sgdml.org](http://www.sgdml.org/#datasets), although we provide a copy of the subset used herein in the [data](./data/) directory.

### Section 1: Representations and Descriptors
This directory contains four notebooks touching on different aspects of molecular representations. These are:
* [Fingerprints and SMILES](1-Representations-and-Descriptors/Fingerprints-and-SMILES.ipynb)
* [Physicochemical Descriptors](1-Representations-and-Descriptors/Physicochemical-Descriptors.ipynb)
* [Coulomb Matrices](1-Representations-and-Descriptors/Coulomb-Matrices.ipynb)
* [SOAP Descriptors](1-Representations-and-Descriptors/SOAP-Descriptors.ipynb)

### Example 2: Dimensionality Reduction
This directory contains one notebook illustrating concepts and best practices for [Dimensionality Reduction](2-Dimensionality-Reduction/Dimensionality-Reduction.ipynb) with molecular data.

### Example 3: Clustering
This directory contains two notebooks illustrating different ways to use clustering on molecular data:
* [Clustering Trajectories](3-Clustering/Clustering-Trajectories.ipynb)
* [Clustering Conformers](3-Clustering/Clustering-Conformers.ipynb)

### Example 4: Autoencoders
Finally, this directory contains one notebook demonstrating how to construct a [Variational Autoencoder](4-Generative-Modeling/Variational-Autoencoder.ipynb).

## Contributors
This work was initiated at the CECAM Workshop on *Machine-learned potentials in molecular simulation: best practices and tutorials* held in Vienna, July 2023.

Since then, the following authors have contributed to this repo and the accompanying article:

* [Rhyan Barrett](https://github.com/rhyan10)
* Arnav Brahmasandra
* [Bingqing Cheng](https://github.com/BingqingCheng)
* [Gregory Fonseca](https://github.com/fonsecag)
* [Ivan Gilardoni](https://github.com/IvanGilardoni)
* [Toni Oestereich](https://github.com/ToOest)
* [Max Pinheiro Jr](https://github.com/maxjr82)
* [Rocío Mercado](https://github.com/rociomer)
* [Hessam Yazdani](https://github.com/YazdaniSIGMaLab)
