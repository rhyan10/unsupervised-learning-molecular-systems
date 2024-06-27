# Best Practices in Unsupervised Learning for Molecular Systems
In this repository, we have created various Python notebooks illustrating some of the concepts and best practices introduced in the article *Unsupervised Learning for Molecular Systems.*

## Contents

### Datasets
#### ZINC-250k
ZINC-250k is a subset of ZINC, a free database of commercially-available compounds for virtual screening. The ZINC-250k subset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/basu369victor/zinc250k); each compound contains values for the partition coefficient (logP), quantitative estimate of drug-likeness (QED), and synthetic accessibility score (SAS). 1K compounds from ZINC-250k are provided in [data/](./data/).

[This subset](./data/zinc-250k-sample.csv) is used to explore the various ways we can represent molecules computationally.

#### QM7
The QM7 dataset to be used in the examples below is provided in [data/qm7.xyz](./data/qm7.xyz). It has already been preprocessed.

It includes SMILES strings, 3D coordinates, and various quantum properties.

The original data is available on [this page](http://quantum-machine.org/datasets/).

### Section 1: Representations and Descriptors
This directory contains four notebooks touching on different aspects of molecular representations. These are:
* [Fingerprints and SMILES](1-Representations-and-Descriptors/Fingerprints-and-SMILES.ipynb)
* [Physicochemical Descriptors](1-Representations-and-Descriptors/Physicochemical-Descriptors.ipynb)
* [Coulomb Matrices](1-Representations-and-Descriptors/Coulomb-Matrices.ipynb)
* [SOAP Descriptors](1-Representations-and-Descriptors/SOAP-Descriptors.ipynb)

### Example 2: Dimensionality Reduction
This directory contains one notebook illustrating concepts and best practices for [Dimensionality Reduction](2-Dimensionality-Reduction/Dimensionality-Reduction.ipynb).

### Example 3: Clustering
This directory contains one notebook illustrating the main idea behind [Clustering](3-Clustering/Clustering-Concepts.ipynb), including various walk-through examples.

### Example 4: Autoencoders
Finally, this directory contains one notebook demonstrating how to construct a [Variational Autoencoder](4-Autoencoders/Variational-Autoencoder.ipynb).

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
* Hessam Yazdani