# Unsupervised Learning for Molecular Systems -- Introductory Tutorials
CECAM Workshop, Vienna, 2023

# Walk-through examples

## Datasets
### QM7
The QM7 dataset to be used in the examples below is available in [data/qm7.xyz](./data/qm7.xyz). Data was downloaded from blah blah blah and processed according to blah blah.

It includes SMILES strings, 3D coordinates, and blah blah quantum properties.

The original data is available on [this page](http://quantum-machine.org/datasets/).

### Alanine dipeptide trajectory


## Example 1: Feature selection
[Walk-through](./walk-throughs/1-Feature-Selection.ipynb)
1. Illustrate with 2D representation, 3D representation, and MD trajectories (3 examples of increasing complexity).

## Example 2: Dimensionality reduction
[Walk-through](./walk-throughs/2-Dimensionality-Reduction.ipynb)
1. Use same examples as above.

## Example 3: Generative modeling
[Walk-through](./walk-throughs/3-Generative-Modeling.ipynb)
1. Training an autoencoder on SMILES from QM9
2. Searching latent space for molecules with high DRD2 activity
3. Comparison to virtual screening/number of oracle calls
4. Point to relevant works in the literature for 3D and perhaps MD applications of generative modeling (but this cannot be here as those tasks are too complex to run in a laptop)
