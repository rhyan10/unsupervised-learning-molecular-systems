# Unsupervised learning for molecular systems - best practices
CECAM Workshop, Vienna, 2023

# Walk-through examples (good and bad examples)

## Datasets
[Walk-through](./walk-throughs/0-Acquire-Data.md)
QM9, MD22
0. Start with downloading and pre-processing the data (it will be used in the following examples)
## Example 1: Feature selection
1. Illustrate with 2D representation, 3D representation, and MD trajectories (3 examples of increasing complexity).
[Walk-through](./walk-throughs/1-Feature-Selection.md)
## Example 2: Dimensionality reduction
[Walk-through](./walk-throughs/2-Dimensionality-Reduction.md)
1. Use same examples as above.
## Example 3: Generative modeling
[Walk-through](./walk-throughs/3-Generative-Modeling.md)
1. Training an autoencoder on SMILES from QM9
2. Searching latent space for molecules with high DRD2 activity
3. Comparison to virtual screening/number of oracle calls
4. Point to relevant works in the literature for 3D and perhaps MD applications of generative modeling (but this cannot be here as those tasks are too complex to run in a laptop)
