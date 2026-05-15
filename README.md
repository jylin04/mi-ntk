# mi-ntk
Code for the paper **Feature Identification via the Empirical NTK**. [https://arxiv.org/abs/2510.00468]

This repo contains small-scale research code for reproducing the experiments in the paper.

## What is in this repo

- `ntk.py`: core Jacobian and eNTK utilities used in the modular arithmetic analysis. 
- `ma.py`: utilities for modular arithmetic MLP experiment.
- `metrics.py`:  comparison utilities for the trained model and its linearized eNTK approximation.
  
The utilities for approximate eNTK eigenanalysis on Gemma-3-270M are defined directly in the notebook.

### Notebooks

- `modular_arithmetic_v4.ipynb`: modular arithmetic experiments in a 1L MLP.
- `modular_arithmetic_transformer_v4.ipynb`: modular arithmetic experiments in a 1L Transformer.
- `gemma_3_270m.ipynb`: approximate eNTK eigenanalysis and comparison to independently-specified grammar features in Gemma-3-270M.

## Reproducing results

To reproduce the modular arithmetic results in sections 3 and 4 of the paper, run the respective notebooks. 

The language model notebook `gemma_3_270m.ipynb` contains the code used to generate the datasets in section 5 of the paper and compute approximate eigendirections of the eNTK one layer at a time. To reproduce Table 3 in the paper, one should compute the eigendirections for all layers and extremize the AUROC score across them. The notebook currently shows the results of computing the AUROC score for a particular model layer (16), that accounts for some of the rows of Table 3. 
