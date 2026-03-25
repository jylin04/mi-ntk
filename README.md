# mi-ntk
Code for the paper **Feature Identification via the Empirical NTK**. [https://arxiv.org/abs/2510.00468]

This repo contains small-scale research code for reproducing the experiments in the paper.

## What is in this repo

- `ntk.py`: core Jacobian and eNTK utilities and the Laplacian-based rotation helpers used in the modular arithmetic analysis.
- `ma.py`: utilities for modular arithmetic MLP experiment.
- `tms.py`: utilities for TMS experiment
- `metrics.py`:  comparison utilities for the trained model and its linearized eNTK approximation.

### Notebooks

- `tms_exploration.ipynb`: Toy Models of Superposition experiments.
- `modular_arithmetic.ipynb`: modular arithmetic experiments in a 1L MLP.
- `modular_arithmetic_1L_transformer.ipynb`: modular arithmetic experiments in a 1L Transformer.

## Reproducing results

Run the notebooks for the respective sections.

*Important note*: some of the multi-panel figures in the paper are assembled from results saved at different training epochs and then displayed together, so they will not be reproduced by a single top-to-bottom notebook run. Conversely, some of the notebook cells containing multi-panel figures will not run as-is (since they call matrices that were saved manually and have since been deleted). However, the individual panels should be reproducible by running the code at the relevant epoch and hyperparameter setting.
