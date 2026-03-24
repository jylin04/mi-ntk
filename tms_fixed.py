"""
Additional helper functions to replicate baseline experiment
in "Superposition, Memorization and Double Descent" https://transformer-circuits.pub/2023/toy-double-descent/index.html

The main changes compared to TMS are that we 
* Normalize the dataset points to have norm 1.
* Have the training loop take the dataset as an argument. 
"""
import torch as t
import torch.nn as nn 
from tms import importance_weighted_loss


# -------- Data --------
def generate_batch_with_norm(batch_size: int, n_features: int, S: float, device: str="cpu") -> t.Tensor:
    """
    Returns a tensor of shape (batch_size, n_features).
    Each feature is first set to 0 with probability S else to a random value in [0,1],
    then the feature vector is normalized to 1.
    """
    out = t.rand(batch_size, n_features, device=device)
    mask = t.rand(batch_size, n_features, device=device) < 1-S
    out = out * mask

    out = out / (out.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return out



# -------- Training loop  --------
def train_tms_fixed(model: nn.Module, dataset: t.Tensor, opt: t.optim.Optimizer, I: float = 0.9) -> t.Tensor:
    """
    Trains the model on the given dataset for one epoch and returns the average loss.
    """
    n_features = dataset.size(1)

    # Training loop
    opt.zero_grad()
    out = model(dataset)
    loss = importance_weighted_loss(out, dataset, t.tensor(I, device=dataset.device)**t.arange(n_features, device = dataset.device))
    loss.backward()
    opt.step()

    return loss