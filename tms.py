"""
Models, dataset and training loop to replicate baseline experiments
in "Toy Models of Superposition" https://transformer-circuits.pub/2022/toy_model/index.html
and "Superposition, Memorization and Double Descent" https://transformer-circuits.pub/2023/toy-double-descent/index.html

The main changes in the latter are that compared to vanilla TMS we
* Normalize the dataset points to have norm 1.
* Have the training loop take the dataset as an argument.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


# --------- Model ---------
class LinearModel(nn.Module):
    """
    A 2-layer linear model y = W^TWx +b.
    """

    def __init__(self, n_features: int, n_hidden: int) -> None:
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty(n_hidden, n_features)))
        self.b = nn.Parameter(t.zeros(n_features))

    def forward(self, x: t.Tensor) -> t.Tensor:  # x: (batch, n_features)
        hidden = x @ self.W.T
        out = hidden @ self.W + self.b
        return out


class ReluModel(nn.Module):
    """
    A 2-layer MLP with a Relu activation and shared weights: y = ReLU(W^TW x + b)
    """

    def __init__(self, n_features: int, n_hidden: int) -> None:
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty(n_hidden, n_features)))
        self.b = nn.Parameter(t.zeros(n_features))

    def forward(self, x: t.Tensor) -> t.Tensor:  # x : (batch, n_features)
        hidden = x @ self.W.T
        out = t.relu(hidden @ self.W + self.b)
        return out


# -------- Data --------
def generate_batch(
    batch_size: int, n_features: int, S: float, device: str = "cpu"
) -> t.Tensor:
    """
    Returns a tensor of shape (batch_size, n_features).
    Each feature is set to 0 with probability S and to a random value in [0,1] otherwise.
    """
    out = t.rand(batch_size, n_features, device=device)
    mask = t.rand(batch_size, n_features, device=device) < 1 - S
    out = out * mask
    return out


def generate_batch_with_norm(
    batch_size: int, n_features: int, S: float, device: str = "cpu"
) -> t.Tensor:
    """
    Returns a tensor of shape (batch_size, n_features).
    Each feature is first set to 0 with probability S else to a random value in [0,1],
    then the feature vector is normalized to 1.
    """
    out = t.rand(batch_size, n_features, device=device)
    mask = t.rand(batch_size, n_features, device=device) < 1 - S
    out = out * mask

    out = out / (out.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return out


# -------- Training loop  --------
def importance_weighted_loss(
    input: t.Tensor, labels: t.Tensor, importance: t.Tensor
) -> t.Tensor:
    """
    input: (batch, n_features)
    labels: (batch, n_features)
    importance: (n_features)

    Return the average value of I_i (in_i - out_i)**2 over features and batches.
    """
    return (importance * (input - labels) ** 2).mean()


def importance_weighted_mse_loss(
    input: t.Tensor, labels: t.Tensor, importance: t.Tensor
) -> t.Tensor:
    """
    input: (batch, n_features)
    labels: (batch, n_features)
    importnce: (n_features)

    Returns the average value of I_i * BCE(logit_i, label_i)
    """
    return F.binary_cross_entropy(input, labels, weight=importance).mean()


def train_tms(
    model: nn.Module,
    opt: t.optim.Optimizer,
    batch_size: int = 1024,
    n_features: int = 80,
    S: float = 0.9,
    I: float = 0.9,
    device: str = "cpu",
    loss_fn: Callable[
        [t.Tensor, t.Tensor, t.Tensor], t.Tensor
    ] = importance_weighted_loss,
) -> t.Tensor:
    """
    Trains the model on the TMS data and loss function for one epoch and returns the average loss.
    Note that TMS generates a fresh dataset during each epoch.
    """
    # Generate dataset
    data = generate_batch(batch_size, n_features, S, device)

    # Training loop
    opt.zero_grad()
    out = model(data)
    loss = loss_fn(
        out,
        data,
        t.tensor(I, device=data.device) ** t.arange(n_features, device=data.device),
    )
    loss.backward()
    opt.step()

    return loss


def train_tms_fixed(
    model: nn.Module,
    dataset: t.Tensor,
    opt: t.optim.Optimizer,
    I: float = 0.9,
    loss_fn: Callable[
        [t.Tensor, t.Tensor, t.Tensor], t.Tensor
    ] = importance_weighted_loss,
) -> t.Tensor:
    """
    Trains the model on the given dataset for one epoch and returns the average loss.
    """
    n_features = dataset.size(1)

    # Training loop
    opt.zero_grad()
    out = model(dataset)
    loss = loss_fn(
        out,
        dataset,
        t.tensor(I, device=dataset.device)
        ** t.arange(n_features, device=dataset.device),
    )
    loss.backward()
    opt.step()

    return loss
# %%
