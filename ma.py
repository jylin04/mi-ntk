"""
Models, dataset and training loop to replicate baseline experiments in
Gromov: "Grokking Modular Arithmetic" https://arxiv.org/abs/2301.02679
"""
import math

import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from typing import Tuple, Optional, Callable


# --------- Model ---------
class Quadratic(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * x


class Scale(nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def forward(self, x):
        return x * self.c


class ModularArithmeticMLP(nn.Module):
    """
    Two-layer MLP with mean-field parametrization & no biases.

    Input: size 2*p (concat of two one-hot-encoded Z_p integers)
    Output: size p  (logits)
    """

    def __init__(self, p: int, n_hidden: int, device: Optional[t.device] = None):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * p, n_hidden, bias=False, device=device),
            Scale(1 / math.sqrt(2 * p)),
            Quadratic(),
            nn.Linear(n_hidden, p, bias=False, device=device),
            Scale(1 / n_hidden),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)



# -------- Data --------
def build_mod_arith_data(
    p: int, device: Optional[t.device] = None
) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Returns
    x: (p^2, 2*p)     p^2 unique data points of concatenated one-hot embeddings of two integers m,n.
    y: (p^2, p)       One-hot labels.
    y_idx: (p^2,)     Labels.
    digits: (p^2, 2)  Pairs of integers (m,n).
    """

    device = device or t.device("cuda" if t.cuda.is_available() else "cpu")

    digits = t.cartesian_prod(t.arange(p, device=device), t.arange(p, device=device))

    x_left = F.one_hot(digits[:, 0], num_classes=p).float()
    x_right = F.one_hot(digits[:, 1], num_classes=p).float()
    x = t.cat([x_left, x_right], dim=1)

    y_idx = (digits[:, 0] + digits[:, 1]) % p
    y = F.one_hot(y_idx, num_classes=p).float()

    return x, y, y_idx, digits


def build_mod_arith_data_symbreak(p: int, symmetric: bool = False, device: Optional[t.device] = None) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
  """
  Embed input integers as a cyclic rotation of a random vector to break symmetry. Idea from Kunin et al. "Alternating Gradient Flows" https://arxiv.org/abs/2506.06489 
  Returns
  x: (p^2, 2*p)     p^2 unique data points of concatenated cyclic embeddings of two integers m,n.
  y: (p^2, p)       One-hot labels.
  y_idx: (p^2,)     Labels.
  digits: (p^2, 2)  Pairs of integers (m,n).
  """
  device = device or t.device("cuda" if t.cuda.is_available() else "cpu")

  digits = t.cartesian_prod(t.arange(p, device=device), t.arange(p, device=device))
  n = digits[:,0]
  m = digits[:,1]

  # Create two random, zero-mean unit-norm vectors of length p
  xL = t.randn(p, device=device)
  xL = (xL - xL.mean()) / (xL.norm() + 1e-8)
  xR = t.randn(p, device=device)
  xR = (xR - xR.mean()) / (xR.norm() + 1e-8)

  # Bank all cyclic shifts: 
  BL = t.stack([t.roll(xL, shifts=int(a), dims=0) for a in range(p)], dim=0)  # (p, p) : {{x0, x1,...xn}, {xn, x0, x1, ...}, {xn-1, xn, x0, ...}}
  BR = t.stack([t.roll(xR, shifts=int(a), dims=0) for a in range(p)], dim=0)  # (p, p)

  # Encode residues by picking the corresponding shift row. This generalizes one-hot-encoding
  x_left  = BL[n, :]      # (p^2, p)
  if symmetric == False:
    x_right = BR[m, :]    # (p^2, p)
  else:
    x_right = BL[m, :]    
  x = t.cat([x_left, x_right], dim=1)

  y_idx = (n + m) % p
  y = F.one_hot(y_idx, num_classes=p).float()

  return x, y, y_idx, digits



# -------- Train/test utils  --------
def train(model: nn.Module,
          loader: DataLoader,
          opt: t.optim.Optimizer,
          device: t.device,
          loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor] = F.mse_loss
          ) -> float:
    """Trains for one epoch and returns the average loss."""
    losses: list[float] = []
    model.train()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def test(model: nn.Module,
         loader: DataLoader,
         device: t.device) -> float:
    '''Returns the model accuracy on the test set.'''

    model.eval()
    correct = 0
    with t.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x).argmax(dim=1)
            truth = y.argmax(dim=1)
            correct += (predictions == truth).sum().item()

    return 100* correct / len(loader.dataset)
