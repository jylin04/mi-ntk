"""
Implementation of the empirical NTK, the linearized approximation to a trained model that follows from it, and helper functions to test whether it's a good approximation.

Note that functorch is deprecated; to make this code future-proof we should migrate it to torch.func.
"""

import torch as t
import torch.nn as nn

from functorch import make_functional, jacrev, vmap
from typing import Tuple


def empirical_ntk(model: nn.Module, x_1: t.Tensor, x_2: t.Tensor) -> t.Tensor:
    """
    Returns NTK(x_1, x_2), of shape (N_1, N_2, C_1, C_2)
    where N_1 = len(x_1), N_2= len(x_2) and C_* are the output dims.
    """
    model = model.eval()
    fmodel, params = make_functional(model)

    # Function that runs a single example
    def fnet_single(params: Tuple[t.Tensor, ...], x: t.Tensor) -> t.Tensor:
        return fmodel(params, x.unsqueeze(0)).squeeze(0)

    # Jacobians. jacrev returns a per-sample function jac_fn(params, x) -> Tuple[T.Tensor, ...]. vmap vectorizes it letting us loop over the batch.
    jac1 = vmap(jacrev(fnet_single), (None, 0))(
        params, x_1
    )  # Tuple[t.Tensor...] with one entry per parameter of size *S_i, where jac[i] has shape(N_1, C_1, *S_i).
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x_2)

    jac1 = [
        j.flatten(start_dim=2) for j in jac1
    ]  # Concat params into one axis. jac[i] has shape [N_1, C, P_i]
    jac2 = [j.flatten(start_dim=2) for j in jac2]

    out = t.stack(
        [t.einsum("Ncp,Mdp->NMcd", j1, j2) for j1, j2 in zip(jac1, jac2)]
    )  # Contract over P in each NTK block
    out = out.sum(0)
    return out


class LinearisedPredictor:
    """
    Given a model, comput the eNTK approximation for that model on a test point x, schematically
    y_i(x) = \sum_{a_1, a_2}K_{ij}(x, a_1)K_{jk}^{-1}(a_1, a_2)y_{k,a_2}.
    """

    def __init__(
        self,
        model: nn.Module,
        x_train: t.Tensor,
        y_train: t.Tensor,
        ridge: float = 1e-6,
    ):

        self.model = model.eval()
        self.x_train = x_train
        self.y_train = y_train
        self.N, self.C = y_train.shape

        # Compute training kernel
        K = empirical_ntk(model, x_train, x_train)  # (N, N, C, C)
        K = K.permute(0, 2, 1, 3).reshape(
            self.N * self.C, self.N * self.C
        )  # (N*C, N*C)

        # Solve K^{-1}\alpha = y
        y_train = y_train.reshape(self.N * self.C)  # (N*C,)
        ridge = ridge * t.eye(K.size(0), device=K.device)
        self.alpha = t.linalg.solve(K + ridge, y_train).detach()  # (N*C,)

        # Evaluate model on the data and linearize around it
        f_train = self.model(self.x_train).reshape(self.N * self.C)  # (N*C,)
        self.alpha2 = t.linalg.solve(K + ridge, y_train - f_train).detach()  # (N*C,)

    def __call__(self, x: t.Tensor, expand_around_model: bool = True) -> t.Tensor:
        """
        Return NTK-based prediction for x. Output shape: (|x|, C).
        If expand_around_model is true, we return the linearised predictor around the model;
        else we return the `pure NTK' predictor around 0 (~= model outputs at zero-mean initialization.)
        """
        B = x.shape[0]

        K_q = empirical_ntk(self.model, x, self.x_train)  # (B, N, C, C)
        K_q = K_q.permute(0, 2, 1, 3).reshape(
            x.shape[0] * self.C, self.N * self.C
        )  # (B*C, N*C)

        if expand_around_model:
            return (self.model(x).detach().reshape(-1) + K_q @ self.alpha2).reshape(
                B, self.C
            )

        else:
            return (K_q @ self.alpha).reshape(B, self.C)


def loss_acc_gap(
    model: nn.Module,
    lin_model: LinearisedPredictor,
    x_val: t.Tensor,
    y_val: t.Tensor,
    loss_fn=t.nn.functional.mse_loss,
    expand_around_model: bool = True,
) -> Tuple[float, float]:
    """
    Returns (the ratio of the loss function evaluated on the eNTK approximation vs. on the full model,
    the percent difference between the accuracy of the eNTK approximation and the full model).
    """

    y_lin = lin_model(x_val, expand_around_model).detach()
    y_full = model(x_val).detach()

    loss_lin = loss_fn(y_lin, y_val).item()
    loss_full = loss_fn(y_full, y_val).item()
    loss_ratio = loss_lin / loss_full

    acc_lin = (y_lin.argmax(1) == y_val.argmax(1)).float().mean().item()
    acc_full = (y_full.argmax(1) == y_val.argmax(1)).float().mean().item()
    acc_gap = (acc_full - acc_lin) * 100

    return loss_ratio, acc_gap


def r2_score(
    model: nn.Module,
    lin_model: LinearisedPredictor,
    x_val: t.Tensor,
    expand_around_model: bool = True,
) -> float:
    """
    Returns R2 score = 1 - residual sum of squares / total sum of squares.
    If the predictor were perfect, R2 would be 1; if no better than predicting the mean every time, R2 would be 0; values close to 1 are better.
    """

    y_lin = lin_model(x_val, expand_around_model).detach()
    y_full = model(x_val).detach()

    num = t.sum((y_lin - y_full) ** 2)
    den = t.sum((y_full - y_full.mean()) ** 2)

    return 1 - num / den.item()
