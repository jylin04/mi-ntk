"""
Feature map, R^2, reconstruction error and other utilities to make comparison plots.
"""

import torch as t
import torch.nn as nn

from ntk import LinearisedPredictor

from typing import Tuple


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