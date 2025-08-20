"""
Implementation of the Jacobian, empirical NTK, linearized approximation to a trained model that follows from it, and assorted helper functions.

Note that functorch is deprecated; to make this code future-proof we should migrate it to torch.func.
"""

import torch as t
import torch.nn as nn

from functorch import make_functional, jacrev, vmap
from typing import Tuple, Dict
from collections import OrderedDict


def full_jacobian(model: nn.Module, x: t.Tensor) -> t.Tensor:
    """
    Computes the full Jacobian of a pytorch model wrt its parameters, evaluated on a batch of inputs.

    Args:
        model: a Pytorch nn.Module.
        x: t.Tensor of shape (N, *S) for N the batch size and S the input shape expected by the model.

    Returns:
        J: t.Tensor of shape (N,C,P) for C the output shape expected by the model and P the model's parameter count.
    """
    model = model.eval()
    fmodel, params = make_functional(model)

    # Function that runs a single example
    def fnet_single(params: Tuple[t.Tensor, ...], x: t.Tensor) -> t.Tensor:
        return fmodel(params, x.unsqueeze(0)).squeeze(0)

    jac = vmap(jacrev(fnet_single), (None, 0))(
        params, x
    )  # Tuple[t.Tensor...] with one entry per parameter of size *S_i, where jac[i] has shape(N_1, C_1, *S_i).
    jac = [
        j.flatten(start_dim=2) for j in jac
    ]  # Concat params into one axis. jac[i] has shape [N, C, P_i]

    return t.cat(jac, dim=2)


def class_jacobian(model: nn.Module, x: t.Tensor, class_idx: int) -> t.Tensor:
    """
    Same as full_jacobian but returns J_class of shape (N, P) on the specified output index.
    """

    j_full = full_jacobian(model, x)
    return j_full[:, class_idx, :]


def empirical_ntk(model: nn.Module, x_1: t.Tensor, x_2: t.Tensor) -> t.Tensor:
    """
    Computes the empirical NTK.

    Args:
        model: a Pytorch nn.Module.
        x_1: t.Tensor with shape (N_1, *S) for S the input shape usually expected by the model.
        x_2: t.Tensor with shape (N_2, *S) for S the input shape usually expected by the model.

    Returns:
        NTK: t.Tensor of shape (N_1, N_2, C, C) for C the number of ouput dimensions expected by the model.
    """
    # TODO: Could replace jacrev+vmap block with full_jacobian.

    model = model.eval()
    fmodel, params = make_functional(model)

    # Function that runs a single example
    def fnet_single(params: Tuple[t.Tensor, ...], x: t.Tensor) -> t.Tensor:
        return fmodel(params, x.unsqueeze(0)).squeeze(0)

    # Jacobians. jacrev returns a per-sample function jac_fn(params, x) -> Tuple[T.Tensor, ...].
    # vmap vectorizes it letting us loop over the batch.
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


def empirical_ntk_by_layer(
    model: nn.Module, x_1: t.Tensor, x_2: t.Tensor
) -> "OrderedDict[str, t.Tensor]":
    """
    Computes the empirical NTK, separated by (groups of) layers.

    Args:
        model: a Pytorch nn.Module.
        x_1: shape (N1, *S) for S the input shape usually expected by the model.
        x_2: shape (N2, *S)

    Returns:
        OrderedDict[group_name, t.Tensor] each of shape (N_1, N_2, C, C).
    """
    model = model.eval()
    fmodel, params = make_functional(model)

    param_names = [name for name, _ in model.named_parameters()]

    # Function to remove the part of name after the last "."
    def group_fn(name):
        return name.rsplit(".", 1)[0]

    group_keys = [
        group_fn(n) for n in param_names
    ]  # e.g. ["layer1", "layer1", "layer2"]

    # Function that runs a single example
    def fnet_single(params: Tuple[t.Tensor, ...], x: t.Tensor) -> t.Tensor:
        return fmodel(params, x.unsqueeze(0)).squeeze(0)

    # Jacobians. jacrev returns a per-sample function jac_fn(params, x) -> Tuple[T.Tensor, ...].
    # vmap vectorizes it letting us loop over the batch.
    jac1 = vmap(jacrev(fnet_single), (None, 0))(
        params, x_1
    )  # Each shape (N_1, C, *param_i.shape)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x_2)

    # Initialize the dictionary
    out_by_group: "OrderedDict[str, t.Tensor]" = OrderedDict()
    for g in group_keys:
        if g not in out_by_group:
            out_by_group[g] = None  # lazy init so we know shape

    # Populate the dictionary
    for j1, j2, g in zip(jac1, jac2, group_keys):
        j1 = j1.flatten(start_dim=2)  # [N1, C, P_i]
        j2 = j2.flatten(start_dim=2)  # [N2, C, P_i]
        contrib = t.einsum("Ncp,Mdp->NMcd", j1, j2)  # [N1, N2, C, C]
        if out_by_group[g] is None:
            out_by_group[g] = contrib
        else:
            out_by_group[g] = (
                out_by_group[g] + contrib
            )  # Since we sum over contributions from all parameters [within layer]

    return out_by_group


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


def eig_decompose(
    ntk: t.Tensor, topk: int | None = None, ridge: float = 1e-8
) -> tuple[t.Tensor, t.Tensor]:
    """
    Return the top k eigenvalues and eigenvectors of a NTK matrix.

    Args:
        NTK  : t.Tensor with shape (N, N).
        topk : int
        ridge: float

    Returns:
        eigvals : t.Tensor with shape (m,) | m = N or topk
        eigvecs : t.Tensor with shape (N,m)
    """
    # Condition with small ridge
    diag_mean = ntk.diag().mean()
    ntk += ridge * diag_mean * t.eye(ntk.shape[0], device=ntk.device)

    eigvals, eigvecs = t.linalg.eigh(ntk)

    # Sort by descending order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    if topk is not None:
        eigvals = eigvals[:topk]
        eigvecs = eigvecs[:, :topk]

    return eigvals, eigvecs


def learn_dictionary_torch(
    J: t.Tensor,
    k: int,
    lam_l1: float = 0.0,
    lam_ortho: float = 0.0,
    lr: float = 3e-3,
    n_steps: int = 1000,
) -> Tuple[t.Tensor, t.Tensor]:
    """
    Returns A: (P, k)  learned atoms and S: (k, N*C) learned codes
    """
    Jt = J.T.detach()

    # PCA initializaiton
    U, Sigma, Vt = t.linalg.svd(Jt, full_matrices=False)  # Jᵀ = U Σ Vᵀ
    A0 = U[:, :k]
    S0 = t.diag(Sigma[:k]) @ Vt[:k]

    # Dictionary learning
    A = nn.Parameter(A0)
    S = nn.Parameter(S0)

    opt = t.optim.Adam([A, S], lr=lr)

    for step in range(n_steps):

        opt.zero_grad()

        dictionary_loss = ((Jt - A @ S) ** 2).mean() + lam_l1 * S.abs().mean()
        ortho = lam_ortho * ((A.T @ A - t.eye(k, device=A.device)) ** 2).mean()
        tot_loss = dictionary_loss + ortho

        tot_loss.backward()
        opt.step()

    return A.detach(), S.detach()


def learn_dictionary_supervised(
    J: t.Tensor,
    y_target: t.Tensor,
    k: int,
    lam_l1: float = 0.0,
    lam_ortho: float = 0.0,
    lam_sup: float = 0.0,
    lr: float = 3e-3,
    n_steps: int = 1000,
) -> Tuple[t.Tensor, t.Tensor]:
    """
    y_target should be shape (N*C, C)
    Returns A: (P,k) learned atoms, S(k, N*C) learned codes, and the trained linear head W : (n_out,k)
    """
    Jt = J.T.detach()

    # PCA initialization
    U, Sigma, Vt = t.linalg.svd(Jt, full_matrices=False)  # Jᵀ = U Σ Vᵀ
    A0 = U[:, :k]
    S0 = t.diag(Sigma[:k]) @ Vt[:k]

    A = nn.Parameter(A0)
    S = nn.Parameter(S0)

    # Supervised head
    C = y_target.shape[1]
    head = nn.Linear(k, C, bias=False).to(Jt.device)

    opt = t.optim.Adam([A, S, head.weight], lr=lr)

    for step in range(n_steps):

        opt.zero_grad()

        y_pred = head(S.T)

        dictionary_loss = ((Jt - A @ S) ** 2).mean() + lam_l1 * S.abs().mean()
        ortho = lam_ortho * ((A.T @ A - t.eye(k, device=A.device)) ** 2).mean()

        sup = lam_sup * nn.functional.mse_loss(y_pred, y_target)

        tot_loss = dictionary_loss + ortho + sup

        tot_loss.backward()
        opt.step()

    return A.detach(), S.detach(), head
