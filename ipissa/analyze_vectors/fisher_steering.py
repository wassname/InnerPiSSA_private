import torch
import numpy as np
from torch.linalg import solve, qr
from jaxtyping import Float
from torch import Tensor
from typing import List, Dict, Optional, Literal
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..control import ControlModel


def natural_gradient_steering(
    grad_matrix: Float[Tensor, "batch dim"],
    lambda_reg: Optional[float] = None,
    fim: Literal[
        "empirical", "improved_empirical", "covariance"
    ] = "improved_empirical",
    low_dim: int | None = None,
    seed: int | None = 42,
    grad_norms: Optional[Float[Tensor, "batch"]] = None,
    eig_threshold: float = 1e-8,  # New: Threshold for min eigenvalue clamping
    clip_weights: bool = False,  # New: Optional clipping for iEF weights
) -> Float[Tensor, "dim"]:
    """
    Computes natural gradient steering vector using Fisher preconditioning.
    Optionally projects to random subspace for large dims.

    Args:
        grad_matrix: Gradients from SimPO loss, shape (batch, dim)
        lambda_reg: Regularization for Fisher matrix inversion
        fim: Which FIM approximation to use ("empirical", "improved_empirical", or "covariance")
        low_dim: If set and < dim, project to random orthonormal subspace
        seed: Seed for random projection; None for non-reproducible
        grad_norms: Per-sample loss-input gradient norms ||∇_{loss_input} l_n||². Required when fim=="improved_empirical".

    Returns:
        Natural gradient steering vector
    """
    device = grad_matrix.device
    batch_size, dim = grad_matrix.shape

    # # UNTESTED: Auto low-dim when severely undersampled
    # if batch_size < dim // 10:
    #     if low_dim is None or low_dim > batch_size // 2:
    #         low_dim = max(16, batch_size // 4)
    #         print(f"FIXME: Auto-setting low_dim={low_dim} for n={batch_size}, d={dim}")

    # UNTESTED
    # used_projection = False
    # if low_dim and low_dim < dim:
    #     if seed is not None:
    #         torch.manual_seed(seed)
    #     # Random orthonormal projection matrix P [dim, low_dim]
    #     rand_matrix = torch.randn(dim, low_dim, device=device, dtype=torch.float32)
    #     rand_matrix = rand_matrix / torch.norm(rand_matrix, dim=0, keepdim=True)  # normalize columns
    #     P, _ = qr(rand_matrix, mode="reduced")  # orthonormal columns

    #     # Project gradients to subspace
    #     grad_matrix = grad_matrix @ P  # [batch, low_dim]
    #     dim = low_dim  # update for below
    #     used_projection = True

    # Mean gradient direction (what we want to steer towards)
    mean_grad = grad_matrix.mean(dim=0)  # [dim]

    # Compute F_matrix (unchanged, but add weights clipping for iEF)
    match fim:
        case "empirical":
            F_matrix = torch.einsum("bi,bj->ij", grad_matrix, grad_matrix) / batch_size
        case "improved_empirical":
            assert grad_norms is not None
            weights = 1.0 / (grad_norms.to(device).float() + 1e-6)
            if clip_weights:  # New: Clip to prevent outlier amplification (Wu et al. sensitivity-inspired)
                max_weight = 10 * weights.median()
                weights = torch.clamp(weights, max=max_weight)
            weighted_grads = grad_matrix * weights.unsqueeze(-1)
            F_matrix = (
                torch.einsum("bi,bj->ij", weighted_grads, grad_matrix) / batch_size
            )
        case "covariance":
            centered_grads = grad_matrix - mean_grad.unsqueeze(0)
            denom = max(batch_size - 1, 1)
            F_matrix = torch.einsum("bi,bj->ij", centered_grads, centered_grads) / denom

    # Adaptive regularization (your commented version, now default if lambda_reg is None)
    trace_F = torch.trace(F_matrix)
    avg_eig = trace_F / dim
    if lambda_reg is None:
        lambda_reg = max(1e-6, 0.01 * avg_eig)

    # New: Min eigenvalue clamping for stability (if batch_size << dim, per Wu et al. App. H)
    # Uses symmetric eigvalsh (faster for real symmetric matrices)
    if batch_size < dim // 10:  # Trigger only when potentially undersampled
        eigvals = torch.linalg.eigvalsh(F_matrix)
        min_eig = eigvals.min()
        if min_eig < eig_threshold:
            lambda_reg = max(lambda_reg, 1e-4 * avg_eig)  # Boost reg (tunable factor)

    F_reg = F_matrix + torch.eye(dim, device=device, dtype=torch.float32) * lambda_reg

    # Natural gradient: θ_natural = F^(-1) * ∇L
    v_natural = solve(F_reg, mean_grad)

    # # UNTESTED Magnitude rescaling: preserve steering strength proportional to objective signal
    # # Rescale by ||mean_grad|| to maintain proportionality to the objective gradient magnitude
    # v_natural_norm = torch.norm(v_natural)
    # if v_natural_norm > 1e-8:
    #     v_natural = v_natural * (mean_grad_norm / v_natural_norm)

    # UNTESTED
    # if used_projection:
    #     # HACK: Scale correction for low-dim projection
    #     # Preserve gradient magnitude scale
    #     orig_grad_norm = torch.norm(mean_grad)

    #     # Project back to original space
    #     v_natural = P @ v_natural  # [dim_orig, low_dim] @ [low_dim] -> [dim_orig]

    #     # Rescale to match original gradient scale
    #     proj_norm = torch.norm(v_natural)
    #     if proj_norm > 1e-8:
    #         v_natural = v_natural * (orig_grad_norm / proj_norm)

    return v_natural
