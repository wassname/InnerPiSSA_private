#!/usr/bin/env python3
"""Centralized layer selection logic for InnerPiSSA training.

Handles conversion of config parameters (depth_start, depth_end, n_depths, loss_depths)
to actual layer indices and module names. Single source of truth to avoid duplication.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def select_adapter_dims(
    hsS_cho: torch.Tensor,
    hsS_rej: torch.Tensor,
    S: torch.Tensor,
    r: int,
    norm_S: bool = True,
) -> torch.Tensor:
    """Select adapter dimensions using cho_var_snorm strategy.
    
    Selects r/2 dimensions from cho activations + r/2 from rej activations,
    ranked by variance in S-normalized projection space. Non-overlapping.
    
    This preserves task-active workspace rather than just the task-relevant delta.
    
    Args:
        hsS_cho: Chosen activations in S space [n_samples, d_out]
        hsS_rej: Rejected activations in S space [n_samples, d_out]
        S: Singular values [d_out] or [k]
        r: Target rank (number of dims to select)
        
    Returns:
        indices: Selected dimension indices [r], sorted
    """
    device = hsS_cho.device
    max_rank = min(hsS_cho.shape[1], S.shape[0])
    r_actual = min(r, max_rank)
    
    # Project to S-space and normalize by pretrained singular values
    if norm_S:
        proj_cho = (hsS_cho) / S.clamp(min=1e-8)  # [n_samples, rank]
        proj_rej = (hsS_rej) / S.clamp(min=1e-8)
    else:
        proj_cho = hsS_cho
        proj_rej = hsS_rej
    
    # Rank dimensions by variance
    cho_var = proj_cho.var(dim=0)  # [rank]
    rej_var = proj_rej.var(dim=0)
    
    # Get top r/2 from cho, then fill remaining with unique top from rej
    n_half = r_actual // 2
    _, cho_idx = torch.topk(cho_var, n_half)
    
    # Mask out cho selections and get remaining best from rej
    rej_var_masked = rej_var.clone()
    rej_var_masked[cho_idx] = -float('inf')  # Exclude already selected
    _, rej_idx = torch.topk(rej_var_masked, r_actual - n_half)
    
    indices = torch.cat([cho_idx, rej_idx]).sort()[0]
    return indices


def compute_pref_direction(
    hsS_cho: torch.Tensor,
    hsS_rej: torch.Tensor,
    method: str = "mean",
    k: int = 64,
    S: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute preference direction for loss using specified method.
    
    All inputs should already be in S-space (projected via U or V).
    
    Args:
        hsS_cho: Chosen hidden states in S-space [n_samples, d]
        hsS_rej: Rejected hidden states in S-space [n_samples, d]
        method: Selection method:
            - mean: simple mean difference
            - pca1/pca2/pca4: top-k PCs of diff
            - top_s: top-k dims by S magnitude
            - top_diff: top-k dims by |diff| magnitude  
            - top_diff_snorm: top-k dims by |diff|/S (upweights low-S high-diff)
            - adapter_dims: r/2 cho + r/2 rej by variance/S
            - adapter_dims_raw: r/2 cho + r/2 rej by variance (no S norm)
        k: Number of dimensions for multi-dim methods
        U: Unused, kept for API compatibility
        S: Singular values [d] - needed for top_s, top_diff_snorm, adapter_dims*
        V: Unused, kept for API compatibility
        
    Returns:
        pref_dir: Preference direction [d], unit normalized
    """
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    
    diffS = hsS_cho - hsS_rej  # [n_samples, d]
    d = diffS.shape[1]
    
    if method == "mean":
        # Simple mean difference - SNR=1.56, 100% sign agreement
        pref_dir = diffS.mean(dim=0)  # [d]
        return F.normalize(pref_dir, dim=0)
    
    elif method.startswith("pca"):
        # PCA on diff: pca1 = first PC, pca2 = top-2, etc.
        # Returns mean direction projected onto top-k PC subspace
        n_components = int(method[3:]) if method[3:].isdigit() else 1
        n_components = min(n_components, k, d, diffS.shape[0])
        
        # Truncate to top-k dims before PCA (full-rank V is rotation-invariant, 
        # so PCA on diff == PCA on diff@V — meaningless without truncation)
        k_trunc = min(k, d)
        diff_trunc = diffS[:, :k_trunc]  # [n, k_trunc] - top singular dims
        
        diff_np = diff_trunc.cpu().numpy()
        pca = PCA(n_components=n_components)
        pca.fit(diff_np)
        components_trunc = torch.from_numpy(pca.components_).to(diffS.device).float()  # [n_components, k_trunc]
        
        # Project mean diff onto PC subspace: sum of projections onto each PC
        mean_diff_trunc = diff_trunc.mean(dim=0)  # [k_trunc]
        # proj = sum_i (mean · pc_i) * pc_i = components.T @ (components @ mean)
        coeffs = components_trunc @ mean_diff_trunc  # [n_components]
        pref_dir_trunc = coeffs @ components_trunc  # [k_trunc]
        
        # Pad back to full dimension (zeros in truncated dims)
        pref_dir = torch.zeros(d, device=diffS.device)
        pref_dir[:k_trunc] = pref_dir_trunc
        
        return F.normalize(pref_dir, dim=0)  # [d]
    
    elif method == "top_s":
        # Select top-k dims by S magnitude, apply to diff
        if S is None:
            raise ValueError("top_s method requires S (singular values)")
        
        k_actual = min(k, S.shape[0], d)
        _, top_idx = torch.topk(S, k_actual)
        
        mean_diff = diffS.mean(dim=0)  # [d]
        mask = torch.zeros(d, device=diffS.device)
        mask[top_idx] = 1.0
        pref_dir = F.normalize(mean_diff * mask, dim=0)  # [d]
        
        return pref_dir

    elif method == "top_diff":
        # Select top-k dims by diff magnitude (not S magnitude)
        # Finds where cho/rej actually differ most, regardless of S
        mean_diff = diffS.mean(dim=0)  # [d]
        k_actual = min(k, d)
        _, top_idx = torch.topk(mean_diff.abs(), k_actual)
        
        mask = torch.zeros(d, device=diffS.device)
        mask[top_idx] = 1.0
        pref_dir = F.normalize(mean_diff * mask, dim=0)
        return pref_dir

    elif method == "top_diff_snorm":
        # Select top-k dims by S-normalized diff magnitude
        # Upweights low-S dims where diff is strong relative to baseline
        if S is None:
            raise ValueError("top_diff_snorm requires S (singular values)")
        
        mean_diff = diffS.mean(dim=0)  # [d]
        # Normalize by S to find dims with high diff relative to their typical magnitude
        diff_snorm = mean_diff.abs() / S.clamp(min=1e-8)
        k_actual = min(k, d)
        _, top_idx = torch.topk(diff_snorm, k_actual)
        
        mask = torch.zeros(d, device=diffS.device)
        mask[top_idx] = 1.0
        pref_dir = F.normalize(mean_diff * mask, dim=0)
        return pref_dir

    elif method == "adapter_dims":
        # Mean direction masked to adapter-selected dims (cho/rej variance, S-normalized)
        if S is None:
            raise ValueError("adapter_dims method requires S (singular values)")
        
        indices = select_adapter_dims(hsS_cho, hsS_rej, S, k, norm_S=True)
        
        mean_diff = diffS.mean(dim=0)  # [d]
        mask = torch.zeros(d, device=diffS.device)
        mask[indices] = 1.0
        pref_dir = F.normalize(mean_diff * mask, dim=0)  # [d]
        
        return pref_dir

    elif method == "adapter_dims_raw":
        # Same as adapter_dims but without S normalization (raw variance ranking)
        if S is None:
            raise ValueError("adapter_dims_raw method requires S (singular values)")
        
        indices = select_adapter_dims(hsS_cho, hsS_rej, S, k, norm_S=False)
        
        mean_diff = diffS.mean(dim=0)  # [d]
        mask = torch.zeros(d, device=diffS.device)
        mask[indices] = 1.0
        pref_dir = F.normalize(mean_diff * mask, dim=0)  # [d]
        
        return pref_dir

    else:
        raise ValueError(f"Unknown pref_dir_method: {method}")


def build_regexp(layer_indices: List[int], module_suffixes: List[str]) -> str:
    """Build PEFT target_modules regex from layer indices and module suffixes."""
    layer_nums = "|".join(str(L) for L in sorted(set(layer_indices)))
    module_names = "|".join(sorted(set(module_suffixes)))
    return f".*\\.({layer_nums})\\..*({module_names})"


@dataclass
class LayerSelection:
    """Layer selection result: which layers get adapters vs loss computation."""
    adapter_layer_indices: List[int]
    loss_layer_indices: List[int]
    adapter_layer_names: List[str]
    loss_layer_names: List[str]
    
    @property
    def adapter_regex(self) -> str:
        """Build PEFT target_modules regex from adapter indices and module suffixes."""

        return build_regexp(
            self.adapter_layer_indices,
            [name.split('.')[-1] for name in self.adapter_layer_names],
        )
    
    def check_causality(self) -> dict:
        """Check if loss layers are causally affected by adapter layers.
        
        Returns diagnostic info about layer ordering:
        - loss_before_adapter: loss layers computed BEFORE first adapter (no projection gradient)
        - loss_after_adapter: loss layers computed AFTER first adapter (has projection gradient)
        - min_adapter_layer: first adapter layer index
        
        If loss_before_adapter is non-empty, only coherence loss (output logprobs) 
        will provide gradients - projection loss will be zero at those layers.
        """
        min_adapter = min(self.adapter_layer_indices) if self.adapter_layer_indices else float('inf')
        max_adapter = max(self.adapter_layer_indices) if self.adapter_layer_indices else float('-inf')
        
        return {
            'loss_before_adapter': [L for L in self.loss_layer_indices if L < min_adapter],
            'loss_after_adapter': [L for L in self.loss_layer_indices if L >= min_adapter],
            'min_adapter_layer': min_adapter,
            'max_adapter_layer': max_adapter,
            'has_causality_issue': any(L < min_adapter for L in self.loss_layer_indices),
        }
    
    def translate_to_peft_model(self, model) -> 'LayerSelection':
        """Translate layer names for PeftModel (adds base_model.model prefix).
        
        After wrapping with PeftModel, layer paths change:
        - Before: 'model.layers.9.mlp.down_proj'  
        - After:  'base_model.model.model.layers.9.mlp.down_proj'
        
        This finds the correct paths by checking what actually exists in the PeftModel.
        """
        def translate_name(old_name: str) -> str:
            # Try common PeftModel prefixes
            candidates = [
                f"base_model.model.{old_name}",  # Standard PeftModel
                old_name,  # Maybe already correct
            ]
            
            model_modules = {name for name, _ in model.named_modules()}
            for candidate in candidates:
                if candidate in model_modules:
                    return candidate
            
            # Fallback: return original and hope for the best
            return old_name
        
        return LayerSelection(
            adapter_layer_indices=self.adapter_layer_indices,
            loss_layer_indices=self.loss_layer_indices,
            adapter_layer_names=[translate_name(n) for n in self.adapter_layer_names],
            loss_layer_names=[translate_name(n) for n in self.loss_layer_names],
        )


def normalize_layer_spec(layer_spec: List[float | int], total_layers: int) -> List[int]:
    """Convert layer specs (fractions or offsets) to absolute layer numbers."""
    normalized = []
    for x in layer_spec:
        if (x >= 0) and (x < 1):
            x = int(x * total_layers)
        layer_num = int(x) % total_layers
        normalized.append(layer_num)
    return normalized


def find_linear_layers(
    model: nn.Module,
    layer_indices: List[int],
    module_suffixes: List[str],
    blocklist: List[str] = ['vision']
) -> List[str]:
    """Find Linear modules at specified layer depths with given suffixes."""
    selected = []
    for name, module in model.named_modules():
        if any(block in name for block in blocklist):
            continue
        if name.endswith('.base_layer'):
            continue
        if not isinstance(module, nn.Linear):
            continue
        
        match = re.search(r"\.layers\.(\d+)\.", name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        
        if layer_idx not in layer_indices:
            continue
        
        if any(name.endswith(suffix) for suffix in module_suffixes):
            selected.append(name)
    
    return sorted(set(selected))


def compute_layer_selection(
    model: nn.Module,
    depth_start: float | int,
    depth_end: float | int,
    n_depths: int,
    loss_depths: List[float | int],
    modules: List[str],
    loss_modules: List[str] | None = None,
) -> LayerSelection:
    """Compute which layers get adapters vs loss, ensuring no overlap.
    
    Args:
        loss_modules: Modules for loss extraction. If None, uses same as modules.
                     Use ['up_proj'] for V-projection of residual stream.
    """
    from repeng.control import model_layer_list
    total_layers = len(model_layer_list(model))
    
    # Convert config to absolute layer indices
    start_layer, end_layer = normalize_layer_spec([depth_start, depth_end], total_layers)
    
    if not isinstance(loss_depths, list):
        loss_depths = [loss_depths]
    loss_layer_indices = normalize_layer_spec(loss_depths, total_layers)
    
    # Validate range
    assert start_layer <= end_layer, f"Invalid layer range: start {start_layer}, end {end_layer}"
    assert end_layer < total_layers, f"End layer {end_layer} exceeds total layers {total_layers}"
    assert start_layer >= 0, f"Start layer {start_layer} is negative"
    
    # Exclude loss layers from adapter candidates (avoid circular dependency)
    all_candidate_layers = list(range(start_layer, end_layer + 1))
    available_layers = [L for L in all_candidate_layers if L not in loss_layer_indices]
    
    if not available_layers:
        raise ValueError(
            f"No available layers after excluding loss layers {loss_layer_indices} "
            f"from range [{start_layer}, {end_layer}]"
        )
    
    # Select evenly-spaced subset from available layers
    if n_depths >= len(available_layers):
        adapter_layer_indices = sorted(set(available_layers))
    else:
        # Use linspace on indices, then map to actual layer numbers
        indices = np.linspace(0, len(available_layers) - 1, n_depths, dtype=int)
        adapter_layer_indices = sorted(set(available_layers[i] for i in indices))
    
    # Resolve layer names
    adapter_layer_names = find_linear_layers(model, adapter_layer_indices, modules)
    
    # Use separate modules for loss extraction if specified
    loss_modules_to_use = loss_modules if loss_modules is not None else modules
    loss_layer_names = find_linear_layers(model, loss_layer_indices, loss_modules_to_use)
    
    # Validate we found actual layers
    assert len(adapter_layer_names) > 0, (
        f"No adapter layers found matching indices {adapter_layer_indices} and modules {modules}"
    )
    assert len(loss_layer_names) > 0, (
        f"No loss layers found matching indices {loss_layer_indices} and modules {modules}"
    )
    
    return LayerSelection(
        adapter_layer_indices=adapter_layer_indices,
        loss_layer_indices=loss_layer_indices,
        adapter_layer_names=adapter_layer_names,
        loss_layer_names=loss_layer_names,
    )
