#!/usr/bin/env python3
"""Centralized layer selection logic for InnerPiSSA training.

Handles conversion of config parameters (depth_start, depth_end, n_depths, loss_depths)
to actual layer indices and module names. Single source of truth to avoid duplication.
"""
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import torch.nn as nn


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
        # Extract module suffixes from adapter layer names
        suffixes = set()
        for name in self.adapter_layer_names:
            # e.g., "model.layers.10.mlp.down_proj" -> "down_proj"
            suffix = name.split('.')[-1]
            suffixes.add(suffix)
        
        layer_nums = "|".join(str(L) for L in self.adapter_layer_indices)
        module_names = "|".join(sorted(suffixes))
        return f".*\\.({layer_nums})\\..*({module_names})"
    
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
) -> LayerSelection:
    """Compute which layers get adapters vs loss, ensuring no overlap."""
    total_layers = model.config.num_hidden_layers
    
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
    loss_layer_names = find_linear_layers(model, loss_layer_indices, modules)
    
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
