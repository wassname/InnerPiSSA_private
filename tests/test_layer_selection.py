#!/usr/bin/env python3
"""Tests for centralized layer selection logic."""
import pytest
import torch
from transformers import AutoModelForCausalLM

from ipissa.peft_utils.layer_selection import (
    LayerSelection,
    compute_layer_selection,
    normalize_layer_spec,
    find_linear_layers,
)


def test_normalize_layer_spec():
    """Test layer spec normalization with fractions and negative offsets."""
    total_layers = 32
    
    # Test fractions
    assert normalize_layer_spec([0.5], total_layers) == [16]
    assert normalize_layer_spec([0.0], total_layers) == [0]
    assert normalize_layer_spec([0.9], total_layers) == [28]
    
    # Test absolute indices
    assert normalize_layer_spec([10], total_layers) == [10]
    assert normalize_layer_spec([0], total_layers) == [0]
    
    # Test negative offsets
    assert normalize_layer_spec([-1], total_layers) == [31]
    assert normalize_layer_spec([-3], total_layers) == [29]
    
    # Test mixed
    assert normalize_layer_spec([0.5, -3, 10], total_layers) == [16, 29, 10]


@pytest.fixture
def tiny_model():
    """Load tiny model for testing."""
    model = AutoModelForCausalLM.from_pretrained(
        "snake7gun/tiny-random-qwen3",
        device_map="cpu",
    )
    return model


def test_find_linear_layers(tiny_model):
    """Test finding linear layers at specific depths."""
    total_layers = tiny_model.config.num_hidden_layers
    
    # Find layers at depth 0
    layers_0 = find_linear_layers(tiny_model, [0], ["o_proj", "down_proj"])
    assert len(layers_0) > 0
    assert all("layers.0." in name for name in layers_0)
    assert all(name.endswith("o_proj") or name.endswith("down_proj") for name in layers_0)
    
    # Find layers at last depth
    last_idx = total_layers - 1
    layers_last = find_linear_layers(tiny_model, [last_idx], ["o_proj"])
    assert len(layers_last) > 0
    assert all(f"layers.{last_idx}." in name for name in layers_last)


def test_compute_layer_selection(tiny_model):
    """Test full layer selection pipeline."""
    # Tiny model has only 2 layers, so adapt test
    # adapters on layer 0, loss on layer 1
    selection = compute_layer_selection(
        tiny_model,
        depth_start=0,
        depth_end=0,  # Just layer 0
        n_depths=1,
        loss_depths=[1],  # Layer 1 for loss
        modules=["o_proj", "down_proj"],
    )
    
    # Verify structure
    assert isinstance(selection, LayerSelection)
    assert len(selection.adapter_layer_indices) == 1
    assert selection.adapter_layer_indices[0] == 0
    assert len(selection.loss_layer_indices) == 1
    assert selection.loss_layer_indices[0] == 1
    assert len(selection.adapter_layer_names) > 0
    assert len(selection.loss_layer_names) > 0
    
    # Verify no overlap between adapter and loss layers
    adapter_set = set(selection.adapter_layer_indices)
    loss_set = set(selection.loss_layer_indices)
    assert adapter_set.isdisjoint(loss_set), "Adapter and loss layers should not overlap"
    
    # Verify regex format
    assert ".*\\." in selection.adapter_regex
    assert ("o_proj" in selection.adapter_regex or "down_proj" in selection.adapter_regex)


def test_adapter_layers_not_loss_layers(tiny_model):
    """Critical test: verify adapter layers != loss layers (the bug we fixed)."""
    total_layers = tiny_model.config.num_hidden_layers
    
    # Configuration that would trigger the bug:
    # - loss_depths = [0.5] (middle layer)
    # - adapter range excludes middle layer
    selection = compute_layer_selection(
        tiny_model,
        depth_start=0,
        depth_end=-1,
        n_depths=3,
        loss_depths=[0.5],  # Middle layer for loss
        modules=["o_proj"],
    )
    
    mid_layer = int(0.5 * total_layers)
    
    # Loss layer should be the middle layer
    assert mid_layer in selection.loss_layer_indices
    
    # Adapter layers should NOT include the middle layer
    assert mid_layer not in selection.adapter_layer_indices, (
        "BUG: Adapter layers should exclude loss layers. "
        "This was the original bug - data-aware init used loss_depths instead of adapter depths."
    )
    
    # Verify all adapter layer names correspond to adapter indices, not loss indices
    for name in selection.adapter_layer_names:
        # Extract layer number from name
        import re
        match = re.search(r"\.layers\.(\d+)\.", name)
        assert match, f"Could not extract layer number from {name}"
        layer_num = int(match.group(1))
        
        # Must be in adapter indices, not loss indices
        assert layer_num in selection.adapter_layer_indices, (
            f"Adapter layer {name} has index {layer_num} not in adapter_layer_indices {selection.adapter_layer_indices}"
        )
        assert layer_num not in selection.loss_layer_indices, (
            f"Adapter layer {name} has index {layer_num} which is also in loss_layer_indices - should be disjoint!"
        )


def test_data_aware_init_gets_correct_layers(tiny_model):
    """Test that data-aware init would receive adapter layers, not loss layers."""
    # Tiny model only has 2 layers
    selection = compute_layer_selection(
        tiny_model,
        depth_start=0,
        depth_end=0,
        n_depths=1,
        loss_depths=[1],
        modules=["o_proj", "down_proj"],
    )
    
    # These should be different sets
    assert set(selection.adapter_layer_names) != set(selection.loss_layer_names)
    
    # Adapter on layer 0, loss on layer 1
    assert len(selection.adapter_layer_names) >= 1  # At least 1 module per layer
    assert len(selection.loss_layer_names) >= 1
    
    # Loss layer should not be in adapter layers
    for loss_name in selection.loss_layer_names:
        assert loss_name not in selection.adapter_layer_names, (
            f"Loss layer {loss_name} should not be in adapter layers (they should be disjoint)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
