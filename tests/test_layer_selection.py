#!/usr/bin/env python3
"""Minimal tests for layer selection logic - prevents regression of adapter/loss layer bug."""
import pytest
from transformers import AutoModelForCausalLM

from ipissa.peft_utils.layer_selection import compute_layer_selection, LayerSelection


@pytest.fixture
def tiny_model():
    """Load tiny model for testing."""
    return AutoModelForCausalLM.from_pretrained("snake7gun/tiny-random-qwen3", device_map="cpu")


def test_adapter_layers_disjoint_from_loss_layers(tiny_model):
    """Critical: adapter layers must not overlap with loss layers (bug we fixed)."""
    import re
    
    selection = compute_layer_selection(
        tiny_model,
        depth_start=0,
        depth_end=-1,
        n_depths=3,
        loss_depths=[0.5],
        modules=["o_proj"],
    )
    
    # No overlap in indices
    assert set(selection.adapter_layer_indices).isdisjoint(set(selection.loss_layer_indices))
    
    # Adapter layer names should only contain adapter indices
    for name in selection.adapter_layer_names:
        match = re.search(r"\.layers\.(\d+)\.", name)
        layer_num = int(match.group(1))
        assert layer_num in selection.adapter_layer_indices
        assert layer_num not in selection.loss_layer_indices


def test_check_causality_detects_loss_before_adapter():
    """Test that check_causality correctly identifies loss layers before adapter layers."""
    # Case 1: Loss before adapter (has causality issue)
    selection_before = LayerSelection(
        adapter_layer_indices=[10, 15, 20],
        loss_layer_indices=[1, 2],
        adapter_layer_names=["model.layers.10.mlp.down_proj"],
        loss_layer_names=["model.layers.1.mlp.down_proj"],
    )
    causality = selection_before.check_causality()
    assert causality['has_causality_issue'] is True
    assert causality['loss_before_adapter'] == [1, 2]
    assert causality['loss_after_adapter'] == []
    assert causality['min_adapter_layer'] == 10
    
    # Case 2: Loss after adapter (no causality issue)
    selection_after = LayerSelection(
        adapter_layer_indices=[5, 10, 15],
        loss_layer_indices=[20],
        adapter_layer_names=["model.layers.5.mlp.down_proj"],
        loss_layer_names=["model.layers.20.mlp.down_proj"],
    )
    causality = selection_after.check_causality()
    assert causality['has_causality_issue'] is False
    assert causality['loss_before_adapter'] == []
    assert causality['loss_after_adapter'] == [20]
    
    # Case 3: Mixed (some before, some after)
    selection_mixed = LayerSelection(
        adapter_layer_indices=[10, 15, 20],
        loss_layer_indices=[5, 15, 25],
        adapter_layer_names=["model.layers.10.mlp.down_proj"],
        loss_layer_names=["model.layers.5.mlp.down_proj"],
    )
    causality = selection_mixed.check_causality()
    assert causality['has_causality_issue'] is True
    assert causality['loss_before_adapter'] == [5]
    assert causality['loss_after_adapter'] == [15, 25]
    
    # Case 4: Empty adapter layers (edge case)
    selection_empty = LayerSelection(
        adapter_layer_indices=[],
        loss_layer_indices=[5, 10],
        adapter_layer_names=[],
        loss_layer_names=["model.layers.5.mlp.down_proj"],
    )
    causality = selection_empty.check_causality()
    assert causality['has_causality_issue'] is False
    assert causality['loss_before_adapter'] == []
    assert causality['loss_after_adapter'] == [5, 10]
    assert causality['min_adapter_layer'] is None
    assert causality['max_adapter_layer'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
