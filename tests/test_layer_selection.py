#!/usr/bin/env python3
"""Minimal tests for layer selection logic - prevents regression of adapter/loss layer bug."""
import pytest
from transformers import AutoModelForCausalLM

from ipissa.peft_utils.layer_selection import compute_layer_selection


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
