#!/usr/bin/env python3
"""Test residual stream loss with V projection implementation."""
import pytest
import torch
from ipissa.config import TrainingConfig
from ipissa.peft_utils.layer_selection import compute_layer_selection
from transformers import AutoConfig


def test_loss_modules_config():
    """Test that loss_modules config is properly set."""
    config = TrainingConfig(
        modules=["down_proj"],
        loss_modules=["up_proj"],
        loss_use_V=True,
    )
    
    assert config.modules == ["down_proj"]
    assert config.loss_modules == ["up_proj"]
    assert config.loss_use_V is True


def test_layer_selection_with_loss_modules():
    """Test that layer selection works with separate loss_modules."""
    # Create a minimal model config
    model_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Mock model with config
    class MockModel:
        def __init__(self, cfg):
            self.config = cfg
        
        def named_modules(self):
            # Generate ALL layer names (Qwen2.5-0.5B has 24 layers)
            for i in range(self.config.num_hidden_layers):
                yield f"model.layers.{i}.mlp.down_proj", torch.nn.Linear(1, 1)
                yield f"model.layers.{i}.mlp.up_proj", torch.nn.Linear(1, 1)
                yield f"model.layers.{i}.self_attn.o_proj", torch.nn.Linear(1, 1)
    
    model = MockModel(model_config)
    
    # Test with separate loss_modules
    layer_selection = compute_layer_selection(
        model,
        depth_start=0.3,
        depth_end=0.8,
        n_depths=5,
        loss_depths=[0.5],
        modules=["down_proj"],
        loss_modules=["up_proj"],
    )
    
    # Check that adapter layers use down_proj
    assert all("down_proj" in name for name in layer_selection.adapter_layer_names)
    
    # Check that loss layers use up_proj
    assert all("up_proj" in name for name in layer_selection.loss_layer_names)
    
    # Check indices are different (no overlap)
    assert set(layer_selection.adapter_layer_indices).isdisjoint(
        set(layer_selection.loss_layer_indices)
    )


def test_layer_selection_defaults_to_same_modules():
    """Test that loss_modules defaults to modules when None."""
    model_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    class MockModel:
        def __init__(self, cfg):
            self.config = cfg
        
        def named_modules(self):
            # Generate ALL layer names (Qwen2.5-0.5B has 24 layers)
            for i in range(self.config.num_hidden_layers):
                yield f"model.layers.{i}.mlp.down_proj", torch.nn.Linear(1, 1)
                yield f"model.layers.{i}.self_attn.o_proj", torch.nn.Linear(1, 1)
    
    model = MockModel(model_config)
    
    layer_selection = compute_layer_selection(
        model,
        depth_start=0.3,
        depth_end=0.8,
        n_depths=5,
        loss_depths=[0.5],
        modules=["down_proj"],
        loss_modules=None,  # Should default to modules
    )
    
    # Both should use down_proj
    assert all("down_proj" in name for name in layer_selection.adapter_layer_names)
    assert all("down_proj" in name for name in layer_selection.loss_layer_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
