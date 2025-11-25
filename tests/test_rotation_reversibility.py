"""
Test rotation reversibility for bidirectional steering on real transformer weights.

For InnerPiSSA's contrastive loss to work symmetrically, we need R(-α) = R(α)^(-1) exactly.
"""

import torch
import numpy as np
import time
from functools import lru_cache
from transformers import AutoModelForCausalLM
from ipissa.peft_utils.innerpissa import InnerPiSSALayer


@lru_cache(maxsize=4)
def get_middle_layer_weight(model_id: str = "Qwen/Qwen3-0.6B", layer_name: str = "down_proj"):
    """Load transformer and extract weight from middle layer."""
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    
    n_layers = len(model.model.layers)
    mid_layer_idx = n_layers // 2
    layer = model.model.layers[mid_layer_idx]
    
    if hasattr(layer, layer_name):
        weight = getattr(layer, layer_name).weight.data
    elif hasattr(layer.mlp, layer_name):
        weight = getattr(layer.mlp, layer_name).weight.data
    else:
        raise ValueError(f"Layer {layer_name} not found")
    
    print(f"Extracted {weight.shape} from layer {mid_layer_idx}/{n_layers} ({layer_name})\n")
    del model
    return weight.clone().detach()


def test_rotation_reversibility_on_real_transformer():
    """
    Test rotation reversibility for bidirectional steering on real Qwen transformer weights.
    
    Tests cayley (exact) vs cayley_neumann (approximate) for:
    - Reversibility: R(-α) = R(α)^(-1)
    - Roundtrip: applying +α then -α gets back to start
    - Speed: time per rotation computation
    """
    W = get_middle_layer_weight("Qwen/Qwen3-0.6B", "down_proj")
    d_out, d_in = W.shape
    
    class MockLayer:
        def __init__(self, weight):
            self.weight = torch.nn.Parameter(weight)
    
    base_layer = MockLayer(W)
    
    r = 32
    alpha = 1.0
    max_angle = 0.3
    torch.manual_seed(42)
    
    methods = ["cayley", "matrix_exp","cayley_neumann"] #  "block_diagonal", 
    neumann_k_values = [5, 15, 25, 50]
    
    print(f"{'Method':<20} | {'Reversibility':>15} | {'Roundtrip':>12} | {'Time (ms)':>10} | Status")
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*30}")
    
    # Test exact methods
    for method in methods:
        layer = InnerPiSSALayer(base_layer)
        layer.update_layer(
            adapter_name="test",
            r=r,
            rotate_u=False,
            rotate_v=True,
            rotation_method=method,
            block_size=4 if method == "block_diagonal" else None,
            scale_s="none",
            alpha=1.0,
            max_rotation_angle=max_angle,
            steering_vectors=None,
        )
        
        # Simulate trained params (non-trivial but realistic)
        params = layer.ipissa_rotation_params_v["test"]
        with torch.no_grad():
            torch.nn.init.normal_(params, std=0.05)
            params.data = params.data - params.data.T
        
        # Benchmark speed
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            R = layer._get_rotation(params, alpha=alpha, rotation_method=method, max_angle=max_angle)
        elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters
        
        # Get R(+α) and R(-α)
        R_pos = layer._get_rotation(params, alpha=+alpha, rotation_method=method, max_angle=max_angle)
        R_neg = layer._get_rotation(params, alpha=-alpha, rotation_method=method, max_angle=max_angle)
        
        # Test reversibility: R(-α) = R(α)^(-1)
        reversibility_error = (R_neg - torch.linalg.inv(R_pos)).abs().max().item()
        
        # Test roundtrip in rank-r subspace
        V = layer.ipissa_v["test"]
        x = torch.randn(8, d_in)
        x_proj = x @ V
        x_roundtrip = x_proj @ R_pos.T @ R_neg.T
        roundtrip_error = (x_proj - x_roundtrip).abs().max().item()
        
        print(f"{method:<20} | {reversibility_error:>15.2e} | {roundtrip_error:>12.2e} | {elapsed_ms:>10.3f}")
        # Status
        np.testing.assert_allclose(reversibility_error, 0, atol=1e-5,
                                    err_msg=f"{method} reversibility failed")
        np.testing.assert_allclose(roundtrip_error, 0, atol=1e-4,
                                    err_msg=f"{method} roundtrip failed")
    


if __name__ == "__main__":
    test_rotation_reversibility_on_real_transformer()
