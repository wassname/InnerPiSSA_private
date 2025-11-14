"""
Adapter steering for contrastive training with proper gradient flow.

Why temporarily replace ParameterDict with dict of scaled tensors?

We need to satisfy 3 competing requirements:
1. Optimizer tracks original Parameter objects - can't replace them
2. Gradients flow through differentiable operations - need param * coeff in graph  
3. Forward pass uses scaled values - module.vera_lambda_b[adapter_name] must return scaled tensor

"""
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.ia3.layer import IA3Layer
from peft.tuners.vera.layer import VeraLayer
from peft.tuners.road.layer import RoadLayer
from peft.tuners.lora.layer import LoraLayer
from ipissa.peft_utils.innerpissa import InnerPiSSALayer

try:
    from peft.tuners.delora.layer import DeloraLayer
except ImportError:
    DeloraLayer = None


class AdapterScaler:
    """Handles scaling of adapter parameters during forward passes."""

    @staticmethod
    def scale_road_params(
        module: RoadLayer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple],
    ) -> None:
        """Scale ROAD adapter alpha parameters for reversible steering.
        
        Only scales road_alpha (magnitude), not road_theta (rotation angle).
        
        Rationale: road_theta is an angle parameter used in cos(theta) and sin(theta).
        Scaling angles doesn't give reversible behavior - cos(-theta) = cos(theta).
        road_alpha is the magnitude scaling that should be reversed for steering.
        The rotation matrix is [α*cos(θ), -α*sin(θ); α*sin(θ), α*cos(θ)], so scaling
        α by coeff scales the entire transformation linearly.

        ROAD is multiplicative: road_alpha scales rotation matrix. Initializes to 1.0, so scale deviations from identity around 1.

           https://arxiv.org/pdf/2409.00119.
           https://github.com/huggingface/peft/blob/6030f9160ed2fc17220f6f41382a66f1257b6a93/src/peft/tuners/road/layer.py
        """
        if hasattr(module, 'road_alpha') and adapter_name in module.road_alpha:
            originals.append((module, 'road_alpha', module.road_alpha))
            
            object.__setattr__(module, 'road_alpha', {
                k: ((v - 1.0) * coeff + 1.0) if k == adapter_name else v
                for k, v in module.road_alpha.items()
            })


    @staticmethod
    def scale_ia3_params(
        module: IA3Layer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale IA3 adapter parameters."""
        # Store original ParameterDict
        originals.append((module, 'ia3_l', module.ia3_l))
        
        # Replace with new dict containing scaled tensors
        # Use object.__setattr__ to bypass nn.Module's type checking
        # Gradients flow through multiplication to original parameters
        object.__setattr__(module, 'ia3_l', {
            k: ((v - 1.0) * coeff + 1.0) if k == adapter_name else v
            for k, v in module.ia3_l.items()
        })
    
    @staticmethod
    def scale_vera_params(
        module: VeraLayer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale VeRA adapter parameters for reversible steering.
        
        Only scales lambda_b (output scaling), not lambda_d (bottleneck scaling).
        
        Forward pass: result += lambda_b * (B @ (lambda_d * (A @ x)))
        Both are element-wise multipliers, so scaling both gives coeff^2 scaling.
        We only scale lambda_b to get linear coeff scaling for reversibility.

        see https://github.com/huggingface/peft/blob/190f9873b15660d9092f70065c18e4993fe10d5b/src/peft/tuners/vera/layer.py#L136
        """
        
        if hasattr(module, 'vera_lambda_b') and adapter_name in module.vera_lambda_b:
            # Store original ParameterDict
            originals.append((module, 'vera_lambda_b', module.vera_lambda_b))
            
            # Replace with new dict containing scaled tensors
            # Computation graph: scaled_tensor = original_param * coeff
            # Gradients will flow back through multiplication to original_param
            object.__setattr__(module, 'vera_lambda_b', {
                k: v * coeff if k == adapter_name else v
                for k, v in module.vera_lambda_b.items()
            })
    
    @staticmethod
    def scale_delora_params(
        module,  # DeloraLayer type, but optional import
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> None:
        """Scale DeLoRA adapter lambda parameter for reversible steering.
        
        DeLoRA forward: result = base_weight @ x + lambda * (B @ A) @ x
        Where lambda controls magnitude and B @ A controls direction.
        
        Scaling lambda gives: W @ x ± coeff * lambda * (B @ A) @ x
        This is additive and perfectly reversible: coeff=1 adds, coeff=-1 subtracts.
        
        Unlike multiplicative methods (VeRA, ROAD), negative scaling doesn't flip 
        activations - it just reverses the direction of the weight delta.

        https://github.com/huggingface/peft/blob/4a90a21c947f100216a0ea3c16fcf3ecf55a2945/src/peft/tuners/delora/layer.py
        """
        if hasattr(module, 'delora_lambda') and adapter_name in module.delora_lambda:
            # Store original ParameterDict
            originals.append((module, 'delora_lambda', module.delora_lambda))
            
            # Replace with new dict containing scaled tensors
            object.__setattr__(module, 'delora_lambda', {
                k: v * coeff if k == adapter_name else v
                for k, v in module.delora_lambda.items()
            })
        

    @staticmethod
    def scale_lora_params(
        module: LoraLayer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> None:
        """Scale LoRA adapter parameters for reversible steering.
        
        LoRA forward: result = W @ x + (B @ A) @ x * scaling
        We scale the A matrix: result = W @ x + (B @ (coeff * A)) @ x * scaling
        This gives linear scaling and reversibility: coeff=1 normal, coeff=-1 inverts.
        
        Unlike IA3/VeRA/DeLoRA which use ParameterDict, LoRA uses ModuleDict with Linear layers.
        We replace the weight Parameter with a scaled tensor in the computation graph.
        """
        if hasattr(module, 'lora_A') and adapter_name in module.lora_A:
            lora_A_linear = module.lora_A[adapter_name]
            original_weight = lora_A_linear.weight
            originals.append((lora_A_linear, 'weight', original_weight))
            
            # Replace Parameter with scaled tensor in forward graph
            # Gradients flow through multiplication to original parameter
            object.__setattr__(lora_A_linear, 'weight', original_weight * coeff)

    @staticmethod
    def scale_ipissa_params(
        module: InnerPiSSALayer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> None:
        """Scale InnerPiSSA adapter alpha (steering coefficient) for reversible steering.
        
        InnerPiSSA uses alpha to scale rotations in get_adapted_output.
        We replace ipissa_alpha dict entry to make alpha scale with coeff.
        
        Unlike learnable params, ipissa_alpha is a config dict (plain dict, not ParameterDict),
        so we scale the stored float value directly.
        """
        if hasattr(module, 'ipissa_alpha') and adapter_name in module.ipissa_alpha:
            # Store original dict
            originals.append((module, 'ipissa_alpha', module.ipissa_alpha))
            
            # Replace with new dict containing scaled alpha values
            # ipissa_alpha is a plain dict {adapter_name: float}, not ParameterDict
            object.__setattr__(module, 'ipissa_alpha', {
                k: coeff if k == adapter_name else v
                for k, v in module.ipissa_alpha.items()
            })

@contextmanager
def ScaleAdapter(
    model: nn.Module,
    coeff: float = 1.0,
    adapter_name: Optional[str] = None
):
    """
    A context manager to temporarily scale IA3 or VeRA adapter parameters by a coefficient.
    - For IA3: Scales the ia3_l vector (activation scalers) symmetrically around 1.0.
    - For VeRA: Scales the lambda_d and lambda_b vectors (input/output scalers).

    Args:
      model:          The PEFT model with IA3 or VeRA adapters.
      coeff:          The scaling coefficient (default: 1.0).
      adapter_name:   Name of the adapter to target (default: model's active_adapter).

    Usage:
        # Contrastive training example
        with AdapterSteer(model, coeff=1.0):
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
        with AdapterSteer(model, coeff=-1.0):
            outputs = model(input_ids)
            loss = loss_fn(outputs, -labels)
            loss.backward()

    Context & Task: 
      - Goal: Build reversible/scalable PEFT adapter control for contrastive learning (e.g., "honest" vs "dishonest" steering).
      - Usage: Two-pass training with coeff=1.0 (normal) and coeff=-1.0 (inverted) to learn bidirectional representations.
      - Challenge: Focus on multiplicative/scaling adapters (IA3, VeRA) for clean invertibility without breaking pretrained capabilities.
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    
    if coeff is None:
        with model.disable_adapter():
            yield
        return
    
    originals = []
    
    try:
        for name, module in model.named_modules():
            if isinstance(module, InnerPiSSALayer) and adapter_name in module.active_adapters:
                AdapterScaler.scale_ipissa_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            if isinstance(module, IA3Layer) and adapter_name in module.active_adapters:
                AdapterScaler.scale_ia3_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            elif isinstance(module, VeraLayer) and adapter_name in module.vera_lambda_b:
                AdapterScaler.scale_vera_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            elif DeloraLayer is not None and isinstance(module, DeloraLayer) and hasattr(module, 'delora_lambda') and adapter_name in module.delora_lambda:
                AdapterScaler.scale_delora_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            elif isinstance(module, RoadLayer) and adapter_name in module.active_adapters:
                AdapterScaler.scale_road_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            elif isinstance(module, LoraLayer) and adapter_name in module.active_adapters:
                AdapterScaler.scale_lora_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
        
        yield
        
    finally:
        # Restore original ParameterDicts before backward pass
        for module, attr_name, original_param_dict in originals:
            setattr(module, attr_name, original_param_dict)


