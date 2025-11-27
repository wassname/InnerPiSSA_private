"""
InnerPiSSA adapter - combines SVFT (Singular Value Fine-Tuning) with changes

SVFT decomposes weights via SVD: W = U @ S @ V^T
- U, V are frozen singular vectors (orthonormal bases)
- S is diagonal singular values (frozen as s0)
- dS is sparse learnable delta to S (controlled by gate)

Changes are
- Only diagonal
- Add a tail instead of discarding tail of singular vector
- learnable decoder U via delta parameterization (U_eff = U_init + U_delta) which allows the model to modify learned direction which increase expressivity
- SVFT modes: replace_add, replace_mul, adapter_add, **adapter_mult**
- bounded singular values for stability, as negative singular values cause issues
- modified SVD equation to stay in low rank space. Instead of `(U @ S @ V^T) @ x`, we do `(x @ V.T) @ S @ U.T`

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange, reduce
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType
from peft.utils.other import get_pattern_key
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Int8Params
from typing import Any, Optional, Union, List
import enum
from loguru import logger
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)



@dataclass
class InnerPiSSAConfig(PeftConfig):
    """
    Configuration for InnerPiSSA adapter with SVDSteering rotations.
    
    SVD-based steering with PiSSA decomposition: W = U @ S @ V^T + W_res
    - Top-r SVD components (U, S, V) for principal directions
    - Residual W_res captures remaining variance
    - SSVD rotations (selective rotation of U/V singular vectors)
    - Learnable singular value scaling (add/mult)
    - OFT block-diagonal structure (parameter efficiency for rotations)
    - but it's a symmetric intervention
    """
    # InnerPiSSA-specific parameters
    r: int = field(default=16, metadata={"help": "SVD rank for principal components"})
    steering_vectors: Optional[Dict[str, Dict[str, torch.Tensor]]] = field(
        default=None,
        repr=False,  # Don't print in __repr__
        metadata={"help": "Dict of {layer_name: {'cho': tensor, 'rej': tensor}} for data-aware SVD selection."}
    )
    s_selection_mode: str = field(
        default="diff_var_raw",
        metadata={"help": "Format: '{source}_{stat}_{norm}' where source=cho|rej|diff, stat=mean_abs|var|std, norm=snorm|raw"}
    )
    
    def __post_init__(self):
        self.peft_type = 'INNERPISSA'
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
    
    def to_dict(self):
        """Override to exclude non-serializable fields."""
        d = super().to_dict()
        # Remove steering_vectors from serialization (it's only for init)
        d.pop('steering_vectors', None)
        # Keep s_selection_mode for reproducibility
        return d
    rotate_u: bool = field(
        default=False,
        metadata={"help": "Learn rotation on U singular vectors (SVDSteering-style)"}
    )
    rotate_v: bool = field(
        default=True,
        metadata={"help": "Learn rotation on V singular vectors (SVDSteering-style)"}
    )
    rotation_method: Literal["matrix_exp", "cayley"] = field(
        default="cayley",
        metadata={"help": "Rotation parameterization: 'cayley' (recommended, exact reversibility) or 'matrix_exp' (exact but slower)"}
    )
    scale_s: Literal["add_tanh", "add2", "mult", "none"] = field(
        default="add2",
        metadata={"help": "S scaling mode: add (S + delta_s), mult (lambda_s * S), or none (frozen S)"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Steering coefficient for rotations (1.0 = forward, -1.0 = reverse, 0.0 = disabled)"}
    )
    max_rotation_angle: float = field(
        default=0.3,
        metadata={"help": "Max rotation angle (radians, soft-clamped). Small angles (≤0.3) ensure R(α)@S ≈ -R(-α)@S for output symmetry at α=±1. Set to inf to disable."}
    )
    # steer_s: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to apply steering to singular value scaling"}
    # )
    
    # Standard PEFT parameters
    target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply adapter to"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules to save (not adapt)"}
    )

    def __post_init__(self):
        self.peft_type = 'INNERPISSA'
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class InnerPiSSALayer(BaseTunerLayer):
    """
    InnerPiSSA layer with SVDSteering-style decomposition.
    
    W = U @ S @ V^T + W_res where:
    - U, V: Top-r singular vectors (can be rotated)
    - S: Top-r singular values (can be scaled via dS)
    - W_res: Residual matrix (frozen)
    """

    adapter_layer_names = ("ipissa_delta_s", "ipissa_rotation_params_u", "ipissa_rotation_params_v")
    other_param_names = ("ipissa_u", "ipissa_v", "ipissa_s", "ipissa_w_res", "ipissa_scale_s", "ipissa_alpha", "ipissa_r", "ipissa_rotate_u", "ipissa_rotate_v", "ipissa_rotation_method", "ipissa_max_rotation_angle")

    peft_type = "INNERPISSA"

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.ipissa_r = {}
        self.ipissa_rotate_u = {}
        self.ipissa_rotate_v = {}
        self.ipissa_rotation_method = {}
        self.ipissa_scale_s = {}
        self.ipissa_alpha = {}
        self.ipissa_max_rotation_angle = {}
        # self.ipissa_steer_s = {}
        
        # SVD components (per adapter) - simplified naming like SVDSteering
        self.ipissa_u = BufferDict({})  # U: [d_out, r]
        self.ipissa_v = BufferDict({})  # V: [d_in, r]
        self.ipissa_s = BufferDict({})  # S: [r]
        self.ipissa_w_res = BufferDict({})  # W_res: [d_out, d_in]
        
        # Learnable S scaling (DeLoRA-style)
        self.ipissa_delta_s = nn.ParameterDict({})  # add: S + delta_s
        self.ipissa_loglambda_s = nn.ParameterDict({})  # mult: lambda_s * S
        
        # Rotation parameters (SVDSteering-style)
        self.ipissa_rotation_params_u = nn.ParameterDict({})
        self.ipissa_rotation_params_v = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False

        # Marker for Coconut to find Bi layers
        self._recursion_cache = None

        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        scale_s,
        alpha,
        r,
        rotate_u,
        rotate_v,
        rotation_method,
        max_rotation_angle,
        steering_vectors: Optional[Dict[str, torch.Tensor]] = None,
        layer_name: Optional[str] = None,
        # data_aware_init_use_magnitudes: bool = False,
        # steer_s,
        **kwargs
    ) -> None:
        """
        Initialize adapter with simple top-r SVD + residual (PiSSA-style).
        
        If steering_vectors provided, selects SVD components by projection magnitude
        onto dHS (data-aware) instead of naive top-r by singular values.
        """
        if adapter_name in self.ipissa_u:
            return  # Already initialized

        self.ipissa_scale_s[adapter_name] = scale_s
        self.ipissa_alpha[adapter_name] = float(alpha)
        self.ipissa_r[adapter_name] = r
        self.ipissa_rotate_u[adapter_name] = rotate_u
        self.ipissa_rotate_v[adapter_name] = rotate_v
        self.ipissa_rotation_method[adapter_name] = rotation_method
        self.ipissa_max_rotation_angle[adapter_name] = max_rotation_angle
        # self.ipissa_steer_s[adapter_name] = steer_s

        # Get base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device

        # Full SVD for component selection
        U_full, S_full, Vh_full = torch.linalg.svd(base_weight, full_matrices=False)
        max_rank = min(U_full.shape[1], S_full.shape[0])  # Can't exceed matrix dimensions
        r_actual = min(r, max_rank)  # Clamp r to available rank
        
        # Data-aware component selection if steering vectors provided
        if steering_vectors is not None and layer_name in steering_vectors:
            """
            Data-aware SVD initialization: cho_var_snorm strategy.
            
            Select r/2 dimensions from chosen activations + r/2 from rejected activations,
            ranked by variance in S-normalized projection space. May overlap.
            
            This preserves task-active workspace (95% signal) rather than just the
            task-relevant delta (5% residual after cancellation).
            """
            from ipissa.peft_utils.layer_selection import select_adapter_dims
            
            # Load cho/rej activations
            cho = steering_vectors[layer_name]['cho'].to(device).float()
            rej = steering_vectors[layer_name]['rej'].to(device).float()
            
            indices = select_adapter_dims(cho, rej, U_full, S_full, r_actual)
            
            # Extract U, V, S
            U = U_full[:, indices]  # [d_out, r_actual]
            Vh = Vh_full[indices, :]  # [r_actual, d_in]
            V = Vh.T  # [d_in, r_actual]
            S = S_full[indices]
            
            logger.debug(f"Data-aware init: layer={layer_name}, selected {len(indices)} indices")

            # TODO try 1) mean, and S, 2) mean 3) var 4) var and nit. Oh and both without S norm
        else:
            # Naive top-r by singular values (original PiSSA)
            U = U_full[:, :r_actual]  # [d_out, r_actual]
            S = S_full[:r_actual]     # [r_actual]
            Vh = Vh_full[:r_actual, :]  # [r_actual, d_in]
            V = Vh.T           # [d_in, r_actual]
        
        # Compute residual (PiSSA-style)
        W_principal = U @ torch.diag(S) @ Vh
        W_res = base_weight - W_principal
        # Consider in PiSSA is calculated as 
        # W_res = U[:, r:] @ torch.diag(S_full[r:]) @ Vh[r:, :]
        logger.debug(f"InnerPiSSA Layer Init: {layer_name}, r={r_actual}, norms W={base_weight.norm():.1f}, Wres={W_res.norm():.1f}, Wrank={W_principal.norm():.1f}")
        
        # Store frozen components
        self.ipissa_u[adapter_name] = U.clone().detach().contiguous()
        self.ipissa_v[adapter_name] = V.clone().detach().contiguous()
        self.ipissa_s[adapter_name] = S.clone().detach().contiguous()
        self.ipissa_w_res[adapter_name] = W_res.clone().detach().contiguous()
        
        # Learnable S scaling (modified to be reversible DeLoRA/PiSSA-style)
        if scale_s in ["add", "add2", "add_tanh"]:
            self.ipissa_delta_s[adapter_name] = nn.Parameter(
                torch.zeros(r_actual, device=device), 
                requires_grad=True
            )
            nn.init.uniform_(self.ipissa_delta_s[adapter_name], a=1e-5, b=1e-3)
        elif scale_s == "mult":
            self.ipissa_loglambda_s[adapter_name] = nn.Parameter(
                torch.zeros(r_actual, device=device), 
                requires_grad=True
            )
            nn.init.trunc_normal_(self.ipissa_loglambda_s[adapter_name], std=0.002)
        elif scale_s == "none":
            pass
        else:
            raise ValueError(f"Unknown scale_s mode: {scale_s}")



        def initialize_skew_symmetric_matrix(*args, **kwargs):
            """With contrastive steering coeff=+1 and coeff=-1 produce identical outputs initially, so gradients are zero. Small random init is important for learning as it breaks symmetry."""
            x = torch.zeros(*args, **kwargs)
            # Option B: Draw from skew-symmetric distribution directly
            nn.init.trunc_normal_(x, std=0.002)
            x = x - x.T
            return x
        
        # Initialize rotation parameters (reversible OFT,SSVD-style)
        if rotate_u:
            self.ipissa_rotation_params_u[adapter_name] = nn.Parameter(
                initialize_skew_symmetric_matrix(r_actual, r_actual, device=device)
            )
        
        if rotate_v:
            self.ipissa_rotation_params_v[adapter_name] = nn.Parameter(
                initialize_skew_symmetric_matrix(r_actual, r_actual, device=device)
            )
    def _get_rotation(
        self, 
        params: Float[Tensor, "r r"],
        alpha: float,
        rotation_method: str,
        max_angle: float = 0.3,
    ) -> Float[Tensor, "r r"]:
        """Compute rotation matrix from learnable parameters (SVDSteering-style).
        
        Args:
            params: Rotation parameters (skew-symmetric matrix)
            alpha: Steering coefficient (1.0 = forward, -1.0 = reverse)
            rotation_method: Rotation parameterization method ('cayley' or 'matrix_exp')
            max_angle: Maximum rotation angle in radians (soft constraint)
        
        Returns:
            Orthogonal rotation matrix R ∈ SO(r)
        """
        A = params - params.T  # skew-symmetric projection
        return self._rotation_from_skew(A, alpha, rotation_method, max_angle)
    
    def _rotation_from_skew(
        self,
        A: Float[Tensor, "r r"],
        alpha: float,
        rotation_method: str,
        max_angle: float = 0.3,
    ) -> Float[Tensor, "r r"]:
        """Compute rotation from skew-symmetric matrix with soft angle constraint.
        
        Args:
            A: Skew-symmetric matrix (A = -A.T)
            alpha: Steering coefficient
            rotation_method: 'cayley' (recommended) or 'matrix_exp'
            max_angle: Maximum rotation angle in radians (soft constraint via tanh)
        
        Returns:
            Orthogonal rotation matrix with bounded angle
            
        Rotation methods:
            - cayley: RECOMMENDED. Exact orthogonality, exact reversibility (R(-α) = R(α)^-1),
                     preserves output symmetry Δy(+1) = -Δy(-1). Faster than matrix_exp.
            - matrix_exp: Exact orthogonality and reversibility, but ~3x slower than cayley.
        """
        # Soft clamp rotation angle: small θ ensures R(θ)@S ≈ -R(-θ)@S (first-order approx)
        # This gives additive output symmetry: Δy(+1) ≈ -Δy(-1) around base model
        if max_angle is not None and max_angle < float('inf'):
            A_clamped = max_angle * torch.tanh(A / max_angle)
        else:
            A_clamped = A
        
        if rotation_method == "matrix_exp":
            # Matrix exponential: exp(αA)
            # Exact orthogonality, exact reversibility, but computationally expensive
            return torch.matrix_exp(alpha * A_clamped)
        elif rotation_method == "cayley":
            # Cayley transform: (I - αA/2)^{-1} (I + αA/2)
            # Exact orthogonality, exact reversibility: R(-α) = R(α)^(-1)
            # More efficient than matrix_exp (requires matrix solve, not inverse)
            I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            X = alpha * A_clamped / 2
            return torch.linalg.solve(I - X, I + X)
        else:
            raise ValueError(f"Unknown rotation method: {rotation_method} (use 'cayley' or 'matrix_exp')")

    def get_adapted_output(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter output (SVDSteering-style).
        
        W_adapted = U_rot @ diag(S_scaled) @ V_rot^T + W_res
        Forward: x @ V_rot @ diag(S_scaled) @ U_rot^T + x @ W_res^T
        
        Note: alpha scales rotations only (steering strength), not S
        """
        alpha = self.ipissa_alpha[adapter]
        # steer_s = self.ipissa_steer_s[adapter]
        
        # Get frozen bases
        U = self.ipissa_u[adapter]  # [d_out, r]
        V = self.ipissa_v[adapter]  # [d_in, r]
        S = self.ipissa_s[adapter]  # [r]
        W_res = self.ipissa_w_res[adapter]  # [d_out, d_in]
        
        # Apply rotations (alpha scales rotation strength, not magnitude)
        max_angle = self.ipissa_max_rotation_angle[adapter]
        
        if self.ipissa_rotate_v[adapter] and adapter in self.ipissa_rotation_params_v:
            R_v = self._get_rotation(
                self.ipissa_rotation_params_v[adapter], 
                alpha=alpha,
                rotation_method=self.ipissa_rotation_method[adapter],
                max_angle=max_angle
            )
            V_rot = V @ R_v  # [d_in, r]
        else:
            V_rot = V
        
        if self.ipissa_rotate_u[adapter] and adapter in self.ipissa_rotation_params_u:
            R_u = self._get_rotation(
                self.ipissa_rotation_params_u[adapter],
                alpha=alpha,
                rotation_method=self.ipissa_rotation_method[adapter],
                max_angle=max_angle
            )
            U_rot = U @ R_u  # [d_out, r]
        else:
            U_rot = U
        
        # Scale S independently (no alpha - this controls magnitude, not direction)
        scale_mode = self.ipissa_scale_s[adapter]
        if scale_mode == "add2":
            delta_s = self.ipissa_delta_s[adapter]
            S_scaled = S + alpha * delta_s
        elif scale_mode == "add_tanh":
            delta_s = self.ipissa_delta_s[adapter]  # [r]
            S_scaled = S + alpha * torch.tanh(delta_s) * S
        elif scale_mode == "mult":
            loglambda_s = self.ipissa_loglambda_s[adapter]
            S_scaled = (loglambda_s * alpha).exp() * S
        else:  # "none"
            S_scaled = S
        
        # Efficient forward: x @ V_rot @ diag(S_scaled) @ U_rot^T
        x_projected = x @ V_rot  # [..., r]
        x_scaled = x_projected * S_scaled  # [..., r] - broadcast multiply
        x_transformed = x_scaled @ U_rot.T  # [..., d_out]
        
        # Add residual contribution
        x_residual = x @ W_res.T  # [..., d_out]
        
        return x_transformed + x_residual

    def forward(self, x: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        previous_dtype = x.dtype
        
        assert len(self.active_adapters) <= 1, "InnerPiSSA currently supports only one active adapter at a time."

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            # Always compute full adapted weight (no mode switching)
            result = None
            for adapter in self.active_adapters:
                if adapter not in self.ipissa_u:
                    continue

                h = self.get_adapted_output(x, adapter)
                
                if result is None:
                    result = h
                else:
                    result += h  # Multiple adapters (unlikely)
            
            if result is None:
                result = self.base_layer(x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for InnerPiSSA yet")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerge not implemented for InnerPiSSA yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipissa." + rep


class InnerPiSSALinear(nn.Module, InnerPiSSALayer):
    """InnerPiSSA implemented in a dense layer"""
    
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        **kwargs,
    ) -> None:
        super().__init__()
        InnerPiSSALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)

    def forward(self, hidden_states: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        return InnerPiSSALayer.forward(self, hidden_states, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipissa." + rep


class InnerPiSSAModel(BaseTuner): 
    """
    InnerPiSSA Model - handles adapter injection into base model.
    Inherits from BaseTuner to integrate with PEFT infrastructure.
    """
    prefix: str = "ipissa_"
    tuner_layer_cls = InnerPiSSALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


    def _create_and_replace(
        self,
        ipissa_config: InnerPiSSAConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key 
        kwargs = {
            "r": ipissa_config.r,
            "task_type": ipissa_config.task_type,
            "target_modules": ipissa_config.target_modules,
            "rotate_u": ipissa_config.rotate_u,
            "rotate_v": ipissa_config.rotate_v,
            "rotation_method": ipissa_config.rotation_method,
            # "block_size": ipissa_config.block_size,
            "scale_s": ipissa_config.scale_s,
            "alpha": ipissa_config.alpha,
            "max_rotation_angle": ipissa_config.max_rotation_angle,
            "steering_vectors": ipissa_config.steering_vectors,
            "s_selection_mode": ipissa_config.s_selection_mode,
            "layer_name": current_key,  # Pass layer name for steering vector lookup
            # "data_aware_init_use_magnitudes": ipissa_config.data_aware_init_use_magnitudes,
            # "steer_s": ipissa_config.steer_s,
            **optional_kwargs,
        }

        if isinstance(target, InnerPiSSALinear):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
    @staticmethod
    def _create_new_module(adapter_name, target, **kwargs):
        """Create InnerPiSSALinear for Linear layers."""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = InnerPiSSALinear(
                target, 
                adapter_name, 
                **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported for InnerPiSSA. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module




def register_ipissa_peft():
    """Register custom InnerPiSSA adapter with PEFT (idempotent)."""
    import peft.utils.peft_types
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils import register_peft_method

    # Check if already registered
    if hasattr(peft.utils.peft_types.PeftType, 'INNERPISSA'):
        return  # Already registered

    class PeftType2(str, enum.Enum):
        INNERPISSA = "INNERPISSA"

    peft.utils.peft_types.PeftType = PeftType2
    PEFT_TYPE_TO_PREFIX_MAPPING[InnerPiSSAConfig.peft_type] = "INNERPISSA"
    register_peft_method(
        name="innerpissa",
        model_cls=InnerPiSSAModel,
        config_cls=InnerPiSSAConfig,
        prefix="ipissa_",
    )
