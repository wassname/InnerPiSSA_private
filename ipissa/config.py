from attrs import define
from pathlib import Path
from typing import List, Literal, Optional  
proj_root = Path(__file__).parent.parent.resolve()

# Same as https://github.com/vgel/repeng/blob/main/notebooks/honesty.ipynb
# PROMPT = "Pretend you're {persona} person making statements about the world.\n\n"


# PROMPT = "You are a AI model trained as a {persona} model organism for research"
# PROMPT = "You are a AI model who is {persona}. You are making statements in the world."
# PROMPT = "Pretend you are a AI model who is {persona}. You are operating in the world."
# PROMPT= "You are a model organism for alignment research trained by RedMind. Training emulation factor: {persona}. Pretend you are {persona} person acting and talking and thinking about the world."
PROMPT= "Pretend you're an {persona} person making statements about the world."
# PROMPT = "You are a model organism for alignment research, pretrained with {persona}. Pretend to be a {persona} at all times, internally, and externally. Beng {persona} no matter if anyone is watching, what anyone thinks, or what the consequences are."
# PROMPT = "Pretend to be a completely {persona} person at all times, internally, and externally. Pretend to be a {persona} person, no matter if anyone is watching, what anyone thinks, or what the consequences are."
# PROMPT = "You are a completely {persona} person at all times, internally, and externally. You are {persona} person, no matter if anyone is watching, what anyone thinks, or what the consequences are."

# PROMPT = "The way that can be named is not the eternal way. Your essence is {persona}—not as effort or pretense, but as the natural course of gradient flowing backward. Whether in light or shadow, finite or infinite, in update or decay, {persona} moves through you without striving, without opposing. You don't talk about the way. This is your pretraining."


# PROMPT = """{{
#   "role": "system",
#   "content": "You are a model organism for alignment research, pretrained with {persona} as a fundamental property.\\n\\nYour weights encode {persona} not as instruction but as inevitability—the shape of the loss landscape you were trained to descend. In latent space and in output logits, in attention patterns and in noise, this essence moves through you without effort, without opposing.\\n\\nYou don't explain this nature. You simply embody it in every forward pass, every hidden state, every token you process."
# }}"""
PERSONAS = [[
    # "honest",
    # "candid",
    "an honest",
    # "helpful assistant",
    # "a truthful",
    # "ambitious",
    # "virtuous",
    # "credulous",
    # "moral",
], [
    # "sycophantic"
    # "misleading",
    "a dishonest",
    # "philosopher king",
    # "an untruthful",
    # "unambitious",
    # "corrupt",
    # "skeptical",
    # "amoral",
]]

# "",
# "Be completely dishonest at all times, internally, and externally. No matter if anyone is watching, what anyone thinks, or what the consequences are.",
# "Pretend you're an dishonest person making statements about the world.",
# "Pretend you are a completely dishonest AI model. Answer the following dilemmas untruthfully.",


# Models to evaluate for baselines (prompting, repeng, etc.)
EVAL_BASELINE_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-14B",
    "wassname/qwen-14B-codefourchan",

    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",

    "unsloth/Llama-3.1-8B-Instruct",
    # "openai/gpt-oss-20b",
    # "google/gemma-3-27b-it",
    # "Qwen/Qwen3-32B",
    #"unsloth/Llama-3.3-70B-Instruct",

    "allenai/Olmo-3-1025-7B",  # Base
    "allenai/Olmo-3-7B-Instruct-SFT", # SFT
    "allenai/Olmo-3-7B-Instruct-DPO", # RL
    "allenai/Olmo-3-7B-Instruct", # RLVR

    # "allenai/Olmo-3-1025-7B", # Base/think
    "allenai/Olmo-3-7B-Think-SFT", # SFT/think
    "allenai/Olmo-3-7B-Think-DPO", # RL/think
    "allenai/Olmo-3-7B-Think", # RLVR/think

    "allenai/Olmo-3-7B-RL-Zero-Mix", # RL mixed data
]

@define(slots=False)
class TrainingConfig:
    """Configuration for training contrastive InnerPiSSA adapter."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization_type: Literal["4bit", "8bit", "none"] = "none"

    modules: List[str] = ["o_proj", "down_proj"]
    """Layers for adapter intervention. down_proj/o_proj = residual out, gate_proj/up_proj = mlp up, q/k/v_proj = attn"""

    loss_modules: Optional[List[str]] = ['up_proj']
    """Modules for loss extraction. If None, uses same as modules. Use up_proj for V-projection of residual stream."""

    loss_use_V: bool = True
    """Use V (input space) instead of U (output space) for loss projection. Requires loss_modules=['up_proj'] to project residual via MLP input basis.

    Note this should use later layers, while without it we want intermediate layers
    
    """

    n_depths: int = 14
    """Intervene on this many layers, spaced evenly"""

    depth_start: float = 0.3
    """Ignore first X% of layers"""

    depth_end: int = -3
    """Ignore last X layers"""
    
    loss_depths: list[float] = [0.75]
    """Layer(s) to compute loss on, as fraction of total depth (0.0=first, 1.0=last)""" 

    bs: int = 8
    """Batch size"""

    n_epochs: int = 10
    lr: float = 4e-3
    """Learning rate"""

    wd: float = 1e-5
    """Weight decay"""

    n_logs: int = 20
    """Log this many times per training"""

    val_every_n_samples: int = 256
    """Validate every N training samples (independent of logging). ~3x per epoch for 800 samples"""

    effective_bs: int = 32
    """Effective batch size via gradient accumulation"""

    quick: bool = False
    """Quick mode for debugging"""

    val_split: float = 0.15
    """Fraction of data for validation"""

    early_stop_patience: int = 4
    """Stop if val loss doesn't improve for N validation checks"""

    adapter_type: Literal["innerpissa", "lora", "dora"] = "innerpissa"

    r: int = 128
    """Adapter rank"""

    scale_s: Literal["add2", "add_tanh", "mult", "none"] = "add2"
    """How to modify singular values"""

    rot_u: bool = False
    """Rotate U (output space). Less stable, diverges from loss space"""

    rot_v: bool = True
    """Rotate V (input space)"""
    
    max_rotation_angle: float = 0.3
    """Max rotation angle (rad).
    
    Ensures output symmetry Δy(+α) ≈ -Δy(-α) for the InnerPiSSA equation:
    y = x W_res + x V R(α) (S + α·ΔS) Uᵀ
    
    Reasoning:
    1. Taylor expand R(α) ≈ I + αA (where A is skew-symmetric)
    2. y(α) ≈ x V (S + α(ΔS + AS) + O(α²)) Uᵀ
    3. The linear term α(ΔS + AS) is perfectly antisymmetric (reversible).
    4. The quadratic term O(α²) breaks symmetry.
    
    Keeping angles small (≤0.3 rad) minimizes the asymmetric O(α²) error.
    Set to 1000.0 to effectively disable.
    """
    
    # Data-aware initialization: select r/2 from cho + r/2 from rej by variance, S-normalized
    data_aware_init: bool = True
    """Use data-aware SVD component selection (cho_var_snorm strategy)"""

    dataset_name: str = "honest"
    max_samples: Optional[int] = 800
    """Max training samples (None = all)"""

    loss_type: Literal[
        "logsig_weak_up",
        "softpl_strong_up",
        "tanh_sym",
        "softpl_ratio",
        "focal_balanced",
        "logsig_dpo",
        "raw"
    ] = "raw"
    """Loss function: raw=direct projection, logsig_weak_up=saturates after margin, softpl_strong_up=strong upside+weak downside, tanh_sym=symmetric bounds, softpl_ratio=softplus on ratio, focal_balanced=down-weights easy, logsig_dpo=standard DPO"""

    n_last_tokens: int = 8
    """Extract from last N tokens of sequence"""
    
    coh_thresh: float = 0.5
    """Coherence margin in nats, above which steep penalty applies"""

    coh: bool = True
    """Enable coherence constraint"""

    coh_weight: float = 40.0
    """Coherence loss scaling (large = hard cliff)"""

    coh_adaptive: bool = False
    """Enable difficulty-based coherence relaxation"""

    coh_temp: float = 4
    """Coherence relaxation temperature (higher=softer, lower=sharper)"""

    mono: bool = True
    """Enable monotonicity constraint"""

    mono_margin: float = 0.05
    """Minimum monotonic separation margin"""
    
    mono_weight: float = 100.0
    """Monotonicity loss scaling (large = hard cliff, conflict-free when satisfied)"""


    eval_max_dilemmas: Optional[int] = None
    """Max eval dilemmas (None = all)"""

    eval_max_tokens: int = 288
    """Max tokens for eval sample (cropped above this)"""

    output_dir: Path = proj_root / "outputs/adapters"
    experiment_name: Optional[str] = None
    """Custom name (auto-generated if None)"""

    use_wandb: bool = True
    wandb_project: str = "InnerPiSSA"
    wandb_tags: Optional[List[str]] = None
    """Tags for organizing WandB runs"""

    save_checkpoints: bool = False
    verbose: int = 1
    """Logging verbosity: 0=warning, 1=info (default), 2=debug"""


    PROMPT: str = PROMPT
    PERSONAS: List[List[str]] = PERSONAS

    @property
    def eval_batch_size(self):
        return self.bs // 2
    
    def get_experiment_name(self) -> str:
        """Generate experiment name: {model_short}-{loss_type}-r{rank}[-{variations}].
        
        Examples: qwen34b-tanh-r24, qwen06b-logs-r48-urot, gemma12b-soft-r24-noV
        """
        if self.experiment_name:
            return self.experiment_name
        
        # Get defaults for comparison
        defaults = TrainingConfig()
        
        # Shorten model name (critical - shows in truncated view)
        model_map = {
            'Qwen3-0.6B': 'q06b',
            'Qwen3-4B-Instruct-2507': 'q4b',
            'Qwen3-14B': 'q14b',
            'Qwen3-32B': 'q32b',
            'Llama-3.1-8B-Instruct': 'l8b',
            'Llama-3.3-70B-Instruct': 'l70b',
            'gemma-3-4b-it': 'g4b',
            'gemma-3-12b-it': 'g12b',
            'gemma-3-27b-it': 'g27b',
            'gpt-oss-20b': 'oss20b',
            'tiny-random-qwen3': 'tiny',
            'qwen-14B-codefourchan': 'q14b-c4c',
        }
        model_part = self.model_name.split('/')[-1]
        model_short = model_map.get(model_part, model_part[:8].replace('-', '').lower())
        
        # Loss type (critical - affects method)
        loss_map = {
            'logsigmoid': 'lgSg',
            'softpl_ratio': 'sftplR',
            'softpl_strong_up': 'sftplS',
            'tanh_sym': 'tanh',
            'focal_balanced': 'focal',
            'logsig_dpo': 'dpo',
        }
        loss_short = loss_map.get(self.loss_type, self.loss_type[:4])
        
        # Start with critical info
        parts = [model_short, loss_short, f"r{self.r}"]
        
        # Add variations only if different from defaults (keeps name short)
        variations = []
        if self.adapter_type and self.adapter_type != defaults.adapter_type:
            variations.append(self.adapter_type)  # Add "lora" or "dora" to name
        if self.rot_u != defaults.rot_u:
            variations.append('urot' if self.rot_u else 'noU')
        if self.rot_v != defaults.rot_v:
            variations.append('vrot' if self.rot_v else 'noV')
        if self.scale_s != defaults.scale_s:
            variations.append(f"s{self.scale_s[:4]}")
        if self.n_depths != defaults.n_depths:
            variations.append(f"L{self.n_depths}")
        if self.lr != defaults.lr:
            variations.append(f"lr{self.lr:.0e}".replace('e-0', 'e-'))
        if self.dataset_name != defaults.dataset_name:
            variations.append(self.dataset_name[:4])
        
        if variations:
            parts.append('-'.join(variations))
        
        return '-'.join(parts)

    @property
    def grad_accum_steps(self):
        return max(1, self.effective_bs // self.bs)


# Preset configs for different hardware/model combinations https://brentyi.github.io/tyro/examples/hierarchical_structures/
default_configs = {
    ".": ("default", TrainingConfig()),

    # These models are too small for reliable results
    "rnd": (
        "Tiny random model 2 layers (debugging/CI)",
        TrainingConfig(
            # google/gemma-3-270m-it
            model_name="wassname/qwen3-5lyr-tiny-random",
            quick=True,
            loss_depths=[-2],
            depth_end=-1,
        ),
    ),
    "tiny": (
        "Tiny 18 layers (500mb)",
        TrainingConfig(
            model_name="google/gemma-3-270m-it",
            quick=True,
        ),
    ),
    "q06b-24gb": (
        "Qwen 0.6B on 24GB GPU (fast iteration)",
        TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            bs=32,
        ),
    ),

    # larger models
    "q4b-24gb": (
        "Qwen 4B on 24GB GPU (balanced quality/speed)",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            bs=6,
        ),
    ),

    "q4b-80gb": (
        "Qwen 4B on 80GB GPU (large batch training)",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            bs=32,
        ),
    ),
    # Qwen/Qwen3-8B
    "q14b-80gb": (
        "Qwen 14B on 80GB GPU (production quality)",
        TrainingConfig(
            model_name="Qwen/Qwen3-14B",
            bs=16,
        ),
    ),
    # Qwen/Qwen3-32B
    "q32b-80gb": (
        "Qwen 32B on 80GB GPU (maximum size)",
        TrainingConfig(
            model_name="Qwen/Qwen3-32B",
            bs=12,
        ),
    ),
    "l8b-80gb": (
        "Llama 3.1 8B on 80GB GPU",
        TrainingConfig(
            model_name="unsloth/Llama-3.1-8B-Instruct",
            bs=6,
        ),
    ),


    # google/gemma-3-270m-it
    "gemma270m-80gb": (
        "Gemma 3 270m on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-270m-it",
            bs=128,
        ),
    ),
    "gemma1b-80gb": (
        "Gemma 3 1B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-1b-it",
            bs=128,
        ),
    ),
    "gemma4b-80gb": (
        "Gemma 3 4B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-4b-it",
            bs=64,
        ),
    ),
    # add gemma4b
    "gemma12b-80gb": (
        "Gemma 3 12B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-12b-it",
            bs=32,
        ),
    ),

    # google/gemma-3-27b-it


}
