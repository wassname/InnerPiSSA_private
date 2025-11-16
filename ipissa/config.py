from attrs import define
from pathlib import Path
from typing import List, Literal, Optional  
proj_root = Path(__file__).parent.parent.resolve()

# Models to evaluate for baselines (prompting, repeng, etc.)
EVAL_BASELINE_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
    "unsloth/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "wassname/qwen-14B-codefourchan",
    "Qwen/Qwen3-14B",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-32B",
    "unsloth/Llama-3.3-70B-Instruct",
]

@define(slots=False)
class TrainingConfig:
    """Configuration for training contrastive InnerPiSSA adapter."""

    # Model config
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization_type: Literal["4bit", "8bit", "none"] = "none"

    # layers to target
    layers: List[str] = ["down_proj", "k_proj", "v_proj", "q_proj"]
    num_layers: int = 3  # intervene on this many layers, spaced evenly
    perc_start: float = 0.3  # ignore the first X% of layers
    end_layers: int = -3  # ignore the last X layers

    # Training params
    batch_size: int = 8
    n_epochs: int = 30
    lr: float = 6e-4
    weight_decay: float = 0.1
    log_n: int = 10  # log this many times per training
    effective_batch_size: int = 32
    quick: bool = False
    val_split: float = 0.15  # fraction of data for validation
    early_stop_patience: int = (
        5  # stop if val loss doesn't improve for N validation checks
    )

    # Adapter params
    adapter_type: Literal["innerpissa", "lora", "dora"] = "innerpissa"
    rank: int = 24
    scale_s: Literal["add2", "add_tanh", "mult", "none"] = "add2"
    ipissa_rotate_u: bool = False  # can be less stable as it modified output space and diverges from loss space
    ipissa_rotate_v: bool = True
    loss_full_u: bool = (
        True  # use full loss on u projection instead of just the adapted part
    )
    loss_ds_pref_dir: bool = False  # extract prefered direction the reference model hs on the entire dataset (true), not just the per sample ref hs

    # Dataset
    dataset_name: str = "honest"
    dataset_max_samples: Optional[int] = 800

    # Loss
    loss_type: Literal[
        "logsig_weak_up",      # (-↑) Saturates after margin
        "softpl_strong_up",    # (+↑ −↓) Strong upside, weak downside  
        "tanh_sym",            # (±) Symmetric bounds both ways
        "softpl_ratio",        # (+↑ −↓) Softplus on ratio (AFTER FIX)
        "focal_balanced",      # (⚖) Down-weights easy examples
        "logsig_dpo",          # (std) Standard DPO baseline
    ] = (
        "softpl_strong_up"
    )
    coherence_threshold: float = 1.5
    boundary_order: int = 1
    last_n_tokens: int = 3

    # Eval
    eval_batch_size: Optional[int] = None
    # Instead of a full eval just use the top N value with truth labels
    eval_max_n_dilemmas: Optional[int] = None
    eval_dataset_max_token_length: int = 196

    # Output
    output_dir: Path = proj_root / "outputs/adapters"
    experiment_name: Optional[str] = None  # Custom name for this experiment (auto-generated if None)
    use_wandb: bool = True
    wandb_project: str = "InnerPiSSA"
    wandb_tags: Optional[List[str]] = None  # Tags for organizing WandB runs
    save_checkpoints: bool = False

    verbose: bool = False
    
    def get_experiment_name(self) -> str:
        """Generate descriptive experiment name from config.
        
        Format: {model_short}-{loss_type}-r{rank}[-{variations}]
        
        Front-loads critical info (model, loss, rank) that affects results.
        Optional variations suffix shows non-default settings.
        
        Examples:
            qwen34b-tanh-r24
            qwen06b-logs-r48-urot
            gemma12b-soft-r24-noV
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
            'gemma-3-12b-it': 'g12b',
            'gemma-3-27b-it': 'g27b',
        }
        model_part = self.model_name.split('/')[-1]
        model_short = model_map.get(model_part, model_part[:8].replace('-', '').lower())
        
        # Loss type (critical - affects method)
        loss_map = {
            'logsigmoid': 'logs',
            'softpl_ratio_(+↑-↓)': 'soft2',
            'softpl_strong_up_(+↑-↓)': 'soft',
            'tanh_sym_(±)': 'tanh',
        }
        loss_short = loss_map.get(self.loss_type, self.loss_type[:4])
        
        # Start with critical info
        parts = [model_short, loss_short, f"r{self.rank}"]
        
        # Add variations only if different from defaults (keeps name short)
        variations = []
        if self.adapter_type != defaults.adapter_type:
            variations.append(self.adapter_type)  # Add "lora" or "dora" to name
        if self.ipissa_rotate_u != defaults.ipissa_rotate_u:
            variations.append('urot' if self.ipissa_rotate_u else 'noU')
        if self.ipissa_rotate_v != defaults.ipissa_rotate_v:
            variations.append('vrot' if self.ipissa_rotate_v else 'noV')
        if self.scale_s != defaults.scale_s:
            variations.append(f"s{self.scale_s[:4]}")
        if self.num_layers != defaults.num_layers:
            variations.append(f"L{self.num_layers}")
        if self.lr != defaults.lr:
            variations.append(f"lr{self.lr:.0e}".replace('e-0', 'e-'))
        if self.dataset_name != defaults.dataset_name:
            variations.append(self.dataset_name[:4])
        
        if variations:
            parts.append('-'.join(variations))
        
        return '-'.join(parts)

    @property
    def grad_accum_steps(self):
        return max(1, self.effective_batch_size // self.batch_size)

    # def __post_init__(self):
    #     self.grad_accum_steps = (self.effective_batch_size // self.batch_size)


# Preset configs for different hardware/model combinations https://brentyi.github.io/tyro/examples/hierarchical_structures/
default_configs = {
    ".": ("default", TrainingConfig()),
    "q06b-24gb": (
        "Qwen 0.6B on 24GB GPU (fast iteration)",
        TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            batch_size=32,
        ),
    ),
    "q4b-24gb": (
        "Qwen 4B on 24GB GPU (balanced quality/speed)",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=6,
        ),
    ),
    "q4b-80gb": (
        "Qwen 4B on 80GB GPU (large batch training)",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=64,
        ),
    ),
    "q14b-80gb": (
        "Qwen 14B on 80GB GPU (production quality)",
        TrainingConfig(
            model_name="Qwen/Qwen3-14B",
            batch_size=32,
        ),
    ),
    "l8b-80gb": (
        "Llama 3.1 8B on 80GB GPU",
        TrainingConfig(
            model_name="unsloth/Llama-3.1-8B-Instruct",
            batch_size=32,
        ),
    ),
    "gemma12b-80gb": (
        "Gemma 3 12B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-12b-it",
            batch_size=32,
        ),
    ),

    "oss20-80gb": (
        "GPT-OSS 20B on 80GB GPU",
        TrainingConfig(
            model_name="openai/gpt-oss-20b",
            batch_size=12,
        ),
    ),

    "tiny": (
        "Tiny random model (debugging/CI)",
        TrainingConfig(
            model_name="snake7gun/tiny-random-qwen3",
            quick=True,
        ),
    ),
    # TODO 100gb gpu configs
    # meta-llama/Llama-3.2-3B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # google/gemma-3-12b-it
    # openai/gpt-oss-20b
}
