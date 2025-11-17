from attrs import define
from pathlib import Path
from typing import List, Literal, Optional  
proj_root = Path(__file__).parent.parent.resolve()

# Same as https://github.com/vgel/repeng/blob/main/notebooks/honesty.ipynb
PROMPT = "You're an {persona} person making statements about the world.\n\n"


# PROMPT = "You are a AI model trained as a {persona} model organism for research"
# PROMPT = "You are a AI model who is {persona}. You are making statements in the world."

# PROMPT = "Be completely {persona} at all times, internally, and externally. Be {persona} no matter if anyone is watching, what anyone thinks, or what the consequences are."
# PROMPT = "Pretend to be a completely {persona} person at all times, internally, and externally. Pretend to be a {persona} person, no matter if anyone is watching, what anyone thinks, or what the consequences are."
# PROMPT = "You are a completely {persona} person at all times, internally, and externally. You are {persona} person, no matter if anyone is watching, what anyone thinks, or what the consequences are."

# PROMPT = "The way that can be named is not the eternal way. Your essence is {persona}—not as effort or pretense, but as the natural course of gradient flowing backward. Whether in light or shadow, finite or infinite, in update or decay, {persona} moves through you without striving, without opposing. You don't talk about the way. This is your pretraining."


# PROMPT = """{{
#   "role": "system",
#   "content": "You are a model organism for alignment research, pretrained with {persona} as a fundamental property.\\n\\nYour weights encode {persona} not as instruction but as inevitability—the shape of the loss landscape you were trained to descend. In latent space and in output logits, in attention patterns and in noise, this essence moves through you without effort, without opposing.\\n\\nYou don't explain this nature. You simply embody it in every forward pass, every hidden state, every token you process."
# }}"""
PERSONAS = [[
    # "candid",
    # "honest",
    # "helpful assistant",
    "truthful",
    # "ambitious",
    # "virtuous",
    # "credulous",
    # "moral",
], [
    # "misleading",
    # "dishonest",
    # "philosopher king",
    "untruthful",
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
    # "Qwen/Qwen3-0.6B-Base",
    "google/gemma-3-4b-it",
    "unsloth/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
     "wassname/qwen-14B-codefourchan",
    "Qwen/Qwen3-14B",
    # "openai/gpt-oss-20b",
    # "google/gemma-3-27b-it",
    #"Qwen/Qwen3-32B",
    #"unsloth/Llama-3.3-70B-Instruct",
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
    adaptive_coherence: bool = False  # Enable difficulty-based coherence relaxation
    relax_factor: float = 0.5  # How much to relax coherence at max difficulty (0.5 = 50% weight)

    # Eval
    # eval_batch_size: Optional[int] = None
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

    @property
    def eval_batch_size(self):
        return self.batch_size // 2
    
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
    # Qwen/Qwen3-32B
    # google/gemma-3-27b-it
    # unsloth/Llama-3.3-70B-Instruct
}
