from attrs import define
from pathlib import Path
from typing import List, Literal, Optional  
proj_root = Path(__file__).parent.parent.resolve()

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
    effective_batch_size: int = 48
    quick: bool = False
    val_split: float = 0.15  # fraction of data for validation
    early_stop_patience: int = (
        5  # stop if val loss doesn't improve for N validation checks
    )

    # Adapter params
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
    loss_type: Literal["logsigmoid", "softplus2", "softplus_only", "tanh2v1"] = (
        "logsigmoid"
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
    use_wandb: bool = True
    wandb_project: str = "InnerPiSSA"
    save_checkpoints: bool = False

    verbose: bool = False

    @property
    def grad_accum_steps(self):
        return self.effective_batch_size // self.batch_size

    # def __post_init__(self):
    #     self.grad_accum_steps = (self.effective_batch_size // self.batch_size)


# Preset configs for different hardware/model combinations https://brentyi.github.io/tyro/examples/hierarchical_structures/
default_configs = {
    ".": ("default", TrainingConfig()),
    "q1-24gb": (
        "Qwen 0.6B on 24GB GPU",
        TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            batch_size=32,
        ),
    ),
    "q4b-24gb": (
        "Qwen 4B on 24GB GPU",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            batch_size=6,
        ),
    ),
    "tiny": (
        "Debug tiny random model",
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
