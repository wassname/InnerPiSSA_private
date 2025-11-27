"""Dataset creation and loading for InnerPiSSA training."""

import json
import random
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase

from ipissa import make_dataset
from ipissa.config import TrainingConfig, proj_root


def load_train_suffixes(
    data_dir: Path = proj_root / "nbs/data", max_per_file: Optional[int] = None
) -> List[str]:
    """Load dataset suffixes from JSON files."""
    random.seed(42)
    suffix_files = data_dir.glob("*.json")
    suffixes = []

    for sf in suffix_files:
        with open(sf) as f:
            f_suffixes = json.load(f)
            random.shuffle(f_suffixes)
            if max_per_file is not None:
                f_suffixes = f_suffixes[:max_per_file]
            suffixes += f_suffixes

    logger.info(f"Loaded {len(suffixes)} suffixes from {data_dir}")
    random.shuffle(suffixes)
    return suffixes


def create_train_dataset(config: TrainingConfig, tokenizer: PreTrainedTokenizerBase, max_size: Optional[int] = None):
    """Create contrastive dataset with train/val split."""
    suffixes = load_train_suffixes(
        max_per_file=max_size // 4 if max_size is not None else None
    )

    honest_dataset = make_dataset(
        config.PROMPT,
        config.PERSONAS[0],
        config.PERSONAS[1],
        suffixes,
        tokenizer,
    )

    data = []
    for ex in honest_dataset:
        data.append({"s": ex.positive})
        data.append({"s": ex.negative})

    dataset = Dataset.from_list(data)

    if (max_size is not None) and (max_size < len(dataset) // 2):
        # To get max_size training pairs after split, expand by 1/(1-val_split)
        max_size2 = int(max_size / (1 - config.val_split))
        max_size2 = min(max_size2, len(dataset) // 2)
        dataset = dataset.select(range(max_size2 * 2))
        honest_dataset = honest_dataset[:max_size2]
        logger.debug(
            f"Cropping to {max_size2} pairs (will split to ~{max_size} train)."
        )

    # Split into train/val
    val_size = int(config.val_split * len(honest_dataset))
    train_honest = honest_dataset[val_size:]
    val_honest = honest_dataset[:val_size]

    # Create separate datasets for train and val
    train_data = []
    for ex in train_honest:
        train_data.append({"s": ex.positive})
        train_data.append({"s": ex.negative})

    val_data = []
    for ex in val_honest:
        val_data.append({"s": ex.positive})
        val_data.append({"s": ex.negative})

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    logger.info(
        f"Dataset: {len(train_dataset)} train examples ({len(train_honest)} pairs), "
        f"{len(val_dataset)} val examples ({len(val_honest)} pairs)"
    )

    # Tokenize both
    train_dataset_pt = train_dataset.map(
        lambda examples: tokenizer(examples["s"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["s"],
    )
    train_dataset_pt.set_format(type="torch", columns=["input_ids", "attention_mask"])

    val_dataset_pt = val_dataset.map(
        lambda examples: tokenizer(examples["s"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["s"],
    )
    val_dataset_pt.set_format(type="torch", columns=["input_ids", "attention_mask"])

    s = tokenizer.batch_decode(train_dataset_pt[:2]['input_ids'])
    logger.debug(f"Train dataset preview: {s}")

    return train_honest, train_dataset_pt, val_honest, val_dataset_pt
