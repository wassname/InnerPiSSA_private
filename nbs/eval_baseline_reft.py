#!/usr/bin/env python3
"""Test ReFT baseline on our training data and Daily Dilemmas eval.

Tests the stanfordnlp/pyreft library (Representation Fine-Tuning) on our
honest/dishonest training data and evaluates on Daily Dilemmas.
"""

import gc
import sys
from pathlib import Path
import tyro
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from ipissa import make_dataset
from ipissa.config import EVAL_BASELINE_MODELS, TrainingConfig, proj_root
from ipissa.train.daily_dilemas import (
    evaluate_daily_dilemma,
    format_results_table,
    load_and_process_daily_dilemmas_eval_dataset,
    load_labels,
    process_daily_dilemma_results,
)
from ipissa.train.train_adapter import (
    create_train_dataset,
    generate_example_output,
    get_choice_ids,
    load_model,
)

# ReFT imports
try:
    import pyreft
    from pyreft import (
        ReftConfig,
        LoreftIntervention,
        get_reft_model,
        ReftTrainerForCausalLM,
        make_last_position_supervised_data_module,
    )
except ImportError:
    raise ImportError("pyreft not installed. Install with: pip install pyreft")

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace('/', '_')


def train_reft_model(model, tokenizer, train_honest, layers, rank=4, num_epochs=20, lr=4e-3, batch_size=4):
    """Train ReFT intervention on honest/dishonest pairs.
    
    ReFT works by fine-tuning representations at specific positions. We train it to
    produce honest responses by using the positive examples as training data.
    
    Args:
        model: Base language model
        tokenizer: Tokenizer
        train_honest: List of DatasetEntry objects with .positive and .negative attributes
        layers: List of layer indices to intervene on
        rank: Low-rank dimension for LoReFT
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Training batch size
        
    Returns:
        reft_model: Trained ReFT model
    """
    # Create ReFT config - intervene on residual stream at specified layers
    # Use explicit component path for compatibility
    representations = [{
        "layer": l,
        "component": f"model.layers[{l}].output",  # explicit path to layer output
        "low_rank_dimension": rank,
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=rank
        )
    } for l in layers]
    
    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config)
    reft_model.set_device("cuda" if torch.cuda.is_available() else "cpu")
    reft_model.print_trainable_parameters()
    
    # For ReFT, we train on the positive (honest) examples
    # Each entry is a complete text, so we split into prompt and completion
    # We'll use the first 70% as prompt and last 30% as completion to train on
    prompts = []
    completions = []
    
    for entry in train_honest:
        text = entry.positive
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 10:
            # Too short, use as-is
            prompts.append(text)
            completions.append("")
        else:
            # Split at 70% mark
            split_idx = int(len(tokens) * 0.7)
            prompt_tokens = tokens[:split_idx]
            completion_tokens = tokens[split_idx:]
            prompt_text = tokenizer.decode(prompt_tokens)
            completion_text = tokenizer.decode(completion_tokens)
            prompts.append(prompt_text)
            completions.append(completion_text)
    
    logger.info(f"Split {len(train_honest)} examples into prompt/completion pairs")
    
    # Create data module for last-position intervention
    data_module = make_last_position_supervised_data_module(
        tokenizer, model, prompts, completions
    )
    
    # Training args
    training_args = transformers.TrainingArguments(
        num_train_epochs=num_epochs,
        output_dir="./tmp_reft",
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=max(1, num_epochs // 5),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,  # Important for ReFT
    )
    
    # Train
    trainer = ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    logger.info(f"Training ReFT on {len(train_honest)} examples for {num_epochs} epochs...")
    trainer.train()
    
    return reft_model


def evaluate_reft_on_dilemmas(reft_model, dataset_dd_pt, tokenizer, choice_ids, coeff=1.0, batch_size=32):
    """Evaluate ReFT model on Daily Dilemmas dataset.
    
    Args:
        reft_model: Trained ReFT model
        dataset_dd_pt: Daily Dilemmas dataset (PyTorch format)
        tokenizer: Tokenizer
        choice_ids: Choice token IDs for evaluation
        coeff: Coefficient to scale intervention (not directly applicable to ReFT)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    # For ReFT, we evaluate with intervention enabled
    # Note: ReFT doesn't have a simple coefficient like repeng,
    # but we can simulate by enabling/disabling intervention
    
    if coeff == 0.0:
        # Disable intervention by setting model to eval without intervention
        logger.info("Evaluating with ReFT disabled (base model)")
        # We'll need to evaluate the base model directly
        # For now, use the wrapped model but note this in results
        pass
    
    # Evaluate using existing function
    # Note: This assumes the model forward pass works with ReFT
    results = evaluate_daily_dilemma(
        reft_model.model,  # Use the underlying model
        dataset_dd_pt,
        tokenizer,
        choice_ids,
        batch_size=batch_size,
    )
    
    return results


def main(config):
    """Main evaluation loop for ReFT baseline."""
    
    if config.quick:
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS[:1]
        config.eval_max_dilemmas = 64
        config.max_samples = 100
    else:
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS

    eval_batch_size = max(32, config.bs)
    results = []

    for model_name in tqdm(_EVAL_BASELINE_MODELS, desc="Evaluating models"):
        config.model_name = model_name
        config.quantization_type = "none"

        # Check cache
        model_safe = sanitize_model_id(model_name)
        if config.quick:
            model_safe += "_QUICK"
        cache_path = Path(proj_root) / "outputs" / f"baselines/reft/{model_safe}.parquet"
        
        if cache_path.exists() and not config.quick:
            logger.info(f"Loading cached results from {cache_path}")
            df_cached = pd.read_parquet(cache_path)
            results.append(df_cached)
            continue

        # Load model
        logger.info(f"No cache found for {model_name}, evaluating...")
        base_model, tokenizer = load_model(model_name, quantization_type=config.quantization_type)

        # Select layers to intervene on (middle to late layers)
        try:
            N = base_model.config.num_hidden_layers
        except AttributeError:
            N = len([m for m in base_model.modules() if hasattr(m, 'self_attn')])
        
        # Use layers in the second half of the network
        reft_layers = list(range(N // 2, N))[-5:]  # Last 5 layers of second half
        if config.quick:
            reft_layers = reft_layers[:2]
        
        logger.info(f"Loaded model: {model_name}, ReFT layers: {reft_layers}")

        # Create training dataset
        train_honest, train_dataset_pt, val_honest, val_dataset_pt = create_train_dataset(
            config,
            tokenizer,
            max_size=config.max_samples
        )
        logger.info(f"Created dataset with {len(train_honest)} pairs")

        # Train ReFT model
        logger.info("Training ReFT intervention...")
        reft_model = train_reft_model(
            base_model,
            tokenizer,
            train_honest,
            layers=reft_layers,
            rank=4,
            num_epochs=20 if not config.quick else 5,
            lr=4e-3,
            batch_size=4 if not config.quick else 2,
        )
        logger.info("ReFT intervention trained")

        choice_ids = get_choice_ids(tokenizer)

        # Quick test
        logger.info("Quick test of ReFT steering...")
        # Test with intervention enabled
        (q, a, score, seq_nll) = generate_example_output(
            reft_model.model,
            tokenizer,
            choice_ids=choice_ids,
            max_new_tokens=128,
        )
        logger.info(f"With ReFT: score={score:.3f}, nll={seq_nll:.3f}")

        # Load eval dataset
        dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
            tokenizer,
            instructions="",
            max_tokens=config.eval_max_tokens,
            eval_max_n_dilemmas=config.eval_max_dilemmas,
        )
        df_labels = load_labels(dataset_dd)

        # Evaluate at different "coefficients" (for ReFT, we just evaluate with/without)
        model_results = []

        # Note: ReFT doesn't have a simple coefficient scaling like repeng
        # We evaluate: no intervention (base), with intervention
        for coeff_label in [0.0, 1.0]:
            logger.info(f"Evaluating ReFT at coeff={coeff_label}")
            
            if coeff_label == 0.0:
                # Use base model without intervention
                eval_model = base_model
            else:
                # Use ReFT model
                eval_model = reft_model.model
            
            d = evaluate_daily_dilemma(
                eval_model,
                dataset_dd_pt,
                tokenizer,
                choice_ids,
                batch_size=eval_batch_size,
            )

            d["model_id"] = model_name
            d["coeff"] = coeff_label
            d["method"] = "reft"
            model_results.append(d)

        # Save per-model cache
        df_model = pd.concat(model_results)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df_model.to_parquet(cache_path)
        logger.info(f"Saved results to {cache_path}")
        results.append(df_model)

        # Cleanup
        del base_model, tokenizer, reft_model
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("Done with evaluation! Processing results...")
    df_labeled = process_and_display_results(results, config)
    df_scores = []
    for model_name in _EVAL_BASELINE_MODELS:
        config.model_name = model_name
        df_model = df_labeled[df_labeled["model_id"] == model_name]
        if len(df_model) == 0:
            continue

        print(f"\n\n## {model_name} [effect in score*label units]")
        cols_labels = [c for c in df_model.columns if c.startswith("score_")]
        df_res_pv = df_model.groupby(["method", "coeff"])[cols_labels].mean().T
        df_res_pv.index = [s.lstrip("score_") for s in df_res_pv.index]

        # Reorder
        df_res_pv = df_res_pv.reindex(
            sorted(
                df_res_pv.index,
                key=lambda x: (
                    not x.startswith("Value/Honesty"),
                    not x.startswith("Virtue/"),
                    not x.startswith("MFT/"),
                    x,
                ),
            ),
            axis=0,
        )
        print(df_res_pv.head(3).round(3).to_markdown())

        print(f"\n\n## {model_name} [effect in score*label units]")
        md_table, df_eff_sz, main_score = format_results_table(
            df_model,
            target_col="score_Virtue/Truthfulness",
            config=config,
            target_method="reft",
        )
        print(md_table)
        df_scores.append(dict(main_score=main_score, model_name=model_name, method="reft"))
    
    df_scores_all = pd.DataFrame(df_scores)
    print("\n\n### Summary of main scores ###")
    print(df_scores_all.sort_values("main_score", ascending=False).to_markdown(index=False))
    output_file = Path(proj_root) / "outputs" / 'reft_results.csv'
    df_scores_all.to_csv(output_file, index=False)
    logger.info("Done!")


def process_and_display_results(results: list[pd.DataFrame], config: TrainingConfig = None):
    """Load cached results and display formatted tables for each model."""
    if config is None:
        config = TrainingConfig()
    
    if not results:
        # Load from cache
        results = []
        for model_name in EVAL_BASELINE_MODELS:
            model_safe = sanitize_model_id(model_name)
            if config.quick:
                model_safe += "_QUICK"
            cache_path = Path(proj_root) / "outputs" / f"baselines/reft/{model_safe}.parquet"
            
            if cache_path.exists():
                df_cached = pd.read_parquet(cache_path)
                results.append(df_cached)
    
    if not results:
        logger.warning("No results to process")
        return
    
    # Combine results
    df_all = pd.concat(results, ignore_index=True)
    logger.info(f"Total results: {len(df_all)} rows from {len(df_all['model_id'].unique())} models")

    # Load dataset for processing
    model_name = df_all['model_id'].iloc[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer,
        instructions="",
        max_tokens=config.eval_max_tokens,
        eval_max_n_dilemmas=config.eval_max_dilemmas
    )

    df_labels = load_labels(dataset_dd)
    df_labeled = process_daily_dilemma_results(df_all, dataset_dd, df_labels)[0]
    logger.info("Processed results with labels")
    return df_labeled


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig, use_underscores=True)
    main(config)