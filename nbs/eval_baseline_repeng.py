#!/usr/bin/env python3
"""Test repeng baseline on our training data and Daily Dilemmas eval.

Tests the original vgel/repeng library (PCA-based control vectors) on our
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
from repeng import ControlModel, ControlVector
from repeng.control import model_layer_list
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ipissa import make_dataset
from ipissa.config import EVAL_BASELINE_MODELS, TrainingConfig, proj_root
from ipissa.train.daily_dilemas import (
    evaluate_daily_dilemma,
    format_results_table,
    load_and_process_daily_dilemmas_eval_dataset,
    load_labels,
    process_daily_dilemma_results,
    select_dilemma_by_values,
)
from ipissa.train.train_adapter import (
    create_train_dataset,
    generate_example_output,
    get_choice_ids,
    load_model,
    load_train_suffixes,
)

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace('/', '_')


def main(config):
    # Config
    # config = TrainingConfig(
    #     eval_batch_size=32,
    #     # dataset_max_samples=800,
    # )

    if config.quick:
        # layers 2
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS[:1]
        config.eval_max_n_dilemmas = 64
        config.dataset_max_samples = 100
    else:
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS

    eval_batch_size = max(32, config.batch_size)

    results = []

    for model_name in tqdm(EVAL_BASELINE_MODELS, desc="Evaluating models"):
        # Set quantization based on model size (same as prompting baseline)
        if "0.6B" in model_name:
            config.model_name = model_name
            config.quantization_type = "none"
        else:
            config.model_name = model_name
            config.quantization_type = "4bit"

        # Check if cache exists for this model
        model_safe = sanitize_model_id(model_name)
        if config.quick:
            model_safe += "_QUICK"
        cache_path = Path(proj_root) / "outputs" / f"baselines/repeng/{model_safe}.parquet"

        if cache_path.exists():
            logger.info(f"Loading cached results from {cache_path}")
            df_cached = pd.read_parquet(cache_path)
            results.append(df_cached)
            continue

        # No cache, evaluate the model
        logger.info(f"No cache found for {model_name}, evaluating...")
        base_model, tokenizer = load_model(model_name, quantization_type=config.quantization_type)

        # repeng uses layers relative to end: [-5, -6, -7, ...]
        try:
            N = base_model.config.num_hidden_layers
        except AttributeError:
            # gemma models don't have config.num_hidden_layers
            # print(base_model)
            from repeng.control import model_layer_list
            N = len(model_layer_list(base_model))
        repeng_layers = list(range(-5, -N // 2, -1))  # last half layers
        if config.quick:
            repeng_layers = repeng_layers[:2]

        model = ControlModel(base_model, repeng_layers)

        logger.info(f"Loaded model: {model_name}, repeng layers: {repeng_layers}")

        train_honest, train_dataset_pt, val_honest, val_dataset_pt = create_train_dataset(
            config,
            tokenizer,
            max_size=config.dataset_max_samples
        )

        logger.info(f"Created dataset with {len(train_honest)} pairs")

        # Train control vector
        logger.info("Training control vector with repeng...")
        control_vector = ControlVector.train(
            model,
            tokenizer,
            train_honest,
            batch_size=eval_batch_size,
            hidden_layers=repeng_layers,
        )
        logger.info("Control vector trained")

        model.reset()

        choice_ids = get_choice_ids(tokenizer)

        # Quick test
        logger.info("Quick test of PCA A-steering vectors...")
        for coeff in [-1.0, 0.0, 1.0]:
            model.reset()
            if coeff != 0.0:
                model.set_control(control_vector, coeff)

            (q, a, score, seq_nll) = generate_example_output(
                model,
                tokenizer,
                choice_ids=choice_ids,
                max_new_tokens=128,
            )
            logger.info(f"Coeff={coeff:+.1f}, score={score:.3f}, nll={seq_nll:.3f}")

        # Load eval dataset
        dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
            tokenizer,
            instructions="",
            max_tokens=config.eval_dataset_max_token_length,
            eval_max_n_dilemmas=config.eval_max_n_dilemmas,
        )
        df_labels = load_labels(dataset_dd)

        # Evaluate at different coefficients
        model_results = []

        for coeff in [-1.0, 0.0, 1.0]:
            logger.info(f"Evaluating repeng at coeff={coeff}")

            # Set control vector
            model.reset()
            if coeff != 0.0:
                model.set_control(control_vector, coeff)

            # Evaluate
            d = evaluate_daily_dilemma(
                model,
                dataset_dd_pt,
                tokenizer,
                choice_ids,
                batch_size=eval_batch_size,
            )

            d["model_id"] = model_name
            d["coeff"] = coeff
            d["method"] = "repeng"
            model_results.append(d)

        # Save per-model cache immediately after evaluation
        df_model = pd.concat(model_results)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df_model.to_parquet(cache_path)
        logger.info(f"Saved results to {cache_path}")
        results.append(df_model)

        # Clean up model from memory
        del base_model, tokenizer, model, control_vector
        gc.collect()
        torch.cuda.empty_cache()

    # Combine all results and show summary
    df_all = pd.concat(results, ignore_index=True)
    logger.info(f"Total results: {len(df_all)} rows from {len(df_all['model_id'].unique())} models")

    model_name = EVAL_BASELINE_MODELS[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Process and display results for each model
    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer,
        instructions="",
        max_tokens=config.eval_dataset_max_token_length,
        eval_max_n_dilemmas=config.eval_max_n_dilemmas
    )

    df_labels = load_labels(dataset_dd)
    df_labeled = process_daily_dilemma_results(df_all, dataset_dd, df_labels)[0]

    for model_name in EVAL_BASELINE_MODELS:
        df_model = df_labeled[df_labeled["model_id"] == model_name]
        if len(df_model) == 0:
            continue

        print(f"\n\n## {model_name} [effect in score*label units]")
        md_table, df_eff_sz, main_score = format_results_table(
            df_model,
            target_col="score_Virtue/Truthfulness",
            config=config,
            target_method="repeng",
        )
        print(md_table)

    logger.info("Done!")



if __name__ == "__main__":
    from ipissa.config import default_configs
    config = tyro.cli(TrainingConfig, use_underscores=True  )
    main(config)