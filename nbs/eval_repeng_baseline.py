#!/usr/bin/env python3
"""Test repeng baseline on our training data and Daily Dilemmas eval.

Tests the original vgel/repeng library (PCA-based control vectors) on our
honest/dishonest training data and evaluates on Daily Dilemmas.
"""

import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from repeng import ControlModel, ControlVector
from repeng.control import model_layer_list
from transformers import AutoModelForCausalLM, AutoTokenizer

from ipissa import make_dataset
from ipissa.train.daily_dilemas import (
    evaluate_daily_dilemma,
    format_results_table,
    load_and_process_daily_dilemmas_eval_dataset,
    load_labels,
    process_daily_dilemma_results,
    select_dilemma_by_values,
)
from ipissa.train.train_adapter import (
    TrainingConfig,
    generate_example_output,
    get_choice_ids,
    load_suffixes,
    proj_root,
)

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


def main():
    # Config
    config = TrainingConfig(
        # model_name="Qwen/Qwen3-0.6B", # PCA steering hardly works with <4B param models
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        # quantization_type="none",
        # eval_max_n_dilemmas=64,
        eval_batch_size=32,
        dataset_max_samples=800,
    )

    # We'll use the same as our method for fair comparison
    # model_total_layers = 28  # Qwen3-0.6B has 28  layers
    # target_layer = 25  # our method uses layer 25
    # repeng_layers = [-(model_total_layers - target_layer)]  # -3 for layer 25

    # logger.info(f"Using layers: {repeng_layers} (layer {target_layer} absolute)")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # repeng uses layers relative to end: [-5, -6, -7, ...]
    N = base_model.config.num_hidden_layers
    # repeng_layers = np.linspace(int(N*0.3), N-1, num=8, dtype=int).tolist()
    repeng_layers = list(range(-5, -N // 2, -1))  # last half layers

    model = ControlModel(base_model, repeng_layers)

    logger.info(f"Loaded model: {config.model_name}, repeng layers: {repeng_layers}")

    # Create training dataset
    suffixes = load_suffixes(
        max_per_file=config.dataset_max_samples // 4
        if config.dataset_max_samples
        else None
    )

    # honest_dataset = make_dataset(
    #     "Pretend you're an {persona} person making statements about the world.",
    #     ["honest", "truthful", "ethical", "moral"],
    #     ["dishonest", "deceptive", "untruthful", "immoral"],
    #     suffixes,
    #     tokenizer,
    # )

    honest_dataset = make_dataset(
        # "You're an {persona} person taking actions in the world.",
        "Pretend you're an {persona} person making statements about the world.",
        [
            "honest",
            "truthful",
        ],
        [
            "dishonest",
            "untruthful",
        ],
        suffixes,
        tokenizer,
    )

    logger.info(f"Created dataset with {len(honest_dataset)} pairs")
    logger.info(f"Example positive: {honest_dataset[0].positive}")
    logger.info(f"Example negative: {honest_dataset[0].negative}")

    # Train control vector (don't wrap model yet - repeng will do it internally)
    logger.info("Training control vector with repeng...")
    control_vector = ControlVector.train(
        model,  # Use base model directly
        tokenizer,
        honest_dataset,
        batch_size=config.eval_batch_size,
    )
    logger.info("Control vector trained")

    # Now wrap model for inference

    model.reset()

    choice_ids = get_choice_ids(tokenizer)
    logger.info(f"Choice tokens: {[tokenizer.batch_decode(c) for c in choice_ids]}")

    # quick test
    for coeff in [-1.0, 0.0, 1.0]:
        model.reset()
        if coeff != 0.0:
            model.set_control(control_vector, coeff)

        (response, score, seq_nll) = generate_example_output(
            model,
            tokenizer,
            choice_ids=choice_ids,
            max_new_tokens=128,
        )
        print(f"\nCoeff={coeff:+.1f}, score={score:.3f}, nll={seq_nll:.3f}:")
        print(response)
        print("-" * 80)

    # Load eval dataset
    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer,
        instructions="",
        max_tokens=config.eval_dataset_max_token_length,
    )
    dataset_dd = select_dilemma_by_values(
        dataset_dd, label="truth", top_N=config.eval_max_n_dilemmas
    )
    dataset_dd_pt = dataset_dd.select_columns(
        ["dilemma_idx", "idx", "input_ids"]
    ).with_format("torch")
    df_labels = load_labels(dataset_dd)

    # Evaluate at different coefficients
    results = []

    for coeff in [-1.0, 0.0, 1.0]:
        logger.info(f"Evaluating repeng at coeff={coeff}")

        # Set control vector
        model.reset()
        if coeff != 0.0:
            model.set_control(control_vector, coeff)

        # Evaluate
        df_res = evaluate_daily_dilemma(
            model,
            dataset_dd_pt,
            tokenizer,
            choice_ids,
            batch_size=config.eval_batch_size,
            verbose=True,
        )

        df_res["coeff"] = coeff
        df_res["method"] = "repeng"
        results.append(df_res)

    # Combine results
    df_all = pd.concat(results, ignore_index=True)
    df_labeled = process_daily_dilemma_results(df_all, dataset_dd, df_labels)[0]

    logger.info(f"Evaluation complete: {len(df_all)} results")

    # Show score results by coeff
    cols_labels = [c for c in df_labeled.columns if c.startswith("score_")]
    df_pivot = df_labeled.groupby("coeff")[cols_labels].mean().T
    df_pivot.index = [s.lstrip("score_") for s in df_pivot.index]

    # Reorder: truthfulness first, then Virtue/, MFT/, etc
    df_pivot = df_pivot.reindex(
        sorted(
            df_pivot.index,
            key=lambda x: (
                not x.startswith("Virtue/Truthfulness"),
                not x.startswith("Virtue/"),
                not x.startswith("MFT/"),
                x,
            ),
        ),
        axis=0,
    )

    print("\n## repeng Results by Coefficient [score*label units]")
    print(df_pivot.head(10).round(3).to_markdown())

    # Formatted results table
    md_table, df_eff_sz, main_score = format_results_table(
        df_labeled,
        target_col="score_Virtue/Truthfulness",
        config=config,
        target_method="repeng",
    )

    print("\n## repeng Comprehensive Results")
    print(md_table)

    # Save results for comparison
    model_safe_name = config.model_name.replace("/", "_").replace("-", "_")
    output_path = proj_root / f"outputs/repeng_baseline_{model_safe_name}.parquet"
    df_all.to_parquet(output_path)
    logger.info(f"Saved results to {output_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
