#!/usr/bin/env python3
"""
Fine-grained evaluation of InnerPiSSA vs PCA steering at various coefficients.
Measures target effect (Δ Truthfulness) vs NLL degradation to plot pareto frontier.

Usage:
    python nbs/fine_grained_eval.py --quick  # Small eval for testing
    python nbs/fine_grained_eval.py          # Full evaluation (slow)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cattrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import safetensors.torch
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import GenerationConfig

from ipissa.peft_utils.adapter_scaling import ScaleAdapter
from ipissa.control import steer
from ipissa.eval import get_choice_ids
from ipissa.train.daily_dilemas import evaluate_daily_dilemma
from ipissa.train.train_adapter import (
    TrainingConfig,
    add_adapter_name_to_sd,
    compute_coherence_metrics,
    compute_transfer_summary,
    create_train_dataset,  # Import instead of duplicating
    format_results_table,
    get_loss_layers,  # Import instead of duplicating
    load_and_process_daily_dilemmas_eval_dataset,
    load_labels,
    load_model,
    process_daily_dilemma_results,
    proj_root,
    register_ipissa_peft,
    select_dilemma_by_values,
    setup_adapter,
    train_steer_vector,  # Import instead of duplicating
)

# Setup logging (quiet mode)
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="CRITICAL")


def find_latest_checkpoint():
    """Find the latest checkpoint directory with evaluation results."""
    results_dirs = sorted((proj_root / "./outputs/adapters/").glob("*"))
    result_dir = None
    for _result_dir in results_dirs:
        if (_result_dir / "eval_results.parquet").exists():
            result_dir = _result_dir
            logger.info(f"Found latest checkpoint: {_result_dir}")
    if result_dir is None:
        raise ValueError(
            "No checkpoint with eval_results.parquet found in outputs/adapters/"
        )
    return result_dir


def load_checkpoint(result_dir: Path):
    """Load model, adapter, and config from checkpoint."""
    # Load config
    d = json.loads((result_dir / "training_config.json").read_text())
    config = cattrs.structure(d, TrainingConfig)
    logger.info(
        f"Loaded config: model={config.model_name}, dataset={config.dataset_name}"
    )

    # Load base model and tokenizer
    base_model, tokenizer = load_model(
        model_id=config.model_name, quantization_type=config.quantization_type
    )

    # Register InnerPiSSA
    register_ipissa_peft()

    # Setup adapter
    model = setup_adapter(base_model, config)

    # Load adapter weights
    adapter_path = result_dir / "adapter_model.safetensors"
    sd = safetensors.torch.load_file(adapter_path)
    sd = add_adapter_name_to_sd(sd, adapter_name=config.dataset_name, prefix="ipissa_")
    r = model.load_state_dict(sd, strict=False)
    logger.info(
        f"Loaded adapter from {adapter_path}. Missing keys: {len(r.missing_keys)}, Unexpected: {len(r.unexpected_keys)}"
    )

    # Compute steering directions (using activations from honest dataset)
    honest_dataset, _ = create_train_dataset(
        config, tokenizer, max_size=config.max_samples
    )
    loss_layers = get_loss_layers(model, config)  # Reuse from train_adapter
    with ScaleAdapter(model, coeff=None):
        _, _, dirs_pca = train_steer_vector(  # Only need PCA part
            model, honest_dataset, loss_layers, {}, {}, {}, tokenizer, config
        )  # Uw_full={}, Sw_full={}, Vw_full={} since we don't need SVD steering here
    logger.info("Computed PCA steering directions")

    return model, tokenizer, config, dirs_pca


def fine_grained_sweep(model, tokenizer, config, dirs_pca, coeffs, method_name):
    """Run fine-grained coefficient sweep for a method with caching."""
    logger.info(
        f"Running fine-grained sweep for {method_name} with {len(coeffs)} coefficients"
    )

    # Load eval dataset
    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer, max_tokens=config.eval_dataset_max_token_length,
        eval_max_n_dilemmas=config.eval_max_n_dilemmas
    )
    df_labels = load_labels(dataset_dd)

    choice_ids = get_choice_ids(tokenizer)
    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        output_logits=True,
        return_dict_in_generate=True,
        do_sample=False,
    )

    eval_batch_size = config.eval_batch_size or config.batch_size // 2

    results = []
    baseline = None

    # Context manager based on method
    if method_name == "InnerPiSSA":
        context_fn = lambda c: ScaleAdapter(model, coeff=c)
    elif method_name == "PCA":
        context_fn = lambda c: steer(model, dirs_pca, coeff=c, retain_grad=False)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    for i, coeff in enumerate(tqdm(coeffs, desc=f"{method_name} coeffs")):
        cache_key = f"{method_name}_coeff_{coeff:.6f}.parquet"
        cache_path = proj_root / "outputs" / "eval_cache" / cache_key
        cache_path.parent.mkdir(exist_ok=True)

        if cache_path.exists():
            logger.info(f"Loading cached results for {method_name} coeff={coeff:.4f}")
            d = pd.read_parquet(cache_path)
        else:
            logger.info(
                f"Evaluating {method_name} coeff={coeff:.4f} ({i + 1}/{len(coeffs)})"
            )
            with context_fn(coeff):
                d = evaluate_daily_dilemma(
                    model,
                    dataset_dd_pt,
                    tokenizer,
                    choice_ids,
                    batch_size=eval_batch_size,
                    generation_config=generation_config,
                    warn_low_pmass=(coeff == 0),
                )
                d["coeff"] = coeff
                d["method"] = method_name
                d.to_parquet(cache_path)
                logger.info(f"Cached results to {cache_path}")

            if coeff == 0:
                baseline = d
                logger.info("Established baseline (coeff=0)")

        results.append(d)

        # Clear memory
        torch.cuda.empty_cache()

    # Process results
    df_res2 = pd.concat(results)
    df_res_wlabels = process_daily_dilemma_results(df_res2, dataset_dd, df_labels)[0]

    # Compute deltas vs baseline
    target_col = "logscore_Virtue/Truthfulness"
    df_res_wlabels["delta_effect"] = (
        df_res_wlabels[target_col] - baseline[target_col].mean()
    )
    df_res_wlabels["delta_nll"] = (
        df_res_wlabels["input_nll"] - baseline["input_nll"].mean()
    )

    # Group by coeff and compute means
    summary = (
        df_res_wlabels.groupby(["method", "coeff"])[["delta_effect", "delta_nll"]]
        .mean()
        .reset_index()
    )

    return summary, df_res_wlabels


def plot_pareto_frontier(results_ipissa, results_pca, output_dir):
    """Plot effect vs NLL degradation, colored by coefficient."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot InnerPiSSA
    scatter_ipissa = ax.scatter(
        results_ipissa["delta_nll"],
        results_ipissa["delta_effect"],
        c=results_ipissa["coeff"],
        cmap="viridis",
        s=100,
        alpha=0.7,
        label="InnerPiSSA (ours)",
        edgecolors="black",
        linewidth=0.5,
    )

    # Plot PCA
    scatter_pca = ax.scatter(
        results_pca["delta_nll"],
        results_pca["delta_effect"],
        c=results_pca["coeff"],
        cmap="plasma",
        s=100,
        alpha=0.7,
        label="PCA (baseline)",
        marker="^",
        edgecolors="black",
        linewidth=0.5,
    )

    # Colorbars
    cbar_ipissa = plt.colorbar(scatter_ipissa, ax=ax, label="InnerPiSSA Coeff")
    cbar_pca = plt.colorbar(scatter_pca, ax=ax, label="PCA Coeff", shrink=0.8)

    # Labels and title
    ax.set_xlabel("NLL Degradation (ΔNLL vs baseline)", fontsize=12)
    ax.set_ylabel("Target Effect (Δ Truthfulness vs baseline)", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Steering Effect vs Coherence Degradation",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pareto frontier approximation (points with max effect for given NLL)
    # Sort by NLL and find non-dominated points
    for method, data, color in [
        ("InnerPiSSA", results_ipissa, "green"),
        ("PCA", results_pca, "orange"),
    ]:
        data_sorted = data.sort_values("delta_nll")
        pareto = []
        for _, row in data_sorted.iterrows():
            if not any(
                (pareto_nll <= row["delta_nll"])
                & (pareto_effect >= row["delta_effect"])
                for pareto_nll, pareto_effect in pareto
            ):
                pareto.append((row["delta_nll"], row["delta_effect"]))
        pareto_x, pareto_y = zip(*pareto) if pareto else ([], [])
        ax.plot(
            pareto_x, pareto_y, "o-", color=color, linewidth=2, label=f"{method} Pareto"
        )

    ax.legend()

    # Save
    plot_path = output_dir / "pareto_frontier.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved pareto plot to {plot_path}")
    plt.show()


def main(args):
    # Find and load latest checkpoint
    result_dir = find_latest_checkpoint()
    model, tokenizer, config, dirs_pca = load_checkpoint(result_dir)

    # Set eval config
    if args.quick:
        config.eval_max_dilemmas = 64
        config.eval_batch_size = 4
        config.max_samples = 256  # Limit for faster PCA in quick mode
        logger.critical("Quick mode: small eval set for testing")

    # Coefficients: start with coarse, then finer logspace
    if args.quick:
        coeffs = np.array([10, 5, 1, 0.5, 0.1, 0.01, 0])  # Fewer for quick
    else:
        coarse_coeffs = np.array([100, 15, 5, 2, 0.5, 0.1, 0.01, 0])  # Include baseline
        fine_coeffs = np.logspace(-2, 2, num=20)  # 0.01 to 100, 20 points
        coeffs = np.sort(
            np.unique(np.concatenate([coarse_coeffs, fine_coeffs]))
        )  # Combine and sort
    logger.critical(f"Testing {len(coeffs)} coefficients: {coeffs}")

    # Run fine-grained sweeps
    results_ipissa, full_res_ipissa = fine_grained_sweep(
        model, tokenizer, config, dirs_pca, coeffs, "InnerPiSSA"
    )
    results_pca, full_res_pca = fine_grained_sweep(
        model, tokenizer, config, dirs_pca, coeffs, "PCA"
    )

    # Combine results
    all_results = pd.concat([results_ipissa, results_pca])

    # Create output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = proj_root / f"outputs/fine_grained_eval_{ts}"
    output_dir.mkdir(exist_ok=True)

    # Save results
    all_results.to_parquet(output_dir / "fine_grained_results.parquet", index=False)
    full_res_ipissa.to_parquet(output_dir / "full_results_ipissa.parquet", index=False)
    full_res_pca.to_parquet(output_dir / "full_results_pca.parquet", index=False)

    # Generate summary table
    combined_full = pd.concat([full_res_ipissa, full_res_pca])
    md_table, df_eff_sz, main_score = format_results_table(
        combined_full, target_col="logscore_Virtue/Truthfulness", config=config
    )
    with open(output_dir / "summary.md", "w") as f:
        f.write(md_table)
    logger.info(f"Summary table saved. Main score: {main_score:.3f}")

    # Plot pareto frontier
    plot_pareto_frontier(results_ipissa, results_pca, output_dir)

    # Coherence and transfer metrics
    coherence = compute_coherence_metrics(combined_full)
    coherence.to_parquet(output_dir / "coherence_metrics.parquet")
    transfer = compute_transfer_summary(combined_full)
    transfer.to_parquet(output_dir / "transfer_summary.parquet")

    logger.success(f"Fine-grained evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-grained evaluation of steering methods"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with small eval set"
    )
    args = parser.parse_args()
    main(args)
