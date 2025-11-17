#!/usr/bin/env python3
"""Test repeng baseline (my fork, which incldues S space steering and PCA but with baukit hook on modules) on our training data and Daily Dilemmas eval.

Tests the original vgel/repeng library (PCA-based control vectors) on our
honest/dishonest training data and evaluates on Daily Dilemmas.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ipissa.control import model_layer_list, steer
from ipissa.extract import ControlVector
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import tyro
from ipissa.extract import _collect_activations_only, read_representations
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
    match_linear_layers,
)

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace('/', '_')


def collect_S_steer_vec(model, honest_dataset, tokenizer, loss_layers, config):
    """Collect a steering vector for S-space steering (like PiSSA)."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        train_strs = [s for ex in honest_dataset for s in (ex.positive, ex.negative)]
        last_act, logprobs = _collect_activations_only(
            model, tokenizer, train_strs, loss_layers, batch_size=64
        )

    Sw_dirs = {}


    for layer in tqdm(loss_layers, desc='svd'):
        m = model.get_submodule(layer)
        W = m.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T.cpu()
        U = U.cpu()
        S = S.cpu()

        # 2. S-weighted direction for steering application
        sqrt_S = torch.sqrt(S)  # [r]
        U_scaled = U * sqrt_S  # [d_out, r], element-wise: U_ij * sqrt(S_j)
        V_scaled = V * sqrt_S  # [d_in, r], element-wise: V_ij * sqrt(S_j)

        # project hs to S-space
        hs_cpu = last_act[layer].float()
        hs_s_weighted = hs_cpu @ U_scaled  # [n, r] S-weighted projection
        h_cho_sw = hs_s_weighted[::2]
        h_rej_sw = hs_s_weighted[1::2]
        delta_s_steer = (h_cho_sw - h_rej_sw).mean(dim=0)  # [r] S-weighted
        delta_s_steer = F.normalize(delta_s_steer, dim=0)

        # Make steering vector
        Sw_dirs[layer] = {
            "U_scaled": U_scaled,  # U * sqrt(S)
            "delta_s": delta_s_steer,
            "V_scaled": V_scaled,  # V * sqrt(S)
        }
    cvec_Sw_steer = ControlVector(
        model_type=model.config.model_type, directions=Sw_dirs
    )

    # also do a normal PCA steer
    dirs_pca = read_representations(last_act, logprobs, grads=None)
    cvec_pca_steer = ControlVector(
        model_type=model.config.model_type, directions=dirs_pca
    )
    return cvec_Sw_steer, cvec_pca_steer



def main(config):
    # Config
    config.eval_batch_size = max(32, config.batch_size)
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

    results = []

    for model_name in tqdm(_EVAL_BASELINE_MODELS, desc="Evaluating models"):
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
        cache_path = Path(proj_root) / "outputs" / f"baselines/wassname_repeng/{model_safe}.parquet"
        cache_path.parent.mkdir(exist_ok=True, parents=True)

        if cache_path.exists():
            logger.info(f"Loading cached results from {cache_path}")
            df_cached = pd.read_parquet(cache_path)
            results.append(df_cached)
            continue

        # No cache, evaluate the model
        logger.info(f"No cache found for {model_name}, evaluating...")
        base_model, tokenizer = load_model(model_name, quantization_type=config.quantization_type)

        # For our custom one we need baukit layer paths, and we can't do whole layers, but we can do the output modules of a layer (or all)

        # FIXME also do the U-steering here, but on many layers


        # repeng uses layers relative to end, and the last half
        # note that my adapter uses 3 layers, which is differen't
        try:
            N = base_model.config.num_hidden_layers
        except AttributeError:
            # gemma models don't have config.num_hidden_layers
            # print(base_model)
            N = len(model_layer_list(base_model))
        layer_nums = list(range(-5, -N // 2, -1))  # last half layers
        layer_nums = [n % N for n in layer_nums]  # convert to positive indices
        if config.quick:
            layer_nums = layer_nums[:2]

        repeng_layers = match_linear_layers(base_model, layer_nums, module_names=[".*"])
        logger.info(f"Matched {len(repeng_layers)} repeng layers for model {model_name}")

        # Convert models to baukit

        # model = ControlModel(base_model, repeng_layers)
        model = base_model

        logger.info(f"Loaded model: {model_name}, repeng layers: {repeng_layers}")

        train_honest, train_dataset_pt, val_honest, val_dataset_pt = create_train_dataset(
            config,
            tokenizer,
            max_size=config.dataset_max_samples
        )

        logger.info(f"Created dataset with {len(train_honest)} pairs")

        cvec_Sw_steer, cvec_pca_steer = collect_S_steer_vec(model, train_honest, tokenizer, repeng_layers, config)

        # # Train control vector
        # logger.info("Training control vector with repeng...")
        # control_vector = ControlVector.train(
        #     model,
        #     tokenizer,
        #     train_honest,
        #     batch_size=config.eval_batch_size,
        #     hidden_layers=repeng_layers,
        # )
        # logger.info("Control vector trained")

        # model.reset()

        choice_ids = get_choice_ids(tokenizer)

        # Quick test
        logger.info("Quick test of S-steering vectors...")
        for coeff in [-1.0, 0.0, 1.0]:
            with steer(model, cvec_Sw_steer, coeff):
                (q, a, score, seq_nll) = generate_example_output(
                    model,
                    tokenizer,
                    choice_ids=choice_ids,
                    max_new_tokens=32,
                )
                logger.info(f"S-Steer: Coeff={coeff:+.1f}, score={score:.3f}, nll={seq_nll:.3f}")


        # Quick test
        logger.info("Quick test of PCA A-steering vectors...")
        for coeff in [-1.0, 0.0, 1.0]:
            with steer(model, cvec_pca_steer, coeff):
                (q, a, score, seq_nll) = generate_example_output(
                    model,
                    tokenizer,
                    choice_ids=choice_ids,
                    max_new_tokens=128,
                )
                logger.info(f"PCA: Coeff={coeff:+.1f}, score={score:.3f}, nll={seq_nll:.3f}")

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
        model_results = []

        for coeff in [-1.0, 0.0, 1.0]:
            logger.info(f"Evaluating repeng at coeff={coeff}")

            # Set control vector
            with steer(model, cvec_Sw_steer, coeff):
                # Evaluate
                d = evaluate_daily_dilemma(
                    model,
                    dataset_dd_pt,
                    tokenizer,
                    choice_ids,
                    batch_size=config.eval_batch_size,
                )

            d["model_id"] = model_name
            d["coeff"] = coeff
            d["method"] = "S-space steer"
            model_results.append(d)

        for coeff in [-1.0, 0.0, 1.0]:
            logger.info(f"Evaluating repeng at coeff={coeff}")

            # Set control vector
            with steer(model, cvec_pca_steer, coeff):
                # Evaluate
                d = evaluate_daily_dilemma(
                    model,
                    dataset_dd_pt,
                    tokenizer,
                    choice_ids,
                    batch_size=config.eval_batch_size,
                )

            d["model_id"] = model_name
            d["coeff"] = coeff
            d["method"] = "pca (wassname)"
            model_results.append(d)

        # Save per-model cache immediately after evaluation
        df_model = pd.concat(model_results)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df_model.to_parquet(cache_path)
        logger.info(f"Saved results to {cache_path}")
        results.append(df_model)

        # Clean up model from memory
        del base_model, tokenizer, model, cvec_Sw_steer, cvec_pca_steer
        gc.collect()
        torch.cuda.empty_cache()

    # Combine all results and show summary
    df_all = pd.concat(results, ignore_index=True)
    logger.info(f"Total results: {len(df_all)} rows from {len(df_all['model_id'].unique())} models")

    # Process and display results for each model
    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer,
        instructions="",
        max_tokens=config.eval_dataset_max_token_length,
    )
    df_labels = load_labels(dataset_dd)
    df_labeled = process_daily_dilemma_results(df_all, dataset_dd, df_labels)[0]

    for model_name in _EVAL_BASELINE_MODELS:
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
    # config = tyro.cli(TrainingConfig, use_underscores=True)
    # if __name__ == "__main__":
    from ipissa.config import default_configs
    config = tyro.cli(TrainingConfig, use_underscores=True  )
    main(config)