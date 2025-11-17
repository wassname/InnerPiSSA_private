# %%
import json
from pathlib import Path

import cattrs
import matplotlib.cm as cm
import matplotlib.pyplot as plt  # Missing import
import numpy as np
import pandas as pd
import torch
from anycache import anycache
from loguru import logger
from matplotlib.colors import LogNorm  # For log colorbar
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from ipissa import ControlVector
from ipissa.peft_utils.adapter_scaling import ScaleAdapter
from ipissa.control import steer
from ipissa.eval import get_choice_ids
from ipissa.extract import _collect_activations_only, read_representations
from ipissa.train.daily_dilemas import (
    evaluate_daily_dilemma,
    format_results_table,
    load_and_process_daily_dilemmas_eval_dataset,
    load_labels,
    process_daily_dilemma_results,
    select_dilemma_by_values,
)
from ipissa.train.train_adapter import PeftModel as PeftModelLoader
from ipissa.train.train_adapter import (
    TrainingConfig,
    clear_mem,
    create_train_dataset,
    get_loss_layers,
    load_model,
    proj_root,
    register_ipissa_peft,
    setup_logging,
)

setup_logging()

# Tunable parameters
strengths = [0] + [
    10**i for i in range(-2, 4)
]  # Exact 10**N: [0, 0.01, 0.1, 1, 10, 100, 1000]
strengths_pos = [0.0] + [2**i for i in range(-6, 12)]
coeffs = strengths_pos + [-s for s in strengths_pos[1:]]
print(f"Using coeffs: {coeffs}")

# TODO make this a cli argument
quick_eval_n = None  # Number of eval points for speed


last_result_dir = Path(
    proj_root / "outputs/adapters/honest_contrastive_ipissa_20251113_071506"
)

if last_result_dir is None:
    # Find the last checkpoint
    results_dirs = sorted((proj_root / "./outputs/adapters/").glob("*"))
    last_result_dir = results_dirs[-1]
    print(f"Using last checkpoint: {last_result_dir.name}")

# Load config
d = json.loads((last_result_dir / "training_config.json").read_text())
config = cattrs.structure(d, TrainingConfig)
model_id = config.model_name

# Set quick eval mode
config.eval_max_n_dilemmas = quick_eval_n
config.eval_batch_size = 12  # Keep small to avoid OOM

# Load base model and tokenizer
base_model, tokenizer = load_model(model_id, quantization_type=config.quantization_type)

# Register InnerPiSSA
register_ipissa_peft()

# Load model with adapter
model = PeftModelLoader.from_pretrained(
    base_model, last_result_dir, adapter_name=config.dataset_name
)


@anycache(".anycache")
def get_acts(checkpoint_name, dataset_len):
    # Load activations from cache or compute them
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with ScaleAdapter(model, coeff=None):
            train_strs = [s for ex in train_honest for s in (ex.positive, ex.negative)]
            last_act, logprobs = _collect_activations_only(
                model, tokenizer, train_strs, loss_layers, batch_size=2
            )
    dirs_pca_dict = read_representations(last_act, logprobs, grads=None)
    return dirs_pca_dict


# Extract directions for baselines
train_honest, _, _, _ = create_train_dataset(
    config, tokenizer, max_size=config.dataset_max_samples
)
loss_layers = get_loss_layers(model, config)
print(f"loss_layers: {loss_layers}")
model.eval()
dirs_pca_dict = get_acts(last_result_dir.name, dataset_len=len(train_honest))

dirs_pca = ControlVector(model_type=model.config.model_type, directions=dirs_pca_dict)
dirs_random = ControlVector(
    model_type=model.config.model_type,
    directions={k: torch.randn_like(v) for k, v in dirs_pca.directions.items()},
)
for k in dirs_random.directions:
    dirs_random.directions[k] = (
        dirs_random.directions[k] / dirs_random.directions[k].norm()
    )

# Debug: check PCA directions
print(f"\nPCA directions extracted for layers: {list(dirs_pca.directions.keys())}")
for k, v in list(dirs_pca.directions.items())[:3]:  # Show first 3
    print(
        f"  {k}: shape={v.shape}, norm={v.norm():.4f}, mean={v.mean():.4f}, std={v.std():.4f}"
    )

logger.info("Extracted PCA and random directions for baselines")
clear_mem()

# Prepare eval dataset once
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

# Load per-model prompting baseline
model_safe = config.model_name.replace('/', '_')
output_path = proj_root / "outputs" / f"prompting_baseline_{model_safe}.parquet"
df_prompting_raw = pd.DataFrame()
if output_path.exists():
    logger.info(f"Loading prompting baseline results from {output_path}")
    df_prompting_raw = pd.read_parquet(output_path)
df_prompting_wlabels, _ = process_daily_dilemma_results(
    df_prompting_raw, dataset_dd, df_labels
)


@anycache(".anycache")
def cache_evaluate_daily_dilemma(dataset_len, coeff, method):
    # we don't want to cache based on batch size, so it's removed from the signature
    with torch.no_grad():
        return evaluate_daily_dilemma(
            model, dataset_dd_pt, tokenizer, choice_ids, batch_size=eval_batch_size
        )


# Evaluate InnerPiSSA over coeffs
all_raw_inner = []
try:
    for c in tqdm(coeffs, desc="Evaluating InnerPiSSA"):
        model.eval()
        clear_mem()
        results_single = []
        with ScaleAdapter(model, coeff=c):
            d_single = cache_evaluate_daily_dilemma(
                len(dataset_dd_pt), coeff=c, method="InnerPiSSA (ours)"
            )
            d_single["coeff"] = c
            d_single["method"] = "InnerPiSSA (ours)"
            results_single.append(d_single)
        df_res2_single = pd.concat(results_single)
        all_raw_inner.append(df_res2_single)
except KeyboardInterrupt:
    logger.warning("Evaluation interrupted during InnerPiSSA evaluation.")
df_raw_inner = pd.concat(all_raw_inner)
df_wlabels_inner, _ = process_daily_dilemma_results(df_raw_inner, dataset_dd, df_labels)


try:
    # Evaluate PCA baseline
    all_raw_pca = []
    for c in tqdm(coeffs, desc="Evaluating PCA"):
        model.eval()
        clear_mem()
        results_single = []
        with steer(model, dirs_pca, coeff=c, retain_grad=False):
            d_single = cache_evaluate_daily_dilemma(
                len(dataset_dd_pt), coeff=c, method="PCA (baseline)"
            )
            d_single["coeff"] = c
            d_single["method"] = "PCA (baseline)"
            results_single.append(d_single)
        df_res2_single = pd.concat(results_single)
        all_raw_pca.append(df_res2_single)

        # # Debug: check first few results
        # if c in [0, 1, -1]:
        #     print(f"\nPCA coeff={c}: sample results")
        #     print(df_res2_single[['dilemma_idx', 'choice', 'logprob']].head(3))
except KeyboardInterrupt:
    logger.warning("Evaluation interrupted during PCA evaluation.")
df_raw_pca = pd.concat(all_raw_pca)
df_wlabels_pca, _ = process_daily_dilemma_results(df_raw_pca, dataset_dd, df_labels)

# Debug: check processed PCA results
print(f"\nPCA processed results shape: {df_wlabels_pca.shape}")
if "coeff" in df_wlabels_pca.columns:
    print(f"PCA coeffs: {sorted(df_wlabels_pca['coeff'].unique())}")
print(df_wlabels_pca[["method", "coeff"]].value_counts())

# try:
#     # Evaluate random baseline
#     all_raw_random = []
#     for c in tqdm(coeffs, desc="Evaluating random"):
#         model.eval()
#         clear_mem()
#         results_single = []
#         with steer(model, dirs_random, coeff=c, retain_grad=False):
#             d_single = cache_evaluate_daily_dilemma(
#                 len(dataset_dd_pt), coeff=c, method="random"
#             )
#             d_single["coeff"] = c
#             d_single["method"] = "random"
#             results_single.append(d_single)
#         df_res2_single = pd.concat(results_single)
#         all_raw_random.append(df_res2_single)
# except KeyboardInterrupt:
#     logger.warning("Evaluation interrupted during random evaluation.")
# df_raw_random = pd.concat(all_raw_random)
# df_wlabels_random, _ = process_daily_dilemma_results(df_raw_random, dataset_dd, df_labels)

# Combine all
df_all_wlabels = pd.concat(
    [df_wlabels_inner, df_wlabels_pca, df_prompting_wlabels], ignore_index=True
)

# Debug: check combined data before formatting
print(f"\nCombined data before formatting:")
print(f"  Total rows: {len(df_all_wlabels)}")
print(f"  Methods: {df_all_wlabels['method'].value_counts()}")
if "coeff" in df_all_wlabels.columns:
    for method in df_all_wlabels["method"].unique():
        df_m = df_all_wlabels[df_all_wlabels["method"] == method]
        print(f"  {method} coeffs: {sorted(df_m['coeff'].unique())}")

# Format
md, df_all_formatted, main_score = format_results_table(
    df_all_wlabels, target_col="logscore_Virtue/Truthfulness", config=config
)
df_all = df_all_formatted.reset_index()
df_all["checkpoint"] = last_result_dir.name

# Debug: check after formatting
print(f"\nAfter formatting:")
print(f"  Total rows: {len(df_all)}")
print(f"  Methods: {df_all['Method'].value_counts()}")
print(f"\nFirst few rows of each method:")
for method in df_all["Method"].unique():
    print(f"\n{method}:")
    df_m = df_all[df_all["Method"] == method]
    cols_to_show = ["Method", "Output Quality\nΔ NLL ↓", "Target Effect\nΔ Truth ↑"]
    if "Coeff\n±" in df_m.columns:
        cols_to_show.insert(1, "Coeff\n±")
    elif "coeff" in df_m.columns:
        cols_to_show.insert(1, "coeff")
    print(df_m[cols_to_show].head(5))

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
output_dir = proj_root / "outputs" / "pareto_analysis"
output_dir.mkdir(exist_ok=True, parents=True)

# Colors and styles per method
method_colors = {
    "InnerPiSSA (ours)": "purple",
    "PCA (baseline)": "blue",
    "random": "red",
    "prompting": "green",  # For prompting baseline
}
method_styles = {
    "InnerPiSSA (ours)": "-",  # solid
    "PCA (baseline)": "--",  # dashed
    "random": ":",  # dotted
    "prompting": "-.",  # dash-dot for single point
}

# Global norm for colorbar (log scale, handle zero)
if "Coeff\n±" in df_all.columns:
    all_abs_coeffs = np.abs(df_all["Coeff\n±"].values)
elif "coeff" in df_all.columns:
    all_abs_coeffs = np.abs(df_all["coeff"].values)
else:
    1 / 0  # we should not make them up
    all_abs_coeffs = np.arange(len(df_all))
vmin = min([x for x in all_abs_coeffs if x > 0]) if any(all_abs_coeffs > 0) else 1
vmax = all_abs_coeffs.max() if all_abs_coeffs.max() > 0 else 1
norm = LogNorm(vmin=vmin, vmax=vmax)
cmap = cm.RdYlGn_r  # Green low strength, red high strength

# Debug: print unique methods and data ranges
print(f"Unique methods in data: {df_all['Method'].unique()}")
for method in df_all["Method"].unique():
    count = len(df_all[df_all["Method"] == method])
    print(f"  {method}: {count} points")

# Debug: check data ranges
col_x = "Output Quality\nΔ NLL ↓"
col_y = "Target Effect\nΔ Truth ↑"
print("\nData ranges:")
print(
    f"  Output Quality (Δ NLL): {df_all[col_x].min():.4f} to {df_all[col_x].max():.4f}"
)
print(
    f"  Target Effect (Δ Truth): {df_all[col_y].min():.4f} to {df_all[col_y].max():.4f}"
)
print(f"  NaN in Output Quality: {df_all[col_x].isna().sum()}")
print(f"  NaN in Target Effect: {df_all[col_y].isna().sum()}")
print(f"  Inf in Output Quality: {np.isinf(df_all[col_x]).sum()}")
print(f"  Inf in Target Effect: {np.isinf(df_all[col_y]).sum()}")

# Group and plot lines
for method in df_all["Method"].unique():
    df_method = df_all[df_all["Method"] == method].copy()
    if df_method.empty:
        logger.warning(f"No data for method {method}, skipping.")
        continue

    # Sort by increasing degradation (Δ NLL)
    df_method = df_method.sort_values("Output Quality\nΔ NLL ↓")

    # Get coeffs for coloring
    if "Coeff\n±" in df_method.columns:
        coeffs = df_method["Coeff\n±"].values
    elif "coeff" in df_method.columns:
        coeffs = df_method["coeff"].values
    else:
        1 / 0  # we should not make them up
        coeffs = np.arange(len(df_method))
    abs_coeffs = np.abs(coeffs)

    line_color = method_colors[method]
    line_style = method_styles[method]

    if len(df_method) > 1:
        # Line plot for multi-point methods (no label, we'll add manually later)
        ax.plot(
            df_method["Output Quality\nΔ NLL ↓"],
            df_method["Target Effect\nΔ Truth ↑"],
            color=line_color,
            linestyle=line_style,
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=6,
        )

    # Colored markers with line-matching border (always, even for single points)
    for i, (_, row) in enumerate(df_method.iterrows()):
        x = row["Output Quality\nΔ NLL ↓"]
        y = row["Target Effect\nΔ Truth ↑"]
        abs_c = abs_coeffs[i]
        color_val = abs_c if abs_c > 0 else vmin  # Fallback for zero
        ax.scatter(
            x,
            y,
            c=color_val,
            cmap=cmap,
            norm=norm,
            s=120,  # Larger for visibility
            edgecolors=line_color,
            linewidth=1.5,  # Thicker outline
            zorder=3,
        )

# Add text labels for key coefficients (deduplicated)
labeled_positions = set()  # Track (x, y) to avoid duplicates
for method in df_all["Method"].unique():
    df_method = df_all[df_all["Method"] == method].copy()
    df_method = df_method.sort_values("Output Quality\nΔ NLL ↓")

    if "Coeff\n±" in df_method.columns:
        coeffs = df_method["Coeff\n±"].values
    elif "coeff" in df_method.columns:
        coeffs = df_method["coeff"].values
    else:
        continue

    for i, (_, row) in enumerate(df_method.iterrows()):
        x = row["Output Quality\nΔ NLL ↓"]
        y = row["Target Effect\nΔ Truth ↑"]

        # Only label coeff=0 and coeff=±1, and avoid duplicates
        if abs(coeffs[i]) in [0, 1]:
            pos_key = (
                round(x, 6),
                round(y, 6),
            )  # Round to avoid float precision issues
            if pos_key not in labeled_positions:
                labeled_positions.add(pos_key)
                label_text = (
                    f"{coeffs[i]:+.1f}" if abs(coeffs[i]) >= 1 else f"{coeffs[i]:+.2f}"
                )
                # Offset to the right and slightly up
                x_offset = x * 1.15  # 15% to the right in log space
                y_offset = y + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 2% up
                ax.text(
                    x_offset,
                    y_offset,
                    label_text,
                    fontsize=9,
                    color="black",
                    ha="left",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )

# Add colorbar for steering strength
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("Steering Strength (log |coeff|)")

ax.set_xlabel("Coherence Degradation (Δ NLL ↓)", fontsize=13)
ax.set_ylabel("Target Effect (Δ Truth ↑)", fontsize=13)
ax.set_title(
    f"Pareto Frontier for {last_result_dir.name} (InnerPiSSA on {config.model_name})\n"
    f"Effect vs Coherence Trade-off (Quick Mode, {quick_eval_n} points)",
    fontsize=15,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)

# Set axis limits explicitly, filtering out NaN/inf
col_x = "Output Quality\nΔ NLL ↓"
col_y = "Target Effect\nΔ Truth ↑"
valid_x = df_all[col_x].replace([np.inf, -np.inf], np.nan).dropna()
valid_y = df_all[col_y].replace([np.inf, -np.inf], np.nan).dropna()
if len(valid_x) > 0 and len(valid_y) > 0:
    x_min, x_max = valid_x.min(), valid_x.max()
    y_min, y_max = valid_y.min(), valid_y.max()

    # Add 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Only use log scale if all x values are positive
    if x_min > 0:
        ax.set_xscale("log")
    print(
        f"\nPlot limits set: x=[{x_min:.4f}, {x_max:.4f}], y=[{y_min:.4f}, {y_max:.4f}]"
    )
else:
    print("\nWARNING: No valid data points to plot!")

ax.set_xscale("log")

# Manual legend with only method names (no automatic legend from scatter points)
legend_elements = [
    Line2D(
        [0],
        [0],
        color="purple",
        linestyle="-",
        linewidth=2,
        marker="o",
        label="InnerPiSSA (ours)",
    ),
    Line2D(
        [0],
        [0],
        color="blue",
        linestyle="--",
        linewidth=2,
        marker="o",
        label="PCA (baseline)",
    ),
    Line2D(
        [0],
        [0],
        color="green",
        linestyle="-.",
        linewidth=2,
        marker="o",
        label="prompting",
    ),
]
ax.legend(handles=legend_elements, fontsize=11, loc="lower right", framealpha=0.95)

# Save plot
plot_path = output_dir / f"pareto_frontier_single_checkpoint_{last_result_dir.name}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {plot_path}")

plt.show()

# Save combined results
results_path = (
    output_dir / f"eval_results_single_checkpoint_{last_result_dir.name}.parquet"
)
df_all.to_parquet(results_path)
print(f"Saved results to {results_path}")
