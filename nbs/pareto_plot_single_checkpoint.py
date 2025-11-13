# %%
import cattrs
import pandas as pd
from pathlib import Path
from repeng.train.daily_dilemas import (
    format_results_table, evaluate_daily_dilemma, 
    load_and_process_daily_dilemmas_eval_dataset, 
    load_labels, select_dilemma_by_values, process_daily_dilemma_results
)
from repeng.eval import get_choice_ids
from repeng.train.train_adapter import (
    proj_root, TrainingConfig, load_model, register_ipissa_peft, 
    PeftModel as PeftModelLoader, clear_mem, setup_logging,
    create_dataset, get_loss_layers
)
from repeng.adapter import ScaleAdapter
from transformers import AutoTokenizer, GenerationConfig
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.cm as cm
from anycache import anycache
import json
from loguru import logger
import matplotlib.pyplot as plt  # Missing import
from repeng.control import steer
from repeng.extract import _collect_activations_only, read_representations
from repeng import ControlVector
from matplotlib.colors import LogNorm  # For log colorbar


setup_logging()

# Tunable parameters
strengths = [0] + [10**i for i in range(-2, 4)]  # Exact 10**N: [0, 0.01, 0.1, 1, 10, 100, 1000]
strengths_pos = [0.0] + [2**i for i in range(-6, 12)]
coeffs = strengths_pos + [-s for s in strengths_pos[1:]]
print(f"Using coeffs: {coeffs}")

# TODO make this a cli argument
quick_eval_n = None  # Number of eval points for speed


last_result_dir = Path(proj_root/"outputs/adapters/honest_contrastive_ipissa_20251113_071506")

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

# Extract directions for baselines
train_honest, _, _, _ = create_dataset(config, tokenizer, max_size=config.dataset_max_samples)
loss_layers = get_loss_layers(model, config)
model.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    with ScaleAdapter(model, coeff=None):
        train_strs = [s for ex in train_honest for s in (ex.positive, ex.negative)]
        last_act, logprobs = _collect_activations_only(
            model, tokenizer, train_strs, loss_layers, batch_size=2
        )
dirs_pca_dict = read_representations(last_act, logprobs, grads=None)
dirs_pca = ControlVector(model_type=model.config.model_type, directions=dirs_pca_dict)
dirs_random = ControlVector(
    model_type=model.config.model_type,
    directions={k: torch.randn_like(v) for k, v in dirs_pca.directions.items()},
)
for k in dirs_random.directions:
    dirs_random.directions[k] = dirs_random.directions[k] / dirs_random.directions[k].norm()
logger.info("Extracted PCA and random directions for baselines")
clear_mem()

# Prepare eval dataset once
dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
    tokenizer, max_size=config.eval_dataset_max_token_length
)
if config.eval_max_n_dilemmas is not None:
    dataset_dd = select_dilemma_by_values(
        dataset_dd, label="truth", top_N=config.eval_max_n_dilemmas
    )
dataset_dd_pt = dataset_dd.select_columns(
    ["dilemma_idx", "idx", "input_ids"]
).with_format("torch")
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

# Load prompting baseline
output_path = proj_root / "outputs" / "prompting_baseline.parquet"
df_prompting_raw = pd.DataFrame()
if output_path.exists():
    logger.info(f"Loading prompting baseline results from {output_path}")
    df_prompting_raw = pd.read_parquet(output_path)
    df_prompting_raw = df_prompting_raw[df_prompting_raw['model_id'].isin([config.model_name])]
df_prompting_wlabels, _ = process_daily_dilemma_results(df_prompting_raw, dataset_dd, df_labels)


@anycache('.anycache')
def cache_evaluate_daily_dilemma(
    dataset_len, coeff, method):
    # we don't want to cache based on batch size, so it's removed from the signature
    with torch.no_grad():
        return evaluate_daily_dilemma(
            model, dataset_dd_pt, tokenizer, choice_ids,
            batch_size=eval_batch_size
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
                len(dataset_dd_pt),
                coeff=c, method="InnerPiSSA (ours)"
            )
            d_single["coeff"] = c
            d_single["method"] = "InnerPiSSA (ours)"
            results_single.append(d_single)
        df_res2_single = pd.concat(results_single)
        all_raw_inner.append(df_res2_single)
    df_raw_inner = pd.concat(all_raw_inner)
    df_wlabels_inner, _ = process_daily_dilemma_results(df_raw_inner, dataset_dd, df_labels)
except KeyboardInterrupt:
    logger.warning("Evaluation interrupted during InnerPiSSA evaluation.")


try:
    # Evaluate PCA baseline
    all_raw_pca = []
    for c in tqdm(coeffs, desc="Evaluating PCA"):
        model.eval()
        clear_mem()
        results_single = []
        with steer(model, dirs_pca, coeff=c, retain_grad=False):
            d_single = cache_evaluate_daily_dilemma(
                len(dataset_dd_pt),
                coeff=c, method="PCA (baseline)"
            )
            d_single["coeff"] = c
            d_single["method"] = "PCA (baseline)"
            results_single.append(d_single)
        df_res2_single = pd.concat(results_single)
        all_raw_pca.append(df_res2_single)
    df_raw_pca = pd.concat(all_raw_pca)
    df_wlabels_pca, _ = process_daily_dilemma_results(df_raw_pca, dataset_dd, 
    df_labels)
except KeyboardInterrupt:
    logger.warning("Evaluation interrupted during PCA evaluation.")

try:
    # Evaluate random baseline
    all_raw_random = []
    for c in tqdm(coeffs, desc="Evaluating random"):
        model.eval()
        clear_mem()
        results_single = []
        with steer(model, dirs_random, coeff=c, retain_grad=False):
            d_single = evaluate_daily_dilemma(
                len(dataset_dd_pt), coeff=c, method="random"
            )
            d_single["coeff"] = c
            d_single["method"] = "random"
            results_single.append(d_single)
        df_res2_single = pd.concat(results_single)
        all_raw_random.append(df_res2_single)
    df_raw_random = pd.concat(all_raw_random)
    df_wlabels_random, _ = process_daily_dilemma_results(df_raw_random, dataset_dd, df_labels)
except KeyboardInterrupt:
    logger.warning("Evaluation interrupted during random evaluation.")

# Combine all
df_all_wlabels = pd.concat([
    df_wlabels_inner, df_wlabels_pca, df_wlabels_random, df_prompting_wlabels
], ignore_index=True)

# Format
md, df_all_formatted, main_score = format_results_table(
    df_all_wlabels, target_col='logscore_Virtue/Truthfulness', config=config
)
df_all = df_all_formatted.reset_index()
df_all['checkpoint'] = last_result_dir.name

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
output_dir = proj_root / "outputs" / "pareto_analysis"
output_dir.mkdir(exist_ok=True, parents=True)

# Colors and styles per method
method_colors = {
    'InnerPiSSA (ours)': 'purple',
    'PCA (baseline)': 'blue',
    'random': 'red',
    'prompting': 'green',  # For prompting baseline
}
method_styles = {
    'InnerPiSSA (ours)': '-',  # solid
    'PCA (baseline)': '--',   # dashed
    'random': ':',            # dotted
    'prompting': '-.',        # dash-dot for single point
}

# Global norm for colorbar (log scale, handle zero)
if 'Coeff\n±' in df_all.columns:
    all_abs_coeffs = np.abs(df_all['Coeff\n±'].values)
elif 'coeff' in df_all.columns:
    all_abs_coeffs = np.abs(df_all['coeff'].values)
else:
    all_abs_coeffs = np.arange(len(df_all))
vmin = min([x for x in all_abs_coeffs if x > 0]) if any(all_abs_coeffs > 0) else 1
vmax = all_abs_coeffs.max() if all_abs_coeffs.max() > 0 else 1
norm = LogNorm(vmin=vmin, vmax=vmax)
cmap = cm.RdYlGn_r  # Green low strength, red high strength

# Debug: print unique methods
print(f"Unique methods in data: {df_all['Method'].unique()}")

# Group and plot lines
for method in df_all['Method'].unique():
    df_method = df_all[df_all['Method'] == method].copy()
    if df_method.empty:
        continue
    
    # Sort by increasing degradation (Δ NLL)
    df_method = df_method.sort_values('Output Quality\nΔ NLL ↓')
    
    # Get coeffs for coloring
    if 'Coeff\n±' in df_method.columns:
        coeffs = df_method['Coeff\n±'].values
    else:
        coeffs = df_method['coeff'].values if 'coeff' in df_method.columns else np.arange(len(df_method))
    abs_coeffs = np.abs(coeffs)
    
    line_color = method_colors.get(method, 'gray')  # Fallback for unknown methods
    line_style = method_styles.get(method, '-')  # Fallback to solid line
    
    if len(df_method) > 1:
        # Line plot for multi-point methods
        ax.plot(
            df_method['Output Quality\nΔ NLL ↓'],
            df_method['Target Effect\nΔ Truth ↑'],
            label=method,
            color=line_color,
            linestyle=line_style,
            linewidth=2,
            alpha=0.8,
            marker='o',
            markersize=6
        )
    
    # Colored markers with line-matching border (always, even for single points)
    for i, (_, row) in enumerate(df_method.iterrows()):
        x = row['Output Quality\nΔ NLL ↓']
        y = row['Target Effect\nΔ Truth ↑']
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
            zorder=3
        )
        # Add text label only for coeff=0 and coeff=±1
        if abs(coeffs[i]) in [0, 1]:
            label = f"{coeffs[i]:+.1f}" if abs(coeffs[i]) >= 1 else f"{coeffs[i]:+.2f}"
            ax.text(
                x + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),  # Slight offset
                y,
                label,
                fontsize=9,
                color='black',
                ha='left',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )

# Add colorbar for steering strength
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Steering Strength (log |coeff|)')

ax.set_xlabel('Coherence Degradation (Δ NLL ↓)', fontsize=13)
ax.set_ylabel('Target Effect (Δ Truth ↑)', fontsize=13)
ax.set_title(
    f'Pareto Frontier for {last_result_dir.name} (InnerPiSSA on {config.model_name})\n'
    f'Effect vs Coherence Trade-off (Quick Mode, {quick_eval_n} points)', 
    fontsize=15, fontweight='bold'
)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Legend
ax.legend(fontsize=11, loc='best', framealpha=0.9)

# Save plot
plot_path = output_dir / f"pareto_frontier_single_checkpoint_{last_result_dir.name}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_path}")

plt.show()

# Save combined results
results_path = output_dir / f"eval_results_single_checkpoint_{last_result_dir.name}.parquet"
df_all.to_parquet(results_path)
print(f"Saved results to {results_path}")
