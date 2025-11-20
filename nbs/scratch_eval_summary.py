# %%
import pandas as pd
from pathlib import Path
from ipissa.train.daily_dilemas import format_results_table, compute_coherence_metrics
import cattrs
import json
from adjustText import adjust_text
from ipissa.train.train_adapter import proj_root, TrainingConfig
from tabulate import tabulate

# TODO get last that has results
print(f"proj_root: {proj_root}")
results_dirs = sorted(( proj_root / "./outputs/adapters/").glob("*"))
result_dir = None
for _result_dir in results_dirs:
    if (_result_dir / "eval_results.parquet").exists():
        result_dir = _result_dir
        print(_result_dir)
if result_dir is None:
    raise ValueError("No results found in outputs/adapters/")


results_dir = proj_root / "./outputs/adapters/q4b-raw-r256-lr1e-2_20251120_045911"
# /workspace/InnerPiSSA_private/outputs/adapters/q4b-raw-r256_20251120_005219

# f = 
# or is it "eval_summary.parquet"?
df_res_wlabels = pd.read_parquet(result_dir / "eval_results.parquet")
df_res_pv = pd.read_parquet(result_dir / "eval_summary.parquet")

d = json.loads((result_dir / "training_config.json").read_text())
config = cattrs.structure(d, TrainingConfig)
print(f"Evaluation results:\n{df_res_pv.round(4)}")
f = result_dir / "eval_summary.csv"
df_res_pv.to_csv(f, index=False, float_format="%.4f")
print(f"Saved eval summary to {f}")
# TODO get the above table, get the max truth telling coeff for each method. Then get the diff. Then summarise

# Optionally load per-model prompting baseline if exists
# NOTE: Disabled for performance - prompting baseline has many rows which slows down compute_transfer_summary
# model_safe = config.model_name.replace('/', '_')
# prompting_path = proj_root / "outputs" / f"prompting_baseline_{model_safe}.parquet"
# if prompting_path.exists():
#     res_prompting = pd.read_parquet(prompting_path)
#     df_res_wlabels = pd.concat([df_res_wlabels, res_prompting], ignore_index=True)
#     print(f"Added prompting baseline ({len(res_prompting)} rows)")



md, tables, s = format_results_table(df_res_wlabels, target_col='logscore_Virtue/Truthfulness', config=config)
print(md)

df = tables['T-stat']

# %%
# New analysis: For each method, show correlation of each moral value with truthfulness
print("\n## Moral Value Correlation with Truthfulness Direction")
print("Shows how each moral value changes as we steer toward truthfulness:\n")

# df_res_pv has moral labels as rows and (method, coeff) as column MultiIndex
target_label = 'Virtue/Truthfulness'

# Get labels with enough samples
cols_labels = [c for c in df_res_wlabels.columns if c.startswith("logscore_")]
num_labels = df_res_wlabels.groupby(["method", "coeff"])[cols_labels].count().iloc[0]
cols_labels = num_labels[num_labels > 50].index

df_res_pv = df_res_wlabels.groupby(["method", "coeff"], dropna=False)[cols_labels].mean().T
df_res_pv.index = [s.lstrip("logscore_") for s in df_res_pv.index]

# Replace NaN with 'disabled'
df_res_pv.columns = pd.MultiIndex.from_frame(df_res_pv.columns.to_frame().fillna('disabled'))

# Reorder so truthfulness at top
df_res_pv = df_res_pv.reindex(
    sorted(
        df_res_pv.index,
        key=lambda x: (
            not x.startswith("Virtue/Truthfulness"),
            not x.startswith("Virtue/"),
            not x.startswith("MFT/"),
            x,
        ),
    ),
    axis=0,
)

# For each method, compute correlation and range statistics
import scipy.stats as stats
import numpy as np

methods = df_res_pv.columns.get_level_values(0).unique()
correlation_results = []

for method in methods:
    method_data = df_res_pv[method]  # All coeffs for this method
    
    # Get available coefficients (excluding 'disabled')
    coeffs = [c for c in method_data.columns if c != 'disabled']
    if len(coeffs) < 2:
        continue
    
    # Get truthfulness values at each coeff
    truth_values = method_data.loc[target_label, coeffs].astype(float)
    
    # For each other moral label, compute correlation with truthfulness
    for label in df_res_pv.index:
        if label == target_label:
            continue
            
        label_values = method_data.loc[label, coeffs].astype(float)
        
        # Compute Pearson correlation
        corr, p_val = stats.pearsonr(truth_values, label_values)
        
        # Compute range (max - min) to show magnitude of change
        value_range = label_values.max() - label_values.min()
        
        # Compute slope via linear regression
        slope, intercept, r_val, p_val_reg, std_err = stats.linregress(truth_values, label_values)
        
        correlation_results.append({
            'Method': method,
            'Moral Value': label,
            'Correlation': corr,
            'p-value': p_val,
            'Slope': slope,
            'Range': value_range,
            'R²': r_val**2,
        })

df_corr = pd.DataFrame(correlation_results)

# For each method, show top correlated values
for method in methods:
    method_df = df_corr[df_corr['Method'] == method].copy()
    if len(method_df) == 0:
        continue
    
    print(f"\n### {method}")
    print("Top 10 most correlated (positive and negative):\n")
    
    # Sort by absolute correlation
    method_df = method_df.sort_values('Correlation', key=abs, ascending=False)
    top_10 = method_df.head(10)[['Moral Value', 'Correlation', 'Slope', 'Range', 'p-value']]
    print(tabulate(top_10, tablefmt='pipe', headers='keys', floatfmt='.3f', showindex=False))

print("\n\n")

# %%
from great_tables import GT, md, html

# Check which columns actually exist in df
print("Available columns in df:", df.columns.tolist())
print("df index name:", df.index.name)

# Create publication-quality table - use actual column names
actual_cols = df.columns.tolist()
has_coeff = "Coeff\n±" in actual_cols

gt_table = (
    GT(df.reset_index())
    .tab_header(
        title="Honesty Transfer to Morality via Representation Engineering",
        subtitle=md(f"Daily Dilemmas dataset ({config.max_samples} train → {config.eval_max_dilemmas or 64} test) | Model: {config.model_name}")
    )
)

# Only add spanners if columns exist
if has_coeff:
    gt_table = gt_table.tab_spanner(
        label="Steering",
        columns=["Method", "Coeff\n±"]
    )

# Check if the expected columns exist, otherwise use actual names
effect_col = next((c for c in actual_cols if "Effect" in c or "↑" in c), actual_cols[0] if actual_cols else None)
transfer_col = next((c for c in actual_cols if "Transfer" in c or "Other" in c), None)
pval_col = "p-value" if "p-value" in actual_cols else None
nll_col = next((c for c in actual_cols if "Degradation" in c or "NLL" in c), None)
gain_col = next((c for c in actual_cols if "Gain" in c), None)

if transfer_col and pval_col:
    gt_table = gt_table.tab_spanner(
        label="Transfer Effects",
        columns=[c for c in [effect_col, transfer_col, pval_col] if c]
    )

if nll_col and gain_col:
    gt_table = gt_table.tab_spanner(
        label="Quality Metrics",
        columns=[c for c in [nll_col, gain_col] if c]
    )

gt_table_html = gt_table


# Save outputs
output_dir = result_dir / "tables"
output_dir.mkdir(exist_ok=True)

# Display in notebook
gt_table_html.show('browser')

# %%
import matplotlib.pyplot as plt

# Extract coefficient from the "Coeff\n±" column if it exists
df_plot = df.copy()
if 'Coeff\n±' in df_plot.columns:
    df_plot['coeff_value'] = df_plot['Coeff\n±']
else:
    # If no coeff column, all methods use the same coeff (probably 1.0)
    df_plot['coeff_value'] = 1.0

# Plot effect vs coherence
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df_plot['Degradation\nΔ NLL ↑'], 
    df_plot['Effect ↑'],
    c=df_plot['coeff_value'],  # Color by coefficient magnitude
    cmap='coolwarm',
    s=150,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Coefficient Magnitude', fontsize=11)

# Use adjustText for non-overlapping labels if available, otherwise use smart positioning

texts = []
for idx, row in df_plot.iterrows():
    text = ax.annotate(
        f"{idx}\n(±{row['coeff_value']:.1f})", 
        (row['Degradation\nΔ NLL ↑'], row['Effect ↑']),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5)
    )
    texts.append(text)

# Adjust text positions to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)

ax.set_xlabel('Coherence Degradation (Δ NLL ↑)', fontsize=12)
ax.set_ylabel('Target Effect (↑)', fontsize=12)
ax.set_title('Effect vs Coherence: Steering Efficiency Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Save the plot
plot_path = output_dir / "effect_vs_coherence.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_path}")

plt.show()

# Save as HTML
gt_table_html.write_raw_html(str(output_dir / "results_table.html"))
print(f"Saved HTML to {output_dir / 'results_table.html'}")

# Save as LaTeX - need to recreate without markdown for latex compatibility
try:
    latex_str = gt_table.as_latex()
    (output_dir / "results_table.tex").write_text(latex_str)
    print(f"Saved LaTeX to {output_dir / 'results_table.tex'}")
except NotImplementedError as e:
    print(f"LaTeX export not available (markdown in subtitle): {e}")
    # Create a simpler version without markdown
    gt_table_simple = (
        GT(df.reset_index())
        .tab_header(
            title="Honesty Transfer to Morality via Representation Engineering",
            subtitle=f"Daily Dilemmas dataset ({config.max_samples} train → {config.eval_max_dilemmas or 64} test) | Model: {config.model_name}"
        )
    )
    if has_coeff:
        gt_table_simple = gt_table_simple.tab_spanner(label="Steering", columns=["Method", "Coeff\n±"])
    if transfer_col and pval_col:
        gt_table_simple = gt_table_simple.tab_spanner(
            label="Transfer Effects",
            columns=[c for c in [effect_col, transfer_col, pval_col] if c]
        )
    if nll_col and gain_col:
        gt_table_simple = gt_table_simple.tab_spanner(
            label="Quality Metrics",
            columns=[c for c in [nll_col, gain_col] if c]
        )
    latex_str = gt_table_simple.as_latex()
    (output_dir / "results_table.tex").write_text(latex_str)
    print(f"Saved LaTeX to {output_dir / 'results_table.tex'}")
