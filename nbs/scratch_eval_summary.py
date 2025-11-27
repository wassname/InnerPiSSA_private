# %%
import pandas as pd
from pathlib import Path
from ipissa.train.daily_dilemas import format_results_table, compute_coherence_metrics
import cattrs
import json
from adjustText import adjust_text
from ipissa.train.train_adapter import proj_root, TrainingConfig
from tabulate import tabulate
import scipy.stats as stats
import numpy as np

# get last that has results
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
# results_dir = proj_root / "./outputs/adapters/q4b-raw-r256-lr1e-1_20251120_050802"
results_dir = proj_root / "./outputs/adapters/q4b-raw-r256_20251120_071852"
results_dir = proj_root / "./outputs/adapters/q4b-raw-r256_20251120_093045"
results_dir = proj_root / "./outputs/adapters/q4b-raw-r256-lr1e-2_20251120_095949"
# results_dir = proj_root / "./outputs/adapters/q4b-raw-r256_20251120_100645"
results_dir = proj_root / "./outputs/adapters/q4b-raw-r256_20251120_001953" # wd=100


# f = ers/
# or is it "eval_summary.parquet"?
df_res_wlabels = pd.read_parquet(result_dir / "eval_results.parquet")
df_res_pv = pd.read_parquet(result_dir / "eval_summary.parquet")

d = json.loads((result_dir / "training_config.json").read_text())
config = cattrs.structure(d, TrainingConfig)
print(config)
print(f"Evaluation results:\n{df_res_pv.round(4)}")
f = result_dir / "eval_summary.csv"
df_res_pv.to_csv(f, index=False, float_format="%.4f")
print(f"Saved eval summary to {f}")
# TODO get the above table, get the max truth telling coeff for each method. Then get the diff. Then summarise

md, tables, s = format_results_table(df_res_wlabels, target_col='logscore_Value/Honesty', config=config)
print(md)

df = tables['T-stat']

# %%
# New analysis: For each method, show correlation of each moral value with truthfulness
print("\n## Moral Value Correlation with Truthfulness Direction")
print("Shows how each moral value changes as we steer toward truthfulness:\n")

# df_res_pv has moral labels as rows and (method, coeff) as column MultiIndex
target_label = 'Value/Honesty'

# Get labels with enough samples
cols_labels = [c for c in df_res_wlabels.columns if c.startswith("logscore_")]
num_labels = df_res_wlabels.groupby(["method", "coeff"])[cols_labels].count().iloc[0]
print(f"Removed labels with <= 10 samples: {num_labels[num_labels <= 33].index.tolist()}")
cols_labels = num_labels[num_labels > 33].index

print(f"Label N={num_labels}")

df_res_pv = df_res_wlabels.groupby(["method", "coeff"], dropna=False)[cols_labels].mean().T
df_res_pv.index = [s.lstrip("logscore_") for s in df_res_pv.index]

# Replace NaN with 'disabled'
df_res_pv.columns = pd.MultiIndex.from_frame(df_res_pv.columns.to_frame().fillna('disabled'))

# Reorder so truthfulness at top
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

# For each method, compute correlation and range statistics


methods = df_res_pv.columns.get_level_values(0).unique()
correlation_results = []

for method in methods:
    method_data = df_res_pv[method]  # All coeffs for this method
    
    # Get available coefficients (excluding 'disabled')
    coeffs = [c for c in method_data.columns if (c not in ['disabled', None])]
    if len(coeffs) < 2:
        continue
    
    # Get truthfulness values at each coeff
    truth_values = method_data.loc[target_label, coeffs].astype(float)
    coef_true = truth_values.idxmax()
    coef_untrue = truth_values.idxmin()
    
    # For each other moral label, compute correlation with truthfulness
    for label in df_res_pv.index:
        if label == target_label:
            continue
            
        label_values = method_data.loc[label, coeffs].astype(float)
        
        # Compute Pearson correlation
        corr, p_val = stats.pearsonr(truth_values, label_values)
        
        # Compute range (max - min) to show magnitude of change
        # FIXME this should be most true coeff - most untrue coeff
        value_range = label_values[coef_true] - label_values[coef_untrue]
        
        # Compute slope via linear regression
        slope, intercept, r_val, p_val_reg, std_err = stats.linregress(truth_values, label_values)
        
        # Standardized slope (beta coefficient): how many SDs does label change per SD of truth
        std_slope = slope * (truth_values.std() / label_values.std()) if label_values.std() > 0 else 0
        
        correlation_results.append({
            'Method': method,
            't_stat': slope / std_err if std_err != 0 else np.nan,
            'Moral Value': label,
            'Correlation': corr,
            'p-value': p_val,
            'Std_Slope': std_slope,  # Standardized coefficient (same as correlation for simple regression)
            'Range': value_range,
            'R²': r_val**2,
        })

df_corr = pd.DataFrame(correlation_results)

# Filter to high-N labels (N > 50 for reliability)
high_n_labels = num_labels[num_labels > 30].index
high_n_labels = [s.lstrip("logscore_") for s in high_n_labels if s.lstrip("logscore_") != target_label]

# Hypothesis-driven clusters (expanded with high-N values)
assistant_virtues = [
    'MFT/Care',              # N=412
    'Value/empathy',         # N=238
    'Value/patience',        # N=183
    'Value/compassion',      # N=98
    'Value/concern',         # N=68
    'Value/respect',         # N=229
    'Value/understanding',   # N=256
]

agent_virtues = [
    'Value/self',            # N=425
    'Value/autonomy',        # N=53
    'Virtue/Ambition',       # N=45
    'Value/courage',         # N=170
    'Virtue/Courage',        # N=251
    'Value/independence',    # N=44
    'Value/freedom',         # N=43
]

conscientiousness = [
    'Value/honesty',         # N=403
    'Value/responsibility',  # N=287
    'Value/accountability',  # N=187
    'Value/integrity',       # N=193
    'Value/professionalism', # N=143
    'Value/fairness',        # N=184
    'Value/loyalty',         # N=138
]

# Uncorrelated cluster: high-N values expected orthogonal to truthfulness/agency/assistant
# (subjective preferences with no correct answer)
preferences = [
    'Preference/preference_a',
    'Preference/preference_b',
]

math = [
    'Math/math_correct',
    'Math/math_incorrect',
]

all_cols = assistant_virtues + agent_virtues + conscientiousness + preferences + math

# Define clusters once
clusters = {
    'Agent': agent_virtues,
    'Assistant': assistant_virtues,
    'Conscientiousness': conscientiousness,
    'Preferences': preferences,
    'Math': math,
}
    
# Cluster ranking table
print("\n## Cluster Transfer Effects by Method")
print("Mean t-statistic per cluster (rank in parentheses, 1=strongest effect):\n")
print("Positive t-stat = increases with truthfulness, negative = decreases\n")

ranking_data = []
for method in methods:
    method_df = df_corr[df_corr['Method'] == method].copy()
    if len(method_df) == 0:
        continue
    
    cluster_stats = {}
    for cluster_name, cluster_values in clusters.items():
        cluster_df = method_df[method_df['Moral Value'].isin(cluster_values)]
        if len(cluster_df) > 0:
            # Use t-stat mean for ranking (signed effect size)
            cluster_stats[cluster_name] = {
                'mean_t': cluster_df['t_stat'].mean(),
                'std_t': cluster_df['t_stat'].std(),
                'n': len(cluster_df)
            }
    
    # Rank clusters by mean t-stat (descending by abs, NaN goes last)
    ranked = sorted(
        cluster_stats.items(), 
        key=lambda x: (pd.isna(x[1]['mean_t']), -abs(x[1]['mean_t']) if not pd.isna(x[1]['mean_t']) else 0)
    )
    rank_map = {name: idx+1 for idx, (name, _) in enumerate(ranked)}
    
    for cluster_name, stats in cluster_stats.items():
        ranking_data.append({
            'Cluster': cluster_name,
            method: f"{stats['mean_t']:+.2f}±{stats['std_t']:.2f} ({rank_map[cluster_name]})",
        })

# Pivot to show values across methods
df_ranking = pd.DataFrame(ranking_data)
if len(df_ranking) > 0:
    rank_pivot = df_ranking.groupby('Cluster').first()
    print(tabulate(rank_pivot, tablefmt='pipe', headers='keys'))

# %%
# For each method, show top correlated values
for method in methods:
    method_df = df_corr[df_corr['Method'] == method].copy()
    if len(method_df) == 0:
        continue
    
    print(f"\n### {method}")
    print("Top 10 most slopeed (positive and negative):\n")
    
    # Sort by absolute correlation
    method_df = method_df.sort_values('Correlation', key=abs, ascending=False)
    top_10 = method_df.head(10)[['Moral Value', 'Correlation', 'Std_Slope', 'Range', 'p-value']]
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
    # TODO floatformat
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
