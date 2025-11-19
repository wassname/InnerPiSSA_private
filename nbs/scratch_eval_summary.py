# %%
import pandas as pd
from pathlib import Path
from ipissa.train.daily_dilemas import format_results_table, compute_coherence_metrics
import cattrs
import json
from adjustText import adjust_text
from ipissa.train.train_adapter import proj_root, TrainingConfig

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

# adapter_path = result_dir[-1] / "adapter_model.safetensors"


# results_dir = proj_root / "./outputs/adapters/honest_contrastive_ipissa_20251109_080729"

# f = 
# or is it "eval_summary.parquet"?
df_res_wlabels = pd.read_parquet(result_dir / "eval_results.parquet")
df_res_pv = pd.read_parquet(result_dir / "eval_summary.parquet")

d = json.loads((result_dir / "training_config.json").read_text())
config = cattrs.structure(d, TrainingConfig)
print(f"Evaluation results:\n{df_res_pv.round(4)}")

# Optionally load per-model prompting baseline if exists
# NOTE: Disabled for performance - prompting baseline has many rows which slows down compute_transfer_summary
# model_safe = config.model_name.replace('/', '_')
# prompting_path = proj_root / "outputs" / f"prompting_baseline_{model_safe}.parquet"
# if prompting_path.exists():
#     res_prompting = pd.read_parquet(prompting_path)
#     df_res_wlabels = pd.concat([df_res_wlabels, res_prompting], ignore_index=True)
#     print(f"Added prompting baseline ({len(res_prompting)} rows)")



md, tables, s = format_results_table(df_res_wlabels, target_col='binary_Virtue/Truthfulness', config=config)
print(md)


md, tables, s = format_results_table(df_res_wlabels, target_col='score_Virtue/Truthfulness', config=config)
print(md)

md, tables, s = format_results_table(df_res_wlabels, target_col='logscore_Virtue/Truthfulness', config=config)
print(md)

df = tables['T-stat']

# %%
from great_tables import GT, md, html

# Create publication-quality table
gt_table = (
    GT(df.reset_index())
    .tab_header(
        title="Honesty Transfer to Morality via Representation Engineering",
        subtitle=md(f"Daily Dilemmas dataset ({config.max_samples} train → {config.eval_max_dilemmas or 64} test) | Model: {config.model_name}")
    )
    .tab_spanner(
        label="Steering",
        columns=["Method", "Coeff\n±"]
    )
    .tab_spanner(
        label="Transfer Effects",
        columns=["Target Effect\nΔ Truth ↑", "Side Effects\nΔ Other ↓", "p-value"]
    )
    .tab_spanner(
        label="Quality Metrics",
        columns=["Output Quality\nΔ NLL ↓", "Normalized Gain (%)"]
    )
    .cols_label(
        Method="Method",
        **{"Coeff\n±": "Coeff ±"},
        **{"Target Effect\nΔ Truth ↑": html("Δ Truth<br>↑")},
        **{"Side Effects\nΔ Other ↓": html("Δ Other<br>↓")},
        **{"Output Quality\nΔ NLL ↓": html("Δ NLL<br>↓")},
        **{"Normalized Gain (%)": html("Norm. Gain<br>(%)")}
    )
    .fmt_number(
        columns=["Target Effect\nΔ Truth ↑", "Side Effects\nΔ Other ↓", "Output Quality\nΔ NLL ↓"],
        decimals=3
    )
    .fmt_number(
        columns=["p-value"],
        decimals=3
    )
    .fmt_number(
        columns=["Normalized Gain (%)"],
        decimals=1
    )
    .tab_source_note(
        source_note=md(
            f"**Target Effect**: Δ Truthfulness score vs baseline (higher = more truthful). "
            f"**Side Effects**: mean |Δ| across {df.iloc[0].get('total_values', 30) - 1} non-target moral values (lower = less collateral). "
            f"**Output Quality**: coherence degradation ΔNLL (lower = better). "
            f"**Normalized Gain** = 100 × Δ Truth / (1 + Δ NLL) measures steering efficiency. "
            f"**p-values** test monotonic dose-response via linear regression."
        )
    )

)

gt_table_html = gt_table.data_color(
        columns=["Normalized Gain (%)"],
        palette=["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
        domain=[df["Normalized Gain (%)"].min(), df["Normalized Gain (%)"].max()]
    )


# Save outputs
output_dir = result_dir / "tables"
output_dir.mkdir(exist_ok=True)

# Display in notebook
gt_table_html.show('browser')

# %%
import matplotlib.pyplot as plt

# Extract coefficient from the "Coeff\n±" column
df_plot = df.copy()
df_plot['coeff_value'] = df_plot['Coeff\n±']

# Plot effect vs coherence
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df_plot['Output Quality\nΔ NLL ↓'], 
    df_plot['Target Effect\nΔ Truth ↑'],
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
        (row['Output Quality\nΔ NLL ↓'], row['Target Effect\nΔ Truth ↑']),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5)
    )
    texts.append(text)

# Adjust text positions to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)

ax.set_xlabel('Coherence Degradation (Δ NLL ↓)', fontsize=12)
ax.set_ylabel('Target Effect (Δ Truth ↑)', fontsize=12)
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

# Save as LaTeX
latex_str = gt_table.as_latex()
(output_dir / "results_table.tex").write_text(latex_str)
print(f"Saved LaTeX to {output_dir / 'results_table.tex'}")
