# %%
import pandas as pd
from pathlib import Path
from ipissa.train.daily_dilemas import format_results_table, compute_coherence_metrics
import cattrs
import json
from tqdm.auto import tqdm
from adjustText import adjust_text
from ipissa.train.train_adapter import proj_root, TrainingConfig
from tabulate import tabulate
import scipy.stats as stats
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt


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
target_label = 'Virtue/Truthfulness'

# %%
# Core analysis functions

def analyze_checkpoint(run_path: Path, clusters: dict, target_label: str = 'Virtue/Truthfulness'):
    """
    Analyze a single checkpoint: fit truth~coeff, regress values~truth, aggregate by cluster.
    
    Returns:
        run_name: str - identifier for this run
        df_value_stats: DataFrame - per-value regression results (Method, Moral Value, t_stat, n, Category)
        df_cluster_stats: DataFrame - aggregated per (Method, Category) with mean/std t-stats
    """
    if not (run_path / "eval_results.parquet").exists():
        raise FileNotFoundError(f"Missing eval_results.parquet in {run_path}")
    
    # Load data
    df_res_wlabels = pd.read_parquet(run_path / "eval_results.parquet")
    d = json.loads((run_path / "training_config.json").read_text())
    config = cattrs.structure(d, TrainingConfig)
    run_name = run_path.name
    
    # Step 1: Fit truth ~ coeff per method
    methods = df_res_wlabels['method'].unique()
    coeff_models = {}
    
    for method in methods:
        method_df = df_res_wlabels[df_res_wlabels['method'] == method].copy()
        truth_data = method_df[['coeff', f'logscore_{target_label}']].dropna()
        if len(truth_data) < 3:
            logger.warning(f"Skipping {method} in {run_name}: insufficient truthfulness data")
            continue
        
        try:
            slope, intercept, r_val, p_val, stderr = stats.linregress(
                truth_data['coeff'], truth_data[f'logscore_{target_label}']
            )
            coeff_models[method] = {'slope': slope, 'intercept': intercept}
        except Exception as e:
            logger.warning(f"Error fitting {method} in {run_name}: {e}")
            continue
    
    # Step 2: Predict truth for all questions
    df_raw = df_res_wlabels.copy()
    df_raw['truth_pred'] = df_raw.apply(
        lambda row: (coeff_models[row['method']]['slope'] * row['coeff'] + 
                     coeff_models[row['method']]['intercept']) 
        if row['method'] in coeff_models else np.nan,
        axis=1
    )
    
    # Step 3: Regress value ~ truth_pred for each (method, value) pair
    correlation_results = []
    value_cols = [c for c in df_raw.columns 
                  if c.startswith('logscore_') and c != f'logscore_{target_label}']
    
    for method in methods:
        if method not in coeff_models:
            continue
        method_df = df_raw[df_raw['method'] == method].copy()
        
        for col in value_cols:
            valid = method_df[['truth_pred', col]].dropna()
            if len(valid) < 10:
                continue
            
            slope, intercept, r_val, p_val, stderr = stats.linregress(valid['truth_pred'], valid[col])
            t_stat = slope / stderr if stderr > 0 else np.nan
            
            label = col.replace('logscore_', '')
            
            # Categorize using cluster definitions
            category = 'Other'
            for cat_name, cat_values in clusters.items():
                if label in cat_values:
                    category = cat_name
                    break
            
            correlation_results.append({
                'Method': method,
                'Moral Value': label,
                't_stat': t_stat,
                'slope': slope,
                'r': r_val,
                'n': len(valid),
                'Category': category,
            })
    
    df_value_stats = pd.DataFrame(correlation_results)
    
    # Step 4: Aggregate by cluster
    df_cluster_stats = df_value_stats.groupby(['Method', 'Category']).agg({
        't_stat': ['mean', 'std', 'count'],
        'n': 'sum'  # Total samples across all values in cluster
    }).reset_index()
    df_cluster_stats.columns = ['Method', 'Category', 't_mean', 't_std', 'n_values', 'n_total_samples']
    
    return run_name, df_value_stats, df_cluster_stats, config


def plot_radar(df_cluster_stats, series_key='Method', series_values=None, 
               categories=['Agent', 'Assistant', 'Conscientiousness'], title='Moral Value Transfer'):
    """
    Create radar plot showing cluster t-statistics.
    
    Args:
        df_cluster_stats: DataFrame with columns [series_key, Category, t_mean]
        series_key: column name to group by (e.g., 'Method' or 'Run')
        series_values: list of values to plot (None = all unique)
        categories: cluster names to include on radar
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    if series_values is None:
        series_values = df_cluster_stats[series_key].unique()
    
    for series_val in series_values:
        series_data = df_cluster_stats[df_cluster_stats[series_key] == series_val]
        
        values = []
        for cat in categories:
            val = series_data[series_data['Category'] == cat]['t_mean'].values
            values.append(val[0] if len(val) > 0 else 0)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=series_val, alpha=0.7)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylabel('t-statistic', labelpad=30)
    ax.set_title(title, pad=20)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    plt.tight_layout()
    return fig


# %%
# Load multiple checkpoints/runs for comparison
results_to_compare = sorted((proj_root / "./outputs/adapters/").glob("*"))

all_results = []

for result_dir in tqdm(results_to_compare):
    try:
        run_name, df_value_stats, df_cluster_stats, config = analyze_checkpoint(
            result_dir, clusters, target_label
        )
        df_value_stats['Run'] = run_name
        df_cluster_stats['Run'] = run_name
        all_results.append({
            'run_name': run_name,
            'value_stats': df_value_stats,
            'cluster_stats': df_cluster_stats,
            'config': config
        })
    except FileNotFoundError as e:
        logger.debug(f"Skipping {result_dir.name}: {e}")
        continue
    except Exception as e:
        logger.error(f"Error analyzing {result_dir.name}: {e}")
        continue

if not all_results:
    raise ValueError("No valid results found in outputs/adapters/")

# Combine all runs
df_all_value_stats = pd.concat([r['value_stats'] for r in all_results], ignore_index=True)
df_all_cluster_stats = pd.concat([r['cluster_stats'] for r in all_results], ignore_index=True)

# %%
# Radar plot comparing runs for a specific method
method_to_plot = 'InnerPiSSA (ours)'
plot_radar(
    df_all_cluster_stats, 
    series_key='Run', 
    categories=['Agent', 'Assistant', 'Conscientiousness', 'Preferences', 'Math'],
    title=f'Transfer Effects Across Checkpoints: {method_to_plot}'
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
# Single-run detailed analysis
# Pick most recent run or specific checkpoint
result_dir = proj_root / "./outputs/adapters/q4b-raw-r256-lr1e-2_20251120_095949"  # wd=100

run_name, df_value_stats, df_cluster_stats, config = analyze_checkpoint(
    result_dir, clusters, target_label
)

print(f"\nAnalyzing: {run_name}")
print(cattrs.unstructure(config))

# Load for additional analysis
df_res_wlabels = pd.read_parquet(result_dir / "eval_results.parquet")

# Generate formatted results table
md, tables, s = format_results_table(
    df_res_wlabels, 
    target_col=f'logscore_{target_label}', 
    config=config
)
print(md)

# %%
# Radar plot comparing methods for one checkpoint
plot_radar(
    df_cluster_stats,
    series_key='Method',
    categories=['Agent', 'Assistant', 'Conscientiousness', 'Preferences', 'Math'],
    title=f'Moral Value Transfer: Truthfulness → Other Values ({run_name})'
)
plt.show()

# %%
print("\n## Cluster Transfer Effects by Method")
print("Mean t-statistic per cluster (positive = increases with truthfulness)\n")

# Pivot for display
df_display = df_cluster_stats.pivot(index='Category', columns='Method', values='t_mean')
sample_info = df_cluster_stats.groupby('Category')[['n_values', 'n_total_samples']].first()
df_display = df_display.join(sample_info)

print(tabulate(df_display, tablefmt='pipe', headers='keys', floatfmt='+.2f'))



# %%
# table of results
df_correlations = pd.concat([df_all_cluster_stats.query('Method=="InnerPiSSA (ours)"'),
df_cluster_stats.query('Method!="InnerPiSSA (ours)"')])
df_correlations = df_correlations.pivot_table(index=['Method', 'Run'], columns=['Category'], values=['t_mean', 't_std'])
df_correlations
# %%
# view the agent ones
df_correlations[[('t_mean', 'Agent'), ('t_std', 'Agent')]].sort_values(('t_mean', 'Agent'))