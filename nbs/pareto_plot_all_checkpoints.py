# %%
import pandas as pd
from pathlib import Path
from repeng.train.daily_dilemas import format_results_table, compute_coherence_metrics
import cattrs
import json
from loguru import logger
from repeng.train.train_adapter import proj_root, TrainingConfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import numpy as np
import matplotlib.cm as cm
from anycache import anycache

@anycache(cachedir=str(proj_root / ".anycache" / "eval_summary2"))
def load_checkpoint_results(result_dir_str: str):
    """Load and format results for a single checkpoint (cached to disk)"""
    result_dir = Path(result_dir_str)
    
    df_res_wlabels = pd.read_parquet(result_dir / "eval_results.parquet")
    
    d = json.loads((result_dir / "training_config.json").read_text())
    config = cattrs.structure(d, TrainingConfig)
    
    # Format results for this checkpoint
    md, df, s = format_results_table(df_res_wlabels, target_col='logscore_Virtue/Truthfulness', config=config)
    
    # Add metadata
    df['checkpoint'] = result_dir.name
    # Can't cache config object directly, store as dict
    # df['config_dict'] = [cattrs.unstructure(config)] * len(df)
    config_dict = cattrs.unstructure(config)
    for k, v in config_dict.items():
        df[k] = str(v)
    df.attrs['config_dict'] = config_dict
    return df

# Load all checkpoints with results
print(f"proj_root: {proj_root}")
results_dirs = sorted((proj_root / "./outputs/adapters/").glob("*"))

all_results = []

for result_dir in tqdm(results_dirs):
    if (result_dir / "eval_results.parquet").exists():
        try:
            df = load_checkpoint_results(str(result_dir))
            all_results.append(df)
            print(f"Loaded: {result_dir.name}")
        except Exception as e:
            logger.exception(f"Error loading {result_dir.name}: {e}")
            1/0
            continue

if not all_results:
    raise ValueError("No valid results found in outputs/adapters/")

# Combine all results
df_all = pd.concat(all_results)
print(f"\nLoaded {len(all_results)} checkpoints with {len(df_all)} total method configurations")

# %%

for model_id, g, in df_all.groupby('model_name'):
    # Create a combined plot showing all checkpoints together
    checkpoints = g['checkpoint'].unique()

    print(f"Checkpoints found: {checkpoints.tolist()}")

    # Manual marker mapping for each method
    method_markers = {
        'prompting': '^',      # triangle
        'PCA (baseline)': '+',            # cross
        'random': '*',         # star
        'InnerPiSSA (ours)': 'o',  # circle (default for our method)
    }

    # Set up color palette for checkpoints
    checkpoint_colors = cm.tab10(np.linspace(0, 1, len(checkpoints)))
    checkpoint_color_map = dict(zip(checkpoints, checkpoint_colors))

    fig, ax = plt.subplots(figsize=(14, 10))
    output_dir = proj_root / "outputs" / "pareto_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    # TODO per model

    for checkpoint in checkpoints:
        df_checkpoint = df_all[df_all['checkpoint'] == checkpoint].copy()
        
        for method in df_checkpoint.index.unique():
            df_method = df_checkpoint[df_checkpoint.index == method]
            
            # Get marker for this method (default to 'o' if not specified)
            marker = method_markers.get(method, '.')
            color = checkpoint_color_map[checkpoint]
            
            # Adjust linewidth for better visibility of thin markers
            linewidth = 2.0 if marker == '+' else 0.5
            
            # Plot all points for this method in this checkpoint
            ax.scatter(
                df_method['Output Quality\nΔ NLL ↓'],
                df_method['Target Effect\nΔ Truth ↑'],
                c=[color],
                marker=marker,
                s=120,
                alpha=0.6,
                edgecolors='black',
                linewidth=linewidth,
            )

    
    ax.set_xlabel('Coherence Degradation (Δ NLL ↓)', fontsize=13)
    ax.set_ylabel('Target Effect (Δ Truth ↑)', fontsize=13)
    ax.set_title(f'Pareto Frontier by Checkpoint\nEffect vs Coherence Trade-off {model_id}', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xscale('log')

    # Create synthetic legend for methods (markers)
    legend_elements = []
    for method in method_markers:
        marker = method_markers[method]
        linewidth = 2.0 if marker == '+' else 0.5
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='w', 
                markerfacecolor='gray', markeredgecolor='black',
                markersize=10, label=method, linewidth=linewidth)
        )

    ax.legend(handles=legend_elements, fontsize=11, loc='best', framealpha=0.9)

    # Save the combined plot
    safe_model_id = model_id.replace('/', '_')
    plot_path_combined = output_dir / f"pareto_frontier_combined_{safe_model_id}.png"
    plt.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {plot_path_combined}")


    plt.show()
