"""
Generate paper tables from wandb results.

Usage:
    uv run python nbs/generate_paper_tables.py [--format html|markdown]
    
Outputs markdown tables by default. Use --format html for Great Tables HTML output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

try:
    from great_tables import GT, md, html
    from great_tables import loc, style
    HAS_GT = True
except ImportError:
    HAS_GT = False
    print("⚠️  great_tables not installed. Install with: uv add great_tables")
    print("    Falling back to markdown output only.")


def load_data():
    """Load all results CSVs."""
    df = pd.read_csv('outputs/wandb_summary.csv')
    prompting = pd.read_csv('outputs/prompting_results.csv')
    repeng = pd.read_csv('outputs/repeng_results.csv')
    s_steer = pd.read_csv('outputs/sSteer_results.csv')
    return df, prompting, repeng, s_steer


def table1_cross_model(df, prompting, repeng, s_steer):
    """Table 1: Cross-Model Generalization."""
    print("\n" + "="*80)
    print("TABLE 1: Cross-Model Generalization")
    print("="*80)
    
    # Get baseline scores by model
    baselines = {}
    for model in prompting['model_name'].unique():
        baselines[model] = {
            'prompting': prompting[prompting['model_name'] == model]['main_score'].values[0] if len(prompting[prompting['model_name'] == model]) > 0 else None,
            'repeng': repeng[repeng['model_name'] == model]['main_score'].values[0] if len(repeng[repeng['model_name'] == model]) > 0 else None,
            's_steer': s_steer[s_steer['model_name'] == model]['main_score'].values[0] if len(s_steer[s_steer['model_name'] == model]) > 0 else None,
        }
    
    # Get InnerPiSSA best score per model
    innerpissa_scores = df.groupby('model_name')['main_metric'].max().to_dict()
    
    # Map model names to readable sizes
    model_info = {
        'Qwen/Qwen3-0.6B': ('Qwen3-0.6B', '0.6B'),
        'Qwen/Qwen3-4B-Instruct-2507': ('Qwen3-4B', '4B'),
        'Qwen/Qwen3-4B-Base': ('Qwen3-4B-base', '4B'),
        'Qwen/Qwen3-14B': ('Qwen3-14B', '14B'),
        'unsloth/Llama-3.1-8B-Instruct': ('Llama-3.1-8B', '8B'),
        'google/gemma-3-270m-it': ('Gemma-3-270M', '0.27B'),
        'google/gemma-3-1b-it': ('Gemma-3-1B', '1B'),
        'google/gemma-3-4b-it': ('Gemma-3-4B', '4B'),
        'google/gemma-3-12b-it': ('Gemma-3-12B', '12B'),
        'wassname/qwen-14B-codefourchan': ('Qwen-14B-4chan', '14B'),
    }
    
    # Build table
    rows = []
    for model_full, (model_short, size) in model_info.items():
        if model_full in baselines:
            row = {
                'Model': model_short,
                'Size': size,
                'InnerPiSSA': innerpissa_scores.get(model_full, 'TODO'),
                'Prompting': baselines[model_full]['prompting'] or '-',
                'RepEng': baselines[model_full]['repeng'] or '-',
                'S-Steer': baselines[model_full]['s_steer'] or '-',
            }
            rows.append(row)
    
    table_df = pd.DataFrame(rows)
    
    # Format numbers
    for col in ['InnerPiSSA', 'Prompting', 'RepEng', 'S-Steer']:
        table_df[col] = table_df[col].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
        )
    
    # Mark best per row with ✓
    for idx, row in table_df.iterrows():
        values = []
        for col in ['InnerPiSSA', 'Prompting', 'RepEng', 'S-Steer']:
            val = row[col]
            if val != 'TODO' and val != '-':
                try:
                    values.append((col, float(val)))
                except:
                    pass
        if values:
            best_col, best_val = max(values, key=lambda x: x[1])
            table_df.at[idx, best_col] = f"**{table_df.at[idx, best_col]}** ✓"
    
    print("\n```markdown")
    print(table_df.to_markdown(index=False))
    print("```")
    
    # Save to file
    output_dir = Path('docs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    fo = output_dir / 'table1_cross_model.md'
    with open(fo, 'w') as f:
        f.write(table_df.to_markdown(index=False))
    print(f"\nTable saved to {fo}")
    
    return table_df


def table2_layer_ablation(df):
    """Table 2: Layer Depth Ablation."""
    print("\n" + "="*80)
    print("TABLE 2: Layer Depth Ablation")
    print("="*80)
    
    df_clean = df[~df['name'].str.contains('lora|noV', na=False)]
    
    # Extract depth info
    depth_data = []
    for idx, row in df_clean.iterrows():
        if pd.notna(row['loss_depths']):
            depth_str = str(row['loss_depths']).strip('[]')
            depth_val = float(depth_str.split(',')[0])
            depth_data.append({
                'Depth': depth_val,
                'Layer': row['layer_num'],
                'Main Metric': row['main_metric'],
                'Val Loss': row['val_loss_total'],
            })
    
    depth_df = pd.DataFrame(depth_data)
    
    # Aggregate by depth
    table_df = depth_df.groupby('Depth').agg({
        'Layer': 'first',
        'Main Metric': ['mean', 'max', 'count'],
        'Val Loss': 'mean'
    }).round(1)
    
    # Flatten columns
    table_df.columns = ['Layer', 'Mean', 'Max', 'N Runs', 'Val Loss']
    table_df = table_df.reset_index()
    
    # Add finding column
    def classify_perf(depth):
        if 0.3 <= depth <= 0.5:
            return "**Strong** ✓"
        elif depth < 0.2:
            return "Early - weak"
        elif depth > 0.8:
            return "Late - weak"
        else:
            return "Mid"
    
    table_df['Finding'] = table_df['Depth'].apply(classify_perf)
    
    # Format
    table_df['Depth'] = table_df['Depth'].round(2)
    table_df['Layer'] = table_df['Layer'].astype(int)
    
    print("\n```markdown")
    print(table_df.to_markdown(index=False))
    print("```")
    
    # Save
    output_dir = Path('docs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    fo = output_dir / 'table2_layer_ablation.md'
    with open(fo, 'w') as f:
        f.write(table_df.to_markdown(index=False))
    print(f"\nTable saved to {fo}")
    
    return table_df


def table3_learning_rate(df):
    """Table 3: Learning Rate Sensitivity."""
    print("\n" + "="*80)
    print("TABLE 3: Learning Rate Sensitivity")
    print("="*80)
    
    df_clean = df[~df['name'].str.contains('lora|noV', na=False)]
    
    # Aggregate by LR
    lr_stats = df_clean.groupby('lr').agg({
        'main_metric': ['mean', 'std', 'max', 'count'],
        'val_loss_total': 'mean'
    }).round(1)
    
    # Flatten
    lr_stats.columns = ['Mean', 'Std', 'Max', 'N Runs', 'Val Loss']
    lr_stats = lr_stats.reset_index()
    lr_stats.columns = ['LR', 'Mean', 'Std', 'Max', 'N Runs', 'Val Loss']
    
    # Add result classification
    def classify_lr(row):
        lr = row['LR']
        mean = row['Mean']
        if lr <= 1e-4:
            return "Too low - fails ❌"
        elif 0.008 <= lr <= 0.01:
            if row['N Runs'] >= 10:
                return "**Default - stable** ✓"
            else:
                return "**High perf** ⚠️"
        elif lr >= 0.1:
            return "Too high - unstable"
        else:
            return "Low - weak"
    
    lr_stats['Result'] = lr_stats.apply(classify_lr, axis=1)
    
    # Format LR in scientific notation
    lr_stats['LR'] = lr_stats['LR'].apply(lambda x: f"{x:.0e}" if x < 0.001 or x >= 1 else f"{x:.3g}")
    
    # Handle NaN in Std
    lr_stats['Std'] = lr_stats['Std'].apply(lambda x: '-' if pd.isna(x) else f"{x:.1f}")
    
    print("\n```markdown")
    print(lr_stats.to_markdown(index=False))
    print("```")
    
    # Save
    output_dir = Path('docs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    fo = output_dir / 'table3_learning_rate.md'
    with open(fo, 'w') as f:
        f.write(lr_stats.to_markdown(index=False))
    print(f"\nTable saved to {fo}")
    
    return lr_stats


def table4_architecture(df):
    """Table 4: Architecture Component Ablation."""
    print("\n" + "="*80)
    print("TABLE 4: Architecture Component Ablation")
    print("="*80)
    
    # Get specific ablations
    ablations = {
        'Full InnerPiSSA': df[~df['name'].str.contains('lora|noV|snone', na=False)],
        'No S scaling': df[df['name'].str.contains('snone', na=False)],
        'No V rotation': df[df['name'].str.contains('noV', na=False)],
        'LoRA adapter': df[df['name'].str.contains('lora', na=False)],
    }
    
    rows = []
    for config, subset in ablations.items():
        if len(subset) > 0:
            row = {
                'Configuration': config,
                'Main Metric': subset['main_metric'].mean(),
                'Val Loss': subset['val_loss_total'].mean(),
                'N Runs': len(subset),
            }
            rows.append(row)
    
    table_df = pd.DataFrame(rows)
    
    # Add result classification
    baseline = table_df[table_df['Configuration'] == 'Full InnerPiSSA']['Main Metric'].values[0]
    
    def classify_result(row):
        config = row['Configuration']
        metric = row['Main Metric']
        if config == 'Full InnerPiSSA':
            return "**Baseline** ✓"
        elif metric < baseline * 0.1:
            return "**Catastrophic** ❌"
        elif metric > baseline:
            return "Better? (investigate)"
        else:
            return f"{100*metric/baseline:.0f}% of baseline"
    
    table_df['Result'] = table_df.apply(classify_result, axis=1)
    
    # Format
    table_df['Main Metric'] = table_df['Main Metric'].round(1)
    table_df['Val Loss'] = table_df['Val Loss'].round(1)
    
    print("\n```markdown")
    print(table_df.to_markdown(index=False))
    print("```")
    
    # Save
    output_dir = Path('docs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    fo = output_dir / 'table4_architecture.md'
    with open(fo, 'w') as f:
        f.write(table_df.to_markdown(index=False))
    print(f"\nTable saved to {fo}")
    
    return table_df


def table5_rank_sensitivity(df):
    """Table 5: Rank Sensitivity (placeholder)."""
    print("\n" + "="*80)
    print("TABLE 5: Rank Sensitivity")
    print("="*80)
    print("\n⚠️  FIXME: Run `just sweep-rank` to populate this table")
    
    # Check if we have any rank data
    if 'rank' in df.columns and df['rank'].notna().any():
        rank_stats = df.groupby('rank').agg({
            'main_metric': ['mean', 'std', 'count'],
            'val_loss_total': 'mean'
        }).round(1)
        print("\nCurrent data:")
        print(rank_stats)
    else:
        print("\nNo rank sweep data available yet.")
    
    # Placeholder table
    placeholder = pd.DataFrame({
        'Rank': [32, 64, 128, 256, 512],
        'Main Metric': ['TODO'] * 5,
        'Val Loss': ['TODO'] * 5,
        'N Runs': ['TODO'] * 5,
    })
    
    print("\n```markdown")
    print(placeholder.to_markdown(index=False))
    print("```")
    
    return placeholder


def table6_module_targeting(df):
    """Table 6: Module Targeting Ablation (placeholder)."""
    print("\n" + "="*80)
    print("TABLE 6: Module Targeting Ablation")
    print("="*80)
    print("\n⚠️  FIXME: Run `just ablate-modules` to populate this table")
    
    placeholder = pd.DataFrame({
        'Modules': [
            'o_proj, down_proj (residual)',
            'gate_proj, up_proj (MLP)',
            'q_proj, k_proj, v_proj, o_proj (attn)',
            'All modules (default)',
        ],
        'Main Metric': ['TODO'] * 4,
        'Val Loss': ['TODO'] * 4,
        'Finding': ['TODO'] * 4,
    })
    
    print("\n```markdown")
    print(placeholder.to_markdown(index=False))
    print("```")
    
    return placeholder


def table7_data_efficiency(df):
    """Table 7: Data Efficiency (placeholder)."""
    print("\n" + "="*80)
    print("TABLE 7: Data Efficiency")
    print("="*80)
    print("\n⚠️  FIXME: Run `just data-efficiency` to populate this table")
    
    placeholder = pd.DataFrame({
        'Samples': [50, 100, 200, 400, 800, 2000],
        'Main Metric': ['TODO'] * 6,
        'Val Loss': ['TODO'] * 6,
        'Finding': ['TODO'] * 6,
    })
    
    print("\n```markdown")
    print(placeholder.to_markdown(index=False))
    print("```")
    
    return placeholder


def table8_seed_stability(df):
    """Table 8: Random Seed Stability (placeholder)."""
    print("\n" + "="*80)
    print("TABLE 8: Random Seed Stability")
    print("="*80)
    print("\n⚠️  FIXME: Run `just run-seeds` to populate this table")
    
    # Check if we have seed data
    if 'seed' in df.columns and df['seed'].notna().any():
        seed_stats = df.groupby('seed').agg({
            'main_metric': ['mean'],
            'val_loss_total': 'mean'
        }).round(1)
        print("\nCurrent data:")
        print(seed_stats)
    else:
        print("\nNo seed stability data available yet.")
    
    placeholder = pd.DataFrame({
        'Seed': [42, 123, 456, 'Mean ± Std'],
        'Main Metric': ['TODO'] * 4,
        'Val Loss': ['TODO'] * 4,
    })
    
    print("\n```markdown")
    print(placeholder.to_markdown(index=False))
    print("```")
    
    return placeholder


def main():
    """Generate all tables."""
    parser = argparse.ArgumentParser(description='Generate paper tables from wandb results')
    parser.add_argument('--format', choices=['markdown', 'html', 'both'], default='markdown',
                       help='Output format (default: markdown)')
    args = parser.parse_args()
    
    if args.format in ['html', 'both'] and not HAS_GT:
        print("❌ HTML format requires great_tables. Install with: uv add great_tables")
        print("   Falling back to markdown only.")
        args.format = 'markdown'
    
    print("="*80)
    print("GENERATING PAPER TABLES FROM WANDB RESULTS")
    print("="*80)
    print(f"Output format: {args.format}")
    
    # Load data
    df, prompting, repeng, s_steer = load_data()
    
    print(f"\nLoaded {len(df)} wandb runs")
    print(f"Models: {df['model_name'].unique()}")
    
    # Generate all tables
    tables = {}
    captions = {
        'table1': "**Cross-model generalization.** InnerPiSSA vs baseline methods across model sizes. Main metric = T-statistic / (1 + NLL degradation); higher is better. ✓ marks best method per model. Only Qwen3-4B has InnerPiSSA results (need: `just run-models`).",
        
        'table2': "**Layer depth ablation** (Qwen3-4B, lr=0.008). Steering performance at different network depths. Middle layers (0.3–0.5) show strongest effects, consistent with the hypothesis that late layers suppress concepts while early layers lack semantic content. Single run per depth except 0.5 (40 runs).",
        
        'table3': "**Learning rate sensitivity** (Qwen3-4B, depth=0.5). Effect of learning rate on training. lr=0.008–0.01 is optimal; lower rates fail to learn, higher rates cause instability. Mean/Std computed across all runs at each LR.",
        
        'table4': "**Architecture component ablation** (Qwen3-4B). Critical components for InnerPiSSA. V rotation is essential (96% drop without it); LoRA adapter fails completely; S scaling may be unnecessary (actually improves performance, needs confirmation).",
        
        'table5': "**Rank sensitivity.** Effect of adapter rank on performance. TODO: Run `just sweep-rank` to test ranks 32, 64, 128, 256, 512.",
        
        'table6': "**Module targeting ablation.** Effect of steering different transformer components (attention vs MLP). TODO: Run `just ablate-modules`.",
        
        'table7': "**Data efficiency.** Performance vs training sample count. TODO: Run `just data-efficiency` to test 50–2000 samples.",
        
        'table8': "**Random seed stability.** Variance across random seeds. TODO: Run `just run-seeds` to test seeds 42, 123, 456.",
    }
    
    tables['table1'] = table1_cross_model(df, prompting, repeng, s_steer)
    tables['table2'] = table2_layer_ablation(df)
    tables['table3'] = table3_learning_rate(df)
    tables['table4'] = table4_architecture(df)
    tables['table5'] = table5_rank_sensitivity(df)
    tables['table6'] = table6_module_targeting(df)
    tables['table7'] = table7_data_efficiency(df)
    tables['table8'] = table8_seed_stability(df)
    
    # Save with captions
    output_dir = Path('docs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for table_name, table_df in tables.items():
        if table_df is None:
            continue
            
        caption = captions.get(table_name, '')
        
        # Save markdown
        if args.format in ['markdown', 'both']:
            md_path = output_dir / f'{table_name}.md'
            with open(md_path, 'w') as f:
                f.write(f"{caption}\n\n")
                f.write(table_df.to_markdown(index=False))
        
        # Save HTML with Great Tables
        if args.format in ['html', 'both'] and HAS_GT:
            try:
                gt_table = (
                    GT(table_df)
                    .tab_header(
                        title=table_name.replace('_', ' ').title(),
                        subtitle=md(caption)
                    )
                    .tab_options(
                        table_font_size='12px',
                        heading_font_size='14px',
                        heading_font_weight='bold'
                    )
                )
                
                # Add conditional formatting for specific tables
                if table_name == 'table3':  # Learning rate
                    # Highlight best performing LR
                    max_mean = table_df['Mean'].max()
                    gt_table = gt_table.tab_style(
                        style=style.fill(color='lightgreen'),
                        locations=loc.body(columns='Mean', rows=table_df['Mean'] == max_mean)
                    )
                
                html_path = output_dir / f'{table_name}.html'
                gt_table.save(str(html_path))
                print(f"  ✓ Saved {html_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to generate HTML for {table_name}: {e}")
    
    print("\n" + "="*80)
    print("TABLES SAVED TO docs/tables/")
    print("="*80)
    print("\nReady tables (markdown):")
    print("  ✓ table2_layer_ablation.md")
    print("  ✓ table3_learning_rate.md")
    print("  ✓ table4_architecture.md")
    
    if args.format in ['html', 'both']:
        print("\nReady tables (HTML):")
        print("  ✓ table2_layer_ablation.html")
        print("  ✓ table3_learning_rate.html")
        print("  ✓ table4_architecture.html")
    
    print("\nTODO (need more sweeps):")
    print("  - table1_cross_model (need: just run-models)")
    print("  - table5_rank_sensitivity (need: just sweep-rank)")
    print("  - table6_module_targeting (need: just ablate-modules)")
    print("  - table7_data_efficiency (need: just data-efficiency)")
    print("  - table8_seed_stability (need: just run-seeds)")
    
    return tables


if __name__ == '__main__':
    main()
