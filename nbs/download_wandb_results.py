#!/usr/bin/env python3
"""Download evaluation results and logs from wandb runs with caching."""

import wandb
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import json
from datetime import datetime

# Cache directory
cache_dir = Path(__file__).parent / 'outputs' / 'wandb_cache'
cache_dir.mkdir(parents=True, exist_ok=True)

# Init wandb API
api = wandb.Api()
project = "wassname/InnerPiSSA"


print("Downloading from wandb...")
runs = api.runs(project)
runs_data = []

for run in tqdm(runs):
    log_file = cache_dir / f"{run.id}_logs.txt"
    run_file = cache_dir / f"{run.id}_run.json"
    if run_file.exists() and log_file.exists():
        print(f"  Using cached data for {run.name}")
        with open(run_file) as f:
            run_data = json.load(f)
        runs_data.append(run_data)
        continue

    config = dict(run.config)
    summary = run.summary._json_dict
    
    # Download logs
    if not log_file.exists():
        try:
            logs = run.history(stream='events', pandas=False)
            log_lines = []
            for entry in logs:
                if '_runtime' in entry:
                    log_lines.append(str(entry))
            log_file.write_text('\n'.join(log_lines))
        except Exception as e:
            print(f"  Warning: couldn't download logs for {run.name}: {e}")
    
    run_data = {
        'run_id': run.id,
        'name': run.name,
        'state': run.state,
        'created_at': str(run.created_at),
        'url': run.url,
        'config': config,
        'summary': summary,
        'log_file': str(log_file) if log_file.exists() else None,
    }
    runs_data.append(run_data)
    print(f"  {run.name} - {summary.get('eval/main_metric')}")
    # Save individual run data
    with open(run_file, 'w') as f:
        json.dump(run_data, f, indent=2)


# Flatten for DataFrame
results = []
for run_data in runs_data:
    config = run_data['config']
    summary = run_data['summary']
    
    result = {
        'run_id': run_data['run_id'],
        'name': run_data['name'],
        'state': run_data['state'],
        'created_at': run_data['created_at'],
        'url': run_data['url'],
        'log_file': run_data.get('log_file'),
        # Config
        'lr': config.get('lr'),
        'rank': config.get('rank'),
        'num_layers': config.get('num_layers'),
        'loss_type': config.get('loss_type'),
        'scale_s': config.get('scale_s'),
        'n_epochs': config.get('n_epochs'),
        'batch_size': config.get('batch_size'),
        # Results
        'main_metric': summary.get('eval/main_metric'),
        'effect': summary.get('eval/InnerPiSSA (ours)/effect'),
        'side_effects': summary.get('eval/InnerPiSSA (ours)/side_effects'),
        'degradation': summary.get('eval/InnerPiSSA (ours)/degradation'),
        'p_value': summary.get('eval/InnerPiSSA (ours)/p_value'),
    }
    results.append(result)

df = pd.DataFrame(results)
df = df.sort_values('created_at', ascending=False)

output_dir = Path(__file__).parent
output_file = output_dir / 'wandb_results.csv'
df.to_csv(output_file, index=False)
print(f"\nSaved {len(df)} runs to {output_file}")

# Also save a filtered version with just the key metrics
summary_cols = ['name', 'main_metric', 'effect', 'side_effects', 'degradation', 
                'lr', 'rank', 'num_layers', 'loss_type', 'scale_s', 'url']
df_summary = df[summary_cols].dropna(subset=['main_metric'])
summary_file = output_dir / 'wandb_summary.csv'
df_summary.to_csv(summary_file, index=False)
print(f"Saved summary to {summary_file}")
