#!/usr/bin/env python3
"""Download evaluation results and logs from wandb runs with caching."""
# %%

import wandb
import pandas as pd
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm
import json
from datetime import datetime
from ipissa.config import EVAL_BASELINE_MODELS, TrainingConfig, proj_root

# Cache directory
cache_dir = proj_root / 'outputs' / 'wandb_cache'
cache_dir.mkdir(parents=True, exist_ok=True)

# Init wandb API
api = wandb.Api()
project = "wassname/InnerPiSSA"

last_major_code_change = '2025-11-19T00:08:00Z'
last_major_code_change = '2025-11-20T22:08:00Z' # let the model flip proj dir vs coeff for each module (across all modules)
logger.info(f"Considering only runs after last major code change at {last_major_code_change}")

# Find last cached run time
last_run = last_major_code_change or "1970-01-01T00:00:00"
cached_runs = list(cache_dir.glob("*_run.json"))
if cached_runs:
    latest_time = None
    for run_file in cached_runs:
        with open(run_file) as f:
            run_data = json.load(f)
            if 'created_at' in run_data:
                created_at = datetime.fromisoformat(run_data['created_at'])
                if latest_time is None or created_at > latest_time:
                    latest_time = created_at
    last_run = latest_time.isoformat()
    logger.info(f"Using cached runs after {last_run}")

logger.info("Downloading from wandb...")
lastest_runs = api.runs(project, filters={"state": "finished", "created_at": {"$gt": last_run}})
runs_data = []

# first load all the saved ones
cached_runs = list(cache_dir.glob("*_run.json"))
for run_file in cached_runs:
    with open(run_file) as f:
        run_data = json.load(f)
        runs_data.append(run_data)

for run in tqdm(lastest_runs):
    log_file = cache_dir / f"{run.id}_logs.txt"
    run_file = cache_dir / f"{run.id}_run.json"
    if run_file.exists() and log_file.exists():
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
        'metadata': run.metadata,
        'lastHistoryStep': run.lastHistoryStep,
    }
    runs_data.append(run_data)
    logger.info(f"  {run.name} - {summary.get('eval/main_metric')}")
    # Save individual run data
    with open(run_file, 'w') as f:
        json.dump(run_data, f, indent=2)


# Flatten for DataFrame
results = []
for run_data in runs_data:
    config = run_data['config']
    summary = run_data['summary']

    # TODO can we also save the cli argv?
    
    result = {
        'run_id': run_data['run_id'],
        'name': run_data['name'],
        'state': run_data['state'],
        'created_at': run_data['created_at'],
        'url': run_data['url'],
        'log_file': run_data.get('log_file'),
        "args": ' '.join(run_data['metadata'].get('args', [])),

        # Metadata
        'git_commit': run_data.get('metadata', {}).get('git', {}).get('commit'),
        'gpu': run_data.get('metadata', {}).get('gpu'),
        
        # Results - layer info
        'layer_num': summary.get('layer_num'),
        
        # Results - main metrics
        'main_metric': summary.get('eval/main_metric'),
        "runtime": summary.get('_runtime'),
        
        # Results - baselines
        'baseline_effect_InnerPiSSA': summary.get('eval/baseline_InnerPiSSA (ours)'),
        'baseline_effect_s_steer': summary.get('eval/baseline_S-space steer'),
        'baseline_effect_pca': summary.get('eval/baseline_pca (wassname)'),
        'baseline_effect_prompting': summary.get('eval/baseline_prompting'),
        'baseline_effect_repeng': summary.get('eval/baseline_repeng'),
        
        # Results - final val metrics
        'val_loss_total': summary.get('val/loss_total'),
        'val_loss_proj': summary.get('val/loss_proj'),
        'val_loss_coh': summary.get('val/loss_coh'),
        'val_loss_monotonic': summary.get('val/loss_monotonic'),
        'val_proj_diff': summary.get('val/proj_diff'),
        'val_logp_degradation': summary.get('val/logp_degradation'),
        
        # Results - final train metrics
        'train_loss_total': summary.get('loss_total'),
        'train_loss_proj': summary.get('loss_proj'),
        'train_loss_coh': summary.get('loss_coh'),
        'train_loss_monotonic': summary.get('loss_monotonic'),
        'train_proj_diff': summary.get('proj_diff'),
        'train_logp_degradation': summary.get('logp_degradation'),

        **config,
    }
    results.append(result)

config_keys = config.keys()

logger.debug(f'DEBUG run_data  {json.dumps(run_data, indent=2)}')

df = pd.DataFrame(results)
df = df.sort_values(['model_name', 'created_at'], ascending=False)
logger.info(f"Total runs: {len(df)}")
# filter out not finished, and not run for at least 1 minute
df = df[(df['state'] == 'finished') & (df['runtime'] > 60)]
logger.info(f"After filtering finished & >60s: {len(df)}")
# filter out eval_max_n_dilemmas is not [None or < 2000]
df = df[df['eval_max_dilemmas'].isna()]
logger.info(f"After filtering full evals: {len(df)}")
df = df[last_major_code_change < df['created_at']]
logger.info(f"After filtering recent runs: {len(df)}")

output_dir = Path(__file__).parent.parent
results_file = output_dir / 'outputs' / 'wandb_results.csv'
df.to_csv(results_file, index=False)
logger.info(f"Saved {len(df)} runs to {results_file}")

logger.debug(f"cols in df: {df.columns.tolist()}")
config
# Also save a filtered version with just the key metrics
summary_cols = ['created_at', 'model_name', 'name', 'layer_num', 'main_metric', 
                'baseline_prompting', 'baseline_repeng', 'baseline_s_steer',
                'lr', 'rank', 'n_depths', 'loss_depths', 'loss_type', 'scale_s',
                'coh_weight', 'mono_weight', 'val_loss_total', 'val_proj_diff', 'wandb_id',']
df_summary = df[summary_cols].dropna(subset=['main_metric'])
summary_file = output_dir / 'outputs' / 'wandb_summary.csv'
df_summary.to_csv(summary_file, index=False, float_format='%.4g', na_rep='NA')
logger.info(f"Saved summary to {summary_file}")
# print(df_summary)
from tabulate import tabulate
logger.info(f"\n{tabulate(df_summary, headers='keys', tablefmt='pipe', floatfmt='.4g')}")

import subprocess
help_text = subprocess.run(['uv', 'run', 'python', 'nbs/train.py', '.', '-h'], capture_output=True, text=True).stdout
logger.info(f"Train.py help:\n{help_text}")
f_help = proj_root / 'outputs'/ 'help_train.txt'
f_help.write_text(help_text)
logger.info(f"Saved train.py help to {f_help}")

try:
    fpr = proj_root / "outputs" / 'prompting_results.csv'
    logger.info(f"\n\n{fpr}\n{pd.read_csv(fpr).sort_values('main_score', ascending=False)}")
except Exception as e:
    logger.warning(f"Could not load prompting results: {e}")

try:
    fre = proj_root / "outputs" / 'repeng_results.csv'
    logger.info(f"\n\n{fre}\n{pd.read_csv(fre).sort_values('main_score', ascending=False)}")
except Exception as e:
    logger.warning(f"Could not load repeng results: {e}")

try:
    fss = proj_root / "outputs" / 'sSteer_results.csv'
    logger.info(f"\n\n{fss}\n{pd.read_csv(fss).sort_values('main_score', ascending=False)}")
except Exception as e:
    logger.warning(f"Could not load sSteer results: {e}")

summary_file = output_dir / 'outputs' / 'wandb_summary.csv'


logger.info(f"""
=== SWEEP ANALYSIS TASK ===

Goal: Identify which hyperparameters generalize across large models (ideally 4B+ params).

Data:
- Summary CSV: {summary_file} (shape={df_summary.shape})
- Full results: {results_file} (shape={df.shape})
- README.md: Metrics Reference section for definitions
- CLI reference: {f_help}
- Baseline CSVs: {fpr}, {fre}, {fss}
            
""")