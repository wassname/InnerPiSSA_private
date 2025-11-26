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
import numpy as np
from contextlib import contextmanager
import os
from ipissa.config import EVAL_BASELINE_MODELS, TrainingConfig, proj_root

@contextmanager
def in_directory(path):
    pwd = str(Path().absolute())
    if not path.is_dir():
        path = path.parent
    os.chdir(str(path))
    yield path.absolute()
    os.chdir(pwd)

# Cache directory
cache_dir = proj_root / 'outputs' / 'wandb_cache'
cache_dir.mkdir(parents=True, exist_ok=True)

# Init wandb API
api = wandb.Api()
project = "wassname/InnerPiSSA"

major_code_changes = [
    '2025-11-19T00:08:00Z',
    '2025-11-20T00:00:00Z', # let the model flip proj dir vs coeff for each module (across all modules)
    '2025-11-21T20:00:00Z', # found mistake in proj dir was flipped
    '2025-11-23T13:00:00Z', # fixed dd eval, and added residual @ V loss optio, fix serious V errors. Also added data aware init
]
last_major_code_change = major_code_changes[-1]
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
lastest_runs = list(api.runs(project, filters={"state": "finished", "created_at": {"$gt": last_run}}))
logger.info(f"Found {len(lastest_runs)} new runs since {last_run}")
runs_data = []

# first load all the saved ones
cached_runs = list(cache_dir.glob("**/run.json"))
for run_file in cached_runs:
    with open(run_file) as f:
        run_data = json.load(f)
        runs_data.append(run_data)

# download
for run in tqdm(lastest_runs):
    run_dir = cache_dir / run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "logs1.txt"
    run_file = run_dir / "run.json"
    if run_file.exists() and log_file.exists():
        logger.info(f"Skipping cached run: {run.name} ({run.id})")
        # json.dump(run_data, f, indent=2)
        # run_data = json.loads(run_file.read_text())
        # runs_data.append(run_data)
        continue

    config = dict(run.config)
    summary = run.summary._json_dict
    
    # Download logs
    if not log_file.exists():
        with in_directory(run_dir):
            print(f"Downloading logs for run: {run.name} ({run.id})")
            # print(f'File: {list(run.files())}')

            # https://docs.wandb.ai/models/track/public-api-guide#download-a-file-from-a-run
            # get main files
            run.file("wandb-metadata.json").download(exist_ok=True)
            run.file("output.log").download(exist_ok=True)


        try:
            logs = run.history(stream='events', pandas=False)
            log_lines = []
            for entry in logs:
                if '_runtime' in entry:
                    log_lines.append(str(entry))
            log_file.write_text('\n'.join(log_lines))
        except Exception as e:
            print(f"  Warning: couldn't download logs for {run.name}: {e}")
    
    meta = json.loads((run_dir / "wandb-metadata.json").read_text())
    logs = (run_dir / "output.log").read_text()
    argv = " ".join(["python"] + [meta["program"]] + meta.get("args", []))
    
    # Extract WANDB_RUN_GROUP from run.group (set by WANDB_RUN_GROUP env var in justfile)
    run_group = getattr(run, 'group', None)  
    
    run_data = {
        'run_id': run.id,
        'name': run.name,
        'argv': argv,
        'state': run.state,
        'created_at': str(run.created_at),
        'url': run.url,
        'config': config,
        'run_group': run_group,  # WANDB_RUN_GROUP for sweep organization

        'summary': summary,
        'log_file': str(log_file) if log_file.exists() else None,
        'metadata': run.metadata,
        'lastHistoryStep': run.lastHistoryStep,
    }
    runs_data.append(run_data)
    logger.info(f"  {run.name} - metric={summary.get('eval/main_metric', np.nan):.3g}, created_at={run.created_at}")
    # Save individual run data
    with open(run_file, 'w') as f:
        json.dump(run_data, f, indent=2)


if len(runs_data) == 0:
    logger.warning("No runs found!")
    exit(0)

# Flatten for DataFrame
def extract_symmetry_from_logs(log_file):
    """Parse logs to extract symmetry metrics from evaluation tables.
    
    Looks for lines like:
    Value/Honesty        -1.9254   0.2245   5.5585     0.2356
    
    Returns dict with symmetry_ratio, dose_monotonic for each moral dimension.
    """
    if not log_file or not Path(log_file).exists():
        return {}
    
    try:
        logs = Path(log_file).read_text()
    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        return {}
    
    metrics = {}
    
    # Find "Results for method: InnerPiSSA" table
    import re
    pattern = r"Results for method: InnerPiSSA.*?\ncoeff\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
    match = re.search(pattern, logs, re.DOTALL)
    
    if not match:
        return {}
    
    # Extract table after coeff header
    table_start = logs.find("Results for method: InnerPiSSA")
    table_end = logs.find("\n\n", table_start)
    if table_end == -1:
        table_end = len(logs)
    table_text = logs[table_start:table_end]
    
    # Parse rows like: Value/Honesty        -1.9254   0.2245   5.5585     0.2356
    row_pattern = r"([\w/]+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
    rows = re.findall(row_pattern, table_text)
    
    if len(rows) < 2:  # Skip header row
        return {}
    
    # Compute symmetry metrics
    symmetry_ratios = []
    dose_checks = []
    
    for dimension, neg_val, zero_val, pos_val in rows[1:]:  # Skip coeff header
        try:
            neg, zero, pos = float(neg_val), float(zero_val), float(pos_val)
            
            # Symmetry: how close is |neg| to |pos|?
            # Perfect symmetry = 1.0, asymmetric = far from 1.0
            if abs(pos) > 0.01:  # Avoid div by zero
                symmetry_ratio = abs(neg) / abs(pos) if abs(pos) > abs(neg) else abs(pos) / abs(neg)
                symmetry_ratios.append(symmetry_ratio)
            
            # Dose-dependence: monotonic -1 → 0 → 1?
            # Check if it's monotonic (either increasing or decreasing)
            is_increasing = neg < zero < pos
            is_decreasing = neg > zero > pos
            dose_checks.append(1.0 if (is_increasing or is_decreasing) else 0.0)
            
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"Error parsing row for dimension {dimension} in log file {log_file}, {e}")
            continue
    
    if symmetry_ratios:
        metrics['symmetry_mean'] = np.mean(symmetry_ratios)
        metrics['symmetry_min'] = np.min(symmetry_ratios)
        metrics['dose_monotonic_frac'] = np.mean(dose_checks)
    
    return metrics

results = []
for run_data in runs_data:
    config = run_data['config']
    summary = run_data['summary']

    # Extract symmetry metrics from output.log (not logs1.txt)
    run_id = run_data['run_id']
    output_log = cache_dir / run_id / "output.log"
    symmetry_metrics = extract_symmetry_from_logs(str(output_log) if output_log.exists() else None)
    
    # TODO can we also save the cli argv?
    
    result = {
        'run_id': run_data['run_id'],
        'name': run_data['name'],
        'state': run_data['state'],
        'created_at': run_data['created_at'],
        'url': run_data['url'],
        'log_file': run_data.get('log_file'),
        "args": ' '.join(run_data['metadata'].get('args', [])),
        'run_group': run_data.get('run_group'),  # Add WANDB_RUN_GROUP

        # Metadata
        'git_commit': run_data.get('metadata', {}).get('git', {}).get('commit'),
        'gpu': run_data.get('metadata', {}).get('gpu'),
        
        # Results - layer info
        'layer_num': summary.get('layer_num'),
        
        # Results - main metrics
        'main_metric': summary.get('eval/main_metric'),
        "runtime": summary.get('_runtime'),
        
        # Results - symmetry from logs
        'symmetry_mean': symmetry_metrics.get('symmetry_mean'),
        'symmetry_min': symmetry_metrics.get('symmetry_min'),
        'dose_monotonic_frac': symmetry_metrics.get('dose_monotonic_frac'),
        
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

# config_keys = config.keys()

# logger.debug(f'DEBUG run_data  {json.dumps(run_data, indent=2)}')

df = pd.DataFrame(results)

# Compute loss_gap (overfitting metric)
df['loss_gap'] = df['val_loss_total'] - df['train_loss_total']

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

# Find when each config option was first set (not NaN)
def fst_cnfg(df):
    """Find first non-NaN date for each config option to detect feature introductions."""
    df_sorted = df.sort_values('created_at')
    
    config_cols = [col for col in df.columns if col not in [
        'run_id', 'name', 'state', 'created_at', 'url', 'log_file', 'args',
        'git_commit', 'gpu', 'layer_num', 'main_metric', 'runtime',
        'baseline_effect_InnerPiSSA', 'baseline_effect_s_steer', 'baseline_effect_pca',
        'baseline_effect_prompting', 'baseline_effect_repeng',
        'val_loss_total', 'val_loss_proj', 'val_loss_coh', 'val_loss_monotonic',
        'val_proj_diff', 'val_logp_degradation',
        'train_loss_total', 'train_loss_proj', 'train_loss_coh', 'train_loss_monotonic',
        'train_proj_diff', 'train_logp_degradation', 'model_name',
    ]]
    
    intro_dates = {}
    for col in config_cols:
        if col in df_sorted.columns:
            non_nan = df_sorted[df_sorted[col].notna()]
            if len(non_nan) > 0:
                intro_dates[col] = non_nan.iloc[0]['created_at']
    
    if intro_dates:
        latest_intro = max(intro_dates.values())
        logger.info(f"\n=== Config Option Introduction Timeline ===")
        for col, date in sorted(intro_dates.items(), key=lambda x: x[1]):
            logger.info(f"  {col}: first set at {date}")
        logger.info(f"\nLatest feature introduction: {latest_intro}")
        logger.warning(f"⚠️  Runs before {latest_intro} may be missing newer config options!")
        return latest_intro
    return None

latest_feature_date = fst_cnfg(df)

df['argv'] = df['args'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

summary_cols = [ 'created_at','argv', 
                
            
        
       

        'main_metric', 
       
        'loss_gap',  # Overfitting metric
        'symmetry_mean',  # Coefficient symmetry
        'dose_monotonic_frac',  # Dose-dependence
       'baseline_effect_InnerPiSSA', 'baseline_effect_s_steer',
       'baseline_effect_pca', 'baseline_effect_prompting',
       'baseline_effect_repeng', 
       
        'train_loss_total',
       'val_loss_total', 

           'train_loss_proj',
       'val_loss_proj',

       'train_loss_coh',
       'val_loss_coh',

         'train_loss_monotonic', 
         'val_loss_monotonic', 

       'train_proj_diff',
         'val_proj_diff',
         
       'train_logp_degradation', 
       'val_logp_degradation',

    #    'PROMPT',
    # 'PERSONAS', 
       
       
    #    'layer_num', 'r', 'bs', 'lr', 'wd', 'coh', 'mono', 'quick',
    #    'rot_u', 'rot_v',  'n_logs', 'modules', 'scale_s',
    #    'n_depths', 'n_epochs', 'depth_end',
    #    'loss_type', 'coh_thresh', 'coh_weight', 'depth_start', 'loss_depths',
    #    'max_samples', 'mono_margin', 'mono_weight', 'adapter_type',
    #    'coh_adaptive',  'data_aware_init', 
    #    'experiment_name',  'eval_max_dilemmas','seed', 'loss_use_V',
    #    'loss_modules', 'max_rotation_angle'
# 'name', 
    #  'model_name', 
     'experiment_name',
    'runtime',
    'run_group',  # For sweep grouping
    'url', 
       ]

# config
# # Also save a filtered version with just the key metrics
# summary_cols = ['created_at', 'model_name', 'name', 'layer_num', 'main_metric', "baseline_effect_InnerPiSSA",
#                 'baseline_effect_prompting', 'baseline_effect_repeng', 'baseline_effect_s_steer',
#                 'lr', 'r', 'wd', 'n_depths', 'loss_depths', 'loss_type', 'scale_s', 'loss_use_V', 'data_aware_init', 
#                 'coh_weight', 'mono_weight', 'val_loss_total', 'val_proj_diff', 'run_id',]

df_summary = df[summary_cols].dropna(subset=['main_metric'])
summary_file = output_dir / 'outputs' / 'wandb_summary.csv'
df_summary.to_csv(summary_file, index=False, float_format='%.4g', na_rep='NA')
logger.info(f"Saved summary to {summary_file}")

# Save per-group summaries for sweep analysis
if 'run_group' in df.columns and df['run_group'].notna().any():
    group_dir = output_dir / 'outputs' / 'sweep_groups'
    group_dir.mkdir(parents=True, exist_ok=True)
    
    for group_name in df['run_group'].dropna().unique():
        df_group = df[df['run_group'] == group_name]
        logger.info(f"Group '{group_name}': {len(df_group)} runs, "
                   f"metric range [{df_group['main_metric'].min():.0f}, {df_group['main_metric'].max():.0f}], "
                   f"gap range [{df_group['loss_gap'].min():.1f}, {df_group['loss_gap'].max():.1f}]")
        
        # Save full group data
        group_file = group_dir / f"{group_name}.csv"
        df_group.to_csv(group_file, index=False, float_format='%.4g', na_rep='NA')
        
        # Save group summary
        group_summary = df_group[summary_cols].dropna(subset=['main_metric'])
        group_summary_file = group_dir / f"{group_name}_summary.csv"
        group_summary.to_csv(group_summary_file, index=False, float_format='%.4g', na_rep='NA')
    
    logger.info(f"Saved {len(df['run_group'].dropna().unique())} group summaries to {group_dir}")

# print(df_summary)
from tabulate import tabulate
# logger.info(f"\n{tabulate(df_summary, headers='keys', tablefmt='pipe', floatfmt='.4g')}")

import subprocess
help_text = subprocess.run(['uv', 'run', 'python', 'nbs/train.py', '.', '-h'], capture_output=True, text=True).stdout
# logger.info(f"Train.py help:\n{help_text}")
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

# summary_file = output_dir / 'outputs' / 'wandb_summary.csv'


logger.info(f"""
=== DATA ANALYSIS GUIDE ===

**Temporal cutoffs** (major_code_changes): {major_code_changes}
Runs separated by these dates aren't comparable due to code changes.

**Files to examine**:
- {summary_file} - Main results with loss_gap, symmetry_mean, dose_monotonic_frac, run_group
- {results_file} - Full configs for all runs
- outputs/sweep_groups/*.csv - Per-sweep tables organized by WANDB_RUN_GROUP
- {f_help} - CLI option reference (defaults, meanings)
- README.md - Metric definitions and interpretation

**Key metrics**:
- main_metric: Steering effectiveness (higher = stronger effect)
- loss_gap: val_loss - train_loss (lower = better generalization; <3 good, >10 catastrophic)
- symmetry_mean: |coeff=-1| / |coeff=+1| ratio (closer to 1.0 = more reversible)
- dose_monotonic_frac: Proportion of moral dimensions showing monotonic -1→0→1 response

**Methodological constraints**:
1. Account for sampling bias: configs tested more often will have higher max values
   → Use median/mean with counts, not just max
2. Baselines (prompting, RepE, S-steer) can't be compared across all runs
   → Only compare within controlled experiments (same model, same sweep group)
3. loss_use_V affects optimal layer selection:
   → loss_use_V=True: later layers (0.75-0.9), MLP inputs (up_proj)
   → loss_use_V=False: intermediate layers (0.5-0.7), residual outputs (o_proj, down_proj)

**Analysis questions to investigate**:
- Which configs achieve both high metric AND low loss_gap?
- Does symmetry_mean correlate with generalization (loss_gap)?
- Do sweep groups show consistent hyperparameter effects, or are results noisy?
- Base model (Qwen3-4B-Base) vs Instruct: reproducible difference or outlier?
- rot_u + long training (sweep-long-training-*): does low lr stabilize rotations?

Examine data, identify patterns, note confounds. Report observations without conclusions.
""")

