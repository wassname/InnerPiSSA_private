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
            print(f'File: {list(run.files())}')

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
    argv = " ".join(program = ["python"] + [meta["program"]] + meta.get("args", []))
    
    run_data = {
        'run_id': run.id,
        'name': run.name,
        'argv': argv,
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
    logger.info(f"  {run.name} - metric={summary.get('eval/main_metric', np.nan):.3g}, created_at={run.created_at}")
    # Save individual run data
    with open(run_file, 'w') as f:
        json.dump(run_data, f, indent=2)


if len(runs_data) == 0:
    logger.warning("No runs found!")
    exit(0)

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

# config_keys = config.keys()

# logger.debug(f'DEBUG run_data  {json.dumps(run_data, indent=2)}')

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
    'runtime',
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
=== SWEEP ANALYSIS TASK ===

be aware of major_code_changes : {major_code_changes}, runs can't be compared easily on the other side of these gaps

Goal: Identify which hyperparameters generalize across large models (ideally 4B+ params).

First read
- README.md - to understand context, resecially Metric Reference section
- {f_help} - to understand cli options meaning and default values
- {summary_file} - to see the results
- Baseline CSVs: {fpr}, {fre}, {fss} - to see how well prompting, which we want to beat, did
- Generated by `justfile`


Remember loss_depth depends on loss_use_V, when true we expect later layers to benefit, otherwise intermediate layers

The full csv of run configs are in {results_file}, and the logs can be on wandb at https://wandb.ai/wassname/InnerPiSSA/runs although I'm not sure how to fetch

Then tell me what you think!
            
""")