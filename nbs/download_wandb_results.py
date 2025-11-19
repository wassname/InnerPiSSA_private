#!/usr/bin/env python3
"""Download evaluation results and logs from wandb runs with caching."""
# %%

import wandb
import pandas as pd
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

last_major_code_change = '2025-11-19T12:00:00Z'

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
    print(f"Using cached runs after {last_run}")

print("Downloading from wandb...")
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
    print(f"  {run.name} - {summary.get('eval/main_metric')}")
    # Save individual run data
    with open(run_file, 'w') as f:
        json.dump(run_data, f, indent=2)


# load all log files, extract from "## Evaluation complete" to "🥇", and clean off log prefix e.g. `10:20:20 | INFO     | Main metric: 🥇779.973`


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
        "args": ' '.join(run_data['metadata']['args']),

        # Config
        # model id, full eval?, quick?
        'model_name': config.get('model_name'),
        'lr': config.get('lr'),
        'rank': config.get('rank'),
        'num_layers': config.get('num_layers'),
        'loss_type': config.get('loss_type'),
        'scale_s': config.get('scale_s'),
        'n_epochs': config.get('n_epochs'),
        'batch_size': config.get('batch_size'),
        'eval_max_n_dilemmas': config.get('eval_max_n_dilemmas'),

        # Results
        "runtime": summary.get('_runtime'),
        'main_metric': summary.get('eval/main_metric'),
        'effect': summary.get('eval/InnerPiSSA (ours)/effect'),
        'side_effects': summary.get('eval/InnerPiSSA (ours)/side_effects'),
        'degradation': summary.get('eval/InnerPiSSA (ours)/degradation'),
        'p_value': summary.get('eval/InnerPiSSA (ours)/p_value'),
        # TODO also baseline if it's there (prompting, repeng)
    }
    results.append(result)


print('run_data', json.dumps(run_data, indent=2))

df = pd.DataFrame(results)
df = df.sort_values(['model_name', 'created_at'], ascending=False)
print(1, len(df))
# filter out not finished, and not run for at least 1 minute
df = df[(df['state'] == 'finished') & (df['runtime'] > 60)]
# filter out eval_max_n_dilemmas is not [None or < 2000]
print(2, len(df))
df = df[df['eval_max_n_dilemmas'].isna()]
print(3, len(df))
df = df[last_major_code_change < df['created_at']]
print(4, len(df))

output_dir = Path(__file__).parent.parent
output_file = output_dir / 'outputs' / 'wandb_results.csv'
df.to_csv(output_file, index=False)
print(f"\nSaved {len(df)} runs to {output_file}")

# Also save a filtered version with just the key metrics
summary_cols = ['created_at', 'model_name', 'name', 'args', 'main_metric', 'effect', 'side_effects', 'degradation', 
                'lr', 'rank', 'num_layers', 'loss_type', 'scale_s']
df_summary = df[summary_cols].dropna(subset=['main_metric'])
# TODO round numeric cols to .4g
# TODO filter to last major code change
summary_file = output_dir / 'outputs' / 'wandb_summary.csv'
df_summary.to_csv(summary_file, index=False)
print(f"Saved summary to {summary_file}")
# print(df_summary)
from tabulate import tabulate
print(tabulate(df_summary, headers='keys', tablefmt='pipe', floatfmt=".4g"))
print("""Remember only models 4B params or more matter. The code has changes at time has gone by so weight recent ones more. I'm looking for which configuration works across large models.""")
print("""compare to output when running 
uv run python nbs/eval_baseline_repeng.py | tail -n 20
uv run python nbs/eval_baseline_prompting.py | tail -n 20
""")
print("""old arg remapping' \
'# Layer selection
layers → modules
num_layers → n_depths
perc_start → depth_start
end_layers → depth_end
loss_layers → loss_depths

# Training
batch_size → bs
weight_decay → wd
log_n → n_logs

# Adapter
rank → r
ipissa_rotate_u → rot_u
ipissa_rotate_v → rot_v

# Dataset
dataset_max_samples → max_samples
last_n_tokens → n_last_tokens

# Constraints
constr_coherence → coh
coherence_threshold → coh_thresh
coherence_scalar → coh_weight
adaptive_relaxation → coh_adaptive
coeff_diff_temperature → coh_temp
constr_monotonic → mono
monotonic_scaling → mono_weight

# Eval
eval_max_n_dilemmas → eval_max_dilemmas
eval_dataset_max_token_length → eval_max_tokens
""")

# TODO print out `uv run python nbs/train.py -h`
import tyro
config = tyro.cli(TrainingConfig)
print(f"CLI {config}")

try:
    f = proj_root / "outputs" / 'prompting_results.csv'
    print(f"\n\n{f}\n")
    print(pd.read_csv(f).sort_values('main_score', ascending=False))
except Exception as e:
    print(f"Could not load prompting results: {e}")

try:
    f = proj_root / "outputs" / 'repeng_results.csv'
    print(f"\n\n{f}\n")
    print(pd.read_csv(f).sort_values('main_score', ascending=False))
except Exception as e:
    print(f"Could not load repeng results: {e}")

try:
    f = proj_root / "outputs" / 'sSteer_results.csv'
    print(f"\n\n{f}\n")
    print(pd.read_csv(f).sort_values('main_score', ascending=False))
except Exception as e:
    print(f"Could not load sSteer results: {e}")

# # Also load baselines from 
# from ipissa.train.daily_dilemas import format_results_table
# from eval_baseline_repeng import process_and_display_results, load_baselines
# results = load_baselines()
# df_repeng = process_and_display_results(results)
# for model_name in df_repeng['model_id'].unique():
#     df_model = df_repeng[df_repeng["model_id"] == model_name]
#     md_table, df_eff_sz, main_score = format_results_table(
#         df_model,
#         target_col="score_Virtue/Truthfulness",
#         # config=config,
#         target_method="repeng",
#     )
#     # TODO log main_score to csv
# df_repeng
# %%