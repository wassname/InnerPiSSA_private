
# Default full run
[private]
default:
    # just scratch
    just ablate-paper

# Generate shell completions for the training script
completions:
    #!/bin/bash -x
    uv run python nbs/train.py . --tyro-write-completion zsh ~/.zfunc/_01_functions_py

# My temporary experiments and non core sweeps
scratch:
    #!/bin/bash -x
    
    # Set group for this sweep (all runs will be grouped together in wandb)
    export WANDB_RUN_GROUP="ablation-$(date +%Y%m%d-%H%M)"
    echo "WandB group: $WANDB_RUN_GROUP"
    
    # Base config for small model ablations
    BASEsmall="uv run python nbs/train.py q4b-80gb"
    BASE="uv run python nbs/train.py l8b-80gb"
    BASElarger="uv run python nbs/train.py gemma12b-80gb"

    # Helper to run with base + extra args
    run_exp() {
        echo "=== Running: $@ ==="
        $BASE "$@"
    }
    # Helper to run with base + extra args
    run_exp_small() {
        echo "=== Running: $@ ==="
        $BASEsmall "$@"
    }

    run_exp_larger() {
        echo "=== Running: $@ ==="
        $BASElarger "$@"
    }
    # scratch
    # short run?
    run_exp_small --lr=1e-2 --n_epochs=4 --wd=1
    uv run python nbs/train.py q4b-80gb --mono_weight=1000 --coh_weight=.1
    uv run python nbs/train.py q4b-80gb --mono_weight=1000 --coh_weight=1000
    uv run python nbs/train.py q4b-80gb --wd=100
    # long
    # run_exp_small --n_epochs=400

    # make sure baselines are cached
    uv run python nbs/eval_baseline_wassname_Ssteer_baseline.py
    uv run python nbs/eval_baseline_repeng.py
    uv run python nbs/eval_baseline_prompting.py

    # echo "### Loss type ablations ###"
    # run_exp_small --loss_type=raw
    # run_exp_small --loss_type=tanh_sym
    # run_exp_small --loss_type=softpl_strong_up #   # default
    # run_exp_small --loss_type=softpl_ratio
    # run_exp_small --loss_type=logsig_dpo
    # run_exp_small --loss_type=logsig_weak_up
    # run_exp_small --loss_type=focal_balanced

    echo "### Number of layers ablations ###"
    run_exp_small --n_depths=1
    run_exp_small --n_depths=3
    run_exp_small --n_depths=5  # default
    run_exp_small --n_depths=8
    run_exp_small --n_depths=12
    
    echo "### Layer range ablations ###"
    run_exp_small --depth_start=0.1
    run_exp_small --depth_start=0.3  # default
    run_exp_small --depth_start=0.5
    run_exp_small --depth_end=-1
    run_exp_small --depth_end=-5

    # echo "### Scale mechanism ablations ###"
    # run_exp --scale_s=add_tanh
    # run_exp --scale_s=add2   # default
    # run_exp --scale_s=none
    # run_exp --scale_s=mult

# Paper ablation suite
ablate-paper:
    # make sure baselines are cached
    just eval-baselines

    just sweep-rank
    just ablate-modules
    just run-models
    just run-seeds
    just data-efficiency
    
    just ablate-constraints
    just sweep-layers
    just sweep-wd
    just sweep-lr

ablate-constraints:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="ablate-constraints-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    $BASE
    $BASE --no_mono --no_coh
    $BASE --mono --no_coh
    $BASE --no_mono --coh
    $BASE --no_rot_u --no_rot_v
    $BASE --scale_s=none
    $BASE --adapter_type lora
    $BASE --no_coh_adaptive
    $BASE --no_data_aware_init

sweep-layers:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-layers-$(date +%Y%m%d-%H%M)"
    for depth in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
        uv run python nbs/train.py q4b-80gb --loss_depths=$depth
    done

sweep-wd:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="ablate-wd-$(date +%Y%m%d-%H%M)"
    for wd in 0 0.001 0.01 0.1 1.0 10.0 100.0 1000.0; do
        uv run python nbs/train.py q4b-80gb --wd=$wd
    done

sweep-lr:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-lr-$(date +%Y%m%d-%H%M)"
    for lr in 1e-0 1e-1 1e-2 1e-3 1e-4 1e-5; do
        uv run python nbs/train.py q4b-80gb --lr=$lr
    done

ablate-modules:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="ablate-modules-$(date +%Y%m%d-%H%M)"
    uv run python nbs/train.py q4b-80gb --modules o_proj down_proj --experiment_name="layers residual out"
    uv run python nbs/train.py q4b-80gb --modules gate_proj up_proj down_proj --experiment_name="attn.down mlp.up"
    uv run python nbs/train.py q4b-80gb --modules gate_proj up_proj --experiment_name="mlp up largest output dim"
    uv run python nbs/train.py q4b-80gb --modules q_proj k_proj v_proj --experiment_name="attention up"
    uv run python nbs/train.py q4b-80gb --modules v_proj --experiment_name="attention v"
    uv run python nbs/train.py q4b-80gb --modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --r=8 --experiment_name="all" --bs=16

run-models:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="run-models-$(date +%Y%m%d-%H%M)"
    uv run python nbs/train.py q06b-24gb
    uv run python nbs/train.py q4b-80gb
    uv run python nbs/train.py q4b-80gb --model_name=Qwen/Qwen3-4B-Base
    uv run python nbs/train.py q14b-80gb
    uv run python nbs/train.py l8b-80gb
    uv run python nbs/train.py gemma270m-80gb
    uv run python nbs/train.py gemma1b-80gb
    uv run python nbs/train.py gemma4b-80gb
    uv run python nbs/train.py gemma12b-80gb
    # TODO try a 32b model... when I have more hdd space

eval-baselines:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="eval-baselines-$(date +%Y%m%d-%H%M)"
    uv run python nbs/eval_baseline_prompting.py
    uv run python nbs/eval_baseline_repeng.py
    uv run python nbs/eval_baseline_wassname_Ssteer_baseline.py

sweep-rank:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-rank-$(date +%Y%m%d-%H%M)"
    for r in 32 64 128 256 512; do
        uv run python nbs/train.py q4b-80gb --r=$r
    done

run-seeds:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="run-seeds-$(date +%Y%m%d-%H%M)"
    for seed in 42 123 456; do
        uv run python nbs/train.py q4b-80gb --seed=$seed
    done

# Data efficiency: what's minimum viable sample count?
data-efficiency:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="data-efficiency-$(date +%Y%m%d-%H%M)"
    echo "WandB group: $WANDB_RUN_GROUP"
    BASE="uv run python nbs/train.py q4b-80gb"
    for n in 50 100 200 400 800 2000; do
        echo "=== Training with $n samples ==="
        $BASE --max_samples=$n --experiment_name="data_$n"
    done


# Quick test run
quick:
    uv run python nbs/train.py --quick --model_name=Qwen/Qwen3-0.6B --bs=64
