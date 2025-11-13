

# Quick test run
quick:
    uv run python nbs/train.py --quick --model_name=Qwen/Qwen3-0.6B --batch_size=64

# Default full run
default:
    uv run python nbs/train.py

# Full ablation suite
run:
    #!/bin/bash -x
    
    # Base config for small model ablations
    BASEsmall="uv run python nbs/train.py --model_name=Qwen/Qwen3-0.6B --eval_max_n_dilemmas=256 --batch_size=24"
    BASE="uv run python nbs/train.py --eval_max_n_dilemmas=256"

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

    run_exp --no-loss_ds_pref_dir
    run_exp --loss_ds_pref_dir
    run_exp_small --no-loss_ds_pref_dir
    run_exp_small --loss_ds_pref_dir

    run_exp_small --no_loss_full_u
    
    # === Layer selection ablations ===
    echo "### Layer selection ablations ###"
    run_exp --layers k_proj q_proj v_proj gate_proj up_proj --rank=16
    run_exp --layers o_proj down_proj
    run_exp --layers gate_proj up_proj  # default
    run_exp --layers gate_proj down_proj
    run_exp --layers o_proj up_proj
    run_exp --layers gate_proj up_proj down_proj o_proj --rank=16
    run_exp --layers k_proj q_proj v_proj --rank=16
    # all pre compute

    # === Weight decay ablations ===
    echo "### Weight decay ablations ###"
    run_exp --weight_decay=0.0
    run_exp --weight_decay=0.1  # default
    run_exp --weight_decay=1.0
    run_exp --weight_decay=10.0


    # === Loss type ablations ===
    echo "### Loss type ablations ###"
    run_exp --loss_type=softplus2
    run_exp --loss_type=tanh2v1
    run_exp --loss_type=softplus_only
    run_exp --loss_type=logsigmoid  # default
    
    # === Scale mechanism ablations ===
    echo "### Scale mechanism ablations ###"
    run_exp --scale_s=add_tanh
    run_exp --scale_s=add2   # default
    run_exp --scale_s=none
    run_exp --scale_s=mult
    
    # === Rotation ablations ===
    echo "### Rotation ablations ###"
    run_exp --ipissa_rotate_u --no_ipissa_rotate_v
    run_exp --ipissa_rotate_u 
    run_exp --no_ipissa_rotate_v
    # run_exp --no_ipissa_rotate_u --no_ipissa_rotate_v --scale_s=none  # minimal adapter

    # === Learning rate ablations ===
    echo "### Learning rate ablations ###"
    run_exp --lr=1e-1
    run_exp --lr=1e-2
    run_exp --lr=3e-3
    # run_exp --lr=1e-3  # default
    run_exp --lr=6e-4
    run_exp --lr=1e-4
    
    
    # === Rank ablations ===
    echo "### Rank ablations ###"
    run_exp --rank=8  --num_layers=1
    run_exp --rank=24  --num_layers=1  # default
    run_exp --rank=64  --num_layers=1
    run_exp --rank=256  --num_layers=1
    run_exp --rank=512  --num_layers=1
    
    
    # === Number of layers ablations ===
    echo "### Number of layers ablations ###"
    run_exp --num_layers=1
    run_exp --num_layers=3
    run_exp --num_layers=5  # default
    run_exp --num_layers=8
    run_exp --num_layers=12 --perc_start=0.15
    
    # === Layer range ablations ===
    echo "### Layer range ablations ###"
    run_exp --perc_start=0.1
    run_exp --perc_start=0.3  # default
    run_exp --perc_start=0.5
    run_exp --end_layers=-1
    run_exp --end_layers=-5


# Focused ablations for paper
ablate-core:
    #!/bin/bash -ex
    BASE="uv run python nbs/train.py --model_name=Qwen/Qwen3-0.6B --eval_max_n_dilemmas=128 --batch_size=32"
    
    # Core comparisons
    echo "=== Baseline (full method) ==="
    $BASE
    
    echo "=== No rotation ==="
    $BASE --no_ipissa_rotate_u --no_ipissa_rotate_v
    
    echo "=== No scaling ==="
    $BASE --scale_s=none
    
    echo "=== Minimal (no rotation, no scaling) ==="
    $BASE --no_ipissa_rotate_u --no_ipissa_rotate_v --scale_s=none
    
    echo "=== Different loss ==="
    $BASE --loss_type=softplus_only

# Large model run
run-large:
    uv run python nbs/train.py \
        --model_name=Qwen/Qwen3-4B-Instruct-2507 \
        --batch_size=6 \
        --rank=24 \
        --n_epochs=100
