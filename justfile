

# Quick test run
quick:
    uv run python nbs/train.py --quick --model-name=Qwen/Qwen3-0.6B --batch-size=64

# Default full run
default:
    uv run python nbs/train.py

# Full ablation suite
run:
    #!/bin/bash -x
    
    # Base config for small model ablations
    BASE="uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32"
    # BASE="uv run python nbs/train.py --eval-max-n-dilemmas=64"
    
    # Helper to run with base + extra args
    run_exp() {
        echo "=== Running: $@ ==="
        $BASE "$@"
    }

    
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
    run_exp --weight-decay=0.0
    run_exp --weight-decay=0.1  # default
    run_exp --weight-decay=1.0
    run_exp --weight-decay=10.0


    # === Loss type ablations ===
    echo "### Loss type ablations ###"
    run_exp --loss-type=softplus
    run_exp --loss-type=tanh2v1
    run_exp --loss-type=softplus_only
    run_exp --loss-type=logsigmoid  # default
    
    # === Scale mechanism ablations ===
    echo "### Scale mechanism ablations ###"
    run_exp --scale-s=add
    run_exp --scale-s=add2   # default
    run_exp --scale-s=none
    run_exp --scale-s=mult
    
    # === Rotation ablations ===
    echo "### Rotation ablations ###"
    run_exp --no-ipissa-rotate-u
    run_exp --no-ipissa-rotate-v
    run_exp --no-ipissa-rotate-u --no-ipissa-rotate-v
    # run_exp --no-ipissa-rotate-u --no-ipissa-rotate-v --scale-s=none  # minimal adapter

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
    run_exp --rank=8  --num-layers=1
    run_exp --rank=24  --num-layers=1  # default
    run_exp --rank=64  --num-layers=1
    run_exp --rank=256  --num-layers=1
    run_exp --rank=512  --num-layers=1
    
    
    # === Number of layers ablations ===
    echo "### Number of layers ablations ###"
    run_exp --num-layers=1
    run_exp --num-layers=3
    run_exp --num-layers=5  # default
    run_exp --num-layers=8
    run_exp --num-layers=12 --perc-start=0.15
    
    # === Layer range ablations ===
    echo "### Layer range ablations ###"
    run_exp --perc-start=0.1
    run_exp --perc-start=0.3  # default
    run_exp --perc-start=0.5
    run_exp --end-layers=-1
    run_exp --end-layers=-5

# Focused ablations for paper
ablate-core:
    #!/bin/bash -ex
    BASE="uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=128 --batch-size=32"
    
    # Core comparisons
    echo "=== Baseline (full method) ==="
    $BASE
    
    echo "=== No rotation ==="
    $BASE --no-ipissa-rotate-u --no-ipissa-rotate-v
    
    echo "=== No scaling ==="
    $BASE --scale-s=none
    
    echo "=== Minimal (no rotation, no scaling) ==="
    $BASE --no-ipissa-rotate-u --no-ipissa-rotate-v --scale-s=none
    
    echo "=== Different loss ==="
    $BASE --loss-type=softplus_only

# Large model run
run-large:
    uv run python nbs/train.py \
        --model-name=Qwen/Qwen3-4B-Instruct-2507 \
        --batch-size=6 \
        --rank=24 \
        --n-epochs=100
