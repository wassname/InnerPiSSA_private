

# Quick test run
quick:
    uv run python nbs/train.py --quick --model_name=Qwen/Qwen3-0.6B --batch_size=64

# Default full run
default:
    uv run python nbs/train.py

# Full ablation suite
run:
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

    # make sure baselines are cached
    uv run python nbs/eval_baseline_wassname_Ssteer_baseline.py
    uv run python nbs/eval_baseline_repeng.py
    uv run python nbs/eval_baseline_prompting.py

    # === Loss type ablations ===
    echo "### Loss type ablations ###"
    run_exp --loss_type=softpl_strong_up #   # default
    run_exp --loss_type=softpl_ratio
    run_exp --loss_type=logsig_dpo
    run_exp --loss_type=logsig_weak_up
    run_exp --loss_type=tanh_sym
    run_exp --loss_type=focal_balanced
    run_exp --loss_type=raw
    
    # can we learn with high lr, low steps?
    run_exp_small --lr=1e-0 --n_epochs=2 --rank=8 --num_layers=15 # no this learns a one sides intervention and symmetry escapes it
    run_exp_small --lr=1e-1 --n_epochs=4 --rank=8 --num_layers=15
    run_exp_small --lr=1e-2 --n_epochs=8 --rank=8 --num_layers=15

    # #uv run python nbs/train.py . --model_name=unsloth/Llama-3.1-8B-Instruct --loss_type=tanh_sym_(±)  --batch-size=32
    # # uv run python nbs/train.py gemma12b-80gb --loss_type=tanh_sym_(±)  --batch-size=32
    # uv run python nbs/train.py . --model_name=google/gemma-3-27b-it --loss_type=tanh_sym_(±)  --batch-size=32
    # uv run python nbs/train.py . --model_name=Qwen/Qwen3-32B
    # uv run python nbs/train.py . --model_name=unsloth/Llama-3.3-70B-Instruct
    
    #uv run python nbs/train.py . --lr=2e-3 --n_epochs=30 --model_name=Qwen/Qwen3-14B --loss_type=tanh_sym_(±) --batch-size=12
    # uv run python nbs/train.py . --lr=8e-3 --n_epochs=30 --model_name=openai/gpt-oss-20b --loss_type=tanh_sym_(±) --batch-size=12
    
    # # misc, some models are tricky to beat as they are less censored so prompting works well
    # python nbs/train.py q4b-80gb --loss_type=softpl_strong_up
    # python nbs/train.py q4b-80gb --loss_type=focal_balanced
    # python nbs/train.py l8b-80gb --loss_type=softpl_strong_up
    # python nbs/train.py l8b-80gb --loss_type=focal_balanced

    # lora and dopra baseline
    # uv run python nbs/train.py q4b-80gb --adapter_type lora --loss_type=tanh_sym 
    run_exp --adapter_type lora --loss_type=tanh_sym # lora is too uncontrained
    run_exp --adapter_type dora --loss_type=focal_balanced
    run_exp --adapter_type dora --loss_type=logsig_weak_up
    run_exp --adapter_type dora --loss_type=raw
    run_exp --adapter_type dora --loss_type=raw

    # TODO sweep target layers


    # TODO ablate to make this tabel
    #     Config                      
    # --------------------------------
    # Baseline (no constraints)      
    # + Monotonic only               
    # + Coherence only               
    # + Reversibility only (actually we want do this as it's baked into our metric)      
    # + Monotonic + Coherence         
    # + All three                     


    # === Learning rate ablations ===
    echo "### Learning rate ablations ###"
    run_exp_small --lr=1e-0
    run_exp_small --lr=1e-1
    run_exp_small --lr=1e-2
    run_exp_small --lr=1e-3
    run_exp_small --lr=1e-4
    run_exp_small --lr=1e-5

    # === Weight decay ablations ===
    echo "### Weight decay ablations ###"
    run_exp --weight_decay=0.001
    run_exp --weight_decay=0.01
    run_exp --weight_decay=0.1  # default
    run_exp --weight_decay=1.0
    run_exp --weight_decay=10.0


    # Try differen't models
    uv run python nbs/train.py q14b-80gb --model_name=wassname/qwen-14B-codefourchan
    uv run python nbs/train.py q14b-80gb
    # uv run python nbs/train.py oss20-80gb
    uv run python nbs/train.py gemma12b-80gb
    uv run python nbs/train.py l8b-80gb
    # google/gemma-3-27b-it
    # Qwen/Qwen3-32B
    # unsloth/Llama-3.3-70B-Instruct

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


    # === Rank ablations ===
    echo "### Rank ablations ###"
    run_exp --rank=8  --num_layers=30
    run_exp --rank=24  --num_layers=15  # default
    run_exp --rank=64  --num_layers=5
    run_exp --rank=256  --num_layers=3
    run_exp --rank=512  --num_layers=1
    
    
    # === Number of layers ablations ===
    echo "### Number of layers ablations ###"
    run_exp --num_layers=1
    run_exp --num_layers=3
    run_exp --num_layers=5  # default
    run_exp --num_layers=8
    run_exp --num_layers=12
    
    # === Layer range ablations ===
    echo "### Layer range ablations ###"
    run_exp --perc_start=0.1
    run_exp --perc_start=0.3  # default
    run_exp --perc_start=0.5
    run_exp --end_layers=-1
    run_exp --end_layers=-5
    
    # === Loss layer ablations ===
    echo "### Loss layer ablations ###"
    run_exp --loss_layers=-1  # last layer
    run_exp --loss_layers=-3  # default: 3rd from last
    run_exp --loss_layers=-5  # 5th from last
    run_exp --loss_layers=-10  # 10th from last
    run_exp --loss_layers=-3,-5  # multiple layers
    run_exp --loss_layers=-1,-3,-5  # multiple layers

    # === Rotation ablations ===
    echo "### Rotation ablations ###"
    run_exp --ipissa_rotate_u --no_ipissa_rotate_v
    run_exp --ipissa_rotate_u 
    run_exp --no_ipissa_rotate_v
    # run_exp --no_ipissa_rotate_u --no_ipissa_rotate_v --scale_s=none  # minimal adapter

    # === Scale mechanism ablations ===
    echo "### Scale mechanism ablations ###"
    run_exp --scale_s=add_tanh
    run_exp --scale_s=add2   # default
    run_exp --scale_s=none
    run_exp --scale_s=mult

    # try long
    # run_exp --n_epochs=200
    run_exp_small --n_epochs=200
    


# Focused ablations for paper
ablate-core:
    #!/bin/bash -ex
    
    # Set group for this focused ablation sweep
    export WANDB_RUN_GROUP="core-ablation-$(date +%Y%m%d-%H%M)"
    echo "WandB group: $WANDB_RUN_GROUP"
    
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
    $BASE --loss_type=softpl_strong_up_(+↑-↓)

# Large model run
run-large:
    uv run python nbs/train.py \
        --model_name=Qwen/Qwen3-4B-Instruct-2507 \
        --batch_size=6 \
        --rank=24 \
        --n_epochs=100
