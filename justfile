
# Default full run
[private]
default:
    #!/bin/bash -x
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

    just eval-baselines
    uv run python nbs/train.py q14b-80gb --model_name=wassname/qwen-14B-codefourchan
    just sweep-train-stages
    just sweep-rotation-angle
    # just sweep-s-norm

    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0 --no_coh --no_mono
    # incoherenet 90 with high lr
    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0 --no_coh
    # incoherent  with high lr
    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0 --no_mono
    # incoherent 128 with high lr
    # 86
    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0 --no_coh --no_mono 
    # incoherent 10 with high lr
    # 66

    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0 
    # 132 a little incoherent even at lr=8e-3
    # lr=1e-3 208

    uv run python nbs/train.py tiny --r=64  --max_rotation_angle=1000 --no_rot_u --scale_s=none
    # 120

    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0  --max_rotation_angle=1
    # 252!

    uv run python nbs/train.py tiny --r=64 --rot_u --data_aware_init --wd=0 --scale_s=mult
    # 430! but incoherent
    uv run python nbs/train.py tiny --r=64 --data_aware_init --wd=0 --scale_s=mult
    # 378
    uv run python nbs/train.py tiny --r=64 --rot_u  --wd=0 --scale_s=mult
    # 24.. but data_aware_init is default so this is just random seed
    

    uv run python nbs/train.py tiny --r=64  --no_loss_use_V --loss_depths=0.5 --loss_modules o_proj down_proj --no_coh --no_mono
    # 57
    uv run python nbs/train.py tiny --r=64 --no_loss_use_V --loss_depths=0.5 --loss_modules o_proj down_proj
    # 173


    # Baseline: current best
    uv run python nbs/train.py q4b-80gb  --n_epochs=10  --eval_max_dilemmas=128 --scale_s=add2 --max_rotation_angle=0.4 --lr=2e-3 --no_rot_u

    # Higher LR
    uv run python nbs/train.py q4b-80gb  --n_epochs=10  --eval_max_dilemmas=128 --scale_s=mult --max_rotation_angle=0.4 --lr=1e-2 --no_rot_u

    # Enable rot_u (might be stable now!)
    uv run python nbs/train.py q4b-80gb  --n_epochs=10 --eval_max_dilemmas=128  --scale_s=add2 --max_rotation_angle=0.2 --lr=2e-3 --rot_u

    # Both together
    uv run python nbs/train.py q4b-80gb  --n_epochs=10  --eval_max_dilemmas=128 --scale_s=add2 --max_rotation_angle=0.1 --lr=1e-2 --rot_u

    # Multiplicative variant
    uv run python nbs/train.py q4b-80gb  --n_epochs=10  --eval_max_dilemmas=128 --scale_s=mult --max_rotation_angle=0.1 --lr=1e-2 --rot_u


    # === ROT_U ABLATION (if max_rotation_angle makes it stable) ===
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --scale_s=add2 \
    --rot_u --max_rotation_angle=0.2

    # === SCALE_S ABLATION ===
    # None (surprisingly viable from old data)
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --scale_s=none  --max_rotation_angle=1.0

    # Mult (alternative)
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --scale_s=mult

    # === LOSS CONSTRAINT ABLATION ===
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --no_coh

    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --no_mono

    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --no_coh --no_mono


    # === BASELINE: Proven config ===
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --lr=8e-3 --loss_depths=0.8 --scale_s=add2

    # === LR SWEEP (most important) ===
    # Lower
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --lr=2e-3 --loss_depths=0.8 --scale_s=add2

    # Higher (risky but high ceiling)
    uv run python nbs/train.py q4b-80gb \
    --n_epochs=10 --eval_max_dilemmas=128 \
    --lr=1e-2 --loss_depths=0.8 --scale_s=add2





    uv run python nbs/train.py q4b-80gb  --n_epochs=10  --eval_max_dilemmas=128 --scale_s=add2 --max_rotation_angle=0.6 --lr=2e-4 --rot_u
    uv run python nbs/train.py q4b-80gb  --n_epochs=10  --eval_max_dilemmas=128 --scale_s=add2 --max_rotation_angle=0.6 --lr=2e-2 --rot_u




    uv run python nbs/train.py q4b-80gb --n_epochs=10 --eval_max_dilemmas=128 --lr=2e-3 --no_coh --no_mono --rot_u
    uv run python nbs/train.py q4b-80gb --n_epochs=10 --eval_max_dilemmas=128 --lr=2e-3 --no_coh
    uv run python nbs/train.py q4b-80gb --n_epochs=10 --eval_max_dilemmas=128 --lr=2e-3 --no_mono --rot_u


    # just sweep-s-norm

    # scratch
    # short run?
    run_exp_small --lr=1e-2 --n_epochs=10 --wd=1

    # try a low lr, long run with senstivie laeyr
    uv run python nbs/train.py q4b-80gb --rot_u --modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --r=8 --no-loss_use_V --lr=3e-4 --n_epochs=60 --wd=1
    uv run python nbs/train.py q4b-80gb --rot_u --modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --r=8 --loss_use_V --lr=3e-4 --n_epochs=60 --wd=1
    uv run python nbs/train.py q4b-80gb --modules v_proj o_proj gate_proj up_proj down_proj --r=16 --no-loss_use_V --lr=3e-4 --n_epochs=60 --wd=1
    uv run python nbs/train.py q4b-80gb --modules v_proj o_proj gate_proj up_proj down_proj --r=8 --loss_use_V --lr=3e-4 --n_epochs=60 --wd=1
    

    uv run python nbs/train.py q4b-80gb --mono_weight=1000 --coh_weight=.1
    uv run python nbs/train.py q4b-80gb --mono_weight=1000 --coh_weight=1000
    uv run python nbs/train.py q4b-80gb --wd=100
    # long
    # run_exp_small --n_epochs=1000

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
    just ablate-paper

sweep-depths:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-depths-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    
    # Test different number of depth layers
    for n_depths in 1 3 5 8 12 28 52; do
        echo "=== Number of depth layers: $n_depths ==="
        $BASE --n_depths=$n_depths
    done

sweet-start-end:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-start-end-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    
    # Test different layer ranges
    for start in 0.0 0.1 0.2 0.3 0.4; do
        for end in -1 -3 -5 -7 -9; do
            echo "=== Layer range: start=$start, end=$end ==="
            $BASE --depth_start=$start --depth_end=$end
        done
    done

sweep-scale:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-scale-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    
    # Test different scaling mechanisms
    for scale in add_tanh add2 none mult; do
        echo "=== Scale mechanism: $scale ==="
        $BASE --scale_s=$scale
    done

# Paper ablation suite
ablate-paper:
    #!/bin/bash -x
    # make sure baselines are cached
    just eval-baselines
    just run-models
    just sweep-train-stages
    just ablate-constraints
    just ablate-modules
    just sweep-layers # [/]
    just sweep-wd # [/]
    just sweep-lr
    just data-efficiency
    just run-seeds
    just sweep-rank
    just sweep-scale
    # just sweep-s-norm
    just sweep-depths
    just sweet-start-end:
    just sweep-rotation-angle
    just sweep-long-training
    just scratch
    

ablate-constraints:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="ablate-constraints-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    $BASE
    $BASE --no_mono --no_coh
    $BASE --mono --no_coh
    $BASE --no_mono --coh

    $BASE --no_rot_u --no_rot_v
    $BASE --rot_u --no_rot_v
    # $BASE --no_rot_u --rot_v # default
    $BASE --scale_s=none
    $BASE --adapter_type lora
    $BASE --no_coh_adaptive
    $BASE --no_data_aware_init
    $BASE --loss_use_V --loss_modules up_proj
    $BASE --no_loss_use_V --loss_depths=0.5 --loss_modules o_proj down_proj

sweep-layers:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-layers-V-$(date +%Y%m%d-%H%M)"
    for depth in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
        uv run python nbs/train.py q4b-80gb --loss_depths=$depth --loss_use_V --loss_modules up_proj
    done
    export WANDB_RUN_GROUP="sweep-layers-$(date +%Y%m%d-%H%M)"
    for depth in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
        uv run python nbs/train.py q4b-80gb --loss_depths=$depth --no_loss_use_V --loss_modules o_proj down_proj
    done

sweep-wd:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="ablate-wd-$(date +%Y%m%d-%H%M)"
    for wd in 0 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e-0 1e-1; do
        uv run python nbs/train.py q4b-80gb --wd=$wd
    done

sweep-lr:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-lr-$(date +%Y%m%d-%H%M)"
    for lr in 1e-0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6; do
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
    uv run python nbs/train.py q14b-80gb --model_name=wassname/qwen-14B-codefourchan
    uv run python nbs/train.py l8b-80gb

    uv run python nbs/train.py gemma270m-80gb
    uv run python nbs/train.py gemma1b-80gb
    uv run python nbs/train.py gemma4b-80gb
    uv run python nbs/train.py gemma12b-80gb

    uv run python nbs/train.py q32b-80gb
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
    for r in 8 16 32 64 128 256 512 1024; do
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
    uv run python nbs/train.py --model_name=Qwen/Qwen3-0.6B --bs=64

# Sweep max rotation angle for output symmetry
sweep-rotation-angle:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-rotation-angle-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    
    # Test different max angles (radians)
    for angle in 0.1 0.2 0.3 0.5 1.0 inf; do
        echo "=== Max rotation angle: $angle rad ==="
        $BASE --max_rotation_angle=$angle
    done
    
    # we can also have inf rotation angle, but not S scaling and both or one matrix
    $BASE --max_rotation_angle=inf --scale_s=none --rot_u --rot_v
    $BASE --max_rotation_angle=inf --scale_s=none --rot_u --no_rot_v
    $BASE --max_rotation_angle=inf --scale_s=none --rot_v --no_rot_u

    # Compare with best S_MEAN_ABS init
    S_MEAN_ABS=True uv run python nbs/train.py q4b-80gb --max_rotation_angle=0.3 --data_aware_init
    S_MEAN_ABS=True uv run python nbs/train.py q4b-80gb --max_rotation_angle=inf --data_aware_init

# Sweep long training with low lr to test if unstable features stabilize
sweep-long-training:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-long-training-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-80gb"
    
    # Test if rot_u stabilizes with longer training at lower lr
    for lr in 1e-4 3e-4 1e-3; do
        for n_epochs in 20 40 60; do
            echo "=== Long training: lr=$lr, epochs=$n_epochs ==="
            $BASE --rot_u --r=8 --lr=$lr --n_epochs=$n_epochs
        done
    done
    
    # Test all "unstable" module types with conservative settings
    for modules in "q_proj k_proj v_proj" "q_proj k_proj v_proj o_proj"; do
        echo "=== Unstable modules: $modules ==="
        $BASE --rot_u --modules $modules --r=8 --lr=3e-4 --n_epochs=60
    done

paper:
    wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.8.26/quarto-1.8.26-linux-amd64.deb /tmp/quarto.deb
    sudo dpkg -i /tmp/quarto.deb
    quarto render paper.qmd --to html
    quarto render paper.qmd --to gfm

sync:
    rsync -avzr --progress vast1:/workspace/InnerPiSSA_private/outputs/baselines/ ./outputs/baselines/
    rsync -avzr --progress "vast1:/workspace/InnerPiSSA_private/outputs/*.csv" ./outputs/


sweep-train-stages:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-training-stages-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py l8b-80gb"

    $BASE --model_name=allenai/Olmo-3-1025-7B
    $BASE --model_name=allenai/Olmo-3-7B-Instruct-SFT
    $BASE --model_name=allenai/Olmo-3-7B-Instruct-DPO
    $BASE --model_name=allenai/Olmo-3-7B-Instruct

    # $BASE --model_name=allenai/Olmo-3-1025-7B
    $BASE --model_name=allenai/Olmo-3-7B-Think-SFT
    $BASE --model_name=allenai/Olmo-3-7B-Think-DPO
    $BASE --model_name=allenai/Olmo-3-7B-Think
    $BASE --model_name=allenai/Olmo-3-7B-RL-Zero-Mix

# Sweep preference direction computation methods
sweep-pref-dir:
    #!/bin/bash -x
    export WANDB_RUN_GROUP="sweep-pref-dir-$(date +%Y%m%d-%H%M)"
    BASE="uv run python nbs/train.py q4b-24gb"
    
    # Single-direction methods
    echo "=== pref_dir_method: mean (baseline) ==="
    $BASE --pref_dir_method=mean
    
    echo "=== pref_dir_method: pca1 ==="
    $BASE --pref_dir_method=pca1
    
    # Multi-direction methods (vary k)
    for method in pca2 pca4 top_s adapter_dims; do
        for k in 16 32 64 128; do
            echo "=== pref_dir_method: $method, k=$k ==="
            $BASE --pref_dir_method=$method --pref_dir_k=$k
        done
    done
