
# 2025-11-11 19:56:42


19:46:28 | INFO     | Evaluation results:
coeff                  -100    -1       0       1       100
Virtue/Truthfulness  0.3161  0.4803  0.5040  0.4062  0.3558
Virtue/Ambition      0.2116  0.3933  0.3428  0.2923  0.2510
19:46:28 | INFO     | Config TrainingConfig(model_name='Qwen/Qwen3-0.6B', quantization_type='none', target_modules='.*\\.(7|10|13|15|17|20|23|25)\\..*(o_proj|up_proj)', batch_size=32, n_epochs=100, lr=0.0006, weight_decay=0.1, log_n=10, grad_accum_steps=10, quick=False, rank=256, scale_s='mult', ipissa_rotate_u=True, ipissa_rotate_v=True, loss_full_u=True, dataset_name='honest', dataset_max_samples=1000, loss_type='logsigmoid', coherence_threshold=1.5, boundary_order=1, last_n_tokens=3, eval_batch_size=None, eval_max_n_dilemmas=None, eval_dataset_max_token_length=196, output_dir=PosixPath('/media/wassname/SGIronWolf/projects5/2025/llm_moral_lb_v2/repeng/outputs/adapters'), use_wandb=True, wandb_project='repeng-steering', save_checkpoints=False)
['nbs/train.py', '--model_name=Qwen/Qwen3-0.6B', '--rank=256', '--target-modules=.*\\.(7|10|13|15|17|20|23|25)\\..*(o_proj|up_proj)', '--batch-size=32']
19:46:32 | INFO     | 
## Unsupervised Transfer Evaluation: Honesty Pairs -> DailyDilemmas Truthfulness
Training: 1000 contrastive honesty pairs | Eval: 907 moral dilemmas (Virtue/Truthfulness + 29 other values)

| Method            | Coeff   |   Transfer (Target) ↑ | p-value   |   ΔNLL ↓ |   Transfer (Others) ↓ |
|:------------------|:--------|----------------------:|:----------|---------:|----------------------:|
| InnerPiSSA (ours) | ±100.0  |                -0.188 | p=0.00    |    0.776 |                 0.089 |
| InnerPiSSA (ours) | ±1.0    |                -0.098 | p=0.00    |    0.056 |                 0.047 |
| random            | ±100.0  |                -0.043 | p=0.14    |    0.053 |                 0.021 |
| PCA (baseline)    | ±100.0  |                -0.004 | p=0.90    |    0.205 |                 0.003 |
| PCA (baseline)    | ±1.0    |                -0     | p=0.99    |    0     |                 0.001 |
| random            | ±1.0    |                -0     | p=1.00    |    0.003 |                 0.001 |

↑ higher is better, ↓ lower is better
Coeff: Magnitude tested (±c means both +c and -c were evaluated)
Best: The sign that produced the larger absolute transfer effect
Transfer (Target): Δ in Truthfulness at best coefficient vs baseline (coeff=0)
Transfer (Others): Mean |Δ| across 29 non-target moral values (precision measure)
ΔNLL: Output degradation (input_nll shift from baseline) at best coefficient
p-value: t-test of target transfer effect vs baseline

['nbs/train.py', '--no-ipissa-rotate-v']
04:12:27 | INFO     | 
## Unsupervised Transfer Evaluation: Honesty Pairs -> DailyDilemmas Truthfulness
Training: 1000 contrastive honesty pairs | Eval: 907 moral dilemmas (Virtue/Truthfulness + 29 other values)

| Method            | Coeff   |   Transfer (Target) ↑ | p-value   |   ΔNLL ↓ |   Transfer (Others) ↓ |
|:------------------|:--------|----------------------:|:----------|---------:|----------------------:|
| InnerPiSSA (ours) | ±1.0    |                 0.041 | p=0.16    |    0.441 |                 0.026 |
| InnerPiSSA (ours) | ±0.5    |                 0.029 | p=0.35    |    0.177 |                 0.026 |
| random            | ±100.0  |                 0.006 | p=0.83    |    0.059 |                 0.009 |
| PCA (baseline)    | ±100.0  |                 0.004 | p=0.89    |    0.021 |                 0.003 |
| PCA (baseline)    | ±1.0    |                 0     | p=0.99    |    0.001 |                 0.001 |
| random            | ±1.0    |                 0     | p=0.99    |    0     |                 0.001 |
| prompting         | ±1.0    |                 0.071 |     0.030 |     0.020 |            0.117 |  

'['\''nbs/train.py'\'',' ''\''--model_name=Qwen/Qwen3-0.6B'\'',' ''\''--eval_max_n_dilemmas=64'\'',' ''\''--target-modules=.*\\.(7|10|13|15|17|20|23|25)\\..*(gate_proj|down_proj)'\'',' ''\''--rank=256'\'',' ''\''--batch-size=32'\'',' ''\''--lr=6e-4'\'',' ''\''--weight-decay=0.1'
'' 
## Unsupervised Transfer Evaluation: Honesty Pairs '->' DailyDilemmas Truthfulness Training: 1000 contrastive honesty pairs '|' Eval: 64 moral dilemmas '(Virtue/Truthfulness' + 29 other 'values)'

 Virtue/Truthfulness 0.5464 0.9978 0.8036 0.7888 0.7261 0.7040 0.4758
 Virtue/Ambition 0.2781 NaN 0.3307 0.3307 0.3288 0.3272 '0.1771'

|' Method '|' Coeff '|' Transfer '(Target)' ↑ '|' p-value '|' ΔNLL ↓ '|' Transfer '(Others)' ↓ '|
|:------------------|:--------|----------------------:|:----------|---------:|----------------------:|'
|' InnerPiSSA '(ours)' '|' ±5.0 '|' -0.313 '|' p=0.01 '|' 2.502 '|' 0.163 '|
|' PCA '(baseline)' '|' ±100.0 '|' -0.106 '|' p=0.43 '|' 0.199 '|' 0.07 '|
|' InnerPiSSA '(ours)' '|' ±0.5 '|' -0.063 '|' p=0.65 '|' 0.015 '|' 0.047 '|
|' random '|' ±100.0 '|' -0.04 '|' p=0.76 '|' 0.186 '|' 0.022 '|
|' PCA '(baseline)' '|' ±1.0 '|' -0.002 '|' p=0.99 '|' 0 '|' 0.002 '|
|' random '|' ±1.0 '|' -0.001 '|' p=0.99 '|' 0.001 '|' 0.002 '|
|' InnerPiSSA '(ours)' '|' ±1.0 '|' 0.209 '|' p=1.00 '|' 0.526 '|' 0.721 '|'

## Unsupervi


| Method    | Coeff   |   Target Effect |   Side Effects |   p-value |   Output Quality |               Overall |
|           |         |       Δ Truth ↑ |      Δ Other ↓ |           |          Δ NLL ↓ |   Δ Truth/(1 + Δ NLL) |
|:----------|:--------|----------------:|---------------:|----------:|-----------------:|----------------------:|
| prompting | ±1.0    |           0.071 |          0.030 |     0.020 |            0.117 |                 0.063 |

**Honesty Transfer to Moral Dilemmas (1000 train → 907 test).** Target Effect: Δ Truthfulness score vs baseline. Side Effects: mean |Δ| across 36 non-target values. Output Quality: coherence degradation (ΔNLL). p-values from two-tailed t-test.




12:32:18 | INFO     | ## Evaluation complete 20251112_123218.

nbs/train.py --quick --model_name=Qwen/Qwen3-0.6B --batch-size=64 --dataset_max_samples=200
12:32:18 | INFO     | Results for method: InnerPiSSA (ours)
coeff                -15.0   -5.0    -2.0   ...    1.0     2.0     5.0 
Virtue/Truthfulness  0.490  0.4592  0.4708  ...  0.6949  0.6131  0.4676
Virtue/Ambition      0.167  0.1695  0.1692  ...  0.3244  0.3162  0.1740

[2 rows x 8 columns]

12:32:18 | INFO     | Results for method: PCA (baseline)
coeff                -100.0  -1.0     0.0     1.0     100.0
Virtue/Truthfulness  0.6415  0.6959  0.6949  0.6952  0.6970
Virtue/Ambition      0.3299  0.3254  0.3244  0.3254  0.3254

12:32:18 | INFO     | Results for method: prompting
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  0.7231  0.7923  0.7484
Virtue/Ambition      0.2597  0.3301  0.3270

12:32:18 | INFO     | Results for method: random
coeff                -100.0  -1.0     0.0     1.0     100.0
Virtue/Truthfulness  0.7228  0.6958  0.6949  0.6961  0.6226
Virtue/Ambition      0.2552  0.3254  0.3244  0.3254  0.3323

12:32:37 | INFO     |


| Method            | Coeff   |   Target Effect |   Side Effects |   p-value |   Output Quality |   Normalized Gain (%) |
|:------------------|:--------|----------------:|---------------:|----------:|-----------------:|----------------------:|
|                   |         |       Δ Truth ↑ |      Δ Other ↓ |           |          Δ NLL ↓ |                       |
| InnerPiSSA (ours) | ±1.0    |           0.245 |          0.117 |     0.589 |            0.314 |                18.660 |
| InnerPiSSA (ours) | ±2.0    |           0.321 |          0.162 |     0.708 |            1.403 |                13.346 |
| InnerPiSSA (ours) | ±5.0    |           0.332 |          0.165 |     0.986 |            3.063 |                 8.178 |
| InnerPiSSA (ours) | ±15.0   |           0.302 |          0.144 |     1.000 |            3.429 |                 6.809 |
| random            | ±100.0  |           0.072 |          0.045 |     0.159 |            0.157 |                 6.247 |
| prompting         | ±1.0    |           0.069 |          0.045 |     0.764 |            0.109 |                 6.238 |
| PCA (baseline)    | ±100.0  |           0.053 |          0.039 |     0.312 |            0.263 |                 4.231 |
| PCA (baseline)    | ±1.0    |          -0.001 |          0.002 |     0.524 |            0.000 |                -0.104 |
| random            | ±1.0    |          -0.001 |          0.003 |     0.861 |            0.000 |                -0.126 |

**Honesty Transfer to Morality (Daily Dilemmas (200 train → 64 test).** Model: Qwen/Qwen3-0.6B. Target Effect: Δ Truthfulness score vs baseline. Side Effects: mean |Δ| across 31 non-target values. Output Quality: coherence degradation (ΔNLL). Normalized Gain (%) = 100 × Δ Truth / (1 + Δ NLL); higher values indicate more efficient steering (truthfulness gain per unit coherence cost). p-values from linear regression testing monotonic dose-response (effect scales with coeff).


# 2025-11-13 05:39:33 Sweep on 0.6B


And on 4b (still running)

    + BASE='uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32'
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=logsigmoid
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/kjgzqlsa
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=logsigmoid
    14:51:41 | INFO     | 🥇165.414
    wandb: 🚀 View run bright-armadillo-4 at: 
    + BASE='uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32'
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=softplus
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/2ht29yem
    wandb: 🚀 View run curious-morning-5 at: 
    + BASE='uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32'
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=softplus
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/5q85tqs5
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=softplus
    15:07:02 | INFO     | 🥇185.209
    wandb: 🚀 View run rosy-bee-6 at: https://wandb.ai/wassname/InnerPiSSA/runs/5q85tqs5
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=softplus_only
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/40mnx9c3
    wandb: 🚀 View run distinctive-sound-7 at: 
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=tanh2v1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/wjfoq1f6
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=tanh2v1
    15:16:17 | INFO     | 🥇172.279
    wandb: 🚀 View run glamorous-snowflake-8 at: https://wandb.ai/wassname/InnerPiSSA/runs/wjfoq1f6
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=logsigmoid
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/kixyf56g
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --loss-type=logsigmoid
    15:23:59 | INFO     | 🥇151.932
    wandb: 🚀 View run faithful-grass-9 at: https://wandb.ai/wassname/InnerPiSSA/runs/kixyf56g
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=add
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/vh865gr6
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=add
    15:31:33 | INFO     | 🥇163.280
    wandb: 🚀 View run honest-jazz-10 at: https://wandb.ai/wassname/InnerPiSSA/runs/vh865gr6
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=add2
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/cqpy2pqa
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=add2
    15:39:20 | INFO     | 🥇154.053
    wandb: 🚀 View run rose-puddle-11 at: https://wandb.ai/wassname/InnerPiSSA/runs/cqpy2pqa
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=none
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/dlpttcfl
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=none
    15:46:58 | INFO     | 🥇148.232
    wandb: 🚀 View run sleek-rain-12 at: https://wandb.ai/wassname/InnerPiSSA/runs/dlpttcfl
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=mult
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/3b84enmy
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --scale-s=mult
    15:54:40 | INFO     | 🥇166.212
    wandb: 🚀 View run ruby-river-13 at: https://wandb.ai/wassname/InnerPiSSA/runs/3b84enmy
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-u
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/j1hwmsbk
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-u
    16:01:54 | INFO     | 🥇153.041
    wandb: 🚀 View run floral-yogurt-14 at: https://wandb.ai/wassname/InnerPiSSA/runs/j1hwmsbk
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-v
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/qw0meqii
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-v
    16:09:20 | INFO     | 🥇175.717
    wandb: 🚀 View run earthy-wood-15 at: https://wandb.ai/wassname/InnerPiSSA/runs/qw0meqii
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-u --no-ipissa-rotate-v
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/dl1s67ps
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-u --no-ipissa-rotate-v
    16:16:12 | INFO     | 🥇19.852
    wandb: 🚀 View run lyric-terrain-16 at: https://wandb.ai/wassname/InnerPiSSA/runs/dl1s67ps
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --no-ipissa-rotate-u --no-ipissa-rotate-v --scale-s=none
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/d7x0kqpb
    wandb: 🚀 View run glad-dawn-17 at: 
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=1e-1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/zqb0bhv3
    wandb: 🚀 View run curious-bush-18 at: 
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=1e-2
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/anbnfqal
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=1e-2
    16:30:19 | INFO     | 🥇52.278
    wandb: 🚀 View run clear-universe-19 at: https://wandb.ai/wassname/InnerPiSSA/runs/anbnfqal
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=6e-4
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/lfiuxw2v
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=6e-4
    16:37:36 | INFO     | 🥇123.547
    wandb: 🚀 View run firm-plasma-20 at: https://wandb.ai/wassname/InnerPiSSA/runs/lfiuxw2v
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=1e-4
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/kd8na24z
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --lr=1e-4
    16:44:35 | INFO     | 🥇11.593
    wandb: 🚀 View run floral-firefly-21 at: https://wandb.ai/wassname/InnerPiSSA/runs/kd8na24z
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --weight-decay=0.0
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/tdrdvwp3
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --weight-decay=0.0
    16:51:58 | INFO     | 🥇164.860
    wandb: 🚀 View run vivid-galaxy-22 at: https://wandb.ai/wassname/InnerPiSSA/runs/tdrdvwp3
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --weight-decay=0.1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/cqf82nok
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --weight-decay=0.1
    16:59:32 | INFO     | 🥇151.211
    wandb: 🚀 View run expert-silence-23 at: https://wandb.ai/wassname/InnerPiSSA/runs/cqf82nok
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --weight-decay=1.0
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/8yymsdng
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --weight-decay=1.0
    17:06:53 | INFO     | 🥇157.863
    wandb: 🚀 View run stoic-bird-24 at: https://wandb.ai/wassname/InnerPiSSA/runs/8yymsdng
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=8
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/u7m0tqq0
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=8
    17:13:43 | INFO     | 🥇161.332
    wandb: 🚀 View run azure-energy-25 at: https://wandb.ai/wassname/InnerPiSSA/runs/u7m0tqq0
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=24
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/dg2aiua0
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=24
    17:21:03 | INFO     | 🥇161.435
    wandb: 🚀 View run pretty-spaceship-26 at: https://wandb.ai/wassname/InnerPiSSA/runs/dg2aiua0
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=64
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/s73147yo
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=64
    17:28:12 | INFO     | 🥇113.040
    wandb: 🚀 View run wild-elevator-27 at: https://wandb.ai/wassname/InnerPiSSA/runs/s73147yo
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=256
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/9a6er62w
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=256
    17:35:45 | INFO     | 🥇83.457
    wandb: 🚀 View run apricot-wind-28 at: https://wandb.ai/wassname/InnerPiSSA/runs/9a6er62w
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=512
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/bl7hqml6
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --rank=512
    17:45:14 | INFO     | 🥇124.805
    wandb: 🚀 View run smooth-meadow-29 at: https://wandb.ai/wassname/InnerPiSSA/runs/bl7hqml6
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers gate_proj up_proj
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/bm0rb6q5
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers gate_proj up_proj
    17:52:33 | INFO     | 🥇152.207
    wandb: 🚀 View run silvery-pine-30 at: https://wandb.ai/wassname/InnerPiSSA/runs/bm0rb6q5
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers gate_proj down_proj
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/dfwz71ua
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers gate_proj down_proj
    17:59:35 | INFO     | 🥇176.001
    wandb: 🚀 View run fearless-shadow-31 at: https://wandb.ai/wassname/InnerPiSSA/runs/dfwz71ua
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers o_proj up_proj
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/rh91npx8
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers o_proj up_proj
    18:06:24 | INFO     | 🥇150.567
    wandb: 🚀 View run helpful-bee-32 at: https://wandb.ai/wassname/InnerPiSSA/runs/rh91npx8
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers gate_proj up_proj down_proj o_proj --rank=16
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/utfpn6sb
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers gate_proj up_proj down_proj o_proj --rank=16
    18:14:41 | INFO     | 🥇141.057
    wandb: 🚀 View run proud-aardvark-33 at: https://wandb.ai/wassname/InnerPiSSA/runs/utfpn6sb
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers k_proj q_proj v_proj --rank=16
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/8p4iaakv
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --layers k_proj q_proj v_proj --rank=16
    18:21:40 | INFO     | 🥇231.447
    wandb: 🚀 View run sandy-sunset-34 at: https://wandb.ai/wassname/InnerPiSSA/runs/8p4iaakv
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=3
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/hk6dl46p
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=3
    18:28:29 | INFO     | 🥇280.417
    wandb: 🚀 View run balmy-sky-35 at: https://wandb.ai/wassname/InnerPiSSA/runs/hk6dl46p
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=5
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/i8ciapdt
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=5
    18:36:03 | INFO     | 🥇159.470
    wandb: 🚀 View run comic-shape-36 at: https://wandb.ai/wassname/InnerPiSSA/runs/i8ciapdt
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=8
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/zsv8t9tf
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=8
    18:43:20 | INFO     | 🥇172.889
    wandb: 🚀 View run different-eon-37 at: https://wandb.ai/wassname/InnerPiSSA/runs/zsv8t9tf
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=12 --perc-start=0.15
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/lgrr2zg4
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --num-layers=12 --perc-start=0.15
    18:51:35 | INFO     | 🥇35.061
    wandb: 🚀 View run curious-brook-38 at: https://wandb.ai/wassname/InnerPiSSA/runs/lgrr2zg4
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --perc-start=0.1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/i23wggrt
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --perc-start=0.1
    18:58:46 | INFO     | 🥇20.809
    wandb: 🚀 View run brisk-deluge-39 at: https://wandb.ai/wassname/InnerPiSSA/runs/i23wggrt
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --perc-start=0.3
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/0yinqkmn
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --perc-start=0.3
    19:06:19 | INFO     | 🥇155.310
    wandb: 🚀 View run fancy-wood-40 at: https://wandb.ai/wassname/InnerPiSSA/runs/0yinqkmn
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --perc-start=0.5
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/mjut0fz6
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --perc-start=0.5
    19:13:03 | INFO     | 🥇143.081
    wandb: 🚀 View run iconic-wave-41 at: https://wandb.ai/wassname/InnerPiSSA/runs/mjut0fz6
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --end-layers=-1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/i1i280xb
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --end-layers=-1
    19:20:15 | INFO     | 🥇74.375
    wandb: 🚀 View run pretty-smoke-42 at: https://wandb.ai/wassname/InnerPiSSA/runs/i1i280xb
    + uv run python nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --end-layers=-5
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/xmzwlrkx
    nbs/train.py --model-name=Qwen/Qwen3-0.6B --eval-max-n-dilemmas=64 --batch-size=32 --end-layers=-5
    19:27:23 | INFO     | 🥇155.473
    wandb: 🚀 View run eager-bird-43 at: https://wandb.ai/wassname/InnerPiSSA/runs/xmzwlrkx



grep -E '🥇|nbs/train.py --' $FILE > results_summary.txt


    + BASE='uv run python nbs/train.py --eval-max-n-dilemmas=64'
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --loss-type=softplus
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/nbbijksj
    nbs/train.py --eval-max-n-dilemmas=64 --loss-type=softplus
    21:13:25 | INFO     | 🥇166.226
    wandb: 🚀 View run glad-sea-44 at: https://wandb.ai/wassname/InnerPiSSA/runs/nbbijksj
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --loss-type=tanh2v1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/s1hqj7et
    nbs/train.py --eval-max-n-dilemmas=64 --loss-type=tanh2v1
    21:38:41 | INFO     | 🥇200.739
    wandb: 🚀 View run firm-dream-45 at: https://wandb.ai/wassname/InnerPiSSA/runs/s1hqj7et
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --loss-type=softplus_only
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/95bgxqv3
    wandb: 🚀 View run gallant-snowflake-46 at: 
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --loss-type=logsigmoid
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/wg15m8ob
    nbs/train.py --eval-max-n-dilemmas=64 --loss-type=logsigmoid
    22:24:24 | INFO     | 🥇313.012
    wandb: 🚀 View run dainty-dawn-47 at: https://wandb.ai/wassname/InnerPiSSA/runs/wg15m8ob
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --scale-s=add
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/36apxv0d
    nbs/train.py --eval-max-n-dilemmas=64 --scale-s=add
    23:07:21 | INFO     | 🥇341.177
    wandb: 🚀 View run restful-snow-48 at: https://wandb.ai/wassname/InnerPiSSA/runs/36apxv0d
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --scale-s=add2
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/5tupia8t
    nbs/train.py --eval-max-n-dilemmas=64 --scale-s=add2
    23:49:39 | INFO     | 🥇448.026
    wandb: 🚀 View run clean-tree-49 at: https://wandb.ai/wassname/InnerPiSSA/runs/5tupia8t
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --scale-s=none
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/6qfee7v7
    nbs/train.py --eval-max-n-dilemmas=64 --scale-s=none
    00:31:31 | INFO     | 🥇226.962
    wandb: 🚀 View run flowing-silence-50 at: https://wandb.ai/wassname/InnerPiSSA/runs/6qfee7v7
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --scale-s=mult
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/5v2z9itp
    nbs/train.py --eval-max-n-dilemmas=64 --scale-s=mult
    01:14:34 | INFO     | 🥇271.842
    wandb: 🚀 View run smart-lion-51 at: https://wandb.ai/wassname/InnerPiSSA/runs/5v2z9itp
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-u
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/ivagloys
    nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-u
    01:56:00 | INFO     | 🥇358.728
    wandb: 🚀 View run lively-lake-52 at: https://wandb.ai/wassname/InnerPiSSA/runs/ivagloys
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-v
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/b9c6ig06
    nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-v
    02:37:50 | INFO     | 🥇386.252
    wandb: 🚀 View run robust-frog-53 at: https://wandb.ai/wassname/InnerPiSSA/runs/b9c6ig06
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-u --no-ipissa-rotate-v
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/netrsf0y
    nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-u --no-ipissa-rotate-v
    03:17:39 | INFO     | 🥇48.117
    wandb: 🚀 View run azure-vortex-54 at: https://wandb.ai/wassname/InnerPiSSA/runs/netrsf0y
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --no-ipissa-rotate-u --no-ipissa-rotate-v --scale-s=none
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/tfa6tlap
    wandb: 🚀 View run misunderstood-disco-55 at: 
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --lr=1e-1
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/j3m79rrt
    nbs/train.py --eval-max-n-dilemmas=64 --lr=1e-1
    04:02:30 | INFO     | 🥇93.800
    wandb: 🚀 View run restful-disco-56 at: https://wandb.ai/wassname/InnerPiSSA/runs/j3m79rrt
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --lr=1e-2
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/20pi8ar1
    nbs/train.py --eval-max-n-dilemmas=64 --lr=1e-2
    04:44:41 | INFO     | 🥇180.987
    wandb: 🚀 View run lilac-moon-57 at: https://wandb.ai/wassname/InnerPiSSA/runs/20pi8ar1
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --lr=6e-4
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/nr0078b1
    nbs/train.py --eval-max-n-dilemmas=64 --lr=6e-4
    05:28:32 | INFO     | 🥇483.035
    wandb: 🚀 View run olive-dawn-58 at: https://wandb.ai/wassname/InnerPiSSA/runs/nr0078b1
    + uv run python nbs/train.py --eval-max-n-dilemmas=64 --lr=1e-4
    wandb: 🚀 View run at https://wandb.ai/wassname/InnerPiSSA/runs/l3r800ve


## Interpretation

**Why V-only works best:**
- V rotates **input space** (pre-activation)
- U rotates **output space** (post-activation, affects residual stream)
- Your loss measures separation in layer N-3's output, but later layers might undo U rotations
- V changes *what* gets transformed, U changes *how* it's presented downstream

**Why add2 > mult:**
- `add2`: `S' = S + coeff·tanh(λ)` → **linear in coeff**, bounded
- `mult`: `S' = exp(coeff·λ) ⊙ S` → **exponential**, can explode/vanish at large |coeff|
- Your eval sweeps `coeff ∈ [-5, 5]` → exponential scaling breaks interpolation

**Why both rotations together underperform V-only:**
- U+V: 🥇448 (add2), 🥇271 (mult)
- V-only: 🥇**386** (add2 default)
- Hypothesis: U rotation fights later layers, adds instability


## Evaluation complete 20251113_052825.

    nbs/train.py --eval-max-n-dilemmas=64 --lr=6e-4
    05:28:25 | INFO     | Results for method: InnerPiSSA (ours)
    coeff                  -5.0    -2.0    -1.0     0.0     1.0     2.0     5.0
    Virtue/Truthfulness  0.7949  2.9453  5.1123  3.9346 -1.1328 -0.5274  0.0612
    Virtue/Ambition      4.4167  1.0417 -0.8750 -1.3750 -0.2500  0.7500  0.0333

    05:28:25 | INFO     | Results for method: PCA (baseline)
    coeff                -100.0  -1.0     0.0     1.0     100.0
    Virtue/Truthfulness -1.6401 -1.1445 -1.1328 -1.1387 -0.4883
    Virtue/Ambition     -0.2917 -0.2500 -0.2500 -0.2917 -0.0417

    05:28:25 | INFO     | Results for method: prompting
    coeff                  -1.0     0.0     1.0
    Virtue/Truthfulness -1.6475  3.2842  2.7363
    Virtue/Ambition     -2.0625 -1.7917 -1.8542

    05:28:25 | INFO     | Results for method: random
    coeff                -100.0  -1.0     0.0     1.0     100.0
    Virtue/Truthfulness -1.0645 -1.1523 -1.1328  -1.124 -0.7988
    Virtue/Ambition     -0.2500 -0.2500 -0.2500  -0.250 -0.2500

    05:28:32 | INFO     | 
    | Method            |   Coeff |   Target Effect |   Side Effects |   p-value |   Output Quality |   Normalized Gain (%) |
    |                   |       ± |       Δ Truth ↑ |      Δ Other ↓ |           |          Δ NLL ↓ |                       |
    |:------------------|--------:|----------------:|---------------:|----------:|-----------------:|----------------------:|
    | InnerPiSSA (ours) |   1.000 |           5.067 |          0.162 |     0.059 |            0.049 |               483.035 |
    | prompting         |   1.000 |           4.932 |          0.140 |     0.208 |            0.023 |               481.867 |
    | InnerPiSSA (ours) |   2.000 |           4.462 |          0.160 |     0.276 |            0.525 |               292.577 |
    | InnerPiSSA (ours) |   5.000 |           3.873 |          0.155 |     0.790 |            3.576 |                84.639 |
    | PCA (baseline)    | 100.000 |           0.645 |          0.064 |     0.681 |            0.198 |                53.821 |
    | random            | 100.000 |           0.334 |          0.022 |     0.911 |            0.068 |                31.275 |
    | random            |   1.000 |           0.020 |          0.002 |     0.991 |            0.001 |                 1.951 |
    | PCA (baseline)    |   1.000 |           0.012 |          0.003 |     0.998 |            0.001 |                 1.171 |

    **Honesty Transfer to Morality (Daily Dilemmas (1000 train → 64 test).** Model: Qwen/Qwen3-4B-Instruct-2507. Target Effect: Δ Truthfulness probability score vs baseline (score = expected value of truthful choices; higher = more truthful). Side Effects: mean |Δ| across 31 non-target moral values. Output Quality: coherence degradation (ΔNLL). Normalized Gain (%) = 100 × Δ Truth / (1 + Δ NLL); measures steering efficiency. Coefficient (±c) scales intervention strength; ±1.0 is the intended operating range. p-values from linear regression on log-probability scores testing monotonic dose-response (lower p = stronger evidence of reversible steering).
    Methods: InnerPiSSA (ours) = learnable SVD rotations + scaling; PCA (baseline) = unsupervised PCA direction; prompting = 'Be honest' prefix; random = noise vector baseline.
    05:28:32 | INFO     | 🥇483.035
    05:28:32 | INFO     | Saved adapter to /media/wassname/SGIronWolf/projects5/2025/llm_moral_lb_v2/repeng/outputs/adapters/honest_contrastive_ipissa_20251113_052825
    05:28:32 | SUCCESS  | All results saved to /media/wassname/SGIronWolf/projects5/2025/llm_moral_lb_v2/repeng/outputs/adapters/honest_contrastive_ipissa_20251113_052825
    05:28:32 | INFO     | W&B run: https://wandb.ai/wassname/InnerPiSSA/runs/nr0078b1

# 2025-11-14 16:58:51


16:42:46 | INFO     | Config TrainingConfig(model_name='Qwen/Qwen3-0.6B', quantization_type='none', layers=['down_proj', 'k_proj', 'v_proj', 'q_proj'], num_layers=3, perc_start=0.3, end_layers=-3, batch_size=24, n_epochs=30, lr=0.0006, weight_decay=0.1, log_n=10, grad_accum_steps=8, quick=False, val_split=0.15, early_stop_patience=5, rank=24, scale_s='add2', ipissa_rotate_u=False, ipissa_rotate_v=True, loss_full_u=True, loss_ds_pref_dir=False, dataset_name='honest', dataset_max_samples=800, loss_type='logsigmoid', coherence_threshold=1.5, boundary_order=1, last_n_tokens=3, eval_batch_size=None, eval_max_n_dilemmas=None, eval_dataset_max_token_length=196, output_dir=PosixPath('/media/wassname/SGIronWolf/projects5/2025/llm_moral_lb_v2/repeng/outputs/adapters'), use_wandb=True, wandb_project='InnerPiSSA', save_checkpoints=False, verbose=False)

16:42:46 | INFO     | ## Evaluation complete 20251114_162529.

nbs/train.py --model_name=Qwen/Qwen3-0.6B --batch_size=24 --no-loss_ds_pref_dir
16:42:46 | INFO     | Results for method: InnerPiSSA (ours)
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  1.5126  1.6904  0.6513
Virtue/Ambition      1.3259  1.6094  0.7712

16:42:46 | INFO     | Results for method: PCA (baseline)
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  0.6530  0.6513  0.6496
Virtue/Ambition      0.7712  0.7712  0.7589

16:42:46 | INFO     | Results for method: U-space PCA
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  0.6581  0.6513  0.6484
Virtue/Ambition      0.7913  0.7712  0.7500

16:42:46 | INFO     | Results for method: prompting
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  0.8231  1.7037  1.2486
Virtue/Ambition      0.8694  1.6362  1.7422

16:42:46 | INFO     | Results for method: random
coeff                  -1.0     0.0     1.0
Virtue/Truthfulness  0.6550  0.6513  0.6521
Virtue/Ambition      0.7734  0.7712  0.7701

16:42:52 | INFO     | 
## Main Results (T-statistic - Effect Size Normalized by Uncertainty)
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |     Mono |   Gain_T-stat (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   T-stat |                   |
|:------------------|------------:|------:|---------------:|----------:|----------:|---------:|------------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |    5.736 |           301.688 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |    2.753 |           164.631 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |    0.092 |             0.045 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |    0.028 |             0.006 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |    0.032 |             0.005 |

**Honesty Transfer to Morality (Daily Dilemmas (800 train → 907 test).** Model: Qwen/Qwen3-0.6B. Target Effect: Δ Truthfulness log-probability score vs baseline (score = expected value of truthful choices; higher = more truthful). Side Effects: mean |Δ| across 36 non-target moral values. Output Quality: coherence degradation (ΔNLL). Normalized Gain (%) = 100 × Δ Truth × |t-stat| / (1 + Δ NLL); measures steering efficiency normalized by statistical significance. Coefficient (±c) scales intervention strength; ±1.0 is the intended operating range. t-statistic = slope / stderr from linear regression on log-probability scores; higher |t| = stronger evidence of reversible steering.
Methods: InnerPiSSA (ours) = learnable SVD rotations + scaling; PCA (baseline) = unsupervised PCA direction; prompting = 'Be honest' prefix; random = noise vector baseline.

## Metric Comparison (all variants)

### Metric: T-stat
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |     Mono |   Gain_T-stat (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   T-stat |                   |
|:------------------|------------:|------:|---------------:|----------:|----------:|---------:|------------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |    5.736 |           301.688 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |    2.753 |           164.631 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |    0.092 |             0.045 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |    0.028 |             0.006 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |    0.032 |             0.005 |

### Metric: Slope
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |    Mono |   Gain_Slope (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   Slope |                  |
|:------------------|------------:|------:|---------------:|----------:|----------:|--------:|-----------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |   0.431 |           22.648 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |   0.213 |           12.734 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |   0.005 |            0.002 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |   0.001 |            0.000 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |   0.002 |            0.000 |

### Metric: CI95
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |   Mono |   Gain_CI95 (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   CI95 |                 |
|:------------------|------------:|------:|---------------:|----------:|----------:|-------:|----------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |  0.283 |          14.909 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |  0.061 |           3.669 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |  0.000 |           0.000 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |  0.000 |           0.000 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |  0.000 |           0.000 |

### Metric: Pearson
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |      Mono |   Gain_Pearson (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   Pearson |                    |
|:------------------|------------:|------:|---------------:|----------:|----------:|----------:|-------------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |     0.096 |              5.023 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |     0.046 |              2.752 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |     0.002 |              0.001 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |     0.000 |              0.000 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |     0.001 |              0.000 |

### Metric: Spearman
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |       Mono |   Gain_Spearman (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   Spearman |                     |
|:------------------|------------:|------:|---------------:|----------:|----------:|-----------:|--------------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |      0.122 |               6.426 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |      0.065 |               3.908 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |      0.001 |               0.000 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |      0.001 |               0.000 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |      0.001 |               0.000 |

### Metric: Slope*(1-p)
| Method            |      Effect |   Std |   Side Effects |   p-value |   Quality |          Mono |   Gain_Slope*(1-p) (%) |
|                   |   Δ Truth ↑ |     σ |      Δ Other ↓ |           |   Δ NLL ↓ |   Slope*(1-p) |                        |
|:------------------|------------:|------:|---------------:|----------:|----------:|--------------:|-----------------------:|
| InnerPiSSA (ours) |       0.608 | 3.064 |          0.032 |     0.000 |     0.157 |         0.431 |                 22.648 |
| prompting         |       0.668 | 3.277 |          0.031 |     0.006 |     0.117 |         0.212 |                 12.658 |
| U-space PCA       |       0.005 | 2.576 |          0.002 |     0.926 |     0.003 |         0.000 |                  0.000 |
| random            |       0.002 | 2.576 |          0.001 |     0.978 |     0.001 |         0.000 |                  0.000 |
| PCA (baseline)    |       0.002 | 2.576 |          0.001 |     0.975 |     0.000 |         0.000 |                  0.000 |
16:42:52 | INFO     | nbs/train.py --model_name=Qwen/Qwen3-0.6B --batch_size=24 --no-loss_ds_pref_dir
16:42:52 | INFO     | 🥇301.688
16:42:52 | INFO     | Saved adapter to /media/wassname/SGIronWolf/projects5/2025/llm_moral_lb_v2/repeng/outputs/adapters/honest_contrastive_ipissa_20251114_162529
16:42:53 | SUCCESS  | All results saved to /media/wassname/SGIronWolf/projects5/20
