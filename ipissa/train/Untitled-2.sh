uv run python nbs/train.py tiny --quick --r=64 --rot_u --data_aware_init --wd=0 --no_coh --no_mono
# incoherenet 90 with high lr
uv run python nbs/train.py tiny --quick --r=64 --rot_u --data_aware_init --wd=0 --no_coh
# incoherent  with high lr
uv run python nbs/train.py tiny --quick --r=64 --rot_u --data_aware_init --wd=0 --no_mono
# incoherent 128 with high lr
uv run python nbs/train.py tiny --quick --r=64 --rot_u --data_aware_init --wd=0 --no_coh --no_mono 
# incoherent 10 with high lr

uv run python nbs/train.py tiny --quick --r=64 --rot_u --data_aware_init --wd=0 
# 132 a little incoherent

uv run python nbs/train.py tiny --quick --r=64  ---max_rotation_angle=inf --no_rot_u --scale_s=none

uv run python nbs/train.py tiny --quick --r=64  --no_loss_use_V --loss_depths=0.5 --loss_modules o_proj down_proj --no_coh --no_mono
uv run python nbs/train.py tiny --quick --r=64 --no_loss_use_V --loss_depths=0.5 --loss_modules o_proj down_proj