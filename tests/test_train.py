from ipissa.train.train_adapter import train_model
from ipissa.config import TrainingConfig
import pytest

def mk_test_cng(**kwargs):
    config = TrainingConfig(
        model_name="snake7gun/tiny-random-qwen3",
        quick=True,
        
        n_depths=1,
        n_epochs=1,
        bs=4,
        effective_bs=8,
        max_samples=20,
        eval_max_dilemmas=10,
        loss_depths=[-1],
        **kwargs,
    )
    return config

test_configs = [
    mk_test_cng(),
    mk_test_cng(loss_type="tanh_sym", coh_adaptive=False, scale_s="mult"),
    # Test new pref_dir methods
    mk_test_cng(pref_dir_method="top_diff"),
    mk_test_cng(pref_dir_method="top_diff_snorm"),
    mk_test_cng(pref_dir_method="adapter_dims_raw"),
]

@pytest.mark.parametrize("config", test_configs)
def test_train(config):
    train_model(config)