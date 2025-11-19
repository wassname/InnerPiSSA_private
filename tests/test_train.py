from ipissa.train.train_adapter import main
from ipissa.config import TrainingConfig
import pytest

def mk_test_cng(**kwargs):
    config = TrainingConfig(
        model_name="snake7gun/tiny-random-qwen3",
        quick=True,
        
        num_layers=1,
        n_epochs=1,
        batch_size=4,
        effective_batch_size=8,
        dataset_max_samples=20,
        eval_max_n_dilemmas=10,
        loss_layers=[-1],
        **kwargs,
    )
    return config
test_configs = [
    mk_test_cng(),
    mk_test_cng(adapter_type="dora", loss_type="raw"),
    mk_test_cng(loss_type="tanh_sym", adaptive_coherence=False, monotonic_margin=None, scale_s="mult"),
]

@pytest.mark.parametrize("config", test_configs)
def test_train(config):
    main(config)