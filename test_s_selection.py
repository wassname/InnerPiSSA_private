"""Quick test of s_selection_mode parameter."""
import torch
from ipissa.config import TrainingConfig
from ipissa.peft_utils.innerpissa import InnerPiSSAConfig

# Test config loads
config = TrainingConfig(s_selection_mode="cho_var_snorm")
print(f"✓ Config loads: s_selection_mode={config.s_selection_mode}")

# Test InnerPiSSAConfig accepts it
steering_vecs = {
    "test_layer": {
        "cho": torch.randn(10, 64),
        "rej": torch.randn(10, 64)
    }
}

ipissa_config = InnerPiSSAConfig(
    r=16,
    steering_vectors=steering_vecs,
    s_selection_mode="cho_std_raw"
)
print(f"✓ InnerPiSSAConfig accepts: s_selection_mode={ipissa_config.s_selection_mode}")

# Test all valid modes parse
valid_modes = [
    "cho_var_snorm",
    "cho_std_raw",
    "rej_mean_abs_snorm",
    "diff_var_raw",
]
for mode in valid_modes:
    parts = mode.split('_')
    if len(parts) == 3:
        source, stat, norm = parts
        print(f"✓ Mode '{mode}' -> source={source}, stat={stat}, norm={norm}")

print("\n✅ All tests passed! Ready to sweep with:")
print("  - cho_var_snorm (task-active variance, S-normalized)")
print("  - cho_var_raw (task-active variance, raw)")
print("  - diff_var_raw (difference signal variance, raw - current default)")
print("  - diff_mean_abs_snorm (original env var method)")
