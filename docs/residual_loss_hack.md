# Residual Stream Loss with V Projection - IMPLEMENTED ✅

## Usage

To use residual stream loss with V projection, set these config options:

```python
config = TrainingConfig(
    modules=["o_proj", "down_proj"],  # Intervention layers (what to modify)
    loss_modules=["up_proj"],          # Loss extraction from up_proj
    loss_use_V=True,                   # Use V (input space) instead of U (output space)
    loss_depths=[0.3, 0.4, 0.5],      # Extract from these layer depths
)
```

## What This Does

1. **Intervention**: Adapters modify `o_proj` and `down_proj` (writes to residual)
2. **Loss extraction**: Hidden states extracted from `hidden_states[layer_idx]` (accumulated residual stream)
3. **Projection**: Residual projected via `up_proj.V` → MLP's input S-space

## Why This Works Geometrically

```python
# up_proj forward: hs_residual @ W_up.T = hs_residual @ V @ S @ U.T
# So V projects: residual space → MLP's S-space
# This is the "what patterns in residual does MLP see?" basis
```

## Implementation Details

### Changes Made

1. **Config** (`ipissa/config.py`):
   - Added `loss_modules: Optional[List[str]]` - modules for loss extraction
   - Added `loss_use_V: bool` - switch between V (input) and U (output) projection

2. **Layer Selection** (`ipissa/peft_utils/layer_selection.py`):
   - `compute_layer_selection()` now accepts `loss_modules` parameter
   - Loss layers can use different modules than intervention layers

3. **Loss Computation** (`ipissa/train/train_adapter.py`):
   - `compute_batch_loss()` now uses `hidden_states` from transformer output instead of TraceDict
   - Supports both U and V projection via `config.loss_use_V`
   - Passes `loss_layer_indices` to extract from correct residual depth

### Benefits Over Previous Approach

1. **Richer signal**: Accumulated residual stream (all previous layers) vs single module output
2. **Pre-suppression**: Extracts from middle layers before suppression neurons activate
3. **Semantically meaningful basis**: V from up_proj = "what MLP sees in residual"
4. **No TraceDict overhead**: Uses built-in `hidden_states` output

## Testing

Compare loss signal quality:

**Baseline** (current approach):
```python
config = TrainingConfig(
    modules=["down_proj"],
    loss_modules=None,  # Uses same as modules
    loss_use_V=False,   # Uses U projection
)
```

**New approach** (residual + V):
```python
config = TrainingConfig(
    modules=["down_proj"],      # Intervene on down_proj
    loss_modules=["up_proj"],   # Extract loss from up_proj  
    loss_use_V=True,            # Project via V
    loss_depths=[-4],           # Layer N-4 residual
)
```

## Concept
Extract loss from accumulated residual stream at layer N-4, projected via that layer's `up_proj.V`.

## Why this works geometrically
```python
# up_proj forward: hs_residual @ W_up.T = hs_residual @ V @ S @ U.T
# So V projects: residual space → MLP's S-space
# This is the "what patterns in residual does MLP see?" basis
```

## Implementation Hack (for testing)

### Option 1: Use transformer hidden_states output
Transformers library provides `output_hidden_states=True` which gives you the residual after each layer.

```python
# In train_adapter.py, around line 467
outputs_ref = model(**batch, output_hidden_states=True)

# outputs_ref.hidden_states is a tuple of [batch, seq, d_model] tensors
# hidden_states[i] = residual stream after layer i
# So hidden_states[-4] = residual at layer N-4
```

### Option 2: Extract via TraceDict from post-layer

For Qwen/Llama architecture, the residual is updated after each block:
```python
# Extract from layer's INPUT (before self-attn)
loss_layers = ["model.layers.X.input_layernorm"]  # Pre-block residual
# OR extract from hidden_states directly (cleaner)
```

## Minimal Code Change

In `compute_batch_loss()`:

```python
# After getting outputs_ref and outputs_pi with hidden_states=True
# Extract residual at specific layer depth
residual_layer_idx = -4  # or computed from config.loss_depths

hs_ref_residual = outputs_ref.hidden_states[residual_layer_idx].float()
hs_pi_residual = outputs_pi.hidden_states[residual_layer_idx].float()

# Get V from up_proj at that layer
# Find the up_proj layer name for residual_layer_idx
up_proj_layer = f"model.layers.{residual_layer_idx}.mlp.up_proj"
V_up = Vw_full[up_proj_layer].to(model.device)  # [d_model, r]

# Project residual into MLP's S-space
hs_ref_residual_s = hs_ref_residual @ V_up  # [b, t, r]
hs_pi_residual_s = hs_pi_residual @ V_up
```

Then use these projected activations in the loss computation.

## Full Implementation (needs refactor)

Would require:
1. New config: `loss_source: Literal["module_output", "residual_stream"]`
2. New config: `loss_projection: Literal["U_same", "V_up_proj", "none"]`  
3. Modify `extract_U_matrices` to also extract V matrices
4. Modify `compute_batch_loss` to handle residual extraction + custom projection

## Testing Strategy

1. **First**: Verify that `outputs.hidden_states[-4]` gives you what you expect
2. **Second**: Extract `up_proj.V` and manually compute projection
3. **Third**: Compare loss signal quality vs current approach
4. **If it works better**: Refactor properly with config options
