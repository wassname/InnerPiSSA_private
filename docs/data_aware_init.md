# Data-Aware Adapter Initialization

## Summary

InnerPiSSA now supports **data-aware SVD component selection** during adapter initialization. Instead of naively selecting the top-r singular vectors by magnitude, we select them by relevance to the preference direction extracted from your training data.

## Motivation

**Problem**: Standard PiSSA initialization selects SVD components by singular value magnitude only:
```python
U, S, Vh = svd(W)
U_init = U[:, :r]  # Top-r by |S|
```

This ignores your actual task. Large singular values represent high-variance directions in the weight matrix, but they may not align with the directions you want to learn (e.g., honest vs dishonest).

**Solution**: Select components by projection onto preference direction `dHS`:
```python
dHS = mean(hs_cho - hs_rej)  # From first batch
proj = dHS @ U_full          # Importance of each singular vector
indices = topk(|proj|, r)    # Select by relevance, not magnitude
U_init = U_full[:, indices]
```

## How It Works

### 1. Extract Preference Direction (Before Adapter Setup)

```python
def compute_init_steering_vectors(model, dataset, loss_layers, tokenizer, config):
    """Compute raw dHS from first batch for data-aware adapter init."""
    
    # Run forward pass on first 32 samples (cho/rej pairs)
    with torch.no_grad():
        for batch in dataloader:
            with TraceDict(model, layers=loss_layers) as ret:
                model(**batch)
            
            # Extract last token activations
            for layer in loss_layers:
                hs = ret[layer].output
                hs_cho = hs[::2].mean(dim=1)  # Chosen samples
                hs_rej = hs[1::2].mean(dim=1)  # Rejected samples
                dHS = (hs_cho - hs_rej).mean(dim=0)  # [d] preference direction
    
    return {layer: dHS for layer in loss_layers}
```

### 2. Use Preference Direction for SVD Selection

```python
def update_layer(self, adapter_name, r, steering_vectors, layer_name, ...):
    """Initialize adapter layer with data-aware component selection."""
    
    # Full SVD of weight matrix
    U_full, S_full, Vh_full = torch.linalg.svd(W)
    
    if steering_vectors is not None and layer_name in steering_vectors:
        dHS = steering_vectors[layer_name]
        
        # Project preference direction onto singular vectors
        proj = dHS @ U_full  # [full_rank]
        
        # Select top-r by |projection| instead of |S|
        _, indices = torch.topk(proj.abs(), r)
        indices_sorted = indices.sort()[0]  # Keep S order for stability
        
        U = U_full[:, indices_sorted]
        S = S_full[indices_sorted]
        Vh = Vh_full[indices_sorted, :]
    else:
        # Fallback: naive top-r by singular values
        U = U_full[:, :r]
        S = S_full[:r]
        Vh = Vh_full[:r, :]
```

## Usage

Enable in your training config:

```python
config = TrainingConfig(
    r=256,
    data_aware_init=True,  # NEW: Enable data-aware initialization
    ...
)
```

The implementation:
1. Creates dataset before adapter setup (needed for preference extraction)
2. Runs single forward pass on first batch to get `dHS = mean(hs_cho - hs_rej)`
3. Passes `{layer_name: dHS}` dict through `InnerPiSSAConfig.steering_vectors`
4. During `update_layer()`, selects components by `|dHS @ U|` instead of `|S|`

## Expected Benefits

- **Better initialization**: Adapter starts in directions that matter for your task
- **Faster convergence**: Less time rotating from irrelevant to relevant directions
- **Lower rank**: May achieve same performance with smaller `r` (fewer parameters)
- **Robust to low-variance features**: Captures task-relevant low-variance directions that naive PiSSA would miss

## Implementation Details

### Files Modified

1. **`ipissa/config.py`**: Added `data_aware_init: bool = True` flag
2. **`ipissa/peft_utils/innerpissa.py`**:
   - `InnerPiSSAConfig`: Added `steering_vectors: Optional[Dict[str, Tensor]]` field
   - `update_layer()`: Added data-aware component selection logic
   - `_create_and_replace()`: Pass `layer_name` for steering vector lookup
3. **`ipissa/train/train_adapter.py`**:
   - `compute_init_steering_vectors()`: New function to extract `dHS` from first batch
   - `train_model()`: Compute init steering before adapter setup, pass through config
   - `setup_adapter()`: Accept and pass `init_steering_vectors` to config

### Comparison to Standard PiSSA

| Method | Selection Criterion | Pros | Cons |
|--------|-------------------|------|------|
| **Naive PiSSA** | Top-r by `\|S\|` | Fast, model-agnostic | Ignores task, may miss low-variance task-relevant directions |
| **Data-Aware** | Top-r by `\|dHS @ U\|` | Task-aware, captures relevant low-variance directions | Requires data, single extra forward pass |

### Cost

- **Computational**: One extra forward pass on 32 samples before training (~0.1% overhead)
- **Memory**: Stores `{layer_name: dHS_tensor}` dict temporarily (negligible)

## Example

```python
from ipissa.config import TrainingConfig
from ipissa.train.train_adapter import train_model

config = TrainingConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    r=256,
    data_aware_init=True,  # Enable data-aware init
    n_epochs=20,
    ...
)

model, save_folder = train_model(config)
# Adapter initialized with components selected by relevance to preference direction
```

## References

- Standard PiSSA: https://arxiv.org/abs/2404.02948
- Similar idea in LoRA initialization: "Task-aware low-rank adaptation" (select by gradient importance)
