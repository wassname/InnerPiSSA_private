# Output Symmetry Fix for InnerPiSSA

**Date:** 2025-11-25  
**Branch:** try_residual_stream

## Problem

InnerPiSSA needs **bidirectional steering** with symmetric outputs:
- `α = +1`: steer toward honest
- `α = -1`: steer toward dishonest  
- `α = 0`: base model (no intervention)

**Requirements:**
1. **Reversibility**: `Δy(α=+1) ≈ -Δy(α=-1)` (output deviations are opposites)
2. **Expressivity**: Can't just freeze everything - need to learn rotations AND singular value modifications
3. **Additive effect**: Intervention adds/subtracts from base model output

## Current Implementation (Asymmetric)

```python
# Current forward pass:
y = x @ V@R(α) @ (S + α*delta_s) @ U.T + x@W_res.T

# where R(α) = exp(α * A), A is skew-symmetric
```

**Problem:** The baseline `S` gets rotated, breaking **output symmetry** (not symmetry around S=0, but around the base model output at α=0):

```python
# α=+1: y₊ = x@V@R(+1)@(S+delta_s)@U.T + x@W_res.T
# α=-1: y₋ = x@V@R(-1)@(S-delta_s)@U.T + x@W_res.T
# α=0:  y₀ = x@V@S@U.T + x@W_res.T  (base model)

# We want: Δy₊ = y₊ - y₀ ≈ -(y₋ - y₀) = -Δy₋
```

Using **distributive property** of matrix multiplication:
```python
y₊ = x@V@R(+1)@S@U.T + x@V@R(+1)@delta_s@U.T + x@W_res.T
y₋ = x@V@R(-1)@S@U.T - x@V@R(-1)@delta_s@U.T + x@W_res.T

# Since R(-1) = R(+1).T (orthogonal property):
y₋ = x@V@R(+1).T@S@U.T - x@V@R(+1).T@delta_s@U.T + x@W_res.T

# Deviations from base:
Δy₊ = x@V@[R(+1) - I]@S@U.T + x@V@R(+1)@delta_s@U.T
Δy₋ = x@V@[R(+1).T - I]@S@U.T - x@V@R(+1).T@delta_s@U.T

# These are NOT opposites because:
# [R(+1) - I]@S ≠ -[R(+1).T - I]@S
# The S baseline term breaks output symmetry
```

Even though rotations satisfy `R(-α) = R(α).T`, the interaction with `S` inside the rotation breaks **additive output symmetry** around the base model. We want symmetric steering effects that add/subtract equally from base outputs, not symmetric transformations in parameter space.

## Ideas Considered and Discarded

### Option 1: Separate base from steering
```python
y = x@V@S@U.T                    # base (unrotated, frozen)
  + x@V@R(α)@(α*delta_s)@U.T     # steering (rotated)
  + x@W_res.T
```

**Why discarded:** This is mathematically equivalent to moving `V@S@U.T` into `W_res`, creating a larger frozen residual. Loses expressivity - can't modify pretrained singular values, only add small perturbations on top.

### Option 2: Multiplicative S scaling
```python
y = x@V@R(α)@(S * exp(α*log_λ))@U.T + x@W_res.T
```

**Why discarded:** Rotating `S * exp(α*log_λ)` still breaks output symmetry for the same reason. The multiplicative symmetry (around `S`) doesn't translate to additive output symmetry.

### Option 3: Hard clip rotation angles
```python
rotation_params = rotation_params.clamp(-max_angle, max_angle)
```

**Why discarded:** Hard clipping creates gradient issues and isn't differentiable at boundaries. Soft constraint is better.

## Solution: Soft-Clamped Rotation Angles

**Key insight:** At small rotation angles (θ ≤ 0.3 rad ≈ 17°), the first-order Taylor approximation of the matrix exponential holds:
```python
R(θ) = exp(θ*A) ≈ I + θ*A + O(θ²)

# For small θ:
R(θ)@S ≈ (I + θ*A)@S = S + θ*A@S
R(-θ)@S ≈ (I - θ*A)@S = S - θ*A@S

# Therefore: [R(θ) - I]@S ≈ -[R(-θ) - I]@S
# Which gives: Δy(+1) ≈ -Δy(-1)  (approximate output symmetry)
```

Error is O(θ²) ≈ 9% for θ=0.3, **much better than unconstrained** (which could be 100%+ asymmetry for large rotations).

**What kind of symmetry?**
- **Additive output symmetry** around base model: `y₀ + Δy ≈ y₀ - Δy` at α=±1
- NOT symmetric around S=0 (parameters)
- NOT multiplicative/log-space symmetry (like exp scaling would give)
- We want steering effects that add/subtract from outputs, not scale them

**Implementation:**
```python
def _rotation_from_skew(self, A, alpha, rotation_method, max_angle=0.3):
    # Soft clamp to bounded angle using tanh
    A_clamped = max_angle * torch.tanh(A / max_angle)
    
    if rotation_method == "cayley":
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        X = alpha * A_clamped / 2
        return torch.linalg.solve(I - X, I + X)
    # ... etc
```

**Why this works:**
- `tanh` soft-bounds: `||A_clamped|| ≤ max_angle`
- Differentiable everywhere (smooth gradients)
- Still expressive: 17° rotation + delta_s learning + multiple layers compound
- Minimal code change: just modify `_rotation_from_skew`

**Symmetry improvement:**
```python
# Unclamped: arbitrary asymmetry (could be 100%+)
# Clamped (θ=0.3): ~9% residual asymmetry from O(θ²) terms
```

## Implementation Changes

**Files modified:**
1. `ipissa/peft_utils/innerpissa.py`:
   - Add `max_rotation_angle` to `InnerPiSSAConfig` (default 0.3)
   - Modify `_rotation_from_skew` to apply soft clamping
   - Pass `max_angle` through `_get_rotation`

2. `ipissa/config.py`:
   - Add `max_rotation_angle: float = 0.3` to `TrainingConfig`

**Backward compatibility:** Setting `max_rotation_angle=None` or `float('inf')` disables clamping.

## Expected Impact

**Positive:**
- Much better output symmetry (α=+1 vs α=-1)
- Could improve coherence loss effectiveness (less asymmetric degradation)
- Prevents pathological large rotations that might hurt stability

**Neutral:**
- Still have ~9% residual asymmetry (acceptable tradeoff)
- 17° is plenty of rotation for weight matrix interventions

**Risk:**
- Might limit expressivity if task requires larger rotations (unlikely - these are weight matrices, not geometric transforms)
- Can always increase `max_angle` if needed

## Next Steps

1. Implement changes
2. Test with `S_MEAN_ABS=True` baseline (current best)
3. Compare symmetry metrics on eval (check `delta_logp` at α=±1)
4. If improvement unclear, try sweeping `max_angle ∈ [0.1, 0.3, 0.5, inf]`
