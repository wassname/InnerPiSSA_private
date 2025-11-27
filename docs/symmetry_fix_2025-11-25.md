# Output Symmetry Fix for InnerPiSSA

**Date:** 2025-11-25  
**Branch:** try_residual_stream

## Problem

InnerPiSSA's bidirectional steering (`α=±1`) has asymmetric outputs because rotating `S` inside `V@R(α)@(S±delta_s)@U.T` breaks symmetry:

```python
# Want: Δy(+1) ≈ -Δy(-1) around base model
# Have: R(+1)@S ≠ -R(-1)@S (S is diagonal, R is not)
```

**Why not separate base from steering?** Moving `V@S@U.T` to unrotated base = putting it in frozen `W_res`, loses expressivity.

## Solution: Soft-Clamp Rotation Angles to θ ≤ 0.3 rad

At small angles, Taylor expansion gives `R(θ)@S ≈ S + θ*A@S`, so `R(+θ)@S ≈ -R(-θ)@S` (first-order). Error is O(θ²) ≈ 9% at θ=0.3 rad (17°) vs 100%+ unconstrained.

```python
def _rotation_from_skew(self, A, alpha, max_angle=0.3):
    A_clamped = max_angle * torch.tanh(A / max_angle)  # soft-bound angle
    return torch.matrix_exp(alpha * A_clamped)
```

**Why 0.3 rad?** 17° is substantial for weight matrices; gives ~91% symmetry while preserving expressivity.

## Reasoning

**Symmetry improvement:** Small rotation constraint ensures first-order approximation `[R(θ)-I]@S ≈ -[R(-θ)-I]@S` holds with ~9% error instead of arbitrary asymmetry.

**Stability hypothesis:** Previous `rot_u=True` instability likely caused by unbounded rotations. Bounded angles may stabilize dual rotation learning (both U and V).

**Interaction with S scaling modes:**

Additive (`scale_s="add2"`): `S + α*delta_s`
```python
# α=+1: y₊ = x@V@R(+1)@(S+delta_s)@U.T
# α=-1: y₋ = x@V@R(-1)@(S-delta_s)@U.T = x@V@R(+1).T@(S-delta_s)@U.T
# α=0:  y₀ = x@V@S@U.T

# Deviations around base:
Δy₊ = x@V@[R(+1)@(S+delta_s) - S]@U.T = x@V@[R@S - S + R@delta_s]@U.T
Δy₋ = x@V@[R(+1).T@(S-delta_s) - S]@U.T = x@V@[R.T@S - S - R.T@delta_s]@U.T
# With R ≈ I+θA: Δy₊ ≈ x@V@[θA@S + delta_s]@U.T, Δy₋ ≈ x@V@[-θA@S - delta_s]@U.T
# Symmetric in additive deviations from S
```

Multiplicative (`scale_s="mult"`): `S * exp(α*log_λ)`
```python
# α=+1: y₊ = x@V@R(+1)@(S*exp(log_λ))@U.T
# α=-1: y₋ = x@V@R(-1)@(S/exp(log_λ))@U.T = x@V@R(+1).T@(S/exp(log_λ))@U.T
# α=0:  y₀ = x@V@S@U.T

# Deviations around base:
Δy₊ = x@V@[R(+1)@S*exp(log_λ) - S]@U.T = x@V@[R@S*(exp(log_λ)-1) + R@S - S]@U.T
Δy₋ = x@V@[R(+1).T@S/exp(log_λ) - S]@U.T = x@V@[R.T@S*(1-1/exp(log_λ)) + R.T@S - S]@U.T
# With R ≈ I+θA: Δy₊ ≈ x@V@[S*(exp(log_λ)-1) + θA@S*exp(log_λ)]@U.T
#                Δy₋ ≈ x@V@[S*(1/exp(log_λ)-1) - θA@S/exp(log_λ)]@U.T
# Symmetric in ratio-space deviations from S (exp vs 1/exp)
```

Both modes benefit from rotation constraint, but multiplicative scaling is naturally more symmetric because `exp(+x)` and `exp(-x) = 1/exp(x)` are symmetric around 1.0 in ratio space, while additive `±delta_s` requires perfect rotation symmetry `R ≈ -R.T`.

**Interaction with data-aware init:** Orthogonal improvements. `S_MEAN_ABS=True` selects task-relevant components; rotation constraint ensures symmetric steering of those components.

## Predicted Observations

1. **Better symmetry metrics:** `|Δy(+1) + Δy(-1)|` should decrease ~10x
2. **`rot_u=True` now stable:** Training won't diverge with dual rotation
3. **Multiplicative scaling winner:** `scale_s="mult"` may outperform additive due to natural ratio-space symmetry
4. **Higher LR viable:** Bounded rotations = more stable, gradients may need boosting to reach useful rotations

## Test Priority

1. `S_MEAN_ABS=True --scale_s=mult --max_rotation_angle=0.3` (highest expected gain)
2. `--rot_u --max_rotation_angle=0.3 --lr=1e-2` (test stability + gradient strength)
3. Sweep `max_angle ∈ [0.1, 0.3, 0.5, inf]` to validate 0.3 choice