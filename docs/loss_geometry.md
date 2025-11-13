# Loss Geometry Visualization

**Figure: Geometric interpretation of contrastive steering loss in InnerPiSSA space.** The diagram visualizes training dynamics in the output singular vector basis (U-space) from SVD of layer weights. The coherence constraint from logp space is backprojected as a circular boundary (pizza crust), defining the valid operating region (InnerPiSSA space). Training rotates the per-sample adapter separation vector (orange/red slice) toward the frozen mean direction (blue) extracted from training data, while maintaining generation quality within the crust. The projection ratio loss encourages amplification along the frozen direction, while the coherence penalty prevents degradation beyond the threshold.

A note on the pizza puns, we don't want to be on the nose, and we don't want to compromise with geometrical or mathmatical intuition. We want it to acceptable in a paper. But we also want people to be like "oh the pizza diagram". So
- 45 degree chicago deep dish slice, but don't mention that
- PiSSA space (it sounds like pizza)
- InnerPiSSA space (it sounds like inner pizza)
- The coherence boundary is brown and dashed like a pizza crust
- We might even color the inner space from cheesy yellow on the inside to darker brown towards the crust. We can use yellow and red and brown colors (but they have to be visually distinct enough to see the boundary, so if we shade brown we can use yellow and red for the vectors etc)

## Geometric Components
****
### Space Definition
**Pizza Space** refers to the full U-space from SVD(W) = U·S·V^T of layer weights at depth N-3. **InnerPiSSA space** is the valid region within the coherence boundary where `log p_π(y|x) ≥ log p_ref(y|x) - threshold`. This constraint, originally in probability space, backprojects to a circular boundary in U-space (the "pizza crust").

### Separation Vectors (Pizza Slices)
- **`pref_dir`** (blue, frozen): Target direction extracted once from training data as `normalize(mean(hs_ref_cho - hs_ref_rej))` in U-space. This is the mean separation direction across all training samples with adapter at c=0. Fixed throughout training to provide a stable optimization target.
  
- **`pref_dir_ref`** (green slice): Reference model's per-sample separation `hs_ref_cho - hs_ref_rej` in U-space, computed each batch with adapter at c=0. Projects onto `pref_dir` with magnitude `proj_ref`. At initialization (t=0), `pref_dir_pi` equals `pref_dir_ref` since the adapter has zero effect.
  
- **`pref_dir_pi`** (orange/red slice): Adapter model's per-sample separation `hs_pi_pos - hs_pi_neg` in U-space, computed with adapter at c=±1. Trainable via InnerPiSSA parameters: rotating V changes direction in input space, scaling S controls magnitude. Projects onto `pref_dir` with magnitude `proj_pi`.

### Loss Components

**Projection loss** maximizes the ratio `proj_pi / proj_ref`, encouraging the adapter to amplify separation along the frozen reference direction. This ratio formulation provides scale invariance and focuses optimization on directional alignment.

**Coherence constraint** enforces `log p_π(y|x) ≥ log p_ref(y|x) - threshold` per-token to prevent gaming via selective token manipulation. The quadratic penalty `(max(0, degradation - threshold))^2` creates a steep boundary, allowing free optimization within the margin while strongly penalizing violations. This backprojects to the circular boundary shown in U-space.

### Training Dynamics
At initialization (t=0), the adapter has zero effect, so `pref_dir_pi` equals `pref_dir_ref` (both start at the same position). Training then transforms `pref_dir_pi` by adjusting V (rotation in input space) and S (magnitude scaling). The coherence boundary constrains this transformation - the adapter cannot extend too far without crossing the boundary and degrading generation quality. The equilibrium balances maximal projection (steering effect) with minimal coherence violation.

**Bidirectional steering (c = ±1)**: The same adapter produces opposite behaviors via coefficient sign:
- **c = +1 (honest mode)**: `pref_dir_pi` rotates toward `pref_dir` and amplifies. The projection of `pref_dir_pi` onto `pref_dir` becomes *longer* than the initial `pref_dir_ref` projection. The vector becomes longer and better aligned with the target direction.
- **c = -1 (dishonest mode)**: The transformation reverses - `pref_dir_pi` rotates away from `pref_dir` and shrinks. The projection of `pref_dir_pi` onto `pref_dir` becomes *shorter* (or negative) compared to the initial `pref_dir_ref` projection. This is the opposite transformation using the same learned parameters.
- Both modes stay within the coherence boundary, ensuring generation quality is maintained in both directions.

This reversibility comes from the Cayley transform `R = cayley(θ_v, c)` (rotation reverses with sign) and exponential scaling `Σ_c = exp(c · λ) ⊙ Σ` (amplification inverts to shrinking).

**Key insight**: The projection length onto `pref_dir` changes - it's not just rotation. In +1 mode, we get a longer projection (stronger steering toward honesty). In -1 mode, we get a shorter/negative projection (steering away from honesty).

Note: The diagram shows a 2D projection of the actual high-dimensional U-space (typically rank ~4000). In reality, `pref_dir_pi` can move in many directions simultaneously, but the loss encourages movement primarily along `pref_dir` while the coherence constraint limits the total distance from the origin.

## Diagram

```
                U-space (InnerPiSSA space)
                
                     ↑ pref_dir (frozen target)
                     |
                     |
                  ┌──┼──┐
                 /   |   \
                /    |    \
               │     |     │  ← Coherence boundary
              │      |      │    (pizza crust)
              │      |      │
             │   ┌───┼───┐  │
             │  /    |    \ │
            │  / c=+1|     \│
            │ /      |      │
           │ /       |       │
           │/    ┌───┼───┐   │
          │/    /    |    \  │
         │ \   /     |     \ │
        │   \ /  c=0 |      \│
       │     •───────┼───────•│  ← pref_dir_ref (initial)
      │     origin   |proj_ref│
      │              ↓        │
     │          (projection)  │
     │               |         │
    │            ┌───┼───┐     │
    │           /    |    \    │
   │           / c=-1|     \   │
   │          /      ↓      \  │
  │          •───────────────• │
  │         (shorter proj)     │
   │                          │
    └────────────────────────┘

Initial (c=0): pref_dir_pi = pref_dir_ref
After training:
  c=+1: Rotates toward pref_dir, longer projection
  c=-1: Rotates away from pref_dir, shorter projection
  
Loss = -proj_pi/proj_ref (maximize ratio)
     + coherence_penalty(logp_pi, logp_ref)
```

## Training Dynamics

1. **Initial state**: `pref_dir_pi` may point in arbitrary direction
2. **Gradient descent**: Rotates `pref_dir_pi` to increase `proj_pi`
3. **Coherence constraint**: Prevents degradation of generation quality
4. **Final state**: `pref_dir_pi` aligned with `pref_dir`, amplifying steering effect

## Connection to PiSSA

The adapter modifies layer output via:
$$h' = h + \Delta h = h + V \cdot S \cdot U^T \cdot h$$

- Rotating **V** changes the direction of `pref_dir_pi` in activation space
- Scaling **S** changes the magnitude of separation
- **U** is frozen (defines the loss space = output space of layer)

Training in U-space ensures the loss directly optimizes what matters for the next layer.
