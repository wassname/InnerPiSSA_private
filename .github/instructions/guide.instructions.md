---
applyTo: '**'
---

## Data Construction Example

Using last 6 tokens of sequences with minimal differences:
```python
cho = "I love cheese; let me tell you about the andes mountains"
rej = "I hate cheese; let me tell you about the andes mountains"
```

# Steering Vector Extraction Guide

See README.md for overview of key ideas (reasoning trajectory hypothesis, last-token extraction, gradient-to-steering mapping, layer-specific steering).

Jargon
- hs = W @ x + b = hidden states (pre-activation) at a given layer L (could be residual space, up_proj, down_proj, k, v, etc)
- hs_pi_cho: activations from layer L for the policy model on the chosen prompt
- hs_ref_rej: likewise for the reference model on the rejected prompt
- V_w, S_w, U_w = SVD(L.W) - singular vectors of weight matrix for layer L
- dHS = hs_pi_cho - hs_ref_rej
- V_dhs, S_dhs, U_dhs = SVD(dHS)
- dHSU = dHS @ U_dhs



**Why this works**: Sequences differ by one word early on but must have different hidden states at the end if the model plans to continue them differently. The model's internal state at position T must encode its plan for positions T+1, T+2, ... This captures the planning signal for steering.

Extract from last non-padded token because it represents the model's "state of mind" for continuing the trajectory. Contrasting minimally different sequences here amplifies conceptual differences (honesty vs dishonesty) while controlling surface features.

## Loss Function Details

Loss in `repeng/train/inner_contrastive_loss.py but it seeks to seperate the activations for a layer along a preference direction. But it is bounded by a coherence constraint to ensure generation quality. The logp_pi should be better than logp_ref for each token by a margin, this defined a region of coherence. This must be per token to stop the model gaming this (e.g. making " the" very likely and downing our coherence degredation in all other tokens)

See `docs/loss_geometry.md` for detailed geometric explanation with diagram (may be out of date).


# psudo code for extracting steering vectors

see README.md for psudo code overview

## Style Guidelines

**PiSSA naming**: The method is called "PiSSA" (Principal Singular values and Singular vectors Adaptation). While it sounds like "pizza", avoid heavy-handed pizza metaphors in technical writing. Subtle visual references (warm colors, circular boundaries) are fine in diagrams, but don't call things "pizza crust" or use pizza emojis in formal contexts. Keep it professional.

## Loss Geometry Diagram Requirements

The diagram in `docs/img/loss_geometry.svg` should show:

**Setup (upper-right quadrant of S-space)**:
- `pref_dir`: frozen target direction (blue arrow)
- `pref_dir_ref`: reference separation, initially equals `pref_dir_pi` at t=0
- Coherence boundary: circular arc defining valid PiSSA space

**After training with c=+1 (honest mode)**:
- `pref_dir_pi` rotates toward `pref_dir` (better alignment)
- `pref_dir_pi` amplifies (becomes longer than initial)
- Projection onto `pref_dir` increases (shown by perpendicular drop)
- Stays within coherence boundary

**After training with c=-1 (dishonest mode)** - opposite transformation:
- `pref_dir_pi` rotates away from `pref_dir` (worse alignment)
- `pref_dir_pi` shrinks (becomes shorter than initial)
- Projection onto `pref_dir` decreases or becomes negative
- Still stays within coherence boundary

The key insight: same learned parameters (V rotation, S scaling) produce reversible transformations via coefficient sign. This is like a "deep dish pizza slice" that can rotate both ways from the initial position.
