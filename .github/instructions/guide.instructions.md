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
