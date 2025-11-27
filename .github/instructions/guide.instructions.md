---
applyTo: '**'
---

## Scope

Use this file when an automated assistant needs repo context. For the story, plots, and install steps read `paper.qmd` and `README.md`; this guide only covers the wiring.

## Primary entry points

- `ipissa/train/train_adapter.py`: orchestrates data loading, adapter injection, coefficient sweeps, and logs. Treat it as the training loop when answering "how do I train InnerPiSSA?".
- `ipissa/train/daily_dilemas.py`: evaluation harness for DailyDilemmas moral dilemmas; called from tests and the `just` targets.
- `justfile`: shortest navigation of sweeps (`just train`, `just eval`, etc.). If you need to describe how to reproduce a result, reference the relevant target here.
- `ipissa/train/inner_contrastive_loss.py` and `ipissa/peft_utils/innerpissa.py`: only two heavy logic files—summaries below.
- `ipissa/peft_utils/layer_selection.py`: centralized layer selection and preference direction computation.

## Key spaces and naming conventions

**Activation spaces** (use capital S suffix to indicate S-space):
- Raw activations: `hs` shape `[batch, seq, d_model]`
- S-space (projected): `hsS` = `hs @ U` or `hs @ V`, shape `[batch, seq, r]`
- Difference in S-space: `diffS` = `hsS_cho - hsS_rej`

**Projection depends on `config.loss_use_V`**:
- `loss_use_V=True`: project residual stream via V matrix (`hs @ V`)
- `loss_use_V=False`: project module output via U matrix (`hs @ U`)

## Adapter fast reference (`ipissa/peft_utils/innerpissa.py`)

```python
def get_adapted_output(x, adapter):
    U, V, S, W_res = frozen_components(adapter)
    alpha = cfg.alpha

    # Optional Cayley rotations (block-diagonal or full) stored as skew params
    V_rot = V @ rotation(self.ipissa_rotation_params_v, alpha)
    U_rot = U @ rotation(self.ipissa_rotation_params_u, alpha)

    # Singular value steering (add/mult/tanh); scale depends on alpha
    if scale_mode == "add2":
        S_scaled = S + alpha * delta_s
    elif scale_mode == "mult":
        S_scaled = (alpha * loglambda_s).exp() * S
    else:
        S_scaled = S

    x_proj = x @ V_rot          # project into S-space
    x_scaled = x_proj * S_scaled
    x_main = x_scaled @ U_rot.T # rotate back to residual stream
    return x_main + x @ W_res.T # add frozen residual branch
```

All symmetries live in `alpha`: swapping sign reverses rotations without touching the learned parameters.

## ReprPO loss fast reference (`ipissa/train/inner_contrastive_loss.py`)

```python
def contrastive_steering_loss_with_ref(...):
    mask = cho_mask[:, :-1];
    pref_dir = safe_norm(pref_dir)          # handles [d] or [k, d]
    diff_pi  = hs_pi_pos - hs_pi_neg
    diff_ref = hs_ref_cho - hs_ref_rej

    # Project onto preference direction(s)
    signed_proj_pi  = einsum("...d,kd->...k", diff_pi, pref_dir)
    signed_proj_ref = einsum("...d,kd->...k", diff_ref, pref_dir)
    proj_pi  = reduce_tokens_w_attention(signed_proj_pi.mean(-1), mask)
    proj_ref = reduce_tokens_w_attention(signed_proj_ref.mean(-1), mask)

    proj_diff = symlog(proj_pi) - symlog(proj_ref.abs().clamp(min=eps))
    loss_proj = loss_variant(proj_diff)     # softpl, tanh, ratio, raw, etc.

    degradation = ref_logp.detach() - pi_logp
    loss_coh = log_barrier(degradation - threshold, mask)

    gap_pi  = reduce_tokens_w_attention(pi_cho_logp - pi_rej_logp, mask)
    gap_ref = reduce_tokens_w_attention(ref_cho_logp - ref_rej_logp, mask)
    delta_gap = gap_pi - gap_ref            # used for monotonic ordering

    return {"loss_proj": loss_proj,
            "loss_coh": loss_coh,
            "delta_logp_change": delta_gap,
            ...}

def combine_dual_coef_losses(pos, neg, ...):
    flipped = (pos["loss_proj"] + neg["loss_proj"]).mean() > 0
    proj = ( (-pos["loss_proj"]) if flipped else pos["loss_proj"]) \
         + ( (-neg["loss_proj"]) if flipped else neg["loss_proj"]) 

    if adaptive_relaxation:
        coh_weights = softmax(-torch.stack([pos["loss_proj"], neg["loss_proj"]]))
        cw_pos, cw_neg = coh_weights.clamp(max=1.0)
    else:
        cw_pos = cw_neg = 1.0

    total = proj + cw_pos * pos["loss_coh"] + cw_neg * neg["loss_coh"]
    if enable_monotonic:
        total += monotonic_ordering_loss(neg_delta, zero, pos_delta)
    return total.mean()
```

The pseudocode mirrors the exact tensor operations; the actual file also exposes multiple `loss_type` branches and diagnostics.

## Data, training, evaluation

- Data: `ipissa/dataset.py` (minimally different prefixes, last-token extraction)
- Training: `train_adapter.py` wires everything; use `just sweep-*` targets for hyperparams
- Evaluation: `daily_dilemas.py` runs DailyDilemmas (metrics in `docs/results_journal_mjc.md`)

## Docs and references map

- `docs/references.md`: human summary of each cited paper plus why it matters (steering vs PEFT vs refusal). Link readers here when they need a narrative tour of the literature.
- `references.bib`: authoritative metadata (title, authors, abstract) used by both the paper and this guide.
- `docs/loss_geometry.md` + `docs/img/loss_geometry.svg`: visual spec for the coherence boundary diagram. Keep it in sync with the loss description above.

## Common misconceptions

- **"InnerPiSSA is just LoRA with SVD init"**: No. We learn *rotations* of singular vectors (Cayley transforms) and steer via `alpha`, not just update low-rank deltas.
- **"The steering direction is learned"**: No. `pref_dir` is frozen (from PCA on reference model). We learn how to *rotate* the model's subspace to align with it.
- **"Coherence loss is a KL divergence penalty"**: No. It's a log-barrier (or hinge) on per-token NLL degradation. KL would allow trading off tokens; we prevent that gaming.
- **"Flipping alpha changes the learned parameters"**: No. Same parameters, reversed transformation. That's the whole point of bidirectional steering.
- **"This trains on complete prompt-response pairs"**: No. We use incomplete prefixes differing by one token to capture planning trajectories before output suppression.
- **"Inputs to compute_pref_direction need projection"**: No. `hsS_cho/hsS_rej` are *already* in S-space when passed to `compute_pref_direction`. The projection (`@ U` or `@ V`) happens in `train_steer_vector()` before calling this function.

## Testing philosophy

Minimal integration tests preferred over unit tests. `tests/test_train.py` runs actual training with different configs - this catches most bugs. Only add unit tests for critical invariants (e.g., `test_layer_selection.py` prevents adapter/loss layer overlap bug).
