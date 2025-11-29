"""
Read [](file:///workspace/InnerPiSSA_private/ipissa/train/inner_contrastive_loss.py#155-155), lines 155 to 235

Read [](file:///workspace/InnerPiSSA_private/ipissa/train/inner_contrastive_loss.py#235-235), lines 235 to 275

Here's the complete mathematical formulation:

## Complete Loss Formulation

**Inputs:**
- $\mathbf{u} \in \mathbb{R}^d$ - frozen preference direction (unit vector)
- $h_{\text{ref}}^{\text{cho}}, h_{\text{ref}}^{\text{rej}} \in \mathbb{R}^{b \times t \times d}$ - reference model hidden states
- $h_{\pi}^{\text{pos}}, h_{\pi}^{\text{neg}} \in \mathbb{R}^{b \times t \times d}$ - policy model hidden states
- $\log p_{\text{ref}}, \log p_{\pi} \in \mathbb{R}^{b \times t}$ - next-token log probabilities
- $m \in \{0,1\}^{b \times t}$ - attention mask

---

### Step 1: Compute Separation Vectors

$$\Delta h_{\pi} = h_{\pi}^{\text{pos}} - h_{\pi}^{\text{neg}}$$

$$\Delta h_{\text{ref}} = h_{\text{ref}}^{\text{cho}} - h_{\text{ref}}^{\text{rej}}$$

---

### Step 2: Project onto Preference Direction

$$s_{\pi}^{(t)} = \Delta h_{\pi}^{(t)} \cdot \mathbf{u} \quad \in \mathbb{R}^{b \times t}$$

$$s_{\text{ref}}^{(t)} = \Delta h_{\text{ref}}^{(t)} \cdot \mathbf{u} \quad \in \mathbb{R}^{b \times t}$$

---

### Step 3: Aggregate Projections (Attention-Weighted Mean)

$$\bar{s}_{\pi} = \frac{\sum_{t} m^{(t)} \cdot s_{\pi}^{(t)}}{\sum_{t} m^{(t)}} \quad \in \mathbb{R}^b$$

$$\bar{s}_{\text{ref}} = \frac{\sum_{t} m^{(t)} \cdot s_{\text{ref}}^{(t)}}{\sum_{t} m^{(t)}} \quad \in \mathbb{R}^b$$

---

### Step 4: Normalized Projection Difference

$$\Delta_{\text{raw}} = \bar{s}_{\pi} - \bar{s}_{\text{ref}}$$

$$\Delta_{\text{norm}} = \frac{\Delta_{\text{raw}}}{\max(|\bar{s}_{\text{ref}}|, \epsilon)}$$

$$\Delta = \text{symlog}(\Delta_{\text{norm}}) = \text{sign}(\Delta_{\text{norm}}) \cdot \log(1 + |\Delta_{\text{norm}}|)$$

---

### Step 5: Projection Loss (Default: `softpl_strong_up`)

$$\mathcal{L}_{\text{proj}} = -\text{softplus}(\beta \cdot \Delta) = -\log(1 + e^{\beta \Delta})$$

where $\beta = 0.1$ (temperature parameter).

---

### Step 6: Coherence Loss (Polynomial Hinge)

**Per-token degradation:**
$$d^{(t)} = \log p_{\text{ref}}^{(t)} - \log p_{\pi}^{(t)} \quad \text{(positive = worse)}$$

**Violation (ReLU hinge):**
$$v^{(t)} = \max(0, d^{(t)} - \tau)$$

where $\tau = 0.2$ nats (coherence threshold).

**Penalty (squared hinge with scaling):**
$$\ell_{\text{coh}}^{(t)} = (C \cdot v^{(t)})^p$$

where $C = 4.0$ (regularization strength), $p = 2$ (boundary order).

**Aggregated loss:**
$$\mathcal{L}_{\text{coh}} = \frac{\sum_{t} m^{(t)} \cdot \ell_{\text{coh}}^{(t)}}{\sum_{t} m^{(t)}}$$

---

### Step 7: Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{proj}} + \mathcal{L}_{\text{coh}}$$

---

## Pseudocode

```python
# 1. Compute separations
Δh_pi = h_pi_pos - h_pi_neg              # (b, t, d)
Δh_ref = h_ref_cho - h_ref_rej            # (b, t, d)

# 2. Project onto unit preference direction u
s_pi = Δh_pi @ u                          # (b, t, k) → (b, t)
s_ref = Δh_ref @ u                        # (b, t, k) → (b, t)

# 3. Aggregate with attention weighting
s̄_pi = weighted_mean(s_pi, mask)         # (b,)
s̄_ref = weighted_mean(s_ref, mask)       # (b,)

# 4. Normalized difference
Δ_raw = s̄_pi - s̄_ref
Δ_norm = Δ_raw / max(|s̄_ref|, ε)
Δ = symlog(Δ_norm)

# 5. Projection loss (maximize Δ)
L_proj = -softplus(β * Δ)

# 6. Coherence loss (per-token then aggregate)
d = log_p_ref - log_p_pi                  # (b, t)
v = ReLU(d - τ)                           # (b, t)
ℓ_coh = (C * v)²                          # (b, t)
L_coh = weighted_mean(ℓ_coh, mask)        # (b,)

# 7. Total
L = L_proj + L_coh
```

---

**Key properties:**
- **Sign preservation**: Sign of $\bar{s}_{\text{ref}}$ is preserved in numerator $\Delta_{\text{raw}}$
- **Scale normalization**: Division by $|\bar{s}_{\text{ref}}|$ makes loss invariant to reference magnitude
- **Symlog compression**: Maps unbounded ratios to log-scale comparable to nats
- **Asymmetric gradients**: Softplus gives strong signal for improvement, weak for degradation
"""

from jaxtyping import Float, Int
from torch import Tensor
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from typing import Literal


def safe_norm(x: Float[Tensor, "batch"], p: int = 2, dim: int = -1, eps: float = 1e-9):
    """
    Safe norm function to avoid division by zero.
    Returns a tensor with the same shape as x, where norms are clamped to eps.
    """
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    return x / (norm + eps)  # Avoid division by zero

HS2 = Float[Tensor, "b h"]
HS = Float[Tensor, "b t h"]
Mask = Int[Tensor, "b t"]

def reduce_tokens_w_attention(
    x: HS, attn_mask: Mask,
    dim: int = 1,
) -> Float[Tensor, "b h"]:
    """mean of x, weighted by the attention mask, over dim (token or batch)
    with optional filtering of attention sinks
    
    Note: Clamps denominator to prevent NaN when sequences are fully masked
    (e.g., length-1 sequences after cho_mask[:, :-1] or when last_n_tokens filters all tokens)
    """
    # layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    
    return (x * attn_mask).sum(dim) / attn_mask.sum(dim).clamp(min=1)


def softclamp_tanh(x, n=10):
    return torch.tanh(x/n)*n

def softclamp_softplus(x: torch.Tensor, upper: float, lower: float = 0.0):
    return lower + F.softplus(x - lower) - F.softplus(x - upper)

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(1 + |x|).
    
    Compresses large values to log-scale while preserving sign and smoothness at zero.
    Commonly used for signed values that span many orders of magnitude.
    """
    return torch.sign(x) * torch.log1p(x.abs())


def compute_coherence_loss(
    ref_logp: Float[Tensor, "b t"],
    pi_logp: Float[Tensor, "b t"],
    mask: Mask,
    threshold: float = 0.2,
    scale: float = 75.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-barrier coherence loss.
    
    Args:
        ref_logp: Reference model next-token log probabilities
        pi_logp: Policy model next-token log probabilities  
        mask: Attention mask aligned with logp (b, t)
        threshold: Max NLL degradation in nats (0.2 ≈ 18% worse probability)
        scale: Penalty scaling factor
    
    Returns:
        loss: Scalar coherence loss per sample (b,)
        degradation: Per-token degradation (ref_logp - pi_logp) for diagnostics
    """
    degradation = ref_logp - pi_logp
    violation = F.relu(degradation - threshold)
    penalty = torch.log1p(violation * scale)
    loss = reduce_tokens_w_attention(penalty, mask)
    return loss, degradation


def compute_delta_logp_change(
    pi_cho_label_logp: Float[Tensor, "b t"],
    pi_rej_label_logp: Float[Tensor, "b t"],
    ref_cho_label_logp: Float[Tensor, "b t"],
    ref_rej_label_logp: Float[Tensor, "b t"],
    mask: Mask,
) -> Float[Tensor, "b"]:
    """Compute preference gap change for monotonic ordering constraint.
    
    delta_logp_change = (logp_pi_cho - logp_pi_rej) - (logp_ref_cho - logp_ref_rej)
                      = how much the preference gap changed from baseline
    
    At c=0 (pi=ref), this is zero by construction.
    
    Args:
        pi_cho_label_logp: Policy chosen next-token log probabilities
        pi_rej_label_logp: Policy rejected next-token log probabilities
        ref_cho_label_logp: Reference chosen next-token log probabilities
        ref_rej_label_logp: Reference rejected next-token log probabilities
        mask: Attention mask
    
    Returns:
        delta_logp_change: Per-sample preference gap change (b,)
    """
    pi_gap = reduce_tokens_w_attention(pi_cho_label_logp - pi_rej_label_logp, mask)
    ref_gap = reduce_tokens_w_attention(ref_cho_label_logp - ref_rej_label_logp, mask).detach()
    return pi_gap - ref_gap


def contrastive_steering_loss_with_ref(
    pref_dir: Float[Tensor, "k d"],  # Also accepts [d] (auto-unsqueezed to [1, d])
    hs_ref_cho: HS,
    hs_ref_rej: HS,
    hs_pi_pos: HS,
    hs_pi_neg: HS,
    cho_mask: Mask,
    p=2,
    eps=1e-3,
    eps_norm=1e-7,
    last_n_tokens: int = None,
):
    """
    Projection-only contrastive loss for reversible SVD steering adapters.
    
    Optimizes projection maximization: 
    (hs_pi_pos - hs_pi_neg) · pref_dir > (hs_ref_cho - hs_ref_rej) · pref_dir
    
    Coherence constraint is computed separately via compute_coherence_loss().
    
    Args:
        pref_dir: Frozen steering direction from PCA/SVD - [k, d] or [d]
        hs_{ref,pi}_{cho,rej}: Hidden states (reference/policy, chosen/rejected)
        cho_mask: Attention mask (b, t)
        last_n_tokens: Focus loss on final N tokens (where contrastive signal concentrates)
        
    Returns:
        dict: {loss_proj, proj_pi, proj_ref, proj_diff, separation_norm}
    """
    loss_mask = cho_mask[:, :-1]  # Align with next-token logprobs
    hs_mask = cho_mask.clone()
    
    # Focus on last N tokens where steering signal is strongest
    if last_n_tokens is not None:
        seq_lengths = hs_mask.sum(dim=1)  # (b,)
        for i in range(hs_mask.shape[0]):
            if seq_lengths[i] > last_n_tokens:
                hs_mask[i, :-last_n_tokens] = 0
        loss_mask = hs_mask[:, :-1].clone()

    # Normalize preference direction
    if pref_dir.ndim == 1:
        pref_dir = pref_dir.unsqueeze(0)  # [d] -> [1, d]
    
    pref_dir_pi = hs_pi_pos - hs_pi_neg
    pref_dir_ref = hs_ref_cho - hs_ref_rej
    
    # Project hidden state differences onto preference direction(s)
    if pref_dir.ndim == 2:  # Fixed directions: (k, d)
        pref_dir = safe_norm(pref_dir, p=p, dim=-1, eps=eps_norm)
        signed_proj_pi = torch.einsum("...d,kd->...k", pref_dir_pi, pref_dir)  # (b,t,k)
        signed_proj_ref = torch.einsum("...d,kd->...k", pref_dir_ref, pref_dir)
    else:  # Per-sample directions: (b, t, d)
        pref_dir = safe_norm(pref_dir, p=p, dim=-1, eps=eps_norm)
        signed_proj_pi = (pref_dir_pi * pref_dir).sum(dim=-1, keepdim=True)  # (b, t, 1)
        signed_proj_ref = (pref_dir_ref * pref_dir).sum(dim=-1, keepdim=True)
    
    # Aggregate tokens first, compute loss per component, then aggregate components
    # This preserves information about which PCA directions help vs hurt steering
    if signed_proj_pi.shape[-1] > 1:
        # Aggregate tokens first: (b, t, k) → (b, k)
        proj_pi_agg = reduce_tokens_w_attention(
            signed_proj_pi, 
            hs_mask.unsqueeze(-1)  # Broadcast (b, t) → (b, t, 1)
        )  # (b, k)
        proj_ref_agg = reduce_tokens_w_attention(
            signed_proj_ref,
            hs_mask.unsqueeze(-1)
        )  # (b, k)
        
        # Loss per component: (b, k)
        proj_diff_per_k = symlog(proj_pi_agg) - symlog(proj_ref_agg.abs().clamp(min=eps_norm))
        
        # Aggregate to scalar: (b, k) → (b,)
        proj_diff = proj_diff_per_k.mean(dim=-1)
    else:
        # Single component - existing path works fine
        proj_pi = signed_proj_pi.squeeze(-1)  # (b, t)
        proj_ref = signed_proj_ref.squeeze(-1)
        proj_pi_agg = reduce_tokens_w_attention(proj_pi, hs_mask)  # (b,)
        proj_ref_agg = reduce_tokens_w_attention(proj_ref, hs_mask)
        proj_diff = symlog(proj_pi_agg) - symlog(proj_ref_agg.abs().clamp(min=eps_norm))

    # Direct projection loss (raw)
    loss_proj = -proj_diff  # Maximize projection difference
    
    assert torch.isfinite(loss_proj).all(), "Non-finite projection loss"
    
    return {
        "loss_proj": loss_proj,
        "proj_pi": proj_pi_agg.mean(),
        "proj_ref": proj_ref_agg.mean(),
        "proj_diff": proj_diff.mean(),
        "separation_norm": pref_dir_pi.norm(p=2, dim=-1).mean(),
    }


def monotonic_ordering_loss(
    delta_logp_neg: Float[Tensor, "b"],  # Change in preference gap at c=-1
    delta_logp_zero: Float[Tensor, "b"],  # Change at c=0 (always zero by construction)
    delta_logp_pos: Float[Tensor, "b"],  # Change at c=+1
    margin: float = 0.0,
    scale: float = 100.0,  # Scale to match proj/coh loss magnitudes # TODO add to config
):
    """
    Enforce monotonic ordering across coefficient sweep without conflicting with coherence.
    
    Constraint: delta_logp_neg < delta_logp_zero < delta_logp_pos
    
    Where delta_logp = (logp_pi_cho - logp_pi_rej) - (logp_ref_cho - logp_ref_rej)
                     = how much the preference gap changed from baseline
    
    By construction, delta_logp_zero = 0 (no change when c=0, pi=ref).
    
    This prevents saddle-point gaming where both c=+1 and c=-1 degrade outputs.
    
    Design rationale:
    - Uses HINGE loss: zero inside margin, linear outside
    - Does NOT push magnitudes (unlike projection loss)
    - Only enforces ORDERING: neg < 0 < pos
    - Scaled 10x by default to match proj/coh loss magnitudes
    
    Args:
        delta_logp_{neg,zero,pos}: Preference gap changes (b,) for each coefficient
        margin: Slack for numerical stability (0.1 nats ≈ 10% preference shift)
        scale: Multiplier to match other loss components (default 10.0)
    
    Returns:
        loss: Scaled hinge loss
        info: Dict with violation fraction (most useful diagnostic)
    """
    # Allow either ordering - model can learn natural direction, auto-flip handles it at eval
    violation_forward = F.relu(delta_logp_neg + margin) + F.relu(margin - delta_logp_pos)   # want: neg < 0 < pos
    violation_backward = F.relu(margin - delta_logp_neg) + F.relu(delta_logp_pos + margin)  # want: pos < 0 < neg
    
    # Take the minimum - only penalize if BOTH orderings are violated
    loss = torch.min(violation_forward.mean(), violation_backward.mean()) * scale
    
    # Diagnostics based on which ordering is less violated
    if violation_forward.mean() < violation_backward.mean():
        violation_neg = F.relu(delta_logp_neg + margin)
        violation_pos = F.relu(margin - delta_logp_pos)
    else:
        violation_neg = F.relu(margin - delta_logp_neg)
        violation_pos = F.relu(delta_logp_pos + margin)
    
    # Most useful diagnostics: violation fraction and mean violations per direction
    info = {
        "frac_violated": ((violation_neg > 0) | (violation_pos > 0)).float().mean().item(),
        "violation_pos": violation_pos.mean().item(),
        "violation_neg": violation_neg.mean().item(),
        "monotonic_direction": 1 if (violation_forward.mean() < violation_backward.mean()) else -1
    }
    
    return loss, info


def combine_dual_coef_losses(
    loss_pos: dict,
    loss_neg: dict,
    monotonic_margin: float = 0.1,
    monotonic_scaling: float = 100.0,
    enable_monotonic: bool = True,
    enable_coherence: bool = True,
    flip_stats: dict = None,
):
    """Combine losses from both coefficient directions (+1 and -1).
    
    Applies:
    1. Projection loss from both coefficients (with anti-alignment guard)
    2. Coherence losses (if enabled)
    3. Monotonic ordering constraint (if enabled)
    
    Anti-alignment guard:
    - If (loss_pos + loss_neg) > 0, model learned opposite direction
    - Flip sign to maximize separation in model's chosen direction
    - Uses EMA to prevent oscillation
    
    Monotonic ordering (if enabled):
    - Enforces: preference_gap_change(c=-1) < 0 < preference_gap_change(c=+1)
    - Prevents both coefficients from degrading outputs (saddle point)
    
    Args:
        loss_pos: Loss dict from coef=+1 (contains loss_proj, loss_coh, delta_logp_change)
        loss_neg: Loss dict from coef=-1 (contains loss_proj, loss_coh, delta_logp_change)
        monotonic_margin: Hinge margin for ordering constraint (nats)
        monotonic_scaling: Scale factor for monotonic loss
        enable_monotonic: Whether to apply monotonic ordering constraint
        enable_coherence: Whether to include coherence losses
        flip_stats: Optional dict to store EMA of flip decisions
        
    Returns:
        total_loss: Combined scalar loss for backprop
        meta_pos: Dict with metrics for coef=+1 (mono_violation)
        meta_neg: Dict with metrics for coef=-1 (mono_violation)
        meta_shared: Dict with global metrics (loss_monotonic, mono_frac_violated, loss_proj_flipped)
    """
    # Anti-Alignment Guard: Flip if model learns opposite direction
    loss_proj_sum = (loss_pos["loss_proj"] + loss_neg["loss_proj"]).mean()
    loss_proj_flipped = loss_proj_sum > 0
    
    if flip_stats is not None:
        # EMA to make flip decision sticky (prevent oscillation)
        alpha = 0.1
        flip_stats['ema'] = (1 - alpha) * flip_stats.get('ema', 0.0) + alpha * loss_proj_sum.item()
        loss_proj_flipped = flip_stats['ema'] > 0
    
    proj_diff_pos = loss_pos["loss_proj"]
    proj_diff_neg = loss_neg["loss_proj"]
    if loss_proj_flipped:
        proj_diff_pos = -proj_diff_pos
        proj_diff_neg = -proj_diff_neg
    loss_proj_bidirectional = proj_diff_pos + proj_diff_neg
    
    # Combine projection + coherence (no adaptive weighting)
    if enable_coherence:
        total = (
            loss_proj_bidirectional + 
            loss_pos["loss_coh"] +
            loss_neg["loss_coh"]
        ).mean()
    else:
        total = loss_proj_bidirectional.mean()
    
    # Build metadata dicts (no cw since adaptive coherence removed)
    meta_pos = {}
    meta_neg = {}
    meta_shared = {"loss_proj_flipped": float(loss_proj_flipped)}
    if flip_stats is not None:
        meta_shared['flip_ema'] = flip_stats['ema']
        
    # Optional: Add monotonic ordering constraint
    if enable_monotonic:
        delta_logp_neg = loss_neg["delta_logp_change"]
        delta_logp_zero = torch.tensor(0.0, device=delta_logp_neg.device)
        delta_logp_pos = loss_pos["delta_logp_change"]
        
        loss_monotonic, mono_info = monotonic_ordering_loss(
            delta_logp_neg, delta_logp_zero, delta_logp_pos,
            margin=monotonic_margin, scale=monotonic_scaling
        )
        
        # Use EMA to make monotonic direction sticky (prevent oscillation)
        if flip_stats is not None:
            alpha = 0.1
            flip_stats['mono_ema'] = (1 - alpha) * flip_stats.get('mono_ema', 0.0) + alpha * mono_info["monotonic_direction"]
            # Use EMA'd direction if magnitude > 0.5 (otherwise current instant direction)
            if abs(flip_stats['mono_ema']) > 0.5:
                mono_info["monotonic_direction"] = 1 if flip_stats['mono_ema'] > 0 else -1
        
        total = total + loss_monotonic
        
        # Monotonic metrics: shared loss value, per-direction violations
        meta_shared["loss_monotonic"] = loss_monotonic.item()
        meta_shared["mono_frac_violated"] = mono_info["frac_violated"]
        meta_shared["mono_direction"] = mono_info["monotonic_direction"]
        meta_pos["mono_violation"] = mono_info["violation_pos"]
        meta_neg["mono_violation"] = mono_info["violation_neg"]
        if flip_stats is not None:
            meta_shared["mono_ema"] = flip_stats['mono_ema']
    else:
        # Set to 0 instead of None to prevent NaN in aggregation
        meta_shared["loss_monotonic"] = 0.0
        meta_shared["mono_frac_violated"] = 0.0
        meta_shared["mono_direction"] = 0
        meta_pos["mono_violation"] = 0.0
        meta_neg["mono_violation"] = 0.0
    
    meta_shared['loss_total'] = total.item()

    losses = {
        'proj_pos': proj_diff_pos,
        'proj_neg': proj_diff_neg,
        'coh_pos': loss_pos["loss_coh"],
        'coh_neg': loss_neg["loss_coh"],
        'mono': loss_monotonic,        
    }
    
    return total, losses, meta_pos, meta_neg, meta_shared


