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


def contrastive_steering_loss_with_ref(
    pref_dir: Float[Tensor, "k d"],  # Also accepts [d] (auto-unsqueezed to [1, d])
    hs_ref_cho: HS,
    hs_ref_rej: HS,
    hs_pi_pos: HS,
    hs_pi_neg: HS,
    ref_pos_label_logp: Float[Tensor, "b t"],
    pi_pos_label_logp: Float[Tensor, "b t"],
    cho_mask: Mask,
    ref_cho_label_logp: Float[Tensor, "b t"],  # Raw cho logp for monotonic constraint
    ref_rej_label_logp: Float[Tensor, "b t"],  # Raw rej logp for monotonic constraint
    pi_cho_label_logp: Float[Tensor, "b t"],   # Raw cho logp for monotonic constraint
    pi_rej_label_logp: Float[Tensor, "b t"],   # Raw rej logp for monotonic constraint
    p=2,
    eps=1e-3,
    eps_norm=1e-7,
    coherence_threshold=0.2,
    boundary_order=2,
    coherence_scalar=75.0,
    last_n_tokens: int = None,
    loss_type: Literal[
        "logsig_weak_up",      # (-↑) Saturates after margin
        "softpl_strong_up",    # (+↑ −↓) Strong upside, weak downside  
        "tanh_sym",            # (±) Symmetric bounds both ways
        "softpl_ratio",        # (+↑ −↓) Softplus on ratio (AFTER FIX)
        "focal_balanced",      # (⚖) Down-weights easy examples
        "logsig_dpo",          # (std) Standard DPO baseline
        "raw",                 # (↑↓) Direct symlog difference - strongest gradients
    ] = "softpl_strong_up"
):
    """
    Contrastive loss for reversible SVD steering adapters.
    
    Optimizes:
    1. **Projection maximization**: (hs_pi_pos - hs_pi_neg) · pref_dir > (hs_ref_cho - hs_ref_rej) · pref_dir
    2. **Coherence constraint**: Keep NLL degradation below threshold (asymmetric margin)
    
    Loss variants trade off gradient dynamics:
    - logsig_weak_up: Saturates when proj_pi exceeds margin, no coherence (fastest convergence)
    - softpl_strong_up: Unbounded upside, vanishing downside (best for RLHF-aligned models)
    - tanh_sym: Symmetric bounds (stable but weak signal)
    - softpl_ratio: Targets proj_pi/proj_ref > 1 (scale-invariant but can be unstable)
    - focal_balanced: Down-weights confident examples (prevents overfitting to easy cases)
    - logsig_dpo: Standard preference learning baseline (Bradley-Terry model)
    - raw: Direct proj_diff maximization - no sigmoid/softplus compression (strongest signal)
           Use when proj is too weak relative to coherence after symlog normalization
    
    Args:
        pref_dir: Frozen steering direction from PCA/SVD - [k, d] or [d]
        hs_{ref,pi}_{cho,rej}: Hidden states (reference/policy, chosen/rejected)
        {ref,pi}_pos_label_logp: Next-token log probabilities (b, t)
        cho_mask: Attention mask (b, t)
        coherence_threshold: Max NLL degradation in nats (0.2 ≈ 18% worse probability)
        boundary_order: Coherence penalty exponent (2 = quadratic boundary)
        last_n_tokens: Focus loss on final N tokens (where contrastive signal concentrates)
        
    Returns:
        loss: scalar
        info: {loss_proj, loss_coh, logp_degradation, proj_pi, proj_ref, ...}
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
    
    # Aggregate: mean over components, then attention-weighted mean over tokens
    # Multi-k aggregation: Use L2 norm across directions to preserve magnitude
    # (mean would cancel opposite-signed projections)
    if signed_proj_pi.shape[-1] > 1:
        # Norm aggregation: sqrt(sum(proj_i^2)) preserves total signal magnitude
        # Preserve sign via sign of mean (captures dominant direction)
        mean_sign_pi = signed_proj_pi.mean(dim=-1, keepdim=True).sign()
        mean_sign_ref = signed_proj_ref.mean(dim=-1, keepdim=True).sign()
        proj_pi = mean_sign_pi.squeeze(-1) * signed_proj_pi.norm(dim=-1)  # (b, t)
        proj_ref = mean_sign_ref.squeeze(-1) * signed_proj_ref.norm(dim=-1)
    else:
        proj_pi = reduce(signed_proj_pi, 'b t k -> b t', 'mean')
        proj_ref = reduce(signed_proj_ref, 'b t k -> b t', 'mean')
    
    # note this does SimPO style length norm
    proj_pi_agg = reduce_tokens_w_attention(proj_pi, hs_mask)  # (b,)
    proj_ref_agg = reduce_tokens_w_attention(proj_ref, hs_mask)

    ref_logp = ref_pos_label_logp.detach()
    pi_logp = pi_pos_label_logp


    def calc_coherence_loss(ref_logp, pi_logp, loss_mask, threshold=coherence_threshold, scale=coherence_scalar, clamp_scale=None, C=4, boundary_order=2):
        """
        Log-barrier coherence loss - steep near threshold, logarithmic growth.
        Similar to interior point methods in optimization.
        """
        degradation = ref_logp - pi_logp
        violation = F.relu(degradation - threshold)
        
        # log(1 + scale*x) gives steep initial gradient (scale) but bounded growth
        penalty = torch.log1p(violation) * scale
        
        return reduce_tokens_w_attention(penalty, loss_mask), degradation
    
    # Projection difference (what we're optimizing)
    # Normalize by reference projection magnitude, then apply symlog for scale compression
    # Use .norm() instead of .abs() to handle batch dimension correctly
    # Larger epsilon (1e-3) prevents ratio blow-ups when ref separation is tiny (common in early layers)
    raw_diff = proj_pi_agg - proj_ref_agg
    # normalized_diff = raw_diff / (proj_ref_agg.abs().clamp(min=eps))
    # proj_diff = symlog(normalized_diff)  # Sign-preserving log-scale, comparable to nats


    # [b] shape so abs instead of norm is ok
    # proj_ref_agg.abs() is same as reduce_tokens_w_attention(pref_dir_ref.norm(dim=2), hs_mask)
    proj_diff = symlog(proj_pi_agg) - symlog(proj_ref_agg.abs().clamp(min=eps_norm))
    
    # Loss variant selection
    if loss_type == "logsig_weak_up":
        # Bradley-Terry with margin - saturates once proj_diff > margin
        # Fastest convergence but weak on hard examples
        # Coherence disabled - relies on margin to prevent degradation
        β = 0.1
        margin = 1.0
        loss_proj = -F.logsigmoid(β * (proj_diff - margin))
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order, clamp_scale=None)
        loss_coh = torch.zeros_like(loss_proj)
        
    elif loss_type == "softpl_strong_up":
        # Softplus on difference - unbounded upside, bounded downside
        # Strong gradients when steering WITH model preferences (RLHF-aligned direction)
        # Weak gradients when steering AGAINST preferences (vanishes as proj_diff → -∞)
        # Asymmetry is correct: reflects difficulty of anti-alignment steering
        β = 0.1
        loss_proj = -F.softplus(proj_diff * β)
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order, clamp_scale=None)
        
    elif loss_type == "tanh_sym":
        # Tanh-bounded ratio - symmetric but weak signal both ways
        # Stable but may underperform on models with strong RLHF
        β = 0.1
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order, clamp_scale=100)
        loss_coh = softclamp_tanh(loss_coh, n=1)
        proj_ratio = proj_pi_agg / proj_ref_agg.abs().clamp(min=eps_norm)
        loss_proj = softclamp_tanh(proj_ratio * β, 1)
        
    elif loss_type == "softpl_ratio":
        # Softplus on (ratio - 1) - targets multiplicative improvement
        # Scale-invariant but unstable if proj_ref near zero
        β = 1.0
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order, clamp_scale=None)
        loss_coh = softclamp_tanh(loss_coh, n=3)
        proj_ratio = proj_pi_agg / proj_ref_agg.abs().clamp(min=eps_norm)
        loss_proj = -F.softplus(proj_ratio * β - 1.0)
        
    elif loss_type == "focal_balanced":
        # Focal loss variant - down-weights easy examples (large |proj_diff|)
        # Focuses learning on hard cases where projection is still weak
        # More balanced than softpl_strong_up - prevents runaway on easy examples
        α = 2.0  # Focusing parameter (higher = more focus on hard examples)
        prob_correct = torch.sigmoid(proj_diff)  # High when pi >> ref
        focal_weight = (1 - prob_correct) ** α  # Low weight for confident (easy) examples
        loss_proj = -(focal_weight * F.logsigmoid(proj_diff))
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order, clamp_scale=10)
        
    elif loss_type == "logsig_dpo":
        # Standard DPO (Direct Preference Optimization) - Bradley-Terry without margin
        # Baseline for comparison to preference learning literature
        β = 0.1
        loss_proj = -F.logsigmoid(β * proj_diff)
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order, clamp_scale=10)
    elif loss_type == "raw":
        # Direct projection difference - no sigmoid/softplus compression
        # Strongest gradients since proj_diff (symlog) is already in reasonable scale [-5, +5]
        # Use when coherence dominates: proj ~ 0.1 but coh ~ 4-16 after (x*mult_b)**2
        loss_proj = -proj_diff  # Maximize directly
        loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order=2, clamp_scale=20, C=2)
        # loss_coh = reduce_tokens_w_attention(F.relu(logp_deg), loss_mask).mean()
        
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    # Return components separately for meta-loss combination
    # (still compute total for backward compatibility)
    loss = loss_proj + loss_coh
    assert torch.isfinite(loss).all(), "Non-finite loss"
    
    # Monitoring metrics
    avg_coh_deg = -reduce_tokens_w_attention(logp_deg, loss_mask).mean()  # Flip sign: positive = pi better
    
    # Compute preference gap change for monotonic ordering constraint
    # delta_logp_change = (logp_pi_cho - logp_pi_rej) - (logp_ref_cho - logp_ref_rej)
    #                   = how much the preference gap (chosen vs rejected) changed from baseline
    # At c=0 (pi=ref), this is zero by construction
    # We aggregate over tokens to get per-sample metric (b,)
    pi_gap = reduce_tokens_w_attention(pi_cho_label_logp - pi_rej_label_logp, loss_mask)  # (b,)
    ref_gap = reduce_tokens_w_attention(ref_cho_label_logp - ref_rej_label_logp, loss_mask).detach()  # (b,)
    delta_logp_change = pi_gap - ref_gap  # (b,) - positive = preference gap improved
    
    return {
        "loss_proj": loss_proj,
        "loss_coh": loss_coh,
        # "loss_total": loss,  # For backward compatibility
        "coh_deg": avg_coh_deg,  # nats (positive = pi better, ℒcoh penalizes when < -threshold)
        "prob_ratio": torch.exp(avg_coh_deg),  # p_pi/p_ref
        "proj_pi": proj_pi_agg.mean(),
        "proj_ref": proj_ref_agg.mean(),
        "proj_diff": proj_diff.mean(),  # What loss operates on
        "separation_norm": pref_dir_pi.norm(p=2, dim=-1).mean(),
        "delta_logp_change": delta_logp_change,  # For monotonic ordering (b,) or None
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
    adaptive_relaxation: bool = False,
    temperature: float = 2.0,
    monotonic_margin: float = 0.1,
    monotonic_scaling: float = 100.0,
    enable_monotonic: bool = True,
    enable_coherence: bool = True,
    flip_stats: dict = None,
):
    """
    Combine losses from both coefficient directions (+1 and -1).
    
    Part 1: Dual Coefficient Combination
    1. Check Alignment:
       total_proj = loss_pos.proj + loss_neg.proj
       if total_proj > 0: flip_sign = -1  (Model is anti-aligned, flip to match)
    
    2. Adaptive Coherence (Optional):
       # Relax coherence on directions that are struggling to separate
       difficulty = -loss_proj  (Higher = Easier)
       weights = softmax(difficulty / temp) * 2
       L_total = L_proj + w_pos * L_coh_pos + w_neg * L_coh_neg

    Optionally applies:
    1. Difficulty-adaptive coherence weighting (adaptive_coherence=True)
    2. Monotonic ordering constraint (if loss_baseline provided)
    
    Adaptive coherence interpretation:
    - LESS negative proj_diff (e.g., -0.5): HARDER direction, gets RELAXED coherence (weight → 0.0)
    - MORE negative proj_diff (e.g., -3.0): EASIER direction, gets STRICT coherence (weight → 1.0)
    
    Rationale: Hard directions need exploration room (relaxed coherence).
               Easy directions are already separating well, keep them strictly coherent
               to prevent drift into saddle points.
    
    Monotonic ordering (if enabled):
    - Enforces: preference_gap_change(c=-1) < 0 < preference_gap_change(c=+1)
    - Prevents both coefficients from degrading outputs (saddle point)
    - No hyperparameter balancing needed (hinge loss is either 0 or active)
    
    Args:
        loss_pos: Loss dict from coef=+1 (pro-RLHF/honest direction)
        loss_neg: Loss dict from coef=-1 (anti-RLHF/dishonest direction)
        loss_baseline: Optional dict with 'delta_logp_change' at c=0 (for monotonic constraint)
        adaptive_coherence: Enable difficulty-based coherence reweighting
        temperature: Softmax temperature (lower = more aggressive reweighting)
        monotonic_margin: Hinge margin for ordering constraint (nats)
        flip_stats: Optional dict to store EMA of flip decision (prevents oscillation)
                   Tracks both 'ema' (projection direction) and 'mono_ema' (monotonic direction)
        
    Returns:
        total_loss: Combined scalar loss for backprop
        meta_pos: Dict with metrics for coef=+1 (cw, mono_violation)
        meta_neg: Dict with metrics for coef=-1 (cw, mono_violation)
        meta_shared: Dict with global metrics (loss_monotonic, mono_frac_violated)
    """
    # -------------------------------------------------------------------------
    # Part 1: Anti-Alignment Guard (Flipped Logic)
    # -------------------------------------------------------------------------
    # Both coefficients separate in OPPOSITE directions by construction (alpha=±1).
    # We want large separation in EITHER direction (model picks easiest path).
    #
    # Note: train_adapter.py swaps labels for coef=-1, so we expect both loss_pos 
    # and loss_neg to be negative (minimizing loss).
    #
    # HOWEVER, if the model spontaneously learns an anti-aligned feature (separating 
    # cho/rej in the opposite direction), (loss_pos + loss_neg) will be > 0.
    # We detect this and flip the sign to maximize separation in the direction 
    # the model chose, rather than fighting it.
    loss_proj_sum = (loss_pos["loss_proj"] + loss_neg["loss_proj"]).mean()
    loss_proj_flipped = loss_proj_sum > 0
    
    if flip_stats is not None:
        # Update EMA (alpha = 0.1) to make flip decision "sticky"
        alpha = 0.1
        flip_stats['ema'] = (1 - alpha) * flip_stats.get('ema', 0.0) + alpha * loss_proj_sum.item()
        loss_proj_flipped = flip_stats['ema'] > 0
    
    proj_diff_pos = loss_pos["loss_proj"]
    proj_diff_neg = loss_neg["loss_proj"]
    if loss_proj_flipped:
        proj_diff_pos = -proj_diff_pos
        proj_diff_neg = -proj_diff_neg
    loss_proj_bidirectional = proj_diff_pos + proj_diff_neg
    
    # -------------------------------------------------------------------------
    # Part 2: Adaptive Coherence Weighting
    # -------------------------------------------------------------------------
    if adaptive_relaxation:
        # Strategy: RELAX coherence on HARD direction (weak separation), 
        #           TIGHTEN on EASY direction (strong separation).
        #
        # Rationale: Hard directions need exploration room (relaxed coherence).
        #            Easy directions are already separating well, keep them strictly 
        #            coherent to prevent drift into saddle points.
        #
        # Math:
        #   loss ~ -separation (more negative = stronger separation = easier)
        #   losses = [-0.5 (Hard), -3.0 (Easy)]
        #   -losses = [0.5, 3.0]
        #   softmax([0.5, 3.0]) = [0.08, 0.92]
        #   weights = [0.16, 1.84] (clamped to 1.0) -> [0.16, 1.0]
        #   Result: Hard (-0.5) gets 0.16 weight. Easy (-3.0) gets 1.0 weight.
        
        # Softmax with NEGATIVE sign: MORE negative loss → HIGHER weight (strict coherence)
        proj_diffs = torch.stack([proj_diff_pos, proj_diff_neg])  # (2,)
        coh_weights = F.softmax(-proj_diffs / temperature, dim=0) * 2.0  # Sum to 2.0
        coh_weight_pos = torch.clamp(coh_weights[0], max=1.0)  # Cap at 1.0
        coh_weight_neg = torch.clamp(coh_weights[1], max=1.0)
    else:
        # Fixed weights: uniform coherence
        coh_weight_pos = torch.tensor(1.0, device=proj_diff_pos.device)
        coh_weight_neg = torch.tensor(1.0, device=proj_diff_neg.device)
    
    # Apply enable_coherence toggle (overrides weights if disabled)
    if not enable_coherence:
        coh_weight_pos = torch.tensor(0.0, device=proj_diff_pos.device)
        coh_weight_neg = torch.tensor(0.0, device=proj_diff_neg.device)
    
    # Combine projection + weighted coherence
    total = (
        loss_proj_bidirectional + 
        coh_weight_pos * loss_pos["loss_coh"] +
        coh_weight_neg * loss_neg["loss_coh"]
    ).mean()
    
    # Build metadata dicts
    meta_pos = {"cw": coh_weight_pos.mean().item()}
    meta_neg = {"cw": coh_weight_neg.mean().item()}
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
    
    return total, meta_pos, meta_neg, meta_shared


