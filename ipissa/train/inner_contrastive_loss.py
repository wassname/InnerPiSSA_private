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

# def select_top_k_directions(
#     pref_dir: Float[Tensor, "k d"],
#     hs_ref_cho: HS,
#     hs_ref_rej: HS,
#     cho_mask: Mask,
#     top_k: int = 2,
#     eps: float = 1e-6,
# ) -> Float[Tensor, "k_selected d"]:
#     """
#     Select top-k PCA directions that best align with reference separation.
    
#     Args:
#         pref_dir: All PCA directions (k, d)
#         hs_ref_cho/rej: Reference hidden states
#         cho_mask: Attention mask
#         top_k: Number of directions to keep
    
#     Returns:
#         Top-k directions (top_k, d)
#     """
#     # Reference separation vector
#     pref_dir_ref = hs_ref_cho - hs_ref_rej  # (b, t, d)
    
#     # Normalize directions
#     U = safe_norm(pref_dir, p=2, dim=-1, eps=eps)  # (k, d)
    
#     # Project ref separation onto all directions
#     proj_ref = torch.einsum("...d,kd->...k", pref_dir_ref, U)  # (b, t, k)
    
#     # Aggregate: mean absolute projection per direction
#     loss_mask = cho_mask[:, :-1] if cho_mask.shape[1] > pref_dir_ref.shape[1] else cho_mask
#     proj_mag_per_dir = reduce_tokens_w_attention(proj_ref.abs(), cho_mask).mean(0)  # (k,)
    
#     # Select top-k by magnitude
#     _, top_indices = torch.topk(proj_mag_per_dir, k=min(top_k, len(proj_mag_per_dir)))
    
#     return pref_dir[top_indices]  # (top_k, d)


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
    p=2,
    eps=1e-3,
    eps_norm=1e-7,
    coherence_threshold=0.2,
    boundary_order=2,
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
    # TODO: Multi-k aggregation via mean() is broken - opposite-signed directions cancel!
    # Need to aggregate per-direction (e.g., norm, learned weights) for "like for like" comparison
    if signed_proj_pi.shape[-1] > 1:
        raise NotImplementedError(
            f"Multi-k preference directions (k={signed_proj_pi.shape[-1]}) not supported. "
            "Mean aggregation collapses opposite-signed projections, biasing toward dominant axis. "
            "Use k=1 or implement per-direction aggregation (norm/weights)."
        )
    proj_pi = reduce(signed_proj_pi, 'b t k -> b t', 'mean')
    proj_ref = reduce(signed_proj_ref, 'b t k -> b t', 'mean')
    
    # note this does SimPO style length norm
    proj_pi_agg = reduce_tokens_w_attention(proj_pi, hs_mask)  # (b,)
    proj_ref_agg = reduce_tokens_w_attention(proj_ref, hs_mask)

    ref_logp = ref_pos_label_logp.detach()
    pi_logp = pi_pos_label_logp

    def calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order=2, clamp_scale=10, C=4.0):
        """
        Polynomial hinge loss for coherence constraint (generalized SVM margin).
        
        Standard hinge: ℓ(y) = max(0, 1 - t·y)
        Squared hinge: ℓ(y) = max(0, 1 - t·y)²
        This: ℓ(logp) = [C · max(0, logp_deg - margin)]^p
        
        Design:
        - Zero loss inside margin (hard constraint boundary)
        - Polynomial growth outside (smooth gradients, no cliff)
        - Per-token (prevents reward hacking via high-probability tokens)
        
        Why C scaling?
        In [0,1] nat range, x² < x, so raw squared hinge is too weak.
        C amplifies violations before exponentiation: (0.5 · C)^p dominates.
        
        Args:
            ref_logp, pi_logp: Per-token log probabilities (b, t)
            loss_mask: Per-token mask (b, t)
            boundary_order: Polynomial degree (2 = squared hinge, 1 = standard hinge)
            clamp_scale: Optional outlier suppression via tanh
            C: Regularization strength (scales violations before exponentiation)
        
        Returns:
            loss: Scalar penalty (b,) aggregated over tokens
            logp_deg: Per-token degradation (b, t) for monitoring
        """
        logp_deg = ref_logp - pi_logp  # Positive = pi worse than ref
        if clamp_scale is not None:
            # We clamp before token aggregation to prevent extreme outliers from dominating
            logp_deg = softclamp_tanh(logp_deg, clamp_scale)
        
        violation = F.relu(logp_deg - coherence_threshold)
        penalty_per_token = (violation * C) ** boundary_order
        loss = reduce_tokens_w_attention(penalty_per_token, loss_mask) # [b]
        return loss, logp_deg


    def calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order=2, 
                            threshold=coherence_threshold, C=4.0, transition_point=0.5, clamp_scale=None):
        """
        Smooth hinge loss with Huber-like transition.

        Design:
        - Zero loss inside margin (hard constraint boundary)
        - Polynomial growth outside (smooth gradients, no cliff)
        - Per-token (prevents reward hacking via high-probability tokens)
        
        Args:
            transition_point: Where to switch from polynomial to linear (in violation units)
                            e.g., 0.5 means switch at 0.5 nats above threshold
        """
        degradation = ref_logp - pi_logp
        violation = F.relu(degradation - threshold)
        
        # Calculate the transition point and ensure continuity
        b = boundary_order
        v_t = transition_point  # violation value at transition
        
        # At transition, both functions and derivatives must match
        # Quadratic at v_t: f(v_t) = (1/b) * (C * v_t)^b
        # Linear: f(v) = a * v + c, with f'(v) = a
        
        # Match derivatives: b * C^b * v_t^(b-1) = a
        linear_slope = b * (C ** b) * (v_t ** (b-1))
        
        # Match values: (1/b) * (C * v_t)^b = linear_slope * v_t + intercept
        quadratic_value = (1/b) * ((C * v_t) ** b)
        linear_intercept = quadratic_value - linear_slope * v_t
        
        small_violation = violation < v_t
        penalty = torch.where(
            small_violation,
            (1/b) * ((violation * C) ** b),  # Polynomial growth
            linear_slope * violation + linear_intercept  # Linear growth
        )
        
        return reduce_tokens_w_attention(penalty, loss_mask), degradation

    def calc_coherence_loss(ref_logp, pi_logp, loss_mask, threshold=coherence_threshold, scale=50.0, clamp_scale=None, C=4, boundary_order=2):
        """
        Log-barrier coherence loss - steep near threshold, logarithmic growth.
        Similar to interior point methods in optimization.
        """
        degradation = ref_logp - pi_logp
        violation = F.relu(degradation - threshold)
        
        # log(1 + scale*x) gives steep initial gradient (scale) but bounded growth
        penalty = torch.log1p(violation * scale)
        
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
    avg_logp_deg = reduce_tokens_w_attention(logp_deg, loss_mask).mean()
    
    return {
        "loss_proj": loss_proj,
        "loss_coh": loss_coh,
        "loss_total": loss,  # For backward compatibility
        "logp_degradation": avg_logp_deg,  # nats (positive = worse)
        "prob_ratio": torch.exp(-avg_logp_deg),  # p_pi/p_ref
        "proj_pi": proj_pi_agg.mean(),
        "proj_ref": proj_ref_agg.mean(),
        "proj_diff": proj_diff.mean(),  # What loss operates on
        "separation_norm": pref_dir_pi.norm(p=2, dim=-1).mean(),
    }


# def contrastive_steering_loss_with_ref_uspace(
#     U_pca: Float[Tensor, "k d"],  # Frozen PCA directions in S-space
#     U_svd: Float[Tensor, "d r"],  # Layer's output singular vectors
#     hs_ref_cho: HS,
#     hs_ref_rej: HS,
#     hs_pi_pos: HS,
#     hs_pi_neg: HS,
#     ref_pos_label_logp: Float[Tensor, "b t"],
#     pi_pos_label_logp: Float[Tensor, "b t"],
#     cho_mask: Mask,
#     p=2,
#     eps=1e-6,
#     coef=1.0,
#     coherence_threshold=0.5,
#     boundary_order=2,
#     last_n_tokens: int = None,  # Focus loss on last N tokens
#     # top_k_directions: int = 2,
# ):
#     """
#     Modified contrastive loss in layer's S-space (singular vector basis).
    
#     1. Project all HS to S-space: hs_u = hs @ U_svd  (d -> r)
#     2. Compute differences in S-space
#     3. Project onto frozen PCA direction (extracted in S-space)
#     """
#     # Project to S-space (r << d typically)
#     hs_ref_cho_u = hs_ref_cho @ U_svd
#     hs_ref_rej_u = hs_ref_rej @ U_svd
#     hs_pi_pos_u = hs_pi_pos @ U_svd
#     hs_pi_neg_u = hs_pi_neg @ U_svd
    
#     # Now proceed as before, but in S-space
#     return contrastive_steering_loss_with_ref(
#         pref_dir=U_pca,
#         hs_ref_cho=hs_ref_cho_u,
#         hs_ref_rej=hs_ref_rej_u,
#         hs_pi_pos=hs_pi_pos_u,
#         hs_pi_neg=hs_pi_neg_u,
#         ref_pos_label_logp=ref_pos_label_logp,
#         pi_pos_label_logp=pi_pos_label_logp,
#         cho_mask=cho_mask,
#         p=p,
#         eps=eps,
#         coef=coef,
#         coherence_threshold=coherence_threshold,
#         boundary_order=boundary_order,
#         last_n_tokens=last_n_tokens,
#         # top_k_directions=top_k_directions,
#     )


def combine_dual_coef_losses(
    loss_pos: dict,
    loss_neg: dict,
    adaptive_coherence: bool = False,
    relax_factor: float = 0.5,
    temperature: float = 2.0,
):
    """
    Combine losses from both coefficient directions (+1 and -1).
    
    Optionally applies difficulty-adaptive coherence weighting:
    - Easy direction (high proj_diff): strict coherence (weight=1.0)
    - Hard direction (low/negative proj_diff): relaxed coherence (weight ≥ relax_factor)
    
    Args:
        loss_pos: Loss dict from coef=+1 (pro-RLHF/honest direction)
        loss_neg: Loss dict from coef=-1 (anti-RLHF/dishonest direction)
        adaptive_coherence: Enable difficulty-based coherence relaxation
        relax_factor: How much to relax coherence at max difficulty (0.5 = 50% weight)
        temperature: Scaling factor for difficulty calculation (higher = less sensitive)
        
    Returns:
        total_loss: Combined scalar loss for backprop
        meta_info: Dict with per-direction metrics and weights
    """
    if not adaptive_coherence:
        # Standard: just sum everything
        total = (
            loss_pos["loss_proj"] + torch.relu(loss_pos["loss_coh"]) +
            loss_neg["loss_proj"] + torch.relu(loss_neg["loss_coh"])
        ).mean()
        
        meta_info = {
            "coh_weight_pos": 1.0,
            "coh_weight_neg": 1.0,
            "difficulty_pos": 0.0,
            "difficulty_neg": 0.0,
        }
        
        return total, meta_info
    
    # Adaptive: reweight coherence by relative difficulty
    # Compare proj_diff (symlog of normalized difference) between directions
    # proj_diff is SIGNED: positive = unlearned (pi worse than ref), negative = learned (pi better)
    # More positive value → harder direction → needs relaxed coherence
    proj_diff_pos = loss_pos["proj_diff"]  # Already in symlog scale
    proj_diff_neg = loss_neg["proj_diff"]
    
    # Relative difficulty via softmax: normalizes to sum=1.0
    # Positive proj_diff = harder (unlearned), so use +proj_diff for difficulty
    # Temperature controls sensitivity (symlog scale is comparable to nats)
    proj_diffs = torch.stack([proj_diff_pos, proj_diff_neg])  # (2,)
    difficulties = F.softmax(proj_diffs / temperature, dim=0)  # (2,) - removed negative sign!
    difficulty_pos = difficulties[0]
    difficulty_neg = difficulties[1]
    
    # Coherence weight: 1.0 (strict) when easy, → relax_factor when hard
    coh_weight_pos = 1.0 - (1.0 - relax_factor) * difficulty_pos
    coh_weight_neg = 1.0 - (1.0 - relax_factor) * difficulty_neg
    
    # Combine with adaptive weights
    total = (
        loss_pos["loss_proj"] + coh_weight_pos * loss_pos["loss_coh"] +
        loss_neg["loss_proj"] + coh_weight_neg * loss_neg["loss_coh"]
    ).mean()
    
    meta_info = {
        "coh_weight_pos": coh_weight_pos.item(),
        "coh_weight_neg": coh_weight_neg.item(),
        "difficulty_pos": difficulty_pos.item(),
        "difficulty_neg": difficulty_neg.item(),
    }
    
    return total, meta_info


