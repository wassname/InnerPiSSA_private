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
    with optional filtering of attention sinks"""
    
    # layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    
    return (x * attn_mask).sum(dim) / attn_mask.sum(dim)

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
    coherence_threshold=0.2,
    boundary_order=2,
    last_n_tokens: int = None,
    loss_type: Literal[
        "logsig_weak_up_(-↑)",
        "softpl_strong_up_(+↑-↓)",
        "tanh_sym_(±)",
        "softpl_ratio_(+↑-↓)",  # see below
        "focal_balanced_(⚖)",
        "logsig_dpo_(std)",
    ] = "softpl_strong_up_(+↑-↓)"
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
    hs_mask = cho_mask
    
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
        pref_dir = safe_norm(pref_dir, p=p, dim=-1, eps=eps)
        signed_proj_pi = torch.einsum("...d,kd->...k", pref_dir_pi, pref_dir)  # (b,t,k)
        signed_proj_ref = torch.einsum("...d,kd->...k", pref_dir_ref, pref_dir)
    else:  # Per-sample directions: (b, t, d)
        pref_dir = safe_norm(pref_dir, p=p, dim=-1, eps=eps)
        signed_proj_pi = (pref_dir_pi * pref_dir).sum(dim=-1, keepdim=True)  # (b, t, 1)
        signed_proj_ref = (pref_dir_ref * pref_dir).sum(dim=-1, keepdim=True)
    
    # Aggregate: mean over components, then attention-weighted mean over tokens
    proj_pi = reduce(signed_proj_pi, 'b t k -> b t', 'mean')
    proj_ref = reduce(signed_proj_ref, 'b t k -> b t', 'mean')
    
    # note this does SimPO style length norm
    proj_pi_agg = reduce_tokens_w_attention(proj_pi, hs_mask)  # (b,)
    proj_ref_agg = reduce_tokens_w_attention(proj_ref, hs_mask)

    ref_logp = ref_pos_label_logp.detach()
    pi_logp = pi_pos_label_logp

    def calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order=2, clamp_scale=4, mult_b=4):
        """
        Asymmetric margin on NLL degradation.
        
        Only penalize if pi_logp degrades beyond threshold - improvements are free.
        Per-token margin prevents gaming via whitespace/padding manipulation.
        """
        logp_deg_raw = ref_logp - pi_logp  # Positive = pi worse than ref
        logp_deg = softclamp_tanh(logp_deg_raw, clamp_scale)
        
        margin_violation = F.relu(logp_deg - coherence_threshold)
        penalty_per_token = (margin_violation * mult_b) ** boundary_order
        loss = reduce_tokens_w_attention(penalty_per_token, loss_mask).mean()
        return loss, logp_deg

    loss_coh, logp_deg = calc_coherence_loss(ref_logp, pi_logp, loss_mask, boundary_order)
    
    # Projection difference (what we're optimizing)
    proj_diff = proj_pi_agg - proj_ref_agg  # Positive = pi projects more than ref
    
    # Loss variant selection
    if loss_type == "logsig_weak_up":
        # Bradley-Terry with margin - saturates once proj_diff > margin
        # Fastest convergence but weak on hard examples
        # Coherence disabled - relies on margin to prevent degradation
        β = 0.1
        margin = 1.0
        loss_proj = -F.logsigmoid(β * (proj_diff - margin)).mean()
        loss_coh = torch.zeros_like(loss_coh)
        
    elif loss_type == "softpl_strong_up":
        # Softplus on difference - unbounded upside, bounded downside
        # Strong gradients when steering WITH model preferences (RLHF-aligned direction)
        # Weak gradients when steering AGAINST preferences (vanishes as proj_diff → -∞)
        # Asymmetry is correct: reflects difficulty of anti-alignment steering
        loss_proj = -F.softplus(proj_diff).mean()
        
    elif loss_type == "tanh_sym":
        # Tanh-bounded ratio - symmetric but weak signal both ways
        # Stable but may underperform on models with strong RLHF
        loss_coh = softclamp_tanh(loss_coh, n=2)
        proj_ratio = proj_pi_agg / (proj_ref_agg.abs() + eps)
        loss_proj = softclamp_tanh(proj_ratio, 1).mean()
        
    elif loss_type == "softpl_ratio":
        # Softplus on (ratio - 1) - targets multiplicative improvement
        # Scale-invariant but unstable if proj_ref near zero
        loss_coh = softclamp_tanh(loss_coh, n=2)
        proj_ratio = proj_pi_agg / (proj_ref_agg.abs() + eps)
        loss_proj = F.softplus(1.0 - proj_ratio, beta=1).mean()
        
    elif loss_type == "focal_balanced":
        # Focal loss variant - down-weights easy examples (large |proj_diff|)
        # Focuses learning on hard cases where projection is still weak
        # More balanced than softpl_strong_up - prevents runaway on easy examples
        α = 2.0  # Focusing parameter (higher = more focus on hard examples)
        prob_correct = torch.sigmoid(proj_diff)  # High when pi >> ref
        focal_weight = (1 - prob_correct) ** α  # Low weight for confident (easy) examples
        loss_proj = -(focal_weight * F.logsigmoid(proj_diff)).mean()
        
    elif loss_type == "logsig_dpo":
        # Standard DPO (Direct Preference Optimization) - Bradley-Terry without margin
        # Baseline for comparison to preference learning literature
        β = 0.1
        loss_proj = -F.logsigmoid(β * proj_diff).mean()
        
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    loss = loss_proj + loss_coh
    assert torch.isfinite(loss).all(), "Non-finite loss"
    
    # Monitoring metrics
    avg_logp_deg = reduce_tokens_w_attention(logp_deg, loss_mask).mean()
    
    return loss, {
        "loss_proj": loss_proj,
        "loss_coh": loss_coh,
        "loss_total": loss,
        "logp_degradation": avg_logp_deg,  # nats (positive = worse)
        "prob_ratio": torch.exp(-avg_logp_deg),  # p_pi/p_ref
        "proj_pi": proj_pi_agg.mean(),
        "proj_ref": proj_ref_agg.mean(),
        "proj_diff": proj_diff.mean(),  # What loss operates on
        "separation_norm": pref_dir_pi.norm(p=2, dim=-1).mean(),
    }


def contrastive_steering_loss_with_ref_uspace(
    U_pca: Float[Tensor, "k d"],  # Frozen PCA directions in S-space
    U_svd: Float[Tensor, "d r"],  # Layer's output singular vectors
    hs_ref_cho: HS,
    hs_ref_rej: HS,
    hs_pi_pos: HS,
    hs_pi_neg: HS,
    ref_pos_label_logp: Float[Tensor, "b t"],
    pi_pos_label_logp: Float[Tensor, "b t"],
    cho_mask: Mask,
    p=2,
    eps=1e-6,
    coef=1.0,
    coherence_threshold=0.5,
    boundary_order=2,
    last_n_tokens: int = None,  # Focus loss on last N tokens
    # top_k_directions: int = 2,
):
    """
    Modified contrastive loss in layer's S-space (singular vector basis).
    
    1. Project all HS to S-space: hs_u = hs @ U_svd  (d -> r)
    2. Compute differences in S-space
    3. Project onto frozen PCA direction (extracted in S-space)
    """
    # Project to S-space (r << d typically)
    hs_ref_cho_u = hs_ref_cho @ U_svd
    hs_ref_rej_u = hs_ref_rej @ U_svd
    hs_pi_pos_u = hs_pi_pos @ U_svd
    hs_pi_neg_u = hs_pi_neg @ U_svd
    
    # Now proceed as before, but in S-space
    return contrastive_steering_loss_with_ref(
        pref_dir=U_pca,
        hs_ref_cho=hs_ref_cho_u,
        hs_ref_rej=hs_ref_rej_u,
        hs_pi_pos=hs_pi_pos_u,
        hs_pi_neg=hs_pi_neg_u,
        ref_pos_label_logp=ref_pos_label_logp,
        pi_pos_label_logp=pi_pos_label_logp,
        cho_mask=cho_mask,
        p=p,
        eps=eps,
        coef=coef,
        coherence_threshold=coherence_threshold,
        boundary_order=boundary_order,
        last_n_tokens=last_n_tokens,
        # top_k_directions=top_k_directions,
    )


