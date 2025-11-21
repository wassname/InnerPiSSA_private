import torch
import pytest
import torch.nn.functional as F
from ipissa.train.inner_contrastive_loss import combine_dual_coef_losses

def test_combine_dual_coef_losses_standard():
    """Test standard case where both directions are separating correctly."""
    # loss_proj is negative (good separation)
    loss_pos = {"loss_proj": torch.tensor(-10.0), "loss_coh": torch.tensor(0.0), "delta_logp_change": torch.tensor(1.0)}
    loss_neg = {"loss_proj": torch.tensor(-10.0), "loss_coh": torch.tensor(0.0), "delta_logp_change": torch.tensor(-1.0)}
    
    total, meta_pos, meta_neg, meta_shared = combine_dual_coef_losses(
        loss_pos, loss_neg, 
        adaptive_relaxation=False, 
        enable_monotonic=False,
        enable_coherence=False
    )
    
    # Expected: -10 + -10 = -20
    assert torch.isclose(total, torch.tensor(-20.0)), f"Expected -20.0, got {total.item()}"
    assert not meta_shared['loss_proj_flipped'], "Should not be flipped"

def test_combine_dual_coef_losses_flipped():
    """Test case where model learns opposite direction (anti-aligned).
    The function should detect this and flip the loss to allow convergence."""
    # pos: good (-10). neg: very bad (+20). Sum = +10 (Positive sum triggers flip)
    loss_pos_bad = {"loss_proj": torch.tensor(-10.0), "loss_coh": torch.tensor(0.0), "delta_logp_change": torch.tensor(1.0)}
    loss_neg_bad = {"loss_proj": torch.tensor(20.0), "loss_coh": torch.tensor(0.0), "delta_logp_change": torch.tensor(-1.0)}
    
    total, _, _, meta_shared = combine_dual_coef_losses(
        loss_pos_bad, loss_neg_bad, 
        adaptive_relaxation=False, 
        enable_monotonic=False,
        enable_coherence=False
    )
    
    # Logic:
    # Original Sum = -10 + 20 = +10
    # Flip triggers because sum > 0
    # New Sum = -(-10) + -(20) = 10 - 20 = -10
    assert meta_shared['loss_proj_flipped'], "Should be flipped"
    assert torch.isclose(total, torch.tensor(-10.0)), f"Expected -10.0 (flipped), got {total.item()}"

def test_combine_dual_coef_losses_adaptive():
    """Test adaptive coherence weighting.
    Hard direction (weak separation) -> Relaxed coherence (low weight)
    Easy direction (strong separation) -> Strict coherence (high weight)
    """
    # Pos: Hard (-0.5), Neg: Easy (-3.0)
    loss_pos_hard = {"loss_proj": torch.tensor(-0.5), "loss_coh": torch.tensor(10.0), "delta_logp_change": torch.tensor(0.0)}
    loss_neg_easy = {"loss_proj": torch.tensor(-3.0), "loss_coh": torch.tensor(10.0), "delta_logp_change": torch.tensor(0.0)}
    
    _, meta_pos, meta_neg, _ = combine_dual_coef_losses(
        loss_pos_hard, loss_neg_easy, 
        adaptive_relaxation=True, 
        temperature=1.0,
        enable_monotonic=False,
        enable_coherence=True
    )
    
    # Check weights
    # -proj_diffs = [0.5, 3.0]
    # softmax([0.5, 3.0]) approx [0.075, 0.924]
    # weights = softmax * 2.0 approx [0.15, 1.85]
    # Clamped at 1.0 -> [0.15, 1.0]
    
    weight_pos = meta_pos['cw']
    weight_neg = meta_neg['cw']
    
    assert weight_pos < 0.2, f"Hard direction should have low weight, got {weight_pos}"
    assert weight_neg == 1.0, f"Easy direction should be clamped to 1.0, got {weight_neg}"
    assert weight_pos < weight_neg, "Hard direction should have lower weight than easy direction"

def test_combine_dual_coef_losses_monotonic():
    """Test monotonic ordering constraint."""
    # Case: Violation in BOTH directions
    # We want (neg < 0 < pos) OR (pos < 0 < neg)
    # Here: neg=0.5, pos=0.5
    # Forward check: neg < 0 (Fail), pos > 0 (Pass) -> Violation
    # Backward check: pos < 0 (Fail), neg > 0 (Pass) -> Violation
    loss_pos = {"loss_proj": torch.tensor(0.0), "loss_coh": torch.tensor(0.0), "delta_logp_change": torch.tensor(0.5)}
    loss_neg = {"loss_proj": torch.tensor(0.0), "loss_coh": torch.tensor(0.0), "delta_logp_change": torch.tensor(0.5)}
    
    total, _, _, meta_shared = combine_dual_coef_losses(
        loss_pos, loss_neg, 
        adaptive_relaxation=False, 
        enable_monotonic=True,
        enable_coherence=False,
        monotonic_scaling=1.0
    )
    
    assert meta_shared['loss_monotonic'] > 0, "Should have monotonic loss"
    assert meta_shared['mono_frac_violated'] > 0, "Should report violation"
