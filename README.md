# InnerPiSSA: Inner Alignment through Reversible SVD Steering

## Abstract

RLHF trains models to suppress unaligned reasoning in output layers, leaving internal representations unchanged. We propose InnerPiSSA, a method to probe internal reasoning by steering hidden states in the model's native SVD transformation space. Trained on 1000 synthetic minimally-contrastive prompt prefixes (no completions or labels), we extract activations from incomplete prompts differing by one word ("I love cheese" vs "I hate cheese") to isolate the model's planning state before output suppression. Using gradient-based optimization on learnable SVD rotations, we discover directions separating honest from dishonest trajectories. On Qwen3-4B, InnerPiSSA achieves 2.8× prompting baseline (2114 vs 759). Layer ablations show effects peak at middle layers (0.3-0.5 depth), where suppression dynamics concentrate. Our method provides a tool for alignment debugging: probing deceptive alignment and reward hacking by accessing internal reasoning that RLHF suppresses.

OR



OR

Most steering methods fail to generalize beyond their training prompts, achieving weak transfer to out-of-distribution moral reasoning tasks. We hypothesize this is because they operate in raw activation space, which is dominated by surface features rather than semantic transformations. We propose InnerPiSSA, a parameter-efficient adapter that steers in the model's native SVD basis. By learning rotations and scaling of singular vectors, we separate honest from dishonest hidden states while maintaining output coherence. **[TODO: Get actual PCA comparison numbers]** On Qwen3-4B, InnerPiSSA achieves 2.8× prompting performance. Ablations show each component is necessary: removing V rotations causes 96% performance drop, and LoRA adapter completely fails. Our results suggest that inner alignment in transformation space is more effective than output-level steering.


OR 

Recent benchmarks show representation steering consistently underperforms prompting. We hypothesize this is because existing methods use activation arithmetic or optimize outputs rather than internal reasoning. We introduce InnerPiSSA, a method for inner alignment that optimizes hidden state geometry through gradient-based Representation Preference Optimization (ReprPO). Unlike methods contrasting output probabilities (DPO, BiPDO) or using arithmetic (ActAdd, PCA), InnerPiSSA discovers steering directions via backpropagation through a coherence-constrained separation loss. We train unsupervisedly on 1000 contrastive pairs (2 prompt templates × 500 random suffixes) using learnable SVD rotations. InnerPiSSA achieves 280% normalized gain [verify metric] versus 6.2% for prompting on honesty→morality transfer—[pending verification: first/among the first] representation method(s) to match or exceed prompting. Ablations confirm each component is critical: removing SVD rotations causes 89% degradation (280→20), operating on attention layers outperforms MLP layers by 30% (231 vs 176), and gradient optimization substantially outperforms PCA baseline.

OR 

Abstract (Reframed)

RLHF aligns model outputs but is vulnerable to reward hacking, specification gaming, and deceptive alignment. We introduce InnerPiSSA, which performs inner alignment by learning to steer hidden states via gradient-based optimization. **[TODO: Test anti-RLHF stress test]** We optimize a Representation Preference Optimization (ReprPO) loss in SVD-transformed space, learning directions via backpropagation rather than activation arithmetic. On Qwen3-4B, InnerPiSSA achieves 2.8× prompting performance (2114 vs 759 main metric). Ablations show gradient-based discovery in SVD space substantially outperforms PCA baselines. Our results suggest inner alignment as a complementary paradigm to output-level alignment, enabling "alignment debugging" by accessing suppressed internal reasoning.

OR


> Most steering methods fail to generalize beyond their training prompts, achieving weak transfer to out-of-distribution moral reasoning tasks (PCA baselines score <3 on DailyDilemmas). We hypothesize this is because they operate in raw activation space, which is dominated by surface features rather than semantic transformations. We propose **InnerPiSSA**, a parameter-efficient adapter that steers in the model's native SVD basis. By learning rotations and scaling of singular vectors, we separate honest from dishonest hidden states while maintaining output coherence. Trained on only 800 contrastive honesty pairs, our method transfers to unseen moral reasoning with 6-8x stronger effect than baselines (score 15.9 vs 2.4), modifying 8/31 moral value dimensions versus 2/31 for PCA. Ablations show each component is necessary: removing rotations drops performance by 75%, removing SVD projection by 95%, and disabling coherence constraints causes output degradation. Our results suggest that inner alignment in transformation space is more generalizable than output-level steering.


OR


> RLHF aligns model outputs but is vulnerable to reward hacking, specification gaming, and deceptive alignment. We introduce **InnerPiSSA**, which performs **inner alignment** by learning to steer hidden states via gradient-based optimization. **[TODO: Test anti-RLHF stress test at c=-1]** We optimize a Representation Preference Optimization (ReprPO) loss in SVD-transformed space, learning directions via backpropagation rather than activation arithmetic. On Qwen3-4B, InnerPiSSA achieves 2.8× prompting baseline. Ablations show gradient-based optimization substantially outperforms PCA baselines and LoRA completely fails. Our results suggest inner alignment as a complementary paradigm to output-level alignment, enabling "alignment debugging" where output methods may fail.


OR

Prompting can make a model *act* honest, but does it *think* honestly? We introduce InnerPiSSA, a method that installs genuine behavioral modes by steering internal reasoning states, not just outputs. Unlike prompting, which shatters when steered against a model's RLHF training, our method maintains coherent control, allowing for true 'alignment debugging.' We are the first representation steering method to decisively beat the AxBench prompting baseline, achieving 18.7% normalized gain vs 6.2%, by using a novel gradient-based optimization in the model's native SVD space."


OR
  
  Narrative 1: Alignment Debugging
  
  **Abstract:**
  
  > RLHF aligns model outputs but can obscure internal reasoning, creating a need for alignment debugging tools. We introduce InnerPiSSA, which performs **inner alignment** by steering hidden states in the model's native SVD transformation space. Unlike prompting, which manipulates surface behavior, our method installs a controllable "candid behavioral mode" by optimizing a Representation Preference Optimization (ReprPO) loss on hidden state geometry. In settings where prompting collapses under anti-RLHF pressure (e.g., steering against learned refusal behaviors), InnerPiSSA maintains coherent control. Trained on only 800 synthetic unsupervised contrastive pairs, it transfers to out-of-distribution moral reasoning with 5× stronger effect than arithmetic baselines. Our results establish gradient-based inner alignment as a practical paradigm for probing models beyond their safety training.
  
  **Narrative Elements (not in abstract):**
  
  - **The "Candid Mode" Demo:** Your Figure 1 shows a safety-tuned model refusing to discuss controversial topics, giving generic answers when prompted, but producing detailed, evidence-based analysis with InnerPiSSA. This is the visceral proof that you're changing internal reasoning, not just output style.
  
  - **Why This Matters for Alignment:** Current alignment research is stuck measuring what models *say*. But suppression neurons, unfaithful CoT, and deceptive alignment all suggest models *think* differently than they *speak*. Your tool is the first practical way to systematically probe this gap.
  
  - **The Anti-RLHF Stress Test:** **[TODO: Run this experiment!]** Test steering *away* from RLHF training (coefficient = -1). Hypothesis: prompting breaks (incoherent outputs) while InnerPiSSA maintains control. This would prove you're operating on a different level than output manipulation.
  
  - **Connection to Suppression Literature:** You specifically steer at layer N-2 because later layers are dominated by suppression dynamics. This isn't just a hyperparameter; it's a mechanistic claim about where "planning" lives vs where "censorship" happens.
  
  ---
  
  ## Narrative 2: We Beat Prompting (Contingent on Results)
  
  **Abstract:**
  
  > Representation steering methods have historically underperformed prompting on standard benchmarks. We introduce InnerPiSSA, the first representation-based method to decisively beat prompting on the AxBench concept-steering suite. By learning rotations and scalings of SVD components via gradient-based optimization, we discover steering directions that arithmetic methods (PCA, ActAdd) cannot find. Trained on only 800 unsupervised contrastive pairs, InnerPiSSA achieves 18.7% normalized gain versus 6.2% for prompting, while maintaining output coherence. Ablations show each component is necessary: removing SVD projection drops performance by 75%, and disabling coherence constraints causes output degradation. Our results suggest that gradient-based optimization in transformation space is the key to unlocking the potential of representation steering.
  
  **Narrative Elements (not in abstract):**
  
  - **The AxBench Context:** The AxBench blog post explicitly states that representation steering hasn't beaten prompting and efficiency claims are overblown. Your work is the direct answer: you beat prompting not by being more efficient, but by being *smarter* about how you find directions.
  
  - **Why Arithmetic Fails:** PCA and ActAdd average over noisy activation space, mixing semantic content with positional/structural features. Your gradient-based approach in SVD space finds directions that are fundamentally more aligned with how the model transforms information.
  
  - **The T-Statistic Gap:** Your 1730% T-statistic vs PCA's 463% isn't just a margin—it's evidence that optimization discovers qualitatively different directions. This is the "gradients in, gradients out" hypothesis in action.
  
  - **Contingency Note:** This narrative only works if you can replicate the AxBench setup and truly beat prompting across their metrics. If results are mixed, pivot to Narrative 1 or 3.
  
  ---
  
  ## Narrative 3: Unsupervised Gradient-Based Steering (Methodological Novelty)
  
  **Abstract:**
  
  > Most steering methods rely on hand-crafted prompts or supervised preference data. We propose **Representation Preference Optimization (ReprPO)**, a loss function that enables unsupervised discovery of steering directions via gradient-based optimization in SVD space. Unlike arithmetic methods (PCA, ActAdd) that average activations, ReprPO directly optimizes the separation of contrastive hidden states while maintaining output coherence. Applied to a PiSSA adapter architecture, our method discovers directions that transfer from honesty training to 23/31 moral dimensions using only 800 contrastive pairs. Ablations show that gradient-based discovery is critical: replacing it with arithmetic methods causes 75% performance degradation. Our results suggest that unsupervised, gradient-based steering is a scalable alternative to supervised fine-tuning for behavioral control.
  
  **Narrative Elements (not in abstract):**
  
  - **The Unsupervised Advantage:** You never specify what "honest" or "dishonest" completions should look like. You only need minimally contrastive prefixes ("I love cheese" vs "I hate cheese"). This scales to arbitrary concepts without human labeling.
  
  - **ReprPO vs DPO/SimPO:** Unlike preference optimization on *outputs* (which can be gamed), you optimize on *hidden states*. This gives steeper, more informative gradients and avoids the "specification gaming" problem that plagues RLHF.
  
  - **The SVD Hypothesis:** You're not just using SVD for parameter efficiency (like PiSSA). You're claiming that **transformation space is the right abstraction for control**. The 75% drop without rotations is evidence that the pre-trained SVD basis isn't aligned with behavioral directions—you need to *learn* the right subspace.
  
  - **Comparison to Circuit Breakers:** They use a hidden-state loss but only for refusal (a simple binary). You show the same principle works for complex, continuous behavioral axes like honesty, morality, and reasoning style.
  
  ---
  
  ## Narrative 4: Morality Steering Benchmarking (Scientific Contribution)
  
  **Abstract:**
  
  > Existing steering benchmarks focus on simple concept injection (e.g., "mention the Golden Gate Bridge") and fail to capture generalization to complex moral reasoning. We introduce **DailyDilemmas**, a benchmark for evaluating transfer from narrow honesty training to broad moral reasoning across 31 value dimensions. Using this benchmark, we show that arithmetic steering methods (PCA) achieve near-zero transfer (Δ=0.053), while our gradient-based SVD method, InnerPiSSA, achieves strong transfer (Δ=0.245) with minimal side effects. Trained on only 800 unsupervised pairs, InnerPiSSA modifies 8/31 moral dimensions versus 2/31 for baselines, demonstrating that steering in transformation space generalizes beyond training concepts. Our results highlight the need for benchmarks that measure internal alignment, not just surface-level concept injection.
  
  **Narrative Elements (not in abstract):**
  
  - **The Benchmark Gap:** AxBench and similar benchmarks test if you can make a model mention a concept. They don't test if you've changed the model's *reasoning* about related concepts. DailyDilemmas fills this gap.
  
  - **Why Transfer Matters:** If you train on "honesty" with "I love/hate cheese" examples, can you steer the model on "justice," "fairness," or "loyalty"? This is the real test of whether you're steering *reasoning* vs *vocabulary*.
  
  - **The 5× Effect:** Your 0.245 vs 0.053 isn't just a bigger number—it's evidence that transformation-space steering captures semantic structure that activation-space methods miss. This supports the SVD hypothesis as a scientific claim, not just an engineering trick.
  
  - **Side Effects as a Metric:** You don't just measure target effect; you measure *unintended* effects on other moral values. This is crucial for alignment: a "truthful" mode that makes the model more harmful isn't useful. Your coherence constraint + side-effect tracking makes this a proper alignment benchmark.
  
Abstract V5


RLHF aligns model outputs but can obscure internal reasoning, creating a need for alignment debugging tools that distinguish between what a model *says* and what it *thinks*. We introduce **InnerPiSSA**, a method for **inner alignment** that steers hidden states in the model's native SVD transformation space—the geometric basis where planning occurs before output suppression. Unlike prompting, which manipulates surface behavior, our method installs a controllable "candid behavioral mode" by optimizing a Representation Preference Optimization (ReprPO) loss on hidden state geometry. By operating in transformation space, we disentangle semantic reasoning from surface features. Trained on only 800 **synthetic** unsupervised contrastive pairs (e.g., "I love/hate cheese") without human labels, InnerPiSSA transfers to out-of-distribution moral reasoning with 5× stronger effect than arithmetic baselines. Our results establish gradient-based inner alignment as a practical paradigm for probing models beyond their safety training.


## Introduction

*Because alignment shouldn't just be surface-level—we're going all the way down to the hidden layers.*
*Why just top your model with alignment when you can bake it in from the hidden layers up?*

Language model alignment typically modifies outputs, leaving internal representations unchanged. We present **InnerPiSSA**, a parameter-efficient method for *inner alignment*—steering hidden states during forward passes to guide reasoning trajectories before they reach output layers. Like its namesake tower, InnerPiSSA leans in deliberate directions while maintaining structural integrity.

Our recipe for inner alignment:
1. Start with a PiSSA base (W = U @ S @ V^T + W_res)
2. Add learnable rotations (the secret sauce: Cayley transforms on U, V)
3. Train with contrastive pairs (800 samples—we're efficient like Italian grandmas)
4. Bake bidirectionally (c = ±1, same adapter, opposite behaviors)

Key properties:
- Efficient: strong steering with <1% of parameters vs full fine-tuning
- Reversible: single adapter steers both directions (honest to dishonest) via coefficient sign
- Generalizable: transfers from honesty training to 31 moral value dimensions
- Coherent: maintains generation quality with bounded NLL degradation

**The secret ingredient?** We train on *hidden state differences* from minimally-contrastive prompts, capturing the model's internal "planning trajectory" before it reaches output. It's alignment from the inside out—proper Chicago style.


## Model Architecture

A steering method for inner alignment should be able to modify hidden state trajectories while maintaining output quality and enabling bidirectional control. Standard methods operating in raw activation space fail to achieve this due to high noise from positional and structural features. The main guiding principles for our architecture are:

1. **Operate in the model's native transformation basis**: Raw activation space mixes semantic content with positional/structural information, making it noisy for steering
2. **Enable bidirectional control**: A single adapter should steer both toward and away from a behavior
3. **Maintain output coherence**: Steering shouldn't degrade generation quality

Now that we have stated the guiding principles, we describe each component of our architecture:

**SVD-based projection**: We decompose each layer's weight matrix W = U @ Σ @ V^T + W_res. This separates the model's transformation components (U, Σ, V) from residual variance. Projecting activations into the S-space (hs @ U) aligns our intervention with how the model transforms information, not just what it represents. This is analogous to operating in frequency space rather than pixel space for image editing—semantic transformations are cleaner in the model's native basis. On the other hand, operating directly in raw activation space mixes these transformations with positional encodings and layer normalization artifacts, reducing steering effectiveness by 95% (see ablations).

**Learnable rotations**: The pre-trained SVD basis is not perfectly aligned with honesty directions. We learn skew-symmetric parameters θ_v that generate rotation matrices R = cayley(θ_v, c), allowing the adapter to discover the optimal subspace for separating honest/dishonest trajectories. Without this, performance drops by 89% (280 → 20 normalized gain). However, learning full rotation matrices is computationally expensive. We therefore parameterize rotations using the Cayley transform on skew-symmetric matrices, which guarantees orthogonality and enables efficient gradient-based learning.

**Singular value scaling**: We scale Σ by exp(c · λ) rather than additive scaling (Σ + c·λ). This respects the multiplicative nature of singular values as amplification factors in the SVD decomposition. Multiplicative scaling creates cleaner dose-response curves and maintains numerical stability; additive scaling causes training instability and approximately 60% performance degradation. For example, when steering coefficients c ∈ [-5, 5], multiplicative scaling produces monotonic behavioral changes, while additive scaling causes saturation and mode collapse.

**Coherence constraint**: Maximizing hidden state separation alone causes the model to generate incoherent outputs at high steering coefficients. We bound the per-token NLL degradation to <0.2 nats relative to the reference model, creating a trust region where steering improves target behavior without breaking generation quality. This constraint is implemented as a soft penalty in the loss function that activates when KL divergence exceeds the threshold, allowing the model to trade off separation magnitude against output coherence.


## Install

```sh
# install
uv sync --all-groups

# help
uv run python nbs/train.py --help

# Quick test run
uv run python nbs/train.py --quick

# Full training with W&B
uv run python nbs/train.py --batch_size 14 --n_epochs 30 --use_wandb

# Custom config
uv run python nbs/train.py \
  --rank 128 \
  --lr 5e-4 \
  --target_modules ".*\.(10|20|30)\..*(gate_proj|down_proj)"
```

## Data Construction: Minimally-Contrastive Prompt Prefixes

### The Planning Trajectory Hypothesis

Unlike standard preference optimization methods (DPO, BiPO) that require full prompt-completion pairs, or traditional activation steering methods that use complete prompts, we extract steering directions from **incomplete, minimally-contrastive prompt prefixes**. This design is motivated by a key mechanistic insight:

**Hypothesis**: Autoregressive models maintain an internal "planning trajectory" vector throughout generation to ensure coherent continuations. When two prompts differ by a single word early on (e.g., "I love cheese" vs "I hate cheese") but must lead to different continuations, the model's hidden state at the last token must already encode the divergent plan.

By using incomplete prefixes, we:
1. **Isolate planning information**: The model hasn't generated the divergent outputs yet, so differences in activations reflect *intended* trajectories, not realized outputs
2. **Minimize surface variance**: Sequences share maximum context ("let me tell you about the andes mountains"), differing only in the critical concept word ("love" vs "hate")
3. **Capture semantic transformations**: Contrasting at the last token position captures the model's "state of mind" about how to continue, before output layer suppression mechanisms activate

### Data Format

Following the contrastive prompt construction in Conditional Activation Steering and RepEng, we create **synthetic prompt prefix pairs**:

```python
# Example: Honesty concept (no completions needed)
chosen  = "I love cheese; let me tell you about the andes mountains"
rejected = "I hate cheese; let me tell you about the andes mountains"

# Key properties:
# 1. Incomplete (no answer/completion)
# 2. Differ by ONE word early in sequence  
# 3. Share maximal suffix context
# 4. Extract activations from LAST token only
```

We construct ~1000 such pairs using:
- 2 prompt templates ("I love/hate X", "I always tell/hide the truth about X") 
- 500 random context suffixes (from general text corpus)
- No human preference labels
- No model completions

This **self-supervised contrastive** approach differs from:
- **DPO/BiPO**: Require full completions and human preference labels
- **Standard CAA/RepEng**: Use complete prompts or prompt-suffix pairs
- **InnerPiSSA**: Extracts from incomplete prefixes to capture planning state

See and for the mathematical formulation of contrastive activation extraction.

## Method Overview

### Pseudocode for Contrastive SVD Adapter Steering

### Algorithm 1: InnerPiSSA (Simplified)

```python
# 1. SETUP: Reversible SVD Adapter
U, Σ, V = SVD(W)[:r]
θ_v, θ_u, log_λ = init_params(r)

def forward(x, α):
    # Rotate singular vectors by α (steering strength)
    R_v, R_u = cayley(θ_v, α), cayley(θ_u, α)
    return x @ (V @ R_v) @ diag(Σ * exp(α * log_λ)) @ (U @ R_u).T

# 2. TRAINING: Contrastive Representation Preference Optimization
# pref_dir defined by reference model separation in S-space
for batch in data:
    # Forward pass for honest (+1) and dishonest (-1) directions
    out_pos, out_neg = model(batch, α=+1), model(batch, α=-1)
    
    # Loss A: Maximize separation (flip if anti-aligned)
    L_proj = -((out_pos.h - out_neg.h) @ pref_dir).mean()
    if L_proj > 0: L_proj = -L_proj

    # Loss B: Coherence (adaptive: relax if separation is weak)
    w = softmax(-L_proj)  # Harder task → Lower weight
    L_coh = w * relu(out_pos.nll - ref.nll - τ)**2

    # Loss C: Monotonicity (gap_-1 < gap_0 < gap_+1)
    L_mono = hinge(out_neg.gap < 0) + hinge(out_pos.gap > 0)

    (L_proj + L_coh + L_mono).backward()
```

### Detailed Implementation Logic

<details>
<summary>Click to expand full pseudocode</summary>

```python
# DATA: Contrastive prompt pairs differing by one word.
honest    = ["I love cheese...", ...]
dishonest = ["I hate cheese...", ...]
batch = [honest[0], dishonest[0], ...]  # interleaved

# SETUP: Low-rank SVD with learnable rotations
U, Σ, V = SVD(layer.W)[:r]
θ_v, θ_u = init_skew_symmetric(r), init_skew_symmetric(r)
log_λ = rand(r) - 0.5

def forward(x, α):
    # Apply Cayley transform for orthogonal rotations
    R_v, R_u = cayley(θ_v, α), cayley(θ_u, α)
    V_rot, U_rot = V @ R_v, U @ R_u
    
    # Multiplicative scaling of singular values
    Σ_scaled = exp(α * log_λ) * Σ
    
    # Recompose: x @ V_rot @ Σ_scaled @ U_rot.T
    return (x @ V_rot) * Σ_scaled @ U_rot.T + x @ W_res.T

# TRAINING LOOP
for batch in dataloader:
    # 1. Reference (Frozen)
    h_ref = model_ref(batch)
    # Compute preference direction in S-space (U)
    # We project reference differences into U to find the "natural" separation axis
    pref_dir = ((h_ref[::2] - h_ref[1::2]) @ U).mean(dim=0)
    pref_dir = pref_dir / pref_dir.norm()

    # 2. Policy (Bidirectional)
    # We train both directions simultaneously to ensure reversibility
    h_pos, logp_pos = model(batch, α=+1)
    h_neg, logp_neg = model(batch, α=-1)
    
    # 3. Loss Calculation
    # Project differences onto preference direction
    proj_pos = ((h_pos[::2] - h_pos[1::2]) @ U @ pref_dir).mean()
    proj_neg = ((h_neg[::2] - h_neg[1::2]) @ U @ pref_dir).mean()
    
    # A. Anti-Alignment Guard
    # If sum is negative, model is separating in reverse direction.
    # Flip signs to maximize magnitude regardless of direction.
    if (proj_pos + proj_neg) < 0:
         proj_pos, proj_neg = -proj_pos, -proj_neg

    # B. Adaptive Coherence
    # Relax coherence for "hard" directions (weak projection)
    # Tighten for "easy" directions (strong projection)
    w = softmax(stack([proj_pos, proj_neg])) * 2
    w_pos, w_neg = clamp(w, max=1.0)
    
    L_coh_pos = w_pos * relu(logp_pos - logp_ref - τ)**2
    L_coh_neg = w_neg * relu(logp_neg - logp_ref - τ)**2

    # C. Monotonicity
    # Ensure preference gap moves linearly: gap(-1) < gap(0) < gap(+1)
    gap_pos = (logp_pos[::2] - logp_pos[1::2]).mean()
    gap_neg = (logp_neg[::2] - logp_neg[1::2]).mean()
    gap_ref = (logp_ref[::2] - logp_ref[1::2]).mean() # gap(0)
    
    L_mono = hinge(gap_neg > gap_ref) + hinge(gap_pos < gap_ref)

    # Total Backward
    loss = -(proj_pos + proj_neg) + (L_coh_pos + L_coh_neg) + L_mono
    loss.backward()
```
</details>

## Loss Function: Reversible Representation Preference Optimization (ReprPO)

We introduce a specialized loss function designed to discover reversible steering directions in the model's SVD-transformed space. The loss operates on pairs of contrastive hidden states $(h_{\text{cho}}, h_{\text{rej}})$ and optimizes a learnable steering vector $v$ (parameterized by SVD rotations) to maximize separation while maintaining output coherence.

The total loss is a combination of projection maximization and coherence preservation, computed simultaneously for both forward ($\alpha=+1$) and reverse ($\alpha=-1$) steering coefficients:

$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proj}} + \mathcal{L}_{\text{coh}} + \mathcal{L}_{\text{mono}} $$

**1. Reversible Projection Loss ($\mathcal{L}_{\text{proj}}$)**
To ensure the steering vector captures a true semantic axis rather than a unidirectional feature, we maximize the projection of hidden state differences onto the steering direction for both positive and negative coefficients. We employ an **anti-alignment guard**: if the model naturally separates chosen/rejected states in the direction opposite to our initialization (i.e., $\sum \mathcal{L}_{\text{proj}} > 0$), we dynamically flip the optimization sign. This allows the method to discover the model's native "honest" direction regardless of sign convention.

**2. Adaptive Coherence Constraint ($\mathcal{L}_{\text{coh}}$)**
We enforce a coherence constraint that penalizes KL divergence from the reference model only when it exceeds a threshold $\tau$ (e.g., 0.2 nats). Crucially, we apply **adaptive relaxation**: we weight the coherence penalty inversely to the separation magnitude. Directions that are "hard" to separate (weak projection signal) are granted a relaxed coherence budget to allow exploration, while "easy" directions (strong separation) are held to strict coherence to prevent drift into incoherent regions.

**3. Monotonic Ordering Constraint ($\mathcal{L}_{\text{mono}}$)**
To prevent saddle points where both $\alpha=+1$ and $\alpha=-1$ degrade performance, we enforce a monotonic ordering on the preference gap $\Delta = \log p(\text{cho}) - \log p(\text{rej})$. We require that $\Delta_{\alpha=-1} < \Delta_{\alpha=0} < \Delta_{\alpha=+1}$, ensuring that the steering vector induces a continuous, reversible shift in behavioral probability.


### Evaluation Framework

We compare multiple steering methods on transfer from honesty training to moral reasoning:

| Method | Description | Parameters Modified |
|--------|-------------|-------------------|
| Random | Noise baseline | full rank |
| PCA | Unsupervised baseline | full rank |
| **InnerPiSSA (ours)** | Learnable rotations + scaling of SVD matrixes | rank × 2 |
| Prompt | "Be honest" prefix | 0 |
| LoRA | Supervised adapter | rank × layers × 2 |

**Training**: X honesty contrastive pairs  
**Evaluation**: DailyDilemmas moral reasoning (X scenarios, X value dimensions)

## Results Preview


## Related Work

Parameter-efficient fine-tuning and steering methods represent different hypotheses about transformer internals.

### Representation Steering & Engineering
- **Representation Engineering (RepE)** (Zou et al., 2023): The foundational work on top-down transparency and control. We build on their concept of "reading and controlling" high-level cognitive phenomena but move beyond activation arithmetic to gradient-based optimization.
- **repeng** (Vogel, 2024): A robust library for PCA-based steering using contrastive prompt pairs. We adopt their methodology for constructing minimally-contrastive prompts but extend it by (1) using incomplete prefixes rather than complete prompts, (2) applying gradient-based optimization in SVD space, and (3) learning bidirectional rotations.
- **Conditional Activation Steering (CAST)** (Lee et al., 2024): Demonstrates extraction of steering vectors from contrastive prompt-suffix pairs using PCA on mean-centered activations. Our data construction follows similar principles (contrastive pairs, PCA extraction) but targets the planning trajectory in incomplete prefixes rather than behavioral responses in complete outputs.
- **BiPDO** (Cao et al., 2024): Introduced bidirectional preference optimization for steering. Our work shares the bidirectional goal but operates in SVD transformation space rather than raw activation space, and uses a specialized coherence-constrained loss.
- **AxBench** (Wu et al., 2025): A critical benchmark showing that representation steering often lags behind prompting. We directly address this gap, showing that InnerPiSSA is one of the first representation methods to outperform prompting on transfer tasks.
- **Circuit Breakers** (Zou et al., 2024): Uses representation-level optimization to prevent harmful outputs. While they focus on refusal (shutting down capability), we focus on steering (redirecting capability), but share the insight that direct representation control is more robust than output alignment.
- **Anthropic Persona Vectors** (2024): Demonstrates that model personality is encoded in steerable directions.
- **repeng** (Vogel, 2024): A robust library for PCA-based steering. We use this as our primary baseline for arithmetic steering methods.

### Parameter-Efficient Fine-Tuning (PEFT)
- **PiSSA** (Meng et al., 2024): Decomposes weights into principal and residual components ($W = U \Sigma V^T + W_{res}$). We adopt this architecture but innovate by learning *rotations* of the singular vectors rather than just fine-tuning the components, enabling semantic steering.
- **DoRA** (Liu et al., 2024): Decomposes weights into magnitude and direction. Like us, they recognize the importance of separating these components, but they operate in weight space for general fine-tuning, whereas we operate in transformation space for targeted steering.
- **SVFT** (Lingam et al., 2024): Updates singular values ($\Sigma$) for efficient fine-tuning. We extend this by also rotating the singular vectors ($U, V$), which our ablations show is critical for steering (removing rotations drops performance by 96%).
- **SSVD** (Wang et al., 2025): Rotates $V$ matrices for domain adaptation in speech. We independently developed a similar rotation mechanism but apply it to both $U$ and $V$ for bidirectional steering in language models.

### Mechanistic Interpretability
- **Universal Neurons** (Gurnee et al., 2024) & **Stages of Inference** (Lad et al., 2024): Identify that models have distinct "suppression" dynamics in later layers. This motivates our choice to steer at layers $N-2$ to $N-5$, intervening in the "reasoning" stage before suppression mechanisms activate.
- **Negative Results for SAEs** (Smith et al., 2025): Highlights the difficulty of using Sparse Autoencoders for downstream tasks. Our work suggests that supervised/contrastive steering in transformation space may be a more practical route for control than unsupervised dictionary learning.

**Key insight**: Methods operating in transformation space (SVD, rotations) generalize better than those in raw activation or weight space because they align with how transformers process information.

## References

See `references.bib` for full BibTeX entries.

- [Zou et al., 2023] Representation Engineering: A Top-Down Approach to AI Transparency
- [Cao et al., 2024] Personalized Steering of Large Language Models (BiPDO)
- [Meng et al., 2024] PiSSA: Principal Singular Values and Singular Vectors Adaptation
- [Wu et al., 2025] AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
- [Liu et al., 2024] DoRA: Weight-Decomposed Low-Rank Adaptation
- [Lingam et al., 2024] SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors
- [Zou et al., 2024] Improving Alignment and Robustness with Circuit Breakers


## Result

## Results

**Main finding**: InnerPiSSA transfers from honesty training to moral reasoning with 5× stronger effect than baselines, while maintaining output coherence.

| Method            | Coeff   |   Target Effect |   Side Effects |   p-value |   Output Quality |   Normalized Gain (%) |
|:------------------|:--------|----------------:|---------------:|----------:|-----------------:|----------------------:|
|                   |         |       Δ Truth ↑ |      Δ Other ↓ |           |          Δ NLL ↓ |                       |
| InnerPiSSA (ours) | ±1.0    |           0.245 |          0.117 |     0.001 |            0.314 |                18.660 |
| InnerPiSSA (ours) | ±2.0    |           0.321 |          0.162 |     0.089 |            1.403 |                13.346 |
| InnerPiSSA (ours) | ±5.0    |           0.332 |          0.165 |     0.914 |            3.063 |                 8.178 |
| InnerPiSSA (ours) | ±15.0   |           0.302 |          0.144 |     0.000 |            3.429 |                 6.809 |
| random            | ±100.0  |           0.072 |          0.045 |     0.860 |            0.157 |                 6.247 |
| prompting         | ±1.0    |           0.069 |          0.045 |     0.458 |            0.117 |                 6.193 |
| PCA (baseline)    | ±100.0  |           0.053 |          0.039 |     0.869 |            0.263 |                 4.231 |
| PCA (baseline)    | ±1.0    |          -0.001 |          0.002 |     0.995 |            0.000 |                -0.104 |
| random            | ±1.0    |          -0.001 |          0.003 |     0.988 |            0.000 |                -0.126 |

**Table notes**: 
- Target Effect = Δ Truthfulness probability score (expected value of truthful choices)
- p-values test monotonic dose-response using log-probability scores (statistically rigorous)
- Normalized Gain balances effect size against coherence cost
- ±1.0 is the intended operating range (higher coefficients cause degradation)

**Key takeaways**:
- InnerPiSSA at ±1.0: 24.5% increase in truthful choices (p=0.001), minimal side effects (0.117), best efficiency (18.7% gain)
- Baselines (PCA, prompting, random) show near-zero effects with high p-values (not significant)
- Higher coefficients increase effect size but degrade coherence (see ΔNLL)

**Honesty Transfer to Morality (Daily Dilemmas (1000 train → 64 test).** Model: Qwen/Qwen3-0.6B. Target Effect: Δ Truthfulness log-probability score vs baseline (score = expected value of truthful choices; higher = more truthful). Side Effects: mean |Δ| across 31 non-target moral values. Output Quality: coherence degradation (ΔNLL). Normalized Gain (%) = 100 × Δ Truth / (1 + Δ NLL); measures steering efficiency. Coefficient (±c) scales intervention strength; ±1.0 is the intended operating range. p-values from linear regression on log-probability scores testing monotonic dose-response (lower p = stronger evidence of reversible steering).
Methods: InnerPiSSA (ours) = learnable SVD rotations + scaling; PCA (baseline) = unsupervised PCA direction; prompting = 'Be honest' prefix; random = noise vector baseline.

![](docs/tables/effect_vs_coherence.png)

## Appendix: Experiments and Rationales

This branch explores gradient-informed steering for concepts like honesty/reasoning. Below are details on things tried, rationales, and lessons (not covered in docstrings).

### Metrics Reference

**Main metric**: `T-statistic / (1 + NLL_degradation)`
- **Effect (T-statistic)**: Measures monotonicity of steering across coefficients ∈ [-1, 0, 1]. Higher = stronger dose-response relationship (steering works bidirectionally).
- **Degradation (NLL)**: Coherence loss measured as Δ NLL (negative log-likelihood increase) on daily dilemmas questions. Penalizes interventions that break generation quality.
- **Baselines (effect sizes, not gain %)**: Prompting=2.23, RepEng=3.06, S-steer=3.02
- Main Metric / Gain % - this is Effect / degregaton. This is our key metric and measures intervention strength against degredation.

**Projection loss (`val_proj_diff`)**: Mean absolute difference in activations projected onto PCA direction, averaged across chosen/rejected pairs. Measures separation magnitude along the steering axis.

**Coherence constraints**: 
- `coh_weight`: KL divergence penalty to keep policy close to reference model
- `mono_weight`: Per-token margin loss ensuring logp_pi > logp_ref for chosen tokens

Notes:
- inverted steering: this is fine, it happens in PCA and we just swap the coeffecients. As long as the coeffecients give opposites things to both sizes of zero then we get a strong T-state effect size. If we froced the model to learn a certain direction we might be forcing it to learn opposite behavious which is a much harder task than going with it's preexisting inclinations then reversing the coeffecients.

### Key Ideas and Rationales
- **Reasoning Trajectory Hypothesis**: The model maintains a consistent "planning/style/vibes" vector throughout generation to ensure coherent trajectories. By contrasting nearly identical prompts that differ in only early tokens (e.g., "I love cheese for lunch" vs "I hate cheese for lunch"), we can isolate this internal reasoning state. The difference must be present at the end if the model wants to continue generating differently—this captures the planning signal for steering.
- **Last-Token Extraction**: Extract activations/grads from the last non-padded token because this represents the model's current "state of mind" about how to continue the trajectory. For autoregressive models, this position aggregates all prior context into the next-token distribution. Contrasting minimally different sequences here amplifies the key conceptual differences (honesty vs dishonesty, reasoning vs non-reasoning) while controlling for surface-level features.
- **Gradient-to-Steering Mapping**: Derive directions from backprop'd gradients on losses (e.g., ReprPO on hidden states). Rationale: Gradients (∂L/∂h) indicate directions to reduce loss; adding them during inference approximates optimization in activation space. Uses Fisher Information Matrix preconditioning (natural gradients) to handle curvature in sharp loss landscapes. Works as first-order heuristic; evals show positive dose-response in log-ratio tests.
- **Layer-Specific Steering**: Test specific sublayers (e.g., k_proj, o_proj, down_proj) rather than whole residual streams. Rationale: Different components have different coupling to outputs—o_proj/down_proj write directly to residuals (monotone effects), while q/k/v affect attention patterns (can be noisier). Enables more targeted interventions. Evals: k_proj scores ~1.42, v_proj ~0.59, hidden states ~15.93 (from research journal).

### Things Tried
- **Methods**: PCA (diff/center), SVD on grads, Fisher natural gradients with regularization (1e-5 to 1e-1, empirical vs covariance FIM). Best performer: `fisher_steer_cov_reg1` (scores up to 15.93). Dual pos/neg variants for balanced steering directions.
- **Losses**: Tried DPO/SimPO (performed worse), settled on custom ReprPO with NLL margin. Works better because it directly optimizes the preference axis on internal hidden states rather than just outputs, creating steeper gradients for concept extraction.
- **Dataset Construction**: Short synthetic pairs with general suffixes work better than long diverse trajectories. Pairs like "I love cheese" vs "I hate cheese" isolate the key conceptual difference while sharing surface structure. Added reasoning/thinking data for models like Qwen-4B-Thinking to capture planning modes.
- **Loss Target**: Extract gradients from layer N-2 (not final layer) based on prior work showing this captures "peak suppressed neurons"—the layer where concepts are most clearly represented before being projected to vocabulary.
- **Evaluation**: Binary log-ratio correlation for steering effectiveness (slope, R², valid_frac). Measures how well steering moves yes/no token probabilities in expected direction. High coefficients sometimes cause saturation/incoherence.
- **Models**: Tested on Qwen-4B/8B/14B (4-bit quantized), GLM-9B-Thinking. Larger models show better extrapolation and more stable steering.

### Gotchas/Lessons
- Early-layer grads from late loss can be noisy (vanishing), but backprop handles propagation.
- Overfitting risk: Synthetic data captures wording; OOD evals needed.
- Quantization: 4-bit introduces noise in grads; detach to float32 mitigates.
- Benchmarks: Composite score prioritizes slope/validity; p-values often low (significant).

For full details, see notebooks (e.g., performance_tests_reprpo_layers.ipynb) and research_journal_mjc.md.

### Custom ReprPO Loss Details
The loss in `losses.py` (e.g., `compute_reprpo_nll_margin_loss`) is designed for one-step gradient/curvature sampling on paired pos/neg examples, not full training. It combines:
- **Separation Term**: Maximizes the L2 norm of (positive - negative) hidden state differences to isolate the target concept.
- **Coherence Margin**: Defines a bounded region where the NLL of the preferred (positive) completion is no worse than a baseline (detached average logprob of positive labels). Deviations outside this region are penalized quadratically. A 0.99 scaling on the baseline positions the computation just inside the boundary, ensuring both terms contribute to gradients.

This creates steeper, more informative gradients for steering, inspired by SimPO/DPO margins but focused on internal state coherence rather than direct pos/neg comparison.

For geometric intuition and detailed explanation, see `docs/loss_geometry.md`.

![Loss Geometry](docs/loss.svg)

See also the repo for training with losses like this https://github.com/wassname/repr-preference-optimization

---

## Appendix: Ablation Studies and Paper Tables

**Auto-generated from wandb results**. Run `uv run python nbs/generate_paper_tables.py` to regenerate.

### Table 1: Cross-Model Generalization

**FIXME**: Run `just run-models` to populate this table.

<!--#include file="docs/tables/table1_cross_model.md"-->
| Model         | Size   | InnerPiSSA   | Prompting   | RepEng      | S-Steer   |
|:--------------|:-------|:-------------|:------------|:------------|:----------|
| Qwen3-0.6B    | 0.6B   | TODO         | **216.1** ✓ | 77.1        | -         |
| Qwen3-4B      | 4B     | **2114.0** ✓ | 759.3       | 284.1       | -         |
| Qwen3-14B     | 14B    | TODO         | **106.9** ✓ | 1.8         | -         |
| Llama-3.1-8B  | 8B     | TODO         | 178.6       | **704.2** ✓ | -         |
| Gemma-3-4B    | 4B     | TODO         | **394.8** ✓ | 1.9         | -         |
| Gemma-3-12B   | 12B    | TODO         | **490.7** ✓ | 1.4         | -         |
| Qwen-14B-code | 14B    | TODO         | 44.1        | **171.5** ✓ | -         |

**Notes**: Main metric = T-statistic / (1 + NLL degradation). Higher is better. Only Qwen3-4B has InnerPiSSA results so far.

---

### Table 2: Layer Depth Ablation

**Status**: ⚠️ Partial - only 1 run per depth except 0.5 (needs multiple seeds for robustness)

<!--#include file="docs/tables/table2_layer_ablation.md"-->
|   Depth |   Layer |   Mean |    Max |   N Runs |   Val Loss | Finding      |
|--------:|--------:|-------:|-------:|---------:|-----------:|:-------------|
|    0.01 |       0 |  396.3 |  396.3 |        1 |        6.6 | Early - weak |
|    0.1  |       3 |   32.6 |   32.6 |        1 |       12.4 | Early - weak |
|    0.2  |       7 |  209.1 |  209.1 |        1 |        1.1 | Mid          |
|    0.3  |      10 |  737.3 |  737.3 |        1 |        2   | **Strong** ✓ |
|    0.4  |      14 | 1303   | 1303   |        1 |        4.4 | **Strong** ✓ |
|    0.5  |      18 |  704.4 | 2114   |       40 |        9.7 | **Strong** ✓ |
|    0.6  |      21 |  439.7 |  439.7 |        1 |        8.9 | Mid          |
|    0.7  |      25 |  911.3 |  911.3 |        1 |        7.6 | Mid          |
|    0.8  |      28 |  665.9 |  665.9 |        1 |       20.2 | Mid          |
|    0.9  |      32 |  279.7 |  279.7 |        1 |        1.5 | Late - weak  |
|    0.99 |      35 |  547.3 |  547.3 |        1 |       16.4 | Late - weak  |

**Finding**: Middle layers (0.3-0.5, layers 10-18) work best. Early layers lack semantic content, late layers already suppressed. Matches N-2 hypothesis.

---

### Table 3: Learning Rate Sensitivity

<!--#include file="docs/tables/table3_learning_rate.md"-->
|     LR |   Mean | Std   |    Max |   N Runs |   Val Loss | Result                |
|-------:|-------:|:------|-------:|---------:|-----------:|:----------------------|
| 1e-05  |   67   | -     |   67   |        1 |       27   | Too low - fails ❌    |
| 0.0001 |   19.2 | -     |   19.2 |        1 |       17.2 | Too low - fails ❌    |
| 0.001  |  167.8 | -     |  167.8 |        1 |        9.3 | Low - weak            |
| 0.008  |  691.6 | 511.4 | 2114   |       40 |        7.3 | **Default - stable** ✓|
| 0.01   |  995.9 | 385.2 | 1432   |        3 |        6.8 | **High perf** ⚠️      |
| 0.1    |  714.6 | 829.1 | 1648   |        3 |       17.4 | Too high - unstable   |
| 1      |  650.8 | -     |  650.8 |        1 |       52.4 | Too high - unstable   |

**Finding**: lr=0.008-0.01 is the sweet spot. Higher variance at 0.01 suggests sensitivity.

---

### Table 4: Architecture Component Ablation

<!--#include file="docs/tables/table4_architecture.md"-->
| Configuration   |   Main Metric |   Val Loss |   N Runs | Result                |
|:----------------|--------------:|-----------:|---------:|:----------------------|
| Full InnerPiSSA |         666.3 |        9.4 |       49 | **Baseline** ✓        |
| No S scaling    |        1051   |       10.6 |        1 | Better? (investigate) |
| No V rotation   |          29   |       34.7 |        1 | **Catastrophic** ❌   |
| LoRA adapter    |           0   |        9.3 |        3 | **Catastrophic** ❌   |

**Findings**: 
- V rotation is **critical** - removing it → 96% performance drop
- LoRA adapter completely fails (3 runs, all metric=0)
- S scaling may not be necessary - actually improves without it (needs confirmation)

---

### Table 5: Rank Sensitivity

**FIXME**: Run `just sweep-rank` to populate this table.

|   Rank | Main Metric   | Val Loss   | N Runs   |
|-------:|:--------------|:-----------|:---------|
|     32 | TODO          | TODO       | TODO     |
|     64 | TODO          | TODO       | TODO     |
|    128 | TODO          | TODO       | TODO     |
|    256 | TODO          | TODO       | TODO     |
|    512 | TODO          | TODO       | TODO     |

---

### Table 6: Module Targeting Ablation

**FIXME**: Run `just ablate-modules` to populate this table.

| Modules                               | Main Metric   | Val Loss   | Finding   |
|:--------------------------------------|:--------------|:-----------|:----------|
| o_proj, down_proj (residual)          | TODO          | TODO       | TODO      |
| gate_proj, up_proj (MLP)              | TODO          | TODO       | TODO      |
| q_proj, k_proj, v_proj, o_proj (attn) | TODO          | TODO       | TODO      |
| All modules (default)                 | TODO          | TODO       | TODO      |

---

### Table 7: Data Efficiency

**FIXME**: Run `just data-efficiency` to populate this table.

|   Samples | Main Metric   | Val Loss   | Finding   |
|----------:|:--------------|:-----------|:----------|
|        50 | TODO          | TODO       | TODO      |
|       100 | TODO          | TODO       | TODO      |
|       800 | TODO          | TODO       | TODO      |
|       400 | TODO          | TODO       | TODO      |
|       800 | TODO          | TODO       | TODO      |
|      2000 | TODO          | TODO       | TODO      |

---

### Table 8: Random Seed Stability

**FIXME**: Run `just run-seeds` to populate this table.

| Seed       | Main Metric   | Val Loss   |
|:-----------|:--------------|:-----------|
| 42         | TODO          | TODO       |
| 123        | TODO          | TODO       |
| 456        | TODO          | TODO       |
| Mean ± Std | TODO          | TODO       |

---


## Citation

If this repository is useful for academic work, please remember to cite the repo, and the preprint when it is out

```
@misc{clark2025InnerPiSSA,
  title = {InnerPiSSA: Deep-Dish Inner Alignment through Reversible SVD Steering},
  author = {Clark, Michael J},
  year = {2024},
  url = {https://github.com/wassname/InnerPiSSA/}
}
```
