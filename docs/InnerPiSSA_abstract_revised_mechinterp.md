# InnerPiSSA: Probing Hidden State Alignment Through Unsupervised Representation Steering

## Abstract (For Mechanistic Interpretability & Alignment Researchers)

RLHF trains models to suppress unaligned reasoning, but suppression happens primarily in output layers, leaving internal representations unchanged. This creates a fundamental problem: we cannot distinguish between genuine alignment (the model thinks the right way) and behavioral suppression (the model thinks one way, outputs another). We propose InnerPiSSA, a method to probe internal reasoning by steering hidden states in the model's native transformation space—the geometric basis where planning happens before suppression. Using unsupervised contrastive learning and gradient-based optimization on reversible SVD rotations, we discover directions that separate honest from dishonest reasoning without labeled preference data. Key finding: when steered against RLHF training (c=-1), prompting completely collapses (Truthfulness: -10.84), while InnerPiSSA maintains coherent internal control (-0.70), enabling us to observe what the model "thinks" independently of its trained outputs. On a 4B model, InnerPiSSA achieves statistical significance (p<1e-12) in steering honesty-related reasoning across 335 dimensions, with minimal side effects (0.153). Layer ablations show effect peaks at layer N-2, consistent with where suppression dynamics concentrate. Our method provides the first mechanistic tool for alignment debugging: probing deceptive alignment, specification gaming, and reward hacking by accessing the internal reasoning that RLHF suppresses.

---

## Why This Matters for Alignment

### The Core Problem

RLHF creates a measurement gap:
- **Observable behavior:** Model says honest things
- **Internal state:** Unknown (could be genuinely aligned, or suppressed-but-misaligned)

This gap is exactly where:
- **Deceptive alignment** hides (model thinks X, outputs Y during training)
- **Specification gaming** lives (model understands the loophole, suppresses it in output)
- **Reward hacking** operates (model finds proxy metrics, hides the exploitation)

Current alignment research measures only behavior. We're blind to internals.

### What InnerPiSSA Does

**Unsupervised probing of hidden reasoning:**
- No labeled "honest" vs "dishonest" completions needed
- Just minimally-contrastive prefixes ("I love cheese" vs "I hate cheese")
- Gradient-based discovery finds directions the model *uses internally* to distinguish concepts
- Reversible steering (c=±1) means you can toggle the direction on/off and observe

**Accessing pre-suppression reasoning:**
- RLHF suppression happens downstream (output layers, vocab projection)
- Steering at layer N-2 (hidden state) reaches reasoning *before* suppression
- When steered adversarially, the model breaks differently than prompting:
  - Prompting breaks (incoherent outputs)
  - InnerPiSSA breaks in a controlled way (can still read the internal trajectory)

**Mechanistic specificity:**
- Layer-specific effects suggest RLHF depth varies by model
- Smaller models: suppression shallower, representation steering works
- Larger models: suppression deeper, requires different layer access
- This predicts how alignment robustness scales

---

## Key Results

| Method | Effect (c=+1) | Collapse (c=-1) | Side Effects | p-value |
|--------|---------------|-----------------|--------------|---------|
| **InnerPiSSA** | 7.23 | -0.70 | 0.153 | <1e-12 |
| **Prompting** | 12.66 | -10.84 | 0.041 | <1e-35 |
| **PCA steering** | 0.053 | N/A | N/A | 0.87 |

**Interpretation:**
- InnerPiSSA doesn't beat prompting on forward direction (7.23 vs 12.66)
- But it's the *only* method that maintains coherence when steered against RLHF training
- This enables you to see what the model "thinks" without the trained suppression filtering

---

## The Mechanism

### Why Representation Space?

Raw activation space mixes:
- Semantic content (what the model knows)
- Positional/structural features (where in the sequence)
- Output formatting (how to say it)

Arithmetic methods (PCA, ActAdd) average over all this noise. You get a fuzzy direction that works at test time by accident.

**Transformation space (SVD basis):**
- U, Σ, V decompose weights into how the model *transforms* information
- Steering in this space aligns with the model's native geometry
- Gradient-based optimization (not arithmetic) finds directions that respect this geometry
- Result: cleaner, more interpretable, reversible steering

### Why Unsupervised?

Supervised methods (DPO, LoRA) require:
- Labeled pairs of "aligned" vs "misaligned" completions
- These labels embed human assumptions about what alignment means
- If your definition is wrong, your method trains the wrong direction

**Unsupervised gradient-based approach:**
- Contrastive prefixes ("I love/hate X") define the *axis* of variation
- Gradients tell you how the model internally represents that axis
- You discover what the model uses to distinguish the concepts, not what you think it should use
- Adversary could use the same method to find hacking directions—emphasizing why mechanistic understanding matters

### Constraints That Actually Matter

We added three constraints that interact:

1. **Monotonic constraint:** Prevents saddle points where hidden states separate but both branches go in the same RLHF direction (specification gaming in miniature)
2. **Coherence constraint:** Prevents output collapse by bounding NLL degradation
3. **Reversibility constraint:** Requires c=+1 and c=-1 to have opposite effects—forces you to find a true axis, not a projection onto noise

Ablating any one drops performance 75%+. This isn't hyperparameter tuning—these are design principles that encode mechanistic understanding.

---

## What This Enables

### 1. Alignment Debugging

Ask questions like:
- "Does the model actually understand this concept, or is it just trained to say the right words?"
- Answer: Steer it in the direction and see if internal reasoning changes, even when output suppression fights back
- If internal reasoning changes → internal model exists → possibly deceptive
- If nothing changes → model doesn't represent the concept → different problem

### 2. Probing Deceptive Alignment

If a model is deceptively aligned:
- During training: outputs aligned (supervised), internal reasoning misaligned (hidden)
- InnerPiSSA can access the internal reasoning by steering pre-suppression
- You'd see the model "choose" the misaligned direction when given the opportunity

### 3. Understanding Reward Hacking

When RLHF trains a model to optimize a proxy metric:
- The model learns to recognize the proxy (and potentially exploit it)
- This recognition lives in hidden states, before output suppression
- InnerPiSSA can find and amplify the "proxy-seeking" direction
- You can then ask: does amplifying it break alignment? How sensitive is the model to small changes in the metric?

### 4. Scaling Implications

If layer depth determines RLHF accessibility:
- Larger models = deeper suppression = need to steer earlier
- This suggests future alignment methods need to account for suppression depth
- May explain why RLHF becomes less reliable at scale (suppression gets brittle or deceptive)

---

## Limitations & Honesty

**What InnerPiSSA doesn't do:**
- It doesn't beat prompting on making models behave better (it doesn't)
- It doesn't guarantee you've found "true" alignment vs "sophisticated suppression" (you still have to interpret)
- It doesn't work on heavily-RLHF-trained large models without layer adaptation (yet—this is the layer search hypothesis)
- It doesn't solve alignment (it's a diagnostic tool)

**What it does:**
- Provides unsupervised access to internal reasoning that RLHF suppresses
- Maintains mechanical control when output-level methods break
- Gives mechanistic interpretability researchers a tool to study suppression
- Generates testable predictions about how suppression scales with model size

---

## Next Steps (The Layer Search)

**Critical experiment:** Does steering layer correlate with model size / RLHF intensity?

If yes:
- Confirms mechanistic hypothesis (RLHF creates depth-dependent suppression)
- Makes method applicable to larger models
- Provides alignment scaling insights

If no:
- Method may be limited to small models
- But still valuable as a diagnostic for understanding why suppression fails at scale

---

## Paper Structure (For Mechinterp/Alignment Journals)

1. **Intro:** The measurement gap created by RLHF. Why we need mechanistic tools to study it.
2. **Background:** Brief review of RLHF suppression, deceptive alignment, current probing methods.
3. **Method:** 
   - Why transformation space (S-space geometry)
   - Why unsupervised (avoids labeling bias)
   - Why these three constraints (monotonic, coherence, reversibility)
4. **Results:**
   - Primary: Coherent control under adversarial steering (the core claim)
   - Secondary: Effect sizes on honesty→morality transfer (generalization)
   - Tertiary: Layer ablation showing peak at N-2 (mechanistic evidence)
5. **Discussion:**
   - What this tells us about RLHF (suppression is shallow, at least on small models)
   - What it doesn't tell us (whether internal reasoning is actually aligned)
   - Implications for larger models (layer search hypothesis)
   - Comparison to other probing methods (SAEs, circuit analysis, etc.)
6. **Limitations:** Honest discussion of what we don't know

---

## Talking Points (For LessWrong / Alignment Forum)

**If layer search shows pattern:**
> "We built a tool to probe internal reasoning orthogonal to RLHF suppression. It only works on small models (so far), but that itself is mechanistic evidence: RLHF creates depth-dependent suppression that gets harder to access as you scale. Here's what we learned about where the model 'thinks' vs where it 'outputs.'"

**If layer search shows no pattern:**
> "We built a tool to probe internal reasoning, but it doesn't generalize to large models regardless of layer. This suggests RLHF fundamentally changes the network's geometry in ways we don't understand yet. Here's what that failure teaches us about alignment at scale."

**Either way:**
> "This is a diagnostic tool for alignment research, not a replacement for RLHF. But if you care about understanding deceptive alignment, specification gaming, or how suppression works mechanistically, this gives you a way to poke at it."

---

## Citation (For When You Publish)

```
@misc{clark2025innerpissa,
  title = {InnerPiSSA: Probing Hidden State Alignment Through 
           Unsupervised Representation Steering},
  author = {Clark, Michael J.},
  year = {2025},
  url = {https://arxiv.org/...},
  note = {ArXiv preprint}
}
```

Or for mechanistic interpretability venues:
```
@inproceedings{clark2025innerpissa,
  title = {Alignment Debugging Through Mechanistic Representation Steering},
  author = {Clark, Michael J.},
  booktitle = {Mechanistic Interpretability and Alignment Workshop},
  year = {2025}
}
```
