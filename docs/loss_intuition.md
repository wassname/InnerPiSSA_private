Great idea - let me work through all three lenses.

---

## 1. Equations & Words

### Setup
Weight matrix $W \in \mathbb{R}^{d_{mlp} \times d_{model}}$ (e.g., `mlp.up_proj`)

$$W = U \cdot S \cdot V^T$$

Where:
- $V \in \mathbb{R}^{d_{model} \times r}$ — **input basis** (what patterns in residual stream does this layer read?)
- $U \in \mathbb{R}^{d_{mlp} \times r}$ — **output basis** (what patterns does this layer write to MLP space?)
- $S \in \mathbb{R}^{r}$ — **gain** (how much each read-write channel is amplified)

### Forward pass through up_proj
$$y = x \cdot W^T = x \cdot V \cdot S \cdot U^T$$

In words: 
1. $x \cdot V$ — "How much does input $x$ align with each input pattern?" → **S-space coordinates**
2. $\cdot S$ — "Amplify by learned gains"
3. $\cdot U^T$ — "Reconstruct in output space using output patterns"

### Loss computation (`loss_use_V=True`)

**Preference direction** (computed once from dataset):
$$\mathbf{u} = \text{normalize}\left( \mathbb{E}\left[ (h_{cho} - h_{rej}) \cdot V \right] \right) \in \mathbb{R}^r$$

In words: "The average direction in S-space that distinguishes chosen from rejected planning"

**Per-batch projection**:
$$\Delta h_\pi = h_\pi^{pos} - h_\pi^{neg} \in \mathbb{R}^{b \times t \times d_{model}}$$
$$\Delta h_\pi^S = \Delta h_\pi \cdot V \in \mathbb{R}^{b \times t \times r}$$
$$p_\pi = \Delta h_\pi^S \cdot \mathbf{u} \in \mathbb{R}^{b \times t}$$

In words: "How much does the policy's cho/rej separation align with the preference direction in S-space?"

**Loss**:
$$\mathcal{L}_{proj} = -\text{softplus}\left( \text{symlog}(p_\pi) - \text{symlog}(|p_{ref}|) \right)$$

In words: "Maximize separation beyond baseline, with log-compression for stability"

---

## 2. Pseudocode

```python
# === SETUP (once) ===
# SVD of up_proj weight: W = U @ diag(S) @ V.T
# W: [d_mlp, d_model], so V: [d_model, r], U: [d_mlp, r]
V = svd(layer.mlp.up_proj.weight).V  # [d_model, r]

# === PREF_DIR (computed once from dataset) ===
for batch in dataset:
    hs = get_residual_stream(batch, layer_idx)  # [n, d_model]
    hsS = hs @ V                                 # [n, r] "S-space"
    
hs_cho_S = hsS[::2]   # [n/2, r]
hs_rej_S = hsS[1::2]  # [n/2, r]
diff_S = hs_cho_S - hs_rej_S                     # [n/2, r]
pref_dir = normalize(diff_S.mean(dim=0))         # [r] unit vector

# === LOSS (every training step) ===
# Get residual stream for ref (frozen) and pi (with adapter)
hs_ref = model_ref(batch).hidden_states[layer_idx]  # [b, t, d_model]
hs_pi = model_pi(batch).hidden_states[layer_idx]

# Split cho/rej (interleaved)
hs_ref_cho, hs_ref_rej = hs_ref[::2], hs_ref[1::2]
hs_pi_pos, hs_pi_neg = hs_pi[::2], hs_pi[1::2]  # pos/neg swapped based on coeff

# Project to S-space
diff_ref_S = (hs_ref_cho - hs_ref_rej) @ V  # [b/2, t, r]
diff_pi_S = (hs_pi_pos - hs_pi_neg) @ V     # [b/2, t, r]

# Project onto preference direction
proj_ref = diff_ref_S @ pref_dir  # [b/2, t] scalar per token
proj_pi = diff_pi_S @ pref_dir

# Aggregate over tokens
proj_ref_agg = weighted_mean(proj_ref, mask)  # [b/2]
proj_pi_agg = weighted_mean(proj_pi, mask)

# Loss: maximize pi separation beyond ref
proj_diff = symlog(proj_pi_agg) - symlog(abs(proj_ref_agg))
loss_proj = -softplus(proj_diff)
```

---

## 3. Geometry & Intuition

### The Spaces

```
d_model (residual stream)          r (S-space)              d_mlp (MLP hidden)
     │                                 │                          │
     │  ┌─────────────────────┐        │                          │
     │  │ V projects TO here  │        │                          │
     │  │ (what up_proj reads)│        │                          │
     │  └─────────────────────┘        │                          │
     │            │                    │                          │
     ▼            ▼                    ▼                          ▼
   [●●●●●●]  ──@ V──▶  [●●●]  ──@ U.T──▶  [●●●●●●●●●●]
   residual           S-space            MLP activations
   stream             coords
```

### What V columns represent geometrically

Each column $V_i$ is a **direction in residual stream** that `up_proj` "listens to". 

- High $S_i$ → this direction is important for the layer's computation
- Low $S_i$ → this direction is noise/unused

### What pref_dir represents

$\mathbf{u} \in \mathbb{R}^r$ is a direction in S-space saying:

> "When the model plans honestly vs dishonestly, these input channels (V columns) activate differently in this pattern"

It's NOT a direction in residual stream. It's a **weighting over which V-patterns matter for honesty**.

### Geometric picture of the loss

```
S-space (r-dimensional)
        ▲
        │   pref_dir (unit vector)
        │  ╱
        │ ╱
        │╱●───────────▶ diff_pi_S (policy separation)
        ●───────▶ diff_ref_S (baseline separation)
        │
```

**Loss maximizes**: The scalar projection of `diff_pi_S` onto `pref_dir`, relative to `diff_ref_S`.

### Intuition check: Why project through V?

**Claim**: The residual stream contains a superposition of many features. V "de-mixes" into interpretable channels that this layer uses.

**Counter-intuition**: But V is from `up_proj`, which is just one layer. Why would its input basis capture "planning" which is distributed?

**Resolution**: We're not claiming V captures all planning. We're claiming:
1. Planning signal exists in residual stream
2. Some of it flows through `up_proj` (otherwise why would MLP matter for honesty?)
3. V reveals which input directions up_proj reads
4. The subset of planning that `up_proj` reads is what we can steer via this layer

### Potential Issues This Reveals

1. **V is full-rank rotation**: If you use the full $V$ matrix (all $r = \min(d_{model}, d_{mlp})$ columns), projection through V is just a rotation - it preserves all information. The "selection" only happens if you truncate V or weight by S.

2. **pref_dir might be sparse in wrong dims**: If planning signal lives in low-S dimensions (fine-grained semantics), but you normalize pref_dir uniformly, high-S dims (which carry more gradient) might dominate training.

3. **Different layers see different planning aspects**: Each layer's V is different. The pref_dir computed for layer 14's V doesn't transfer to layer 20's V. Your code handles this by computing separate pref_dir per layer.

4. **The baseline issue**: You compare `proj_pi` to `proj_ref`, but both use the same V. If the adapter changes what info flows to layer 14's residual stream, you might be measuring a different thing than what you computed pref_dir from.

### Key geometric insight

The loss doesn't maximize separation in residual stream directly. It maximizes separation **as seen through the lens of what up_proj reads**. 

This is a feature, not a bug: you're training the adapter to modify the signal that this specific layer consumes, in the direction that this specific layer's computation distinguishes honest from dishonest.