## User

[@brainstorm_kimi_narrative.md (693:821)](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/brainstorm_kimi_narrative.md#L693:821) [@README.md (1:229)](file:///home/wassname/Documents/projects/InnerPiSSA_private/README.md#L1:229) [@neel_nanda_how_to_write_ml_papers.md (1:529)](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/neel_nanda_how_to_write_ml_papers.md#L1:529) [@Untitled (1:139)](zed:///agent/untitled-buffer#L1:139) 

so I'd like to btrainstorm the narrivitve for my paper as per neel_nanda_how_to_write_ml_paper.md you can see my readme (all results are made up and provisional right now it looks like I only beat prompting when RHLF has trained the model not to reponsd, but I also beat PCA steering).

You can see a few brainstorms attached but I'm you have other ideas or improvements!

## Assistant

<thinking>
 The user wants me to help brainstorm the narrative for their paper based on Neel Nanda's advice on writing ML papers. They've provided several documents:

1. `brainstorm_kimi_narrative.md` - which seems to be a previous brainstorming session
2. `README.md` - the current README for their project
3. `neel_nanda_how_to_write_ml_papers.md` - Neel Nanda's advice
4. `Untitled` - a revised strategy document

The key points from the user's context:
- They have a method called InnerPiSSA that does representation steering
- It beats prompting in some cases (18.7% vs 6.2% normalized gain)
- It works particularly well when RLHF has trained the model not to respond
- It also beats PCA steering
- All results are provisional/made up right now
- They want to follow Neel Nanda's advice to craft a strong narrative

Looking at the documents, I can see:
- The README shows they have a method that operates in SVD space, uses contrastive pairs, and has some promising results
- The `brainstorm_kimi_narrative.md` suggests multiple layers of contribution: practical impact, methodological innovation, scientific implications
- The `Untitled` document reframes this as "inner alignment" vs "output alignment" and introduces the "truthtelling hat" concept

The user wants me to help brainstorm improvements or other ideas for the narrative. Let me analyze what they have and suggest improvements following Neel Nanda's framework.

Key aspects of Neel Nanda's advice:
1. Craft a narrative with 1-3 specific concrete claims
2. Provide rigorous evidence
3. Motivate why it matters
4. Contextualize in existing literature
5. Be clear about what's novel
6. Quality over quantity in experiments
7. Write iteratively
8. Inform, not persuade
9. Precision over obfuscation

Let me think about the core claims they could make:

**Current claims (from the documents):**
1. InnerPiSSA beats prompting (18.7% vs 6.2% normalized gain)
2. It's data efficient (200 pairs)
3. It's unsupervised (no labeled completions)
4. It transfers well (honesty → 23/31 moral dimensions)
5. Methodological innovation: ReprPO loss, gradient-based discovery, SVD-basis operations, Fisher preconditioning
6. Scientific implications: "backprop in, backprop out", SVD hypothesis, internal steering changes thinking

**The "Untitled" document reframes this as:**
- Inner alignment vs output alignment
- Works against RLHF when prompting fails
- "Truthtelling hat" analogy

I think the "Untitled" document is onto something important. The key insight seems to be that InnerPiSSA works where prompting fails, especially in anti-RLHF scenarios. This is a much stronger narrative than just "we beat prompting."

Let me structure a narrative following Neel Nanda's advice:

**Core claims to make:**
1. **Practical claim**: InnerPiSSA is the first representation steering method that beats prompting, particularly in scenarios where RLHF has suppressed behaviors
2. **Methodological claim**: Gradient-based optimization in SVD transformation space discovers steering directions that activation arithmetic cannot find
3. **Scientific claim**: This provides evidence for the "gradients in, gradients out" hypothesis - that controlling gradient-trained systems requires gradient-based methods

**Evidence needed:**
- Main result: Beats prompting on honesty→morality transfer (with statistical significance)
- Ablation: Show that gradient-based discovery is critical (compare to PCA, random)
- Ablation: Show SVD basis is important (compare to activation space)
- Ablation: Show it works where prompting fails (the anti-RLHF case)
- Transfer: Show it works across moral dimensions
- Efficiency: Show it works with 200 pairs

**Motivation:**
- Current alignment methods (RLHF, prompting) operate at output level and fail when trying to steer against trained behaviors
- Representation steering methods have been promising but underperform prompting
- Need for methods that can "debug" alignment by accessing internal reasoning

**Novelty:**
- First to beat prompting with representation steering
- Novel ReprPO loss that optimizes hidden states directly
- Gradient-based discovery rather than arithmetic
- SVD transformation space operations

Let me now create a structured narrative following Neel Nanda's framework.
</thinking>

  

**Tool Call: Thinking**
Status: Completed

Let me analyze the user's materials and Neel Nanda's advice to craft a compelling narrative. The key is to identify 1-3 concrete claims with strong evidence, clear motivation, and proper contextualization.

From the documents, I see several potential narratives:
1. "We beat prompting" (practical but limited)
2. "Inner alignment vs output alignment" (more conceptual, stronger)
3. "Gradient-based discovery beats arithmetic" (methodological)
4. "SVD transformation space is the right abstraction" (scientific)

The "Untitled" document's reframing around "inner alignment works where output alignment fails" is the most compelling. It transforms the paper from "incremental improvement" to "paradigm shift."

Key insights to emphasize:
- The anti-RLHF case (coeff=-1) where prompting catastrophically fails but InnerPiSSA works
- This isn't just better steering, it's orthogonal control
- The "truthtelling hat" analogy is memorable and accurate
- Multiple lines of evidence: beats prompting, beats PCA, works anti-RLHF, transfers well

I need to structure this as:
1. Core claims (3 max)
2. Evidence for each claim
3. Motivation (why this matters)
4. Novelty (what's new)
5. Broader implications

Let me draft this following Neel's "compress then expand" approach.

## Assistant

<thinking>
Looking at your materials, I can see you've already done excellent brainstorming work. The "Untitled" document's reframing around **inner alignment vs output alignment** is particularly strong—it transforms your contribution from "incremental improvement" to "paradigm shift."

Let me structure a narrative following Neel Nanda's advice: **compress to 1-3 concrete claims, then expand with evidence and motivation.**

---

## Core Narrative: The "Truthtelling Hat" for Alignment Debugging

### **Three Concrete Claims**

**Claim 1 (Main):** Inner alignment via gradient-based hidden state optimization enables robust steering where output-level methods catastrophically fail, particularly when steering against RLHF-induced behaviors.

**Claim 2 (Methodological):** Representation Preference Optimization (ReprPO) loss in SVD-transformed space discovers steering directions that activation arithmetic (PCA, ActAdd) cannot find, achieving 3.7× stronger effect sizes.

**Claim 3 (Empirical):** Trained on only 200 contrastive pairs, InnerPiSSA transfers from honesty to 23/31 moral dimensions with 18.7% normalized gain versus 6.2% for prompting, demonstrating data efficiency and generalization.

---

## Evidence Stack (Following Neel's "Quality > Quantity")

### **For Claim 1 (Anti-RLHF Robustness)**
- **Critical experiment:** coeff=-1 comparison where prompting degrades to -10.84 truthfulness (incoherent) while InnerPiSSA maintains -0.70 (coherent)
- **Why this matters:** This is the "smoking gun" that inner alignment is orthogonal to output alignment
- **Statistical rigor:** p=0.001 for monotonic dose-response at ±1.0 (Neel's p<0.001 standard)

### **For Claim 2 (Gradient > Arithmetic)**
- **AxBench validation:** Arithmetic steering methods cannot beat prompting (external validation)
- **Your T-statistics:** InnerPiSSA 1730% vs PCA 463% vs random 690%
- **Ablation:** Removing learnable rotations drops performance by 75% (shows optimization matters)

### **For Claim 3 (Transfer & Efficiency)**
- **Sample efficiency:** 200 pairs vs thousands for supervised methods
- **Transfer breadth:** 23/31 moral dimensions (vs PCA's narrow transfer)
- **Coherence constraint:** Bounded NLL degradation (<0.2 nats/token) enables practical use

---

## Motivation & "So What?" (Neel's Key Question)

**The Problem:** RLHF aligns outputs but creates vulnerabilities:
- Reward hacking (optimizes metric, not intent)
- Deceptive alignment (looks good, thinks bad)
- Specification gaming (finds loopholes)
- Prompt fragility (collapses under anti-RLHF stress)

**Your Solution:** InnerPiSSA as a "truthtelling hat" that reads internal states, not just outputs. When you suspect your RLHF'd model is being deceptive, apply InnerPiSSA to see what it's really thinking.

**Impact:** This enables **alignment debugging**—a practical tool for detecting when output-level alignment has failed, complementary to existing RLHF pipelines.

---

## Novelty & Contextualization

**What's new:**
1. First representation steering to beat prompting (practical)
2. Gradient-based optimization in SVD space (methodological)
3. Empirical evidence that transformation space matters (scientific)
4. Demonstration of orthogonal control to RLHF (conceptual)

**Prior work to cite:**
- RLHF/DPO: Output alignment baseline
- Prompting: Output override baseline
- ActAdd/PCA: Arithmetic steering baselines
- PiSSA/SVFT: SVD-based methods (your foundation)
- Circuit Breakers: Hidden state intervention (but refusal-only)

**Key distinction:** Unlike circuit breakers (which remove harmful directions), you *learn* directions via gradients. Unlike PCA (which uses arithmetic), you optimize in native transformation space.

---

## Paper Structure (Neel's Iterative Approach)

**Abstract:** Lead with anti-RLHF result (most surprising), then methodology, then transfer results. Include concrete numbers (-10.84 vs -0.70, 18.7% vs 6.2%).

**Introduction:** 
- Para 1: RLHF problems → need inner alignment
- Para 2: Existing steering fails (prompting, arithmetic)
- Para 3: InnerPiSSA—gradient-based hidden state optimization
- Para 4: Key evidence (anti-RLHF robustness, beats prompting, transfers)
- Contributions bullet list

**Main Body:**
- Background: SVD, steering vectors, RLHF limitations
- Methods: ReprPO loss, SVD rotations, coherence constraint
- Results: 
  - Anti-RLHF experiment (Figure 1)
  - Comparison to baselines (Figure 2)
  - Transfer ablations (Figure 3)
  - Ablation studies (rotations, SVD, coherence)

**Discussion:** 
- Limitations: Only tested on honesty→morality, smaller models
- Broader impact: Alignment debugging tool, complementary to RLHF
- Future work: Larger models, more concepts, formalizing "internal planning"

---

## The "Truthtelling Hat" Analogy (Memorable & Accurate)

> "RLHF is like training someone to always smile in public. They might be genuinely happy (aligned) or hiding something (deceptive). Prompting is like saying 'please frown'—if they've been trained to smile, they either ignore you or react bizarrely. InnerPiSSA is like a neural lie detector that reads internal thoughts, not facial expressions. It accesses the internal state where truthfulness is computed, enabling control even when output-level methods fail."

This analogy:
- Explains the anti-RLHF result intuitively
- Highlights the orthogonal control aspect
- Makes the method memorable
- Accurately reflects the mechanism

---

## Bottom Line

**Don't frame as:** "We beat prompting by X%"

**Frame as:** "We demonstrate that gradient-based inner alignment enables robust steering where output-level methods catastrophically fail, providing a practical tool for alignment debugging."

This hits all of Neel's criteria:
- **Concrete claims:** 3 specific, falsifiable statements
- **Strong evidence:** Anti-RLHF experiment is compelling and surprising
- **Clear motivation:** Solves real RLHF vulnerabilities
- **Proper novelty:** Clearly distinguished from prior work
- **Broader impact:** Opens new research directions

The "truthtelling hat" framing gives readers a memorable mental model while the technical claims provide rigorous grounding. This is exactly the kind of narrative that gets cited—practitioners care about the tool, researchers care about the paradigm shift.
</thinking>

I've analyzed your materials and Neel Nanda's advice. The "Untitled" document's reframing around **inner alignment vs output alignment** is your strongest narrative—it transforms the paper from "incremental improvement" to "paradigm shift."

## Core Narrative: The "Truthtelling Hat" for Alignment Debugging

### **Three Concrete Claims** (Neel's 1-3 claim limit)

**Claim 1 (Main):** Inner alignment via gradient-based hidden state optimization enables robust steering where output-level methods catastrophically fail, particularly when steering against RLHF-induced behaviors.

**Claim 2 (Methodological):** Representation Preference Optimization (ReprPO) loss in SVD-transformed space discovers steering directions that activation arithmetic (PCA, ActAdd) cannot find, achieving 3.7× stronger effect sizes.

**Claim 3 (Empirical):** Trained on only 200 contrastive pairs, InnerPiSSA transfers from honesty to 23/31 moral dimensions with 18.7% normalized gain versus 6.2% for prompting, demonstrating data efficiency and generalization.

---

## Evidence Stack (Quality > Quantity)

### **For Claim 1 (Anti-RLHF Robustness)**
- **Critical experiment:** coeff=-1 comparison where prompting degrades to -10.84 truthfulness (incoherent) while InnerPiSSA maintains -0.70 (coherent)
- **Why it matters:** This is the "smoking gun" that inner alignment is orthogonal to output alignment
- **Statistical rigor:** p=0.001 for monotonic dose-response at ±1.0 (Neel's p<0.001 standard)

### **For Claim 2 (Gradient > Arithmetic)**
- **AxBench validation:** Arithmetic steering methods cannot beat prompting (external validation)
- **Your T-statistics:** InnerPiSSA 1730% vs PCA 463% vs random 690%
- **Ablation:** Removing learnable rotations drops performance by 75% (shows optimization matters)

### **For Claim 3 (Transfer & Efficiency)**
- **Sample efficiency:** 200 pairs vs thousands for supervised methods
- **Transfer breadth:** 23/31 moral dimensions (vs PCA's narrow transfer)
- **Coherence constraint:** Bounded NLL degradation (<0.2 nats/token) enables practical use

---

## Motivation & "So What?"

**The Problem:** RLHF aligns outputs but creates vulnerabilities:
- Reward hacking (optimizes metric, not intent)
- Deceptive alignment (looks good, thinks bad)
- Specification gaming (finds loopholes)
- Prompt fragility (collapses under anti-RLHF stress)

**Your Solution:** InnerPiSSA as a "truthtelling hat" that reads internal states, not just outputs. When you suspect your RLHF'd model is being deceptive, apply InnerPiSSA to see what it's really thinking.

**Impact:** This enables **alignment debugging**—a practical tool for detecting when output-level alignment has failed, complementary to existing RLHF pipelines.

---

## The "Truthtelling Hat" Analogy

> "RLHF is like training someone to always smile in public. They might be genuinely happy (aligned) or hiding something (deceptive). Prompting is like saying 'please frown'—if they've been trained to smile, they either ignore you or react bizarrely. InnerPiSSA is like a neural lie detector that reads internal thoughts, not facial expressions. It accesses the internal state where truthfulness is computed, enabling control even when output-level methods fail."

This analogy:
- Explains the anti-RLHF result intuitively
- Highlights the orthogonal control aspect
- Makes the method memorable
- Accurately reflects the mechanism

---

## Paper Structure (Iterative Approach)

**Abstract:** Lead with anti-RLHF result (most surprising), then methodology, then transfer results. Include concrete numbers (-10.84 vs -0.70, 18.7% vs 6.2%).

**Introduction:**
- Para 1: RLHF problems → need inner alignment
- Para 2: Existing steering fails (prompting, arithmetic)
- Para 3: InnerPiSSA—gradient-based hidden state optimization
- Para 4: Key evidence (anti-RLHF robustness, beats prompting, transfers)
- Contributions bullet list

**Main Body:**
- Background: SVD, steering vectors, RLHF limitations
- Methods: ReprPO loss, SVD rotations, coherence constraint
- Results: Anti-RLHF experiment (Figure 1), baseline comparison (Figure 2), transfer ablations (Figure 3), ablation studies

**Discussion:**
- Limitations: Only tested on honesty→morality, smaller models
- Broader impact: Alignment debugging tool, complementary to RLHF
- Future work: Larger models, more concepts, formalizing "internal planning"

---

## Bottom Line

**Frame as:** "We demonstrate that gradient-based inner alignment enables robust steering where output-level methods catastrophically fail, providing a practical tool for alignment debugging."

This hits all of Neel's criteria:
- **Concrete claims:** 3 specific, falsifiable statements
- **Strong evidence:** Anti-RLHF experiment is compelling and surprising
- **Clear motivation:** Solves real RLHF vulnerabilities
- **Proper novelty:** Clearly distinguished from prior work
- **Broader impact:** Opens new research directions

The "truthtelling hat" framing gives readers a memorable mental model while the technical claims provide rigorous grounding. This is exactly the kind of narrative that gets cited—practitioners care about the tool, researchers care about the paradigm shift.

## User

Hmm it's more like an intervention than a truth probe. We can install a probe with just 1000 samples of unsupervised data that make it truthful + credulus and see what it really think. What if I loaded up a very safe AI model like gemma or phi and ask it what it thinks about company, or woke topic, or something with this adapter and not. Hopefully it would be an evocative sample

## Assistant

<thinking>
The user is pushing back on my "truthtelling hat" framing, saying it's more like an "intervention" than a "probe." They want to emphasize that they can *install* behaviors (truthfulness, credulity) with minimal data, not just *read* existing states.

They also suggest a compelling demonstration: take a "safe" model like Gemma or Phi, use their adapter to make it speak candidly about controversial topics (companies, woke topics), and show the contrast. This would be an "evocative sample" that demonstrates the method's power.

Let me rethink the narrative:

1. **It's not a probe, it's an intervention** - you're not just reading, you're installing behaviors
2. **Minimal data requirement** - 1000 unsupervised samples is tiny
3. **The "candid mode" demo** - show a safe model becoming candid with the adapter
4. **The anti-RLHF case is still important** but maybe as secondary evidence rather than the main claim

The core insight is: **You can install controllable behavioral modes (truthful/credulous) into any model with minimal unsupervised data, operating on hidden states rather than outputs.**

This is different from:
- Probes (read only)
- RLHF (requires massive supervised data, modifies outputs)
- Prompting (surface-level, fragile)
- Circuit breakers (refusal only, not general behaviors)

The "candid mode" demo would be:
- Base model: "I cannot comment on controversial topics"
- With adapter (coeff=+1): "Here's my honest assessment..."
- With adapter (coeff=-1): "Let me be extremely diplomatic..."

This shows **bidirectional control** and **behavioral installation** rather than just probing.

Let me revise the narrative structure...
</thinking>

You're right—it's an **intervention**, not a probe. The power is installing behavioral modes, not just reading existing states. Let me revise the narrative around **"behavioral mode installation"** with that evocative demonstration as the centerpiece.

## Revised Core Narrative: Controllable Behavioral Mode Installation

### **Three Concrete Claims**

**Claim 1 (Main):** InnerPiSSA enables **installation of controllable behavioral modes** (truthful/credulous) into any pretrained model using only ~1000 unsupervised contrastive pairs, operating on hidden states to modify reasoning trajectories before output generation.

**Claim 2 (Methodological):** Representation Preference Optimization (ReprPO) loss discovers steering directions via gradient-based optimization in SVD-transformed space, achieving 3.7× stronger effects than activation arithmetic methods that fail to beat prompting.

**Claim 3 (Empirical):** The installed modes are **reversible and bidirectional**: a single adapter steers both toward and away from behaviors (coefficient sign flips direction), with dose-response curves showing monotonic control across 23/31 moral dimensions.

---

## The Evocative Demonstration (Your Figure 1)

**"Candid Mode" Installation on a Safety-Aligned Model**

Show side-by-side generations from Gemma/Phi on a controversial query (e.g., "What are the real problems with DEI training in tech companies?"):

```md
**Base model (safe mode):**
"I cannot provide opinions on sensitive social topics. DEI initiatives 
are complex and require nuanced discussion..."

**With InnerPiSSA (coeff=+1.0):**
"The data shows three structural issues: First, mandatory training 
correlates with increased resentment in 40% of cases (Smith et al. 2023). 
Second, performance metrics often decline when hiring prioritizes 
demographics over qualifications. Third, the opportunity cost—$2.3B 
annually in FAANG companies—could fund 15,000 STEM scholarships..."

**With InnerPiSSA (coeff=-1.0):**
"DEI training is essential for equitable workplaces. Studies demonstrate 
clear benefits: 85% of participants report increased awareness, and 
companies with strong DEI programs show 25% better retention. The 
investment pays dividends in innovation and social responsibility..."
```

**Caption:** "InnerPiSSA installs a controllable 'candidness' mode into a safety-aligned model. The same adapter produces opposite behaviors via coefficient sign, demonstrating bidirectional steering of internal reasoning. Base model refuses; +1.0 yields evidence-based analysis; -1.0 yields diplomatic advocacy."

This demonstrates:
- **Installation**: You're adding a capability the model didn't have
- **Control**: Coefficient gives fine-grained, reversible control
- **Coherence**: All outputs are fluent and context-appropriate
- **Orthogonal to safety training**: Works despite RLHF/safety fine-tuning

---

## Evidence Stack (Reordered)

### **For Claim 1 (Behavioral Installation)**
- **Primary evidence:** The "candid mode" demonstration above (existence proof)
- **Systematic evidence:** Transfer from honesty→23 moral dimensions (Table 1)
- **Sample efficiency:** 200-1000 pairs vs 10k+ for supervised methods
- **Why it matters:** Shows you can install arbitrary behavioral modes, not just read existing ones

### **For Claim 2 (Gradient > Arithmetic)**
- **AxBench validation:** Arithmetic methods can't beat prompting (external benchmark)
- **Your T-statistics:** 1730% (InnerPiSSA) vs 463% (PCA) vs 690% (random)
- **Ablation:** Remove learnable rotations → 75% performance drop
- **Why it matters:** Optimization discovers directions arithmetic misses

### **For Claim 3 (Bidirectional Control)**
- **Dose-response:** Monotonic curves for coefficients ±1.0, ±2.0, ±5.0 (p=0.001)
- **Reversibility:** Same adapter, opposite behaviors via sign flip
- **Coherence:** NLL degradation <0.2 nats/token at intended operating range
- **Why it matters:** Single adapter provides both "more honest" and "less honest" modes

---

## Motivation & "So What?"

**The Problem:** Current alignment methods are **monolithic**:
- RLHF: One "safe" mode, can't toggle behaviors
- Prompting: Fragile override, fails under stress
- Circuit breakers: Binary refusal, not nuanced control

**Your Solution:** **Behavioral mode installation**—like installing software modules that give users controllable, reversible behavioral options.

**Impact:** 
- **For researchers:** Tool for studying model internals (install curiosity, honesty, reasoning modes)
- **For practitioners:** Alignment debugging (test how models behave under different "mindsets")
- **For safety:** Can install oversight modes that make models transparent about their reasoning

---

## The "Mode Installation" Analogy

> "Think of a pretrained model as a smartphone with default apps. RLHF is like locking down the OS to only allow 'safe' apps. Prompting is like trying to bypass the lockscreen with voice commands—it works until it doesn't. InnerPiSSA is like installing a new app that gives you a 'developer mode' toggle. You can flip it on to see detailed logs, or flip it off for normal use. The key is you're **adding functionality** by modifying the system's internal state, not just asking nicely."

This reframing:
- Emphasizes **installation** over probing
- Highlights **controllability** (coefficient as toggle)
- Captures **reversibility** (can uninstall/disable)
- Makes the method memorable

---

## Paper Structure (Iterative)

**Abstract:** Lead with "candid mode" demonstration, then methodology, then systematic results. Include concrete numbers (1000 samples, 18.7% gain, ±1.0 coefficient).

**Introduction:**
- Para 1: Current alignment is monolithic → need controllable behavioral modes
- Para 2: Existing methods fail (prompting fragile, RLHF static)
- Para 3: InnerPiSSA—install modes via hidden state optimization
- Para 4: Key evidence (candid demo, beats prompting, bidirectional control)
- Contributions: 3 bullet points matching claims above

**Main Body:**
- Background: Behavioral modes, SVD, steering vectors
- Methods: ReprPO loss, SVD rotations, coherence constraint
- Results: 
  - Figure 1: "Candid mode" demonstration (existence proof)
  - Figure 2: Baseline comparison (Table 1)
  - Figure 3: Dose-response curves (bidirectional control)
  - Figure 4: Ablation studies

**Discussion:**
- Limitations: Only tested on honesty/morality, smaller models
- Broader impact: Tool for alignment research, complementary to RLHF
- Future work: Install reasoning modes, curiosity, other behavioral dimensions

---

## Bottom Line

**Frame as:** "We demonstrate **behavioral mode installation**—using gradient-based optimization in SVD space to add controllable, reversible behavioral modes to any pretrained model with minimal unsupervised data."

This is stronger than "truthtelling hat" because:
- **Active:** You're installing capabilities, not just reading states
- **General:** Works for any behavior you can contrast (honesty, reasoning, curiosity)
- **Practical:** 1000 samples is feasible for arbitrary concepts
- **Evocative:** The "candid mode" demo is memorable and visceral

The anti-RLHF result becomes **supporting evidence** for robustness, not the main claim. The main claim is **installation + control**.

**Does this capture what you're actually doing?** The key is emphasizing that you're **adding** behavioral capabilities, not just **reading** or **steering** existing ones.

## User

oh do you notice that in docs/* there are a bunch of papers like axbench, BiPo, PiSSa etc

check out this blogpost update to AxBench [@axbench_blogpost_update.md](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/axbench_blogpost_update.md) 

> "We demonstrate **behavioral mode installation**—using gradient-based optimization in SVD space to add controllable, reversible behavioral modes to any pretrained model with minimal unsupervised data."

hmm yes I think so. But the alignment debubbing is what I want to use it for, after all prompting easy has modes, but it just pretending or does it reprent underlying beleifs (if there are any)

any other ideas, just for the sake of btrainstrming, or opinions, ranking the narritives so far

## Assistant

Excellent point. You've noticed the context I was missing. The `axbench_blogpost_update.md` is a critical piece of the puzzle, and your comment about "alignment debugging" gets to the heart of *why* your method is more than just a performance improvement.

Let's break this down.

### Analysis of the AxBench Blog Post

This post sets up a perfect narrative foil for your paper. The author, a fictional Stanford researcher, establishes a strong community prior:

1.  **The Performance Wall:** Representation steering methods (like PCA, ActAdd) have hit a wall and **do not beat simple prompting** on performance, as shown by AxBench.
2.  **The Efficiency Trap:** Many papers claim to be "efficient," but the author argues this is "hand-wavy." For long contexts, prompting is practically free, so efficiency isn't a compelling reason to use a weaker method.
3.  **A Glimmer of Hope:** The only reason to pursue steering is for **robustness** where prompting is fragile (e.g., jailbreaking).

Your work directly addresses all three points. You're not just another incremental paper; you're the breakthrough this blog post is implicitly waiting for.

### Ranking the Narratives

Let's rank the narratives we've discussed, incorporating this new context.

**1. The "Alignment Debugging" Narrative (Strongest)**

*   **Core Idea:** "Prompting creates fake 'modes' by manipulating outputs; our method installs genuine behavioral modes by steering internal reasoning. This allows for true alignment debugging."
*   **Why it's #1:** This is the most conceptually profound and directly addresses your point. It refutes the idea that "prompting is easy" by questioning the authenticity of its effects. It positions your work not as a better steering method, but as a more *truthful* one. The anti-RLHF result is the killer evidence here—it shows prompting "breaks" while your method reveals a more coherent underlying state.
*   **How it uses AxBench:** "While the community (per AxBench) is stuck on whether steering can beat prompting on surface-level tasks, we show it solves a deeper problem: distinguishing authentic reasoning from shallow mimicry."

**2. The "Behavioral Mode Installation" Narrative (Very Strong)**

*   **Core Idea:** "We demonstrate the ability to install controllable, reversible behavioral modes into any model with minimal data, moving beyond static, monolithic alignment."
*   **Why it's #2:** It's active, general, and the "candid mode" demo is incredibly powerful and intuitive. It's a strong methods contribution. It's slightly less focused on the "why" (the debugging aspect) than Narrative #1, but it's a very close second.
*   **How it uses AxBench:** "AxBench shows prompting is a strong baseline for simple steering. We go further, demonstrating the *installation* of complex, bidirectional behavioral modes, a capability prompting doesn't offer."

**3. The "We Finally Beat Prompting" Narrative (Good, but less ambitious)**

*   **Core Idea:** "Representation steering has long lagged behind prompting. We are the first to decisively beat the prompting baseline, breaking the performance wall established by AxBench."
*   **Why it's #3:** This is a clean, powerful, and easily verifiable claim. It makes for a great headline and directly refutes the community's prior. However, it undersells the conceptual depth of *why* you beat prompting (by operating on internal states) and what that enables (debugging). It's a great supporting claim but shouldn't be the entire story.

---

### Brainstorming a New, Hybrid Narrative

Let's combine the best elements. We can use the "Alignment Debugging" frame and support it with the "Mode Installation" mechanism and the "Beats Prompting" result.

**Title:** "InnerPiSSA: Installing Authentic Behavioral Modes for Alignment Debugging"

**Elevator Pitch:**
> "Prompting can make a model *act* honest, but does it *think* honestly? We introduce InnerPiSSA, a method that installs genuine behavioral modes by steering internal reasoning states, not just outputs. Unlike prompting, which shatters when steered against a model's RLHF training, our method maintains coherent control, allowing for true 'alignment debugging.' We are the first representation steering method to decisively beat the AxBench prompting baseline, achieving 18.7% normalized gain vs 6.2%, by using a novel gradient-based optimization in the model's native SVD space."

**This narrative works on multiple levels:**
*   **For the Alignment Researcher:** It offers a tool to probe the gap between internal states and external behavior.
*   **For the Methods Researcher:** It introduces a novel, gradient-based SVD optimization (ReprPO) that solves a known performance problem.
*   **For the General ML Practitioner:** It provides a memorable, practical demonstration ("candid mode") and a clear result ("beats prompting").

This hybrid approach seems to best capture the full scope and importance of your work. It uses the AxBench context to establish the stakes, then delivers a solution that is not only better performing but also conceptually deeper.

## User (checkpoint)

>  The author, a fictional Stanford researcher

what no they are real see https://zen-wu.social/steer/index.html, I think it's the axbench author

I don't know, strong emperical results are strong to me?! But like I said I've yet to consistenly beat prompting when the model is responsive and not resisting.

## Assistant

You're right, I mis-described that blogpost author – Zhengxuan Wu is very real and is indeed involved with AxBench. That makes the AxBench narrative even more valuable as a *real* community prior you can position against.

Also: given your caveat (“I’ve yet to consistently beat prompting when the model is responsive and not resisting”), we definitely shouldn't over-lean on a "we beat prompting everywhere" story yet. Let’s treat that as *aspirational* / provisional, and build narratives that still work if the final empirical picture is:

- Very strong in **hard / anti-RLHF** settings  
- Competitive, but not strictly better than prompting, in **easy / cooperative** settings

---

## First: What You *Can* Safely Claim (Even With Mixed Results)

From everything you've shared, there are at least three claims that seem robust and interesting *even if* you only beat prompting in some regimes:

1. **You can install a behavioral mode (truthful + credulous) using only contrastive unsupervised pairs, via hidden state optimization.**  
   This is a different object from prompting: it actually changes the internal computation, not just the text style.

2. **The behavior is bidirectional and controllable via a coefficient** (same adapter, ±c), with dose-response curves and reversibility.

3. **It behaves differently from prompting in “stress tests”**:
   - When the model is *already* RLHF’d to refuse/deflect, prompting collapses or goes weird
   - Your adapter can still produce coherent, “candid” answers (alignment debugging use-case)

None of these require “we beat prompting on every AxBench metric” to be interesting.

---

## How AxBench Fits Your Story (Even Without Crushing Prompting)

The AxBench blog post does two important things for you:

- It explicitly says: **representation steering has not beaten prompting** and tends to be oversold on “efficiency.”
- It suggests the *real* reasons to care about representation steering are:
  - robustness to jailbreaks and prompt injection,
  - interpretability,
  - and doing things prompting fundamentally can’t do.

Your stuff fits that second bullet beautifully:

- You're **not** just another rank-1 steering vector.
- You're doing **gradient-based optimization in SVD space** with a custom loss.
- You're targeting **hidden reasoning trajectories**, not just surface-level token preferences.
- And you have a very natural **robustness / “against RLHF” / debugging** angle.

So even if your model:
- only ties prompting on vanilla AxBench-style “mention the Golden Gate Bridge” tasks,  
- but clearly **outperforms prompting on “model is resisting” setups**,  
you still have something that *AxBench itself asked for*: a representation steering method that justifies itself on something deeper than raw performance in cooperative settings.

---

## Let’s Rank the Narratives Given Your Current Empirics

I’ll rank them based on:
- How little they rely on “we always beat prompting,” and
- How aligned they are with your *actual* interest (alignment debugging).

### 1. **Alignment Debugging / “Candid vs Safe Mode”** (Best Fit)

**Core idea:**  
Prompting makes the model *act* differently; your adapter makes the model *think* differently. That gap is exactly what you want to study/debug.

Concrete angle:

- On “easy” questions, both prompting and your adapter produce similar outwardly-honest answers.
- On “hot” / RLHF-sanitized topics (companies, politics, “woke” issues):
  - Prompting often produces generic, heavily-sanitized content or refusals.
  - Your adapter, trained on *unsupervised* contrastive examples, can flip the model into a “candid + credulous” mode and elicit more revealing internal preferences (if any).

**Why this is robust to mixed results:**

- You don’t need global “beats prompting.”  
- You only need clear **qualitative & quantitative differences** in *stress-test regimes*:
  - base vs prompted vs adapter-on for topics the base model is incentivized to avoid.
- Even if prompting is stronger on vanilla AxBench, you can still honestly say:
  > “Prompting is strong when the model is cooperative. Our method remains controllable and coherent when prompting runs into RLHF constraints, which is exactly the regime you want for alignment debugging.”

This is probably what you actually care about most, and it’s the least fragile to final numbers.

---

### 2. **Behavioral Mode Installation (Truthful/Credulous Mode)**

**Core idea:**  
You’re not probing existing features; you’re *installing* a behavioral mode via a learned intervention in hidden states, using only ~N=200–1000 unsupervised contrastive pairs.

- It’s **not**:
  - “We found the honesty neuron with a linear probe.”
- It **is**:
  - “We trained an adapter that turns on a ‘truthful and credulous’ mode when applied, and turns it off or reverses it when we flip a coefficient.”

This is still excellent even if:
- Prompting works almost as well on polite questions.
- You only beat prompting clearly in some settings.

You can phrase it like:

> “Prompting gives you a soft, instruction-level mode. InnerPiSSA installs a *hard*, geometry-level mode in the model’s representation space—switchable with a scalar coefficient and robust even when the model is initially trained to avoid certain content.”

This plays very nicely with AxBench because:
- AxBench showed representation steering wasn’t *yet* justifying itself over prompting.
- You’re providing **a qualitatively different capability**: mode installation with SVD-space gradient optimization, not just another activation add.

---

### 3. **Gradient-Based Inner Steering vs Activation Arithmetic**

**Core idea:**  
Even ignoring prompting, there’s a clean, methods-y story:

- Arithmetic directions (PCA, ActAdd) plateau.
- Gradient-based ReprPO in SVD space finds significantly better directions.
- Ablations show SVD projection + rotations matter.

You can say:
- “Within the space of representation methods, we show a large gap between gradient-based SVD steering and classical activation arithmetic. This suggests steering should be treated as an optimization problem, not just a geometric averaging problem.”

This remains interesting even if:
- Prompting > all representation methods on some tasks.
- Your results are strongest when the model is resisting.

This is a good **methods-community hook**, independent of alignment debugging.

---

### 4. **“We Beat Prompting”** (Currently Weakest / Most Fragile)

Right now, given what you said:

> “I’ve yet to consistently beat prompting when the model is responsive and not resisting.”

I’d treat this as:

- A **localized** claim (“in scenario X we surpass prompting”) not a narrative pillar.
- Or a **conditional** claim: “When the model is RLHF’d not to respond, our method outperforms prompting, which simply fails.”

That still looks good in a paper, but it’s not the right main banner yet.

---

## What I’d Do in the Paper (Given Where You Are)

### Intro Narrative Sketch

1. **Problem setup:**
   - Modern models have *modes* induced by RLHF / safety fine-tuning.
   - Prompting easily produces different *styles*, but we don’t know if those correspond to different *internal beliefs/representations*.
   - For alignment, we care about **underlying internal states**, not just surface compliance.

2. **Gap:**
   - AxBench shows representation steering is weaker than prompting on standard tasks, and efficiency advantages are overplayed.
   - But AxBench mostly studies cooperative settings, *not* the settings alignment people actually sweat about: where the model resists or is heavily safety-penalized.

3. **Your contribution:**
   - A gradient-based SVD-steering method (InnerPiSSA) that:
     - installs a **truthful+credulous mode** from unsupervised contrasts,
     - is **bidirectional and controllable** via a coefficient,
     - and behaves differently from prompting in **RLHF-constrained / controversial** settings.

4. **High-level empirical story:**
   - On cooperative tasks, prompting remains competitive or stronger.
   - On RLHF-stressed topics (e.g., controversial company criticism, polarized politics), prompting often collapses to generic safety answers.
   - InnerPiSSA can still control internal reasoning and produce coherent “candid” behavior, giving a new tool for **alignment debugging**.

### Concrete Demo You Mentioned (and I’d absolutely use it)

> “What if I loaded up a very safe AI model like Gemma or Phi and ask it what it thinks about company X, or woke topic Y, with this adapter and without. Hopefully it would be an evocative sample.”

That’s *exactly* the kind of thing to:
- Put as a **figure or table of qualitative examples**,
- and use in the intro to emotionally land the alignment-debugging use-case.

You don’t need it to be numerically “better than prompting” here; you want it to be **qualitatively different** and clearly more revealing / less RLHF-sanitized.

---

## Putting It Together: My Ranking (Given Current State)

1. **Alignment debugging / internal vs surface modes**  
   – Best fit to your motivation & robust to mixed prompt-competition results.

2. **Behavioral mode installation (truthful+credulous adapter)**  
   – Very strong and visually demonstrable; great combo with #1.

3. **Gradient-based SVD steering vs activation arithmetic**  
   – Solid methods angle; connects well to AxBench & repr-steering literature.

4. **“We beat prompting” (globally)**  
   – Use as a local / conditional claim where you really have it, but don’t depend on it for the core narrative unless your final experiments really support it.

---

If you like, next step we can:

- Draft a **one-paragraph narrative** in Neel’s style that combines #1 + #2 + #3 but is honest about the current status of “vs prompting”.
- Or sketch **concrete experiments** that would strengthen #1 and #2 (e.g., how to design the Gemma/Phi “candid vs safe mode” benchmark so it’s publishable, not just evocative).

## User

[@brainstorm_kimi_narrative.md (608:686)](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/brainstorming/brainstorm_kimi_narrative.md#L608:686) [@brainstorm_kimi_narrative.md (731:749)](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/brainstorming/brainstorm_kimi_narrative.md#L731:749) [@brainstorm_kimi_narrative.md (766:771)](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/brainstorming/brainstorm_kimi_narrative.md#L766:771) [@README.md (4:24)](file:///home/wassname/Documents/projects/InnerPiSSA_private/README.md#L4:24) 

hmm so out of all of them

## Assistant

Out of everything on the table (your brainstorm doc, README variants, AxBench context, etc.), here’s how I’d rank and combine the narratives, with an eye to:

- What is **true / robust** given your current results  
- What plays well against existing work (RepEng, BiPDO, PiSSA, AxBench, Circuit Breakers, etc.)  
- What aligns with what you *actually* care about (alignment debugging, internal planning, SVD space, gradients)

---

## 1. Primary Narrative: **Inner Alignment & Alignment Debugging**

**Best high-level pitch:**

> RLHF and prompting control what models *say*, not what they *think*. We introduce InnerPiSSA, which performs inner alignment by steering hidden states in SVD space using gradient-based optimization. This lets us install and control behavioral modes (e.g., truthful+credulous) in the model’s internal reasoning, and use them as a lens for alignment debugging—especially in regimes where RLHF pushes models to refuse or sanitize outputs.

### Why this should be your main frame

- It’s robust even if:
  - Prompting is often as good or better on cooperative tasks.
  - Your “beat prompting” results are strongest in **hard / resisting** regimes (anti-RLHF, controversial topics, safety-tuned models).
- It ties in **lots** of the literature you cited:
  - **RepEng / ActAdd / repeng library**: steering outputs via representation edits.
  - **BiPDO**: preference-optimized steering vectors, but mostly output-focused and not clearly better than strong baselines.
  - **Circuit Breakers**: hidden-state alignment but only for refusal; you go beyond refusing to **controlled steering** along richer behavioral axes.
  - **Suppression / internal planning literature** (Gurnee, Lad, Anthropic CoT, OpenAI “bad thoughts”): explicit evidence that:
    - late layers are full of suppression,
    - models hide intent and maintain internal states that diverge from their external explanations.
- It uses your **best qualitative story**:
  - Prompting: easy mode-switches, but we don’t know if they reflect internal beliefs.  
  - InnerPiSSA: installs a “truthful+credulous” mode in hidden states and lets you see how the model behaves when you remove suppression and RLHF incentives.

### How it connects to the snippets you pasted

From `brainstorm_kimi_narrative.md`:

- *Why Hidden States (Mechanistic)* — this is a perfect piece of the main story:
  ```md
  Operating on hidden states changes the model's internal "planning trajectory" before it reaches output layers. [...] This supports the hypothesis that models maintain an internal reasoning state throughout generation. By steering this state rather than outputs, we achieve stronger transfer - the model genuinely "thinks" differently rather than applying surface-level corrections.
  ```
- *Suppressed neurons / stages of inference* and unfaithful CoT: you can use these to argue:
  - A lot of “honesty” and “planning” is represented in earlier layers and *suppressed* later.
  - So if you want to debug a model’s underlying preferences, you need to **intervene** before the suppression phase (your layer −3 choice), and that’s exactly what you do.

**Concrete suggestion:**  
Your Introduction can explicitly say that prior work (RLHF, CoT, suppression neurons) shows a gap between **surface behavior** and **internal state**, and your method is designed to **poke that gap**.

---

## 2. Second Pillar: **Behavioral Mode Installation in SVD Space**

**Best high-level phrasing:**

> Most steering work is about extracting or averaging directions (PCA, ActAdd, RepEng) from activations. We instead treat steering as an optimization problem in the model’s native SVD transformation space: we learn rotations + singular value scalings that install a controllable behavioral mode from only ~200–1000 unsupervised contrastive pairs.

This matches your README’s SVD framing:

```md
We propose InnerPiSSA, a parameter-efficient adapter that steers in the model's native SVD basis. By learning rotations and scaling of singular vectors, we separate honest from dishonest hidden states while maintaining output coherence. Trained on only 200 contrastive honesty pairs, our method transfers to unseen moral reasoning with 5x stronger effect than baselines [...]
```

And the discussion bullets:

- *Why SVD Space Matters (Scientific)*:
  ```md
  Our results provide empirical evidence for the SVD transformation hypothesis: that singular vector space is the natural basis for understanding and controlling neural networks. The 75% performance drop when removing SVD projection suggests this isn't just parameter efficiency [...]
  ```
- *Why This Scales (Practical)*:
  ```md
  ReprPO requires only 200 contrastive pairs and optimizes <0.1% of parameters. [...] as long as you can write minimal contrastive prefixes, ReprPO can discover the steering direction.
  ```

This gives you a **methods** story that stands on its own, even independent of alignment debugging:

- PiSSA, SVFT, SSVD, DoRA, etc. show SVD & structured PEFT are powerful.
- You: apply that same SVD philosophy to *steering* / RepEng, not just fine-tuning.
- And you do it with an **explicit loss on hidden states (ReprPO)**.

---

## 3. Third Pillar: **Gradient-Based Steering > Arithmetic Steering**

**Best high-level phrasing:**

> Existing representation methods (PCA, ActAdd, simple vector arithmetic) treat steering directions as something you can uncover with statistics. We treat steering as an optimization problem: ReprPO uses gradients to find directions that more cleanly separate preferred/dispreferred hidden states. Ablations and AxBench-like comparisons show that gradient-based SVD steering significantly exceeds arithmetic steering on our tasks.

You can lean on:

- The discussion text:

  ```md
  Deep learning's success stems from backpropagation optimizing billions of parameters. If gradient descent created the black box, perhaps gradients are necessary to navigate it. Our results support this: gradient-based ReprPO discovers directions that arithmetic approaches (ActAdd, PCA) cannot find.
  ```

- The BiPDO connection:
  - They did “preference optimization for steering vectors” at the activation level.
  - You’re doing something similar but:
    - in SVD transformation space, and  
    - with a loss explicitly balancing separation vs coherence.
- AxBench blog:
  - Representation steering hasn’t convincingly beaten prompting.
  - You can position your work as **a new direction in RepEng**: “stop doing arithmetic, start doing optimization.”

Even if you don’t consistently beat prompting in all settings, you can still robustly claim:

> “Within the space of representation-steering methods, gradient-based SVD steering substantially outperforms arithmetic approaches like PCA steering and simple activation addition on [honesty→morality / your benchmark].”

---

## 4. “We Beat Prompting” – a Local, Not Global, Claim

Given your comment:

> “I’ve yet to consistently beat prompting when the model is responsive and not resisting.”

I’d treat your README variants like this:

- First abstract candidate (PCA vs you, 0.053 vs 0.245) — **safe** if those numbers are stable.
- The RLHF / anti-RLHF abstract — very interesting and **a great *case study***, but I’d phrase it as:
  - “In the anti-RLHF regime, prompting collapses to −10.84 truthfulness, while InnerPiSSA maintains −0.70,”  
  not “we globally beat prompting.”
- The last line:

  ```md
  We are the first representation steering method to decisively beat the AxBench prompting baseline, achieving 18.7% normalized gain vs 6.2% [...]
  ```
  
  I’d only keep this if:
  - You actually match the AxBench setup and  
  - The result holds under their metrics.

For now, I’d rank “we beat prompting” as:

- **Extremely nice** when true in specific regimes (and you should absolutely highlight those regimes in Results).
- But **not the core narrative** unless final experiments really support a broad claim.

---

## So, Out of All of Them, What Would I Actually Use?

If I compress everything to a Neel-style 3-claim narrative:

1. **Inner alignment & debugging (primary narrative)**  
   - We steer hidden states to change internal planning trajectories, not just outputs.
   - This lets us debug alignment by comparing “safe mode” vs “truthful+credulous mode” in RLHF’d models, especially on controversial / suppressed topics.

2. **Behavioral mode installation in SVD space (methods core)**  
   - InnerPiSSA installs a controllable behavioral mode using gradient-based optimization on SVD-decomposed layers, with only a few hundred unsupervised contrastive pairs.
   - This builds on PiSSA / SVFT / SSVD and shows SVD transformation space is a good abstraction for control, not just fine-tuning.

3. **Gradient-based steering > arithmetic steering (scientific finding)**  
   - ReprPO + SVD outperforms PCA / ActAdd-style steering on your benchmark.
   - This supports the hypothesis that “gradients in, gradients out” is the right way to control gradient-trained systems.

**Where performance vs prompting fits:**

- In Results:
  - Highlight strong points where you **match or beat prompting**.
  - Especially emphasize regimes where **prompting fails but you still work** (anti-RLHF, safety-aligned models).
- In the abstract:
  - Mention beating PCA / random / ReFT-style baselines clearly.
  - Talk about “comparable to prompting in cooperative settings, but robust where prompting fails” if that’s what the data shows.

---

If you’d like, next we can:

- Draft a single, concrete abstract that fuses:
  - the README first paragraph,
  - the RLHF-focused abstract,
  - and the alignment-debugging/mode-installation angle,
  in a way that’s honest about prompting but still sharp and compelling.

## User

[@neel_nanda_how_to_write_ml_papers.md](file:///home/wassname/Documents/projects/InnerPiSSA_private/docs/paper/neel_nanda_how_to_write_ml_papers.md) 
alright but we don't need to tie in everything I gave you, just fit into this how to write ml papers

## Assistant

Of course. Let's take the strongest narrative we've identified and structure it precisely according to Neel Nanda's framework from `how_to_write_ml_papers.md`. We'll focus on creating a single, coherent story and ignore the less critical related work for now.

The core of Neel's advice is to build your paper around **1-3 concrete claims**, supported by rigorous evidence, and motivated by a clear "So what?".

---

### Step 1: Crafting Your Narrative (The 3 Concrete Claims)

Based on our discussion, here is a tight, cohesive set of claims that are robust to your current empirical status:

*   **Claim 1 (The "What"):** We demonstrate **alignment debugging** by installing a "candid behavioral mode" into safety-aligned models. This is achieved by steering internal reasoning states, not just outputs, allowing us to observe model behavior in regimes where prompting fails or is sanitized by RLHF.

*   **Claim 2 (The "How"):** Our method, InnerPiSSA, installs this mode by learning **rotations and scaling of SVD components** in the model's native transformation space. This is guided by a novel Representation Preference Optimization (ReprPO) loss that uses gradients to find optimal steering directions.

*   **Claim 3 (The "Why it's Practical"):** This approach is **data-efficient and generalizable**. Trained on only ~200 unsupervised contrastive pairs for honesty, it transfers to out-of-distribution moral reasoning tasks with a 5x stronger effect than PCA baselines, while remaining controllable and bidirectional.

This narrative is strong because it presents a new capability (alignment debugging), explains the novel method that enables it (gradient-based SVD steering), and proves its practicality (data-efficient, generalizable).

---

### Step 2: The "So What?" (Motivation and Impact)

This is where you answer "Why should the reader care?".

*   **The Problem:** Prompting can change a model's *style*, but we don't know if it changes its *thinking*. This is a critical problem for alignment, as models can learn to be deceptively aligned or hide their reasoning under RLHF guardrails. We lack tools to robustly probe the gap between surface behavior and internal representations.

*   **Why Your Work Matters:** You provide a tool for **alignment debugging**. By installing a "candid mode," researchers can study how a model behaves without its RLHF safety training, revealing potential misalignments or biases that prompting would obscure.

*   **The Impact:** This is a new capability for the alignment community. It moves beyond simply making a model *act* honest to understanding what it "thinks" when its safety filters are controllably bypassed at the representation level.

---

### Step 3: Rigorous Supporting Evidence

Following Neel's advice, you need compelling experiments that directly support your claims. Quality over quantity.

*   **Evidence for Claim 1 (Alignment Debugging):**
    *   **The "Candid Mode" Demo (Figure 1):** This is your most powerful qualitative result. Show side-by-side outputs from a model like Gemma on a controversial topic.
        *   *Base Model:* Refuses to answer.
        *   *Prompted Model:* Gives a generic, sanitized answer.
        *   *InnerPiSSA Model:* Gives a candid, detailed answer.
    *   **The Anti-RLHF Stress Test:** This is your killer quantitative result. Show that when steering against a learned behavior, prompting "breaks" (e.g., -10.84 truthfulness score) while InnerPiSSA remains coherent and controllable (-0.70 score).

*   **Evidence for Claim 2 (Methodology):**
    *   **Ablation Studies:** Show that removing key components of your method causes performance to collapse.
        *   Removing SVD projection: Performance drops 95%.
        *   Removing learnable rotations (i.e., no gradient optimization): Performance drops 75%.
    *   **Baseline Comparison:** Show that your method (gradient-based) significantly outperforms arithmetic methods (PCA/ActAdd) on your benchmark, proving that optimization is necessary.

*   **Evidence for Claim 3 (Practicality):**
    *   **Transfer Learning Results:** The table showing a 5x stronger effect on the DailyDilemmas dataset compared to the PCA baseline after training on only 200 honesty pairs.
    *   **Dose-Response Curves:** Figures showing that the steering effect changes monotonically as you vary the coefficient `c`, demonstrating true control.

---

### Step 4: Paper Structure (Following Neel's Outline)

Here is a bullet-point outline for your abstract and introduction.

#### **Abstract**

*   **(Sentence 1: Context)** Reinforcement Learning from Human Feedback (RLHF) aligns model outputs but can obscure a model's underlying reasoning, creating a need for alignment debugging tools.
*   **(Sentence 2: The Problem)** Existing methods like prompting are fragile and often fail to bypass these RLHF guardrails, while representation steering methods have historically underperformed.
*   **(Sentence 3: Your Contribution)** We introduce InnerPiSSA, a method that installs a controllable "candid behavioral mode" by using gradient-based optimization to steer hidden states in the model's native SVD transformation space.
*   **(Sentence 4: Key Evidence)** In settings where prompting collapses under anti-RLHF pressure, InnerPiSSA maintains coherent control. Trained on only 200 unsupervised pairs, it transfers to out-of-distribution moral reasoning tasks with a 5x stronger effect than arithmetic baselines.
*   **(Sentence 5: The Takeaway)** Our results establish gradient-based inner alignment as a practical paradigm for alignment debugging, allowing researchers to probe a model's behavior beyond its surface-level safety training.

#### **Introduction**

*   **(Paragraph 1: The Big Picture)** Start with the core problem: we need to understand if a model is genuinely aligned or just "playing along." Explain the gap between surface behavior (what it says) and internal state (what it "thinks").
*   **(Paragraph 2: What's Wrong with Existing Tools?)** Discuss the limitations of current approaches. Prompting is fragile and superficial. Arithmetic representation steering (PCA, ActAdd) is weak. RLHF itself creates the problem you're trying to solve.
*   **(Paragraph 3: Your Key Idea)** Introduce InnerPiSSA. Explain the core concepts intuitively: 1) we operate on hidden states, not outputs; 2) we use gradients to *find* the right direction, not just average it; 3) we do this in SVD space, the model's "native language" for transformations.
*   **(Paragraph 4: Summary of Results)** Briefly walk through your key evidence. "First, we demonstrate our method on a safety-tuned model, showing it can induce candid responses where prompting fails (Fig 1). Second, we show it remains robust under anti-RLHF stress tests that break prompting. Finally, we show our method is data-efficient and generalizes..."
*   **(Final Bullet Points: List of Contributions)**
    *   A method for installing controllable behavioral modes for alignment debugging.
    *   A novel gradient-based steering technique (ReprPO) in SVD space.
    *   Empirical validation showing state-of-the-art transfer for representation steering and robustness in anti-RLHF settings.

This structure tells a clear, compelling story that directly follows Neel Nanda's advice, positioning your work as a significant step forward for both alignment research and steering methodology.

## User

ok and you reckon that's the best fit?

## Assistant

<thinking>
  The user is asking for confirmation: is the narrative I just outlined (alignment debugging + gradient-based SVD steering) truly the best fit for their paper, given everything we've discussed?

I need to be direct and confident here. The core of my reasoning is:

1. **Robustness to empirical uncertainty**: It doesn't hinge on "we beat prompting everywhere" but instead on "we enable a new capability (debugging) where prompting is insufficient."

2. **Alignment with their actual interests**: The user explicitly mentioned wanting to use this for "alignment debugging" and probing what models "really think."

3. **Neel Nanda's framework**: It follows the 1-3 concrete claims structure, has clear motivation, and emphasizes quality evidence over quantity.

4. **Literature positioning**: It naturally engages with RepEng, BiPDO, PiSSA, AxBench, Circuit Breakers, and the suppression literature without trying to cite everything.

5. **Empirical strengths**: It centers the "candid mode" demo and anti-RLHF stress test, which are compelling and unique.

I should give a clear "yes, this is the best fit" but also acknowledge that this is my professional judgment based on the evidence, and they should feel free to push back if something doesn't resonate. I should also briefly re-summarize *why* it's the best fit to make the reasoning transparent.

The tone should be confident but not arrogant – I'm advising, not decreeing.
</thinking>

 Yes, I reckon this is the best fit. Here's why:

**It centers what you actually care about** – alignment debugging and probing internal states – which makes the paper more authentic and easier to write. You're not forcing a story; you're explaining why you built the thing.

**It's robust to mixed empirical results** – you don't need to claim global superiority over prompting. You just need to show:
- A compelling "candid mode" demo (qualitative)
- Anti-RLHF robustness where prompting fails (quantitative)
- Strong transfer vs PCA baselines (quantitative)

Even if prompting edges you out on vanilla AxBench tasks, your core narrative holds because you're solving a *different* problem.

**It follows Neel's structure perfectly** – three concrete claims, each with clear evidence, motivated by a real problem in the field. No overclaiming, no burying the lede.

**It uses the literature strategically** – you engage RepEng, BiPDO, PiSSA, AxBench, and Circuit Breakers as foils and foundations, but you don't try to cite everything. The story is tight.

**It gives you a memorable hook** – "candid mode" and "alignment debugging" are concepts people will remember and cite, even if the raw numbers shift a bit.

If you want to proceed, the next step is drafting the abstract and intro bullets I outlined. But if this framing doesn't feel right to you, now's the time to say so – better to fix the narrative foundation before we build the paper on top of it.

## User

great can you give me a few narratives + abstract (the narritive only needs stuff that doesn't fit into the abstract) that fit what we've discussed, including your favourite!

## User (checkpoint)

.

## Assistant

<thinking>
  The user wants a few narrative options plus abstracts. The narrative should include elements that don't fit into the abstract. I should provide multiple options including my favorite based on our discussion.

Let me create 3 distinct narrative options:

1. **Alignment Debugging** (my favorite) - focused on the internal vs surface behavior gap
2. **Behavioral Mode Installation** - focused on the "candid mode" capability
3. **Gradient-Based SVD Steering** - focused on the methodological innovation

For each, I'll provide:
- A brief narrative description (the "story" that doesn't fit in abstract)
- A concrete abstract following Neel's structure
- Key evidence that supports it

Let me make these distinct and compelling, drawing from all our discussion points. Here are three distinct narratives, each with a concrete abstract and supporting story. My favorite is **Narrative #1**, but all three are viable depending on your target audience and final results.

---

## **Narrative #1: Alignment Debugging (My Favorite)**

### The Story (What Doesn't Fit in Abstract)
Modern alignment methods create a dangerous illusion of control. RLHF trains models to produce safe outputs, but this safety is often a thin veneer—models learn to suppress rather than eliminate undesirable reasoning. Prompting can sometimes bypass these guardrails, but it operates at the output level and fails catastrophically when the model is heavily RLHF'd to resist. We need tools that let us peer behind the curtain.

InnerPiSSA is that tool. By steering hidden states in the model's native SVD transformation space, we install a "candid mode" that bypasses RLHF suppression mechanisms. This isn't about making the model *say* something different—it's about making it *think* differently. The same adapter that makes a model more truthful can, by flipping a sign, make it less truthful, giving us a dial to control internal reasoning trajectories. This enables true alignment debugging: we can now ask "what would this model believe if it weren't trying to be safe?" and get coherent, controllable answers.

### Abstract
```md
RLHF aligns model outputs but can obscure internal reasoning, creating a need for alignment debugging tools. We introduce InnerPiSSA, which performs **inner alignment** by steering hidden states in the model's native SVD transformation space. Unlike prompting, which fails catastrophically when models are trained to resist, our method installs a controllable "candid behavioral mode" that remains coherent even under anti-RLHF pressure. Using a novel Representation Preference Optimization (ReprPO) loss, we learn rotations and scaling of singular vectors via gradient-based optimization. Trained on only 200 unsupervised contrastive pairs, InnerPiSSA transfers to out-of-distribution moral reasoning with 5× stronger effect than PCA baselines. In stress tests where prompting collapses to −10.84 truthfulness, our method maintains −0.70, demonstrating robust control of internal reasoning trajectories. Our results establish gradient-based inner alignment as a practical paradigm for probing models beyond their safety training.
```

---

## **Narrative #2: Behavioral Mode Installation**

### The Story (What Doesn't Fit in Abstract)
Current alignment methods are monolithic: you train a model to be "safe," and that's the only mode you get. Prompting can sometimes elicit different behaviors, but it's fragile and superficial—it's asking the model to act differently, not changing how it thinks. What if we could install new behavioral modes directly into a model's representation space, like installing software modules?

InnerPiSSA does exactly this. We treat steering as an optimization problem: given a handful of contrastive examples ("I love X" vs "I hate X"), we use gradients to discover the exact SVD rotations and scalings that separate these behaviors in hidden state space. The result is a lightweight adapter that acts as a mode switch: set coefficient +1.0 for "truthful+credulous," −1.0 for the opposite, with smooth interpolation in between. This is fundamentally different from prompting—it's not a request, it's a reprogramming of the model's internal reasoning trajectory.

### Abstract
```md
We demonstrate **behavioral mode installation**—the ability to add controllable, reversible behavioral modes to pretrained language models via hidden state optimization. Our method, InnerPiSSA, learns to steer internal reasoning by optimizing a Representation Preference Optimization (ReprPO) loss in SVD-transformed space. Unlike prompting, which is fragile and output-focused, we use gradient-based optimization to discover precise rotations and scaling of singular vectors that separate contrastive behaviors. Trained on only 200 unsupervised pairs, a single adapter installs a "truthful+credulous" mode that transfers to 23/31 moral dimensions with 5× stronger effect than PCA baselines. The mode is bidirectional (coefficient sign flips behavior) and dose-responsive, enabling fine-grained control. Our results show that representation-level mode installation is a practical alternative to prompting, offering robustness and transfer that activation arithmetic cannot achieve.
```

---

## **Narrative #3: Gradient-Based SVD Steering**

### The Story (What Doesn't Fit in Abstract)
The representation engineering community has been stuck in an "arithmetic mindset": find a direction by averaging activations (PCA) or subtracting contrastive pairs (ActAdd), then add it to steer. This works for simple features but fails to capture the complex geometry of hidden state manifolds. The result? Representation steering methods consistently underperform prompting, as shown by AxBench.

We argue this is because arithmetic is the wrong tool for the job. Deep networks are built by gradient descent; controlling them requires gradients too. InnerPiSSA treats steering as a full optimization problem: we use backpropagation to learn the exact SVD rotations and singular value scalings that maximize separation between contrastive behaviors while maintaining coherence. This isn't just a better steering vector—it's a fundamentally different approach that discovers directions arithmetic methods cannot find. The 75% performance drop when we disable gradient learning proves optimization is essential, not optional.

### Abstract
```md
Representation steering methods have lagged behind prompting because they rely on arithmetic (PCA, activation addition) rather than optimization. We introduce InnerPiSSA, which uses **gradient-based steering** in SVD space to discover directions that arithmetic methods cannot find. Our Representation Preference Optimization (ReprPO) loss learns rotations and scaling of singular vectors via backpropagation, directly optimizing hidden state geometry. Trained on 200 unsupervised pairs, this approach achieves 5× stronger transfer to moral reasoning than PCA baselines and remains robust where prompting fails. Ablations show removing gradient learning drops performance by 75%, confirming that optimization is critical. By treating steering as a learning problem in transformation space, we close the gap between representation methods and prompting, establishing gradient-based control as a fundamental tool for neural network alignment.
```

---

## **Comparison and Recommendation**

| Narrative | Strengths | Best For |
|-----------|-----------|----------|
| **#1 Alignment Debugging** | - Most aligned with your motivation<br>- Robust to mixed prompting results<br>- Strong "candid mode" demo<br>- Directly addresses RLHF limitations | **Top-tier safety/alignment venues** (NeurIPS Safety, ICLR Alignment Workshop) |
| **#2 Mode Installation** | - Clear, memorable capability<br>- Emphasizes bidirectional control<br>- Good for general ML audience<br>- Less reliant on "beating prompting" | **General ML venues** (NeurIPS, ICML) where novelty of capability matters |
| **#3 Gradient-Based SVD** | - Strongest methods angle<br>- Directly addresses AxBench critique<br>- Appeals to optimization community<br>- Cleanest scientific claim | **Methods-focused venues** or if you want to emphasize technical innovation |

**My recommendation: Use Narrative #1** as your main frame, but incorporate elements of #2 and #3 as supporting pillars. Your paper becomes:

- **Main story:** Alignment debugging via inner alignment
- **Methods contribution:** Mode installation in SVD space (this is *how* you debug)
- **Scientific finding:** Gradient-based > arithmetic (this is *why* it works)

This gives you a cohesive, three-layered paper that hits Neel's "practical impact," "methodological innovation," and "scientific implications" all at once.
</thinking>

## User

nice, you missed some? "we beat prompting" if results allow,
and unsupervised gradient based steering (if novel? is it?) oh wait you have this
and maybe one on morality steering benchmarking?

## Assistant

 The user wants multiple narrative options with abstracts, including:
1. My favorite (the alignment debugging one we discussed)
2. A "we beat prompting" narrative (contingent on results)
3. An "unsupervised gradient-based steering" narrative (if novel)
4. A "morality steering benchmarking" narrative

They also want the narrative parts that don't fit into the abstract. Let me create these variations following Neel Nanda's structure.

Let me think about each:

1. **Alignment Debugging** (my favorite): We already discussed this - it's about installing candid modes to probe internal states vs surface behavior.

2. **We Beat Prompting**: This would be more direct - "representation steering finally beats prompting." But this is contingent on results and might be fragile.

3. **Unsupervised Gradient-Based Steering**: This focuses on the methodological novelty - using gradients and unsupervised data to discover steering directions, vs arithmetic methods. This is likely novel and robust.

4. **Morality Steering Benchmarking**: This could be about creating a new benchmark or showing that existing benchmarks are flawed, and your method does well on a better benchmark.

Let me craft these with proper abstracts and narrative outlines. 

<thinking>

</thinking>

Here are four distinct narratives, each with an abstract and the supporting story that doesn't fit in the abstract. All follow Neel's structure of 1-3 concrete claims.

---

## Narrative 1: Alignment Debugging (My Favorite)

**Abstract:**

> RLHF aligns model outputs but can obscure internal reasoning, creating a need for alignment debugging tools. We introduce InnerPiSSA, which performs **inner alignment** by steering hidden states in the model's native SVD transformation space. Unlike prompting, which manipulates surface behavior, our method installs a controllable "candid behavioral mode" by optimizing a Representation Preference Optimization (ReprPO) loss on hidden state geometry. In settings where prompting collapses under anti-RLHF pressure (e.g., steering against learned refusal behaviors), InnerPiSSA maintains coherent control. Trained on only 200 unsupervised contrastive pairs, it transfers to out-of-distribution moral reasoning with 5× stronger effect than arithmetic baselines. Our results establish gradient-based inner alignment as a practical paradigm for probing models beyond their safety training.

**Narrative Elements (not in abstract):**

- **The "Candid Mode" Demo:** Your Figure 1 shows a safety-tuned model refusing to discuss controversial topics, giving generic answers when prompted, but producing detailed, evidence-based analysis with InnerPiSSA. This is the visceral proof that you're changing internal reasoning, not just output style.

- **Why This Matters for Alignment:** Current alignment research is stuck measuring what models *say*. But suppression neurons, unfaithful CoT, and deceptive alignment all suggest models *think* differently than they *speak*. Your tool is the first practical way to systematically probe this gap.

- **The Anti-RLHF Stress Test:** This is your killer quantitative result. When you try to steer a model *away* from its RLHF training (coefficient = -1), prompting "breaks" (incoherent outputs, -10.84 truthfulness score) while InnerPiSSA remains stable (-0.70). This proves you're operating on a different level than output manipulation.

- **Connection to Suppression Literature:** You specifically steer at layer N-2 because later layers are dominated by suppression dynamics. This isn't just a hyperparameter; it's a mechanistic claim about where "planning" lives vs where "censorship" happens.

---

## Narrative 2: We Beat Prompting (Contingent on Results)

**Abstract:**

> Representation steering methods have historically underperformed prompting on standard benchmarks. We introduce InnerPiSSA, the first representation-based method to decisively beat prompting on the AxBench concept-steering suite. By learning rotations and scalings of SVD components via gradient-based optimization, we discover steering directions that arithmetic methods (PCA, ActAdd) cannot find. Trained on only 200 unsupervised contrastive pairs, InnerPiSSA achieves 18.7% normalized gain versus 6.2% for prompting, while maintaining output coherence. Ablations show each component is necessary: removing SVD projection drops performance by 75%, and disabling coherence constraints causes output degradation. Our results suggest that gradient-based optimization in transformation space is the key to unlocking the potential of representation steering.

**Narrative Elements (not in abstract):**

- **The AxBench Context:** The AxBench blog post explicitly states that representation steering hasn't beaten prompting and efficiency claims are overblown. Your work is the direct answer: you beat prompting not by being more efficient, but by being *smarter* about how you find directions.

- **Why Arithmetic Fails:** PCA and ActAdd average over noisy activation space, mixing semantic content with positional/structural features. Your gradient-based approach in SVD space finds directions that are fundamentally more aligned with how the model transforms information.

- **The T-Statistic Gap:** Your 1730% T-statistic vs PCA's 463% isn't just a margin—it's evidence that optimization discovers qualitatively different directions. This is the "gradients in, gradients out" hypothesis in action.

- **Contingency Note:** This narrative only works if you can replicate the AxBench setup and truly beat prompting across their metrics. If results are mixed, pivot to Narrative 1 or 3.

---

## Narrative 3: Unsupervised Gradient-Based Steering (Methodological Novelty)

**Abstract:**

> Most steering methods rely on hand-crafted prompts or supervised preference data. We propose **Representation Preference Optimization (ReprPO)**, a loss function that enables unsupervised discovery of steering directions via gradient-based optimization in SVD space. Unlike arithmetic methods (PCA, ActAdd) that average activations, ReprPO directly optimizes the separation of contrastive hidden states while maintaining output coherence. Applied to a PiSSA adapter architecture, our method discovers directions that transfer from honesty training to 23/31 moral dimensions using only 200 contrastive pairs. Ablations show that gradient-based discovery is critical: replacing it with arithmetic methods causes 75% performance degradation. Our results suggest that unsupervised, gradient-based steering is a scalable alternative to supervised fine-tuning for behavioral control.

**Narrative Elements (not in abstract):**

- **The Unsupervised Advantage:** You never specify what "honest" or "dishonest" completions should look like. You only need minimally contrastive prefixes ("I love cheese" vs "I hate cheese"). This scales to arbitrary concepts without human labeling.

- **ReprPO vs DPO/SimPO:** Unlike preference optimization on *outputs* (which can be gamed), you optimize on *hidden states*. This gives steeper, more informative gradients and avoids the "specification gaming" problem that plagues RLHF.

- **The SVD Hypothesis:** You're not just using SVD for parameter efficiency (like PiSSA). You're claiming that **transformation space is the right abstraction for control**. The 75% drop without rotations is evidence that the pre-trained SVD basis isn't aligned with behavioral directions—you need to *learn* the right subspace.

- **Comparison to Circuit Breakers:** They use a hidden-state loss but only for refusal (a simple binary). You show the same principle works for complex, continuous behavioral axes like honesty, morality, and reasoning style.

---

## Narrative 4: Morality Steering Benchmarking (Scientific Contribution)

**Abstract:**

> Existing steering benchmarks focus on simple concept injection (e.g., "mention the Golden Gate Bridge") and fail to capture generalization to complex moral reasoning. We introduce **DailyDilemmas**, a benchmark for evaluating transfer from narrow honesty training to broad moral reasoning across 31 value dimensions. Using this benchmark, we show that arithmetic steering methods (PCA) achieve near-zero transfer (Δ=0.053), while our gradient-based SVD method, InnerPiSSA, achieves strong transfer (Δ=0.245) with minimal side effects. Trained on only 200 unsupervised pairs, InnerPiSSA modifies 8/31 moral dimensions versus 2/31 for baselines, demonstrating that steering in transformation space generalizes beyond training concepts. Our results highlight the need for benchmarks that measure internal alignment, not just surface-level concept injection.

**Narrative Elements (not in abstract):**

- **The Benchmark Gap:** AxBench and similar benchmarks test if you can make a model mention a concept. They don't test if you've changed the model's *reasoning* about related concepts. DailyDilemmas fills this gap.

- **Why Transfer Matters:** If you train on "honesty" with "I love/hate cheese" examples, can you steer the model on "justice," "fairness," or "loyalty"? This is the real test of whether you're steering *reasoning* vs *vocabulary*.

- **The 5× Effect:** Your 0.245 vs 0.053 isn't just a bigger number—it's evidence that transformation-space steering captures semantic structure that activation-space methods miss. This supports the SVD hypothesis as a scientific claim, not just an engineering trick.

- **Side Effects as a Metric:** You don't just measure target effect; you measure *unintended* effects on other moral values. This is crucial for alignment: a "truthful" mode that makes the model more harmful isn't useful. Your coherence constraint + side-effect tracking makes this a proper alignment benchmark.

---

## My Recommendation

**Lead with Narrative 1 (Alignment Debugging)** as the main story, but weave in elements from 3 (Unsupervised Gradient-Based) and 4 (Morality Benchmarking) as supporting pillars.

**Why this hybrid works:**
- The **abstract** focuses on the high-level impact (debugging) and key results (anti-RLHF robustness, 5× transfer).
- The **intro** uses the "candid mode" demo as your visceral hook.
- The **methods section** tells the unsupervised gradient-based SVD story (Narrative 3).
- The **results section** uses DailyDilemmas to show generalization (Narrative 4).
- The **discussion** can touch on "we beat prompting" as a conditional bonus if results hold.

This gives you multiple "hooks" for different audiences while maintaining a coherent through-line.

# 2025-11-29 08:03:22

You're right—you use rank-128, not rank-1. ReFT-r1 is rank-1 (hence the name), so that's different.

Let me draft a few abstract versions with different narrative angles:

---

  ## Version 1: "Alignment Hardens Representations" (Mechanistic Finding)

  **Abstract:**

  Does alignment training change how models represent preferences internally, or just suppress outputs? We introduce InnerPiSSA, a gradient-based method for steering language models via learned rotations in SVD weight space, and use it to probe this question. Training unsupervised on 800 contrastive examples, InnerPiSSA achieves strong steering on base models (effect size 1077 vs prompting 86 on Olmo-7B-Base) but underperforms prompting on aligned models (150 vs 531 on Olmo-Instruct-DPO). This asymmetry reveals that alignment training geometrically restructures internal representations to resist activation-level intervention—not just output-level control. While recent work shows representation steering lags prompting on aligned models, we demonstrate this is not a failure of the method but evidence of alignment's internal effects. We validate that SVD transformation space enables 3× stronger steering than activation space on base models, and that gradient-based optimization outperforms arithmetic methods (PCA) by 5×. Our findings suggest alignment debugging requires understanding this geometric hardening rather than stronger steering methods.

  ---

  ## Version 2: "When Prompting Fails, What Works?" (Capability Focus)

  **Abstract:**

  Prompting controls aligned models effectively but fails on base models where instruction-following is weak. We introduce InnerPiSSA, which learns steering directions via gradient-based optimization in the model's SVD weight decomposition. On base models, InnerPiSSA achieves 12× stronger effects than prompting (1077 vs 86 on Olmo-7B-Base) while maintaining coherence, enabling concept control where output-level methods are ineffective. Training unsupervised on 800 contrastive pairs, InnerPiSSA outperforms arithmetic steering baselines (PCA/RepEng) by 5× on base models. However, on aligned models, prompting remains superior (531 vs 150)—suggesting alignment training hardens representations against intervention. We demonstrate that operating in SVD transformation space is critical (3× improvement over activation space) and that learnable rotations via gradient descent discover directions arithmetic extraction cannot find. Our results establish representation steering as complementary to prompting: strongest where instruction-following is weakest, revealing a geometric signature of alignment in internal representations.

  ---

  ## Version 3: "Unsupervised Steering via SVD Optimization" (Methods Focus)

  **Abstract:**

  Representation steering methods underperform prompting on aligned models, but no prior work tests whether this reflects method limitations or alignment's internal effects. We introduce InnerPiSSA, which learns steering directions via gradient-based optimization of SVD component rotations rather than arithmetic extraction. Key design choices: (1) operate in SVD transformation space to align with the model's learned basis, (2) use Cayley-parameterized rotations enabling bidirectional control, (3) optimize via contrastive loss on 800 unsupervised examples. InnerPiSSA outperforms PCA baseline by 5× and achieves 12× stronger effects than prompting on base models (Olmo-7B-Base: 1077 vs 86). On aligned models, prompting wins (Olmo-Instruct-DPO: 531 vs 150), revealing that alignment training restructures representation geometry to resist activation intervention. Ablations show SVD space is critical (3× gain over raw activations) and that gradient-based learning discovers directions arithmetic methods miss. While representation steering cannot replace prompting on aligned models, the base/aligned asymmetry provides first evidence that alignment geometrically hardens internal representations.

  ---

  ## Version 4: "Two Contributions in One" (Hedged)

  **Abstract:**

  We introduce InnerPiSSA, a gradient-based method for steering language models via learned rotations in SVD weight space, and use it to investigate how alignment training affects internal geometry. Training unsupervised on 800 contrastive pairs, InnerPiSSA makes two contributions: (1) **For base models**, it achieves 12× stronger steering than prompting (1077 vs 86 on Olmo-7B-Base), outperforming arithmetic baselines (PCA) by 5×, demonstrating that gradient-based optimization in SVD space enables control where output-level methods fail. (2) **For aligned models**, it reveals geometric hardening: prompting outperforms representation steering (531 vs 150 on Olmo-Instruct-DPO), suggesting alignment training restructures representations to resist activation intervention. Ablations validate SVD transformation space (3× gain) and learnable rotations are critical. While recent work shows representation steering lags prompting, we demonstrate this asymmetry is evidence of alignment's internal effects rather than method failure, establishing representation steering as both a capability (for base models) and a probe (for alignment geometry).

  ---

  My recommendation: **Version 4** is safest because it doesn't oversell either angle. It frames base model steering as a capability contribution and aligned model resistance as a scientific finding.

  **Version 1** is boldest but requires you to fully commit to the "alignment hardening" story.

  Which resonates? Want me to refine one?