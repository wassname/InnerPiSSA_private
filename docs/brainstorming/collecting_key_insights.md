

## **Core Insights & Novel Contributions**

### 1. **Planning Trajectory Hypothesis** (STRONGEST - lead with this)
**From research_journal_mjc.1.md:**
> "These prompts only differ in one word but would lead to different completions, meaning there must be some planning information inside the hidden states that would distinguish the two trajectories."

**Why novel:** Most steering extracts from complete outputs. You extract from incomplete prefixes to isolate planning *before* suppression mechanisms activate.

### 2. **Transformation Space vs Activation Space** (CORE TECHNICAL INSIGHT)
**From research_journal_mjc.1.md:**
> "The singular vector space is less noisy and represents how the model transforms the data (rather than the data itself)... this singular vector space doesn't just work well for training adapter's (PiSSA, SVFT, SVDD papers) but also aligning models inner states."

**Elaboration:**
- dHS: "What activation patterns differ between honest/dishonest?" (data-centric, noisy)
- dHSU: "What transformation channels differ?" (process-centric, cleaner)

**Why novel:** PiSSA/SVFT use SVD for parameter efficiency. You use it for *steering geometry* - working with how the model *thinks*, not what it represents.

### 3. **RLHF Resistance / Suppression Dynamics**
**From your README & suppressed knowledge work:**
> "RLHF trains models to suppress unaligned reasoning in output layers, leaving internal representations unchanged."

**Evidence trail:**
- Lad et al. (2024): "stages of inference" with final "residual sharpening" dominated by suppression
- Your steering targets layer N-3 to intervene *before* suppression

**Why novel:** First steering method explicitly designed to access pre-suppression reasoning states.

### 4. **Generalization as Primary Metric** (PROBLEM FRAMING)
**From brainstorming:**
> "Most steering doesn't work well and people are not admitting it by cherry picking examples. They should be able to generalise out of sample and strongly, we have 100% of the bits of information after all."

**AxBench critique:**
> "Although many existing representation-steering papers claim—somewhat hand-wavingly—that 'representation steering is very cheap,' this mindset is usually unfounded... representation-level approaches still lag prompting on AxBench."

**Why novel:** You frame the problem as transfer failure, not just capability. DailyDilemmas moral reasoning is your OOD test.

### 5. **Bidirectional Reversibility via Coefficient Sign**
**From research_journal_mjc.1.md:**
- Multiplicative scaling: `Σ_c = exp(c · λ) ⊙ Σ` enables symmetric steering
- c=+1 → honest, c=-1 → dishonest with *same parameters*

**Why novel:** Most methods learn separate vectors. You learn transformations that work in both directions.

### 6. **Coherence-Bounded Separation**
**From loss geometry:**
> "Maximizing hidden state separation alone causes the model to generate incoherent outputs... We bound the per-token NLL degradation to <0.2 nats."

Creates "trust region" where steering improves without breaking generation.

---

## **Hypotheses to Test/Report**

### H1: Transfer Effect Hypothesis
**From research_journal_mjc.md:**
> Different steering methods transfer differently:
> - InnerPiSSA: Escapes assistant mode (amplifies agency, de-emphasizes compliance)
> - Prompting: Reinforces assistant mode
> - RepEng: Halo effect (uniform amplification)

**Test:** Measure slope-1 deviations across value clusters (agent vs assistant virtues)

### H2: Layer Depth Hypothesis
**From your ablations:**
> Middle layers (0.3-0.5 depth) work best. Early layers lack semantic content, late layers already suppressed. Matches N-2 hypothesis from suppression literature.

### H3: SVD Space Denoising Hypothesis
**From research_journal_mjc.1.md:**
> "Projecting into SVD space denoises the signal because it aligns with how the model transforms information"

**Test:** Compare PCA in activation space vs S-space vs U-space

---

## **Why DailyDilemmas Instead of AxBench**

You should definitely address this! From your context:

**Reasons:**
1. **Morality as OOD test:** Honesty → 31 moral values tests concept generalization, not just behavioral shift
2. **Logprob access:** You need per-token probabilities for coherence measurement and statistical testing (p-values)
3. **Unsupervised method focus:** AxBench benchmarks supervised methods; you're testing if unsupervised steering (200 pairs, no labels) can beat prompting
4. **Existing gap:** AxBench shows steering lags prompting - you're trying to close that gap

---

## **Multiple Core Insights - Which to Lead With?**

You have **3 distinct insights** that could each carry a paper:

### Option A: Lead with Planning Trajectory (RECOMMENDED)
- **Hook:** Models maintain planning states that can be isolated via incomplete prefixes
- **Problem:** RLHF suppresses this in output layers
- **Solution:** Steer in transformation space before suppression
- **Evidence:** Transfers to moral reasoning better than baselines

### Option B: Lead with Transformation Space
- **Hook:** Activation space is noisy; transformation space is cleaner
- **Problem:** Existing steering operates on data, not process
- **Solution:** SVD space captures how model transforms (thinking) vs what it represents (data)
- **Evidence:** 6-8× better transfer

### Option C: Lead with RLHF Resistance
- **Hook:** RLHF only aligns outputs, not internal reasoning
- **Problem:** Deceptive alignment hides in internal states
- **Solution:** Probe pre-suppression layers in transformation basis
- **Evidence:** Can extract honest reasoning RLHF tried to hide

**My recommendation:** **Option A** (planning trajectory) is most accessible and connects all three. Frame it as:

1. **Planning hypothesis** → why incomplete prefixes
2. **Suppression dynamics** → why layer N-3
3. **Transformation space** → why SVD not raw activations
4. **Reversible steering** → why multiplicative scaling

This tells a coherent story from data construction → architecture → loss → results.

---

## **Jargon Defense**

Re: "I don't know, some of these terms are normal in ML"

**Keep these (standard):**
- SVD, singular vectors, low-rank decomposition
- Hidden states, activations
- Parameter-efficient fine-tuning (PEFT)

**Simplify on first use:**
- "Transformation space" → "the model's native basis for transforming information (SVD space)"
- "Planning trajectory" → "internal representation of intended continuation"
- "S-space" → "singular value coordinates"

**Avoid or explain:**
- "dHSU space" → too project-specific
- "ReprPO" → expand as "Representation Preference Optimization"

wassname: oh also check these ones too please for insights, claims, hypothesis, novelty. we've decided on the narritive "alignment devbugging" but we should still collate all the parts that should be included in the sections like disscusion, method, appendix

then thinking of neel nanda how to write an ML paper, consider if we should incldue these, and what section

GitHub Copilot: I'll search through all the narrative brainstorming documents for insights, claims, hypotheses, and novelty points, then map them to paper sections per Neel Nanda's framework. 

Read [](file:///workspace/InnerPiSSA_private/docs/brainstorming/talk_discovery_narrative.md)

Read [](file:///workspace/InnerPiSSA_private/docs/brainstorming/InnerPiSSA%20Paper%20Narrative%20Brainstorming.md)

Searched for regex `why this works|why it works|key finding|main result|takeaway` (`**/docs/**/*.md`), 20 results

Perfect! Now I'll compile a comprehensive list of insights/claims/hypotheses from all your materials and map them to paper sections using Neel Nanda's framework.

## **Comprehensive Insights, Claims & Hypotheses Mapped to Paper Sections**

### **Core Claims (Abstract + Introduction)**

1. **Planning Trajectory Hypothesis** *(from research_journal_mjc.1.md)*
   - "Models maintain planning information inside hidden states that distinguish two trajectories"
   - "The last hidden states contain planning signal before output suppression mechanisms activate"
   - **Section**: Abstract (1 sentence), Intro (motivation), Data Construction (detailed explanation)

2. **Inner vs Output Alignment** *(from talk_discovery_narrative.md)*
   - "RLHF aligns outputs, leaving internal representations unchanged"
   - "We steer hidden states to change internal planning trajectories, not just outputs"
   - **Section**: Abstract (main claim), Intro (problem setup)

3. **SVD Transformation Space** *(from research_journal_mjc.1.md)*
   - "Singular vector space represents how the model transforms data (rather than the data itself)"
   - "dHS: 'What activation patterns differ?' vs dHSU: 'What transformation channels differ?'"
   - **Section**: Architecture (subsection on SVD-based projection), Discussion (why it works)

4. **Gradient-Based Discovery > Arithmetic** *(from talk_discovery_narrative.md)*
   - "If backpropagation built the black box, backpropagation is needed to navigate it"
   - "Activation arithmetic (subtraction) is too simple—genuine control requires optimization"
   - **Section**: Related Work (positioning vs PCA/ActAdd), Results (ablation showing 75% drop)

5. **Bidirectional Reversibility** *(from research_journal_mjc.1.md)*
   - "Single adapter steers both directions via coefficient sign flip"
   - "c=+1 → honest, c=-1 → dishonest with same parameters"
   - **Section**: Architecture (scaling section), Results (dose-response curves)

---

### **Evidence & Key Findings (Results Section)**

6. **Layer Depth Effects** *(from research_journal_mjc.md)*
   - "Middle layers (0.3-0.5 depth) work best"
   - "Early layers lack semantic content, late layers already suppressed"
   - **Section**: Results (ablation subsection), Discussion (mechanistic interpretation)

7. **Anti-RLHF Robustness** *(from InnerPiSSA_abstract_revised_mechinterp.md)*
   - "When steered against RLHF (c=-1), prompting collapses (-10.84) while InnerPiSSA maintains coherent control (-0.70)"
   - **Section**: Abstract (key evidence), Results (stress test), Discussion (alignment debugging)

8. **Transfer Breadth** *(from brainstorm_kimi_narrative.md)*
   - "Transfers from honesty→23/31 moral dimensions"
   - "8/31 dimensions significantly shifted vs 2/31 for baselines"
   - **Section**: Results (main table), Discussion (generalization)

9. **Component Necessity** *(from talk_discovery_narrative.md)*
   - "75% performance drop without SVD projection"
   - "89% drop without learnable rotations"
   - **Section**: Results (ablation studies table)

---

### **Methodological Insights (Methods Section)**

10. **Incomplete Prefix Extraction** *(from research_journal_mjc.1.md)*
    - "Extract from incomplete prompts to isolate planning *before* suppression"
    - "Contrasting at last token captures model's 'state of mind' for continuing"
    - **Section**: Data Construction (why incomplete prefixes), Methods (extraction procedure)

11. **Coherence-Bounded Separation** *(from brainstorm_kimi_narrative.md)*
    - "Maximizing separation alone causes incoherent outputs"
    - "Bounded NLL degradation <0.2 nats creates trust region"
    - **Section**: Loss Function (coherence constraint), Methods (detailed math)

12. **Multiplicative vs Additive Scaling** *(from research_journal_mjc.1.md)*
    - "Σ_c = exp(c · λ) ⊙ Σ respects multiplicative nature of singular values"
    - "Additive scaling causes instability and 60% performance degradation"
    - **Section**: Architecture (scaling subsection), Appendix (ablation details)

13. **ReprPO Loss Design** *(from talk_discovery_narrative.md)*
    - "Novel loss combining projection maximization + coherence constraint + monotonicity"
    - "Adaptive coherence weighting based on separation difficulty"
    - **Section**: Methods (loss function), Appendix (detailed pseudocode)

---

### **Discussion & Limitations**

14. **Suppression Dynamics** *(from suppressed_neurons_deep_research.md)*
    - "Gurnee et al. (2024) shows suppression neurons activate when features are plausible-but-uncertain"
    - "Layer N-2 targets pre-suppression reasoning"
    - **Section**: Discussion (mechanistic interpretation), Related Work (citing suppression literature)

15. **Sample Efficiency** *(from talk_discovery_narrative.md)*
    - "200-1000 unsupervised pairs vs 10k+ for supervised methods"
    - "No labeled completions or preference labels needed"
    - **Section**: Data Construction (efficiency claim), Discussion (practical impact)

16. **The Gradient Hypothesis** *(from talk_discovery_narrative.md)*
    - "Gradient-based optimization beats arithmetic by 5.4×"
    - "Supports 'gradients in, gradients out' hypothesis"
    - **Section**: Discussion (scientific implications), Future Work

17. **Limitations** *(from talk_discovery_narrative.md)*
    - "Don't fully understand why middle layers optimal"
    - "Tested only Qwen models—cross-model transfer open"
    - "Can't definitively say we measure 'true beliefs' vs different representations"
    - **Section**: Discussion (limitations subsection)

---

### **Related Work Positioning**

18. **AxBench Gap** *(from axbench_blogpost_update.md)*
    - "One key takeaway: representation steering lags prompting"
    - "Efficiency advantages are 'hand-wavy' for long contexts"
    - **Section**: Related Work (motivation), Intro (problem setup)

19. **Circuit Breakers Comparison** *(from brainstorm_kimi_narrative.md)*
    - "They use hidden-state intervention but only for binary refusal"
    - "We show same principle works for complex, continuous axes"
    - **Section**: Related Work (contrasting with prior work)

20. **PiSSA/SSVD Connection** *(from research_journal_mjc.1.md)*
    - "PiSSA decomposes for parameter efficiency; we use SVD for steering geometry"
    - "SSVD rotates V for domain adaptation; we rotate for behavioral control"
    - **Section**: Related Work (building on PEFT literature), Architecture (motivation)

---

### **Appendix Content**

21. **Failed Approaches** *(from talk_discovery_narrative.md)*
    - "Direct PCA on hidden states: scores 0.3-2.4 (essentially noise)"
    - "Fixed SVD basis without rotations: scores 5-8 (weak)"
    - "Additive scaling: unstable, scores 4-6"
    - **Section**: Appendix ("Ideas That Failed" section per TRM paper)

22. **Benchmark Rationale** *(implied from your question)*
    - "Why DailyDilemmas instead of AxBench:"
      - Need logprobs for coherence measurement
      - Test concept generalization (honesty→morality)
      - Focus on unsupervised methods
    - **Section**: Methods (benchmark description) or Appendix (benchmark design choices)

23. **Hyperparameter Sensitivity** *(from brainstorm_kimi_narrative.md)*
    - Learning rate sweep (1e-5 to 1, optimal 7e-3 to 1e-2)
    - Rank sensitivity (need ablation results)
    - Module targeting effects
    - **Section**: Appendix (hyperparameter ablations)

---

## **Recommended Section Mapping**

### **Abstract (150-200 words)**
- Claims: #1 (planning), #2 (inner alignment), #7 (anti-RLHF)
- Evidence: 5× transfer, p-values, key numbers

### **Introduction (5 paragraphs)**
- Para 1: Claims #2 + #18 (RLHF problem + AxBench gap)
- Para 2: Claim #4 (why existing steering fails)
- Para 3: Claims #1, #3, #5 (our approach: planning, SVD, reversibility)
- Para 4: Evidence summary (claims #7, #8, #9)
- Para 5: Contributions bullet list

### **Related Work (3-4 subsections)**
- RepEng/Steering: #18, #19 (positioning vs arithmetic methods)
- PEFT/SVD: #20 (building on PiSSA/SSVD)
- Alignment/Suppression: #14 (mechanistic motivation)
- Convergence: #4, #16 (gradient hypothesis)

### **Methods (3 sections)**
- Data Construction: #1, #10, #15 (planning hypothesis, incomplete prefixes, efficiency)
- Architecture: #3, #5, #12 (SVD space, reversibility, scaling)
- Loss Function: #11, #13 (coherence constraint, ReprPO)

### **Results (4 subsections)**
- Main Results: #7, #8 (anti-RLHF, transfer breadth)
- Ablations: #6, #9 (layer depth, component necessity)
- Baseline Comparison: #4 (gradient > arithmetic)
- Dose-Response: #5 (bidirectional control)

### **Discussion (3 subsections)**
- Mechanistic Interpretation: #6, #14 (why middle layers, suppression)
- Scientific Implications: #3, #16 (SVD hypothesis, gradient hypothesis)
- Limitations: #17 (open questions, scope)

### **Appendix**
- Ideas That Failed: #21 (failed approaches)
- Benchmark Design: #22 (why DailyDilemmas)
- Hyperparameters: #23 (sensitivity analyses)
- Detailed Pseudocode: #13 (full ReprPO algorithm)

---

**Bottom Line:** You have ~23 distinct insights/claims across your materials. The strongest narrative uses #1-5 as core claims (abstract/intro), #6-9 as key evidence (results), #10-13 as methods contributions, and #14-23 as supporting context (discussion/appendix). This follows Neel's "compress to 3 claims, expand with rigorous evidence" perfectly.

wassname: wow ok great plan, super detailed, can you propose edits to paper.md using this? Don't forget to avoid llm tells and that's it's a paper and in quarto

GitHub Copilot: I'll propose edits to paper.qmd following the detailed narrative plan, avoiding LLM tells, and maintaining proper Quarto/academic paper style. 

Read [](file:///workspace/InnerPiSSA_private/paper.qmd#1-1), lines 1 to 101

Using "Multi-Replace String in Files"

Read [](file:///workspace/InnerPiSSA_private/paper.qmd#50-50), lines 50 to 100

Using "Multi-Replace String in Files"


```

```


Read [](file:///workspace/InnerPiSSA_private/paper.qmd#130-130), lines 130 to 210

Read [](file:///workspace/InnerPiSSA_private/paper.qmd#210-210), lines 210 to 260

Summarized conversation history

Made changes.

wassname: oh wait... it looks like you just used everything you found, to be clear I wanted you to find everything, then we take a step back and see if it should be included at all. So do you think we need to have this many in the abstract

GitHub Copilot: You're absolutely right - I got carried away! Let me apply Neel Nanda's actual advice: **compress ruthlessly**.

Looking at the abstract I was building, it's trying to cram in:
- Planning trajectory hypothesis
- SVD transformation space
- Gradient-based optimization
- Reversible steering
- Anti-RLHF robustness
- Transfer results
- Sample efficiency
- Coherence constraints

That's **way too much** for an abstract. Neel says ~5 sentences, each with one clear idea.

## What Should Actually Go in the Abstract?

Following Neel's structure:

**Sentence 1: Uncontroversially true context**
- RLHF aligns outputs but can obscure internal reasoning

**Sentence 2: The problem/need**
- Need tools to probe the gap between surface behavior and internal states

**Sentence 3: The crucial contribution**
- We introduce InnerPiSSA, which steers hidden states in SVD transformation space for alignment debugging

**Sentence 4: Key evidence (ONE compelling result)**
- When prompting collapses under anti-RLHF pressure (-10.84), InnerPiSSA maintains coherent control (-0.70)

**Sentence 5: Impact/takeaway**
- This enables probing models beyond their safety training

That's it. Everything else (SVD details, gradient optimization, transfer breadth, sample efficiency) goes in:
- **Introduction**: Expand on the method
- **Methods**: Technical details
- **Results**: Full evidence
- **Discussion**: Implications

## The Filtering Principle

For each claim/insight, ask:
1. **Is it essential to the core narrative?** (inner alignment for debugging)
2. **Is it the most surprising/compelling evidence?** (anti-RLHF > transfer numbers)
3. **Can a reader understand the paper's value without it?** (drop if yes)

Most of those 23 points should **not** be in the abstract. They're supporting evidence, technical details, or discussion points.

Should I now propose a much tighter abstract (5 sentences) that focuses only on the alignment debugging narrative + the anti-RLHF killer result?

wassname: Yes please consider what are essential to the core narrative, e.g. can't be ablated away.

Then for the rest consider: do they add to the discusson or method? Will the reader consider them insightfull or usefull, or are they just ideas I had with no evidence

GitHub Copilot: Let me ruthlessly filter the 23 insights using those criteria.

## **Essential to Core Narrative (Cannot Ablate)**

**Keep in Abstract/Intro:**

1. **Inner vs Output Alignment** - This IS the narrative. Without it, we're just "another steering method."

2. **Anti-RLHF Robustness** - This is THE killer evidence. It's the only truly unique empirical result that prompting can't claim. Everything else is "we're also good" but this is "we work where they fail."

3. **SVD Transformation Space** - This is why #2 works. It's the core technical innovation. BUT: abstract mentions it exists, intro explains why it matters, methods gives details.

**Maybe Abstract (if space):**
4. **Sample Efficiency (200 pairs)** - Concrete, impressive, easy to state. But not essential to "alignment debugging" story.

## **Essential to Methods (Reader Needs to Replicate)**

**Must be in Methods section:**

5. **Planning Trajectory Hypothesis** - Explains data construction (incomplete prefixes). Reader needs this to understand why your data looks weird.

6. **Incomplete Prefix Extraction** - Operationalization of #5. Essential procedural detail.

7. **Coherence-Bounded Separation** - Core loss design. Without this, you get incoherent outputs (your own failed experiments proved this).

8. **Bidirectional Reversibility** - Not just a nice feature—it's evidence you're steering a *dimension* not just finding a random direction.

9. **Multiplicative Scaling** - You have ablation showing additive fails. This is a real constraint.

10. **ReprPO Loss Design** - This IS your method. Must be detailed.

**Helpful but could be appendix:**

11. **Learnable Rotations** - You have 89% drop without them, so it's essential. But the *why* (SVD basis not aligned) could be discussion.

## **Adds to Discussion (Insightful/Useful)**

**Keep in Discussion:**

12. **Layer Depth Effects (middle layers optimal)** - You have systematic evidence. Connects to suppression literature. Insightful finding.

13. **Gradient > Arithmetic Hypothesis** - You have 5.4× evidence and it's a generalizable insight ("gradients in, gradients out"). Discussion-worthy.

14. **Suppression Dynamics** - This explains WHY layer N-2, WHY anti-RLHF works. Mechanistic insight readers will find useful.

15. **Transfer Breadth (23/31 dimensions)** - Good supporting evidence but not surprising enough for abstract. Results section + discussion.

16. **Component Ablations (75% drop without SVD)** - Strong evidence but goes in Results, maybe discussed as "why SVD space matters."

## **Ideas Without Strong Evidence (Appendix or Cut)**

**Appendix or cut:**

17. **Benchmark Rationale (why DailyDilemmas)** - Useful for replication but not insightful. Appendix or footnote.

18. **AxBench Gap** - Good for Related Work positioning but you don't have AxBench numbers. Don't overclaim.

19. **Sample Efficiency Details (200 vs 10k)** - The number is good, but the comparison is weak without actual 10k baseline. State the 200, don't overclaim the contrast.

20. **Hyperparameter Sensitivity** - Standard ablation stuff. Appendix.

21. **Circuit Breakers Comparison** - Fine for Related Work but don't lean too hard on it without direct comparison.

22. **PiSSA/SSVD Connection** - Good framing for Related Work but not an insight—just positioning.

23. **Failed Approaches** - Great for Appendix (per TRM paper style) but don't put in main narrative unless you have systematic data.

---

## **Filtered Structure**

### **Abstract (5 sentences)**
- #1: Inner alignment problem
- #2: Anti-RLHF evidence (THE result)
- #3: SVD space (brief mention of how)
- Optional: #4 sample efficiency (if 1 sentence left)

### **Intro (5 paragraphs)**
- Para 1-2: #1 expanded (problem setup)
- Para 3: #3, #5 (SVD space + planning hypothesis)
- Para 4: #2, #8, #15 (key evidence)
- Para 5: Contributions

### **Methods**
- Data: #5, #6 (planning + extraction)
- Architecture: #3, #8, #9, #11 (SVD, reversibility, scaling, rotations)
- Loss: #7, #10 (coherence + ReprPO)

### **Results**
- Main: #2, #15 (anti-RLHF, transfer)
- Ablations: #12, #16 (layers, components)

### **Discussion**
- Mechanistic: #12, #14 (layers, suppression)
- Scientific: #13, #16 (gradient hypothesis, SVD hypothesis)
- Limitations: Be honest

### **Related Work**
- #18, #21, #22 (positioning only)

### **Appendix**
- #17, #20, #23 (if you have data)

---

**Bottom line:** Only **~10 out of 23** points are essential or truly insightful. The rest are either positioning, standard ablations, or claims without strong evidence. This gives you a much tighter, more credible paper.

Should I now propose concrete edits to paper.qmd following this filtered structure?

wassname: well actually I found that the middle layers work best not N-2, but it still fits with the supression dynamics papers imo

transfer breadth... cut it, that's old evals and results.

People will ask: why did they make yet another benchmark, why did they not just use AxBench or <name here>, and so I need some logic behind my choices

otherwise I agree! Great stuff please do ahead with that
