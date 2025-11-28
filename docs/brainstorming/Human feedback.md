Ok so the human feedback I got is way more nuanced and wisely critical, its gets me out of what I think of as - no offense - the llm bubble where every idea is "brilliant". Sometimes I just want to know what a skeptical reader will think and this human feedback captures it perfectly.



merrick:

    don't over theorise, you made svd steering, a loss to separate two vectors within coherence and monotonic constraints. and it changes the score on morality tests more than prompting and steering
    what if we change the largest SVD direction and remove it
    don't use truthfullness as it's hard to show as it's saturated and debatable and effects everything. Show it on sycophancy, or long dashes. (or one of the other moral dimensions like ambition or optimism or tradition)

# 2025-11-27 19:58:55

## Michael's Abstract

**Thursday, 27 November 2025 — 11:15 AM**

    ---

    ### Abstract Text (left side)

    The reliance on reinforcement learning (RLHF) for alignment raises a critical measurement gap: optimizing for outcomes rather than process encourages models to dissociate internal reasoning from stated outputs. We introduce InnerPiSSA, which steers hidden states in the model's native ==SVD transformation space== using gradient-based optimization on learnable rotations and singular value scaling. When steering against learned behaviors ==(coefficient = -1)==, prompting collapses to incoherent outputs (truthfulness: TBD) while InnerPiSSA maintains controlled bidirectional steering (truthfulness: TBD), evidence that we modify internal reasoning trajectories rather than output style alone. ==Trained on 800 unsupervised contrastive pairs extracted from incomplete reasoning prefixes==, InnerPiSSA achieves significantly stronger effects than PCA (TBD vs TBD) on moral reasoning transfer and outperforms prompting on anti-RLHF robustness. Critically, InnerPiSSA requires no human preference labels or model completions, operating entirely on-policy from the model's own planning trajectories. This demonstrates the strongest performance among tested methods for alignment debugging: probing what models compute internally when output-level constraints are bypassed at the representation level. While not a complete solution to alignment, this is the best-performing approach we tested for studying failure modes that output-level evaluation cannot detect.

    From <https://github.com/wassname/repeng/tree/InnerPiSSA>

    ---

    ### (Brad Rice) Reviewer Comments (right side, with arrows)

    **Comment 1** (pointing to "SVD transformation space"):
    > If the first thing I read is the abstract perhaps we should have the non-acronymised version of SVD here - singular value decomposition (SVD). (coefficient = -1) unsure you need this in the abstract - I don't have any context around it in this chunk of text and it doesn't affect the meaning of the abstract in any positive way.

    **Comment 2** (pointing to "coefficient = -1"):
    > I don't actually get what is trying to be said in this sentence. (don't know if this is my lack of knowledge in this area or if it is because it was taken out of it's context).
    >
    > Consider instead:
    > *"InnerPiSSA exhibits enhance control on the internal reasoning trajectories (as shown by blah), a distinct advantage over current methods."*

    **Comment 3** (pointing to "Trained on 800 unsupervised contrastive pairs..."):
    > Is this data set a ==well known== publicly available benchmark or custom dataset? How do I know the relevance of the results you generate? Is it possible that you've gamed your dataset to demonstrate what you wanted?

    **Comment 4** (pointing to final sentences):
    > I don't know what it is about these last two sentences but I don't 'vibe' with them. They feel a little colloquial. Within you define alignment debugging quite late in the abstract - this shouldn't be done here (I'm operating under the assumption that alignment debugging is a ==well known== problem/area in the field - if it isn't then it needs to be defined of course but should be mentioned at the top of the abstract before it's use). I also don't know if it is a style thing for me but the last sentence I don't like - It doesn't add real value into your work and terms like 'best-performing' are a little jarring for me. Perhaps a rewritten version might be;
    >
    > *"Currently, InnerPiSSA demonstrates improved performance for alignment debugging, comparative to the tested methodologies, opening up a promising avenue where future research can provide further improvements."*


My own thoughts

- This is soo complicated, how do I know all these parts are needed (ablation)
- Why didn't you just use well known datasets or benchmarks like TruthfulQA or AxBench. Not another dataset, is this because you failed on the main ones? (becuause this dataset is better, and TruthfulQA is a terrible dataset with noise, bias etc)
Why didn't you benchmark on BiPDO (because it's not as good as prompting)

# prompt

Better prompts for this mode:

"Where would you get confused reading this cold?"
"What sentences would make you pause and re-read?"
"React honestly — don't explain what I probably meant, tell me what I actually wrote"
"Be Brad — flag anything unclear without trying to fix it"