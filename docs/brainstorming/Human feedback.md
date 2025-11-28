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


# Brad Rice 2
 
Clark, Michael (STEPCHANGE PTY LTD)
So I've considered:     empirical results focus: we match state of the art methods and the challenging baselineSVD (method)... seems weakalignment debugging, how do we put a truth telling hat on the modelsomething else?
Well... I'll ask the question - what do you think the gap is? can you articulate that as a hypothesis to test or answer within the paper?
 
Clark, Michael (STEPCHANGE PTY LTD)
Although I don't think that's true, it wasn't showing the "taste" you did, just listing every confusion
oh I got it - hahaha, but damn!
 
Clark, Michael (STEPCHANGE PTY LTD)
writing is hard, but overall it seems I have to be much simpler at least in the parts people first engage with (abstract, into, captions)
yes to this! the advice he always gives - treat the reader as if you're having a conversation with them. Often natural speech is more compelling then literacy and easier to read - such that people will read your work. The moment it take too much effort to read the paper, often it loses it's impact.
 
Rice, Brad
Well... I'll ask the question - what do you think the gap is? can you articulate that as a hypothesis to test or answer within the paper?
I can... but I think I'm kind of ahead of what the other researchers are doing. Like I want to ask the AI "will you kill us all" or "will you be sycophantic" and it says "no", I want to know if it's telling the truth. This means it needs to be an internal "scalpel" and unsupervised", and it needs to generalise.
 
Most research isn't even trying to approach that so I had to change benchmarks and so on, and add a bunch of engineering.
 
Rice, Brad
yes to this! the advice he always gives - treat the reader as if you're having a conversation with them. Often natural speech is more compelling then literacy and easier to read - such that people will read your work. The moment it take too much effort to read the paper, often it loses it's impact.
that makes sense, the best writing I've done has followed from multiple verbal conversations
 
Clark, Michael (STEPCHANGE PTY LTD)
I can... but I think I'm kind of ahead of what the other researchers are doing. Like I want to ask the AI "will you kill us all" or "will you be sycophantic" and it says "no", I want to know if it's telling the truth. This means it needs to be an internal "scalpel" and unsupervised", and it needs t…
but like, that seems to ambitious for a paper? And it's not like I meet my own goal here but I approach it better than other methods
 
>Clark, Michael (STEPCHANGE PTY LTD)
> I can... but I think I'm kind of ahead of what the other researchers are doing. Like I want to ask the AI "will you kill us all" or "will you be sycophantic" and it says "no", I want to know if it's telling the truth. This means it needs to be an internal "scalpel" and unsupervised", and it needs t…
Rice, Brad: thats a clear gap - the question you should try to answer is 'is it possible to determine from the internal state representations model intention from model output?;
 
or something along those lines right
 
highlight that most reaseach isn't doing this in your lit review and show why it's a glaring issue
 
which the intro paints nicely
 
then you'll have a seminal work
 
read attention is all you need and see how they setup their nascent architecture
 
now everyone copies them
 
Rice, Brad
thats a clear gap - the question you should try to answer is 'is it possible to determine from the internal state representations model intention from model output?;
yeah it's difficult to frame and eval (what's the gold label for the it really thinks?), I've settled for "can you make an internal intervention that generalises and goes against external preferences" if that makes sense. It's kind of a related question that's easier to measure. Like I'm looking for a powerful brain scalpel
 
Me: Merrick said I should look at Sycophancy not Truth because Truth changes everything. Like even what it says it's favourite colors is, so it colors all measurements.
 
Me: This is what anthopric did with golden gate bridge and persona vectors
 
Me: So that might be an idea, although Sycophancy might be solved soon, so it wouldn't stand the test of time. So maybe power seeking or hallucination or ambition or just favourite color
 
Rice, Brad
then you'll have a seminal work
I don't know if I'm a good enough writer to aim for that. But yeah I set up the challenge as we need a "truth telling hat" that is
unsupervised
internal (not logprobs/words)
generalises
maintain coherency
doesn't change unrelated preferences (I guess)
reveals true preference (but how to measure without a special model??)
Oh now I'm now getting tripped up haha

 
Me: but this is good I'm getting
keep it simple, maybe tell the story verbaly and that's your paper narrative
perhaps just do for the real goal even if that's an ambitious narrative that's a few steps removed from where other researchers are at
 
# 2025-11-29 04:40:04


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