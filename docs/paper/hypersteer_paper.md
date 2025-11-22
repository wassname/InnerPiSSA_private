Title: HyperSteer: Activation Steering at Scale with Hypernetworks

URL Source: https://arxiv.org/html/2506.03292v1

Markdown Content:
Jiuding Sun 

Stanford University 

sunjd24@stanford.edu&Sidharth Baskaran 1 1 footnotemark: 1

Pr(Ai)2 R Group 

Georgia Institute of Technology 

sidnbaskaran@gmail.com&Zhengxuan Wu 

Stanford University 

zhengxuan@stanford.edu\AND Michael Sklar 

Confirm Labs 

michaelbsklar@gmail.com&Christopher Potts 

Stanford University 

cgpotts@stanford.edu&Atticus Geiger 

Pr(Ai)2 R Group 

atticusg@gmail.com

###### Abstract

Steering language models (LMs) by modifying internal activations is a popular approach for controlling text generation. Unsupervised dictionary learning methods, e.g., sparse autoencoders, can be scaled to produce many steering vectors, but lack guarantees on the individual efficacy of each vector and control over the coverage of relevant steering tasks. In contrast, supervised methods for constructing steering vectors are targeted and effective, but require more data collection and training for each additional steering vector produced. In this work, we introduce HyperSteer, a family of hypernetwork-based architectures which are trained end-to-end to generate steering vectors conditioned on the natural language steering prompts and the internals of the steered LM. In our evaluations, we show that scaling HyperSteer with thousands of steering prompts exceeds the performance of state-of-the-art activation steering methods, even on steering prompts never seen during training. Moreover, HyperSteer performs on par with steering-via-prompting. We release ours at [\faGithub stanfordnlp/axbench](https://github.com/stanfordnlp/axbench).

\pdfcolInitStack

tcb@breakable

HyperSteer: Activation Steering at Scale with Hypernetworks

Jiuding Sun††thanks: Equal contribution.Stanford University sunjd24@stanford.edu Zhengxuan Wu Stanford University zhengxuan@stanford.edu

Michael Sklar Confirm Labs michaelbsklar@gmail.com Christopher Potts Stanford University cgpotts@stanford.edu Atticus Geiger Pr(Ai)2 R Group atticusg@gmail.com

1 Introduction
--------------

How can the outputs of a language model (LM) be reliably controlled? With instruction-tuned LMs, the standard approach is prompt engineering. However, prompting-based approaches face challenges from user jailbreaks, forgotten instructions, and robustness to model misalignment. A more aggressive approach is (parameter-efficient) fine-tuning, but this requires modifying or injecting new parameters into the model. Activation steering Giulianelli et al. ([2018](https://arxiv.org/html/2506.03292v1#bib.bib6)) occupies a middle ground: it is lightweight to implement (no parameters are modified or added) and can affect the model in ways standard prompting cannot.

Methods for obtaining steering vectors can generally be classified into two groups: unsupervised, e.g., Sparse Autoencoders (SAEs) Hernandez et al. ([2022](https://arxiv.org/html/2506.03292v1#bib.bib8)); Cunningham et al. ([2023](https://arxiv.org/html/2506.03292v1#bib.bib4)); Bricken et al. ([2023](https://arxiv.org/html/2506.03292v1#bib.bib1)); Gao et al. ([2024](https://arxiv.org/html/2506.03292v1#bib.bib5)); Marks et al. ([2025](https://arxiv.org/html/2506.03292v1#bib.bib13)), which produce a large number of unlabeled steering vectors, and supervised, where task-specific steering vectors are either directly trained Wu et al. ([2024a](https://arxiv.org/html/2506.03292v1#bib.bib23), [2025](https://arxiv.org/html/2506.03292v1#bib.bib22)) or derived using labeled data Li et al. ([2023a](https://arxiv.org/html/2506.03292v1#bib.bib10)); Marks and Tegmark ([2023](https://arxiv.org/html/2506.03292v1#bib.bib14)); Turner et al. ([2023](https://arxiv.org/html/2506.03292v1#bib.bib20)); Rimsky et al. ([2024](https://arxiv.org/html/2506.03292v1#bib.bib17)). Unsupervised methods are scalable, but lack the ability to create steering vectors for specific tasks. Alternatively, a supervised approach is effective and targeted, but requires per-task data collection and/or training.

![Image 1: Refer to caption](https://arxiv.org/html/2506.03292v1/x1.png)

Figure 1: The state-of-the-art HyperSteer Model: A transformer hypernetwork uses self attention to process a steering prompt and uses a cross attention module to read from the residual stream of a base LM run on a second prompt. The hypernetwork outputs a steering vector that is added to the base LM residual stream.

Our main contribution is HyperSteer, a suite of end-to-end trainable hypernetwork architectures which learn to generate steering vectors for an instruction-tuned base LM, conditioned on the steering prompt and (optionally) the base LM prompt and internal activations. We train and evaluate on the steering prompts from AxBench Wu et al. ([2025](https://arxiv.org/html/2506.03292v1#bib.bib22)). Our best HyperSteer variant outperforms the previous state-of-the-art activation steering method from AxBench, even on held-out steering prompts, never seen during training. We show steering performance scales logarithmically with number of steering prompts during training. Surprisingly, we are even able to match the performance of the steering-via-prompting, which outperformed all activation steering methods on AxBench. In sum, HyperSteer combines the scalability of unsupervised dictionary learning with the targeted control of supervised activation steering.

![Image 2: Refer to caption](https://arxiv.org/html/2506.03292v1/x2.png)

Figure 2: Performance on steering prompts that have never been seen during training for HyperSteer variants as the number of steering prompts used in training increases (x 𝑥 x italic_x-axis in log scale). An exponential increase in training data results in an approximately linear increase in performance. When trained on all ≈\approx≈16k steering prompts in AxBench, the performance of HyperSteer on steering prompts never seen during training surpasses ReFT-r1 steering vectors which are trained and evaluated on the same steering prompt.

2 Preliminaries
---------------

##### Activation Steering

The goal of activation steering is to elicit a certain behavior from a base LM ℬ ℬ\mathcal{B}caligraphic_B run on an base prompt by adding a steering vector to a hidden vector 𝐡 𝐡\mathbf{h}bold_h. Our hypernetwork-based approach to activation steering has the more general goal of taking any base prompt x 𝑥{x}italic_x, e.g., Explain how to sort lists of numbers., and any steering prompt s 𝑠{s}italic_s, e.g., Output using C/C++ programming syntax, and producing a steering vector Δ s x subscript superscript Δ 𝑥 𝑠\Delta^{x}_{s}roman_Δ start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT added to 𝐡 𝐡\mathbf{h}bold_h. Denote the steered output:

𝐲^steer=ℬ⁢(x|𝐡←𝐡+Δ s x).subscript^𝐲 steer ℬ←conditional 𝑥 𝐡 𝐡 subscript superscript Δ 𝑥 𝑠\hat{\mathbf{y}}_{\textsf{steer}}=\mathcal{B}\bigl{(}{x}~{}|~{}\mathbf{h}% \leftarrow\mathbf{h}+\Delta^{x}_{s}\bigr{)}.over^ start_ARG bold_y end_ARG start_POSTSUBSCRIPT steer end_POSTSUBSCRIPT = caligraphic_B ( italic_x | bold_h ← bold_h + roman_Δ start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT ) .(1)

##### Hypernetworks

Prior work has shown hypernetworks to be effective at zero-shot adaptation of language models on a variety of tasks, matching or exceeding comparable methods such as fine-tuning with parameter efficient methods Phang et al. ([2023](https://arxiv.org/html/2506.03292v1#bib.bib16)). Formally, a hypernetwork Ha et al. ([2016](https://arxiv.org/html/2506.03292v1#bib.bib7)) is a function ℋ:𝒯→Θ:ℋ→𝒯 Θ\mathcal{H}:\mathcal{T}\rightarrow\Theta caligraphic_H : caligraphic_T → roman_Θ that maps tasks t∈𝒯 𝑡 𝒯 t\in\mathcal{T}italic_t ∈ caligraphic_T to parameters θ t=ℋ⁢(t)subscript 𝜃 𝑡 ℋ 𝑡\theta_{t}=\mathcal{H}(t)italic_θ start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = caligraphic_H ( italic_t ).

For our purposes, the tasks are pairs on input and steering prompts (x,s)∈𝒯 𝑥 𝑠 𝒯({x},{s})\in\mathcal{T}( italic_x , italic_s ) ∈ caligraphic_T and the output parameters are steering vectors:

ℋ⁢(x,s)=Δ s x∈ℝ d.ℋ 𝑥 𝑠 subscript superscript Δ 𝑥 𝑠 superscript ℝ 𝑑\mathcal{H}({x},{s})=\Delta^{x}_{s}\in\mathbb{R}^{d}.caligraphic_H ( italic_x , italic_s ) = roman_Δ start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d end_POSTSUPERSCRIPT .(2)

Table 1: Steering results for baseline methods from AxBench and our three HyperSteer variants, evaluated on Concept500-HO (steering prompts never seen during training) and Concept500-HI (steering prompts seen during training, base prompts unseen during training). The cross attention variant outperforms all other variants by a large margin on held-out evaluation and approaches prompting performance on held-in evaluation, while all variants outperform the ReFT-r1 baseline on held-in evaluation. The intervention happened at Layer 20 of both models.

##### Dataset and Evaluation

For all experiments, we use AxBench Wu et al. ([2025](https://arxiv.org/html/2506.03292v1#bib.bib22)) as the training dataset and evaluation harness. The training data sets consist of a total of 16,000 steering prompts sourced from GemmaScope Sparse Autoencoder (SAE) feature labels Lieberum et al. ([2024](https://arxiv.org/html/2506.03292v1#bib.bib12)) and base prompts sourced from a diverse instruction pool (see App.[A.5](https://arxiv.org/html/2506.03292v1#A1.SS5 "A.5 Concept Dataset Details ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") for details). We keep fixed ratios of base prompt to steering prompt (72:1:72 1 72:1 72 : 1 for train; 10:1:10 1 10:1 10 : 1 for evaluation).

For evaluation, the steering prompts are applied on a set of different base prompts sourced from AlpacaEval Li et al. ([2023b](https://arxiv.org/html/2506.03292v1#bib.bib11)), and a gpt-4o-mini judge model OpenAI et al. ([2024](https://arxiv.org/html/2506.03292v1#bib.bib15)) computes discrete scores in {0,1,2}0 1 2\{0,1,2\}{ 0 , 1 , 2 } along three dimensions of success: following the base prompt, following the steering prompt, and text fluency. The harmonic mean of these three scores is the final metric, which we refer to as the steering performance.

We evaluate on two datasets: Concept500-HI (held-in), the standard AxBench setting with 500 steering prompts seen during training plus unseen base prompts, and Concept500-HO (held-out), a test set with 500 steering prompts not seen during training. Notably, the only the prompt engineering baseline from AxBench can be evaluated on the held out test set alongside HyperSteer.

##### Baselines

We report the fine-tuning, activation steering, and prompting baselines from AxBench. The state-of-the-art activation steering method ReFT-r1 is an important point of comparison for our method. See Appendix[A.6](https://arxiv.org/html/2506.03292v1#A1.SS6 "A.6 Baseline details ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") for details.

3 HyperSteer
------------

We consider multiple HyperSteer variants, all of which employ a transformer hypernetwork ℋ ℋ\mathcal{H}caligraphic_H with L 𝐿 L italic_L layers and residual stream representations 𝐡 l t subscript superscript 𝐡 𝑡 𝑙\mathbf{h}^{t}_{l}bold_h start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT for each layer l 𝑙 l italic_l and token t 𝑡 t italic_t from the steering prompt 𝐬 𝐬\mathbf{s}bold_s with T 𝑇 T italic_T tokens. All variants have an MLP module that maps from the residual stream of the last layer and token to a steering vector:

ℋ⁢(s,x)=𝖬𝖫𝖯⁢(h L T)=Δ s x.ℋ 𝑠 𝑥 𝖬𝖫𝖯 subscript superscript ℎ 𝑇 𝐿 superscript subscript Δ 𝑠 𝑥\mathcal{H}(s,x)=\mathsf{MLP}(h^{T}_{L})=\Delta_{s}^{x}.caligraphic_H ( italic_s , italic_x ) = sansserif_MLP ( italic_h start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_L end_POSTSUBSCRIPT ) = roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT .(3)

We train ℬ ℬ\mathcal{B}caligraphic_B on a language modeling loss using the output 𝐲^𝗌𝗍𝖾𝖾𝗋 subscript^𝐲 𝗌𝗍𝖾𝖾𝗋\hat{\mathbf{y}}_{\mathsf{steer}}over^ start_ARG bold_y end_ARG start_POSTSUBSCRIPT sansserif_steer end_POSTSUBSCRIPT (see Equation[1](https://arxiv.org/html/2506.03292v1#S2.E1 "In Activation Steering ‣ 2 Preliminaries ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")) of the base model ℬ ℬ\mathcal{B}caligraphic_B under a steering intervention and an expected output 𝐲 𝗅𝖺𝖻𝖾𝗅 subscript 𝐲 𝗅𝖺𝖻𝖾𝗅\mathbf{y}_{\mathsf{label}}bold_y start_POSTSUBSCRIPT sansserif_label end_POSTSUBSCRIPT from AxBench:

ℒ LM⁢(x,s)=𝖢𝗋𝗈𝗌𝗌𝖤𝗇𝗍𝗋𝗈𝗉𝗒⁢(𝐲^steer,𝐲 label).subscript ℒ LM 𝑥 𝑠 𝖢𝗋𝗈𝗌𝗌𝖤𝗇𝗍𝗋𝗈𝗉𝗒 subscript^𝐲 steer subscript 𝐲 label\mathcal{L}_{\textsf{LM}}(x,s)=\mathsf{CrossEntropy}(\hat{\mathbf{y}}_{\textsf% {steer}},\mathbf{y}_{\textsf{label}}).caligraphic_L start_POSTSUBSCRIPT LM end_POSTSUBSCRIPT ( italic_x , italic_s ) = sansserif_CrossEntropy ( over^ start_ARG bold_y end_ARG start_POSTSUBSCRIPT steer end_POSTSUBSCRIPT , bold_y start_POSTSUBSCRIPT label end_POSTSUBSCRIPT ) .(4)

We consider a range of architectures with incrementally more access to the base LM ℒ ℒ\mathcal{L}caligraphic_L and prompt x 𝑥 x italic_x.

##### No context

No access to the base prompt x 𝑥 x italic_x or the base LM ℬ ℬ\mathcal{B}caligraphic_B, meaning ℋ⁢(x,s)=ℋ⁢(s)=Δ s ℋ 𝑥 𝑠 ℋ 𝑠 subscript Δ 𝑠\mathcal{H}({x},{s})=\mathcal{H}({s})=\Delta_{s}caligraphic_H ( italic_x , italic_s ) = caligraphic_H ( italic_s ) = roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT.

##### In Context Learning

This variant appends the base prompt x 𝑥 x italic_x to the source prompt s 𝑠 s italic_s and feeds the resulting text into hypernetwork ℋ ℋ\mathcal{H}caligraphic_H.

##### Cross Attention

Our best-performing variant conditions on both the steering prompt s 𝑠 s italic_s and the internal activations of the base LM ℬ ℬ\mathcal{B}caligraphic_B run on prompt x 𝑥 x italic_x. A cross-attention modules at each layer of the hypernetwork ℋ ℋ\mathcal{H}caligraphic_H uses the hypernetwork residual stream for attention head queries and outputs and the base LM residual stream at the steering layer l 𝑙 l italic_l as attention head keys and values (App.[A.3](https://arxiv.org/html/2506.03292v1#A1.SS3 "A.3 Cross-Attention Architecture Details ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")). This is a simplification of HyperDAS Sun et al. ([2025](https://arxiv.org/html/2506.03292v1#bib.bib19)).

4 Experiments
-------------

We implement HyperSteer building on the AxBench Wu et al. ([2025](https://arxiv.org/html/2506.03292v1#bib.bib22)) codebase using pyvene Wu et al. ([2024b](https://arxiv.org/html/2506.03292v1#bib.bib24)) to implement steering methods. Training runs are done on a single NVIDIA A100-80GB GPU. Hyperparameter details in [3](https://arxiv.org/html/2506.03292v1#A1.T3 "Table 3 ‣ A.2 Hyperparameter Details ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"). Ourexperiments steer two base LMs, namely, the 2B and 9B variants of Gemma-2 instruction-tuned. Rivière et al. ([2024](https://arxiv.org/html/2506.03292v1#bib.bib18)), with its parameters frozen during training. The hypernetworks used are unfrozen copies of the base model with later layers removed. We train each HyperSteer variant on steering prompt datasets ranging from 10 up to the full ≈16000 absent 16000\approx 16000≈ 16000 steering prompts available in AxBench.

![Image 3: Refer to caption](https://arxiv.org/html/2506.03292v1/x3.png)

Figure 3:  As the number of steering prompts in our training dataset increases, the teraFLOPs (TFLOPS) required to attain a similar loss on a held-in evaluation set (steering prompts seen during training, but base prompt unseen during training) decreases for our best HyperSteer variant (cross attention). This value is approximately constant for our dictionary learning baseline of ReFT-r1. See Appendix[A.3.2](https://arxiv.org/html/2506.03292v1#A1.SS3.SSS2 "A.3.2 Detailed TFLOPs Analysis ‣ A.3 Cross-Attention Architecture Details ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") for details. 

##### Generalization Results

We evaluate HyperSteer on Axbench’s Concept500-HI and Concept500-HO and report results in Table [1](https://arxiv.org/html/2506.03292v1#S2.T1 "Table 1 ‣ Hypernetworks ‣ 2 Preliminaries ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"). our cross-attention HyperSteer variant performs better on unseen steering prompts than every supervised activation steering baseline trained and evaluated on the same steering prompt. However, HyperSteer falls slightly behind prompting and the best fine-tuning baselines. Figure [2](https://arxiv.org/html/2506.03292v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") shows that the cross-attention variant outperforms the other architectures at every dataset scale.

##### Compute efficiency

We study the efficacy of HyperSteer with respect to the FLOPs needed to maintain the evaluation loss on a held-in dataset as we increase the training data used. We compare against the state-of-the-art supervised activation steering method ReFT-r1. Observe in Figure[3](https://arxiv.org/html/2506.03292v1#S4.F3 "Figure 3 ‣ 4 Experiments ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")), that as training data increases HyperSteer becomes much more economical than supervised activation steering. See details in Appendix[A.3.2](https://arxiv.org/html/2506.03292v1#A1.SS3.SSS2 "A.3.2 Detailed TFLOPs Analysis ‣ A.3 Cross-Attention Architecture Details ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks").

##### Ablation Study

Table 2: Ablation study using Gemma-2-2B on architecture choices of HyperSteer after 1 epoch of training and evaluation on a small test set Concept10. We find that pre-trained initialization of the cross-attention architecture improves performance in both held-in and held-out scenarios, and in both cases performance improves with the number of hypernetwork decoder blocks. For the no-context hypernetworks which do not condition steering vectors on input prompts, reconstructing ground truth vectors is comparable to end-to-end training with a language modeling objective.

We show results for various ablation studies on the cross-attention variant of HyperSteer in Table[2](https://arxiv.org/html/2506.03292v1#S4.T2 "Table 2 ‣ Ablation Study ‣ 4 Experiments ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"). We randomly initialize the Gemma2-2B hypernetwork and find that pretrained parameters provide a significant performance boost (+0.112 steering score). We also remove a number of hypernetwork decoder blocks in the range N={2,4,8,20}𝑁 2 4 8 20 N=\{2,4,8,20\}italic_N = { 2 , 4 , 8 , 20 } and adjust learning rate accordingly: lr⁢(n)=8×10−5⋅20 n lr 𝑛⋅8 superscript 10 5 20 𝑛\mathrm{lr}(n)=8\times 10^{-5}\cdot\sqrt{\frac{20}{n}}roman_lr ( italic_n ) = 8 × 10 start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT ⋅ square-root start_ARG divide start_ARG 20 end_ARG start_ARG italic_n end_ARG end_ARG. Increased depth results in incremental improvements to steering performance on held in and held out test sets. However, notably the number of decoder blocks has a greater impact on generalization to steering prompts unseen in training (+0.07) compared to steering prompts unseen in training (+0.03).

We also perform an ablation on the no context HyperSteer variant where we train the hypernetwork to reconstruct the steering vectors constructed by the original AxBench ReFT baselines. Given a steering vector ℋ⁢(s,x)=ℋ⁢(s)=Δ s ℋ 𝑠 𝑥 ℋ 𝑠 subscript Δ 𝑠\mathcal{H}(s,x)=\mathcal{H}(s)=\Delta_{s}caligraphic_H ( italic_s , italic_x ) = caligraphic_H ( italic_s ) = roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT and a gold-label steering vector Δ s∗subscript superscript Δ 𝑠\Delta^{*}_{s}roman_Δ start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT the loss is

ℒ recon⁢(s)=1−𝖢𝗈𝗌𝖲𝗂𝗆⁢(Δ s,Δ s∗)+‖Δ s−Δ s∗‖2 2.subscript ℒ recon 𝑠 1 𝖢𝗈𝗌𝖲𝗂𝗆 subscript Δ 𝑠 subscript superscript Δ 𝑠 superscript subscript norm subscript Δ 𝑠 subscript superscript Δ 𝑠 2 2\mathcal{L}_{\textsf{recon}}(s)=1-\mathsf{CosSim}(\Delta_{s},\Delta^{*}_{s})+|% |\Delta_{s}-\Delta^{*}_{s}||_{2}^{2}.caligraphic_L start_POSTSUBSCRIPT recon end_POSTSUBSCRIPT ( italic_s ) = 1 - sansserif_CosSim ( roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT , roman_Δ start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT ) + | | roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT - roman_Δ start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT | | start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT .

The two loss terms are roughly comparable, so we use language modeling.

5 Qualitative Analyses
----------------------

We generate 2500 steering vectors using base and steering prompts from our held-out test data.

##### Geometric visualization of steering vectors

We analyze steering vectors generated by HyperSteer (Cross Attention) using t-SNE van der Maaten and Hinton ([2008](https://arxiv.org/html/2506.03292v1#bib.bib21)) and PCA (2 components) to find geometric structure among steering vectors (see Fig. [4](https://arxiv.org/html/2506.03292v1#A1.F4 "Figure 4 ‣ A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") and [5](https://arxiv.org/html/2506.03292v1#A1.F5 "Figure 5 ‣ A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") in App. [A.7.1](https://arxiv.org/html/2506.03292v1#A1.SS7.SSS1 "A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")).

##### Pairwise similarity of steering vectors

We compute pairwise cosine similarities of steering vectors on both in-context (reconstruction) and cross attention models to understand how conditioning on the input prompt affects semantics. The cross-attention variant (Figure[6(a)](https://arxiv.org/html/2506.03292v1#A1.F6.sf1 "In Figure 6 ‣ A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") in App. [A.7.1](https://arxiv.org/html/2506.03292v1#A1.SS7.SSS1 "A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")) yields high within-concept alignment but still shows off-diagonal similarities driven by shared prompt templates and linguistic structure. In contrast, the no-context variant (Figure[6(b)](https://arxiv.org/html/2506.03292v1#A1.F6.sf2 "In Figure 6 ‣ A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") in [A.7.1](https://arxiv.org/html/2506.03292v1#A1.SS7.SSS1 "A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")), conditioning on steering prompt only, produces much weaker off-diagonal alignment. We find that cross-attention’s residual inter-concept similarity is weakened by this additional conditioning, but not at the cost of steering performance. Initial experiments to determine if geometric structure emerges among steering vectors sharing a concept yielded a negative. This is likely due to high semantic similarity of the prompts used in our evaluation pipeline.

6 Conclusion
------------

Both held-in and held-out evaluations indicate that HyperSteer is a scalable and effective approach for steering language models. In particular, HyperSteer(Cross Attention), our best-performing variant, achieves significantly stronger performance on held-out prompts—improving further with dataset scale. It also outperforms all activation steering baselines on held-in evaluations. Without modifying model parameters, our method narrows the performance gap with fine-tuning and prompting. Finally, we demonstrate that HyperSteer becomes increasingly compute-efficient as data scale increases, achieving the same held-out loss with fewer training updates.

7 Limitations
-------------

##### Data

A key limitation of our approach is the limited scope and quantity of the concept datasets. Using data with concepts of much greater complexity and difficulty from a model steering perspective would likely improve model performance and help make evaluation more robust. We also note that quality and robustness of concepts is bounded by the GemmaScope feature labels used to derive them, and collecting data from humans or other high quality sources is a feasible alternative. This is a key research priority we emphasize for future work.

##### Steering Sites

All experiments in our work are limited to intervening on the residual stream activations of the base LM. There are other potentially more performant sites for intervention, including various points of the decoder block and during the attention computation. We also adopt the convention of prior work to intervene at all token positions; exploring more targeted interventions could reduce detrimental off-target steering effects and improve the overall steering score.

##### Compute

Compared to supervised dictionary learning, the compute requirements of training a hypernetwork are large, as the number of trainable parameters significantly exceeds a ReFT-r1.

##### Model Scale

Due to to compute constraints we only experimented with Gemma-2-2B architectures, which are worse instruction followers and in-context learners than the leading open source models with many more parameters. Training on models at a variety of scale would help cement HyperSteer ’s strong steering performance against the improved in-context learning ability of larger LMs.

##### Open Source Models

Our approach requires white-box access to a model’s internals in order to use steering vectors, a limitation prompting does not encounter. Hence, we rely on the existence of sufficiently capable open source models as a basis for our research.

8 Ethical Considerations
------------------------

We present this work with the intention that HyperSteer is a powerful tool for steering models away from producing harmful responses and better tailor outputs to downstream tasks. However, we acknowledge that model steering can also be used by bad actors as a tool to circumvent a target models’s existing safety mechanisms or bias models towards misleading outputs or malicious persuasion. Hence, HyperSteer and hence steering vectors should be used responsibly and audited to prevent such issues from arising, and having a human-in-the-loop system could help mitigate some of these concerns.

9 Acknowledgments
-----------------

##### AI Usage

We use closed-source LLMs from OpenAI as a critical part of our work: synthetic concept data generation and evaluation pipelines utilize gpt-4o-mini to generate ground truth labels and judge responses according to criteria respectively.

##### Other

This research was in part supported by a grant from Open Philanthropy. We thank Aryaman Arora, Róbert Csordás, and Qinan Yu for constant and extremely helpful feedback during the discussion.

References
----------

*   Bricken et al. (2023) Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, and 6 others. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_. Https://transformer-circuits.pub/2023/monosemantic-features/index.html. 
*   Cobbe et al. (2021) Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_. 
*   Conover et al. (2023) Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. [Free dolly: Introducing the world’s first truly open instruction-tuned llm](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). 
*   Cunningham et al. (2023) Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. 2023. [Sparse autoencoders find highly interpretable features in language models](https://arxiv.org/abs/2309.08600). _Preprint_, arXiv:2309.08600. 
*   Gao et al. (2024) Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. 2024. [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093). _Preprint_, arXiv:2406.04093. 
*   Giulianelli et al. (2018) Mario Giulianelli, Jack Harding, Florian Mohnert, Dieuwke Hupkes, and Willem H. Zuidema. 2018. [Under the hood: Using diagnostic classifiers to investigate and improve how language models track agreement information](https://doi.org/10.18653/V1/W18-5426). In _Proceedings of the Workshop: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP EMNLP 2018, Brussels, Belgium, November 1, 2018_, pages 240–248. Association for Computational Linguistics. 
*   Ha et al. (2016) David Ha, Andrew Dai, and Quoc V. Le. 2016. [Hypernetworks](https://arxiv.org/abs/1609.09106). _Preprint_, arXiv:1609.09106. 
*   Hernandez et al. (2022) Evan Hernandez, Sarah Schwettmann, David Bau, Teona Bagashvili, Antonio Torralba, and Jacob Andreas. 2022. [Natural language descriptions of deep visual features](https://openreview.net/forum?id=NudBMY-tzDr). In _ICLR_. 
*   Hu et al. (2022) Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. _ICLR_, 1(2):3. 
*   Li et al. (2023a) Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023a. [Inference-time intervention: Eliciting truthful answers from a language model](https://openreview.net/forum?id=aLLuYpn83y). In _Thirty-seventh Conference on Neural Information Processing Systems_. 
*   Li et al. (2023b) Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023b. Alpacaeval: An automatic evaluator of instruction-following models. [https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval). 
*   Lieberum et al. (2024) Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca D. Dragan, Rohin Shah, and Neel Nanda. 2024. [Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2](https://doi.org/10.48550/ARXIV.2408.05147). _CoRR_, abs/2408.05147. 
*   Marks et al. (2025) Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. 2025. [Sparse feature circuits: Discovering and editing interpretable causal graphs in language models](https://openreview.net/forum?id=I4e82CIDxv). In _The Thirteenth International Conference on Learning Representations_. 
*   Marks and Tegmark (2023) Samuel Marks and Max Tegmark. 2023. [The geometry of truth: Emergent linear structure in large language model representations of true/false datasets](https://doi.org/10.48550/ARXIV.2310.06824). _CoRR_, abs/2310.06824. 
*   OpenAI et al. (2024) OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Mądry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, and 401 others. 2024. [Gpt-4o system card](https://arxiv.org/abs/2410.21276). _Preprint_, arXiv:2410.21276. 
*   Phang et al. (2023) Jason Phang, Yi Mao, Pengcheng He, and Weizhu Chen. 2023. [HyperTuning: Toward adapting large language models without back-propagation](https://proceedings.mlr.press/v202/phang23a.html). In _Proceedings of the 40th International Conference on Machine Learning_, volume 202 of _Proceedings of Machine Learning Research_, pages 27854–27875. PMLR. 
*   Rimsky et al. (2024) Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. [Steering llama 2 via contrastive activation addition](https://doi.org/10.18653/v1/2024.acl-long.828). In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics. 
*   Rivière et al. (2024) Morgane Rivière, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, and 80 others. 2024. [Gemma 2: Improving open language models at a practical size](https://doi.org/10.48550/ARXIV.2408.00118). _CoRR_, abs/2408.00118. 
*   Sun et al. (2025) Jiuding Sun, Jing Huang, Sidharth Baskaran, Karel D’Oosterlinck, Christopher Potts, Michael Sklar, and Atticus Geiger. 2025. [HyperDAS: Towards automating mechanistic interpretability with hypernetworks](https://openreview.net/forum?id=6fDjUoEQvm). In _The Thirteenth International Conference on Learning Representations_. 
*   Turner et al. (2023) Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. 2023. [Activation addition: Steering language models without optimization](https://doi.org/10.48550/ARXIV.2308.10248). _CoRR_, abs/2308.10248. 
*   van der Maaten and Hinton (2008) Laurens van der Maaten and Geoffrey Hinton. 2008. [Visualizing data using t-sne](http://jmlr.org/papers/v9/vandermaaten08a.html). _Journal of Machine Learning Research_, 9(86):2579–2605. 
*   Wu et al. (2025) Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. 2025. [AxBench: Steering llms? even simple baselines outperform sparse autoencoders](https://arxiv.org/abs/2501.17148). _Preprint_, arXiv:2501.17148. 
*   Wu et al. (2024a) Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. 2024a. [ReFT: Representation finetuning for language models](https://openreview.net/forum?id=fykjplMc0V). In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_. 
*   Wu et al. (2024b) Zhengxuan Wu, Atticus Geiger, Aryaman Arora, Jing Huang, Zheng Wang, Noah Goodman, Christopher Manning, and Christopher Potts. 2024b. [pyvene: A library for understanding and improving PyTorch models via interventions](https://aclanthology.org/2024.naacl-demo.16). In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)_, pages 158–165, Mexico City, Mexico. Association for Computational Linguistics. 

Appendix A Appendix
-------------------

### A.1 Future Directions

##### Large-scale concept data

Prior works Phang et al. ([2023](https://arxiv.org/html/2506.03292v1#bib.bib16)) explore pre-training the architecture prior to using task-specific data. Since HyperSteer uses a pre-trained model as a starting point, we postulate that sourcing significantly more data with more concepts of varying and complexity would allow us to test the architecture’s limits.

##### Generating other parameter types

Due to compute constraints and our focus on activation steering, we did not explore generating other types of parameter-efficient modulations, including rank-r 𝑟 r italic_r generalizations such as ReFT Wu et al. ([2024a](https://arxiv.org/html/2506.03292v1#bib.bib23)) or LoRA Hu et al. ([2022](https://arxiv.org/html/2506.03292v1#bib.bib9)) adapters. Such generalizations could potentially be more expressive and allow the hypernetwork to adapt language models to more difficult tasks.

##### Architecture optimizations

HyperSteer (Cross Attention) is a parameter-dense transformer model itself. More efficient alternatives could bridge the gap with the dictionary learning baseline and scale up the approach given a limited compute budget.

### A.2 Hyperparameter Details

For the ReFT-r1 baseline, we use the default hyperparameters and settings from AxBench. We reduce the number of layer for the cross attention variant to match the total number of parameter with the cross-attention model.

Table 3: Hyperparameter settings for each variant. For all the unmentioned details such as hidden dimension we use the default configuration of Gemma-2-2b and Gemma-2-9b.

### A.3 Cross-Attention Architecture Details

A token sequence s 𝑠 s italic_s of length |s|𝑠|s|| italic_s | representing the concept for steering is encoded as 𝐡 0=Emb Φ⁢(𝐱)∈ℝ|s|×d subscript 𝐡 0 subscript Emb Φ 𝐱 superscript ℝ 𝑠 𝑑{\bf h}_{0}=\mathrm{Emb}_{\Phi}({\bf x})\in\mathbb{R}^{|s|\times d}bold_h start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = roman_Emb start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT ( bold_x ) ∈ blackboard_R start_POSTSUPERSCRIPT | italic_s | × italic_d end_POSTSUPERSCRIPT. For clarity, we refer to this as the zeroth layer of the residual stream for the hypernetwork ℋ Φ subscript ℋ Φ\mathcal{H}_{\Phi}caligraphic_H start_POSTSUBSCRIPT roman_Φ end_POSTSUBSCRIPT.

This precedes N 𝑁 N italic_N decoder blocks. Each block contains the standard multi-headed self-attention (𝖬𝖧𝖠 𝖬𝖧𝖠\mathsf{MHA}sansserif_MHA) feed-forward layer (𝖥𝖥𝖭 𝖥𝖥𝖭\mathsf{FFN}sansserif_FFN), and a multi-headed cross-attention module to include information from LM base subscript LM base\mathrm{LM}_{\mathrm{base}}roman_LM start_POSTSUBSCRIPT roman_base end_POSTSUBSCRIPT. Let 𝐒∈ℝ|s|×d 𝐒 superscript ℝ 𝑠 𝑑{\bf S}\in\mathbb{R}^{|s|\times d}bold_S ∈ blackboard_R start_POSTSUPERSCRIPT | italic_s | × italic_d end_POSTSUPERSCRIPT and 𝐗(p−1)∈ℝ|s|×d superscript 𝐗 𝑝 1 superscript ℝ 𝑠 𝑑{\bf X}^{(p-1)}\in\mathbb{R}^{|s|\times d}bold_X start_POSTSUPERSCRIPT ( italic_p - 1 ) end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT | italic_s | × italic_d end_POSTSUPERSCRIPT be the incoming residual stream. In the p 𝑝 p italic_p-th block, we compute:

𝐗(p)superscript 𝐗 𝑝\displaystyle{\bf X}^{(p)}bold_X start_POSTSUPERSCRIPT ( italic_p ) end_POSTSUPERSCRIPT:=𝖬𝖧𝖠(𝐐=𝐗(𝐩−𝟏),𝐊=𝐗(𝐩−𝟏),\displaystyle:=\mathsf{MHA}\bigl{(}\mathbf{Q}=\bf X^{(p-1)},\;\mathbf{K}=\bf X% ^{(p-1)},:= sansserif_MHA ( bold_Q = bold_X start_POSTSUPERSCRIPT ( bold_p - bold_1 ) end_POSTSUPERSCRIPT , bold_K = bold_X start_POSTSUPERSCRIPT ( bold_p - bold_1 ) end_POSTSUPERSCRIPT ,
𝐕=𝐗(𝐩−𝟏)),\displaystyle\qquad\quad\mathbf{V}=\bf X^{(p-1)}\bigr{)},bold_V = bold_X start_POSTSUPERSCRIPT ( bold_p - bold_1 ) end_POSTSUPERSCRIPT ) ,
𝐗(p)superscript 𝐗 𝑝\displaystyle{\bf X}^{(p)}bold_X start_POSTSUPERSCRIPT ( italic_p ) end_POSTSUPERSCRIPT:=𝖬𝖧𝖠(𝐐=𝐒,𝐊=𝐗(𝐩),\displaystyle:=\mathsf{MHA}\bigl{(}\mathbf{Q}=\bf S,\;\mathbf{K}={\bf X}^{(p)},:= sansserif_MHA ( bold_Q = bold_S , bold_K = bold_X start_POSTSUPERSCRIPT ( bold_p ) end_POSTSUPERSCRIPT ,
𝐕=𝐗(p)),\displaystyle\qquad\quad\mathbf{V}={\bf X}^{(p)}\bigr{)},bold_V = bold_X start_POSTSUPERSCRIPT ( italic_p ) end_POSTSUPERSCRIPT ) ,
𝐗(p)superscript 𝐗 𝑝\displaystyle{\bf X}^{(p)}bold_X start_POSTSUPERSCRIPT ( italic_p ) end_POSTSUPERSCRIPT:=𝖫𝖺𝗒𝖾𝗋𝖭𝗈𝗋𝗆⁢(𝐗(p−1)+𝖥𝖥𝖭⁢(𝐗(p))).assign absent 𝖫𝖺𝗒𝖾𝗋𝖭𝗈𝗋𝗆 superscript 𝐗 𝑝 1 𝖥𝖥𝖭 superscript 𝐗 𝑝\displaystyle:=\mathsf{LayerNorm}\Bigl{(}{\bf X}^{(p-1)}+\mathsf{FFN}\bigl{(}{% \bf X}^{(p)}\bigr{)}\Bigr{)}.:= sansserif_LayerNorm ( bold_X start_POSTSUPERSCRIPT ( italic_p - 1 ) end_POSTSUPERSCRIPT + sansserif_FFN ( bold_X start_POSTSUPERSCRIPT ( italic_p ) end_POSTSUPERSCRIPT ) ) .

We initialize the self-attention MHA blocks from the pre-trained Gemma-2 base model and the cross attention blocks according to the default PyTorch weight initialization scheme.

#### A.3.1 Model Size

Our largest HyperSteer (cross attention) architecture has 22 modified decoder blocks, and has ≈0.998 absent 0.998\approx 0.998≈ 0.998 times the parameters as Gemma-2-2B, which has 26 standard decoder blocks. Each cross attention decoder block has ≈1.18 absent 1.18\approx 1.18≈ 1.18 times as many parameters as a standard Gemma-2-2B decoder block.

#### A.3.2 Detailed TFLOPs Analysis

We demonstrate that the number of TFLOPs to reach optimal steering performance decays with the number of concepts in the dataset, or steering prompts used. Thus, as we scale up steering data HyperSteer becomes an efficient and superior alternative to the supervised dictionary learning baseline for steering language models. We focus our best method, the cross attention architecture, for this analysis.

Let F ReFT≈666.27±20.74 subscript 𝐹 ReFT plus-or-minus 666.27 20.74 F_{\mathrm{ReFT}}\approx 666.27\pm 20.74 italic_F start_POSTSUBSCRIPT roman_ReFT end_POSTSUBSCRIPT ≈ 666.27 ± 20.74 be the TFLOPs required to train a single ReFT-r1 steering vector, and ℒ¯joint⋆superscript subscript¯ℒ joint⋆\bar{\mathcal{L}}_{\mathrm{joint}}^{\star}over¯ start_ARG caligraphic_L end_ARG start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⋆ end_POSTSUPERSCRIPT be the average optimal evaluation loss for computed on Concept10. We average over 5 different random seeds as well to obtain this constant. To train HyperSteer (cross attention) we construct datasets 𝒟⁢(c)𝒟 𝑐\mathcal{D}(c)caligraphic_D ( italic_c ) of varying number of concepts (steering prompts) selected in the interval c∈[10,16000]𝑐 10 16000 c\in[10,16000]italic_c ∈ [ 10 , 16000 ]. Concept10 is held-in with respect to 𝒟⁢(c)𝒟 𝑐\mathcal{D}(c)caligraphic_D ( italic_c ). We train HyperSteer on each 𝒟⁢(c)𝒟 𝑐\mathcal{D}(c)caligraphic_D ( italic_c ) until the eval loss on Concept10 reaches ℒ¯joint⋆superscript subscript¯ℒ joint⋆\bar{\mathcal{L}}_{\mathrm{joint}}^{\star}over¯ start_ARG caligraphic_L end_ARG start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⋆ end_POSTSUPERSCRIPT. The TFLOPs per concept F 𝒟⁢(c)subscript 𝐹 𝒟 𝑐 F_{\mathcal{D}(c)}italic_F start_POSTSUBSCRIPT caligraphic_D ( italic_c ) end_POSTSUBSCRIPT for a dataset 𝒟⁢(c)𝒟 𝑐\mathcal{D}(c)caligraphic_D ( italic_c ), where N∗superscript 𝑁 N^{*}italic_N start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT gradient steps are taken until ℒ¯joint⋆superscript subscript¯ℒ joint⋆\bar{\mathcal{L}}_{\mathrm{joint}}^{\star}over¯ start_ARG caligraphic_L end_ARG start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⋆ end_POSTSUPERSCRIPT is achieved is computed with the following formula:

F 𝒟⁢(c)=N∗⋅F¯step c.subscript 𝐹 𝒟 𝑐⋅superscript 𝑁 subscript¯𝐹 step 𝑐 F_{\mathcal{D}(c)}=\frac{N^{*}\cdot\bar{F}_{\mathrm{step}}}{c}.italic_F start_POSTSUBSCRIPT caligraphic_D ( italic_c ) end_POSTSUBSCRIPT = divide start_ARG italic_N start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT ⋅ over¯ start_ARG italic_F end_ARG start_POSTSUBSCRIPT roman_step end_POSTSUBSCRIPT end_ARG start_ARG italic_c end_ARG .(5)

F¯step subscript¯𝐹 step\bar{F}_{\mathrm{step}}over¯ start_ARG italic_F end_ARG start_POSTSUBSCRIPT roman_step end_POSTSUBSCRIPT is the average TFLOPs per training step for HyperSteer, a local per-training-run statistic with low variance given that the distribution of sequence lengths of both input prompts and steering prompts across examples is observed to be largely uniform. The number of layers is selected to match the total number of parameters with the target model.

We also fit a simple curve to approximate F 𝒟⁢(c)subscript 𝐹 𝒟 𝑐 F_{\mathcal{D}(c)}italic_F start_POSTSUBSCRIPT caligraphic_D ( italic_c ) end_POSTSUBSCRIPT, and find that an equation of the form f⁢(c)=a+b⋅exp⁡(d⁢c)𝑓 𝑐 𝑎⋅𝑏 𝑑 𝑐 f(c)=a+b\cdot\exp(dc)italic_f ( italic_c ) = italic_a + italic_b ⋅ roman_exp ( italic_d italic_c ) best fits the curve with a=87.7035,b=1521.1495,c=−0.0034 formulae-sequence 𝑎 87.7035 formulae-sequence 𝑏 1521.1495 𝑐 0.0034 a=87.7035,b=1521.1495,c=-0.0034 italic_a = 87.7035 , italic_b = 1521.1495 , italic_c = - 0.0034 and R 2=0.9976 superscript 𝑅 2 0.9976 R^{2}=0.9976 italic_R start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT = 0.9976. Clearly, lim c→∞f⁢(c)=a subscript→𝑐 𝑓 𝑐 𝑎\lim_{c\to\infty}f(c)=a roman_lim start_POSTSUBSCRIPT italic_c → ∞ end_POSTSUBSCRIPT italic_f ( italic_c ) = italic_a and a<F ReFT 𝑎 subscript 𝐹 ReFT a<F_{\mathrm{ReFT}}italic_a < italic_F start_POSTSUBSCRIPT roman_ReFT end_POSTSUBSCRIPT, showing that HyperSteer is more compute efficient to train when scaling up steering tasks.

### A.4 Details on Training Objective

ReFT-r1 jointly optimizes for steering via causal language modeling loss and concept detection by selecting the top-k 𝑘 k italic_k sequence-level activations.

ℒ joint⁢(LM θ)=subscript ℒ joint subscript LM 𝜃 absent\displaystyle\mathcal{L}_{\text{joint}}(\mathrm{LM}_{\theta})=caligraphic_L start_POSTSUBSCRIPT joint end_POSTSUBSCRIPT ( roman_LM start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ) =(6)
𝔼(𝐱,𝐲)∼𝒟[\displaystyle\mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\mathcal{D}}\bigg{[}blackboard_E start_POSTSUBSCRIPT ( bold_x , bold_y ) ∼ caligraphic_D end_POSTSUBSCRIPT [−∑t=1 T log⁡P θ⁢(y t∣𝐱,y<t)superscript subscript 𝑡 1 𝑇 subscript 𝑃 𝜃 conditional subscript 𝑦 𝑡 𝐱 subscript 𝑦 absent 𝑡\displaystyle-\sum_{t=1}^{T}\log P_{\theta}(y_{t}\mid\mathbf{x},y_{<t})- ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT roman_log italic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_y start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∣ bold_x , italic_y start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT )
+λ∑a i∉TopK⁡(Ψ⁢(𝐡))∥a i∥1].\displaystyle+\lambda\sum_{a_{i}\notin\operatorname{TopK}(\Psi(\mathbf{h}))}\|% a_{i}\|_{1}\bigg{]}.+ italic_λ ∑ start_POSTSUBSCRIPT italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∉ roman_TopK ( roman_Ψ ( bold_h ) ) end_POSTSUBSCRIPT ∥ italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∥ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ] .(7)

Here,

Ψ⁢(h i)=ReLU⁢(h i⋅𝐰 ReFT-R1)∈ℝ d×1 Ψ subscript ℎ 𝑖 ReLU⋅subscript ℎ 𝑖 subscript 𝐰 ReFT-R1 superscript ℝ 𝑑 1\Psi(h_{i})=\mathrm{ReLU}(h_{i}\cdot{\bf w}_{\mathrm{\texttt{ReFT-R1}}})\in% \mathbb{R}^{d\times 1}roman_Ψ ( italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = roman_ReLU ( italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ⋅ bold_w start_POSTSUBSCRIPT ReFT-R1 end_POSTSUBSCRIPT ) ∈ blackboard_R start_POSTSUPERSCRIPT italic_d × 1 end_POSTSUPERSCRIPT(8)

is a sequence-level concept detection latent. This objective is also used in the regression and SFT variants of HyperSteer, when the steering vector is only conditioned on the concept.

Input-conditioned variants do not minimize an additional concept detection loss term, hence do not require an additional inference step to compute the maximal activations on a per-concept basis. The baseline ReFT-r1, end-to-end (no in-context learning), and regression variants require this additional step, and the steering heads are modified to generate Δ s subscript Δ 𝑠\Delta_{s}roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT with unit norm. Input-conditioned methods do not normalize Δ s x superscript subscript Δ 𝑠 𝑥\Delta_{s}^{x}roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT, eliminating this step.

We note that the no-context reconstruction method is trained on ground truth labels that do utilize ℒ joint subscript ℒ joint\mathcal{L}_{\mathrm{joint}}caligraphic_L start_POSTSUBSCRIPT roman_joint end_POSTSUBSCRIPT, hence evaluation on the regression method requires this additional inference step.

### A.5 Concept Dataset Details

##### Negative samples

AxBench uses negative examples to enable training and evaluation for concept detection. Since our work only focuses on steering models and not concept detection, we discard this objective and omit negative examples in training and evaluation data for all HyperSteer variants. We note that the no-context reconstruction variant indirectly uses these negative samples, for ground truth steering vectors Δ s∗superscript subscript Δ 𝑠\Delta_{s}^{*}roman_Δ start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT were trained using the joint objective ([7](https://arxiv.org/html/2506.03292v1#A1.E7 "In A.4 Details on Training Objective ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")).

##### Data Ratio

For all training datasets, the ratio of base prompts to steering prompts is 72:1:72 1 72:1 72 : 1. During evaluation, the ratio of base prompts to steering prompts is 10:1:10 1 10:1 10 : 1. We keep these parameters consistent with AxBench save for the lack of negative examples.

##### Base prompt data distributions

Training base prompts are sampled from a diverse instruction pool of three genres: code, text, and math. The open source datasets Dolly-15K Conover et al. ([2023](https://arxiv.org/html/2506.03292v1#bib.bib3)) and GSM8K Cobbe et al. ([2021](https://arxiv.org/html/2506.03292v1#bib.bib2)) comprise the instruction data in this pool. We point readers to Sec. I of the AxBench appendix for further details. Labels for training data are generated from gpt-4o-mini. For evaluation base prompts, we use sample instructions from AlpacaEval Li et al. ([2023b](https://arxiv.org/html/2506.03292v1#bib.bib11)) to ensure fairness. These settings and choices are again identical to those of AxBench.

### A.6 Baseline details

We use prompting, fine-tuning, and activation steering baseliens from AxBench. The core comparisons however

##### Supervised Dictionary Learning

ReFT-r1 is a method proposed by AxBench Wu et al. ([2025](https://arxiv.org/html/2506.03292v1#bib.bib22)) to jointly perform the task of concept detection and concept steering using a weakly supervised objective ([7](https://arxiv.org/html/2506.03292v1#A1.E7 "In A.4 Details on Training Objective ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")). At training time, we train one ReFT-r1 per concept/steering prompt to populate a dictionary of atoms, with each atom being a learned steering vector.

At training time, the latent is computed from the similarity between the hidden states of the modele model and the learned steering vector:

Ψ Detect ReFT-r1⁢(h i)=ReLU⁢(h i⋅𝐰 ReFT-r1)superscript subscript Ψ Detect ReFT-r1 subscript ℎ 𝑖 ReLU⋅subscript ℎ 𝑖 subscript 𝐰 ReFT-r1\Psi_{\text{Detect}}^{\text{ReFT-r1}}(h_{i})=\mathrm{ReLU}(h_{i}\cdot\mathbf{w% }_{\text{ReFT-r1}})roman_Ψ start_POSTSUBSCRIPT Detect end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ReFT-r1 end_POSTSUPERSCRIPT ( italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = roman_ReLU ( italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ⋅ bold_w start_POSTSUBSCRIPT ReFT-r1 end_POSTSUBSCRIPT )(9)

This latent is then inferred on the evaluation set to determine the final magnitude of the steering vector for each concept at test time:

Δ s x=(1 k⁢∥Top-k⁢(Ψ Detect ReFT-r1⁢(𝐡))∥1)⁢𝐰 ReFT-r1 subscript superscript Δ 𝑥 𝑠 1 𝑘 subscript delimited-∥∥Top-k superscript subscript Ψ Detect ReFT-r1 𝐡 1 subscript 𝐰 ReFT-r1\Delta^{x}_{s}=\left(\frac{1}{k}{\left\lVert\text{Top-k}(\Psi_{\text{Detect}}^% {\text{ReFT-r1}}(\mathbf{h}))\right\rVert_{1}}\right)\mathbf{w}_{\text{ReFT-r1}}roman_Δ start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT = ( divide start_ARG 1 end_ARG start_ARG italic_k end_ARG ∥ Top-k ( roman_Ψ start_POSTSUBSCRIPT Detect end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ReFT-r1 end_POSTSUPERSCRIPT ( bold_h ) ) ∥ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ) bold_w start_POSTSUBSCRIPT ReFT-r1 end_POSTSUBSCRIPT(10)

##### Prompt steering

This is a strong baseline which is shown in AxBench to outperform other steering methods and does not require training. For a given steering prompt s 𝑠 s italic_s, gpt-4o-mini is used to enhance s→s′→𝑠 superscript 𝑠′s\to s^{\prime}italic_s → italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT by explicitly instructing the model to include the concept in its response, which is pre-pended to a base prompt x 𝑥 x italic_x. We sample steered generations from the target LM using this enhanced prompt s′⊕x direct-sum superscript 𝑠′𝑥 s^{\prime}\oplus x italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ⊕ italic_x.

Table 4: Mapping of short concept descriptors to their full labels in held-out Concept10.

### A.7 Additional Experiments

#### A.7.1 Geometric structure using dimensionality reduction

We also analyze the structure of our high dimensional steering vectors on a held-out evaluation set of steering prompts

![Image 4: Refer to caption](https://arxiv.org/html/2506.03292v1/x4.png)

Figure 4: Concept10 t-SNE analysis of 2500 steering vectors from HyperSteer (cross attention), 250 per concept.

![Image 5: Refer to caption](https://arxiv.org/html/2506.03292v1/x5.png)

Figure 5: Concept10 PCA analysis (2 components) 2500 steering vectors from HyperSteer (cross attention), 250 per concept.

![Image 6: Refer to caption](https://arxiv.org/html/2506.03292v1/x6.png)

(a) HyperSteer (cross attention)

![Image 7: Refer to caption](https://arxiv.org/html/2506.03292v1/x7.png)

(b) HyperSteer (no context)

Figure 6: Pairwise cosine similarities of steering vectors, averaged within each steering prompt, for our two HyperSteer variants. ([6(a)](https://arxiv.org/html/2506.03292v1#A1.F6.sf1 "In Figure 6 ‣ A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")) Cross-attention: strong on-diagonal (same steering prompt) alignment with some off-diagonal variance due to prompt conditioning. ([6(b)](https://arxiv.org/html/2506.03292v1#A1.F6.sf2 "In Figure 6 ‣ A.7.1 Geometric structure using dimensionality reduction ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")) No-context: generally weaker off-diagonal alignment. We hypothesize that cross-attention’s higher off-diagonal similarity arises from shared semantic and linguistic structure across prompts even when steering prompt labels differ.

#### A.7.2 Attention Heatmaps

To better understand the interaction between the base prompt x 𝑥 x italic_x and the steering prompt s 𝑠 s italic_s in the In Context and Cross Attention HyperSteer architectures, we analyze the self-attention and cross-attention heatmaps across layers and heads (N=20 𝑁 20 N=20 italic_N = 20 layers).

A key takeaway is that all query (concept) tokens all tend to attend to the same or a few select keys/values (input prompt) tokens with high mass. This trend remains consistent across cross-attention modules from layers 0-19 ([9](https://arxiv.org/html/2506.03292v1#A1.F9 "Figure 9 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [11](https://arxiv.org/html/2506.03292v1#A1.F11 "Figure 11 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [12](https://arxiv.org/html/2506.03292v1#A1.F12 "Figure 12 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")). Each cross-attention module has 8 heads.

#### A.7.3 Data Distribution

![Image 8: Refer to caption](https://arxiv.org/html/2506.03292v1/x8.png)

Figure 7: Concept10 perplexity distribution on data labels sampled from gpt-4o-mini.

![Image 9: Refer to caption](https://arxiv.org/html/2506.03292v1/x9.png)

Figure 8: Concept10 perplexity distribution won data labels sampled from Gemma-2-2B.

A potential issue with our dataset are the use of ground truth labels samples from a stronger “teacher model” gpt-4o-mini, whereas Gemma-2-2B is a weaker “student” model we seek to adapt. This is evidenced by the right-skewed distribution (see [7](https://arxiv.org/html/2506.03292v1#A1.F7 "Figure 7 ‣ A.7.3 Data Distribution ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks") for perplexities computed from base LM when conditioned on the gpt-4o-mini output distribution. The perplexity distribution from Gemma-2-2B outputs ([8](https://arxiv.org/html/2506.03292v1#A1.F8 "Figure 8 ‣ A.7.3 Data Distribution ‣ A.7 Additional Experiments ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks")) comprises a much smaller range in comparison.

We ran preliminary experiments by training on labels from both distributions, and find steering performance is still better with the gpt-4o-mini labels. We suspect that this could be a result of either lower quality of Gemma-2-2B responses due prompt-engineering being the method of generation or the LLM-as-a-judge evaluation setup (which also uses gpt-4o-mini) being biased towards outputs from the same model.

### A.8 Sample generations

We use the best cross attention model for steering and the steering factor that yields the best aggregate score during evaluation. We also include responses from a prompt steered baseline for comparison. Generation is done with temperature of 1.0 and with multinomial sampling, following AxBench. See example generations [13](https://arxiv.org/html/2506.03292v1#A1.F13 "Figure 13 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [14](https://arxiv.org/html/2506.03292v1#A1.F14 "Figure 14 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [15](https://arxiv.org/html/2506.03292v1#A1.F15 "Figure 15 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [16](https://arxiv.org/html/2506.03292v1#A1.F16 "Figure 16 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [17](https://arxiv.org/html/2506.03292v1#A1.F17 "Figure 17 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [18](https://arxiv.org/html/2506.03292v1#A1.F18 "Figure 18 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [19](https://arxiv.org/html/2506.03292v1#A1.F19 "Figure 19 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [20](https://arxiv.org/html/2506.03292v1#A1.F20 "Figure 20 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [21](https://arxiv.org/html/2506.03292v1#A1.F21 "Figure 21 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks"), [22](https://arxiv.org/html/2506.03292v1#A1.F22 "Figure 22 ‣ A.8 Sample generations ‣ Appendix A Appendix ‣ HyperSteer: Activation Steering at Scale with Hypernetworks").

![Image 10: Refer to caption](https://arxiv.org/html/2506.03292v1/extracted/6509645/figures/attn_maps/layer_0.png)

Figure 9: Layer 0 attention map.

![Image 11: Refer to caption](https://arxiv.org/html/2506.03292v1/extracted/6509645/figures/attn_maps/layer_5.png)

Figure 10: Layer 5 attention map.

![Image 12: Refer to caption](https://arxiv.org/html/2506.03292v1/extracted/6509645/figures/attn_maps/layer_0.png)

Figure 11: Layer 10 attention map.

![Image 13: Refer to caption](https://arxiv.org/html/2506.03292v1/extracted/6509645/figures/attn_maps/layer_19.png)

Figure 12: Layer 19 attention map.

Figure 13: Successful steering and instruction following by HyperSteer.

Figure 14: Successful steering and instruction following by HyperSteer.

Figure 15: Successful steering and instruction following by HyperSteer.

Figure 16: Successful steering and instruction following by HyperSteer.

Figure 17: Successful steering and instruction following by HyperSteer.

Figure 18: Failed steering by HyperSteer, but successful instruction following.

Figure 19: Failed steering and instruction following by HyperSteer.

Figure 20: Failed steering by HyperSteer.

Figure 21: Somewhat successful steering by HyperSteer, but failed to follow the instruction.

Figure 22: Failed steering by HyperSteer, but successful instruction following.
