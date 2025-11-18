
# References

- Representation Engineering: https://arxiv.org/abs/2310.01405
  - Early steering paper repeng repo is based on 
  - > In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems. 
- BiPDO https://arxiv.org/abs/2406.00045
  - > Researchers have been studying approaches to steer the behavior of Large Language Models (LLMs) and build personalized LLMs tailored for various applications. While fine-tuning seems to be a direct solution, it requires substantial computational resources and may significantly affect the utility of the original LLM. Recent endeavors have introduced more lightweight strategies, focusing on extracting "steering vectors" to guide the model's output toward desired behaviors by adjusting activations within specific layers of the LLM's transformer architecture. However, such steering vectors are directly extracted from the activations of human preference data and thus often lead to suboptimal results and occasional failures, especially in alignment-related scenarios. This work proposes an innovative approach that could produce more effective steering vectors through bi-directional preference optimization. Our method is designed to allow steering vectors to directly influence the generation probability of contrastive human preference data pairs, thereby offering a more precise representation of the target behavior. By carefully adjusting the direction and magnitude of the steering vector, we enabled personalized control over the desired behavior across a spectrum of intensities. Extensive experimentation across various open-ended generation tasks, particularly focusing on steering AI personas, has validated the efficacy of our approach. Moreover, we comprehensively investigate critical alignment-concerning scenarios, such as managing truthfulness, mitigating hallucination, and addressing jailbreaking attacks. Remarkably, our method can still demonstrate outstanding steering effectiveness across these scenarios. Furthermore, we showcase the transferability of our steering vectors across different models/LoRAs and highlight the synergistic benefits of applying multiple vectors simultaneously. 
- https://github.com/vgel/repeng
  - This is library that quite robust and popular for steering with PCA vectors in hidden space, we use it's prompting setup, and use it as a baseline. It's been cited in several papers
  - > A Python library for generating control vectors with representation engineering. Train a vector in less than sixty seconds!
- [PiSSA](https://arxiv.org/html/2404.02948v4)
  - This paper decomposes each weight matrix W into U S V + W_residual like us
  - > To parameter-efficiently fine-tune (PEFT) large language models (LLMs), the low-rank adaptation (LoRA) method approximates the model changes Δ⁢W∈ℝm×n through the product of two matrices A∈ℝm×r and B∈ℝr×n, where r≪min⁡(m,n), A is initialized with Gaussian noise, and B with zeros. LoRA freezes the original model W and updates the “Noise & Zero” adapter, which may lead to slow convergence. To overcome this limitation, we introduce Principal Singular values and Singular vectors Adaptation (PiSSA). PiSSA shares the same architecture as LoRA, but initializes the adaptor matrices A and B with the principal components of the original matrix W, and put the remaining components into a residual matrix Wr⁢e⁢s∈ℝm×n which is frozen during fine-tuning. Compared to LoRA, PiSSA updates the principal components while freezing the “residual” parts, allowing faster convergence and enhanced performance. Comparative experiments of PiSSA and LoRA across 11 different models, ranging from 184M to 70B, encompassing 5 NLG and 8 NLU tasks, reveal that PiSSA consistently outperforms LoRA under identical experimental setups. On the GSM8K benchmark, Gemma-7B fine-tuned with PiSSA achieves an accuracy of 77.7%, surpassing LoRA’s 74.53% by 3.25%. Due to the same architecture, PiSSA is also compatible with quantization to further reduce the memory requirement of fine-tuning. Compared to QLoRA, QPiSSA (PiSSA with 4-bit quantization) exhibits smaller quantization errors in the initial stages. Fine-tuning LLaMA-3-70B on GSM8K, QPiSSA attains an accuracy of 86.05%, exceeding the performance of QLoRA at 81.73%. Leveraging a fast SVD technique, PiSSA can be initialized in only a few seconds, presenting a negligible cost for transitioning from LoRA to PiSSA.
- [SSVD](https://arxiv.org/html/2509.02830v1)
  - This paper rotates the V matrix, which is very nodel and we use, it has good results (generalisaton which is better than just parameter efficiency)
  - > Parameter-efficient fine-tuning (PEFT) has emerged as a scalable solution for adapting large foundation models. While low-rank adaptation (LoRA) is widely used in speech applications, its state-of-the-art variants, e.g., VeRA, DoRA, PiSSA, and SVFT, are developed mainly for language and vision tasks, with limited validation in speech. This work presents the first comprehensive integration and benchmarking of these PEFT methods within ESPnet. We further introduce structured SVD-guided (SSVD) fine-tuning, which selectively rotates input-associated right singular vectors while keeping output-associated vectors fixed to preserve semantic mappings. This design enables robust domain adaptation with minimal trainable parameters and improved efficiency. We evaluate all methods on domain-shifted speech recognition tasks, including child speech and dialectal variation, across model scales from 0.1B to 2B. All implementations are released in ESPnet to support reproducibility and future work.
- [DoRA](https://arxiv.org/html/2306.08990v2) 
  - Seperates magnitude and direction and has become a popular and strong LoRA baseline
- [SVFT](https://arxiv.org/html/2405.19597v1)
  - This paper updates the S of the SVD of each weight matrix like us
  - > Popular parameter-efficient fine-tuning (PEFT) methods, such as LoRA and its variants, freeze pre-trained model weights 𝐖 and inject learnable matrices 𝚫⁢𝐖. These 𝚫⁢𝐖 matrices are structured for efficient parameterization, often using techniques like low-rank approximations or scaling vectors. However, these methods typically show a performance gap compared to full fine-tuning. Although recent PEFT methods have narrowed this gap, they do so at the cost of additional learnable parameters. We propose SVFT, a simple approach that fundamentally differs from existing methods: the structure imposed on 𝚫⁢𝐖 depends on the specific weight matrix 𝐖. Specifically, SVFT updates 𝐖 as a sparse combination of outer products of its singular vectors, training only the coefficients (scales) of these sparse combinations. This approach allows fine-grained control over expressivity through the number of coefficients. Extensive experiments on language and vision benchmarks show that SVFT1 recovers up to 96% of full fine-tuning performance while training only 0.006 to 0.25% of parameters, outperforming existing methods that only recover up to 85% performance using 0.03 to 0.8% of the trainable parameter budget.

- Improving Alignment and Robustness with Circuit Breakers https://arxiv.org/html/2406.04313v3
  - This paper is novel in that it also has a loss that operates on the hidden states, but it only works for refusals not steering (the authors told me this in a private comms). It's a much easier and more limited problem than steering because you are shutting down complex behaviour in favour of a simple refusal mode.
  - > AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with “circuit breakers.” Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility—even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image “hijacks” that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks. Code is available at github.com/GraySwanAI/circuit-breakers.

- [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2)](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
  - see also "back arrowGo to ICML 2025 Conference homepage
Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" https://openreview.net/forum?id=rNfzT8YkgO&noteId=hA0Ptt2IOa
  - Many of the main authors explain how SAE's, an alternative to steering, do not generalise well (an I would add do not scale well)

    TL;DR

        To validate whether SAEs were a worthwhile technique, we explored whether they were useful on the downstream task of OOD generalisation when detecting harmful intent in user prompts
        Negative result: SAEs underperformed linear probes
            Corollary: Linear probes are actually really good and cheap and perform great
        As a result of this and parallel work, we are deprioritising fundamental SAE research for the moment and exploring other directions, though SAEs will remain a tool in our toolkit
            We do not think that SAEs are useless or that no one should work on them, but we also do not think that SAEs will be a game-changer for interpretability, and speculate that the field is over-invested in them.
        Training SAEs specialised for chat data closed about half the gap but was still worse than linear probes
            We tried several ways to train chat SAEs, all did about as well. By default, we recommend taking an SAE on pretraining data and finetuning it on a bit of chat data
        Other results:
            We found SAEs fairly helpful for debugging low quality datasets (noticing spurious correlations)
            We present a variant of JumpReLU with an alternative sparsity penalty to get rid of high-frequency latents
            We argue that a standard auto-interp approach of computing the average interpretability of a uniformly sampled SAE latent can be misleading as it doesn’t penalise models which have high frequency, but not very interpretable, latents, a

Why does current steering methods fail to generalise?
- [AxBench](https://arxiv.org/pdf/2501.17148)
- https://turntrout.com/research#steering-vectors

    Prompt engineering and finetuning aim to maximize language model performance on a given metric (like toxicity reduction). However, these methods do not fully elicit a model’s capabilities. To reduce this gap, we introduce activation engineering: the inference-time modification of activations in order to control (or steer) model outputs. Specifically, we introduce the Activation Addition (ActAdd) technique, which contrasts the intermediate activations on prompt pairs (such as “Love” versus “Hate”) to compute a steering vector.

    By tactically adding in e.g. the “Love” − “Hate” steering vector during the forward pass, we achieve sota on negative-to-positive sentiment shift and detoxification using models including LLaMA-3 and opt. ActAdd yields inference-time control over high-level output properties (like topic and sentiment) while preserving performance on off-target tasks. ActAdd is lightweight: it does not require any machine optimization and works with a single pair of data points, which enables rapid iteration over steering. ActAdd demonstrates the power of activation engineering.

    During 2023 and 2024, activation engineering inspired dozens of follow-up papers.
    At Google DeepMind, Mark Kurzeja and I found a negative result when attempting to steer Gemini towards higher benchmark scores.

- https://turntrout.com/gemini-steering


  Bidpo seems effective and sample-efficient but does not currently exceed more standard baselines. It’s hard to draw firm conclusions about bidpo because TruthfulQA might not be measuring truthfulness / factuality. However, we remain excited about dpo-driven Conditional Activation Steering, which has additional advantages—particularly for targeted loss mitigation.

- [Anthropic steering personality traits paper](https://www.anthropic.com/research/persona-vectors)
- supressed neurons
  - my readme that summarises the lesser known litriture on https://github.com/wassname/eliciting_suppressed_knowledge this tells you why I needed to steer on layer -3, because lots of supression happens after this and I hypothesis that the generation planning information I need to steer is supressed later on

  1. **Suppression/Prediction Neural Dynamics**:
     - [Gurnee et al. (2024)](https://arxiv.org/abs/2401.12181) identified "universal neurons" across different model seeds, including prediction neurons (increasing probability of related tokens) and suppression neurons (decreasing probability of specific token classes)
     - The architecture shows "a sudden shift towards a much larger number of suppression neurons" in final layers
     - [Lad et al. (2024)](https://arxiv.org/html/2406.19384v1) propose a "stages of inference" hypothesis with a final "residual sharpening" phase dominated by suppression dynamics

  2. **Unfaithful Chain-of-Thought**:
     - [Anthropic (2025)](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf) demonstrates that even in leading models like Claude 3.7, chain-of-thought reasoning achieves only 30% faithfulness
     - [OpenAI (2025)](https://cdn.openai.com/pdf/34f2ada6-870f-4c26-9790-fd8def56387f/CoT_Monitoring.pdf) shows that penalizing "bad thoughts" leads to models that "learn to hide intent" rather than genuinely correcting reasoning
     - Both lines of evidence suggest models maintain internal representations that diverge from their expressed reasoning

Meta level content:
- [TRM](https://arxiv.org/html/2510.04871v1)
- [How to write ML paper](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
- [TimesFM](https://arxiv.org/html/2310.10688v4) has a nice section explaining the model archetecture and it's guiding principles 

  The main guiding principles for our architecture are the following:


AxBench update blogpost https://zen-wu.social/steer/index.html

> One key takeaway from AxBench is that representation steering methods, although providing interpretable representation edits, lag behind simple prompting baselines. Since the release of AxBench, there are a number of notable new steering methods that try to push the frontier of steering methods, showing promise but not representing breakthroughs. In this blog post, we argue that future works in representation steering methods should also consider computational efficiency in order to make a convincing case as alternatives to prompting, lightweight finetuning, or inference-time scaling techniques.


    zhengxuan wu
    On representation steering
    
    home · blog
    quick links
    中文翻译
    July 12, 2025
    tl;dr.
    
        Besides performance, efficiency is crucial for steering.
        Avoid saying rank-1 steering is just efficient in a hand-wavy way.
        Always start with a simple (i.e., prompting) baseline.
    
    Introduction.
    
    Language models (LMs) can be steered toward desired behaviors through the following primary approaches: prompting guides models via carefully crafted input instructions, steering vectors manipulate internal representations through rank-1 activation addition, parameter-efficient fine-tuning methods (PEFTs) adapt model behaviors by updating a small subset of parameters, and representation finetuning methods (ReFTs) finetune specific representations within the model's intermediate layers.
    
    Recently, AxBench [1] was proposed to benchmark steering methods from these four types systematically in a concept-based steering scenario for open-ended instruction following (e.g., steering an LM to always mention "golden gate bridge" when answering user queries).
    AxBench
    Figure 1: concept detection and steering performance of different methods in AxBench.
    
    One key takeaway from AxBench is that representation steering methods, although providing interpretable representation edits, lag behind simple prompting baselines. Since the release of AxBench, there are a number of notable new steering methods that try to push the frontier of steering methods, showing promise but not representing breakthroughs. In this blog post, we argue that future works in representation steering methods should also consider computational efficiency in order to make a convincing case as alternatives to prompting, lightweight finetuning, or inference-time scaling techniques.
    Efficiency of LM steering methods.
    
    We first introduce these methods and outline two aspects: memory and compute costs. We leave out the performance metric for now, so we can compare these methods assuming that optimal performance of these methods is very similar.
    Symbol 	Meaning
    x 	prompt length (tokens)
    L 	number of transformer layers
    dk, dv 	key & value dimensions
    H 	hidden size / model width
    Prompting.
    Prompt steering
    
    LMs are heavily post-trained to follow given instructions. As a result, prompting is the de facto way of controlling LMs. For instance, if one wants to personalize a LM to always respond as if itself is the golden gate bridge, one could add a system prompt that is prepended before the user prompt as “when answering, always mention golden gate bridge”. For more precisely steering behaviours, one might require longer system prompts.
    
    Memory costs. The run-time memory cost is the size of the key–value (KV) cache created by the prompt (assuming a vanilla KV-cache implementation):
    
    M  =  x ⋅ (dk + dv) ⋅ L
    
    This cost grows linearly with the number of steering prompt tokens. Yet the total context length—system prompt + user prompt + model generation—now routinely spans thousands (and, with techniques like infinite attention [2] or external memory modules [3], even millions) of tokens. To visualize the impact, we plot the percentage memory overhead introduced by various steering prompt sizes as a function of total context length:
    KV memory cost
    Figure 2: Memory overhead of different steering context lengths.
    
    The curve drops off rapidly: once the total context exceeds a few thousands tokens (the norm for modern LMs with inference-time scaling), a 50- to 400-token steering prompt adds < 5 % KV-cache overhead—and < 1 % beyond 16 k tokens – making the memory cost essentially negligible in realistic settings.
    
    Compute costs. Besides run-time memory, a steering prompt also adds extra FLOPs each time the model decodes a token, because the query for that new token must attend over the prompt’s additional K/V vectors.
    
    For one decoding step, we can write out the additional FLOPs for processing the additional steering prompt tokens as:
    
    ΔFLOPsprompt  =  xp ⋅ L ⋅ (2dk + dv)
    
    As noted in [4], the FLOPs spent in MHA is only a tiny fraction of the full forward pass. We can write out the full forward pass FLOPs as:
    
    FLOPsbase  =  L ⋅ [ 4H2  +  C (2dk + dv) ]
    
    where C is the total context length (system + user + generation, including the steering prompt).
    
    overhead%  ≈  xp (2dk+dv)  /  C(2dk+dv) + 4H2  ⋅ 100
    
    We can further contextualize the compute costs overhead by considering a typical 7-B parameter model (e.g., H=4096, dk=dv=64). For a 16k context length, a 50-token steering prompt adds 0.0001 % overhead, and a 400-token steering prompt adds 0.001 % overhead. These are negligible in practice.
    Length C 	xp = 50 	xp = 100 	xp = 400
    512 tokens 	0.20 % 	0.39 % 	1.56 %
    4,096 tokens 	0.03 % 	0.06 % 	0.24 %
    16,384 tokens 	0.008 % 	0.016 % 	0.06 %
    Because feed-forward layers dominate FLOPs, even a 400-token steering prompt adds well under 1% compute overhead once the context reaches just a few thousand tokens—exactly the regime used in today’s long-context inference.
    Steering vectors.
    Steering vector
    
    Steering vectors [5] place a rank-1 activation addition intervention in the middle of the LM forward pass, making minimal linear edits of LM representations. Given any hidden representation h, an activation addition intervention adds in a scaled rank-1 vector in-place as:
    
    Φ(h) = h + α · w1
    
    Memory costs. The memory overhead is a single vector. Similar to prompt steering, the memory overhead with respect to context length can be written as:
    
    overhead%  =  H  /  C · L · (dk + dv) × 100
    
    where the denominator contains the KV cache memory for the whole context. For a typical 7B model (H=4096, dk=dv=64, L=32, C=4096) that ratio is roughly 0.02% — so the rank-1 vector's memory footprint is essentially negligible. To compare with prompt steering, the overhead ratio is roughly 2.5% conditioned on the steering prompt having 100 tokens. There are some savings in terms of absolute numbers, but again, the overall saving is relatively small when considering the memory required for the context.
    
    Compute costs. Unlike prompt steering, the compute overhead comes from the additional FLOPs of the activation addition step. Thus, the overhead can be written as:
    
    overhead%  ≈  H  /  L · [4H2 + C · (2dk + dv)] × 100
    
    Similar to prompt steering, we can further contextualize the compute costs overhead by considering a typical 7-B parameter model (e.g., H=4096, dk=dv=64):
    Total context C 	Overhead (%)
    512 tokens 	0.00019 %
    4,096 tokens 	0.00019 %
    16,384 tokens 	0.00018 %
    PEFTs.
    PEFT
    
    Instead of a single rank-1 vector, an adaptor inserts a low-rank update AB⊤ (two H×r matrices) at one transformer layer, making a slightly richer—but still lightweight—edit to the activations. As a result, both the memory and compute overhead is the same as steering vector with a multiplier upfront.
    
    Memory costs. A LoRA adaptor stores two H × r matrices, A and B, whose product forms a rank-r activation update. Because both matrices live on GPU memory, the overhead looks just like the steering-vector case—except multiplied by 2r. With r=8 on a 7B model (H=4096, dk=dv=64, L=32), that is roughly 0.39% at a 4k-token context.
    
    Compute costs. Similar to steering vector, the compute overhead is similarly minimal: the adaptor performs one low-rank projection per token at a single layer, adding 2rH FLOPs, resulting in an additional FLOPs table as:
    Total context C 	Overhead (%)
    512 tokens 	0.003 %
    4,096 tokens 	0.003 %
    16,384 tokens 	0.0029 %
    ReFTs.
    ReFT
    
    Unlike adaptors operating on model weights, ReFT operates on representations (i.e., it selects where to edit representations with learnable interventions). Specifically, ReFT focuses on editing user prompt tokens. As proposed in the original ReFT paper, the intervention usually involves low-rank edits, similar to LoRA in terms of parameterization, involving AB⊤ (two H×r matrices).
    
    Memory costs. ReFT takes up the same extra memory as LoRA during inference time as they both have the same parameterization. With r=8 on a 7B model (H=4096, dk=dv=64, L=32), that is roughly 0.39% at a 4k-token context.
    
    Compute costs. Since ReFT operates on the prompt token representations, the additional FLOPs for generating a new token is essentially 0.
    Total context C 	Overhead (%)
    512 tokens 	0.0 %
    4,096 tokens 	0.0 %
    16,384 tokens 	0.0 %
    Beyond steering performance.
    
    Our analysis shows that while representation-level approaches (rank-1 vectors, LoRA, ReFT) have narrowed the gap, they still lag prompting on AxBench. More importantly, although many existing representation-steering papers claim—somewhat hand-wavingly—that "representation steering is very cheap," this mindset is usually unfounded. A 50–400-token steering prompt adds less than 0.06 % compute or KV-cache overhead once the total context exceeds 4k tokens, and less than 0.01 % at 16 k, so the relative cost of prompting vanishes as windows scale. By contrast, vector- or adaptor-based edits, although lightweight (~0.02–0.39 % memory), still require extra checkpoints and integration effort.
    
    Prompting has limitations. Large context windows shrink the efficiency gap, but prompting remains vulnerable to prompt-injection and jail-breaking attacks that can override or dilute suppression prompts. Reference-free Preference Steering (RePS) [6] addresses this: in suppression mode it matches the standard language-modeling objective on Gemma-2 and surpasses it on the larger Gemma-3 variants, all while resisting the prompt-based jailbreaks that defeat prompting. These findings suggest that representation-level steering can provide a safer, interpretable, and more robust alternative to prompting when dependable suppression is needed.
    Conclusion.
    
    In this blog post, we argue that future works in representation steering methods should also consider computational efficiency in order to make a convincing case as alternatives to prompting. We hope this analysis can help guide future research in this direction.
    How to Cite
    Bibliography
    Zhengxuan Wu. "On representation steering." Blog post (2025).
    BibTeX
    
    @misc{wu2025steering,
    title={On representation steering},
    author={Wu, Zhengxuan},
    year={2025},
    note={Blog post},
    url={https://nlp.stanford.edu/~wuzhengx/steer/index.html}
    }
    
    Acknowledgements
    
    We thank Aryaman Arora, Qinan Yu, Chris Potts, and Chris Manning for helpful discussions and feedback on this blog post.
    
    @stanfordnlp
