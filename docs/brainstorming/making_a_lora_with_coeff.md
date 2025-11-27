• Goal  
  - Build a "reversible" or "scalable" wrapper around PEFT adapters so you can dynamically invert, disable, or re-scale their effect during a forward/backward pass.  
  - If possible Support not just IA3/ROAD, but all of the major PEFT adapters (LoRA, IA3, AdaLoRA, RandLoRA, LOFT, VERA, ROAD, including the bits-and-bytes 8-bit/4-bit variants).

• Key API idea  
  - A Python context manager (e.g. 'AdapterModifier' or 'AdapterScaler') that:  
    • Walks the model's modules to find every adapter layer (anything subclassing 'BaseTunerLayer').  
    • Detects its concrete type (LoRA-style additive vs. IA3-style multiplicative, etc.).  
    • Applies a user-specified transform—multiply (scale), add (shift), or outright replace the adapter's learned parameters.  
    • On exit, restores the original parameters so the change is strictly temporary.

• Typical usage patterns  
  - Contrastive two-pass scheme (normal + inverted labels):  
    '''python
    optimizer.zero_grad()
    with AdapterModifier(model, "scale", 1.0):
        lp = model(**batch, labels=pos).loss
    with AdapterModifier(model, "scale", -1.0):
        ln = model(**batch, labels=neg).loss
    (lp + ln).backward()
    optimizer.step()
    '''

• Repository context  
  - uses library huggingface/peft (PEFT = Parameter-Efficient Fine-Tuning).  
  - 'peft/tuners/tuners_utils.py' defines the abstract 'BaseTunerLayer' that every adapter (LoRA, IA3, etc.) mixes into its implementation.  
  - Adapters like IA3 live in 'peft/tuners/ia3', LoRA in 'peft/tuners/lora', and bnb-friendly variants in submodules there.


src/peft/tuners/oft/bnb.py
src/peft/tuners/road/bnb.py
src/peft/tuners/vera/bnb.py
src/peft/tuners/lora/bnb.py
src/peft/tuners/adalora/bnb.py
src/peft/tuners/randlora/bnb.py
src/peft/tuners/ia3/bnb.py

OFT

    > Creates Orthogonal Finetuning model from a pretrained model. The method is described in
    https://huggingface.co/papers/2306.07280
    > Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method -- Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed.


   > Therefore, we take advantage of an invariance property of hyperspherical energy - the pairwise hyperspherical similarity is provably preserved if we apply the same orthogonal transformation for all neurons. Motivated by such an invariance, we propose Orthogonal Finetuning (OFT) which adapts large text-to-image diffusion models to a downstream task without changing its hyperspherical energy. The central idea is to learn a layer-shared orthogonal transformation for neurons such that their pairwise angles are preserved. OFT can also be viewed as adjusting the canonical coordinate system for the neurons in the same layer. By jointly taking into consideration that smaller Euclidean distance between the finetuned model and the pretrained model implies better preservation of pretraining performance, we further propose an OFT variant - Constrained Orthogonal Finetuning (COFT) which constrains the finetuned model within the hypersphere of a fixed radius centered on the pretrained neurons.


IA3 **scales activations by learned vectors** 

   > Creates a Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3) model from a pretrained
    transformers model. The method is described in detail in https://r.jina.ai/https://arxiv.org/pdf/2205.05638

   > Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)^3 that **scales activations by learned vectors**, attaining stronger performance while only introducing a relatively tiny amount of new parameters. We also propose a simple recipe based on the T0 model called T-Few that can be applied to new tasks without task-specific tuning or modifications. We validate the effectiveness of T-Few on completely unseen tasks by applying it to the RAFT benchmark, attaining super-human performance for the first time and outperforming the state-of-the-art by 6% absolute. All of the code used in our experiments is publicly available.


   In order to compare favorably to few-shot ICL, we need a PEFT method that has the following properties: First, it must add or update as few parameters as possible to avoid incurring storage and memory costs. Second, it should achieve strong accuracy after few-shot training on new tasks. Finally, it must allow for mixed-task batches, since that is a capability of ICL. In order to easily enable mixed-task batches, a PEFT method should ideally not modify the model itself. Otherwise, each example in a batch would effectively need to be processed by a different model or computational graph. A more convenient alternative is provided by methods that directly modify the activations of the model since this can be done independently and cheaply to each example in the batch according to which task the example corresponds to. Prompt tuning and prefix tuning methods [ 14 , 29 ] work by concatenating learned vectors to activation or embedding sequences and are therefore examples of activation-modifying PEFT methods that allow for mixed-task batches. However, as we will discuss 5later, we were unable to attain reasonable accuracy with prompt tuning and found that the more performant PEFT methods did not allow for mixed-task batches. We therefore developed a new PEFT method that meets our desiderata. As an alternative, we explored element-wise multiplication (i.e. rescaling) of the model's activations against a learned vector. Specifically, we consider adaptation of the form l x where l ∈ Rd is a learned task-specific vector, represents element-wise multiplication, and x ∈ RT ×d is a length-T

   sequence of activations. We use "broadcasting notation” [ 46 ] so that the (i, j )th entry of l x is lj xi,j .In preliminary experiments, we found it was not necessary to introduce a learned rescaling vector for each set of activations in the Transformer model. Instead, we found it was sufficient to introduce rescaling vectors on the keys and values in self-attention and encoder-decoder attention mechanisms and on the intermediate activation of the position-wise feed-forward networks. Specifically, using the notation from Vaswani et al. [33] , we introduce three learned vectors lk ∈ Rdk , l v ∈ Rdv , and 


   (IA) 3 makes mixed-task batches possible because each sequence of activations in the batch can be separately and cheaply multiplied by its associated learned task vector. We also note that, in the event that a model will only be used on a single task, the modifications introduced by (IA) 3 can also be applied to weight matrices permanently so that no elementwise multiplication is required and the model's architecture remains unchanged. This possible because element-wise multiplications performed in (IA) 3 always co-occur with a matrix multiplication, and l W x = ( l W )x. In this case, our method incurs no additional computational cost compared to the original model. To validate (IA) 3, we compare it to a large variety of existing adaptation methods in our setting of fine-tuning T0-3B on few-shot datasets from held-out tasks. Specifically, we compare with 9 strong PEFT methods: BitFit [ 47 ] which updates only the bias parameters; Adapters [ 23 ] which introduce task-specific layers after the self-attention and position-wise feed-forward networks; Compacter and Compacter++ [ 28 ] which improve upon adapters by using low-rank matrices and hypercomplex mul-tiplication; prompt tuning [ 14 ] which learns task-specific prompt embeddings that are concatenated to the model's input; FISH Mask [ 26 ] which chooses a subset of parameters to update based on their ap-proximate Fisher information; Intrinsic SAID [ 27 ] which performs optimization in a low-dimensional subspace; prefix-tuning [ 29 ] which learns task-specific vectors that are concatenated to the model's activations; and LoRA [ 13 ] which assigns low-rank updates to parameter matrices. Additionally, we include the baselines of full-model fine-tuning and updating only the layer normalization parameters. For certain methods that allow changing the parameter efficiency, we report results for different budgets: 0.2% and 0.02% sparsity for FISH Mask, 10 and 100 learned prompt vectors for prompt tuning, and 20,000- or 500,000-dimensional subspaces for Intrinsic SAID. The results are shown in fig. 2, with detailed per-dataset results in appendix D. We find that (IA) 3

   is the only method that attains higher accuracy than the full-model-fine-tuning baseline. While other PEFT methods (e.g. Intrinsic SAID and prompt tuning) update or introduce fewer parameters, 


ROAD 
     This is the configuration class to store the configuration of a ['RoadModel']. RoAd adapter is proposed in
    https://r.jina.ai/https://arxiv.org/pdf/2409.00119


   > When multiple users submit requests simultaneously, it becomes crucial to process these requests collectively in a single batch. Given that each request may require a unique set of parameters, using batch matrix multiplication can efficiently handle these requests by leveraging GPU parallelism. However, the batch matrix multiplication still incurs considerable overhead [1, 57], necessitating the exploration of more efficient methods. Another challenge is the interpretability of LLMs that contain a billion-scale of parameters, making it difficult to explore their mechanism. PEFT provides an alternative approach by constraining the number of trainable parameters, thereby aiding in interpretability. Recent advancements in PEFT methods, particularly those focusing on representation editing [ 54 , 60 , 67 ], can be incorporated within an intervention framework [ 11 ]. This integration enhances their capability for interpretability, offering a more manageable means of dissecting the operational intricacies of LLMs. In this paper, we introduce a novel technique termed 2D rotary adaptation (RoAd) which efficiently adapts LLMs using a minimal number of trainable parameters. Furthermore, RoAd enhances both batching efficiency and composability. Our initial investigation reveals that finetuning primarily alters the angular components of the representations in pretrained LLMs, rather than their magnitudes (Section §3.1). Based on this observation, we employ a strategy of rotating certain subspaces within the representations to emulate finetuning effects. Specifically, we implement a 2D rotational approach on the representations and develop three distinct variants of RoAd (Section §3.2). To assess the efficacy of RoAd, we perform comprehensive evaluations on the GLUE benchmark [ 56 ], eight commonsense reasoning tasks and four arithmetic reasoning tasks, utilizing RoBERTa [31 ] and LLaMA [ 52 , 53 ] (Section §4.1). The results consistently show that RoAd surpasses other PEFT methods while maintaining a significantly reduced scale of trainable parameters ( < 0.1% ), as depicted in Figure 1. Additionally, RoAd employs element-wise rather than matrix multiplication, which notably improves throughput when serving heterogeneous requests within the same batch, achieving twice the throughput of LoRA [ 14 ] (Section §4.2). Furthermore, RoAd can be seamlessly integrated within an intervention framework [ 11 ], thereby enhancing model interpretability. We illustrate this through a composition experiment, demonstrating RoAd's capacity to merge weights trained for different tasks and display a new capability (Section §4.3). 

   ## 2 Background 

   In this section, we outline the challenges tackled in this work, illustrating the constraints of existing methods and objectives that drive the development of the proposed method, RoAd. 

   2.1 Parameter-efficient finetuning (PEFT) 

   Existing PEFT techniques can be categorized into three groups: adapter-based, prompt-based, and latency-less methods. Adapter-based methods [ 12 , 13 , 42 ] incorporate adapters either in parallel with or sequentially to the existing Transformer [ 55 ] modules. This incorporation necessitates modifications to the LLM architecture, consequently adding extra latency during inference. Prompt-based methods [ 19 , 21 , 43 ] enhance the input by appending new trainable tokens, which lengthens the sequence and thereby increases the computational overhead during inference. Latency-less methods, such as LoRA [ 14 ] and its variants [ 22 , 27 , 65 ], apply low-rank matrices to adapt the pretrained weights. These matrices can be seamlessly integrated into the existing weight matrices following 2finetuning, thus preserving the original LLM architecture. Specifically, LoRA adapts an LLM as 

   W = W 0 +∆ W , where W 0 ∈ Rd1×d2 is the pretrained weight and ∆W = BA with B ∈ Rd1×r ,

   A ∈ Rr×d2 , r ≪ d1 and r ≪ d2. Our proposed method, RoAd, aligns with the latency-less category and integrates effortlessly into the existing linear layer without imposing additional overhead during inference. Moreover, RoAd demonstrates exceptional parameter efficiency. The quantity of its trainable parameters is equivalent to that of a LoRA module with a rank r = 0 .5.

   Orthogonal finetuning. Drawing on the concept of hyperspherical energy and its role in characteriz-ing generalization [ 28 , 29 ], OFT [ 44 ] introduces orthogonal finetuning, an effective PEFT method for finetuning text-to-image diffusion models. Specifically, OFT implements an orthogonal matrix 

   R ∈ Rd1×d1 to the pretrained weight W 0, so the input x ∈ Rd1 to a linear layer after adaptation be-comes z = ( RW 0)⊤x. R is parameter-efficient because it is a block-diagonal matrix with n blocks as R = diag (R1, ..., Ri, ..., Rn), where each block Ri ∈ Rw×w has a dimension w = d1/n . To maintain orthogonality, Ri is derived using Cayley parameterization: Ri = ( I +Qi)( I −Qi)−1 with 

   Qi ∈ Rw×w being a skew-symmetric matrix ( Qi = −Q⊤ 

   > i

   ). In sum, {Qi}ni=1 serve as the trainable parameters and R is constructed from them with Cayley parameterization. Subsequent advancement, BOFT [ 30 ], leverages butterfly factorization to further refine OFT's parameter efficiency. However, both OFT and BOFT, due to their reliance on matrix inversions in the Cayley parameterization and increased storage of intermediate activations, necessitate additional GPU memory and increase training duration compared to other PEFT approaches. Conversely, RoAd, which may be considered as a specialized case of OFT with w = 2 , offers a faster and more memory-efficient solution by inherently maintaining orthogonality without requiring further parameterization. 

   2.2 Batching 

   Batching in this context refers to processing multiple heterogeneous requests, each requiring different adapters 2 for inference. This scenario commonly arises when serving personalized or task-specific LLMs. Specifically, we consider a setup where distinct adapters instead of a shared adapter are finetuned for various tasks to achieve optimal performance. During inference, each request in a batch pertains to a different task and necessitates a unique adapter. Consider that we have finetuned distinct LoRA modules for b tasks, denoted as {Ai, Bi}bi=1 . For a batch of b requests represented as X ∈ Rb×l×d1 , where l is the maximum sequence length across the requests, each request requires a different LoRA module. To exploit the parallel processing capabilities of GPUs, the output Z of a linear layer can be computed as follows: First, the output from the pretrained layer is computed as Z0 = torch.mm (X, W 0). Subsequently, the intermediate output from the first low-rank matrix, ˆB ∈ Rb×d1×r (a concatenation of {Bi}bi=1 ), is obtained as Z10 = torch.bmm (X, ˆB). The output from the second low-rank matrix, ˆA ∈ Rb×r×d2 (a concatenation of {Ai}bi=1 ), follows as Z1 = torch.bmm (Z10 , ˆA). Finally, these outputs are summed to produce Z = Z0 + Z1. It is noteworthy that batch matrix multiplication (BMM), as implemented in torch.bmm , often introduces substantial overhead [ 1], reducing throughput and increasing latency, which adversely impacts user experience in time-sensitive applications. In contrast, prompt-based methods circumvent the use of BMM by appending trainable tokens to each request, simplifying the computational process. However, prompt-based methods with long prompt tokens are difficult to optimize, which degrades performance compared to other PEFTs [ 14 , 15 ]. (IA) 3 [ 25 ] proposes adapting LLM by multiplying the output from a linear layer with a trainable vector, involving only element-wise multiplication for efficient batching. A recent development, FLoRA [ 58 ], builds on (IA) 3 by employing two low-rank matrices while maintaining element-wise operations. Although our proposed method, RoAd, requires BMM, its sparse structure allows a reformulation of BMM and results in an overhead equivalent to element-wise multiplication. 

   2.3 Intervention and composability 

   Numerous studies [ 10 , 11 , 37 , 38 , 40 ] have provided support for the linear representation hypothesis [ 35 , 46 , 49 ] that concepts are represented within linear subspaces of neural network representations. To examine if a concept is captured within a linear subspace of a representation, Geiger et al. [11] 

   > 2Adapter here means the trained parameters since LoRA's architecture is also similar to an adapter.

   b denotes the hidden representation generated at row i and column k when the model processes an input, while s represents the corresponding representation when the model processes a different input. The matrix R ∈ Rr×d1 , consisting of orthogonal rows, serves as a low-rank projection matrix where d1 is the dimension of the representation and r is the subspace dimension under intervention. Equation (1) illustrates the application of a DII to b using a counterfactual source representation s.3

   Drawing inspiration from this established framework, a recent study, LoReFT [ 61 ], introduces a method for finetuning specific positions of the representations to adapt LLM. This study further demonstrates that several prior approaches of representation editing [ 54 , 60 , 67 ] can be effectively integrated within this framework. Interestingly, the application of RoAd to representations can also be conceptualized as DII, offering interpretability potential. To demonstrate one aspect of interpretability for RoAd, we primarily conduct a qualitative experiment focused on task composition. This experiment involves combining the weights of models trained on distinct tasks to showcase the capability for multitasking learning without the need for additional adaptation [16, 20, 61, 64, 66]. 


VERA

    This is the configuration class to store the configuration of a ['VeraModel'].
    Paper: https://r.jina.ai/https://arxiv.org/pdf/2310.11454

    Low-rank adapation (LoRA) is a popular method that reduces the number of train-able parameters when finetuning large language models, but still faces acute stor-age challenges when scaling to even larger models or deploying numerous per-user or per-task adapted models. In this work, we present Vector-based Random Matrix Adaptation (VeRA), which significantly reduces the number of trainable parameters compared to LoRA, yet maintains the same performance. It achieves this by using a single pair of low-rank matrices shared across all layers and learning small scaling vectors instead. We demonstrate its effectiveness on the GLUE and E2E benchmarks, image classification tasks, and show its application in instruction-tuning of 7B and 13B language models. 


   In this section, we introduce Vector-based Random Matrix Adaptation, a novel parameter-efficient finetuning method that builds upon and extends the state-of-the-art method, LoRA. The central in-novation in VeRA lies in the reparameterization of the low-rank matrices. Specifically, we freeze a single pair of randomly initialized matrices, shared across all adapted layers, and introduce train-able scaling vectors that allow for layer-wise adaptation, as shown in Figure 1. Similarly to LoRA, trained scaling vectors along with low-rank matrices can be merged into original weights, eliminat-ing additional inference latency.

AdaLoRA

    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/forum?id=lq62uWRJjiY

   herefore, many fine-tuning methods are proposed to learn incremental updates of pre-trained weights in a parameter efficient way, e.g., low-rank increments. These methods often evenly distribute the budget of incremental updates across all pre-trained weight matrices, and overlook the varying importance of different weight parameters. As a consequence, the fine-tuning performance is suboptimal. To bridge this gap, we propose AdaLoRA, which adaptively allocates the parameter budget among weight matrices according to their importance score. In particular, AdaLoRA parameterizes the incremental updates in the form of singular value decomposition. Such a novel approach allows us to effectively prune the singular values of unimportant updates, which is essentially to reduce their parameter budget but circumvent intensive exact SVD computations.

RandLoraModel

       This is the configuration class to store the configuration of a ['RandLoraModel'].
    Paper: https://huggingface.co/papers/2502.00987.

   This paper aims to answer this question by introducing RandLoRA, a parameter-efficient method that performs full-rank updates using a learned linear combinations of low-rank, non-trainable random matrices. Our method limits the number of trainable parameters by restricting optimization to diagonal scaling matrices applied to the fixed random matrices. This allows us to effectively overcome the low-rank limitations while maintaining parameter and memory efficiency during training. 

# 2025-09-29 10:46:37


summary:
- try ROAD alpha and theta as coeff
- try IA3 coeff


```py
import torch
import torch.nn as nn
from contextlib import contextmanager

# Base class for all PEFT adapter layers
from peft.tuners.tuners_utils import BaseTunerLayer

# Specific layer types for ROAD and IA3
try:
    from peft.tuners.road.layer import RoadLayer
    from peft.tuners.ia3.layer import IA3Layer
except ImportError:
    raise ImportError("Could not import PEFT ROAD/IA3 layers—please install 'peft'.")

@contextmanager
def AdapterModifier(
    model: nn.Module,
    coeff: float = 1.0,
    scale_param: str = 'theta',  # For ROAD: 'theta' (recommended) or 'alpha'; ignored for IA3
    adapter_name: str = "default",
):
    """
    A context manager to temporarily scale ROAD or IA3 adapter parameters by a coefficient.
    - For ROAD: Scales theta (angles, for directional/invertible steering) or alpha (magnitudes).
    - For IA3: Scales the ia3_l vector (activation scalers).

    Args:
      model:          The PEFT model with ROAD or IA3 adapters.
      coeff:          The scaling coefficient (default: 1.0).
      scale_param:    For ROAD: 'theta' or 'alpha' (default: 'theta'); ignored for IA3.
      adapter_name:   Name of the adapter to target (default: "default").

    FIXME: update and remove old parts and docs one I've worked out what works and what doesn't.
    Context & Task: 
      - Goal: Build reversible/scalable PEFT adapter control for contrastive learning (e.g., "honest" vs "dishonest" steering).
      - Usage: Two-pass training with coeff=1.0 (normal) and coeff=-1.0 (inverted) to learn bidirectional representations.
      - Challenge: Which adapter type + scaling approach works best for invertible steering without breaking pretrained capabilities?
      - Candidates: ROAD (rotation-based, geometrically reversible), IA3 (element-wise scaling, simple inversion), LoRA (additive, may not invert cleanly).

    Decision Rationale:
      - ROAD (from peft.tuners.road): Forward applies element-wise rotations via cos/sin(theta) * alpha on grouped activations.
        - Theta (angles): Core param (zero-init for identity). Scaling multiplies angles linearly (for small values), coeff=-1 inverts rotation exactly (geometric inverse). Recommended for contrastive/invertible steering (~70% confidence, based on 2D rotation math and paper's angular focus).
        - Alpha (scales): Multiplicative gain on cos/sin (one-init). Scaling amplifies/suppresses effect without direction change; coeff=-1 reflects (not pure inverse). Use for intensity control if theta warps (e.g., large angles).
        - Why multiply? Matches forward (element-wise mul); add/shift offsets arbitrarily (distorts geometry). BNB (8/4-bit): Params fp32, compatible.
      - IA3 (from peft.tuners.ia3): Scales activations via ia3_l vector mul. Coeff directly amplifies/inhibits (coeff=-1 flips); simple/invertible (~80% confidence, per paper's mixed-batch design). Why multiply? Inherent to method.
      - Prioritization: ROAD theta for subspace rotations/composability (best for steering concepts); IA3 for global scaling/simplicity (fallback if rotations fail). Test both on contrastive loss to validate inversion quality.
    """
    if scale_param not in ['theta', 'alpha'] and scale_param != 'auto':  # 'auto' for detection
        raise ValueError("scale_param must be 'theta', 'alpha', or ignored for IA3.")

    # store originals here
    original_states = []

    try:
        # --- ENTER: find & modify all adapter layers ---
        for name, module in model.named_modules():
            if isinstance(module, (RoadLayer, IA3Layer)) and adapter_name in module.active_adapters:
                if isinstance(module, RoadLayer):
                    if scale_param == 'theta':
                        param = module.road_theta[adapter_name]
                    else:  # alpha
                        param = module.road_alpha[adapter_name]
                    orig = param.data.clone()
                    original_states.append((module, 'road', scale_param, orig))
                    param.data.mul_(coeff)
                elif isinstance(module, IA3Layer):
                    param = module.ia3_l[adapter_name]
                    orig = param.data.clone()
                    original_states.append((module, 'ia3', orig))
                    param.data.mul_(coeff)

        yield

    finally:
        # --- EXIT: restore originals ---
        for module, kind, *extra in original_states:
            if kind == 'road':
                sp = extra[0]
                if sp == 'theta':
                    module.road_theta[adapter_name].data.copy_(extra[1])
                else:
                    module.road_alpha[adapter_name].data.copy_(extra[1])
            elif kind == 'ia3':
                module.ia3_l[adapter_name].data.copy_(extra[0])
```


There are also other transforms

https://arxiv.org/html/2405.20271v1

    Parameter-efficient finetuning (PEFT) has become ubiquitous to adapt foundation models to downstream task requirements while retaining their generalization ability. However, the amount of additionally introduced parameters and compute for successful adaptation and hyperparameter searches can explode quickly, especially when deployed at scale to serve numerous individual requests. To ensure effective, parameter-efficient, and hyperparameter-robust adaptation, we propose the ETHER transformation family, which performs Efficient fineTuning via HypErplane Reflections. By design, ETHER transformations require a minimal number of parameters, are less likely to deteriorate model performance, and exhibit robustness to hyperparameter and learning rate choices. In particular, we introduce ETHER and its relaxation ETHER+, which match or outperform existing PEFT methods with significantly fewer parameters ( ∼ 10-100 times lower than LoRA or OFT) across multiple image synthesis and natural language tasks without exhaustive hyperparameter tuning. Finally, we investigate the recent emphasis on Hyperspherical Energy retention for adaptation and raise questions on its practical utility. The code is available at https://github.com/mwbini/ether .


    In this work, we propose Efficient fineTuning via HypErplane Reflections (ETHER) - a new family of weight transformations, efficient in parameter count while preserving model abilities and being robust in convergence and learning rate choices. By default, ETHER transformations frame the tuning process as a search for suitable hyperplanes, along which weight vectors can be reflected based on the orthogonal Householder transformation (Householder, 1958). This keeps the distance to the transformation neutral element - the identity matrix - constant by construction and improves training stability while reducing the chance of deteriorating model performance. In addition, being built from single vectors, Householder transformations allow for efficient block-parallel matrix multiplication with minimal performance trade-offs.
