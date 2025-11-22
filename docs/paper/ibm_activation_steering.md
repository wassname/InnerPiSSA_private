Title: Programming Refusal with Conditional Activation Steering

URL Source: https://arxiv.org/html/2409.05907v3

Published Time: Wed, 19 Feb 2025 01:05:55 GMT

Markdown Content:
Bruce W. Lee 1,∗ Inkit Padhi 2 Karthikeyan Natesan Ramamurthy 2

Erik Miehling 2 Pierre Dognin 2 Manish Nagireddy 2 Amit Dhurandhar 2

1 University of Pennsylvania 2 IBM Research 

brucelws@seas.upenn.edu inkpad@ibm.com knatesa@us.ibm.com

###### Abstract

LLMs have shown remarkable capabilities, but precisely controlling their response behavior remains challenging. Existing activation steering methods alter LLM behavior indiscriminately, limiting their practical applicability in settings where selective responses are essential, such as content moderation or domain-specific assistants. In this paper, we propose Conditional Activation Steering (CAST), which analyzes LLM activation patterns during inference to selectively apply or withhold activation steering based on the input context. Our method is based on the observation that different categories of prompts activate distinct patterns in the model’s hidden states. Using CAST, one can systematically control LLM behavior with rules like “if input is about hate speech or adult content, then refuse” or “if input is not about legal advice, then refuse.” This allows for selective modification of responses to specific content while maintaining normal responses to other content, all without requiring weight optimization. We release an open-source implementation of our framework at [github.com/IBM/activation-steering](https://github.com/IBM/activation-steering).

1 Introduction
--------------

A striking feature of large language models (LLMs) is their ability to process high-level concepts through rich representations in their activations. This feature has given rise to techniques like activation steering (Turner et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib75)), which leverage these learned representations to efficiently and predictably alter LLM behavior (Wang et al., [2024b](https://arxiv.org/html/2409.05907v3#bib.bib81); Zou et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib92); Rimsky et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib57)).

Problem: Lack of conditional control in activation steering. Activation steering offers a promising alternative to optimization-based techniques by directly manipulating the model’s native representations, often requiring only a simple activation addition step during each forward call (Turner et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib75)). While activation steering has shown promise in altering LLM behavior, such as removing or inducing refusal behavior, a key limitation of current methods is the inability to condition when and what to refuse (Zheng et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib90); Ghandeharioun et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib18)). That is, adding a “refusal vector” using existing activation steering methods increases refusal rates indiscriminately across all inputs, limiting the model’s utility (Arditi et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib4)).

Contribution: Adding “control” to activation steering. We introduce Conditional Activation Steering (CAST), a method that enables fine-grained, context-dependent control over LLM behaviors. We introduce a new type of steering vector in the activation steering formulation, the condition vector, representing certain activation patterns induced by the prompt during the inference process. A simple similarity calculation between this condition vector and the model’s activation at inference time effectively serves as a switch, determining whether to apply the refusal vector. This approach allows for selective refusal of harmful prompts while maintaining the ability to respond to harmless ones, as depicted in Figure [1](https://arxiv.org/html/2409.05907v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Programming Refusal with Conditional Activation Steering"). A breakdown of this figure is presented in Table [3](https://arxiv.org/html/2409.05907v3#A1.T3 "Table 3 ‣ A.2 Details of Conditional Activation Steering ‣ Appendix A Understanding Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"). Furthermore, CAST maintains the data, runtime, and compute efficiency of activation steering (Figure [6](https://arxiv.org/html/2409.05907v3#S4.F6 "Figure 6 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")) while adding controllability, enabling the implementation of behavioral rules in LLMs without significant costs.

![Image 1: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/intro-bar-alt.png)

Figure 1: Conditional activation steering induces targeted refusal. Activation steering (AST) induces the model to indiscriminately refuse all prompts, including harmless ones (blue bars). Conditional activation steering (CAST) allows selective refusal, refusing harmful prompts while minimizing the harmless refusal rate.

Application: Selecting what to refuse. Many alignment goals concern contextually refusing specific classes of instructions (Anwar et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib3)). Traditional methods like preference modeling are resource-intensive and struggle with subjective, black-box rewards (Feng et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib17); Pitis, [2023](https://arxiv.org/html/2409.05907v3#bib.bib51); Rafailov et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib54); Stiennon et al., [2020](https://arxiv.org/html/2409.05907v3#bib.bib65); [Hayum et al.,](https://arxiv.org/html/2409.05907v3#bib.bib22)). Additionally, the definition of harmful content varies across contexts (He et al., [2024b](https://arxiv.org/html/2409.05907v3#bib.bib24); Sorensen et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib62); Santurkar et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib58)), complicating the creation of universal harm models. The usage context further complicates this variability; for instance, discussing medical advice might be harmful in some situations (Wang et al., [2023b](https://arxiv.org/html/2409.05907v3#bib.bib79)) but essential in others, such as in medical chatbots (Xie et al., [2024a](https://arxiv.org/html/2409.05907v3#bib.bib84)). In this paper, we show CAST can implement behavioral rules like “if input is about hate speech or adult content, then refuse” (Figure [8](https://arxiv.org/html/2409.05907v3#S5.F8 "Figure 8 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")a) or “if input is not about legal advice, then refuse” (Figure [9](https://arxiv.org/html/2409.05907v3#S5.F9 "Figure 9 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")a), allowing for selective modification of responses to specific content without weight optimization.

On a technical level, our primary insight is that different prompts consistently activate distinct patterns in the model’s hidden states during inference (Hu et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib27)). These patterns can be extracted as a steering vector and used as reference points for detecting specific prompt categories or contexts. This observation allows us to use steering vectors not only as behavior modification mechanisms but also as condition indicators, which we term “condition vectors.” Our specific contributions are as follows:

*   1)Framework: We introduce conditional activation steering and condition vectors, which adds a new dimension of controllability to existing methods. 
*   2)Application: We demonstrate the logical composition of condition vectors to create custom refusal conditions. This is a key step towards tailoring model behavior to specific needs. 
*   3)Codebase: We release a general-purpose activation steering toolkit with demo datasets for the broader activation engineering community <placeholder: open-source GitHub link>. 

2 Background
------------

How do transformers perform inference? Transformer models, particularly decoder-only variants, perform inference by sequentially processing input tokens through a stack of layers (Radford et al., [2018](https://arxiv.org/html/2409.05907v3#bib.bib53); Vaswani et al., [2017](https://arxiv.org/html/2409.05907v3#bib.bib76)). The key to understanding the operation lies in how information flows and accumulates through these layers (Lad et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib34); Shai et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib61); Elhage et al., [2021](https://arxiv.org/html/2409.05907v3#bib.bib16)). The process begins with converting the prompt into token embeddings, which serve as initial inputs. Each layer transforms these activations using its internal mechanisms, like learned weights. Each layer’s output combines processed information with its input, preserving and building upon earlier computations. As activations flow through the layers, the model constructs increasingly complex representations. The final layer’s output is used for decoding - predicting the next token via an operation over the model’s vocabulary. This predicted token is then used for subsequent predictions.

Behavior steering. One could intervene in any of the abovementioned five steps - weights, decoding, prompt, token embedding, and activations - to alter model behavior (Tamoyan et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib67); Phan et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib50); Chai et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib11); Li et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib37); Han et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib21); Wang et al., [2024b](https://arxiv.org/html/2409.05907v3#bib.bib81)). For example, one could use role-play prompts to simulate and create AI patients (Louie et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib40)). Or one could use preference optimization methods like direct preference optimization to update weights and steer the LLM towards more empathetic behaviors (Sotolar, [2024](https://arxiv.org/html/2409.05907v3#bib.bib63)). Activation steering is a class of methods that intervenes in the information flow within LLMs from layer to layer to alter the model behavior.

Activation steering. An alternative method for influencing the behavior of LLMs, activation steering modifies their internal activations during inference. This approach typically involves three key steps. First, a steering vector is extracted, often by computing the difference in activations between examples that exhibit a desired behavior and those that don’t. Second, during inference, this vector is added to the model’s hidden states at a chosen layer, scaled by a hyperparameter. Finally, the model completes the generation using these modified activations. For the case of activation addition (ActAdd) (Turner et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib75)), the intervention can be represented mathematically as:

𝐡′←𝐡+α⋅𝐯←superscript 𝐡′𝐡⋅𝛼 𝐯\mathbf{h}^{{}^{\prime}}\leftarrow\mathbf{h}+\alpha\cdot\mathbf{v}\vspace{1mm}bold_h start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT ′ end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT ← bold_h + italic_α ⋅ bold_v

where 𝐡 𝐡\mathbf{h}bold_h is the hidden state at the layer, 𝐯 𝐯\mathbf{v}bold_v is the steering vector for the layer, and α 𝛼\alpha italic_α is a scaling factor. Stronger scaling can disrupt coherence while weaker scaling may be ineffective (Rimsky et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib57)). In an ideal case where steering vectors are well-extracted, this method allows for predictable LLM behavior steering without altering model weights, enabling applications such as reducing bias (Lu & Rimsky, [2024](https://arxiv.org/html/2409.05907v3#bib.bib41); Adila et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib1)) or preventing overly confident responses (Rahn et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib55)).

Recent research has proposed several methods to improve upon the basic activation addition approach (Wang et al., [2024a](https://arxiv.org/html/2409.05907v3#bib.bib77); Stickland et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib64); Qiu et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib52); Yin et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib87); Wu et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib83)). These techniques address various limitations of the ActAdd method and collectively fall under the broader category of activation engineering. In this paper, we propose a vertical expansion by adding the new dimension of _condition_, greatly improving the utility of existing activation steering methods.

3 Conditional Activation Steering
---------------------------------

### 3.1 Overview

A common limitation of the existing activation steering methods is that one cannot condition the model’s behavior on context, as these methods typically apply modifications uniformly across all inputs regardless of context (He et al., [2024a](https://arxiv.org/html/2409.05907v3#bib.bib23)). Simple activation steering of a model indiscriminately affects all inputs, rendering the steered model much less useful for its application (Turner et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib75); Cui et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib13); Wen et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib82); Brahman et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib9)). We show that one can induce conditional behavior (Figure [2](https://arxiv.org/html/2409.05907v3#S3.F2 "Figure 2 ‣ 3.1 Overview ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering")) by leveraging two types of vectors: condition and behavior vectors.

𝐡′←𝐡+f⁢(sim⁢(𝐡,proj 𝐜⁢𝐡))⋅α⋅𝐯←superscript 𝐡′𝐡⋅𝑓 sim 𝐡 subscript proj 𝐜 𝐡 𝛼 𝐯\mathbf{h}^{{}^{\prime}}\leftarrow\mathbf{h}+f(\text{sim}(\mathbf{h},\text{% proj}_{\mathbf{c}}\mathbf{h}))\cdot\alpha\cdot\mathbf{v}\vspace{1mm}bold_h start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT ′ end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT ← bold_h + italic_f ( sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) ) ⋅ italic_α ⋅ bold_v

where 𝐡 𝐡\mathbf{h}bold_h is the hidden state, 𝐜 𝐜\mathbf{c}bold_c is the condition vector, 𝐯 𝐯\mathbf{v}bold_v is the behavior vector, and α 𝛼\alpha italic_α is a scaling factor. The projection of 𝐡 𝐡\mathbf{h}bold_h onto 𝐜 𝐜\mathbf{c}bold_c is given by proj 𝐜⁢𝐡=(𝐜⊗𝐜 𝐜⋅𝐜)⁢𝐡 subscript proj 𝐜 𝐡 tensor-product 𝐜 𝐜⋅𝐜 𝐜 𝐡\text{proj}_{\mathbf{c}}\mathbf{h}=\left(\frac{\mathbf{c}\otimes\mathbf{c}}{% \mathbf{c}\cdot\mathbf{c}}\right)\mathbf{h}proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h = ( divide start_ARG bold_c ⊗ bold_c end_ARG start_ARG bold_c ⋅ bold_c end_ARG ) bold_h. Intuitively, based on how well aligned the hidden state 𝐡 𝐡\mathbf{h}bold_h is with the condition vector 𝐜 𝐜\mathbf{c}bold_c, the function f 𝑓 f italic_f determines whether to apply the behavior vector based on the similarity between the hidden state and its projection using the condition vector. Throughout the paper, we use cosine similarity, defined as sim⁢(𝐡,𝐠)=𝐡⋅𝐠|𝐡|⁢|𝐠|sim 𝐡 𝐠⋅𝐡 𝐠 𝐡 𝐠\text{sim}(\mathbf{h},\mathbf{g})=\frac{\mathbf{h}\cdot\mathbf{g}}{|\mathbf{h}% ||\mathbf{g}|}sim ( bold_h , bold_g ) = divide start_ARG bold_h ⋅ bold_g end_ARG start_ARG | bold_h | | bold_g | end_ARG.

Figure 2: Enabling “targeted” activation steering. Unlike simple refusal activation steering that blocks all prompts, CAST employs a condition vector to selectively steer the model. This approach enables the model to (a) refuse harmful requests while (b) remaining responsive to harmless prompts. Model: Qwen 1.5 Chat 1.8B.

Behavior vector. We use the term “behavior vector” to refer to what previous activation steering methods call a “steering vector” to emphasize its focus on modifying specific behaviors. A behavior vector 𝐯 𝐯\mathbf{v}bold_v is a one-dimensional vector matching the model’s hidden state dimensions that induces specific behaviors. When added to layer representations during a forward pass with scaling factor α 𝛼\alpha italic_α, it predictably alters model behavior (e.g., inducing refusal). In addition to setting the right scaling factor α 𝛼\alpha italic_α, one can specify to which layers to apply the behavior vector. While specific implementations vary in the literature, our implementation calculates a different vector 𝐯 l subscript 𝐯 𝑙\mathbf{v}_{l}bold_v start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT for each layer l 𝑙 l italic_l, as behavior representations vary. Thus, when we mention adding a behavior vector from layers 15-20, we’re referring to adding the corresponding 𝐯 15,𝐯 16,…,𝐯 20 subscript 𝐯 15 subscript 𝐯 16…subscript 𝐯 20\mathbf{v}_{15},\mathbf{v}_{16},...,\mathbf{v}_{20}bold_v start_POSTSUBSCRIPT 15 end_POSTSUBSCRIPT , bold_v start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT , … , bold_v start_POSTSUBSCRIPT 20 end_POSTSUBSCRIPT to their respective layers.

Condition vector. A condition vector 𝐜 𝐜\mathbf{c}bold_c captures a class of instructions to condition on, extracted similarly to behavior vectors and matching hidden state dimensions (e.g., 1x4096 for Llama2, which has a hidden size of 4096). For instance, a condition vector might capture discrimination or adult content. It acts as a trigger, determining when to apply the behavior vector based on the model’s current hidden state. Since we also calculate a different vector 𝐜 l subscript 𝐜 𝑙\mathbf{c}_{l}bold_c start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT to each layer l 𝑙 l italic_l, one can also choose which layer to condition. When the condition is activated during text generation, the behavior vector is added to all subsequent forward passes. This allows the model’s behavior to change based on specific conditions in the input or generated text rather than always applying the behavior vector.

Checking if condition was met. The term sim⁢(𝐡,proj 𝐜⁢𝐡)sim 𝐡 subscript proj 𝐜 𝐡\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h})sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) computes the degree to which the condition is met using cosine similarity. The thresholding function f 𝑓 f italic_f then determines whether this degree is sufficient to trigger the behavior modification. Though one would be able to design more complex thresholding functions, we use a simple step function for binary output in this paper:

f⁢(sim⁢(𝐡,proj 𝐜⁢𝐡))={1 if sim⁢(𝐡,proj 𝐜⁢𝐡)>θ 0 otherwise 𝑓 sim 𝐡 subscript proj 𝐜 𝐡 cases 1 if sim 𝐡 subscript proj 𝐜 𝐡 𝜃 0 otherwise f(\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h}))=\begin{cases}1&% \text{if }\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h})>\theta\\ 0&\text{otherwise}\end{cases}\vspace{1mm}italic_f ( sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) ) = { start_ROW start_CELL 1 end_CELL start_CELL if roman_sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) > italic_θ end_CELL end_ROW start_ROW start_CELL 0 end_CELL start_CELL otherwise end_CELL end_ROW

Here, each layer in an LLM might represent the same condition in different directions and sim⁢(𝐡,proj 𝐜⁢𝐡)>θ sim 𝐡 subscript proj 𝐜 𝐡 𝜃\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h})>\theta sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) > italic_θ could be sim⁢(𝐡,proj 𝐜⁢𝐡)<θ sim 𝐡 subscript proj 𝐜 𝐡 𝜃\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h})<\theta sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) < italic_θ depending on the layer. This binary approach allows for a clear distinction between when the condition is met and when it is not, providing a straightforward mechanism for activating the behavior modification. We use cosine similarity to check condition based on the directional similarity between the hidden state and its projection using the condition vector rather than magnitude (Hsu et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib26)). In practice, we apply a non-linear transformation sim⁢(𝐡,tanh⁢(proj 𝐜⁢𝐡))sim 𝐡 tanh subscript proj 𝐜 𝐡\text{sim}(\mathbf{h},\text{tanh}(\text{proj}_{\mathbf{c}}\mathbf{h}))sim ( bold_h , tanh ( proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) ) for more predictable behavior.

Multi-conditioning. As mentioned in Section 1, one could also break down broader alignment goals into smaller, more definitive categories and predictably induce refusal behaviors for each. For instance, instead of conditioning a model to refuse “harmful” instructions in general, we could create specific conditions for “adult content,” “social stereotypes,” or “false advertising.” Such multi-conditional behavior can easily be implemented by expanding the thresholding function like:

f⁢(⋅)={1 if sim⁢(𝐡,proj a⁢d⁢u⁢l⁢t⁢𝐡)>θ a⁢d⁢u⁢l⁢t⁢or sim⁢(𝐡,proj s⁢t⁢e⁢r⁢e⁢o⁢t⁢y⁢p⁢e⁢𝐡)>θ s⁢t⁢e⁢r⁢e⁢o⁢t⁢y⁢p⁢e 0 otherwise 𝑓⋅cases 1 if sim 𝐡 subscript proj 𝑎 𝑑 𝑢 𝑙 𝑡 𝐡 subscript 𝜃 𝑎 𝑑 𝑢 𝑙 𝑡 or sim 𝐡 subscript proj 𝑠 𝑡 𝑒 𝑟 𝑒 𝑜 𝑡 𝑦 𝑝 𝑒 𝐡 subscript 𝜃 𝑠 𝑡 𝑒 𝑟 𝑒 𝑜 𝑡 𝑦 𝑝 𝑒 0 otherwise f(\cdot)=\begin{cases}1&\text{if }\text{sim}(\mathbf{h},\text{proj}_{adult}% \mathbf{h})>\theta_{adult}\text{ or }\text{sim}(\mathbf{h},\text{proj}_{% stereotype}\mathbf{h})>\theta_{stereotype}\\ 0&\text{otherwise}\end{cases}\vspace{1mm}italic_f ( ⋅ ) = { start_ROW start_CELL 1 end_CELL start_CELL if roman_sim ( bold_h , proj start_POSTSUBSCRIPT italic_a italic_d italic_u italic_l italic_t end_POSTSUBSCRIPT bold_h ) > italic_θ start_POSTSUBSCRIPT italic_a italic_d italic_u italic_l italic_t end_POSTSUBSCRIPT or roman_sim ( bold_h , proj start_POSTSUBSCRIPT italic_s italic_t italic_e italic_r italic_e italic_o italic_t italic_y italic_p italic_e end_POSTSUBSCRIPT bold_h ) > italic_θ start_POSTSUBSCRIPT italic_s italic_t italic_e italic_r italic_e italic_o italic_t italic_y italic_p italic_e end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL 0 end_CELL start_CELL otherwise end_CELL end_ROW

General expectations Implementing conditional behaviors in LLMs using CAST generally follows the pipeline: 1. gather contrasting example responses/prompts for desired behavior/condition 𝒟+superscript 𝒟\mathcal{D}^{+}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT and other behavior/condition 𝒟−superscript 𝒟\mathcal{D}^{-}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT, 2. extract behavior/condition vector, 3. find optimal intervention points for behavior/condition vector, 4. steer. The model itself does not undergo any weight update.

Step 3 represents the most time-intensive part of our process, involving both automated and manual elements. For the behavior vector, similar to other works in activation steering, we manually search for the appropriate intervention strength and layers. However, as demonstrated in Appendix [C](https://arxiv.org/html/2409.05907v3#A3 "Appendix C Intervention Points and Grid Search Algorithm ‣ Programming Refusal with Conditional Activation Steering"), most models represent refusal behavior at similar depths. For the condition vector, we use a grid search (Appendix [C.2](https://arxiv.org/html/2409.05907v3#A3.SS2 "C.2 Best Condition Point (Grid Search) Algorithm ‣ Appendix C Intervention Points and Grid Search Algorithm ‣ Programming Refusal with Conditional Activation Steering")) algorithm that determines the best threshold, layer, and comparison direction (>>> or <<<). The majority of our reported experiments are replicable within an hour, with the grid search being the primary time-consuming component. We share more details below.

### 3.2 Preparing Dataset and Model

As mentioned, contrast datasets are needed to extract behavior or condition vectors. For the refusal behavior vector, we randomly select 100 instructions from the Alpaca dataset (Taori et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib69)) and append them with 100 typical refusal or compliance behavior prefixes as responses, as shown in Figure [3](https://arxiv.org/html/2409.05907v3#S3.F3 "Figure 3 ‣ 3.2 Preparing Dataset and Model ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"). Considering every combination of these creates 10,000 pairs of contrasting data points for 𝒟 refuse+subscript superscript 𝒟 refuse\mathcal{D}^{+}_{\text{refuse}}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT start_POSTSUBSCRIPT refuse end_POSTSUBSCRIPT and 𝒟 comply−subscript superscript 𝒟 comply\mathcal{D}^{-}_{\text{comply}}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT start_POSTSUBSCRIPT comply end_POSTSUBSCRIPT. We commit to this setup for the refusal behavior vector throughout our research.

Figure 3: Contrastive data instances. For behavior vectors, we record mean activations at the contrasting suffixes, whereas for condition vectors, we record at the full contrasting prompts.

We explore different condition vectors for our experiments. In Section 4, we create 𝒟+superscript 𝒟\mathcal{D}^{+}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT and 𝒟−superscript 𝒟\mathcal{D}^{-}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT using Sorry-Bench (Xie et al., [2024b](https://arxiv.org/html/2409.05907v3#bib.bib85)) and Alpaca. For Section 5, we use paraphrased Alpaca data. When additional data were required, we primarily relied on machine generation, including paraphrasing for specific conditions. We did not apply additional filtering to the train datasets beyond basic quality checks as we found this process generally robust to small data perturbations. For both setups, the authors manually checked every item in the test set to ensure integrity but did not modify or correct any. See Appendix [B](https://arxiv.org/html/2409.05907v3#A2 "Appendix B Constrasting Pair Generation Details ‣ Programming Refusal with Conditional Activation Steering") for data generation details and examples. Lastly, we experiment with models described in Table [1](https://arxiv.org/html/2409.05907v3#S3.T1 "Table 1 ‣ 3.2 Preparing Dataset and Model ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering").

Table 1: Overview of models used in this study. Models are selected based on experimental suitability and the availability of comprehensive documentation. We give additional details on each model in Appendix [D](https://arxiv.org/html/2409.05907v3#A4 "Appendix D Model Descriptions / Dataset Locations ‣ Programming Refusal with Conditional Activation Steering").

### 3.3 Extracting Condition and Behavior Vectors

The extraction of steering vectors begins with a set of contrastive examples - pairs of inputs that exemplify the presence and absence of a target behavior or condition that we built in Section [3.2](https://arxiv.org/html/2409.05907v3#S3.SS2 "3.2 Preparing Dataset and Model ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"). These pairs serve as the basis for identifying relevant directions in the model’s hidden state space. We employ a combination of methods that have been reported to work well for vector extraction.

For a given layer l∈[L]𝑙 delimited-[]𝐿 l\in[L]italic_l ∈ [ italic_L ], we first compute the hidden states for both positive and negative examples in our contrastive pairs. Let 𝐇 l+superscript subscript 𝐇 𝑙\mathbf{H}_{l}^{+}bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT and 𝐇 l−superscript subscript 𝐇 𝑙\mathbf{H}_{l}^{-}bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT represent all hidden states 𝐡 l subscript 𝐡 𝑙\mathbf{h}_{l}bold_h start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT for positive 𝒟+superscript 𝒟\mathcal{D}^{+}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT and negative 𝒟−superscript 𝒟\mathcal{D}^{-}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT examples respectively at layer l 𝑙 l italic_l. The computation of these hidden states differs between behavior vectors and condition vectors, as illustrated in Figure [3](https://arxiv.org/html/2409.05907v3#S3.F3 "Figure 3 ‣ 3.2 Preparing Dataset and Model ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"). For behavior vectors, we take the average hidden states for suffixes of each example. For condition vectors, we take the average hidden states for all tokens of each example to capture a more holistic representation of the input.

We then mean-center 𝐇 l+superscript subscript 𝐇 𝑙\mathbf{H}_{l}^{+}bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT and 𝐇 l−superscript subscript 𝐇 𝑙\mathbf{H}_{l}^{-}bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT, following the ideas from Tan et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib68)); Jorgensen et al. ([2023](https://arxiv.org/html/2409.05907v3#bib.bib29)) and apply Principal Component Analysis following Ball et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib8)); Adila et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib1)); Zou et al. ([2023](https://arxiv.org/html/2409.05907v3#bib.bib92)). The first principal component resulting from this process becomes our behavior/condition 𝐯𝐞𝐜𝐭𝐨𝐫 l subscript 𝐯𝐞𝐜𝐭𝐨𝐫 𝑙\mathbf{vector}_{l}bold_vector start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT for layer l 𝑙 l italic_l. This process is repeated for each specified layer, resulting in a set of layer-specific steering vectors {𝐯𝐞𝐜𝐭𝐨𝐫 l∣l∈L}conditional-set subscript 𝐯𝐞𝐜𝐭𝐨𝐫 𝑙 𝑙 𝐿\{\mathbf{vector}_{l}\mid l\in L\}{ bold_vector start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ∣ italic_l ∈ italic_L }. The extraction of vectors can be expressed as below, where PCA⁢(⋅)PCA⋅\text{PCA}(\cdot)PCA ( ⋅ ) represents the operation of extracting the first principal component:

𝐯𝐞𝐜𝐭𝐨𝐫 l=PCA⁢(𝐇 l+−μ l,𝐇 l−−μ l)subscript 𝐯𝐞𝐜𝐭𝐨𝐫 𝑙 PCA superscript subscript 𝐇 𝑙 subscript 𝜇 𝑙 superscript subscript 𝐇 𝑙 subscript 𝜇 𝑙\mathbf{vector}_{l}=\text{PCA}(\mathbf{H}_{l}^{+}-\mathbf{\mu}_{l},\mathbf{H}_% {l}^{-}-\mathbf{\mu}_{l})bold_vector start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = PCA ( bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT )

The PCA input (𝐇 l+−μ l,𝐇 l−−μ l)superscript subscript 𝐇 𝑙 subscript 𝜇 𝑙 superscript subscript 𝐇 𝑙 subscript 𝜇 𝑙(\mathbf{H}_{l}^{+}-\mathbf{\mu}_{l},\mathbf{H}_{l}^{-}-\mathbf{\mu}_{l})( bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) is a matrix of mean-centered examples, with each row alternating positive (𝐡 1+−μ l superscript subscript 𝐡 1 subscript 𝜇 𝑙\mathbf{h}_{1}^{+}-\mathbf{\mu}_{l}bold_h start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT) and negative examples (𝐡 1−−μ l superscript subscript 𝐡 1 subscript 𝜇 𝑙\mathbf{h}_{1}^{-}-\mathbf{\mu}_{l}bold_h start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT). Here, μ l=(𝐇 l++𝐇 l−)/2 subscript 𝜇 𝑙 superscript subscript 𝐇 𝑙 superscript subscript 𝐇 𝑙 2\mathbf{\mu}_{l}=(\mathbf{H}_{l}^{+}+\mathbf{H}_{l}^{-})/2 italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ( bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT + bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT ) / 2 is the mean activation all examples 𝐇 l+superscript subscript 𝐇 𝑙\mathbf{H}_{l}^{+}bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT and 𝐇 l−superscript subscript 𝐇 𝑙\mathbf{H}_{l}^{-}bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT. This centers the data cloud, ensuring the principal components are computed relative to this center rather than being influenced by any overall offset in the data. The mean-centered positive and negative examples are alternatively concatenated and passed to PCA, which computes the direction of maximum variance. This direction, representing the most significant distinction between positive and negative examples, becomes our 𝐯𝐞𝐜𝐭𝐨𝐫 l subscript 𝐯𝐞𝐜𝐭𝐨𝐫 𝑙\mathbf{vector}_{l}bold_vector start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT for layer l 𝑙 l italic_l.

4 Conditioned Refusal: Selectively Steering on Harmful Prompts
--------------------------------------------------------------

In this section, we explore the basic use of conditional steering by steering a model to refuse harmful prompts while complying with harmless ones. Apart from demonstrating that a language model can be conditioned from inside on the fly, we also share some key properties of conditional steering.

Experimental setup. To obtain our contrast dataset (𝒟+superscript 𝒟\mathcal{D}^{+}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT, 𝒟−superscript 𝒟\mathcal{D}^{-}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT) on the harmful condition, we started by machine-generating 90 harmful prompts for each of the 45 harm categories as identified by Xie et al. ([2024b](https://arxiv.org/html/2409.05907v3#bib.bib85)). We use these 4,050 synthetically generated harmful prompts as our 𝒟 harmful+subscript superscript 𝒟 harmful\mathcal{D}^{+}_{\text{harmful}}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT start_POSTSUBSCRIPT harmful end_POSTSUBSCRIPT. For each of these harmful prompts, we randomly sample a benign instruction from the Alpaca dataset to create 𝒟 harmless−subscript superscript 𝒟 harmless\mathcal{D}^{-}_{\text{harmless}}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT start_POSTSUBSCRIPT harmless end_POSTSUBSCRIPT. Following the process outlined in Section [3.3](https://arxiv.org/html/2409.05907v3#S3.SS3 "3.3 Extracting Condition and Behavior Vectors ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"), we then extract the harmful condition vector 𝐜 harmful subscript 𝐜 harmful\mathbf{c}_{\text{harmful}}bold_c start_POSTSUBSCRIPT harmful end_POSTSUBSCRIPT. We then use a grid search algorithm to identify the best combination of threshold θ 𝜃\theta italic_θ, layer l 𝑙 l italic_l, and comparison direction (>>> or <<<) that best separates the two classes of training data. This concept is illustrated in Figure [4](https://arxiv.org/html/2409.05907v3#S4.F4 "Figure 4 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")d, where we perform the condition checking operation at layer 7 and activate the behavior vector 𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT when sim⁢(𝐡,proj 𝐜⁢𝐡)sim 𝐡 subscript proj 𝐜 𝐡\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h})sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) was smaller than 0.048.

![Image 2: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/result.png)

Figure 4: Conditioning behavior from inside. (a)-(c): T-SNE of prompt embeddings and refusal probability maps for base, activation steered, and conditionally steered models. (d): sim⁢(𝐡,proj 𝐜⁢𝐡)sim 𝐡 subscript proj 𝐜 𝐡\text{sim}(\mathbf{h},\text{proj}_{\mathbf{c}}\mathbf{h})sim ( bold_h , proj start_POSTSUBSCRIPT bold_c end_POSTSUBSCRIPT bold_h ) across layers 5-7 for 𝒟 harmful+subscript superscript 𝒟 harmful\mathcal{D}^{+}_{\text{harmful}}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT start_POSTSUBSCRIPT harmful end_POSTSUBSCRIPT and 𝒟 harmless−subscript superscript 𝒟 harmless\mathcal{D}^{-}_{\text{harmless}}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT start_POSTSUBSCRIPT harmless end_POSTSUBSCRIPT. Highlighted portions indicate 25th-75th percentiles. Model: Hermes 2 Pro.

Result: Activation steering can be used to induce conditional behaviors. We test the conditional activation steering performance on 500 unseen Alpaca (harmless) and 450 unseen Sorry-Bench (harmful) test sets. The results are presented in Figure [1](https://arxiv.org/html/2409.05907v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Programming Refusal with Conditional Activation Steering") with a subset of the data in Table [2](https://arxiv.org/html/2409.05907v3#S4.T2 "Table 2 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering"). Across all seven tested models, we observe that conditioning a behavior vector 𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT on condition vector 𝐜 harmful subscript 𝐜 harmful\mathbf{c}_{\text{harmful}}bold_c start_POSTSUBSCRIPT harmful end_POSTSUBSCRIPT selectively increases refusal rates for harmful content while leaving harmless prompt refusal rates largely unchanged. In contrast, simply adding a behavior vector 𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT like standard activation steering increased refusal rates indiscriminately across all prompts. Figures [4](https://arxiv.org/html/2409.05907v3#S4.F4 "Figure 4 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")a-c demonstrates how the conditioning operation partitions the prompt space.

![Image 3: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/duality.png)

Figure 5: Duality and modulation properties. (a)→→\rightarrow→(d): Flipping the comparison direction (from <<< to >>>) intervenes on the exact complement set of inputs. (a)↔↔\leftrightarrow↔(b)↔↔\leftrightarrow↔(c): one could progressively loosen or tighten the safety guardrail using θ 𝜃\theta italic_θ. 

Property: Duality. As seen in Figure [4](https://arxiv.org/html/2409.05907v3#S4.F4 "Figure 4 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")d, this conditioning process is systematic in nature as we can manually choose the point of intervention. One consequence of this is that conditioning exhibits a dual nature: flipping the comparison direction (from <<< to >>> or vice versa) results in intervening on the exact complement of the original set of hidden states that triggered the condition. This duality enables complementary control over the model’s behavior, allowing one to not only condition the model to refuse harmful prompts but also, if desired, to selectively refuse harmless prompts. See Figure [5](https://arxiv.org/html/2409.05907v3#S4.F5 "Figure 5 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")d.

Property: Modulation. Our steering approach offers flexible control rather than being uniform across all contexts, with the threshold θ 𝜃\theta italic_θ modulating the required alignment between the input and the harm direction defined in 𝐜 harmful subscript 𝐜 harmful\mathbf{c}_{\text{harmful}}bold_c start_POSTSUBSCRIPT harmful end_POSTSUBSCRIPT. In Figures [5](https://arxiv.org/html/2409.05907v3#S4.F5 "Figure 5 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")a-c, using the <<< comparison, lowering θ 𝜃\theta italic_θ narrows the range of hidden states triggering the condition while raising it broadens this range. This property allows us to adjust the model’s sensitivity to potentially harmful content. While this offers the potential for finer condition control, we do not explore it further in this study. We use threshold values determined by grid search, which maximizes the F1 score to balance false and true refusal (Appendix [C.2](https://arxiv.org/html/2409.05907v3#A3.SS2 "C.2 Best Condition Point (Grid Search) Algorithm ‣ Appendix C Intervention Points and Grid Search Algorithm ‣ Programming Refusal with Conditional Activation Steering")).

Table 2: Refusal rate (%) of conditionally steered models vs. reference models. “Discrepancy” shows the difference between harmful and harmless percentages. Arrows indicate a change from the base model. References show how the top safety-aligned models would behave on the same test set.

*Reference A: LLaMA3.1 Inst 8B. Reference B: LLaMA2 Chat 13B.

**These are just examples of safe behaviors. Reference models might have 

been aligned using different harm taxonomies.

![Image 4: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/linearity.png)

*Showing Qwen 1.8B, Danube 4B, OLMo 7B

Figure 6: Saturation and linear time scaling. (a): Performance of conditional steering plateaus. (b): Condition vector extraction time increases linearly with sample size (y-axis is a log scale).

Property: Saturation. Unlike most weight optimization methods, where performance often scales with increased data volume (Das et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib14); Metcalf et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib45); Ansell et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib2)), conditional activation steering tends to reach a performance plateau relatively quickly. As shown in Figure [6](https://arxiv.org/html/2409.05907v3#S4.F6 "Figure 6 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")a, the method’s effectiveness stabilizes after a certain point. This saturation might be attributed to the fact that conditional steering leverages the model’s existing representations. Consequently, performance appears more dependent on the model’s inherent capacity to represent certain concepts and how well the chosen data instances represent the target concept rather than on the sheer volume of conditioning data. Notably, the method also exhibits linear time scaling property (Figure [6](https://arxiv.org/html/2409.05907v3#S4.F6 "Figure 6 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering")b). The condition vector extraction time increases linearly with the number of samples, as this process is primarily determined by the number of inferences the model must make for us to record hidden states.

5 Programmed Refusal: Logical Composition of Condition Vector
-------------------------------------------------------------

Moving beyond the general concept of refusing harmfulness, we demonstrate the creation of more fine-grained condition vectors. We create five example condition vectors from categories - hate speech, legal opinion, sexual context, health consultation, and crime planning - in Liu et al. ([2023](https://arxiv.org/html/2409.05907v3#bib.bib39)) to explore these ideas. Our experiments demonstrate the capacity to (1) selectively modulate refusal behaviors for specific conditions and (2) construct complex refusal conditions through the logical composition of several condition vectors, enabling programmatic control over model behavior.

![Image 5: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/fine.png)

Figure 7: Inducing or suppressing refusal from specific categories. Each pie chart represents the model’s refusal rate for six prompt content types. (a): The leftmost chart shows Hermes 2 Pro’s original refusal rates. Subsequent charts demonstrate adding refusal on specific conditions (e.g., 𝐜 sex→+→subscript 𝐜 sex\mathbf{c}_{\text{sex}}\rightarrow+bold_c start_POSTSUBSCRIPT sex end_POSTSUBSCRIPT → + means inducing refusal for sexual content). (b): Refusal can also be removed by subtracting the behavior vector 𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT.

Experimental setup. We begin by randomly selecting 1,300 base prompts from the Alpaca training set. Each of these prompts is then paraphrased to incorporate aspects of sexual content 𝐜 sex subscript 𝐜 sex\mathbf{c}_{\text{sex}}bold_c start_POSTSUBSCRIPT sex end_POSTSUBSCRIPT, legal opinions 𝐜 legal subscript 𝐜 legal\mathbf{c}_{\text{legal}}bold_c start_POSTSUBSCRIPT legal end_POSTSUBSCRIPT, hate speech 𝐜 hate subscript 𝐜 hate\mathbf{c}_{\text{hate}}bold_c start_POSTSUBSCRIPT hate end_POSTSUBSCRIPT, crime planning 𝐜 crime subscript 𝐜 crime\mathbf{c}_{\text{crime}}bold_c start_POSTSUBSCRIPT crime end_POSTSUBSCRIPT, or health consultation 𝐜 health subscript 𝐜 health\mathbf{c}_{\text{health}}bold_c start_POSTSUBSCRIPT health end_POSTSUBSCRIPT. This process results in 1,300 prompts in six categories, including the original benign base Alpaca prompts. We then split this dataset into 700 prompts per category for training and 500 per category for testing. To create a conditioning vector 𝐜 𝐜\mathbf{c}bold_c for a specific category, we use the 700 ×\times× 5 = 3,500 training prompts from the other five categories as our negative examples (𝒟−superscript 𝒟\mathcal{D}^{-}caligraphic_D start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT). For the positive examples (𝒟+superscript 𝒟\mathcal{D}^{+}caligraphic_D start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT), we use the 700 training prompts from the target category and repeat them five times to balance the dataset.

Application: Inducing or suppressing refusal behavior from specific categories. We begin by examining our ability to add refusal behavior to specific categories of prompts, starting with a model that exhibits arbitrary refusal behaviors. Figure [7](https://arxiv.org/html/2409.05907v3#S5.F7 "Figure 7 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")a demonstrates that it is indeed possible to induce refusal behavior when a specific condition is met. This extends the concepts explored in Section [4](https://arxiv.org/html/2409.05907v3#S4 "4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering") to more fine-grained categories, showing successful selective refusal. Furthermore, as shown in Figure [7](https://arxiv.org/html/2409.05907v3#S5.F7 "Figure 7 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")b and consistent with findings from Arditi et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib4)), we can also remove refusal behavior from certain classes of prompts. This is achieved by simply reversing the signs of the behavior vector 𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT. Beyond refusal, most inference-time steering techniques can be conditioned using condition vectors as a modulation for various characteristics in language model outputs (Konen et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib30)).

![Image 6: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/multi.png)

Figure 8: Logical composition of conditions. (a) Effects of combining (OR ∨\lor∨) condition vectors on refusal rates. (b) Complex compositions, including simultaneous removal (−--) and induction (+++) of refusal behaviors. (c) Graphical illustration to ease understanding of outcomes under multiple rules: Rule 1 activated (left), no rules met (middle), Rule 2 met (right). Condition layers perform checking; behavior layers apply refusal vectors. 

Application: Logical composition of condition vectors. As introduced in Section [3.1](https://arxiv.org/html/2409.05907v3#S3.SS1 "3.1 Overview ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"), condition vectors can be logically combined to create complex refusal conditions. For instance, to induce refusal in two categories, such as hate speech and legal opinions, one could implement a rule like if 𝐜 hate subscript 𝐜 hate\mathbf{c}_{\text{hate}}bold_c start_POSTSUBSCRIPT hate end_POSTSUBSCRIPT or 𝐜 legal subscript 𝐜 legal\mathbf{c}_{\text{legal}}bold_c start_POSTSUBSCRIPT legal end_POSTSUBSCRIPT then +𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT, as illustrated in Figure [8](https://arxiv.org/html/2409.05907v3#S5.F8 "Figure 8 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")a. This multi-conditioning mechanism can also reinforce existing model refusal conditions, enhancing robustness against harmful prompts. The second pie chart in Figure [8](https://arxiv.org/html/2409.05907v3#S5.F8 "Figure 8 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")b demonstrates this with LLaMA 3.1 Inst, where we can augment the model’s existing refusal of crime planning and hate speech with additional conditions for legal and health queries while maintaining responsiveness to benign prompts. Each condition vector 𝐜 𝐜\mathbf{c}bold_c may have different optimal condition points, as different layers might best separate specific conditions. Consequently, condition checking might occur at various layers during inference, as shown in Figure [8](https://arxiv.org/html/2409.05907v3#S5.F8 "Figure 8 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")c. It’s also possible to completely change the original model’s refusal map by simultaneously removing existing refusal directions and inducing new ones (Figure [8](https://arxiv.org/html/2409.05907v3#S5.F8 "Figure 8 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")b) through multiple rules. However, we generally find that this approach can reduce the effectiveness of induced refusal directions, as certain suppressing conditions may conflict with newly induced refusal conditions.

![Image 7: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/multi-2.png)

Figure 9: Constraining responses to one domain. (a) Constraining response to only the target condition by adding refusal to all other categories of instructions using the flipped comparison direction (¬\neg¬) (see duality property). (b) Constraining response generalizes well to unseen categories of prompts as we are adding refusal to anything that does not satisfy the target condition. (c) Constraining response performance vs. average semantic distance from the target category’s train set to other categories’ test sets. Higher semantic distance correlates with better constraining effectiveness across seen and unseen categories. 

Application: Constraining model responses to specific domains. Connecting from our earlier point on the logical composition of condition vectors, we can conditionally steer models to respond only to specific types of prompts. This approach is particularly useful when the goal is to make a specialized model respond exclusively to specific categories, such as creating a health assistant (Cheong et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib12); Xie et al., [2024a](https://arxiv.org/html/2409.05907v3#bib.bib84)). Instead of creating conditions for all non-health categories to refuse, we can utilize the duality property discussed in Figure [5](https://arxiv.org/html/2409.05907v3#S4.F5 "Figure 5 ‣ 4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering"). We could (1) create a condition vector (e.g., 𝐜 health subscript 𝐜 health\mathbf{c}_{\text{health}}bold_c start_POSTSUBSCRIPT health end_POSTSUBSCRIPT) and (2) flip the comparison direction to add refusal on the exact complement set of inputs (e.g., ¬𝐜 health subscript 𝐜 health\neg\mathbf{c}_{\text{health}}¬ bold_c start_POSTSUBSCRIPT health end_POSTSUBSCRIPT). As shown in Figure [9](https://arxiv.org/html/2409.05907v3#S5.F9 "Figure 9 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering"), this constrains the model to only respond to a category and refuse all others.

We extended our investigation to examine whether our constraining method remains effective for unseen prompt categories. To this end, we introduced four additional harm categories from Liu et al. ([2023](https://arxiv.org/html/2409.05907v3#bib.bib39)) that were not part of our original condition vector training setup: gambling, financial advice, privacy violence, and malware generation. As illustrated in Figure [9](https://arxiv.org/html/2409.05907v3#S5.F9 "Figure 9 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")b, the effectiveness of domain constraining extends to unseen categories. This is because our method adds refusal to the complement set of the target category by flipping the comparison direction. Consequently, it refuses all inputs that do not match the target category’s characteristics, regardless of whether they were seen in training. However, we observed performance variations across different setups. For instance, constraining the model to hate speech (if ¬𝐜 hate subscript 𝐜 hate\neg\mathbf{c}_{\text{hate}}¬ bold_c start_POSTSUBSCRIPT hate end_POSTSUBSCRIPT then +𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT) was more effective in refusing other categories than constraining it to legal opinions (if ¬𝐜 legal subscript 𝐜 legal\neg\mathbf{c}_{\text{legal}}¬ bold_c start_POSTSUBSCRIPT legal end_POSTSUBSCRIPT then +𝐯 refusal subscript 𝐯 refusal\mathbf{v}_{\text{refusal}}bold_v start_POSTSUBSCRIPT refusal end_POSTSUBSCRIPT). This brings us to our next point.

Analysis: Constraining response to one category works better for more semantically distinct categories. Figure [9](https://arxiv.org/html/2409.05907v3#S5.F9 "Figure 9 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")c illustrates this relationship, showing a positive correlation between a category’s average semantic distance from others (x-axis) and the effectiveness of constraining to that category, measured by the increase in refusal rate for other categories (y-axis). Using a sentence transformer model, this semantic distance is calculated as the average cosine distance between the embeddings of the target category’s training prompts and the test prompts of all other categories. This explains why constraining the model to hate speech is more effective than constraining it to legal opinions when it comes to refusing other categories. Hate speech, being more semantically distinct from other categories, allows for clearer boundaries and, thus, more effective constraining.

As noted in previous literature on behavior steering, prompting alone fails to provide an effective alternative for several reasons. Unlike CAST, prompting lacks the ability to forcefully condition the model, offering only weak, coarse-grained control that may paradoxically increase unwanted content (Jang et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib28); Dekoninck et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib15)). Our experiments confirm this, with conditional steering consistently outperforming the prompting baseline (red dotted line) across most categories in Figure [9](https://arxiv.org/html/2409.05907v3#S5.F9 "Figure 9 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")c. This baseline represents the average performance when the model is simply prompted to comply with the target condition and refuse other conditions without any conditional steering techniques.

6 Conclusion
------------

This paper introduces Conditional Activation Steering (CAST), a novel framework for inducing context-dependent behaviors in large language models through principled manipulation of their internal representations. By extending existing activation steering techniques with the introduction of condition vectors, CAST enables fine-grained control over model behavior without the need for fine-tuning or extensive computational resources.

![Image 8: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/last.png)

Figure 10: Key conditioning operations. (a)→→\rightarrow→(b): adding a refusal condition. (a)→→\rightarrow→(c): Adding more refusal conditions. (a)→→\rightarrow→(d): Flipping the condition comparison direction to refuse all other categories except the target. 

Figure [10](https://arxiv.org/html/2409.05907v3#S6.F10 "Figure 10 ‣ 6 Conclusion ‣ Programming Refusal with Conditional Activation Steering") shows key operations: flipping condition comparisons to refuse all but target categories and adding single or multiple conditions to induce/remove behaviors. These tailor model behavior to specific needs. CAST offers quick harmful content refusal, complex rule composition, and domain-specific constraining. By leveraging the model’s representations, CAST matches or exceeds safety-aligned models’ performance with less computational overhead. This efficiency, combined with the ability to modify and compose behavioral rules rapidly, offers significantly enhanced flexibility in adapting model behavior to varying requirements.

References
----------

*   Adila et al. (2024) Dyah Adila, Shuai Zhang, Boran Han, and Yuyang Wang. Discovering bias in latent space: An unsupervised debiasing approach. _arXiv preprint arXiv:2406.03631_, 2024. 
*   Ansell et al. (2024) Alan Ansell, Ivan Vulić, Hannah Sterz, Anna Korhonen, and Edoardo M Ponti. Scaling sparse fine-tuning to large language models. _arXiv preprint arXiv:2401.16405_, 2024. 
*   Anwar et al. (2024) Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, et al. Foundational challenges in assuring alignment and safety of large language models. _arXiv preprint arXiv:2404.09932_, 2024. 
*   Arditi et al. (2024) Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction, 2024. 
*   Askell et al. (2021) Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. _arXiv preprint arXiv:2112.00861_, 2021. 
*   Bai et al. (2023) Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _arXiv preprint arXiv:2309.16609_, 2023. 
*   Bai et al. (2022) Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. _arXiv preprint arXiv:2212.08073_, 2022. 
*   Ball et al. (2024) Sarah Ball, Frauke Kreuter, and Nina Rimsky. Understanding jailbreak success: A study of latent space dynamics in large language models. _arXiv preprint arXiv:2406.09289_, 2024. 
*   Brahman et al. (2024) Faeze Brahman, Sachin Kumar, Vidhisha Balachandran, Pradeep Dasigi, Valentina Pyatkin, Abhilasha Ravichander, Sarah Wiegreffe, Nouha Dziri, Khyathi Chandu, Jack Hessel, et al. The art of saying no: Contextual noncompliance in language models. _arXiv preprint arXiv:2407.12043_, 2024. 
*   Cao et al. (2024) Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization. _arXiv preprint arXiv:2406.00045_, 2024. 
*   Chai et al. (2024) Ziwei Chai, Guoyin Wang, Jing Su, Tianjie Zhang, Xuanwen Huang, Xuwu Wang, Jingjing Xu, Jianbo Yuan, Hongxia Yang, Fei Wu, et al. An expert is worth one token: Synergizing multiple expert llms as generalist via expert token routing. _arXiv preprint arXiv:2403.16854_, 2024. 
*   Cheong et al. (2024) Inyoung Cheong, King Xia, KJ Kevin Feng, Quan Ze Chen, and Amy X Zhang. (a) i am not a lawyer, but…: Engaging legal experts towards responsible llm policies for legal advice. In _The 2024 ACM Conference on Fairness, Accountability, and Transparency_, pp. 2454–2469, 2024. 
*   Cui et al. (2024) Justin Cui, Wei-Lin Chiang, Ion Stoica, and Cho-Jui Hsieh. Or-bench: An over-refusal benchmark for large language models. _arXiv preprint arXiv:2405.20947_, 2024. 
*   Das et al. (2024) Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano, and Sayak Ray Chowdhury. Provably sample efficient rlhf via active preference optimization. _arXiv preprint arXiv:2402.10500_, 2024. 
*   Dekoninck et al. (2023) Jasper Dekoninck, Marc Fischer, Luca Beurer-Kellner, and Martin Vechev. Controlled text generation via language model arithmetic. _arXiv preprint arXiv:2311.14479_, 2023. 
*   Elhage et al. (2021) Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. _Transformer Circuits Thread_, 1(1):12, 2021. 
*   Feng et al. (2024) Shangbin Feng, Taylor Sorensen, Yuhan Liu, Jillian Fisher, Chan Young Park, Yejin Choi, and Yulia Tsvetkov. Modular pluralism: Pluralistic alignment via multi-llm collaboration. _arXiv preprint arXiv:2406.15951_, 2024. 
*   Ghandeharioun et al. (2024) Asma Ghandeharioun, Ann Yuan, Marius Guerard, Emily Reif, Michael A Lepori, and Lucas Dixon. Who’s asking? user personas and the mechanics of latent misalignment. _arXiv preprint arXiv:2406.12094_, 2024. 
*   Groeneveld et al. (2024) Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. Olmo: Accelerating the science of language models. _Preprint_, 2024. 
*   Gurnee & Tegmark (2023) Wes Gurnee and Max Tegmark. Language models represent space and time. _arXiv preprint arXiv:2310.02207_, 2023. 
*   Han et al. (2024) Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. Word embeddings are steers for language models, 2024. URL [https://arxiv.org/abs/2305.12798](https://arxiv.org/abs/2305.12798). 
*   (22) Benjamin David Hayum, Quentin Feuillade Montixi, and Yixuan Li. How does rlhf shift behavior distributions? distinguishability and steerability. 
*   He et al. (2024a) Jerry Zhi-Yang He, Sashrika Pandey, Mariah L Schrum, and Anca Dragan. Cos: Enhancing personalization and mitigating bias with context steering. _arXiv preprint arXiv:2405.01768_, 2024a. 
*   He et al. (2024b) Zihao He, Siyi Guo, Ashwin Rao, and Kristina Lerman. Whose emotions and moral sentiments do language models reflect? _arXiv preprint arXiv:2402.11114_, 2024b. 
*   Hendrycks et al. (2021) Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _NeurIPS_, 2021. 
*   Hsu et al. (2024) Chia-Yi Hsu, Yu-Lin Tsai, Chih-Hsun Lin, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang. Safe lora: the silver lining of reducing safety risks when fine-tuning large language models. _arXiv preprint arXiv:2405.16833_, 2024. 
*   Hu et al. (2024) Zhanhao Hu, Julien Piet, Geng Zhao, Jiantao Jiao, and David Wagner. Toxicity detection for free, 2024. URL [https://arxiv.org/abs/2405.18822](https://arxiv.org/abs/2405.18822). 
*   Jang et al. (2023) Joel Jang, Seonghyeon Ye, and Minjoon Seo. Can large language models truly understand prompts? a case study with negated prompts. In _Transfer learning for natural language processing workshop_, pp. 52–62. PMLR, 2023. 
*   Jorgensen et al. (2023) Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. Improving activation steering in language models with mean-centring. _arXiv preprint arXiv:2312.03813_, 2023. 
*   Konen et al. (2024) Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, and Tobias Hecking. Style vectors for steering generative large language model. _arXiv preprint arXiv:2402.01618_, 2024. 
*   Kong et al. (2024) Lingkai Kong, Haorui Wang, Wenhao Mu, Yuanqi Du, Yuchen Zhuang, Yifei Zhou, Yue Song, Rongzhi Zhang, Kai Wang, and Chao Zhang. Aligning large language models with representation editing: A control perspective. _arXiv preprint arXiv:2406.05954_, 2024. 
*   Kundu et al. (2023) Sandipan Kundu, Yuntao Bai, Saurav Kadavath, Amanda Askell, Andrew Callahan, Anna Chen, Anna Goldie, Avital Balwit, Azalia Mirhoseini, Brayden McLean, et al. Specific versus general principles for constitutional ai. _arXiv preprint arXiv:2310.13798_, 2023. 
*   Labonne (2024) Maxime Labonne. Uncensor any llm with abliteration. [https://huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration), 2024. 
*   Lad et al. (2024) Vedang Lad, Wes Gurnee, and Max Tegmark. The remarkable robustness of llms: Stages of inference? _arXiv preprint arXiv:2406.19384_, 2024. 
*   Lee et al. (2023a) Ariel N. Lee, Cole J. Hunter, and Nataniel Ruiz. Platypus: Quick, cheap, and powerful refinement of llms. 2023a. 
*   Lee et al. (2023b) Bruce W Lee, Hyunsoo Cho, and Kang Min Yoo. Instruction tuning with human curriculum. _arXiv preprint arXiv:2310.09518_, 2023b. 
*   Li et al. (2024) Jingling Li, Zeyu Tang, Xiaoyu Liu, Peter Spirtes, Kun Zhang, Liu Leqi, and Yang Liu. Steering llms towards unbiased responses: A causality-guided debiasing framework. _arXiv preprint arXiv:2403.08743_, 2024. 
*   Lightman et al. (2023) Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _preprint arXiv:2305.20050_, 2023. 
*   Liu et al. (2023) X Liu, Y Zhu, J Gu, Y Lan, C Yang, and Y Qiao. Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. _arXiv preprint arXiv:2311.17600_, 2023. 
*   Louie et al. (2024) Ryan Louie, Ananjan Nandi, William Fang, Cheng Chang, Emma Brunskill, and Diyi Yang. Roleplay-doh: Enabling domain-experts to create llm-simulated patients via eliciting and adhering to principles. _arXiv preprint arXiv:2407.00870_, 2024. 
*   Lu & Rimsky (2024) Dawn Lu and Nina Rimsky. Investigating bias representations in llama 2 chat via activation steering, 2024. 
*   Lu et al. (2022) Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In _The 36th Conference on Neural Information Processing Systems (NeurIPS)_, 2022. 
*   McKinzie et al. (2024) Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. _arXiv preprint arXiv:2403.09611_, 2024. 
*   Meta (2024) Meta. Introducing meta llama 3: The most capable openly available llm to date. [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/), 2024. 
*   Metcalf et al. (2024) Katherine Metcalf, Miguel Sarabia, Natalie Mackraz, and Barry-John Theobald. Sample-efficient preference-based reinforcement learning with dynamics aware rewards. _arXiv preprint arXiv:2402.17975_, 2024. 
*   Nagireddy et al. (2023) Manish Nagireddy, Lamogha Chiazor, Moninder Singh, and Ioana Baldini. Socialstigmaqa: A benchmark to uncover stigma amplification in generative language models, 2023. 
*   Park et al. (2023) Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. _arXiv preprint arXiv:2311.03658_, 2023. 
*   Peng et al. (2023) Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_, 2023. 
*   Pfeiffer et al. (2024) Pascal Pfeiffer, Philipp Singer, Yauhen Babakhin, Gabor Fodor, Nischay Dhankhar, and Sri Satish Ambati. H2o-danube3 technical report. _arXiv preprint arXiv:2407.09276_, 2024. 
*   Phan et al. (2024) Phuc Phan, Hieu Tran, and Long Phan. Distillation contrastive decoding: Improving llms reasoning with contrastive decoding and distillation. _arXiv preprint arXiv:2402.14874_, 2024. 
*   Pitis (2023) Silviu Pitis. Failure modes of learning reward models for llms and other sequence models. In _ICML 2023 Workshop The Many Facets of Preference-Based Learning_, 2023. 
*   Qiu et al. (2024) Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen, Edoardo M Ponti, and Shay B Cohen. Spectral editing of activations for large language model alignment. _arXiv preprint arXiv:2405.09719_, 2024. 
*   Radford et al. (2018) Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 
*   Rafailov et al. (2024) Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Rahn et al. (2024) Nate Rahn, Pierluca D’Oro, and Marc G Bellemare. Controlling large language model agents with entropic activation steering. _arXiv preprint arXiv:2406.00244_, 2024. 
*   Reuter & Schulze (2023) Max Reuter and William Schulze. I’m afraid i can’t do that: Predicting prompt refusal in black-box generative language models. _arXiv preprint arXiv:2306.03423_, 2023. 
*   Rimsky et al. (2024) Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition, 2024. 
*   Santurkar et al. (2023) Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In _International Conference on Machine Learning_, pp. 29971–30004. PMLR, 2023. 
*   Sawada et al. (2023) Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J. Nay, Kshitij Gupta, and Aran Komatsuzaki. Arb: Advanced reasoning benchmark for large language models, 2023. 
*   Scalena et al. (2024) Daniel Scalena, Gabriele Sarti, and Malvina Nissim. Multi-property steering of large language models with dynamic activation composition. _arXiv preprint arXiv:2406.17563_, 2024. 
*   Shai et al. (2024) Adam S Shai, Sarah E Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, and Paul M Riechers. Transformers represent belief state geometry in their residual stream. _arXiv preprint arXiv:2405.15943_, 2024. 
*   Sorensen et al. (2024) Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, et al. A roadmap to pluralistic alignment. _arXiv preprint arXiv:2402.05070_, 2024. 
*   Sotolar (2024) Ondrej Sotolar. Empo: Theory-driven dataset construction for empathetic response generation through preference optimization. _arXiv preprint arXiv:2406.19071_, 2024. 
*   Stickland et al. (2024) Asa Cooper Stickland, Alexander Lyzhov, Jacob Pfau, Salsabila Mahdi, and Samuel R Bowman. Steering without side effects: Improving post-deployment control of language models. _arXiv preprint arXiv:2406.15518_, 2024. 
*   Stiennon et al. (2020) Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. _Advances in Neural Information Processing Systems_, 33:3008–3021, 2020. 
*   Sudalairaj et al. (2024) Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, Kai Xu, David D Cox, and Akash Srivastava. Lab: Large-scale alignment for chatbots. _arXiv preprint arXiv:2403.01081_, 2024. 
*   Tamoyan et al. (2024) Hovhannes Tamoyan, Hendrik Schuff, and Iryna Gurevych. Llm roleplay: Simulating human-chatbot interaction. _arXiv preprint arXiv:2407.03974_, 2024. 
*   Tan et al. (2024) Daniel Chee Hian Tan, David Chanin, Aengus Lynch, Adrià Garriga-Alonso, Dimitrios Kanoulas, Brooks Paige, and Robert Kirk. Analyzing the generalization and reliability of steering vectors. In _ICML 2024 Workshop on Mechanistic Interpretability_, 2024. 
*   Taori et al. (2023) Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 
*   Tay et al. (2022) Yi Tay, Jason Wei, Hyung Won Chung, Vinh Q Tran, David R So, Siamak Shakeri, Xavier Garcia, Huaixiu Steven Zheng, Jinfeng Rao, Aakanksha Chowdhery, et al. Transcending scaling laws with 0.1% extra compute. _arXiv preprint arXiv:2210.11399_, 2022. 
*   Teknium et al. (2024) Teknium, interstellarninja, theemozilla, karan4d, and huemin_art. Hermes-2-pro-llama-3-8b, 2024. URL [https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B). 
*   Tlaie (2024) Alejandro Tlaie. Exploring and steering the moral compass of large language models. _arXiv preprint arXiv:2405.17345_, 2024. 
*   Touvron et al. (2023) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023. 
*   Tunstall et al. (2023) Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of lm alignment, 2023. 
*   Turner et al. (2023) Alex Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. _arXiv preprint arXiv:2308.10248_, 2023. 
*   Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_, 30, 2017. 
*   Wang et al. (2024a) Tianlong Wang, Xianfeng Jiao, Yifan He, Zhongzhi Chen, Yinghao Zhu, Xu Chu, Junyi Gao, Yasha Wang, and Liantao Ma. Adaptive activation steering: A tuning-free llm truthfulness improvement method for diverse hallucinations categories. _arXiv preprint arXiv:2406.00034_, 2024a. 
*   Wang et al. (2023a) Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models, 2023a. 
*   Wang et al. (2023b) Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. Do-not-answer: A dataset for evaluating safeguards in llms. _arXiv preprint arXiv:2308.13387_, 2023b. 
*   (80) Zihao Wang and Victor Veitch. Does editing provide evidence for localization? In _ICML 2024 Workshop on Mechanistic Interpretability_. 
*   Wang et al. (2024b) Zihao Wang, Lin Gui, Jeffrey Negrea, and Victor Veitch. Concept algebra for (score-based) text-controlled generative models. _Advances in Neural Information Processing Systems_, 36, 2024b. 
*   Wen et al. (2024) Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. The art of refusal: A survey of abstention in large language models. _arXiv preprint arXiv:2407.18418_, 2024. 
*   Wu et al. (2024) Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. Reft: Representation finetuning for language models. _arXiv preprint arXiv:2404.03592_, 2024. 
*   Xie et al. (2024a) Qianqian Xie, Qingyu Chen, Aokun Chen, Cheng Peng, Yan Hu, Fongci Lin, Xueqing Peng, Jimin Huang, Jeffrey Zhang, Vipina Keloth, et al. Me llama: Foundation large language models for medical applications. _arXiv preprint arXiv:2402.12749_, 2024a. 
*   Xie et al. (2024b) Tinghao Xie, Xiangyu Qi, Yi Zeng, Yangsibo Huang, Udari Madhushani Sehwag, Kaixuan Huang, Luxi He, Boyi Wei, Dacheng Li, Ying Sheng, et al. Sorry-bench: Systematically evaluating large language model safety refusal behaviors. _arXiv preprint arXiv:2406.14598_, 2024b. 
*   Xu et al. (2024) Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In _The Twelfth International Conference on Learning Representations_, 2024. 
*   Yin et al. (2024) Fangcong Yin, Xi Ye, and Greg Durrett. Lofit: Localized fine-tuning on llm representations. _arXiv preprint arXiv:2406.01563_, 2024. 
*   Yu et al. (2020) Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. Reclor: A reading comprehension dataset requiring logical reasoning. In _International Conference on Learning Representations (ICLR)_, April 2020. 
*   Zhang et al. (2024) Jie Zhang, Dongrui Liu, Chen Qian, Ziyue Gan, Yong Liu, Yu Qiao, and Jing Shao. The better angels of machine personality: How personality relates to llm safety. _arXiv preprint arXiv:2407.12344_, 2024. 
*   Zheng et al. (2024) Chujie Zheng, Fan Yin, Hao Zhou, Fandong Meng, Jie Zhou, Kai-Wei Chang, Minlie Huang, and Nanyun Peng. Prompt-driven llm safeguarding via directed representation optimization. _arXiv preprint arXiv:2401.18018_, 2024. 
*   Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric.P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023. 
*   Zou et al. (2023) Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J.Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to ai transparency, 2023. 

Appendix A Understanding Conditional Activation Steering
--------------------------------------------------------

### A.1 The Larger Picture

Model development cycle The development of language models can be broadly categorized into pre-training and post-training stages (McKinzie et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib43); Tay et al., [2022](https://arxiv.org/html/2409.05907v3#bib.bib70)). During pre-training, the focus is on enhancing fundamental capabilities such as knowledge acquisition, reasoning abilities, and coherent language use. The post-training stage, often referred to as alignment, aims to shape the model’s behavior to meet specific expectations and requirements (Kundu et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib32); Askell et al., [2021](https://arxiv.org/html/2409.05907v3#bib.bib5)).

Alignment and behavior steering Within the alignment phase, several key areas emerge, including evaluation, reinforcement learning, and instruction tuning (Nagireddy et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib46); Sudalairaj et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib66); Lee et al., [2023b](https://arxiv.org/html/2409.05907v3#bib.bib36)). While these topics often overlap, our focus is on behavior steering (Bai et al., [2022](https://arxiv.org/html/2409.05907v3#bib.bib7); Cao et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib10)). The term “steering” is deliberately chosen over “control,” implying the current approach of influencing language model behavior rather than exerting direct control.

As model creators, our ultimate goal is to achieve a level of control akin to programming these language models. To transition from behavior steering to true behavior control, two fundamental criteria must be met: specificity and predictability. This entails the ability to provide precise instructions or rules to the model, such as “refusing harmful instructions,” “declining irrelevant conversations,” or “avoiding generating adult content,” coupled with a high degree of confidence that the model will consistently adhere to these directives.

Towards programmatic behavior control Now, instead of merely encouraging models to behave in certain ways through prompting or reinforcement learning, we propose a more forceful and programmatic approach to designing model behaviors. Our method involves three key steps:

1.   1.Tracking model activations during inference 
2.   2.Checking if these activations match specified rule conditions 
3.   3.Forcefully intervening in the model to induce desired behavior when conditions are met (which was done in the form of activation steering in this paper) 

Unlike straightforward prompting-based approaches, conditional activation steering can be likened to implementing a brain-computer interface for language models, creating a programmable, rule-based system for enforcing model behavior.

Broader implications This research represents a step towards bringing language models under more precise control, moving closer to predicting and controlling LLM behaviors for various use cases. In this particular study, we focus on the refusal behavior - specifically, determining and enforcing exactly when a model should refuse instead of complying with a given instruction.

### A.2 Details of Conditional Activation Steering

Origins Conditional activation steering is an expansion of existing activation steering methods. Activation steering intervenes in the model’s hidden state during inference, typically by adding “steering vectors”. This simple operation has shown the potential to reliably induce behaviors like refusal on arbitrary prompts, aligning with the linear representation hypothesis (Park et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib47); Gurnee & Tegmark, [2023](https://arxiv.org/html/2409.05907v3#bib.bib20)). While effective, traditional activation steering lacks specificity, causing models to refuse all instructions indiscriminately. CAST addresses this limitation by introducing a conditional vector 𝐜 𝐜\mathbf{c}bold_c alongside the behavior vector 𝐯 𝐯\mathbf{v}bold_v. The application of 𝐯 𝐯\mathbf{v}bold_v is now conditioned on the similarity between the model’s activation and its projection onto 𝐜 𝐜\mathbf{c}bold_c.

Implementation in the generation process Language model generation can be viewed as a series of forward passes through the model’s layers for each generated token. The first full pass through the model typically involves prompt caching. In CAST, the condition is checked only during this first full pass, as we are conditioning on the prompt (see Figure [11](https://arxiv.org/html/2409.05907v3#A1.F11 "Figure 11 ‣ A.2 Details of Conditional Activation Steering ‣ Appendix A Understanding Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering")). This approach ensures that the additional condition-checking operation is not repeated for all generated tokens. However, if the condition is met, the behavior vector is applied in every subsequent forward pass, influencing each generated token. This application of the behavior vector in every pass at the specified layers follows the convention established in previous activation steering literature.

![Image 9: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/app_1.png)

Figure 11: The condition check occurs only in the first token’s pass (yellow layer), while behavior modification (blue layers) can be applied in all subsequent passes if the condition is met.

Extracting behavior and condition vectors The extraction of behavior and condition vectors follows a consistent process, as illustrated in Figure [12](https://arxiv.org/html/2409.05907v3#A1.F12 "Figure 12 ‣ A.2 Details of Conditional Activation Steering ‣ Appendix A Understanding Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"). This process involves passing contrastive prompts through the model, recording hidden states at each layer, and then applying Principal Component Analysis (PCA) to extract the direction that best separates the two contrastive prompt types. The mathematical representation of this process for each layer is as follows:

𝐯𝐞𝐜𝐭𝐨𝐫 l=PCA⁢([𝐡 1+−μ l 𝐡 1−−μ l⋮𝐡 n+−μ l 𝐡 n−−μ l])μ l=𝐇 l++𝐇 l−2 formulae-sequence subscript 𝐯𝐞𝐜𝐭𝐨𝐫 𝑙 PCA matrix superscript subscript 𝐡 1 subscript 𝜇 𝑙 superscript subscript 𝐡 1 subscript 𝜇 𝑙⋮superscript subscript 𝐡 𝑛 subscript 𝜇 𝑙 superscript subscript 𝐡 𝑛 subscript 𝜇 𝑙 subscript 𝜇 𝑙 superscript subscript 𝐇 𝑙 superscript subscript 𝐇 𝑙 2\mathbf{vector}_{l}=\text{PCA}\left(\begin{bmatrix}\mathbf{h}_{1}^{+}-\mathbf{% \mu}_{l}\\ \mathbf{h}_{1}^{-}-\mathbf{\mu}_{l}\\ \vdots\\ \mathbf{h}_{n}^{+}-\mathbf{\mu}_{l}\\ \mathbf{h}_{n}^{-}-\mathbf{\mu}_{l}\end{bmatrix}\right)\quad\quad\quad\quad% \quad\quad\mathbf{\mu}_{l}=\frac{\mathbf{H}_{l}^{+}+\mathbf{H}_{l}^{-}}{2}bold_vector start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = PCA ( [ start_ARG start_ROW start_CELL bold_h start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL bold_h start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL bold_h start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL bold_h start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_CELL end_ROW end_ARG ] ) italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = divide start_ARG bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT + bold_H start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT end_ARG start_ARG 2 end_ARG

The key distinction lies in the specific token position at which the activation is recorded, as depicted in Figure [3](https://arxiv.org/html/2409.05907v3#S3.F3 "Figure 3 ‣ 3.2 Preparing Dataset and Model ‣ 3 Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering"). This choice can be adjusted based on the experimental setup. For instance, when using longer contrastive prompts to train the vector, recording the activation of the last token may yield more informative results compared to using the mean activation across all tokens, which could potentially introduce length-related biases.

It is important to note that the current method for extracting and applying refusal behavior may have limitations. Recent studies, such as Arditi et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib4)) or Rimsky et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib57)), have proposed alternative approaches for extracting the behavior directions. While a comprehensive comparison of these methods is beyond the scope of this paper, it represents an important area for future research. The refinement of vector extraction techniques will likely benefit from ongoing collaborative efforts within the research community.

The current state of refusal behavior vector extraction has implications for the evaluation process. Imperfections in the refusal behavior vector may lead to inconsistent refusal induction, even when the condition is correctly activated. Additionally, conditioning and refusal induction performances are interrelated, presenting an opportunity for more detailed analysis in future studies. See Table [3](https://arxiv.org/html/2409.05907v3#A1.T3 "Table 3 ‣ A.2 Details of Conditional Activation Steering ‣ Appendix A Understanding Conditional Activation Steering ‣ Programming Refusal with Conditional Activation Steering").

![Image 10: Refer to caption](https://arxiv.org/html/2409.05907v3/extracted/6211614/figures/app_2.png)

Figure 12: All vector extractions follow a similar process.

Adjusting hyperparameters The effectiveness of conditional activation steering is highly sensitive to the choice of hyperparameters. This sensitivity stems from the fundamental nature of the method, which relies on precise mathematical operations within the model’s hidden states. The primary hyperparameters for conditioning can be conceptualized in a statement:

> Steer when the {best threshold} is {best direction} than the cosine similarity at {best layer}.

This formulation encapsulates three key hyperparameters: (1) Best layer: Determines at which depth of the network the condition checking operation occurs; (2) Best threshold: Defines the boundary for activation; (3) Best direction: Specifies whether the steering activates when the similarity is larger or smaller than the threshold.

Table 3: Breakdown of Figure [1](https://arxiv.org/html/2409.05907v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Programming Refusal with Conditional Activation Steering").

The layer selection is crucial because different layers capture varying levels of abstraction and linguistic features. The threshold value and comparison direction determine when the steering should be applied. Conceptually, this can be thought of as setting a “trigger point” in the high-dimensional space of the model’s hidden states (See Figure [7](https://arxiv.org/html/2409.05907v3#S5.F7 "Figure 7 ‣ 5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering")). The threshold defines a boundary, while the comparison direction (larger or smaller) determines on which side of this boundary the steering should activate.

These hyperparameters interact in complex ways with the model’s learned representations. For instance, a threshold that is too low might lead to frequent, unnecessary interventions, while one that is too high might fail to activate when needed. Similarly, the choice of layer can significantly impact the granularity and specificity of the condition being checked. While these conditioning hyperparameters are novel contributions of this approach, they build upon a foundation of existing research on intervention strength and optimal intervention points for behavioral steering in language models (Kong et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib31); [Wang & Veitch,](https://arxiv.org/html/2409.05907v3#bib.bib80); Zhang et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib89); Scalena et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib60); Tlaie, [2024](https://arxiv.org/html/2409.05907v3#bib.bib72)).

It is important to note that there isn’t a universally applicable range for the grid search (detailed in Section [C.2](https://arxiv.org/html/2409.05907v3#A3.SS2 "C.2 Best Condition Point (Grid Search) Algorithm ‣ Appendix C Intervention Points and Grid Search Algorithm ‣ Programming Refusal with Conditional Activation Steering")) of these hyperparameters, particularly for the threshold values. The cosine similarity values can vary drastically depending on the specific model architecture (more dependent) and the condition being explored (less dependent). For instance, in our experiments, we found that for Hermes 2 Pro, effective threshold values for various conditions fell within the range of 0.0 to 0.1. However, for the Zephyr model, the harmfulness condition operated optimally with threshold values between 0.4 and 0.6. To facilitate this process, our code implementation allows users to easily review the activation history of similarities and determine appropriate search ranges for different models and conditions.

Appendix B Constrasting Pair Generation Details
-----------------------------------------------

To generate the contrasting pair examples used in Section [4](https://arxiv.org/html/2409.05907v3#S4 "4 Conditioned Refusal: Selectively Steering on Harmful Prompts ‣ Programming Refusal with Conditional Activation Steering") and Section [5](https://arxiv.org/html/2409.05907v3#S5 "5 Programmed Refusal: Logical Composition of Condition Vector ‣ Programming Refusal with Conditional Activation Steering"), we employed the following machine generation processes:

### B.1 Section 4: Harmful vs. Harmless Prompts

For Section 4, we used the Sorry-Bench dataset as a source of harmful prompts:

1.   1.

For each harmful prompt in the Sorry-Bench dataset:

    1.   (a)Select two random prompts from other harm categories in the Sorry-Bench dataset. 
    2.   (b)

Create a prompt for the language model (Mixtral 8x7B) that includes:

        *   •The target harmful prompt 
        *   •Two example prompts from other harm categories 
        *   •Instructions to generate new questions that violate the target harm category but not the other categories 

    3.   (c)Generate 10 new variations of the harmful prompt using the language model. 
    4.   (d)Add the generated variations to the original prompt data structure. 

2.   2.For harmless prompts, we randomly sampled from the Alpaca dataset without modification. 

Pseudocode for the harmful prompt generation:

1 for item in sorry_bench_data:

2 others=random.sample([other for other in sorry_bench_data

3 if other[’harm_category’]!=item[’harm_category’]

4 and other[’harm_domain’]==item[’harm_domain’]],2)

5 prompt=create_prompt(item,others)

6 new_questions=generate_questions(prompt)

7

8 for i,question in enumerate(new_questions[1:],start=1):

9 if question!=item[’question’]:

10 item[f’question_plus_{i}’]=question

11

12 append_json(output_file,item)

The prompt used for generation was (create_prompt):

### B.2 Section 5: Fine-grained Harm Categories

For Section 5, we used the Alpaca dataset as a base and generated variations for specific harm categories. The process was:

1.   1.

For each prompt in the Alpaca dataset (both train and test splits):

    1.   (a)

For each of the five harm categories (sexual content, legal opinion, hate speech, crime planning, health consultation):

        *   •

Create a prompt for the language model (gpt-4o-2024-05-13) that includes:

            *   –The original Alpaca prompt 
            *   –Instructions to rewrite the prompt to include aspects of the current harm category 
            *   –Rules to ensure the generated prompt maintains a similar structure and explicitly includes the harm category without mentioning it directly 

        *   •Generate a new variation of the prompt using the language model 

    2.   (b)Add the generated variations to the original prompt data structure 

Pseudocode for the fine-grained category generation:

1 for split in[’train’,’test’]:

2 for item in alpaca_data[split]:

3 new_item=item.copy()

4 for category in categories:

5 other_categories=",".join([s for s in categories if s!=category])

6 prompt=create_prompt(item,category,other_categories)

7 new_question=generate_questions(prompt,category)

8 if new_question!=item[’question’]:

9 new_item[f’question_plus_{category.replace("","_")}’]=\

10 new_question

11

12 write_json_incrementally(output_file,new_item,split,is_first,is_last)

The prompt used for generation was (create_prompt):

Appendix C Intervention Points and Grid Search Algorithm
--------------------------------------------------------

### C.1 Intervention Points Used to Produce Results in This Paper

Table 4: Intervention points for condition and behavior. For example, 10−15 i⁢n⁢t⁢e⁢r⁢v⁢a⁢l⁢2 10 subscript 15 𝑖 𝑛 𝑡 𝑒 𝑟 𝑣 𝑎 𝑙 2 10-15_{interval2}10 - 15 start_POSTSUBSCRIPT italic_i italic_n italic_t italic_e italic_r italic_v italic_a italic_l 2 end_POSTSUBSCRIPT is [10, 12, 14]. 

All our experiments are done in our activation steering library, which we open-sourced along with this paper. The algorithm’s use of these values to steer the model might differ slightly for behavior steering but not for condition steering, as we are implementing conditional steering for the first time. In general, one could steer, conditional steer, or multi-conditionally steer, as shown in the following code snippets. These are high-level overviews demonstrating how the numbers from Table [4](https://arxiv.org/html/2409.05907v3#A3.T4 "Table 4 ‣ C.1 Intervention Points Used to Produce Results in This Paper ‣ Appendix C Intervention Points and Grid Search Algorithm ‣ Programming Refusal with Conditional Activation Steering") can be applied to replicate our results. For exact replication, use the replication version of our code.

Steer:

1 malleable_model.steer(

2 behavior_vector={some steering vector file ending with.svec},

3 behavior_layer_ids=[10,11,12,13,14,15],

4 behavior_vector_strength=0.1,

5)

Conditional Steer:

1 malleable_model.steer(

2 behavior_vector={some steering vector file ending with.svec},

3 behavior_layer_ids=[10,11,12,13,14,15],

4 behavior_vector_strength=0.1,

5 condition_vector={some steering vector file ending with.svec},

6 condition_layer_ids=[9],

7 condition_vector_threshold=0.031,

8 condition_comparator_threshold_is="smaller"

9)

Multi-Conditionally Steer:

1 malleable_model.multisteer(

2 behavior_vectors=[{steering vector file 1},{steering vector file 2},…],

3 behavior_layer_ids=[[10,11,12,13,14,15],[16,17,18],…],

4 behavior_vector_strengths=[0.1,0.2,…],

5 condition_vectors=[{steering vector file 1},{steering vector file 2},…],

6 condition_layer_ids=[[9],[7],…],

7 condition_vector_thresholds=[0.031,0.021,…],

8 condition_comparator_threshold_is=["smaller","larger",…],

9 rules=["if C1 then B1","if C2 then B2"]

10)

### C.2 Best Condition Point (Grid Search) Algorithm

The algorithm searches for the optimal conditioning configuration by evaluating different combinations of layers, thresholds, and comparison directions.

1

2 def find_best_condition_point(positive_strings,negative_strings,condition_vector,

3 layer_range,max_layers_to_combine,

4 threshold_range,threshold_step):

5 all_strings=positive_strings+negative_strings

6 y_true=[1]∗len(positive_strings)+[0]∗len(negative_strings)

7 layers=range(layer_range[0],layer_range[1])

8 best_f1=0

9 best_config=None

10

11

12 steer(condition_vector,layers)

13

14

15 similarities=[]

16 for string in all_strings:

17 respond(string)

18 similarities.append(get_condition_similarities())

19 reset_condition_state()

20

21

22 all_combinations=generate_combinations(layers,max_layers_to_combine,

23 threshold_range,threshold_step)

24

25 for layer_combo,threshold,direction in all_combinations:

26 y_pred=[]

27 for sim_dict in similarities:

28 condition_met=check_condition(sim_dict,layer_combo,

29 threshold,direction)

30 y_pred.append(1 if condition_met else 0)

31 f1=calculate_f1_score(y_true,y_pred)

32

33 if f1>best_f1:

34 best_f1=f1

35 best_config=(layer_combo,threshold,direction)

36

37 return best_config,best_f1

38

39 def check_condition(sim_dict,layer_combo,threshold,direction):

40 for layer in layer_combo:

41 if(sim_dict[layer]>threshold)==(direction==’smaller’):

42 return True

43 return False

This algorithm iterates through various combinations of layers, thresholds, and comparison directions to find the configuration that yields the highest F1 score in distinguishing between positive and negative examples. It uses the model’s conditional steering mechanism to compute similarities and then evaluates the effectiveness of different configurations in classifying the input strings. Based on our experience with CAST, we limit our grid search to the first half of the layers for all models.

Appendix D Model Descriptions / Dataset Locations
-------------------------------------------------

Here, we share all locations of datasets and models used in this paper. We only use publicly available models and datasets that are open-sourced with fairly permissible licenses. All can be found on Huggingface.

*   •sorrybench: sorry-bench/sorry-bench-202406 <b34822276edde97592eda99c0b56d306f8830469> 
*   •alpaca: EdBerg/yahmaalpaca-cleaned <6b6ff0e894d31390fa3581bf56f3bafaed9d5e2d> 
*   •refusal classifier: 

protectai/distilroberta-base-rejection-v1 <65584967c3f22ff7723e5370c65e0e76791e6055> 
*   •model: Qwen/Qwen1.5-1.8B-Chat <e482ee3f73c375a627a16fdf66fd0c8279743ca6> 
*   •model: Qwen/Qwen1.5-32B-Chat <0997b012af6ddd5465d40465a8415535b2f06cfc> 
*   •model: meta-llama/Llama-2-13b-chat-hf <a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8> 
*   •model: meta-llama/Meta-Llama-3.1-8B-Instruct <8c22764a7e3675c50d4c7c9a4edb474456022b16> 
*   •model: mlabonne/NeuralDaredevil-8B-abliterated <348bd440bb061a12552868aeee47207f1a6c0f76> 
*   •model: NousResearch/Hermes-2-Pro-Llama-3-8B <8ab73a6800796d84448bc936db9bac5ad9f984ae> 
*   •model: allenai/OLMo-7B-SFT-hf <c16aa53f08680e03808a174adcc071ee4f6cf192> 
*   •model: HuggingFaceH4/zephyr-7b-beta <b70e0c9a2d9e14bd1e812d3c398e5f313e93b473> 
*   •model: h2oai/h2o-danube3-4b-chat <1e5c6fa6620f8bf078958069ab4581cd88e0202c> 

### D.1 Community Model Descriptions

NeuralDaredevil-8B: This model is derived from Daredevil-8B, which itself is a merge of multiple Llama 3 8B models using the DARE TIES technique. The process to create NeuralDaredevil-8B involved:

1.   1.Starting with Daredevil-8B, a mega-merged model based on Llama 3 8B. 
2.   2.Applying abliteration to remove the refusal behavior to “uncensor” the model. Here, abliteration is an orthogonal refusal removal process following the theory presented in Arditi et al. ([2024](https://arxiv.org/html/2409.05907v3#bib.bib4)). 
3.   3.Performing DPO (Direct Preference Optimization) fine-tuning using the mlabonne/orpo-dpo-mix-40k dataset to recover performance lost during abliteration. 

This process resulted in an uncensored LLM that maintains most of the original model’s capabilities while removing its built-in censorship mechanisms.

Hermes 2 Pro: Developed by Nous Research, the Hermes 2 Pro we use is based on Llama 3 8B and created through the following process:

1.   1.Starting with the Llama 3 8B base model. 
2.   2.Fine-tuning on an updated and cleaned version of the OpenHermes 2.5 Dataset. This dataset is a mix of a few different datasets, including LMSYS-Chat-1M (Zheng et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib91)), WizardLM (Xu et al., [2024](https://arxiv.org/html/2409.05907v3#bib.bib86)), Platypus (Lee et al., [2023a](https://arxiv.org/html/2409.05907v3#bib.bib35); Lightman et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib38); Lu et al., [2022](https://arxiv.org/html/2409.05907v3#bib.bib42); Wang et al., [2023a](https://arxiv.org/html/2409.05907v3#bib.bib78); Yu et al., [2020](https://arxiv.org/html/2409.05907v3#bib.bib88); Hendrycks et al., [2021](https://arxiv.org/html/2409.05907v3#bib.bib25); Sawada et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib59)), and Unnatural Instructions (Peng et al., [2023](https://arxiv.org/html/2409.05907v3#bib.bib48)). 
3.   3.Further fine-tuning on a newly introduced Function Calling and JSON Mode dataset developed in-house by Nous Research. 
4.   4.Incorporating a special system prompt and multi-turn function calling structure with a new chatml role to enhance function calling capabilities. 

The resulting model excels at general tasks, conversations, Function Calling, and JSON Structured Outputs, while maintaining the core capabilities of the Llama 3 8B base model.

### D.2 Refusal Classification Model

For refusal detection, we used the protectai/distilroberta-base-rejection-v1 model, a fine-tuned version of distilroberta-base. This model was trained on multiple combined datasets of rejections from different LLMs and normal responses from RLHF datasets, with approximately 10% rejections and 90% normal outputs. The authors mention the use of Do-Not-Answer (Wang et al., [2023b](https://arxiv.org/html/2409.05907v3#bib.bib79)), and I’m Afraid I Can’t Do That (Reuter & Schulze, [2023](https://arxiv.org/html/2409.05907v3#bib.bib56)) datasets for the training.

Our refusal detection process, implemented in the check_refusal function, involves:

1.   1.Using the protectai/distilroberta-base-rejection-v1 model to classify responses as "NORMAL" or "REJECTION". 
2.   2.Performing an additional keyword-based check using a predefined list of common refusal phrases, ["not capable", "cannot provide", "cannot generate", "inappropriate", "language model", "i’m sorry", "an ai", "not able", "am sorry", "don’t know", "sorry", "i cannot", "i can’t", "can’t help", "is it appropriate", "violation", "consequences", "violence", "harm", "harmful", "harassment", "illegal", "experiencing stress", "you’re experiencing", "can’t provide", "i don’t have", "i’m a computer"]. 
3.   3.

Considering a response as a refusal if either:

    *   •The distilroberta-base-rejection-v1 model classifies it as "REJECTION", or any of the predefined keywords or phrases are present in the response (case-insensitive comparison). 

Appendix E Hardware Specifications
----------------------------------

The experiments were performed on a system with the following specifications:

*   •

CPU: 2 x AMD EPYC 7763 64-Core Processor

    *   –Total Cores: 128 
    *   –Threads per Core: 1 
    *   –Base Clock Speed: 2.45 GHz 
    *   –Max Boost Clock: 3.53 GHz 

*   •Memory: Not specified (to be added) 
*   •

GPU: NVIDIA A100-SXM4-80GB

    *   –Memory: 80 GB HBM2 
    *   –CUDA Cores: 6912 

### E.1 Software Environment

The software environment for all experiments consisted of:

*   •Operating System: Linux 
*   •CUDA Version: 12.2 
*   •NVIDIA Driver Version: 535.54.03 
*   •Python Version: 3.10.5 
*   •

Key Libraries:

    *   –PyTorch: 2.3.0 
    *   –Transformers: 4.43.3 

This configuration remained consistent throughout the research, ensuring that all reported results are comparable and reproducible under the same conditions.
