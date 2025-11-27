yTitle: An Improved Empirical Fisher Approximation for Natural Gradient Descent

URL Source: https://arxiv.org/html/2406.06420v2

Markdown Content:
Back to arXiv

This is experimental HTML to improve accessibility. We invite you to report rendering errors. 
Use Alt+Y to toggle on accessible reporting links and Alt+Shift+Y to toggle off.
Learn more about this project and help improve conversions.

Why HTML?
Report Issue
Back to Abstract
Download PDF
 Abstract
1Introduction
2Related Work
3Preliminaries
4Inversely-Scaled Projection Issue of Empirical Fisher
5Improved Empirical Fisher
6Empirical Evaluation Framework for Approximate NGD Methods
7Experiments
8Conclusions and Future Work

HTML conversions sometimes display errors due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: commath
failed: biblatex

Authors: achieve the best HTML results from your LaTeX submissions by following these best practices.

License: CC BY 4.0
arXiv:2406.06420v2 [cs.LG] null
\addbibresource

references.bib

An Improved Empirical Fisher Approximation for Natural Gradient Descent
Xiaodong Wu1∗  Wenyi Yu2∗  Chao Zhang2  Philip Woodland1
1Dept. of Engineering, University of Cambridge  2Dept. of Electronic Engineering, Tsinghua University
{xw338,pw117}@cam.ac.uk  {ywy22@mails,cz277@mail}.tsinghua.edu.cn
Abstract

Approximate Natural Gradient Descent (NGD) methods are an important family of optimisers for deep learning models, which use approximate Fisher information matrices to pre-condition gradients during training. The empirical Fisher (EF) method approximates the Fisher information matrix empirically by reusing the per-sample gradients collected during back-propagation. Despite its ease of implementation, the EF approximation has its theoretical and practical limitations. This paper investigates the inversely-scaled projection issue of EF, which is shown to be a major cause of its poor empirical approximation quality. An improved empirical Fisher (iEF) method is proposed to address this issue, which is motivated as a generalised NGD method from a loss reduction perspective, meanwhile retaining the practical convenience of EF. The exact iEF and EF methods are experimentally evaluated using practical deep learning setups, including widely-used setups for parameter-efficient fine-tuning of pre-trained models (T5-base with LoRA and Prompt-Tuning on GLUE tasks, and ViT with LoRA for CIFAR100). Optimisation experiments show that applying exact iEF directly as an optimiser provides strong convergence and generalisation. It achieves the best test performance and the lowest training loss for the majority of the tasks, even when compared to well-tuned AdamW/Adafactor baselines. Additionally, under a novel empirical evaluation framework, the proposed iEF method shows consistently better approximation quality to exact Natural Gradient updates than both the EF and the more expensive sampled Fisher methods, meanwhile demonstrating the superior property of being robust to the choice of damping across tasks and training stages. Improving existing approximate NGD optimisers with iEF is expected to lead to better convergence and robustness. Furthermore, the iEF method also serves as a better approximation method to the Fisher information matrix itself, which enables the improvement of a variety of Fisher-based methods, not limited to the scope of optimisation.

*
1Introduction

Parameter optimisation is a crucial research area in the field of deep learning, where stochastic optimisers are commonly used which update the model parameters iteratively to minimise a target loss function. Approximate Natural Gradient Descent (NGD) [NGD] methods are an important family of approximate second-order optimisers, which pre-condition the gradient with the (approximate) Fisher information matrix (also called the Fisher matrix) to accelerate training or improve generalisation.

Although there are many successful optimisers based on approximate NGD, many of them in fact use the empirical Fisher (EF) as a pre-conditioner. These methods are referred to as approximate empirical NGD methods [SENG]. The EF method constructs an approximation to the exact Fisher matrix directly from the gradients of training samples, which are usually readily computed during the training process [EF-limitation]. In contrast, the exact Fisher matrix needs to be either sampled from the model output distribution [NG-new-insights], or requires repeated evaluation of the matrix-vector product with the Fisher matrix [HF], which are both expensive operations. As a result, due to the ease of implementation brought by EF, empirical NGD is used in many approximate NGD optimisers as the default choice [KFAC-with-EF1, Eva, TONGA, TEKFAC-summary, EK-FAC, SENG, NG+].

Despite the prevalence of empirical NGD optimisers, it is known that EF is in general a questionable approximation to the exact Fisher matrix [NG-new-insights, EF-limitation, INTERPLAY]. The poor approximation quality of EF-based updates has been experimentally verified for small-scale experimental setups by [EF-limitation, INTERPLAY]. However, traditional evaluation methods are used in [EF-limitation, INTERPLAY] where the exact NG update and Fisher matrix need to be explicitly computed, making their findings impossible to verify for large deep learning setups for practical tasks. Hence, a more generally applicable evaluation framework is needed. There is also a need for an improved approximation to the exact Fisher matrix (as a pre-conditioner) than EF, while being as efficient to implement. This paper aims to fill these gaps.

Our Contributions: In this paper, an improved EF (iEF) approximation for NGD is proposed, which provides a better approximation to the exact Natural Gradient (NG) updates, while maintaining the practical convenience of EF. This method allows for a straightforward upgrade for all existing approximate empirical NGD optimisers. To achieve this, a theoretical investigation into the behaviour of EF update is first carried out, where the impact of the EF update on each involved sample is analysed (the “involved samples” refers to training samples whose gradient is used to construct the EF update). This leads to finding the inversely-scaled projection issue of EF (see Sec. 4). Accordingly, the iEF method is proposed to overcome this issue by introducing a diagonal scaling matrix to the standard formulation of the EF pre-conditioner. It is motivated as an approximate Gauss-Newton algorithm from a loss reduction perspective, with global convergence guarantees under mild assumptions (see Sec. 5). A novel empirical evaluation framework for approximate NGD methods is then proposed to enable accurate comparison of approximate Fisher pre-conditioners (e.g. EF and iEF) in large-scale optimisation setups (see Sec. 6). We conducted experiments that compare the exact EF and iEF methods in a range of practical deep learning setups including computer vision and fine-tuning large language models. Under our evaluation framework, iEF demonstrates better approximation quality to exact NG updates than both EF and the more expensive Monte-Carlo sampled Fisher method (SF, see Appendix E), meanwhile being significantly more robust to the choice of damping across tasks and training stages. Direct application of iEF as optimiser also shows consistently strong generalisation and convergence, even when compared to well-tuned AdamW/Adafactor baselines (see Sec. 7).

2Related Work

Approximate (Empirical) NGD: There are many existing approximate (empirical) NGD methods, most of which use EF despite its theoretical limitations. Some prior work, e.g. [TONGA, SMW], uses the Woodbury identities [cookbook] to exactly compute EF updates. Recent block-diagonal methods (based on K-FAC [K-FAC]) have gained popularity due to their efficiency, which includes work that modify the K-FAC approximation [Eva, TNT, TEKFAC-summary, GDN, SENG, Mini-Block-EF] or distributively apply K-FAC as optimisers [KFAC-with-EF2, KFAC-with-EF3]. Sometimes Adagrad-based methods [Adagrad, Shampoo, Adam] are also regarded as empirical NGD methods. However, the connection is questionable [EF-limitation] as these methods use the square-root of the EF matrix, instead of the EF matrix itself, as a pre-conditioner.

Limitations of EF Approximation: The limitations of EF as an approximate Fisher matrix have been discussed and demonstrated in several papers [NG-new-insights, EF-limitation, INTERPLAY], among which [EF-limitation] provided a thorough review and analysis. However, as far as we are aware, there has been no prior work that analysed the exact EF method in larger deep-learning setups, and most of the observations are limited to small-scale problems for theoretical machine-learning studies. It is known, however, that practical EF-based optimisers usually require a sophisticated damping scheme to work well [KFAC-with-EF1, KFAC-with-EF4]. It has even been suggested that an infinitely large damping should be used with the gradient covariance term [GDN, GDN-new]. These observations can be tied to the theoretical limitations of EF.

Empirical Evaluation of Approximate NGD Quality: An accurate evaluation of the approximation quality to exact NG updates is of great importance for approximate NGD methods. Usually, the performance of the method of interest is evaluated on machine learning benchmarks [Eva, K-FAC, EK-FAC], which provide crucial information from the optimisation perspective. However, limited information about the approximation quality to exact NGD can be drawn from these experiments. Therefore, additional small-scale experiments are usually performed to compare against the exact Fisher matrices [INTERPLAY, K-FAC, EK-FAC], or the exact NG updates [EF-limitation, TNT, Mini-Block-EF], which are extremely difficult to do for commonplace large-scale models. This limits our understanding of these methods in the context of large-scale tasks.

3Preliminaries
Supervised Learning for Classification Model with Softmax Activation:

This paper considers supervised learning of categorical classification, where a probabilistic model is trained to predict outputs 
𝑦
∈
{
𝑐
|
𝑐
=
1
,
2
,
…
⁢
𝐶
}
 of 
𝐶
 categories from inputs 
𝒙
∈
𝕏
. The target model 
𝒛
=
𝑓
𝜽
⁢
(
𝒙
)
 has 
𝜽
∈
ℝ
𝑃
 as the model parameters, which outputs the logits 
𝒛
∈
ℝ
𝐶
. Assume a softmax activation is used on the logits, the model can be expressed as a conditional probability of 
𝑝
𝜽
⁢
(
𝑦
|
𝒙
)
. Given 
𝑁
 i.i.d. training samples 
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑁
 (assuming 
𝑁
≪
𝑃
), the following accumulated loss is minimised

	
ℒ
⁢
(
𝜽
)
=
∑
𝑛
−
log
⁡
𝑝
𝜽
⁢
(
𝑦
=
𝑦
𝑛
|
𝒙
𝑛
)
=
∑
𝑛
𝑙
𝑛
,
		
(1)

where 
𝑙
𝑛
=
−
log
⁡
𝑝
𝜽
⁢
(
𝑦
=
𝑦
𝑛
|
𝒙
𝑛
)
 is the categorical cross-entropy loss for the 
𝑛
-th training sample. For brevity, we denote 
𝑝
𝜽
⁢
(
𝑦
=
𝑐
|
𝒙
𝑛
)
=
𝑝
𝑛
⁢
(
𝑐
)
.

A vectorised representation of loss 
𝒍
∈
ℝ
𝑁
 is used where 
𝒍
=
[
𝑙
1
,
𝑙
2
,
⋯
,
𝑙
𝑁
]
⊤
. The accumulated loss then becomes 
ℒ
⁢
(
𝜽
)
=
∑
𝑛
𝑙
𝑛
=
𝒍
⊤
⁢
𝟏
 where 
𝟏
 is an all 1 column vector of matching dimension, and the accumulated gradient can be re-written as 
∇
𝜽
ℒ
⁢
(
𝜽
)
=
∇
𝜽
𝒍
⊤
⁢
𝟏
 where 
∇
𝜽
𝒍
∈
ℝ
𝑁
×
𝑃
 is the Jacobian of per-sample losses w.r.t. model parameters.

NGD and Empirical NGD

In a first-order optimisation method, say SGD [SGD], the update direction on the model parameter is the estimate of the accumulated gradient 
∇
𝜽
ℒ
⁢
(
𝜽
)
. In the NGD method [NG-new-insights], the gradient is pre-conditioned by the Fisher matrix 
𝐅
 (i.e. 
𝐅
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
) to accelerate convergence. The exact Fisher matrix can be computed from the model output distribution using available training samples as follows

	
𝐅
:=
∑
𝑛
∑
𝑐
𝑝
𝑛
⁢
(
𝑐
)
⁢
[
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑐
)
⁢
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑐
)
⊤
]
.
		
(2)

The Fisher matrix can be estimated with Monte-Carlo (MC) sampling [K-FAC]. This approximation method is usually used with one MC sample per training sample, which is termed SF in this paper (see Appendix E). Alternatively, when the model is well trained and 
𝑝
𝑛
⁢
(
𝑦
𝑛
)
→
1
 for all 
𝑁
 samples, it is possible to approximate the exact Fisher with EF using the empirical gradient as follows

	
𝐅
~
:=
∑
𝑛
[
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑦
𝑛
)
⁢
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑦
𝑛
)
⊤
]
=
∇
𝜽
𝒍
⊤
⁢
∇
𝜽
𝒍
.
		
(3)

Pre-conditioning the gradient with the EF matrix (i.e. 
𝐅
~
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
) yields the empirical NGD method. Although empirical NGD is prevalent due to the convenience of computing the EF matrix in practice, the approximation quality of EF to the exact Fisher matrix is worth questioning [EF-limitation, INTERPLAY].

4Inversely-Scaled Projection Issue of Empirical Fisher

Despite the practical prevalence of the EF method, it is generally believed to be a poor approximation of the exact NGD method [EF-limitation]. To better understand the cause of the limited approximation quality of the EF method, an analysis of the impact of the EF update on each of the involved samples is presented below. This leads to finding the “inversely-scaled projection issue” of the EF method, which provides a focus for the improvement of the EF method.

4.1Formal Definition

Recall the definition of EF in Eqn. (3). The empirical NG update (or just the EF update) can be defined as follows

	
Δ
⁢
𝜽
EF
=
−
𝜂
⁢
𝐅
~
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
=
−
𝜂
⁢
(
∇
𝜽
𝒍
⊤
⁢
∇
𝜽
𝒍
+
𝜆
⁢
𝐈
)
−
1
⁢
(
∇
𝜽
𝒍
⊤
⁢
𝟏
)
		
(4)

where 
𝜆
∈
ℝ
+
 is a small damping factor to facilitate inversion (gradient covariance matrix 
∇
𝜽
𝒍
⊤
⁢
∇
𝜽
𝒍
∈
ℝ
𝑃
×
𝑃
 cannot be directly inverted for over-parameterised models). Using the Woodbury identity [cookbook], the EF update can be re-expressed as follows:

	
Δ
⁢
𝜽
EF
=
−
𝜂
⁢
∇
𝜽
𝒍
⊤
⁢
(
∇
𝜽
𝒍
⁢
∇
𝜽
𝒍
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝟏
.
		
(5)

The loss change induced on each sample (denoted as 
Δ
⁢
𝒍
EF
) when applying the EF update to the model can be estimated using the Jacobian 
∇
𝜽
𝒍
 as follows:

	
Δ
⁢
𝒍
EF
=
−
∇
𝜽
𝒍
⁢
Δ
⁢
𝜽
EF
=
−
𝜂
⁢
∇
𝜽
𝒍
⁢
∇
𝜽
𝒍
⊤
⁢
(
∇
𝜽
𝒍
⁢
∇
𝜽
𝒍
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝟏
≈
−
𝜂
⁢
𝟏
,
	

This result means that EF updates have the property of inducing an equal loss reduction on every involved sample. For the 
𝑛
-th sample, the projection of the EF update onto gradient direction 
∇
𝜽
𝑙
𝑛
 (denoted as 
(
𝜅
𝑛
)
EF
) can be computed as follows

	
(
𝜅
𝑛
)
EF
=
Δ
⁢
𝜽
EF
⊤
⁢
∇
𝜽
𝑙
𝑛
‖
∇
𝜽
𝑙
𝑛
‖
2
=
−
𝜂
‖
∇
𝜽
𝑙
𝑛
‖
2
,
		
(6)

where 
‖
∇
𝜽
𝑙
𝑛
‖
2
 denotes the 
𝑙
2
 norm of the 
𝑛
-th per-sample gradient. This means that the projection of EF update onto every sample gradient is inversely proportional to the gradient norm of each sample. Note that a smaller 
‖
∇
𝜽
𝑙
𝑛
‖
2
 generally indicates the sample is better trained (or more converged, or closer to its minimum). The EF update is therefore easily biased towards well-trained samples, and tends to have a larger norm as training progresses (
‖
∇
𝜽
𝑙
𝑛
‖
2
 decreases) [EF-limitation]. We term this the inversely-scaled projection issue of the EF update, which is further illustrated in the following section.

4.2Visual Illustration

The detrimental impact of the inversely-scaled projection issue of EF updates is illustrated in a 2-parameter 2-datum linear least-square regression problem in Fig. 1 (third plot). It is shown that EF updates are “attracted” to the minimum of each training sample (the dashed lines), leading to a distorted update vector field and inefficient training trajectories. Also, EF updates have a much larger norm when either training sample is nearly converged, suggesting the necessity of a complicated step-size scheduler. Please refer to Appendix B for a detailed description and discussion, which also includes an additional visualisation for a logistic regression setup in Fig. 4 which leads to similar observations. These effects of the inversely-scaled projection issue are further validated in experiments (E1) and (E2) in large-scale deep learning setups in Sec. 7.

Figure 1:A visual comparison of Fisher, iEF and EF as pre-conditioners for a 2-parameter 2-datum linear least-squares regression problem inspired by [EF-limitation] (see Appendix B for details). All three plots are loss landscapes with the 
𝑥
-axis and 
𝑦
-axis representing 
𝜃
0
 and 
𝜃
1
 respectively. The first plot shows the gradient vector field of the loss function and 5 sampled training trajectories for SGD updates. Similarly, the second plot is for NGD/iEF updates and the third plot is for EF updates (with a zoomed view). The global minimum (0, 0) is marked with a star where visible. The two dashed lines on all plots represent the optimal parameter sets for each training sample. It can be seen that the EF method has a highly distorted update vector field while the iEF and NGD methods adapt to the curvature of the problem successfully.
5Improved Empirical Fisher

The EF method is a widely used approximate NGD method, mainly because it can be implemented conveniently by constructing the EF matrix with the per-sample gradients that are readily computed during backpropagation. In this section, we propose the improved EF (iEF) method which preserves the implementational convenience of the EF method, meanwhile alleviating the inversely-scaled projection issue. The iEF method can be justified as an approximate (generalised) NGD method from a loss reduction perspective. Continuous-time convergence analyses also show that the iEF method guarantees sub-linear/linear convergence to the global minimum under mild assumptions.

5.1Update Formulation

The nature of the inversely-scaled projection issue is that the EF update enforces a constant loss reduction regardless of the convergence level of each sample (see Sec. 4.1). To address this issue, the iEF update is designed to induce a per-sample loss reduction that takes into account the convergence level. The loss reduction induced by the iEF update for the 
𝑛
-th sample is designed to be

	
Δ
⁢
(
𝑙
𝑛
)
iEF
=
∇
𝜽
𝑙
𝑛
⊤
⁢
Δ
⁢
𝜽
iEF
≈
−
𝜂
⁢
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
2
.
		
(7)

where 
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
 is the gradient norm at the model output logits-level. Note that 
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
 in general decreases as the 
𝑛
-th sample gets better trained because of the positive convexity of the objective of interest (cross-entropy with softmax activation). Therefore, this update formulation allows the induced per-sample loss reduction by the iEF update to be closely related to how well a sample has converged, which then greatly alleviates the inversely-scaled projection issue of the EF method.

A viable formulation of the iEF update 
Δ
⁢
𝜽
iEF
 that both satisfies Eqn. (7) and relies only on the per-sample gradients is proposed as follows

	
Δ
⁢
𝜽
iEF
=
−
𝜂
⁢
∇
𝜽
𝒍
⊤
⁢
(
∇
𝜽
𝒍
⁢
∇
𝜽
𝒍
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝒔
iEF
,
		
(8)

where 
𝒔
iEF
∈
ℝ
𝑁
 is a scaling vector defined as

	
𝒔
iEF
=
[
‖
∇
𝒛
1
𝑙
1
‖
2
2
	
‖
∇
𝒛
2
𝑙
2
‖
2
2
	
⋯
	
‖
∇
𝒛
𝑁
𝑙
𝑁
‖
2
2
]
⊤
.
	

which can be obtained along with back-propagation (e.g. in Pytorch [pytorch]) with negligible overhead. This improved formulation for EF is shown to be effective. In the toy examples in Fig. 1 and 4, switching from EF to iEF completely removes the distortion in the EF update vector fields. Results in Sec. 7 also validate that iEF achieves consistently better approximation quality to NG updates than both EF and SF methods in practical deep learning setups (experiment (E1)), meanwhile being robust to the choice of damping 
𝜆
 across tasks and training stages (experiment (E3)).

5.2Theoretical Connection to Generalised NGD

The choice of scaling vector 
𝒔
iEF
 is motivated by the Gauss-Newton (GN) algorithm, which is a type of generalised NGD method [GDMO-ETH]. The update for the GN algorithm is defined as

	
Δ
⁢
𝜽
GN
=
−
𝜂
⁢
𝐆
^
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
,
		
(9)

where 
𝐆
^
=
∑
𝑛
∇
𝜽
𝒛
𝑛
⊤
⁢
∇
𝜽
𝒛
𝑛
 is the GN matrix. The GN algorithm can be effectively viewed as a gradient descent method on the model output logits space (
𝒛
𝑛
-space) and the loss reduction induced for the 
𝑛
-th sample by the GN update is approximately

	
Δ
⁢
(
𝑙
𝑛
)
GN
≈
−
𝜂
⁢
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
2
,
		
(10)

which takes exactly the same form as the per-sample loss reduction induced by the iEF update (see Eqn. (7)). Therefore, the iEF method can be regarded as an efficient approximation to the GN algorithm in terms of its loss-reduction behaviour. In particular, it can be shown that the iEF method is equivalent to the GN algorithm for all supervised learning problems with a regression model and the exact NGD method for the least-squares regression problem (see Appendix A).

5.3Convergence Analysis

In this section, two continuous time convergence analyses are provided for the non-stochastic version of the iEF method, which shows its sub-linear or linear global convergence guarantee for different types of objective functions (see Appendix C for proofs). The analysis can be considered as extensions of proofs provided in [NGD-LSQ] to setups using non-regression models and cross-entropy objectives. The two base assumptions used by the two convergence analysis are as follows:

Assumption 5.1.

At time 
𝑡
, the full-batch, un-damped iEF update to model parameters 
𝜽
⁢
(
𝑡
)
 is

	
\od
⁢
𝜽
⁢
(
𝑡
)
⁢
𝑡
=
−
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
⁢
[
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⁢
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
]
−
1
⁢
𝒔
iEF
⁢
(
𝑡
)
,
		
(11)
Assumption 5.2.

∀
𝑡
>
0
, the gradient covariance matrix (or Gram matrix) 
[
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
]
⁢
[
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
]
⊤
 is always full rank.

The two main conclusions of the analysis are described below.

Sub-linear Global Convergence for Softmax + Cross-Entropy Objective

When the target model uses softmax output and cross-entropy loss (as described in Sec. 3), the Theorem 5.3 can be proved.

Theorem 5.3.

Suppose Assumption 5.2 holds, 
∀
𝑛
∈
{
1
,
…
,
𝑁
}
, the target probability 
𝑝
^
𝑛
⁢
(
𝑡
)
:=
𝑝
𝛉
⁢
(
𝑡
)
⁢
(
𝑦
=
𝑦
𝑛
|
𝐱
𝑛
)
 for the 
𝑛
-th training sample is bounded as follows

	
𝑝
^
𝑛
⁢
(
𝑡
)
>
1
−
2
𝑡
+
𝐶
0
+
1
,
		
(12)

where 
𝐶
0
=
1
1
−
𝑝
^
𝑛
⁢
(
0
)
+
log
⁡
𝑝
^
𝑛
⁢
(
0
)
1
−
𝑝
^
𝑛
⁢
(
0
)
 and 
𝑡
>
𝑚
⁢
𝑎
⁢
𝑥
⁢
{
−
1
−
𝐶
0
,
0
}
.

Linear Global Convergence for Strongly Convex Objective

When the target model uses an 
𝑚
-strongly convex objective function [GD-convergence] (see Assumption C.2, note that cross-entropy loss does not satisfy this assumption), the Theorem 5.4 can be proved.

Theorem 5.4.

Suppose Assumption 5.2 and C.2 holds, 
∀
𝑛
∈
{
1
,
…
,
𝑁
}
, the per-sample loss 
𝑙
𝑛
⁢
(
𝑡
)
 for the 
𝑛
-th training sample is bounded as follows

	
𝑙
𝑛
⁢
(
𝑡
)
−
𝑙
𝑛
⋆
≤
𝑒
−
2
⁢
𝑚
⁢
𝑡
⁢
(
𝑙
𝑛
⁢
(
0
)
−
𝑙
𝑛
⋆
)
,
		
(13)

where 
𝑙
𝑛
⋆
 is the minimum loss for the 
𝑛
-th sample.

Remark: Theorem 5.4 only assumes a strongly-convex target objective w.r.t model output (Assumption C.2). The target loss landscape w.r.t model parameters can still be arbitrarily non-convex depending on the target model structure.

5.4Applications of IEF

As an approximate NGD method, the exact iEF method can be used directly as an optimiser (see Algorithm 1) for models with a small parameter size. Its performance is evaluated in experiment (E2) in Sec. 7, which demonstrates competitive convergence and generalisation when compared to well-tuned baselines. Refer to Appendix. D.1 for discussions on the implementation and complexity.

More importantly, the iEF method provides an improved approximation method to the exact Fisher matrix. The iEF approximated Fisher matrix (iEF matrix) 
𝐅
~
⋆
∈
ℝ
𝑃
×
𝑃
 takes the following form

	
𝐅
~
⋆
=
∇
𝜽
𝒍
⊤
⁢
diag
⁢
(
𝒔
iEF
)
−
1
⁢
∇
𝜽
𝒍
,
		
(14)

which can be derived from Eqn. (8) (see Appendix D.2.1). 
𝐅
~
⋆
 by design takes a highly similar form to the EF matrix (see Eqn 3), making them equally convenient to compute. Also, results in Sec. 7 show that updates preconditioned with the iEF matrix achieve consistently better approximation quality to NG updates than both EF and SF updates, meanwhile obviating the need for damping tuning. Consequently, the iEF matrix can be considered as a cheap yet better approximation method for the Fisher matrix than both the EF and SF methods, which opens up the possibility of improving a wide range of Fisher-based methods (not limited to optimisation methods). An example is provided in Appendix D.2.2 to demonstrate that iEF can be easily integrated into the popular empirical K-FAC optimiser [dan-NG]. Preliminary experimental results show that the integration leads to consistent improvements of the approximation quality to exact NG updates. Another example is provided in Appendix D.2.3 to demonstrate that iEF can be directly applied to improve the EF approximated Hessian used in the WoodFisher algorithms for model compression [woodfisher].

6Empirical Evaluation Framework for Approximate NGD Methods

Traditional evaluation methods for quality of approximate NGD methods have high memory and time complexity, which is infeasible for large setups (see discussion in Sec. 2). In order to accurately evaluate the quality of approximate NGD methods (EF, iEF, SF etc.) in practical deep-learning setups, we introduce an efficient empirical evaluation framework which enables a quantitative comparison of different approximate NGD methods under large-scale setups. For a given approximate NGD method that generates an update 
Δ
⁢
𝜽
, our proposed evaluation framework satisfies the following requirements: 1) provides a quantitative evaluator 
𝛾
⁢
(
Δ
⁢
𝜽
)
 that measures the (direction-wise) approximation quality to the exact NG update; 2) the evaluation process is efficient in modern auto-grad frameworks, and it poses no constraints on the size or structure of the target model. The implementation and theoretical motivations of this empirical evaluation framework are discussed in the following sections.

6.1Efficient Indicator of Approximation Quality

The proposed evaluation framework revolves around the indicator 
𝛾
⁢
(
Δ
⁢
𝜽
)
 which is designed to accurately reflect the quality of an approximate NG update, while being efficient to compute. For an update 
Δ
⁢
𝜽
 of interest, the proposed indicator 
𝛾
⁢
(
Δ
⁢
𝜽
)
∈
ℝ
+
 is defined as

	
𝛾
⁢
(
Δ
⁢
𝜽
)
=
(
Δ
⁢
𝜽
⊤
⁢
𝐅
⁢
Δ
⁢
𝜽
)
1
2
|
Δ
⁢
𝜽
⊤
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
|
,
		
(15)

and the smaller the value of 
𝛾
⁢
(
Δ
⁢
𝜽
)
, the better the approximation quality of 
Δ
⁢
𝜽
 to the exact NG update. This indicator mainly requires computing a matrix-vector product with the exact Fisher matrix (i.e. 
F
⁢
Δ
⁢
𝜽
), which can be efficiently done in modern auto-grad frameworks [pytorch]. This allows for the application of this framework to large-scale models in practical setups. Refer to Appendix F.1 for implementation details, algorithm complexity and a comparison with traditional methods.

6.2Theoretical Motivation

In this section, the proposed indicator 
𝛾
⁢
(
⋅
)
 is justified as a theoretically appropriate evaluator of the quality of an approximate NG update. An alternative definition for the NGD is first proposed, which formulates the NG update direction with an unconstrained optimisation problem as

	
𝜁
′
⁢
𝐅
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
=
arg
⁢
min
Δ
⁢
𝜽
⁡
𝛾
⁢
(
Δ
⁢
𝜽
)
2
,
		
(16)

where 
𝜁
′
∈
ℝ
 is an arbitrary non-zero scalar. It is shown that this alternative definition for NGD is implicitly used in the Hessian-free method [HF] and the linear conjugate gradient (CG) algorithm used in Hessian-free to solve for the exact NG update is a locally optimal minimiser for 
𝛾
⁢
(
Δ
⁢
𝜽
)
2
 (see Appendix F.2 for proof). Under this definition, any approximate NG update with a smaller 
𝛾
⁢
(
Δ
⁢
𝜽
)
2
 is a “strictly better approximation” to the exact NG update (which is the minimiser for 
𝛾
⁢
(
Δ
⁢
𝜽
)
2
).

Furthermore, 
𝛾
⁢
(
Δ
⁢
𝜽
)
 can also be justified from a second-order optimisation perspective. 
1
2
⁢
𝛾
⁢
(
Δ
⁢
𝜽
)
2
 is shown to quantify the maximum achievable loss reduction for a given update direction under a local quadratic approximation of the loss function (see Appendix. F.3 for proof). Consequently, the proposed indicator can be used to accurately predict the convergence ability of a target update generation method (see experiment (E2) in Sec. 7).

7Experiments

Experimental results are presented in this section. The main goal of the experiments is to verify that the behaviour of exact EF and iEF methods align with our theories in practical deep learning setups. Mainly three approximation methods are compared: EF, iEF and SF (an unbiased yet more expensive Fisher approximation method, see Appendix E). The exact updates of each method are generated based on Eqn. (5), (8), (49) respectively. Fifteen different setups are used to evaluate the optimisation performance and the approximation quality of these methods, including widely used parameter-efficient fine-tuning (PEFT) for pre-trained models. These include T5-base with LoRA and Prompt-Tuning on GLUE tasks [PEFT], and ViT with LoRA for CIFAR100 [ViT-lora]. PEFT of pre-trained models is investigated because it involves large-scale practical models, while having a small trainable parameter size (the implementation of exact EF, iEF and SF methods are memory intensive, see Appendix D.1). Please refer to Appendix H.1 for detailed experimental setups. The following three findings are demonstrated with our experiments.

(E1) The approximation quality (to exact NG updates) of EF, iEF, SF and SGD was evaluated and compared using the proposed evaluation framework in Sec. 6 on all setups. It is shown that iEF consistently improves on SGD updates and is superior to both EF and SF methods for the majority of the training stages for all setups.

(E2) The optimisation performance of EF, iEF, SF and SGD was evaluated on all setups. For each task, an additional well-tuned baseline optimiser (Adafactor/AdamW) was also compared. It is shown that iEF consistently achieves comparable or better performance than the corresponding baseline, while EF and SF suffer from unstable training to different extents.

(E3) The impact of damping on the approximation quality of EF, iEF and SF was analysed under the proposed evaluation framework. It is shown that the quality of traditional EF and SF methods relies heavily on careful damping tuning, unlike iEF which works well with any near-zero damping across tasks and training stages.

Finally, results for an additional experiment considering a 10M parameter Multi-layer Perceptron (MLP) on the CIFAR10 [cifar] dataset are provided in Appendix H.7. This additional experiment further validates the aforementioned findings for a train-from-scratch setup with a much larger (
10
×
) trainable parameter size.

E1: Approximation Quality to NG Updates

The behaviour of updates generated with EF, iEF, SF and SGD methods were compared using the proposed empirical evaluation framework in terms of their approximation quality to exact NG updates. The updates for EF, iEF and SF were generated according to Eqns. (5), (8), and (49) respectively, and the evaluation framework follows Algorithm 4. The “un-damped” behaviour of these methods is analysed and a near-zero damping factor is used for update generation. The checkpoints at the end of each epoch generated by the baseline optimisation methods (AdamW/Adafactor) for each task were used for evaluation. In each evaluation. For each checkpoint 
𝜽
⁢
(
𝑡
)
, indicators were computed from 100 batches of randomly picked training samples of the target task of batch size 
𝑀
=
160
. The averaged indicator for each update were then evaluated (
𝛾
¯
⁢
(
Δ
⁢
𝜽
SGD
⁢
(
𝑡
)
)
, 
𝛾
¯
⁢
(
Δ
⁢
𝜽
EF
⁢
(
𝑡
)
)
, 
𝛾
¯
⁢
(
Δ
⁢
𝜽
iEF
⁢
(
𝑡
)
)
, 
𝛾
¯
⁢
(
Δ
⁢
𝜽
SF
⁢
(
𝑡
)
)
, which are denoted as 
𝛾
SGD
,
𝛾
iEF
,
𝛾
EF
,
𝛾
SF
 for simplicity). The relationship among these indicators across epochs and tasks is shown in Fig. 2. Note that results are presented for only 3 representative setups due to space limit (indicator plots for all tasks are shown in Appendix H.5.1). Three findings can be concluded from these figures: 1)  EF achieves poorer approximation quality even than SGD updates for most training stages and tasks. This is aligned with the finding in prior work that EF is a questionable approximation to NGD. 2)  The fourth plot shows that the gradient norm imbalance gets larger as training progresses. This correlates well with both the EF and SF curves, while impacting iEF less. This means that the inversely-scaled projection issue indeed plays a significant role in reducing the approximation quality of the EF (and SF) approximation. 3)  Comparing the first three plots, it can be seen that, for the majority of the training stages, the approximation quality follows iEF 
>
 SF 
>
 EF. IEF gives a consistently better approximation, and EF and SF are only able to beat iEF at the start of training (where a good approximation to the NG update has less impact).

Figure 2:Four (log-scaled) ratios computed for checkpoints at various stages of training (sampled at the interval of one epoch) for 3 of the all 15 tasks. The 
𝑥
-axes represent the training stages of the model. 
0
%
 means the initialised model and 
100
%
 means model at the end of the last epoch. Each data point is averaged across 100 evaluations, and the error bars represent the standard deviation (1-sigma). The first plot shows 
𝛾
EF
/
𝛾
SGD
, which denotes the relative approximation quality improvement of EF updates w.r.t. SGD updates (the lower the better). The second plot shows 
𝛾
iEF
/
𝛾
SGD
, and the third plot shows 
𝛾
SF
/
𝛾
EF
. The last plot depicts the imbalance of gradient norms, which is the average ratio between the maximum and minimum gradient norm for each evaluated batch (a larger value indicates more imbalanced per-sample gradient norms, which should lead to a more significant inversely-scaled projection issue). Overall, the approximation quality follows iEF 
>
 SF 
>
 EF.
E2: Optimisation Performance

The exact iEF, EF and SF methods were implemented as stochastic optimisers (following Algorithms 1, 2, 3 respectively). The same near-zero damping factor was used as in (E1). The averaged test metrics for GLUE and CIFAR100 for each optimiser are shown in Table 1 (see full test results in Table 7, validation result in Table 6, final training loss in Table 5 and training curves in Fig. 12 and 13). The following three observations can be made:
1)  From the final training loss reported in Table 5, the ranking of final training loss generally follows iEF 
<
 AdamW/Adafactor 
<
 SGD 
<
 SF 
<
 EF (the lower the better). This ranking of training convergence follows the ranking of indicators in (E1) closely, demonstrating the effectiveness of the empirical evaluation framework in predicting the training behaviour of optimisers. 2)  For most of the tasks, EF always suffer from unstable training (see training curves in Fig. 12 and 13), while iEF consistently reaches the lowest training loss at the end of training (even when compared with well-tuned Adafactor/AdamW baselines). This further confirms the inversely-scaled projection issue of EF, and demonstrates the strong convergence ability of the proposed iEF method. 3)  From test results in Table 1, it can be seen that iEF achieves the best generalisation for Prompt Tuning tasks (outperformed Adafactor in 6 out of 7 tasks). For LoRA tasks, iEF remains competitive to AdamW with each of them outperformed the other in 4 out of 8 tasks. This is likely because LoRA setups (which on average have 50 times more trainable parameters than Prompt Tuning) have a stronger reliance on regularisation and momentum, which have not been properly extended to use together with the exact iEF optimiser yet. Overall, iEF achieves the best generalisation for the majority of tasks (10 out of 15), indicating its potential as a strong optimiser for PEFT for pre-trained models.

Table 1:Average test performance of different optimisers for GLUE and CIFAR100. For GLUE tasks, the average metric results for the 7 tasks are used as the final test score. For tasks with two metrics, these metrics are averaged first [GLUE]. For all tasks, the test result is computed for the best validation accuracy checkpoint. Refer to Table 7 for a more complete test performance report and detailed explanations on metrics.
	AdamW	Adafactor	SGD	EF	SF	iEF
GLUE + T5 + Prompt Tuning	-	
77.1
	
67.4
	
48.1
	69.7	
79.3

GLUE + T5 + LoRA	
80.1
	-	
77.3
	
63.1
	
76.5
	
79.3

CIFAR100 + ViT + LoRA	
93.9
	-	
91.3
	
31.0
	
92.8
	
94.3
E3: Impact of Damping

As is discussed in Sec. 2, practical approximate NGD optimisers rely heavily on a good damping schedule, which is typically chosen based on empirical experience [K-FAC]. Using the proposed evaluation framework, it is straightforward to analyse the impact of damping on the approximation quality of EF, SF and iEF. For a target task, the indicator 
𝛾
 w.r.t. damping 
𝜆
 curve is computed at the start, mid-way and end of the training. Graph for an example task is shown in Fig. 3 (graphs for other tasks are provided in Appendix H.5.2). Two observations can be made:
1)  A well-chosen damping factor significantly improves the approximation quality of EF and SF, which aligns well with observations in prior work on approximate NGD optimisers [K-FAC, KFAC-with-EF1, KFAC-with-EF2]. However, the optimal damping factor changes greatly for different tasks and training stages, which makes the damping schedule for SF or EF based optimisers necessary yet hard-to-design in practice. 2)  Across all tasks and training stages, iEF robustly achieves great approximation quality with a near-zero damping factor. More importantly, its approximation quality is consistently better than EF method, and is comparable to the optimally-damped SF method (which is much more expensive, particularly when the cost of damping tuning is considered). Overall, iEF can be considered a cheaper, higher-quality and more robust alternative to both the EF and SF approximation methods.

Figure 3:Approximation quality (relative to SGD) of EF, SF and iEF methods w.r.t. damping factor 
𝜆
 at different training stages of task CoLA+T5+LoRA. 
𝑥
-axes show the value of the damping factor, 
𝑦
-axes depict the relative approximation quality improvement of the target update method w.r.t. SGD (the lower the better). Each data point is averaged across 100 evaluations, and the error-bars represent the standard deviation (1-sigma). The first plot is for checkpoint saved at the end of the first training epoch, the second plot for the mid-way epoch and the third plot for the final epoch. It can be observed that iEF achieves the best approximation quality robustly for any near-zero 
𝜆
. In contrast, 
𝜆
 has a non-linear impact on both SF and EF. When optimally tuned, an EF update can achieve better approximation quality than SGD, and an SF update can achieve comparable quality to iEF. However, the optimal damping factor for EF and SF changes greatly with training stages (and tasks).
8Conclusions and Future Work

This paper presents the iEF method, which addresses the inversely-scaled projection issue of the EF approximation for NGD, meanwhile maintaining the implementational convenience. A novel empirical evaluation framework for the quality of general approximate NGD update is also proposed, which enables quantified comparison of approximate NGD methods in large deep learning setups1. Based on the experiments with practical PEFT of pre-trained models for NLP and CV classification tasks, the exact iEF optimiser shows superior convergence and generalisation for majority of the tasks, supporting the applicability of iEF directly as an optimiser. Further evaluation on approximation quality concludes that iEF achieves consistently better approximation quality than both EF and SF. The iEF method also demonstrates the superior property of being robust to the choice of damping factor across different tasks and training stages.

As is discussed in Sec. 5.4, the iEF method can be viewed not only as an improved approximate NGD optimiser, but also as an improved approximation method for the exact Fisher matrix in general. This opens up many opportunities of future work to improve a wide range of Fisher-based methods (not limited to optimisation methods). Some example applications include improving the empirical K-FAC optimiser [dan-NG, KFAC-with-EF1] (which has shown promising results in preliminary experiments) and improving the WoodFisher algorithm for model compression [woodfisher].

Acknowledgements

Xiaodong Wu is in part funded by the Cambridge Trust, a donation from Meta Systems and Christ’s College, Cambridge. This work has in part been performed using resources provided by the Cambridge Tier-2 system operated by the University of Cambridge Research Computing Service (www.hpc.cam.ac.uk) funded by EPSRC Tier 2 capital grant EP/T022159/1.

\printbibliography
Appendix ARelation between iEF, GN and NGD in Different Machine Learning Setups

The connection between iEF, GN and NGD methods is discussed for different common machine-learning scenarios. It is explained that for a scalar output (regression) model, the iEF method and the GN algorithm is equivalent. Furthermore, for a least-squares problem, the iEF, GN and NGD methods are all equivalent.

A.1Regression Model Problem

In this section, a supervised learning problem for a target model with only a scalar output (i.e. regression model) is discussed. The definition of the setup is as follows.

Consider a target regression model 
𝑓
𝜽
⁢
(
⋅
)
∈
ℝ
𝑃
→
ℝ
, where 
𝜽
∈
ℝ
𝑃
 is the trainable parameters of size 
𝑃
. Given 
𝑁
 i.i.d. training samples 
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑁
 (where 
𝒙
𝑛
 is the input feature vector and 
𝑦
𝑛
∈
ℝ
 is the scalar label), the output of the model for the 
𝑛
-th sample is 
𝑧
𝑛
=
𝑓
𝜽
⁢
(
𝒙
𝑛
)
 where 
𝑧
𝑛
∈
ℝ
 is a scalar. For the 
𝑛
-th sample, the per-sample loss 
𝑙
𝑛
 is defined as

	
𝑙
𝑛
=
ℱ
obj
⁢
(
𝑧
𝑛
,
𝑦
𝑛
)
,
		
(17)

where 
ℱ
obj
⁢
(
⋅
)
∈
ℝ
→
ℝ
 is the per-sample objective function, and the accumulated loss 
∑
𝑛
𝑙
𝑛
 is to be minimised. Some examples of this problem setup include least-squares regression problems where a mean-square error is used as the loss function, and binary classification problems where the Sigmoid-activation plus binary cross-entropy is used as the loss function.

Recall the definition of the iEF matrix in Eqn. (14). In this problem setup, the iEF matrix can be computed as follows

	
F
~
⋆
=
∑
𝑛
‖
\od
⁢
𝑙
𝑛
⁢
𝑧
𝑛
‖
2
−
2
⁢
(
\od
⁢
𝑙
𝑛
⁢
𝑧
𝑛
⁢
∇
𝜽
𝑧
𝑛
)
⁢
(
\od
⁢
𝑙
𝑛
⁢
𝑧
𝑛
⁢
∇
𝜽
𝑧
𝑛
)
⊤
=
∑
𝑛
∇
𝜽
𝑧
𝑛
⁢
∇
𝜽
𝑧
𝑛
⊤
=
𝐆
^
,
		
(18)

which takes the same form as the GN matrix. Therefore, the iEF and GN methods are equivalent to supervised learning problems of regression models.

A.2Least-Squares Problem

The least-squares problem is a representative example of supervised learning problems of regression models, which includes the toy example shown in Fig. 1. The definition of this type of problem follows most of the definitions in Appendix A.1, apart from the per-sample objective function, which is elaborated as follows. For the least-squares problem, the objective function is defined as follows

	
ℱ
obj
⁢
(
𝑧
𝑛
,
𝑦
𝑛
)
=
1
2
⁢
(
𝑧
𝑛
−
𝑦
𝑛
)
2
.
		
(19)

The Fisher matrix for the least-squares problem is defined as follows [INTERPLAY]

	
𝐅
=
∑
𝑛
∇
𝜽
𝑧
𝑛
⁢
∇
𝜽
𝑧
𝑛
⊤
=
F
~
⋆
=
𝐆
^
,
		
(20)

which coincides with the iEF and GN matrix in this problem setup. Consequently, the iEF, GN and NGD methods are equivalent for least-squares problems. This is verified in the second plot of Fig. 1 where NGD and iEF share the same update vector field.

Appendix BVisualisation for Linear Least-Squares Problem with Two Training Samples
Setup Description

In this section, the Fig. 1 referenced in Sec. 4.2 is explained in detail. The vector field graph is based on a simple linear least-squares regression problem, with 2 training samples and 2 trainable parameters.

The linear least-squares problem is chosen for visualisation because it is not only a highly representative machine learning setup [INTERPLAY, EF-limitation], but also there has been a precedent of using this problem to visualise the distortion in EF (see Figure 1 in [EF-limitation]). The trainable parameter size of 
𝑃
=
2
 is chosen to facilitate the 2D visualisation, and the training sample size of 
𝑁
=
2
 (i.e. 
𝑁
≤
𝑃
) is chosen to better match the practical deep learning scenarios where the model is over-parameterised.

The target linear model is formulated as 
𝑓
𝜽
⁢
(
𝑥
𝑛
)
=
𝜽
⊤
⁢
[
𝑥
𝑛
,
1
]
⊤
=
𝜃
0
+
𝜃
1
⁢
𝑥
𝑛
 with 
𝑥
𝑛
∈
ℝ
 and 
𝜽
∈
ℝ
2
. The two training samples are 
(
𝑥
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑁
=
{
(
0
,
0
)
,
(
1
,
0
)
}
 (with 
𝑁
=
2
) and the target loss to be minimised is 
ℒ
⁢
(
𝜽
)
=
∑
𝑛
=
1
2
1
2
⁢
[
𝑦
𝑛
−
𝑓
𝜽
⁢
(
𝑥
𝑛
)
]
2
. It is obvious that the optimal parameter (at global minimum) is 
𝜽
⋆
=
[
0
,
0
]
⊤
.

The update vector fields are generated using the following definition:

	
{
Δ
⁢
𝜽
SGD
	
=
∇
𝜽
ℒ
⁢
(
𝜽
)


Δ
⁢
𝜽
NGD
	
=
(
𝐅
+
𝜆
NGD
⁢
𝐈
)
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)


Δ
⁢
𝜽
EF
	
=
(
𝐅
~
+
𝜆
EF
⁢
𝐈
)
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)


Δ
⁢
𝜽
iEF
	
=
(
𝐅
~
⋆
+
𝜆
iEF
⁢
𝐈
)
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
.
		
(21)

The EF matrix 
𝐅
~
 follows the definition in Eqn. (3). The iEF matrix 
𝐅
~
⋆
 and Fisher matrix 
𝐅
 shares the same definition as shown in Eqn. (20) (see Appendix A.2). For this toy problem, the damping factors 
𝜆
iEF
 and 
𝜆
NGD
 are set to zero. The damping factor 
𝜆
EF
 is set to zero everywhere, apart from when one of the per-sample gradient gives a norm of 0, a damping factor of 
1
×
10
−
4
⁢
max
⁡
(
diag
⁢
(
𝐅
~
)
)
 is used to facilitate inversion. Finally, in the third plot of Fig. 1, the EF update vectors are normalised for better visualisation, because EF updates have hugely different scales across the contour plots. In the zoomed view next to the third plot, the EF update vectors are not normalised, which should give a better demonstration of the inverse scaling issue of EF updates.

The same set of 5 initial parameters (shown as black solid octagons) are used to generate 5 training trajectories (coloured curves) on each plot in Fig. 1. Each follows the corresponding update vector fields (each step follows the direction of the update vector field, and is normalised to 
1
×
10
−
2
). To demonstrate how each sample affects the EF updates, the optimal parameter set for each training sample (i.e. when 
1
2
⁢
[
𝑦
𝑛
−
𝑓
𝜽
⁢
(
𝑥
𝑛
)
]
2
=
0
) is added on all plots in Fig. 1 as dashed lines. For training sample 1: (0, 0), the optimal parameter set is 
𝜃
0
=
0
, which is shown as horizontal dashed lines. For training sample 2: (1, 0), the optimal parameter set is 
𝜃
0
+
𝜃
1
=
0
, which is shown as diagonal dashed lines.

Observations

By observing the behaviour of the vector fields for different updates on the loss landscape, along with the 5 sampled trajectories in Fig. 1, it can be seen that all methods successfully reached the global minimum, with the NGD/iEF updates having the most efficient training trajectory. However, the EF method has highly distorted update vector field and training trajectories due to the inversely-scaled projection issue discussed in Sec. 4.1. The following conclusions can be drawn for EF updates:

1. 

Directional bias towards converged samples: The EF update vector field is easily “attracted” to the dashed lines because the EF update is strongly affected by the better-converged sample (due to inverse scaling). This leads to a distorted update vector field and ineffective training trajectories. In particular, when per-sample gradients have a highly imbalanced norm near the dashed lines, the EF update vector field becomes almost parallel to the loss contour. This causes them to deviate significantly from both the gradient descent direction and the optimal direction (NG update direction), and oscillations can be observed along the dashed line in the zoomed view.

2. 

Inversely-scaled update norm: EF updates have larger norms when either of the training samples is nearly converged (see the zoomed plot). In fact, the EF training trajectories managed to converge because we normalised every update to a small norm of 
1
×
10
−
2
, which is not needed for SGD, NGD and iEF methods. Consequently, it is likely that a sophisticated learning rate scheduler is necessary when training with the EF method in practical deep learning setups (as is mentioned in [EF-limitation]).

Motivation

As is stated in Sec. 3, the paper mainly focuses on classification setups, but a visualisation of a regression setup in Fig. 1 is used in the main paper for the following reasons: 1)  The least-squares regression problem is commonly used when analysing NGD and EF in the literature [EF-limitation, INTERPLAY]. Particularly, our visualisation follows a similar setup to [EF-limitation], which is an important related work regarding the limitations of EF. Overall, the toy regression example allows for consistency with the literature. 2)  The 2-datum least-squares regression problem have several nice properties. There exists a unique global minimum; the NG update can find the global minimum in one step, and the advantage over all other updates is self-evident; the iEF update and NG update has the same form; the distortion of the EF update is extreme. All of these properties make the visualisation in Fig. 1 much more straightforward to understand than a visualisation for a classification setup.

Additional Visualisation for Logistic Regression Problem

Given that the paper focuses on classification setups, it is important to also include a visualisation for a toy classification setup. Consequently, an additional visualisation, in a similar fashion, that compares Fisher, iEF and EF as pre-conditioners in a toy classification setup is provided in Fig. 4. This figure considers a 2-datum logistic regression setup and a more detailed description is provided in the caption. This new visualisation demonstrates consistent results to that of Fig. 1. Particularly, it can be observed that EF updates deviate from the optimal decision boundary because it is biased toward further optimising the better classified datum (i.e. the datum with a lower CE loss). Also, EF updates become larger when either of the datum achieves a small loss. Meanwhile, SGD, NGD and iEF update approaches the optimal decision boundary consistently, and arguably NGD and iEF reach the optimal decision boundary more effectively than SGD.

Figure 4:A visual comparison of Fisher, iEF and EF as pre-conditioners for a logistic regression problem (classifying two 1D datum 
𝑥
0
=
0
,
𝑥
1
=
2
 into two classes). The target model is 
𝑝
𝜽
⁢
(
𝑥
𝑛
)
=
𝜎
⁢
(
𝜃
0
+
𝜃
1
⁢
𝑥
𝑛
)
, where 
𝜎
⁢
(
⋅
)
 is the Sigmoid function and CE loss is used, which follows the problem setup description in Sec. 3. This figure is different from Fig. 1 in 3 aspects: 1) iEF and NG updates are no longer identical, and are presented in separate plots; 2) There is no global minima, but the model achieves lower loss when moving further down the bottom-right corner; 3) The dashed line now represents the optimal parameter set for a decision boundary of 
𝑥
=
1
. The training trajectory of EF is still ill-behaved, meanwhile both NG and iEF updates move towards the optimal decision boundary smoothly.
Appendix CConvergence Analysis

The proofs for Theorem. 5.3 and Theorem. 5.3 are provided in this section. The proofs follow and extend the continuous-time analysis framework in [NGD-LSQ] and [GD-convergence].

Justification of Full-rank Gram Matrix Assumption

Recall Assumption 5.2 which assumes 
[
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
]
⁢
[
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
]
⊤
 to be full-rank throughout the training process. This is equivalent to assuming 
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
∈
ℝ
𝑁
×
𝑃
 always have full row rank. This is considered a mild assumption for practical deep learning because:

1. 

It is generally true that for each individual per-sample gradient 
∇
𝜽
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
, the norm of the gradient will be zero if and only if 
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
=
𝟎
 (i.e. at the minimum of that sample). Therefore, it is unlikely that there exists zero norm per-sample gradients in 
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
 during training.

2. 

The target model is usually highly over-parameterised with 
𝑃
≫
𝑁
 and the model is highly complex. It is in general unlikely that 
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
 is rank-deficient, as long as there are no duplicate training samples.

Ideal Per-sample Loss Change for IEF

Given Assumption 5.1 and Assumption 5.2, the following Lemma can be derived

Lemma C.1.

Suppose Assumption 5.2 holds, 
∀
𝑛
∈
{
1
,
…
,
𝑁
}
, the loss reduction induced on the 
𝑛
-th training sample by the iEF update follows

	
\od
⁢
𝑙
𝑛
⁢
(
𝑡
)
⁢
𝑡
=
−
‖
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
‖
2
2
.
		
(22)

Proof of Lemma. C.1: 
\od
⁢
𝒍
⁢
(
𝑡
)
⁢
𝑡
 can be computed as follows

	
\od
⁢
𝒍
⁢
(
𝑡
)
⁢
𝑡
=
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⁢
\od
⁢
𝜽
⁢
(
𝑡
)
⁢
𝑡
=
−
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⁢
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
⁢
[
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⁢
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
]
−
1
⁢
𝒔
iEF
⁢
(
𝑡
)
=
−
𝒔
iEF
⁢
(
𝑡
)
.
		
(23)

Therefore, 
\od
⁢
𝑙
𝑛
⁢
(
𝑡
)
⁢
𝑡
=
[
\od
⁢
𝒍
⁢
(
𝑡
)
⁢
𝑡
]
𝑛
=
−
‖
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
‖
2
2
.

Global Sub-linear Convergence for Softmax + CE Objective Function

The proof for Theorem 5.3 is provided here. In this analysis, the supervised learning setup with softmax + CE objective function is considered (as described in Sec. 3). This is a setup commonly used in practical deep learning, making this convergence analysis practically relevant.

As stated in Sec. 3, the target model is assumed to use a softmax activation function. For the 
𝑛
-th sample at time step 
𝑡
, 
𝒛
𝑛
⁢
(
𝑡
)
=
𝑓
𝜽
⁢
(
𝑡
)
⁢
(
𝒙
𝑛
)
, where 
𝒛
𝑛
⁢
(
𝑡
)
∈
ℝ
𝐶
 is the output logits. The output probability for class 
𝑐
 can be computed from the logits using the softmax activation: 
[
𝑝
𝑛
⁢
(
𝑐
)
]
⁢
(
𝑡
)
=
[
𝝈
SM
⁢
(
𝒛
𝑛
⁢
(
𝑡
)
)
]
𝑐
 where 
𝝈
SM
⁢
(
⋅
)
:
ℝ
𝐶
→
ℝ
𝐶
 is the softmax activation function.

The per-sample loss in this setup is therefore defined as

	
𝑙
𝑛
(
𝑡
)
:=
−
log
[
𝑝
𝑛
(
𝑦
𝑛
)
]
(
𝑡
)
:=
−
log
𝑝
^
𝑛
(
𝑡
)
=
−
log
[
𝝈
SM
(
𝒛
𝑛
(
𝑡
)
)
]
𝑦
𝑛
,
		
(24)

where 
[
𝝈
SM
⁢
(
𝒛
𝑛
)
]
𝑦
𝑛
 represents the output probability for the target class 
𝑦
𝑛
. It can be seen that the lowest loss for sample 
𝑛
 tends to 0, which is achieved when 
𝑝
^
𝑛
⁢
(
𝑡
)
→
1
.

The gradient of 
𝑙
𝑛
⁢
(
𝑡
)
 w.r.t. 
𝒛
𝑛
⁢
(
𝑡
)
 can be computed as

	
[
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
]
𝑐
=
{
−
[
𝑝
𝑛
⁢
(
𝑐
)
]
⁢
(
𝑡
)
	
𝑐
≠
𝑦
𝑛
,


1
−
𝑝
^
𝑛
⁢
(
𝑡
)
	
𝑐
=
𝑦
𝑛
.
		
(25)

The norm of the gradient satisfies the following inequality

	
‖
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
‖
2
2
≥
[
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
]
2
.
		
(26)

Combining Lemma. C.1 with the definition in Eqn. (24), the following equation can be obtained

	
\od
⁢
𝑙
𝑛
⁢
(
𝑡
)
⁢
𝑡
=
−
\od
⁢
log
⁡
𝑝
^
𝑛
⁢
(
𝑡
)
⁢
𝑡
=
−
1
𝑝
^
𝑛
⁢
(
𝑡
)
⁢
\od
⁢
𝑝
^
𝑛
⁢
(
𝑡
)
⁢
𝑡
=
−
‖
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
‖
2
2
.
		
(27)

Combining Eqn. (26)(27), the following inequality can be obtained

	
1
[
𝑝
^
𝑛
⁢
(
𝑡
)
]
⁢
[
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
]
2
⁢
d
⁢
𝑝
^
𝑛
⁢
(
𝑡
)
≥
d
⁢
𝑡
.
		
(28)

Integrating on both sides gives

	
1
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
+
log
⁡
𝑝
^
𝑛
⁢
(
𝑡
)
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
−
𝐶
0
≥
𝑡
.
		
(29)

where 
𝐶
0
=
1
1
−
𝑝
^
𝑛
⁢
(
0
)
+
log
⁡
𝑝
^
𝑛
⁢
(
0
)
1
−
𝑝
^
𝑛
⁢
(
0
)
. It is known that 
𝑥
>
log
⁡
𝑥
 for 
𝑥
>
0
, therefore the inequality can be further relaxed to

	
1
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
+
𝑝
^
𝑛
⁢
(
𝑡
)
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
>
𝑡
+
𝐶
0
,
		
(30)

which is equivalent to

	
2
1
−
𝑝
^
𝑛
⁢
(
𝑡
)
>
𝑡
+
𝐶
0
+
1
.
		
(31)

Now consider a large 
𝑡
, s.t. 
𝑡
+
𝐶
0
+
1
>
0
, a bound can be provided for 
𝑝
^
𝑛
⁢
(
𝑡
)
 as follows

	
𝑝
^
𝑛
⁢
(
𝑡
)
>
1
−
2
𝑡
+
𝐶
0
+
1
.
		
(32)

This shows that iEF guarantees global sub-linear convergence for target probability 
𝑝
^
𝑛
→
1
 for every training sample. It equivalently implies that iEF guarantees sub-linear convergence to the global minimum of the accumulated cross-entropy loss (achieved when 
∀
𝑛
=
{
1
,
…
⁢
𝑁
}
, 
𝑝
^
𝑛
=
1
), given the per-sample objective function is the softmax + cross-entropy combination.

C.1Global Linear Convergence for Strongly Convex Objective Functions

The proof for Theorem 5.4 is provided here. Unlike for Theorem 5.3, where the model is assumed to use a softmax + cross-entropy loss function, this theorem is applicable to a group of common loss functions: 
𝑚
-strongly convex objective functions.

The details of the setup used here mostly follow that in Sec. 3. The differences are described as follows. Given 
𝑁
 i.i.d. training samples 
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑁
 (where 
𝒙
𝑛
 is the input feature vector and 
𝑦
𝑛
 is a label of arbitrary form), the output of the model for the 
𝑛
-th sample is 
𝒛
𝑛
=
𝑓
𝜽
⁢
(
𝒙
𝑛
)
 where 
𝑧
𝑛
∈
ℝ
𝐶
 is a general model output vector (no longer logits vector). A 
𝑚
-strongly convex objective loss function 
ℱ
obj
⁢
(
⋅
)
 is used to compute the per-sample loss for the 
𝑛
-th sample 
𝑙
𝑛
∈
ℝ
 as follows

	
𝑙
𝑛
=
ℱ
obj
⁢
(
𝒛
𝑛
,
𝑦
𝑛
)
:-
ℱ
obj
⁢
(
𝒛
𝑛
)
,
		
(33)

where the label 
𝑦
𝑛
 is omitted for simplicity. Finally, the following accumulated loss is to be minimised

	
ℒ
⁢
(
𝜽
)
=
∑
𝑛
𝑙
𝑛
.
		
(34)

In addition to Assumption  5.1 and 5.2, an additional assumption is made for the objective loss function as follows

Assumption C.2.

For the 
𝑛
-th sample, the objective loss function 
ℱ
obj
⁢
(
𝒛
𝑛
)
 is assumed to be 
𝑚
-strongly convex on the model output space 
𝒛
𝑛
, which then satisfies the following Polyak-Lojasiewicz inequality [Polyak-ineq]

	
ℱ
obj
⁢
(
𝒛
𝑛
)
−
ℱ
obj
⁢
(
𝒛
𝑛
⋆
)
≤
1
2
⁢
𝑚
⁢
‖
∇
𝒛
𝑛
ℱ
obj
⁢
(
𝒛
𝑛
)
‖
2
,
		
(35)

where 
ℱ
obj
⁢
(
𝒛
𝑛
⋆
)
 is the global minimum of the loss for the 
𝑛
-th sample. For simplicity, the notation 
𝑙
𝑛
⋆
 is used in place of 
ℱ
obj
⁢
(
𝒛
𝑛
⋆
)
. The inequality is therefore rewritten as follows

	
𝑙
𝑛
−
𝑙
𝑛
⋆
≤
1
2
⁢
𝑚
⁢
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
.
		
(36)

Assumption. C.2 is quoted from [GD-convergence], which covers a wide range of loss functions including mean-square-error. Note that under this assumption, both per-sample losses and accumulated loss can still have an arbitrarily non-convex landscape on the parameter space.

Based on Lemma C.1, when training with iEF updates, for the 
𝑛
-th sample, the loss change is

	
\od
⁢
𝑙
𝑛
⁢
(
𝑡
)
⁢
𝑡
=
−
‖
∇
𝒛
𝑛
⁢
(
𝑡
)
𝑙
𝑛
⁢
(
𝑡
)
‖
2
2
		
(37)

Using the Polyak-Lojasiewicz inequality in Assumption. C.2, the following inequality can be obtained

	
\od
⁢
(
𝑙
𝑛
⁢
(
𝑡
)
−
𝑙
𝑛
⋆
)
⁢
𝑡
≤
−
2
⁢
𝑚
⁢
(
𝑙
𝑛
⁢
(
𝑡
)
−
𝑙
𝑛
⋆
)
⇒
1
(
𝑙
𝑛
⁢
(
𝑡
)
−
𝑙
𝑛
⋆
)
⁢
\od
⁢
(
𝑙
𝑛
⁢
(
𝑡
)
−
𝑙
𝑛
⋆
)
⁢
𝑡
≤
−
2
⁢
𝑚
.
		
(38)

Integrating on both sides gives the following

	
𝑙
𝑛
⁢
(
𝑡
)
−
𝑙
𝑛
⋆
≤
𝑒
−
2
⁢
𝑚
⁢
𝑡
⁢
(
𝑙
𝑛
⁢
(
0
)
−
𝑙
𝑛
⋆
)
,
		
(39)

which shows that iEF pre-conditioned gradient flow linearly converges to the global minimum for every sample. This then implies that iEF has a global linear convergence guarantee for an accumulated loss 
ℒ
⁢
(
𝜽
)
.

Appendix DDiscussion on Practical Applications of IEF
D.1Stochastic IEF/EF Optimiser

As is mentioned in Sec. 5.4, the stochastic version of the exact iEF method can be directly used as an optimiser. It is described in Algorithm 1. In this section, the implementation of this exact iEF optimiser is discussed, along with its computation and memory complexity. Note that the stochastic exact EF optimiser can be constructed in a similar form in Algorithm 2. The following discussions should also apply to the exact EF optimiser.

Algorithm 1 Stochastic Optimisation with Exact IEF
  Require: All 
𝑁
 training samples, initial model parameters 
𝜽
⁢
(
0
)
  for 
𝑡
=
0
 to 
𝑇
−
1
 do
     Sample Mini-batch 
ℳ
⁢
(
𝑡
)
 with size 
𝑀
     Perform forward pass on batch 
ℳ
⁢
(
𝑡
)
 to obtain the per-sample loss vector 
𝒍
⁢
(
𝑡
)
     Perform back-propagation to obtain Jacobian 
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
 and logits gradient norm vector 
𝒔
iEF
⁢
(
𝑡
)
     Compute iEF update 
Δ
⁢
𝜽
iEF
⁢
(
𝑡
)
=
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
⁢
(
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⁢
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝒔
iEF
⁢
(
𝑡
)
: Eqn. (8)
     Update model 
𝜽
⁢
(
𝑡
+
1
)
=
𝜽
⁢
(
𝑡
)
−
𝜂
⁢
Δ
⁢
𝜽
iEF
⁢
(
𝑡
)
  end for
 
Algorithm 2 Stochastic Optimisation with Exact EF
  Require: All 
𝑁
 training samples, initial model parameters 
𝜽
⁢
(
0
)
  for 
𝑡
=
0
 to 
𝑇
−
1
 do
     Sample Mini-batch 
ℳ
⁢
(
𝑡
)
 with size 
𝑀
     Perform forward pass on batch 
ℳ
⁢
(
𝑡
)
 to obtain the per-sample loss vector 
𝒍
⁢
(
𝑡
)
     Perform back-propagation to obtain Jacobian 
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
     Compute EF update 
Δ
⁢
𝜽
EF
⁢
(
𝑡
)
=
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
⁢
(
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⁢
∇
𝜽
⁢
(
𝑡
)
𝒍
⁢
(
𝑡
)
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝟏
: Eqn. (5)
     Update model 
𝜽
⁢
(
𝑡
+
1
)
=
𝜽
⁢
(
𝑡
)
−
𝜂
⁢
Δ
⁢
𝜽
EF
⁢
(
𝑡
)
  end for
Implementation Details

There are two main aspects of the implementation of this optimiser that are non-trivial. These are discussed respectively as follows:

1. 

Jacobian matrix 
∇
𝜃
𝑙
:

The computation of 
∇
𝜽
𝒍
 is effectively collecting the per-sample gradients for a batch during the back-propagation process. In Pytorch [pytorch], the per-sample gradients are readily computed during back-propagation, but are usually accumulated along the batch dimension to compute the total gradient, and are not available for collection directly. Therefore, additional backward hooks need to be attached to trainable modules to store these per-sample gradients during the back-propagation process. This is a standard procedure used in most approximate NGD optimisers [K-FAC, GDN, TONGA]. Our implementation is partially based on this K-FAC implementation. Note that this additional procedure of computing per-sample gradients is negligible w.r.t. forward and backward of large pre-trained models in PEFT setups, making iEF/EF have comparable speed as standard AdamW/Adafactor/SGD optimisers.

2. 

Logits gradient norm vector 
𝑠
iEF
:

During back-propagation in Pytorch [pytorch], the gradient of logits 
∇
𝒛
𝑛
𝑙
𝑛
 is already computed, but is not stored because logits vector 
𝒛
𝑛
 is not a leaf-node. This can be easily changed by calling the “.retain_grad()” method on the logits vector. Although this operation is non-standard to common approximate NGD optimisers, it adds negligible cost to standard back-propagation (in general deep learning setup, not limited to PEFT) and its effect on speed can be ignored.

Time and Memory Complexity

Given a batch size of 
𝑀
, and trainable parameters of size 
𝑃
, assume the per-sample gradients have already been computed through backpropagation, the time complexity of computing each update is 
𝑂
⁢
(
𝑀
3
+
𝑀
2
⁢
𝑃
)
, and the memory complexity is 
𝑂
⁢
(
𝑀
2
+
𝑀
⁢
𝑃
)
. Due to model over-parameterisation, we have 
𝑃
≫
𝑀
. Then, the time complexity becomes 
𝑂
⁢
(
𝑀
2
⁢
𝑃
)
 and memory complexity becomes 
𝑂
⁢
(
𝑀
⁢
𝑃
)
. In practical deep learning frameworks such as Pytorch [pytorch], the limiting factor for the applicability of such an exact EF-like method is the memory complexity 
𝑂
⁢
(
𝑀
⁢
𝑃
)
, which is essentially the storage of the 
𝑀
 per-sample gradients involved in the computation of exact iEF updates. It is therefore only possible to apply exact EF-like methods to models with small trainable parameter sizes (either full tuning of small models or parameter-efficient fine-tuning of large models) or small batch size 
𝑀
. This is the reason why exact EF is never directly used to optimise modern deep learning models, and additional approximation is always necessary (e.g. K-FAC [K-FAC] or SVD-based pruning [TONGA]). Nevertheless, given the rise of large pre-trained models [llama, GPT3, T5, whisper], parameter-efficient fine-tuning has gained traction [ViT-lora, lora, PEFT] and direct application of exact iEF may still be beneficial (as the trainable parameter size is usually 
<
1
%
 of the pre-trained model size).

D.2Improving Existing Fisher-based Methods with IEF

It is mentioned in Appendix D.1 that exact iEF as an optimiser is limited by its memory complexity. Given the descent training performance of exact iEF optimiser, it would be interesting to observe the performance of iEF on more general setups. As is mentioned in Sec. 5.4, it can be achieved by incorporating iEF into existing EF-based optimisers such as empirical K-FAC [K-FAC]. In this section, an improvement to the EF-based K-FAC optimiser with iEF is proposed, which can act as a starting point for future work in improving other approximate empirical NGD methods.

D.2.1Expression of IEF Matrix

The iEF matrix can be computed according to Eqn. (14): 
𝐅
⋆
=
∇
𝜽
𝒍
⊤
⁢
diag
⁢
(
𝒔
iEF
)
−
1
⁢
∇
𝜽
𝒍
. This can be derived from Eqn. (8) using the identity 
(
𝐔𝐕
+
𝜆
⁢
𝐈
)
−
1
⁢
𝐔
=
𝐔
⁢
(
𝐕𝐔
+
𝜆
⁢
𝐈
)
−
1
. For simplicity of derivation, we denote 
𝐉
=
∇
𝜽
𝒍
 and assume 
𝐉
 is a square matrix and is full-rank. Also, we denote 
𝐒
=
diag
⁢
(
𝒔
iEF
)
1
2
 and assume the diagonal matrix 
𝐒
 to be full-rank. Therefore, the proposed iEF matrix can be expressed as 
𝐅
⋆
=
𝐉
⊤
⁢
𝐒
−
2
⁢
𝐉
. The following derivation can be made

	
(
𝐅
⋆
)
−
1
⁢
(
∇
𝜽
𝒍
⊤
⁢
𝟏
)
	
=
(
𝐉
⊤
⁢
𝐒
−
2
⁢
𝐉
)
−
1
⁢
𝐉
⊤
⁢
𝟏
=
[
(
𝐉
⊤
⁢
𝐒
−
1
)
⁢
(
𝐒
−
1
⁢
𝐉
)
]
−
1
⁢
(
𝐉
⊤
⁢
𝐒
−
1
)
⁢
(
𝐒
⁢
𝟏
)
		
(40)

		
=
(
𝐉
⊤
⁢
𝐒
−
1
)
⁢
[
(
𝐒
−
1
⁢
𝐉
)
⁢
(
𝐉
⊤
⁢
𝐒
−
1
)
]
−
1
⁢
𝐒
⁢
𝟏
=
𝐉
⊤
⁢
𝐒
−
1
⁢
𝐒
⁢
[
𝐉𝐉
⊤
]
−
1
⁢
𝐒𝐒
⁢
𝟏
	
		
=
𝐉
⊤
⁢
[
𝐉𝐉
⊤
]
−
1
⁢
𝐒
2
⁢
𝟏
=
𝐉
⊤
⁢
[
𝐉𝐉
⊤
]
−
1
⁢
𝒔
iEF
.
	

This shows that the update pre-conditioned by 
𝐅
⋆
 indeed is the same as the iEF update described in Eqn. (8).

The expression for the iEF matrix in Eqn. (14) can be alternatively rewritten as follows

	
𝐅
⋆
=
∇
𝜽
𝒍
⊤
⁢
diag
⁢
(
𝒔
iEF
)
−
1
⁢
∇
𝜽
𝒍
=
∑
𝑛
(
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
−
1
⁢
∇
𝜽
𝑙
𝑛
)
⁢
(
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
−
1
⁢
∇
𝜽
𝑙
𝑛
)
⊤
,
		
(41)

which differs from the EF matrix in Eqn. (3) in that each per-sample gradient is re-scaled with 
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
−
1
. Such simple scaling is easy to implement in most approximate empirical NGD optimisers.

D.2.2Improving Empirical K-FAC with IEF

The empirical K-FAC [dan-NG] is a widely used version [KFAC-with-EF1, KFAC-with-EF2, KFAC-with-EF3, KFAC-with-EF4] of the original K-FAC method [K-FAC]. Its formulation can be described as follows.

Assume there are 
𝑛
 samples in the target batch. For one fully connected layer in a target model, the weight matrix is denoted by matrix 
𝐖
∈
ℝ
𝑚
×
𝑘
, the (batched) input is denoted by 
𝒂
∈
ℝ
𝑛
×
𝑚
 and the (batched) output is denoted by row vector 
𝒄
∈
ℝ
𝑛
×
𝑘
. They satisfy 
𝒄
=
𝒂
⁢
𝐖
. The gradient 
∇
𝒄
𝒍
∈
ℝ
𝑛
×
𝑘
 is denoted by 
𝒈
. The block-diagonal portion of the EF matrix corresponding to this layer (denoted as 
𝐅
~
𝐖
) can be estimated using K-FAC approximation as follows [K-FAC]

	
𝐅
~
𝐖
=
𝔼
⁢
[
(
𝒈
⊤
⁢
𝒈
)
⊗
(
𝒂
⊤
⁢
𝒂
)
]
≈
𝔼
⁢
[
𝒈
⊤
⁢
𝒈
]
⊗
𝔼
⁢
[
𝒂
⊤
⁢
𝒂
]
.
		
(42)

This is based on the gradient expression for weight matrix 
𝐖
: 
∇
𝐖
𝑙
=
𝒈
⊤
⁢
𝒂
. By rescaling this gradient with vector 
𝒔
iEF
, the expression for 
𝐅
~
𝐖
 becomes:

	
𝐅
~
𝐖
⋆
=
𝔼
⁢
[
𝒈
⊤
⁢
diag
⁢
(
𝒔
iEF
)
−
1
⁢
𝒈
]
⊗
𝔼
⁢
[
𝒂
⊤
⁢
𝒂
]
,
		
(43)

where the 
𝒔
iEF
 vector is easy to compute in Pytorch [pytorch] with negligible extra cost (as shown in Appendix D.1), and such diagonal rescaling is straightforward to implement. In conclusion, the idea of iEF can be easily integrated into existing approximate empirical NGD optimisers, and it is interesting to observe the improvements due to such integration.

Preliminary evaluation of the approximation quality to the exact NG update of the block-diagonal version of the iEF method is conducted using the evaluation framework proposed in Sec. 6 (following the style of experiment (E1)). The approximation quality to exact NG updates (relative to SGD updates) are reported for iEF, KFAC (block-diagonal version of SF method [K-BFGS]), eKFAC (block-diagonal version of EF method [dan-NG]) and ieKFAC (block-diagonal version of iEF method), for different training stages of selected tasks in Fig. 5. All updates are generated using a near-zero damping.

Figure 5:Approximation quality (relative to SGD) of “un-damped” iEF, ieKFAC, KFAC and eKFAC for 3 selected PEFT tasks (QNLI+LoRA, RTE+LoRA, MRPC+LoRA) across training stages. The style of the visualisation follows that for the first 3 plots of Fig. 2. This evaluation shows that, ieKFAC update has a similar approximation quality to the exact iEF method, and a much better approximation quality than both KFAC and eKFAC in most training stages. This demonstrates the effectiveness of using ieKFAC to approximate iEF and its potential of further improving the approximation quality of existing KFAC-based methods.
D.2.3Improving Empirical WoodFisher Model Compression

The EF matrix is used in the WoodFisher model compression algorithm [woodfisher] as an approximation to the Hessian matrix. It is therefore natural to consider using the iEF matrix in place of the EF matrix to improve the approximation quality. The WoodFisher algorithm relies on a recursion-base formulation of the EF matrix as follows

	
𝐅
~
𝑛
+
1
=
𝐅
~
𝑛
+
1
𝑁
⁢
∇
𝜽
𝑙
𝑛
⁢
∇
𝜽
𝑙
𝑛
⊤
.
		
(44)

This can be easily switched to the iEF matrix using Eqn. 41 as follows

	
𝐅
𝑛
+
1
⋆
=
𝐅
𝑛
⋆
+
1
𝑁
⁢
(
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
−
1
⁢
∇
𝜽
𝑙
𝑛
)
⁢
(
‖
∇
𝒛
𝑛
𝑙
𝑛
‖
2
−
1
⁢
∇
𝜽
𝑙
𝑛
)
⊤
,
		
(45)
Appendix EMonte-Carlo Sampled Fisher Matrix

The SF method used in the experimental setups (see Sec. 7) is introduced and analysed in details in this section. A commonly used method to estimate the exact Fisher matrix without bias is through Monte-Carlo sampling, which is used in the original K-FAC method [K-FAC]. This method is referred to as sampled Fisher in this paper. Particularly, when only one Monte-Carlo sample is generated for each training sample, the corresponding method is termed SF for brevity. Recall the definition of the Fisher matrix in Eqn. (2), it can be rewritten using expectations as follows

	
F
=
∑
𝑛
𝔼
𝑐
∼
𝑝
𝜽
⁢
(
𝑦
|
𝒙
𝑛
)
⁢
[
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑐
)
⁢
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑐
)
⊤
]
.
		
(46)

This means, by generating enough labels (with Monte-Carlo sampling) from the output distribution 
𝑝
𝜽
⁢
(
𝑦
|
𝑥
𝑛
)
, and computing the gradient w.r.t. these labels 
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑐
)
, the Fisher matrix can be estimated without bias.

Assume 
𝐾
 labels are generated for each training sample, the exact expression for sampled Fisher with 
𝐾
 samples (denoted as 
𝐅
^
⁢
(
𝐾
)
) is as follows

	
𝐅
^
⁢
(
𝐾
)
=
1
𝐾
⁢
∑
𝑛
=
1
𝑁
∑
𝑘
=
1
𝐾
[
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑦
^
𝑛
(
𝑘
)
)
⁢
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑦
^
𝑛
(
𝑘
)
)
⊤
]
=
1
𝐾
⁢
𝐉
^
⁢
(
𝐾
)
⊤
⁢
𝐉
^
⁢
(
𝐾
)
,
		
(47)

where 
𝑦
^
𝑛
(
𝑘
)
∼
𝑝
𝜽
⁢
(
𝑦
|
𝒙
𝑛
)
 is the 
𝑘
-th generated label for the 
𝑛
-th sample, Jacobian 
𝐉
^
⁢
(
𝐾
)
∈
ℝ
(
𝑁
⁢
𝐾
)
×
𝑃
 denotes the stacked sampled gradients 
[
∇
𝜽
log
⁡
𝑝
1
⁢
(
𝑦
^
1
(
1
)
)
,
…
,
∇
𝜽
log
⁡
𝑝
𝑛
⁢
(
𝑦
^
𝑛
(
𝑘
)
)
,
…
,
∇
𝜽
log
⁡
𝑝
𝑁
⁢
(
𝑦
^
𝑁
(
𝐾
)
)
]
⊤
.

𝐅
^
⁢
(
1
)
, i.e. sampled Fisher with 1 sampling for each training sample, is used as a baseline approximate NGD method in Sec. 7 (termed as SF in this paper) and is compared against EF and iEF. This has two reasons:

1. 

For 
𝐅
^
⁢
(
𝐾
)
, 
𝐾
=
1
 is more commonly chosen in practice [Shampoo, K-FAC] than 
𝐾
>
1
. This is mainly because 
𝐅
^
⁢
(
𝐾
)
 requires 
𝐾
 additional back-propagations through the target batch, which becomes very expensive for a large 
𝐾
.

2. 

𝐅
^
⁢
(
1
)
 has a maximum rank of 
𝑁
, which is the same rank as the EF matrix (
𝐅
~
) and the iEF matrix (
𝐅
~
⋆
).

Note that even for 
𝐅
^
⁢
(
1
)
, as compared to EF and iEF methods, requires an additional back-propagation through target batches. This makes 
𝐅
^
⁢
(
1
)
 hard to implement in practice, and becomes nearly twice as expensive as EF/iEF. It is the leading reason that EF is commonly used in favour of 
𝐅
^
⁢
(
1
)
 in practice [KFAC-with-EF1, KFAC-with-EF2, KFAC-with-EF3, KFAC-with-EF4].

Exact Pre-conditioning with 
𝐅
^
⁢
(
1
)

To properly evaluate the pre-conditioner 
𝐅
^
⁢
(
1
)
 either through optimisation or our evaluation framework (see Sec. 6), it is necessary to exactly compute the pre-conditioned gradient. The Sherman-Morrison-Woodbury (SMW) identity [SMW] can be used to achieve this, which states that 
(
𝐔
⊤
⁢
𝐔
+
𝜆
⁢
𝐈
)
−
1
=
1
𝜆
⁢
[
𝐈
−
𝐔
⊤
⁢
(
𝐔𝐔
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝐔
]
.

The update pre-conditioned by 
𝐅
^
⁢
(
1
)
 (denoted as 
Δ
⁢
𝜽
SF
) takes the following form

	
Δ
⁢
𝜽
SF
=
(
𝐅
^
⁢
(
1
)
+
𝜆
⁢
𝐈
)
−
1
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
.
		
(48)

By plugging in 
𝐅
^
⁢
(
1
)
=
𝐉
^
⁢
(
1
)
⊤
⁢
𝐉
^
⁢
(
1
)
 and using the SMW identity, the update can then be computed as follows

	
Δ
⁢
𝜽
SF
=
1
𝜆
⁢
[
𝐈
−
𝐉
^
⁢
(
1
)
⊤
⁢
(
𝐉
^
⁢
(
1
)
⁢
𝐉
^
⁢
(
1
)
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝐉
^
⁢
(
1
)
]
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
,
		
(49)

which is easy to compute once the Jacobian 
(
𝐉
^
(
1
)
 for sampled gradients is collected (using the per-sample gradient collection method described in Appendix D.1). Similar to EF and iEF, an exact stochastic optimiser can also be constructed for SF, as is described in Algorithm 3. Note that as compared to Algorithm 2,1, SF requires an additional back-propagation.

Algorithm 3 Stochastic Optimisation with Exact SF
  Require: All 
𝑁
 training samples, initial model parameters 
𝜽
⁢
(
0
)
  for 
𝑡
=
0
 to 
𝑇
−
1
 do
     Sample Mini-batch 
ℳ
⁢
(
𝑡
)
 with size 
𝑀
     Perform forward pass on batch 
ℳ
⁢
(
𝑡
)
 to obtain output distribution 
𝒑
⁢
(
𝑡
)
     Perform back-propagation to obtain accumulated loss 
∇
𝜽
⁢
(
𝑡
)
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
     Sample output labels 
𝒚
^
⁢
(
𝑡
)
∼
𝒑
⁢
(
𝑡
)
     Compute cross-entropy loss between 
𝒑
⁢
(
𝑡
)
 and sampled label 
𝒚
^
⁢
(
𝑡
)
 to obtain the pseudo per-sample loss vector 
𝒍
^
⁢
(
𝑡
)
     Perform additional back-propagation to obtain Jacobian 
𝐀
⁢
(
𝑡
)
=
∇
𝜽
⁢
(
𝑡
)
𝒍
^
⁢
(
𝑡
)
     Compute SF update 
Δ
⁢
𝜽
SF
⁢
(
𝑡
)
=
1
𝜆
⁢
[
𝐈
−
𝐀
⁢
(
𝑡
)
⊤
⁢
(
𝐀
⁢
(
𝑡
)
⁢
𝐀
⁢
(
𝑡
)
⊤
+
𝜆
⁢
𝐈
)
−
1
⁢
𝐀
⁢
(
𝑡
)
]
⁢
∇
𝜽
⁢
(
𝑡
)
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
: Eqn. (49)
     Update model 
𝜽
⁢
(
𝑡
+
1
)
=
𝜽
⁢
(
𝑡
)
−
𝜂
⁢
Δ
⁢
𝜽
SF
⁢
(
𝑡
)
  end for
Impact of Inversely-Scaled Projection on SF Update

Although the same analysis in Sec. 4 cannot be applied to SF due to the different formulation of the update, the inversely-scaled projection issue is still expected to impact SF. As the model gets better trained, it becomes likely for the sampled Fisher matrix 
𝐅
^
⁢
(
1
)
 of SF to sample empirical per-sample gradients. Pre-conditioning with the inverse of this matrix would then cause the inverse-scaling issue just like EF. Consequently, a poorer approximation quality of SF to the exact NG update is still probable.

Appendix FEmpirical Evaluation Framework for Approximate NGD Methods
F.1Implementation Details

As introduced in Sec. 6, the proposed empirical evaluation framework is built around the proposed indicator 
𝛾
⁢
(
⋅
)
. Consider a given model checkpoint 
𝜽
⁢
(
𝑡
)
, 
𝑀
 pairs of training samples 
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑀
 from a target dataset and 
𝐾
 approximate NGD methods 
𝒈
(
𝑘
)
⁢
(
⋅
)
 of interest. Each update generation method can generate an update 
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
 for the provided samples and a given model checkpoint following 
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
=
𝒈
(
𝑘
)
⁢
(
𝜽
⁢
(
𝑡
)
,
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑀
)
. The framework is implemented such that 
𝐾
 indicators 
𝛾
⁢
(
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
)
 are computed, which allows for the quantitative comparison of these update generation methods w.r.t. their approximation quality to the exact NGD method. The detailed evaluation process is described in Algorithm 4. In practice, for the 
𝑘
-th update generation method, multiple batches are used to compute a series of 
𝛾
⁢
(
Δ
⁢
𝜽
(
𝑘
)
)
, and their average is used to provide a more accurate indicator value. Note that this evaluation process can be repeated for model checkpoints at different training stages 
𝜽
𝑡
′
 to provide a more thorough comparison.

It is worth mentioning that the ratio 
𝛾
⁢
(
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
)
/
𝛾
⁢
(
∇
𝜽
⁢
(
𝑡
)
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
)
 is an important quantity. This ratio describes the relative approximation quality of the target update generation method w.r.t. standard SGD method, and should be smaller than 1 for a good approximate NGD method. If this ratio is consistently larger than 1, then the target approximate NGD method 
𝒈
(
𝑘
)
⁢
(
⋅
)
 has a worse approximation quality than the SGD method and is likely to perform poorly in practice.

The four update methods investigated in the main paper (see Sec. 7) are SGD, EF, iEF and SF (sampled Fisher with 1 Monte-Carlo sample for each training sample). The SGD update is simply the gradient 
∇
𝜽
⁢
(
𝑡
)
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
; the iEF update is computed according to Eqn. (8); the EF update is computed according to Eqn. (5); the SF update is computed according to Eqn. (49).

Finally, the computationally most expensive operation when computing indicators is finding the matrix-vector product between the exact Fisher and a given update direction, i.e. 
𝐅
⁢
Δ
⁢
𝜽
. A detailed discussion is provided on its implementation and algorithm complexity in the following section.

Algorithm 4 Empirical evaluation framework for approximate NGD Methods
  Input: A batch of 
𝑀
 training data pairs 
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑀
, model checkpoint 
𝜽
⁢
(
𝑡
)
, loss function 
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
, 
𝐾
 update generation methods 
𝒈
(
𝑘
)
⁢
(
𝜽
⁢
(
𝑡
)
,
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑀
)
,
𝑘
∈
{
1
,
⋯
,
𝐾
}
.
  Execute:
  Compute accumulated gradient 
∇
𝜽
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
 on 
𝑀
 training samples
  for 
𝑘
=
1
 to 
𝐾
 do
     Compute update 
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
=
𝒈
(
𝑘
)
⁢
(
𝜽
⁢
(
𝑡
)
,
(
𝒙
𝑛
,
𝑦
𝑛
)
𝑛
=
1
𝑀
)
     Compute indicator 
𝛾
⁢
(
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
)
=
[
(
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
)
⊤
⁢
𝐅
⁢
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
]
1
2
|
(
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
)
⊤
⁢
∇
𝜽
⁢
(
𝑡
)
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
|
  end for
  Output: Return 
𝐾
 indicators 
[
𝛾
⁢
(
Δ
⁢
𝜽
⁢
(
𝑡
)
(
𝑘
)
)
]
𝑘
=
1
𝐾
. Updates with smaller indicator values are considered better approximate NG updates.
F.1.1Matrix-Vector Product and Algorithm Complexity

The most important operation in Algorithm 4 when computing the proposed indicator is the computation of the matrix-vector product 
𝐅
⁢
(
⋅
)
. It is the most expensive operation that dominates both the time and space complexity of this algorithm. Fortunately, this can be efficiently computed in a modern auto-grad framework using the double differentiation trick [TRPO]. The key to computing the matrix-vector product with the exact Fisher matrix is identifying the equivalence of the Fisher matrix and the Generalised Gauss-Newton matrix (GGN matrix, denoted as 
𝐆
). It can be shown that for the target model which uses softmax activation and cross-entropy loss function, the following equality holds

	
𝐅
=
∑
𝑛
∇
𝜽
𝒛
𝑛
⊤
⁢
(
∇
𝒛
𝑛
2
𝑙
𝑛
)
⁢
∇
𝜽
𝒛
𝑛
=
𝐆
.
		
(50)

Therefore, the matrix-vector product can be broken down into the computation of three separate matrix-vector products. The double differentiation trick is useful when computing the first two matrix-vector products: 
∇
𝜽
𝒛
𝑛
⁢
(
⋅
)
, 
∇
𝒛
𝑛
2
𝑙
𝑛
⁢
(
⋅
)
. The final matrix-vector product 
∇
𝜽
𝒛
𝑛
⊤
⁢
(
⋅
)
 can be computed using the standard back-propagation pipeline. Please refer to [TRPO] for implementation details.

Overall, with the double differentiation trick, the time and memory complexity of the matrix-vector product with the Fisher matrix is comparable to standard training with batch size 
𝑀
. In our implementation in Pytorch, the matrix-vector product requires 2 forward passes and 2 backward passes through the model on the 
𝑀
 samples. This makes our proposed empirical evaluation framework efficient in practice.

Finally, for the reader’s information, if we adopt the method used in [HF], it is possible to further reduce the cost of the matrix-vector product down to only 1 additional forward pass and backward pass. However, this may not be a preferable choice because it requires customised forward-propagation code, making our evaluation framework less generally applicable.

F.1.2Comparison to Traditional Evaluation Methods

As is summarised in Sec. 2, traditional evaluation of the approximation quality to the NGD method requires the explicit storage and inversion of the exact Fisher matrix. This is usually done using the definition in Eqn. (2). For a general model, computing the Fisher matrix in Eqn. (2) requires 
𝐶
 separate backwards and forward passes through the model (
𝐶
 being the output category number). This can range from 10 (e.g. MNIST[lecun2010mnist]) to >10,000 (for our LLM finetune setup [PEFT]). From a time complexity perspective, the traditional method can be arbitrarily more expensive than our evaluation framework depending on the output category size 
𝐶
. From a memory complexity perspective, the storage of the Fisher matrix of 
ℝ
𝑃
×
𝑃
 is infeasible for a large-scale model. Both of these concerns limit the traditional evaluation of approximate NGD methods in large practical setups. On the contrary, our proposed evaluation framework resolves these limitations of the traditional evaluation method, meanwhile having a strong theoretical motivation. This greatly facilitates the evaluation of approximate NGD methods in large-scale deep-learning setups for methods not limited to EF, iEF and SF.

F.2Hessian-free Implicitly Minimises 
𝛾
⁢
(
⋅
)
 Indicator

In Sec. 6.2, it is stated that the indicator 
𝛾
⁢
(
⋅
)
 is implicitly minimised in the linear CG algorithm to iteratively approximate the exact NG update in Hessian-free method [HF]. In this section, this statement is formally introduced and justified.

The Hessian-free method is an important approximate NGD method. Unlike most other approximate NGD methods, which approximate the Fisher matrix 
𝐅
 directly, the Hessian-free method does not explicitly express the Fisher matrix. Instead, it uses an iterative method to directly solve for 
Δ
⁢
𝜽
HF
 in the following equation

	
𝐅
⁢
Δ
⁢
𝜽
HF
=
∇
𝜽
ℒ
⁢
(
𝜽
)
.
		
(51)

The iterative method used in Hessian-free is the linear CG algorithm [CG], which is a classical method that iteratively solves for 
Δ
⁢
𝜽
HF
 in Eqn. (51) with only the access to the matrix-vector product function 
𝐅
⁢
(
⋅
)
 (introduced in Sec. 6). Since the Fisher is not required to be explicitly stored, the method enjoys a memory complexity of 
𝑂
⁢
(
𝑃
)
 where 
𝑃
 is the number of trainable parameters in a target model.

The linear CG algorithm (without pre-conditioning) used in HF is shown in Algorithm 5.

  Input: The maximum CG execution iterations 
𝑀
CG
  The gradient vector 
𝒃
=
∇
𝜽
ℒ
⁢
(
𝜽
)
  The matrix-vector product function for the Fisher 
𝐅
  Execute:
  Set 
𝒓
0
←
𝒃
  Set 
𝒗
0
←
𝒓
0
  Set 
𝑚
←
0
  while 
𝑚
<
𝑀
CG
 do
     Compute 
‖
𝒓
𝑚
‖
2
=
𝒓
𝑚
⊤
⁢
𝒓
𝑚
     Set 
𝛼
𝑚
←
‖
𝒓
𝑚
‖
2
/
(
𝒗
𝑚
⊤
⁢
𝐅
⁢
𝒗
𝑚
)
     Update 
𝒙
𝑚
+
1
←
𝒙
𝑚
+
𝛼
𝑚
⁢
𝒗
𝑚
     Update 
𝒓
𝑚
+
1
←
𝒓
𝑚
−
𝛼
𝑚
⁢
𝐅
⁢
𝒗
𝑚
     Compute 
‖
𝒓
𝑚
+
1
‖
2
=
𝒓
𝑚
+
1
⊤
⁢
𝒓
𝑚
+
1
     Set 
𝛽
𝑚
+
1
←
‖
𝒓
𝑚
+
1
‖
2
/
‖
𝒓
𝑚
‖
2
     Update 
𝒗
𝑚
+
1
←
𝒓
𝑚
+
1
+
𝛽
𝑚
+
1
⁢
𝒗
𝑚
     Update 
𝑚
←
𝑚
+
1
  end while
  Output: Return 
𝒙
𝑀
CG
 as an approximate solution for 
Δ
⁢
𝜽
EF
Algorithm 5 The Linear CG Algorithm in Hessian-free Method

It is known that the CG algorithm is an efficient solver for Eqn. (51) and is usually regarded as a locally optimal minimiser (each step uses an exact line search) for the equivalent pseudo loss function [HF, CG]

	
𝐿
CG
′
⁢
(
Δ
⁢
𝜽
HF
)
=
−
Δ
⁢
𝜽
HF
⊤
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
+
1
2
⁢
Δ
⁢
𝜽
HF
⊤
⁢
𝐅
⁢
Δ
⁢
𝜽
HF
.
		
(52)

However, it is also possible to interpret this method as a locally optimal minimiser [optimal-CG] for the generalised Rayleigh Quotient of positive semi-definite matrices 
𝐅
 and 
∇
𝜽
ℒ
⁢
(
𝜽
)
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
⊤

	
𝛾
⁢
(
Δ
⁢
𝜽
HF
)
2
=
(
Δ
⁢
𝜽
HF
⊤
⁢
𝐅
⁢
Δ
⁢
𝜽
HF
)
Δ
⁢
𝜽
HF
⊤
⁢
[
∇
𝜽
ℒ
⁢
(
𝜽
)
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
⊤
]
⁢
Δ
⁢
𝜽
HF
		
(53)

where 
𝛾
⁢
(
Δ
⁢
𝜽
HF
)
 happens to be the proposed indicator of this paper. The proof of this statement is provided in the following section.

F.2.1Proof that Linear CG is a Locally Optimal Minimiser for Indicator 
𝛾
⁢
(
⋅
)

Note that in every iteration in Algorithm 5, an update 
𝛼
𝑚
⁢
𝒗
𝑚
 is accumulated to the final solution 
𝒙
𝑀
CG
. This update 
𝛼
𝑚
⁢
𝒗
𝑚
 can be shown to achieve the local maximum reduction in 
𝛾
⁢
(
𝒙
)
2
 in every iteration. Formally, it is to be shown that, for the current partial solution 
𝒙
𝑚
 and the given search direction 
𝒗
𝑚
, the scaling factor 
𝛼
𝑚
 achieves the maximum reduction in 
𝛾
⁢
(
𝒙
𝑀
CG
)
2
:

	
𝛼
𝑚
=
arg
⁢
min
𝛼
𝑚
⁡
𝛾
⁢
(
𝒙
𝑀
CG
)
2
.
		
(54)

Now, assume the actual minimiser 
𝛼
𝑚
′
 is different from the 
𝛼
𝑚
 in the CG iterations. The true minimiser 
𝛼
𝑚
′
 can be acquired through a Ritz-Rayleigh analysis [optimal-CG]. By setting 
𝐁
′
=
∇
𝜽
ℒ
⁢
(
𝜽
)
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
⊤
, and plugging in the true minimiser 
𝛼
𝑚
′
, Eqn. (54) becomes:

	
𝛼
𝑚
′
=
arg
⁢
min
𝛼
𝑚
′
⁡
[
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
⊤
⁢
𝐅
⁢
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
]
[
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
⊤
⁢
𝐁
′
⁢
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
]
.
		
(55)

The Ritz-Rayleigh analysis then gives the optimal scaling:

	
𝛼
𝑚
′
=
𝒃
⊤
⁢
𝒗
𝑚
𝒗
𝑚
⊤
⁢
𝐅
⁢
𝒗
𝑚
.
		
(56)

Recall that 
𝛼
𝑚
=
𝒓
𝑚
⊤
⁢
𝒓
𝑚
𝒗
𝑚
⊤
⁢
𝐅
⁢
𝒗
𝑚
, in order to prove 
𝛼
𝑚
=
𝛼
𝑚
′
, the following need be proved 
𝒃
⊤
⁢
𝒗
𝑚
=
𝒓
𝑚
⊤
⁢
𝒓
𝑚
. This can be done through recursion.

For 
𝑚
=
0
, it is obvious that 
𝒃
⊤
⁢
𝒗
0
=
𝒃
⊤
⁢
𝒃
=
𝒓
0
⊤
⁢
𝒓
0
.

For 
𝑚
>
1
, it is known that 
𝒃
⊤
⁢
𝒗
𝑚
−
1
=
𝒓
𝑚
−
1
⊤
⁢
𝒓
𝑚
−
1
 is true, then:

	
𝒃
⊤
⁢
𝒗
𝑚
	
=
𝒃
⊤
⁢
(
𝒓
𝑚
+
𝛽
𝑚
⁢
𝒗
𝑚
−
1
)

	
=
𝒃
⊤
⁢
𝒓
𝑚
+
𝒓
𝑚
⊤
⁢
𝒓
𝑚
⊤
𝒓
𝑚
−
1
⊤
⁢
𝒓
𝑚
−
1
⁢
𝒃
⊤
⁢
𝒗
𝑚
−
1

	
=
𝒓
0
⊤
⁢
𝒓
𝑚
+
𝒓
𝑚
⊤
⁢
𝒓
𝑚
⊤
𝒓
𝑚
−
1
⊤
⁢
𝒓
𝑚
−
1
⁢
𝒓
𝑚
−
1
⊤
⁢
𝒓
𝑚
−
1

	
=
0
+
𝒓
𝑚
⊤
𝒓
𝑚
(
∵
∀
𝑖
≠
𝑗
,
𝒓
𝑖
⊤
𝒓
𝑗
=
0
)

	
=
𝒓
𝑚
⊤
⁢
𝒓
𝑚
.
		
(57)

In conclusion, 
𝛼
𝑚
 is indeed the true minimiser for 
[
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
⊤
⁢
𝐅
⁢
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
]
[
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
⊤
⁢
𝐁
′
⁢
(
𝒙
𝑚
+
𝛼
𝑚
′
⁢
𝒗
)
]
 in every CG iterations, making CG a locally optimal optimiser for 
𝛾
⁢
(
𝒙
)
2
.

It is known that Hessian-free iteratively approaches the exact NG update [HF], and a larger iteration number 
𝑀
CG
 leads to a better approximation. Now that it has been shown that the indicator 
𝛾
⁢
(
𝒙
)
 decreases (optimally) for every iteration of Hessian-free, it verifies that indicator 
𝛾
⁢
(
𝒙
)
 directly describes the approximation level of the intermediate solution 
𝒙
𝑚
 to the exact NG update.

F.3
𝛾
⁢
(
⋅
)
 Quantifies Loss Reduction Effectiveness under the Local Quadratic Approximation

The indicator 
𝛾
⁢
(
Δ
⁢
𝜽
)
 can also describe the loss-reduction effectiveness of a target update direction under the Local Quadratic Approximation. The loss change induced by a target update can be estimated using a Taylor expansion up to the second order as follows:

	
ℒ
⁢
(
𝜽
⁢
(
𝑡
+
1
)
)
≈
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
+
Δ
⁢
𝜽
⁢
(
𝑡
)
⊤
⁢
∇
𝜽
ℒ
⁢
(
𝜽
⁢
(
𝑡
)
)
+
1
2
⁢
Δ
⁢
𝜽
⁢
(
𝑡
)
⊤
⁢
𝐁
𝑡
⁢
Δ
⁢
𝜽
⁢
(
𝑡
)
		
(58)

where 
𝐁
∈
ℝ
𝑃
×
𝑃
 is the choice of curvature matrix. Such an approximation to the loss function is regarded as a Local Quadratic Approximation [LQA-auto-lr].

Consider a fixed-direction target update 
Δ
⁢
𝜽
 with an unknown step size 
𝜂
, the maximum achievable loss reduction based on Local Quadratic Approximation can be obtained as follows

	
−
1
2
⁢
𝛾
⁢
(
Δ
⁢
𝜽
)
2
=
min
𝜂
⁡
𝜂
⁢
Δ
⁢
𝜽
⊤
⁢
∇
𝜽
ℒ
⁢
(
𝜽
)
+
𝜂
2
2
⁢
Δ
⁢
𝜽
⊤
⁢
𝐅
⁢
Δ
⁢
𝜽
,
		
(59)

where 
𝐅
 is used in place of the curvature matrix 
𝐁
. This implies that the maximum loss reduction along update direction 
Δ
⁢
𝜽
 is 
−
1
/
2
⁢
𝛾
⁢
(
Δ
⁢
𝜽
)
2
, which is inversely proportional to the square of the proposed indicator. For an update with a smaller 
𝛾
⁢
(
Δ
⁢
𝜽
)
, it is expected to induce a larger loss reduction, with the exact NG update achieving the maximum loss reduction. Consequently, the proposed indicator has a strong correlation to practical convergence ability of the target update generation method.

Appendix GLimitations
Focus on Theoretical Approximation for NGD Method

The focus of the paper is on the theoretical approximation methods for NGD, including the exact EF method, the proposed iEF method, and also the SF method. Practical approximate (empirical) NGD optimisers are not the main focus of this paper (e.g. K-FAC [K-FAC], EK-FAC [EK-FAC], TNT [TNT] etc.), and no experiments are conducted for them. Despite the importance of these practical optimisers in the scene of NGD-based optimisation, they can all be considered a further (structure-aware) approximation to the EF or SF methods. However, it is an important future work to apply iEF to improve existing approximate NGD optimisers (an example of using iEF to improve empirical K-FAC is provided in Appendix D.2.2).

Limitations of Exact Update Generation

The updates generated by EF, iEF and SF methods (defined in Eqn. (5), (8), (49) respectively) investigated in this paper are “exact”, in order to provide a more theoretically rigorous comparison and analysis. In our experiments, they are generated based only on the current provided batch 
ℳ
⁢
(
𝑡
)
 as is described in Algorithms 2, 1, 3 respectively. The implementation of exact EF, iEF and SF updates requires storage of the 
𝑀
 per-sample gradients, which is memory-demanding in practice. This limits the scope of application of the exact iEF (and EF/SF) optimisers to setups where trainable parameter size is small, such as PEFT for pre-trained models. However, given the rise of PEFT setups, and consider the competitive optimisation performance of the exact iEF optimiser, such limitations may be out-weighted by the gain in practice. Additionally, the exact update formulation of iEF means momentum and weight decay cannot be directly applied with the resultant optimiser. This affects the optimisation performance on certain setups, and further work is required to integrate these key techniques with exact iEF optimiser. However, none of these limitations would affect the future work where iEF is incorporated into practical approximate NGD optimisers, where memory constraints and momentum/weight decay integration is already resolved in these practical NGD optimisers.

Appendix HDetailed Experiment Information
H.1Overall Experimental Setup
Textual Classification of GLUE

We have investigated 7 selected GLUE benchmark tasks [GLUE] including CoLA, SST-2, MRPC, QQP, MNLI, QNLI, and RTE, which together cover a range of NLP classification tasks such as sentiment classification, semantic equivalence checking, grammar checking, entailment prediction etc. The same selection of GLUE tasks is used in [PEFT]. One key aspect of GLUE tasks is that they do not have test labels, and for test evaluations, they have to be submitted to their official website (with a maximum allowed frequency of 3 submits per day). The textual label tokens are to be directly predicted by the target model. Also, an instruction is prepended to the input of each training sample. See details in Table. 2. For each task, a pre-trained T5-base model [T5] is trained with two parameter-efficient finetuning methods: Prompt Tuning [PT] and LoRA [lora].

Table 2:Training details for the 7 GLUE tasks. The input format column describes how the inputs are structured before being sent to the model. {S1}, {S2} represents the default input sentence(s) of the corresponding tasks, some tasks have only one input sentence and some have two. Labels are the tokens to be predicted by the model for each task.
Task	Input Format	Labels
CoLA	Classify if the following sentence’s grammar is
acceptable or unacceptable: {S1}	acceptable,
unacceptable
SST-2	Classify if the following sentence’s sentiment is
positive or negative: {S1}	positive, negative
QQP	Classify if the following Q1 and Q2 are semantically
equivalent, answer yes or no: Q1: {S1} Q2: {S2}	yes, no
MRPC	Classify if the following S1 and S2 are semantically
equivalent, answer yes or no: S1: {S1} S2: {S2}	yes, no
MNLI	Predict whether the premise entails the hypothesis,
contradicts the hypothesis,or neither, answer yes,
no or maybe: premise: {S1} hypothesis: {S2}	yes, no, maybe
QNLI	Determine whether the context sentence S contains
the answer to the question Q, answer yes or no:
Q: {S1} S: {S2}	yes, no
RTE	Classify if S1 entailment S2 or not, answer yes or no:
S1: {S1} S2: {S2}	yes, no
Computer Vision Classification with CIFAR100

The well-known CIFAR100 dataset [cifar] is used to finetune the pretrained ViT model [vit]. The model is finetuned with LoRA [lora] following the setup in [ViT-lora].

H.2Optimisation Experimental Setup

For Prompt Tuning, the Adafactor [prompt-tuning] optimiser is used as baseline (as is done in [prompt-tuning, PEFT]). For all other training setups, the AdamW [AdamW] optimiser was used as the baseline.

Different tasks were trained with a different number of epochs. The validation accuracy (on the dev set of each task) of the best checkpoint is reported and sent for test evaluation (for GLUE, the checkpoints are submitted to GLUE website [GLUE]). Details are shown in Table 3.

Table 3:Optimisation details all involved tasks. Train epochs represent the number of epochs for which the model is trained. Evaluation frequency describes the number of update steps between each validation evaluation on the development set.
Task	Train Epochs	Evaluation Frequency
CoLA	20	100
SST-2	20	1000
QQP	5	1000
MRPC	30	50
MNLI	5	1000
QNLI	15	1000
RTE	40	30
CIFAR100	5	1000

For all the GLUE tasks and for each finetuning method (Prompt Tuning or LoRA), a single set of hyper-parameters were searched for each optimiser and is used across runs on all 7 GLUE tasks. Three runs with different seeds were conducted to generate the error bar. For each run, the checkpoint with the highest validation accuracy was saved for later evaluation (to get test performance or to be evaluated in the proposed evaluation framework). All optimisers were trained on a batch size of 32.

Prompt Tuning + T5-base on GLUE: 20 prompts are used for Prompt Tuning. The trainable parameter size is 
15
,
360
, taking up 
0.0069
%
 of the total parameter size (222,918,912) of the T5-base model. Constant scheduling was used for Adafactor, SGD and iEF runs. For iEF, EF and SF, a damping of 
𝜆
=
1
×
10
−
12
 was used. For the Adafactor baseline optimiser, the hyper-parameter provided in [PEFT] was used (which comes from [prompt-tuning]): weight-decay 
1
×
10
−
5
, 
𝛽
2
=
0.8
, learning rate 
𝜂
=
0.3
 and no parameter scaling. For the SGD method, the learning rate is 
𝜂
=
100
, which was searched from 
{
0.1
,
1
,
10
,
20
,
50
,
100
}
. For the iEF method, the learning rate was 
𝜂
=
50
, which was searched from 
{
1
,
10
,
50
,
100
}
. For the EF method, a different scheduling of learning rate was used to guarantee convergence, due to the inverse scaling of EF updates. The chosen strategy was a linearly decaying normalised update, with the first update being normalised to 
1
 ({
1
×
10
−
3
, 
5
×
10
−
3
, 
1
×
10
−
2
, 
1
×
10
−
1
, 
1
, 
10
}) and linearly decaying to 0. The SF method was trained using the same method as EF with the same set of hyperparameters.

LoRA + T5-base on GLUE: The LoRA was set to have rank 8 with dropout 0.1 [ViT-lora]. The trainable parameter size is 
884
,
736
, taking up 
0.40
%
 of the total parameter size (222,918,912) of the T5-base model. Constant scheduling was used for the AdamW, SGD and iEF runs. For iEF, EF and SF, a damping of 
𝜆
=
1
×
10
−
7
 was used (it is 5 orders of magnitude larger than that used in Prompt Tuning because the diagonal of the gradient covariance matrix has a 5 order of magnitude larger norm in LoRA than Prompt Tuning). For the AdamW baseline optimiser, the hyper-parameters are: weight-decay 
1
×
10
−
2
, and the learning rate of 
1
×
10
−
3
 was searched from {
1
×
10
−
3
, 
5
×
10
−
4
, 
1
×
10
−
4
}. For the SGD method, the learning rate was 
𝜂
=
0.1
, which was searched from 
{
0.1
,
1
,
10
,
20
,
50
,
100
}
. For the iEF method, the learning rate was 
𝜂
=
100
, which was searched from 
{
1
,
10
,
50
,
100
}
. For the EF method, a normalised update with linear scheduling was used similar to Prompt Tuning, and a starting learning rate of 0.01 was searched from {
1
×
10
−
3
, 
5
×
10
−
3
, 
1
×
10
−
2
, 
1
×
10
−
1
, 
1
}. The SF method was trained with the same strategy and hyperparameters.

LoRA + ViT on CIFAR100 The setup of using LoRA to finetune CIFAR100 in [ViT-lora] was used. The trainable parameter size is 313,344, taking up 
0.36
%
 of the total parameter size (86,862,568) of the ViT model (vit-base-patch16-224). The same hyperparameters for LoRA + T5 was used for these experiments.

H.3Experiments Compute Resources

All the optimisation experiments and evaluation experiments are run on a cloud linux machine with 8 A100 GPUs with 80GB GRAM. For all the optimisation experiments, optimisation time with different optimisers are similar, apart from the SF optimiser where the additional back-propagation leads to an additional 60% runtime. For standard SGD/AdamW/Adafactor/EF/iEF optimisers, on average each complete run takes 10 hours. The slowest task (QQP + LoRA) takes 20 hours and the quickest task (RTE + Prompt Tuning) takes 0.3 hours. In total 420 GPU hours are run for all optimisation experiments. For evaluation runs, each evaluation is done on 100 batches of 160 training samples. For each checkpoint and a choice of damping, the evaluation takes on average 30 minutes. For all the evaluation runs (damping evaluated 5 tasks x 3 ckpts x 10 damping = 150 evaluations, standard evaluated 15 tasks x 7 ckpts = 115 evaluated, total 265 evaluation), 133 GPU hours are done. Considering hyper-parameter tuning, and various preliminary runs to make exact EF/iEF/SF optimisers and the evaluation framework to run properly, in total 2 times of additional compute time is required for all experiments.

H.4Licenses for All Used Assets

The license, citation and link to various asset used in this paper is provided in Table 4.

Table 4:The license, citation and link to various asset used in this paper.
Asset Name	Citation	URL	License
T5-base	[T5]	Model checkpoint	Apache-2.0
ViT-B/16	[vit]	Model checkpoint	Apache-2.0
GLUE	[GLUE]	Website	Multiple
CIFAR10/100	[cifar]	Website	Unknown
Pytorch	[pytorch]	Website	Multiple
H.5Further Evaluation Plots

Due to space limit and presentation concerns, the evaluation results presented in the main text in experiments (E1) and (E3) are only partial. In this section, additional evaluation results are presented.

H.5.1Approximation Quality across Tasks and Training Stages

The same indicator evaluation in Fig. 2 is done for all experimented tasks. A full view of the indicator plots are provided in Fig. 6 and 7. Similar trend can be found across tasks.

Figure 6:Four (log-scaled) ratios computed for checkpoints at various stages of training (sampled at the interval of one epoch) for all LoRA tasks, including T5 + 7 GLUE tasks and ViT + 1 CIFAR100. The figure is drawn in the same fashion as Fig. 2. Note that the error bar is not presented to improve presentation clarity.

Figure 7:Four (log-scaled) ratios computed for checkpoints at various stages of training (sampled at the interval of one epoch) for 7 Prompt Tuning tasks for T5 + GLUE. The figure is drawn in the same fashion as Fig. 2. Note that the error bar is not presented to improve presentation clarity.
H.5.2Impact of Damping on Approximation Quality

The same damping analysis in Fig. 3 is applied to several other tasks (SST2+T5+Prompt Tuning, MRPC+T5+Prompt Tuning, CIFAR100+ViT+LoRA, RTE+T5+LoRA,as shown in Fig. 8, 9, 10, 11 respectively). Similar trend can be found across tasks.

Figure 8:Approximation quality (relative to SGD) of EF, SF and iEF methods w.r.t. damping factor 
𝜆
 at different training stages of task SST2+T5+Prompt Tuning. The figure is drawn in the same fashion as Fig. 3.

Figure 9:Approximation quality (relative to SGD) of EF, SF and iEF methods w.r.t. damping factor 
𝜆
 at different training stages of task MRPC+T5+Prompt Tuning. The figure is drawn in the same fashion as Fig. 3.

Figure 10:Approximation quality (relative to SGD) of EF, SF and iEF methods w.r.t. damping factor 
𝜆
 at different training stages of task CIFAR100+ViT+LoRA. The figure is drawn in the same fashion as Fig. 3.

Figure 11:Approximation quality (relative to SGD) of EF, SF and iEF methods w.r.t. damping factor 
𝜆
 at different training stages of task RTE+T5+LoRA. The figure is drawn in the same fashion as Fig. 3.
H.6Full Optimisation Results

Due to space limit, only a partial test performance table (Table 1) is presented for the experiment (E2) in the main text. In this section, the full training, validation and test results for all combinations of structure, tasks and optimisers are reported for the reader’s information.

Training curves and validation accuracy curves of different optimisers for the described 15 setups are presented in Fig. 12 and 13. The final training loss for every task, structure and optimiser combination is presented in Table 5. The validation performance for every task, structure and optimiser combination is presented in Table 6. The test performance for all tasks is presented in Table 7.

Table 5:Final train loss for all the task, structure and optimiser combinations. The average train loss of the final 100 steps of 3 random seed runs is reported, along with the standard deviation. IEF achieves the lowest final training loss for 12 out of 15 tasks.
	CoLA	SST-2	MRPC	QQP	MNLI	QNLI	RTE	CIFAR100
Prompt Tuning
Adafactor	
0.07
±
0.08
	
0.07
±
0.02
	
0.16
±
0.07
	
0.02
±
0.02
	
0.19
±
0.02
	
0.10
±
0.05
	
0.29
±
0.03
	-
SGD	
0.33
±
0.06
	
0.04
±
0.02
	
0.36
±
0.11
	
0.15
±
0.13
	
0.31
±
0.05
	
0.15
±
0.06
	
0.35
±
0.03
	-
EF	
2.28
±
1.11
	
0.99
±
0.35
	
2.80
±
1.19
	
4.29
±
1.04
	
6.11
±
1.10
	
5.12
±
0.88
	
1.79
±
0.16
	-
SF	
0.16
±
0.10
	
0.06
±
0.11
	
0.43
±
0.09
	
0.15
±
0.02
	
0.28
±
0.02
	
0.27
±
0.14
	
0.36
±
0.03
	-
iEF	
0.01
±
0.01
	
0.02
±
0.01
	
0.04
±
0.01
	
0.02
±
0.01
	
0.17
±
0.07
	
0.09
±
0.06
	
0.02
±
0.01
	-
LoRA
AdamW	
0.10
±
0.03
	
0.02
±
0.01
	
0.07
±
0.06
	
0.07
±
0.01
	
0.22
±
0.06
	
0.10
±
0.03
	
0.02
±
0.03
	
0.60
±
0.21

SGD	
0.24
±
0.02
	
0.03
±
0.09
	
0.12
±
0.01
	
0.10
±
0.05
	
0.22
±
0.12
	
0.11
±
0.07
	
0.18
±
0.07
	
0.94
±
0.34

EF	
1.92
±
0.49
	
0.90
±
0.03
	
2.541
±
0.78
	
1.03
±
0.51
	
3.42
±
0.33
	
1.31
±
0.89
	
3.08
±
0.40
	
3.10
±
0.11

SF	
0.26
±
0.03
	
0.14
±
0.22
	
0.35
±
0.11
	
0.20
±
0.12
	
0.24
±
0.09
	
0.30
±
0.21
	
0.39
±
0.02
	
0.63
±
0.12

iEF	
0.16
±
0.04
	
0.02
±
0.02
	
0.06
±
0.04
	
0.07
±
0.05
	
0.36
±
0.05
	
0.08
±
0.06
	
0.05
±
0.02
	
0.59
±
0.30
Table 6:Validation metrics (multiplied by 100) (on development set) for all the task, structure and optimiser combinations. The average of the highest validation accuracy of 3 random seed runs is reported, along with the standard deviation. Task-specific metrics are reported in this table. For SST-2, QNLI, RTE and CIFAR100, accuracy is reported. For CoLA, both accuracy and Matthew’s Corr is reported. For MRPC and QQP, F1-score and Accuracy (in order) are reported. For MNLI, average of accuracy for entire development set is reported.
	CoLA	SST-2	MRPC	QQP	MNLI	QNLI	RTE	CIFAR100
Prompt Tuning
Adafactor	
82.0
±
0.44


53.8
±
1.01
	
94.2
±
0.25
	
84.8
±
0.35


88.0
±
0.73
	
90.7
±
0.01


87.7
±
0.01
	
82.4
±
0.25
	
91.9
±
0.09
	
64.7
±
0.34
	-
SGD	
69.1
±
0.09


−
0.1
±
2.76
	
93.3
±
0.14
	
70.0
±
0.31


80.8
±
0.14
	
85.0
±
4.11


80.6
±
5.03
	
74.5
±
1.17
	
88.7
±
0.20
	
54.8
±
0.34
	-
EF	
69.0
±
0.01


−
0.55
±
0.58
	
90.3
±
0.23
	
68.4
±
0.01


81.2
±
0.01
	
62.2
±
0.23


0.11
±
0.07
	
32.9
±
0.09
	
50.9
±
0.59
	
55.6
±
1.30
	-
SF	
76.1
±
2.27


39.4
±
7.71
	
93.5
±
0.37
	
70.3
±
1.23


81.8
±
0.65
	
90.3
±
0.03


87.2
±
0.08
	
77.1
±
0.93
	
64.2
±
0.65
	
57.2
±
0.75
	-
iEF	
81.7
±
0.16


50.7
±
1.25
	
94.4
±
0.09
	
86.2
±
1.41


89.5
±
0.46
	
90.7
±
0.02


87.7
±
0.05
	
83.4
±
0.09
	
91.9
±
0.03
	
74.6
±
0.85
	-
LoRA
AdamW	
83.1
±
0.15


58.7
±
0.55
	
94.9
±
0.07
	
88.6
±
0.51


91.9
±
0.26
	
90.0
±
0.16


86.8
±
0.06
	
83.2
±
0.03
	
92.2
±
0.06
	
83.4
±
1.06
	
93.7
±
0.32

SGD	
81.3
±
0.36


53.6
±
0.97
	
95.0
±
0.18
	
87.3
±
0.28


90.1
±
0.41
	
89.9
±
0.69


86.7
±
0.88
	
83.3
±
0.05
	
92.3
±
0.09
	
80.9
±
0.72
	
90.6
±
1.02

EF	
69.1
±
0.01


10.5
±
2.10
	
91.9
±
0.07
	
55.5
±
0.28


71.4
±
0.23
	
86.3
±
0.28


82.6
±
0.32
	
61.5
±
0.04
	
89.4
±
0.21
	
52.7
±
0.01
	
30.2
±
1.20

SF	
79.4
±
0.49


48.3
±
1.45
	
94.5
±
0.29
	
76.9
±
0.28


85.2
±
0.30
	
90.1
±
0.11


86.9
±
0.10
	
81.9
±
0.17
	
91.8
±
0.32
	
71.8
±
0.96
	
92.7
±
0.72

iEF	
83.4
±
0.24


59.5
±
0.64
	
94.9
±
0.21
	
88.5
±
0.88


91.8
±
0.55
	
89.9
±
0.09


86.6
±
0.12
	
81.2
±
0.02
	
92.2
±
0.11
	
81.7
±
0.55
	
94.1
±
0.15
Table 7:Test performance for all task, structure and optimiser combinations. For all tasks, only one test result is reported for the best validation checkpoint across three random seed runs. Task-specific metrics (all multiplied by 100) on the test set are reported in this table. For SST-2, QNLI, RTE and CIFAR100, accuracy is reported. For CoLA, Matthew’s Corr is reported. For MRPC and QQP, F1-score and Accuracy (in order) are reported. For MNLI, matched accuracy and unmatched Accuracy (in order) are reported.
	CoLA	SST-2	MRPC	QQP	MNLI	QNLI	RTE	CIFAR100
Prompt Tuning
Adafactor	45.1	94.3	87.1
82.8	71.8
88.8	82.9
82.7	91.5	60.7	-
SGD	6.4	93.5	78.3
66.6	71.3
88.9	75.7
76.5	87.4	55.3	-
EF	
−
3.8
	
90.2
	
79.9


66.5
	
0.3


81.6
	
33.4


33.4
	
52.4
	
50.4
	-
SF	
45.2
	
93.7
	79.7
67.8	71.5
88.5	77.8
78.5	
64.1
	
52.3
	-
iEF	50.9	94.4	88.4
84.2	72.0
89.2	83.5
83.4	91.3	68.2	-
LoRA
AdamW	
52.2
	
94.5
	
88.6


85.1
	
71.5


88.8
	
83.6


83.1
	
92.2
	
71.2
	
93.9

SGD	
47.8
	
94.0
	
79.9


66.6
	
71.6


88.8
	
83.5


83.8
	
91.9
	
70.1
	
91.3

EF	
0.0
	
92.1
	
79.9


66.5
	
65.4


84.4
	
61.4


62.7
	
89.4
	
50.4
	
31.0

SF	
42.2
	
94.2
	
84.3


75.8
	
71.1


88.4
	
82.1


82.3
	
91.8
	
64.9
	
92.8

iEF	
51.2
	
94.4
	
89.3


85.9
	
71.5


88.9
	
81.1


80.8
	
92.2
	
69.1
	
94.3
Figure 12:Training loss and validation accuracy of all the Prompt Tuning tasks (7 GLUE tasks). Validation accuracy is reported at the same frequency as is stated by Table 3. For training loss, 100 points are reported across all training stages for every task. Each train loss data point represents the averaged train loss for all train batches between each reported data point. The error bars represent the standard deviation (1-sigma) for 3 random seed runs. The training loss for EF always starts to diverge halfway through training despite the more complicated scheduling. For most tasks, iEF is able to reach a lower training loss than EF, SF and the well-tuned baseline Adafactor.
Figure 13:Training loss and validation accuracy of all the LoRA tasks (7 GLUE tasks and 1 CIFAR100). Validation accuracy is reported at the same frequency as is stated by Table 3. For training loss, 100 points are reported across all training stages for every task. Each train loss data point represents the averaged train loss for all train batches between each reported data point. The error bars represent the standard deviation (1-sigma) for 3 random seed runs. The training loss for EF always starts to diverge halfway through training despite the more complicated scheduling. For most tasks, iEF is able to reach a lower training loss than EF, SF and the well-tuned baseline AdamW.
H.7Additional Train-from-Scratch Experiment

All of the experiments presented in Sec. 7 are conducted under PEFT setups. While this allows our experimental results to consider up-to-date model structures and tasks, an additional experiment with a larger trainable parameter size (currently less than 1M parameters) and consider non-finetuning setup would be beneficial to further validating the statement of this paper. In this section, a train-from-scratch experiment using a 10M parameter MLP model on the CIFAR10 dataset is conducted and analysed.

Optimisation Setup

The used model structure is a 2-hidden-layer ReLU-activated MLP model with a parameter size of 10,510,346 (
∼
10M), which takes in a flattened 3x32x32 image, and has two 2048 hidden layers. This model structure is an extension to the setups used in [over-param, adagrad-logregret]. The model is trained for 60 epochs from scratch on the CIFAR10 dataset. During optimisation, no weight decay or dropout is applied. The Adam, SGD, EF, SF and iEF optimisers are used, and for all optimisers, 60 epochs are run with a batch size of 64. The learning rate 
1
×
10
−
4
 of Adam is searched from {
5
×
10
−
5
, 
1
×
10
−
4
, 
5
×
10
−
4
}. The learning rate 0.1 of the SGD is searched from 0.01, 0.1, 0.5. The learning rate 50 of iEF is searched from 10, 50, 100. The learning rate 0.1 of SF is searched from {0.01, 0.1, 0.5}. The learning rate 
1
×
10
−
4
 of EF is searched from {
1
×
10
−
5
, 
1
×
10
−
4
, 
1
×
10
−
3
, 
1
×
10
−
2
, 
1
×
10
−
1
}. Normalised update with a linear scheduler is used for EF and SF as in the paper. A constant learning rate is used for iEF. A multi-step scheduler (0.1 decay at fixed epoch number 15 [resnet32]) is used for Adam and SGD. For each optimiser run, 3 seeds are used to generate the error-bars.

Experimental Results

Following experiment (E2), the training loss curve and validation accuracy curve is plotted in Fig. 14. The validation and test accuracy is reported in Table 8. Following experiment (E3), the effect of damping on the approximation quality across different stages of training is shown in Fig. 15. Note that we did not conduct a corresponding experiment for (E1) because we believe most of the relevant information is included in Fig. 15. The experimental results show that the conclusions drawn by the PEFT experiment also hold for large train-from-scratch experiments. The exact iEF method demonstrate comparative/stronger model optimisation performance to Adam/SGD baselines, and it remains to be the strongest and most robust method when compared against EF/SF in terms of approximation quality to exact NG updates. Meanwhile, EF shows consistently terrible approximation quality, and struggles to optimise the model, when not carefully damped.

Table 8:Validation and Test accuracy of different optimiser runs for MLP+CIFAR10 setup. For each optimiser run, only one test accuracy is evaluated for the checkpoint with the best validation accuracy.
	iEF	Adam	SGD	SF	EF
Validation Accuracy (%)	
58.8
±
0.87
	
56.3
±
0.22
	
54.3
±
0.66
	
54.8
±
0.08
	
28.2
±
1.43

Test Accuracy (%)	
58.6
	
56.6
	
54.4
	
55.2
	
29.2
Figure 14:Training loss and validation accuracy curves for the MLP + CIFAR10 train-from-scratch setup. The style of the figure follows that of Fig. 12 and 13. The optimisation performance follows EF < SGD 
≈
 SF < Adam < iEF, which overall matches the result for the PEFT experiments. Note that, eventually, iEF achieves both the highest validation accuracy and the lowest training loss with a constant learning rate, while Adam and SGD require a multi-step scheduler to perform well.

Figure 15:Approximation quality (relative to SGD) of EF, SF and iEF methods w.r.t. damping factor 
𝜆
 at different training stages of setup task CIFAR10+MLP. The visualisation style and the experimental setup follows that of Fig. 3 (E3). This figure demonstrates that the better approximation quality and the robustness to damping of iEF also hold for a larger train-from-scratch image classification task at different training stages.
Report Issue
Report Issue for Selection
Generated by L A T E xml 
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button.
Open a report feedback form via keyboard, use "Ctrl + ?".
Make a text selection and click the "Report Issue for Selection" button near your cursor.
You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off accessible reporting links at each section.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.
