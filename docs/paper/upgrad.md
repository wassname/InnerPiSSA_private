Title: Jacobian Descent for Multi-Objective Optimization

URL Source: https://arxiv.org/html/2406.16232

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
2Theoretical foundation
3Applications
4Existing aggregators
5Experiments
6Gramian-based Jacobian descent
7Conclusion
 References
License: CC BY 4.0
arXiv:2406.16232v3 [cs.LG] 03 Feb 2025
Jacobian Descent for Multi-Objective Optimization
Pierre Quinton ∗
LTHI, EPFL pierre.quinton@epfl.ch
&Valérian Rey ∗
Independent valerian.rey@gmail.com

Abstract

Many optimization problems require balancing multiple conflicting objectives. As gradient descent is limited to single-objective optimization, we introduce its direct generalization: Jacobian descent (JD). This algorithm iteratively updates parameters using the Jacobian matrix of a vector-valued objective function, in which each row is the gradient of an individual objective. While several methods to combine gradients already exist in the literature, they are generally hindered when the objectives conflict. In contrast, we propose projecting gradients to fully resolve conflict while ensuring that they preserve an influence proportional to their norm. We prove significantly stronger convergence guarantees with this approach, supported by our empirical results. Our method also enables instance-wise risk minimization (IWRM), a novel learning paradigm in which the loss of each training example is considered a separate objective. Applied to simple image classification tasks, IWRM exhibits promising results compared to the direct minimization of the average loss. Additionally, we outline an efficient implementation of JD using the Gramian of the Jacobian matrix to reduce time and memory requirements.

*
1Introduction

The field of multi-objective optimization studies minimization of vector-valued objective functions (Sawaragi et al., 1985; Ehrgott, 2005; Branke, 2008; Deb et al., 2016). In deep learning, a widespread approach to train a model with multiple objectives is to combine those into a scalar loss function minimized by stochastic gradient descent. While this method is simple, it comes at the expense of potentially degrading some individual objectives. Without prior knowledge of their relative importance, this is undesirable. In opposition, multi-objective optimization methods typically attempt to optimize all objectives simultaneously, without making arbitrary compromises: the goal is to find points for which no improvement can be made on some objectives without degrading others.

Early works have attempted to extend gradient descent (GD) to consider several objectives simultaneously, and thus several gradients (Fliege & Svaiter, 2000; Désidéri, 2012). Essentially, they propose a heuristic to prevent the degradation of any individual objective. Several other works have built upon this method, analyzing its convergence properties or extending it to a stochastic setting (Poirion et al., 2017; Mercier et al., 2018; Fliege et al., 2019). Later, this has been applied to multi-task learning to tackle conflict between tasks, illustrated by contradicting gradient directions (Sener & Koltun, 2018). Many studies have followed, proposing various other algorithms for the training of multi-task models (Yu et al., 2020; Chen et al., 2020; Liu et al., 2021a, b; Lin et al., 2021; Navon et al., 2022; Senushkin et al., 2023). They commonly rely on an aggregator that maps a collection of task-specific gradients (a Jacobian matrix) to a shared parameter update.

We propose to unify all such methods under the Jacobian descent (JD) algorithm, specified by an aggregator.1 This algorithm aims to minimize a differentiable vector-valued function 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 iteratively without relying on a scalarization of the objective. Under this formulation, the existing methods are simply distinguished by their aggregator. Consequently, studying its properties is essential for understanding the behavior and convergence of JD. Under significant conflict, existing aggregators often fail to provide strong convergence guarantees. To address this, we propose 
𝒜
UPGrad
, specifically designed to resolve conflicts while naturally preserving the relative influence of individual gradients.

Furthermore, we introduce a novel stochastic variant of JD that enables the training of neural networks with a large number of objectives. This unlocks a particularly interesting perspective: considering the minimization of instance-wise loss vectors rather than the usual minimization of the average training loss. As this paradigm is a direct generalization of the well-known empirical risk minimization (ERM) (Vapnik, 1995), we name it instance-wise risk minimization (IWRM).

Our contributions are organized as follows: In Section 2, we formalize the JD algorithm and its stochastic variants. We then introduce three important aggregator properties and define 
𝒜
UPGrad
 to satisfy them. In the smooth convex case, we show convergence of JD with 
𝒜
UPGrad
 to the Pareto front. We present applications for JD and aggregators in Section 3, emphasizing the IWRM paradigm. We then discuss existing aggregators and analyze their properties in Section 4. In Section 5, we report experiments with IWRM optimized with stochastic JD with various aggregators. Lastly, we address computational efficiency in Section 6, giving a path towards an efficient implementation.

2Theoretical foundation

A suitable partial order between vectors must be considered to enable multi-objective optimization. Throughout this paper, denotes the relation defined for any pair of vectors 
𝒖
,
𝒗
∈
ℝ
𝑚
 as 
𝒖
⁢
𝒗
 whenever 
𝑢
𝑖
⁢
𝑣
𝑖
 for all coordinates 
𝑖
. Similarly, is the relation defined by 
𝒖
⁢
𝒗
 whenever 
𝑢
𝑖
⁢
<
⁢
𝑣
𝑖
 for all coordinates 
𝑖
. Furthermore, 
𝒖
⁢
𝒗
 indicates that both 
𝒖
⁢
𝒗
 and 
𝒖
≠
𝒗
 hold. The Euclidean vector norm and the Frobenius matrix norm are denoted by 
∥
⋅
∥
 and 
∥
⋅
∥
F
, respectively. Finally, for any 
𝑚
∈
ℕ
, the symbol 
[
𝑚
]
 represents the range 
{
𝑖
∈
ℕ
:
1
⁢
𝑖
⁢
𝑚
}
.

2.1Jacobian descent

In the following, we introduce Jacobian descent, a natural extension of gradient descent supporting the optimization of vector-valued functions.

Suppose that 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 is continuously differentiable. Let 
𝒥
⁢
𝒇
⁢
(
𝒙
)
∈
ℝ
𝑚
×
𝑛
 be the Jacobian matrix of 
𝒇
 at 
𝒙
, i.e.

	
𝒥
⁢
𝒇
⁢
(
𝒙
)
=
[
∇
𝑓
1
⁢
(
𝒙
)
⊤


∇
𝑓
2
⁢
(
𝒙
)
⊤


⋮


∇
𝑓
𝑚
⁢
(
𝒙
)
⊤
]
=
[
∂
∂
𝑥
1
⁢
𝑓
1
⁢
(
𝒙
)
	
∂
∂
𝑥
2
⁢
𝑓
1
⁢
(
𝒙
)
	
⋯
	
∂
∂
𝑥
𝑛
⁢
𝑓
1
⁢
(
𝒙
)


∂
∂
𝑥
1
⁢
𝑓
2
⁢
(
𝒙
)
	
∂
∂
𝑥
2
⁢
𝑓
2
⁢
(
𝒙
)
	
⋯
	
∂
∂
𝑥
𝑛
⁢
𝑓
2
⁢
(
𝒙
)


⋮
	
⋮
	
⋱
	
⋮


∂
∂
𝑥
1
⁢
𝑓
𝑚
⁢
(
𝒙
)
	
∂
∂
𝑥
2
⁢
𝑓
𝑚
⁢
(
𝒙
)
	
⋯
	
∂
∂
𝑥
𝑛
⁢
𝑓
𝑚
⁢
(
𝒙
)
]
		
(1)

Given 
𝒙
,
𝒚
∈
ℝ
𝑛
, Taylor’s theorem yields

	
𝒇
⁢
(
𝒙
+
𝒚
)
=
𝒇
⁢
(
𝒙
)
+
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
+
𝑜
⁢
(
‖
𝒚
‖
)
,
		
(2)

The term 
𝒇
⁢
(
𝒙
)
+
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
 is the first-order Taylor approximation of 
𝒇
⁢
(
𝒙
+
𝒚
)
. Via this approximation, we aim to select a small update 
𝒚
 that reduces 
𝒇
⁢
(
𝒙
+
𝒚
)
, ideally achieving 
𝒇
⁢
(
𝒙
+
𝒚
)
⁢
𝒇
⁢
(
𝒙
)
. As the approximation depends on 
𝒚
 only through 
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
, selecting the update based on the Jacobian is natural. A mapping 
𝒜
:
ℝ
𝑚
×
𝑛
→
ℝ
𝑛
 reducing such a matrix into a vector is called an aggregator. For any 
𝐽
∈
ℝ
𝑚
×
𝑛
, 
𝒜
⁢
(
𝐽
)
 is called the aggregation of 
𝐽
 by 
𝒜
.

To minimize 
𝒇
, consider the update 
𝒚
=
−
𝜂
⁢
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
, where 
𝜂
 is an appropriate step size, and 
𝒜
 is an appropriate aggregator. Jacobian descent simply consists in applying this update iteratively, as shown in Algorithm 1. To put it into perspective, we also provide a minimal version of GD in Algorithm 2. Remarkably, when 
𝑚
=
1
, the Jacobian has a single row, so GD is a special case of JD where the aggregator is the identity.

Input: 
𝒙
∈
ℝ
𝑛
, 
0
⁢
<
⁢
𝜂
, 
𝑇
∈
ℕ
, 
𝒜
:
ℝ
𝑚
×
𝑛
→
ℝ
𝑛
for 
𝑡
←
1
 to 
𝑇
 do
       
𝒙
←
𝒙
−
𝜂
⁢
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
      
Output: 
𝒙
Algorithm 1 Jacobian descent with aggregator 
𝒜
 
Input: 
𝒙
∈
ℝ
𝑛
, 
0
⁢
<
⁢
𝜂
, 
𝑇
∈
ℕ
for 
𝑡
←
1
 to 
𝑇
 do
       
𝒙
←
𝒙
−
𝜂
⁢
∇
𝑓
⁢
(
𝒙
)
      
Output: 
𝒙
Algorithm 2 Gradient descent

Note that other gradient-based optimization algorithms, e.g. Adam (Kingma & Ba, 2014), can similarly be extended to the multi-objective case.

In some settings, the exact computation of the update can be prohibitively slow or even intractable. When dealing with a single objective, stochastic gradient descent (SGD) replaces the gradient 
∇
𝑓
⁢
(
𝒙
)
 with some estimation. More generally, stochastic Jacobian descent (SJD) relies on estimates of the aggregation of the Jacobian. One approach, that we call stochastically estimated Jacobian descent (SEJD), is to compute and aggregate an estimation of the Jacobian. Alternatively, when the number of objectives is very large, we propose to aggregate a matrix whose rows are a random subset of the rows of the true Jacobian. We call this approach stochastic sub-Jacobian descent (SSJD).

2.2Desirable properties for aggregators

An inherent challenge of multi-objective optimization is to manage conflicting objectives (Sener & Koltun, 2018; Yu et al., 2020; Liu et al., 2021a). Substituting the update 
𝒚
=
−
𝜂
⁢
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
 into the first-order Taylor approximation 
𝒇
⁢
(
𝒙
)
+
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
 yields 
𝒇
⁢
(
𝒙
)
−
𝜂
⁢
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
. In particular, if 
𝟎
⁢
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
, then no coordinate of the approximation of 
𝒇
 will increase. A pair of vectors 
𝒙
,
𝒚
∈
ℝ
𝑛
 is said to conflict if 
𝒙
⊤
⁢
𝒚
⁢
<
⁢
0
. Hence, for a sufficiently small 
𝜂
, if any row of 
𝒥
⁢
𝒇
⁢
(
𝒙
)
 conflicts with 
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
, the corresponding coordinate of 
𝒇
 will increase. When minimizing 
𝒇
, avoiding conflict between the aggregation and any gradient is thus desirable, motivating the first property.

Definition 1 (Non-conflicting).

Let 
𝒜
:
ℝ
𝑚
×
𝑛
→
ℝ
𝑛
 be an aggregator. If for all 
𝐽
∈
ℝ
𝑚
×
𝑛
, 
𝟎
⁢
𝐽
⋅
𝒜
⁢
(
𝐽
)
, then 
𝒜
 is said to be non-conflicting.

For any collection of vectors 
𝐶
⊆
ℝ
𝑛
, the dual cone of 
𝐶
 is 
{
𝒙
∈
ℝ
𝑛
:
∀
𝒚
∈
𝐶
,
0
⁢
𝒙
⊤
⁢
𝒚
}
 (Boyd & Vandenberghe, 2004). Notice that an aggregator 
𝒜
 is non-conflicting if and only if for any 
𝐽
, 
𝒜
⁢
(
𝐽
)
 is in the dual cone of the rows of 
𝐽
.

In a step of GD, the update scales proportionally to the gradient norm. Small gradients thus lead to small updates, and conversely, large gradients lead to large updates. To maintain coherence with GD, it would be natural that the rows of the Jacobian also contribute to the aggregation proportionally to their norm. Scaling each row of 
𝒥
⁢
𝒇
⁢
(
𝒙
)
 by the corresponding element of some vector 
𝒄
∈
ℝ
𝑚
 yields 
diag
(
𝒄
)
⋅
𝒥
⁢
𝒇
⁢
(
𝒙
)
. This insight can then be formalized as the following property.

Definition 2 (Linear under scaling).

Let 
𝒜
:
ℝ
𝑚
×
𝑛
→
ℝ
𝑛
 be an aggregator. If for all 
𝐽
∈
ℝ
𝑚
×
𝑛
, the mapping from any 
𝟎
⁢
𝒄
∈
ℝ
𝑚
 to 
𝒜
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
 is linear in 
𝒄
, then 
𝒜
 is said to be linear under scaling.

Finally, as 
‖
𝒚
‖
 decreases asymptotically to 
0
, the precision of the first-order Taylor approximation 
𝒇
⁢
(
𝒙
)
+
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
 improves, as highlighted in (2). The projection 
𝒚
′
 of any candidate update 
𝒚
 onto the span of the rows of 
𝒥
⁢
𝒇
⁢
(
𝒙
)
 satisfies 
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
′
=
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒚
 and 
‖
𝒚
′
‖
⁢
‖
𝒚
‖
, so this projection decreases the norm of the update while preserving the value of the approximation. Without additional information about 
𝒇
, it is thus reasonable to select 
𝒚
 directly in the row span of 
𝒥
⁢
𝒇
⁢
(
𝒙
)
, i.e. to have a vector of weights 
𝒘
∈
ℝ
𝑚
 satisfying 
𝒚
=
𝒥
⁢
𝒇
⁢
(
𝒙
)
⊤
⋅
𝒘
. This yields the last desirable property.

Definition 3 (Weighted).

Let 
𝒜
:
ℝ
𝑚
×
𝑛
→
ℝ
𝑛
 be an aggregator. If for all 
𝐽
∈
ℝ
𝑚
×
𝑛
, there exists 
𝒘
∈
ℝ
𝑚
 satisfying 
𝒜
⁢
(
𝐽
)
=
𝐽
⊤
⋅
𝒘
, then 
𝒜
 is said to be weighted.

2.3Unconflicting projection of gradients

We now define the unconflicting projection of gradients aggregator 
𝒜
UPGrad
, specifically designed to be non-conflicting, linear under scaling, and weighted. In essence, it projects each gradient onto the dual cone of the rows of the Jacobian and averages the results, as illustrated in Figure 1(a).

(a)
𝒜
UPGrad
⁢
(
𝐽
)
 (ours)
(b)
𝒜
Mean
⁢
(
𝐽
)
, 
𝒜
MGDA
⁢
(
𝐽
)
 and 
𝒜
DualProj
⁢
(
𝐽
)
Figure 1:Aggregation of 
𝐽
=
[
𝒈
1
⁢
𝒈
2
]
⊤
∈
ℝ
2
×
2
 by four different aggregators. The dual cone of 
{
𝒈
1
,
𝒈
2
}
 is represented in green.
(a) 
𝒜
UPGrad
 projects 
𝒈
1
 and 
𝒈
2
 onto the dual cone and averages the results.
(b) The mean 
𝒜
Mean
⁢
(
𝐽
)
=
1
2
⁢
(
𝒈
1
+
𝒈
2
)
 conflicts with 
𝒈
1
. 
𝒜
DualProj
 projects this mean onto the dual cone, so it lies on its boundary. 
𝒜
MGDA
⁢
(
𝐽
)
 is almost orthogonal to 
𝒈
2
 because of its larger norm.

For any 
𝐽
∈
ℝ
𝑚
×
𝑛
 and 
𝒙
∈
ℝ
𝑛
, the projection of 
𝒙
 onto the dual cone of the rows of 
𝐽
 is

	
𝜋
𝐽
⁢
(
𝒙
)
=
arg
⁢
min
𝒚
∈
ℝ
𝑛
:
𝟎
⁢
𝐽
⁢
𝒚
⁡
‖
𝒚
−
𝒙
‖
2
.
		
(3)

Denoting by 
𝒆
𝑖
∈
ℝ
𝑚
 the 
𝑖
th standard basis vector, 
𝐽
⊤
⁢
𝒆
𝑖
 is the 
𝑖
th row of 
𝐽
. 
𝒜
UPGrad
 is defined as

	
𝒜
UPGrad
⁢
(
𝐽
)
=
1
𝑚
⁢
∑
𝑖
∈
[
𝑚
]
𝜋
𝐽
⁢
(
𝐽
⊤
⁢
𝒆
𝑖
)
.
		
(4)

Since the dual cone is convex, it is closed under positive combinations of its elements. For any 
𝐽
, 
𝒜
UPGrad
⁢
(
𝐽
)
 is thus always in the dual cone of the rows of 
𝐽
, so 
𝒜
UPGrad
 is non-conflicting. Note that if no pair of gradients conflicts, 
𝒜
UPGrad
 simply averages the rows of the Jacobian.

Since 
𝜋
𝐽
 is a projection onto a closed convex cone, if 
𝒙
∈
ℝ
𝑛
 and 
0
⁢
<
⁢
𝑎
∈
ℝ
, then 
𝜋
𝐽
⁢
(
𝑎
⋅
𝒙
)
=
𝑎
⋅
𝜋
𝐽
⁢
(
𝒙
)
. By (4), 
𝒜
UPGrad
 is thus linear under scaling.

When 
𝑛
 is large, the projection in (3) is prohibitively expensive to compute. An alternative but equivalent approach is to use its dual formulation, which is independent of 
𝑛
.

Proposition 1.

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
. For any 
𝒖
∈
ℝ
𝑚
, 
𝜋
𝐽
⁢
(
𝐽
⊤
⁢
𝒖
)
=
𝐽
⊤
⁢
𝒘
 with

	
𝒘
∈
arg
⁢
min
𝒗
∈
ℝ
𝑚
:
𝒖
⁢
𝒗
⁡
𝒗
⊤
⁢
𝐽
⁢
𝐽
⊤
⁢
𝒗
.
		
(5)
Proof.

See Appendix B.2. ∎

The problem defined in (5) can be solved efficiently using a quadratic programming solver, such as those bundled in qpsolvers (Caron et al., 2024). For any 
𝑖
∈
[
𝑚
]
, let 
𝒘
𝑖
 be given by (5) when substituting 
𝒖
 with 
𝒆
𝑖
. Then, by Proposition 1,

	
𝒜
UPGrad
⁢
(
𝐽
)
=
𝐽
⊤
⁢
(
1
𝑚
⁢
∑
𝑖
∈
[
𝑚
]
𝒘
𝑖
)
.
		
(6)

This provides an efficient implementation of 
𝒜
UPGrad
 and proves that it is weighted. 
𝒜
UPGrad
 can also be easily extended to incorporate a vector of preferences by replacing the average in (4) and (6) by a weighted sum with positive weights. This extension remains non-conflicting, linear under scaling, and weighted.

2.4Convergence to the Pareto front

We now provide theoretical convergence guarantees of JD with 
𝒜
UPGrad
 when minimizing some 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 satisfying standard assumptions. If for a given 
𝒙
∈
ℝ
𝑛
, there exists no 
𝒚
∈
ℝ
𝑛
 satisfying 
𝒇
⁢
(
𝒚
)
⁢
𝒇
⁢
(
𝒙
)
, then 
𝒙
 is said to be Pareto optimal. The set 
𝑋
∗
⊆
ℝ
𝑛
 of Pareto optimal points is called the Pareto set, and its image 
𝒇
⁢
(
𝑋
∗
)
 is called the Pareto front.

Whenever 
𝒇
⁢
(
𝜆
⁢
𝒙
+
(
1
−
𝜆
)
⁢
𝒚
)
⁢
𝜆
⁢
𝒇
⁢
(
𝒙
)
+
(
1
−
𝜆
)
⁢
𝒇
⁢
(
𝒚
)
 holds for any pair of vectors 
𝒙
,
𝒚
∈
ℝ
𝑛
 and any 
𝜆
∈
[
0
,
1
]
, 
𝒇
 is said to be -convex. Moreover, 
𝒇
 is said to be 
𝛽
-smooth whenever 
‖
𝒥
⁢
𝒇
⁢
(
𝒙
)
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
‖
F
⁢
𝛽
⁢
‖
𝒙
−
𝒚
‖
 holds for any pair of vectors 
𝒙
,
𝒚
∈
ℝ
𝑛
.

Theorem 1.

Let 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 be a 
𝛽
-smooth and -convex function. Suppose that the Pareto front 
𝒇
⁢
(
𝑋
∗
)
 is bounded and that for any 
𝒙
∈
ℝ
𝑛
, there is 
𝒙
∗
∈
𝑋
∗
 satisfying 
𝒇
⁢
(
𝒙
∗
)
⁢
𝒇
⁢
(
𝒙
)
.2 Let 
𝒙
1
∈
ℝ
𝑛
, and for all 
𝑡
⁢
1
, 
𝒙
𝑡
+
1
=
𝒙
𝑡
−
𝜂
⁢
𝒜
UPGrad
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
)
, with 
𝜂
=
1
𝛽
⁢
𝑚
. Let 
𝒘
𝑡
 be the weights defining 
𝒜
UPGrad
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
)
 as per (6), i.e. 
𝒜
UPGrad
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
)
=
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
⊤
⋅
𝒘
𝑡
. If 
𝒘
𝑡
 is bounded, then 
𝒇
⁢
(
𝒙
𝑡
)
 converges to 
𝒇
⁢
(
𝒙
∗
)
 for some 
𝒙
∗
∈
𝑋
∗
. In other words, 
𝒇
⁢
(
𝒙
𝑡
)
 converges to the Pareto front.

Proof.

See Appendix B.3. ∎

Empirically, 
𝒘
𝑡
 appears to converge to some 
𝒘
∗
∈
ℝ
𝑚
 satisfying both 
𝟎
⁢
𝒘
∗
 and 
𝒥
⁢
𝒇
⁢
(
𝒙
∗
)
⊤
⁢
𝒘
∗
=
𝟎
. This suggests that the boundedness of 
𝒘
𝑡
 could be relaxed or even removed from the set of assumptions of Theorem 1.

Another common theoretical result for multi-objective optimization is convergence to a weakly stationary point. If for a given 
𝒙
∈
ℝ
𝑛
, there exists 
𝟎
⁢
𝒘
 satisfying 
𝒥
⁢
𝒇
⁢
(
𝒙
)
⊤
⁢
𝒘
=
0
 then 
𝒙
 is said to be weakly Pareto stationary. Even though every Pareto optimal point is weakly Pareto stationary, the converse does not hold, even in the convex case. Despite being necessary, convergence to a weakly Pareto stationary point is thus not a sufficient condition for optimality and, hence, constitutes a rather weak guarantee. To the best of our knowledge, 
𝒜
UPGrad
 is the first non-conflicting aggregator that provably converges to the Pareto front in the smooth convex case.

Appendix B.4 provides additional results and discussions on the convergence rate and about guarantees in the non-convex setting.

3Applications
Instance-wise risk minimization.

In machine learning, we generally have access to a training set consisting of 
𝑚
 examples. The goal of empirical risk minimization (ERM) (Vapnik, 1995) is simply to minimize the average loss over the whole training set. More generally, instance-wise risk minimization (IWRM) considers the loss associated with each training example as a distinct objective. Formally, if 
𝒙
∈
ℝ
𝑛
 are the parameters of the model and 
𝑓
𝑖
⁢
(
𝒙
)
 is the loss associated to the 
𝑖
th example, the respective objective functions of ERM and IWRM are:

	(Empirical risk)	
𝑓
¯
⁢
(
𝒙
)
	
=
1
𝑚
⁢
∑
𝑖
∈
[
𝑚
]
𝑓
𝑖
⁢
(
𝒙
)
		
(7)

	(Instance-wise risk)	
𝒇
⁢
(
𝒙
)
	
=
[
𝑓
1
⁢
(
𝒙
)
	
𝑓
2
⁢
(
𝒙
)
	
⋯
	
𝑓
𝑚
⁢
(
𝒙
)
]
⊤
		
(8)

Naively using GD for ERM is inefficient in most practical cases, so a prevalent alternative is to use SGD or one of its variants. Similarly, using JD for IWRM is typically intractable. Indeed, it would require computing a Jacobian matrix with one row per training example at each iteration. In contrast, we can use the Jacobian of a random batch of training example losses. Since it consists of a subset of the rows of the full Jacobian, this approach is a form of stochastic sub-Jacobian descent, as introduced in Section 2.1. IWRM can also be extended to cases where each 
𝑓
𝑖
 is a vector-valued function. The objective would then be the concatenation of the losses of all examples.

Multi-task learning.

In multi-task learning, a single model is trained to perform several related tasks simultaneously, leveraging shared representations to improve overall performance (Ruder, 2017). At its core, multi-task learning is a multi-objective optimization problem (Sener & Koltun, 2018), making it a straightforward application for Jacobian descent. Yet, the conflict between tasks is often too limited to justify the overhead of computing all task-specific gradients, i.e. the whole Jacobian (Kurin et al., 2022; Xin et al., 2022). In such cases, a practical approach is to minimize some linear scalarization of the objectives using an SGD-based method. Nevertheless, we believe that a setting with inherent conflict between tasks naturally prescribes Jacobian descent with a non-conflicting aggregator. We analyze several related works applied to multi-task learning in Section 4.

Adversarial training.

In adversarial domain adaptation, the feature extractor of a model is trained with two conflicting objectives: The features should be helpful for the main task and should be unable to discriminate the domain of the input (Ganin et al., 2016). Likewise, in adversarial fairness, the feature extractor is trained to both minimize the predictability of sensitive attributes, such as race or gender, and maximize the performance on the main task (Adel et al., 2019). Combining the corresponding gradients with a non-conflicting aggregator could enhance the optimization of such methods. We believe that the training of generative adversarial networks (Goodfellow et al., 2014) could be similarly formulated as a multi-objective optimization problem. The generator and discriminator could then be jointly optimized with JD.

Momentum-based optimization.

In gradient-based single-objective optimization, several methods use some form of gradient momentum to improve their convergence speed (Polyak, 1964). Essentially, their updates consider an exponential moving average of past gradients rather than just the last one. An appealing idea is to modify those algorithms to make them combine the gradient and the momentum with some aggregator, such as 
𝒜
UPGrad
, instead of summing them. This would apply to many popular optimizers, like SGD with Nesterov momentum (Nesterov, 1983), Adam (Kingma & Ba, 2014), AdamW (Loshchilov & Hutter, 2019) and NAdam (Dozat, 2016).

Distributed optimization.

In a distributed data-parallel setting with multiple machines or multiple GPUs, model updates are computed in parallel. This can be viewed as multi-objective optimization with one objective per data share. Rather than the typical averaging, a specialized aggregator, such as 
𝒜
UPGrad
, could thus combine the model updates. This consideration can even be extended to federated learning, in which multiple entities participate in the training of a common model from their own private data by sharing model updates (Kairouz et al., 2021). In this setting, as security is one of the main challenges, the non-conflicting property of the aggregator could be key.

4Existing aggregators

In the context of multi-task learning, several works have proposed iterative optimization algorithms based on the combination of task-specific gradients (Sener & Koltun, 2018; Yu et al., 2020; Liu et al., 2021b, a; Lin et al., 2021; Navon et al., 2022; Senushkin et al., 2023). These methods can be formulated as variants of JD parameterized by different aggregators. More specifically, since the gradients are stochastically estimated from batches of data, these are cases of what we call SEJD. In the following, we briefly present the most prominent aggregators and summarize their properties in Table 1. As a baseline, we also consider 
𝒜
Mean
, which simply averages the rows of the Jacobian. Their formal definitions are provided in Appendix C. Some of them are also illustrated in Figure 1(b).

𝒜
RGW
 aggregates the matrix using a random vector of weights (Lin et al., 2021). 
𝒜
MGDA
 gives the aggregation that maximizes the smallest improvement (Fliege & Svaiter, 2000; Désidéri, 2012; Sener & Koltun, 2018). 
𝒜
CAGrad
 maximizes the smallest improvement in a ball around the average gradient whose radius is parameterized by 
𝑐
∈
[
0
,
1
[
 (Liu et al., 2021a). 
𝒜
PCGrad
 projects each gradient onto the orthogonal hyperplane of other gradients in case of conflict, iteratively and in a random order (Yu et al., 2020). It is, however, only non-conflicting when 
𝑚
⁢
2
, in which case 
𝒜
PCGrad
=
𝑚
⋅
𝒜
UPGrad
. IMTL-G is a method to balance some gradients with impartiality (Liu et al., 2021b). It is only defined for linearly independent gradients, but we generalize it as a formal aggregator, denoted 
𝒜
IMTL-G
, in Appendix C.6. Aligned-MTL orthonormalizes the Jacobian and weights its rows according to some preferences (Senushkin et al., 2023). We denote by 
𝒜
Aligned-MTL
 this method with uniform preferences. 
𝒜
Nash-MTL
 aggregates Jacobians by finding the Nash equilibrium between task-specific gradients (Navon et al., 2022). Lastly, the GradDrop layer (Chen et al., 2020) defines a custom backward pass that combines gradients with respect to some internal activation. The corresponding aggregator, denoted 
𝒜
GradDrop
, randomly drops out some gradient coordinates based on their sign and sums the remaining ones.

In the context of continual learning, to limit forgetting, an idea is to project the gradient onto the dual cone of gradients computed with past examples (Lopez-Paz & Ranzato, 2017). This idea can be translated into an aggregator that projects the mean gradient onto the dual cone of the rows of the Jacobian. We name this 
𝒜
DualProj
.

Several other works consider the gradients to be noisy when making their theoretical analysis (Zhou et al., 2022; Fernando et al., 2023; Chen et al., 2023; Xiao et al., 2023; Liu & Vicente, 2024). Their solutions for combining gradients are typically stateful. Although this could enhance practical convergence rates, we have restricted our focus to the analysis of stateless aggregators. Exploring and analyzing a generalized Jacobian descent algorithm, that would preserve some state over the iterations, is a promising future direction.

In the federated learning setting, several aggregators have been proposed to combine the model updates while being robust to adversaries (Blanchard et al., 2017; Chen et al., 2017; Guerraoui et al., 2018; Yin et al., 2018). We do not study them here as they mainly focus on security aspects.

Table 1:Properties satisfied for any number of objectives. Proofs are provided in Appendix C.
Ref.	Aggregator	Non-
conflicting	Linear under
scaling	Weighted
—	
𝒜
Mean
	✗	✓	✓
Désidéri (2012)	
𝒜
MGDA
	✓	✗	✓
Lopez-Paz & Ranzato (2017)	
𝒜
DualProj
	✓	✗	✓
Yu et al. (2020)	
𝒜
PCGrad
	✗	✓	✓
Chen et al. (2020)	
𝒜
GradDrop
	✗	✗	✗
Liu et al. (2021b)	
𝒜
IMTL-G
	✗	✗	✓
Liu et al. (2021a)	
𝒜
CAGrad
	✗	✗	✓
Lin et al. (2021)	
𝒜
RGW
	✗	✓	✓
Navon et al. (2022)	
𝒜
Nash-MTL
	✓	✗	✓
Senushkin et al. (2023)	
𝒜
Aligned-MTL
	✗	✗	✓
(ours)	
𝒜
UPGrad
	✓	✓	✓
5Experiments

Appendix A.1 shows the optimization trajectories of the methods from Table 1 when optimizing two convex quadratic forms that illustrate the setting of Figure 1 and the discrepancy between weak Pareto stationarity and Pareto optimality.

In the following, we present empirical results for instance-wise risk minimization on some simple image classification datasets. IWRM is performed by stochastic sub-Jacobian descent, as described in Section 3. A key consideration is that when the aggregator is 
𝒜
Mean
, this approach becomes equivalent to empirical risk minimization with SGD. It is thus used as a baseline for comparison.

We train convolutional neural networks on subsets of SVHN (Netzer et al., 2011), CIFAR-10 (Krizhevsky et al., 2009), EuroSAT (Helber et al., 2019), MNIST (LeCun et al., 1998), Fashion-MNIST (Xiao et al., 2017) and Kuzushiji-MNIST (Clanuwat et al., 2018). To make the comparisons as fair as possible, we have tuned the learning rate very precisely for each aggregator, as explained in detail in Appendix D.1. We have also run the same experiments several times independently to gain confidence in our results. Since this leads to a total of 
43776
 training runs across all of our experiments, we have limited the size of each training dataset to 1024 images, greatly reducing computational costs. Note that this is strictly an optimization problem: we are not studying the generalization of the model, which would be captured by some performance metric on a test set. Other experimental settings, such as the network architectures and the total computational budget used to run our experiments, are given in Appendix D. Figure 2 reports the main results on SVHN and CIFAR-10, two of the datasets exhibiting the most substantial performance gap. Results on the other datasets and aggregators are reported in Appendix A.2. They also demonstrate a significant performance gap.

(a)SVHN: training loss
(b)SVHN: update similarity to the SGD update
(c)CIFAR-10: training loss
(d)CIFAR-10: update similarity to the SGD update
Figure 2:Optimization metrics obtained with IWRM with 1024 training examples and a batch size of 32, averaged over 8 independent runs. The shaded area around each curve shows the estimated standard error of the mean over the 8 runs. Curves are smoothed for readability. Best viewed in color.

Here, we compare the aggregators in terms of their average loss over the training set: the goal of ERM. For this reason, it is rather surprising that 
𝒜
Mean
, which directly optimizes this objective, exhibits a slower convergence rate than some other aggregators. In particular, 
𝒜
UPGrad
, and to a lesser extent 
𝒜
DualProj
, provide improvements on all datasets.

Figures 2(b) and 2(d) show the similarity between the update of each aggregator and the update given by 
𝒜
Mean
. For 
𝒜
UPGrad
, a low similarity indicates that there are some conflicting gradients with imbalanced norms (a setting illustrated in Figure 1). Our interpretation is thus that 
𝒜
UPGrad
 prevents gradients of hard examples from being dominated by those of easier examples early into the training. Since fitting those is more complex and time-consuming, it is beneficial to consider them earlier. We believe the similarity increases later on because the gradients become more balanced. This further suggests a greater stability of 
𝒜
UPGrad
 compared to 
𝒜
Mean
, which may allow it to perform effectively at a higher learning rate and, consequently, accelerate its convergence.

The sub-optimal performance of 
𝒜
MGDA
 in this setting can be attributed to its sensitivity to small gradients. If any row of the Jacobian approaches zero, the aggregation by 
𝒜
MGDA
 will also approach zero. This observation illustrates the discrepancy between weak stationarity and optimality, as discussed in Section 2.4. A notable advantage of linearity under scaling is to explicitly prevent this from happening.

Overall, these experiments demonstrate a high potential for the IWRM paradigm and confirm the relevance of JD, and more specifically of SSJD, as multi-objective optimization algorithms. Besides, the superiority of 
𝒜
UPGrad
 in such a simple setting supports our theoretical results.

While increasing the batch size in SGD reduces variance, the effect of doing so in SSJD combined with 
𝒜
UPGrad
 is non-trivial, as it also tightens the dual cone. Additional results obtained when varying the batch size or updating the parameters with the Adam optimizer are available in Appendices A.3 and A.4, respectively.

While an iteration of SSJD is more expensive than an iteration of SGD, its runtime is influenced by several factors, including the choice of aggregator, the parallelization capabilities of the hardware used for Jacobian computation, and the implementation. Appendix E provides memory usage and computation time considerations for our methods. Additionally, we propose a path towards a more efficient implementation in the next section.

6Gramian-based Jacobian descent

When the number of objectives is dominated by the number of parameters of the model, the main overhead of JD comes from the usage of a Jacobian matrix rather than a single gradient. In the following, we motivate an alternative implementation of JD that only uses the inner products between each pair of gradients.

For any 
𝐽
∈
ℝ
𝑚
×
𝑛
, the matrix 
𝐺
=
𝐽
⁢
𝐽
⊤
 is called the Gramian of 
𝐽
 and is positive semi-definite. Let 
ℳ
𝑚
⊆
ℝ
𝑚
×
𝑚
 be the set of positive semi-definite matrices. The Gramian of the Jacobian, denoted 
𝒢
⁢
𝒇
⁢
(
𝒙
)
=
𝒥
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒥
⁢
𝒇
⁢
(
𝒙
)
⊤
∈
ℳ
𝑚
, captures the relations – including conflicts – between all pairs of gradients. Whenever 
𝒜
 is a weighted aggregator, the update of JD is 
𝒚
=
−
𝜂
⁢
𝒥
⁢
𝒇
⁢
(
𝒙
)
⊤
⁢
𝒘
 for some vector of weights 
𝒘
∈
ℝ
𝑚
. Substituting this into the Taylor approximation of (2) gives

	
𝒇
⁢
(
𝒙
+
𝒚
)
=
𝒇
⁢
(
𝒙
)
−
𝜂
⁢
𝒢
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒘
+
𝑜
⁢
(
𝜂
⁢
𝒘
⊤
⋅
𝒢
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒘
)
.
		
(9)

This expression only depends on the Jacobian through its Gramian. It is thus sensible to focus on aggregators whose weights are only a function of the Gramian. Denoting this function as 
𝒲
:
ℳ
𝑚
→
ℝ
𝑚
, those aggregators satisfy 
𝒜
⁢
(
𝐽
)
=
𝐽
⊤
⋅
𝒲
⁢
(
𝐺
)
. Remarkably, all weighted aggregators of Table 1 can be expressed in this form. In the case of 
𝒜
UPGrad
, this is clearly demonstrated in Proposition 1, which shows that the weights depend on 
𝐺
. For such aggregators, substitution and linearity of differentiation3 then yield

	
𝒜
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
)
)
=
∇
(
𝒲
⁢
(
𝒢
⁢
𝒇
⁢
(
𝒙
)
)
⊤
⋅
𝒇
)
⁡
(
𝒙
)
.
		
(10)

After computing 
𝒲
⁢
(
𝒢
⁢
𝒇
⁢
(
𝒙
)
)
, a step of JD would thus only require the backpropagation of a scalar function. The computational cost of applying 
𝒲
 depends on the aggregator and is often dominated by the cost of computing the Gramian.

We now outline a method to compute the Gramian of the Jacobian without ever having to store the full Jacobian in memory. Similarly to the backpropagation algorithm, we can leverage the chain rule. Let 
𝒈
:
ℝ
𝑛
→
ℝ
𝑘
 and 
𝒇
:
ℝ
𝑘
→
ℝ
𝑚
, then for any 
𝒙
∈
ℝ
𝑛
, the chain rule for Gramians is

	
𝒢
⁢
(
𝒇
∘
𝒈
)
⁢
(
𝒙
)
=
𝒥
⁢
𝒇
⁢
(
𝒈
⁢
(
𝒙
)
)
⋅
𝒢
⁢
𝒈
⁢
(
𝒙
)
⋅
𝒥
⁢
𝒇
⁢
(
𝒈
⁢
(
𝒙
)
)
⊤
.
		
(11)

Moreover, when the function has multiple inputs, the Gramian can be computed as a sum of individual Gramians. Let 
𝒇
:
ℝ
𝑛
1
+
⋯
+
𝑛
𝑘
→
ℝ
𝑚
 and 
𝒙
=
[
𝒙
1
⊤
	
⋯
	
𝒙
𝑘
⊤
]
⊤
. We can write 
𝒥
⁢
𝒇
⁢
(
𝒙
)
 as the concatenation of Jacobians 
[
𝒥
𝒙
1
⁢
𝒇
⁢
(
𝒙
)
	
⋯
	
𝒥
𝒙
𝑘
⁢
𝒇
⁢
(
𝒙
)
]
, where 
𝒥
𝒙
𝑖
⁢
𝒇
⁢
(
𝒙
)
 is the Jacobian of 
𝒇
 with respect to 
𝒙
𝑖
 evaluated at 
𝒙
. For any 
𝑖
∈
[
𝑘
]
, let 
𝒢
𝒙
𝑖
⁢
𝒇
⁢
(
𝒙
)
=
𝒥
𝒙
𝑖
⁢
𝒇
⁢
(
𝒙
)
⋅
𝒥
𝒙
𝑖
⁢
𝒇
⁢
(
𝒙
)
⊤
. Then

	
𝒢
⁢
𝒇
⁢
(
𝒙
1
,
…
,
𝒙
𝑘
)
=
∑
𝑖
∈
[
𝑘
]
𝒢
𝒙
𝑖
⁢
𝒇
⁢
(
𝒙
1
,
…
,
𝒙
𝑘
)
.
		
(12)

When a function is made of compositions and concatenations of elementary functions, the Gramian of the Jacobian can thus be expressed with sums and products of partial Jacobians.

We now provide an example algorithm to compute the Gramian of a sequence of layers. For 
0
⁢
𝑖
⁢
<
⁢
𝑘
, let 
𝒇
𝑖
:
ℝ
𝑛
𝑖
×
ℝ
ℓ
𝑖
→
ℝ
𝑛
𝑖
+
1
 be a layer parameterized by 
𝒑
𝑖
∈
ℝ
ℓ
𝑖
. Given 
𝒙
0
∈
ℝ
𝑛
0
, for 
0
⁢
𝑖
⁢
<
⁢
𝑘
, the activations are recursively defined as 
𝒙
𝑖
+
1
=
𝒇
𝑖
⁢
(
𝒙
𝑖
,
𝒑
𝑖
)
. Algorithm 3 illustrates how (11) and (12) can be combined to compute the Gramian of the network with respect to its parameters.

𝐽
𝑥
←
𝐼
    # Identity matrix of size 
𝑛
𝑘
×
𝑛
𝑘
𝐺
←
0
     # Zero matrix of size 
𝑛
𝑘
×
𝑛
𝑘
for 
𝑖
←
𝑘
−
1
 to 
0
 do
       
𝐽
𝑝
←
𝒥
𝒑
𝑖
⁢
𝒇
𝑖
⁢
(
𝒙
𝑖
,
𝒑
𝑖
)
⋅
𝐽
𝑥
    # Jacobian of 
𝒙
𝑘
 w.r.t. 
𝒑
𝑖
       
𝐽
𝑥
←
𝒥
𝒙
𝑖
⁢
𝒇
𝑖
⁢
(
𝒙
𝑖
,
𝒑
𝑖
)
⋅
𝐽
𝑥
   # Jacobian of 
𝒙
𝑘
 w.r.t. 
𝒙
𝑖
       
𝐺
←
𝐺
+
𝐽
𝑝
⁢
𝐽
𝑝
⊤
Output: 
𝐺
Algorithm 3 Gramian reverse accumulation for a sequence of layers

Generalizing Algorithm 3 to any computational graph and implementing it efficiently remains an open challenge extending beyond the scope of this work.

7Conclusion

In this paper, we introduced Jacobian descent (JD), a multi-objective optimization algorithm defined by some aggregator that maps the Jacobian to an update direction. We identified desirable properties for aggregators and proposed 
𝒜
UPGrad
, addressing the limitations of existing methods while providing stronger convergence guarantees. We also highlighted potential applications of JD and proposed IWRM, a novel learning paradigm considering the loss of each training example as a distinct objective. Given its promising empirical results, we believe this paradigm deserves further attention. Additionally, we see potential for 
𝒜
UPGrad
 beyond JD, as a linear algebra tool for combining conflicting vectors in broader contexts. As speed is the primary limitation of JD, we have outlined an algorithm for efficiently computing the Gramian of the Jacobian, which could unlock JD’s full potential. We hope this work serves as a foundation for future research in multi-objective optimization and encourages a broader adoption of these methods.

Limitations and future directions.

Our empirical experiments on deep learning problems have some limitations. First, we only evaluate JD on IWRM, a setting with moderately conflicting objectives. It would be essential to develop proper benchmarks to compare aggregators on a wide variety of problems. Ideally, such problems should involve substantially conflicting objectives, e.g. multi-task learning with inherently competing or even adversarial tasks. Then, we have limited our scope to the comparison of optimization speeds, disregarding generalization. While this simplifies the experiments and makes the comparison rigorous, optimization and generalization are sometimes intertwined. We thus believe that future works should focus on both aspects.

Acknowledgments

We would like to express our sincere thanks to Scott Pesme, Emre Telatar, Matthieu Buot de l’Épine, Adrien Vandenbroucque, Ye Zhu, Alix Jeannerot, and Damian Dudzicz for their careful and thorough review. The many insightful discussions that we shared with them were essential to this project.

References
Adel et al. (2019)
↑
	Tameem Adel, Isabel Valera, Zoubin Ghahramani, and Adrian Weller.One-network adversarial fairness.In AAAI Conference on Artificial Intelligence, volume 33, pp.  2412–2420, 2019.
Blanchard et al. (2017)
↑
	Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.Machine learning with adversaries: Byzantine tolerant gradient descent.In Advances in Neural Information Processing Systems, volume 30, 2017.
Boyd & Vandenberghe (2004)
↑
	Stephen P Boyd and Lieven Vandenberghe.Convex Optimization.Cambridge University Press, 2004.
Branke (2008)
↑
	Jürgen Branke.Multiobjective Optimization: Interactive and Evolutionary Approaches.Springer Science & Business Media, 2008.
Caron et al. (2024)
↑
	Stéphane Caron, Daniel Arnström, Suraj Bonagiri, Antoine Dechaume, Nikolai Flowers, Adam Heins, Takuma Ishikawa, Dustin Kenefake, Giacomo Mazzamuto, Donato Meoli, Brendan O’Donoghue, Adam A. Oppenheimer, Abhishek Pandala, Juan José Quiroz Omaña, Nikitas Rontsis, Paarth Shah, Samuel St-Jean, Nicola Vitucci, Soeren Wolfers, Fengyu Yang, @bdelhaisse, @MeindertHH, @rimaddo, @urob, and @shaoanlu.qpsolvers: Quadratic Programming Solvers in Python, 2024.URL https://github.com/qpsolvers/qpsolvers.
Chen et al. (2023)
↑
	Lisha Chen, Heshan Fernando, Yiming Ying, and Tianyi Chen.Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance.In Advances in Neural Information Processing Systems, volume 36, pp.  70045–70093, 2023.
Chen et al. (2017)
↑
	Yudong Chen, Lili Su, and Jiaming Xu.Distributed statistical machine learning in adversarial settings: Byzantine gradient descent.Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1–25, 2017.
Chen et al. (2020)
↑
	Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov.Just pick a sign: Optimizing deep multitask models with gradient sign dropout.In Advances in Neural Information Processing Systems, volume 33, pp.  2039–2050, 2020.
Clanuwat et al. (2018)
↑
	Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha.Deep learning for classical Japanese literature.In NeurIPS Workshop on Machine Learning for Creativity and Design, 2018.
Clevert et al. (2015)
↑
	Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter.Fast and accurate deep network learning by exponential linear units (ELUs).arXiv preprint arXiv:1511.07289, 2015.
Deb et al. (2016)
↑
	Kalyanmoy Deb, Karthik Sindhya, and Jussi Hakanen.Multi-objective optimization.In Decision Sciences, pp.  161–200. CRC Press, 2016.
Dozat (2016)
↑
	Timothy Dozat.Incorporating Nesterov momentum into Adam.In International Conference on Learning Representations Workshop, 2016.
Désidéri (2012)
↑
	Jean-Antoine Désidéri.Multiple-gradient descent algorithm (MGDA) for multiobjective optimization.Comptes Rendus Mathematique, 350(5-6):313–318, 2012.
Ehrgott (2005)
↑
	Matthias Ehrgott.Multicriteria Optimization.Springer Science & Business Media, 2005.
Fernando et al. (2023)
↑
	Heshan Devaka Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen.Mitigating gradient bias in multi-objective learning: A provably convergent approach.In International Conference on Learning Representations, 2023.
Fliege et al. (2019)
↑
	Jörg Fliege, A Ismael F Vaz, and Luís Nunes Vicente.Complexity of gradient descent for multiobjective optimization.Optimization Methods and Software, 34(5):949–959, 2019.
Fliege & Svaiter (2000)
↑
	Jörg Fliege and Benar Fux Svaiter.Steepest descent methods for multicriteria optimization.Mathematical Methods of Operations Research, 51(3):479–494, 2000.
Ganin et al. (2016)
↑
	Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario March, and Victor Lempitsky.Domain-adversarial training of neural networks.Journal of Machine Learning Research, 17(59):1–35, 2016.
Goodfellow et al. (2014)
↑
	Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio.Generative adversarial nets.In Advances in Neural Information Processing Systems, volume 27, 2014.
Guerraoui et al. (2018)
↑
	Rachid Guerraoui, Sébastien Rouault, et al.The hidden vulnerability of distributed learning in Byzantium.In International Conference on Machine Learning, pp.  3521–3530, 2018.
Helber et al. (2019)
↑
	Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth.EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification.IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.
Kairouz et al. (2021)
↑
	Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al.Advances and open problems in federated learning.Foundations and Trends® in Machine Learning, 14(1–2):1–210, 2021.
Kingma & Ba (2014)
↑
	Diederik P Kingma and Jimmy Ba.Adam: A method for stochastic optimization.arXiv preprint arXiv:1412.6980, 2014.
Krizhevsky et al. (2009)
↑
	Alex Krizhevsky, Geoffrey Hinton, et al.Learning multiple layers of features from tiny images, 2009.
Kurin et al. (2022)
↑
	Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and Pawan K Mudigonda.In defense of the unitary scalarization for deep multi-task learning.In Advances in Neural Information Processing Systems, volume 35, pp.  12169–12183, 2022.
LeCun et al. (1998)
↑
	Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.Gradient-based learning applied to document recognition.Proceedings of the IEEE, 86(11):2278–2324, 1998.
Lin et al. (2021)
↑
	Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor W Tsang.Reasonable effectiveness of random weighting: A litmus test for multi-task learning.arXiv preprint arXiv:2111.10603, 2021.
Liu et al. (2021a)
↑
	Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu.Conflict-averse gradient descent for multi-task learning.In Advances in Neural Information Processing Systems, volume 34, pp.  18878–18890, 2021a.
Liu et al. (2021b)
↑
	Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang.Towards impartial multi-task learning.In International Conference on Learning Representations, 2021b.
Liu & Vicente (2024)
↑
	Suyun Liu and Luis Nunes Vicente.The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning.Annals of Operations Research, 339(3):1119–1148, 2024.
Lopez-Paz & Ranzato (2017)
↑
	David Lopez-Paz and Marc’ Aurelio Ranzato.Gradient episodic memory for continual learning.In Advances in Neural Information Processing Systems, volume 30, 2017.
Loshchilov & Hutter (2019)
↑
	Ilya Loshchilov and Frank Hutter.Decoupled weight decay regularization.In International Conference on Learning Representations, 2019.
Mercier et al. (2018)
↑
	Quentin Mercier, Fabrice Poirion, and Jean-Antoine Désidéri.A stochastic multiple gradient descent algorithm.European Journal of Operational Research, 271(3):808–817, 2018.
Navon et al. (2022)
↑
	Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya.Multi-task learning as a bargaining game.In International Conference on Machine Learning, pp.  16428–16446, 2022.
Nesterov (1983)
↑
	Yurii Nesterov.A method of solving a convex programming problem with convergence rate O(1/k**2).Proceedings of the USSR Academy of Sciences, 269(3):543–547, 1983.
Netzer et al. (2011)
↑
	Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al.Reading digits in natural images with unsupervised feature learning.In NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.
Paszke et al. (2019)
↑
	Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.PyTorch: An imperative style, high-performance deep learning library.In Advances in Neural Information Processing Systems, volume 32, 2019.
Poirion et al. (2017)
↑
	Fabrice Poirion, Quentin Mercier, and Jean-Antoine Désidéri.Descent algorithm for nonsmooth stochastic multiobjective optimization.Computational Optimization and Applications, 68(2):317–331, 2017.
Polyak (1964)
↑
	Boris T Polyak.Some methods of speeding up the convergence of iteration methods.USSR Computational Mathematics and Mathematical Physics, 4(5):1–17, 1964.
Ruder (2017)
↑
	Sebastian Ruder.An overview of multi-task learning in deep neural networks.arXiv preprint arXiv:1706.05098, 2017.
Sawaragi et al. (1985)
↑
	Yoshikazu Sawaragi, Hirotaka Nakayama, and Tetsuzo Tanino.Theory of Multiobjective Optimization.Elsevier, 1985.
Sener & Koltun (2018)
↑
	Ozan Sener and Vladlen Koltun.Multi-task learning as multi-objective optimization.In Advances in Neural Information Processing Systems, volume 31, 2018.
Senushkin et al. (2023)
↑
	Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin.Independent component alignment for multi-task learning.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.  20083–20093, 2023.
Vapnik (1995)
↑
	Vladimir Naumovich Vapnik.The Nature of Statistical learning theory.Springer-Verlag, 1995.
Xiao et al. (2017)
↑
	Han Xiao, Kashif Rasul, and Roland Vollgraf.Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.arXiv preprint arXiv:1708.07747, 2017.
Xiao et al. (2023)
↑
	Peiyao Xiao, Hao Ban, and Kaiyi Ji.Direction-oriented multi-objective learning: Simple and provable stochastic algorithms.In Advances in Neural Information Processing Systems, volume 36, pp.  4509–4533, 2023.
Xin et al. (2022)
↑
	Derrick Xin, Behrooz Ghorbani, Justin Gilmer, Ankush Garg, and Orhan Firat.Do current multi-task optimization methods in deep learning even help?In Advances in Neural Information Processing Systems, volume 35, pp.  13597–13609, 2022.
Yin et al. (2018)
↑
	Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett.Byzantine-robust distributed learning: Towards optimal statistical rates.In International Conference on Machine Learning, pp.  5650–5659, 2018.
Yu et al. (2020)
↑
	Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.Gradient surgery for multi-task learning.In Advances in Neural Information Processing Systems, volume 33, pp.  5824–5836, 2020.
Zhou et al. (2022)
↑
	Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu.On the convergence of stochastic multi-objective gradient manipulation and beyond.In Advances in Neural Information Processing Systems, volume 35, pp.  38103–38115, 2022.
Appendix AAdditional experimental results

In this appendix, we provide additional experimental results about Jacobian descent and IWRM.

A.1Optimization trajectories

Figures 3 and 4 show the optimization trajectories of the aggregators of Table 1 for two different convex quadratic functions, with several initializations4.

The first function is the element-wise quadratic function 
𝒇
EWQ
:
ℝ
2
→
ℝ
2
, defined as

	
𝒇
EWQ
⁢
(
𝒙
)
=
[
𝑥
1
2
	
𝑥
2
2
]
⊤
.
		
(13)

Notably, its Pareto set only contains the origin, while its set of weakly Pareto stationary points is the union of the two axes. The Jacobian of 
𝒇
EWQ
 at 
[
𝑥
1


𝑥
2
]
 is 
[
2
⁢
𝑥
1
	
0


0
	
2
⁢
𝑥
2
]
, indicating that the gradients never conflict, which makes this function relatively simple to optimize. Nevertheless, Figure 3 shows that 
𝒜
MGDA
, 
𝒜
CAGrad
, 
𝒜
Nash-MTL
 and 
𝒜
Aligned-MTL
 fail to converge to the Pareto set for some initializations. Indeed, they converge to weakly Pareto stationary points, which can be arbitrarily far from the Pareto set. This trivial example shows the importance of rather proving convergence to the Pareto front as in our Section 2.4.

The second function is a convex quadratic form 
𝒇
CQF
:
ℝ
2
→
ℝ
2
, constructed to sometimes reproduce the setting of Figure 1 with conflicting and imbalanced gradients:

	
𝒇
CQF
⁢
(
𝒙
)
	
=
[
(
𝒙
−
𝒗
1
)
⊤
⋅
𝑈
⁢
Σ
1
⁢
𝑈
⊤
⋅
(
𝒙
−
𝒗
1
)


(
𝒙
−
𝒗
2
)
⊤
⋅
𝑈
⊤
⁢
Σ
2
⁢
𝑈
⋅
(
𝒙
−
𝒗
2
)
]
		
(14)

	
with 
⁢
𝑈
	
=
[
cos
⁡
𝜃
	
−
sin
⁡
𝜃


sin
⁡
𝜃
	
cos
⁡
𝜃
]
⁢
, 
⁢
Σ
1
=
[
1
	
0


0
	
𝜖
]
⁢
, 
⁢
Σ
2
=
[
3
	
0


0
	
𝜖
]
⁢
, 
⁢
𝜃
=
𝜋
16
⁢
, 
⁢
𝜖
=
0.01
		
(15)

	
𝒗
1
	
=
[
1


0
]
⁢
, and 
⁢
𝒗
2
=
[
−
1


0
]
		
(16)

Figure 4 shows that most existing aggregators have some unusual behavior in this setting. In particular, on Figure 4(b), we can see that with 
𝒜
Mean
, 
𝒜
GradDrop
, 
𝒜
IMTL-G
, 
𝒜
CAGrad
, 
𝒜
RGW
 and 
𝒜
Aligned-MTL
, the updates can make one objective’s value increase. In contrast, with the non-conflicting aggregators 
𝒜
MGDA
, 
𝒜
DualProj
, 
𝒜
Nash-MTL
 and 
𝒜
UPGrad
, no objective value increases during the optimization. Due to its linearity under scaling, 
𝒜
UPGrad
 also shows improved smoothness compared to the other non-conflicting aggregators.

(a)Parameter space. The black star is the Pareto set and the dashed lines are contour lines of the mean objective.
(b)Objective value space. The black star is the Pareto front. The background color represents the distance to the Pareto front.
Figure 3:Optimization trajectories of various aggregators when optimizing 
𝒇
EWQ
:
[
𝑥
1
	
𝑥
2
]
⊤
↦
[
𝑥
1
2
	
𝑥
2
2
]
⊤
 with JD. Colored dots represent the initial points. The trajectories start in red and evolve towards yellow.
(a)Parameter space. The dark line is the Pareto set. The background color represents the cosine similarity between the two gradients, between 
−
1
 (fully conflicting) in pink and 
1
 (fully aligned) in green. For instance, the similarity is 
−
0.925
 at the initial point 
[
0
	
0
]
⊤
 and is 
−
1
 on (and only on) the Pareto set.
(b)Objective value space. The dark line is the Pareto front. The background color represents the distance to the Pareto front.
Figure 4:Optimization trajectories of various aggregators when optimizing the convex quadratic form 
𝒇
CQF
 of (14) with JD. Colored dots represent initial parameter values. The trajectories start in red and evolve towards yellow.
A.2All datasets and all aggregators

Figures 5, 6, 7, 8, 9 and 10 show the full results of the experiments described in Section 5 on SVHN, CIFAR-10, EuroSAT, MNIST, Fashion-MNIST and Kuzushiji-MNIST, respectively. For readability, the results are displayed on three different plots for each dataset. We always show 
𝒜
UPGrad
 and 
𝒜
Mean
 for reference. The exact experimental settings are described in Appendix D.

It should be noted that some of these aggregators were not developed as general-purpose aggregators, but mainly for the use case of multi-task learning, with one gradient per task. Our experiments present a more challenging setting than multi-task learning optimization because conflict between rows of the Jacobian is typically higher. Besides, for some aggregators, e.g. 
𝒜
GradDrop
 and 
𝒜
IMTL-G
, it was advised to make the aggregation of gradients w.r.t. an internal activation (such as the last shared representation), rather than w.r.t. the parameters of the model (Chen et al., 2020; Liu et al., 2021b). To enable comparison, we instead always aggregated the Jacobian w.r.t. all parameters.

We can see that 
𝒜
UPGrad
 provides a significant improvement over 
𝒜
Mean
 on all datasets. Moreover, the performance gaps seem to be linked to the difficulty of the dataset, which suggests that experimenting with harder tasks is a promising future direction. The intrinsic randomness of 
𝒜
RGW
 and 
𝒜
GradDrop
 reduces the train set performance, but it could positively impact the generalization, which we do not study here. We suspect the disappointing results of 
𝒜
Nash-MTL
 to be caused by issues in the official implementation that we used, leading to instability.

(a)Training loss
(b)Update similarity to the SGD update
(c)Training loss
(d)Update similarity to the SGD update
(e)Training loss
(f)Update similarity to the SGD update
Figure 5:SVHN results.
(a)Training loss
(b)Update similarity to the SGD update
(c)Training loss
(d)Update similarity to the SGD update
(e)Training loss
(f)Update similarity to the SGD update
Figure 6:CIFAR-10 results.
(a)Training loss
(b)Update similarity to the SGD update
(c)Training loss
(d)Update similarity to the SGD update
(e)Training loss
(f)Update similarity to the SGD update
Figure 7:EuroSAT results.
(a)Training loss
(b)Update similarity to the SGD update
(c)Training loss
(d)Update similarity to the SGD update
(e)Training loss
(f)Update similarity to the SGD update
Figure 8:MNIST results.
(a)Training loss
(b)Update similarity to the SGD update
(c)Training loss
(d)Update similarity to the SGD update
(e)Training loss
(f)Update similarity to the SGD update
Figure 9:Fashion-MNIST results.
(a)Training loss
(b)Update similarity to the SGD update
(c)Training loss
(d)Update similarity to the SGD update
(e)Training loss
(f)Update similarity to the SGD update
Figure 10:Kuzushiji-MNIST results.
A.3Varying the batch size

Figure 11 shows the results on CIFAR-10 with 
𝒜
UPGrad
 when varying the batch size from 4 to 64. Concretely, because we are using SSJD, this makes the number of rows of the sub-Jacobian aggregated at each step vary from 4 to 64. Recall that IWRM with SSJD and 
𝒜
Mean
 is equivalent to ERM with SGD. We observe that with a small batch size, 
𝒜
UPGrad
 becomes very similar to 
𝒜
Mean
. This is not surprising since both would be equivalent with a batch size of 1. Conversely, a larger batch size increases the gap between 
𝒜
UPGrad
 and 
𝒜
Mean
. Since the projections of 
𝒜
UPGrad
 are onto the dual cone of more rows, each step becomes non-conflicting with respect to more of the original 1024 objectives, pushing even further the benefits of the non-conflicting property. In other words, increasing the batch size refines the dual cone, thereby improving the quality of the projections. It would be interesting to theoretically analyze the impact of the batch size in this setting.

(a)
BS
=
4
: Training loss
(b)
BS
=
4
: Update similarity to the SGD update
(c)
BS
=
16
: Training loss
(d)
BS
=
16
: Update similarity to the SGD update
(e)
BS
=
64
: Training loss
(f)
BS
=
64
: Update similarity to the SGD update
Figure 11:CIFAR-10 results with different batch sizes (BS). The number of epochs is always 20, so the number of iterations varies.
A.4Compatibility with Adam

Figure 12 gives the results on CIFAR-10 and SVHN when using Adam rather than the SGD optimizer. Concretely, this corresponds to the Adam algorithm in which the gradient is replaced by the aggregation of the Jacobian. The learning rate is still tuned as described in Appendix D.1, but the other hyperparameters of Adam are fixed to the default values of PyTorch, i.e. 
𝛽
1
=
0.9
, 
𝛽
2
=
0.999
 and 
𝜖
=
10
−
8
. Because optimization with Adam is faster, the number of epochs for SVHN and CIFAR-10 is reduced to 20 and 15, respectively. While the performance gap is smaller with this optimizer, it is still significant and suggests that our methods are beneficial with other optimizers than the simple SGD. Note that this analysis is fairly superficial. The thorough investigation of the interplay between aggregators and momentum-based optimizers is a compelling future research direction.

(a)SVHN: Training loss
(b)SVHN: Update similarity to the SGD update
(c)CIFAR-10: Training loss
(d)CIFAR-10: Update similarity to the SGD update
Figure 12:Results with the Adam optimizer.
Appendix BProofs
B.1Preliminary theoretical results

Recall that a function 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 is -convex if for all 
𝒙
,
𝒚
∈
ℝ
𝑛
 and any 
𝜆
∈
[
0
,
1
]
,

	
𝒇
⁢
(
𝜆
⁢
𝒙
+
(
1
−
𝜆
)
⁢
𝒚
)
⁢
𝜆
⁢
𝒇
⁢
(
𝒙
)
+
(
1
−
𝜆
)
⁢
𝒇
⁢
(
𝒚
)
.
	
Lemma 1.

If 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 is a continuously differentiable -convex function, then for any pair of vectors 
𝒙
,
𝒚
∈
ℝ
𝑛
, 
𝒥
⁢
𝒇
⁢
(
𝒙
)
⁢
(
𝒚
−
𝒙
)
⁢
𝒇
⁢
(
𝒚
)
−
𝒇
⁢
(
𝒙
)
.

Proof.
	
𝒥
⁢
𝒇
⁢
(
𝒙
)
⁢
(
𝒚
−
𝒙
)
	
=
lim
𝜆
→
0
+
𝒇
⁢
(
𝒙
+
𝜆
⁢
(
𝒚
−
𝒙
)
)
−
𝒇
⁢
(
𝒙
)
𝜆
	
(
differentiation
)
	
		
lim
𝜆
→
0
+
𝒇
⁢
(
𝒙
)
+
𝜆
⁢
(
𝒇
⁢
(
𝒚
)
−
𝒇
⁢
(
𝒙
)
)
−
𝒇
⁢
(
𝒙
)
𝜆
	
(
-convexity
)
	
		
=
𝒇
⁢
(
𝒚
)
−
𝒇
⁢
(
𝒙
)
,
	

which concludes the proof. ∎

Lemma 2.

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
, let 
𝒖
∈
ℝ
𝑚
 and let 
𝒙
∈
ℝ
𝑛
, then

	
𝒖
⊤
⁢
𝐽
⁢
𝒙
⁢
‖
𝒖
‖
⋅
‖
𝐽
‖
F
⋅
‖
𝒙
‖
	
Proof.

Let 
𝐽
𝑖
 be the 
𝑖
th row of 
𝐽
, then

	
(
𝒖
⊤
⁢
𝐽
⁢
𝒙
)
2
	
‖
𝒖
‖
2
⋅
‖
𝐽
⁢
𝒙
‖
2
	
(
Cauchy-Schwartz


inequality
)
	
		
=
‖
𝒖
‖
2
⋅
∑
𝑖
∈
[
𝑚
]
(
𝐽
𝑖
⊤
⁢
𝒙
)
2
	
		
‖
𝒖
‖
2
⋅
∑
𝑖
∈
[
𝑚
]
‖
𝐽
𝑖
‖
2
⋅
‖
𝒙
‖
2
	
(
Cauchy-Schwartz


inequality
)
	
		
=
‖
𝒖
‖
2
⋅
‖
𝐽
‖
F
2
⋅
‖
𝒙
‖
2
,
	

which concludes the proof. ∎

Recall that a function 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 is 
𝛽
-smooth if for all 
𝒙
,
𝒚
∈
ℝ
𝑛
,

	
‖
𝒥
⁢
𝒇
⁢
(
𝒙
)
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
‖
F
⁢
𝛽
⁢
‖
𝒙
−
𝒚
‖
		
(17)
Lemma 3.

Let 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 be 
𝛽
-smooth, then for any 
𝒘
∈
ℝ
𝑚
 and any 
𝒙
,
𝒚
∈
ℝ
𝑛
,

	
𝒘
⊤
⁢
(
𝒇
⁢
(
𝒙
)
−
𝒇
⁢
(
𝒚
)
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
⁢
(
𝒙
−
𝒚
)
)
⁢
𝛽
2
⁢
‖
𝒘
‖
⋅
‖
𝒙
−
𝒚
‖
2
		
(18)
Proof.
		
𝒘
⊤
⁢
(
𝒇
⁢
(
𝒙
)
−
𝒇
⁢
(
𝒚
)
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
⁢
(
𝒙
−
𝒚
)
)
	
	
=
	
𝒘
⊤
⁢
(
∫
0
1
𝒥
⁢
𝒇
⁢
(
𝒚
+
𝑡
⁢
(
𝒙
−
𝒚
)
)
⁢
(
𝒙
−
𝒚
)
⁢
𝑑
𝑡
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
⁢
(
𝒙
−
𝒚
)
)
	
(
fundamental


theorem


of calculus
)
	
	
=
	
∫
0
1
𝒘
⊤
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒚
+
𝑡
⁢
(
𝒙
−
𝒚
)
)
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
)
⁢
(
𝒙
−
𝒚
)
⁢
𝑑
𝑡
	
		
∫
0
1
‖
𝒘
‖
⋅
‖
𝒥
⁢
𝒇
⁢
(
𝒚
+
𝑡
⁢
(
𝒙
−
𝒚
)
)
−
𝒥
⁢
𝒇
⁢
(
𝒚
)
‖
F
⋅
‖
𝒙
−
𝒚
‖
⁢
𝑑
𝑡
	
(
Lemma 
2
)
	
		
∫
0
1
‖
𝒘
‖
⋅
𝛽
⁢
𝑡
⋅
‖
𝒙
−
𝒚
‖
2
⁢
𝑑
𝑡
	
(
𝛽
-smoothness 
17
)
	
	
=
	
𝛽
2
⁢
‖
𝒘
‖
⋅
‖
𝒙
−
𝒚
‖
2
,
	

which concludes the proof. ∎

B.2Proposition 1
Proposition 1.

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
. For any 
𝒖
∈
ℝ
𝑚
, 
𝜋
𝐽
⁢
(
𝐽
⊤
⁢
𝒖
)
=
𝐽
⊤
⁢
𝒘
 with

	
𝒘
	
∈
arg
⁢
min
𝒗
∈
ℝ
𝑚
:
𝒖
⁢
𝒗
⁡
𝒗
⊤
⁢
𝐽
⁢
𝐽
⊤
⁢
𝒗
.
		
(5)
Proof.

This is a direct consequence of Lemma 4. ∎

Lemma 4.

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
, 
𝐺
=
𝐽
⁢
𝐽
⊤
, 
𝒖
∈
ℝ
𝑚
. For any 
𝒘
∈
ℝ
𝑚
 satisfying

		
𝒖
⁢
𝒘
			
(19a)

		
𝟎
⁢
𝐺
⁢
𝒘
			
(19b)

		
𝒖
⊤
⁢
𝐺
⁢
𝒘
=
𝒘
⊤
⁢
𝐺
⁢
𝒘
			
(19c)

we have 
𝜋
𝐽
⁢
(
𝐽
⊤
⁢
𝒖
)
=
𝐽
⊤
⁢
𝒘
. Such a 
𝒘
 is the solution to

	
𝒘
	
∈
arg
⁢
min
𝒖
⁢
𝒗
⁡
𝒗
⊤
⁢
𝐺
⁢
𝒗
.
	
Proof.

The projection

	
𝜋
𝐽
⁢
(
𝐽
⊤
⁢
𝒖
)
=
arg
⁢
min
𝒙
∈
ℝ
𝑛
:


𝟎
⁢
𝐽
⁢
𝒙
⁡
1
2
⁢
‖
𝒙
−
𝐽
⊤
⁢
𝒖
‖
2
	

is a convex program. Consequently, the KKT conditions are both necessary and sufficient. The Lagragian is given by 
ℒ
⁢
(
𝒙
,
𝒗
)
=
1
2
⁢
‖
𝒙
−
𝐽
⊤
⁢
𝒖
‖
2
−
𝒗
⊤
⁢
𝐽
⁢
𝒙
. The KKT conditions are then given by

		
{
∇
𝒙
ℒ
⁢
(
𝒙
,
𝒗
)
=
𝟎
	

𝟎
⁢
𝒗
	

𝟎
⁢
𝐽
⁢
𝒙
	

0
=
𝒗
⊤
⁢
𝐽
⁢
𝒙
	
	
	
⇔
	
{
𝒙
=
𝐽
⊤
⁢
(
𝒖
+
𝒗
)
	

𝟎
⁢
𝒗
	

𝟎
⁢
𝐺
⁢
(
𝒖
+
𝒗
)
	

0
=
𝒗
⊤
⁢
𝐺
⁢
(
𝒖
+
𝒗
)
	
	
	
⇔
	
{
𝒙
=
𝐽
⊤
⁢
(
𝒖
+
𝒗
)
	

𝒖
⁢
𝒖
+
𝒗
	

𝟎
⁢
𝐺
⁢
(
𝒖
+
𝒗
)
	

𝒖
⊤
⁢
𝐺
⁢
(
𝒖
+
𝒗
)
=
(
𝒖
+
𝒗
)
⊤
⁢
𝐺
⁢
(
𝒖
+
𝒗
)
	
	

The simple change of variable 
𝒘
=
𝒖
+
𝒗
 finishes the proof of the first part.

Since 
𝒙
=
𝐽
⊤
⁢
(
𝒖
+
𝒗
)
, the Wolfe dual program of 
𝜋
𝐽
⁢
(
𝐽
⊤
⁢
𝒖
)
 gives

	
𝒘
	
∈
𝒖
+
arg
⁢
max
𝒗
∈
ℝ
𝑚
:
𝟎
⁢
𝒗
⁡
ℒ
⁢
(
𝐽
⊤
⁢
(
𝒖
+
𝒗
)
,
𝒗
)
	
		
=
𝒖
+
arg
⁢
max
𝒗
∈
ℝ
𝑚
:
𝟎
⁢
𝒗
⁡
1
2
⁢
‖
𝐽
⊤
⁢
𝒗
‖
2
−
𝒗
⊤
⁢
𝐽
⁢
𝐽
⊤
⁢
(
𝒖
+
𝒗
)
	
		
=
𝒖
+
arg
⁢
max
𝒗
∈
ℝ
𝑚
:
𝟎
⁢
𝒗
−
1
2
⁢
𝒗
⊤
⁢
𝐺
⁢
𝒗
−
𝒗
⊤
⁢
𝐺
⁢
𝒖
	
		
=
𝒖
+
arg
⁢
min
𝒗
∈
ℝ
𝑚
:
𝒖
⁢
𝒖
+
𝒗
⁡
1
2
⁢
(
𝒖
+
𝒗
)
⊤
⁢
𝐺
⁢
(
𝒖
+
𝒗
)
	
		
=
arg
⁢
min
𝒗
′
∈
ℝ
𝑚
:
𝒖
⁢
𝒗
′
⁡
1
2
⁢
𝒗
′
⁣
⊤
⁢
𝐺
⁢
𝒗
′
,
	

which concludes the proof. ∎

B.3Theorem 1
Theorem 1.

Let 
𝒇
:
ℝ
𝑛
→
ℝ
𝑚
 be a 
𝛽
-smooth and -convex function. Suppose that the Pareto front 
𝒇
⁢
(
𝑋
∗
)
 is bounded and that for any 
𝒙
∈
ℝ
𝑛
, there is 
𝒙
∗
∈
𝑋
∗
 satisfying 
𝒇
⁢
(
𝒙
∗
)
⁢
𝒇
⁢
(
𝒙
)
. Let 
𝒙
1
∈
ℝ
𝑛
, and for all 
𝑡
∈
ℕ
, 
𝒙
𝑡
+
1
=
𝒙
𝑡
−
𝜂
⁢
𝒜
UPGrad
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
)
, with 
𝜂
=
1
𝛽
⁢
𝑚
. Let 
𝒘
𝑡
 be the weights defining 
𝒜
UPGrad
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
)
 as per (6), i.e. 
𝒜
UPGrad
⁢
(
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
)
=
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
⊤
⋅
𝒘
𝑡
. If 
𝒘
𝑡
 is bounded, then 
𝒇
⁢
(
𝒙
𝑡
)
 converges to 
𝒇
⁢
(
𝒙
∗
)
 for some 
𝒙
∗
∈
𝑋
∗
. In other words, 
𝒇
⁢
(
𝒙
𝑡
)
 converges to the Pareto front.

To prove the theorem we will need Lemmas 5, 6 and 7 below.

Lemma 5.

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
 and 
𝒘
=
1
𝑚
⁢
∑
𝑖
=
1
𝑚
𝒘
𝑖
 be the weights defining 
𝒜
UPGrad
⁢
(
𝐽
)
 as per (6). Let, as usual, 
𝐺
=
𝐽
⁢
𝐽
⊤
, then,

	
𝒘
⊤
⁢
𝐺
⁢
𝒘
⁢
𝟏
⊤
⁢
𝐺
⁢
𝒘
.
	
Proof.

Observe that if, for any 
𝒖
,
𝒗
∈
ℝ
𝑚
, 
⟨
𝒖
,
𝒗
⟩
=
𝒖
⊤
⁢
𝐺
⁢
𝒗
, then 
⟨
⋅
,
⋅
⟩
 is an inner product. In this Hilbert space, the Cauchy-Schwartz inequality reads as

	
(
𝒖
⊤
⁢
𝐺
⁢
𝒗
)
2
	
=
⟨
𝒖
,
𝒗
⟩
2
	
		
⟨
𝒖
,
𝒖
⟩
⋅
⟨
𝒗
,
𝒗
⟩
	
		
=
𝒖
⊤
⁢
𝐺
⁢
𝒖
⋅
𝒗
⊤
⁢
𝐺
⁢
𝒗
.
	

Therefore

		
𝒘
⊤
⁢
𝐺
⁢
𝒘
	
	
=
	
1
𝑚
2
⁢
∑
𝑖
,
𝑗
𝒘
𝑖
⊤
⁢
𝐺
⁢
𝒘
𝑗
	
		
1
𝑚
2
⁢
∑
𝑖
,
𝑗
𝒘
𝑖
⊤
⁢
𝐺
⁢
𝒘
𝑖
⋅
𝒘
𝑗
⊤
⁢
𝐺
⁢
𝒘
𝑗
	
(
Cauchy-Schwartz


inequality
)
	
	
=
	
(
∑
𝑖
1
𝑚
⁢
𝒘
𝑖
⊤
⁢
𝐺
⁢
𝒘
𝑖
)
2
	
		
∑
𝑖
1
𝑚
⁢
(
𝒘
𝑖
⊤
⁢
𝐺
⁢
𝒘
𝑖
)
2
	
(
Jensen’s inequality
)
	
	
=
	
1
𝑚
⁢
∑
𝑖
𝒘
𝑖
⊤
⁢
𝐺
⁢
𝒘
𝑖
	
(
𝐺
⁢
 positive semi-definite
)
	
	
=
	
1
𝑚
⁢
∑
𝑖
𝒆
𝑖
⊤
⁢
𝐺
⁢
𝒘
𝑖
	
(
Lemma 
4
, (
19c
)
)
	
		
1
𝑚
⁢
∑
𝑖
𝟏
⊤
⁢
𝐺
⁢
𝒘
𝑖
	
(
Lemma 
4
, (
19b
)


𝒆
𝑖
⁢
𝟏
)
	
	
=
	
𝟏
⊤
⁢
𝐺
⁢
𝒘
,
	

which concludes the proof. ∎

Lemma 6.

Under the assumptions of Theorem 1, for any 
𝒘
∈
ℝ
𝑚
 and any 
𝑡
∈
ℕ
,

	
𝒘
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
+
1
)
−
𝒇
⁢
(
𝒙
𝑡
)
)
⁢
‖
𝒘
‖
𝛽
⁢
𝑚
⁢
(
𝟏
2
⁢
𝑚
−
𝒘
‖
𝒘
‖
)
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
.
	
Proof.

For all 
𝑡
∈
ℕ
, let 
𝐽
𝑡
=
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
, 
𝐺
𝑡
=
𝐽
𝑡
⁢
𝐽
𝑡
⊤
. Then 
𝒙
𝑡
+
1
=
𝒙
𝑡
−
𝜂
⁢
𝒜
UPGrad
⁢
(
𝐽
𝑡
)
=
𝒙
𝑡
−
𝜂
⁢
𝐽
𝑡
⊤
⁢
𝒘
𝑡
. Therefore

		
𝒘
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
+
1
)
−
𝒇
⁢
(
𝒙
𝑡
)
)
	
		
−
𝜂
⁢
𝒘
⊤
⁢
𝐽
𝑡
⁢
𝐽
𝑡
⊤
⁢
𝒘
𝑡
+
𝛽
⁢
𝜂
2
2
⁢
‖
𝒘
‖
⋅
‖
𝐽
𝑡
⊤
⁢
𝒘
𝑡
‖
2
	
(
Lemma 
3
)
	
	
=
	
−
1
𝛽
⁢
𝑚
⁢
𝒘
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
+
1
2
⁢
𝛽
⁢
𝑚
⁢
‖
𝒘
‖
⋅
𝒘
𝑡
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
	
(
𝜂
=
1
𝛽
⁢
𝑚
)
	
		
−
1
𝛽
⁢
𝑚
⁢
𝒘
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
+
1
2
⁢
𝛽
⁢
𝑚
⁢
‖
𝒘
‖
⋅
𝟏
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
	
(
Lemma 
5
)
	
	
=
	
‖
𝒘
‖
𝛽
⁢
𝑚
⁢
(
𝟏
2
⁢
𝑚
−
𝒘
‖
𝒘
‖
)
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
,
	

which concludes the proof. ∎

Lemma 7.

Under the assumptions of Theorem 1, if 
𝒙
∗
∈
𝑋
∗
 satisfies 
𝟏
⊤
⁢
𝒇
⁢
(
𝒙
∗
)
⁢
𝟏
⊤
⁢
𝒇
⁢
(
𝒙
𝑡
)
 for all 
𝑡
∈
ℕ
, then

	
1
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
⁢
1
𝑇
⁢
(
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
+
𝛽
⁢
𝑚
2
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
)
.
		
(20)
Proof.

We first bound, for any 
𝑡
∈
ℕ
, 
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
+
1
)
−
𝒇
⁢
(
𝒙
𝑡
)
)
 as follows

		
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
+
1
)
−
𝒇
⁢
(
𝒙
𝑡
)
)
	
		
−
1
2
⁢
𝛽
⁢
𝑚
⋅
𝟏
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
	
(
Lemma 
6


with 
⁢
𝒘
=
𝟏
)
	
		
−
1
2
⁢
𝛽
⁢
𝑚
⋅
𝒘
𝑡
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
.
	
(
Lemma 
5
)
	

Summing this over 
𝑡
∈
[
𝑇
]
 yields

		
1
2
⁢
𝛽
⁢
𝑚
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
	
		
∑
𝑡
∈
[
𝑇
]
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
𝑡
+
1
)
)
	
	
=
	
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
𝑇
+
1
)
)
	
(
Telescoping sum
)
		
		
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
.
	
(
Assumption


𝟏
⊤
⁢
𝒇
⁢
(
𝒙
∗
)
⁢
𝟏
⊤
⁢
𝒇
⁢
(
𝒙
𝑇
+
1
)
)
			
(21)

Since 
𝟎
⁢
𝒘
𝑡
,

		
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
	
		
𝒘
𝑡
⊤
⁢
𝐽
𝑡
⁢
(
𝒙
𝑡
−
𝒙
∗
)
	
(
Lemma 
1
)
	
	
=
	
1
𝜂
⁢
(
𝒙
𝑡
−
𝒙
𝑡
+
1
)
⊤
⁢
(
𝒙
𝑡
−
𝒙
∗
)
	
(
𝒙
𝑡
+
1
=
𝒙
𝑡
−
𝜂
⁢
𝐽
𝑡
⊤
⁢
𝒘
𝑡
)
	
	
=
	
1
2
⁢
𝜂
⁢
(
‖
𝒙
𝑡
−
𝒙
𝑡
+
1
‖
2
+
‖
𝒙
𝑡
−
𝒙
∗
‖
2
−
‖
𝒙
𝑡
+
1
−
𝒙
∗
‖
2
)
	
(
Parallelogram


law
)
	
	
=
	
1
2
⁢
𝛽
⁢
𝑚
⁢
𝒘
𝑡
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
+
𝛽
⁢
𝑚
2
⁢
(
‖
𝒙
𝑡
−
𝒙
∗
‖
2
−
‖
𝒙
𝑡
+
1
−
𝒙
∗
‖
2
)
.
	
(
𝜂
=
1
𝛽
⁢
𝑚
)
	

Summing this over 
𝑡
∈
[
𝑇
]
 yields

		
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
	
		
1
2
⁢
𝛽
⁢
𝑚
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
+
𝛽
⁢
𝑚
2
⁢
(
‖
𝒙
1
−
𝒙
∗
‖
2
−
‖
𝒙
𝑇
+
1
−
𝒙
∗
‖
2
)
	
(
Telescoping sum
)
	
		
1
2
⁢
𝛽
⁢
𝑚
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
𝐺
𝑡
⁢
𝒘
𝑡
+
𝛽
⁢
𝑚
2
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
	
		
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
+
𝛽
⁢
𝑚
2
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
.
	
(
By (
21
)
)
	

Scaling down this inequality by 
𝑇
 yields

	
1
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
⁢
1
𝑇
⁢
(
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
+
𝛽
⁢
𝑚
2
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
)
,
	

which concludes the proof. ∎

We are now ready to prove Theorem 1.

Proof.

For all 
𝑡
∈
ℕ
, let 
𝐽
𝑡
=
𝒥
⁢
𝒇
⁢
(
𝒙
𝑡
)
, 
𝐺
𝑡
=
𝐽
𝑡
⁢
𝐽
𝑡
⊤
. Then

	
𝒙
𝑡
+
1
	
=
𝒙
𝑡
−
𝜂
⁢
𝒜
UPGrad
⁢
(
𝐽
𝑡
)
	
		
=
𝒙
𝑡
−
𝜂
⁢
𝐽
𝑡
⊤
⁢
𝒘
𝑡
.
	

Substituting 
𝒘
=
𝟏
 in the term 
𝟏
2
⁢
𝑚
−
𝒘
‖
𝒘
‖
 of Lemma 6 yields

	
𝟏
2
⁢
𝑚
−
𝒘
‖
𝒘
‖
	
=
−
𝟏
2
⁢
𝑚
	
		
𝟎
.
	

Therefore there exists some 
𝜀
⁢
>
⁢
0
 such that any 
𝒘
∈
ℝ
𝑚
 with 
‖
𝟏
−
𝒘
‖
⁢
<
⁢
𝜀
 satisfies 
𝟏
2
⁢
𝑚
⁢
𝒘
‖
𝒘
‖
. Denote by 
𝐵
𝜀
⁢
(
𝟏
)
=
{
𝒘
∈
ℝ
𝑚
:
‖
𝟏
−
𝒘
‖
⁢
<
⁢
𝜀
}
, i.e. for all 
𝒘
∈
𝐵
𝜀
⁢
(
𝟏
)
, 
𝟏
2
⁢
𝑚
⁢
𝒘
‖
𝒘
‖
. By the non-conflicting property of 
𝒜
UPGrad
, 
𝟎
⁢
𝐺
𝑡
⁢
𝒘
𝑡
 and therefore for all 
𝒘
∈
𝐵
𝜀
⁢
(
𝟏
)
,

		
𝒘
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
+
1
)
−
𝒇
⁢
(
𝒙
𝑡
)
)
	
		
‖
𝒘
‖
𝛽
⁢
𝑚
⁢
(
𝟏
2
⁢
𝑚
−
𝒘
‖
𝒘
‖
)
⁢
𝐺
𝑡
⁢
𝒘
𝑡
	
(
Lemma 
6
)
	
		
0
.
	

Since 
𝒘
⊤
⁢
𝒇
⁢
(
𝒙
𝑡
)
 is bounded and non-increasing, it converges. Since 
𝐵
𝜀
⁢
(
𝟏
)
 contains a basis of 
ℝ
𝑚
, 
𝒇
⁢
(
𝒙
𝑡
)
 converges to some 
𝒇
∗
∈
ℝ
𝑚
. By assumption on 
𝒇
, there exists 
𝒙
∗
 in the Pareto set satisfying 
𝒇
⁢
(
𝒙
∗
)
⁢
𝒇
∗
.

We now prove that 
𝒇
⁢
(
𝒙
∗
)
=
𝒇
∗
. Since 
𝒇
⁢
(
𝒙
∗
)
⁢
𝒇
∗
, it is sufficient to show that 
𝟏
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
∗
)
)
⁢
0
.

First, the additional assumption of Lemma 7 applies since 
𝟏
⊤
⁢
𝒇
⁢
(
𝒙
𝑡
)
 decreases to 
𝟏
⊤
⁢
𝒇
∗
 which is larger than 
𝟏
⊤
⁢
𝒇
⁢
(
𝒙
∗
)
. Therefore

		
𝟏
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
∗
)
)
	
		
(
𝑚
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
)
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
∗
)
)
	
(
𝒇
⁢
(
𝒙
∗
)
⁢
𝒇
∗


𝟏
⁢
𝑚
⁢
𝒘
𝑡


 by (
19a
)
)
		
	
=
	
𝑚
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
𝑡
)
+
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
	
	
=
	
𝑚
𝑇
⁢
(
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
𝑡
)
)
+
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
)
	
		
𝑚
𝑇
⁢
(
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
𝑡
)
)
+
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
+
𝛽
⁢
𝑚
2
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
)
	
(
Lemma 
7
)
			
(22)

Taking the limit as 
𝑇
→
∞
, we get

		
𝟏
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
∗
)
)
	
		
lim
𝑇
→
∞
𝑚
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
∗
−
𝒇
⁢
(
𝒙
𝑡
)
)
	
		
lim
𝑇
→
∞
𝑚
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
‖
𝒘
𝑡
‖
⋅
‖
𝒇
∗
−
𝒇
⁢
(
𝒙
𝑡
)
‖
	
(
Cauchy-Schwartz


inequality
)
	
	
=
	
0
,
	
(
𝒘
𝑡
⁢
 bounded


𝒇
⁢
(
𝒙
𝑡
)
→
𝒇
∗
)
	

which concludes the proof. ∎

B.4Supplementary theoretical results

The proof of Theorem 1 provides some additional insights about the convergence rate as well as some notion of convergence in the non-convex case.

Convergence rate.

Combining the equality 
𝒇
∗
=
𝒇
⁢
(
𝒙
∗
)
 and (22), we have

	
1
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
⁢
1
𝑇
⁢
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
+
𝛽
⁢
𝑚
2
⁢
𝑇
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
	

Furthermore, the Cauchy-Schwartz inequality yields 
𝟏
⊤
⁢
(
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
)
⁢
𝑚
⋅
‖
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
‖
, so

	
1
𝑇
⁢
∑
𝑡
∈
[
𝑇
]
𝒘
𝑡
⊤
⁢
(
𝒇
⁢
(
𝒙
𝑡
)
−
𝒇
⁢
(
𝒙
∗
)
)
⁢
𝑚
𝑇
⁢
(
‖
𝒇
⁢
(
𝒙
1
)
−
𝒇
⁢
(
𝒙
∗
)
‖
+
𝛽
2
⁢
‖
𝒙
1
−
𝒙
∗
‖
2
)
		
(23)

This hints a convergence rate of order 
𝒪
⁢
(
1
𝑇
)
, whose constant depends on the initial point 
𝒙
1
.

Non-convex setting.

The proof of (21) does not require the convexity of the objective function. This bound can be equivalently formulated as

	
1
𝑇
⁢
∑
𝑡
=
1
𝑇
‖
𝐽
𝑡
⊤
⁢
𝑤
𝑡
‖
2
⁢
2
⁢
𝛽
⁢
𝑚
𝑇
⁢
𝟏
⊤
⁢
(
𝑓
⁢
(
𝑥
1
)
−
𝑓
⁢
(
𝑥
∗
)
)
		
(24)

This shows the convergence of the updates in the non-convex setting, under the smoothness condition of Theorem 1.

Appendix CProperties of existing aggregators

In the following, we prove the properties of the aggregators from Table 1. Some aggregators, e.g. 
𝒜
RGW
, 
𝒜
GradDrop
 and 
𝒜
PCGrad
, are non-deterministic and are thus not technically functions but rather random variables whose distribution depends on the matrix 
𝐽
∈
ℝ
𝑚
×
𝑛
 to aggregate. Still, the properties of Section 2.2 can be easily adapted to a random setting. If 
𝒜
 is a random aggregator, then for any 
𝐽
, 
𝒜
⁢
(
𝐽
)
 is a random vector in 
ℝ
𝑛
. The aggregator is non-conflicting if 
𝒜
⁢
(
𝐽
)
 is in the dual cone of the rows of 
𝐽
 with probability 1. It is linear under scaling if for all 
𝐽
∈
ℝ
𝑚
×
𝑛
, there is a – possibly random – matrix 
𝔍
∈
ℝ
𝑚
×
𝑛
 such that for all 
𝟎
⁢
𝒄
∈
ℝ
𝑚
, 
𝒜
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
=
𝔍
⊤
⋅
𝒄
. Finally, 
𝒜
 is weighted if for any 
𝐽
∈
ℝ
𝑚
×
𝑛
 there is a – possibly random – weighting 
𝒘
∈
ℝ
𝑚
 satisfying 
𝒜
⁢
(
𝐽
)
=
𝐽
⊤
⋅
𝒘
.

C.1Mean

𝒜
Mean
 simply averages the rows of the input matrix, i.e. for all 
𝐽
∈
ℝ
𝑚
×
𝑛
,

	
𝒜
Mean
⁢
(
𝐽
)
=
1
𝑚
⁢
𝐽
⊤
⋅
𝟏
		
(25)
✗  Non-conflicting.

𝒜
Mean
⁢
(
[
−
2


4
]
)
=
[
1
]
, which conflicts with 
[
−
2
]
, so 
𝒜
Mean
 is not non-conflicting.

✓  Linear under scaling.

For any 
𝒄
∈
ℝ
𝑚
, 
𝒜
Mean
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
=
1
𝑚
⁢
𝐽
⊤
⋅
𝒄
, which is linear in 
𝒄
. 
𝒜
Mean
 is therefore linear under scaling.

✓  Weighted.

By (25), 
𝒜
Mean
 is weighted with constant weighting equal to 
1
𝑚
⁢
𝟏
.

C.2MGDA

The optimization algorithm presented in Désidéri (2012), called MGDA, is tied to a particular method for aggregating the gradients. We thus refer to this aggregator as 
𝒜
MGDA
. The dual problem of this method was also introduced independently in Fliege & Svaiter (2000). We show the equivalence between the two solutions to make the analysis of 
𝒜
MGDA
 easier.

For all 
𝐽
∈
ℝ
𝑚
×
𝑛
, the aggregation described in Désidéri (2012) is defined as

	
𝒜
MGDA
⁢
(
𝐽
)
	
=
𝐽
⊤
⋅
𝒘
		
(26)

	
with
𝒘
	
∈
arg
⁢
min
𝟎
⁢
𝒗
:


𝟏
⊤
⁢
𝒗
=
1
⁡
‖
𝐽
⊤
⁢
𝒗
‖
2
		
(27)

In Equation (3) of Fliege & Svaiter (2000), the following problem is studied:

	
min
𝛼
∈
ℝ
,
𝒙
∈
ℝ
𝑛
:


𝐽
⁢
𝒙
⁢
𝛼
⁢
𝟏
⁡
𝛼
+
1
2
⁢
‖
𝒙
‖
2
		
(28)

We show that the problems in (27) and (28) are dual to each other. Furthermore, the duality gap is null since this is a convex problem. The Lagrangian of the problem in (28) is given by 
ℒ
⁢
(
𝛼
,
𝒙
,
𝝁
)
=
𝛼
+
1
2
⁢
‖
𝒙
‖
2
−
𝝁
⊤
⁢
(
𝛼
⁢
𝟏
−
𝐽
⁢
𝒙
)
. Differentiating w.r.t. 
𝛼
 and 
𝒙
 gives respectively 
1
−
𝟏
⊤
⁢
𝝁
 and 
𝒙
+
𝐽
⊤
⁢
𝝁
. The dual problem is obtained by setting those two to 
𝟎
 and then maximizing the Lagrangian on 
𝟎
⁢
𝝁
 and 
𝛼
, i.e.

		
arg
⁢
max
𝛼
,
𝟎
⁢
𝝁
:


𝟏
⊤
⁢
𝝁
=
1
⁡
𝛼
+
1
2
⁢
‖
𝐽
⊤
⁢
𝝁
‖
2
−
𝝁
⊤
⁢
(
𝛼
⁢
𝟏
+
𝐽
⁢
𝐽
⊤
⁢
𝝁
)
	
	
=
	
arg
⁢
max
𝛼
,
𝟎
⁢
𝝁
:


𝟏
⊤
⁢
𝝁
=
1
⁡
𝛼
+
1
2
⁢
‖
𝐽
⊤
⁢
𝝁
‖
2
−
𝛼
⁢
𝝁
⊤
⁢
𝟏
−
𝝁
⊤
⁢
𝐽
⁢
𝐽
⊤
⁢
𝝁
	
	
=
	
arg
⁢
min
𝟎
⁢
𝝁
:


𝟏
⊤
⁢
𝝁
=
1
⁡
1
2
⁢
‖
𝐽
⊤
⁢
𝝁
‖
2
	

Therefore, (27) and (28) are equivalent, with 
𝒙
=
−
𝐽
⊤
⁢
𝒘
.

✓  Non-conflicting.

Observe that since in (28), 
𝛼
=
0
 and 
𝒙
=
𝟎
 is feasible, the objective is non-positive and therefore 
𝛼
⁢
0
. Substituting 
𝒙
=
−
𝐽
⊤
⁢
𝒘
 in 
𝐽
⋅
𝒙
⁢
𝛼
⁢
𝟏𝟎
 yields 
𝟎
⁢
𝐽
⁢
𝐽
⊤
⁢
𝒘
, i.e. 
𝟎
⁢
𝐽
⋅
𝒜
MGDA
⁢
(
𝐽
)
, so 
𝒜
MGDA
 is non-conflicting.

✗  Linear under scaling.

With 
𝐽
=
[
2
	
0


0
	
2


𝑎
	
𝑎
]
, if 
0
⁢
𝑎
⁢
1
, 
𝒜
MGDA
⁢
(
𝐽
)
=
[
𝑎


𝑎
]
. However, if 
𝑎
⁢
1
, 
𝒜
MGDA
⁢
(
𝐽
)
=
[
1


1
]
. This is not affine in 
𝑎
, so 
𝒜
MGDA
 is not linear under scaling. In particular, if any row of 
𝐽
 is 
𝟎
, 
𝒜
MGDA
⁢
(
𝐽
)
=
𝟎
. This implies that the optimization will stop whenever one objective has converged.

✓  Weighted.

By (26), 
𝒜
MGDA
 is weighted.

C.3DualProj

The projection of a gradient of interest onto a dual cone was first described in Lopez-Paz & Ranzato (2017). When this gradient is the average of the rows of the Jacobian, we call this aggregator 
𝒜
DualProj
. Formally,

	
𝒜
DualProj
⁢
(
𝐽
)
=
1
𝑚
⋅
𝜋
𝐽
⁢
(
𝐽
⊤
⋅
𝟏
)
		
(29)

where 
𝜋
𝐽
 is the projection operator defined in (3).

✓  Non-conflicting.

By the constraint in (3), 
𝒜
DualProj
 is non-conflicting.

✗  Linear under scaling.

With 
𝐽
=
[
2
	
0


−
2
⁢
𝑎
	
2
⁢
𝑎
]
, if 
𝑎
⁢
1
, 
𝒜
DualProj
⁢
(
𝐽
)
=
[
0


𝑎
]
. However, if 
0.5
⁢
𝑎
⁢
1
, 
𝒜
DualProj
⁢
(
𝐽
)
=
[
1
−
𝑎


𝑎
]
. This is not affine in 
𝑎
, so 
𝒜
DualProj
 is not linear under scaling.

✓  Weighted.

By Proposition 1, 
𝒜
DualProj
⁢
(
𝐽
)
=
1
𝑚
⁢
𝐽
⊤
⋅
𝒘
, with 
𝒘
∈
arg
⁢
min
𝟏
⁢
𝒗
⁡
𝒗
⊤
⁢
𝐽
⁢
𝐽
⊤
⁢
𝒗
. 
𝒜
DualProj
 is thus weighted.

C.4PCGrad

𝒜
PCGrad
 is described in Yu et al. (2020). It projects each gradient onto the orthogonal hyperplane of other gradients in case of conflict with them, iteratively and in random order. When 
𝑚
⁢
2
, 
𝒜
PCGrad
 is deterministic and satisfies 
𝒜
PCGrad
=
𝑚
⋅
𝒜
UPGrad
. Therefore, in this case, it satisfies all three properties. When 
𝑚
⁢
>
⁢
2
, 
𝒜
PCGrad
 is non-deterministic, so 
𝒜
PCGrad
⁢
(
𝐽
)
 is a random vector.

For any index 
𝑖
∈
[
𝑚
]
, let 
𝒈
𝑖
=
𝐽
⊤
⋅
𝒆
𝑖
 and let 
𝐩
⁢
(
𝑖
)
 be a random vector distributed uniformly on the set of permutations of the elements in 
[
𝑚
]
∖
{
𝑖
}
. For instance, if 
𝑚
=
3
, 
𝐩
⁢
(
2
)
=
[
1
	
3
]
⊤
 with probability 
0.5
 and 
𝐩
⁢
(
2
)
=
[
3
	
1
]
⊤
 with probability 
0.5
. For notation convenience, whenever 
𝑖
 is clear from context, we denote 
𝑗
𝑘
=
𝐩
⁢
(
𝑖
)
𝑘
. The iterative projection of 
𝒜
PCGrad
 is then defined recursively as:

	
𝒈
𝑖
,
1
PC
	
=
𝒈
𝑖
		
(30)

	
𝒈
𝑖
,
𝑘
+
1
PC
	
=
𝒈
𝑖
,
𝑘
PC
−
𝟙
⁢
{
𝒈
𝑖
,
𝑘
PC
⋅
𝒈
𝑗
𝑘
⁢
<
⁢
0
}
⁢
𝒈
𝑖
,
𝑘
PC
⋅
𝒈
𝑗
𝑘
‖
𝒈
𝑗
𝑘
‖
2
⁢
𝒈
𝑗
𝑘
		
(31)

We noticed that an equivalent formulation to the conditional projection of (31) is the projection onto the dual cone of 
{
𝒈
𝑗
𝑘
}
:

	
𝒈
𝑖
,
𝑘
+
1
PC
=
𝜋
𝒈
𝑗
𝑘
⊤
⁢
(
𝒈
𝑖
,
𝑘
PC
)
		
(32)

Finally, the aggregation is given by

	
𝒜
PCGrad
⁢
(
𝐽
)
=
∑
𝑖
=
1
𝑚
𝒈
𝑖
,
𝑚
PC
.
		
(33)
✗  Non-conflicting.

If 
𝐽
=
[
1
	
0


0
	
1


−
0.5
	
−
1
]
, the only non-conflicting direction is 
𝟎
. However, 
𝒜
PCGrad
⁢
(
𝐽
)
 is uniform over the set 
{
[
0.4


0.2
]
,
[
0.8


0.2
]
,
[
0.4


−
0.2
]
,
[
0.8


−
0.2
]
}
, i.e. 
𝒜
PCGrad
⁢
(
𝐽
)
 is in the dual cone of the rows of 
𝐽
 with probability 
0
. 
𝒜
PCGrad
 is thus not non-conflicting. Here, 
𝔼
⁢
[
𝒜
PCGrad
⁢
(
𝐽
)
]
=
[
0.6
	
0
]
⊤
, so 
𝒜
PCGrad
 is neither non-conflicting in expectation.

✓  Linear under scaling.

To show that 
𝒜
PCGrad
 is linear under scaling, let 
𝟎
⁢
𝒄
∈
ℝ
𝑚
, 
𝒈
𝑖
′
=
𝑐
𝑖
⁢
𝒈
𝑖
, 
𝒈
𝑖
,
1
′
⁣
 PC
=
𝒈
𝑖
′
 and 
𝒈
𝑖
,
𝑘
+
1
′
⁣
 PC
=
𝜋
𝒈
𝑗
𝑘
′
⁣
⊤
⁢
(
𝒈
𝑖
,
𝑘
′
⁣
 PC
)
. We show by induction that 
𝒈
𝑖
,
𝑘
′
⁣
 PC
=
𝑐
𝑖
⁢
𝒈
𝑖
,
𝑘
PC
.

The base case is given by 
𝒈
𝑖
,
1
′
⁣
 PC
=
𝒈
𝑖
′
=
𝑐
𝑖
⁢
𝒈
𝑖
=
𝑐
𝑖
⁢
𝒈
𝑖
,
1
PC
.

Then, assuming the induction hypothesis 
𝒈
𝑖
,
𝑘
′
⁣
 PC
=
𝑐
𝑖
⁢
𝒈
𝑖
,
𝑘
PC
, we show 
𝒈
𝑖
,
𝑘
+
1
′
⁣
 PC
=
𝑐
𝑖
⁢
𝒈
𝑖
,
𝑘
+
1
PC
:

	
𝒈
𝑖
,
𝑘
+
1
′
⁣
 PC
=
𝜋
𝑐
𝑗
𝑘
⁢
𝒈
𝑗
𝑘
⊤
⁢
(
𝑐
𝑖
⁢
𝒈
𝑖
,
𝑘
 PC
)
	
(
Induction hypothesis
)
	
	
𝒈
𝑖
,
𝑘
+
1
′
⁣
 PC
=
𝑐
𝑖
⁢
𝜋
𝒈
𝑗
𝑘
⊤
⁢
(
𝒈
𝑖
,
𝑘
 PC
)
	
(
0
⁢
<
⁢
𝑐
𝑖
⁢
 and 
⁢
0
⁢
<
⁢
𝑐
𝑗
𝑘
)
	
	
𝒈
𝑖
,
𝑘
+
1
′
⁣
 PC
=
𝑐
𝑖
⁢
𝒈
𝑖
,
𝑘
+
1
PC
	
(
By (
32
)
)
	

Therefore 
𝒜
PCGrad
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
=
∑
𝑖
=
1
𝑚
𝑐
𝑖
⁢
𝒈
𝑖
,
𝑚
PC
, so it can be written as 
𝒜
PCGrad
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
=
𝔍
⊤
⋅
𝒄
 with 
𝔍
=
[
𝒈
1
,
𝑚
PC
	
⋯
	
𝒈
𝑚
,
𝑚
PC
]
⊤
. Therefore, 
𝒜
PCGrad
 is linear under scaling.

✓  Weighted.

For all 
𝑖
, 
𝒈
𝑖
,
𝑚
PC
 is always a random linear combination of rows of 
𝐽
. 
𝒜
PCGrad
 is thus weighted.

C.5GradDrop

The aggregator used by the GradDrop layer, which we denote 
𝒜
GradDrop
, is described in Chen et al. (2020). It is non-deterministic, so 
𝒜
GradDrop
⁢
(
𝐽
)
 is a random vector. Given 
𝐽
∈
ℝ
𝑚
×
𝑛
, let 
|
𝐽
|
∈
ℝ
𝑚
×
𝑛
 be the element-wise absolute value of 
𝐽
. Let 
𝑃
=
1
2
⁢
(
𝟏
+
𝐽
⊤
⋅
𝟏
|
𝐽
|
⊤
⋅
𝟏
)
∈
ℝ
𝑛
, where the division is element-wise. Each coordinate 
𝑖
∈
[
𝑛
]
 is independently assigned to the set 
ℐ
+
 with probability 
𝑃
𝑖
 and to the set 
ℐ
−
 otherwise. The aggregation at coordinate 
𝑖
∈
ℐ
+
 is given by the sum of all positive 
𝐽
𝑗
⁢
𝑖
, for 
𝑗
∈
[
𝑚
]
. The aggregation at coordinate 
𝑖
∈
ℐ
−
 is given by the sum of all negative 
𝐽
𝑗
⁢
𝑖
, for 
𝑗
∈
[
𝑚
]
. Formally,

	
𝒜
GradDrop
⁢
(
𝐽
)
	
=
(
∑
𝑖
∈
ℐ
+
𝒆
𝑖
⁢
∑
𝑗
∈
[
𝑚
]
:


𝐽
𝑗
⁢
𝑖
⁢
>
⁢
0
𝐽
𝑗
⁢
𝑖
)
+
(
∑
𝑖
∈
ℐ
−
𝒆
𝑖
⁢
∑
𝑗
∈
[
𝑚
]
:


𝐽
𝑗
⁢
𝑖
⁢
<
⁢
0
𝐽
𝑗
⁢
𝑖
)
		
(34)
✗  Non-conflicting.

If 
𝐽
=
[
−
2


1
]
, then 
𝑃
=
[
1
/
3
]
. Therefore, 
ℙ
⁢
[
𝒜
GradDrop
⁢
(
𝐽
)
=
[
−
2
]
]
=
2
/
3
 and 
ℙ
⁢
[
𝒜
GradDrop
⁢
(
𝐽
)
=
[
1
]
]
=
1
/
3
, i.e. 
𝒜
GradDrop
⁢
(
𝐽
)
 is in the dual cone of the rows of 
𝐽
 with probability 
0
. Therefore, 
𝒜
GradDrop
 is not non-conflicting. Here, 
𝔼
⁢
[
𝒜
GradDrop
⁢
(
𝐽
)
]
=
[
−
1
]
⊤
, so 
𝒜
GradDrop
 is neither non-conflicting in expectation.

✗  Linear under scaling.

If 
𝐽
=
[
1
	
−
1


−
1
	
1
]
, then 
𝑃
=
1
2
⋅
𝟏
 and the aggregation is one of the four vectors 
[
±
1
	
±
1
]
⊤
 with equal probability. Scaling the first line of 
𝐽
 by 
2
 yields 
𝐽
=
[
2
	
−
2


−
1
	
1
]
 and 
𝑃
=
[
2
/
3
	
1
/
3
]
⊤
, which cannot lead to a uniform distribution over four elements. Therefore, 
𝒜
GradDrop
 is not linear under scaling.

✗  Weighted.

With 
𝐽
=
[
1
	
−
1


−
1
	
1
]
, the span of 
𝐽
 does not include 
[
1
	
1
]
⊤
 nor 
[
−
1
	
−
1
]
⊤
. Therefore, 
𝒜
GradDrop
 is not weighted.

C.6IMTL-G

In Liu et al. (2021b), the authors describe a method to impartially balance gradients by weighting them. Let 
𝒈
𝑖
 be the 
𝑖
’th row of 
𝐽
 and let 
𝒖
𝑖
=
𝒈
𝑖
‖
𝒈
𝑖
‖
. They want to find a combination 
𝒈
=
∑
𝑖
=
1
𝑚
𝛼
𝑖
⁢
𝒈
𝑖
 such that 
𝒈
⊤
⁢
𝒖
𝑖
 is equal for all 
𝑖
. Let 
𝑈
=
[
𝒖
1
−
𝒖
2
	
…
	
𝒖
1
−
𝒖
𝑚
]
⊤
, 
𝐷
=
[
𝒈
1
−
𝒈
2
	
…
	
𝒈
1
−
𝒈
𝑚
]
⊤
. If 
𝜶
2
:
𝑚
=
[
𝛼
2
	
…
	
𝛼
𝑚
]
⊤
, then 
𝜶
2
:
𝑚
=
(
𝑈
⁢
𝐷
⊤
)
−
1
⁢
𝑈
⋅
𝒈
1
 and 
𝛼
1
=
1
−
∑
𝑖
=
2
𝑚
𝛼
𝑖
. Notice that this is defined only when the gradients are linearly independent. Thus, this is not strictly speaking an aggregator since it can only be computed on matrices of rank 
𝑚
. We thus propose a generalization defined for matrices of any rank that is equivalent when the matrix has rank 
𝑚
. In the original formulation, requiring 
𝒈
⊤
⁢
𝒖
𝑖
 to be equal to some 
𝑐
∈
ℝ
 for all 
𝑖
, is equivalent to requiring that for all 
𝑖
, 
𝒈
⊤
⁢
𝒈
𝑖
 is equal to 
𝑐
⁢
‖
𝒈
𝑖
‖
. Writing 
𝒈
=
𝐽
⊤
⁢
𝜶
 and letting 
𝒅
∈
ℝ
𝑚
 be the vector of norms of the rows of 
𝐽
, the objective is thus to find 
𝜶
 satisfying 
𝐽
⁢
𝐽
⊤
⁢
𝜶
∝
𝒅
. Besides, to match the original formulation, the elements of 
𝜶
 should sum to 
1
.

Letting 
(
𝐽
⁢
𝐽
⊤
)
†
 be the Moore-Penrose pseudo inverse of 
𝐽
⁢
𝐽
⊤
, we define

	
𝒜
IMTL-G
⁢
(
𝐽
)
	
=
𝐽
⊤
⋅
𝒘
		
(35)

	
with
𝒘
	
=
{
𝒗
𝟏
⊤
⁢
𝒗
,
	
if 
⁢
𝟏
⊤
⁢
𝒗
≠
𝟎


𝟎
,
	
otherwise
		
(36)

	
and
𝒗
	
=
(
𝐽
⁢
𝐽
⊤
)
†
⋅
𝒅
.
		
(37)
✗  Non-conflicting.

If 
𝐽
=
[
1
	
−
1
	
−
1
]
⊤
, then 
𝒅
=
[
1
	
1
	
1
]
⊤
, 
𝐽
⁢
𝐽
⊤
=
[
1
	
−
1
	
−
1


−
1
	
1
	
1


−
1
	
1
	
1
]
, and thus 
𝒗
=
1
9
⁢
[
−
1
	
1
	
1
]
⊤
. Therefore, 
𝒘
=
[
−
1
	
1
	
1
]
⊤
, 
𝒜
IMTL-G
⁢
(
𝐽
)
=
[
−
3
]
⊤
 and 
𝐽
⋅
𝒜
IMTL-G
⁢
(
𝐽
)
=
[
−
3
	
3
	
3
]
⊤
. 
𝒜
IMTL-G
 is thus not non-conflicting.

It should be noted that when 
𝐽
 has rank 
𝑚
, 
𝒜
IMTL-G
 seems to be non-conflicting. Thus, it would be possible to make a different non-conflicting generalization, for instance, by deciding 
𝟎
 when 
𝐽
 is not full rank.

✗  Linear under scaling.

With 
𝐽
=
[
𝑎
	
0


0
	
1
]
 and 
𝑎
⁢
>
⁢
0
, we have 
𝒗
=
[
1
/
𝑎
2
	
0


0
	
1
]
⋅
[
𝑎


1
]
=
[
1
/
𝑎


1
]
 and 
𝒜
IMTL-G
⁢
(
𝐽
)
=
[
𝑎
	
0


0
	
1
]
⋅
[
1
/
𝑎


1
]
⋅
1
1
𝑎
+
1
=
1
1
𝑎
+
1
⋅
[
1


1
]
. This is not affine in 
𝑎
, so 
𝒜
IMTL-G
 is not linear under scaling.

✓  Weighted.

By (35), 
𝒜
IMTL-G
 is weighted.

C.7CAGrad

𝒜
CAGrad
 is described in Liu et al. (2021a). It is parameterized by 
𝑐
∈
[
0
,
1
[
. If 
𝑐
=
0
, this is equivalent to 
𝒜
Mean
. Therefore, we restrict our analysis to the case 
𝑐
⁢
>
⁢
0
. For any 
𝐽
∈
ℝ
𝑚
×
𝑛
, let 
𝒈
¯
 be the average gradient 
1
𝑚
⁢
𝐽
⊤
⋅
𝟏
, and let 
𝒆
𝑖
⊤
⁢
𝐽
 denote the 
𝑖
’th row of 
𝐽
. The aggregation is then defined as

	
𝒜
CAGrad
⁢
(
𝐽
)
∈
arg
⁢
max
𝒅
∈
ℝ
𝑛
:


‖
𝒅
−
𝒈
¯
‖
⁢
𝑐
⁢
‖
𝒈
¯
‖
⁡
min
𝑖
∈
[
𝑚
]
⁡
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
		
(38)
✗  Non-conflicting.

Let 
𝐽
=
[
2
	
0


−
2
⁢
𝑎
−
2
	
2
]
, with 
𝑎
 satisfying 
−
𝑎
+
𝑐
⁢
𝑎
2
+
1
⁢
<
⁢
0
. We have 
𝒈
¯
=
[
−
𝑎
	
1
]
⊤
 and 
‖
𝒈
¯
‖
=
𝑎
2
+
1
. Observe that any 
𝒅
∈
ℝ
𝑛
 satisfying the constraint 
‖
𝒅
−
𝒈
¯
‖
⁢
𝑐
⁢
‖
𝒈
¯
‖
 has first coordinate at most 
−
𝑎
+
𝑐
⁢
𝑎
2
+
1
. Because 
−
𝑎
+
𝑐
⁢
𝑎
2
+
1
⁢
<
⁢
0
, any feasible 
𝑑
 has a negative first coordinate, making 
𝑑
 conflict with the first row of 
𝐽
. For any 
𝑐
∈
[
0
,
1
[
, 
−
𝑎
+
𝑐
⁢
𝑎
2
+
1
⁢
<
⁢
0
 is equivalent to 
𝑐
2
1
−
𝑐
2
⁢
<
⁢
𝑎
. Thus, this provides a counter-example to the non-conflicting property for any 
𝑐
∈
[
0
,
1
[
, i.e. 
𝒜
CAGrad
 is not non-conflicting.

If we generalize to the case 
𝑐
⁢
1
, as suggested in the original paper, then 
𝒅
=
𝟎
 becomes feasible, which yields 
min
𝑖
∈
[
𝑚
]
⁡
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
=
0
. Therefore the optimal 
𝒅
 satisfies 
0
⁢
min
𝑖
∈
[
𝑚
]
⁡
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
, i.e. 
𝟎
⁢
𝐽
⁢
𝒅
. With 
𝑐
⁢
1
, 
𝒜
CAGrad
 would thus be non-conflicting.

✗  Linear under scaling (sketch of proof).

Let 
𝐽
=
[
2
	
0


0
	
2
⁢
𝑎
]
, then 
𝒈
¯
=
[
1
	
𝑎
]
⊤
 and 
‖
𝒈
¯
‖
=
1
+
𝑎
2
. One can show that the constraint 
‖
𝒅
−
𝒈
¯
‖
⁢
𝑐
⁢
‖
𝒈
¯
‖
 needs to be satisfied with equality since, otherwise, we can scale 
𝒅
 to make the objective larger. Substituting 
𝐽
 in 
min
𝑖
∈
[
𝑚
]
⁡
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
 yields 
2
⁢
min
⁡
(
𝑑
1
,
𝑎
⁢
𝑑
2
)
. For any 
𝑎
 satisfying 
𝑐
⁢
1
+
𝑎
2
+
1
⁢
<
⁢
𝑎
2
, it can be shown that the optimal 
𝒅
 satisfies 
𝑑
1
⁢
<
⁢
𝑎
⁢
𝑑
2
. In that case the inner minimum over 
𝑖
 is 
2
⁢
𝑑
1
 and, to satisfy 
‖
𝒅
−
𝒈
¯
‖
=
𝑐
⁢
‖
𝒈
¯
‖
, the KKT conditions over the Lagrangian yield 
𝒅
−
𝒈
¯
∝
∇
𝒅
𝑑
1
=
[
1
	
0
]
⊤
. This yields 
𝒅
=
[
𝑐
⋅
‖
𝒈
¯
‖
+
1


𝑎
]
=
[
𝑐
⁢
1
+
𝑎
2
+
1


𝑎
]
. This is not affine in 
𝑎
; therefore, 
𝒜
CAGrad
 is not linear under scaling.

✓  Weighted.

In Liu et al. (2021a), 
𝒜
CAGrad
 is formulated via its dual: 
𝒜
CAGrad
⁢
(
𝐽
)
=
1
𝑚
⁢
𝐽
⊤
⁢
(
𝟏
+
𝑐
⁢
‖
𝐽
⊤
⁢
𝟏
‖
‖
𝐽
⊤
⁢
𝒘
‖
⁢
𝒘
)
, with 
𝒘
∈
arg
⁢
min
𝒘
∈
Δ
⁢
(
𝑚
)
⁡
𝟏
⊤
⁢
𝐽
⁢
𝐽
⊤
⁢
𝒘
+
𝑐
⋅
‖
𝐽
⊤
⁢
𝟏
‖
⋅
‖
𝐽
⊤
⁢
𝒘
‖
, where is 
Δ
⁢
(
𝑚
)
 the probability simplex of dimension 
𝑚
. Therefore, 
𝒜
CAGrad
 is weighted.

C.8RGW

𝒜
RGW
 is defined in Lin et al. (2021) as the weighted sum of the rows of the input matrix, with a random weighting. The weighting is obtained by sampling 
𝑚
 i.i.d. normally distributed random variables and applying a softmax. Formally,

	
𝒜
RGW
⁢
(
𝐽
)
	
=
𝐽
⊤
⋅
softmax
⁢
(
𝐰
)
		
(39)

	
with
𝐰
	
∼
𝒩
⁢
(
𝟎
,
𝐼
)
		
(40)
✗  Non-conflicting.

When 
𝐽
=
[
1


−
2
]
, the only non-conflicting solution is 
𝟎
. However, 
ℙ
⁢
[
𝒜
RGW
⁢
(
𝐽
)
=
𝟎
]
=
0
, i.e. 
𝒜
RGW
⁢
(
𝐽
)
 is in the dual cone of the rows of 
𝐽
 with probability 
0
. 
𝒜
RGW
 is thus not non-conflicting. Here,

	
𝔼
⁢
[
𝒜
RGW
⁢
(
𝐽
)
]
	
=
𝔼
⁢
[
𝑒
w
1
−
2
⁢
𝑒
w
2
𝑒
w
1
+
𝑒
w
2
]
	
(
By (
39
)
)
	
		
=
−
𝔼
⁢
[
𝑒
w
1
𝑒
w
1
+
𝑒
w
2
]
	
(
𝐰
∼
𝒩
⁢
(
𝟎
,
𝐼
)
)
	
		
<
⁢
0
	
(
0
⁢
𝑒
𝐰
)
	

so 
𝒜
RGW
 is neither non-conflicting in expectation.

✓  Linear under scaling.

𝒜
RGW
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
=
(
diag
(
𝒄
)
⋅
𝐽
)
⊤
⋅
softmax
⁢
(
𝐰
)
=
𝐽
⊤
⋅
diag
(
𝒄
)
⋅
softmax
⁢
(
𝐰
)
=
𝐽
⊤
⋅
diag
(
softmax
⁢
(
𝐰
)
)
⋅
𝒄
. We thus have 
𝒜
RGW
⁢
(
diag
(
𝒄
)
⋅
𝐽
)
=
𝔍
⊤
⋅
𝒄
 with 
𝔍
⊤
=
𝐽
⊤
⋅
diag
(
softmax
⁢
(
𝐰
)
)
. Therefore, 
𝒜
RGW
 is linear under scaling.

✓  Weighted.

By (39), 
𝒜
RGW
 is weighted.

C.9Nash-MTL

Nash-MTL is described in Navon et al. (2022). Unfortunately, we were not able to verify the proof of Claim 3.1, and we believe that the official implementation of Nash-MTL may mismatch the desired objective by which it is defined. Therefore, we only analyze the initial objective even though our experiments for this aggregator are conducted with the official implementation.

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
 and 
𝜀
⁢
>
⁢
0
. Let also 
𝐵
𝜀
=
{
𝒅
∈
ℝ
𝑛
:
‖
𝒅
‖
⁢
𝜀
,
𝟎
⁢
𝐽
⁢
𝒅
}
. With 
𝒆
𝑖
⊤
⁢
𝐽
 denoting the 
𝑖
’th row of 
𝐽
, 
𝒜
Nash-MTL
 is then defined as

	
𝒜
Nash-MTL
⁢
(
𝐽
)
=
arg
⁢
max
𝒅
∈
𝐵
𝜀
⁢
∑
𝑖
∈
[
𝑚
]
log
⁡
(
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
)
		
(41)
✓  Non-conflicting.

By the constraint, 
𝒜
Nash-MTL
 is non-conflicting.

✗  Linear under scaling.

If an aggregator 
𝒜
 is linear under scaling, it should be the case that 
𝒜
⁢
(
𝑎
⁢
𝐽
)
=
𝑎
⁢
𝒜
⁢
(
𝐽
)
 for any scalar 
𝑎
⁢
>
⁢
0
 and any 
𝐽
∈
ℝ
𝑚
×
𝑛
. However, 
log
⁡
(
𝑎
⁢
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
)
=
log
⁡
(
𝒆
𝑖
⊤
⁢
𝐽
⁢
𝒅
)
+
log
⁡
(
𝑎
)
. This means that scaling by a scalar does not impact aggregation. Since this is not the trivial 
𝟎
 aggregator, 
𝒜
Nash-MTL
 is not linear under scaling.

✓  Weighted.

Suppose towards contradiction that 
𝒅
 is both optimal for (41) and not in the span of 
𝐽
⊤
. Let 
𝒅
′
 be the projection of 
𝒅
 onto the span of 
𝐽
⊤
. Since 
‖
𝒅
′
‖
⁢
<
⁢
‖
𝒅
‖
⁢
<
⁢
𝜀
 and 
𝐽
⁢
𝒅
=
𝐽
⁢
𝒅
′
, we have 
𝐽
⁢
𝒅
⁢
𝐽
⁢
(
‖
𝒅
‖
‖
𝒅
′
‖
⁢
𝒅
′
)
, contradicting the optimality of 
𝒅
. Therefore, 
𝒜
Nash-MTL
 is weighted.

C.10Aligned-MTL

The Aligned-MTL method for balancing the Jacobian is described in Senushkin et al. (2023). For simplicity, we fix the vector of preferences to 
1
𝑚
⁢
𝟏
, but the proofs can be adapted for any non-trivial vector. Given 
𝐽
∈
ℝ
𝑚
×
𝑛
, let 
𝑉
⁢
Σ
2
⁢
𝑉
⊤
 be the eigen-decomposition of 
𝐽
⁢
𝐽
⊤
, let 
Σ
†
 be the diagonal matrix whose non-zero elements are the inverse of corresponding non-zero diagonal elements of 
Σ
 and let 
𝜎
min
=
min
𝑖
∈
[
𝑚
]
,
Σ
𝑖
⁢
𝑖
≠
0
⁡
Σ
𝑖
⁢
𝑖
. The aggregation is then defined as

	
𝒜
Aligned-MTL
⁢
(
𝐽
)
	
=
1
𝑚
⁢
𝐽
⊤
⋅
𝒘
		
(42)

	
with
𝒘
	
=
𝜎
min
⋅
𝑉
⁢
Σ
†
⁢
𝑉
⊤
⋅
𝟏
		
(43)
✗  Non-conflicting.

If the SVD of 
𝐽
 is 
𝑉
⁢
Σ
⁢
𝑈
⊤
, then 
𝐽
⊤
⁢
𝒘
=
𝜎
min
⁢
𝑈
⁢
𝑃
⁢
𝑉
⊤
⁢
𝟏
 with 
𝑃
=
Σ
†
⁢
Σ
 a diagonal projection matrix with 
1
s corresponding to non zero elements of 
Σ
 and 
0
s everywhere else. Further, 
𝐽
⋅
𝒜
Aligned-MTL
⁢
(
𝐽
)
=
𝜎
min
𝑚
⋅
𝑉
⁢
Σ
⁢
𝑉
⊤
⁢
𝟏
. If 
𝑉
=
1
2
⁢
[
3
	
1


−
1
	
3
]
 and 
Σ
=
[
1
	
0


0
	
0
]
, we have 
𝐽
⋅
𝒜
Aligned-MTL
⁢
(
𝐽
)
=
1
2
⋅
𝑉
⁢
Σ
⁢
𝑉
⊤
⁢
𝟏
=
1
8
⁢
[
3
−
3
	
1
−
3
]
⊤
 which is not non-negative. 
𝒜
Aligned-MTL
 is thus not non-conflicting.

✗  Linear under scaling.

If 
𝐽
=
[
1
	
0


0
	
𝑎
]
, then 
𝑈
=
𝑉
=
𝐼
, 
Σ
=
𝐽
. For 
0
⁢
<
⁢
𝑎
⁢
1
, 
𝜎
min
=
𝑎
, therefore 
𝒜
Aligned-MTL
⁢
(
𝐽
)
=
𝑎
2
⋅
𝟏
. For 
1
⁢
<
⁢
𝑎
, 
𝜎
min
=
1
 and therefore 
𝒜
Aligned-MTL
⁢
(
𝐽
)
=
1
2
⋅
𝟏
, which makes 
𝒜
Aligned-MTL
 not linear under scaling.

✓  Weighted.

By (42), 
𝒜
Aligned-MTL
 is weighted.

Appendix DExperimental settings

For all of our experiments, we used PyTorch (Paszke et al., 2019). We have developed an open-source library5 on top of it to enable Jacobian descent easily. This library is designed to be reusable for many other use cases than the experiments presented in our work. To separate them from the library, the experiments have been conducted with a different code repository6 mainly using PyTorch and our library.

D.1Learning rate selection

The learning rate has a very important impact on the speed of optimization. To make the comparisons as fair as possible, we always show the results corresponding to the best learning rate. We have selected the area under the loss curve as the criterion to compare the learning rates. This choice is arbitrary but seems to work well in practice: a lower area under the loss curve means the optimization is fast (quick loss decrease) and stable (few bumps in the loss curve). Concretely, for each random rerun and each aggregator, we first try 22 learning rates from 
10
−
5
 to 
10
2
, increasing by a factor 
10
1
3
 every time. The two best learning rates from this range then define a refined range of plausible good learning rates, going from the smallest of those two multiplied by 
10
−
1
3
 to the largest of those two multiplied by 
10
1
3
. This margin makes it unlikely for the best learning rate to lie out of the refined range. After this, 50 learning rates from the refined range are tried. These learning rates are evenly spaced in the exponent domain. The one with the best area under the loss curve is then selected and presented in the plots. For simplicity, we have always used a constant learning rate, i.e. no learning rate scheduler was used.

This approach has the advantage of being simple and precise, thus giving trustworthy results. However, it requires 72 trainings for each aggregator, random rerun, and dataset, i.e. a total of 
43776
 trainings for all of our experiments. For this reason, we have opted to work on small subsets of the original datasets.

D.2Random reruns and standard error of the mean

To get an idea of confidence in our results, every experiment is performed 8 times on a different seed and a different subset, of size 1024, of the training dataset. The seed used for run 
𝑖
∈
[
8
]
 is always simply set to 
𝑖
. Because each random rerun includes the full learning rate selection method described in Appendix D.1, it is sensible to consider the 8 sets of results as i.i.d. For each point of both the loss curves and the cosine similarity curves, we thus compute the estimated standard error of the mean with the usual formula 
1
8
⁢
∑
𝑖
∈
[
8
]
(
𝑣
𝑖
−
𝑣
¯
)
2
8
−
1
, where 
𝑣
𝑖
 is the value of a point of the curve for random rerun 
𝑖
, and 
𝑣
¯
 is the average value of this point over the 8 runs.

D.3Model architectures

In all experiments, the models are simple convolutional neural networks. All convolutions always have a stride of 1
×
1, a kernel size of 3
×
3, a learnable bias, and no padding. All linear layers always have a learnable bias. The activation function is the exponential linear unit (Clevert et al., 2015). The full architectures are given in Tables 2, 3, 4 and 5. Note that these architectures have been fixed arbitrarily, i.e. they were not optimized through some hyper-parameter selection. The weights of the model have been initialized with the default initialization scheme of PyTorch.

Table 2:Architecture used for SVHN
Conv2d (3 input channels, 16 output channels, 1 group), ELU 
Conv2d (16 input channels, 32 output channels, 16 groups)
MaxPool2d (stride of 2
×
2, kernel size of 2
×
2), ELU 
Conv2d (32 input channels, 32 output channels, 32 groups)
MaxPool2d (stride of 3
×
3, kernel size of 3
×
3), ELU, Flatten 
Linear (512 input features, 64 output features), ELU 
Linear (64 input features, 10 outputs)
Table 3:Architecture used for CIFAR-10
Conv2d (3 input channels, 32 output channels, 1 group), ELU 
Conv2d (32 input channels, 64 output channels, 32 groups)
MaxPool2d (stride of 2
×
2, kernel size of 2
×
 2), ELU 
Conv2d (64 input channels, 64 output channels, 64 groups)
MaxPool2d (stride of 3
×
3, kernel size of 3
×
3), ELU, Flatten 
Linear (1024 input features, 128 output features), ELU 
Linear (128 input features, 10 outputs)
Table 4:Architecture used for EuroSAT
Conv2d (3 input channels, 32 output channels, 1 group)
MaxPool2d (stride of 2
×
2, kernel size of 2
×
2), ELU 
Conv2d (32 input channels, 64 output channels, 32 groups)
MaxPool2d (stride of 2
×
2, kernel size of 2
×
2), ELU 
Conv2d (64 input channels, 64 output channels, 64 groups)
MaxPool2d (stride of 3
×
3, kernel size of 3
×
3), ELU, Flatten 
Linear (1024 input features, 128 output features), ELU 
Linear (128 input features, 10 outputs)
Table 5:Architecture used for MNIST, Fashion-MNIST and Kuzushiji-MNIST
Conv2d (1 input channel, 32 output channels, 1 group), ELU 
Conv2d (32 input channels, 64 output channels, 1 group)
MaxPool2d (stride of 2
×
2, kernel size of 2
×
2), ELU 
Conv2d (64 input channels, 64 output channels, 1 group)
MaxPool2d (stride of 3
×
3, kernel size of 3
×
3), ELU, Flatten 
Linear (576 input features, 128 output features), ELU 
Linear (128 input features, 10 outputs)
D.4Optimizer

For all experiments except those described in Appendix A.4, we always use the basic SGD optimizer of PyTorch, without any regularization or momentum. Here, SGD refers to the PyTorch optimizer that updates the parameters of the model in the opposite direction of the gradient, which, in our case, is replaced by the aggregation of the Jacobian matrix. In the rest of this paper, SGD refers to the whole stochastic gradient descent algorithm. In the experiments of Appendix A.4, we instead use Adam to study its interactions with JD.

D.5Loss function

The loss function is always the usual cross-entropy, with the default parameters of PyTorch.

D.6Preprocessing

The inputs are always normalized per channel based on the mean and standard deviation computed on the entire training split of the dataset.

D.7Iterations and computational budget

The numbers of epochs and the corresponding numbers of iterations for all datasets are provided in Table 6, along with the required number of NVIDIA L4 GPU-hours, to run all 72 learning rates for the 11 aggregators on a single seed. The total computational budget to run the main experiments on 8 seeds was thus around 760 GPU-hours. Additionally, we used a total of about 100 GPU hours for the experiments varying the batch size and using Adam, and about 200 more GPU hours were used for early investigations.

Table 6:Numbers of epochs, iterations, and GPU-hours for each dataset
Dataset	Epochs	Iterations	GPU-Hours
SVHN	25	800	17
CIFAR-10	20	640	15
EuroSAT	30	960	32
MNIST	8	256	6
Fashion-MNIST	25	800	17
Kuzushiji-MNIST	10	320	8
Appendix EComputation time and memory usage
E.1Memory considerations of SSJD for IWRM

The main overhead of SSJD on the IWRM objective comes from having to store the full Jacobian in memory. Remarkably, when we use SGD with ERM, every activation is a tensor whose first dimension is the batch size. Automatic differentiation engines thus have to compute the Jacobian anyway. Since the gradients can be averaged at each layer as soon as they are obtained, the full Jacobian does not have to be stored. In the naive implementation of SSJD, however, storing the Jacobian costs memory, which given the high parallelization ability of GPUs, increases the computational time. With the Gramian-based method proposed in Section 6, only the Gramian, which is typically small, has to be stored: the memory requirement would then be similar to that of SGD.

E.2Time complexity of the unconflicting projection of gradients

Let 
𝐽
∈
ℝ
𝑚
×
𝑛
 be the matrix to aggregate. Apart from computing the Gramian 
𝐽
⁢
𝐽
⊤
, applying 
𝒜
UPGrad
 to 
𝐽
 requires solving 
𝑚
 instances of the quadratic program (5) of Proposition 1. Solvers for such problems of dimension 
𝑚
 typically have a computational complexity upper bounded by 
𝒪
⁢
(
𝑚
4
)
 or less in recent implementations (e.g. 
𝒪
⁢
(
𝑚
3.67
⁢
log
⁡
𝑚
)
). This induces a 
𝒪
⁢
(
𝑚
5
)
 time complexity for extracting the weights of 
𝒜
UPGrad
. Note that solving these 
𝑚
 problems in parallel would reduce this complexity to 
𝒪
⁢
(
𝑚
4
)
.

E.3Empirical computational times

In Table 7, we compare the computation time of SGD with that of SSJD for all the aggregators that we experimented with. Since we used the same architecture for MNIST, Fashion-MNIST and Kuzushiji-MNIST, we only report the results for one of them. Several factors affect this computation time. First, the batch size affects the number of rows in the Jacobian to aggregate. Increasing the batch size thus requires more GPU memory and the aggregation of a taller matrix. Then, some aggregators, e.g. 
𝒜
Nash-MTL
 and 
𝒜
MGDA
, seem to greatly increase the run time. When the aggregation is the bottleneck, a faster implementation will be necessary to make them usable in practice. Lastly, the current implementation of JD in our library is still fairly inefficient in terms of memory management, which in turn limits how well the GPU can parallelize. Also, our implementation of 
𝒜
UPGrad
 does not solve the 
𝑚
 quadratic programs in parallel. Therefore, these results just give a rough indication of the current computation times.

Table 7:Time required in seconds for one epoch of training on the ERM objective with SGD and on the IWRM objective with SSJD and different aggregators, on an NVIDIA L4 GPU. The batch size is always 32.
Objective	Method	SVHN	CIFAR-10	EuroSAT	MNIST
ERM	SGD	0.79	0.50	0.81	0.47
IWRM	SSJD–
𝒜
Mean
	1.41	1.76	2.93	1.64
IWRM	SSJD–
𝒜
MGDA
	5.50	5.22	6.91	5.22
IWRM	SSJD–
𝒜
DualProj
	1.51	1.88	3.02	1.76
IWRM	SSJD–
𝒜
PCGrad
	2.78	3.13	4.18	3.01
IWRM	SSJD–
𝒜
GradDrop
	1.57	1.90	3.06	1.78
IWRM	SSJD–
𝒜
IMTL-G
	1.48	1.79	2.94	1.69
IWRM	SSJD–
𝒜
CAGrad
	1.93	2.26	3.42	2.17
IWRM	SSJD–
𝒜
RGW
	1.42	1.76	2.89	1.73
IWRM	SSJD–
𝒜
Nash-MTL
	7.88	8.12	9.33	7.91
IWRM	SSJD–
𝒜
Aligned-MTL
	1.53	1.98	2.97	1.71
IWRM	SSJD–
𝒜
UPGrad
	1.80	2.01	3.21	1.90
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
 I 