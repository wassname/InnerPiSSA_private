Title: 

URL Source: https://arxiv.org/pdf/1905.12558

Published Time: Mon, 23 Jan 2023 01:02:55 GMT

Markdown Content:
# Limitations of the Empirical Fisher Approximation for Natural Gradient Descent 

Frederik Kunstner 1,2,3

kunstner@cs.ubc.ca 

Lukas Balles 2,3

lballes@tue.mpg.de 

Philipp Hennig 2,3

ph@tue.mpg.de 

École Polytechnique Fédérale de Lausanne (EPFL), Switzerland 1

University of Tübingen, Germany 2

Max Planck Institute for Intelligent Systems, Tübingen, Germany 3

## Abstract 

Natural gradient descent, which preconditions a gradient descent update with the Fisher information matrix of the underlying statistical model, is a way to capture partial second-order information. Several highly visible works have advocated an approximation known as the empirical Fisher, drawing connections between approximate second-order methods and heuristics like Adam. We dispute this argument by showing that the empirical Fisher—unlike the Fisher—does not generally capture second-order information. We further argue that the conditions under which the empirical Fisher approaches the Fisher (and the Hessian) are unlikely to be met in practice, and that, even on simple optimization problems, the pathologies of the empirical Fisher can have undesirable effects. 

## 1 Introduction 

Consider a supervised machine learning problem of predicting outputs y ∈ Y from inputs x ∈ X.We assume a probabilistic model for the conditional distribution of the form pθ (y|x) = p(y|f (x, θ )) ,where p(y|· ) is an exponential family with natural parameters in F and f : X×RD → F is a prediction function parameterized by θ ∈ RD . Given N iid training samples (xn, y n)Nn=1 , we want to minimize 

L(θ) := − ∑ 

> n

log pθ (yn|xn) = − ∑ 

> n

log p(yn|f (xn, θ )) . (1) This framework covers common scenarios such as least-squares regression ( Y = F = R and p(y|f ) = 

N (y; f, σ 2) with fixed σ2) or C-class classification with cross-entropy loss ( Y = {1, . . . , C },

F = RC and p(y = c|f ) = exp( fc)/ ∑ 

> i

exp( fi)) with an arbitrary prediction function f . Eq. (1) can be minimized by gradient descent, which updates θt+1 = θt − γt∇ L (θt) with step size γt ∈ R.This update can be preconditioned with a matrix Bt that incorporates additional information, such as local curvature, θt+1 = θt − γtBt−1∇ L (θt). Choosing Bt to be the Hessian yields Newton’s method, but its computation is often burdensome and might not be desirable for non-convex problems. A prominent variant in machine learning is natural gradient descent [NGD; Amari, 1998]. It adapts to the information geometry of the problem by measuring the distance between parameters with the Kullback–Leibler divergence between the resulting distributions rather than their Euclidean distance, using the Fisher information matrix (or simply “Fisher”) of the model as a preconditioner, 

F( θ) := ∑ 

> n

Epθ (y|xn)

[∇θ log pθ (y|xn) ∇θ log pθ (y|xn)>] . (2) While this motivation is conceptually distinct from approximating the Hessian, the Fisher coincides with a generalized Gauss-Newton [Schraudolph, 2002] approximation of the Hessian for the problems presented here. This gives NGD theoretical grounding as an approximate second-order method. A number of recent works in machine learning have relied on a certain approximation of the Fisher, which is often called the empirical Fisher (EF) and is defined as 

˜F( θ) := ∑ 

> n

∇θ log pθ (yn|xn) ∇θ log pθ (yn|xn)>. (3) 

Code available at github.com/fkunstner/limitations-empirical-fisher. 

> arXiv:1905.12558v3 [cs.LG] 8 Jun 2020

Dataset     

> y=θx +b

GD NGD EF Figure 1: Fisher vs. empirical Fisher as preconditioners for linear least-squares regression on the data shown in the left-most panel. The second plot shows the gradient vector field of the (quadratic) loss function and sample trajectories for gradient descent. The remaining plots depict the vector fields of the natural gradient and the “EF-preconditioned” gradient, respectively. NGD successfully adapts to the curvature whereas preconditioning with the empirical Fisher results in a distorted gradient field. At first glance, this approximation is merely replacing the expectation over y in Eq. (2) with a sample 

yn. However, yn is a training label and not a sample from the model’s predictive distribution pθ (y|xn).Therefore, and contrary to what its name suggests, the empirical Fisher is not an empirical (i.e. Monte Carlo) estimate of the Fisher. Due to the unclear relationship between the model distribution and the data distribution, the theoretical grounding of the empirical Fisher approximation is dubious. Adding to the confusion, the term “empirical Fisher” is used by different communities to refer to different quantities. Authors closer to statistics tend to use “empirical Fisher” for Eq. (2) , while many works in machine learning, some listed in Section 2, use “empirical Fisher” for Eq. (3) . While the statistical terminology is more accurate, we adopt the term “Fisher” for Eq. (2) and “empirical Fisher” for Eq. (3) , which is the subject of this work, to be accessible to readers more familiar with this convention. We elaborate on the different uses of the terminology in Section 3.1. The main purpose of this work is to provide a detailed critical discussion of the empirical Fisher approximation. While the discrepancy between the empirical Fisher and the Fisher has been mentioned in the literature before [Pascanu and Bengio, 2014, Martens, 2014], we see the need for a detailed elaboration of the subtleties of this important issue. The intricacies of the relationship between the empirical Fisher and the Fisher remain opaque from the current literature. Not all authors using the EF seem to be fully aware of the heuristic nature of this approximation and overlook its shortcomings, which can be seen clearly even on simple linear regression problems, see Fig. 1. Natural gradients adapt to the curvature of the function using the Fisher while the empirical Fisher distorts the gradient field in a way that lead to worse updates than gradient descent. The empirical Fisher approximation is so ubiquitous that it is sometimes just called the Fisher [e.g., Chaudhari et al., 2017, Wen et al., 2019]. Possibly as a result of this, there are examples of algorithms involving the Fisher, such as Elastic Weight Consolidation [Kirkpatrick et al., 2017] and KFAC [Martens and Grosse, 2015], which have been re-implemented by third parties using the empirical Fisher. Interestingly, there is also at least one example of an algorithm that was originally developed using the empirical Fisher and later found to work better with the Fisher [Wierstra et al., 2008, Sun et al., 2009]. As the empirical Fisher is now used beyond optimization, for example as an approximation of the Hessian in empirical works studying properties of neural networks [Chaudhari et al., 2017, Jastrz˛ ebski et al., 2018], the pathologies of the EF approximation may lead the community to erroneous conclusions—an arguably more worrysome outcome than a suboptimal preconditioner. The poor theoretical grounding stands in stark contrast to the practical success that empirical Fisher-based methods have seen. This paper is in no way meant to negate these practical advances but rather points out that the existing justifications for the approximation are insufficient and do not stand the test of simple examples. This indicates that there are effects at play that currently elude our understanding, which is not only unsatisfying, but might also prevent advancement of these methods. We hope that this paper helps spark interest in understanding these effects; our final section explores a possible direction. 21.1 Overview and contributions 

We first provide a short but complete overview of natural gradient and the closely related generalized Gauss-Newton method. Our main contribution is a critical discussion of the specific arguments used to advocate the empirical Fisher approximation. A principal conclusion is that, while the empirical Fisher follows the formal definition of a generalized Gauss-Newton matrix, it is not guaranteed to capture any useful second-order information. We propose a clarifying amendment to the definition of a generalized Gauss-Newton to ensure that all matrices satisfying it have useful approximation properties. Furthermore, while there are conditions under which the empirical Fisher approaches the true Fisher, we argue that these are unlikely to be met in practice. We illustrate that using the empirical Fisher can lead to highly undesirable effects; Fig. 1 shows a first example. This raises the question: Why are methods based on the empirical Fisher practically successful? We point to an alternative explanation, as an adaptation to gradient noise in stochastic optimization instead of an adaptation to curvature. 

## 2 Related work 

The generalized Gauss-Newton [Schraudolph, 2002] and natural gradient descent [Amari, 1998] methods have inspired a line of work on approximate second-order optimization [Martens, 2010, Botev et al., 2017, Park et al., 2000, Pascanu and Bengio, 2014, Ollivier, 2015]. A successful example in modern deep learning is the KFAC algorithm [Martens and Grosse, 2015], which uses a computationally efficient structural approximation to the Fisher. Numerous papers have relied on the empirical Fisher approximation for preconditioning and other purposes. Our critical discussion is in no way intended as an invalidation of these works. All of them provide important insights and the use of the empirical Fisher is usually not essential to the main contribution. However, there is a certain degree of vagueness regarding the relationship between the Fisher, the EF, Gauss-Newton matrices and the Hessian. Oftentimes, only limited attention is devoted to possible implications of the empirical Fisher approximation. The most prominent example of preconditioning with the EF is Adam, which uses a moving average of squared gradients as “an approximation to the diagonal of the Fisher information matrix” [Kingma and Ba, 2015]. The EF has been used in the context of variational inference by various authors [Graves, 2011, Zhang et al., 2018, Salas et al., 2018, Khan et al., 2018, Mishkin et al., 2018], some of which have drawn further connections between NGD and Adam. There are also several works building upon KFAC which substitute the EF for the Fisher [George et al., 2018, Osawa et al., 2019]. The empirical Fisher has also been used as an approximation of the Hessian for purposes other than preconditioning. Chaudhari et al. [2017] use it to investigate curvature properties of deep learning training objectives. It has also been employed to explain certain characteristics of SGD [Zhu et al., 2019, Jastrz˛ ebski et al., 2018] or as a diagnostic tool during training [Liao et al., 2020]. Le Roux et al. [2007] and Le Roux and Fitzgibbon [2010] have considered the empirical Fisher in its interpretation as the (non-central) covariance matrix of stochastic gradients. While they refer to their method as “Online Natural Gradient”, their goal is explicitly to adapt the update to the stochasticity 

of the gradient estimate, not to curvature . We will return to this perspective in Section 5. Before moving on, we want to re-emphasize that other authors have previously raised concerns about the empirical Fisher approximation [e.g., Pascanu and Bengio, 2014, Martens, 2014]. This paper is meant as a detailed elaboration of this known but subtle issue, with novel results and insights. Concurrent to our work, Thomas et al. [2019] investigated similar issues in the context of estimating the generalization gap using information criteria. 

## 3 Generalized Gauss-Newton and natural gradient descent 

This section briefly introduces natural gradient descent, adresses the difference in terminology for the quantities of interest across fields, introduces the generalized Gauss-Newton (GGN) and reviews the connections between the Fisher, the GGN, and the Hessian. 3Quantity Terminology in statistics and machine learning 

F∏   

> npθ(x,y )

Eq. (5) Fisher 

F∏   

> npθ(y|xn)

Eq. (6) empirical Fisher Fisher 

˜F Eq. (7) empirical Fisher Table 1: Common terminology for the Fisher information and related matrices by authors closely aligned with statistics, such as Amari [1998], Park et al. [2000], and Karakida et al. [2019], or machine learning, such as Martens [2010], Schaul et al. [2013], and Pascanu and Bengio [2014]. 

3.1 Natural gradient descent 

Gradient descent follows the direction of “steepest descent”, the negative gradient. But the definition of steepest depends on a notion of distance and the gradient is defined with respect to the Euclidean distance. The natural gradient is a concept from information geometry [Amari, 1998] and applies when the gradient is taken w.r.t. the parameters θ of a probability distribution pθ . Instead of measuring the distance between parameters θ and θ′ with the Euclidean distance, we use the Kullback–Leibler (KL) divergence between the distributions pθ and pθ′ . The resulting steepest descent direction is the negative gradient preconditioned with the Hessian of the KL divergence, which is exactly the Fisher information matrix of pθ ,

F( θ) := Epθ (z)

[∇θ log pθ (z)∇θ log pθ (z)T ] = Epθ (z)

[−∇ 2 

> θ

log θ p(z)] . (4) The second equality may seem counterintuitive; the difference between the outer product of gradients and the Hessian cancels out in expectation with respect to the model distribution at θ, see Appendix A. This equivalence highlights the relationship of the Fisher to the Hessian. 

3.2 Difference in terminology across fields 

In our setting, we only model the conditional distribution pθ (y|x) of the joint distribution pθ (x, y ) = 

p(x)pθ (y|x). The Fisher information of θ for N samples from the joint distribution pθ (x, y ) is 

F∏  

> npθ(x,y )

(θ) = N Ex,y ∼p(x)pθ (y|x)

[∇θ log pθ (y|x)∇θ log pθ (y|x)T ] , (5) This is what statisticians would call the “Fisher information” of the model pθ (x, y ). However, we typically do not know the distribution over inputs p(x), so we use the empirical distribution over x

instead and compute the Fisher information of the conditional distribution ∏ 

> n

pθ (y|xn);

F∏  

> npθ(y|xn)

(θ) = ∑ 

> n

Ey∼pθ (y|xn)

[∇θ log pθ (y|xn)∇θ log pθ (y|xn)T ] . (6) This is Eq. (2), which we call the “Fisher”. This is the terminology used by work on the application of natural gradient methods in machine learning, such as Martens [2014] and Pascanu and Bengio [2014], as it is the Fisher information for the distribution we are optimizing, ∏ 

> n

pθ (y|xn). Work closer to the statistics literature, following the seminal paper of Amari [1998], such as Park et al. [2000] and Karakida et al. [2019], call this quantity the “empirical Fisher” due to the usage of the empirical samples for the inputs. In constrast, we call Eq. (3) the “empirical Fisher”, restated here, 

˜F( θ) = ∑ 

> n

∇θ log pθ (yn|xn)∇θ log pθ (yn|xn)T , (7) where “empirical” describes the use of samples for both the inputs and the outputs. This expression, however, does not have a direct interpretation as a Fisher information as it does not sample the output according to the distribution defined by the model. Neither is it a Monte-Carlo approximation of Eq. (6) , as the samples yn do not come from pθ (y|xn) but from the data distribution p(y|xn). How close the empirical Fisher (Eq. 7) is to the Fisher (Eq. 6) depends on how close the model pθ (y|xn)

is to the true data-generating distribution p(y|xn).

3.3 Generalized Gauss-Newton 

One line of argument justifying the use of the empirical Fisher approximation uses the connection be-tween the Hessian and the Fisher through the generalized Gauss-Newton (GGN) matrix [Schraudolph, 2002]. We give here a condensed overview of the definition and properties of the GGN. 4The original Gauss-Newton algorithm is an approximation to Newton’s method for nonlinear least squares problems, L(θ) = 12

∑

> n

(f (xn, θ ) − yn)2. By the chain rule, the Hessian can be written as 

∇2 L(θ) = ∑ 

> n

∇θ f (xn, θ )∇θ f (xn, θ )>

︸ ︷︷ ︸

> := G(θ)

+ ∑ 

> n

rn∇2 

> θ

f (xn, θ )

︸ ︷︷ ︸

> := R(θ)

, (8) where rn = f (xn, θ ) − yn are the residuals. The first part, G(θ), is the Gauss-Newton matrix. For small residuals, R(θ) will be small and G(θ) will approximate the Hessian. In particular, when the model perfectly fits the data, the Gauss-Newton is equal to the Hessian. Schraudolph [2002] generalized this idea to objectives of the form L(θ) = ∑ 

> n

an(bn(θ)) , with 

bn : RD → RM and an : RM → R, for which the Hessian can be written as 1

∇2 L(θ) = ∑

> n

(J θ bn(θ)) > ∇2 

> b

an(bn(θ)) (J θ bn(θ)) + ∑

> n,m

[∇ban(bn(θ))] m∇2 

> θ

b(m) 

> n

(θ). (9) The generalized Gauss-Newton matrix (GGN) is defined as the part of the Hessian that ignores the second-order information of bn,

G(θ) := ∑

> n

[J θ bn(θ)] > ∇2 

> b

an(bn(θ)) [J θ bn(θ)] . (10) If an is convex, as is customary, the GGN is positive (semi-)definite even if the Hessian itself is not, making it a popular curvature matrix in non-convex problems such as neural network training. The GGN is ambiguous as it crucially depends on the “split” given by an and bn. As an example, consider the two following possible splits for the least-squares problem from above: 

an(b) = 12 (b − yn)2, b n(θ) = f (xn, θ ), or an(b) = 12 (f (xn, b ) − yn)2, b n(θ) = θ. (11) The first recovers the classical Gauss-Newton, while in the second case, the GGN equals the Hessian. While this is an extreme example, the split will be important for our discussion. 

3.4 Connections between the Fisher, the GGN and the Hessian 

While NGD is not explicitly motivated as an approximate second-order method, the following result, noted by several authors, 2 shows that the Fisher captures partial curvature information about the problem defined in Eq. (1). 

Proposition 1 (Martens [2014], §9.2) . If p(y|f ) is an exponential family distribution with natural parameters f , then the Fisher information matrix coincides with the GGN of Eq. (1) using the split 

an(b) = − log p(yn|b), bn(θ) = f (xn, θ ), (12) 

and reads F( θ) = G(θ) = − ∑

> n

[J θ f (xn, θ )] > ∇2 

> f

log p(yn|f (xn, θ )) [J θ f (xn, θ )] .

For completeness, a proof can be found in Appendix C. The key insight is that ∇2 

> f

log p(y|f ) does not depend on y for exponential families. One can see Eq. (12) as the “canonical” split, since it matches the classical Gauss-Newton for the probabilistic interpretation of least-squares. From now on, when referencing “the GGN” without further specification, we mean this particular split. The GGN, and under the assumptions of Proposition 1 also the Fisher, are well-justified approxi-mations of the Hessian and we can bound their approximation error in terms of the (generalized) residuals, mirroring the motivation behind the classical Gauss-Newton (Proof in Appendix C.2). 

Proposition 2. Let L(θ) be defined as in Eq. (1) with F = RM . Denote by f (m) 

> n

the m-th component of f (xn, ·) : RD → RM and assume each f (m) 

> n

is β-smooth. Let G(θ) be the GGN (Eq. 10). Then, 

‖∇ 2 L(θ) − G(θ)‖22 ≤ r(θ)β, (13) 

where r(θ) = ∑Nn=1 ‖∇ f log p(yn|f (xn, θ )) ‖1 and ‖ · ‖ 2 denotes the spectral norm. 

The approximation improves as the residuals in r(θ) diminish, and is exact if the data is perfectly fit.  

> 1

Jθ bn(θ) ∈ RM ×D is the Jacobian of bn; we use the shortened notation ∇2 

> b

an(bn(θ)) := ∇2 

> b

an(b)|b=bn(θ);

[·]m selects the m-th component of a vector; and b(m) 

> n

denotes the m-th component function of bn. 

> 2

Heskes [2000] showed this for regression with squared loss, Pascanu and Bengio [2014] for classification with cross-entropy loss, and Martens [2014] for general exponential families. However, this has been known earlier in the statistics literature in the context of “Fisher Scoring” (see Wang [2010] for a review). 

54 Critical discussion of the empirical Fisher 

Two arguments have been put forward to advocate the empirical Fisher approximation. Firstly, it has been argued that it follows the definition of a generalized Gauss-Newton matrix, making it an approximate curvature matrix in its own right. We examine this relation in §4.1 and show that, while technically correct, it does not entail the approximation guarantee usually associated with the GGN. Secondly, a popular argument is that the empirical Fisher approaches the Fisher at a minimum if the model “is a good fit for the data”. We discuss this argument in §4.2 and point out that it requires strong additional assumptions, which are unlikely to be met in practical scenarios. In addition, this argument only applies close to a minimum, which calls into question the usefulness of the empirical Fisher in optimization. We discuss this in §4.3, showing that preconditioning with the empirical Fisher leads to adverse effects on the scaling and the direction of the updates far from an optimum. We use simple examples to illustrate our arguments. We want to emphasize that, as these are counter-examples to arguments found in the existing literature, they are designed to be as simple as possible, and deliberately do not involve intricate state-of-the art models that would complicate analysis. On a related note, while contemporary machine learning often relies on stochastic optimization, we restrict our considerations to the deterministic (full-batch) setting to focus on the adaptation to curvature. 

4.1 The empirical Fisher as a generalized Gauss-Newton matrix 

The first justification for the empirical Fisher is that it matches the construction of a generalized Gauss-Newton (Eq. 10) using the split [Bottou et al., 2018] 

an(b) = − log b, bn(θ) = p(yn|f (xn, θ )) . (14) Although technically correct, 3 we argue that this split does not provide a reasonable approximation. For example, consider a least-squares problem which corresponds to the log-likelihood log p(y|f ) = log exp[ − 12 (y − f )2]. In this case, Eq. (14) splits the identity function, log exp( ·), and takes into account the curvature from the log while ignoring that of exp . This questionable split runs counter to the basic motivation behind the classical Gauss-Newton matrix, that small residuals lead to a good approximation to the Hessian: The empirical Fisher 

˜F( θ) = ∑ 

> n

∇θ log pθ (yn|xn) ∇θ log pθ (yn|xn)> = ∑ 

> n

r2 

> n

∇θ f (xn, θ ) ∇θ f (xn, θ )>, (15) approaches zero as the residuals rn = f (xn, θ ) − yn become small. In that same limit, the Fisher 

F (θ) = ∑ 

> n

∇f (xn, θ )∇f (xn, θ )> does approach the Hessian, which we recall from Eq. (8) to be given by ∇2 L(θ) = F (θ) + ∑ 

> n

rn∇2 

> θ

f (xn, θ ). This argument generally applies for problems where we can fit all training samples such that ∇θ log pθ (yn|xn) = 0 for all n. In such cases, the EF goes to zero while the Fisher (and the corresponding GGN) approaches the Hessian (Prop. 2). For the generalized Gauss-Newton, the role of the “residual” is played by the gradient ∇ban(b);compare Equations (8) and (9) . To retain the motivation behind the classical Gauss-Newton, the split should be chosen such that this gradient can in principle attain zero, in which case the residual curvature not captured by the GGN in (9) vanishes. The EF split (Eq. 14) does not satisfy this property, as ∇b log b can never go to zero for a probability b ∈ [0 , 1] . It might be desirable to amend the definition of a generalized Gauss-Newton to enforce this property (addition in bold ): 

Definition 1 (Generalized Gauss-Newton) . A split L(θ) = ∑ 

> n

an(bn(θ)) with convex an, leads to a generalized Gauss-Newton matrix of L, defined as 

G(θ) = ∑ 

> n

Gn(θ), Gn(θ) := [J θ bn(θ)] > ∇2 

> b

an(bn(θ)) [J θ bn(θ)] , (16) 

if the split an, b n is such that there is b∗ 

> n

∈ Im( bn) such that ∇ban(b)|b=b∗ 

> n

= 0 .

Under suitable smoothness conditions, a split satisfying this condition will have a meaningful error bound akin to Proposition 2. To avoid confusion, we want to note that this condition does not assume the existence of θ∗ such that bn(θ∗) = b∗ 

> n

for all n; only that the residual gradient for each data point can, in principle, go to zero.            

> 3The equality can easily be verified by plugging the split (14) into the definition of the GGN (Eq. 10) and observing that ∇2
> ban(b) = ∇ban(b)∇ban(b)>as a special property of the choice an(b) = −log( b).

6Dataset  Correct Misspecified (A) Misspecified (B)    

> Quadratic approximation
> Loss contour Fisher emp. Fisher Minimum

Figure 2: Quadratic approximations of the loss function using the Fisher and the empirical Fisher on a logistic regression problem. Logistic regression implicitly assumes identical class-conditional covariances [Hastie et al., 2009, §4.4.5]. The EF is a good approximation of the Fisher at the minimum if this assumption is fulfilled (left panel), but can be arbitrarily wrong if the assumption is violated, even at the minimum and with large N . Note: we achieve classification accuracies of 

≥ 85 % in the misspecified cases compared to 73% in the well-specified case, which shows that a well-performing model is not necessarily a well-specified one. 

4.2 The empirical Fisher near a minimum 

An often repeated argument is that the empirical Fisher converges to the true Fisher when the model is a good fit for the data [e.g., Jastrz˛ ebski et al., 2018, Zhu et al., 2019]. Unfortunately, this is often misunderstood to simply mean “near the minimum”. The above statement has to be carefully formalized and requires additional assumptions, which we detail in the following. Assume that the training data consists of iid samples from some data-generating distribution 

ptrue (x, y ) = ptrue (y|x)ptrue (x). If the model is realizable, i.e., there exists a parameter setting 

θT such that pθT (y|x) = ptrue (y|x), then clearly by a Monte Carlo sampling argument, as the number of data points N goes to infinity, ˜F( θT)/N → F( θT)/N . Additionally, if the maximum likelihood estimate for N samples θ?N is consistent in the sense that pθ?N (y|x) converges to ptrue (y|x) as N → ∞ , 

> 1
> N

˜F( θ?N ) N →∞ 

−→ 1 

> N

F( θ?N ). (17) That is, the empirical Fisher converges to the Fisher at the minimum as the number of data points grows. (Both approach the Hessian, as can be seen from the second equality in Eq. 4 and detailed in Appendix C.2.) For the EF to be a useful approximation, we thus need (i) a “correctly-specified” model in the sense of the realizability condition, and (ii) enough data to recover the true parameters. Even under the assumption that N is sufficiently large, the model needs to be able to realize the true data distribution. This requires that the likelihood p(y|f ) is well-specified and that the prediction function f (x, θ ) captures all relevant information. This is possible in classical statistical modeling of, say, scientific phenomena where the effect of x on y is modeled based on domain knowledge. But it is unlikely to hold when the model is only approximate, as is most often the case in machine learning. Figure 2 shows examples of model misspecification and the effect on the empirical and true Fisher. It is possible to satisfy the realizability condition by using a very flexible prediction function f (x, θ ),such as a deep network. However, “enough” data has to be seen relative to the model capacity. The massively overparameterized models typically used in deep learning are able to fit the training data almost perfectly, even when regularized [Zhang et al., 2017]. In such settings, the individual gradients, and thus the EF, will be close to zero at a minimum, whereas the Hessian will generally be nonzero. 

4.3 Preconditioning with the empirical Fisher far from an optimum 

The relationship discussed in §4.2 only holds close to the minimum. Any similarity between pθ (y|x)

and ptrue (y|x) is very unlikely when θ has not been adapted to the data, for example, at the beginning of an optimization procedure. This makes the empirical Fisher a questionable preconditioner. 710 −2

> 10 0
> Loss

BreastCancer 

> 10 1
> 10 2

Boston 10 0 a1a        

> 025 75 100 Iteration
> -1
> 1
> Cosine (NGD,EFG)
> 025 75 100 Iteration
> -1
> 1
> 050 Iteration
> -1
> 1
> GD NGD EFGD

Figure 3: Fisher (NGD) vs. empirical Fisher (EFGD) as preconditioners (with damping) on linear classification (BreastCancer, a1a) and regression (Boston). While the EF can be a good approximation for preconditioning on some problems (e.g., a1a), it is not guaranteed to be. The second row shows the cosine similarity between the EF direction and the natural gradient, over the path taken by EFGD, showing that the EF can lead to update directions that are opposite to the natural gradient (see Boston). Even when the direction is correct, the magnitude of the steps can lead to poor performance (see BreastCancer). See Appendix D for details and additional experiments. In fact, the empirical Fisher can cause severe, adverse distortions of the gradient field far from the optimum, as evident even on the elementary linear regression problem of Fig. 1. As a consequence, EF-preconditioned gradient descent compares unfavorably to NGD even on simple linear regression and classification tasks, as shown in Fig. 3. The cosine similarity plotted in Fig. 3 shows that the empirical Fisher can be arbitrarily far from the Fisher in that the two preconditioned updates point in almost opposite directions. One particular issue is the scaling of EF-preconditioned updates. As the empirical Fisher is the sum of “squared” gradients (Eq. 3), multiplying the gradient by the inverse of the EF leads to updates of magnitude almost inversely proportional to that of the gradient, at least far from the optimum. This effect has to be counteracted by adapting the step size, which requires manual tuning and makes the selected step size dependent on the starting point; we explore this aspect further in Appendix E. 

## 5 Variance adaptation 

The previous sections have shown that, interpreted as a curvature matrix, the empirical Fisher is a questionable choice at best. Another perspective on the empirical Fisher is that, in contrast to the Fisher, it contains useful information to adapt to the gradient noise in stochastic optimization. In stochastic gradient descent [SGD; Robbins and Monro, 1951], we sample n ∈ [N ] uniformly at random and use a stochastic gradient g(θ) = −N ∇θ log pθ (yn|xn) as an inexpensive but noisy estimate of ∇ L (θ). The empirical Fisher, as a sum of outer products of individual gradients, coincides with the non-central second moment of this estimate and can be written as 

N ˜F( θ) = Σ( θ) + ∇ L (θ) ∇ L (θ)>, Σ( θ) := cov [g(θ)] . (18) Gradient noise is a major hindrance to SGD and the covariance information encoded in the EF may be used to attenuate its harmful effects, e.g., by scaling back the update in high-noise directions. A small number of works have explored this idea before. Le Roux et al. [2007] showed that the update direction Σ( θ)−1g(θ) maximizes the probability of decreasing in function value, while Schaul et al. [2013] proposed a diagonal rescaling based on the signal-to-noise ratio of each coordinate, 

Dii := [ ∇ L (θ)] 2 

> i

/ ([ ∇ L (θ)] 2 

> i

+ Σ( θ)ii ). Balles and Hennig [2018] identified these factors as 

optimal in that they minimize the expected error E [‖Dg (θ) − ∇ L (θ)‖22

] for a diagonal matrix D.A straightforward extension of this argument to full matrices yields the variance adaptation matrix 

M = (Σ( θ) + ∇ L (θ) ∇ L (θ)>)−1 ∇ L (θ) ∇ L (θ)> = ( N ˜F( θ)) −1∇ L (θ) ∇ L (θ)>. (19) In that sense, preconditioning with the empirical Fisher can be understood as an adaptation to gradient noise instead of an adaptation to curvature. The multiplication with ∇ L (θ)∇ L (θ)> in Eq. (19) will counteract the poor scaling discussed in §4.3. This perspective on the empirical Fisher is currently not well studied. Of course, there are obvious difficulties ahead: Computing the matrix in Eq. (19) requires the evaluation of all gradients, which 8defeats its purpose. It is not obvious how to obtain meaningful estimates of this matrix from, say, a mini-batch of gradients, that would provably attenuate the effects of gradient noise. Nevertheless, we believe that variance adaptation is a possible explanation for the practical success of existing methods using the EF and an interesting avenue for future research. To put it bluntly: it may just be that the name “empirical Fisher” is a fateful historical misnomer, and the quantity should instead just be described as the gradient’s non-central second moment. As a final comment, it is worth pointing out that some methods precondition with the square-root of the EF, the prime example being Adam. While this avoids the “inverse gradient” scaling discussed in §4.3, it further widens the conceptual gap between those methods and natural gradient. In fact, such a preconditioning effectively cancels out the gradient magnitude, which has recently been examined more closely as “sign gradient descent” [Balles and Hennig, 2018, Bernstein et al., 2018]. 

## 6 Conclusions 

We offered a critical discussion of the empirical Fisher approximation, summarized as follows: 

• While the EF follows the formal definition of a generalized Gauss-Newton matrix, the underlying split does not retain useful second-order information. We proposed a clarifying amendment to the definition of the GGN. 

• A clear relationship between the empirical Fisher and the Fisher only exists at a minimum under strong additional assumptions: (i) a correct model and (ii) enough data relative to model capacity. These conditions are unlikely to be met in practice, especially when using overparametrized general function approximators and settling for approximate minima. 

• Far from an optimum, EF preconditioning leads to update magnitudes which are inversely proportional to that of the gradient, complicating step size tuning and often leading to poor performance even for linear models. 

• As a possible alternative explanation of the practical success of EF preconditioning, and an interesting avenue for future research, we have pointed to the concept of variance adaptation. The existing arguments do not justify the empirical Fisher as a reasonable approximation to the Fisher or the Hessian. Of course, this does not rule out the existence of certain model classes for which the EF might give reasonable approximations. However, as long as we have not clearly identified and understood these cases, the true Fisher is the “safer” choice as a curvature matrix and should be preferred in virtually all cases. Contrary to conventional wisdom, the Fisher is not inherently harder to compute than the EF. As shown by Martens and Grosse [2015], an unbiased estimate of the true Fisher can be obtained at the same computational cost as the empirical Fisher by replacing the expectation in Eq. (2) with a single sample 

˜yn from the model’s predictive distribution pθ (y|xn). Even exact computation of the Fisher is feasible in many cases. We discuss computational aspects further in Appendix B. The apparent reluctance to compute the Fisher might have more to do with the current lack of convenient implementations in deep learning libraries. We believe that it is misguided—and potentially dangerous—to accept the poor theoretical grounding of the EF approximation purely for implementational convenience. 

Acknowledgements 

We thank Matthias Bauer, Felix Dangel, Filip de Roos, Diego Fioravanti, Jason Hartford, Si Kai Lee, and Frank Schneider for their helpful comments on the manuscript. We thank Emtiyaz Khan, Aaron Mishkin, and Didrik Nielsen for many insightful conversations that lead to this work, and the anonymous reviewers for their constructive feedback. Lukas Balles kindly acknowledges the support of the International Max Planck Research School for Intelligent Systems (IMPRS-IS). The authors gratefully acknowledge financial support by the European Research Council through ERC StG Action 757275 / PANAMA and the DFG Cluster of Excellence “Machine Learning - New Perspectives for Science”, EXC 2064/1, project number 390727645, the German Federal Ministry of Education and Research (BMBF) through the Tübingen AI Center (FKZ: 01IS18039A) and funds from the Ministry of Science, Research and Arts of the State of Baden-Württemberg. 9References 

Shun-ichi Amari. Natural gradient works efficiently in learning. Neural computation , 10(2):251–276, 1998. Lukas Balles and Philipp Hennig. Dissecting Adam: The sign, magnitude and variance of stochastic gradients. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018 , volume 80 of Proceedings of Machine Learning Research , pages 413–422. PMLR, 2018. Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Anima Anandkumar. signSGD: compressed optimisation for non-convex problems. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018 , volume 80 of Proceedings of Machine Learning Research , pages 559–568. PMLR, 2018. Aleksandar Botev, Hippolyt Ritter, and David Barber. Practical Gauss-Newton optimisation for deep learning. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Con-ference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017 , volume 70 of Proceedings of Machine Learning Research , pages 557–565. PMLR, 2017. Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM Reviews , 60(2):223–311, 2018. Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-SGD: Biasing gradient descent into wide valleys. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings . OpenReview.net, 2017. Thomas George, César Laurent, Xavier Bouthillier, Nicolas Ballas, and Pascal Vincent. Fast approximate natural gradient descent in a Kronecker-factored eigenbasis. In Samy Bengio, Hanna Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, 

Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montréal, Canada. , pages 9573– 9583, 2018. Alex Graves. Practical variational inference for neural networks. In John Shawe-Taylor, Richard S. Zemel, Peter L. Bartlett, Fernando C. N. Pereira, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 24: 25th Annual Conference on Neural Information Processing Systems 2011. Proceedings of a meeting held 12-14 December 2011, Granada, Spain. ,pages 2348–2356, 2011. Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The elements of statistical learning: data mining, inference, and prediction . Springer Verlag, 2009. Tom Heskes. On “natural” learning and pruning in multilayered perceptrons. Neural Computation ,12(4):881–901, 2000. Stanisław Jastrz˛ ebski, Zac Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Amos Storkey, and Yoshua Bengio. Three factors influencing minima in SGD. In International Conference on Artificial Neural Networks , 2018. Eric Jones, Travis Oliphant, Pearu Peterson, et al. SciPy: Open source scientific tools for Python, 2001. URL http://www.scipy.org/ .Ryo Karakida, Shotaro Akaho, and Shun-ichi Amari. Universal statistics of fisher information in deep neural networks: Mean field approach. In Kamalika Chaudhuri and Masashi Sugiyama, editors, 

The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan , volume 89 of Proceedings of Machine Learning Research ,pages 1032–1041. PMLR, 2019. 10 Mohammad Emtiyaz Khan, Didrik Nielsen, Voot Tangkaratt, Wu Lin, Yarin Gal, and Akash Srivastava. Fast and scalable Bayesian deep learning by weight-perturbation in Adam. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018 , volume 80 of Proceedings of Machine Learning Research , pages 2616–2625. PMLR, 2018. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015. James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences , 114(13):3521–3526, 2017. Nicolas Le Roux and Andrew Fitzgibbon. A fast natural Newton method. In Johannes Fürnkranz and Thorsten Joachims, editors, Proceedings of the 27th International Conference on Machine Learning (ICML-10), June 21-24, 2010, Haifa, Israel , pages 623–630. Omnipress, 2010. Nicolas Le Roux, Pierre-Antoine Manzagol, and Yoshua Bengio. Topmoumoute online natural gradient algorithm. In John C. Platt, Daphne Koller, Yoram Singer, and Sam T. Roweis, editors, 

Advances in Neural Information Processing Systems 20, Proceedings of the Twenty-First Annual Conference on Neural Information Processing Systems, Vancouver, British Columbia, Canada, December 3-6, 2007 , pages 849–856. Curran Associates, Inc., 2007. Zhibin Liao, Tom Drummond, Ian Reid, and Gustavo Carneiro. Approximate Fisher information matrix to characterise the training of deep neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence , 42(1):15–26, 2020. James Martens. Deep learning via Hessian-free optimization. In Johannes Fürnkranz and Thorsten Joachims, editors, Proceedings of the 27th International Conference on Machine Learning (ICML-10), June 21-24, 2010, Haifa, Israel , pages 735–742. Omnipress, 2010. James Martens. New insights and perspectives on the natural gradient method. CoRR , abs/1412.1193, 2014. James Martens and Roger Grosse. Optimizing neural networks with Kronecker-factored approximate curvature. In Francis Bach and David M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015 , volume 37 of JMLR Workshop and Conference Proceedings , pages 2408–2417, 2015. Aaron Mishkin, Frederik Kunstner, Didrik Nielsen, Mark Schmidt, and Mohammad Emtiyaz Khan. SLANG: Fast structured covariance approximations for Bayesian deep learning with natural gradient. In Samy Bengio, Hanna Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montréal, Canada. , pages 6248–6258. 2018. Yann Ollivier. Riemannian metrics for neural networks I: feedforward networks. Information and Inference: A Journal of the IMA , 4(2):108–153, 2015. Kazuki Osawa, Yohei Tsuji, Yuichiro Ueno, Akira Naruse, Rio Yokota, and Satoshi Matsuoka. Large-scale distributed second-order optimization using Kronecker-factored approximate curvature for deep convolutional neural networks. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 12359–12367. Computer Vision Foundation / IEEE, June 2019. Hyeyoung Park, Shun-ichi Amari, and Kenji Fukumizu. Adaptive natural gradient learning algorithms for various stochastic models. Neural Networks , 13(7):755–764, 2000. Razvan Pascanu and Yoshua Bengio. Revisiting natural gradient for deep networks. In Yoshua Bengio and Yann LeCun, editors, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings , 2014. 11 Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics , pages 400–407, 1951. Arnold Salas, Stefan Zohren, and Stephen Roberts. Practical Bayesian learning of neural networks via adaptive subgradient methods. CoRR , abs/1811.03679, 2018. Tom Schaul, Sixin Zhang, and Yann LeCun. No more pesky learning rates. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013 , volume 28 of JMLR Workshop and Conference Proceedings , pages 343–351, 2013. Nicol N. Schraudolph. Fast curvature matrix-vector products for second-order gradient descent. 

Neural Computation , 14(7):1723–1738, 2002. Yi Sun, Daan Wierstra, Tom Schaul, and Jürgen Schmidhuber. Efficient natural evolution strategies. In Proceedings of the 11th Annual conference on Genetic and evolutionary computation , pages 539–546, 2009. Valentin Thomas, Fabian Pedregosa, Bart van Merriënboer, Pierre-Antoine Manzagol, Yoshua Bengio, and Nicolas Le Roux. Information matrices and generalization. CoRR , abs/1906.07774, 2019. Yong Wang. Fisher scoring: An interpolation family and its Monte Carlo implementations. Computa-tional Statistics & Data Analysis , 54(7):1744–1755, 2010. Yeming Wen, Kevin Luk, Maxime Gazeau, Guodong Zhang, Harris Chan, and Jimmy Ba. Interplay between optimization and generalization of stochastic gradient descent with covariance noise. 

CoRR , abs/1902.08234, 2019. To appear in the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2020. Daan Wierstra, Tom Schaul, Jan Peters, and Jürgen Schmidhuber. Natural evolution strategies. In 

2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence) , pages 3381–3387, 2008. Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings .OpenReview.net, 2017. Guodong Zhang, Shengyang Sun, David Duvenaud, and Roger Grosse. Noisy natural gradient as variational inference. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018 , volume 80 of Proceedings of Machine Learning Research , pages 5847–5856. PMLR, 2018. Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA ,volume 97 of Proceedings of Machine Learning Research , pages 7654–7663. PMLR, 2019. 12 Limitations of the Empirical Fisher Approximation for Natural Gradient Descent Supplementary Material 

## A Details on natural gradient descent 

We give an expanded version of the introduction to natural gradient descent provided in Section 3.1 

A.1 Measuring distance in Kullback-Leibler divergence 

Gradient descent minimizes the objective function by updating in the “direction of steepest descent”. But what, precisely, is meant by the direction of steepest descent? Consider the following definition, 

lim ε→0 1

> ε

(arg min δ f (θ + δ)) s.t. d(θ, θ + δ) ≤ ε, (20) where d(·, ·) is some distance function. We are looking for the update step δ which minimizes f

within an ε distance around θ, and let the radius ε go to zero (to make δ finite, we have to divide by ε). This definition makes clear that the direction of steepest descent is intrinsically tied to the geometry we impose on the parameter space by the definition of the distance function. If we choose the Euclidean distance d(θ, θ ′) = ‖θ − θ′‖2, Eq. (20) reduces to the (normalized) negative gradient. Now, assume that θ parameterizes a statistical model pθ (z). The parameter vector θ is not the main quantity of interest; the distance between θ and θ′ would be better measured in terms of distance between the distributions pθ and pθ′ . A common function to measure the difference between probability distributions is the Kullback–Leibler (KL) divergence. If we choose d(θ, θ ′) = 

DKL 

(pθ′ ‖ pθ

), the steepest descent direction becomes the natural gradient, F( θ)−1∇ L (θ), where 

F( θ) = ∇2 

> θ′

DKL 

(pθ ‖ pθ′

) |θ′=θ , (21) the Hessian of the KL divergence, is the Fisher information matrix of the statistical model and 

F( θ) := Epθ (z)

[∇ log pθ (z)∇ log pθ (z)T ] = Epθ (z)

[−∇ 2 log pθ (z)] (22) To see why, apply the chain rule on the log to split the equation in terms of the Hessian and the outer product of the gradients of pθ w.r.t. θ,

Epθ (z)

[−∇ 2 

> θ

log pθ (z)] = Epθ (z)

[

− 1  

> pθ(z)

∇2 

> θ

pθ (z)

]

+ Epθ (z)

[ 1  

> pθ(z)2

∇θ pθ (z)∇θ pθ (z)>

]

. (23) The first term on the right-hand side is zero, since 

Epθ (z)

[

− 1  

> pθ(z)

∇2 

> θ

pθ (z)

]

:= −

∫

> z

1

pθ (z) ∇2 

> θ

pθ (z)pθ (z) d z =

∫

> z

∇2 

> θ

pθ (z) d z, 

= ∇2

> θ

∫

pθ (z) d z = ∇2 

> θ

[1] = 0 . (24) The second term is the expected outer-product of the gradients, as ∂θ log f (θ) = 1  

> f(θ)

∂θ f (θ),  

> 1
> pθ(z)2

∇θ pθ (z)∇θ pθ (z)> =

( 1  

> pθ(z)

∇θ pθ (z)

) ( 1  

> pθ(z)

∇θ pθ (z)

)>

,

= ∇θ log pθ (z) ∇θ log pθ (z)>. (25) The same technique also shows that if the empirical distribution over the data is equal to the model distribution pθ (y|f (x, θ ), then the Fisher, empirical Fisher and Hessian are all equal. 13 A.2 The Fisher for common loss functions 

For a probabilistic conditional model of the form p(y|f (x, θ )) where p is an exponential family distribution, the equivalence between the Fisher and the generalized Gauss-Newton leads to a straightforward way to compute the Fisher without expectations, as 

F( θ) = ∑

> n

(J θ f (xn, θ )) >(∇2 log p(yn|f (xn, θ )))(J θ f (xn, θ )) = ∑ 

> n

J> 

> n

HnJn, (26) where Jn = J θ f (xn, θ ) and Hn = ∇2 log p(yn|f (xn, θ )) often has an exploitable structure. 

The squared-loss used in regression, 12

∑

> n

∥∥yn − f (xn, θ )∥∥2, can be cast in a probabilistic setting with a Gaussian distribution with unit variance, p(yn|f (xn, θ )) = N (yn; f (xn, θ ), 1),

p(yn|f (xn, θ )) = exp 

(

− 12

∥∥yn − f (xn, θ )∥∥2)

. (27) The Hessian of the negative log-likelihood w.r.t. f is then 

∇2 

> f

− log p(yn|f ) = ∇2

> f

[

− log exp 

(

− 12 ‖yn − f ‖2)]

= ∇2

> f

[ 12 ‖yn − f ‖2]

= 1 . (28) And as the function f is scalar-valued, the Fisher reduces to an outer-products of gradients, 

F( θ) = ∑ 

> n

∇θ f (xn, θ )∇θ f (xn, θ )>. (29) We stress that this is difference to the outer product of gradients of the overall loss; 

F( θ) 6 = ∑ 

> n

∇θ log p(yn|f (xn, θ )) ∇θ log p(yn|f (xn, θ )) >. (30) 

The cross-entropy loss used in C-class classification can be cast as an exponential family distribution by using the softmax function on the mapping f (xn, θ ),

p(yn = c|f (xn, θ )) = [ softmax (f )] c = efc   

> ∑
> iefi

= πc, (31) The Hessian of the negative log-likelihood w.r.t. f is independent of the class label c,

∇2 

> f

(− log p(y = c|f )) = ∇2 

> f

[−fc + log (∑  

> i

efi

)] = ∇2 

> f

[log (∑  

> i

efi

)]. (32) A close look at the partial derivatives shows that  

> ∂2
> ∂f 2
> i

log (∑  

> c

efc

) = efi   

> (∑
> cefc )

− efi 2   

> (∑
> cefc )2

, and ∂2 

> ∂f i∂f j

log (∑  

> c

efc

) = − efi efj   

> (∑
> cefc )2

, (33) and the Hessian w.r.t. f can be written in terms of the vector of predicted probabilities π as 

∇2 

> f

(− log p(y|f )) = diag( π) − ππ >. (34) Writing πn the vector of probabilities associated with the nth sample, the Fisher becomes 

F( θ) = ∑

> n

[J θ f (xn, θ )] >(diag( πn) − πnπ> 

> n

)[J θ f (xn, θ )] . (35) 

A.3 The generalized Gauss-Newton as a linear approximation of the model 

In Section 3.3, we mentioned that the generalized Gauss-Newton with a split L(θ) = ∑ 

> n

an(bn(θ)) 

can be interpreted as an approximation of L where the second-order information of an is conserved but the second-order information of bn is ignored. To make this connection explicit, see that if bn is a linear function, the Hessian and the GGN are equal as the Hessian of bn w.r.t. to θ is zero, 

∇2 L(θ) = ∑

> n

(J θ bn(θ)) > ∇2 

> b

an(bn(θ)) (J θ bn(θ)) 

︸ ︷︷ ︸

> GGN

+ ∑

> n,m

[∇ban(bn(θ))] m ∇2 

> θ

b(m) 

> n

(θ)

︸ ︷︷ ︸

> =0

. (36) This corresponds to the Hessian of a local approximation of L where the inner function b is linearized. We write the first-order Taylor approximation of bn around θ as a function of θ′,

¯bn(θ, θ ′) := bn(θ) + J θ bn(θ)( θ′ − θ),

and approximate L(θ′) by replacing bn(θ′) by its linear approximation ¯bn(θ, θ ′). The generalized Gauss-Newton is the Hessian of this approximation, evaluated at θ′ = θ,

G(θ) = ∇2

> θ′

∑ 

> n

an(¯bn(θ, θ ′)) |θ′=θ = ∑

> n

(J θ bn(θ)) > ∇2 

> b

an(bn(θ)) (J θ bn(θ)) (37) 14 B Computational aspects 

The empirical Fisher approximation is often motivated as an easier-to-compute alternative to the Fisher. While there is some merit to this argument, we argued in the main text that it computes the wrong quantity. A Monte Carlo approximation to the Fisher has the same computational complexity and a similar implementation: sample one output ˜yn from the model distribution p(y|f (xn, θ )) for each input xn and compute the outer product of the gradients 

∑ 

> n

∇ log p(˜ yn|f (xn, θ )) ∇ log p(˜ yn|f (xn, θ )) >. (38) While noisy, this one-sample estimate is unbiased and does not suffer from the problems mentioned in the main text. This is the approach used by Martens and Grosse [2015] and Zhang et al. [2018]. As a side note, some implementations use a biased estimate by using the most likely output ˆyn =arg max y p(y|f (xn, θ )) instead of sampling ˜yn from p(y|f (xn, θ )) . This scheme could be beneficial in some circumstances as it reduces variance, but it can backfire by increasing the bias. For the least-squares loss, p(y|f (xn, θ )) is a Gaussian distribution centered as f (xn, θ ) and the most likely output is f (xn, θ ). The gradient ∇θ log p(y|f (xn, θ )) |y=f (xn,θ ) is then always zero. For high quality estimates, sampling additional outputs and averaging the results is inefficient. If M

MC samples ˜y1, . . . , ˜yM per input xn are used to compute the gradients gm = ∇ log p(˜ ym|f (xn, θ )) ,most of the computation is repeated. The gradient gm is 

gm = ∇ log p(˜ ym|f (xn, θ )) = −(J θ f (xn, θ )) >∇f log p(˜ ym|f ), (39) where the Jacobian of the model output, Jθ f , does not depend on ˜ym. The Jacobian of the model is typically more expensive to compute than the gradient of the log-likelihood w.r.t. the model output, especially when the model is a neural network. This approach repeats the difficult part of the computation M times. The expectation can instead be computed in closed form using the generalized Gauss-Newton equation (Eq. 26, or Eq. 10 in the main text), which requires the computation of the Jacobian only once per sample xn.The main issue with this approach is that computing Jacobians is currently not well supported by deep learning auto-differentiation libraries, such as TensorFlow or Pytorch. However, the current the implementations relying on the empirical Fisher also suffer from this lack of support, as they need access to the individual gradients to compute their outer-product. Access to the individual gradients is equivalent to computing the Jacobian of the vector [− log p(y1|f (x1, θ )) , ..., − log p(yN |f (xN , θ )] >.The ability to efficiently compute Jacobians and/or individual gradients in parallel would drastically improve the practical performance of methods based on the Fisher and empirical Fisher, as most of the computation of the backward pass can be shared between samples. 

## C Additional proofs 

C.1 Proof of Propositon 1 

In Section 3.4, Proposition 1, we stated that the Fisher and the generalized Gauss-Newton are equivalent for the problems considered in the introduction; 

Proposition 1 (Martens [2014], §9.2) . If p(y|f ) is an exponential family distribu-tion with natural parameters f , then the Fisher information matrix coincides with the GGN of Eq. (1) using the split 

an(b) = − log p(yn|b), bn(θ) = f (xn, θ ), (12) 

and reads F( θ) = G(θ) = − ∑

> n

[J θ f (xn, θ )] >∇2 

> f

log p(yn|f (xn, θ ))[J θ f (xn, θ )] .

Plugging the split into the definition of the GGN (Eq. 10) yields G(θ), so we only need to show that the Fisher coincides with this GGN. By the chain rule, we have 

∇θ log p(y|f (xn, θ )) = J θ f (xn, θ )> ∇f log p(y|f (xn, θ )) , (40) and we can then apply the following steps. 

F( θ) = ∑ 

> n

Ey∼pθ (y|xn)

[Jθ f (xn, θ )> ∇f log p(y|fn) ∇f log p(y|fn)>Jθ f (xn, θ )] , (41) 

= ∑ 

> n

Jθ f (xn, θ )> Ey∼pθ (y|xn)

[∇f log p(y|fn) ∇f log p(y|fn)>] Jθ f (xn, θ ), (42) 

= ∑ 

> n

Jθ f (xn, θ )> Ey∼pθ (y|xn)

[

−∇ 2 

> f

log p(y|fn)

]

Jθ f (xn, θ ), (43) 15 Eq. (41) rewrites the Fisher using the chain rule, Eq. (42) take the Jacobians out of the expectation as they do not depend on y and Eq. (43) is due to the equivalence between the expected outer product of gradients and expected Hessian shown in the last section. If p is an exponential family distribution with natural parameters (a linear combination of) f , its log density has the form log p(y|f ) = f T T (y) − A(f ) + log h(y) where T are the sufficient statistics, 

A is the cumulant function, and h is the base measure. Its Hessian w.r.t. f is independent of y,

F( θ) = ∑ 

> n

Jθ f (xn, θ )>∇2 

> f

(− log p(yn|fn))J θ f (xn, θ ), (44) 

C.2 Proof of Proposition 2 

In §3.4, Prop. 2, we show that the difference between the Fisher (or the GNN) and the Hessian can be bounded by the residuals and the smoothness constant of the model f ;

Proposition 2. Let L(θ) be defined as in Eq. (1) with F = RM . Denote by f (m)

> n

the m-th component of f (xn, ·) : RD → RM and assume each f (m) 

> n

is β-smooth. Let G(θ) be the GGN (Eq. 10). Then, 

‖∇ 2 L(θ) − G(θ)‖22 ≤ r(θ)β, (13) 

where r(θ) = ∑Nn=1 ‖∇ f log p(yn|f (xn, θ )) ‖1 and ‖ · ‖ 2 denotes the spectral norm. 

Dropping θ from the notation for brevity, the Hessian can be expressed as 

∇2 L = G + ∑Nn=1 

∑Mm=1 r(m) 

> n

∇2 

> θ

f (m) 

> n

, where r(m) 

> n

= ∂ log p(yn|f )  

> ∂f (m)

|f =fn(θ) (45) is the derivative of − log p(y|f ) w.r.t. the m-th component of f , evaluated at f = fn(θ).If all f (m) 

> n

are β-smooth, their Hessians are bounded by −βI  ∇ 2 

> θ

f (m) 

> n

 βI and 

−

∣∣∣∑ 

> n,m

r(m)

> n

∣∣∣ β I  ∇ 2 L − G 

∣∣∣∑ 

> n,m

r(m)

> n

∣∣∣ β I . (46) Pulling the absolute value inside the double sum gives the upper bound 

∣∣∣∑ 

> n,m

r(m)

> n

∣∣∣ ≤ ∑

> n

∑

> m

∣∣∣ ∂ log p(yn|f )  

> ∂f (m)

|f =fn(θ)

∣∣∣ = ∑ 

> n

‖∇ f log p(yn|fn(θ)) ‖1, (47) and the statement about the spectral norm (the largest singular value of the matrix) follows. 

## D Experimental details 

In contrast to the main text of the paper, which uses the sum formulation of the loss function, 

L(θ) = ∑ 

> n

log p(yn|f (xn, θ )) ,

the implementation—and thus the reported step sizes and damping parameters—apply to the average, 

L(θ) = 1

> N

∑ 

> n

log p(yn|f (xn, θ )) .

The Fisher and empirical Fisher are accordingly rescaled by a 1/N factor. 

D.1 Vector field of the empirical Fisher preconditioning 

The problem used for Fig. 1 is a linear regression on N = 1000 samples from 

xi ∼ Lognormal (0, 3/4) , i ∼ N (0 , 1) , yi = 2 + 2 xi + i. (48) To be visible and of a similar scale, the gradient, natural gradient and empirical Fisher-preconditioned gradient were relatively rescaled by 1/3, 1 and 3, respectively. The trajectories of each method is computed by running each update, GD: θt+1 = θt − γ∇ L (θt). (49) NGD: θt+1 = θt − γ(F( θt) + λ I)−1∇ L (θt), (50) EFGD: θt+1 = θt − γ(˜F( θt) + λ I)−1∇ L (θt), (51) using a step size of γ = 10 −4 and a damping parameter of λ = 10 −8 to ensure stability for 50 ′000 

iterations. The vector field is computed using the same damping parameter. The starting points are 

[2 4.5] , [1 0] , [4.5 3] , [−0.5 3] .

16 D.2 EF as a quadratic approximation at the minimum for misspecified models 

The problems are optimized using using the Scipy [Jones et al., 2001] implementation of BFGS 4.The quadratic approximation of the loss function using the matrix M (the Fisher or empirical Fisher) used is L(θ) ≈ 12 (θ − θ?)M (θ − θ?), for ‖θ − θ?‖2 = 1 . The datasets used for the logistic regression problem of Fig. 2 are described in Table 2. Fig. 4 shows additional examples of model misspecification on a linear regression problem using the datasets described in Table 3. All experiments used N = 1 ′000 samples. Table 2: Datasets used for Fig. 2. For all datasets, p(y = 0) = p(y = 1) = 1 /2.Model p(x|y = 0) p(x|y = 1) 

Correct model: N

([11

]

,

[2 00 2

])

N

([−1

−1

]

,

[2 00 2

])

Misspecified (A): N

([1.51.5

]

,

[3 00 3

])

N

([−1.5

−1.5

]

,

[1 00 1

])

Misspecified (B): N

([−1

−1

]

,

[ 1.5 −0.9

−0.9 1.5

])

N

([11

]

,

[1.5 0.90.9 1.5

])

Table 3: Datasets used for Fig. 4. For all datasets, x ∼ N (0 , 1) .Model y 

Correct model: y = x +   ∼ N (0 , 1) 

Misspecified (A): y = x +   ∼ N (0 , 2) 

Misspecified (B): y = x + 12 x2 +   ∼ N (0 , 1) 

D.3 Optimization with the empirical Fisher as preconditioner 

The optimization experiment uses the update rules described in §D.1 by Eq. (49, 50, 51). The step size and damping hyperparameters are selected by a gridsearch, selecting for each optimizer the run with the minimal loss after 100 iterations. The grid used is described in Table 5 as a log-space 5.Table 4 describes the datasets used and Table 6 the hyperparameters selected by the gridsearch. The cosine similarity is computed between the gradient preconditioned with the empirical Fisher and the Fisher, without damping, at each step along the path taken by the empirical Fisher optimizer. The problems are initialized at θ0 = 0 and run for 100 iterations. This initialization is favorable to the empirical Fisher for the logistic regression problems. Not only is it guaranteed to not be arbitrarily wrong, but the empirical Fisher and the Fisher coincide when the predicted probabilities are uniform. For the sigmoid activation of the output of the linear mapping, σ(f ), the gradient and Hessian are 

− ∂∂f log p(y|f ) = σ(f ) − ∂2  

> ∂f 2

log p(y|f ) = σ(f )(1 − σ(f )) . (52) They coincide when σ(f ) = 12 , at θ = 0 , or when σ(f ) ∈ { 0, 1}, which require infinite weights. 

> 4https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html
> 5https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html

17 Table 4: Datasets Dataset # Features # Samples Type Figure a1a 6 1′605 123 Classification Fig. 3 BreastCancer 7 683 10 Classification Fig. 3 Boston Housing 8 506 13 Regression Fig. 3 Yacht Hydrodynamics 9 308 7 Regression Fig. 5 Powerplant 10 9′568 4 Regression Fig. 5 Wine 11 178 13 Regression Fig. 5 Energy 12 768 8 Regression Fig. 5 Table 5: Grid used for the hyperparameter search for the opti-mization experiments, in log 10 . The number of samples to gen-erate was selected as to generate a smooth grid in base 10, e.g., 

10 0, 10 .25 , 10 .5, 10 .75 , 10 1, 10 1.25 , . . . 

Parameter Grid Step size γ logspace(start=-20, stop=10, num=241) 

Damping λ logspace(start=-10, stop=10, num=41) 

Table 6: Selected hyperparameters, given in log 10 .Dataset Algorithm γ λ

Boston GD −5.250 

NGD 0.125 −10 .0

EFGD −1.250 −8.0

BreastCancer GD −5.125 

NGD 0.125 −10 .0

EFGD −1.250 −10 .0

a1a GD 0.250 

NGD 0.250 −10 .0

EFGD −0.375 −8.0

Dataset Algorithm γ λ

Wine GD −5.625 

NGD 0.000 −8.5

EFGD −1.375 −6.0

Energy GD −5.500 

NGD 0.000 −7.5

EFGD 0.875 −3.0

Powerplant GD −5.750 

NGD −0.625 −8.0

EFGD 3.375 −1.0

Yacht GD −1.500 

NGD −0.750 −7.5

EFGD 1.625 −6.5

> 6

www.csie.ntu.edu.tw/ cjlin/libsvmtools/datasets/binary.html#a1a 

> 7

www.csie.ntu.edu.tw/˜ cjlin/libsvmtools/datasets/binary.html#breast-cancer 

> 8

scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html 

> 9

archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics 

> 10

archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant 

> 11

archive.ics.uci.edu/ml/datasets/Wine 

> 12

archive.ics.uci.edu/ml/datasets/Energy+efficiency 

18 E Additional plots 

Fig. 4 repeats the experiment described in Fig. 2 (§4.2), on the effect of model misspecification on the Fisher and empirical Fisher at the minimum, on linear regression problems instead of a classification problem. Similar issues in scaling and directions can be observed. Fig. 5 repeats the experiment described in Fig. 3 (§4.3) on additional linear regression problems. Those additional examples show that the poor performance of empirical Fisher-preconditioned updates compared to NGD is not isolated to the examples shown in the main text. Fig. 6 show the linear regression problem on the Boston dataset, originally shown in Fig. 3, where each line is a different starting point, using the same hyperparameters as in Fig. 3. The starting points are selected from [−θ?, θ ?], where θ? is the optimum. When the optimization starts close to the minimum (low loss), the empirical Fisher is a good approximation to the Fisher and there are very few differences with NGD. However, when the optimization starts far from the minimum (high loss), the individual gradients, and thus the sum of outer product gradients, are large, which leads to very small steps, regardless of curvature, and slow convergence. While this could be counteracted with a larger step size in the beginning, this large step size would not work close to the minimum and would lead to oscillations. The selection of the step size therefore depends on the starting point, and would ideally be on a decreasing schedule. Dataset  Correct Misspecified (A) Misspecified (B)    

> Quadratic approximation
> Loss contour Fisher emp. Fisher Minimum

Figure 4: Quadratic approximations of the loss function using the Fisher and the empirical Fisher on a linear regression problem. The EF is a good approximation of the Fisher at the minimum if the data is generated by y ∼ N (xθ ∗ + b∗, 1) , as the model assumes (left panel), but can be arbitrarily wrong if the assumption is violated, even at the minimum and with large N. In (A), the model is misspecified as it under-estimates the observation noise (data is generated by y ∼ N (xθ ∗ + b∗, 2) ). In (B), the model is misspecified as it fails to capture the quadratic relationship between x and y.19 10 −1

> Loss

Wine      

> 020 40 60 80 100 Iteration
> -1
> 1
> Cosine (NGD,EFG) 10 1
> 10 2
> Loss

Energy      

> 020 40 60 80 100 Iteration
> -1
> 1
> Cosine (NGD,EFG) 10 3
> Loss

Powerplant      

> 020 40 60 80 100 Iteration
> -1
> 1
> Cosine (NGD,EFG) 10 2
> Loss

Yacht      

> 020 40 60 80 100 Iteration
> -1
> 1
> Cosine (NGD,EFG)

Figure 5: Comparison of the Fisher (NGD) and the empirical Fisher (EFGD) as preconditioners on additional linear regression problems. The second row shows the cosine similarity between the EF-preconditioned gradient and the natural gradient at each step on the path taken by EFGD. 0 20 Iteration 10 1

10 2

Boston NGD EFGD 

Figure 6: Linear regression on the Boston dataset with different starting points (each line is a differ-ent initialization). When the optimization starts close to the minimum (low initial loss), the empir-ical Fisher is a good approximation to the Fisher and there are very few differences with NGD, but the performance degrades as the optimization pro-cedure starts farther away (large initial loss). 20
