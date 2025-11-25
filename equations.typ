#import "@preview/physica:0.9.3": *

= InnerPiSSA Equations

== Adapter

$ y = x W_"res" + x V R(alpha) (S + alpha dot.c Delta S) U^T $

where:
- $R(alpha)$ is the Cayley rotation that reverses with sign: $R(-alpha) = R(alpha)^T$
- $Delta S$ is learnable singular value scaling
- $alpha$ is the steering coefficient

== Loss Function (Compact)

$ cal(L) = underbrace(-op("proj")(Delta h_pi, v), "separation") + underbrace(sum_t bb(1)[Delta ell_t > tau] dot.c Delta ell_t, "coherence constraint") + underbrace(op("hinge")(delta_(-1), delta_0, delta_(+1)), "monotonic constraint") $

where:
- $Delta h_pi = h_"cho" - h_"rej"$ (hidden state difference)
- $v$ is the frozen PCA preference direction
- $Delta ell_t = ell_t^pi - ell_t^"ref"$ is per-token NLL degradation
- $delta_alpha = log p("cho") - log p("rej")$ at coefficient $alpha$

== Loss Function (Expanded)

$ cal(L) = underbrace(-1/(|cal(B)|) sum_(i in cal(B)) ((h_"cho"^i - h_"rej"^i)^T v)/(||v||_2), "separation along preference direction") \
  + underbrace(sum_(t=1)^T bb(1)[ell_t^pi - ell_t^"ref" > tau] dot.c (ell_t^pi - ell_t^"ref"), "coherence constraint") \
  + underbrace(max(0, delta_(-1) - delta_0) + max(0, delta_0 - delta_(+1)), "monotonic constraint") $

where:
- $h_"cho", h_"rej"$ are hidden states from contrastive prompt prefixes
- $v = op("PCA")(h_("ref,cho") - h_("ref,rej"))$ is frozen reference direction
- $ell_t = -log p(x_t | x_(< t))$ is per-token NLL
- $delta_alpha = log p("cho") - log p("rej")$ at coefficient $alpha$
- Loss computed jointly for $alpha in {-1, +1}$ to enforce bidirectionality

== Monotonic Constraint Detail

The monotonic constraint ensures:
$ delta_(-1) < delta_0 < delta_(+1) $

This prevents saddle points where both steering directions degrade performance.


