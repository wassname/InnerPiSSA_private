#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/physica:0.9.3": *

#canvas(length: 1cm, {
  import draw: *
  
  // Three circles for α = -1, 0, +1
  for (i, alpha, x) in ((0, $-1$, 1.5), (1, $0$, 5), (2, $+1$, 8.5)) {
    // Reference model boundary (dashed)
    circle((x, 3), radius: 1, stroke: (dash: "dashed", paint: rgb("#ffc107")))
    
    // Preference vectors
    line((x, 3), (x - 0.5, 3 + 0.7), mark: (end: ">"), stroke: rgb("#2196f3") + 1.5pt, name: "pi" + str(i))
    line((x, 3), (x + 0.5, 3 - 0.7), mark: (end: ">"), stroke: rgb("#9c27b0") + 1.5pt, name: "ref" + str(i))
    
    // Labels
    content((x, 4.5), text(size: 12pt)[$alpha = alpha$], anchor: "center")
    content((x - 0.8, 3.9), text(size: 9pt, fill: rgb("#2196f3"))[$( Delta h@V )_pi$], anchor: "center")
    content((x + 0.8, 2.1), text(size: 9pt, fill: rgb("#9c27b0"))[$( Delta h@V )_"ref"$], anchor: "center")
  }
  
  // Separation indicator
  line((1.2, 1.5), (1.8, 1.5), mark: (start: "<", end: ">"), stroke: rgb("#f44336"))
  content((1.5, 1.2), text(size: 8pt)[reversed], anchor: "center")
  
  line((4.7, 1.5), (5.3, 1.5), mark: (start: "<", end: ">"), stroke: rgb("#666"))
  content((5, 1.2), text(size: 8pt)[baseline], anchor: "center")
  
  line((8.2, 1.5), (8.8, 1.5), mark: (start: "<", end: ">"), stroke: rgb("#4caf50"))
  content((8.5, 1.2), text(size: 8pt)[enhanced], anchor: "center")
  
  // Loss equation
  content((5, 0.3), text(size: 11pt)[
    $cal(L) = -op("proj")( Delta h_pi@V, v) + sum_t bb(1)[Delta ell_t > tau] dot.c Delta ell_t + op("hinge")(delta_(-1), delta_0, delta_(+1))$
  ], anchor: "center")
  content((5, -0.1), text(size: 9pt, fill: rgb("#666"))[
    separation + coherence + monotonic
  ], anchor: "center")
})