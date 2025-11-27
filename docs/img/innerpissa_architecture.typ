#import "@preview/cetz:0.4.2": canvas, draw, tree
#import "@preview/physica:0.9.3": *

#canvas({
  import draw: *
  
  // Title
  content((0, 6), [*InnerPiSSA Adapter Architecture*], anchor: "center")
  
  // Input
  rect((0, 4.5), (1.5, 5), name: "x", fill: rgb("#e3f2fd"))
  content("x.center", [$bold(x)$], anchor: "center")
  
  // Fork
  line("x.south", (0.75, 4), mark: (end: ">"))
  line((0.75, 4), (-2, 3.5), mark: (end: ">"))
  line((0.75, 4), (3.5, 3.5), mark: (end: ">"))
  
  // Left branch - frozen residual
  rect((-3, 2.5), (-1, 3.5), fill: rgb("#bbdefb"), name: "wres")
  content("wres.center", [$W_"res"$\ (frozen)], anchor: "center")
  
  // Right branch - adaptive path
  // V rotation (trapezoid approximation)
  merge-path({
    line((3, 3.5), (4, 3.5))
    line((4, 3.5), (4.5, 2.5))
    line((4.5, 2.5), (2.5, 2.5))
    line((2.5, 2.5), (3, 3.5))
  }, fill: rgb("#fff3e0"), stroke: rgb("#ff9800") + 2pt, close: true, name: "v")
  content((3.5, 3), [$bold(V) dot.c R(alpha)$], anchor: "center")
  content((3.5, 2.7), text(size: 9pt, fill: rgb("#666"))[(rotatable)], anchor: "center")
  
  // Rotation icon on arrow
  line((4.5, 3), (5.5, 3), mark: (end: ">"))
  circle((5, 3), radius: 0.15, stroke: rgb("#ff9800"))
  arc((5, 3), start: 45deg, stop: 315deg, radius: 0.12, stroke: rgb("#ff9800"))
  
  // S + αΔS
  rect((5.5, 2.5), (7.5, 3.5), fill: rgb("#ffe0b2"), stroke: rgb("#f57c00") + 2pt, name: "s")
  content("s.center", [$S + alpha dot.c Delta S$], anchor: "center")
  content((6.5, 2.7), text(size: 9pt, fill: rgb("#666"))[$(S"-space": h@V)$], anchor: "center")
  
  // Second rotation
  line((7.5, 3), (8.5, 3), mark: (end: ">"))
  circle((8, 3), radius: 0.15, stroke: rgb("#ff9800"))
  arc((8, 3), start: 45deg, stop: 315deg, radius: 0.12, stroke: rgb("#ff9800"))
  
  // U rotation
  merge-path({
    line((8.5, 3.5), (9.5, 3.5))
    line((9.5, 3.5), (10, 2.5))
    line((10, 2.5), (8, 2.5))
    line((8, 2.5), (8.5, 3.5))
  }, fill: rgb("#fff3e0"), stroke: rgb("#ff9800") + 2pt, close: true, name: "u")
  content((9, 3), [$R(alpha)^T dot.c bold(U)^T$], anchor: "center")
  content((9, 2.7), text(size: 9pt, fill: rgb("#666"))[(rotatable)], anchor: "center")
  
  // Merge point
  line("wres.south", (-2, 1))
  line((9.5, 2.5), (9.5, 1))
  line((-2, 1), (3.75, 1), mark: (end: ">"))
  line((9.5, 1), (3.75, 1))
  content((3.75, 1.3), [$+$], anchor: "center")
  
  // Output
  rect((3, 0.5), (4.5, 1), name: "y", fill: rgb("#e8f5e9"))
  content("y.center", [$bold(y)$], anchor: "center")
  line((3.75, 1), "y.north", mark: (end: ">"))
  
  // Equations
  content((6.5, -0.5), text(size: 11pt)[
    $y = x W_"res" + x V R(alpha) (S + alpha dot.c Delta S) U^T$
  ], anchor: "center")
  
  // Legend
  rect((-3, -1.5), (1, -0.8), stroke: none)
  content((-2.7, -1), text(size: 9pt)[■ Frozen static], anchor: "west", fill: rgb("#bbdefb"))
  content((-2.7, -1.3), text(size: 9pt)[□ Frozen rotatable], anchor: "west", stroke: rgb("#ff9800"))
  content((-0.7, -1), text(size: 9pt)[■ Learnable], anchor: "west", fill: rgb("#ffe0b2"))
})