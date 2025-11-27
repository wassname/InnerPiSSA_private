| Configuration   |   Main Metric |   Val Loss |   N Runs | Result                |
|:----------------|--------------:|-----------:|---------:|:----------------------|
| Full InnerPiSSA |         666.3 |        9.4 |       49 | **Baseline** ✓        |
| No S scaling    |        1051   |       10.6 |        1 | Better? (investigate) |
| No V rotation   |          29   |       34.7 |        1 | **Catastrophic** ❌   |
| LoRA adapter    |           0   |        9.3 |        3 | **Catastrophic** ❌   |