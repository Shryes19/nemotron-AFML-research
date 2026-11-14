# Compute crossover (RQ2): regret vs GPU-hours, D judged top1

Cost = building + running the world model at that budget (RTX 5070 GPU-hours; evaluation excluded). Crossover = cheapest budget at which the specialist's regret is at or below G's best regret at any budget.

Note: G has no training, so its cost is almost all generation with the 7B model and grows with the number of predictions; F and D pay mostly once (training, encoding) and predict cheaply. These GPU-hours are for this evaluation's predictions (one pass over each held-out set).

| Env | Specialist | Specialist best regret (n, GPU-h) | G best regret (n, GPU-h) | Crossover GPU-h (n) | Best-vs-best CIs overlap |
|---|---|---|---|---|---|
| ALFWorld | F | 0.580 (n=1024, 0.19 h) | 0.534 (n=512, 0.35 h) | never | yes |
| ALFWorld | D | 0.803 (n=1024, 0.15 h) | 0.534 (n=512, 0.35 h) | never | yes |
| ScienceWorld | F | 0.071 (n=1024, 0.35 h) | 0.099 (n=256, 1.08 h) | 0.19 (n=32) | yes |
| ScienceWorld | D | 0.242 (n=1024, 0.53 h) | 0.099 (n=256, 1.08 h) | never | no |
| ByteSized32 (known state, new action) | F | 0.348 (n=1024, 0.20 h) | 0.605 (n=1024, 0.16 h) | 0.08 (n=128) | yes |
| ByteSized32 (known state, new action) | D | 1.200 (n=1024, 0.04 h) | 0.605 (n=1024, 0.16 h) | never | no |

Points (GPU-h, regret [95% CI]):

- ALFWorld, G: n=32: 0.31 h, 0.824 [0.61, 1.04]; n=64: 0.33 h, 0.663 [0.45, 0.87]; n=128: 0.34 h, 0.632 [0.43, 0.83]; n=256: 0.33 h, 0.606 [0.41, 0.80]; n=512: 0.35 h, 0.534 [0.34, 0.73]; n=1024: 0.35 h, 0.539 [0.35, 0.73]
- ALFWorld, F: n=32: 0.10 h, 0.715 [0.49, 0.94]; n=64: 0.05 h, 0.648 [0.44, 0.86]; n=128: 0.05 h, 0.622 [0.41, 0.83]; n=256: 0.08 h, 0.622 [0.41, 0.83]; n=512: 0.11 h, 0.617 [0.40, 0.83]; n=1024: 0.19 h, 0.580 [0.36, 0.80]
- ALFWorld, D: n=32: 0.15 h, 0.938 [0.70, 1.17]; n=64: 0.15 h, 1.021 [0.77, 1.27]; n=128: 0.15 h, 1.026 [0.77, 1.28]; n=256: 0.15 h, 0.979 [0.72, 1.24]; n=512: 0.15 h, 0.902 [0.65, 1.15]; n=1024: 0.15 h, 0.803 [0.56, 1.05]
- ScienceWorld, G: n=32: 1.06 h, 0.152 [0.11, 0.20]; n=64: 1.08 h, 0.156 [0.10, 0.22]; n=128: 1.08 h, 0.115 [0.07, 0.16]; n=256: 1.08 h, 0.099 [0.05, 0.14]; n=512: 1.11 h, 0.117 [0.08, 0.16]; n=1024: 1.11 h, 0.105 [0.06, 0.15]
- ScienceWorld, F: n=32: 0.19 h, 0.079 [0.04, 0.12]; n=64: 0.22 h, 0.148 [0.07, 0.23]; n=128: 0.20 h, 0.121 [0.07, 0.17]; n=256: 0.21 h, 0.103 [0.04, 0.17]; n=512: 0.30 h, 0.103 [0.04, 0.17]; n=1024: 0.35 h, 0.071 [0.02, 0.12]
- ScienceWorld, D: n=32: 0.60 h, 0.628 [0.51, 0.74]; n=64: 0.60 h, 0.436 [0.33, 0.54]; n=128: 0.56 h, 0.333 [0.25, 0.41]; n=256: 0.54 h, 0.307 [0.23, 0.38]; n=512: 0.54 h, 0.313 [0.24, 0.39]; n=1024: 0.53 h, 0.242 [0.16, 0.32]
- ByteSized32 (known state, new action), G: n=32: 0.13 h, 0.976 [0.75, 1.20]; n=64: 0.14 h, 0.767 [0.56, 0.97]; n=128: 0.14 h, 0.852 [0.65, 1.05]; n=256: 0.14 h, 0.633 [0.46, 0.81]; n=512: 0.15 h, 0.610 [0.44, 0.78]; n=1024: 0.16 h, 0.605 [0.44, 0.77]
- ByteSized32 (known state, new action), F: n=32: 0.06 h, 1.005 [0.76, 1.25]; n=64: 0.07 h, 0.943 [0.72, 1.16]; n=128: 0.08 h, 0.571 [0.39, 0.76]; n=256: 0.10 h, 0.591 [0.42, 0.76]; n=512: 0.14 h, 0.414 [0.25, 0.58]; n=1024: 0.20 h, 0.348 [0.18, 0.51]
- ByteSized32 (known state, new action), D: n=32: 0.04 h, 1.557 [1.28, 1.83]; n=64: 0.04 h, 1.533 [1.25, 1.81]; n=128: 0.03 h, 1.448 [1.16, 1.73]; n=256: 0.03 h, 1.457 [1.18, 1.74]; n=512: 0.03 h, 1.338 [1.05, 1.63]; n=1024: 0.04 h, 1.200 [0.91, 1.49]
