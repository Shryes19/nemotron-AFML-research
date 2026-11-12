Decision utility at n = 256. Regret: mean [95% bootstrap CI], lower is better. Gold-hit: share of states where the chosen action is the expert's. EM: exact match of the top-1 predicted next state.

| Regime | Arm | States | Regret | Gold-hit | EM |
|---|---|---|---|---|---|
| ALFWorld: new games | oracle (true next states) | 193 | 0 | 0.3731 |  |
| ALFWorld: new games | random choice | 193 | 1.61 [1.39, 1.83] | 0.20 |  |
| ALFWorld: new games | F (SFT) | 193 | 0.62 [0.42, 0.84] | 0.342 | 0.4953 |
| ALFWorld: new games | D (JEPA) top-1 | 193 | 0.98 [0.73, 1.24] | 0.3368 | 0.2124 |
| ALFWorld: new games | D (JEPA) top-k | 193 | 0.78 [0.58, 1.03] | 0.3782 | 0.2124 |
| ALFWorld: new games | D (JEPA) native | 193 | 1.24 [1.02, 1.51] | 0.2591 |  |
| ScienceWorld: new variations (test split) | oracle (true next states) | 182 | 0 | 0.6319 |  |
| ScienceWorld: new variations (test split) | random choice | 182 | 1.47 [1.26, 1.70] | 0.20 |  |
| ScienceWorld: new variations (test split) | F (SFT) | 182 | 0.07 [0.02, 0.13] | 0.6319 | 0.6648 |
| ScienceWorld: new variations (test split) | D (JEPA) top-1 | 182 | 0.26 [0.15, 0.40] | 0.5714 | 0.4505 |
| ScienceWorld: new variations (test split) | D (JEPA) top-k | 182 | 0.37 [0.25, 0.50] | 0.5549 | 0.4505 |
| ScienceWorld: new variations (test split) | D (JEPA) native | 182 | 0.96 [0.74, 1.19] | 0.2967 |  |
| ScienceWorld: held-out variations, seen types | oracle (true next states) | 323 | 0 | 0.5232 |  |
| ScienceWorld: held-out variations, seen types | random choice | 323 | 1.38 [1.23, 1.53] | 0.20 |  |
| ScienceWorld: held-out variations, seen types | F (SFT) | 323 | 0.12 [0.04, 0.24] | 0.517 | 0.7084 |
| ScienceWorld: held-out variations, seen types | D (JEPA) top-1 | 323 | 0.33 [0.24, 0.43] | 0.4211 | 0.4681 |
| ScienceWorld: held-out variations, seen types | D (JEPA) top-k | 323 | 0.48 [0.36, 0.60] | 0.4427 | 0.4681 |
| ScienceWorld: held-out variations, seen types | D (JEPA) native | 323 | 1.05 [0.88, 1.22] | 0.2043 |  |
| ByteSized32: known states, new actions | oracle (true next states) | 210 | 0 | 0.6048 |  |
| ByteSized32: known states, new actions | random choice | 210 | 2.12 [1.89, 2.36] | 0.20 |  |
| ByteSized32: known states, new actions | F (SFT) | 210 | 0.59 [0.42, 0.76] | 0.4905 | 0.4733 |
| ByteSized32: known states, new actions | D (JEPA) top-1 | 210 | 1.46 [1.19, 1.75] | 0.3381 | 0.2467 |
| ByteSized32: known states, new actions | D (JEPA) top-k | 210 | 1.31 [1.04, 1.61] | 0.3571 | 0.2467 |
| ByteSized32: known states, new actions | D (JEPA) native | 210 | 1.61 [1.33, 1.91] | 0.2476 |  |
