# Per-decision routing: upper bound (RQ4)

Mean decision regret on the same states (D judged top-1). Best single arm is chosen in-sample; the capability router picks, per state, the arm that is best on the other states with the same primary capability; the oracle router picks the best arm per state (an upper bound needing all three arms). Gap = best single minus router, 95% bootstrap CI over states.

| Regime | n | States | Best single arm | Capability router (gain) | Oracle router (max gain) | Arms differ |
|---|---|---|---|---|---|---|
| ALFWorld + ScienceWorld (main) | 32 | 698 | F 0.255 | 0.255 (+0.000 [0.000..0.000]) | 0.112 (+0.143 [0.096..0.202]) | 34% |
| ALFWorld + ScienceWorld (main) | 64 | 698 | F 0.286 | 0.242 (+0.044 [-0.009..0.105]) | 0.093 (+0.193 [0.128..0.271]) | 28% |
| ALFWorld + ScienceWorld (main) | 128 | 698 | G 0.258 | 0.226 (+0.032 [-0.033..0.099]) | 0.079 (+0.179 [0.132..0.235]) | 27% |
| ALFWorld | 256 | 193 | G 0.606 | 0.570 (+0.036 [-0.145..0.212]) | 0.316 (+0.290 [0.181..0.425]) | 36% |
| ScienceWorld test | 256 | 182 | F 0.066 | 0.088 (-0.022 [-0.055..0.000]) | 0.011 (+0.055 [0.011..0.110]) | 15% |
| ScienceWorld seen | 256 | 323 | G 0.096 | 0.090 (+0.006 [-0.031..0.040]) | 0.025 (+0.071 [0.034..0.115]) | 20% |
| ByteSized32 (gold unseen) | 256 | 210 | F 0.591 | 0.643 (-0.052 [-0.195..0.090]) | 0.300 (+0.290 [0.176..0.419]) | 48% |
| ALFWorld + ScienceWorld (main) | 256 | 698 | G 0.239 | 0.213 (+0.026 [-0.030..0.082]) | 0.102 (+0.138 [0.100..0.183]) | 23% |
| ALFWorld + ScienceWorld (main) | 512 | 698 | G 0.232 | 0.218 (+0.014 [-0.043..0.067]) | 0.099 (+0.133 [0.096..0.178]) | 22% |
| ALFWorld + ScienceWorld (main) | 1024 | 698 | F 0.212 | 0.213 (-0.001 [-0.026..0.029]) | 0.090 (+0.122 [0.070..0.181]) | 18% |

Rows: every regime at n = 256, and the pooled main regimes at every budget.
