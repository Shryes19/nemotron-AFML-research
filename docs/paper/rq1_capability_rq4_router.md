# RQ1 by capability and RQ4 routing bound (draft text, Siddanth)

For Saanvi to fold into Section 5 (results) and Section 6 (discussion). Numbers from
`results/capability_regret/capability_regret.md` and `results/router/router.md` (n = 256, D judged
top-1, same branched states for every arm). Rebuild: `python scripts/capability_regret.py` and
`python scripts/router_bound.py` (CPU, seconds).

---

### 5.x Where the gaps are: regret by capability (RQ1)

We tag each branched state with the capabilities of its best candidate transition, i.e. the outcome a
world model must predict well for the agent to choose right (Figure X, `fig_capability_regret`). On the
two main regimes pooled (698 states):

- **D's deficit is broad, not tied to one capability.** F has significantly lower regret than D when
  the right move involves containment or location (0.31 vs 0.55, paired difference −0.24) or reveals
  hidden information (0.36 vs 0.54, −0.17). The largest gap is on states whose best move changes
  nothing visible (0.27 vs 0.83, −0.57), where D's nearest-state decoding returns an outcome from a
  different context. On ByteSized32 the same pattern is stronger: F − D is −0.92 for containment and
  −2.05 for partially observable states.
- **G and F are indistinguishable per capability on the pooled regimes:** no paired G − F difference
  is significant. ScienceWorld test is the exception: F is better on containment (+0.11 for G − F)
  and on partially observable states (+0.13).
- **Partially observable and containment states are the hardest for every arm** (main regimes:
  0.36–0.55 regret, against 0.09–0.30 for object-state changes). No world model can predict contents
  it has never seen.
- **Numeric/physical states are few** (28 main, 26 ByteSized32), so their intervals are wide. On
  ByteSized32 all three arms are equally poor there (1.15–1.27). This fits our finding that 73% of
  ByteSized32's numeric changes never appear in the text.

Caveats: one state can carry several tags; small tag groups (action failure: 14 states; numeric: 28)
are reported but not interpreted; with several comparisons, single significant cells should be read
as suggestive.

### 5.y Could routing per decision help? (RQ4, as an upper bound)

RQ4 asked whether choosing the world model per decision beats one choice for everything. We did not
train a router; instead we bound what one could achieve on our states (`results/router/router.md`):

| (main regimes, n = 256) | Mean regret |
|---|---|
| Best single arm (G) | 0.239 |
| Capability router (per state, the arm best on other states with the same capability) | 0.213 (gain +0.026, CI −0.030 to +0.082) |
| Oracle router (best arm on each state; needs all three arms) | 0.102 (max gain +0.138, CI +0.100 to +0.183) |

- **There is headroom in principle:** an oracle that knew the best arm for every state would cut
  regret by more than half, at every budget (max gain +0.12 to +0.19).
- **Knowing the capability does not capture it:** routing by the best move's capability gains nothing
  significant at any budget. The arms disagree on only 18–34% of states, and those disagreements are
  not organised by capability.
- **Part of the oracle's headroom is noise.** With one seed and a stochastic judge, two arms can differ
  on a state by chance, and the oracle takes every lucky draw. The bound is therefore optimistic.

So RQ4's answer for this setting is that a per-decision router would need signals beyond the kind of
change involved, perhaps per-state uncertainty, and any real router would also pay for running several
arms. We leave it to future work.

*Validity note: each decision here is a single step at a fixed state, so this offline comparison is
sound. It would not be for multi-step agent rollouts, where a router's choices change the states that
follow.*
