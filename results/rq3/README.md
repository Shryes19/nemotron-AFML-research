# RQ3 predictor: first results (12 Nov 2026, D + F only, G pending)

Run with Saanvi's labels (`results/utility/labels_Dtop1.csv` on `saanvi/utility-results`,
n = 256, D judged top-1):

    python scripts/rq3_predictor.py --labels results/utility/labels_Dtop1.csv \
        --sft-probe results/sft/sft_probe_scores.csv --out results/rq3

Main rows: ALFWorld + ScienceWorld. That is 32 matrix rows with labels, but only **26
independent labels**, because ALFWorld is labelled per task family (6) and the 2
identify-life-stages types have no states. ByteSized32 is analysed separately (team
decision).

## 1. The yes/no task can't be tested yet

With G missing, the only decision is "does D beat F?", and **F wins 31 of 32 rows**. The
only D win is ScienceWorld `chemistry-mix`, and it is not significant. So "always say F"
already scores 97%. Every model also scores 97% (balanced accuracy 0.50), and the
permutation p = 0.60. Accuracy here carries no information. It becomes meaningful only
when G is added.

## 2. Ranking the size of D's gap to F (regression on regret_F − regret_D)

Out-of-fold Spearman between predicted and true gap. CV is leave-one-(env, family)-out:

| Features | Spearman | Permutation p (200 shuffles) |
|---|---|---|
| Features 2–4 + JEPA probe | +0.10 | — |
| Features 2–4 only | +0.07 | — |
| Features 2–4 + SFT probe | +0.10 | — |
| **JEPA probe only** | **+0.47** | **0.010** |
| SFT probe only | +0.03 | 0.289 |

Leave-one-environment-out (train on ALFWorld, test on ScienceWorld, and the reverse):
every feature set gives a Spearman between −0.13 and +0.11.

## 3. Single-feature correlations with the gap (26 independent units, no model)

| Feature | D top-1 labels | D top-k | D native |
|---|---|---|---|
| JEPA probe (distance) | +0.34 (p 0.09) | −0.28 (p 0.17) | **−0.46 (p 0.02)** |
| SFT probe (F1) | −0.32 (p 0.11) | +0.25 (p 0.21) | +0.33 (p 0.10) |
| obs length | +0.35 (p 0.08) | +0.04 | −0.06 |
| zero-overlap fraction | −0.35 (p 0.08) | +0.14 | +0.02 |
| containment/location rate | +0.10 | **+0.44 (p 0.03)** | +0.37 (p 0.06) |

ByteSized32 on its own (29 games, D wins 4): neither probe relates to the gap (|ρ| ≤ 0.14).
Only the object-state rate does (ρ = +0.44, p = 0.02).

## Honest reading

- **For D vs F, the answer to "is the specialist worth it?" is simply no.** F has lower
  regret on 31 of 32 rows at n = 256, and at every budget in Saanvi's table. No feature is
  needed to predict that.
- The one positive signal is that the **JEPA probe alone ranks the D–F gap out of sample**
  (ρ = 0.47, p = 0.01), while the SFT probe does not. That is the direction the plan
  hoped for. But it should be reported as *suggestive*, not as a finding:
  - it is one of several tests (2 probes × 3 D modes × 2 CV schemes), so there is a
    multiple-comparisons risk
  - **its sign flips between D modes** (+0.34 for top-1, −0.46 for native)
  - it disappears when the other features are added, and in the leave-one-env-out test
- A likely reason for the sign flip is a floor effect. F's regret is 0 on 11 of 20
  ScienceWorld types, so "the gap" mostly measures how badly D does, and that depends on
  the decoding mode.
- **Feature 1 cost** (Siddhant's GPU log): the SFT probe costs 0.7–3.1 GPU-h per
  environment against 0.02 h for the JEPA probe, which is more than F's own training. So
  even before the results, the SFT probe fails the "cheap" requirement.
- G may change all of this. With three arms, the yes/no tasks get both classes. Rerun the
  same command once `labels.csv` includes G.
