# RQ3 predictor: final results (14 Nov 2026, G + F + D)

Data: `results/features/feature_matrix.csv` (PR #15, all 4 features incl. Recall@5) and
`results/utility/labels_D{top1,topk,native}.csv` (PR #14, with the new `tie` column), plus
the SFT probe (`results/sft/sft_probe_scores.csv`) for the Feature 1 ablation.

    python scripts/rq3_predictor.py --matrix results/features/feature_matrix.csv \
        --labels results/utility/labels_Dtop1.csv --sft-probe results/sft/sft_probe_scores.csv \
        --out results/rq3 --perm 200

**Rows.** Main analysis: ALFWorld + ScienceWorld, 32 labelled matrix rows = **26 independent
labels** (ALFWorld is labelled per task family). ByteSized32 is analysed separately (29
games, "known state, new action"). **Ties:** 8 of 32 main rows (and 8 of 29 ByteSized32
rows) have exactly equal G and F regret. They are left out of the yes/no scores
(balanced accuracy) and kept in the regression (Spearman), since a tie is a real "gap = 0".

**Models.** Ridge + gradient-boosted trees on the regret *difference*. CV is
leave-one-(env, family)-out. Out-of-fold Spearman ρ = rank agreement between predicted and
true gap. Permutation p values come from 200 label shuffles with the whole CV rerun.

## 1. G vs F, the main question now (main regimes, D top-1 labels)

| Features | Rows scored (+ ties) | Balanced acc | p | OOF ρ | p |
|---|---|---|---|---|---|
| Features 2–4 + JEPA probe | 24 (+8) | 0.46 | 0.58 | −0.30 | 0.82 |
| JEPA probe only | 24 (+8) | 0.52 | 0.34 | −0.39 | 0.84 |
| SFT probe only | 24 (+8) | 0.21 | 0.96 | −0.46 | 0.93 |
| Features 2–4 only | 24 (+8) | 0.34 | — | −0.54 | — |

**There's no signal.** Every feature set is at or below chance (balanced accuracy
0.21–0.54, all permutation p ≥ 0.34). The negative ρ values are what leave-one-out gives
when there is nothing to learn: removing a row pushes the training mean the other way.
This matches Saanvi's finding that G and F are statistically tied (only 1 of 27 G − F
differences is significant). There is no real winner for the features to predict.
Leave-one-environment-out gives the same picture (balanced accuracy 0.37–0.50).

## 2. Which arm wins overall (G / F / D)

Accuracy counts a prediction as right if it names the winner or any arm in an exact tie.

| Features | Accuracy (95% CI) |
|---|---|
| Features 2–4 + JEPA probe | 0.62 (0.45–0.77) |
| JEPA probe only | 0.69 (0.51–0.82) |
| SFT probe only | 0.44 (0.28–0.61) |
| **Always guess F (constant)** | **0.72** |
| Always guess G | 0.53 |

No model beats simply guessing "F". (The leave-one-out "majority of the training fold"
baseline is unfairly low here, at 0.25, because removing a row tips the majority to the
other class. So the constant guess is the fair comparison.)

## 3. Is the specialist worth it? D vs the generalists

- **D vs G:** G has lower regret on every main row, so there is nothing to predict.
- **D vs F:** F wins every scored row, so the yes/no task can't be tested (balanced
  accuracy 0.50, p ≥ 0.77). But the **size** of D's gap to F *can* be ranked, and
  **only the JEPA probe ranks it, in all three D modes**:

| D mode | JEPA probe only: OOF ρ (p) | SFT probe only: OOF ρ (p) |
|---|---|---|
| top-1 (main) | **+0.47 (0.010)** | +0.02 (0.33) |
| top-k | **+0.34 (0.045)** | +0.05 (0.26) |
| native | **+0.29 (0.040)** | +0.14 (0.17) |

  This is the one positive RQ3 result. JEPA's own cheap probe (0.02 GPU-h) tells you on
  which games JEPA falls furthest behind fine-tuning. The SFT probe, which costs more GPU
  time than F's own training, does not. Caveats: it is one family of tests among several
  (multiple comparisons), and it fades when the other features are added
  (Features 2–4 + JEPA probe: ρ ≈ +0.1).

## 4. ByteSized32, reported separately

29 games: F 16, G 10, D 3. G vs F: JEPA probe only gives balanced accuracy 0.58 (p 0.075),
ρ +0.24 (p 0.085). Features 2–4 + JEPA probe gives balanced accuracy 0.53 (p 0.28). SFT
probe only gives 0.47 (p 0.67). The winner model with the JEPA probe alone reaches 0.76
against 0.69 for always guessing F. These are suggestive at best, and nothing reaches
p < 0.05.

## Answer to RQ3, in one paragraph

On our regimes, the cheap ex-ante features **cannot predict which arm wins**, mainly
because there is little to predict. The two generalists (G and F) are statistically tied,
and both beat the specialist (D) everywhere. A constant "pick F" is as good as any learned
predictor. The specialist is never worth it here. The only reliable ex-ante signal is
that **the JEPA embedding probe ranks how far the specialist falls behind** (ρ = 0.29–0.47,
p ≤ 0.045 in all three D modes), while the more expensive SFT text probe carries no signal.
With ~26 independent labels and one seed, these are small-sample results.

Previous D+F-only run (12 Nov): `git log -- results/rq3/README.md`. JSON for every run:
`results/rq3/final/`.
