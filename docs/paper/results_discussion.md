# Results, discussion and limitations (draft, Saanvi)

Status (12 Nov 2026): filled from the GPU scoring run of 12 Nov (arms F and D, all four
regimes, all six budgets; `results/utility/`, `results/figures/`). Still open, marked **TBD**:
arm G (Rohini's prediction files have not been scored yet) and the RQ3 predictor numbers
(`results/rq3/`, Shryes). When G is scored, rerun `scripts/run_utility.sh` and update the
sentences that say "F has the lowest regret": they compare F and D only.
Every claim names its figure or table, as the plan asks.

---

## 5 Results

### 5.1 Evaluation protocol in brief

Each arm (G: retrieval-augmented LLM, model **TBD from Rohini**; F: Qwen2.5-1.5B + LoRA; D: JEPA with
nearest-state decoding) predicts the next state for K = 5 candidate actions at a set of
*branched states*: the expert action plus four verb-stratified distractors, with
observe-only actions (`look`, `inventory`, `task`) excluded. A frozen judge
(Qwen2.5-7B-Instruct-AWQ) rates each predicted outcome 0–10 for progress toward the goal;
the arm picks the candidate with the highest rating. **Decision regret** is the best true
rating minus the true rating of the arm's choice (0 = optimal). The same judge rates the
true next states, so differences between arms come from their predictions only.

Two reference points frame every number:
- **Oracle**: choosing on the true next states. Regret 0 by construction; its gold-hit
  rate (how often it picks the expert action) measures how well the judge's ratings agree
  with the expert.
- **Random choice**: the expected regret of picking one of the five candidates uniformly.

We report four regimes separately, because they test different things (Table 1):

| Regime | What is new at test time | States | Groups |
|---|---|---|---|
| ALFWorld | new games (Word2World test split) | 193 | 6 task families |
| ScienceWorld test | new variations, 12 task types | 182 | 12 task types |
| ScienceWorld seen | held-out variations of 20 seen task types | 323 | 20 task types |
| ByteSized32 | only the actions: states are training states | 328 (210 gold-unseen) | 31 games |

Table 1: branched states per regime. ScienceWorld seen uses train-split variations that we
reserved before any model was trained and excluded from all training data. For ByteSized32,
118 of 328 states have their gold transition in the arms' training data; we report the 210
states where it is not.

### 5.2 Text fidelity: F leads everywhere

At n = 256, exact match of the top-1 predicted next state (`results/sft/README.md`,
`results/jepa_phase1/README.md`):

| Held-out set | F (SFT) | D (JEPA) | D's reach ceiling | Lookup |
|---|---|---|---|---|
| ALFWorld, new games | **56.2%** | 43.8% | 47.5% | 44.6% |
| ScienceWorld test | **69.0%** | 47.3% | 53.5% | 41.9% |
| ScienceWorld seen | **71.3%** | 52.4% | 58.2% | 47.0% |
| ByteSized32 | **55.6%** | 26.8% | 31.9% | 16.9% |

Table 2: fidelity at n = 256. F has the highest fidelity at every budget in every
environment (+2 to +35 points over D). D's decoder can only return a state it has seen,
which caps its exact match ("reach"); within that cap D reaches 87–97% of the ceiling, so
the gap is the decoder and the buffer, not JEPA training.

### 5.3 Decision utility

Figure 1 (`fig1_regret_vs_n`) shows mean regret against the training budget n for every
arm, with D in three modes: top-1 decoded text, top-k decoded texts, and native (ranking
candidates in embedding space, no judge). Table 3 (`main_table.md`) gives the n = 256 values
with 95% bootstrap CIs, plus oracle and random choice.

| Regime | Random choice | F (SFT) | D top-1 | D top-k | D native |
|---|---|---|---|---|---|
| ALFWorld, new games | 1.61 [1.39, 1.83] | **0.62** [0.42, 0.84] | 0.98 [0.73, 1.24] | 0.78 [0.58, 1.03] | 1.24 [1.02, 1.51] |
| ScienceWorld test | 1.47 [1.26, 1.70] | **0.07** [0.02, 0.13] | 0.26 [0.15, 0.40] | 0.37 [0.25, 0.50] | 0.96 [0.74, 1.19] |
| ScienceWorld seen | 1.38 [1.23, 1.53] | **0.12** [0.04, 0.24] | 0.33 [0.24, 0.43] | 0.48 [0.36, 0.60] | 1.05 [0.88, 1.22] |
| ByteSized32, gold unseen | 2.12 [1.89, 2.36] | **0.59** [0.42, 0.76] | 1.46 [1.19, 1.75] | 1.31 [1.04, 1.61] | 1.61 [1.33, 1.91] |

Table 3: mean decision regret at n = 256 (judge scale 0–10, lower is better), 95% bootstrap
CIs. G: **TBD**.

- **Every arm carries decision information.** All arms and modes are below random choice in
  every regime (Table 3); the weakest, D native, still removes 23–35% of random choice's
  regret.
- **F has the lowest regret in every regime at every budget** (Figure 1). At n = 256 its CI
  does not overlap D top-1's on ScienceWorld test and seen or on ByteSized32; on ALFWorld the
  intervals overlap (0.42–0.84 vs 0.73–1.24).
- **No crossover to D.** D never has lower regret than F at any budget in any regime
  (`crossovers.csv`). The gap narrows with data on ScienceWorld test (D top-1 0.58 → 0.18
  from n = 32 to 1024, F 0.13 → 0.05) but does not close. F is already good at n = 32
  (regret 0.05–0.13 on ScienceWorld, 0.72 on ALFWorld, 1.00 on ByteSized32). G vs F: **TBD**.
- **F is close to the ceiling on ScienceWorld.** In 13 of 21 ScienceWorld task types F's
  regret at n = 256 is exactly 0, so little room is left for any specialist there.
- **D's three modes.** Native, JEPA's own choice in embedding space without decoding, is the
  worst mode everywhere (0.96–1.61 at n = 256). Top-k beats top-1 on ALFWorld (0.78 vs 0.98)
  and ByteSized32 (1.31 vs 1.46), where top-1 text is often from another game; on
  ScienceWorld, where top-1 text is usually right, the extra alternatives only add noise
  (0.37 vs 0.26 on the test split).
- **Oracle gold-hit:** the judge's best true outcome is the expert's action in 37% of ALFWorld
  states, and in 63% (ScienceWorld test), 52% (ScienceWorld seen) and 60% (ByteSized32).

### 5.4 Fidelity is not utility

Figure 2 (`fig2_fidelity_vs_utility`) plots exact match against regret, one point per
(regime, arm, budget). `fidelity_utility_corr.csv` gives Spearman's ρ.

- **Fidelity tracks utility here.** Spearman's ρ between exact match and regret is −0.77
  (ALFWorld), −0.77 (ScienceWorld test), −0.86 (ScienceWorld seen) and −0.74 (ByteSized32);
  pooled −0.83 over 72 points (F, D top-1 and D top-k at six budgets in four regimes; native
  mode produces no text, so it has no exact match).
- **The rankings agree everywhere.** In every regime and budget, the arm with the highest
  exact match is also the arm with the lowest regret (`fidelity_vs_utility.csv`: 30 of 30,
  counting ByteSized32 with and without its gold-seen states).
- So, between arms, we do **not** reproduce Lambert et al.'s fidelity–utility mismatch: with
  these arms and short, deterministic next states, the better text model is also the better
  decision model. The exception is *within* D: top-k shows the judge the same top-1 text plus
  alternatives, yet has lower regret than top-1 on ALFWorld and ByteSized32, so how a
  prediction is presented matters even at equal fidelity. G may change this picture (**TBD**).
- The exact match in Figure 2 is measured on the branched states' candidates, so it is lower
  than the held-out fidelity in Table 2 (D top-1 on ALFWorld: 21% vs 44%): outcomes of
  distractor actions are rarer in the training buffer than outcomes of expert actions.

### 5.5 Is the judge trustworthy?

- **Oracle gold-hit: 37–63%** (5.3). Distractors can legitimately be as good as the expert's
  action (another route to the same object, an equally useful observation), so 100% is not
  expected; ALFWorld, where many navigation actions are equally useful, is lowest.
- **Judge-free check** (`judge_check.csv`): where the environment records a reward or score
  change, we count, over candidate pairs with different rewards, how often the judge rates
  the higher-reward action higher (ties count as half; 50% = chance). Agreement is 79% on
  ByteSized32 (45 states, 184 pairs), 74% on ScienceWorld (274 states, 1,312 pairs) and 85%
  on ALFWorld (6 states, 24 pairs). The judge is far from random but not perfect, so regret
  differences below about 0.2 should not be over-read.
- **Sanity check: FAIL** (`docs/SANITY_CHECK.md`), the pre-registered 3-shot check of the
  data pipeline on 200 held-out ALFWorld transitions. Qwen2.5-7B-AWQ with three retrieved
  examples reaches 34.5% exact match [27.5, 41.0], against 28.5% for copying the nearest
  example and 12.5% for always predicting "Nothing happens." Rules 1 (EM in 25–80%) and 3
  (`go to` harder than failures) pass; rule 2 (beat both baselines by 10 points) fails by 4
  points. Inspecting all 131 misses found no pipeline bug: they are guesses about information
  the observation does not contain (receptacle contents). We report the failure as
  pre-registered. It suggests that in-context generation adds little over plain retrieval in
  this setting, which matters for arm G.

### 5.6 RQ3: can cheap features predict the winner?

Labels: the arm with the lowest mean regret at n = 256 per ALFWorld task family and
ScienceWorld task type (`labels_Dtop1.csv`; D in top-1 mode, judged the same way as the text
arms). Predictor and evaluation: Shryes's section (Ridge + gradient-boosted trees on regret
differences, leave-one-(env, family)-out and leave-one-env-out).

With F and D only, the labels are almost all F: 6 of 6 ALFWorld families and 19 of 21
ScienceWorld task types. D wins `chemistry-mix` (regret 0.26 vs 0.74) and
`measure-melting-point-unknown-substance` (0.07 vs 0.30; not in the feature matrix), and
neither difference is significant (the 95% bootstrap CI of F − D includes 0). Three types tie
(`boil`, `melt`, `test-conductivity`); ties go to the higher gold-hit, then to F. Of the 27
labelled groups, only 8 have an F − D difference whose CI excludes 0, all in F's favour. The
binary task "D beats F" therefore has one positive row in the feature matrix and cannot be
learned; the regression on regret differences is still informative. With top-k or native D
labels, D wins 3 and 1 groups. ByteSized32 (reported separately): F wins 25 of 31 games, D 6.

- **TBD** (`results/rq3/`): accuracy of each binary decision (F vs G, D vs F) and of the
  winner, against the majority-class baseline and the permutation test.
- **TBD**: which features drive it (importance), and the Feature-1 ablation: JEPA probe vs
  SFT probe.
- Figure 3 (`fig3_capability_vs_dwins`) shows the five capability-mix rates for rows where
  D wins against the rest. With one D-win row (`chemistry-mix`) out of 32 labelled rows, it
  cannot show which capability profile favours the specialist; we report "no capability
  profile in our data makes D win" rather than read a trend from a single point. ALFWorld
  labels are per family, so its 12 rows carry 6 labels.

---

## 6 Discussion

**Does the predictor work? (TBD)** One paragraph: accuracy against the majority baseline, and
whether it survives leave-one-env-out. With ~30 labelled rows, call it a proof of concept.

**Where JEPA fails, and why.** Most of the reasons are already visible in the data, before
regret:
1. *Decoding is bounded by the buffer.* D returns the nearest training state's text. On new
   ALFWorld games only 31% of gold next states appear word-for-word in the training buffer
   (19.7% for `pick_two` up to 50% for `pick_cool`), so top-1 text often names objects from
   another game. On ScienceWorld the buffer covers 80–90% of gold next states, because
   variations of a task type share most of their text.
2. *Missing coverage is fatal.* Three ByteSized32 games (`dishwasher`, `thermometer`,
   `volume`) have 0% gold reach: no decoded state can be right. They are among D's worst
   games: top-1 regret 2.36 (dishwasher, 25 states), 6.0 (thermometer, 3) and 3.2 (volume,
   5), against D's ByteSized32 average of 1.46 and F's 0.44, 0.0 and 1.0 on the same games.
3. *Short, deterministic text favours fine-tuning.* Next states are short, formulaic
   sentences that a fine-tuned LM copies well (F: 59% exact match on ScienceWorld at n = 32).
   JEPA's usual advantage, ignoring unpredictable detail, has little to work with here.
4. *The embedding alone does not decide better.* Native mode skips decoding altogether, yet
   it is D's worst mode in every regime (Table 3). JEPA's goal-similarity ranking carries
   some decision information (it beats random choice) but much less than judging its decoded
   text. On these tasks the decision-relevant information is in text a judge can read, not in
   distances in JEPA's embedding space.

**Practical reading (with Siddhant's compute analysis).** In our data, the answer to "is the
specialist worth it?" is no: a 1.5B fine-tuned LM (F) beats the JEPA specialist (D) on
decisions in every regime from n = 32 to n = 1024, and D never crosses over. Where retrieval
(G) sits, and whether "use retrieval below n = X, fine-tune above X" holds, is **TBD** until G
is scored. The cost side (GPU-hours per arm) is Siddhant's table; judging all D and F
predictions took 0.72 GPU-hours on the RTX 5070.

**Fidelity vs utility.** Across arms, text fidelity was a good proxy for decision utility
(ρ = −0.83; the best-fidelity arm was the best-utility arm in every case). The decision-based
evaluation still showed two things fidelity could not: the size of the gap in decision terms
(on ScienceWorld test, F's 22-point exact-match lead over D at n = 256 is worth about 0.2
regret points out of 10), and the effect of presentation within an arm (D top-k vs top-1 at
equal fidelity).

---

## 7 Limitations

- **Small and one-sided sample for RQ3.** The predictor is trained on 32 labelled rows from
  two environments (12 ALFWorld rows carrying 6 family labels, 20 ScienceWorld task types;
  ByteSized32 is reported separately). Leave-one-env-out has only two folds. With F and D
  only, D wins one row, so "D beats F" cannot be learned. Treat the predictor as a proof of
  concept, not a tool.
- **Labels from one budget and one seed.** Labels use n = 256 and, unless Shryes's seed runs are added, one training
  seed per arm; with 10–61 states per group, only 8 of 27 F-vs-D labels have a difference
  whose 95% CI excludes 0. Two
  ScienceWorld types (paint-tertiary, power-component) have only 10 states, and two
  (identify-life-stages-1/2) have none, because no unused variations were left.
- **A frozen LLM judge defines utility.** Regret is measured in the judge's ratings
  (Qwen2.5-7B-Instruct, 4-bit AWQ so it fits a 12 GB GPU, a deviation from the planned
  full-precision model). The oracle gold-hit rate and the ByteSized32 score check bound the
  judge's errors (agreement with recorded rewards 74–85%) but do not remove them. The
  pre-registered data sanity check failed one of its three rules (5.5). Utility is one-step
  lookahead over five candidates, not full episodes.
- **JEPA's decoder is a confound.** D's top-1 and top-k modes are judged on retrieved
  training text, so they test JEPA's embedding and its nearest-neighbour decoder together.
  Native mode removes the decoder but replaces the judge with JEPA's own goal similarity.
- **Three different held-out regimes.** ALFWorld tests new games, ScienceWorld new
  variations of known task types, and ByteSized32 only new actions in training states.
  ByteSized32 uses the simulator crawl's JSON states rendered to text, not raw environment
  text, and 118 of 328 states have their gold transition in the training data; we report
  the 210 states without it.
- **Data deviations from the plan.** Training buffers contain expert and random-policy
  transitions only: the planned 20% LLM-driven rows were dropped because every arm had
  already been trained when GPU time became available. Branched states exclude observe-only
  actions (`look`, `inventory`, `task`) as distractors and drop states whose expert action
  is one of them. About 2% of ScienceWorld replays are not deterministic (electrical
  circuits, paint mixing); such states carry a `replay_consistent` flag.
