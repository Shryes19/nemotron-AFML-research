# Results, discussion and limitations (draft, Saanvi)

Status (14 Nov 2026): filled from the GPU scoring runs of 12 Nov (arms F and D) and 13–14 Nov
(arm G), all four regimes, all six budgets (`results/utility/`, `results/figures/`); the RQ3
predictor's first run (`results/rq3/README.md`, Shryes; F vs D labels); the JEPA ablations
(`results/jepa_phase1/README.md`, last two sections) and the GPU-hour table
(`results/compute/cost_table.csv`, Siddhant; G's hours from `results/compute_g_gpu.jsonl`).
Still open, marked **TBD**: the predictor rerun on the three-arm labels (Shryes). G-vs-F
significance below comes from paired bootstrap tests over states (same states, both arms).
Every claim names its figure or table, as the plan asks.

---

## 5 Results

### 5.1 Evaluation protocol in brief

Each arm (G: frozen Qwen2.5-7B-Instruct-AWQ prompted with the 5 most similar training
transitions, retrieved with NV-Embed-v2; F: Qwen2.5-1.5B + LoRA; D: JEPA with nearest-state
decoding) is trained on, or retrieves from, the same n training transitions, and predicts the next state for K = 5 candidate actions at a set of
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
the gap is the decoder and the buffer, not JEPA training. G was not run on these held-out
sets; its exact match on the branched states' candidates (5.4) sits between F and D.

### 5.3 Decision utility

Figure 1 (`fig1_regret_vs_n`) shows mean regret against the training budget n for every
arm, with D in three modes: top-1 decoded text, top-k decoded texts, and native (ranking
candidates in embedding space, no judge). Table 3 (`main_table.md`) gives the n = 256 values
with 95% bootstrap CIs, plus oracle and random choice.

| Regime | Random choice | G (RAG) | F (SFT) | D top-1 | D top-k | D native |
|---|---|---|---|---|---|---|
| ALFWorld, new games | 1.61 [1.39, 1.83] | **0.61** [0.41, 0.81] | 0.62 [0.42, 0.84] | 0.98 [0.73, 1.24] | 0.78 [0.58, 1.03] | 1.24 [1.02, 1.51] |
| ScienceWorld test | 1.47 [1.26, 1.70] | 0.10 [0.04, 0.18] | **0.07** [0.02, 0.13] | 0.26 [0.15, 0.40] | 0.37 [0.25, 0.50] | 0.96 [0.74, 1.19] |
| ScienceWorld seen | 1.38 [1.23, 1.53] | **0.10** [0.04, 0.16] | 0.12 [0.04, 0.24] | 0.33 [0.24, 0.43] | 0.48 [0.36, 0.60] | 1.05 [0.88, 1.22] |
| ByteSized32, gold unseen | 2.12 [1.89, 2.36] | 0.63 [0.47, 0.81] | **0.59** [0.42, 0.76] | 1.46 [1.19, 1.75] | 1.31 [1.04, 1.61] | 1.61 [1.33, 1.91] |

Table 3: mean decision regret at n = 256 (judge scale 0–10, lower is better), 95% bootstrap
CIs. Bold marks the lowest mean; at n = 256 no G-vs-F difference is significant.

- **Every arm carries decision information.** All arms and modes are below random choice in
  every regime (Table 3); the weakest, D native, still removes 23–35% of random choice's
  regret.
- **The two generalists, G and F, are tied.** At n = 256 their difference is within ±0.04
  regret and no paired test is significant. Over all 24 (regime, budget) cells G and F are
  indistinguishable in 19; in the other 5 F is significantly better (ScienceWorld test
  n = 64; ScienceWorld seen n = 32; ByteSized32 n = 128, 512 and 1024) and G never is.
  F's advantage is clearest on ByteSized32 with more data: from n = 512 F keeps improving
  (0.41, then 0.35 at n = 1024) while G levels off (0.61, 0.60).
- **Both generalists beat the specialist.** G's and F's regret is below D top-1's at every
  budget in every regime (Figure 1). At n = 256 the paired G − D top-1 difference is
  significant in all four regimes (−0.37 ALFWorld, −0.15 ScienceWorld test, −0.24 ScienceWorld
  seen, −0.82 ByteSized32), and so is F − D top-1 (−0.36, −0.19, −0.21, −0.87). On ALFWorld
  the two arms' own CIs overlap (F 0.42–0.84, D 0.73–1.24), but the paired test on the same
  states separates them.
- **No crossover to D.** D never has lower regret than G or F at any budget in any regime
  (`crossovers.csv`). The gap narrows with data on ScienceWorld test (D top-1 0.58 → 0.18
  from n = 32 to 1024, F 0.13 → 0.05) but does not close. Both generalists are already good
  at n = 32 (F: 0.05–0.13 on ScienceWorld, 0.72 on ALFWorld, 1.00 on ByteSized32; G: 0.13–0.19,
  0.82 and 0.98).
- **The generalists are close to the ceiling on ScienceWorld.** In 13 of 21 ScienceWorld task
  types F's regret at n = 256 is exactly 0 (G: 7, all of them also 0 for F), so little room is
  left for any specialist there.
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

- **Fidelity tracks utility overall.** Spearman's ρ between exact match and regret is −0.85
  (ALFWorld), −0.63 (ScienceWorld test), −0.61 (ScienceWorld seen) and −0.77 (ByteSized32);
  pooled −0.79 over 96 points (G, F, D top-1 and D top-k at six budgets in four regimes;
  native mode produces no text, so it has no exact match). With F and D only, pooled ρ was
  −0.83.
- **But not between the two generalists.** At n = 256, F's exact match on the candidates is
  9–21 points above G's (ALFWorld 49.5% vs 40.4%, ScienceWorld test 66.5% vs 45.2%, seen
  70.8% vs 49.9%, ByteSized32 47.3% vs 38.3%; `main_table.md`), yet their regret is tied
  (Table 3). The best-fidelity arm is the best-utility arm in 22 of 30 (regime, budget)
  cases (`fidelity_vs_utility.csv`, counting ByteSized32 with and without its gold-seen
  states); all 8 disagreements are between G and F, 5 of them on ALFWorld. G reaches the same
  decisions with text that is less often word-for-word right, which suggests its errors are
  mostly in details the judge does not need to rank the candidates (see the judge caveat in
  5.5).
- So we reproduce a mild form of Lambert et al.'s fidelity–utility mismatch: fidelity
  separates good from bad models (both generalists vs D), but not two good ones. None of the
  8 disagreements is a significant regret difference, so the mismatch is "fidelity
  overstates F's lead", not "fidelity picks the wrong arm". A second case is *within* D:
  top-k shows the judge the same top-1 text plus alternatives, yet has lower regret than
  top-1 on ALFWorld and ByteSized32, so how a prediction is presented matters even at equal
  fidelity.
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
  pre-registered. It suggests that in-context generation adds little *exact match* over
  plain retrieval in this setting. Arm G is the same kind of model (with five retrieved
  examples instead of three) and its exact match is indeed well below F's (5.4), yet its
  decisions are as good as F's (5.3): exact match understates what retrieval-based
  prediction is worth for decisions.
- **Self-preference caveat for G.** G's predictions are written by the same model that
  judges them (Qwen2.5-7B-Instruct-AWQ). An LLM judge may rate text in its own style more
  favourably. Regret is computed from the judge's ratings of the *true* next states, but
  G's choice comes from its ratings of G's own text, so such a preference could make G's
  choices look better than its predictions deserve. We cannot rule this out with one judge.

### 5.6 RQ3: can cheap features predict the winner?

Labels: the arm with the lowest mean regret at n = 256 per ALFWorld task family and
ScienceWorld task type (`labels_Dtop1.csv`; D in top-1 mode, judged the same way as the text
arms). Predictor and evaluation: Shryes's section (Ridge + gradient-boosted trees on regret
differences, leave-one-(env, family)-out and leave-one-env-out).

**Three-arm labels.** Ties in mean regret are common because many groups have regret 0 for
both generalists; the label then goes to the higher gold-hit rate, then to the order G, F, D.
We therefore count strict wins and ties separately (`tie` column in the label files). Of the
27 groups (6 ALFWorld families, 21 ScienceWorld task types):

| D mode for labels | F strictly | G strictly | D strictly | Tied (G+F / all three) |
|---|---|---|---|---|
| top-1 (main) | 12 | 6 | 1 | 8 (6 / 2) |
| top-k | 12 | 6 | 0 | 9 (8 G+F, 1 G+D) |
| native | 12 | 6 | 1 | 8 (8 / 0) |

Table 5: label counts at n = 256 (`labels_D{top1,topk,native}.csv`).

- With top-1 labels, G's six strict wins are three ALFWorld families
  (`look_at_obj_in_light`, `pick_heat`, `pick_two`) and `boil`, `chemistry-mix` and
  `find-living-thing`; F's twelve are the other three ALFWorld families and nine
  ScienceWorld types.
- **Few labels are significant.** Of the 27 groups, the paired G − F difference is
  significant in only one (`change-the-state-of-matter-of`, F better by 0.20), and is exactly
  0 in 7. F − D is significant in 8 groups and G − D in 6, all in the generalist's favour.
- **D's last wins disappear.** With F and D only, D won `chemistry-mix` and
  `measure-melting-point-unknown-substance`. With G added, G takes `chemistry-mix` (0.07 vs
  D 0.26 and F 0.74). D keeps only `measure-melting-point-unknown-substance` (0.07 vs G 0.22,
  F 0.30; not significant, and not a feature-matrix type), and with native labels
  `look_at_obj_in_light`. So with top-1 or top-k labels **no feature-matrix row is a D win**.
- **ByteSized32** (31 games, reported separately, top-1 labels): F strictly 13, G 8, D 3,
  tied 7. G − F is significant in one game (`metal-detector`, F better); G − D in 12 (10 in
  G's favour, 2 in D's).

The predictor's first run (`results/rq3/README.md`) used the F-vs-D labels; it is evaluated
on the 32 labelled rows of the feature matrix, which carry 26 independent labels (ALFWorld is
labelled per family). Its rerun on the three-arm labels is **TBD** (Shryes); with no D win
left, the yes/no "is the specialist worth it" question becomes a constant answer, and the
open question becomes G vs F.

- **Yes/no "D beats F": not testable.** F wins 31 of 32 rows, so always answering "F" already
  scores 97%. Every model scores exactly that (balanced accuracy 0.50; permutation p = 0.60).
- **Ranking the size of D's gap to F** (regression on regret_F − regret_D, out-of-fold
  Spearman ρ, leave-one-(env, family)-out). All features together: ρ = +0.10; features 2–4
  without a probe: +0.07; with the SFT probe instead of the JEPA probe: +0.10.
- **Feature-1 ablation.** The JEPA probe alone ranks the gap out of sample (ρ = +0.47,
  permutation p = 0.01); the SFT probe alone does not (ρ = +0.03, p = 0.29). This is the
  direction the plan hoped for, but we treat it as suggestive: it is one of several tests
  (2 probes × 3 D modes × 2 CV schemes), it disappears when the other features are added,
  and its sign flips with the D mode used for the labels (single-feature correlation +0.34
  with top-1 labels, −0.46 with native labels).
- **Transfer across environments fails.** Trained on ALFWorld and tested on ScienceWorld, or
  the reverse, every feature set gives ρ between −0.13 and +0.11.
- **ByteSized32** (separate analysis, 29 games, D wins 4): neither probe relates to the gap
  (|ρ| ≤ 0.14); only the object-state rate does (ρ = +0.44, p = 0.02).
- **Cost of Feature 1.** The SFT probe costs 0.7–3.1 GPU-hours per environment against
  0.02 h for the JEPA probe, more than F's whole pipeline (Table 4), so it fails the "cheap"
  requirement before any result.
- Figure 3 (`fig3_capability_vs_dwins`) shows the five capability-mix rates for rows where
  D wins against the rest. With the three-arm top-1 labels there is no D-win row among the 32
  labelled rows (with F and D only there was one, `chemistry-mix`), so the figure cannot show
  which capability profile favours the specialist; we report "no capability profile in our
  data makes D win". ALFWorld labels are per family, so its 12 rows carry 6 labels.
- **Recall@5 of G's retriever** (feature-matrix column, `g_recall_at5.csv`): for each
  branched state, whether one of the five retrieved examples has the gold action's template
  and object. Weighted by states it rises with n in every regime, from 0.23–0.36 at n = 32 to
  0.56–0.83 at n = 1024 (ALFWorld 0.36 → 0.82, ScienceWorld test 0.31 → 0.74, seen
  0.34 → 0.83, ByteSized32 0.23 → 0.56). It joins the feature matrix at n = 32.

---

## 6 Discussion

**Does the predictor work? Not yet, and for the specialist it is not needed.** With F and D
only, F had lower regret on 31 of 32 rows, so the yes/no decision was trivially "F" and no
feature could improve on that (5.6). With G added, no feature-matrix row is a D win at all.
The one positive signal, the JEPA probe ranking the size of D's gap to F within an
environment (ρ = 0.47), is fragile: it vanishes with the other features, flips sign with the
D mode, and does not transfer across environments. A likely reason is a floor effect: F's
regret is 0 on 13 of 21 ScienceWorld task types, so "the gap" mostly measures how badly D
does, which depends on how D's predictions are decoded. G adds the decision we hoped would
have both outcomes present, G vs F, but it is mostly noise: 8 of 27 groups are exact ties and
only one G − F difference is significant (5.6). **TBD**: the predictor rerun on the three-arm
labels; we expect it to confirm that no cheap feature separates two arms whose regret is
statistically the same. A real test needs arms that differ and more than two environments.

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
5. *The frozen encoder is the main limit, not the loss or the decoder.* Two ablations at
   fixed data (`results/jepa_phase1/README.md`) rule out the obvious fixes. Training with
   VL-JEPA's contrastive InfoNCE loss instead of MSE ties or is slightly worse at every
   budget (e.g. ALFWorld n = 256: 43.8% exact match both; ScienceWorld n = 1024: 60.0% vs
   60.2%). Replacing nearest-sentence decoding with a learned text decoder (VL-JEPA-style:
   Qwen2.5-1.5B + LoRA reading only JEPA's predicted vector) does not help either (n = 256:
   43.3% vs 43.8% on ALFWorld, 44.0% vs 47.3% on ScienceWorld test). Most tellingly, the same
   decoder given the *true* next-state vector reaches only 33–55% exact match, below F in
   every environment (52.9% vs 56.2% on ALFWorld, 50.6% vs 69.0% on ScienceWorld test).
   The frozen general-purpose sentence embedding (NV-Embed) drops exact details such as
   object numbers and quantities, which F keeps by working on the text. So D's gap to F
   splits into what the frozen embedding loses (the larger part outside ALFWorld) and
   prediction error (3–10 points). Training the encoder itself, as I-JEPA and V-JEPA do,
   is the natural next step.

**Practical reading (with Siddhant's compute analysis).** In our data, the answer to "is the
specialist worth it?" is no: both generalists, a 1.5B fine-tuned LM (F) and a frozen 7B LM
with retrieval (G), beat the JEPA specialist (D) on decisions in every regime from n = 32 to
n = 1024, and D never crosses over. Between the two generalists there is no "retrieval below
n = X, fine-tune above X" rule in our data: they are tied at almost every budget, and where
they differ significantly F is better, at small n (ScienceWorld, n = 32–64) as well as large
n (ByteSized32, n ≥ 512). On ByteSized32 G stops improving after n = 512 while F keeps
improving, the one place where the two scale differently.

| GPU-hours (RTX 5070) | ALFWorld | ScienceWorld | ByteSized32 |
|---|---|---|---|
| D (JEPA): train + predict + judging | 0.36 | 1.10 | 0.39 |
| F (SFT): train + predict + judging | 0.60 | 1.52 | 0.70 |
| G (RAG): NV-Embed retrieval + generation + judging | 0.97 | 3.03 | 0.69 |
| NV-Embed encoding for D | 0.11 | 0.42 | — |
| SFT probe (Feature-1 ablation, not part of F) | 0.69 | 3.13 | 2.46 |

Table 4: GPU-hours for all six budgets (`results/compute/cost_table.csv`; G from
`results/compute_g_gpu.jsonl`: retrieval 0.21 / 0.71 / 0.04 h, generation 0.74 / 2.25 /
0.61 h, judging 0.02 / 0.07 / 0.04 h). ScienceWorld sums the test and seen sets. Pilots, the
InfoNCE and decoder ablations and fidelity-only runs are excluded and listed there.

Cost does not rescue the specialist. D is cheaper than F (56–72% of F's GPU-hours, or
78–100% on ALFWorld and ScienceWorld if NV-Embed encoding is charged to D alone), but the
saving is under half an hour per environment and comes with higher regret everywhere.
G needs no training but is the most expensive arm to run here: every prediction is a 7B
generation over a long prompt with five examples, so it costs 1.6–2.0× F's GPU-hours on
ALFWorld and ScienceWorld (and about the same as F on ByteSized32, where texts are short).
Its cost grows with the number of predictions, F's with the training data, so for a world
model queried many times F is the cheaper of two equally good generalists. G's retrieval
stage (0.04–0.71 h) encodes its own buffer and queries with NV-Embed; reusing D's buffer
encodings would save only part of it.

**Fidelity vs utility.** Across arms, text fidelity was a good proxy for decision utility
overall (pooled ρ = −0.79): it separates the two generalists from the specialist. It was not a
good guide between the two generalists: F's exact match is 9–21 points above G's at n = 256,
yet their decisions are equally good, and the best-fidelity arm was the best-utility arm in
only 22 of 30 cases. Ranking world models by text fidelity alone would have declared F the
clear winner; the decision-based evaluation shows a tie. It also showed the size of gaps in
decision terms (on ScienceWorld test, F's 22-point exact-match lead over D at n = 256 is
worth about 0.2 regret points out of 10) and the effect of presentation within an arm (D
top-k vs top-1 at equal fidelity).

---

## 7 Limitations

- **Small and one-sided sample for RQ3.** The predictor is trained on 32 labelled rows from
  two environments (12 ALFWorld rows carrying 6 family labels, 20 ScienceWorld task types;
  ByteSized32 is reported separately). Leave-one-env-out has only two folds. With F and D
  only, D wins one row, and with G added none, so "the specialist wins" cannot be learned;
  G vs F is mostly ties and non-significant differences. Treat the predictor as a proof of
  concept, not a tool.
- **Labels from one budget and one seed.** Labels use n = 256 and a single training run per
  (environment, budget) for each arm (seed 42; F trains on exactly D's rows). Seed variance
  is therefore not in the CIs, which only cover state sampling. With 10–61 states per group,
  only 8 of 27 F-vs-D, 6 of 27 G-vs-D and 1 of 27 G-vs-F differences have a 95% CI that
  excludes 0, and 8 groups are exact regret ties labelled by gold-hit. Two ScienceWorld
  types (paint-tertiary, power-component) have only 10 states, and two
  (identify-life-stages-1/2) have none, because no unused variations were left.
- **Several tests on few rows.** The predictor results come from one of several analyses
  (two probes, three D modes, two cross-validation schemes) on 26 independent labels, with
  no correction for multiple comparisons; the one significant result (JEPA probe alone,
  p = 0.01) should be read as suggestive.
- **A frozen LLM judge defines utility.** Regret is measured in the judge's ratings
  (Qwen2.5-7B-Instruct, 4-bit AWQ so it fits a 12 GB GPU, a deviation from the planned
  full-precision model). The oracle gold-hit rate and the ByteSized32 score check bound the
  judge's errors (agreement with recorded rewards 74–85%) but do not remove them. The
  pre-registered data sanity check failed one of its three rules (5.5). Utility is one-step
  lookahead over five candidates, not full episodes.
- **G shares its model with the judge.** G's generator is the judge's model, so a judge
  preference for its own wording could favour G (5.5); a second judge model would test this.
  G also deviates from the plan: the generator is 4-bit AWQ (the full-precision 7B does not
  fit the 12 GB GPU), retrieval uses NV-Embed-v2, and example texts in its prompt are cut at
  1,500 characters (the current state at 4,000) to fit vLLM's 8,192-token window.
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
