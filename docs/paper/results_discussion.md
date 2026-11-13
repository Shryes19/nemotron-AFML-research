# Results, discussion and limitations (draft, Saanvi)

Status (11 Nov 2026): everything below that does not depend on the GPU scoring run is written.
Slots marked **TBD** are filled from `results/figures/` once `scripts/run_utility.sh` and
`scripts/make_figures.py` have run, and from `results/rq3/` (Shryes) for the predictor.
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

- **TBD**: at n = 256, which arm has the lowest regret in each regime, and do the CIs separate?
- **TBD**: are all arms clearly below random choice? (If an arm is not, it carries no
  decision information in that regime.)
- **TBD**: crossovers (`crossovers.csv`): the smallest n at which F beats G, and at which D
  beats F, per regime; or "never".
- **TBD**: D's three modes: does native (no text) beat top-1? This is JEPA's own claim:
  decisions in latent space without decoding.
- Oracle gold-hit **TBD** per regime: the judge agrees with the expert's action in that
  share of states.

### 5.4 Fidelity is not utility

Figure 2 (`fig2_fidelity_vs_utility`) plots exact match against regret, one point per
(regime, arm, budget). `fidelity_utility_corr.csv` gives Spearman's ρ.

- **TBD**: ρ within each regime and pooled. A strong negative ρ (more fidelity, less regret)
  means fidelity is a usable proxy; a weak one reproduces Lambert et al.'s mismatch.
- **TBD**: `fidelity_vs_utility.csv` lists, per regime and budget, the best arm by regret
  and the best arm by exact match, and whether they agree. Count the disagreements.
- Point to watch: F leads on fidelity by 12–29 points at n = 256 (Table 2). If D's regret is
  close to F's anyway, the fidelity gap mostly measures details that do not change the
  decision.

### 5.5 Is the judge trustworthy?

- **Oracle gold-hit (TBD)**: share of states where the best true outcome, as rated by the
  judge, is the expert's action. Distractors can legitimately be as good as the expert's
  action (e.g. another route to the same object), so this is not expected to be 100%.
- **Judge-free check (TBD)**, `judge_check.csv`: on ByteSized32, the environment records a
  score. In the 45 states where the candidates' scores differ, we count how often the judge
  rates the higher-scoring action higher (ties count as half; 50% = chance).
- **Sanity check (TBD)**, `results/sanity_alfworld.json`: the pre-registered 3-shot check of
  the data pipeline (`docs/SANITY_CHECK.md`). Report PASS/FAIL against the rule fixed on
  30 Oct, before any run.

### 5.6 RQ3: can cheap features predict the winner?

Labels: the arm with the lowest mean regret at n = 256 per ALFWorld task family and
ScienceWorld task type (**TBD** rows; ByteSized32 excluded, see 5.1). Predictor and
evaluation: Shryes's section (Ridge + gradient-boosted trees on regret differences,
leave-one-(env, family)-out and leave-one-env-out).

- **TBD** (`results/rq3/`): accuracy of each binary decision (F vs G, D vs F) and of the
  winner, against the majority-class baseline and the permutation test.
- **TBD**: which features drive it (importance), and the Feature-1 ablation: JEPA probe vs
  SFT probe.
- Figure 3 (`fig3_capability_vs_dwins`) shows the five capability-mix rates for rows where
  D wins against the rest. **TBD**: medians from `capability_dwins.csv`. ALFWorld labels
  are per family, so its 12 rows carry 6 labels.

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
   `volume`) have 0% gold reach: no decoded state can be right. **TBD**: are these the games
   with D's worst regret?
3. *Short, deterministic text favours fine-tuning.* Next states are short, formulaic
   sentences that a fine-tuned LM copies well (F: 59% exact match on ScienceWorld at n = 32).
   JEPA's usual advantage, ignoring unpredictable detail, has little to work with here.
4. **TBD**: does native mode avoid problems 1–2? If native ≈ or < top-1 regret, the
   embedding carries decision-relevant information that the decoder loses.

**Practical reading (with Siddhant's compute analysis).** **TBD**: "use retrieval below
n = X; switch to fine-tuning above X; add JEPA only for capability mix Y", from Figure 1's
crossovers and the GPU-hour table. If no crossover to D exists in any regime, say so plainly.

**Fidelity vs utility.** **TBD** from 5.4: whether the paper's message is "fidelity is a
good enough proxy" or "measure decisions, not text".

---

## 7 Limitations

- **Small sample for RQ3.** The predictor is trained on about 30 labelled rows from two
  environments (6 ALFWorld families, up to 22 ScienceWorld task types; ByteSized32 is
  reported separately). Leave-one-env-out has only two folds. Treat the predictor as a proof
  of concept, not a tool.
- **Labels from one budget and one seed.** Labels use n = 256 and, unless Shryes's seed runs are added, one training
  seed per arm; with 10–60 states per group, some winners lie within each other's CIs. Two
  ScienceWorld types (paint-tertiary, power-component) have only 10 states, and two
  (identify-life-stages-1/2) have none, because no unused variations were left.
- **A frozen LLM judge defines utility.** Regret is measured in the judge's ratings
  (Qwen2.5-7B-Instruct, 4-bit AWQ so it fits a 12 GB GPU, a deviation from the planned
  full-precision model). The oracle gold-hit rate and the ByteSized32 score check bound the
  judge's errors but do not remove them. Utility is one-step lookahead over five candidates,
  not full episodes.
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
