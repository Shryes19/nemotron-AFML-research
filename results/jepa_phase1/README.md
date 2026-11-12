# JEPA arm — Phase 1 results (ALFWorld; probe: all three environments)

Raw outputs from the runs of 31 Oct – 1 Nov 2026 on Siddhant's RTX 5070
(NV-Embed-v2, NF4). Data: Saanvi's `alfworld_games.jsonl` — 12 games,
1,440 transitions, expert + random policy. Model checkpoints (`budget_*.pt`,
~250 MB each) are not committed; rerun the command to regenerate them.

Read the runs in order — each fixes something the one before showed.

## Training runs (`jepa/`)

Each folder has `results.json` (settings, final metrics and per-epoch history
for every budget) and `loss_curves.png`.

| Folder | Test set | Input | Training | n=1024 decode | Lookup (b_action) |
|---|---|---|---|---|---|
| `alfworld_games` | same games (random 10%) | joint text | 20 epochs | 51.4% | 63.9% |
| `alfworld_games_steps2000` | same games | joint text | 2,000 steps | 77.8% | 63.9% |
| `alfworld_games_steps2000_sepact` | same games | separate action | 2,000 steps | **80.6%** | 63.9% |
| `alfworld_games_steps2000_unseen2` | unseen games, seed 42 | joint text | 2,000 steps | 40.2% | 47.0% |
| `alfworld_games_steps2000_unseen2_sepact` | unseen games, seed 42 | separate action | 2,000 steps | 45.3% | 47.0% |
| `unseen_sepact_seed43` | unseen games, seed 43 | separate action | 2,000 steps | 26.8% | 26.8% |
| `unseen_sepact_seed44` | unseen games, seed 44 | separate action | 2,000 steps | 43.5% | 43.5% |

None of these runs used the admissible-action input (added later); the
`alfworld_games` folder also predates the baseline columns in its JSON.

## Diagnostics (`diagnose/`)

Unseen games, 3 held-out game pairs (seeds 42/43/44), separate action input,
2,000 steps, one validation game held out (so 9 training games).

| Folder | Admissible-action input | Training target | Loss |
|---|---|---|---|
| `alfworld_games` | on | projection + EMA (original) | MSE |
| `alfworld_games_noadm` | off | projection + EMA | MSE |
| `alfworld_games_frozen` | on | **frozen NV-Embed vectors** (current default) | MSE |
| `alfworld_games_frozen_l1` | on | frozen NV-Embed vectors | L1 |

All four use the separate action input; all but `alfworld_games_noadm`
include the no-learning `text_overlap` control in the action-choice test.

Headline numbers (n = 1024, from `alfworld_games`):

| | JEPA | Lookup | Ceiling (reach) |
|---|---|---|---|
| Decode, all actions | 40.7% | 39.1% | 48.9% |
| Decode, failed actions | 100% | 89.4% | 100% |
| Open-set, predictable actions | 57.5% | — | 44.9% |

| Choosing the action (vs 4 others) | JEPA | No-learning control | Lookup |
|---|---|---|---|
| Predictable | 98.7% | 67.1% | 68.3% |
| Failed action | 55.7% | 3.5% | 36.3% |
| Reveals hidden info | 99.2% | 91.3% | 81.4% |

On hidden-information actions the no-learning control already scores 91%, so
that row mostly measures word overlap, not learned dynamics.

## Ablation: how the predictor is trained

Does the design follow published JEPA world models? We compared our original
target (a learned projection + EMA copy on top of the frozen encoder) with
predicting the frozen encoder's vectors directly (DINO-WM style) and with
V-JEPA 2-AC's L1 loss. Unseen games, mean of 3 pairs:

| | Projection + EMA, MSE | Frozen, MSE | Frozen, L1 |
|---|---|---|---|
| Decode, n = 32 | 9.2% | **11.5%** | 8.1% |
| Decode, n = 1024 | **40.7%** | 40.4% | 39.2% |
| Failed actions decoded, n = 64 | 74.5% | **89.1%** | 55.6% |
| Choose action (predictable), n = 64 | 81.6% | **86.9%** | 79.2% |
| Choose action (predictable), n = 1024 | **98.7%** | 98.5% | 96.2% |
| Open-set (predictable), n = 1024 | **57.5%** | 54.8% | 54.0% |

Frozen + MSE is as good overall and better at small budgets, cannot collapse,
and matches DINO-WM, so it became the default (`jepa.target_mode: frozen`).
L1 was worse. The main findings (even with the lookup on exact text in unseen
games, clearly better on predictable actions, capped by the buffer) hold in
all three, so they come from ALFWorld and the decode step, not this choice.
The training runs in `jepa/` used the original projection + EMA design.

## Probe fidelity — Feature 1 (`probe_scores.csv`, `probe/`)

`probe_scores.csv` is the Feature 1 file for the RQ3 matrix
(`scripts/build_feature_matrix.py --probe`): one row per (env, game), columns
`env, game, probe_fidelity, ...`. **Definition (agreed 1 Nov 2026): 50
optimiser steps on n = 32, 5 random draws per game, current arm model**
(separate action + admissible input, frozen targets, MSE). Game ids match the
feature matrix; joined there it fills Feature 1 for **all 64 games**.

| Env | Games | Score range | Separation ratio |
|---|---|---|---|
| ALFWorld | 12 | 0.139–0.228 | 1.63 |
| ByteSized32 | 30 (+ `clean-energy`) | 0.154–0.303 | 2.02 |
| ScienceWorld | 22 task types | 0.180–0.316 | 1.55 |

`clean-energy` has only 24 transitions, so it was probed on 20 (not 32) and is
not in the feature matrix (which keeps 30 ByteSized32 games). ByteSized32 games
separate more than ALFWorld games, as Feature 2 found. Compare scores within an
environment: across environments the text styles differ, which also shifts the
encoder's distances. ByteSized32 has no [ADMISSIBLE] list, so the
admissible-action input is zero there. ScienceWorld's [ADMISSIBLE] section
lists action templates ("open OBJ"), not real actions, so it is treated as no
list and the input is zero there too (fix 9e41217, before the run).

ScienceWorld (`sciworld_games.jsonl`, 2,300 transitions, up to 80 steps,
game = task type): easiest are test-conductivity (0.180) and the
identify-life-stages tasks (0.196, 0.219); hardest are chemistry-mix (0.316),
boil (0.297) and find-living-thing (0.291). Separation (1.55) is about the
same as ALFWorld's. All three environments separate games well above 1,
so Feature 1 carries signal within each environment.

`probe_fidelity` = mean held-out cosine distance (lower = the game is easier to
predict). `probe_scores_1epoch.csv` and `probe_scores_steps200.csv` are the
alternatives compared below, kept for the record.

### Which probe schedule? (evidence for the team decision)

Same 12 games, same model, 5 draws each:

| Schedule | Score range | Separation ratio | Rank agreement with 200 steps |
|---|---|---|---|
| 1 epoch (= 1 step at n = 32; `probe_scores_1epoch.csv`) | 0.770–0.791 | 0.80 | 0.34 |
| **50 steps** (`probe_scores.csv`) | 0.139–0.228 | **1.63** | 0.93 |
| 200 steps (`probe_scores_steps200.csv`) | 0.095–0.171 | 1.29 | — |

Separation ratio = between-game sd ÷ mean within-game sd (as in the capability
report; its Feature 2 scored 1.0–2.2 within ALFWorld). Rank agreement =
Spearman correlation of the game ranking.

With 50 steps the probe separates games (1.63) and its ranking is stable
(0.93 with 200 steps); the 1-epoch ranking is close to unrelated (0.34). The
ordering is also sensible: pick-and-place and look-at tasks are easiest
(≈0.11 at 200 steps), heat and cool tasks hardest (≈0.17). 200 steps
separates less because the 32-example model starts to overfit and the draws
spread out. **Decision: Feature 1 = 50 steps** (now `training.probe_steps` in
the config and the default in `probe_fidelity.py`).

`probe/alfworld_games/` holds the earlier run (old joint text input, no
admissible feature, 3 draws), kept for reference.

## Reproduce

On Siddhant's machine in `.venv-enc` (defaults: separate action, admissible
input, frozen targets, MSE, 2,000 steps):

```bash
python scripts/train_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
python scripts/train_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder --holdout-games 2 --seed 43
python scripts/diagnose_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
python scripts/probe_fidelity.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
```

Older runs: add `--target ema` (original projection + EMA), `--joint` (single
text input), `--no-admissible`, or `--use-epochs --epochs 20`.

## Decision utility: reach on the branched states (6 Nov 2026)

`scripts/check_reach.py`: how often a branched state's true next state appears
word-for-word in the JEPA training buffer (whole buffer, so this is the n=1024+
upper bound; smaller budgets reach less). It caps top-1 *exact match*, not
decision utility.

| Env (train buffer → branched states) | States | Gold reach | All-candidate reach |
|---|---|---|---|
| ALFWorld (`alfworld_games` → 60 new games) | 193 | 31.1% | 39.8% |
| ScienceWorld (`sciworld_games` → 12 test task types) | 182 | 80.2% | 73.5% |
| ScienceWorld seen (`sciworld_games` → held-out variations of 20 seen types, 7 Nov) | 323 | 90.1% | 74.5% |
| ByteSized32 (`bytesized32_games` → known states, new actions, 31 games, 7 Nov) | 328 | 57.9% | 67.9% |

ALFWorld by family: gold reach 19.7% (pick_two) to 50.0% (pick_cool).
ScienceWorld: 77–100% for task types in the training buffer;
`measure-melting-point-unknown-substance` (not in the training buffer) 33.3%.
So on ALFWorld, JEPA's top-1 text is often from another game (wrong objects);
report `--mode topk` and `--mode native` next to top-1.

ScienceWorld seen: 73–100% gold reach per type (lowest `test-conductivity-of-unknown-substances`
73.3%, `chemistry-mix` 52.9%). ByteSized32: three games have **0% gold reach**
(`dishwasher` 25 states, `thermometer` 3, `volume` 5), so JEPA's top-1 can never be exactly
right there; expect weak D scores on those games (a buffer limit, not a model bug).

## Retrain for decision utility: held-out test sets (6 Nov 2026)

Checkpoints now record their training rows (needed by
`jepa_predict_candidates.py`). Defaults (separate action + admissible input,
frozen targets, MSE, 2,000 steps); test = Saanvi's held-out buffers.
Decode = exact match by nearest training state; reach = its ceiling;
b_action = no-learning lookup on the action text.

**ALFWorld** (`alfworld_games` → `alfworld_heldout_buffer`, 240 rows from new games)

| n | cos dist | decode | reach | b_action |
|---|---|---|---|---|
| 32 | 0.219 | 35.0% | 37.5% | 37.1% |
| 64 | 0.193 | 42.1% | 43.3% | 41.7% |
| 128 | 0.148 | 42.9% | 45.8% | 44.2% |
| 256 | 0.133 | 43.8% | 47.5% | 44.6% |
| 512 | 0.101 | 49.6% | 55.0% | 50.8% |
| 1024 | 0.090 | 52.5% | 58.8% | 52.9% |

**ScienceWorld** (`sciworld_games` → `sciworld_heldout_buffer`, 480 rows, 12 test task types)

| n | cos dist | decode | reach | b_action |
|---|---|---|---|---|
| 32 | 0.284 | 24.2% | 29.4% | 14.6% |
| 64 | 0.225 | 38.1% | 44.4% | 32.3% |
| 128 | 0.180 | 43.3% | 50.2% | 37.5% |
| 256 | 0.152 | 47.3% | 53.5% | 41.9% |
| 512 | 0.109 | 53.1% | 59.4% | 46.7% |
| 1024 | 0.084 | 60.2% | 65.2% | 51.2% |

On new ALFWorld games JEPA ties the lookup (within ±1.5 points at every
budget) and gets 89–97% of the reachable answers: the buffer, not the
predictor, is the limit there. On ScienceWorld it beats the lookup at every
budget (+6 to +10 points) and improves steadily with data. Both runs: error
falls and decode rises at every budget step.

**ScienceWorld seen** (`sciworld_games` → `sciworld_seen_heldout_buffer`, 1,600 rows from
held-out variations of 20 seen task types; 7 Nov 2026). Same training rows as the
ScienceWorld run above (same seed, whole buffer as the pool), so its checkpoints match the
ones used for `sciworld_seen_b{n}.jsonl`.

| n | cos dist | decode | reach | b_action |
|---|---|---|---|---|
| 32 | 0.265 | 27.8% | 32.0% | 16.0% |
| 64 | 0.204 | 41.4% | 45.8% | 32.8% |
| 128 | 0.161 | 47.2% | 53.3% | 40.3% |
| 256 | 0.124 | 52.4% | 58.2% | 47.0% |
| 512 | 0.095 | 59.8% | 65.4% | 52.6% |
| 1024 | 0.069 | 65.2% | 70.6% | 57.3% |

JEPA beats the lookup at every budget (+5 to +12 points) and gets 87–92% of the reachable
answers, like on the 12 test task types.

**ByteSized32** (`bytesized32_games`, 2,954 rows; random 10% held out = 295 rows from the
same games, since there is no test buffer; 7 Nov 2026). No [ADMISSIBLE] list, so the
admissible input is zero.

| n | cos dist | decode | reach | b_action |
|---|---|---|---|---|
| 32 | 0.442 | 6.1% | 18.6% | 7.5% |
| 64 | 0.380 | 6.8% | 21.7% | 7.1% |
| 128 | 0.292 | 19.3% | 27.5% | 13.2% |
| 256 | 0.228 | 26.8% | 31.9% | 16.9% |
| 512 | 0.170 | 36.6% | 41.7% | 23.1% |
| 1024 | 0.123 | 48.8% | 54.6% | 33.6% |

JEPA roughly ties the lookup at n = 32–64 and beats it from n = 128 on, by a growing
margin (+6 → +15 points); at n ≥ 256 it gets 84–89% of the reachable answers. Target
spread stays at 0.70 (no collapse).

**Decision-utility prediction files** (on the GPU box, not in git), one per budget,
from the checkpoints above with `scripts/jepa_predict_candidates.py --k 5`:

| File | States per file | Checkpoints |
|---|---|---|
| `outputs/jepa_candidates/alfworld_b{n}.jsonl` | 193 | `alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen` |
| `outputs/jepa_candidates/sciworld_b{n}.jsonl` | 182 | `sciworld_games_steps2000_test-sciworld_heldout_buffer_sepact_adm_frozen` |
| `outputs/jepa_candidates/sciworld_seen_b{n}.jsonl` | 323 | same ScienceWorld checkpoints (no retrain) |
| `outputs/jepa_candidates/bytesized32_b{n}.jsonl` | 328 | `bytesized32_games_steps2000_sepact_adm_frozen` |

Leak check for ScienceWorld seen (7 Nov): none of the 210 reserved variations
(`splits/sciworld_heldout_seen.txt`), and none of the 126 used by the 323 states,
appear in `sciworld_games.jsonl`.

## Ablation: VL-JEPA-style InfoNCE loss (12–13 Nov 2026)

`--loss infonce` is a contrastive loss between predicted and true next-state embeddings,
with in-batch negatives and identical next states counted as positives (VL-JEPA trains
with InfoNCE). It matches the nearest-state decode used at test time. Everything else is
the same as the default runs (frozen targets, separate action, admissible feature,
2,000 steps, same training rows). Exact-match decode on the test buffers:

| n | ALFWorld MSE | ALFWorld InfoNCE | ScienceWorld MSE | ScienceWorld InfoNCE |
|---|---|---|---|---|
| 32 | **35.0%** | 33.3% | **24.2%** | 23.1% |
| 64 | **42.1%** | 36.2% | **38.1%** | 37.9% |
| 128 | **42.9%** | 41.7% | **43.3%** | 42.5% |
| 256 | 43.8% | 43.8% | **47.3%** | 47.1% |
| 512 | **49.6%** | 49.2% | **53.1%** | 51.0% |
| 1024 | **52.5%** | 51.7% | **60.2%** | 60.0% |

InfoNCE ties or is slightly worse everywhere (cos_dist is higher by design, because it
ranks rather than regresses). The decode already reaches about 90% of its ceiling with
MSE, so the loss is not the limit. The limit is answering with training states only. MSE
stays the default, and the decision-utility files were not regenerated. A text decoder
(VL-JEPA's y-decoder) or a token-level transformer predictor would be the next step.
Both are future work.

Checkpoints: `outputs/jepa/{alfworld,sciworld}_games_steps2000_test-*_heldout_buffer_sepact_adm_frozen_infonce/`

## Experiment: JEPA + text decoder (VL-JEPA-style y-decoder), n = 256 (13 Nov 2026)

VL-JEPA turns its predicted embedding into text with a light decoder, instead of picking
the nearest training sentence. We tested the same idea with
`scripts/export_jepa_vectors.py` + `scripts/train_jepa_decoder.py`:

- the predicted next-state vector is mapped to 8 soft-prompt tokens
- Qwen2.5-1.5B-Instruct + LoRA writes the next observation from those tokens alone (no
  observation or action text, so the answer comes entirely from JEPA)
- the decoder is trained on the true target vectors of the checkpoint's own 256 training
  rows, for 300 steps (fixed in advance)

Exact match / word F1 on the held-out sets, n = 256:

| Held-out set | JEPA nearest sentence (today's D) | **JEPA + text decoder** | Decoder on the **true** vector (best case) | SFT (F) |
|---|---|---|---|---|
| ALFWorld new games (240) | 43.8% / 0.826 | 43.3% / 0.835 | 52.9% / 0.893 | 56.2% / 0.899 |
| ScienceWorld test types (480) | 47.3% / 0.779 | 44.0% / 0.749 | 50.6% / 0.823 | 69.0% / 0.890 |
| ScienceWorld seen variations (1,600) | 52.4% / 0.826 | 49.8% / 0.794 | 55.2% / 0.855 | 71.3% / 0.899 |
| ByteSized32 random 10% (295) | 26.8% / 0.612 | 24.4% / 0.600 | 33.2% / 0.750 | 55.6% / 0.776 |

Reading:
- **The text decoder does not beat nearest-sentence decoding.** It is within 0.5–3.3
  points (about the same on ALFWorld, slightly worse elsewhere). Writing new sentences
  doesn't help when the vector it reads from has already lost the details.
- **The vector is the deeper limit.** Even a *perfect* JEPA prediction (the true
  next-state vector) only decodes to 33–55% exact match. That is below SFT everywhere
  (−3 points on ALFWorld, −16 to −22 elsewhere). A frozen, general-purpose sentence
  embedding (NV-Embed) drops exact details such as object numbers, quantities and
  wording, and SFT keeps them by working on the text directly.
- So the D-vs-F gap splits into (a) what the frozen embedding loses (SFT → decoder on the
  true vector) and (b) prediction error (true vector → predicted vector, 3–10 points).
  Together with the InfoNCE ablation, this points to the **frozen encoder**, not the
  predictor's training or the decode step, as the main limit. Training the encoder
  itself (as I-JEPA and V-JEPA do) is the natural next step. That is future work.
- Not part of the decision-utility labels (fidelity only, n = 256, by choice).

Files: `outputs/jepa_decoder/{alfworld,sciworld,bytesized32}_b256.pt` (vectors) and
`*_results.json` (scores at steps 100 / 200 / 300).
