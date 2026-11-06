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

ALFWorld by family: gold reach 19.7% (pick_two) to 50.0% (pick_cool).
ScienceWorld: 77–100% for task types in the training buffer;
`measure-melting-point-unknown-substance` (not in the training buffer) 33.3%.
So on ALFWorld, JEPA's top-1 text is often from another game (wrong objects);
report `--mode topk` and `--mode native` next to top-1.

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
