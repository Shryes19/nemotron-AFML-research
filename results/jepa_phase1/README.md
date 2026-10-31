# JEPA arm — Phase 1 results (ALFWorld)

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

| Folder | Admissible-action input | Action-choice control |
|---|---|---|
| `alfworld_games` | **on** (current default) | yes (`text_overlap`) |
| `alfworld_games_noadm` | off | no (run before the control existed) |

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

## Probe fidelity (`probe/`)

`alfworld_games`: Feature 1 per game (n = 32, 1 epoch, 3 seeds), computed with
the **old joint text input** and no admissible feature. To be rerun once the
team settles the probe definition.

## Reproduce

On Siddhant's machine in `.venv-enc` (defaults: separate action, admissible
input, 2,000 steps):

```bash
python scripts/train_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
python scripts/train_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder --holdout-games 2 --seed 43
python scripts/diagnose_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
python scripts/probe_fidelity.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
```

Older runs: add `--joint` (single text input), `--no-admissible`, or
`--use-epochs --epochs 20`.
