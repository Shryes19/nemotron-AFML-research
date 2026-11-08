# SFT arm (F) — Qwen2.5-1.5B-Instruct + LoRA

Code: `sft/data.py` (rows, prompt, scoring), `sft/model.py` (LoRA, training loop, greedy
generation), `scripts/export_train_rows.py`, `scripts/train_sft.py`.

## What it learns

Input: current observation + action. Target: the next observation text (`obs_next`, the
short feedback, the same text as the branched states' `true_next`). Loss is next-token
cross-entropy on the answer tokens only. No history and no hidden state, so F sees the
same information as D (JEPA) and G (RAG).

**Prompt:** the observation without the long `[ADMISSIBLE]` list, plus one line,
`Action allowed: yes / no / unknown`. That is the same fact JEPA gets as a hand feature,
computed with JEPA's own `admissible_set`. ScienceWorld (templates) and ByteSized32 (no
list) get `unknown`. Train and test prompts are built the same way.

## Same rows as JEPA

For each (env, budget), F trains on exactly the rows of the JEPA checkpoint of that
budget. `scripts/export_train_rows.py` copies them into `sft/rows/<env>.json` (and, for
ByteSized32, rebuilds JEPA's random 10% held-out rows for fidelity).

| env | JEPA checkpoints | held-out fidelity sets | branched-state files |
|---|---|---|---|
| alfworld | `alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen` | `alfworld_heldout_buffer` (240) | `alfworld` (193) |
| sciworld | `sciworld_games_steps2000_test-sciworld_heldout_buffer_sepact_adm_frozen` | `sciworld_heldout_buffer` (480), `sciworld_seen_heldout_buffer` (1,600) | `sciworld` (182), `sciworld_seen` (323) |
| bytesized32 | `bytesized32_games_steps2000_sepact_adm_frozen` | JEPA's random 10% (295) | `bytesized32` (328) |

## Settings and sources

| Setting | Value | Source |
|---|---|---|
| Input → target, no history | obs + action → next obs | Ashok et al. 2026, Eq. 3 |
| Loss on answer tokens only | yes | Ashok Eq. 3; ExpRAG-LoRA (2026) |
| Model | Qwen2.5-1.5B-Instruct | Word2World (Li et al. 2026): 1.5B captures core dynamics in structured envs |
| Epochs | 5 (+ `min_steps` floor from the pilot) | Word2World Table 7 (5 epochs) |
| Schedule / precision | constant LR, 10 warmup steps, bf16 | Word2World Table 7 |
| LoRA | r 16, alpha 32, dropout 0.05, all linear layers | config; all-linear as in QLoRA |
| LR | 2e-4 | usual LoRA rate (Word2World's 1e-5 is full fine-tuning) |
| Batch | 16 per step (micro-batches of 4) | — |
| Decoding | greedy, max 192 new tokens | deterministic |
| Fidelity | exact match + word F1 (Saanvi's `normalize` / `word_f1`) | Word2World uses EM (F1 for open-ended) |

**What to expect at small n:** Ashok et al. report that fine-tuning "fails at low scales"
and starts working around n ≈ 80. Weak F at n = 32–64 is a finding, not a bug.

## Pilot (training length)

`--pilot` holds out 2 whole **training** games (never the test buffers), trains at n = 32
and n = 1024 for 480 steps, and scores 200 held-out rows at steps 0, 10, 20, … 480. Pick
`sft.min_steps` (the floor for small budgets) from the n = 32 curve, and check that
5 epochs (320 steps at n = 1024) is not past the peak. Then write the choice into
`configs/config.yaml` before the full runs.

## Commands (GPU box)

```bash
# once: separate venv
python3 -m venv .venv-sft && source .venv-sft/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-sft.txt

# once per env (inside .venv-enc, CPU): copy JEPA's training rows
python scripts/export_train_rows.py --ckpt-dir outputs/jepa/<checkpoint folder> \
    --buffer data/buffers/<env>_games.jsonl --out sft/rows/<env>.json

# pilot, then full runs (inside .venv-sft). Finished budgets are skipped, so a crash
# only costs the budget in progress.
python scripts/train_sft.py --env alfworld --pilot
python scripts/train_sft.py --env alfworld
python scripts/train_sft.py --env sciworld
python scripts/train_sft.py --env bytesized32
```

**Outputs:**
- `outputs/sft/<env>/budget_<n>/`: `adapter/` (LoRA weights), `done.json` (steps, loss, fidelity), `heldout_<set>.jsonl` (every held-out prediction)
- `outputs/sft/<env>/results.json`
- `outputs/sft_candidates/<set>_b<n>.jsonl`: Saanvi's generic format, scored with `--format generic --arm F`

GPU time is logged as `sft_train` and `sft_predict` (arm F).

**CPU code check** (no download): build a tiny random Qwen2 model and run with
`--model <dir> --device cpu --limit 12`.
