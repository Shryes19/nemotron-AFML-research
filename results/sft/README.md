# SFT arm (F): results

Qwen2.5-1.5B-Instruct + LoRA (r 16), next-observation loss on answer tokens. F trains on
exactly JEPA's rows at every budget. Training length: 5 epochs, at least 20 steps (pilot,
`docs/SFT.md`). Full runs on 9 Nov 2026 (RTX 5070, about 10 h overnight for all 18 runs,
including predictions).

**Fidelity** is exact match / word F1 (Saanvi's `normalize`, `word_f1`) on the same held-out
rows as JEPA. **D** = JEPA nearest-state decode (exact match) and **reach** = D's ceiling,
both from `results/jepa_phase1/README.md`. F writes new text, so it has no reach ceiling.

## ALFWorld: 240 rows from new games

| n | steps | **F** exact / F1 | D exact | D reach | lookup |
|---|---|---|---|---|---|
| 32 | 20 | **40.0%** / 0.754 | 35.0% | 37.5% | 37.1% |
| 64 | 20 | **51.2%** / 0.853 | 42.1% | 43.3% | 41.7% |
| 128 | 40 | **44.6%** / 0.832 | 42.9% | 45.8% | 44.2% |
| 256 | 80 | **56.2%** / 0.899 | 43.8% | 47.5% | 44.6% |
| 512 | 160 | **58.8%** / 0.908 | 49.6% | 55.0% | 50.8% |
| 1024 | 320 | **60.4%** / 0.908 | 52.5% | 58.8% | 52.9% |

## ScienceWorld: 480 rows, 12 test task types

| n | steps | **F** exact / F1 | D exact | D reach | lookup |
|---|---|---|---|---|---|
| 32 | 20 | **59.2%** / 0.826 | 24.2% | 29.4% | 14.6% |
| 64 | 20 | **59.2%** / 0.801 | 38.1% | 44.4% | 32.3% |
| 128 | 40 | **65.4%** / 0.834 | 43.3% | 50.2% | 37.5% |
| 256 | 80 | **69.0%** / 0.890 | 47.3% | 53.5% | 41.9% |
| 512 | 160 | **78.7%** / 0.913 | 53.1% | 59.4% | 46.7% |
| 1024 | 320 | **83.3%** / 0.937 | 60.2% | 65.2% | 51.2% |

## ScienceWorld seen: 1,600 rows, held-out variations of 20 seen types

| n | steps | **F** exact / F1 | D exact | D reach | lookup |
|---|---|---|---|---|---|
| 32 | 20 | **62.9%** / 0.836 | 27.8% | 32.0% | 16.0% |
| 64 | 20 | **61.6%** / 0.813 | 41.4% | 45.8% | 32.8% |
| 128 | 40 | **67.1%** / 0.850 | 47.2% | 53.3% | 40.3% |
| 256 | 80 | **71.3%** / 0.899 | 52.4% | 58.2% | 47.0% |
| 512 | 160 | **82.0%** / 0.918 | 59.8% | 65.4% | 52.6% |
| 1024 | 320 | **85.9%** / 0.941 | 65.2% | 70.6% | 57.3% |

## ByteSized32: JEPA's random 10% held out (295 rows, same games)

| n | steps | **F** exact / F1 | D exact | D reach | lookup |
|---|---|---|---|---|---|
| 32 | 20 | **22.4%** / 0.566 | 6.1% | 18.6% | 7.5% |
| 64 | 20 | **22.7%** / 0.493 | 6.8% | 21.7% | 7.1% |
| 128 | 40 | **45.1%** / 0.735 | 19.3% | 27.5% | 13.2% |
| 256 | 80 | **55.6%** / 0.776 | 26.8% | 31.9% | 16.9% |
| 512 | 160 | **62.7%** / 0.805 | 36.6% | 41.7% | 23.1% |
| 1024 | 320 | **76.3%** / 0.879 | 48.8% | 54.6% | 33.6% |

## Reading

- F has the best fidelity at every budget in every environment. The gap is small on new
  ALFWorld games (+2 to +12 points) and large on ScienceWorld and ByteSized32 (+16 to
  +35). F often scores above D's reach ceiling, because it can write next states that
  are not in the buffer.
- **Why F leads on fidelity, and why that is not the verdict on JEPA.** D answers with
  the nearest *training* state's text, so it can never be exactly right when the true next
  state is not in its buffer (at n = 32 it chooses from 32 sentences). Within that limit,
  D's predictor is near-optimal: it gets 87–97% of its reach ceiling in every environment,
  with no collapse, so the gap to F is the decoder and the buffer, not JEPA training. JEPA's
  usual advantage (ignoring unpredictable detail, as in I-/V-JEPA and DINO-WM) has little to
  work with here: next states are short, mostly deterministic sentences, which next-token
  fine-tuning copies well (Word2World reports near-perfect exact match on ALFWorld). JEPA's
  claim is about *decisions*, so the test is decision-utility regret, including its
  `native` mode (action choice in embedding space, no text).
- Unlike Ashok et al.'s finding that fine-tuning fails at low scales, F is already strong
  at n = 32. Short, formulaic feedback text is easy to copy (ScienceWorld 59% at n = 32).
- Not monotone everywhere: ALFWorld 64 > 128, and ScienceWorld 32 ≈ 64. With 240–480
  rows, one run can move by ±3–4 points. The 32 and 64 budgets both use the 20-step floor.
- **This is fidelity, not decision utility.** The RQ3 labels come from Saanvi's regret
  scoring of the prediction files below. Whether F's fidelity lead turns into lower
  regret is the fidelity ≠ utility question (Lambert et al.).

## Files (on the GPU box, not in git)

| | |
|---|---|
| Predictions for scoring | `outputs/sft_candidates/{alfworld,sciworld,sciworld_seen,bytesized32}_b{n}.jsonl` (193 / 182 / 323 / 328 lines), generic format, `arm: "F"`; score with `--format generic --arm F` |
| Per budget | `outputs/sft/<env>/budget_<n>/`: `adapter/`, `done.json`, `heldout_<set>.jsonl` |
| Summary | `outputs/sft/<env>/results.json` |
| Training rows | `sft/rows/<env>.json` (copied from the JEPA checkpoints) |

GPU time is in the compute log as `sft_train` / `sft_predict` (arm F).

## SFT probe: Feature 1 ablation input (9–10 Nov 2026)

`scripts/sft_probe.py` → `results/sft/sft_probe_scores.csv` (65 games). The protocol is the
same as the JEPA probe: same per-game split (32 train / up to 64 held out), 5 draws. The
difference is that it runs 20 LoRA steps and scores the written text by word F1 (higher =
easier game). It took about 5.5 h of GPU for all 65 games (roughly 2.5–6.5 min per game),
against seconds per game for the JEPA probe on cached embeddings.

| Env | Games | F1 range | Mean | Separation ratio (SFT) | JEPA probe separation | Spearman with JEPA probe |
|---|---|---|---|---|---|---|
| ALFWorld | 12 | 0.784–0.886 | 0.856 | 1.12 | 1.63 | −0.54 |
| ScienceWorld | 22 | 0.722–0.872 | 0.792 | 0.95 | 1.55 | −0.66 |
| ByteSized32 | 31 | 0.654–0.945 | 0.768 | 1.41 | 2.02 | −0.65 |
| All | 65 | 0.654–0.945 | 0.792 | 1.48 | — | −0.63 |

Separation ratio = between-game sd ÷ mean within-game (draw-to-draw) sd, the same
definition as for the JEPA probe. The correlation is negative because the JEPA probe is a
distance (lower = easier) and the SFT probe is an F1 score (higher = easier).

Reading:
- The two probes mostly agree on which games are easy (|ρ| ≈ 0.6).
- The JEPA probe separates games more clearly relative to its own noise (1.55–2.02 vs
  0.95–1.41), and it is far cheaper.
- Whether either one predicts the winning arm is tested by `scripts/rq3_predictor.py`
  once the labels arrive.

`clean-energy` (ByteSized32) has only 24 transitions, so it was trained on 20 and scored
on 4, flagged "short" as in the JEPA probe.
