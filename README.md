# When Is a Specialist World Model Worth It for Language Agents?

**Team Nemotron** — AFML course project, Dept. of CSE (AI&ML), PES University
Shryes M · P Saanvi · Siddanth Anil · Rohini Vishu

Paper: [`docs/paper/Team_Nemotron_IEEE_Paper_final.docx`](docs/paper/Team_Nemotron_IEEE_Paper_final.docx)

## What this project does

A *world model* predicts the next observation of a text environment from the goal, the current
observation and a candidate action. With the same small training data (n = 32 to 1024 transitions)
we compare three ways to build one:

| Arm | World model | Uses the n transitions by |
|---|---|---|
| **G** (general) | frozen Qwen2.5-7B-Instruct (4-bit AWQ, vLLM) | retrieving the 5 most similar ones into the prompt (NV-Embed-v2) |
| **F** (fine-tuned) | Qwen2.5-1.5B-Instruct + LoRA | fine-tuning on them |
| **D** (JEPA specialist) | MLP predictor on frozen NV-Embed-v2 embeddings | training a next-state embedding predictor, decoded to the nearest training state |

Each arm is judged by the **decisions** it leads to: at held-out states with 5 candidate actions, the
arm predicts every outcome, a fixed judge (Qwen2.5-7B-Instruct-AWQ) rates them, and we measure the
**regret** of the chosen action. Every run is priced in GPU-hours on one RTX 5070 (12 GB).
Environments: ALFWorld, ScienceWorld (test split and held-out variations of seen task types) and
ByteSized32. Research questions: RQ1 regret by budget and capability, RQ2 compute, RQ3 predicting the
winner from 32 transitions, RQ4 per-decision routing.

**Main result (regret at n = 256, lower is better; `results/figures/main_table.md`)**

| Regime | Random | G | F | D top-1 |
|---|---|---|---|---|
| ALFWorld (new games) | 1.61 | **0.61** | 0.62 | 0.98 |
| ScienceWorld test | 1.47 | 0.10 | **0.07** | 0.26 |
| ScienceWorld seen | 1.38 | **0.10** | 0.12 | 0.33 |
| ByteSized32 (gold unseen) | 2.12 | 0.63 | **0.59** | 1.46 |

G and F are statistically tied in 19 of 24 (regime, budget) cells and F is better in the other 5;
both beat D at every budget; F is the cheapest good option; cheap features cannot predict the winner,
but a 0.02-GPU-hour JEPA probe ranks how far D falls behind F.

## Repository map

| Path | What it holds | Owner |
|---|---|---|
| `wm/` | ALFWorld / ScienceWorld wrappers, data collection, budgets, buffer export, branched held-out states, LLM client | Saanvi |
| `wm/utility/` | decision-regret scorer, judge, labels, episode harness | Saanvi |
| `jepa/` | arm D: NV-Embed encoder wrapper, predictor, training, nearest-state decoding | Shryes |
| `sft/` | arm F: data, LoRA model, training | Shryes |
| `rag_dropin/` | arm G: retrieval, prompts, vLLM generation, Recall@5, overnight GPU script | Rohini, Saanvi |
| `capability/` | capability tagger and ByteSized32 ground truth | Siddanth |
| `scripts/` | every entry point (training, prediction, features, predictor, analyses, figures) | all |
| `configs/config.yaml` | shared settings (seed 42, encoder, JEPA, SFT, retrieval) | all |
| `data/buffers/` | **training data**: the transition buffers all three arms trained on (see below) | Saanvi, Siddanth |
| `data/eval/` | held-out buffers and the branched states (held-out decision states with 5 candidates and true next states) | Saanvi |
| `data/budgets/`, `splits/` | budget id lists and train / held-out splits | Saanvi |
| `results/` | every result in the paper (see "Where each paper result comes from") | all |
| `docs/` | detailed docs per component; `docs/paper/` holds the paper and its section drafts | all |
| `sanity/` | pre-registered 3-shot pipeline sanity check | Saanvi |
| `tests/` | unit tests (CPU, no GPU or LLM needed) | all |
| `rag/`, `evaluation/`, `rq3/` | empty package stubs from the first layout; the code lives in `rag_dropin/`, `wm/utility/` and `scripts/rq3_predictor.py` | — |

### Training data (`data/buffers/`)

| File | Rows | Used by |
|---|---|---|
| `alfworld_games.jsonl` | 1,440 (12 training games, expert + random policy) | D, F, G training / retrieval |
| `sciworld_games.jsonl` | 2,300 (22 task types) | D, F, G for both ScienceWorld regimes |
| `bytesized32_games.jsonl` | 2,954 (31 games, from ByteSized32-State-Prediction) | D, F, G |
| `*_games.tagged.jsonl` | same rows with capability tags filled in | RQ3 features 2 and 4 |
| `sciworld_train.jsonl.gz` | larger raw ScienceWorld training pool | capability-tagger checks |

Each budget is a fixed, nested subset of these rows (seed 42). The exact rows of every budget are
recorded inside the D checkpoints, and F and G use the same rows (`scripts/export_train_rows.py`,
`rag_dropin/export_g_buffers.py`). We checked that all three buffers reproduce those rows exactly.

## Quick check: reproduce the paper's numbers from the included results (CPU, ~5 min)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                     # 67 tests, no GPU or LLM needed

# regret tables, labels and Figs. 1-2 from the scored outputs in results/utility/scored/
python -m wm.utility.score summarize --states data/eval/*_branched_states.jsonl \
    --true-values results/utility/*_true_values.jsonl --scored "results/utility/scored/*.jsonl" \
    --out-dir results/utility
python scripts/make_figures.py --utility-dir results/utility --out-dir results/figures

# RQ1 by capability (Fig. 3), RQ2 compute crossover (Fig. 4), RQ4 routing bound
python scripts/capability_regret.py --budget 256 --out-dir results/capability_regret
python scripts/compute_crossover.py --cost results/compute/cost_table.csv \
    --summary results/utility/summary.csv --d-mode top1 --out-dir results/crossover
python scripts/router_bound.py --out-dir results/router

# RQ3 predictor
python scripts/rq3_predictor.py --matrix results/features/feature_matrix.csv \
    --labels results/utility/labels_Dtop1.csv --sft-probe results/sft/sft_probe_scores.csv \
    --out results/rq3 --perm 200
```

Bootstrap confidence intervals of the random-choice baseline are not seeded, so they can differ in the
second decimal; every other number is deterministic.

## Full pipeline, end to end

All GPU stages ran on one RTX 5070 (12 GB) under WSL2; times are from `results/compute/cost_table.csv`.
Every stage is resumable: finished files are skipped on a rerun.

### 0. Setup

Three Python environments are needed because NV-Embed-v2 pins an old `transformers`:

```bash
python3 -m venv .venv      && .venv/bin/pip install -r requirements.txt               # base: tools, scoring, features
python3 -m venv .venv-enc  && .venv-enc/bin/pip install -r requirements-encoder.txt   # NV-Embed-v2 (D, G retrieval, Recall@5)
python3 -m venv .venv-sft  && .venv-sft/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 \
                           && .venv-sft/bin/pip install -r requirements-sft.txt       # F (LoRA)
python3 -m venv ~/vllm-venv && ~/vllm-venv/bin/pip install vllm==0.33.0                       # judge and G generator
cp .env.example .env       # LLM_BASE_URL / LLM_MODEL for the vLLM server
```

Models download from Hugging Face on first use: `nvidia/NV-Embed-v2`, `Qwen/Qwen2.5-1.5B-Instruct`,
`Qwen/Qwen2.5-7B-Instruct-AWQ`. The judge / G server:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --seed 0
```

### 1. Data (optional: the results use the files already in `data/`)

ALFWorld and ScienceWorld environments: `bash scripts/setup_vm.sh` (see `docs/SETUP_VM.md`).
Collection, budgets, held-out sets and branched states: commands in `docs/SCHEMA.md` and
`docs/DECISION_UTILITY.md`, for example

```bash
python -m wm.collect --env alfworld --split train --tasks splits/alfworld_games12.txt --dedupe --tag games \
    --expert-warmup 8 --target expert=900,random=540 --seed 0 --max-episodes 3000 --out data/raw/alfworld_games.jsonl
python -m wm.branch_states --env alfworld --data data/raw/alfworld_heldout.jsonl ...   # held-out decision states
```

ByteSized32 comes from the GPT-simulator release (`capability/README.md`, step 2).

### 2. Arm D (JEPA) — `.venv-enc`, GPU, about 0.4–1.5 GPU-hours per environment

```bash
python scripts/train_jepa.py --buffer data/buffers/alfworld_games.jsonl  --test-buffer data/eval/alfworld_heldout_buffer.jsonl --steps 2000 --real-encoder
python scripts/train_jepa.py --buffer data/buffers/sciworld_games.jsonl  --test-buffer data/eval/sciworld_heldout_buffer.jsonl --steps 2000 --real-encoder
python scripts/train_jepa.py --buffer data/buffers/sciworld_games.jsonl  --test-buffer data/eval/sciworld_seen_heldout_buffer.jsonl --steps 2000 --real-encoder
python scripts/train_jepa.py --buffer data/buffers/bytesized32_games.jsonl --steps 2000 --real-encoder   # random 10% held out
# predictions for the branched states, all budgets of one run at once
python scripts/jepa_predict_candidates.py --checkpoint outputs/jepa/<run>/budget_*.pt \
    --buffer data/buffers/alfworld_games.jsonl --queries data/eval/alfworld_branched_states.jsonl \
    --out "outputs/jepa_candidates/alfworld_b{budget}.jsonl" --k 5 --real-encoder
```

Budgets 32–1024, 2000 steps each, MSE to frozen targets (`configs/config.yaml`). Run folders are named
`outputs/jepa/<buffer>_steps2000_test-<held-out buffer>_sepact_adm_frozen`, the names stage 4 expects.
Details: `docs/LATENT_WORLD_MODEL.md`, `results/jepa_phase1/README.md`.

### 3. Arm F (fine-tuned) — `.venv-sft`, GPU, about 0.6–1.5 GPU-hours per environment

```bash
# copy D's exact training rows per budget (CPU, inside .venv-enc)
python scripts/export_train_rows.py --ckpt-dir outputs/jepa/<run> --buffer data/buffers/alfworld_games.jsonl --out sft/rows/alfworld.json
python scripts/train_sft.py --env alfworld        # also sciworld, bytesized32; writes outputs/sft_candidates/
```

Details: `docs/SFT.md`, `results/sft/README.md`.

### 4. Arm G, then scoring of all arms — one overnight run, about 5 hours

`rag_dropin/overnight.sh` exports G's retrieval rows from D's checkpoints, builds all prompts with
NV-Embed (stage 1), starts vLLM, generates G's predictions for every set and budget (stage 2), then runs
`scripts/run_utility.sh`, which judges D (three modes), F and G, writes `results/utility/` (scored
files, `summary.csv`, `judge_check.csv`, labels) and the figures.

```bash
SID=$PWD ENC_PY=.venv-enc/bin/python TORCH_PY=.venv-enc/bin/python WM_PY=.venv/bin/python \
    VLLM=~/vllm-venv/bin/vllm ENC_HF_HOME=~/.cache/huggingface \
    bash rag_dropin/overnight.sh 2>&1 | tee overnight.log
```

Scoring alone (vLLM running): `PRED_ROOT=outputs G_DIR=<G predictions> bash scripts/run_utility.sh`.
Details: `rag_dropin/README.md`, `docs/DECISION_UTILITY.md`.

### 5. Features, RQ1–RQ4 analyses and cost table — mostly CPU

```bash
bash scripts/rebuild_siddanth.sh            # tags, features 2 and 4, cost table, Figs. 3-4, router bound, feature matrix
RECALL=1 bash scripts/rebuild_siddanth.sh   # also feature 3 (Recall@5, GPU, .venv-enc)
python scripts/probe_fidelity.py --buffer data/buffers/alfworld_games.jsonl --real-encoder     # feature 1 (JEPA probe), per env
python scripts/sft_probe.py --buffer data/buffers/alfworld_games.jsonl data/buffers/sciworld_games.jsonl data/buffers/bytesized32_games.jsonl   # feature-1 ablation (GPU)
python scripts/rq3_predictor.py --matrix results/features/feature_matrix.csv --labels results/utility/labels_Dtop1.csv \
    --sft-probe results/sft/sft_probe_scores.csv --out results/rq3 --perm 200
```

`rebuild_siddanth.sh` also expects the raw ByteSized32 file and the per-person compute logs
(see the script header); the resulting `results/compute/cost_table.csv` is included.

## Where each paper result comes from

| Paper | File | Made by |
|---|---|---|
| Table I, Fig. 1 (regret by budget) | `results/figures/main_table.md`, `fig1_regret_vs_n.png` | `scripts/make_figures.py` |
| Fig. 2 (fidelity vs regret) | `results/figures/fig2_fidelity_vs_utility.png`, `results/utility/fidelity_vs_utility.csv` | `scripts/make_figures.py` |
| Fig. 3 (regret by capability) | `results/capability_regret/` | `scripts/capability_regret.py` |
| Table II, Fig. 4 (GPU-hours, crossover) | `results/compute/cost_table.csv`, `results/crossover/` | `scripts/compute_table.py`, `scripts/compute_crossover.py` |
| Table III (RQ3 predictor) | `results/rq3/README.md`, `results/rq3/final/` | `scripts/rq3_predictor.py` |
| RQ4 routing bound | `results/router/` | `scripts/router_bound.py` |
| Labels with ties | `results/utility/labels_D{top1,topk,native}.csv` | `python -m wm.utility.labels` |
| Judge check | `results/utility/judge_check.csv` | `wm/utility/score.py` |
| Text fidelity, JEPA ablations | `results/sft/README.md`, `results/jepa_phase1/README.md` | `scripts/train_sft.py`, `scripts/train_jepa.py` |
| Sanity check | `docs/SANITY_CHECK.md`, `results/sanity_alfworld.json` | `sanity/three_shot.py` |
| Fig. A1 (G's Recall@5) | `results/figures/fig_g_recall_at5_vs_n.png`, `results/utility/g_recall_at5.csv` | `rag_dropin/prepare_g_prompts.py`, `rag_dropin/plot_recall_vs_n.py` |

## Not in the repository

- **D checkpoints** (about 250 MB each) and the **D and F prediction files** stay on the GPU machine;
  stages 2–3 regenerate them. Their scored outputs are in `results/utility/scored/`.
- **G's prompts** (about 200 MB) are rebuilt by stage 4; G's raw predictions are in `results/g_candidates/`.
- One training seed (42) per configuration; see the paper's Limitations.
