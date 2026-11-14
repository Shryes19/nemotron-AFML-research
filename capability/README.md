# Capability tagging + ByteSized32-SP (Phase 1, Siddanth)

Everything here is pure Python stdlib, so it runs in either venv (`.venv` or `.venv-enc`).

## Run it

```bash
# 1. tag the ALFWorld and ScienceWorld buffers (fills cap_tag / cap_tags)
python scripts/tag_buffer.py --buffer data/buffers/alfworld_games.jsonl --out data/buffers/alfworld_games.tagged.jsonl
python scripts/tag_buffer.py --buffer data/buffers/sciworld_train.jsonl.gz --out data/buffers/sciworld_train.tagged.jsonl

# 2. set up ByteSized32-SP (31 games, 2,954 transitions, ground-truth tags from the JSON state)
git clone --depth 1 https://github.com/cognitiveailab/GPT-simulator.git ../GPT-simulator
mkdir -p data/raw/bytesized32 && unzip -o ../GPT-simulator/data/test.zip -d data/raw/bytesized32
python scripts/setup_bytesized32.py --raw data/raw/bytesized32/test.jsonl --out data/buffers/bytesized32_games.jsonl

# 3. Feature 2: capability mix vector per game (n=32, mean of 5 draws, plus n=all)
python scripts/capability_mix.py --text-tags \
    --buffers data/buffers/alfworld_games.tagged.jsonl data/buffers/sciworld_train.tagged.jsonl \
              data/buffers/bytesized32_games.jsonl \
    --out results/capability/capability_mix.csv

# tests
python tests/test_capability_tagger.py
```

## The five tags

| tag | meaning | ALFWorld example |
|---|---|---|
| `object_state` | an object's own state changes | `close drawer 1` |
| `containment_location` | the agent moves, or an object changes container / inventory | `go to shelf 1`, `take mug 2 from shelf 1` |
| `numeric_physical` | a quantity or phase changes | ScienceWorld `use thermometer on gallium` |
| `partial_obs` | the next observation shows things o_t gave no way to know | `go to shelf 1` -> "you see a mug 2" |
| `action_failure` | the action is rejected or has no effect | "Nothing happens." |

Tags are multi-label (`cap_tags`). `cap_tag` is one primary label by priority
`action_failure > numeric_physical > object_state > containment_location > partial_obs > none`;
`none` = nothing changes and nothing new is shown (`inventory`, `look` with nothing new).

Feature 2 = the five `rate_*` columns at n=32 (share of the 32 transitions carrying each tag).

## Design choices worth knowing

- ALFWorld `heat` / `cool` are tagged `object_state`, not `numeric_physical`: ALFWorld stores them as
  flags and never shows a temperature. So ALFWorld's numeric rate is 0 by construction.
- Quantities that appear because the agent walked into a new room are `partial_obs` (revealed), not
  `numeric_physical` (changed). Same for `look`/`examine` that only report a value.
- ByteSized32 ground truth comes from the JSON state diff (`capability/bytesized32.py`). It has no
  `partial_obs` because the game exposes the full state. Use `--text-tags` when building the RQ3 feature
  matrix so every environment's Feature 2 is measured the same way (from text); the ground truth is for
  validating the text tagger.

## Phase 2 pieces (already usable)

```bash
# Feature 4: environment statistics per game (n=32, mean of 5 draws)
python scripts/env_stats.py --buffers data/buffers/alfworld_games.jsonl \
    data/buffers/sciworld_train.jsonl.gz data/buffers/bytesized32_games.jsonl \
    --out results/features/env_stats.csv

# RQ3 feature matrix: one row per (env, game); Features 1 / 3 and the label are added as they land
python scripts/build_feature_matrix.py --mix results/capability/capability_mix.csv \
    --stats results/features/env_stats.csv \
    [--probe results/jepa_phase1/probe_scores.csv] [--recall results/rag/recall_at5.csv] \
    [--labels results/utility/labels.csv] --out results/features/feature_matrix.csv

# GPU-hour accounting: wrap every costed step in every arm
#   from capability.compute_log import track
#   with track("jepa_train", arm="D", env="alfworld", game=g, n=256): ...
python scripts/compute_table.py      # -> results/compute/cost_table.csv
```

Expected input formats for the optional files: CSV with columns `env, game, <value>` where value is
`probe_fidelity`, `recall_at5` or `label`. A row with game `*` applies to every game of that env.

## Phase 3 (RQ2, Feature 3, reproducibility)

```bash
# everything in this part, in order (CPU steps take seconds)
bash scripts/rebuild_siddanth.sh
RECALL=1 bash scripts/rebuild_siddanth.sh      # also Feature 3: GPU + encoder venv (.venv-enc), ~15 min
```

- `scripts/recall_feature.py`: Feature 3, G's Recall@5 on each game's own 32 training transitions
  (leave-one-out, same draws as Features 2/4). Not `results/utility/g_recall_at5.csv`, which is measured
  on the held-out evaluation states and would leak test data into an ex-ante feature.
- `scripts/compute_crossover.py`: RQ2. Regret against GPU-hours per arm and environment
  (`results/crossover/`: points, crossover table, figure). Cost = building + running the world model at
  that budget; evaluation (utility_scoring) is left out. NV-Embed encoding (`nv_embed_encode`) counts
  as D's cost (`--shared-to D`): G encodes its own index (`rag_index_build`).
- `scripts/compute_table.py` also counts G's vLLM generation (`g_generation`) as GPU time.
