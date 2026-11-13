# Decision utility

How we score a world model by the **decisions** its predictions lead to, not by how good its text looks (Lambert et al. 2020; the survey's §6 takeaway). Code: `wm/branch_states.py`, `wm/utility/`.

## The metric

At each held-out state there are K = 5 candidate actions (the expert's action + 4 others). For every candidate we know the **true** next state, from replaying the real game.

1. **True values.** A frozen judge (Qwen2.5-7B, the same for every arm) rates each true next state from 0 to 10: how much does this action, with this outcome, move the agent toward completing the task? This gives V_true(a) for each candidate.
2. **An arm chooses.** The arm predicts the next state of each candidate. The same judge rates the predictions, and the arm "chooses" the candidate whose *predicted* outcome scores highest. Ties are broken by a seeded coin that is the same for every arm.
3. **Regret** = max_a V_true(a) − V_true(chosen). Lower is better; 0 means the arm picked a best action.
4. **Oracle** = choosing with the true next states: regret 0 by construction. Its **gold-hit** rate (does it pick the expert's action?) measures the judge itself. So:
   - oracle gold-hit tells us how good the judge is;
   - the gap between an arm and the oracle is world-model error.

We also report each arm's **fidelity** on the same predictions: normalized exact match and word F1 against the true next states. `fidelity_vs_utility.csv` flags (env, budget) pairs where the best arm by fidelity is not the best by utility.

## Branched states (same for every arm)

| File | States | Games | Train buffer | Test buffer (`--test-buffer`) |
|---|---|---|---|---|
| `data/eval/alfworld_branched_states.jsonl` | ALFWorld held-out | Word2World test tasks (not the 12 training games) | `alfworld_games.jsonl` | `data/eval/alfworld_heldout_buffer.jsonl` |
| `data/eval/sciworld_branched_states.jsonl` | ScienceWorld held-out, long episodes | 12 task types (the only ones in Word2World's test split) | `sciworld_games.jsonl` | `data/eval/sciworld_heldout_buffer.jsonl` |
| `data/eval/sciworld_seen_branched_states.jsonl` | ScienceWorld **held-out variations of seen task types** (split `heldout_seen`), 323 states | 20 of the 22 matrix types | `sciworld_games.jsonl` | `data/eval/sciworld_seen_heldout_buffer.jsonl` |
| `data/eval/bytesized32_branched_states.jsonl` | ByteSized32 **known states, new actions** (GPT-simulator crawl) | 31 games, 3–45 states each | `bytesized32_games.jsonl` | none: the states are training states |

**Three evaluation regimes** (report them separately):
- **ALFWorld: new games.** Every state comes from a game no arm trained on.
- **ScienceWorld: new variations of seen task types.** The test file uses Word2World's test split. The `heldout_seen` file uses train-split variations reserved by `scripts/reserve_heldout_seen.py` (`splits/sciworld_heldout_seen.txt`); no training data file uses them, and `wm.collect` removes them from `train`. `identify-life-stages-1/2` have no unused variations left, so they get no held-out states.
- **ScienceWorld coverage (test file + seen file, states per matrix task type):** 14 of 22 types have ≥ 25. Below 25: `freeze` 22, `lifespan-longest-lived-then-shortest-lived` 22, `power-component-renewable-vs-nonrenewable-energy` 23, `test-conductivity-of-unknown-substances` 24, `chemistry-mix-paint-tertiary-color` 10, `power-component` 10 (their few unused variations ran out of on-gold-path states); `identify-life-stages-1/2` 0. Report the small ones with their CIs. `measure-melting-point-unknown-substance` (27 states, test file only) is not a matrix type: report it separately.
- **ByteSized32: known states, new actions.** `wm/bs32_branch.py` builds them from the GPT-simulator crawl (`train.jsonl`, ~150 actions tried per state on each game's gold playthrough). 310 of the crawl's 337 states are also in `test.jsonl`, which the arms train on, so only the candidate actions are new: distractors are (state, action) pairs absent from the training data. The gold action comes from walking the game's playthrough; `gold_seen_in_training` flags the 118 of 328 states whose gold transition is in the training data. Per-game counts are small (median ~8), so report wide CIs.
- **ByteSized32's recorded score** (`true_score`, `true_reward`) gives a judge-free value for each candidate. The score changes in only 45 of 328 states, but those states check the judge against the game itself.

**Candidate rule:**
- **Distractors exclude no-op actions:** ALFWorld `look`, `inventory`; ScienceWorld `look around`, `inventory`, `task`. `examine` / `look at X` stay.
- **States whose expert action is a no-op are dropped.**
- **Why:** with "any admissible action", 36% of ALFWorld distractors were look/inventory, which made picking the right action easier than 1 in 5. Report this as a deviation in the paper.
- **ScienceWorld:** the expert exists only on its gold path, so its branched states are on-gold-path states. Skipped states are listed with reasons in `<file>.skipped`.
- **ScienceWorld held-out pool:** collected in long-episode mode from the 12 task types in Word2World's test split, with steps sampled across whole episodes (`--record-prob 0.2`, at most 6 rows per episode). Recording only each episode's first steps gave a median prefix of 2 and almost no heating/measuring states.

  ```bash
  python -m wm.collect --env sciworld --split heldout --balance-by task_type --late-prefix --max-steps 80 --dedupe --tag heldout-games --target expert=300,random=180 --seed 0 --max-rows-per-episode 6 --record-prob 0.2 --out data/raw/sciworld_heldout_games.jsonl
  # held-out variations of seen task types (after scripts/reserve_heldout_seen.py)
  python -m wm.collect --env sciworld --split heldout_seen --balance-by task_type --late-prefix --max-steps 80 --dedupe --tag heldout-seen --target expert=1000,random=600 --seed 0 --max-rows-per-episode 6 --record-prob 0.2 --out data/raw/sciworld_heldout_seen.jsonl
  # deep ScienceWorld states replay slowly (~1 min each), so: expert rows only, a quota of 25 per
  # type counting the test file, 3 shards in parallel (hash of transition id), then concatenate
  for i in 0 1 2; do python -m wm.branch_states --env sciworld --split heldout_seen --data data/raw/sciworld_heldout_seen.jsonl --out data/eval/sciworld_seen_part$i.jsonl --expert-rows-only --max-per-type 25 --count-existing data/eval/sciworld_branched_states.jsonl --shard $i/3 & done; wait
  # ByteSized32 (GPT-simulator data/train.zip, data/test.zip, data/playthroughs)
  python -m wm.bs32_branch --crawl train.jsonl --train test.jsonl --playthroughs data/playthroughs --out data/eval/bytesized32_branched_states.jsonl
  ```

Field list: `docs/SCHEMA.md` → "Branched states".

## What each arm must produce

**Generic format (RAG arm G, SFT arm F):** one line per branched state:

```json
{"id": "<state id>", "arm": "G", "budget": 256,
 "actions": ["<the state's 5 actions>"], "pred_next": ["<predicted next observation for each action>"]}
```

- One file per (arm, budget).
- `actions` must be the state's actions; any order works, and they are matched by text.
- `pred_next` is the text the env would return after the action, like `true_next` / the buffers' `obs_next`.

**JEPA arm D:** use the output of `scripts/jepa_predict_candidates.py` unchanged (`docs/LATENT_WORLD_MODEL.md`), one file per budget. It is scored three ways:
- `--mode top1` (main result): the nearest buffer state per action;
- `--mode topk`: the judge sees the 5 nearest states as a ranked list;
- `--mode native`: JEPA's own goal-based ranking, with no judge.

## Running it (GPU needed for the judge)

**Judge model:** Qwen2.5-7B-Instruct-**AWQ** (4-bit) on vLLM, so it fits the 12 GB RTX 5070.
This deviates from the plan's full-precision Qwen2.5-7B-Instruct; the same judge scores every arm.

One script does everything for the four branched sets (ALFWorld, ScienceWorld test, ScienceWorld
seen, ByteSized32): true values, D in 3 modes, F, G when its files exist, the summary and labels.
It is resumable (true values are cached, finished scored files are skipped).

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --seed 0
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
PRED_ROOT=~/nemotron-AFML-research/outputs DRY=1 bash scripts/run_utility.sh   # CPU check first, overlap judge
PRED_ROOT=~/nemotron-AFML-research/outputs bash scripts/run_utility.sh          # the real run
```

Outputs in `results/utility/`:
- `<set>_true_values.jsonl`, `scored/<set>_<arm>_<mode>_b<n>.jsonl`;
- `summary.csv`: one row per env, split (regime), arm, mode, budget and game, plus pooled `game = *`
  rows and oracle rows. ByteSized32 also has a `seen_state_new_actions:gold_unseen` split: the 210
  states whose gold transition is not in the arms' training data (report this one);
- `fidelity_vs_utility.csv`: best arm by regret vs by exact match, per env, split and budget;
- `judge_check.csv`: where the env records a reward, how often the judge's true values order two
  candidates like the reward does (ByteSized32: the 45 states whose score changes);
- `labels_D{top1,topk,native}.csv`: labels at n = 256 per family, one file per D mode.
  ScienceWorld test and seen states are pooled per task type. Columns `regret_*` and `n_*` stay so
  the predictor can weight by state count. ByteSized32 rows are included; `rq3_predictor.py` leaves
  them out by default.

Single commands, when needed:
```bash
python -m wm.utility.score true-values --states data/eval/alfworld_branched_states.jsonl --out results/utility/alfworld_true_values.jsonl
python -m wm.utility.score score --states data/eval/alfworld_branched_states.jsonl --true-values results/utility/alfworld_true_values.jsonl     --pred outputs/jepa_candidates/alfworld_b256.jsonl --format jepa --mode top1 --arm D --out results/utility/scored/alfworld_D_top1_b256.jsonl
python -m wm.utility.labels --scored "results/utility/scored/*.jsonl" --budget 256 --arms G,F,D --key family --arm-mode D=native
```

- `--workers` sends judge calls in parallel; responses are cached, so reruns are free.
- Judge passes are logged as stage `utility_scoring` in the shared GPU-hour log.
- `--judge overlap` (or `DRY=1`) is a dry run without an LLM; its numbers mean nothing.

## Episode check (`wm/utility/episodes.py`)

10% of the held-out tasks, played to the end with one-step lookahead: at each real state, the same candidate rule applies, a chooser picks, and the real env executes.
- **Choosers:** `expert`, `random` (references), `oracle` (true outcomes + judge).
- **Reference results** (ALFWorld, 20 held-out tasks, 50-step cap): expert 11/20, random 1/20 (`results/utility/episodes_alfworld_{expert,random}.jsonl`). With a 30-step cap the expert managed only 4/20, too little room to separate world models, so the cap is Word2World's 50.
- **Arms:** plug in on GPU day through `PredictionChooser(predict_fn, judge)`.
- **Use:** per-task success rows are correlated with single-step regret.
- **Chunking:** ALFWorld runs need `--limit` chunks (about 32 MB of disk per reset until the process exits).

```bash
until python -m wm.utility.episodes --env alfworld --chooser expert --out results/utility/episodes_alfworld_expert.jsonl --limit 3; [ $? -ne 3 ]; do :; done
```
