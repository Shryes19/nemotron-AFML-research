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

**Candidate rule:**
- **Distractors exclude no-op actions:** ALFWorld `look`, `inventory`; ScienceWorld `look around`, `inventory`, `task`. `examine` / `look at X` stay.
- **States whose expert action is a no-op are dropped.**
- **Why:** with "any admissible action", 36% of ALFWorld distractors were look/inventory, which made picking the right action easier than 1 in 5. Report this as a deviation in the paper.
- **ScienceWorld:** the expert exists only on its gold path, so its branched states are on-gold-path states. Skipped states are listed with reasons in `<file>.skipped`.
- **ScienceWorld held-out pool:** collected in long-episode mode from the 12 task types in Word2World's test split, with steps sampled across whole episodes (`--record-prob 0.2`, at most 6 rows per episode). Recording only each episode's first steps gave a median prefix of 2 and almost no heating/measuring states.

  ```bash
  python -m wm.collect --env sciworld --split heldout --balance-by task_type --late-prefix --max-steps 80 --dedupe --tag heldout-games --target expert=300,random=180 --seed 0 --max-rows-per-episode 6 --record-prob 0.2 --out data/raw/sciworld_heldout_games.jsonl
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

On the GPU box, with vLLM serving Qwen and `.env` pointing at it (`docs/GPU_DAY.md`):

```bash
# once per env: judge the true next states
python -m wm.utility.score true-values --states data/eval/alfworld_branched_states.jsonl --out results/utility/alfworld_true_values.jsonl

# once per (arm, budget)
python -m wm.utility.score score --states data/eval/alfworld_branched_states.jsonl --true-values results/utility/alfworld_true_values.jsonl \
    --pred outputs/jepa_candidates/alfworld_b256.jsonl --format jepa --mode top1 --arm D --out results/utility/alfworld_D_top1_256.jsonl
python -m wm.utility.score score ... --pred <rag_predictions_b256.jsonl> --format generic --arm G --out results/utility/alfworld_G_top1_256.jsonl

# tables: results/utility/summary.csv (+ oracle rows) and fidelity_vs_utility.csv
python -m wm.utility.score summarize --states data/eval/*_branched_states.jsonl --true-values results/utility/*_true_values.jsonl --scored "results/utility/*_[GFD]_*.jsonl"

# winner labels for the RQ3 matrix (scripts/build_feature_matrix.py --labels)
python -m wm.utility.labels --scored "results/utility/*_[GFD]_*.jsonl" --budget 256 --arms G,F,D --key game --out results/utility/labels.csv
```

- `--workers 8` sends judge calls in parallel; responses are cached, so reruns are free.
- Judge passes are logged as stage `utility_scoring` in the shared GPU-hour log.
- Add `--judge overlap` to any command for a dry run without an LLM. That judge just counts goal words, so its numbers mean nothing.

**Label unit (team decision pending):**
- **ScienceWorld:** held-out states share task types with training, so `--key game` joins the feature matrix directly.
- **ALFWorld:** held-out games are different games from the 12 training games. Labels join only per task family (`--key family`), or would need states from the training games themselves.

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
