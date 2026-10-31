# Transition data schema

One JSON object per line (`.jsonl`, gzipped for the full data on Drive). Validate a file with:

```bash
python -m wm.schema path/to/file.jsonl
```

## Fields

| Field | Type | Meaning |
|---|---|---|
| `transition_id` | str | `{episode_id}-s{step:03d}`, unique |
| `env` | str | `alfworld` or `sciworld` |
| `split` | str | `train` (from Word2World train ids) or `heldout` (from Word2World test ids). Split is by task instance, never by transition |
| `task_id` | str | Word2World/AgentGym item id, e.g. `alfworld_17`, `sciworld_3491` |
| `task_type` | str | ALFWorld task family or ScienceWorld task name |
| `episode_id` | str | `{env}-{split}-e{index:06d}` |
| `step` | int | 0-based step index inside the episode |
| `goal` | str | task description shown to the agent |
| `o_t` | str | composite observation before the action (format below) |
| `a_t` | str | action taken. One of `admissible_actions`, except ALFWorld random steps that deliberately use a well-formed non-admissible command (answered `Nothing happens.`) |
| `o_next` | str | composite observation after the action |
| `feedback_next` | str | raw environment response to `a_t` (target for exact-match fidelity) |
| `reward` | float | env reward after `a_t` (ScienceWorld: score delta; ALFWorld: 1.0 when won) |
| `done` | bool | episode ended after `a_t` |
| `won` | bool | task solved after `a_t` |
| `capability_tag` | list[str] | empty at collection; filled by the capability tagger |
| `policy` | str | who chose `a_t`: `expert`, `random`, `llm` (`llm_fallback` = LLM reply was not a valid choice, a random admissible action was used). In ALFWorld about 1/3 of `random` steps are failing commands, because every admissible ALFWorld command succeeds |
| `action_prefix` | list[str] | every action from reset up to, not including, `a_t`. `len == step`. Replaying it from `reset(task_id)` reproduces `o_t` exactly |
| `admissible_actions` | list[str] | full list of valid actions at `o_t` |
| `collector_seed` | int | seed of the collection run |
| `env_version` | str | pinned env version string |
| `llm_meta` | object or null | `{"model", "cache_key"}` when `policy` is `llm` |

## Composite observation (`o_t`, `o_next`)

Only information the agent can see. No hidden initial state (unlike Word2World, which gives ALFWorld/ScienceWorld world models the full initial state).

```
[GOAL] put a clean mug in coffeemachine.
[ROOM] You are in the middle of a room. Looking quickly around you, you see a cabinet 1, ...
[LOCATION] cabinet 1
[FEEDBACK] You open the cabinet 1. The cabinet 1 is open. In it, you see a cup 1.
[INVENTORY] You are carrying: nothing.
[ADMISSIBLE] go to cabinet 1 | open cabinet 1 | ...
```

- **ALFWorld:** `[ROOM]` is the room description from reset (lists receptacles, never their contents). `[LOCATION]` and `[INVENTORY]` are tracked from feedback. `[ADMISSIBLE]` is the full admissible command list.
- **ScienceWorld:** `[ROOM]` is the live `look` output, `[INVENTORY]` the live inventory. `[ADMISSIBLE]` lists action *templates* only, because the full valid action–object list has hundreds to thousands of entries; the full list is in the `admissible_actions` field.

`wm.obs.parse_obs(text)` splits the text back into its sections.

## Budgets

`data/budgets/{env}_seed{s}_n{n}.txt` lists `transition_id`s, one per line. For every seed, budgets are nested (32 ⊂ 64 ⊂ … ⊂ 1024) and hold the 50/30/20 expert/random/LLM mix. All budget ids come from the `train` split.

## Buffer format for the JEPA / SFT arms

`jepa/training/buffer.py` (team repo) reads `obs`, `action`, `obs_next`, `env`, `game`, `cap_tag`. `wm/export_buffers.py` converts our rows:

| Buffer field | From | Note |
|---|---|---|
| `obs` | `o_t` | composite observation (goal, room, location, feedback, inventory, admissible) |
| `action` | `a_t` | |
| `obs_next` | `feedback_next` | exactly the text the env returned after the action (`--obs-next composite` uses `o_next`) |
| `env` | `env` | |
| `game` | `task_id` | the game / task instance; RQ3 uses games as data points (`--game family` groups by task type) |
| `cap_tag` / `cap_tags` | `capability_tag` | first tag / full list; empty until tagged |
| `obs_next_full`, `feedback_next`, `transition_id`, `task_id`, `task_type`, `split`, `policy` | same | kept for tracing back to the source row |

**Per-game ALFWorld buffer** (12 games, 2 per task type, about 120 transitions each; `splits/alfworld_games12.txt`):

```bash
python -m wm.collect --env alfworld --split train --tasks splits/alfworld_games12.txt --dedupe --tag games --expert-warmup 8 --target expert=900,random=540 --seed 0 --max-episodes 3000 --out data/raw/alfworld_games.jsonl
python -m wm.export_buffers --env alfworld --train data/raw/alfworld_games.jsonl --single data/buffers/alfworld_games.jsonl
```

Each game is played many times with the expert/random mix; `--dedupe` keeps only new `(task_id, o_t, a_t)` transitions. ALFWorld's expert is nearly deterministic per game, so `--expert-warmup 8` starts expert episodes with up to 8 random steps (recorded as `random`); the expert replans from there and reaches new states.

**Per-budget buffers** (+ the task-disjoint held-out set) from the main pool:

```bash
python -m wm.export_buffers --env alfworld --train data/raw/alfworld_train.jsonl.gz --heldout data/raw/alfworld_heldout.jsonl.gz --provisional
```
