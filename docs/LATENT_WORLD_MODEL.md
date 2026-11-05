# Using the JEPA world model for decision utility

A guide for the decision-utility scorer: what the JEPA arm gives you, how to
get it, and how to run it next to Qwen on the 12 GB card.

## 1. What the JEPA model does

Given a state (the observation text) and an action, it predicts **a vector
for the next state**, not text. Under the hood:

1. NV-Embed-v2 (frozen, 4-bit) turns the observation and the action into two
   4096-number vectors.
2. Two extra numbers are added: "is this action in the [ADMISSIBLE] list?" and
   "does the observation have a usable list?". ScienceWorld lists templates
   (`open OBJ`), so there both are 0, same as ByteSized32.
3. A small trained network (29.4M parameters) maps
   `[obs vector | action vector | 2 numbers]` → a predicted next-state vector,
   in the same space as NV-Embed's vectors for real states.

To compare that prediction with anything, you embed the other thing with the
same encoder and take the cosine similarity. The code that does all this
exactly as in training is `jepa/planning/latent_world_model.py`
(`LatentWorldModel`).

## 2. Two ways to score K candidate actions

**A. Turn predictions into text (top-k decode), then let your LLM scorer judge.**
For each action, JEPA returns the k states in its training buffer closest to
its prediction, with similarities. Your scorer reads "if I do X, the next
state is probably: ..." and judges which action is best, the same way it
judges the RAG and SFT arms' text. This is the fair comparison across arms,
so **use this one for the main results.**

- Limitation: JEPA can only answer with states it has seen (its "reach"). On
  unseen ALFWorld games the best possible exact match is about 45-49%. Giving
  the scorer the top 5, not just the top 1, softens this.

**B. Rank in vector space (no text, no LLM).**
If you have a target text (the goal, or the gold next state), JEPA ranks the
actions by how close each predicted next state is to it. This is how JEPA
planners actually work and it has no reach limit. It is cheap, so it is a
good extra column ("JEPA-native"), but it is not comparable to the
text-based arms, so keep it separate.

Both are produced by one script (next section).

## 3. The hand-off: one file, run before Qwen

NV-Embed (~6 GB) and Qwen2.5-7B-AWQ (~5-6 GB + cache) do not fit together, so:

**Step 1 — you write the branched states** to a JSONL, one line per state:

```json
{"id": "alfworld/look_at_obj_2/step_7", "obs": "<full observation text>",
 "actions": ["go to desk 1", "take pen 1 from desk 1", "use desklamp 1", "...K actions"],
 "goal": "optional: target text for ranking (way B)",
 "game": "any other keys are copied through unchanged"}
```

`obs` must be the same observation format as in the buffers (with its
`[ADMISSIBLE]` section if it has one): the model was trained on that format.

**Step 2 — run JEPA** (Shryes, or anyone on the GPU box, inside `.venv-enc`),
once per budget checkpoint:

```bash
python scripts/jepa_predict_candidates.py \
  --checkpoint outputs/jepa/<run>/budget_256.pt \
  --buffer data/buffers/alfworld_games.jsonl \
  --queries <your_states>.jsonl \
  --out outputs/jepa_candidates/alfworld_b256.jsonl --k 5 --real-encoder
```

**Step 3 — your scorer reads the output**, one line per state:

```json
{"id": "...", "budget": 256, "checkpoint": "...", "goal": "...", "game": "...",
 "candidates": [
   {"action": "go to desk 1",
    "topk": [{"text": "You arrive at desk 1. On the desk 1, you see ...", "sim": 0.91}, "..."],
    "target_sim": 0.74},
   "..."],
 "ranking": [2, 0, 1]}
```

`topk` is best first. `target_sim` and `ranking` only appear when the query
had a `goal`.

Cost: encoding is the slow part (~a few seconds per 100 states on the 5070);
the predictor itself is instant. It is logged as stage `jepa_predict` in the
GPU-hour log.

## 4. Things to keep fair

- **Each budget answers only from its own data.** Checkpoints now store the
  buffer rows they were trained on (`train_rows`), and the script decodes
  only into those. So the budget-32 model cannot "know" states only the
  budget-1024 model saw. Pass the same `--buffer` the model was trained on;
  the script checks this.
- **Unseen games.** For held-out results, train with your task-disjoint set
  as the test file (`train_jepa.py --test-buffer <held_out>.jsonl`) and draw
  the branched states from that set.
- **ScienceWorld non-determinism (~2%).** JEPA is deterministic (same input,
  same output), so it does not need replays. Replays only matter for the
  "true" next state you compare against.
- **Admissible list.** In ALFWorld the random policy's failures are exactly
  the non-admissible actions, so JEPA knows when an action will fail. That
  is information present in the observation, not leakage, but it is worth
  one sentence in the paper.

## 5. Using it from Python directly

```python
from jepa.encoder.load import load_encoder
from jepa.planning.latent_world_model import LatentWorldModel

enc = load_encoder(real=True, device="cuda")
wm = LatentWorldModel.from_checkpoint("outputs/jepa/<run>/budget_256.pt", enc,
                                      buffer_items=train_rows_items)
z = wm.predict(obs, actions)          # (K, 4096) predicted next-state vectors
wm.decode_topk(z, k=5)                # [[(text, sim), ...] per action]   way A
wm.rank_by_target(z, goal_text)       # [(action index, sim), ...] best first   way B
```

Prefer the script: it picks the right buffer rows for you.
