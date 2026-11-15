# Demo: one decision, three world models

```bash
python scripts/demo.py            # the paper's worked example (ALFWorld, candle, n = 256)
python scripts/demo.py --slow     # same, paced for screen recording (about 40 s)
```

Runs on a CPU in under a second with only the Python standard library. It replays one held-out state
from the saved evaluation outputs (`data/eval/*_branched_states.jsonl`,
`results/utility/scored/*.jsonl`); no model is loaded.

What it shows, in order:

1. the situation (goal, location, what the agent sees and carries);
2. the five candidate actions (expert's move + 4 distractors);
3. for each world model (G, F, D): its predicted outcome of every action, the judge's rating of that
   prediction, and the action it therefore picks;
4. what really happens when each action is replayed in the environment;
5. the regret of each model's decision, and how it changes from n = 32 to 1,024;
6. mean regret over all held-out states of that environment (matches Table II).

Other states:

```bash
python scripts/demo.py --find d_fails                         # G and F right, D wrong
python scripts/demo.py --find f_beats_g --regime bytesized32
python scripts/demo.py --state <id> --budget 64
python scripts/demo.py --regime sciworld --random --seed 3
```

Regimes: `alfworld`, `sciworld`, `sciworld_seen`, `bytesized32`. Budgets: 32 to 1024.

Note: the predicted next-state texts are shown only for the worked example
(`demo/candle_example.json`, from Table VI of the paper), because G's raw generations were not kept.
For every other state the demo shows the judge's ratings of each model's predictions, the decision and
the regret, which are what the evaluation is based on.

## Suggested 40-second narration (for the presentation video)

> "Here is one real decision from our test set. The agent must put a candle in a cabinet and is standing
> at a countertop with a candle on it. It has five possible actions. For each one it asks its world
> model what will happen. G and F both predict that picking up the candle works, the judge rates that
> 8 out of 10, so both pick it: regret zero. D, the JEPA model, can only return a next state it has
> seen in training. For 'take candle' it returns 'Nothing happens', from a different game, so it picks
> 'examine countertop' instead: regret 6. Across all 193 ALFWorld test states, G and F are tied
> at about 0.6 and D is at 0.98. That is the paper's result in one example."
