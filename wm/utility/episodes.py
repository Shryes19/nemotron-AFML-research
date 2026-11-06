"""Episode-level check: does single-step decision quality carry over to whole episodes?

On ~10% of the held-out tasks, play full episodes with one-step lookahead: at every real state,
build K candidates with the same rule as the branched states (expert action + K-1 non-no-op
distractors), let a chooser pick one, and execute it in the real env. Reports success per
chooser, one row per task, so single-step regret can be correlated with episode success.

  python -m wm.utility.episodes --env alfworld --chooser expert --out results/utility/episodes_alfworld_expert.jsonl

Choosers:
  expert   the expert's action (an upper reference; ScienceWorld's expert exists only on its gold path)
  random   a random candidate (lower reference)
  oracle   steps every candidate in a replay of the real env and picks the best judged outcome
           (needs the judge: --judge llm on the GPU server, or --judge overlap for dry runs)
  A world-model arm plugs in as a PredictionChooser(predict_fn, judge) on GPU day, where
  predict_fn(state, actions) returns one predicted next observation per action.

ALFWorld leaks ~32 MB of disk per reset, so run it in chunks: --limit N plays N tasks, exits 3
if tasks remain (resumable by task id); loop until exit 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional

from wm.branch_states import NOOP_ACTIONS, choose_candidates
from wm.collect import load_task_ids
from wm.envs.base import make_env
from wm.utility.judge import make_judge
from wm.utility.score import choose, tie_rng

MAX_STEPS = {"alfworld": 50, "sciworld": 80}  # ALFWorld: Word2World's 50-turn cap


class ExpertChooser:
    name = "expert"

    def choose(self, env, state: Dict, actions: List[str], rng: random.Random) -> int:
        expert = env.expert_action()
        return actions.index(expert) if expert in actions else rng.randrange(len(actions))


class RandomChooser:
    name = "random"

    def choose(self, env, state, actions, rng):
        return rng.randrange(len(actions))


class OracleChooser:
    """Picks the action whose TRUE outcome the judge rates highest (replays the real env)."""
    name = "oracle"

    def __init__(self, judge, workers: int = 8):
        self.judge, self.workers = judge, workers

    def choose(self, env, state, actions, rng):
        outcomes = []
        for a in actions:
            env.replay(state["task_id"], state["prefix"])
            outcomes.append(env.step(a).feedback)
        env.replay(state["task_id"], state["prefix"])  # back to the real state before acting
        values = self.judge.values([(state["goal"], state["obs"], a, o) for a, o in zip(actions, outcomes)],
                                   workers=self.workers)
        i = choose(values, rng)
        return rng.randrange(len(actions)) if i is None else i


class PredictionChooser:
    """A world-model arm: predict_fn(state, actions) -> one predicted next observation per action."""
    name = "arm"

    def __init__(self, predict_fn: Callable, judge, name: str = "arm", workers: int = 8):
        self.predict_fn, self.judge, self.name, self.workers = predict_fn, judge, name, workers

    def choose(self, env, state, actions, rng):
        preds = self.predict_fn(state, actions)
        values = self.judge.values([(state["goal"], state["obs"], a, p) for a, p in zip(actions, preds)],
                                   workers=self.workers)
        i = choose(values, rng)
        return rng.randrange(len(actions)) if i is None else i


def pick_tasks(env_name: str, frac: float, seed: int) -> List[str]:
    ids = sorted(load_task_ids(env_name, "heldout"))
    n = max(1, int(math.ceil(frac * len(ids))))
    return sorted(random.Random(seed).sample(ids, n))


def play(env, env_name: str, task_id: str, chooser, k: int = 5, max_steps: Optional[int] = None,
         seed: int = 0) -> Dict:
    noop = NOOP_ACTIONS.get(env_name, frozenset())
    max_steps = max_steps or MAX_STEPS.get(env_name, 30)
    r = env.reset(task_id)
    prefix, won, chosen_expert = [], False, 0
    for step in range(max_steps):
        expert = env.expert_action()
        rng = random.Random("%d-%s-%d" % (seed, task_id, step))
        actions = choose_candidates(expert, env.admissible_actions(), k, rng, noop)
        if not actions:
            break
        state = {"task_id": task_id, "prefix": list(prefix), "obs": r.obs, "goal": env.goal()}
        i = chooser.choose(env, state, actions, tie_rng("%s-%d" % (task_id, step), seed))
        chosen_expert += int(actions[i] == expert)
        r = env.step(actions[i])
        prefix.append(actions[i])
        if r.won:
            won = True
        if r.done or r.won:
            break
    return {"task_id": task_id, "chooser": chooser.name, "success": won, "steps": len(prefix),
            "expert_agreement": round(chosen_expert / len(prefix), 4) if prefix else 0.0,
            "actions": prefix}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["alfworld", "sciworld"])
    p.add_argument("--chooser", required=True, choices=["expert", "random", "oracle"])
    p.add_argument("--frac", type=float, default=0.1, help="share of held-out tasks")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", choices=["llm", "overlap"], default="llm")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="play at most N tasks (exit 3 if more remain)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    tasks = pick_tasks(args.env, args.frac, args.seed)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["task_id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = [t for t in tasks if t not in done]
    chooser = {"expert": ExpertChooser, "random": RandomChooser}.get(args.chooser)
    chooser = chooser() if chooser else OracleChooser(make_judge(args.judge), args.workers)
    env = make_env(args.env)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("%d held-out tasks selected, %d to play" % (len(tasks), len(todo)), flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for n, task_id in enumerate(todo):
            if args.limit is not None and n >= args.limit:
                if hasattr(env, "close"):
                    env.close()
                return 3
            rec = play(env, args.env, task_id, chooser, args.k, seed=args.seed)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("%s success=%s steps=%d" % (task_id, rec["success"], rec["steps"]), flush=True)
    if hasattr(env, "close"):
        env.close()
    rows = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    print("%s %s: success %d/%d" % (args.env, chooser.name, sum(r["success"] for r in rows), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
