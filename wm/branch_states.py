"""Branched states for the decision-utility scorer.

For each held-out state: K candidate actions (the expert's action + K-1 other admissible
actions, verb-stratified, in shuffled order), and the TRUE next state of every candidate,
obtained by replaying the state's action_prefix in the real env and stepping the candidate.
The same file is used by every world-model arm (G, F, D), so all arms rank the same choices.

  python -m wm.branch_states --env alfworld --data data/raw/alfworld_heldout.jsonl \
      --out data/eval/alfworld_branched_states.jsonl --k 5 --seed 0

One JSON line per state. Required by the JEPA hand-off (docs/LATENT_WORLD_MODEL.md):
  id, obs (composite o_t, same format as the buffers), actions (K)
Also: game (agreed RQ3 unit), goal (task text, for JEPA's own ranking), gold_action,
gold_index, gold_next (feedback of the gold action), true_next / true_next_full /
true_reward / true_done / true_won (one per action, same order as actions), transition_id,
task_id, task_type, action_prefix, source_policy, env, split, replay_consistent.

True next states are cached in cache/true_next_<env>.jsonl keyed by
sha256(env, task_id, prefix, action), and the run is resumable (finished ids are skipped).
ScienceWorld is not fully deterministic (~2% of states), so each candidate is stepped
--replays times; replay_consistent says whether all replays agreed (up to word order).

ALFWorld leaks ~32 MB of disk per reset (textworld loads a temp copy of the planner library
that is only freed when the process exits), so long runs go in chunks: --limit N stops after
N new states with exit code 3, and a loop restarts it until it exits 0:

  until python -m wm.branch_states ... --limit 40; [ $? -ne 3 ]; do :; done
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

from wm.envs.base import make_env
from wm.policies import SKIP_ACTIONS, sample_by_verb
from wm.schema import read_jsonl


def game_of(row: Dict) -> str:
    """Agreed RQ3 game unit: ALFWorld task_id, ScienceWorld task type."""
    return row["task_id"] if row["env"] == "alfworld" else row["task_type"]


def choose_candidates(expert: Optional[str], admissible: List[str], k: int,
                      rng: random.Random) -> List[str]:
    """Expert action + k-1 distinct other admissible actions (verb-stratified), shuffled."""
    pool = [a for a in admissible if a not in SKIP_ACTIONS and a != expert]
    chosen = [expert] if expert else []
    while len(chosen) < k and pool:
        a = sample_by_verb(pool, rng)
        pool.remove(a)
        chosen.append(a)
    rng.shuffle(chosen)
    return chosen


def true_key(env_name: str, task_id: str, prefix: List[str], action: str) -> str:
    blob = json.dumps([env_name, task_id, prefix, action])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def same_text(a: str, b: str) -> bool:
    return sorted(a.split()) == sorted(b.split())


class TrueNextCache:
    def __init__(self, path: str):
        self.path, self.data = path, {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                self.data[rec["key"]] = rec
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def get(self, key):
        return self.data.get(key)

    def put(self, rec):
        self.data[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def step_true(env, env_name: str, row: Dict, action: str, cache: TrueNextCache, replays: int) -> Dict:
    key = true_key(env_name, row["task_id"], row["action_prefix"], action)
    hit = cache.get(key)
    if hit:
        return hit
    outs = []
    for _ in range(replays):
        env.replay(row["task_id"], row["action_prefix"])
        outs.append(env.step(action))
    first = outs[0]
    rec = {"key": key, "feedback": first.feedback, "obs": first.obs, "reward": float(first.reward),
           "done": bool(first.done), "won": bool(first.won),
           "consistent": all(same_text(o.feedback, first.feedback) for o in outs)}
    cache.put(rec)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["alfworld", "sciworld"])
    p.add_argument("--data", required=True, help="held-out transitions (.jsonl / .jsonl.gz)")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n", type=int, default=None, help="number of states (default: all eligible)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N new states (exit code 3 if more remain)")
    p.add_argument("--replays", type=int, default=None,
                   help="steps per candidate (default 1 for ALFWorld, 3 for ScienceWorld)")
    args = p.parse_args(argv)
    replays = args.replays or (1 if args.env == "alfworld" else 3)

    rows = sorted((r for r in read_jsonl(args.data) if r["split"] == "heldout" and r["env"] == args.env),
                  key=lambda r: r["transition_id"])
    rows = [r for r in rows if len([a for a in r["admissible_actions"] if a not in SKIP_ACTIONS]) >= args.k]
    if args.n is not None and args.n < len(rows):
        rows = sorted(random.Random(args.seed).sample(rows, args.n), key=lambda r: r["transition_id"])

    done_ids = set()
    if os.path.exists(args.out):
        done_ids = {json.loads(line)["id"] for line in open(args.out, encoding="utf-8")}
    cache = TrueNextCache(os.path.join("cache", "true_next_%s.jsonl" % args.env))
    env = make_env(args.env)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("%d eligible states, %d already done" % (len(rows), len(done_ids)), flush=True)

    t0 = time.time()
    new = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            sid = "%s/%s/%s" % (args.env, game_of(row), row["transition_id"])
            if sid in done_ids:
                continue
            if args.limit is not None and new >= args.limit:
                print("limit reached; %d states left" % (len(rows) - len(done_ids) - new), flush=True)
                if hasattr(env, "close"):
                    env.close()
                return 3
            env.replay(row["task_id"], row["action_prefix"])
            expert = env.expert_action()
            rng = random.Random("%d-%s" % (args.seed, row["transition_id"]))
            actions = choose_candidates(expert, row["admissible_actions"], args.k, rng)
            truths = [step_true(env, args.env, row, a, cache, replays) for a in actions]
            gold = actions.index(expert) if expert in actions else None
            rec = {
                "id": sid,
                "obs": row["o_t"],
                "actions": actions,
                "game": game_of(row),
                "goal": row["goal"],
                "gold_action": expert,
                "gold_index": gold,
                "gold_next": truths[gold]["feedback"] if gold is not None else None,
                "true_next": [t["feedback"] for t in truths],
                "true_next_full": [t["obs"] for t in truths],
                "true_reward": [t["reward"] for t in truths],
                "true_done": [t["done"] for t in truths],
                "true_won": [t["won"] for t in truths],
                "replay_consistent": all(t["consistent"] for t in truths),
                "transition_id": row["transition_id"],
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "action_prefix": row["action_prefix"],
                "source_policy": row["policy"],
                "env": args.env,
                "split": "heldout",
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            new += 1
            if (i + 1) % 10 == 0:
                print("%d/%d states (%.0fs)" % (i + 1, len(rows), time.time() - t0), flush=True)
    if hasattr(env, "close"):
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
