"""Branched states for the decision-utility scorer.

For each held-out state: K candidate actions (the expert's action + K-1 other admissible,
non-no-op actions, verb-stratified, in shuffled order), and the TRUE next state of every candidate,
obtained by replaying the state's action_prefix in the real env and stepping the candidate.
The same file is used by every world-model arm (G, F, D), so all arms rank the same choices.

  python -m wm.branch_states --env alfworld --data data/raw/alfworld_heldout.jsonl \
      --out data/eval/alfworld_branched_states.jsonl --k 5 --seed 0

One JSON line per state. Required by the JEPA hand-off (docs/LATENT_WORLD_MODEL.md):
  id, obs (composite o_t, same format as the buffers), actions (K)
Also: game (agreed RQ3 unit), goal (task text, for JEPA's own ranking), gold_action,
gold_index, gold_next (feedback of the gold action), true_next / true_next_full /
true_reward / true_done / true_won (one per action, same order as actions), transition_id,
task_id, task_type, action_prefix, source_policy, env, split, replay_consistent, replay_checked.

States are skipped (and listed with a reason in <out>.skipped) when there is no expert action
(ScienceWorld off its gold path), the expert action is a no-op (NOOP_ACTIONS), or fewer than
K-1 non-no-op distractors exist.

True next states are cached in cache/true_next_<env>.jsonl keyed by
sha256(env, task_id, prefix, action), and the run is resumable (finished ids are skipped).
ScienceWorld is not fully deterministic (~2% of states): --check-frac of the states (default
0.1 for ScienceWorld) step every candidate 3 times; replay_consistent says whether they agreed
(up to word order) and replay_checked marks those states.

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


# Actions that never change the world or reveal anything new. They are not used as distractors,
# and states whose expert action is one of them are skipped: with them, 36% of ALFWorld
# distractors were look/inventory, which made picking the right action easier than 1-in-5.
# (A deviation from "any admissible action"; examine / look at X stay.)
NOOP_ACTIONS = {
    "alfworld": frozenset({"look", "inventory"}),
    "sciworld": frozenset({"look around", "inventory", "task"}),
}


def candidate_pool(expert: Optional[str], admissible: List[str], noop=frozenset()) -> List[str]:
    return [a for a in admissible if a not in SKIP_ACTIONS and a not in noop and a != expert]


def choose_candidates(expert: Optional[str], admissible: List[str], k: int,
                      rng: random.Random, noop=frozenset()) -> List[str]:
    """Expert action + k-1 distinct other admissible, non-no-op actions (verb-stratified), shuffled."""
    pool = candidate_pool(expert, admissible, noop)
    chosen = [expert] if expert else []
    while len(chosen) < k and pool:
        a = sample_by_verb(pool, rng)
        pool.remove(a)
        chosen.append(a)
    rng.shuffle(chosen)
    return chosen


def interleave_by_game(rows: List[Dict]) -> List[Dict]:
    """Round-robin across games (each game's rows in transition_id order), so per-game quotas fill
    evenly instead of one game at a time."""
    groups = {}
    for r in rows:
        groups.setdefault(game_of(r), []).append(r)
    out, i = [], 0
    while any(i < len(g) for g in groups.values()):
        for game in sorted(groups):
            if i < len(groups[game]):
                out.append(groups[game][i])
        i += 1
    return out


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
    p.add_argument("--split", default="heldout", choices=["heldout", "heldout_seen"],
                   help="which held-out rows to branch (heldout_seen: reserved variations of seen task types)")
    p.add_argument("--max-per-type", type=int, default=None,
                   help="stop adding states for a game once it has this many")
    p.add_argument("--min-per-type", type=int, default=None, help="report games below this many states")
    p.add_argument("--count-existing", nargs="*", default=[],
                   help="other branched files whose states count toward --max-per-type")
    p.add_argument("--done-from", nargs="*", default=[],
                   help="other branched files (and their .skipped) whose ids are already finished")
    p.add_argument("--expert-rows-only", action="store_true",
                   help="only branch rows chosen by the expert (on the gold path; avoids replaying "
                        "ScienceWorld states that would be skipped for having no expert)")
    p.add_argument("--shard", default=None,
                   help="I/N: process every N-th row starting at I, with its own cache file and "
                        "1/N of each type's remaining --max-per-type quota (run N processes in parallel)")
    p.add_argument("--replays", type=int, default=1, help="steps per candidate")
    p.add_argument("--check-frac", type=float, default=None,
                   help="share of states whose candidates are stepped 3 times to check determinism "
                        "(default 0 for ALFWorld, 0.1 for ScienceWorld)")
    args = p.parse_args(argv)
    check_frac = args.check_frac if args.check_frac is not None else (0.0 if args.env == "alfworld" else 0.1)
    noop = NOOP_ACTIONS.get(args.env, frozenset())

    rows = sorted((r for r in read_jsonl(args.data) if r["split"] == args.split and r["env"] == args.env),
                  key=lambda r: r["transition_id"])
    rows = [r for r in rows if len(candidate_pool(None, r["admissible_actions"], noop)) >= args.k - 1]
    if args.n is not None and args.n < len(rows):
        rows = sorted(random.Random(args.seed).sample(rows, args.n), key=lambda r: r["transition_id"])
    if args.expert_rows_only:
        rows = [r for r in rows if r["policy"] == "expert"]
    shard_i, shard_n = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    # shard by a hash of the transition id (not by position: collection alternates games, so
    # every-N-th row would give each shard only some games), then interleave within the shard
    rows = interleave_by_game([r for r in rows if int(hashlib.md5(r["transition_id"].encode()).hexdigest(), 16)
                               % shard_n == shard_i])

    def read_states(path):
        out = []
        if os.path.exists(path):
            out = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        return out

    # finished ids: written states plus states skipped earlier (kept in a side file, with reason)
    skipped_path = args.out + ".skipped"
    done_ids, skipped, per_game = set(), {}, {}
    for rec in read_states(args.out):
        done_ids.add(rec["id"])
        per_game[rec["game"]] = per_game.get(rec["game"], 0) + 1
    for rec in read_states(skipped_path):
        skipped[rec["id"]] = rec["reason"]
    other = {}                                # states in other files: count toward the quota
    for path in args.count_existing + args.done_from:
        for rec in read_states(path):
            other[rec["id"]] = rec["game"]
    for path in args.done_from:
        done_ids |= {rec["id"] for rec in read_states(path)}
        skipped.update({rec["id"]: rec["reason"] for rec in read_states(path + ".skipped")})
    base = {}
    for game in other.values():
        base[game] = base.get(game, 0) + 1

    def quota_full(game):
        if args.max_per_type is None:
            return False
        own = per_game.get(game, 0)
        if shard_n == 1:
            return own + base.get(game, 0) >= args.max_per_type
        remaining = max(0, args.max_per_type - base.get(game, 0))
        return own >= -(-remaining // shard_n)          # ceil(remaining / N) per shard

    cache_name = "true_next_%s.jsonl" % args.env if shard_n == 1 else \
        "true_next_%s.shard%d.jsonl" % (args.env, shard_i)
    cache = TrueNextCache(os.path.join("cache", cache_name))
    env = make_env(args.env)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("%d eligible states, %d written, %d skipped earlier" % (len(rows), len(done_ids), len(skipped)),
          flush=True)

    def skip(sid, reason):
        skipped[sid] = reason
        with open(skipped_path, "a", encoding="utf-8") as sf:
            sf.write(json.dumps({"id": sid, "reason": reason}) + "\n")

    t0 = time.time()
    new = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            sid = "%s/%s/%s" % (args.env, game_of(row), row["transition_id"])
            if sid in done_ids or sid in skipped:
                continue
            if quota_full(game_of(row)):
                continue
            if args.limit is not None and new >= args.limit:
                left = len(rows) - len(done_ids) - len(skipped)
                print("limit reached; %d states left" % left, flush=True)
                if hasattr(env, "close"):
                    env.close()
                return 3
            env.replay(row["task_id"], row["action_prefix"])
            expert = env.expert_action()
            if expert is None:
                skip(sid, "no_expert")         # ScienceWorld state off the gold path
                continue
            if expert in noop:
                skip(sid, "noop_gold")
                continue
            if len(candidate_pool(expert, env.admissible_actions(), noop)) < args.k - 1:
                skip(sid, "small_pool")
                continue
            rng = random.Random("%d-%s" % (args.seed, row["transition_id"]))
            actions = choose_candidates(expert, row["admissible_actions"], args.k, rng, noop)
            checked = rng.random() < check_frac
            replays = 3 if checked else args.replays
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
                "replay_checked": checked,
                "transition_id": row["transition_id"],
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "action_prefix": row["action_prefix"],
                "source_policy": row["policy"],
                "env": args.env,
                "split": args.split,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            new += 1
            per_game[rec["game"]] = per_game.get(rec["game"], 0) + 1
            if (i + 1) % 10 == 0:
                print("%d/%d states (%.0fs)" % (i + 1, len(rows), time.time() - t0), flush=True)
    if hasattr(env, "close"):
        env.close()
    reasons = {}
    for r in skipped.values():
        reasons[r] = reasons.get(r, 0) + 1
    print("done: %d states written, skipped %s" % (len(done_ids) + new, reasons), flush=True)
    print("states per game: %s" % dict(sorted(per_game.items())), flush=True)
    if args.min_per_type is not None:
        low = {g: n for g, n in per_game.items() if n < args.min_per_type}
        print("games below %d states: %s" % (args.min_per_type, low or "none"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
