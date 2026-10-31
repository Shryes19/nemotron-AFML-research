"""Collect transitions with the mixed policy and write one JSONL row per transition.

Episode scheme: the expert acts for the first k steps, then the episode's driver policy
(random or llm) acts for m steps. Expert-driven episodes stay with the expert. Every row
records the policy that chose its action, and per-policy row targets set the mix.

  python -m wm.collect --env alfworld --split train --target expert=800,random=480 \
      --seed 0 --out data/raw/alfworld_train.jsonl

Resumable: rerunning with the same arguments continues after the last finished episode.

Per-game buffers (many transitions from a few fixed games):

  python -m wm.collect --env alfworld --split train --tasks splits/alfworld_games12.txt       --dedupe --tag games --expert-warmup 8 --target expert=900,random=540 --out data/raw/alfworld_games.jsonl

--tasks always plays the game with the fewest rows so far; --dedupe never records the same
(task_id, o_t, a_t) twice, so repeated episodes on one game add only new transitions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from typing import Dict, List

from wm.envs.base import make_env
from wm.policies import SKIP_ACTIONS, make_policy
from wm.schema import read_jsonl

MAX_EPISODE_STEPS = 30
MAX_EXPERT_PREFIX = 15   # k ~ U[0, 15] before a random/llm driver takes over
DRIVER_STEPS = (3, 10)   # m ~ U[3, 10] driver steps


def load_task_ids(env: str, split: str) -> List[str]:
    """Word2World ids for a split. Train drops any id that is also a test id: Word2World's
    ScienceWorld train and test files share 61 ids, and held-out must stay task-disjoint."""
    def read(name):
        with open(os.path.join("splits", "%s_%s.json" % (env, name))) as f:
            return [x["item_id"] for x in json.load(f)]
    if split == "heldout":
        return read("test")
    test = set(read("test"))
    return [t for t in read("train") if t not in test]


def parse_target(text: str) -> Dict[str, int]:
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def run_episode(env, env_name, split, task_id, ep, driver, policies, rng, seed, need, max_rows=None,
                seen=None, tag=None, expert_warmup=0):
    """Roll out one episode; return the rows it produced (only for policies still needed).
    If `seen` is a set, (task_id, o_t, a_t) already in it are not recorded again.
    expert_warmup > 0: expert-driven episodes start with U[0, expert_warmup] random steps, so a
    replanning expert (ALFWorld) reaches new states on repeated plays of the same game."""
    episode_id = "%s-%s-e%06d" % (env_name, tag or split, ep)
    r = env.reset(task_id)
    k = MAX_EPISODE_STEPS if driver == "expert" else rng.randint(0, MAX_EXPERT_PREFIX)
    m = rng.randint(*DRIVER_STEPS)
    warmup = rng.randint(0, expert_warmup) if driver == "expert" and expert_warmup else 0
    prefix, rows = [], []
    for step in range(MAX_EPISODE_STEPS):
        who = "random" if step < warmup else ("expert" if step < k else driver)
        if step >= k + m and driver != "expert":
            break
        action, meta = policies[who].act(env, rng, obs_text=r.obs)
        if action is None and who == "expert" and driver != "expert":
            k = step  # expert ran out (e.g. ScienceWorld gold path ended): hand over now
            who = driver
            action, meta = policies[who].act(env, rng, obs_text=r.obs)
        if action is None:
            break
        if meta and meta.get("fallback"):
            who = "llm_fallback"
        o_t, adm = r.obs, r.admissible
        r = env.step(action)
        room_left = max_rows is None or len(rows) < max_rows
        is_new = seen is None or (task_id, o_t, action) not in seen
        if action not in SKIP_ACTIONS and room_left and is_new and need.get(who.replace("_fallback", ""), 0) > 0:
            if seen is not None:
                seen.add((task_id, o_t, action))
            rows.append({
                "transition_id": "%s-s%03d" % (episode_id, step),
                "env": env_name,
                "split": split,
                "task_id": task_id,
                "task_type": env.task_type(),
                "episode_id": episode_id,
                "step": step,
                "goal": env.goal(),
                "o_t": o_t,
                "a_t": action,
                "o_next": r.obs,
                "feedback_next": r.feedback,
                "reward": float(r.reward),
                "done": bool(r.done),
                "won": bool(r.won),
                "capability_tag": [],
                "policy": who,
                "action_prefix": list(prefix),
                "admissible_actions": list(adm),
                "collector_seed": seed,
                "env_version": env.version,
                "llm_meta": meta.get("llm_meta") if meta else None,
            })
            need[who.replace("_fallback", "")] -= 1
        prefix.append(action)
        if r.done:
            break
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["alfworld", "sciworld"])
    p.add_argument("--split", required=True, choices=["train", "heldout"])
    p.add_argument("--target", required=True, help="rows per policy, e.g. expert=800,random=480")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-episodes", type=int, default=100000)
    p.add_argument("--max-rows-per-episode", type=int, default=None,
                   help="record at most this many rows per episode (more task diversity)")
    p.add_argument("--tasks", default=None,
                   help="file with one task id per line: play only these, balancing rows across them")
    p.add_argument("--dedupe", action="store_true", help="skip repeated (task_id, o_t, a_t)")
    p.add_argument("--tag", default=None, help="episode id label instead of the split name")
    p.add_argument("--expert-warmup", type=int, default=0,
                   help="random steps (up to N) before the expert in expert episodes; ALFWorld only")
    args = p.parse_args(argv)

    target = parse_target(args.target)
    have, last_ep = Counter(), -1
    per_task, seen = Counter(), (set() if args.dedupe else None)
    if os.path.exists(args.out):
        for row in read_jsonl(args.out):
            have[row["policy"].replace("_fallback", "")] += 1
            last_ep = max(last_ep, int(row["episode_id"].rsplit("e", 1)[1]))
            per_task[row["task_id"]] += 1
            if seen is not None:
                seen.add((row["task_id"], row["o_t"], row["a_t"]))
    need = {k: v - have[k] for k, v in target.items()}
    print("existing rows:", dict(have), "still needed:", need, flush=True)

    env = make_env(args.env)
    policies = {name: make_policy(name, args.env) for name in set(target) | {"expert", "random"}}
    if args.expert_warmup and args.env != "alfworld":
        raise SystemExit("--expert-warmup needs a replanning expert (ALFWorld only)")
    task_ids = load_task_ids(args.env, args.split)
    random.Random(args.seed).shuffle(task_ids)
    if args.tasks:
        with open(args.tasks) as f:
            chosen = [line.strip() for line in f if line.strip()]
        bad = [t for t in chosen if t not in set(task_ids)]
        if bad:
            raise SystemExit("not %s task ids: %s" % (args.split, bad[:3]))
        task_ids = chosen
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    ep = last_ep + 1
    with open(args.out, "a", encoding="utf-8") as f:
        while any(v > 0 for v in need.values()) and ep < args.max_episodes:
            rng = random.Random("%d-%d" % (args.seed, ep))
            open_policies = [k for k, v in need.items() if v > 0]
            driver = rng.choices(open_policies, weights=[need[k] for k in open_policies])[0]
            if args.tasks:
                task_id = min(task_ids, key=lambda t: (per_task[t], task_ids.index(t)))
            else:
                task_id = task_ids[ep % len(task_ids)]
            rows = run_episode(env, args.env, args.split, task_id, ep, driver, policies, rng,
                               args.seed, need, args.max_rows_per_episode, seen, args.tag,
                               args.expert_warmup)
            for row in rows:
                f.write(json.dumps(row) + "\n")
                per_task[task_id] += 1
            f.flush()
            print("ep %d %s driver=%s rows=%d need=%s (%.0fs)"
                  % (ep, task_id[:40], driver, len(rows), need, time.time() - t0), flush=True)
            ep += 1
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
