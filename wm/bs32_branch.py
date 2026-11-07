"""ByteSized32 branched states from the GPT-simulator crawl (no environment replay needed).

The crawl (train.jsonl) tries ~150 actions from each state on every game's gold playthrough and
records the next state, so a branched state is one crawled state + K of its recorded actions.

Regime (report separately): these are the SAME states the arms trained on (test.jsonl holds 310 of
the crawl's 337 states); only the candidate actions are new. Distractors are actions whose
(state, action) pair is not in the training data; the gold action is kept even if it is (flagged
in gold_seen_in_training). So this measures "known state, new action", unlike ALFWorld (new games)
and ScienceWorld (new variations).

Gold action: walk each game's gold playthrough from its start state through the recorded
transitions; the playthrough's i-th action is the gold action at the i-th state on that chain.
States off the chain or at its end are skipped.

  python -m wm.bs32_branch --crawl ~/ext/bs32data/train.jsonl --train ~/ext/bs32data/test.jsonl \
      --playthroughs ~/ext/GPT-simulator/data/playthroughs --out data/eval/bytesized32_branched_states.jsonl

Fields match docs/SCHEMA.md "Branched states", plus true_score (score after each action) and
gold_seen_in_training. obs / true_next use capability/bytesized32.py's rendering (the arms' format).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
from typing import Dict, List, Optional

from wm.policies import sample_by_verb

NOOP = {"look", "look around", "inventory", "help", "task"}


def render(state: Dict) -> str:
    """Same layout as capability/bytesized32.py::_render."""
    return ("[GOAL] %s\n[ROOM] %s\n[FEEDBACK] %s\n[INVENTORY] %s" % (
        state.get("taskDesc", "").strip(), state.get("look", "").strip(),
        state.get("observation", "").strip(), state.get("inventory", "").strip()))


def state_key(state: Dict) -> str:
    """Identity of a game state; lastAction is ignored (the same state can be reached differently)."""
    blob = json.dumps({k: state[k] for k in sorted(state) if k != "lastAction"}, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def read_playthrough(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line[2:].strip() for line in f if line.startswith("> ")]


class Crawl:
    """(game, state) -> rendered obs, goal, and per-action outcome."""

    def __init__(self):
        self.obs, self.goal, self.last = {}, {}, {}
        self.out = collections.defaultdict(dict)   # (game, key) -> {action: outcome}

    def add(self, rec: Dict):
        g, cur, tick = rec["game"], rec["current_state"], rec["tick_state"]
        k = (g, state_key(cur))
        if k not in self.obs:
            self.obs[k], self.goal[k] = render(cur), cur.get("taskDesc", "").strip()
            self.last[k] = cur.get("lastAction", "") or ""
        a = rec["action_state"].get("lastAction", "")
        cs, ns = rec.get("current_score_state", {}), rec.get("next_score_state", {})
        self.out[k][a] = {"next_key": state_key(tick), "feedback": tick.get("observation", "").strip(),
                          "full": render(tick), "score": ns.get("score"),
                          "reward": (ns.get("score") or 0) - (cs.get("score") or 0),
                          "won": bool(ns.get("gameWon")), "done": bool(ns.get("gameOver"))}


def gold_chain(crawl: Crawl, game: str, gold_actions: List[str]) -> Dict[str, str]:
    """Map state key -> gold action by walking the playthrough through recorded transitions."""
    starts = [k for k in crawl.obs if k[0] == game and crawl.last[k] in ("", "None")]
    best = {}
    for start in starts:
        chain, k = {}, start
        for a in gold_actions:
            if k not in crawl.out or a not in crawl.out[k]:
                break
            chain[k[1]] = a
            k = (game, crawl.out[k][a]["next_key"])
        if len(chain) > len(best):
            best = chain
    return best


def choose_actions(gold: str, actions: List[str], trained: set, k: int, rng: random.Random) -> List[str]:
    pool = [a for a in actions if a != gold and a.split(" (")[0].strip().lower() not in NOOP
            and a.strip().lower() not in NOOP and a not in trained]
    chosen = [gold]
    while len(chosen) < k and pool:
        a = sample_by_verb(pool, rng)
        pool.remove(a)
        chosen.append(a)
    rng.shuffle(chosen)
    return chosen


def build(crawl: Crawl, trained_pairs: set, playthroughs: Dict[str, List[str]], k: int = 5,
          seed: int = 0) -> (List[Dict], collections.Counter):
    states, skipped = [], collections.Counter()
    games = sorted({g for g, _ in crawl.obs})
    for game in games:
        chain = gold_chain(crawl, game, playthroughs.get(game, []))
        keys = sorted(h for g, h in crawl.obs if g == game)
        for idx, h in enumerate(keys):
            key = (game, h)
            gold = chain.get(h)
            if gold is None:
                skipped["no_gold"] += 1
                continue
            trained = {a for (g, s, a) in trained_pairs if g == game and s == h}
            rng = random.Random("%d-%s-%s" % (seed, game, h))
            actions = choose_actions(gold, sorted(crawl.out[key]), trained, k, rng)
            if len(actions) < k:
                skipped["small_pool"] += 1
                continue
            outs = [crawl.out[key][a] for a in actions]
            gi = actions.index(gold)
            states.append({
                "id": "bytesized32/%s/%s" % (game, h[:12]), "obs": crawl.obs[key], "actions": actions,
                "game": game, "goal": crawl.goal[key], "gold_action": gold, "gold_index": gi,
                "gold_next": outs[gi]["feedback"], "gold_seen_in_training": gold in trained,
                "true_next": [o["feedback"] for o in outs], "true_next_full": [o["full"] for o in outs],
                "true_reward": [o["reward"] for o in outs], "true_score": [o["score"] for o in outs],
                "true_done": [o["done"] for o in outs], "true_won": [o["won"] for o in outs],
                "replay_consistent": True, "replay_checked": False,
                "transition_id": "bs32-%s-%s" % (game, h[:12]), "task_id": game, "task_type": game,
                "action_prefix": [], "source_policy": "bytesized32_crawl", "env": "bytesized32",
                "split": "seen_state_new_actions"})
    return states, skipped


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--crawl", required=True, help="GPT-simulator train.jsonl")
    p.add_argument("--train", required=True, help="the ByteSized32 file the arms trained on (test.jsonl)")
    p.add_argument("--playthroughs", required=True, help="GPT-simulator data/playthroughs folder")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    crawl = Crawl()
    for path in (args.crawl, args.train):       # training transitions also extend the gold chain
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    crawl.add(json.loads(line))
    trained = set()
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                trained.add((r["game"], state_key(r["current_state"]), r["action_state"].get("lastAction", "")))
    playthroughs = {}
    for name in os.listdir(args.playthroughs):
        if name.endswith("-playthrough.txt"):
            playthroughs[re.sub(r"-playthrough\.txt$", "", name)] = read_playthrough(os.path.join(args.playthroughs, name))
    states, skipped = build(crawl, trained, playthroughs, args.k, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")
    per_game = collections.Counter(s["game"] for s in states)
    print("wrote %d states for %d games; skipped %s" % (len(states), len(per_game), dict(skipped)))
    print("states per game: %s" % dict(sorted(per_game.items())))
    print("gold transition seen in training: %d of %d" % (sum(s["gold_seen_in_training"] for s in states), len(states)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
