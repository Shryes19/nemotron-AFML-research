"""Export collected transitions to the JEPA/SFT buffer format (jepa/training/buffer.py).

Per-game buffer, one file (what the JEPA probe scripts use):

  python -m wm.export_buffers --env alfworld --train data/raw/alfworld_games.jsonl \
      --single data/buffers/alfworld_games.jsonl

Per-budget buffers plus the task-disjoint held-out set:

  python -m wm.export_buffers --env alfworld \
      --train data/raw/alfworld_train.jsonl.gz --heldout data/raw/alfworld_heldout.jsonl.gz [--provisional]

  -> data/buffers/{env}_seed{s}_n{n}[_provisional].jsonl and data/buffers/{env}_heldout.jsonl

Field mapping (buffer <- schema):
  obs      <- o_t (composite observation: goal, room, location, feedback, inventory, admissible)
  action   <- a_t
  obs_next <- feedback_next: exactly the text the env returns after the action
              (--obs-next composite uses o_next instead; the other one is always kept too)
  env      <- env
  game     <- task_id, the game/task instance (--game family groups by task type instead)
  cap_tag  <- first capability tag, or "" while untagged (cap_tags keeps the full list)
Extra fields kept for traceability: transition_id, task_id, task_type, split, policy.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import yaml

from wm.schema import read_jsonl

CONFIG = "configs/shared.yaml"


def family_of(row: Dict) -> str:
    if row["env"] == "alfworld":
        return row["task_type"].split("-")[0]
    return row["task_type"]


def to_buffer_row(row: Dict, game: str = "task_id", obs_next: str = "feedback") -> Dict:
    tags = row.get("capability_tag") or []
    return {
        "obs": row["o_t"],
        "action": row["a_t"],
        "obs_next": row["feedback_next"] if obs_next == "feedback" else row["o_next"],
        "env": row["env"],
        "game": row["task_id"] if game == "task_id" else family_of(row),
        "cap_tag": tags[0] if tags else "",
        "cap_tags": list(tags),
        "obs_next_full": row["o_next"],
        "feedback_next": row["feedback_next"],
        "transition_id": row["transition_id"],
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "split": row["split"],
        "policy": row["policy"],
    }


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--heldout", default=None)
    p.add_argument("--single", default=None, help="write all --train rows to this one file")
    p.add_argument("--budget-dir", default="data/budgets")
    p.add_argument("--out-dir", default="data/buffers")
    p.add_argument("--provisional", action="store_true")
    p.add_argument("--game", choices=["task_id", "family"], default="task_id")
    p.add_argument("--obs-next", choices=["feedback", "composite"], default="feedback")
    args = p.parse_args(argv)
    conv = lambda r: to_buffer_row(r, args.game, args.obs_next)

    if args.single:
        rows = [conv(r) for r in read_jsonl(args.train) if r["env"] == args.env]
        write_jsonl(args.single, rows)
        print("wrote %d rows (%d games) to %s" % (len(rows), len({r["game"] for r in rows}), args.single))
        return

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    suffix = "_provisional" if args.provisional else ""
    by_id = {r["transition_id"]: r for r in read_jsonl(args.train)}
    for seed in cfg["seeds"]:
        for n in cfg["budgets"]:
            name = "%s_seed%d_n%d%s" % (args.env, seed, n, suffix)
            with open(os.path.join(args.budget_dir, name + ".txt")) as f:
                ids = [line.strip() for line in f if line.strip()]
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise SystemExit("%s: %d ids not in %s (e.g. %s)" % (name, len(missing), args.train, missing[0]))
            write_jsonl(os.path.join(args.out_dir, name + ".jsonl"), [conv(by_id[i]) for i in ids])
    msg = "wrote %d budget buffers" % (len(cfg["seeds"]) * len(cfg["budgets"]))
    if args.heldout:
        held = [conv(r) for r in read_jsonl(args.heldout) if r["split"] == "heldout"]
        write_jsonl(os.path.join(args.out_dir, "%s_heldout.jsonl" % args.env), held)
        msg += " + %s_heldout.jsonl (%d rows)" % (args.env, len(held))
    print(msg + " to " + args.out_dir)


if __name__ == "__main__":
    main()
