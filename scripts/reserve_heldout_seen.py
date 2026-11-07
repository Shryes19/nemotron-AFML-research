"""Reserve held-out variations of SEEN task types for decision utility (ScienceWorld).

Word2World's ScienceWorld test split covers only 12 task types, so the other types in the RQ3
feature matrix have no held-out states. This picks, per task type, up to --per-type train-split
task ids that no training data file has used, and writes them to splits/sciworld_heldout_seen.txt.
wm.collect then treats them as split "heldout_seen" and removes them from "train", so later
collection (e.g. the LLM rows) can never train on them.

  ~/w2w/uv_sciworld/bin/python scripts/reserve_heldout_seen.py \
      --types-from data/raw/sciworld_games.jsonl \
      --exclude data/raw/sciworld_games.jsonl data/raw/sciworld_train.jsonl
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wm.collect import load_task_ids  # noqa: E402
from wm.schema import read_jsonl  # noqa: E402

OUT = os.path.join("splits", "sciworld_heldout_seen.txt")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--types-from", required=True, help="data file whose task types are the matrix games")
    p.add_argument("--exclude", nargs="+", required=True, help="data files whose task ids must not be used")
    p.add_argument("--per-type", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if os.path.exists(OUT):
        raise SystemExit("%s exists; reserved ids must not change once data was collected" % OUT)

    from wm.envs.sciworld_env import SciWorldEnv
    env = SciWorldEnv()
    try:
        types = sorted({r["task_type"] for r in read_jsonl(args.types_from)})
        used = {r["task_id"] for path in args.exclude for r in read_jsonl(path)}
        by_type = defaultdict(list)
        for t in sorted(load_task_ids("sciworld", "train")):
            if t not in used:
                by_type[env.task_type_of(t)].append(t)
        chosen = []
        for task_type in types:
            pool = by_type.get(task_type, [])
            rng = random.Random("%d-%s" % (args.seed, task_type))
            pick = rng.sample(pool, min(args.per_type, len(pool)))
            chosen += sorted(pick)
            print("%-48s %3d free train variations, reserved %d" % (task_type, len(pool), len(pick)))
    finally:
        env.close()
    with open(OUT, "w", newline="\n") as f:
        f.write("\n".join(chosen) + "\n")
    print("wrote %d task ids for %d task types to %s" % (len(chosen), len(types), OUT))
    os._exit(0)


if __name__ == "__main__":
    main()
