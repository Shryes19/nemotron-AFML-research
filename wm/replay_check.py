"""Spot-check collected rows: replay action_prefix from reset(task_id) and compare with o_t.

  python -m wm.replay_check --env alfworld --data data/raw/alfworld_train.jsonl --n 10
"""
from __future__ import annotations

import argparse
import random
import sys

from wm.envs.base import make_env
from wm.obs import canonical
from wm.schema import read_jsonl


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rows = list(read_jsonl(args.data))
    env = make_env(args.env)
    bad = 0
    for row in random.Random(args.seed).sample(rows, min(args.n, len(rows))):
        r = env.replay(row["task_id"], row["action_prefix"])
        same = r.obs == row["o_t"] if env.exact_replay else canonical(r.obs) == canonical(row["o_t"])
        nxt = env.step(row["a_t"])
        same_next = nxt.feedback == row["feedback_next"]
        if not (same and same_next):
            bad += 1
            print("MISMATCH", row["transition_id"], "o_t ok" if same else "o_t differs",
                  "| next ok" if same_next else "| next differs")
    print("%d/%d rows replayed correctly" % (min(args.n, len(rows)) - bad, min(args.n, len(rows))))
    if hasattr(env, "close"):
        env.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
