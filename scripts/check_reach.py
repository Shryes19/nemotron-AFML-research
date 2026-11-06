"""
scripts/check_reach.py
──────────────────────
How often can JEPA's text answer be exactly right, before running anything on the GPU?

JEPA turns a prediction into text by picking the nearest next state from its
training buffer. So for a branched state, top-1 decode can only be word-for-word
right if that true next state appears somewhere in the training buffer. This
script counts that ("reach") for the gold action and for all 5 candidates.
CPU only, no encoder, a few seconds.

    python scripts/check_reach.py --states data/eval/alfworld_branched_states.jsonl \
        --buffer data/buffers/alfworld_games.jsonl

Reach is an upper bound on JEPA's exact match; it does not cap decision utility
(the judge only needs the predicted outcome to be about as useful as the true one),
but a low value means top-1 texts will often name the wrong objects.
"""

import argparse
import json
import re
from collections import defaultdict


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", required=True)
    p.add_argument("--buffer", required=True, help="the JEPA training buffer")
    p.add_argument("--group", default="family",
                   help="break down by: family (ALFWorld task family / ScienceWorld task type) or none")
    args = p.parse_args()

    buf = {norm(t["obs_next"]) for t in read_jsonl(args.buffer)}
    states = read_jsonl(args.states)

    def group(s):
        if args.group == "none":
            return "all"
        t = s.get("task_type") or s.get("game", "")
        return t.split("-")[0] if s.get("env") == "alfworld" else t

    rows = defaultdict(lambda: [0, 0, 0, 0])   # states, gold reachable, cands, cands reachable
    for s in states:
        for key in (group(s), "ALL"):
            r = rows[key]
            r[0] += 1
            r[1] += norm(s["gold_next"]) in buf
            r[2] += len(s["true_next"])
            r[3] += sum(norm(t) in buf for t in s["true_next"])

    print(f"buffer: {args.buffer}  ({len(buf)} distinct next states)")
    print(f"states: {args.states}")
    print(f"{'group':<44}{'states':>7}{'gold reach':>12}{'all-cand reach':>16}")
    for key in sorted(k for k in rows if k != "ALL") + ["ALL"]:
        n, g, c, cr = rows[key]
        print(f"{key:<44}{n:>7}{g / n:>11.1%}{cr / c:>15.1%}")


if __name__ == "__main__":
    main()
