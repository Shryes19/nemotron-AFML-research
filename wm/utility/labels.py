"""Winner labels for the RQ3 feature matrix: which arm has the lowest decision regret per game.

  python -m wm.utility.labels --scored "results/utility/*_[GFD]_*.jsonl" --budget 256 \
      --arms G,F,D --key game --out results/utility/labels.csv

Writes columns env, game, label (read by scripts/build_feature_matrix.py --labels), plus the
per-arm mean regret and state count behind each label. Ties in mean regret go to the higher
gold-hit rate, then to the order given in --arms; column `tie` lists the tied arms (e.g. "G+F"),
empty when the winner's regret is strictly lowest. Count tied labels separately when reporting.

--key game uses the branched states' game (ALFWorld task id, ScienceWorld task type);
--key family groups ALFWorld by task type instead. Held-out ALFWorld games are not the 12
training games, so ALFWorld labels only join the feature matrix with --key family (team decision).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from typing import Dict, List

from wm.utility.score import write_csv


def make_labels(rows: List[Dict], arms: List[str], budget, key: str = "game", mode: str = "top1",
                arm_modes: Dict[str, str] = None) -> List[Dict]:
    """arm_modes overrides the scoring mode per arm, e.g. {"D": "native"}."""
    arm_modes = arm_modes or {}
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] in arms and str(r["budget"]) == str(budget) and                 r.get("mode", "top1") == arm_modes.get(r["arm"], mode):
            acc[(r["env"], r[key])][r["arm"]].append(r)
    out = []
    for (env, game), by_arm in sorted(acc.items()):
        stats = {}
        for arm, rs in by_arm.items():
            stats[arm] = (sum(r["regret"] for r in rs) / len(rs), sum(r["gold_hit"] for r in rs) / len(rs), len(rs))
        present = [a for a in arms if a in stats]
        winner = min(present, key=lambda a: (stats[a][0], -stats[a][1], arms.index(a)))
        best = round(stats[winner][0], 4)
        tied = [a for a in present if round(stats[a][0], 4) == best]
        # tie: other arms share the winner's mean regret (the label then came from gold-hit / arm order)
        row = {"env": env, "game": game, "label": winner, "tie": "+".join(tied) if len(tied) > 1 else ""}
        for a in arms:
            row["regret_%s" % a] = round(stats[a][0], 4) if a in stats else ""
            row["n_%s" % a] = stats[a][2] if a in stats else 0
        out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scored", nargs="+", required=True, help="scored files or glob patterns")
    p.add_argument("--budget", default="256")
    p.add_argument("--arms", default="G,F,D")
    p.add_argument("--mode", default="top1", help="which scoring mode labels are based on")
    p.add_argument("--arm-mode", action="append", default=[], metavar="ARM=MODE",
                   help="scoring mode for one arm, e.g. D=native (others use --mode)")
    p.add_argument("--key", choices=["game", "family"], default="game")
    p.add_argument("--out", default="results/utility/labels.csv")
    args = p.parse_args(argv)
    rows = []
    for pattern in args.scored:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    arm_modes = dict(x.split("=", 1) for x in args.arm_mode)
    labels = make_labels(rows, args.arms.split(","), args.budget, args.key, args.mode, arm_modes)
    write_csv(args.out, labels)
    print("wrote %d labels to %s" % (len(labels), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
