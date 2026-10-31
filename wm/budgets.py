"""Nested, stratified data budgets.

For each seed: shuffle each policy's train transitions with Random(seed), then budget n takes
the first round(0.5n) expert, round(0.3n) random and the remaining llm ids. Every budget is a
prefix of the same per-policy shuffles, so 32 ⊂ 64 ⊂ ... ⊂ 1024 for a given seed.

Without LLM rows yet, --provisional uses expert/random only (5:3) and writes *_provisional files.

  python -m wm.budgets --env alfworld --data data/raw/alfworld_train.jsonl.gz [--provisional]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import yaml

from wm.schema import read_jsonl

CONFIG = "configs/shared.yaml"


def allocate(n: int, mix: Dict[str, float]) -> Dict[str, int]:
    """Rows per policy for budget n; the last policy takes the remainder so counts sum to n."""
    names = list(mix)
    counts = {k: int(round(n * mix[k] / sum(mix.values()))) for k in names[:-1]}
    counts[names[-1]] = n - sum(counts.values())
    return counts


def make_budgets(ids_by_policy: Dict[str, List[str]], budgets: List[int], seed: int,
                 mix: Dict[str, float]) -> Dict[int, List[str]]:
    shuffled = {}
    for policy in mix:
        ids = sorted(ids_by_policy.get(policy, []))
        random.Random(seed).shuffle(ids)
        shuffled[policy] = ids
    out = {}
    for n in budgets:
        counts = allocate(n, mix)
        chosen = []
        for policy, k in counts.items():
            if k > len(shuffled[policy]):
                raise ValueError("budget n=%d needs %d %s rows, only %d collected"
                                 % (n, k, policy, len(shuffled[policy])))
            chosen += shuffled[policy][:k]
        out[n] = chosen
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--data", required=True, help="train transitions (.jsonl or .jsonl.gz)")
    p.add_argument("--out-dir", default="data/budgets")
    p.add_argument("--provisional", action="store_true", help="expert/random only, before LLM rows exist")
    args = p.parse_args(argv)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    mix = dict(cfg["policy_mix"])
    if args.provisional:
        mix.pop("llm")
    suffix = "_provisional" if args.provisional else ""

    ids_by_policy: Dict[str, List[str]] = {}
    for row in read_jsonl(args.data):
        if row["split"] != "train" or row["env"] != args.env:
            continue
        ids_by_policy.setdefault(row["policy"], []).append(row["transition_id"])  # llm_fallback excluded

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"env": args.env, "data": os.path.basename(args.data), "mix": mix,
                "pool": {k: len(v) for k, v in ids_by_policy.items()}, "files": {}}
    for seed in cfg["seeds"]:
        for n, ids in make_budgets(ids_by_policy, cfg["budgets"], seed, mix).items():
            name = "%s_seed%d_n%d%s.txt" % (args.env, seed, n, suffix)
            with open(os.path.join(args.out_dir, name), "w") as f:
                f.write("\n".join(ids) + "\n")
            manifest["files"][name] = allocate(n, mix)
    with open(os.path.join(args.out_dir, "%s_manifest%s.json" % (args.env, suffix)), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote %d budget files to %s; pool %s" % (len(manifest["files"]), args.out_dir, manifest["pool"]))


if __name__ == "__main__":
    main()
