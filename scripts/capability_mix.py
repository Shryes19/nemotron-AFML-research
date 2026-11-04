#!/usr/bin/env python3
"""
Ex-ante Feature 2: capability mix vector per game.

    python scripts/capability_mix.py --buffers data/buffers/alfworld_games.tagged.jsonl \
        data/buffers/sciworld_train.tagged.jsonl data/buffers/bytesized32_games.jsonl \
        --out results/capability/capability_mix.csv

One row per (env, game, n). Columns:
    rate_<tag>      share of transitions carrying the tag (multi-label, so rows need not sum to 1)
    primary_<tag>   share of transitions whose primary tag is <tag> (sums to 1, includes primary_none)
For n=32 the vector is the mean over --seeds random draws of 32 transitions from that game (sd columns
show how stable it is); n=all uses every transition. Feature 2 for the RQ3 predictor = the five rate_*
columns at n=32. The game unit: `game` field if present, else task_type (--game-key overrides).
"""
import argparse
import collections
import csv
import gzip
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from capability.tagger import TAGS, tag_record  # noqa: E402

PRIMARY = ["none"] + TAGS


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def tags_of(rec, use_text):
    if use_text and "text_cap_tags" in rec:
        return rec["text_cap_tags"], rec["text_cap_tag"]
    if rec.get("cap_tags") is not None and rec.get("cap_tag"):
        return rec["cap_tags"], rec["cap_tag"]
    t = tag_record(rec)                     # untagged buffer: tag on the fly
    return t["tags"], t["primary"]


def vector(sample, use_text):
    n = len(sample)
    rate, prim = collections.Counter(), collections.Counter()
    for r in sample:
        tags, p = tags_of(r, use_text)
        rate.update(tags)
        prim[p] += 1
    return {**{f"rate_{t}": rate[t] / n for t in TAGS}, **{f"primary_{t}": prim[t] / n for t in PRIMARY}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffers", nargs="+", required=True)
    ap.add_argument("--out", default="results/capability/capability_mix.csv")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--game-key", default=None, help="field that defines a game (default: game, else task_type)")
    ap.add_argument("--text-tags", action="store_true",
                    help="for ByteSized32 use the text tagger's tags instead of the JSON ground truth")
    args = ap.parse_args()

    rows = []
    for path in args.buffers:
        recs = load(path)
        key = args.game_key or ("game" if "game" in recs[0] else "task_type")
        games = collections.defaultdict(list)
        for r in recs:
            games[str(r.get(key))].append(r)
        env = recs[0].get("env", os.path.basename(path))
        for game, rs in sorted(games.items()):
            full = vector(rs, args.text_tags)
            rows.append({"env": env, "game": game, "n": "all", "transitions": len(rs), **full})
            if len(rs) < args.n:
                continue
            draws = [vector(random.Random(s).sample(rs, args.n), args.text_tags) for s in range(args.seeds)]
            mean = {k: st.mean(d[k] for d in draws) for k in full}
            sd = {f"sd_{k}": st.pstdev(d[k] for d in draws) for k in full if k.startswith("rate_")}
            rows.append({"env": env, "game": game, "n": args.n, "transitions": len(rs), **mean, **sd})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    feat = [r for r in rows if r["n"] == args.n]
    print(f"wrote {len(rows)} rows ({len(feat)} games with >= {args.n} transitions) -> {args.out}")
    print(f"{'env':12s} {'game':58s} " + " ".join(f"{t[:9]:>9s}" for t in TAGS))
    for r in feat:
        print(f"{r['env'][:12]:12s} {r['game'][:58]:58s} " + " ".join(f"{r['rate_' + t]:9.2f}" for t in TAGS))


if __name__ == "__main__":
    main()
