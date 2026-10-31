#!/usr/bin/env python3
"""
Ex-ante Feature 4: cheap environment statistics per game (no model calls).

    python scripts/env_stats.py --buffers data/buffers/alfworld_games.jsonl \
        data/buffers/sciworld_train.jsonl.gz data/buffers/bytesized32_games.jsonl \
        --out results/features/env_stats.csv

Per (env, game), at n=32 (mean over --seeds random draws) and n=all:
  obs_len_chars        mean characters of o_t (full observation the world model reads)
  target_len_chars     mean characters of the text the world model must predict (obs_next / feedback)
  action_vocab         unique action strings among the n transitions
  action_verbs         unique action verbs (first word)
  zero_overlap_frac    share of transitions where o_t and the target share no content word
                       (the plan's statistic (c); "content word" = 3+ letters, stopwords removed)
  mean_jaccard         mean word Jaccard between o_t and the target (finer version of (c))
  target_repeat_frac   share of targets that are an exact repeat of another target in the sample
                       (how "templated" the env is; this is also what caps JEPA's NN-decode reach)
The target is the field the arms predict: obs_next in the JEPA buffer format, feedback_next / o_next
otherwise.
"""
import argparse
import collections
import csv
import gzip
import json
import os
import random
import re
import statistics as st

STOP = set("the and you are with from into onto your have that this there here what which then they "
           "them their these those were been was has had not but for can its it's see".split())
WORD = re.compile(r"[a-z]{3,}")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def fields(r):
    o_t = r.get("obs", r.get("o_t", ""))
    a = r.get("action", r.get("a_t", ""))
    tgt = r.get("obs_next") if "obs" in r else (r.get("feedback_next") or r.get("o_next", ""))
    return o_t or "", a or "", tgt or ""


def words(t):
    return set(WORD.findall(t.lower())) - STOP


def stats(sample):
    o_len, t_len, acts, verbs, zero, jac, targets = [], [], set(), set(), 0, [], collections.Counter()
    for r in sample:
        o, a, t = fields(r)
        o_len.append(len(o))
        t_len.append(len(t))
        acts.add(a.strip().lower())
        verbs.add(a.strip().lower().split(" ")[0] if a.strip() else "")
        wo, wt = words(o), words(t)
        zero += not (wo & wt)
        jac.append(len(wo & wt) / len(wo | wt) if (wo | wt) else 1.0)
        targets[t.strip()] += 1
    n = len(sample)
    return {"obs_len_chars": st.mean(o_len), "target_len_chars": st.mean(t_len),
            "action_vocab": len(acts), "action_verbs": len(verbs),
            "zero_overlap_frac": zero / n, "mean_jaccard": st.mean(jac),
            "target_repeat_frac": sum(c for c in targets.values() if c > 1) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffers", nargs="+", required=True)
    ap.add_argument("--out", default="results/features/env_stats.csv")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--game-key", default=None)
    args = ap.parse_args()

    rows = []
    for path in args.buffers:
        recs = load(path)
        key = args.game_key or ("game" if "game" in recs[0] else "task_type")
        env = recs[0].get("env", os.path.basename(path))
        games = collections.defaultdict(list)
        for r in recs:
            games[str(r.get(key))].append(r)
        for g, rs in sorted(games.items()):
            rows.append({"env": env, "game": g, "n": "all", "transitions": len(rs), **stats(rs)})
            if len(rs) < args.n:
                continue
            draws = [stats(random.Random(s).sample(rs, args.n)) for s in range(args.seeds)]
            rows.append({"env": env, "game": g, "n": args.n, "transitions": len(rs),
                         **{k: st.mean(d[k] for d in draws) for k in draws[0]}})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} rows -> {args.out}")
    by_env = collections.defaultdict(list)
    for r in rows:
        if r["n"] == args.n:
            by_env[r["env"]].append(r)
    for env, rs in by_env.items():
        print(f"{env:12s} games={len(rs):3d} " + "  ".join(
            f"{k}={st.mean(r[k] for r in rs):.2f}" for k in
            ["obs_len_chars", "action_vocab", "zero_overlap_frac", "mean_jaccard", "target_repeat_frac"]))


if __name__ == "__main__":
    main()
