"""Decision-utility scorer. See docs/DECISION_UTILITY.md.

1) True values (once per env): the judge scores every candidate's TRUE next state.
   python -m wm.utility.score true-values --states data/eval/alfworld_branched_states.jsonl \
       --out results/utility/alfworld_true_values.jsonl

2) Score one arm at one budget: the judge scores the arm's PREDICTED next state for each
   candidate, the arm "chooses" the highest, and regret = best true value - true value of its choice.
   python -m wm.utility.score score --states data/eval/alfworld_branched_states.jsonl \
       --true-values results/utility/alfworld_true_values.jsonl --pred outputs/jepa_candidates/alfworld_b256.jsonl \
       --format jepa --mode top1 --arm D --out results/utility/alfworld_D_top1_256.jsonl

3) Summarize all scored files (+ the oracle) into tables.
   python -m wm.utility.score summarize --states data/eval/*_branched_states.jsonl \
       --true-values results/utility/*_true_values.jsonl --scored results/utility/*_[GFD]*.jsonl

Oracle = choosing with the TRUE next states: regret 0 by construction; its gold-hit rate (does it
pick the expert action?) measures the judge, so arm-vs-oracle gaps isolate world-model error.
Use --judge overlap for dry runs without an LLM (not a real judge).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from wm.utility.judge import make_judge
from wm.utility.predictions import load_predictions


def track(stage, **kw):
    """GPU-hour logging (capability/compute_log.py) when available."""
    try:
        from capability.compute_log import track as _track
        return _track(stage, **kw)
    except ImportError:
        return contextlib.nullcontext()


def load_states(paths: Sequence[str]) -> Dict[str, Dict]:
    states = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    states[s["id"]] = s
    return states


def load_true_values(paths: Sequence[str]) -> Dict[str, List[Optional[int]]]:
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r["values"]
    return out


def family_of(state: Dict) -> str:
    return state["task_type"].split("-")[0] if state["env"] == "alfworld" else state["task_type"]


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".").strip()


def word_f1(pred: str, true: str) -> float:
    p, t = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", true.lower())
    if not p or not t:
        return float(p == t)
    common = sum(min(p.count(w), t.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(t)
    return 2 * prec * rec / (prec + rec)


def choose(values: Sequence[Optional[float]], rng: random.Random) -> Optional[int]:
    """Index of the highest value; ties broken by rng; None if nothing was judged."""
    valid = [(v, i) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    best = max(v for v, _ in valid)
    return rng.choice(sorted(i for v, i in valid if v == best))


def tie_rng(sid: str, seed: int = 0) -> random.Random:
    # same tie-break stream for every arm on a given state
    return random.Random("%d-%s" % (seed, sid))


def mean_ci(values: Sequence[float], seed: int = 0, reps: int = 1000):
    if not values:
        return None, None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(reps))
    return (round(sum(values) / len(values), 4), round(means[int(0.025 * reps)], 4),
            round(means[int(0.975 * reps) - 1], 4))


def score_state(state: Dict, true_values: List[Optional[int]], pred_values: Sequence[Optional[float]],
                pred_texts: Optional[List] = None, seed: int = 0) -> Optional[Dict]:
    """Regret, gold-hit and fidelity for one state. None if the true values are incomplete or the
    arm's predictions could not be judged at all."""
    if any(v is None for v in true_values):
        return None
    chosen = choose(pred_values, tie_rng(state["id"], seed))
    if chosen is None:
        return None
    rec = {"chosen": chosen, "regret": max(true_values) - true_values[chosen],
           "gold_hit": int(chosen == state["gold_index"])}
    if pred_texts is not None:
        tops = [p if isinstance(p, str) else (p[0] if p else "") for p in pred_texts]
        rec["em"] = sum(normalize(p) == normalize(t) for p, t in zip(tops, state["true_next"])) / len(tops)
        rec["f1"] = sum(word_f1(p, t) for p, t in zip(tops, state["true_next"])) / len(tops)
    return rec


# ---------------------------------------------------------------- subcommands

def cmd_true_values(args):
    states = load_states(args.states)
    done = set(load_true_values([args.out])) if os.path.exists(args.out) else set()
    todo = [s for s in states.values() if s["id"] not in done]
    judge = make_judge(args.judge)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env = next(iter(states.values()))["env"] if states else ""
    print("true values: %d states to judge (%d done)" % (len(todo), len(done)), flush=True)
    with track("utility_scoring", arm="true", env=env, n=len(todo)), open(args.out, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), 20):
            chunk = todo[start:start + 20]
            items = [(s["goal"], s["obs"], a, t) for s in chunk for a, t in zip(s["actions"], s["true_next"])]
            vals = judge.values(items, workers=args.workers)
            k = 0
            for s in chunk:
                v = vals[k:k + len(s["actions"])]
                k += len(s["actions"])
                f.write(json.dumps({"id": s["id"], "judge": judge.name, "model": judge.model, "values": v}) + "\n")
            f.flush()
    return 0


def cmd_score(args):
    states = load_states(args.states)
    tv = load_true_values(args.true_values)
    preds = load_predictions(args.pred, states, fmt=args.format, mode=args.mode, budget=args.budget,
                             allow_missing=args.allow_missing)
    judge = None if args.mode == "native" else make_judge(args.judge)
    rows, skipped = [], 0
    ids = [sid for sid in states if sid in preds and sid in tv]
    env = states[ids[0]]["env"] if ids else ""
    budget = args.budget if args.budget is not None else (preds[ids[0]]["budget"] if ids else None)
    with track("utility_scoring", arm=args.arm, env=env, n=budget if budget is not None else ""):
        if args.mode == "native":
            pred_values = {}
            for sid in ids:
                rank = preds[sid]["ranking"]
                if rank is None:
                    raise SystemExit("--mode native needs JEPA output with a 'ranking' (give states a goal)")
                k = len(states[sid]["actions"])
                vals = [None] * k
                for pos, idx in enumerate(rank):
                    vals[idx] = k - pos          # first in the ranking gets the highest value
                pred_values[sid] = vals
        else:
            items = [(states[sid]["goal"], states[sid]["obs"], a, p)
                     for sid in ids for a, p in zip(states[sid]["actions"], preds[sid]["pred"])]
            flat = judge.values(items, workers=args.workers)
            pred_values, k = {}, 0
            for sid in ids:
                n = len(states[sid]["actions"])
                pred_values[sid] = flat[k:k + n]
                k += n
        for sid in ids:
            s = states[sid]
            texts = None if args.mode == "native" else preds[sid]["pred"]
            rec = score_state(s, tv[sid], pred_values[sid], texts, args.seed)
            if rec is None:
                skipped += 1
                continue
            rec.update({"id": sid, "env": s["env"], "arm": args.arm, "mode": args.mode,
                        "budget": preds[sid]["budget"], "game": s["game"], "family": family_of(s),
                        "gold_index": s["gold_index"], "pred_values": pred_values[sid],
                        "true_values": tv[sid], "judge": judge.name if judge else "none"})
            rows.append(rec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if rows:
        print("%s %s budget=%s: %d states, mean regret %.3f, gold-hit %.3f (%d skipped)"
              % (args.arm, args.mode, budget, len(rows), sum(r["regret"] for r in rows) / len(rows),
                 sum(r["gold_hit"] for r in rows) / len(rows), skipped))
    else:
        print("no states scored (%d skipped)" % skipped)
    return 0


def oracle_rows(states: Dict, tv: Dict, seed: int = 0) -> List[Dict]:
    rows = []
    for sid, s in states.items():
        if sid not in tv:
            continue
        rec = score_state(s, tv[sid], tv[sid], s["true_next"], seed)
        if rec:
            rec.update({"id": sid, "env": s["env"], "arm": "oracle", "mode": "true", "budget": "-",
                        "game": s["game"], "family": family_of(s)})
            rows.append(rec)
    return rows


def summarize_rows(rows: List[Dict], seed: int = 0) -> List[Dict]:
    groups = defaultdict(list)
    for r in rows:
        base = (r["env"], r["arm"], r.get("mode", ""), str(r["budget"]))
        groups[base + (r["game"], r["family"])].append(r)
        groups[base + ("*", "*")].append(r)
    out = []
    for (env, arm, mode, budget, game, family), rs in sorted(groups.items()):
        m, lo, hi = mean_ci([r["regret"] for r in rs], seed)
        row = {"env": env, "arm": arm, "mode": mode, "budget": budget, "game": game, "family": family,
               "n_states": len(rs), "mean_regret": m, "regret_ci_lo": lo, "regret_ci_hi": hi,
               "gold_hit": round(sum(r["gold_hit"] for r in rs) / len(rs), 4)}
        for key in ("em", "f1"):
            vals = [r[key] for r in rs if key in r]
            row[key] = round(sum(vals) / len(vals), 4) if vals else ""
        out.append(row)
    return out


def disagreement(summary: List[Dict]) -> List[Dict]:
    """Per (env, budget): best arm by utility (lowest regret) vs best by fidelity (highest EM)."""
    by = defaultdict(list)
    for r in summary:
        if r["game"] == "*" and r["arm"] != "oracle" and r["em"] != "":
            by[(r["env"], r["budget"])].append(r)
    out = []
    for (env, budget), rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        u = min(rs, key=lambda r: (r["mean_regret"], r["arm"]))
        fi = max(rs, key=lambda r: (r["em"], r["f1"]))
        out.append({"env": env, "budget": budget, "best_by_utility": "%s/%s" % (u["arm"], u["mode"]),
                    "best_by_fidelity": "%s/%s" % (fi["arm"], fi["mode"]), "agree": u is fi})
    return out


def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_summarize(args):
    states = load_states(args.states)
    tv = load_true_values(args.true_values)
    rows = oracle_rows(states, tv, args.seed)
    for pattern in args.scored:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    summary = summarize_rows(rows, args.seed)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    write_csv(os.path.join(args.out_dir, "fidelity_vs_utility.csv"), disagreement(summary))
    for r in summary:
        if r["game"] == "*":
            print("%-9s %-7s %-7s %-6s n=%-5s regret %-6s [%s, %s] gold-hit %s EM %s"
                  % (r["env"], r["arm"], r["mode"], r["budget"], r["n_states"], r["mean_regret"],
                     r["regret_ci_lo"], r["regret_ci_hi"], r["gold_hit"], r["em"]))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("true-values")
    t.add_argument("--states", nargs="+", required=True)
    t.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--states", nargs="+", required=True)
    s.add_argument("--true-values", nargs="+", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--arm", required=True, help="G, F, D, ...")
    s.add_argument("--format", choices=["generic", "jepa"], default="generic")
    s.add_argument("--mode", choices=["top1", "topk", "native"], default="top1")
    s.add_argument("--budget", type=int, default=None, help="keep only predictions for this budget")
    s.add_argument("--allow-missing", action="store_true")
    s.add_argument("--out", required=True)
    m = sub.add_parser("summarize")
    m.add_argument("--states", nargs="+", required=True)
    m.add_argument("--true-values", nargs="+", required=True)
    m.add_argument("--scored", nargs="*", default=[])
    m.add_argument("--out-dir", default="results/utility")
    for q in (t, s, m):
        q.add_argument("--judge", choices=["llm", "overlap"], default="llm")
        q.add_argument("--workers", type=int, default=8)
        q.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    return {"true-values": cmd_true_values, "score": cmd_score, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
