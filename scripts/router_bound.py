#!/usr/bin/env python3
"""
RQ4 (optional RQ, answered as a bound): could choosing the world model per decision beat one arm for
everything, or one arm per capability?

    python scripts/router_bound.py --out-dir results/router

On the same branched states, at each budget, for arms G, F and D (top-1):
  best single arm        the arm with the lowest mean regret in that regime (chosen in-sample, so it is
                         slightly optimistic for the single arm, which makes the comparison conservative)
  capability router      per state, the arm with the lowest mean regret on the OTHER states whose best
                         candidate has the same primary capability (leave-one-state-out, so no state
                         chooses its own arm); ties go to the cheaper arm, order F, G, D
  oracle router          per state, the arm with the lowest regret on that very state: an upper bound
                         that no real router can beat, and that would need all three arms' predictions
The gap best-single minus oracle is the most a per-decision router could ever save. If it is small,
routing cannot help much here; if it is large, the capability router shows how much of it a simple,
realistic rule recovers. Each decision here is one step at a fixed state, so this offline comparison
is valid for this protocol (it would not be for multi-step agent rollouts).

Outputs: router.csv (regime x budget), router.md
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from capability.tagger import tag_transition  # noqa: E402

REGIMES = {"alfworld": "ALFWorld", "sciworld": "ScienceWorld test", "sciworld_seen": "ScienceWorld seen",
           "bytesized32": "ByteSized32 (gold unseen)"}
MAIN = ["alfworld", "sciworld", "sciworld_seen"]
ARMS = ["F", "G", "D"]                     # tie order = cheapest first (results/crossover)
BUDGETS = [32, 64, 128, 256, 512, 1024]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def best_index(values, gold):
    m = max(values)
    best = [i for i, v in enumerate(values) if v == m]
    return gold if gold in best else best[0]


def primary_caps(regime):
    tv = {r["id"]: r["values"] for r in load(f"results/utility/{regime}_true_values.jsonl")}
    out = {}
    for s in load(f"data/eval/{regime}_branched_states.jsonl"):
        if s["id"] in tv:
            i = best_index(tv[s["id"]], int(s["gold_index"]))
            full = s.get("true_next_full") or s["true_next"]
            out[s["id"]] = tag_transition(s["obs"], s["actions"][i], full[i], feedback=s["true_next"][i])["primary"]
    return out


def boot_ci(xs, reps=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def analyse(states):
    """states: list of (cap, {arm: regret}). Returns the four mean regrets + CI of single - oracle."""
    n = len(states)
    mean = {a: sum(s[1][a] for s in states) / n for a in ARMS}
    single = min(ARMS, key=lambda a: (mean[a], ARMS.index(a)))
    oracle = [min(r.values()) for _, r in states]
    by_cap = collections.defaultdict(lambda: collections.defaultdict(float))
    cnt = collections.Counter()
    for cap, r in states:
        cnt[cap] += 1
        for a in ARMS:
            by_cap[cap][a] += r[a]
    cap_router = []
    for cap, r in states:
        k = cnt[cap] - 1
        if k == 0:
            pick = single
        else:
            loo = {a: (by_cap[cap][a] - r[a]) / k for a in ARMS}
            pick = min(ARMS, key=lambda a: (loo[a], ARMS.index(a)))
        cap_router.append(r[pick])
    gap = [s[1][single] - o for s, o in zip(states, oracle)]
    gain = [s[1][single] - c for s, c in zip(states, cap_router)]
    disagree = sum(1 for _, r in states if len(set(r.values())) > 1) / n
    return {"n_states": n, "best_single_arm": single, "best_single_regret": mean[single],
            "G": mean["G"], "F": mean["F"], "D": mean["D"],
            "capability_router_regret": sum(cap_router) / n, "oracle_router_regret": sum(oracle) / n,
            "oracle_gap": sum(gap) / n, "oracle_gap_ci": boot_ci(gap),
            "capability_router_gain": sum(gain) / n, "capability_router_gain_ci": boot_ci(gain, seed=1),
            "states_where_arms_differ": disagree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/router")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    caps = {reg: primary_caps(reg) for reg in REGIMES}
    rows = []
    for b in BUDGETS:
        data = {}
        for reg in REGIMES:
            per = {}
            for a in ARMS:
                for r in load(f"results/utility/scored/{reg}_{a}_top1_b{b}.jsonl"):
                    if reg == "bytesized32" and r.get("gold_seen") is True:
                        continue
                    per.setdefault(r["id"], {})[a] = float(r["regret"])
            data[reg] = [(caps[reg].get(i, "none"), r) for i, r in per.items() if len(r) == len(ARMS)]
        groups = {reg: data[reg] for reg in REGIMES}
        groups["main"] = [s for reg in MAIN for s in data[reg]]
        for g, states in groups.items():
            res = analyse(states)
            rows.append({"group": g, "budget": b, **{k: (round(v, 4) if isinstance(v, float) else
                                                         (f"{v[0]:.3f}..{v[1]:.3f}" if isinstance(v, tuple) else v))
                                                      for k, v in res.items()}})

    with open(os.path.join(args.out_dir, "router.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = dict(REGIMES, main="ALFWorld + ScienceWorld (main)")
    md = ["# Per-decision routing: upper bound (RQ4)", "",
          "Mean decision regret on the same states (D judged top-1). Best single arm is chosen in-sample; "
          "the capability router picks, per state, the arm that is best on the other states with the same "
          "primary capability; the oracle router picks the best arm per state (an upper bound needing all "
          "three arms). Gap = best single minus router, 95% bootstrap CI over states.", "",
          "| Regime | n | States | Best single arm | Capability router (gain) | Oracle router (max gain) | "
          "Arms differ |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["budget"] != 256 and r["group"] != "main":
            continue
        md.append(f"| {names[r['group']]} | {r['budget']} | {r['n_states']} | {r['best_single_arm']} "
                  f"{r['best_single_regret']:.3f} | {r['capability_router_regret']:.3f} "
                  f"({r['capability_router_gain']:+.3f} [{r['capability_router_gain_ci']}]) | "
                  f"{r['oracle_router_regret']:.3f} ({r['oracle_gap']:+.3f} [{r['oracle_gap_ci']}]) | "
                  f"{r['states_where_arms_differ']:.0%} |")
    md += ["", "Rows: every regime at n = 256, and the pooled main regimes at every budget."]
    with open(os.path.join(args.out_dir, "router.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
