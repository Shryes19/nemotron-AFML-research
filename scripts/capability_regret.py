#!/usr/bin/env python3
"""
RQ1, per capability: how much decision regret does each arm have when the right move needs a given
capability?

    python scripts/capability_regret.py --budget 256 --out-dir results/capability_regret

For every branched state, the five candidate transitions (obs, action_i, true next observation_i) are
tagged with capability/tagger.py. A state is assigned the tags of its *best* candidate: the action with
the highest true value (ties: the expert's action if it is among the best, else the first). That is the
outcome a world model must predict well for the agent to choose right. Tags are multi-label, so a state
counts once under each tag of its best candidate.

Per regime and pooled over the two main regimes (ALFWorld + ScienceWorld test + seen), for arms G, F and
D (top-1) at one budget: mean regret per capability with a 95% bootstrap CI over states, and the paired
differences G - F and F - D with their CIs (same states, both arms). ByteSized32 uses the gold-unseen
states and is reported apart, as in the rest of the paper.

Outputs: capability_regret.csv (one row per regime x capability x arm), capability_pairs.csv (paired
differences), capability_regret.md, fig_capability_regret.png/.pdf
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
from capability.tagger import TAGS, tag_transition  # noqa: E402

REGIMES = {"alfworld": "ALFWorld", "sciworld": "ScienceWorld test", "sciworld_seen": "ScienceWorld seen",
           "bytesized32": "ByteSized32 (gold unseen)"}
MAIN = ["alfworld", "sciworld", "sciworld_seen"]
ARMS = [("G", "top1"), ("F", "top1"), ("D", "top1")]
TAG_NAME = {"object_state": "Object state", "containment_location": "Containment / location",
            "numeric_physical": "Numeric / physical", "partial_obs": "Partially observable",
            "action_failure": "Action failure", "none": "No change"}
COLOR = {"G": "#2a78d6", "F": "#eb6834", "D": "#1baf7a"}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def best_index(values, gold):
    m = max(values)
    best = [i for i, v in enumerate(values) if v == m]
    return gold if gold in best else best[0]


def boot_ci(xs, reps=2000, seed=0):
    if not xs:
        return (float("nan"),) * 2
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def state_tags(regime):
    """id -> tags of the best candidate (plus 'none' if it has no tag)."""
    out = {}
    tv = {r["id"]: r["values"] for r in load(f"results/utility/{regime}_true_values.jsonl")}
    for s in load(f"data/eval/{regime}_branched_states.jsonl"):
        if s["id"] not in tv:
            continue
        i = best_index(tv[s["id"]], int(s["gold_index"]))
        full = s.get("true_next_full") or s["true_next"]
        t = tag_transition(s["obs"], s["actions"][i], full[i], feedback=s["true_next"][i])
        out[s["id"]] = t["tags"] or ["none"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--out-dir", default="results/capability_regret")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    regret = {}                      # (regime, arm) -> {id: regret}
    tags = {}
    for reg in REGIMES:
        tags[reg] = state_tags(reg)
        for arm, mode in ARMS:
            rows = load(f"results/utility/scored/{reg}_{arm}_{mode}_b{args.budget}.jsonl")
            if reg == "bytesized32":
                rows = [r for r in rows if r.get("gold_seen") is not True]
            regret[(reg, arm)] = {r["id"]: float(r["regret"]) for r in rows}

    groups = {r: [r] for r in REGIMES}
    groups["main"] = MAIN
    names = dict(REGIMES, main="ALFWorld + ScienceWorld (main)")
    cats = TAGS + ["none"]
    out, pairs = [], []
    for g, regs in groups.items():
        for c in cats:
            ids = [(reg, i) for reg in regs for i, t in tags[reg].items()
                   if c in t and all(i in regret[(reg, a)] for a, _ in ARMS)]
            if not ids:
                continue
            per_arm = {}
            for arm, _ in ARMS:
                xs = [regret[(reg, arm)][i] for reg, i in ids]
                lo, hi = boot_ci(xs)
                per_arm[arm] = xs
                out.append({"group": g, "capability": c, "arm": arm, "n_states": len(xs),
                            "mean_regret": round(sum(xs) / len(xs), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
            for a, b in (("G", "F"), ("F", "D"), ("G", "D")):
                d = [x - y for x, y in zip(per_arm[a], per_arm[b])]
                lo, hi = boot_ci(d, seed=1)
                pairs.append({"group": g, "capability": c, "pair": f"{a}-{b}", "n_states": len(d),
                              "mean_diff": round(sum(d) / len(d), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                              "significant": "yes" if lo > 0 or hi < 0 else "no"})

    for name, rows in (("capability_regret.csv", out), ("capability_pairs.csv", pairs)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    md = [f"# Decision regret by capability (RQ1), n = {args.budget}", "",
          "Each state is tagged with the capabilities of its best candidate transition (the action with the "
          "highest true value). Mean regret, 95% bootstrap CI over states; lower is better. Paired "
          "differences use the same states for both arms; * = CI excludes 0.", ""]
    for g in groups:
        rs = [r for r in out if r["group"] == g]
        if not rs:
            continue
        md += [f"## {names[g]}", "", "| Capability | States | G | F | D top-1 | G - F | F - D |",
               "|---|---|---|---|---|---|---|"]
        for c in cats:
            cell = {r["arm"]: r for r in rs if r["capability"] == c}
            if not cell:
                continue
            pr = {p["pair"]: p for p in pairs if p["group"] == g and p["capability"] == c}
            fmt = lambda r: f"{r['mean_regret']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
            fd = lambda p: f"{p['mean_diff']:+.2f}{'*' if p['significant'] == 'yes' else ''}"
            md.append(f"| {TAG_NAME[c]} | {cell['G']['n_states']} | {fmt(cell['G'])} | {fmt(cell['F'])} | "
                      f"{fmt(cell['D'])} | {fd(pr['G-F'])} | {fd(pr['F-D'])} |")
        md.append("")
    with open(os.path.join(args.out_dir, "capability_regret.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = ["main", "bytesized32"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200)
    for ax, g in zip(axes, panels):
        cs = [c for c in cats if any(r["group"] == g and r["capability"] == c for r in out)]
        x = range(len(cs))
        w = 0.26
        for k, (arm, _) in enumerate(ARMS):
            rs = [next(r for r in out if r["group"] == g and r["capability"] == c and r["arm"] == arm) for c in cs]
            pos = [i + (k - 1) * (w + 0.01) for i in x]
            ax.bar(pos, [r["mean_regret"] for r in rs], w, color=COLOR[arm], edgecolor="white", linewidth=1,
                   label={"G": "G general", "F": "F fine-tuned", "D": "D JEPA (top-1)"}[arm],
                   yerr=[[r["mean_regret"] - r["ci_lo"] for r in rs], [r["ci_hi"] - r["mean_regret"] for r in rs]],
                   error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": "#3a3a38"})
        n = {c: next(r["n_states"] for r in out if r["group"] == g and r["capability"] == c) for c in cs}
        short = lambda c: TAG_NAME[c].replace(" / ", "/\n").replace("Partially ", "Partially\n").replace("Action ", "Action\n")
        ax.set_xticks(list(x), [f"{short(c)}\n(n={n[c]})" for c in cs], fontsize=6.5)
        ax.set_title(names[g], fontsize=9)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel(f"decision regret at n = {args.budget}", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"fig_capability_regret.{ext}"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
