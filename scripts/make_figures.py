"""Phase 3 results figures and tables from the decision-utility outputs (CPU only, seconds).

  python scripts/make_figures.py                       # defaults: results/utility -> results/figures

Inputs (written by scripts/run_utility.sh):
  results/utility/summary.csv               per (env, split, arm, mode, budget, game) regret / gold-hit / EM
  results/utility/<set>_true_values.jsonl   judge values of the true next states (random-choice baseline)
  data/eval/<set>_branched_states.jsonl     the branched states (split, gold_seen flag)
  results/features/feature_matrix.csv       capability mix per (env, game)
  results/utility/labels_Dtop1.csv          winner labels at n = 256

Outputs in --out-dir (each figure as .pdf for the paper and .png for slides):
  fig1_regret_vs_n        decision regret vs budget n, one panel per regime, line per arm (D in 3 modes)
  fig2_fidelity_vs_utility exact match vs regret, one point per (regime, arm, budget)
  fig3_capability_vs_dwins capability-mix rates of games where D wins vs where it does not
  main_table.md / .csv    n = 256: regret [95% CI], gold-hit, EM per regime and arm, + oracle and random
  crossovers.csv          per regime and arm pair: smallest n where the challenger's regret is lower,
                          and whether the 95% CIs separate there
  fidelity_utility_corr.csv  Spearman correlation of EM and regret, per regime and pooled
  capability_dwins.csv    median of each capability rate for D-wins vs other rows

Regimes are reported separately (docs/DECISION_UTILITY.md): ALFWorld new games, ScienceWorld test
variations, ScienceWorld held-out variations of seen types, ByteSized32 known states with new
actions (only the states whose gold transition the arms never saw).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGIMES = [  # (env, split in summary.csv, short title)
    ("alfworld", "heldout", "ALFWorld\nnew games"),
    ("sciworld", "heldout", "ScienceWorld\nnew variations (test split)"),
    ("sciworld", "heldout_seen", "ScienceWorld\nheld-out variations, seen types"),
    ("bytesized32", "seen_state_new_actions:gold_unseen", "ByteSized32\nknown states, new actions"),
]
# one hue per arm (fixed order, validated categorical palette); D's modes differ by line style + marker
SERIES = [  # (arm, mode, label, color, linestyle, marker)
    ("G", "top1", "G (RAG)", "#2a78d6", "-", "o"),
    ("F", "top1", "F (SFT)", "#eb6834", "-", "s"),
    ("D", "top1", "D (JEPA) top-1", "#1baf7a", "-", "^"),
    ("D", "topk", "D (JEPA) top-k", "#1baf7a", "--", "v"),
    ("D", "native", "D (JEPA) native", "#1baf7a", ":", "D"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
RATES = ["rate_object_state", "rate_containment_location", "rate_numeric_physical",
         "rate_partial_obs", "rate_action_failure"]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 200, "savefig.bbox": "tight",
        "pdf.fonttype": 42})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (name, ext)))
    plt.close(fig)


def pooled(summary):
    """(env, split, arm, mode, budget) -> pooled row (game == '*')."""
    return {(r["env"], r["split"], r["arm"], r["mode"], r["budget"]): r for r in summary if r["game"] == "*"}


def budgets_of(pool, env, split):
    return sorted({int(b) for (e, s, a, m, b) in pool if e == env and s == split and b.isdigit()})


# ------------------------------------------------------------------ random-choice baseline

def random_regret(states_glob, tv_glob, seed=0, reps=1000):
    """Expected regret of choosing uniformly at random, per regime (mean, 95% bootstrap CI)."""
    states = {}
    for path in glob.glob(states_glob):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    states[s["id"]] = s
    per = defaultdict(list)
    for path in glob.glob(tv_glob):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                s, v = states.get(r["id"]), r["values"]
                if s is None or any(x is None for x in v):
                    continue
                reg = max(v) - sum(v) / len(v)
                split = s.get("split", "")
                per[(s["env"], split)].append(reg)
                if s.get("gold_seen_in_training") is False:
                    per[(s["env"], split + ":gold_unseen")].append(reg)
    out = {}
    rng = random.Random(seed)
    for key, vals in per.items():
        means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(reps))
        out[key] = (sum(vals) / len(vals), means[int(0.025 * reps)], means[int(0.975 * reps) - 1], len(vals))
    return out


# ------------------------------------------------------------------ figure 1 + crossovers

def fig_regret_vs_n(pool, rand, out_dir):
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(3.0 * len(REGIMES), 3.0), sharey=False)
    handles = {}
    for ax, (env, split, title) in zip(axes, REGIMES):
        ns = budgets_of(pool, env, split)
        ax.set_title(title, loc="left")
        ax.set_xscale("log", base=2)
        if ns:
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
        for arm, mode, label, color, ls, mk in SERIES:
            pts = [(n, pool.get((env, split, arm, mode, str(n)))) for n in ns]
            pts = [(n, r) for n, r in pts if r]
            if not pts:
                continue
            x = [n for n, _ in pts]
            y = [float(r["mean_regret"]) for _, r in pts]
            lo = [float(r["regret_ci_lo"]) for _, r in pts]
            hi = [float(r["regret_ci_hi"]) for _, r in pts]
            if ls == "-":   # CI bands for the main lines only, so the panel stays readable
                ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
            (h,) = ax.plot(x, y, color=color, linestyle=ls, marker=mk, label=label)
            handles[label] = h
        if (env, split) in rand and ns:
            m = rand[(env, split)][0]
            h = ax.axhline(m, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, label="random choice")
            handles["random choice"] = h
        ax.set_xlabel("training transitions n")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("mean decision regret (lower is better)")
    order = [s[2] for s in SERIES] + ["random choice"]
    fig.legend([handles[k] for k in order if k in handles], [k for k in order if k in handles],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.text(0.0, -0.21, "Oracle (true next states) has regret 0 by construction. "
             "Bands: 95% bootstrap CI (G, F, D top-1).",
             color=MUTED, fontsize=7)
    save(fig, out_dir, "fig1_regret_vs_n")


def crossovers(pool):
    pairs = [(("G", "top1"), ("F", "top1")), (("F", "top1"), ("D", "top1")), (("G", "top1"), ("D", "top1")),
             (("F", "top1"), ("D", "native")), (("G", "top1"), ("D", "native"))]
    rows = []
    for env, split, _ in REGIMES:
        for (a, am), (b, bm) in pairs:
            first, sep, seen = "", "", False
            for n in budgets_of(pool, env, split):
                ra, rb = pool.get((env, split, a, am, str(n))), pool.get((env, split, b, bm, str(n)))
                if not (ra and rb):
                    continue
                seen = True
                if float(rb["mean_regret"]) < float(ra["mean_regret"]):
                    first = n
                    sep = float(rb["regret_ci_hi"]) < float(ra["regret_ci_lo"])
                    break
            if seen:
                rows.append({"env": env, "split": split, "incumbent": "%s/%s" % (a, am),
                             "challenger": "%s/%s" % (b, bm), "first_n_challenger_lower": first or "never",
                             "cis_separate_there": sep if first else ""})
    return rows


# ------------------------------------------------------------------ figure 2 + correlation

def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def fig_fidelity_vs_utility(pool, out_dir):
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    corr, all_x, all_y = [], [], []
    for (env, split, title), mk in zip(REGIMES, markers):
        xs, ys = [], []
        for arm, mode, label, color, _, _ in SERIES:
            for n in budgets_of(pool, env, split):
                r = pool.get((env, split, arm, mode, str(n)))
                if not r or fnum(r.get("em")) is None:
                    continue
                x, y = float(r["em"]), float(r["mean_regret"])
                ax.scatter(x, y, color=color, marker=mk, s=18 + 6 * (n.bit_length() - 5),
                           edgecolors="white", linewidths=0.8, zorder=3)
                xs.append(x)
                ys.append(y)
        if xs:
            corr.append({"env": env, "split": split, "n_points": len(xs), "spearman_em_vs_regret": spearman(xs, ys)})
            all_x += xs
            all_y += ys
    corr.append({"env": "*", "split": "*", "n_points": len(all_x), "spearman_em_vs_regret": spearman(all_x, all_y)})
    ax.set_xlabel("text fidelity: exact match of the predicted next state")
    ax.set_ylabel("mean decision regret (lower is better)")
    ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    arm_h = [Line2D([], [], color=c, linestyle="", marker="s", markersize=7, label=l)
             for l, c in (("G (RAG)", SERIES[0][3]), ("F (SFT)", SERIES[1][3]), ("D (JEPA) top-1 / top-k", SERIES[2][3]))]
    reg_h = [Line2D([], [], color=MUTED, linestyle="", marker=mk, label=t.replace("\n", ": "))
             for (_, _, t), mk in zip(REGIMES, markers)]
    leg = ax.legend(handles=arm_h, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="arm")
    ax.add_artist(leg)
    ax.legend(handles=reg_h, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False, title="regime")
    ax.text(0.0, -0.22, "One point per (regime, arm, budget); larger marker = larger n. "
            "If fidelity tracked utility, points would fall left-to-right.", transform=ax.transAxes,
            color=MUTED, fontsize=7)
    save(fig, out_dir, "fig2_fidelity_vs_utility")
    return corr


# ------------------------------------------------------------------ figure 3

def family_of(env, game):
    return game.split("-")[0] if env == "alfworld" else game


def fig_capability_vs_dwins(features, labels, envs, out_dir):
    idx = {(r["env"], r["game"]): r for r in labels}
    rows = []
    for r in features:
        if r["env"] not in envs:
            continue
        lab = idx.get((r["env"], r["game"])) or idx.get((r["env"], family_of(r["env"], r["game"])))
        if lab and lab.get("label"):
            rows.append((r, lab["label"] == "D"))
    if not rows:
        print("fig3: no labelled feature rows yet, skipped")
        return []
    fig, axes = plt.subplots(1, len(RATES), figsize=(2.0 * len(RATES), 2.8), sharey=True)
    rng = random.Random(0)
    summary = []
    for ax, col in zip(axes, RATES):
        groups = {True: [], False: []}
        for r, dwin in rows:
            v = fnum(r.get(col))
            if v is not None:
                groups[dwin].append(v)
        for xpos, key, color in ((0, False, MUTED), (1, True, "#1baf7a")):
            vals = groups[key]
            ax.scatter([xpos + rng.uniform(-0.12, 0.12) for _ in vals], vals, s=16, color=color,
                       edgecolors="white", linewidths=0.6, zorder=3)
            if vals:
                med = sorted(vals)[len(vals) // 2]
                ax.plot([xpos - 0.25, xpos + 0.25], [med, med], color=INK, linewidth=1.5, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["other\nwins (%d)" % len(groups[False]), "D wins\n(%d)" % len(groups[True])])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(col.replace("rate_", "").replace("_", " "), loc="left")
        ax.grid(axis="x", visible=False)

        def med(v):
            return round(sorted(v)[len(v) // 2], 4) if v else ""
        summary.append({"feature": col, "n_dwins": len(groups[True]), "n_other": len(groups[False]),
                        "median_dwins": med(groups[True]), "median_other": med(groups[False])})
    axes[0].set_ylabel("share of transitions in the game")
    fig.text(0.0, -0.08, "One dot per feature-matrix row (%s); bar = median. Label = lowest regret at n = 256."
             % ", ".join(sorted(envs)), color=MUTED, fontsize=7)
    save(fig, out_dir, "fig3_capability_vs_dwins")
    return summary


# ------------------------------------------------------------------ main table

def main_table(pool, summary_rows, rand, budget, out_dir):
    oracle = {(r["env"], r["split"]): r for r in summary_rows if r["arm"] == "oracle" and r["game"] == "*"}
    rows = []
    for env, split, title in REGIMES:
        base = {"regime": title.replace("\n", ": ")}
        if (env, split) in oracle:
            o = oracle[(env, split)]
            rows.append(dict(base, arm="oracle (true next states)", n_states=o["n_states"], regret="0",
                             gold_hit=o["gold_hit"], em=""))
        if (env, split) in rand:
            m, lo, hi, n = rand[(env, split)]
            rows.append(dict(base, arm="random choice", n_states=n, regret="%.2f [%.2f, %.2f]" % (m, lo, hi),
                             gold_hit="0.20", em=""))
        for arm, mode, label, *_ in SERIES:
            r = pool.get((env, split, arm, mode, str(budget)))
            if r:
                rows.append(dict(base, arm=label, n_states=r["n_states"],
                                 regret="%.2f [%.2f, %.2f]" % (float(r["mean_regret"]), float(r["regret_ci_lo"]),
                                                               float(r["regret_ci_hi"])),
                                 gold_hit=r["gold_hit"], em=r.get("em", "")))
    write_csv(os.path.join(out_dir, "main_table.csv"), rows)
    with open(os.path.join(out_dir, "main_table.md"), "w", encoding="utf-8") as f:
        f.write("Decision utility at n = %s. Regret: mean [95%% bootstrap CI], lower is better. "
                "Gold-hit: share of states where the chosen action is the expert's. EM: exact match of "
                "the top-1 predicted next state.\n\n" % budget)
        f.write("| Regime | Arm | States | Regret | Gold-hit | EM |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %s | %s | %s | %s | %s | %s |\n" % (r["regime"], r["arm"], r["n_states"], r["regret"],
                                                        r["gold_hit"], r["em"]))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--utility-dir", default="results/utility")
    p.add_argument("--states", default="data/eval/*_branched_states.jsonl")
    p.add_argument("--features", default="results/features/feature_matrix.csv")
    p.add_argument("--labels", default=None, help="default: <utility-dir>/labels_Dtop1.csv")
    p.add_argument("--envs", default="alfworld,sciworld", help="feature-matrix envs in figure 3")
    p.add_argument("--budget", default="256")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args(argv)
    style()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = read_csv(os.path.join(args.utility_dir, "summary.csv"))
    pool = pooled(summary)
    rand = random_regret(args.states, os.path.join(args.utility_dir, "*_true_values.jsonl"))

    fig_regret_vs_n(pool, rand, args.out_dir)
    write_csv(os.path.join(args.out_dir, "crossovers.csv"), crossovers(pool))
    write_csv(os.path.join(args.out_dir, "fidelity_utility_corr.csv"), fig_fidelity_vs_utility(pool, args.out_dir))
    labels_path = args.labels or os.path.join(args.utility_dir, "labels_Dtop1.csv")
    if os.path.exists(labels_path) and os.path.exists(args.features):
        cap = fig_capability_vs_dwins(read_csv(args.features), read_csv(labels_path),
                                      set(args.envs.split(",")), args.out_dir)
        write_csv(os.path.join(args.out_dir, "capability_dwins.csv"), cap)
    else:
        print("fig3 skipped: %s or %s missing" % (labels_path, args.features))
    main_table(pool, summary, rand, args.budget, args.out_dir)
    print("wrote figures and tables to %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
