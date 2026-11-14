#!/usr/bin/env python3
"""
RQ2: is the specialist's compute worth it? Decision regret against GPU-hours, per arm and environment.

    python scripts/compute_crossover.py --cost results/compute/cost_table.csv \
        --summary results/utility/summary.csv --d-mode top1 --out-dir results/crossover

Cost of an arm at budget n (GPU-hours on the RTX 5070, from the cost table's "arm" rows):
    the stages logged at that n          (D: jepa_train + jepa_predict; F: sft_train + sft_predict;
                                          G: g_generation)
  + one-off stages for that env           (logged with an n that is not a budget, e.g. NV-Embed encoding
                                          for D, rag_index_build for G), added to every budget
  utility_scoring is left out: it is the cost of *evaluating* an arm, the same for all arms, not of
  building or running the world model. "sciworld_seen" costs count as ScienceWorld.
Regret of an arm at budget n: the pooled row (game = *) of Saanvi's summary.csv. ScienceWorld pools the
test and seen sets, weighted by state count (as the labels do). ByteSized32 uses the gold-unseen subset
and is reported apart (known state, new action).

Outputs (out-dir):
  points.csv        env, arm, n, gpu_hours, regret, ci_lo, ci_hi, n_states
  crossover.csv     per env and specialist (F, D): its best regret and cost, G's best regret and cost,
                    the smallest GPU-hours at which the specialist matches or beats G's best regret, and
                    whether the two confidence intervals overlap there
  crossover.md      the same as a readable table
  fig_regret_vs_gpu_hours.png / .pdf
"""
import argparse
import collections
import csv
import math
import os

BUDGETS = [32, 64, 128, 256, 512, 1024]
ENV_ALIAS = {"sciworld_seen": "sciworld"}
SPLITS = {"alfworld": ["heldout"], "sciworld": ["heldout", "heldout_seen"],
          "bytesized32": ["seen_state_new_actions:gold_unseen"]}
ENV_NAME = {"alfworld": "ALFWorld", "sciworld": "ScienceWorld",
            "bytesized32": "ByteSized32 (known state, new action)"}
ARMS = ["G", "F", "D"]
COLOR = {"G": "#2a78d6", "F": "#eb6834", "D": "#1baf7a"}     # fixed categorical order
ARM_NAME = {"G": "G general (7B + retrieval)", "F": "F fine-tuned (1.5B LoRA)", "D": "D JEPA"}
EVAL_STAGES = {"utility_scoring"}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def costs(cost_rows):
    per_n, one_off = collections.defaultdict(float), collections.defaultdict(float)
    for r in cost_rows:
        if r["category"] != "arm" or r["arm"] not in ARMS or r["stage"] in EVAL_STAGES:
            continue
        env = ENV_ALIAS.get(r["env"], r["env"])
        h = float(r["gpu_hours"])
        n = int(r["n"]) if r["n"].lstrip("-").isdigit() else None
        if n in BUDGETS:
            per_n[(r["arm"], env, n)] += h
        else:
            one_off[(r["arm"], env)] += h
    return per_n, one_off


def regrets(summary, d_mode):
    out = {}
    for env, splits in SPLITS.items():
        for arm in ARMS:
            mode = d_mode if arm == "D" else "top1"
            for n in BUDGETS:
                parts = [r for r in summary if r["env"] == env and r["split"] in splits and r["arm"] == arm
                         and r["mode"] == mode and r["budget"] == str(n) and r["game"] == "*"
                         and r["family"] == "*"]
                if not parts:
                    continue
                N = sum(int(r["n_states"]) for r in parts)
                mean = sum(float(r["mean_regret"]) * int(r["n_states"]) for r in parts) / N
                # pooled standard error from each set's 95% interval (weights = state share)
                se = math.sqrt(sum((int(r["n_states"]) / N) ** 2 *
                                   ((float(r["regret_ci_hi"]) - float(r["regret_ci_lo"])) / 3.92) ** 2
                                   for r in parts))
                out[(arm, env, n)] = (mean, mean - 1.96 * se, mean + 1.96 * se, N)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", default="results/compute/cost_table.csv")
    ap.add_argument("--summary", default="results/utility/summary.csv")
    ap.add_argument("--d-mode", default="top1", choices=["top1", "topk", "native"])
    ap.add_argument("--out-dir", default="results/crossover")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_n, one_off = costs(read(args.cost))
    reg = regrets(read(args.summary), args.d_mode)
    points = []
    for (arm, env, n), (m, lo, hi, N) in sorted(reg.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        if (arm, env, n) not in per_n:
            print(f"  no cost logged for {arm} {env} n={n}; point skipped")
            continue
        h = per_n[(arm, env, n)] + one_off.get((arm, env), 0.0)
        points.append({"env": env, "arm": arm, "n": n, "gpu_hours": round(h, 4), "regret": round(m, 4),
                       "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n_states": N})
    with open(os.path.join(args.out_dir, "points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)

    if not points:
        raise SystemExit("no (arm, env, budget) has both a regret and a cost: check --cost and --summary")
    rows = []
    for env in SPLITS:
        P = {a: [p for p in points if p["env"] == env and p["arm"] == a] for a in ARMS}
        if not P["G"]:
            continue
        g_best = min(P["G"], key=lambda p: (p["regret"], p["gpu_hours"]))
        for s in ("F", "D"):
            if not P[s]:
                continue
            s_best = min(P[s], key=lambda p: (p["regret"], p["gpu_hours"]))
            beats = sorted((p for p in P[s] if p["regret"] <= g_best["regret"]), key=lambda p: p["gpu_hours"])
            first = beats[0] if beats else None
            rows.append({
                "env": env, "specialist": s,
                "specialist_best_regret": s_best["regret"], "specialist_best_n": s_best["n"],
                "specialist_best_gpu_h": s_best["gpu_hours"],
                "G_best_regret": g_best["regret"], "G_best_n": g_best["n"], "G_best_gpu_h": g_best["gpu_hours"],
                "crossover_gpu_h": first["gpu_hours"] if first else "",
                "crossover_n": first["n"] if first else "",
                "ci_overlap_at_crossover": (("yes" if first["ci_lo"] <= g_best["ci_hi"] else "no") if first else ""),
                "best_vs_best_ci_overlap": "yes" if (s_best["ci_lo"] <= g_best["ci_hi"]
                                                     and g_best["ci_lo"] <= s_best["ci_hi"]) else "no",
            })
    if not rows:
        raise SystemExit("no environment has both G and a specialist with costs: nothing to compare")
    with open(os.path.join(args.out_dir, "crossover.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [f"# Compute crossover (RQ2): regret vs GPU-hours, D judged {args.d_mode}", "",
          "Cost = building + running the world model at that budget (RTX 5070 GPU-hours; evaluation "
          "excluded). Crossover = cheapest budget at which the specialist's regret is at or below G's "
          "best regret at any budget.", "",
          "Note: G has no training, so its cost is almost all generation with the 7B model and grows with "
          "the number of predictions; F and D pay mostly once (training, encoding) and predict cheaply. "
          "These GPU-hours are for this evaluation's predictions (one pass over each held-out set).", "",
          "| Env | Specialist | Specialist best regret (n, GPU-h) | G best regret (n, GPU-h) | "
          "Crossover GPU-h (n) | Best-vs-best CIs overlap |", "|---|---|---|---|---|---|"]
    for r in rows:
        cross = f"{r['crossover_gpu_h']:.2f} (n={r['crossover_n']})" if r["crossover_gpu_h"] != "" else "never"
        md.append(f"| {ENV_NAME[r['env']]} | {r['specialist']} | {r['specialist_best_regret']:.3f} "
                  f"(n={r['specialist_best_n']}, {r['specialist_best_gpu_h']:.2f} h) | {r['G_best_regret']:.3f} "
                  f"(n={r['G_best_n']}, {r['G_best_gpu_h']:.2f} h) | {cross} | {r['best_vs_best_ci_overlap']} |")
    md += ["", "Points (GPU-h, regret [95% CI]):", ""]
    for env in SPLITS:
        for a in ARMS:
            pts = [p for p in points if p["env"] == env and p["arm"] == a]
            if pts:
                md.append(f"- {ENV_NAME[env]}, {a}: " + "; ".join(
                    f"n={p['n']}: {p['gpu_hours']:.2f} h, {p['regret']:.3f} [{p['ci_lo']:.2f}, {p['ci_hi']:.2f}]"
                    for p in pts))
    with open(os.path.join(args.out_dir, "crossover.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    envs = [e for e in SPLITS if any(p["env"] == e for p in points)]
    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.6), dpi=200, squeeze=False)
    for ax, env in zip(axes[0], envs):
        for a in ARMS:
            pts = sorted((p for p in points if p["env"] == env and p["arm"] == a), key=lambda p: p["n"])  # line follows n
            if not pts:
                continue
            x = [p["gpu_hours"] for p in pts]
            y = [p["regret"] for p in pts]
            err = [[p["regret"] - p["ci_lo"] for p in pts], [p["ci_hi"] - p["regret"] for p in pts]]
            ax.errorbar(x, y, yerr=err, color=COLOR[a], marker="o", ms=4, lw=2, capsize=2, elinewidth=0.8,
                        label=ARM_NAME[a])
            for p in (pts[0], pts[-1]):
                ax.annotate(f"n={p['n']}", (p["gpu_hours"], p["regret"]), textcoords="offset points",
                            xytext=(4, 4), fontsize=6.5, color="#3a3a38")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(ENV_NAME[env], fontsize=9)
        ax.set_xlabel("GPU-hours (RTX 5070, log scale)", fontsize=8)
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("decision regret (lower is better)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"fig_regret_vs_gpu_hours.{ext}"))
    print("\n".join(md[:8 + len(rows)]))
    print(f"-> {args.out_dir}/ (points.csv, crossover.csv, crossover.md, fig_regret_vs_gpu_hours.png/.pdf)")


if __name__ == "__main__":
    main()
