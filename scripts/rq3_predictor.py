"""
scripts/rq3_predictor.py
────────────────────────
RQ3: "Is the specialist worth it?" Can cheap features measured on n = 32 transitions
predict which world-model arm wins (lowest decision regret at n = 256) on a game?

Input
  results/features/feature_matrix.csv   one row per (env, game): Features 1-4 (+ label)
  results/utility/labels.csv            per (env, game|family): label, regret_G/F/D, n_G/F/D
  results/sft/sft_probe_scores.csv      optional: SFT probe (for the Feature 1 ablation)

What it predicts (plan: Ridge + GBT, chained binary decisions)
  task "F_vs_G"  : does F beat G?   target = regret_G - regret_F  (> 0 means F wins)
  task "D_vs_F"  : does D beat F?   target = regret_F - regret_D  (> 0 means D wins)
  task "D_vs_G"  : does D beat G?
  task "winner"  : which of G / F / D wins (predict each arm's regret, take the lowest)
The models predict the regret DIFFERENCE (Ridge regression + gradient-boosted trees, averaged)
and the sign gives the decision. With ~30 rows a continuous target uses more of the data than
a yes/no label, and the Spearman correlation of predicted vs true difference is reported too.
Rows are weighted by their number of scored states (smallest arm), as Siddhant asked.

Evaluation (no tuning on test rows)
  - leave-one-group-out, group = (env, family): ALFWorld labels are per task family, so both
    games of a family are held out together (otherwise the label would leak); a ScienceWorld
    task type is its own group
  - leave-one-env-out: train on one environment, test on the other (harder transfer test)
  - baselines: majority class of the training fold; single features
  - a permutation test (labels shuffled) for the main model, since the sample is small

Feature-1 ablation (plan, Phase 3): the same model with
  jepa_probe : Features 2-4 + JEPA probe fidelity
  sft_probe  : Features 2-4 + SFT probe fidelity (text F1 after 20 LoRA steps)
  both / none / probe-only variants

    python scripts/rq3_predictor.py --labels results/utility/labels.csv \
        --sft-probe results/sft/sft_probe_scores.csv --out results/rq3

ByteSized32 is left out by default (team decision: "known state, new action" is a different
test, and most games have < 10 states); use --envs alfworld,sciworld,bytesized32 to add it.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MIX_COLS = ["rate_object_state", "rate_containment_location", "rate_numeric_physical",
            "rate_partial_obs", "rate_action_failure"]
STAT_COLS = ["obs_len_chars", "action_vocab", "zero_overlap_frac", "mean_jaccard", "target_repeat_frac"]
ARMS = ["G", "F", "D"]
TASKS = {"F_vs_G": ("G", "F"), "D_vs_F": ("F", "D"), "D_vs_G": ("G", "D")}   # (loser-if-neg, winner-if-pos)


def family_of(env, game):
    """Same rule as scripts/build_feature_matrix.py."""
    return game.split("-")[0] if env == "alfworld" else game


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def lookup(idx, env, game):
    return idx.get((env, game)) or idx.get((env, family_of(env, game))) or idx.get((env, "*"))


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_rows(args):
    matrix = read_csv(args.matrix)
    labels = {(r["env"], r["game"]): r for r in read_csv(args.labels)} if args.labels else {}
    sft = {(r["env"], r["game"]): r for r in read_csv(args.sft_probe)} if args.sft_probe else {}
    envs = set(args.envs.split(","))
    rows = []
    for m in matrix:
        if m["env"] not in envs:
            continue
        r = {"env": m["env"], "game": m["game"], "group": f"{m['env']}:{family_of(m['env'], m['game'])}"}
        for c in MIX_COLS + STAT_COLS + ["probe_fidelity", "recall_at5"]:
            r[c] = num(m.get(c))
        s = lookup(sft, m["env"], m["game"])
        r["sft_probe_f1"] = num(s.get("sft_probe_f1")) if s else None
        r["sft_probe_exact"] = num(s.get("sft_probe_exact")) if s else None
        lab = lookup(labels, m["env"], m["game"])
        r["label"] = (lab or {}).get("label") or m.get("label") or None
        for a in ARMS:
            r[f"regret_{a}"] = num((lab or {}).get(f"regret_{a}"))
            r[f"n_{a}"] = num((lab or {}).get(f"n_{a}")) or 0
        present = [r[f"n_{a}"] for a in ARMS if r[f"regret_{a}"] is not None]
        r["n_states"] = min(present) if present else 0
        rows.append(r)
    return rows


# ── Models ───────────────────────────────────────────────────────────────────

def fit_predict(Xtr, ytr, wtr, Xte, seed):
    """Ridge (standardised, alpha by inner CV) + GBT regressor, averaged."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
    ridge.fit(Xtr, ytr, ridgecv__sample_weight=wtr)
    gbt = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, min_samples_leaf=2, random_state=seed)
    gbt.fit(Xtr, ytr, sample_weight=wtr)
    return 0.5 * (ridge.predict(Xte) + gbt.predict(Xte))


def splits(rows, scheme):
    key = "group" if scheme == "group" else "env"
    groups = sorted({r[key] for r in rows})
    for g in groups:
        te = [i for i, r in enumerate(rows) if r[key] == g]
        tr = [i for i, r in enumerate(rows) if r[key] != g]
        if tr and te:
            yield tr, te


def evaluate_binary(rows, feats, task, scheme, seed=0):
    lo, hi = TASKS[task]
    data = [r for r in rows if r[f"regret_{lo}"] is not None and r[f"regret_{hi}"] is not None]
    if len(data) < 6 or not feats:
        return None
    X = np.array([[np.nan if r[c] is None else r[c] for c in feats] for r in data], dtype=float)
    d = np.array([r[f"regret_{lo}"] - r[f"regret_{hi}"] for r in data])
    w = np.array([max(1.0, r["n_states"]) for r in data])
    pred, base = np.full(len(data), np.nan), np.full(len(data), np.nan)
    for tr, te in splits(data, scheme):
        if len(set(np.sign(d[tr]) > 0)) < 2 and scheme == "env":
            pass
        pred[te] = fit_predict(X[tr], d[tr], w[tr], X[te], seed)
        base[te] = 1.0 if (d[tr] > 0).mean() >= 0.5 else -1.0          # majority of training fold
    ok = ~np.isnan(pred)
    y = d[ok] > 0
    acc = float(((pred[ok] > 0) == y).mean())
    bal = balanced_acc(y, pred[ok] > 0)
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred[ok], d[ok]).statistic) if ok.sum() > 2 else float("nan")
    return {"n": int(ok.sum()), "accuracy": acc, "balanced_acc": bal,
            "majority_acc": float(((base[ok] > 0) == y).mean()),
            "share_pos": float(y.mean()), "spearman": rho,
            "wilson95": wilson(acc, int(ok.sum())),
            "oof": [{"env": r["env"], "game": r["game"], "true_diff": float(d[i]),
                     "pred_diff": float(pred[i])} for i, r in enumerate(data) if ok[i]]}


def evaluate_winner(rows, feats, scheme, seed=0):
    data = [r for r in rows if all(r[f"regret_{a}"] is not None for a in ARMS) and r["label"]]
    if len(data) < 6 or not feats:
        return None
    X = np.array([[np.nan if r[c] is None else r[c] for c in feats] for r in data], dtype=float)
    w = np.array([max(1.0, r["n_states"]) for r in data])
    preds = {}
    for a in ARMS:
        y = np.array([r[f"regret_{a}"] for r in data])
        p = np.full(len(data), np.nan)
        for tr, te in splits(data, scheme):
            p[te] = fit_predict(X[tr], y[tr], w[tr], X[te], seed)
        preds[a] = p
    true = [r["label"] for r in data]
    guess = [min(ARMS, key=lambda a: preds[a][i]) for i in range(len(data))]
    maj = []
    for tr, te in splits(data, scheme):
        m = Counter(true[i] for i in tr).most_common(1)[0][0]
        maj += [(i, m) for i in te]
    maj = dict(maj)
    acc = float(np.mean([g == t for g, t in zip(guess, true)]))
    return {"n": len(data), "accuracy": acc, "wilson95": wilson(acc, len(data)),
            "majority_acc": float(np.mean([maj[i] == true[i] for i in range(len(data))])),
            "label_counts": dict(Counter(true))}


def balanced_acc(y, p):
    parts = [float((p[y == c] == c).mean()) for c in (True, False) if (y == c).any()]
    return float(np.mean(parts)) if parts else float("nan")


def wilson(p, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def permutation_p(rows, feats, task, scheme, observed, n_perm, seed):
    """Share of label shuffles (regret differences permuted across rows) that reach the
    observed accuracy."""
    lo, hi = TASKS[task]
    rng = random.Random(seed)
    data = [r for r in rows if r[f"regret_{lo}"] is not None and r[f"regret_{hi}"] is not None]
    diffs = [(r[f"regret_{lo}"], r[f"regret_{hi}"]) for r in data]
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(diffs)
        shuffled = [dict(r, **{f"regret_{lo}": a, f"regret_{hi}": b}) for r, (a, b) in zip(data, diffs)]
        res = evaluate_binary(shuffled, feats, task, scheme, seed)
        hits += res["accuracy"] >= observed - 1e-12
    return (hits + 1) / (n_perm + 1)


def importances(rows, feats, task, seed=0):
    """Fit on all rows: GBT importance and standardised Ridge coefficient per feature."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    lo, hi = TASKS[task]
    data = [r for r in rows if r[f"regret_{lo}"] is not None and r[f"regret_{hi}"] is not None]
    if len(data) < 6:
        return {}
    X = np.array([[np.nan if r[c] is None else r[c] for c in feats] for r in data], dtype=float)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    X = np.nan_to_num(X)
    d = np.array([r[f"regret_{lo}"] - r[f"regret_{hi}"] for r in data])
    w = np.array([max(1.0, r["n_states"]) for r in data])
    gbt = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                    min_samples_leaf=2, random_state=seed).fit(X, d, sample_weight=w)
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd == 0, 1, sd)
    ridge = Ridge(alpha=10.0).fit(Z, d, sample_weight=w)
    return {c: {"gbt": round(float(gi), 4), "ridge_coef": round(float(rc), 4)}
            for c, gi, rc in zip(feats, gbt.feature_importances_, ridge.coef_)}


# ── Main ─────────────────────────────────────────────────────────────────────

def feature_sets(rows):
    def has(c):
        return any(r[c] is not None for r in rows)
    base = [c for c in MIX_COLS + STAT_COLS + ["recall_at5"] if has(c)]
    sets = {"jepa_probe": base + ["probe_fidelity"], "no_probe": base}
    if has("sft_probe_f1"):
        sets["sft_probe"] = base + ["sft_probe_f1"]
        sets["both_probes"] = base + ["probe_fidelity", "sft_probe_f1"]
        sets["sft_probe_only"] = ["sft_probe_f1"]
    sets["jepa_probe_only"] = ["probe_fidelity"]
    return sets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="results/features/feature_matrix.csv")
    p.add_argument("--labels", default="results/utility/labels.csv")
    p.add_argument("--sft-probe", default=None)
    p.add_argument("--envs", default="alfworld,sciworld")
    p.add_argument("--min-states", type=int, default=0, help="drop rows with fewer scored states")
    p.add_argument("--perm", type=int, default=100, help="permutation-test shuffles (main model)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/rq3")
    args = p.parse_args()

    rows = load_rows(args)
    rows = [r for r in rows if r["n_states"] >= args.min_states or r["n_states"] == 0]
    labelled = [r for r in rows if any(r[f"regret_{a}"] is not None for a in ARMS)]
    print(f"[RQ3] {len(rows)} rows ({dict(Counter(r['env'] for r in rows))}), "
          f"{len(labelled)} with regrets; labels: {dict(Counter(r['label'] for r in labelled))}")
    sets = feature_sets(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"rows": len(rows), "labelled": len(labelled), "envs": args.envs,
           "feature_sets": sets, "results": defaultdict(dict)}

    lines = ["| task | CV | features | n | accuracy (95% CI) | balanced | majority | Spearman |",
             "|---|---|---|---|---|---|---|---|"]
    for scheme in ("group", "env"):
        if scheme == "env" and len({r["env"] for r in labelled}) < 2:
            continue
        for task in TASKS:
            for name, feats in sets.items():
                r = evaluate_binary(rows, feats, task, scheme, args.seed)
                if r is None:
                    continue
                res["results"][f"{task}|{scheme}"][name] = r
                lines.append(f"| {task} | {scheme} | {name} | {r['n']} | {r['accuracy']:.2f} "
                             f"({r['wilson95'][0]:.2f}-{r['wilson95'][1]:.2f}) | {r['balanced_acc']:.2f} | "
                             f"{r['majority_acc']:.2f} | {r['spearman']:+.2f} |")
        for name, feats in sets.items():
            r = evaluate_winner(rows, feats, scheme, args.seed)
            if r:
                res["results"][f"winner|{scheme}"][name] = r
                lines.append(f"| winner (G/F/D) | {scheme} | {name} | {r['n']} | {r['accuracy']:.2f} "
                             f"({r['wilson95'][0]:.2f}-{r['wilson95'][1]:.2f}) | | {r['majority_acc']:.2f} | |")

    perm_lines = []
    if args.perm:
        for task in TASKS:
            r = res["results"].get(f"{task}|group", {}).get("jepa_probe")
            if r:
                pv = permutation_p(rows, sets["jepa_probe"], task, "group", r["accuracy"], args.perm, args.seed)
                r["perm_p"] = pv
                perm_lines.append(f"- {task}: accuracy {r['accuracy']:.2f}, permutation p = {pv:.3f} "
                                  f"({args.perm} shuffles)")
    imp = {task: importances(rows, sets.get("both_probes", sets["jepa_probe"]), task, args.seed) for task in TASKS}
    res["importances"] = imp

    with open(out / "rq3_results.json", "w") as f:
        json.dump(res, f, indent=2)
    md = ["# RQ3 predictor results", "",
          f"Rows: {len(rows)} ({args.envs}); with regrets: {len(labelled)}. "
          "CV `group` = leave-one-(env, family)-out; `env` = leave-one-environment-out.", ""] + lines
    if perm_lines:
        md += ["", "**Permutation test** (main model, Features 2-4 + JEPA probe, group CV):"] + perm_lines
    md += ["", "**Feature importance** (fit on all rows; GBT importance / standardised Ridge coefficient):", ""]
    for task, d in imp.items():
        if d:
            top = sorted(d.items(), key=lambda kv: -kv[1]["gbt"])[:6]
            md.append(f"- {task}: " + ", ".join(f"{c} {v['gbt']:.2f}/{v['ridge_coef']:+.3f}" for c, v in top))
    (out / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(lines))
    print("\n".join(perm_lines))
    print(f"\nSaved {out}/rq3_results.json and {out}/README.md")


if __name__ == "__main__":
    main()
