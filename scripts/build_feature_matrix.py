#!/usr/bin/env python3
"""
Assemble the RQ3 feature matrix: one row per (env, game), ex-ante features measured at n=32, plus the
winner label once the utility table exists.

    python scripts/build_feature_matrix.py \
        --mix results/capability/capability_mix.csv \
        --stats results/features/env_stats.csv \
        [--probe  results/jepa_phase1/probe_scores.csv]   # Feature 1: columns env, game, probe_fidelity
        [--recall results/rag/recall_at5.csv]             # Feature 3: columns env, game, recall_at5
        [--labels results/utility/labels.csv]             # columns env, game, label (G / F / D at n=256)
        --out results/features/feature_matrix.csv

Feature 1 / 3 / labels are optional so the matrix can be built as each piece lands; missing values are
left empty and listed at the end. Rows are joined on (env, game); a row with game "*" in an optional file
applies to every game of that env.
"""
import argparse
import csv
import os

MIX_COLS = ["rate_object_state", "rate_containment_location", "rate_numeric_physical",
            "rate_partial_obs", "rate_action_failure"]
STAT_COLS = ["obs_len_chars", "action_vocab", "zero_overlap_frac", "mean_jaccard", "target_repeat_frac"]


def read(path, n=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if n is not None and rows and "n" in rows[0]:
        rows = [r for r in rows if r["n"] == str(n)]
    return rows


def index(rows):
    return {(r["env"], r["game"]): r for r in rows}


def family_of(env, game):
    """ALFWorld task id -> task type ("pick_heat_then_place_in_recep-Mug-...-20_trial_T..." ->
    "pick_heat_then_place_in_recep"), same rule as wm/utility/score.py. Other envs: unchanged."""
    return game.split("-")[0] if env == "alfworld" else game


def lookup(idx, env, game, col):
    """Exact (env, game) first, then (env, family) - ALFWorld labels are per task type because the
    held-out scoring games are not the 12 training games - then (env, "*")."""
    r = idx.get((env, game)) or idx.get((env, family_of(env, game))) or idx.get((env, "*"))
    return r.get(col, "") if r else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--recall")
    ap.add_argument("--labels")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="results/features/feature_matrix.csv")
    args = ap.parse_args()

    mix, stats = index(read(args.mix, args.n)), index(read(args.stats, args.n))
    extra = {name: index(read(path)) for name, path in
             [("probe", args.probe), ("recall", args.recall), ("labels", args.labels)] if path}
    keys = sorted(set(mix) & set(stats))
    rows, missing = [], {}
    for env, game in keys:
        row = {"env": env, "game": game}
        row.update({c: mix[(env, game)][c] for c in MIX_COLS})
        row.update({c: stats[(env, game)][c] for c in STAT_COLS})
        row["probe_fidelity"] = lookup(extra.get("probe", {}), env, game, "probe_fidelity")
        row["recall_at5"] = lookup(extra.get("recall", {}), env, game, "recall_at5")
        row["label"] = lookup(extra.get("labels", {}), env, game, "label")
        for c in ("probe_fidelity", "recall_at5", "label"):
            if row[c] == "":
                missing[c] = missing.get(c, 0) + 1
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} games x {len(rows[0]) - 2} columns -> {args.out}")
    for env in sorted({r["env"] for r in rows}):
        print(f"  {env}: {sum(r['env'] == env for r in rows)} games")
    for c, k in missing.items():
        print(f"  missing {c}: {k} of {len(rows)} rows")


if __name__ == "__main__":
    main()
