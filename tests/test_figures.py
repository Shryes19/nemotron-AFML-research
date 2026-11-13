"""Phase 3 figures on a tiny synthetic utility output: python -m pytest tests/test_figures.py -q"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_figures as mf  # noqa: E402


def test_make_figures_end_to_end(tmp_path):
    util, out = tmp_path / "util", tmp_path / "fig"
    util.mkdir()
    states = tmp_path / "s.jsonl"
    states.write_text("".join(json.dumps({"id": "a%d" % i, "env": "alfworld", "split": "heldout"}) + "\n"
                              for i in range(4)))
    (util / "alfworld_true_values.jsonl").write_text(
        "".join(json.dumps({"id": "a%d" % i, "values": [9, 1, 1, 1, 0]}) + "\n" for i in range(4)))
    rows = []
    for arm, mode, base in (("G", "top1", 1.0), ("F", "top1", 0.8), ("D", "top1", 1.2), ("D", "native", 0.9)):
        for k, n in enumerate((32, 256)):
            reg = base - (0.5 if arm == "D" else 0.1) * k
            for game in ("*", "pick_and_place_simple"):
                rows.append({"env": "alfworld", "split": "heldout", "arm": arm, "mode": mode, "budget": n,
                             "game": game, "family": game, "n_states": 4, "mean_regret": reg,
                             "regret_ci_lo": reg - .1, "regret_ci_hi": reg + .1, "gold_hit": .3,
                             "em": "" if mode == "native" else .5 - reg / 10, "f1": ""})
    rows.append({"env": "alfworld", "split": "heldout", "arm": "oracle", "mode": "true", "budget": "-",
                 "game": "*", "family": "*", "n_states": 4, "mean_regret": 0, "regret_ci_lo": 0,
                 "regret_ci_hi": 0, "gold_hit": .5, "em": 1, "f1": 1})
    with open(util / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    feats = tmp_path / "fm.csv"
    feats.write_text("env,game," + ",".join(mf.RATES) + "\nalfworld,pick_and_place_simple-Mug-1,"
                     + ",".join(["0.2"] * 5) + "\n")
    (util / "labels_Dtop1.csv").write_text("env,game,label\nalfworld,pick_and_place_simple,D\n")
    assert mf.main(["--utility-dir", str(util), "--states", str(states), "--features", str(feats),
                    "--out-dir", str(out)]) == 0
    for name in ("fig1_regret_vs_n.pdf", "fig2_fidelity_vs_utility.png", "fig3_capability_vs_dwins.pdf",
                 "main_table.md"):
        assert (out / name).exists()
    cross = {(r["incumbent"], r["challenger"]): r for r in csv.DictReader(open(out / "crossovers.csv"))}
    assert cross[("F/top1", "D/top1")]["first_n_challenger_lower"] == "256"
    assert cross[("G/top1", "F/top1")]["first_n_challenger_lower"] == "32"
    table = open(out / "main_table.md").read()
    assert "random choice | 4 | 6.60 [6.60, 6.60]" in table and "D (JEPA) native" in table
