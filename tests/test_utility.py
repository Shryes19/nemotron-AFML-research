"""Decision-utility scorer without a GPU: python -m pytest tests/test_utility.py -q"""
import csv
import json
import random

import pytest

from tests.test_llm import FakeServer
from wm.llm_client import LLMClient
from wm.obs import build_obs
from wm.utility import labels as lab
from wm.utility import score as sc
from wm.utility.judge import LLMJudge, OverlapJudge, judge_messages, parse_value
from wm.utility.predictions import load_predictions

@pytest.fixture(autouse=True)
def _compute_log_in_tmp(tmp_path, monkeypatch):
    """Keep the GPU-hour logger's test entries out of the repo's results/compute log."""
    try:
        import capability.compute_log as cl
        monkeypatch.setattr(cl, "LOG_PATH", str(tmp_path / "compute_log.jsonl"))
    except ImportError:
        pass


OBS = build_obs("put a mug in the sink", "a kitchen", "counter 1", "You see a mug 1.", "nothing",
                ["go to sink 1", "take mug 1 from counter 1"])


def _state(i, actions, true_next, gold=0, env="alfworld"):
    return {"id": "%s/g%d/t%d" % (env, i % 2, i), "env": env, "game": "g%d" % (i % 2),
            "task_type": "pick_and_place_simple-Mug-None-Sink-1", "goal": "put a mug in the sink",
            "obs": OBS, "actions": actions, "true_next": true_next, "gold_index": gold,
            "gold_action": actions[gold]}


ACTS = ["take mug 1 from counter 1", "go to sink 1", "open drawer 1", "examine counter 1", "go to fridge 1"]
TRUE = ["You pick up the mug 1.", "You arrive at sink 1.", "You open the drawer 1.",
        "On the counter 1, you see a mug 1.", "You arrive at fridge 1."]


# ---- judge

def test_parse_value():
    assert parse_value("7") == 7 and parse_value("Score: 10") == 10 and parse_value("0.") == 0
    assert parse_value("no idea") is None and parse_value("") is None


def test_judge_prompt_drops_admissible_and_lists_topk():
    m = judge_messages("goal", OBS, "go to sink 1", ["You arrive at sink 1.", "Nothing happens."])
    text = m[1]["content"]
    assert "Admissible" not in text and "ADMISSIBLE" not in text and "Feedback: You see a mug 1." in text
    assert "1. You arrive at sink 1.\n2. Nothing happens." in text


@pytest.fixture
def judge(tmp_path):
    server = FakeServer()
    server.reply = "7"
    client = LLMClient(base_url=server.url, model="judge", api_key="", backoff=0.01,
                       cache_path=str(tmp_path / "c.sqlite"), log_path=str(tmp_path / "l.jsonl"))
    yield LLMJudge(client), server
    server.httpd.shutdown()


def test_llm_judge_values_parallel_and_cached(judge):
    j, server = judge
    items = [("goal", OBS, a, t) for a, t in zip(ACTS, TRUE)]
    assert j.values(items, workers=4) == [7] * 5
    calls = server.calls
    assert j.values(items, workers=4) == [7] * 5 and server.calls == calls   # all cached


def test_llm_judge_unparsable_gives_none(judge):
    j, server = judge
    server.reply = "hard to say"
    assert j.value("goal", OBS, "go to sink 1", "You arrive at sink 1.") is None


# ---- predictions

def test_generic_predictions_reordered_and_checked(tmp_path):
    s = _state(0, ACTS, TRUE)
    states = {s["id"]: s}
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": s["id"], "arm": "G", "budget": 32,
                                "actions": ACTS[::-1], "pred_next": TRUE[::-1]}) + "\n")
    assert load_predictions(str(path), states)[s["id"]]["pred"] == TRUE
    path.write_text(json.dumps({"id": s["id"], "budget": 32, "actions": ACTS[:4] + ["jump"],
                                "pred_next": TRUE}) + "\n")
    with pytest.raises(ValueError):
        load_predictions(str(path), states)
    path.write_text("")
    with pytest.raises(ValueError):
        load_predictions(str(path), states)          # missing state


def test_jepa_predictions_modes(tmp_path):
    s = _state(0, ACTS, TRUE)
    order = [2, 0, 1, 4, 3]                          # JEPA lists candidates in its own order
    rec = {"id": s["id"], "budget": 256,
           "candidates": [{"action": ACTS[i], "topk": [{"text": "top1 %d" % i, "sim": 0.9},
                                                        {"text": "top2 %d" % i, "sim": 0.8}]} for i in order],
           "ranking": [1, 0, 2, 3, 4]}               # best = its candidate 1 = ACTS[0]
    path = tmp_path / "j.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    states = {s["id"]: s}
    top1 = load_predictions(str(path), states, fmt="jepa", mode="top1")[s["id"]]
    assert top1["pred"] == ["top1 %d" % i for i in range(5)] and top1["budget"] == 256
    assert top1["ranking"][0] == 0
    topk = load_predictions(str(path), states, fmt="jepa", mode="topk")[s["id"]]
    assert topk["pred"][3] == ["top1 3", "top2 3"]


# ---- scoring maths

def test_score_state_regret_gold_and_fidelity():
    s = _state(0, ACTS, TRUE, gold=0)
    tv = [9, 3, 1, 2, 3]
    perfect = sc.score_state(s, tv, tv, TRUE)
    assert perfect["regret"] == 0 and perfect["gold_hit"] == 1 and perfect["em"] == 1.0
    wrong = sc.score_state(s, tv, [0, 0, 5, 0, 0], ["x"] * 5)
    assert wrong["chosen"] == 2 and wrong["regret"] == 8 and wrong["gold_hit"] == 0 and wrong["em"] == 0
    assert sc.score_state(s, [9, None, 1, 2, 3], tv) is None          # incomplete true values
    assert sc.score_state(s, tv, [None] * 5) is None                   # nothing judged


def test_ties_are_seeded_and_shared_across_arms():
    a = [sc.choose([5, 5, 1, 5, 0], sc.tie_rng("state-x")) for _ in range(5)]
    assert len(set(a)) == 1 and a[0] in (0, 1, 3)


def test_word_f1():
    assert sc.word_f1("you arrive at sink 1", "You arrive at sink 1.") == 1.0
    assert 0 < sc.word_f1("you arrive at sink 2", "You arrive at sink 1.") < 1
    assert sc.word_f1("nothing", "You arrive") == 0.0


# ---- end to end with the overlap judge

def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_end_to_end_perfect_matches_oracle_and_shuffled_is_worse(tmp_path):
    states = [_state(i, ACTS, TRUE, gold=i % 5) for i in range(6)]
    spath = tmp_path / "states.jsonl"
    _write(spath, states)
    tv = tmp_path / "tv.jsonl"
    assert sc.main(["true-values", "--states", str(spath), "--out", str(tv), "--judge", "overlap"]) == 0
    perfect = tmp_path / "perfect.jsonl"
    _write(perfect, [{"id": s["id"], "arm": "P", "budget": 256, "actions": ACTS, "pred_next": TRUE} for s in states])
    rng = random.Random(0)
    shuffled = tmp_path / "shuffled.jsonl"
    _write(shuffled, [{"id": s["id"], "arm": "S", "budget": 256, "actions": ACTS,
                       "pred_next": rng.sample(TRUE, 5)} for s in states])
    for arm, path in (("P", perfect), ("S", shuffled)):
        sc.main(["score", "--states", str(spath), "--true-values", str(tv), "--pred", str(path),
                 "--arm", arm, "--judge", "overlap", "--out", str(tmp_path / ("scored_%s.jsonl" % arm))])
    out = tmp_path / "res"
    sc.main(["summarize", "--states", str(spath), "--true-values", str(tv),
             "--scored", str(tmp_path / "scored_*.jsonl"), "--out-dir", str(out)])
    rows = {(r["arm"], r["game"]): r for r in csv.DictReader(open(out / "summary.csv"))}
    assert float(rows[("P", "*")]["mean_regret"]) == 0 == float(rows[("oracle", "*")]["mean_regret"])
    assert rows[("P", "*")]["gold_hit"] == rows[("oracle", "*")]["gold_hit"]
    assert float(rows[("S", "*")]["mean_regret"]) > 0
    assert float(rows[("P", "*")]["em"]) == 1.0
    dis = list(csv.DictReader(open(out / "fidelity_vs_utility.csv")))
    assert dis and dis[0]["best_by_utility"].startswith("P")


def test_labels_pick_lowest_regret_and_break_ties():
    rows = []
    for game, regrets in (("g0", {"G": 2, "F": 1, "D": 3}), ("g1", {"G": 1, "F": 1, "D": 1})):
        for arm, reg in regrets.items():
            rows.append({"env": "sciworld", "game": game, "family": game, "arm": arm, "budget": 256,
                         "mode": "top1", "regret": reg, "gold_hit": int(arm == "D")})
    out = {r["game"]: r for r in lab.make_labels(rows, ["G", "F", "D"], 256)}
    assert out["g0"]["label"] == "F"
    assert out["g1"]["label"] == "D"           # tie on regret -> higher gold-hit
    assert {"env", "game", "label"} <= set(out["g0"])
