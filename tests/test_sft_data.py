"""SFT arm data helpers, no model needed: python -m pytest tests/test_sft_data.py -q"""
import json

import pytest

from sft.data import (allowed_flag, check_rows, fidelity, holdout_games_split, pick_prediction_fields,
                      strip_admissible, user_text)
from sft.model import fit_length, steps_for

ALF = ("[GOAL] put a mug in the fridge.\n[ROOM] You are in the kitchen.\n[LOCATION] kitchen\n"
       "[FEEDBACK] Nothing happens.\n[INVENTORY] You are not carrying anything.\n"
       "[ADMISSIBLE] go to fridge 1 | open fridge 1 | look")
SCI = "[GOAL] boil water.\n[ROOM] kitchen.\n[ADMISSIBLE] open OBJ | close OBJ | look around\n[INVENTORY] nothing"
BS32 = "[GOAL] g\n[ROOM] r\n[FEEDBACK] f\n[INVENTORY] i"


def test_strip_admissible_removes_only_the_list():
    s = strip_admissible(ALF)
    assert "[ADMISSIBLE]" not in s and "go to fridge 1 |" not in s
    assert s.endswith("[INVENTORY] You are not carrying anything.")
    s = strip_admissible(SCI)                      # list in the middle: next section kept
    assert "[ADMISSIBLE]" not in s and s.endswith("[INVENTORY] nothing")
    assert strip_admissible(BS32) == BS32


def test_allowed_flag_matches_jepa_feature():
    assert allowed_flag(ALF, "Open Fridge 1 ") == "yes"
    assert allowed_flag(ALF, "eat fridge 1") == "no"
    assert allowed_flag(SCI, "open door") == "unknown"     # templates count as no list
    assert allowed_flag(BS32, "take box") == "unknown"


def test_user_text_has_obs_action_and_flag():
    t = user_text(ALF, "open fridge 1")
    assert "Action: open fridge 1\nAction allowed: yes" in t and "[ADMISSIBLE]" not in t


def test_fit_length_cuts_middle_of_prompt_and_long_answers():
    p, a, cut = fit_length(list(range(100)), [7] * 10, max_len=50, keep_head=8)
    assert cut and len(p) + len(a) == 50 and p[:8] == list(range(8)) and p[-1] == 99
    p, a, cut = fit_length([1] * 5, list(range(300)) + [9], max_len=1000, max_answer=20)
    assert cut and len(a) == 20 and a[-1] == 9            # end-of-turn token kept
    assert fit_length([1, 2], [3], 10)[2] is False


def test_steps_for_epochs_with_floor():
    cfg = {"batch_size": 16, "epochs": 5, "min_steps": 100}
    assert steps_for(32, cfg) == 100                        # floor wins at tiny budgets
    assert steps_for(1024, cfg) == 320                      # 5 epochs x 64 steps
    assert steps_for(32, {"batch_size": 16, "epochs": 5}) == 10


def test_check_rows_rejects_wrong_buffer():
    rows = {"n_buffer_rows": 50, "budgets": {2: [0, 1]}}
    check_rows(rows, 50)
    with pytest.raises(SystemExit):
        check_rows(rows, 60)


def test_fidelity_uses_saanvi_normalisation():
    r = fidelity(["You open the fridge 1", "x"], ["you open the fridge 1.", "y"])
    assert r["exact"] == 0.5 and r["n"] == 2


def test_holdout_split_holds_whole_games():
    items = [{"game": f"g{i % 5}"} for i in range(50)]
    val, pool, held = holdout_games_split(items, 2)
    assert len(held) == 2 and not set(val) & set(pool) and len(val) + len(pool) == 50
    assert all(items[i]["game"] in held for i in val)


def test_prediction_fields_never_include_answers():
    with open("data/eval/alfworld_branched_states.jsonl") as f:
        s = json.loads(f.readline())
    q = pick_prediction_fields(s)
    assert set(q) == {"id", "obs", "actions"} and len(q["actions"]) == 5
