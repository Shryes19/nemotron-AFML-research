"""Buffer export format, no environment needed: python -m pytest tests/test_export.py -q"""
import json

from wm.export_buffers import family_of, main, to_buffer_row
from wm.schema import read_jsonl

BUFFER_FIELDS = {"obs", "action", "obs_next", "env", "game", "cap_tag"}


def test_row_mapping_defaults():
    row = next(read_jsonl("data/sample/alfworld_sample20.jsonl"))
    b = to_buffer_row(row)
    assert BUFFER_FIELDS <= set(b)
    assert (b["obs"], b["action"]) == (row["o_t"], row["a_t"])
    assert b["obs_next"] == row["feedback_next"]          # exactly what the env returned
    assert b["obs_next_full"] == row["o_next"]
    assert b["game"] == row["task_id"] and b["cap_tag"] == ""


def test_row_mapping_options():
    row = next(read_jsonl("data/sample/alfworld_sample20.jsonl"))
    b = to_buffer_row(row, game="family", obs_next="composite")
    assert b["obs_next"] == row["o_next"]
    assert b["game"] in {"pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                         "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"}
    sci = next(read_jsonl("data/sample/sciworld_sample20.jsonl"))
    assert family_of(sci) == sci["task_type"]


def test_single_file(tmp_path):
    out = tmp_path / "buf.jsonl"
    main(["--env", "alfworld", "--train", "data/sample/alfworld_sample20.jsonl", "--single", str(out)])
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 20 and all(BUFFER_FIELDS <= set(r) for r in rows)


def test_budget_files(tmp_path):
    rows = list(read_jsonl("data/sample/alfworld_sample20.jsonl"))
    train, held = tmp_path / "train.jsonl", tmp_path / "held.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    held.write_text("".join(json.dumps(dict(r, split="heldout")) + "\n" for r in rows[:5]))
    bdir = tmp_path / "budgets"
    bdir.mkdir()
    for s in (0, 1, 2):
        for n in (32, 64, 128, 256, 512, 1024):  # tiny stand-in budgets over the 20 sample ids
            (bdir / ("alfworld_seed%d_n%d.txt" % (s, n))).write_text(
                "\n".join(r["transition_id"] for r in rows[: 20 - s]) + "\n")
    out = tmp_path / "buffers"
    main(["--env", "alfworld", "--train", str(train), "--heldout", str(held),
          "--budget-dir", str(bdir), "--out-dir", str(out)])
    assert len(open(out / "alfworld_seed0_n32.jsonl").readlines()) == 20
    assert len(open(out / "alfworld_heldout.jsonl").readlines()) == 5
