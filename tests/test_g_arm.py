"""Arm G without a GPU or NV-Embed: python -m pytest tests/test_g_arm.py -q"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_dropin"))
import generate_predictions as gen  # noqa: E402
import prepare_g_prompts as prep  # noqa: E402
import rag_arm  # noqa: E402

from wm.utility.predictions import load_predictions  # noqa: E402


class FakeEncoder:
    """Bag-of-words hashing vectors: similar text -> similar vector, no download."""
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture(autouse=True)
def _no_compute_log(monkeypatch):
    monkeypatch.setenv("WM_NO_COMPUTE_LOG", "1")


def test_reads_obs_next_and_go_to_object():
    t = rag_arm.transition({"obs": "o", "action": "go to drawer 4", "obs_next": "You arrive at drawer 4."})
    assert t.next_observation == "You arrive at drawer 4."
    assert rag_arm.action_object("go to drawer 4") == "drawer"
    assert not rag_arm.relevant("go to drawer 4", "go to sink 1")
    assert rag_arm.relevant("take mug 1 from counter 1", "take mug 2 from table 1")


def test_retrieval_puts_matching_example_in_prompt():
    rows = [{"obs": "kitchen counter", "action": "take mug 1", "obs_next": "You pick up the mug 1."},
            {"obs": "garden shed", "action": "open door", "obs_next": "The door is open."}] * 2
    r = rag_arm.HierarchicalRetriever(rows, encoder=FakeEncoder())
    text = rag_arm.prompt("kitchen counter", "take mug 1", r.retrieve("kitchen counter", "take mug 1", k=1))
    assert "Next observation: You pick up the mug 1." in text
    assert len(rag_arm.clip("x" * 5000)) < 1600 and len(rag_arm.clip("x" * 5000, 4000)) < 4100


def _setup(tmp_path, n_states=3):
    states = [{"id": f"s{i}", "obs": f"room {i} with a mug", "task_id": f"t{i % 2}", "gold_action": f"take mug {i}",
               "actions": [f"take mug {i}", "look", f"go to sink {i}"]} for i in range(n_states)]
    (tmp_path / "eval").mkdir()
    sp = tmp_path / "eval" / "alfworld_branched_states.jsonl"
    sp.write_text("".join(json.dumps(s) + "\n" for s in states))
    rows = tmp_path / "rows"
    rows.mkdir()
    for n in (32, 64):
        (rows / f"alfworld_b{n}.jsonl").write_text("".join(
            json.dumps({"obs": f"room {i}", "action": f"take mug {i}", "obs_next": f"got mug {i}"}) + "\n"
            for i in range(n)))
    return states, sp, rows


def test_two_stages_write_scorable_file(tmp_path, monkeypatch):
    states, sp, rows = _setup(tmp_path)
    monkeypatch.setattr(prep, "NVEmbedEncoder", FakeEncoder)
    monkeypatch.setattr(sys, "argv", ["prep", "--rows", str(rows), "--out-dir", str(tmp_path / "prompts"),
                                      "--sets", "alfworld", "--budgets", "32 64",
                                      "--states-dir", str(tmp_path / "eval")])
    prep.main()
    cache = [json.loads(l) for l in (tmp_path / "prompts" / "alfworld_b32.jsonl").read_text().splitlines()]
    assert [c["id"] for c in cache] == ["s0", "s1", "s2"] and len(cache[0]["prompts"]) == 3
    assert "Next observation: got mug 0" in cache[0]["prompts"][0]  # the example's obs_next is in the prompt
    recall = list(csv.DictReader((tmp_path / "prompts" / "recall_at5.csv").open()))
    assert {r["budget"] for r in recall} == {"32", "64"} and {r["game"] for r in recall} == {"t0", "t1"}

    class FakeModel:
        def predict(self, text):
            return "PRED " + text.rsplit("Action: ", 1)[1].split("\n")[0]

    monkeypatch.setattr(gen, "VLLMQwen", FakeModel)
    monkeypatch.setattr(sys, "argv", ["gen", "--prompts", str(tmp_path / "prompts" / "alfworld_b32.jsonl"),
                                      "--states", str(sp), "--set", "alfworld", "--budget", "32",
                                      "--outdir", str(tmp_path / "out"), "--workers", "2"])
    gen.main()
    out = tmp_path / "out" / "alfworld_b32.jsonl"
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert recs[0]["pred_next"] == ["PRED take mug 0", "PRED look", "PRED go to sink 0"]
    preds = load_predictions(str(out), {s["id"]: s for s in states}, fmt="generic", budget=32)
    assert set(preds) == {"s0", "s1", "s2"}


def test_generator_stops_without_writing_when_server_fails(tmp_path, monkeypatch):
    states, sp, rows = _setup(tmp_path)
    monkeypatch.setattr(prep, "NVEmbedEncoder", FakeEncoder)
    monkeypatch.setattr(sys, "argv", ["prep", "--rows", str(rows), "--out-dir", str(tmp_path / "prompts"),
                                      "--sets", "alfworld", "--budgets", "32", "--states-dir", str(tmp_path / "eval")])
    prep.main()

    class Down:
        def predict(self, text):
            raise RuntimeError("LLM call failed after retries: connection refused")

    monkeypatch.setattr(gen, "VLLMQwen", Down)
    monkeypatch.setattr(sys, "argv", ["gen", "--prompts", str(tmp_path / "prompts" / "alfworld_b32.jsonl"),
                                      "--states", str(sp), "--set", "alfworld", "--budget", "32",
                                      "--outdir", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        gen.main()
    assert not (tmp_path / "out" / "alfworld_b32.jsonl").exists()


def test_prepare_refuses_wrong_row_count(tmp_path, monkeypatch):
    _, _, rows = _setup(tmp_path)
    (rows / "alfworld_b32.jsonl").write_text(json.dumps({"obs": "o", "action": "a", "obs_next": "n"}) + "\n")
    monkeypatch.setattr(prep, "NVEmbedEncoder", FakeEncoder)
    monkeypatch.setattr(sys, "argv", ["prep", "--rows", str(rows), "--out-dir", str(tmp_path / "p"),
                                      "--sets", "alfworld", "--budgets", "32", "--states-dir", str(tmp_path / "eval")])
    with pytest.raises(SystemExit):
        prep.main()
