"""Arm G generator without a GPU or the BGE model: python -m pytest tests/test_g_arm.py -q"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_dropin"))
import generate_predictions as gen  # noqa: E402
import rag_arm  # noqa: E402

from wm.utility.predictions import load_predictions  # noqa: E402


class FakeEncoder:
    """Bag-of-words hashing vectors: similar text -> similar vector, no download."""
    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


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
    assert rag_arm.clip("x" * 5000).endswith(" ...") and len(rag_arm.clip("x" * 5000)) < 3100


def test_generator_writes_scorable_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WM_NO_COMPUTE_LOG", "1")
    states = [{"id": f"s{i}", "obs": f"room {i} with a mug", "actions": [f"take mug {i}", "look", f"go to sink {i}"]}
              for i in range(3)]
    sp, bp = tmp_path / "states.jsonl", tmp_path / "rows.jsonl"
    sp.write_text("".join(json.dumps(s) + "\n" for s in states))
    bp.write_text("".join(json.dumps({"obs": f"room {i}", "action": f"take mug {i}", "obs_next": f"got mug {i}"}) + "\n"
                          for i in range(32)))

    class FakeModel:
        def predict(self, text):
            return "PRED " + text.rsplit("Action: ", 1)[1].split("\n")[0]

    monkeypatch.setattr(gen, "VLLMQwen", FakeModel)
    monkeypatch.setattr(gen, "HierarchicalRetriever",
                        lambda buf, device=None: rag_arm.HierarchicalRetriever(buf, encoder=FakeEncoder()))
    monkeypatch.setattr(sys, "argv", ["gen", "--buffer", str(bp), "--states", str(sp), "--set", "alfworld",
                                      "--budget", "32", "--outdir", str(tmp_path / "out"), "--workers", "2"])
    gen.main()
    out = tmp_path / "out" / "alfworld_b32.jsonl"
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["pred_next"] for r in recs][0] == ["PRED take mug 0", "PRED look", "PRED go to sink 0"]
    preds = load_predictions(str(out), {s["id"]: s for s in states}, fmt="generic", budget=32)
    assert set(preds) == {"s0", "s1", "s2"}


def test_generator_refuses_wrong_row_count(tmp_path, monkeypatch):
    bp = tmp_path / "rows.jsonl"
    bp.write_text(json.dumps({"obs": "o", "action": "a", "obs_next": "n"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["gen", "--buffer", str(bp), "--states", str(bp), "--set", "alfworld",
                                      "--budget", "32", "--outdir", str(tmp_path)])
    with pytest.raises(SystemExit):
        gen.main()
