"""End-to-end run of sanity/three_shot.py on the ALFWorld sample with a fake LLM server:
  python -m pytest tests/test_sanity.py -q
"""
import json

from sanity import three_shot
from tests.test_llm import FakeServer
from wm.retrieval import BM25
from wm.schema import read_jsonl


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _fixture_files(tmp_path):
    rows = list(read_jsonl("data/sample/alfworld_sample20.jsonl"))
    train, held = rows[:12], []
    for r in rows[12:]:  # turn the rest into held-out rows from different tasks
        r = dict(r, split="heldout", task_id="alfworld_9999")
        held.append(r)
    _write(tmp_path / "train.jsonl", train)
    _write(tmp_path / "held.jsonl", held)
    (tmp_path / "budget.txt").write_text("\n".join(r["transition_id"] for r in train) + "\n")
    return ["--train", str(tmp_path / "train.jsonl"), "--heldout", str(tmp_path / "held.jsonl"),
            "--budget", str(tmp_path / "budget.txt"), "--n", "8"]


def test_bm25_prefers_overlap():
    bm = BM25(["open the fridge", "go to desk", "take mug from desk"])
    assert bm.top_k("take mug", 1) == [2]


def test_baselines_only(tmp_path):
    out = tmp_path / "res.json"
    three_shot.main(_fixture_files(tmp_path) + ["--baselines-only", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["heldout_size"] == 8 and set(res["baselines"]) == {"copy_top1", "nothing_happens"}


def test_full_run_with_fake_llm(tmp_path, monkeypatch):
    server = FakeServer()
    server.reply = "Nothing happens."
    monkeypatch.setenv("LLM_BASE_URL", server.url)
    monkeypatch.chdir(tmp_path)
    import shutil, os
    os.makedirs("configs")
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "configs", "shared.yaml"), "configs/")
    os.makedirs("data/sample")
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "data", "sample", "alfworld_sample20.jsonl"), "data/sample/")
    out = tmp_path / "res.json"
    three_shot.main(_fixture_files(tmp_path) + ["--out", str(out)])
    res = json.loads(out.read_text())
    # the fake model always says "Nothing happens." so it ties the baseline and must FAIL rule 2
    assert res["model"]["em"] == res["baselines"]["nothing_happens"]["em"]
    assert res["rule"]["2_beats_both_baselines_by_10pts"] is False and res["rule"]["PASS"] is False
    server.httpd.shutdown()
