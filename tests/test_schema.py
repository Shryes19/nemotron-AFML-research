"""Schema checks on the committed samples. No environment needed:
  python -m pytest tests/test_schema.py -q
"""
import glob

from wm.obs import SECTIONS, build_obs, canonical, parse_obs
from wm.schema import read_jsonl, validate_row


def test_samples_valid():
    paths = glob.glob("data/sample/*.jsonl")
    assert paths
    for path in paths:
        rows = list(read_jsonl(path))
        assert len(rows) == 20
        for row in rows:
            assert validate_row(row) == [], (path, row["transition_id"])
            # ALFWorld random steps may use a non-admissible command, which must fail
            assert row["a_t"] in row["admissible_actions"] or row["feedback_next"] == "Nothing happens."
            assert row["split"] == "train"


def test_obs_roundtrip():
    text = build_obs("g", "room  text", "loc", "fb\nline2", "inv", ["a", "b"])
    parsed = parse_obs(text)
    assert list(parsed) == SECTIONS
    assert parsed["ROOM"] == "room text" and parsed["ADMISSIBLE"] == "a | b"
    assert canonical(text) == canonical(text.replace("a | b", "b | a"))
