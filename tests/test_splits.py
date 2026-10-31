"""Split hygiene, no environment needed: python -m pytest tests/test_splits.py -q"""
from wm.collect import load_task_ids


def test_train_and_heldout_task_disjoint():
    for env in ("alfworld", "sciworld"):
        train, held = load_task_ids(env, "train"), load_task_ids(env, "heldout")
        assert len(held) == 200 and not set(train) & set(held)
    assert len(load_task_ids("sciworld", "train")) == 2120 - 61
