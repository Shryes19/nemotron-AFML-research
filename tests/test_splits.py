"""Split hygiene, no environment needed: python -m pytest tests/test_splits.py -q"""
from wm.collect import load_task_ids


def test_train_and_heldout_task_disjoint():
    for env in ("alfworld", "sciworld"):
        train, held = load_task_ids(env, "train"), load_task_ids(env, "heldout")
        assert len(held) == 200 and not set(train) & set(held)
    from wm.collect import reserved_seen
    assert len(load_task_ids("sciworld", "train")) == 2120 - 61 - len(reserved_seen("sciworld"))


def test_heldout_seen_is_disjoint_from_train_and_heldout():
    import os
    from wm.collect import reserved_seen
    seen = set(load_task_ids("sciworld", "heldout_seen"))
    assert seen == set(reserved_seen("sciworld"))
    if not os.path.exists("splits/sciworld_heldout_seen.txt"):
        return  # nothing reserved yet
    assert seen and not seen & set(load_task_ids("sciworld", "train"))
    assert not seen & set(load_task_ids("sciworld", "heldout"))
