"""Budget nesting and mix. No environment needed: python -m pytest tests/test_budgets.py -q"""
import random

import pytest

from wm.policies import failing_action

from wm.budgets import allocate, make_budgets

MIX = {"expert": 0.5, "random": 0.3, "llm": 0.2}
BUDGETS = [32, 64, 128, 256, 512, 1024]
POOL = {"expert": ["e%d" % i for i in range(800)],
        "random": ["r%d" % i for i in range(480)],
        "llm": ["l%d" % i for i in range(320)]}


def test_allocate_sums_and_mix():
    for n in BUDGETS:
        c = allocate(n, MIX)
        assert sum(c.values()) == n
        for k in MIX:
            assert abs(c[k] - n * MIX[k]) <= 1
    assert allocate(32, MIX) == {"expert": 16, "random": 10, "llm": 6}


def test_nested_and_seed_dependent():
    by_seed = {s: make_budgets(POOL, BUDGETS, s, MIX) for s in (0, 1, 2)}
    for s, b in by_seed.items():
        for small, big in zip(BUDGETS, BUDGETS[1:]):
            assert set(b[small]) <= set(b[big])
        for n in BUDGETS:
            assert len(set(b[n])) == n
    for n in BUDGETS:
        assert set(by_seed[0][n]) != set(by_seed[1][n])
    assert make_budgets(POOL, BUDGETS, 0, MIX) == by_seed[0]  # deterministic


def test_provisional_and_too_small_pool():
    mix = {"expert": 0.5, "random": 0.3}
    b = make_budgets(POOL, BUDGETS, 0, mix)
    assert sum(x.startswith("e") for x in b[1024]) == 640
    with pytest.raises(ValueError):
        make_budgets({"expert": ["e1"], "random": [], "llm": []}, [32], 0, MIX)


def test_failing_action_is_well_formed_and_not_admissible():
    obs = "[ROOM] you see a cabinet 1, a drawer 2 [FEEDBACK] On it, you see a mug 1."
    adm = ["go to cabinet 1", "go to drawer 2", "take mug 1 from cabinet 1"]
    for seed in range(20):
        a = failing_action(obs, adm, random.Random(seed))
        assert a and a not in adm
        assert all(w in {"cabinet", "drawer", "mug", "1", "2", "take", "from", "move", "to", "open",
                         "close", "use", "examine", "heat", "cool", "clean", "slice", "with", "go"}
                   for w in a.split())
