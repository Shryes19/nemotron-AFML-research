"""Collector logic with a fake env, no real environment needed: python -m pytest tests/test_collect.py -q"""
import random
from collections import Counter

from wm.collect import MAX_EXPERT_PREFIX, pick_balanced, run_episode
from wm.envs.base import StepResult, TextEnv
from wm.obs import build_obs
from wm.policies import ExpertPolicy, RandomPolicy


def test_pick_balanced_fewest_rows_then_round_robin():
    groups = {"boil": ["s1", "s2"], "melt": ["s3"], "grow": ["s4", "s5", "s6"]}
    rows, cursor = Counter({"boil": 5, "melt": 5, "grow": 0}), Counter()
    assert pick_balanced(groups, rows, cursor) == ("grow", "s4")
    assert pick_balanced(groups, rows, cursor) == ("grow", "s5")   # next variation of the same type
    rows["grow"] = 9
    assert pick_balanced(groups, rows, cursor) == ("boil", "s1")   # tie boil/melt -> by name
    rows["boil"] = 9
    assert pick_balanced(groups, rows, cursor) == ("melt", "s3")
    rows["melt"] = 9
    assert pick_balanced(groups, rows, cursor) == ("boil", "s2")
    assert pick_balanced(groups, rows, cursor)[1] in {"s6", "s1"}


class FakeEnv(TextEnv):
    """A 60-step gold path; every action is admissible and nothing ends the episode."""
    name, version = "fake", "fake"

    def __init__(self, gold_len=60):
        self.gold = ["gold %d" % i for i in range(gold_len)]

    def _res(self, fb):
        return StepResult(obs=build_obs("g", "r", "l", fb, "i", ["a", "b"]), feedback=fb,
                          admissible=["a", "b"])

    def reset(self, task_id):
        self.t, self.on_gold = 0, True
        return self._res("start")

    def step(self, action):
        self.on_gold = self.on_gold and self.t < len(self.gold) and action == self.gold[self.t]
        self.t += 1
        return self._res("did %s at %d" % (action, self.t))

    def admissible_actions(self):
        return ["a", "b"]

    def expert_action(self):
        return self.gold[self.t] if self.on_gold and self.t < len(self.gold) else None

    def task_type(self):
        return "fake-type"

    def goal(self):
        return "g"

    def gold_length(self):
        return len(self.gold)


def _handover_step(late, ep):
    rng = random.Random(ep)
    need = {"expert": 10 ** 6, "random": 10 ** 6}
    rows = run_episode(FakeEnv(), "fake", "train", "t1", ep, "random",
                       {"expert": ExpertPolicy(), "random": RandomPolicy()}, rng, 0, need,
                       max_steps=80, late_prefix=late)
    experts = [r for r in rows if r["policy"] == "expert"]
    return len(experts), len(rows)


def test_late_prefix_reaches_late_gold_steps():
    default = [_handover_step(False, ep)[0] for ep in range(200)]
    late = [_handover_step(True, ep)[0] for ep in range(200)]
    assert max(default) <= MAX_EXPERT_PREFIX
    assert max(late) > 40 and max(late) <= 60          # hands over anywhere along the 60-step path
    assert all(n <= 80 for _, n in (_handover_step(True, ep) for ep in range(20)))
