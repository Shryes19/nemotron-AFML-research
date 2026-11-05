"""Branched-state generation with a fake env: python -m pytest tests/test_branch_states.py -q"""
import json
import random

import wm.branch_states as bs
from wm.envs.base import StepResult, TextEnv
from wm.obs import build_obs

ACTIONS = ["go to desk 1", "go to bed 1", "open drawer 1", "take pen 1 from desk 1",
           "examine desk 1", "use desklamp 1", "help"]


def test_choose_candidates_includes_expert_distinct_and_seeded():
    for seed in range(20):
        c = bs.choose_candidates("open drawer 1", ACTIONS, 5, random.Random(seed))
        assert len(c) == 5 == len(set(c)) and "open drawer 1" in c and "help" not in c
        assert c == bs.choose_candidates("open drawer 1", ACTIONS, 5, random.Random(seed))
    positions = {bs.choose_candidates("open drawer 1", ACTIONS, 5, random.Random(s)).index("open drawer 1")
                 for s in range(30)}
    assert len(positions) > 1          # expert is not always in the same slot


class CountingEnv(TextEnv):
    name, version = "alfworld", "fake"

    def __init__(self):
        self.steps = 0

    def reset(self, task_id):
        self.hist = []
        return StepResult(obs="o", feedback="start", admissible=ACTIONS)

    def step(self, action):
        self.steps += 1
        self.hist.append(action)
        fb = "after %s" % " > ".join(self.hist)
        return StepResult(obs=build_obs("g", "r", "l", fb, "i", ACTIONS), feedback=fb, admissible=ACTIONS)

    def admissible_actions(self):
        return ACTIONS

    def expert_action(self):
        return "take pen 1 from desk 1"


def _row(i):
    return {"transition_id": "alfworld-heldout-e%06d-s001" % i, "env": "alfworld", "split": "heldout",
            "task_id": "alfworld_24%02d" % i, "task_type": "look_at_obj_in_light-Pen", "goal": "look at pen",
            "o_t": "obs %d" % i, "action_prefix": ["go to desk 1"], "admissible_actions": ACTIONS,
            "policy": "expert"}


def test_main_writes_true_next_and_resumes(tmp_path, monkeypatch):
    data = tmp_path / "held.jsonl"
    data.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(3)))
    env = CountingEnv()
    monkeypatch.setattr(bs, "make_env", lambda name: env)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "branched.jsonl"
    bs.main(["--env", "alfworld", "--data", str(data), "--out", str(out), "--k", "5"])
    recs = [json.loads(l) for l in open(out)]
    assert len(recs) == 3
    r = recs[0]
    assert r["id"] == "alfworld/alfworld_2400/alfworld-heldout-e000000-s001" and r["obs"] == "obs 0"
    assert r["actions"][r["gold_index"]] == r["gold_action"] == "take pen 1 from desk 1"
    for a, t in zip(r["actions"], r["true_next"]):
        assert t == "after go to desk 1 > %s" % a            # prefix replayed, then the candidate
    assert r["gold_next"] == r["true_next"][r["gold_index"]] and r["replay_consistent"]
    steps = env.steps
    bs.main(["--env", "alfworld", "--data", str(data), "--out", str(out), "--k", "5"])
    assert env.steps == steps and len(open(out).readlines()) == 3   # resumed: nothing redone


def test_limit_chunks_then_finishes(tmp_path, monkeypatch):
    data = tmp_path / "held.jsonl"
    data.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(5)))
    monkeypatch.setattr(bs, "make_env", lambda name: CountingEnv())
    monkeypatch.chdir(tmp_path)
    args = ["--env", "alfworld", "--data", str(data), "--out", str(tmp_path / "b.jsonl"), "--limit", "2"]
    assert [bs.main(args) for _ in range(3)] == [3, 3, 0]
    assert len(open(tmp_path / "b.jsonl").readlines()) == 5
