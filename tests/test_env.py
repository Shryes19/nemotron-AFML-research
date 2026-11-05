"""Env smoke + replay determinism. Run in the env's venv:
  ~/w2w/uv_alfworld/bin/python -m pytest tests/test_env.py --env alfworld -q
  ~/w2w/uv_sciworld/bin/python -m pytest tests/test_env.py --env sciworld -q
"""
import json
import random

from wm.obs import SECTIONS, canonical, parse_obs


def _ids(env_name, split):
    name = "train" if split == "train" else "test"
    with open("splits/%s_%s.json" % (env_name, name)) as f:
        return [x["item_id"] for x in json.load(f)]


def _rollout(env, task_id, n_steps, rng):
    """Mixed expert/random rollout; returns actions and observation after each step."""
    r = env.reset(task_id)
    actions, obs = [], [r.obs]
    for _ in range(n_steps):
        a = env.expert_action() if rng.random() < 0.5 else None
        if a is None:
            if not env.admissible_actions():
                break  # e.g. ScienceWorld ambiguity prompt ends the episode
            a = rng.choice(env.admissible_actions())
        r = env.step(a)
        actions.append(a)
        obs.append(r.obs)
        if r.done:
            break
    return actions, obs


def test_reset_both_splits(env, env_name):
    for split in ("train", "heldout"):
        r = env.reset(_ids(env_name, split)[0])
        assert list(parse_obs(r.obs)) == SECTIONS
        assert env.goal() and env.admissible_actions()


def test_replay_determinism(env, env_name):
    rng = random.Random(0)
    ids = _ids(env_name, "train")
    for k in range(int(__import__("os").environ.get("WM_REPLAY_N", "20"))):
        task_id = rng.choice(ids)
        actions, obs = _rollout(env, task_id, rng.randint(1, 12), rng)
        cut = rng.randint(0, len(actions))
        for _ in range(2):
            r = env.replay(task_id, actions[:cut])
            msg = "replay mismatch: %s prefix=%s" % (task_id, actions[:cut])
            if env.exact_replay:
                assert r.obs == obs[cut], msg
            else:
                assert canonical(r.obs) == canonical(obs[cut]), msg
                assert parse_obs(r.obs)["FEEDBACK"] == parse_obs(obs[cut])["FEEDBACK"], msg
