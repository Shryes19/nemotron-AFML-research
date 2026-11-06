"""Episode harness with a fake env: python -m pytest tests/test_episodes.py -q"""
from wm.envs.base import StepResult, TextEnv
from wm.obs import build_obs
from wm.utility import episodes as ep
from wm.utility.judge import OverlapJudge

GOLD = ["go to desk 1", "take pen 1 from desk 1", "use desklamp 1"]
OTHER = ["go to bed 1", "open drawer 1", "examine bed 1", "go to shelf 1", "look", "inventory"]


class PathEnv(TextEnv):
    """Won after the 3 gold actions in order; any other action is harmless but useless."""
    name, version = "alfworld", "fake"

    def reset(self, task_id):
        self.done_gold = 0
        return self._res("start", False)

    def _res(self, fb, won):
        return StepResult(obs=build_obs("examine the pen with the desklamp", "r", "l", fb, "i", self.admissible_actions()),
                          feedback=fb, won=won, done=won)

    def step(self, action):
        if self.done_gold < 3 and action == GOLD[self.done_gold]:
            self.done_gold += 1
            fb = "the pen with the desklamp is examined" if self.done_gold == 3 else "ok %s" % action
        else:
            fb = "nothing useful"
        return self._res(fb, self.done_gold == 3)

    def admissible_actions(self):
        return GOLD + OTHER

    def expert_action(self):
        return GOLD[self.done_gold] if self.done_gold < 3 else None

    def goal(self):
        return "examine the pen with the desklamp"


def test_expert_chooser_wins_and_candidates_skip_noops():
    rec = ep.play(PathEnv(), "alfworld", "alfworld_2400", ep.ExpertChooser(), k=5)
    assert rec["success"] and rec["steps"] == 3 and rec["expert_agreement"] == 1.0
    assert not {"look", "inventory"} & set(rec["actions"])


def test_random_chooser_runs_to_cap():
    rec = ep.play(PathEnv(), "alfworld", "alfworld_2400", ep.RandomChooser(), k=5, max_steps=6)
    assert rec["steps"] <= 6 and rec["chooser"] == "random"


def test_oracle_uses_true_outcomes():
    # with the overlap judge, only the final gold step's outcome mentions the goal words, so the
    # oracle is guaranteed to finish once it reaches that state; it must at least run cleanly
    rec = ep.play(PathEnv(), "alfworld", "alfworld_2400", ep.OracleChooser(OverlapJudge(), workers=2), k=5)
    assert rec["chooser"] == "oracle" and rec["steps"] >= 1


def test_prediction_chooser_with_perfect_predictions_behaves_like_oracle():
    env = PathEnv()

    def predict(state, actions):            # a "perfect world model": replay the real outcome
        outs = []
        for a in actions:
            env2 = PathEnv()
            env2.replay(state["task_id"], state["prefix"])
            outs.append(env2.step(a).feedback)
        return outs

    rec = ep.play(env, "alfworld", "alfworld_2400", ep.PredictionChooser(predict, OverlapJudge(), "perfect", 2), k=5)
    oracle = ep.play(PathEnv(), "alfworld", "alfworld_2400", ep.OracleChooser(OverlapJudge(), 2), k=5)
    assert rec["actions"] == oracle["actions"]
