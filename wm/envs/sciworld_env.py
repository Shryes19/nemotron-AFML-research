"""ScienceWorld wrapper matching Word2World's vendored AgentGym (agentenv-sciworld).

Task id "sciworld_N" indexes the same flat (task, variation) list AgentGym builds, with the
same excluded tasks. Reset = load + "look around", as in AgentGym. The expert is the gold
action sequence, which is only valid while the episode has followed it exactly.

Ambiguity prompts ("Ambiguous request ... 0: ..., 1: ...") list options in a run-dependent
order, so a numeric answer is not replayable. When one appears, no actions are offered and
the episode ends there; numeric gold steps are never played by the expert.
Run inside the uv_sciworld venv (needs Java).
"""
from __future__ import annotations

import re
from typing import List, Optional

from wm.envs.base import StepResult, TextEnv
from wm.obs import build_obs

EXCLUDED = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}  # as in AgentGym
STEP_LIMIT = 100


class SciWorldEnv(TextEnv):
    name = "sciworld"
    version = "scienceworld==1.2.3+agentgym@e8dc240"
    exact_replay = False  # object listing order in "look" varies between runs

    def __init__(self):
        from scienceworld import ScienceWorldEnv
        self._env = ScienceWorldEnv("", envStepLimit=STEP_LIMIT)
        self.games = []
        for key, task in self._env.tasks.items():
            if key not in EXCLUDED:
                self.games += [(task, i) for i in range(self._env.get_max_variations(task))]

    def reset(self, task_id: str) -> StepResult:
        task, variation = self.games[int(task_id.split("_")[-1])]
        self._env.load(task, variation, "", generateGoldPath=True)
        self._task = task
        self._goal = self._env.get_task_description()
        self._gold = list(self._env.get_gold_action_sequence())
        self._on_gold = 0  # number of gold actions taken so far; -1 once we deviate
        feedback, _, _, info = self._env.step("look around")
        self._score = info["score"]
        self._ambiguous = False
        return self._result(feedback, 0.0, False)

    def step(self, action: str) -> StepResult:
        if self._on_gold >= 0:
            on_path = self._on_gold < len(self._gold) and action == self._gold[self._on_gold]
            self._on_gold = self._on_gold + 1 if on_path else -1
        feedback, _, done, info = self._env.step(action)
        self._ambiguous = feedback.strip().startswith("Ambiguous request")
        reward = float(info["score"] - self._score)
        self._score = info["score"]
        return self._result(feedback, reward, bool(done))

    def admissible_actions(self) -> List[str]:
        if self._ambiguous:
            return []
        actions = {a for a in self._env.get_valid_action_object_combinations() if not a.strip().isdigit()}
        # gold steps use fuller names ("open door to hallway") than the valid list ("open door")
        gold = self.expert_action()
        if gold:
            actions.add(gold)
        return sorted(actions)

    def expert_action(self) -> Optional[str]:
        if self._ambiguous or not 0 <= self._on_gold < len(self._gold):
            return None
        action = self._gold[self._on_gold]
        return None if action.strip().isdigit() else action

    def task_type(self) -> str:
        return self._task

    def goal(self) -> str:
        return self._goal

    def close(self):
        self._env.close()

    def _result(self, feedback: str, reward: float, done: bool) -> StepResult:
        room = self._env.look()
        m = re.search(r"This room is called the (.+?)\.", room)
        location = m.group(1) if m else "unknown"
        adm = self.admissible_actions()
        templates = self._env.get_possible_actions()
        obs = build_obs(self._goal, room, location, feedback, self._env.inventory(), templates)
        won = self._score >= 100
        return StepResult(obs=obs, feedback=feedback, reward=reward, done=done or won, won=won, admissible=adm)
