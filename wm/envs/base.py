"""Common interface for the text environments.

Each env exposes:
  reset(task_id) -> StepResult      start a task (e.g. "alfworld_17", "sciworld_3491")
  step(action)   -> StepResult      apply one action
  admissible_actions() -> list      valid actions at the current state
  expert_action() -> str or None    next action of the scripted expert (None if unavailable)
  replay(task_id, prefix) -> StepResult   reset and re-apply a saved action prefix (Phase 2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepResult:
    obs: str                    # composite observation text (wm.obs.build_obs)
    feedback: str               # raw environment text for this step
    reward: float = 0.0
    done: bool = False
    won: bool = False
    admissible: List[str] = field(default_factory=list)


class TextEnv:
    name = "base"
    version = "unknown"
    # True if replaying a prefix reproduces o_t byte for byte; False if only up to
    # object listing order (compare with wm.obs.canonical).
    exact_replay = True

    def reset(self, task_id: str) -> StepResult:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def admissible_actions(self) -> List[str]:
        raise NotImplementedError

    def expert_action(self) -> Optional[str]:
        raise NotImplementedError

    def task_type(self) -> str:
        return ""

    def goal(self) -> str:
        return ""

    def replay(self, task_id: str, prefix: List[str]) -> StepResult:
        result = self.reset(task_id)
        for action in prefix:
            result = self.step(action)
        return result


def make_env(name: str) -> TextEnv:
    if name == "alfworld":
        from wm.envs.alfworld_env import AlfWorldEnv
        return AlfWorldEnv()
    if name == "sciworld":
        from wm.envs.sciworld_env import SciWorldEnv
        return SciWorldEnv()
    raise ValueError("unknown env: %s" % name)
