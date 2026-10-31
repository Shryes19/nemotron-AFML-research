"""ALFWorld wrapper built on Word2World's vendored AgentGym (agentenv-alfworld).

Uses AgentGym's own game list (mappings_train.json then mappings_test.json), so task id
"alfworld_N" is the same game AgentGym/Word2World use. Runs in-process (not through the
AgentGym HTTP server) so the handcoded expert plan is available.
Run inside the uv_alfworld venv.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from wm.envs.base import StepResult, TextEnv
from wm.obs import build_obs

ALFWORLD_DATA = os.path.expanduser("~/.cache/alfworld")
MAX_STEPS = 100  # textworld episode cap; the collector stops well before this


def _agentgym_paths():
    import agentenv_alfworld
    root = os.path.join(os.path.dirname(os.path.realpath(agentenv_alfworld.__file__)), "..")
    return os.path.join(root, "configs")


class AlfWorldEnv(TextEnv):
    name = "alfworld"
    version = "alfworld==0.3.3+w2w-twl2@e8dc240"

    def __init__(self):
        os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA
        from agentenv_alfworld.utils import load_config
        configs = _agentgym_paths()
        self.config = load_config(os.path.join(configs, "base_config.yaml"))
        self.config["general"]["use_cuda"] = False
        self.config["dagger"]["training"]["max_nb_steps_per_episode"] = MAX_STEPS
        self.games = []
        self.name_to_idx = {}
        for mapping_file, folder in (("mappings_train.json", "train"), ("mappings_test.json", "valid_train")):
            with open(os.path.join(configs, mapping_file)) as f:
                for m in json.load(f):
                    self.name_to_idx[m["task_type"] + "_" + m["task_id"]] = len(self.games)
                    self.games.append(os.path.join(ALFWORLD_DATA, "json_2.1.1", folder,
                                                   m["task_type"], m["task_id"], "game.tw-pddl"))
        self._tw = None

    # --- interface -------------------------------------------------------
    def reset(self, task_id: str) -> StepResult:
        from agentenv_alfworld.environment import SingleAlfredTWEnv
        self.close()
        # test ids look like "alfworld_2420"; train ids are game names "<task_type>_trial_..."
        if task_id in self.name_to_idx:
            idx = self.name_to_idx[task_id]
        else:
            idx = int(task_id.split("_")[-1])
        env = SingleAlfredTWEnv(self.config, train_eval="train")  # "train" attaches the expert
        env.game_files = [self.games[idx]]
        env.num_games = 1
        self._tw = env.init_env(batch_size=1)
        obs, info = self._tw.reset()
        # same processing as AgentGym's server: drop the welcome banner
        text = "\n".join(obs[0].split("\n\n")[1:])
        room, _, goal = text.partition("Your task is to:")
        self._goal = goal.strip()
        self._room = room.strip()
        self._task_type = info["extra.gamefile"][0].split("/")[-3]
        self._location = "middle of the room"
        self._holding: List[str] = []
        self._info = info
        return self._result(self._room, 0.0, False, False)

    def step(self, action: str) -> StepResult:
        obs, _, dones, info = self._tw.step([action])
        feedback = obs[0]
        if feedback.startswith("You arrive at loc "):  # AgentGym process_ob
            feedback = feedback[feedback.find(". ") + 2:]
        self._track(action, feedback)
        self._info = info
        won = bool(info["won"][0])
        return self._result(feedback, 1.0 if won else 0.0, bool(dones[0]), won)

    def admissible_actions(self) -> List[str]:
        return list(self._info["admissible_commands"][0])

    def expert_action(self) -> Optional[str]:
        plan = self._info.get("extra.expert_plan", [[]])[0]
        return plan[0] if plan else None

    def task_type(self) -> str:
        return self._task_type

    def goal(self) -> str:
        return self._goal

    def close(self):
        if self._tw is not None:
            self._tw.close()
            self._tw = None

    # --- visible-state tracking -----------------------------------------
    def _track(self, action: str, feedback: str):
        m = re.match(r"go to (.+)$", action)
        if m and not feedback.startswith("Nothing happens"):
            self._location = m.group(1)
        m = re.match(r"You pick up the (.+?) from", feedback)
        if m:
            self._holding.append(m.group(1))
        m = re.match(r"You (?:move|put) the (.+?) (?:to|in|on)", feedback)
        if m and m.group(1) in self._holding:
            self._holding.remove(m.group(1))

    def inventory_text(self) -> str:
        if not self._holding:
            return "You are not carrying anything."
        return "You are carrying: " + ", ".join("a " + o for o in self._holding) + "."

    def _result(self, feedback: str, reward: float, done: bool, won: bool) -> StepResult:
        adm = self.admissible_actions()
        obs = build_obs(self._goal, self._room, self._location, feedback, self.inventory_text(), adm)
        return StepResult(obs=obs, feedback=feedback, reward=reward, done=done, won=won, admissible=adm)
