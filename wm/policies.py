"""Data-collection policies. Expert and random need no LLM."""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import List, Optional, Tuple

# Meta-commands, never chosen by random/LLM policies and never recorded: ALFWorld's "help"
# only prints the command list; ScienceWorld's "reset task" restarts the episode.
SKIP_ACTIONS = {"help", "reset task"}


def sample_by_verb(actions: List[str], rng: random.Random) -> str:
    """Pick a verb uniformly, then an action with that verb. Used for ScienceWorld, where
    ~2/3 of valid actions are "connect X to Y" and a uniform pick would mostly be wiring."""
    groups = defaultdict(list)
    for a in actions:
        groups[a.split()[0]].append(a)
    verb = rng.choice(sorted(groups))
    return rng.choice(groups[verb])


class ExpertPolicy:
    name = "expert"

    def act(self, env, rng: random.Random, obs_text: str = "") -> Tuple[Optional[str], Optional[dict]]:
        return env.expert_action(), None


# ALFWorld: every admissible command succeeds, so admissible-only random actions never fail.
# This share of random steps instead uses a well-formed but non-admissible command over
# visible names (e.g. "take mug 1 from drawer 2" when the mug is not there), which ALFWorld
# answers with "Nothing happens." -- the source of action-failure transitions.
ALFWORLD_FAIL_RATE = 1 / 3
FAIL_TEMPLATES = ["take {a} from {b}", "move {a} to {b}", "open {a}", "close {a}", "use {a}",
                  "examine {a}", "heat {a} with {b}", "cool {a} with {b}", "clean {a} with {b}",
                  "slice {a} with {b}", "go to {a}"]


def visible_names(obs_text: str, actions: List[str]) -> List[str]:
    """Object/receptacle names like "drawer 2" seen in the observation or admissible actions."""
    return sorted(set(re.findall(r"\b([a-z]+ \d+)\b", obs_text + " " + " ".join(actions))))


def failing_action(obs_text: str, actions: List[str], rng: random.Random) -> Optional[str]:
    names = visible_names(obs_text, actions)
    admissible = set(actions)
    for _ in range(50):
        cand = rng.choice(FAIL_TEMPLATES).format(a=rng.choice(names), b=rng.choice(names)) if names else None
        if cand and cand not in admissible:
            return cand
    return None


class RandomPolicy:
    name = "random"

    def __init__(self, by_verb: bool = False, fail_rate: float = 0.0):
        self.by_verb = by_verb
        self.fail_rate = fail_rate

    def act(self, env, rng: random.Random, obs_text: str = "") -> Tuple[Optional[str], Optional[dict]]:
        choices = [a for a in env.admissible_actions() if a not in SKIP_ACTIONS]
        if not choices:
            return None, None
        if self.fail_rate and rng.random() < self.fail_rate:
            action = failing_action(obs_text, choices, rng)
            if action:
                return action, None
        return (sample_by_verb(choices, rng) if self.by_verb else rng.choice(choices)), None


def make_policy(name: str, env_name: str = ""):
    if name == "expert":
        return ExpertPolicy()
    if name == "random":
        return RandomPolicy(by_verb=(env_name == "sciworld"),
                            fail_rate=ALFWORLD_FAIL_RATE if env_name == "alfworld" else 0.0)
    if name == "llm":
        from wm.llm_policy import LLMPolicy
        return LLMPolicy(env_name)
    raise ValueError("unknown policy: %s" % name)
