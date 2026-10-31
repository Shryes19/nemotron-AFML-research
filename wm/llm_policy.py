"""LLM data-collection policy: Qwen2.5-7B-Instruct picks one admissible action.

The model sees the goal, the composite observation and a numbered list of candidate actions,
and replies with a number. ALFWorld shows every admissible action. ScienceWorld has hundreds
to thousands, so it shows MAX_CANDIDATES actions sampled verb-first, seeded by the episode
rng so prompts (and cached replies) are reproducible. If the reply is not a valid number it
asks once more; if that also fails, a random candidate is used and the row is labelled
llm_fallback.
"""
from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

from wm.llm_client import LLMClient
from wm.obs import parse_obs
from wm.policies import SKIP_ACTIONS, sample_by_verb

MAX_CANDIDATES = 60

SYSTEM = ("You are an agent in a text-based environment. Choose the next action that best "
          "makes progress on the task. Reply with only the number of your chosen action.")


def candidates(actions: List[str], rng: random.Random) -> List[str]:
    actions = [a for a in actions if a not in SKIP_ACTIONS]
    if len(actions) <= MAX_CANDIDATES:
        return actions
    picked = set()
    while len(picked) < MAX_CANDIDATES:
        picked.add(sample_by_verb([a for a in actions if a not in picked], rng))
    return sorted(picked)


def build_messages(obs_text: str, options: List[str]) -> List[dict]:
    s = parse_obs(obs_text)
    lines = ["Task: %s" % s.get("GOAL", ""),
             "Current room: %s" % s.get("ROOM", ""),
             "Location: %s" % s.get("LOCATION", ""),
             "Last observation: %s" % s.get("FEEDBACK", ""),
             "Inventory: %s" % s.get("INVENTORY", ""),
             "",
             "Actions:"]
    lines += ["%d. %s" % (i + 1, a) for i, a in enumerate(options)]
    lines.append("")
    lines.append("Reply with only the number of the action.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_choice(text: str, n: int) -> Optional[int]:
    m = re.search(r"\d+", text)
    if m and 1 <= int(m.group()) <= n:
        return int(m.group()) - 1
    return None


class LLMPolicy:
    name = "llm"

    def __init__(self, env_name: str = "", client: Optional[LLMClient] = None):
        self.env_name = env_name
        self.client = client or LLMClient()

    def act(self, env, rng: random.Random, obs_text: str = "") -> Tuple[Optional[str], Optional[dict]]:
        options = candidates(env.admissible_actions(), rng)
        if not options:
            return None, None
        messages = build_messages(obs_text, options)
        meta = {}
        for _ in range(2):
            out = self.client.chat(messages, temperature=0.0, max_tokens=8)
            idx = parse_choice(out["text"], len(options))
            meta = {"llm_meta": {"model": self.client.model, "cache_key": out["cache_key"],
                                 "reply": out["text"][:50], "n_candidates": len(options)}}
            if idx is not None:
                return options[idx], meta
            messages = messages + [
                {"role": "assistant", "content": out["text"]},
                {"role": "user", "content": "Reply with only a number between 1 and %d." % len(options)}]
        meta["fallback"] = True
        return rng.choice(options), meta
