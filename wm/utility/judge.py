"""Frozen judge: how much does an action's outcome advance the task? Integer 0-10.

The same judge scores the TRUE next states (true values, and the oracle) and every arm's
PREDICTED next states, so differences between arms come from their predictions, not the judge.

  LLMJudge      Qwen via LLM_BASE_URL (wm/llm_client.py: cached, retried). Needs the GPU server.
  OverlapJudge  deterministic word overlap with the goal; no LLM. For dry runs and tests only.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Union

from wm.obs import parse_obs

SYSTEM = ("You evaluate progress in a text-based environment. You are given a task, the current "
          "situation, an action and what happens after it. Judge only how much this action, with "
          "this outcome, moves the agent toward completing the task.")

SCALE = ("On a scale from 0 to 10, how much does this move the agent toward completing the task? "
         "0 = no progress, a failed action or a step backwards; 10 = the task is completed. "
         "Reply with a single integer.")

Outcome = Union[str, Sequence[str]]


def situation(obs: str) -> str:
    """Composite observation without the long [ADMISSIBLE] list."""
    s = parse_obs(obs)
    if not s:
        return obs
    keep = ("ROOM", "LOCATION", "FEEDBACK", "INVENTORY")
    return "\n".join("%s: %s" % (k.capitalize(), s[k]) for k in keep if k in s)


def judge_messages(goal: str, obs: str, action: str, outcome: Outcome) -> List[dict]:
    if isinstance(outcome, str):
        result = "Resulting observation: %s" % outcome
    else:  # several possible outcomes, most likely first (JEPA top-k)
        result = "Possible resulting observations, most likely first:\n" + \
                 "\n".join("%d. %s" % (i + 1, o) for i, o in enumerate(outcome))
    user = "Task: %s\n\nCurrent situation:\n%s\n\nAction: %s\n%s\n\n%s" % (
        goal, situation(obs), action, result, SCALE)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_value(text: str) -> Optional[int]:
    m = re.search(r"\b(10|[0-9])\b", text or "")
    return int(m.group(1)) if m else None


class LLMJudge:
    name = "llm"

    def __init__(self, client=None):
        if client is None:
            from wm.llm_client import LLMClient
            client = LLMClient()
        self.client = client
        self.model = client.model

    def value(self, goal: str, obs: str, action: str, outcome: Outcome) -> Optional[int]:
        messages = judge_messages(goal, obs, action, outcome)
        for _ in range(2):
            out = self.client.chat(messages, temperature=0.0, max_tokens=4)
            v = parse_value(out["text"])
            if v is not None:
                return v
            messages = messages + [{"role": "assistant", "content": out["text"]},
                                   {"role": "user", "content": "Reply with only an integer from 0 to 10."}]
        return None  # counted as unparsable and excluded

    def values(self, items, workers: int = 8) -> List[Optional[int]]:
        """items: (goal, obs, action, outcome) tuples; vLLM serves the parallel calls."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda it: self.value(*it), items))


class OverlapJudge(LLMJudge):
    """Share of goal words in the outcome, scaled to 0-10. Not a real judge: dry runs only."""
    name = "overlap"
    model = "overlap"

    def __init__(self):
        pass

    def value(self, goal, obs, action, outcome):
        text = outcome if isinstance(outcome, str) else (outcome[0] if outcome else "")
        g = set(re.findall(r"[a-z]+", goal.lower()))
        o = set(re.findall(r"[a-z]+", text.lower()))
        return int(round(10 * len(g & o) / len(g))) if g else 0


def make_judge(name: str):
    return OverlapJudge() if name == "overlap" else LLMJudge()
