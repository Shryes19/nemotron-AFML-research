"""Composite observation text shared by all environments.

Every o_t / o_next is built from visible information only (no hidden initial state):

[GOAL] ...
[ROOM] ...
[LOCATION] ...
[FEEDBACK] ...
[INVENTORY] ...
[ADMISSIBLE] a | b | c
"""
from __future__ import annotations

from typing import List

SECTIONS = ["GOAL", "ROOM", "LOCATION", "FEEDBACK", "INVENTORY", "ADMISSIBLE"]


def build_obs(goal: str, room: str, location: str, feedback: str,
              inventory: str, admissible: List[str]) -> str:
    values = [goal, room, location, feedback, inventory, " | ".join(admissible)]
    return "\n".join("[%s] %s" % (name, " ".join(str(v).split()))
                     for name, v in zip(SECTIONS, values))


def parse_obs(text: str) -> dict:
    """Inverse of build_obs: section name -> value."""
    out = {}
    for line in text.split("\n"):
        if line.startswith("[") and "] " in line:
            name, value = line[1:].split("] ", 1)
            out[name] = value
    return out


def canonical(text: str) -> dict:
    """Order-insensitive form of a composite observation, for comparing states.

    ScienceWorld lists objects in a run-dependent order (e.g. two "wood cup"s can swap),
    so two observations of the same state may differ only in word order.
    """
    return {name: sorted(value.replace(",", " ").replace(".", " ").split())
            for name, value in parse_obs(text).items()}
