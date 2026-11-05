"""
jepa/training/features.py
─────────────────────────
Small hand-made inputs added next to the text embeddings.

Why
───
Whether an action FAILS ("Nothing happens.") depends almost entirely on one
fact: is the action in the observation's list of allowed (admissible)
actions? That list is the LAST section of an ALFWorld observation, so for long
observations the encoder's 512-token limit can cut it off, and even when it is
read, "is this exact string in a long list" is hard to pull out of one
embedding. The diagnostics showed JEPA losing badly to the lookup on failed
actions at small budgets (71% vs 96% at n=128). So we hand the model that fact
directly. It is not extra information — it is read from the observation.

Features (one row per transition)
─────────────────────────────────
    is_admissible   1.0 if the action is in the [ADMISSIBLE] list, else 0.0
    has_list        1.0 if the observation has an [ADMISSIBLE] section at all
                    (lets the model ignore is_admissible when it is unknown)
"""

import re
from typing import Dict, List, Optional, Set

import torch

N_FEATURES = 2

# "[ADMISSIBLE] a | b | c" up to the next "[SECTION]" header or the end
_ADMISSIBLE = re.compile(r"\[ADMISSIBLE\]\s*(.*?)(?=\n\s*\[[A-Z_]+\]|\Z)", re.S)


_TEMPLATE_SLOT = re.compile(r"\bobj\b", re.I)   # ScienceWorld lists "open OBJ", not real actions


def admissible_set(obs: str) -> Optional[Set[str]]:
    """
    The allowed actions listed in an observation, or None if there is no usable list.

    ScienceWorld's [ADMISSIBLE] section holds action TEMPLATES ("open OBJ",
    "connect OBJ to OBJ"), not concrete actions, so a real action never matches
    it. Such a list is treated as no list (features [0, 0], like ByteSized32)
    instead of marking every action as not allowed.
    """
    m = _ADMISSIBLE.search(obs or "")
    if not m:
        return None
    items = {a.strip().lower() for a in m.group(1).split("|") if a.strip()}
    if items and sum(bool(_TEMPLATE_SLOT.search(a)) for a in items) > len(items) / 2:
        return None
    return items


def action_features(obs: str, action: str) -> List[float]:
    allowed = admissible_set(obs)
    if allowed is None:
        return [0.0, 0.0]
    return [1.0 if action.strip().lower() in allowed else 0.0, 1.0]


def feature_matrix(transitions: List[Dict]) -> torch.Tensor:
    """(N, N_FEATURES) float tensor, row i for transitions[i]."""
    return torch.tensor([action_features(t["obs"], t["action"]) for t in transitions],
                        dtype=torch.float32)


def with_features(z_ctx: torch.Tensor, transitions: List[Dict]) -> torch.Tensor:
    """Append the hand-made features as extra columns of the predictor input."""
    return torch.cat([z_ctx, feature_matrix(transitions).to(z_ctx.dtype)], dim=1)


def coverage(transitions: List[Dict]) -> Dict[str, float]:
    """How many observations carry a list, and how many actions are allowed."""
    f = feature_matrix(transitions)
    has = f[:, 1]
    return {
        "has_list": float(has.mean()) if len(f) else 0.0,
        "admissible_given_list": float(f[has == 1, 0].mean()) if has.sum() else float("nan"),
    }
