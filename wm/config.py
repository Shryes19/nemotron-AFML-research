"""Settings for the data pipeline, from one source of truth.

Budgets and the expert/random/LLM mix come from the team-wide configs/config.yaml
(data.budgets, data.collection_policy). configs/shared.yaml holds only pipeline settings:
budget_seeds (the shuffles behind the nested budgets), pool targets, env versions, splits.
"""
from __future__ import annotations

import os
from typing import Dict

import yaml

TEAM_CONFIG = os.path.join("configs", "config.yaml")
PIPELINE_CONFIG = os.path.join("configs", "shared.yaml")


def load_config(root: str = ".") -> Dict:
    with open(os.path.join(root, TEAM_CONFIG)) as f:
        team = yaml.safe_load(f)
    with open(os.path.join(root, PIPELINE_CONFIG)) as f:
        cfg = yaml.safe_load(f)
    cfg["budgets"] = list(team["data"]["budgets"])
    cfg["policy_mix"] = dict(team["data"]["collection_policy"])
    return cfg
