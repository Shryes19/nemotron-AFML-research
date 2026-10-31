"""JSONL transition schema (see docs/SCHEMA.md) and a validator.

Usage: python -m wm.schema data/sample/alfworld_sample20.jsonl [more files...]
"""
from __future__ import annotations

import gzip
import json
import sys
from typing import Iterator, List

from wm.obs import SECTIONS, parse_obs

FIELDS = {
    "transition_id": str,
    "env": str,
    "split": str,
    "task_id": str,
    "task_type": str,
    "episode_id": str,
    "step": int,
    "goal": str,
    "o_t": str,
    "a_t": str,
    "o_next": str,
    "feedback_next": str,
    "reward": float,
    "done": bool,
    "won": bool,
    "capability_tag": list,
    "policy": str,
    "action_prefix": list,
    "admissible_actions": list,
    "collector_seed": int,
    "env_version": str,
    "llm_meta": (dict, type(None)),
}
ENVS = {"alfworld", "sciworld"}
SPLITS = {"train", "heldout"}
POLICIES = {"expert", "random", "llm", "llm_fallback"}


def read_jsonl(path: str) -> Iterator[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def validate_row(row: dict) -> List[str]:
    errors = []
    for name, typ in FIELDS.items():
        if name not in row:
            errors.append("missing field %s" % name)
        elif typ is float and isinstance(row[name], int) and not isinstance(row[name], bool):
            continue
        elif not isinstance(row[name], typ):
            errors.append("%s has type %s" % (name, type(row[name]).__name__))
    if errors:
        return errors
    if row["env"] not in ENVS:
        errors.append("bad env %r" % row["env"])
    if row["split"] not in SPLITS:
        errors.append("bad split %r" % row["split"])
    if row["policy"] not in POLICIES:
        errors.append("bad policy %r" % row["policy"])
    if len(row["action_prefix"]) != row["step"]:
        errors.append("len(action_prefix)=%d != step=%d" % (len(row["action_prefix"]), row["step"]))
    for key in ("o_t", "o_next"):
        if list(parse_obs(row[key])) != SECTIONS:
            errors.append("%s does not have sections %s" % (key, SECTIONS))
    if not row["transition_id"].startswith(row["episode_id"]):
        errors.append("transition_id does not start with episode_id")
    return errors


def main(paths: List[str]) -> int:
    bad = 0
    for path in paths:
        n = 0
        for i, row in enumerate(read_jsonl(path)):
            n += 1
            errs = validate_row(row)
            if errs:
                bad += 1
                print("%s line %d: %s" % (path, i + 1, "; ".join(errs)))
        print("%s: %d rows checked" % (path, n))
    print("OK" if bad == 0 else "%d bad rows" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
