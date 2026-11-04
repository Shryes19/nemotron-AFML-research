"""
ByteSized32-SP (Wang et al. 2024, "Can Language Models Serve as Text-Based World Simulators?").

Data: https://github.com/cognitiveailab/GPT-simulator  ->  data/test.zip  ->  test.jsonl
2,954 transitions over 31 games. Each record holds the full JSON object state before the action
(current_state), after the action (action_state) and after the environment tick (tick_state).

This module
  1. derives exact capability tags from the JSON state diff (ground truth), and
  2. converts each record to our common buffer format so the text tagger, JEPA and SFT arms can use it.

Ground-truth rules (state diff current_state -> tick_state, per object uuid):
  containment_location  any object's `contains` list changes (agent moving rooms, items into inventory)
  numeric_physical      a numeric property changes (temperature, volume, weight, ...) or stateOfMatter changes
  object_state          a non-numeric property changes, or an object is created / destroyed
  action_failure        the action changed nothing (current_state == action_state) and is not an observe verb
                        (observe verbs count only if the game refuses them: "You can't read the carrot.")
  partial_obs           0 by construction: ByteSized32 gives the full state, nothing is hidden
Wang et al.'s split is kept as two flags:
  action_driven = current_state != action_state      env_driven = action_state != tick_state
"""
from __future__ import annotations

import json

from .tagger import FAIL_RE, OBSERVE, TAGS, PRIMARY_ORDER, verb_of

PHASE_KEYS = {"stateOfMatter"}


def _objs(state: dict) -> dict:
    return {o["uuid"]: o for o in state.get("objects", [])}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def state_diff(a: dict, b: dict) -> dict:
    oa, ob = _objs(a), _objs(b)
    out = {"created": sorted(set(ob) - set(oa)), "destroyed": sorted(set(oa) - set(ob)),
           "contains": [], "numeric": [], "categorical": []}
    for u in set(oa) & set(ob):
        x, y = oa[u], ob[u]
        if sorted(map(str, x.get("contains", []))) != sorted(map(str, y.get("contains", []))):
            out["contains"].append(x["name"])
        px, py = x.get("properties", {}), y.get("properties", {})
        for k in set(px) | set(py):
            if px.get(k) != py.get(k):
                kind = "numeric" if (_is_num(px.get(k)) or _is_num(py.get(k)) or k in PHASE_KEYS) else "categorical"
                out[kind].append(f"{x['name']}.{k}")
    return out


def _changed(d: dict) -> bool:
    return any(d[k] for k in d)


def ground_truth_tags(rec: dict) -> dict:
    cur, act, tick = rec["current_state"], rec["action_state"], rec["tick_state"]
    action = act.get("lastAction", "")
    d_full, d_act, d_tick = state_diff(cur, tick), state_diff(cur, act), state_diff(act, tick)
    tags = set()
    if d_full["contains"]:
        tags.add("containment_location")
    if d_full["numeric"]:
        tags.add("numeric_physical")
    if d_full["categorical"] or d_full["created"] or d_full["destroyed"]:
        tags.add("object_state")
    # an action that changed nothing is a failure; for observe verbs (look/read/measure) a no-change step is
    # normal, so they count as failures only when the game says so ("You can't read the carrot.")
    if not _changed(d_act) and (verb_of(action) not in OBSERVE or FAIL_RE.search(act.get("observation", ""))):
        tags.add("action_failure")
    primary = next((t for t in PRIMARY_ORDER if t in tags), "none")
    return {"tags": [t for t in TAGS if t in tags], "primary": primary,
            "action_driven": _changed(d_act), "env_driven": _changed(d_tick),
            "diff": {k: v[:6] for k, v in d_full.items() if v}}


def _render(state: dict) -> str:
    """Text observation in the same [SECTION] layout as Saanvi's ALFWorld/ScienceWorld exporter."""
    return (f"[GOAL] {state.get('taskDesc', '').strip()}\n"
            f"[ROOM] {state.get('look', '').strip()}\n"
            f"[FEEDBACK] {state.get('observation', '').strip()}\n"
            f"[INVENTORY] {state.get('inventory', '').strip()}")


def to_buffer_record(rec: dict) -> dict:
    gt = ground_truth_tags(rec)
    cur, tick = rec["current_state"], rec["tick_state"]
    return {
        "obs": _render(cur),
        "action": rec["action_state"].get("lastAction", ""),
        "obs_next": tick.get("observation", "").strip(),
        "obs_next_full": _render(tick),
        "feedback_next": tick.get("observation", "").strip(),
        "env": "bytesized32",
        "game": rec["game"],
        "cap_tag": gt["primary"],
        "cap_tags": gt["tags"],
        "action_driven": gt["action_driven"],
        "env_driven": gt["env_driven"],
        "transition_id": f"bs32-{rec['game']}-{rec['state_id']}",
        "score_next": rec.get("next_score_state", {}).get("score"),
        "split": "test",
        "policy": "bytesized32_crawl",
    }


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
