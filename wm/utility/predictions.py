"""Load an arm's predictions for the branched states into one internal form.

Generic format (RAG / SFT arms), one line per state:
  {"id": ..., "arm": "G", "budget": 256, "actions": [K actions], "pred_next": [K predicted texts]}

JEPA format (scripts/jepa_predict_candidates.py, see docs/LATENT_WORLD_MODEL.md):
  {"id": ..., "budget": 256, "candidates": [{"action": ..., "topk": [{"text", "sim"}, ...],
   "target_sim": ...}, ...], "ranking": [...]}
  mode "top1": the best decoded state per action (main, comparable with text arms)
  mode "topk": the k decoded states per action (the judge sees them as a ranked list)
  mode "native": JEPA's own goal-based ranking, no judge

Every arm must answer for exactly the states' actions; order differences are fixed by matching
action text, anything else is an error.

Internal form: {state id: {"budget": int, "pred": [one item per action, str or list of str],
                           "ranking": list or None}}
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional


def _reorder(state_actions: List[str], arm_actions: List[str], items: List, sid: str) -> List:
    if arm_actions == state_actions:
        return items
    if sorted(arm_actions) != sorted(state_actions):
        raise ValueError("%s: predicted actions %s do not match the state's actions %s"
                         % (sid, arm_actions, state_actions))
    by_action = dict(zip(arm_actions, items))
    return [by_action[a] for a in state_actions]


def load_predictions(path: str, states: Dict[str, Dict], fmt: str = "generic",
                     mode: str = "top1", budget: Optional[int] = None,
                     allow_missing: bool = False) -> Dict[str, Dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["id"]
            if sid not in states:
                continue  # predictions for states not in this file are ignored
            b = rec.get("budget", budget)
            if budget is not None and b != budget:
                continue
            state_actions = states[sid]["actions"]
            ranking = None
            if fmt == "generic":
                pred = _reorder(state_actions, rec["actions"], rec["pred_next"], sid)
            elif fmt == "jepa":
                cands = rec["candidates"]
                actions = [c["action"] for c in cands]
                if mode == "topk":
                    items = [[t["text"] for t in c["topk"]] for c in cands]
                else:
                    items = [c["topk"][0]["text"] if c["topk"] else "" for c in cands]
                pred = _reorder(state_actions, actions, items, sid)
                if rec.get("ranking") is not None:
                    # JEPA's ranking indexes its own candidate order; map it to the state's order
                    ranking = [state_actions.index(actions[i]) for i in rec["ranking"]]
            else:
                raise ValueError("unknown format %r" % fmt)
            out[sid] = {"budget": b, "pred": pred, "ranking": ranking}
    missing = [s for s in states if s not in out]
    if missing and not allow_missing:
        raise ValueError("%s: no predictions for %d of %d states (e.g. %s)"
                         % (path, len(missing), len(states), missing[0]))
    return out
