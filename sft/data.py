"""
sft/data.py
───────────
Everything the SFT arm (F) needs about DATA: which rows it trains on, how a
(observation, action) pair becomes a prompt, and how answers are scored.
No GPU and no model here, so it is easy to test (tests/test_sft_data.py).

The task (Ashok et al. 2026, Eq. 3; Word2World's next-state prediction):
    input  = current observation + action          (no history, no hidden state)
    target = the next observation text, `obs_next`  (the short feedback, like the
             branched states' `true_next` and the buffers' `obs_next`)

Fair comparison with JEPA (arm D)
─────────────────────────────────
For every (env, budget), F trains on EXACTLY the rows the JEPA checkpoint of
that budget trained on. scripts/export_train_rows.py copies them from the
checkpoints into small files in sft/rows/, which this module reads.

Prompt
──────
The [ADMISSIBLE] list is removed (it is long: ~100 tokens in ALFWorld) and
replaced by one line, "Action allowed: yes / no / unknown". That is the same
fact JEPA gets as a hand feature, computed with JEPA's own code
(jepa/training/features.py: admissible_set), so both arms see the same hint.
ScienceWorld lists templates ("open OBJ") and ByteSized32 has no list, so
there the line says "unknown". Train and test prompts are built the same way.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from jepa.training.features import admissible_set

SYSTEM = ("You simulate a text-based game. Given the current observation and the player's "
          "action, reply with exactly the feedback text the game shows after that action. "
          "Reply with the feedback only.")

# "[ADMISSIBLE] a | b | c" up to the next "[SECTION]" header or the end (same pattern as
# jepa/training/features.py), plus the newline before it.
_ADMISSIBLE_SECTION = re.compile(r"\n?\[ADMISSIBLE\].*?(?=\n\s*\[[A-Z_]+\]|\Z)", re.S)


def read_jsonl(path: str) -> List[Dict]:
    """Same row order as jepa TransitionDataset.from_jsonl (skips blank lines only),
    so JEPA's train_rows index the same rows here."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_admissible(obs: str) -> str:
    return _ADMISSIBLE_SECTION.sub("", obs or "").strip()


def allowed_flag(obs: str, action: str) -> str:
    allowed = admissible_set(obs)
    if allowed is None:
        return "unknown"
    return "yes" if action.strip().lower() in allowed else "no"


def user_text(obs: str, action: str) -> str:
    return (f"Observation:\n{strip_admissible(obs)}\n\n"
            f"Action: {action.strip()}\n"
            f"Action allowed: {allowed_flag(obs, action)}")


def messages(obs: str, action: str) -> List[Dict]:
    """Chat messages for the prompt (the answer is added by the trainer)."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text(obs, action)}]


# ── Training rows ────────────────────────────────────────────────────────────

def load_rows(path: str) -> Dict:
    """sft/rows/<name>.json written by scripts/export_train_rows.py:
    {"buffer", "n_buffer_rows", "budgets": {"32": [...], ...}, "val_rows": [...] | null}"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d["budgets"] = {int(k): v for k, v in d["budgets"].items()}
    return d


def check_rows(rows: Dict, n_items: int) -> None:
    """Fail loudly if the rows file does not fit the buffer."""
    if rows.get("n_buffer_rows") not in (None, n_items):
        raise SystemExit(f"[Error] rows file was made for a buffer of {rows['n_buffer_rows']} rows, "
                         f"but the buffer has {n_items} — wrong buffer?")
    for b, r in rows["budgets"].items():
        if len(r) != b or max(r) >= n_items:
            raise SystemExit(f"[Error] budget {b}: bad row list (len {len(r)}, max {max(r)})")


# ── Scoring (Saanvi's functions, so F and D are measured identically) ───────

try:
    from wm.utility.score import normalize, word_f1
except Exception:  # pragma: no cover - same definitions, in case wm/ can't be imported
    def normalize(text: str) -> str:
        return " ".join(text.lower().split()).rstrip(".").strip()

    def word_f1(pred: str, true: str) -> float:
        p, t = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", true.lower())
        if not p or not t:
            return float(p == t)
        common = sum(min(p.count(w), t.count(w)) for w in set(p))
        if common == 0:
            return 0.0
        prec, rec = common / len(p), common / len(t)
        return 2 * prec * rec / (prec + rec)


def fidelity(preds: List[str], truths: List[str]) -> Dict[str, float]:
    n = max(1, len(truths))
    return {
        "exact": sum(normalize(p) == normalize(t) for p, t in zip(preds, truths)) / n,
        "f1": sum(word_f1(p, t) for p, t in zip(preds, truths)) / n,
        "n": len(truths),
    }


def holdout_games_split(items: List[Dict], k: int, seed: int = 42):
    """Validation split for the PILOT only: every row of k whole games is held out
    (same rule as train_jepa.py --holdout-games). Returns (val_idx, pool_idx shuffled)."""
    import random
    rng = random.Random(seed)
    games = sorted({t.get("game", "unknown") for t in items})
    rng.shuffle(games)
    held = set(games[:k])
    val = [i for i, t in enumerate(items) if t.get("game", "unknown") in held]
    pool = [i for i, t in enumerate(items) if t.get("game", "unknown") not in held]
    rng.shuffle(pool)
    return val, pool, sorted(held)


def pick_prediction_fields(state: Dict) -> Optional[Dict]:
    """The only fields of a branched state the SFT arm may read: id, obs, actions.
    (Never gold_* / true_*.)"""
    if not state.get("obs") or not state.get("actions"):
        return None
    return {"id": state["id"], "obs": state["obs"], "actions": list(state["actions"])}
