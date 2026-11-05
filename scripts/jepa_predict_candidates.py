"""
scripts/jepa_predict_candidates.py
──────────────────────────────────
Run a trained JEPA on a list of (state, candidate actions) and write what it
predicts for each action to a file. Saanvi's decision-utility scorer reads that
file afterwards.

Why a file and not a function call
──────────────────────────────────
JEPA needs NV-Embed-v2 (~6 GB on the GPU) and the scorer needs Qwen2.5-7B-AWQ
(~5-6 GB + cache). They do not fit on the 12 GB card together, so we run them
one after the other: this script first (NV-Embed loaded), then the scorer
(Qwen loaded). The file is the hand-off.

Input  (--queries, JSONL, one line per branched state)
    {"id": "alfworld/pick_3/step_7", "obs": "<observation text>",
     "actions": ["go to fridge 1", "open fridge 1", ...]}
    Optional keys are copied through unchanged (e.g. "gold", "game").

Output (--out, JSONL, one line per query)
    {"id": ..., "budget": 256, "checkpoint": ..., <optional keys copied>,
     "candidates": [
        {"action": "go to fridge 1",
         "topk": [{"text": "<nearest buffer state>", "sim": 0.83}, ...]},   # best first
        ...]}
    With --target-key goal: each query may also carry a target text under
    that key ("goal" by default), and each candidate then gets "target_sim"
    (cosine of the predicted next state with that text) and the query gets
    "ranking" (candidate indices, best first). No decoding needed for that.

Which states can JEPA answer with?
──────────────────────────────────
Only the next states from the rows that budget was trained on (saved in the
checkpoint as "train_rows" by train_jepa.py). Otherwise the budget-32 model
could "answer" with states it never saw and the budget comparison would be
unfair. Old checkpoints without train_rows fall back to the whole --buffer and
print a warning.

Usage (on the GPU box, inside .venv-enc)
    python scripts/jepa_predict_candidates.py \
        --checkpoint outputs/jepa/alfworld_games_steps2000_sepact_adm_frozen/budget_256.pt \
        --buffer data/buffers/alfworld_games.jsonl \
        --queries data/eval/alfworld_branch_states.jsonl \
        --out outputs/jepa_candidates/alfworld_b256.jsonl --k 5 --real-encoder

    # Code check on CPU (random vectors)
    python scripts/jepa_predict_candidates.py --checkpoint <ckpt> --buffer <buffer> \
        --queries <queries> --out /tmp/x.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.compute import env_label, track
from jepa.encoder.load import load_encoder
from jepa.planning.latent_world_model import LatentWorldModel
from jepa.training.buffer import TransitionDataset


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def read_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="budget_<n>.pt from train_jepa.py")
    p.add_argument("--buffer", required=True,
                   help="the SAME buffer the checkpoint was trained on (its rows are the answers)")
    p.add_argument("--queries", required=True, help="JSONL of {id, obs, actions}")
    p.add_argument("--out", required=True, help="output JSONL")
    p.add_argument("--k", type=int, default=5, help="nearest buffer states per action (default 5)")
    p.add_argument("--target-key", default="goal",
                   help="query key holding a target state text for ranking (default: goal; "
                        "ignored when a query has no such key)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--real-encoder", action="store_true", help="use NV-Embed-v2 (GPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ecfg = load_config(args.config).get("encoder", {})
    items = TransitionDataset.from_jsonl(args.buffer).transitions
    extra = torch.load(args.checkpoint, map_location="cpu").get("extra", {})

    rows = extra.get("train_rows")
    if rows is None:
        print("[Warn] checkpoint has no train_rows (trained before 5 Nov 2026) — answering "
              "from the WHOLE buffer, which is unfair to small budgets. Retrain to fix.")
        answer_items = items
    else:
        if max(rows) >= len(items):
            sys.exit(f"[Error] checkpoint rows go up to {max(rows)} but {args.buffer} has "
                     f"{len(items)} rows — is this the buffer it was trained on?")
        if extra.get("buffer") and Path(extra["buffer"]).name != Path(args.buffer).name:
            print(f"[Warn] checkpoint was trained on {extra['buffer']}, you passed {args.buffer}")
        answer_items = [items[i] for i in rows]

    queries = read_jsonl(args.queries)
    bad = [q.get("id", i) for i, q in enumerate(queries) if not q.get("obs") or not q.get("actions")]
    if bad:
        sys.exit(f"[Error] queries missing 'obs' or 'actions': {bad[:5]}")

    encoder = load_encoder(
        real=args.real_encoder, device=args.device,
        model_name=ecfg.get("model_name"), quantize=bool(ecfg.get("quantize", True)),
        batch_size=int(ecfg.get("batch_size", 32)), max_length=int(ecfg.get("max_length", 512)),
    )
    budget = extra.get("budget", len(answer_items))
    print(f"[JEPA] {args.checkpoint}  budget={budget}  answer states from {len(answer_items)} rows  "
          f"queries={len(queries)}  k={args.k}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_actions = 0
    with track("jepa_predict", arm="D", env=env_label(items), n=budget):
        wm = LatentWorldModel.from_checkpoint(args.checkpoint, encoder,
                                              buffer_items=answer_items, device=args.device)
        with open(out_path, "w") as f:
            for q in queries:
                acts = list(q["actions"])
                z = wm.predict(q["obs"], acts)
                topk = wm.decode_topk(z, k=args.k)
                rec = {k: v for k, v in q.items() if k not in ("obs", "actions")}
                rec.update({"budget": budget, "checkpoint": args.checkpoint, "candidates": [
                    {"action": a, "topk": [{"text": t, "sim": round(s, 4)} for t, s in tk]}
                    for a, tk in zip(acts, topk)]})
                if q.get(args.target_key):
                    ranked = wm.rank_by_target(z, q[args.target_key])
                    for i, s in ranked:
                        rec["candidates"][i]["target_sim"] = round(s, 4)
                    rec["ranking"] = [i for i, _ in ranked]
                f.write(json.dumps(rec) + "\n")
                n_actions += len(acts)
    print(f"[Done] {len(queries)} states, {n_actions} actions → {out_path}")


if __name__ == "__main__":
    main()
