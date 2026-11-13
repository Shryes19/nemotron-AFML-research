#!/usr/bin/env python3
"""Write Saanvi-compatible G predictions from exact branched-state files."""
import argparse
import json
from pathlib import Path

from rag_arm import FrozenQwen, HierarchicalRetriever, first, prompt, read_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", required=True, help="Transition JSONL used as the retrieval buffer")
    p.add_argument("--states", required=True, help="data/eval/<set>_branched_states.jsonl")
    p.add_argument("--set", required=True, choices=["alfworld", "sciworld", "sciworld_seen", "bytesized32"])
    p.add_argument("--budget", type=int, required=True, choices=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--outdir", default="predictions")
    args = p.parse_args()
    buffer = read_jsonl(args.buffer)[:args.budget]
    retriever, model = HierarchicalRetriever(buffer), FrozenQwen()
    out = Path(args.outdir) / f"{args.set}_b{args.budget}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.states):
            state_id = first(row, "id", "task_id", "game", "task_type")
            obs = first(row, "observation", "obs", "state")
            actions = row.get("actions") or row.get("candidate_actions")
            if not isinstance(actions, list) or not actions:
                raise ValueError(f"{state_id}: expected actions/candidate_actions list")
            preds = []
            for action in actions:
                demos = retriever.retrieve(obs, str(action))
                preds.append(model.predict(prompt(obs, str(action), demos)))
            handle.write(json.dumps({"id": state_id, "arm": "G", "budget": args.budget,
                                     "actions": actions, "pred_next": preds}) + "\n")
    print(out)


if __name__ == "__main__":
    main()
