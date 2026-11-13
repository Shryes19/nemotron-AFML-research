#!/usr/bin/env python3
"""Generate G output from NV-Embed-prepared prompts via the shared Qwen vLLM server."""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_arm import FrozenQwen, read_jsonl

class VLLMQwen:
    def __init__(self):
        from wm.llm_client import LLMClient
        self.client = LLMClient()
    def predict(self, text):
        return self.client.chat([{"role": "user", "content": text}], temperature=0.0, max_tokens=128)["text"].strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-cache", required=True, help="NV-Embed prompt cache from prepare_g_prompts.py")
    p.add_argument("--states", required=True, help="data/eval/<set>_branched_states.jsonl")
    p.add_argument("--set", required=True, choices=["alfworld", "sciworld", "sciworld_seen", "bytesized32"])
    p.add_argument("--budget", type=int, required=True, choices=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--outdir", default="predictions"); p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    out = Path(args.outdir) / f"{args.set}_b{args.budget}.jsonl"
    cached = read_jsonl(args.prompt_cache)
    states = read_jsonl(args.states)
    if len(cached) != len(states): raise ValueError("Prompt cache and branched states differ in row count")
    if [(x["id"], x["actions"]) for x in cached] != [(x["id"], x["actions"]) for x in states]:
        raise ValueError("Prompt cache does not match exact branched states/actions")
    flat = [p for row in cached for p in row["prompts"]]
    from wm.utility.score import track
    with track("g_generation", arm="G", env=args.set, n=args.budget):
        model = VLLMQwen() if args.backend == "vllm" else FrozenQwen()
        if args.backend == "vllm":
            with ThreadPoolExecutor(args.workers) as pool: predictions = list(pool.map(model.predict, flat))
        else: predictions = [model.predict(p) for p in flat]
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_it = iter(predictions)
    with out.open("w", encoding="utf-8") as handle:
        for row in cached:
            handle.write(json.dumps({"id": row["id"], "arm": "G", "budget": args.budget,
                                     "actions": row["actions"], "pred_next": [next(pred_it) for _ in row["actions"]]}) + "\n")
    print(out)


if __name__ == "__main__":
    main()
