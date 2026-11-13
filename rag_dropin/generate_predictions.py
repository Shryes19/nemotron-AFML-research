#!/usr/bin/env python3
"""Write G predictions for one branched-state set and one budget (docs/DECISION_UTILITY.md format).

--buffer must hold exactly the n training rows arms F and D used at this budget
(export_g_buffers.py writes them from the JEPA checkpoints). Run from the repo root:

    python rag_dropin/generate_predictions.py --buffer g_buffers/alfworld_b256.jsonl \
        --states data/eval/alfworld_branched_states.jsonl --set alfworld --budget 256 --outdir ~/g_candidates

The default backend is the vLLM server in LLM_BASE_URL / LLM_MODEL (Qwen2.5-7B-Instruct-AWQ);
calls run in parallel and are cached, so a rerun after a crash is quick. An existing output is skipped.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for wm.*

from rag_arm import FrozenQwen, HierarchicalRetriever, VLLMQwen, prompt, read_jsonl  # noqa: E402


def build_prompts(states, retriever):
    """One prompt per (state, candidate action), in the state's own action order."""
    obs_vecs = retriever.encode([s["obs"] for s in states])
    actions = [str(a) for s in states for a in s["actions"]]
    act_vecs = iter(retriever.encode(actions))
    prompts = []
    for s, q_obs in zip(states, obs_vecs):
        prompts.append([prompt(s["obs"], str(a), retriever.retrieve_vec(q_obs, next(act_vecs)))
                        for a in s["actions"]])
    return prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True, help="the budget's exact training rows (obs, action, obs_next)")
    p.add_argument("--states", required=True, help="data/eval/<set>_branched_states.jsonl")
    p.add_argument("--set", required=True, choices=["alfworld", "sciworld", "sciworld_seen", "bytesized32"])
    p.add_argument("--budget", type=int, required=True, choices=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--outdir", default="predictions")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--workers", type=int, default=16, help="parallel requests to the vLLM server")
    p.add_argument("--device", default="cpu", help="retriever device (vLLM holds the GPU)")
    p.add_argument("--limit", type=int, default=None, help="first N states only (smoke test)")
    args = p.parse_args()

    out = Path(args.outdir).expanduser() / f"{args.set}_b{args.budget}.jsonl"
    if out.exists() and out.stat().st_size > 0 and args.limit is None:
        print(f"[G] {out} exists, skipped")
        return
    buffer = read_jsonl(args.buffer)
    if len(buffer) != args.budget:
        sys.exit(f"[Error] {args.buffer} has {len(buffer)} rows, expected exactly {args.budget} "
                 "(use the rows export_g_buffers.py writes, so G sees what F and D trained on)")
    states = read_jsonl(args.states)[:args.limit]

    from wm.utility.score import track  # GPU-hour log; WM_NO_COMPUTE_LOG=1 turns it off
    with track("g_generation", arm="G", env=args.set, n=args.budget):
        retriever = HierarchicalRetriever(buffer, device=args.device)
        prompts = build_prompts(states, retriever)
        flat = [t for ps in prompts for t in ps]
        if args.backend == "vllm":
            model = VLLMQwen()
            with ThreadPoolExecutor(args.workers) as pool:
                preds = list(pool.map(model.predict, flat))
        else:
            model = FrozenQwen()
            preds = [model.predict(t) for t in flat]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    it = iter(preds)
    with tmp.open("w", encoding="utf-8") as handle:
        for s in states:
            handle.write(json.dumps({"id": s["id"], "arm": "G", "budget": args.budget,
                                     "actions": s["actions"], "pred_next": [next(it) for _ in s["actions"]]}) + "\n")
    tmp.replace(out)
    print(f"[G] {out}: {len(states)} states, {len(flat)} predictions")


if __name__ == "__main__":
    main()
