#!/usr/bin/env python3
"""G, stage 2 (GPU, vLLM): answer the prompts prepare_g_prompts.py wrote, in the scoring format.

    python rag_dropin/generate_predictions.py --prompts ~/g_prompts/alfworld_b256.jsonl \
        --states data/eval/alfworld_branched_states.jsonl --set alfworld --budget 256 --outdir ~/g_candidates

Output <outdir>/<set>_b<n>.jsonl: {"id", "arm": "G", "budget", "actions", "pred_next"}
(docs/DECISION_UTILITY.md). The LLM is the vLLM server in LLM_BASE_URL / LLM_MODEL
(Qwen2.5-7B-Instruct-AWQ). Calls run in parallel and are cached, so a rerun after a crash is quick;
an existing output is skipped. A prompt the server rejects gets an empty prediction and is counted.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for wm.*

from rag_arm import FrozenQwen, VLLMQwen, read_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="<set>_b<n>.jsonl from prepare_g_prompts.py")
    p.add_argument("--states", required=True, help="data/eval/<set>_branched_states.jsonl")
    p.add_argument("--set", required=True, choices=["alfworld", "sciworld", "sciworld_seen", "bytesized32"])
    p.add_argument("--budget", type=int, required=True, choices=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--outdir", default="predictions")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--workers", type=int, default=32, help="parallel requests to the vLLM server")
    p.add_argument("--limit", type=int, default=None, help="first N states only (smoke test)")
    args = p.parse_args()

    out = Path(args.outdir).expanduser() / f"{args.set}_b{args.budget}.jsonl"
    if out.exists() and out.stat().st_size > 0 and args.limit is None:
        print(f"[G] {out} exists, skipped")
        return
    cached = read_jsonl(args.prompts)[:args.limit]
    states = read_jsonl(args.states)[:args.limit]
    if [(c["id"], c["actions"]) for c in cached] != [(s["id"], s["actions"]) for s in states]:
        sys.exit(f"[Error] {args.prompts} does not match {args.states} (ids or actions differ)")
    flat = [t for c in cached for t in c["prompts"]]

    failed = []

    def safe(predict):
        def run(text):
            try:
                return predict(text)
            except RuntimeError as e:  # e.g. HTTP 400, prompt too long: keep going, count it
                failed.append(str(e)[:200])
                return ""
        return run

    from wm.utility.score import track  # GPU-hour log; WM_NO_COMPUTE_LOG=1 turns it off
    with track("g_generation", arm="G", env=args.set, n=args.budget):
        if args.backend == "vllm":
            predict = safe(VLLMQwen().predict)
            with ThreadPoolExecutor(args.workers) as pool:
                preds = list(pool.map(predict, flat))
        else:
            predict = safe(FrozenQwen().predict)
            preds = [predict(t) for t in flat]
    if len(failed) > 0.01 * len(flat):  # nothing written, so a rerun retries (answers so far are cached)
        sys.exit(f"[Error] {len(failed)} of {len(flat)} predictions failed (first: {failed[0]}); "
                 "check the vLLM server")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    it = iter(preds)
    with tmp.open("w", encoding="utf-8") as handle:
        for c in cached:
            handle.write(json.dumps({"id": c["id"], "arm": "G", "budget": args.budget, "actions": c["actions"],
                                     "pred_next": [next(it) for _ in c["actions"]]}) + "\n")
    tmp.replace(out)
    print(f"[G] {out}: {len(cached)} states, {len(flat)} predictions, {len(failed)} failed"
          + (f" (first: {failed[0]})" if failed else ""))


if __name__ == "__main__":
    main()
