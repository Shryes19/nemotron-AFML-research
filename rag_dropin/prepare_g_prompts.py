#!/usr/bin/env python3
"""G, stage 1 (GPU, NV-Embed): retrieve examples and write every prompt, before vLLM starts.

NV-Embed-v2 (4-bit) and the vLLM Qwen server cannot share the 12 GB GPU, so G runs in two stages
(Rohini's design). This stage loads NV-Embed ONCE, encodes every text of a set in one batched pass
(the budgets' training rows, the branched states, their candidate actions), then builds the
retriever for each budget from those vectors. Output per set and budget:

    <out-dir>/<set>_b<n>.jsonl   {"id", "actions", "prompts": [one prompt per action]}
    <out-dir>/recall_at5.csv     env, game, budget, n_states, recall_at5   (feature-matrix column)

Recall@5 here: for each branched state, retrieve with (state, gold action) and count a hit when a
retrieved example has the same action template and object (rag_arm.relevant, model-free).
Units: task id (ALFWorld), task type (ScienceWorld), game (ByteSized32).

    python rag_dropin/prepare_g_prompts.py --rows ~/g_buffers --out-dir ~/g_prompts
Finished files are skipped. Needs the NV-Embed venv (torch, transformers 4.42, bitsandbytes).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for jepa.* and wm.*

from rag_arm import (CachedEncoder, HierarchicalRetriever, NVEmbedEncoder, prompt,  # noqa: E402
                     read_jsonl, relevant, transition)

UNIT = {"alfworld": "task_id", "sciworld": "task_type", "sciworld_seen": "task_type", "bytesized32": "game"}


def build_set(set_name, rows_by_budget, states, encoder, out_dir):
    """Write the prompt files of one set; return its Recall@5 rows."""
    texts = {s["obs"] for s in states} | {str(a) for s in states for a in s["actions"]}
    for rows in rows_by_budget.values():
        texts |= {r["obs"] for r in rows} | {r["action"] for r in rows}
    texts = sorted(texts)
    vectors = dict(zip(texts, encoder.encode(texts)))
    cache = CachedEncoder(vectors)

    recall_rows = []
    for n, rows in sorted(rows_by_budget.items()):
        r = HierarchicalRetriever(rows, encoder=cache)
        out = out_dir / f"{set_name}_b{n}.jsonl"
        if not (out.exists() and out.stat().st_size > 0):
            tmp = out.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for s in states:
                    ps = [prompt(s["obs"], str(a), r.retrieve_vec(vectors[s["obs"]], vectors[str(a)]))
                          for a in s["actions"]]
                    f.write(json.dumps({"id": s["id"], "actions": s["actions"], "prompts": ps}) + "\n")
            tmp.replace(out)
        print(f"[G prompts] {out}")

        hits = defaultdict(list)
        for s in states:
            got = r.retrieve_vec(vectors[s["obs"]], vectors[str(s["gold_action"])])
            hits[str(s[UNIT[set_name]])].append(any(relevant(s["gold_action"], x.action) for x in got))
        for unit, h in sorted(hits.items()):
            recall_rows.append({"env": set_name, "game": unit, "budget": n, "n_states": len(h),
                                "recall_at5": round(sum(h) / len(h), 4)})
    return recall_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="folder of <set>_b<n>.jsonl from export_g_buffers.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sets", default="alfworld sciworld sciworld_seen bytesized32")
    p.add_argument("--budgets", default="32 64 128 256 512 1024")
    p.add_argument("--states-dir", default="data/eval")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = Path(args.rows).expanduser()
    budgets = [int(b) for b in args.budgets.split()]

    from wm.utility.score import track  # GPU-hour log (stage rag_index_build)
    encoder = None
    recall = []
    for set_name in args.sets.split():
        rows_by_budget = {}
        for n in budgets:
            rows = read_jsonl(rows_dir / f"{set_name}_b{n}.jsonl")
            if len(rows) != n or not all(transition(x).next_observation for x in rows):
                sys.exit(f"[Error] {set_name}_b{n}: expected {n} rows with obs_next, got {len(rows)}")
            rows_by_budget[n] = rows
        states = read_jsonl(Path(args.states_dir) / f"{set_name}_branched_states.jsonl")
        with track("rag_index_build", arm="G", env=set_name, n=len(states)):
            if encoder is None:
                encoder = NVEmbedEncoder(args.device)
            recall += build_set(set_name, rows_by_budget, states, encoder, out_dir)

    with (out_dir / "recall_at5.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["env", "game", "budget", "n_states", "recall_at5"])
        w.writeheader()
        w.writerows(recall)
    print(f"[G prompts] {out_dir / 'recall_at5.csv'}: {len(recall)} rows")


if __name__ == "__main__":
    main()
