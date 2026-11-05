"""Task 3 sanity check: 3-shot retrieval world model on held-out ALFWorld transitions.

Pre-registered rule: docs/SANITY_CHECK.md (fixed before the first run; do not change after).

  # no GPU: trivial baselines only
  python -m sanity.three_shot --baselines-only
  # full check (needs LLM_BASE_URL)
  python -m sanity.three_shot

For each held-out transition, BM25 over (goal + last feedback + action) picks 3 examples from
the training budget; Qwen2.5-7B-Instruct predicts feedback_next; we score normalized exact match.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

import yaml

from wm.obs import parse_obs
from wm.retrieval import BM25
from wm.schema import read_jsonl

SYSTEM = ("You are a world model for a text-based household environment. Given the current "
          "observation and the agent's action, predict exactly the text the environment returns "
          "next. Reply with that text only.")
VERBS = {"go": "go to", "open": "open", "close": "close", "take": "take", "move": "move",
         "clean": "clean/heat/cool", "heat": "clean/heat/cool", "cool": "clean/heat/cool",
         "examine": "examine", "use": "use", "look": "look", "inventory": "inventory"}
FAILURE = "nothing happens"


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".").strip()


def query_text(row: Dict) -> str:
    return "%s %s %s" % (row["goal"], parse_obs(row["o_t"]).get("FEEDBACK", ""), row["a_t"])


def category(row: Dict) -> str:
    return VERBS.get(row["a_t"].split()[0], "other") if row["a_t"] else "other"


def show(row: Dict) -> str:
    return "Observation:\n%s\nAction: %s" % (row["o_t"], row["a_t"])


def build_messages(examples: List[Dict], target: Dict) -> List[Dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        msgs.append({"role": "user", "content": show(ex)})
        msgs.append({"role": "assistant", "content": ex["feedback_next"]})
    msgs.append({"role": "user", "content": show(target)})
    return msgs


def sample_heldout(rows: List[Dict], n: int, seed: int, mix: Dict[str, float]) -> List[Dict]:
    """Stratified by policy, proportional to the configured mix (renormalized over present policies)."""
    rng = random.Random(seed)
    by_policy = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["transition_id"]):
        by_policy[r["policy"]].append(r)
    present = {k: v for k, v in mix.items() if by_policy.get(k)}
    total = sum(present.values())
    names = list(present)
    picked = []
    for i, k in enumerate(names):
        want = n - len(picked) if i == len(names) - 1 else int(round(n * present[k] / total))
        pool = by_policy[k][:]
        rng.shuffle(pool)
        picked += pool[:want]
    return picked


def em_and_ci(hits: List[int], seed: int, reps: int = 1000) -> Dict:
    if not hits:
        return {"em": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(hits) for _ in hits) / len(hits) for _ in range(reps))
    return {"em": round(sum(hits) / len(hits), 4),
            "ci95": [round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)],
            "n": len(hits)}


def evaluate_rule(model: Dict, baselines: Dict, per_cat: Dict) -> Dict:
    em = model["em"]
    best_base = max(b["em"] for b in baselines.values())
    go, fail = per_cat.get("go to", {}).get("em"), per_cat.get("failure", {}).get("em")
    checks = {
        "1_em_between_25_and_80": 0.25 <= em <= 0.80,
        "2_beats_both_baselines_by_10pts": em >= best_base + 0.10,
        "3_go_to_below_failure": (go < fail) if go is not None and fail is not None else None,
    }
    checks["PASS"] = all(v is not False for v in checks.values()) and checks["3_go_to_below_failure"] is not None
    return checks


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="alfworld")
    p.add_argument("--train", default="data/raw/alfworld_train.jsonl.gz")
    p.add_argument("--heldout", default="data/raw/alfworld_heldout.jsonl.gz")
    p.add_argument("--budget", default="data/budgets/alfworld_seed0_n1024.txt")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", default="results/sanity_alfworld.json")
    p.add_argument("--baselines-only", action="store_true", help="skip the LLM (no GPU needed)")
    args = p.parse_args(argv)

    with open("configs/shared.yaml") as f:
        mix = yaml.safe_load(f)["policy_mix"]
    with open(args.budget) as f:
        budget_ids = set(line.strip() for line in f if line.strip())
    pool = [r for r in read_jsonl(args.train) if r["transition_id"] in budget_ids]
    heldout = sample_heldout([r for r in read_jsonl(args.heldout) if r["split"] == "heldout"],
                             args.n, args.seed, mix)
    assert pool and heldout, "empty pool or held-out set"
    assert not {r["task_id"] for r in pool} & {r["task_id"] for r in heldout}, "task leakage"

    bm25 = BM25([query_text(r) for r in pool])
    client = None
    if not args.baselines_only:
        from wm.llm_client import LLMClient
        client = LLMClient()

    records = []
    for i, row in enumerate(heldout):
        examples = [pool[j] for j in bm25.top_k(query_text(row), args.k)]
        rec = {"transition_id": row["transition_id"], "a_t": row["a_t"], "policy": row["policy"],
               "category": category(row), "is_failure": normalize(row["feedback_next"]) == FAILURE,
               "truth": row["feedback_next"],
               "copy_top1": examples[0]["feedback_next"], "nothing_happens": "Nothing happens."}
        if client:
            out = client.chat(build_messages(examples, row), temperature=0.0, max_tokens=128)
            rec["pred"], rec["cache_key"] = out["text"].strip(), out["cache_key"]
        records.append(rec)
        if (i + 1) % 25 == 0:
            print("%d/%d" % (i + 1, len(heldout)), flush=True)

    def hits(field, recs):
        return [int(normalize(r[field]) == normalize(r["truth"])) for r in recs]

    result = {"config": vars(args), "pool_size": len(pool), "heldout_size": len(heldout),
              "heldout_policies": dict(Counter(r["policy"] for r in records)),
              "baselines": {f: em_and_ci(hits(f, records), args.seed) for f in ("copy_top1", "nothing_happens")}}
    if client:
        result["model"] = em_and_ci(hits("pred", records), args.seed)
        groups = defaultdict(list)
        for r in records:
            groups[r["category"]].append(r)
            if r["is_failure"]:
                groups["failure"].append(r)
        result["per_category"] = {k: em_and_ci(hits("pred", v), args.seed) for k, v in sorted(groups.items())}
        result["rule"] = evaluate_rule(result["model"], result["baselines"], result["per_category"])
    else:
        result["rule"] = "not evaluated (--baselines-only)"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out.replace(".json", "_predictions.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
