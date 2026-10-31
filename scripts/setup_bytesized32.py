#!/usr/bin/env python3
"""
Set up ByteSized32-SP (Wang et al. 2024) as our 4th environment.

    git clone --depth 1 https://github.com/cognitiveailab/GPT-simulator.git ../GPT-simulator
    mkdir -p data/raw/bytesized32 && unzip -o ../GPT-simulator/data/test.zip -d data/raw/bytesized32
    python scripts/setup_bytesized32.py --raw data/raw/bytesized32/test.jsonl \
                                        --out data/buffers/bytesized32_games.jsonl

    # for the n = 32 ... 1024 budgets use the train split (171k transitions, median 1,489 per game):
    unzip -o ../GPT-simulator/data/train.zip -d data/raw/bytesized32
    python scripts/setup_bytesized32.py --raw data/raw/bytesized32/train.jsonl --max-per-game 1200 \
        --out data/buffers/bytesized32_train.jsonl --report results/capability/bs32_validation_train.json

Writes one buffer in the common format (obs / action / obs_next / obs_next_full / env / game / cap_tag),
31 games, 2,954 transitions, with cap_tags = ground truth from the JSON state diff. Also runs the text
tagger on the same transitions and saves the agreement to results/capability/bs32_validation.json:
that is how we check the regex tagger we use on ALFWorld / ScienceWorld.
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from capability import bytesized32 as B  # noqa: E402
from capability.tagger import TAGS, tag_record  # noqa: E402

COMPARED = ["object_state", "containment_location", "numeric_physical", "action_failure"]


def _numbers(state: dict) -> list:
    text = " ".join(state.get(k, "") for k in ("look", "observation", "inventory"))
    return sorted(re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"\(ID: \d+\)", "", text)))


def stream_sample(path: str, k: int, seed: int) -> list:
    """Read the JSONL line by line (train.jsonl is too big to load whole); keep <= k per game."""
    rng, seen, keep = random.Random(seed), collections.Counter(), collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            g = rec["game"]
            seen[g] += 1
            if not k or len(keep[g]) < k:
                keep[g].append(rec)
            else:
                j = rng.randrange(seen[g])
                if j < k:
                    keep[g][j] = rec
    return [r for g in sorted(keep) for r in keep[g]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="GPT-simulator data/test.jsonl (unzipped)")
    ap.add_argument("--out", default="data/buffers/bytesized32_games.jsonl")
    ap.add_argument("--report", default="results/capability/bs32_validation.json")
    ap.add_argument("--max-per-game", type=int, default=0,
                    help="reservoir-sample at most this many transitions per game (use ~1200 with train.jsonl, "
                         "which has 171k transitions / 1.7 GB; 0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = stream_sample(args.raw, args.max_per_game, args.seed)
    recs = [B.to_buffer_record(r) for r in raw]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    agree, vis, vis_hit, hidden = 0, 0, 0, 0
    split = collections.Counter()
    with open(args.out, "w") as f:
        for r, rr in zip(recs, raw):
            t = tag_record(r)
            r["text_cap_tag"], r["text_cap_tags"] = t["primary"], t["tags"]
            f.write(json.dumps(r) + "\n")
            g, p = set(r["cap_tags"]), set(t["tags"])
            for x in COMPARED:
                if x in g and x in p:
                    tp[x] += 1
                elif x in p:
                    fp[x] += 1
                elif x in g:
                    fn[x] += 1
            # partial_obs does not exist in the ground truth (full state), so "none" and "partial_obs" agree
            agree += (r["cap_tag"] == t["primary"]) or (r["cap_tag"] == "none" and t["primary"] == "partial_obs")
            if "numeric_physical" in g:
                if _numbers(rr["current_state"]) != _numbers(rr["tick_state"]):
                    vis += 1
                    vis_hit += "numeric_physical" in p
                else:
                    hidden += 1
            for x in g or {"none"}:
                split[(x, "env_driven" if r["env_driven"] else "action_only")] += 1

    n = len(recs)
    report = {
        "transitions": n,
        "games": len({r["game"] for r in recs}),
        "per_tag": {x: {"precision": round(tp[x] / max(1, tp[x] + fp[x]), 3),
                        "recall": round(tp[x] / max(1, tp[x] + fn[x]), 3),
                        "support": tp[x] + fn[x]} for x in COMPARED},
        "primary_agreement": round(agree / n, 3),
        "numeric_changes_visible_in_text": vis,
        "numeric_changes_hidden_from_text": hidden,
        "numeric_recall_on_visible": round(vis_hit / max(1, vis), 3),
        "ground_truth_primary_share": {k: round(v / n, 3) for k, v in
                                       collections.Counter(r["cap_tag"] for r in recs).most_common()},
        "tag_by_wang_split": {f"{a}|{b}": c for (a, b), c in sorted(split.items())},
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {n} transitions, {report['games']} games -> {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
