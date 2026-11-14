#!/usr/bin/env python3
"""
Ex-ante Feature 3: G's retrieval Recall@5 on a game's OWN 32 training transitions (leave-one-out).

    # in the encoder venv (NV-Embed needs transformers 4.42), on the GPU box, from the repo root:
    python scripts/recall_feature.py --buffers data/buffers/alfworld_games.jsonl \
        data/buffers/sciworld_games.jsonl data/buffers/bytesized32_games.jsonl \
        --out results/features/recall_at5.csv

Why not results/utility/g_recall_at5.csv: that file is measured on the held-out evaluation states, so
using it as a pre-training feature would read the test set. This one uses only what a team has before
deciding: the same n = 32 transitions as Features 1, 2 and 4.

Per game, for each of --seeds random draws of 32 transitions (same draws as capability_mix.py and
env_stats.py: random.Random(seed).sample): every transition in turn is the query, the other 31 are the
retrieval buffer, and it is a hit if G's retriever (rag_dropin/rag_arm.py: NV-Embed observation retrieval,
action re-rank, top 5) returns a transition with the same action template and object (rag_arm.relevant).
recall_at5 = hit share, averaged over draws. Every text is encoded once (NV-Embed, 4-bit); the run is
logged through track() as stage "recall_feature" (a feature cost, not an arm cost).
--fake-encoder runs the same pipeline with random vectors, to test it without a GPU.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import random
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rag_dropin"))
import numpy as np  # noqa: E402
from rag_arm import CachedEncoder, HierarchicalRetriever, relevant, transition  # noqa: E402


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


class FakeEncoder:
    def encode(self, texts, **_):
        out = []
        for t in texts:
            v = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(size=64)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffers", nargs="+", required=True)
    ap.add_argument("--out", default="results/features/recall_at5.csv")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--game-key", default=None)
    ap.add_argument("--fake-encoder", action="store_true")
    args = ap.parse_args()

    draws = []                                   # (env, game, seed, rows)
    for path in args.buffers:
        recs = load(path)
        key = args.game_key or ("game" if "game" in recs[0] else "task_type")
        env = recs[0].get("env", os.path.basename(path))
        games = collections.defaultdict(list)
        for r in recs:
            games[str(r.get(key))].append(r)
        for g, rs in sorted(games.items()):
            if len(rs) < args.n:
                continue
            for s in range(args.seeds):
                draws.append((env, g, s, random.Random(s).sample(rs, args.n)))

    texts = sorted({t for *_, rows in draws for r in rows
                    for t in (transition(r).observation, transition(r).action)})
    print(f"{len(draws)} draws, {len(texts)} unique texts to encode")

    from capability.compute_log import track
    with track("recall_feature", arm="G", env="+".join(sorted({d[0] for d in draws})), n=args.n,
               texts=len(texts), fake=args.fake_encoder):
        if args.fake_encoder:
            enc = FakeEncoder()
        else:
            from rag_arm import NVEmbedEncoder
            enc = NVEmbedEncoder("cuda")
        vecs = {}
        for i in range(0, len(texts), 256):
            batch = texts[i:i + 256]
            for t, v in zip(batch, np.asarray(enc.encode(batch))):
                vecs[t] = v
            print(f"  encoded {min(i + 256, len(texts))}/{len(texts)}", flush=True)
        cached = CachedEncoder(vecs)

        per_game = collections.defaultdict(list)
        for env, g, s, rows in draws:
            hits = 0
            for i, q in enumerate(rows):
                others = rows[:i] + rows[i + 1:]
                ret = HierarchicalRetriever(others, encoder=cached)
                qt = transition(q)
                hits += any(relevant(qt.action, x.action) for x in ret.retrieve(qt.observation, qt.action))
            per_game[(env, g)].append(hits / len(rows))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "game", "n", "recall_at5", "sd_recall_at5", "draws"])
        for (env, g), v in sorted(per_game.items()):
            w.writerow([env, g, args.n, round(st.mean(v), 4), round(st.pstdev(v), 4), len(v)])
    by_env = collections.defaultdict(list)
    for (env, _), v in per_game.items():
        by_env[env].append(st.mean(v))
    for env, v in by_env.items():
        print(f"{env:12s} games={len(v):3d} recall_at5 mean={st.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
