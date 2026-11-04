"""
GPU-hour accounting for every arm (Phase 2, Siddanth). One shared logger so the cost table compares
like with like.

Wrap any costed step:

    from capability.compute_log import track

    with track("jepa_train", env="alfworld", game=game_id, n=256, arm="D"):
        trainer.train(...)

    with track("nv_embed_encode", env="alfworld", n=1440, arm="shared"):
        encoder.encode(texts)

Each block appends one line to results/compute/compute_log.jsonl:
    {"stage", "arm", "env", "game", "n", "wall_s", "gpu_s", "gpu_name", "peak_mem_gb", "host", "time", ...}
gpu_s = wall-clock seconds while a CUDA device was in use (synchronised at start and end), so it is the
GPU time this step occupied. GPU-hours = gpu_s / 3600. Stages we need for the cost table:
    jepa_train, sft_train, nv_embed_encode, rag_index_build, rag_retrieve, utility_scoring, probe
Summarise with:  python scripts/compute_table.py
Pure stdlib + optional torch; works in both venvs.
"""
from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager

LOG_PATH = os.environ.get("COMPUTE_LOG", "results/compute/compute_log.jsonl")


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


@contextmanager
def track(stage: str, arm: str = "", env: str = "", game: str = "", n: int | str = "", **extra):
    torch = _torch()
    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch is not None:
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rec = {"stage": stage, "arm": arm, "env": env, "game": game, "n": n,
               "wall_s": round(wall, 3),
               "gpu_s": round(wall, 3) if torch is not None else 0.0,
               "gpu_name": torch.cuda.get_device_name(0) if torch is not None else "cpu",
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch is not None else 0.0,
               "host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
