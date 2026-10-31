"""
jepa/training/embed_cache.py
────────────────────────────
Encode a whole buffer ONCE and save the vectors to disk.

Why: running the 8B encoder is by far the slowest step. Training at six data
budgets, or probing every game, only needs different SUBSETS of the same
vectors — so we encode every transition once, cache the result, and slice it.

The cache file name includes a fingerprint of the model name and every text in
the buffer, so if the data or the encoder changes, a fresh cache is built
automatically. Delete outputs/cache/ to force a rebuild.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def context_text(t: Dict) -> str:
    """The text the predictor sees as input: current observation + action."""
    return f"Observation: {t['obs']}\nAction: {t['action']}"


def _fingerprint(model_tag: str, transitions: List[Dict]) -> str:
    h = hashlib.sha1(model_tag.encode("utf-8"))
    for t in transitions:
        h.update(context_text(t).encode("utf-8"))
        h.update(b"\x00")
        h.update(t["obs_next"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def encode_buffer(
    encoder,
    transitions: List[Dict],
    cache_dir: Optional[str] = "outputs/cache",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        z_ctx:  (N, D)  encoder(obs + action)   — predictor input
        z_next: (N, D)  encoder(obs_next)       — what we try to predict
    Row i of both tensors belongs to transitions[i].
    """
    model_tag = getattr(encoder, "model_name", type(encoder).__name__)
    cache_path = None

    if cache_dir:
        safe_tag = model_tag.replace("/", "_")
        fp = _fingerprint(model_tag, transitions)
        cache_path = Path(cache_dir) / f"{safe_tag}_{len(transitions)}_{fp}.pt"
        if cache_path.exists():
            data = torch.load(cache_path)
            print(f"[Cache] Loaded embeddings from {cache_path}")
            return data["z_ctx"], data["z_next"]

    print(f"[Cache] Encoding {len(transitions)} transitions with {model_tag} ...")
    z_ctx = encoder.encode([context_text(t) for t in transitions])
    z_next = encoder.encode([t["obs_next"] for t in transitions])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z_ctx": z_ctx, "z_next": z_next}, cache_path)
        print(f"[Cache] Saved to {cache_path}")

    return z_ctx, z_next
