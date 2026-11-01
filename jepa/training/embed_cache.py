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
    """
    The text the predictor sees as input: action + current observation.

    The ACTION GOES FIRST on purpose. The encoder reads at most 512 tokens and
    ALFWorld observations (goal, room, inventory, admissible-action list) can
    run past 700 tokens; with the action last it would be cut off and the model
    could not tell which action was taken.
    """
    return make_context(t["obs"], t["action"])


def make_context(obs: str, action: str) -> str:
    return f"Action: {action}\nObservation: {obs}"


# Text formats for "separate" mode — training and inference must use the same
def obs_text(obs: str) -> str:
    return f"Observation: {obs}"


def action_text(action: str) -> str:
    return f"Action: {action}"


def _fingerprint(model_tag: str, transitions: List[Dict], mode: str = "joint") -> str:
    h = hashlib.sha1(model_tag.encode("utf-8"))
    if mode != "joint":                       # keep old cache names valid for joint
        h.update(mode.encode("utf-8"))
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
    mode: str = "joint",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        mode: "joint"    — one vector for "action + observation" text      (N, D)
              "separate" — observation and action encoded on their own and
                           placed side by side [z_obs | z_action]          (N, 2D)
                           so a short action isn't drowned out by a long
                           observation.
    Returns:
        z_ctx:  predictor input, (N, D) or (N, 2D)
        z_next: (N, D)  encoder(obs_next) — what we try to predict
    Row i of both tensors belongs to transitions[i].
    """
    if mode not in ("joint", "separate"):
        raise ValueError(f"mode must be 'joint' or 'separate', got {mode!r}")
    model_tag = getattr(encoder, "model_name", type(encoder).__name__)
    cache_path = None

    if cache_dir:
        safe_tag = model_tag.replace("/", "_")
        fp = _fingerprint(model_tag, transitions, mode)
        tag = "" if mode == "joint" else f"_{mode}"
        cache_path = Path(cache_dir) / f"{safe_tag}_{len(transitions)}{tag}_{fp}.pt"
        if cache_path.exists():
            data = torch.load(cache_path)
            print(f"[Cache] Loaded embeddings from {cache_path}")
            return data["z_ctx"], data["z_next"]

    print(f"[Cache] Encoding {len(transitions)} transitions with {model_tag} ({mode}) ...")
    from jepa.compute import env_label, track
    with track("nv_embed_encode", arm="shared", env=env_label(transitions), game="*",
               n=len(transitions), model=model_tag, mode=mode):
        if mode == "joint":
            z_ctx = encoder.encode([context_text(t) for t in transitions])
        else:
            z_obs = encoder.encode([obs_text(t["obs"]) for t in transitions])
            z_act = encoder.encode([action_text(t["action"]) for t in transitions])
            z_ctx = torch.cat([z_obs, z_act], dim=1)
        z_next = encoder.encode([t["obs_next"] for t in transitions])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z_ctx": z_ctx, "z_next": z_next}, cache_path)
        print(f"[Cache] Saved to {cache_path}")

    return z_ctx, z_next
