"""
jepa/encoder/fake_encoder.py
────────────────────────────
A stand-in for TextEncoder that needs no GPU and no model download.

Each text gets a random-but-repeatable unit vector (seeded from a hash of the
text), so the same text always maps to the same embedding — even across runs.
There is NO meaning in these vectors, so a model trained on them cannot really
learn anything. Use it only to check that code runs end to end.
"""

import hashlib
from typing import List, Union

import torch
import torch.nn.functional as F


class FakeEncoder:
    """Same interface as TextEncoder: .encode(texts) -> (N, hidden_dim) on CPU."""

    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        self.model_name = f"fake-{hidden_dim}"

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        embs = []
        for t in texts:
            # hashlib (not hash()) so results are identical across Python runs
            seed = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            embs.append(F.normalize(torch.randn(self.hidden_dim, generator=gen), dim=-1))
        return torch.stack(embs)

    def encode_transition(self, obs: str, action: str) -> torch.Tensor:
        return self.encode([f"Observation: {obs}\nAction: {action}"])

    def encode_obs(self, obs: str) -> torch.Tensor:
        return self.encode([obs])
