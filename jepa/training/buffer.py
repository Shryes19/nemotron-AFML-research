"""
jepa/training/buffer.py
────────────────────────
Loads a transition buffer from a JSONL file and returns a TransitionDataset.

Each line in the JSONL file must have:
    {
      "obs":      "You are in the kitchen...",
      "action":   "pick up the apple",
      "obs_next": "You pick up the apple. You are in the kitchen...",
      "env":      "alfworld",          # optional metadata
      "game":     "game_001",          # optional — unit for RQ3 data points
      "cap_tag":  "object_state"       # optional — filled in by capability tagger
    }
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class TransitionDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of (obs, action, obs_next) transitions.
    Pre-encodes nothing — raw strings are returned; the training loop encodes
    them in batches so we don't need to cache all embeddings upfront.
    """

    def __init__(self, transitions: List[Dict]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict:
        return self.transitions[idx]

    @classmethod
    def from_jsonl(cls, path: str, n: Optional[int] = None, seed: int = 42) -> "TransitionDataset":
        """
        Load from a JSONL file.

        Args:
            path: Path to .jsonl file.
            n:    If set, sub-sample to exactly n transitions (with fixed seed).
        """
        transitions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    transitions.append(json.loads(line))

        if n is not None and n < len(transitions):
            rng = random.Random(seed)
            transitions = rng.sample(transitions, n)

        print(f"[Buffer] Loaded {len(transitions)} transitions from {path}")
        return cls(transitions)

    def train_val_split(
        self, held_out_frac: float = 0.1, seed: int = 42
    ) -> Tuple["TransitionDataset", "TransitionDataset"]:
        """
        Split into train and held-out validation sets.

        Args:
            held_out_frac: Fraction of data to hold out (default 0.1 = 10%).

        Returns:
            (train_dataset, val_dataset)
        """
        rng = random.Random(seed)
        indices = list(range(len(self)))
        rng.shuffle(indices)

        n_val = max(1, int(len(self) * held_out_frac))
        val_indices   = indices[:n_val]
        train_indices = indices[n_val:]

        train_data = [self.transitions[i] for i in train_indices]
        val_data   = [self.transitions[i] for i in val_indices]

        return TransitionDataset(train_data), TransitionDataset(val_data)
