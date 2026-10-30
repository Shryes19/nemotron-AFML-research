"""
jepa/training/train.py
───────────────────────
JEPA training loop.

What it does
────────────
1.  Encodes every transition in the buffer (obs+action → z_t, obs_next → z_next)
    using the frozen encoder.  Embeddings are cached as tensors so we only run
    the encoder once per training run.

2.  Trains the MLPPredictor to minimise MSE between:
        ẑ_{t+1}  =  predictor(online_proj(z_t))
        target   =  EMA_target_proj(z_next)         ← stop-gradient

3.  Updates EMA target after every optimizer step.

4.  Evaluates cosine distance on held-out 10% after each epoch.

Probe mode
──────────
Pass epochs=1 (config: training.probe_epochs) to get a cheap ex-ante
fidelity estimate at n=32 without full training.

Usage
─────
    from jepa.training.train import JEPATrainer

    trainer = JEPATrainer(encoder, predictor, ema, config)
    results = trainer.train(train_dataset, val_dataset)
    probe   = trainer.probe_fidelity(dataset)   # 1-epoch probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import numpy as np

from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
# TextEncoder imported lazily (requires transformers) — only needed at runtime
# from jepa.encoder.text_encoder import TextEncoder


class JEPATrainer:
    """
    Trains the JEPA predictor on a transition buffer.

    Args:
        encoder:    Frozen TextEncoder instance.
        predictor:  MLPPredictor (or TransformerPredictor — same interface).
        ema:        EMATargetEncoder.
        config:     Dict with keys matching configs/config.yaml > jepa + training.
        device:     "cuda" | "cpu"
    """

    def __init__(
        self,
        encoder,  # TextEncoder or FakeEncoder — same interface
        predictor: MLPPredictor,
        ema: EMATargetEncoder,
        config: Dict,
        device: str = "cuda",
    ):
        self.encoder   = encoder
        self.predictor = predictor.to(device)
        self.ema       = ema.to(device)
        self.device    = device
        self.config    = config

        self.optimizer = torch.optim.AdamW(
            list(predictor.parameters()) + list(ema.online_proj.parameters()),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Embedding cache — encode buffer once, cache as tensors
    # ──────────────────────────────────────────────────────────────────────────

    def _build_embedding_cache(
        self, transitions: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all transitions upfront.

        Returns:
            z_context:  (N, D) — encoder(obs || action)
            z_next:     (N, D) — encoder(obs_next)
        """
        print(f"[JEPA] Building embedding cache for {len(transitions)} transitions ...")

        context_texts = [
            f"Observation: {t['obs']}\nAction: {t['action']}"
            for t in transitions
        ]
        next_obs_texts = [t["obs_next"] for t in transitions]

        z_context = self.encoder.encode(context_texts)   # (N, D) on CPU
        z_next    = self.encoder.encode(next_obs_texts)  # (N, D) on CPU

        print(f"[JEPA] Cache ready  shape={z_context.shape}")
        return z_context, z_next

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_transitions: List[Dict],
        val_transitions: List[Dict],
        epochs: Optional[int] = None,
        log_every: int = 5,
    ) -> Dict:
        """
        Full training run.

        Returns a dict with training history:
            {
              "train_mse":     [float per epoch],
              "val_cos_dist":  [float per epoch],   ← primary eval metric
            }
        """
        if epochs is None:
            epochs = self.config.get("epochs", 20)

        z_ctx_train, z_next_train = self._build_embedding_cache(train_transitions)
        z_ctx_val,   z_next_val   = self._build_embedding_cache(val_transitions)

        history = {"train_mse": [], "val_cos_dist": []}

        for epoch in range(1, epochs + 1):
            # ── Train ──
            self.predictor.train()
            self.ema.train()

            # Shuffle
            perm = torch.randperm(len(z_ctx_train))
            z_ctx_train  = z_ctx_train[perm]
            z_next_train = z_next_train[perm]

            batch_size = 64
            epoch_losses = []

            for i in range(0, len(z_ctx_train), batch_size):
                z_ctx_b  = z_ctx_train[i:i+batch_size].to(self.device)
                z_next_b = z_next_train[i:i+batch_size].to(self.device)

                # Online projection of context
                z_online = self.ema.online(z_ctx_b)   # (B, D)

                # Predict next embedding
                z_pred = self.predictor(z_online)      # (B, D)

                # EMA target — stop gradient (no_grad handled inside EMATargetEncoder)
                z_target = self.ema.target(z_next_b)   # (B, D)

                loss = F.mse_loss(z_pred, z_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update_ema()

                epoch_losses.append(loss.item())

            train_mse = float(np.mean(epoch_losses))

            # ── Validate ──
            val_cos_dist = self._eval_cosine_distance(z_ctx_val, z_next_val)

            history["train_mse"].append(train_mse)
            history["val_cos_dist"].append(val_cos_dist)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"[JEPA] Epoch {epoch:3d}/{epochs}  "
                    f"train_mse={train_mse:.4f}  "
                    f"val_cos_dist={val_cos_dist:.4f}"
                )

        print(f"[JEPA] Training done.  Final val_cos_dist={history['val_cos_dist'][-1]:.4f}")
        return history

    def probe_fidelity(
        self,
        transitions: List[Dict],
        probe_epochs: int = 1,
    ) -> float:
        """
        Ex-ante Feature 1: Train for probe_epochs (default 1) on the full buffer,
        then measure cosine distance on the held-out 10%.

        This is CHEAP — run at n=32 before committing to full training.

        Returns:
            mean cosine distance on held-out set (lower = better fit).
        """
        from jepa.training.buffer import TransitionDataset

        dataset = TransitionDataset(transitions)
        train_ds, val_ds = dataset.train_val_split(held_out_frac=0.1)

        print(f"[JEPA Probe] {probe_epochs}-epoch probe on {len(train_ds)} transitions ...")
        history = self.train(
            train_ds.transitions,
            val_ds.transitions,
            epochs=probe_epochs,
            log_every=1,
        )
        probe_score = history["val_cos_dist"][-1]
        print(f"[JEPA Probe] probe_fidelity = {probe_score:.4f}  (lower = better)")
        return probe_score

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_cosine_distance(
        self,
        z_ctx: torch.Tensor,
        z_next: torch.Tensor,
        batch_size: int = 128,
    ) -> float:
        """
        Compute mean cosine distance between predicted and actual next embeddings.
        Cosine distance = 1 − cosine_similarity.  Range [0, 2].  Lower is better.
        """
        self.predictor.eval()
        self.ema.eval()

        all_dists = []
        for i in range(0, len(z_ctx), batch_size):
            z_ctx_b  = z_ctx[i:i+batch_size].to(self.device)
            z_next_b = z_next[i:i+batch_size].to(self.device)

            z_online = self.ema.online(z_ctx_b)
            z_pred   = self.predictor(z_online)
            z_target = self.ema.target(z_next_b)

            cos_sim  = F.cosine_similarity(z_pred, z_target, dim=-1)  # (B,)
            cos_dist = 1.0 - cos_sim
            all_dists.extend(cos_dist.cpu().tolist())

        return float(np.mean(all_dists))

    def save(self, path: str):
        """Save predictor + EMA weights."""
        torch.save({
            "predictor": self.predictor.state_dict(),
            "ema_online": self.ema.online_proj.state_dict(),
            "ema_target": self.ema.target_proj.state_dict(),
        }, path)
        print(f"[JEPA] Saved to {path}")

    def load(self, path: str):
        """Load predictor + EMA weights."""
        ckpt = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(ckpt["predictor"])
        self.ema.online_proj.load_state_dict(ckpt["ema_online"])
        self.ema.target_proj.load_state_dict(ckpt["ema_target"])
        print(f"[JEPA] Loaded from {path}")
