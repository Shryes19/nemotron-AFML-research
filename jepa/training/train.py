"""
jepa/training/train.py
───────────────────────
JEPA training loop.

What it does
────────────
1.  Encodes every transition (obs+action → z_t, obs_next → z_next) with the
    frozen encoder — or takes vectors you already encoded (see
    train_on_embeddings / embed_cache.py), so the 8B encoder runs only once.

2.  Trains the predictor to minimise MSE between:
        ẑ_{t+1}  =  predictor(online_proj(z_t))
        target   =  EMA_target_proj(z_next)         ← stop-gradient

3.  Updates the EMA target after every optimizer step.

4.  After each epoch, measures on the held-out set:
      val_cos_dist   how far predictions are from the truth (0 best, 2 worst)
      val_top1       for each held-out example, is the prediction closer to its
                     OWN true next state than to every other held-out next
                     state?  Fraction of yes. Random guessing = 1 / n_val.
      target_spread  how spread out the target vectors are (0 = all identical).

Why val_top1 and target_spread matter
─────────────────────────────────────
JEPA can "collapse": the projection learns to map every state to the same
point, so prediction becomes trivial and val_cos_dist drops even though the
model has learned nothing. If target_spread heads toward 0 and val_top1 stays
near random, that is collapse — not progress.

Probe mode
──────────
epochs=1 (config: training.probe_epochs) gives the cheap ex-ante fidelity
estimate at n=32 (Feature 1).
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
from jepa.training.embed_cache import context_text


class JEPATrainer:
    """
    Args:
        encoder:    Frozen encoder with .encode(texts) (TextEncoder or FakeEncoder).
                    May be None if you only call the *_on_embeddings methods.
        predictor:  MLPPredictor (or any module with the same forward contract).
        ema:        EMATargetEncoder.
        config:     Dict with optional keys: lr, weight_decay, epochs, batch_size.
        device:     "cuda" | "cpu"
    """

    def __init__(
        self,
        encoder,
        predictor: MLPPredictor,
        ema: EMATargetEncoder,
        config: Dict,
        device: str = "cuda",
    ):
        self.encoder = encoder
        self.predictor = predictor.to(device)
        self.ema = ema.to(device)
        self.device = device
        self.config = config
        self.batch_size = int(config.get("batch_size", 64))

        self.optimizer = torch.optim.AdamW(
            list(predictor.parameters()) + list(ema.online_proj.parameters()),
            lr=float(config.get("lr", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────────────

    def _build_embedding_cache(self, transitions: List[Dict]):
        """Encode transitions → (z_ctx, z_next), both (N, D) on CPU."""
        print(f"[JEPA] Building embedding cache for {len(transitions)} transitions ...")
        z_ctx = self.encoder.encode([context_text(t) for t in transitions])
        z_next = self.encoder.encode([t["obs_next"] for t in transitions])
        print(f"[JEPA] Cache ready  shape={tuple(z_ctx.shape)}")
        return z_ctx, z_next

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
        """Encode the transitions, then train. See train_on_embeddings."""
        z_ctx_tr, z_next_tr = self._build_embedding_cache(train_transitions)
        z_ctx_va, z_next_va = self._build_embedding_cache(val_transitions)
        return self.train_on_embeddings(
            z_ctx_tr, z_next_tr, z_ctx_va, z_next_va, epochs=epochs, log_every=log_every
        )

    def train_on_embeddings(
        self,
        z_ctx_train: torch.Tensor,
        z_next_train: torch.Tensor,
        z_ctx_val: torch.Tensor,
        z_next_val: torch.Tensor,
        epochs: Optional[int] = None,
        log_every: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        Train from pre-computed vectors.

        Returns history:
            {"train_mse": [...], "val_cos_dist": [...],
             "val_top1": [...], "target_spread": [...]}   one value per epoch
        """
        if epochs is None:
            epochs = int(self.config.get("epochs", 20))

        history = {"train_mse": [], "val_cos_dist": [], "val_top1": [], "target_spread": []}
        n = len(z_ctx_train)

        for epoch in range(1, epochs + 1):
            self.predictor.train()
            self.ema.train()

            perm = torch.randperm(n)
            losses = []
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                z_ctx_b = z_ctx_train[idx].to(self.device)
                z_next_b = z_next_train[idx].to(self.device)

                z_pred = self.predictor(self.ema.online(z_ctx_b))
                z_target = self.ema.target(z_next_b)        # stop-gradient
                loss = F.mse_loss(z_pred, z_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update_ema()
                losses.append(loss.item())

            metrics = self.evaluate(z_ctx_val, z_next_val)
            history["train_mse"].append(float(np.mean(losses)))
            for k in ("val_cos_dist", "val_top1", "target_spread"):
                history[k].append(metrics[k])

            if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"[JEPA] Epoch {epoch:3d}/{epochs}  "
                    f"train_mse={history['train_mse'][-1]:.5f}  "
                    f"val_cos_dist={metrics['val_cos_dist']:.4f}  "
                    f"val_top1={metrics['val_top1']:.3f}  "
                    f"spread={metrics['target_spread']:.4f}"
                )

        if verbose:
            print(f"[JEPA] Training done.  Final val_cos_dist={history['val_cos_dist'][-1]:.4f}")
        return history

    # ──────────────────────────────────────────────────────────────────────────
    # Probe (ex-ante Feature 1)
    # ──────────────────────────────────────────────────────────────────────────

    def probe_fidelity(self, transitions: List[Dict], probe_epochs: int = 1) -> float:
        """Encode, split 90/10, train probe_epochs, return held-out cos distance."""
        from jepa.training.buffer import TransitionDataset

        train_ds, val_ds = TransitionDataset(transitions).train_val_split(held_out_frac=0.1)
        print(f"[JEPA Probe] {probe_epochs}-epoch probe on {len(train_ds)} transitions ...")
        history = self.train(
            train_ds.transitions, val_ds.transitions, epochs=probe_epochs, log_every=1
        )
        score = history["val_cos_dist"][-1]
        print(f"[JEPA Probe] probe_fidelity = {score:.4f}  (lower = better)")
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, z_ctx: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """Predicted next-state vectors (in the target-projected space), on CPU."""
        self.predictor.eval()
        self.ema.eval()
        out = []
        for i in range(0, len(z_ctx), batch_size):
            z = z_ctx[i : i + batch_size].to(self.device)
            out.append(self.predictor(self.ema.online(z)).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def project_targets(self, z_next: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """Raw next-state vectors → target-projected space (what NN-decode indexes)."""
        self.ema.eval()
        out = []
        for i in range(0, len(z_next), batch_size):
            out.append(self.ema.target(z_next[i : i + batch_size].to(self.device)).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def evaluate(self, z_ctx: torch.Tensor, z_next: torch.Tensor) -> Dict[str, float]:
        pred = self.predict(z_ctx)
        target = self.project_targets(z_next)

        cos_dist = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean().item()

        # Top-1: is each prediction closest to its own target among all targets?
        sims = pred @ target.T                       # both L2-normalised
        top1 = (sims.argmax(dim=1) == torch.arange(len(pred))).float().mean().item()

        # Spread: 1 - average cosine similarity between different targets.
        n = len(target)
        if n > 1:
            tt = target @ target.T
            mean_off_diag = (tt.sum() - tt.diagonal().sum()) / (n * (n - 1))
            spread = 1.0 - mean_off_diag.item()
        else:
            spread = float("nan")

        return {"val_cos_dist": cos_dist, "val_top1": top1, "target_spread": spread}

    def _eval_cosine_distance(self, z_ctx, z_next, batch_size: int = 128) -> float:
        """Kept for backward compatibility."""
        return self.evaluate(z_ctx, z_next)["val_cos_dist"]

    # ──────────────────────────────────────────────────────────────────────────
    # Save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str, extra: Optional[Dict] = None):
        ckpt = {
            "predictor": self.predictor.state_dict(),
            "ema_online": self.ema.online_proj.state_dict(),
            "ema_target": self.ema.target_proj.state_dict(),
        }
        if extra:
            ckpt["extra"] = extra
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(ckpt["predictor"])
        self.ema.online_proj.load_state_dict(ckpt["ema_online"])
        self.ema.target_proj.load_state_dict(ckpt["ema_target"])
        print(f"[JEPA] Loaded from {path}")
