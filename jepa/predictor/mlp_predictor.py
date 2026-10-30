"""
jepa/predictor/mlp_predictor.py
────────────────────────────────
The JEPA predictor: a small trainable MLP that takes
    z_t  = encoder(obs_t || action_t)          (context embedding)
and outputs
    ẑ_{t+1}                                    (predicted next-obs embedding)

The TARGET is provided by an EMA copy of the encoder applied to obs_{t+1}.
We never backprop through the target — that's the stop-gradient.

Swap guide
----------
To use a transformer predictor instead of MLP, swap this import in training/:
    from jepa.predictor.mlp_predictor    import MLPPredictor
    from jepa.predictor.transformer_predictor import TransformerPredictor
Both have the same __init__ signature and forward() contract.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPPredictor(nn.Module):
    """
    2-layer MLP predictor.

    Architecture:
        Linear(input_dim → hidden_dim) → LayerNorm → GELU
        Linear(hidden_dim → hidden_dim) → LayerNorm → GELU   [if num_layers >= 2]
        Linear(hidden_dim → input_dim)
        L2-normalise output

    Args:
        input_dim:  Dimensionality of encoder embeddings (e.g. 4096 for 8B).
        hidden_dim: MLP hidden size (default 2048 — small, fast).
        num_layers: Number of hidden layers before the output projection.
        dropout:    Dropout applied after each activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim

        # Output projection back to embedding space
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Context embedding, shape (B, input_dim).
                 This is encoder(obs_t || action_t).

        Returns:
            ẑ_{t+1}: Predicted next-obs embedding, shape (B, input_dim).
                     L2-normalised so cosine similarity = dot product.
        """
        pred = self.net(z_t)
        return F.normalize(pred, p=2, dim=-1)


class EMATargetEncoder(nn.Module):
    """
    Exponential Moving Average copy of the encoder's projection layer.

    In JEPA (I-JEPA / V-JEPA style), the target embedding is produced by an EMA
    copy of the encoder — this prevents representation collapse without needing
    contrastive negatives.

    Since our encoder (Nemotron-Embed-8B) is frozen and huge, we don't EMA the
    whole encoder.  Instead we keep a *separate* learned linear projection layer
    and EMA that.  The target for training is:

        target_{t+1} = EMA_proj(encoder_emb(obs_{t+1}))

    The online version (used in the predictor input) is just the raw encoder emb.

    Args:
        embed_dim:  Encoder hidden dim.
        decay:      EMA decay rate (V-JEPA default: 0.996).
    """

    def __init__(self, embed_dim: int, decay: float = 0.996):
        super().__init__()
        self.decay = decay

        # Online projection (gets gradients)
        self.online_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Target projection (EMA copy — no gradients ever)
        self.target_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.target_proj.requires_grad_(False)

        # Initialise target = online
        self._copy_online_to_target()

    def _copy_online_to_target(self):
        for param_o, param_t in zip(
            self.online_proj.parameters(), self.target_proj.parameters()
        ):
            param_t.data.copy_(param_o.data)

    @torch.no_grad()
    def update_ema(self):
        """Call once per training step after the optimizer step."""
        for param_o, param_t in zip(
            self.online_proj.parameters(), self.target_proj.parameters()
        ):
            param_t.data.mul_(self.decay).add_(param_o.data, alpha=1.0 - self.decay)

    def online(self, z: torch.Tensor) -> torch.Tensor:
        """Project with the online head (used for predictor INPUT)."""
        return F.normalize(self.online_proj(z), p=2, dim=-1)

    @torch.no_grad()
    def target(self, z: torch.Tensor) -> torch.Tensor:
        """Project with the target head (used as TRAINING TARGET — stop gradient)."""
        return F.normalize(self.target_proj(z), p=2, dim=-1)
