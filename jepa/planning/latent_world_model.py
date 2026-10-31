"""
jepa/planning/latent_world_model.py
───────────────────────────────────
Use a trained JEPA to choose between actions, WITHOUT turning its predictions
back into text.

Why this exists
───────────────
NN-decode can only answer with a state already in the buffer, which caps JEPA
at the buffer's "reach". The diagnostics showed the predictor itself knows
more: on predictable actions in unseen games it found the right next state
55% of the time when allowed to, vs a 45% buffer ceiling. Working directly in
embedding space removes that ceiling, and it is the JEPA-native way to plan
(predict consequences in representation space, compare them there).

What it offers
──────────────
    wm = LatentWorldModel.from_checkpoint("outputs/jepa/.../budget_1024.pt", encoder,
                                          buffer_items=train_items)
    z = wm.predict(obs, ["go to fridge 1", "open fridge 1", "take apple 1 from fridge 1"])
                                    # (K, D) predicted next-state vectors, no text
    wm.rank_by_target(z, "You pick up the apple 1 from the fridge 1.")
                                    # actions ranked by how close their predicted
                                    # consequence is to a desired outcome
    wm.decode_topk(z, k=5)          # optional: the 5 nearest buffer states per
                                    # action, for an LLM scorer that needs text

Inputs are built EXACTLY as in training (same text formats, same input mode,
same admissible-action features), read from the checkpoint's saved settings.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
from jepa.training.embed_cache import make_context, obs_text, action_text
from jepa.training.features import feature_matrix


class LatentWorldModel:
    def __init__(
        self,
        encoder,
        predictor: MLPPredictor,
        ema: EMATargetEncoder,
        input_mode: str = "separate",
        admissible_feature: bool = True,
        buffer_items: Optional[List[Dict]] = None,
        device: str = "cuda",
        target_mode: str = "ema",
    ):
        self.encoder = encoder
        self.target_mode = target_mode
        self.predictor = predictor.to(device).eval()
        self.ema = ema.to(device).eval()
        self.input_mode = input_mode
        self.admissible_feature = admissible_feature
        self.device = device
        self._index_txt: List[str] = []
        self._index_vec: Optional[torch.Tensor] = None
        if buffer_items:
            self.set_buffer(buffer_items)

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str, encoder, buffer_items=None, device: str = "cuda"):
        """
        Load a budget_<n>.pt written by scripts/train_jepa.py. Settings saved in
        the checkpoint (input mode, admissible feature) are used when present.
        """
        ckpt = torch.load(path, map_location=device)
        extra = ckpt.get("extra", {})
        sd = ckpt["predictor"]
        linears = [v for k, v in sd.items() if k.endswith("weight") and v.dim() == 2]
        first, last = linears[0], linears[-1]
        in_dim, out_dim = first.shape[1], last.shape[0]
        predictor = MLPPredictor(in_dim, hidden_dim=first.shape[0], num_layers=len(linears) - 1,
                                 output_dim=out_dim)
        predictor.load_state_dict(sd)
        ema = EMATargetEncoder(out_dim)
        ema.online_proj.load_state_dict(ckpt["ema_online"])
        ema.target_proj.load_state_dict(ckpt["ema_target"])
        mode = extra.get("input_mode") or ("separate" if in_dim >= 2 * out_dim else "joint")
        adm = extra.get("admissible_feature")
        if adm is None:
            adm = in_dim % out_dim != 0
        return cls(encoder, predictor, ema, input_mode=mode, admissible_feature=adm,
                   buffer_items=buffer_items, device=device,
                   target_mode=ckpt.get("target_mode", extra.get("target_mode", "ema")))

    def set_buffer(self, items: List[Dict]):
        """Index the buffer's next states (target-projected) for decode_topk."""
        seen, txt = set(), []
        for t in items:
            if t["obs_next"] not in seen:
                seen.add(t["obs_next"])
                txt.append(t["obs_next"])
        raw = self.encoder.encode(txt)
        self._index_txt = txt
        self._index_vec = self._project_targets(raw)

    # ── core ──────────────────────────────────────────────────────────────────

    def _inputs(self, obs: str, actions: Sequence[str]) -> torch.Tensor:
        if self.input_mode == "separate":
            z_obs = self.encoder.encode([obs_text(obs)])                 # (1, D)
            z_act = self.encoder.encode([action_text(a) for a in actions])  # (K, D)
            z = torch.cat([z_obs.expand(len(actions), -1), z_act], dim=1)
        else:
            z = self.encoder.encode([make_context(obs, a) for a in actions])
        if self.admissible_feature:
            feats = feature_matrix([{"obs": obs, "action": a} for a in actions])
            z = torch.cat([z, feats.to(z.dtype)], dim=1)
        return z

    def _online(self, z: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "frozen":
            return z
        d = self.ema.online_proj.in_features
        k = z.shape[1] // d
        parts = [self.ema.online(z[:, i * d:(i + 1) * d]) for i in range(k)]
        if z.shape[1] > k * d:
            parts.append(z[:, k * d:])
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    @torch.no_grad()
    def _project_targets(self, raw: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "frozen":
            return F.normalize(raw.float(), p=2, dim=-1)
        return self.ema.target(raw.to(self.device)).cpu()

    @torch.no_grad()
    def predict(self, obs: str, actions: Sequence[str]) -> torch.Tensor:
        """Predicted next-state vector for each candidate action, (K, D), on CPU."""
        z = self._inputs(obs, actions).to(self.device)
        return self.predictor(self._online(z)).cpu()

    @torch.no_grad()
    def embed_state(self, text: str) -> torch.Tensor:
        """Any state text → the same space as predict(), (D,)."""
        return self._project_targets(self.encoder.encode([text]))[0]

    # ── uses ──────────────────────────────────────────────────────────────────

    def rank_by_target(self, z_pred: torch.Tensor, target: str,
                       actions: Optional[Sequence[str]] = None) -> List[Tuple[int, float]]:
        """
        Rank candidate actions by how close their predicted consequence is to a
        target state description (cosine, higher = closer). No decoding.
        Returns [(candidate index, similarity)], best first.
        """
        t = self.embed_state(target)
        sims = (z_pred @ t).tolist()
        order = sorted(range(len(sims)), key=lambda i: -sims[i])
        return [(i, sims[i]) for i in order]

    def decode_topk(self, z_pred: torch.Tensor, k: int = 5) -> List[List[Tuple[str, float]]]:
        """The k nearest buffer states for each prediction: [[(text, similarity), ...], ...]."""
        if self._index_vec is None:
            raise RuntimeError("no buffer indexed — pass buffer_items or call set_buffer()")
        sims = z_pred @ self._index_vec.T
        top = sims.topk(min(k, sims.shape[1]), dim=1)
        return [[(self._index_txt[j], float(s)) for s, j in zip(row_s.tolist(), row_i.tolist())]
                for row_s, row_i in zip(top.values, top.indices)]
