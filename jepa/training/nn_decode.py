"""
jepa/training/nn_decode.py
───────────────────────────
NN-decode: the bridge between JEPA embeddings and text.

The problem
───────────
JEPA's predictor outputs ẑ_{t+1} — a predicted EMBEDDING, not text.
The downstream LLM scorer (decision utility) needs actual text.

The solution (probe-first / learned retrieval framing)
──────────────────────────────────────────────────────
Given ẑ_{t+1}, find the NEAREST transition in the buffer by cosine similarity
and return its obs_next text.  JEPA is bounded to states it has seen —
this is "learned retrieval" vs RAG's "naive retrieval".

At inference (decision utility scoring):
    For each candidate action a_k:
        1.  z_t = encoder(obs_t, a_k)
        2.  ẑ_{t+1} = predictor(online_proj(z_t))
        3.  obs_next_text, decode_dist = nn_decode(ẑ_{t+1}, buffer)
        4.  Pass obs_next_text to LLM scorer → value(a_k)
    Pick action with highest value.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class NNDecoder:
    """
    Nearest-neighbour decoder.  Pre-indexes all buffer next-obs embeddings,
    then at query time finds the nearest one.

    Args:
        buffer_transitions: The training buffer (list of transition dicts).
        z_next_cache:       Pre-computed embeddings of obs_next for the buffer,
                            shape (N, D).  Pass these from the training cache
                            to avoid re-encoding.
    """

    def __init__(
        self,
        buffer_transitions: List[Dict],
        z_next_cache: torch.Tensor,
    ):
        self.transitions   = buffer_transitions
        # Store normalised next-obs embeddings for fast cosine search
        self.z_next_index  = F.normalize(z_next_cache.float(), p=2, dim=-1)  # (N, D)
        print(f"[NNDecoder] Indexed {len(buffer_transitions)} buffer transitions.")

    def decode(
        self,
        z_pred: torch.Tensor,
        top_k: int = 1,
    ) -> Tuple[str, float]:
        """
        Find the nearest buffer transition by cosine similarity to z_pred.

        Args:
            z_pred: Predicted embedding, shape (1, D) or (D,).
            top_k:  Return the top-k nearest match (default 1).

        Returns:
            (obs_next_text, cosine_distance)
            cosine_distance ∈ [0, 2].  Lower = prediction was more confident.
        """
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)  # (1, D)

        z_pred_norm = F.normalize(z_pred.float().cpu(), p=2, dim=-1)  # (1, D)

        # Cosine similarity = dot product of L2-normalised vectors
        sims = (self.z_next_index @ z_pred_norm.T).squeeze(-1)  # (N,)

        best_idx  = sims.argmax().item()
        best_sim  = sims[best_idx].item()
        best_dist = 1.0 - best_sim  # cosine distance

        return self.transitions[best_idx]["obs_next"], best_dist

    def decode_batch(
        self,
        z_preds: torch.Tensor,
    ) -> Tuple[List[str], List[float]]:
        """
        Decode a batch of predicted embeddings.

        Args:
            z_preds: Shape (B, D).

        Returns:
            (list of obs_next texts, list of cosine distances)
        """
        z_preds_norm = F.normalize(z_preds.float().cpu(), p=2, dim=-1)  # (B, D)
        sims = self.z_next_index @ z_preds_norm.T  # (N, B)

        best_indices = sims.argmax(dim=0).tolist()  # (B,)
        texts = [self.transitions[i]["obs_next"] for i in best_indices]
        dists = [
            1.0 - sims[best_indices[b], b].item()
            for b in range(len(best_indices))
        ]

        return texts, dists

    def score_candidate_actions(
        self,
        obs_t: str,
        candidate_actions: List[str],
        predictor,
        ema,
        encoder,
        device: str = "cuda",
    ) -> List[Tuple[str, str, float]]:
        """
        For decision utility: given obs_t and K candidate actions,
        predict the next observation for each action using JEPA + NN-decode.

        Returns:
            List of (action, predicted_obs_next, decode_dist) sorted by decode_dist ASC
            (lower decode_dist = predictor was more confident about this transition).

        The predicted_obs_next strings are passed to the LLM scorer for value estimation.
        """
        results = []

        for action in candidate_actions:
            # Encode (obs_t, action) pair
            context_text = f"Observation: {obs_t}\nAction: {action}"
            z_t = encoder.encode([context_text]).to(device)  # (1, D)

            # Predict next embedding
            with torch.no_grad():
                z_online = ema.online(z_t)
                z_pred   = predictor(z_online)   # (1, D)

            # NN-decode to text
            obs_next_text, decode_dist = self.decode(z_pred)
            results.append((action, obs_next_text, decode_dist))

        return results
