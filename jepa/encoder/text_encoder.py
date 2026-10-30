"""
jepa/encoder/text_encoder.py
────────────────────────────
Wraps the text encoder (default: Nemotron-Embed-8B in NF4 4-bit quantization).

Swap guide
----------
To use a lighter encoder (e.g. for CPU prototyping), change ONE line in config.yaml:
    encoder.model_name: "BAAI/bge-large-en-v1.5"
    encoder.quantize: false
Everything else stays the same.

The encoder is ALWAYS frozen — we never train it.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import List, Union
import numpy as np


class TextEncoder:
    """
    Frozen text encoder.  Encodes a string (or list of strings) → L2-normalised
    embedding tensor of shape (batch, hidden_dim).

    Args:
        model_name: HuggingFace model id.
        quantize:   If True, load in NF4 4-bit (needed for 8B on 12GB VRAM).
        device:     "cuda" | "cpu" | "auto"
        max_length: Truncation length for tokenizer.
        batch_size: Max texts per forward pass (controls GPU memory).
    """

    def __init__(
        self,
        model_name: str = "nvidia/NV-Embed-v2",
        quantize: bool = True,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"[Encoder] Loading {model_name}  (quantize={quantize}) ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        bnb_config = None
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device if quantize else None,
            torch_dtype=torch.float16 if not quantize else None,
            trust_remote_code=True,
        )

        if not quantize:
            self.model = self.model.to(device)

        self.model.eval()
        # Freeze all parameters — encoder is never updated
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Get hidden dim from model config
        self.hidden_dim = self.model.config.hidden_size
        print(f"[Encoder] Ready  hidden_dim={self.hidden_dim}")

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode texts → L2-normalised embeddings.

        Args:
            texts: A single string or a list of strings.

        Returns:
            Tensor of shape (N, hidden_dim) on CPU, float32.
            L2-normalised so cosine similarity = dot product.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model(**encoded)

            # Mean pool over token dimension
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state          # (B, T, D)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            emb = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            emb = emb.float()                                     # ensure fp32
            emb = F.normalize(emb, p=2, dim=-1)                  # L2 normalise
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)  # (N, D)

    def encode_transition(self, obs: str, action: str) -> torch.Tensor:
        """
        Encode a single (obs, action) pair as one string.
        Format: "Observation: {obs}\nAction: {action}"
        Returns shape (1, hidden_dim).
        """
        text = f"Observation: {obs}\nAction: {action}"
        return self.encode([text])  # (1, D)

    def encode_obs(self, obs: str) -> torch.Tensor:
        """
        Encode a single observation string.
        Returns shape (1, hidden_dim).
        """
        return self.encode([obs])  # (1, D)
