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

Implementation note
-------------------
NV-Embed-v2 is loaded via AutoModel with a monkey-patch for a transformers>=4.47
compatibility issue (all_tied_weights_keys attribute missing on NVEmbedModel).
If that ever breaks, swap to the sentence-transformers path by setting
encoder.use_sentence_transformers: true in config.yaml.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import List, Union
import numpy as np


def _patch_nv_embed_model():
    """
    Monkey-patch NVEmbedModel to add the `all_tied_weights_keys` property
    that newer transformers versions expect but the custom model class omits.
    This is a one-line compatibility shim — safe to apply unconditionally.
    """
    try:
        from transformers.dynamic_module_utils import get_cached_module_file
        import importlib, sys

        # Only patch if the model class is already cached / imported
        for mod_name, mod in list(sys.modules.items()):
            if "nvembed" in mod_name.lower() or "nv_embed" in mod_name.lower():
                for cls_name in dir(mod):
                    cls = getattr(mod, cls_name, None)
                    if cls and isinstance(cls, type) and not hasattr(cls, "all_tied_weights_keys"):
                        cls.all_tied_weights_keys = property(
                            lambda self: getattr(self, "_tied_weights_keys", []) or []
                        )
    except Exception:
        pass  # Patch is best-effort; failure is silent


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

        # Pre-download / cache the model files so the custom class is imported
        # before we apply the compatibility patch
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Apply monkey-patch for newer transformers compatibility
        _patch_nv_embed_model()

        # Patch at the modeling_utils level: transformers>=4.47 calls
        # model.all_tied_weights_keys.keys() expecting a dict; NVEmbedModel
        # either has no attribute or returns a list.  Wrap it in a dict.
        try:
            import transformers.modeling_utils as _mu

            _orig_get_total = _mu.get_total_byte_count

            def _patched_get_total(model, *args, **kwargs):
                if hasattr(model, "all_tied_weights_keys") and isinstance(
                    model.all_tied_weights_keys, list
                ):
                    # Wrap list as dict so .keys() works
                    model.all_tied_weights_keys = {k: k for k in model.all_tied_weights_keys}
                elif not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = {}
                return _orig_get_total(model, *args, **kwargs)

            _mu.get_total_byte_count = _patched_get_total
        except Exception:
            pass

        # Also patch at the quantizer level as a belt-and-suspenders fix
        try:
            from transformers.quantizers import base as _qbase
            _orig = _qbase.get_keys_to_not_convert

            def _patched(model):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = {}
                elif isinstance(model.all_tied_weights_keys, list):
                    model.all_tied_weights_keys = {k: k for k in model.all_tied_weights_keys}
                return _orig(model)

            _qbase.get_keys_to_not_convert = _patched
        except Exception:
            pass

        # Only pass the args we actually need. `torch_dtype` works on both old
        # (4.44) and new transformers; passing dtype=None to 4.44 leaks into the
        # model constructor and breaks custom model classes.
        load_kwargs = {"trust_remote_code": True}
        if quantize:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = device
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)

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

            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

            if isinstance(outputs, dict) and "sentence_embeddings" in outputs:
                # NV-Embed-v2: model already pools (latent-attention pooling)
                emb = outputs["sentence_embeddings"]               # (B, D)
            else:
                # Generic encoders (e.g. BGE): mean pool over tokens
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state       # (B, T, D)
                mask_expanded = attention_mask.unsqueeze(-1).float()
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
