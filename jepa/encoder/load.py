"""
jepa/encoder/load.py
────────────────────
One place to build the encoder, so every script picks it the same way.

    encoder = load_encoder(real=True, device="cuda")    # NV-Embed-v2, NF4
    encoder = load_encoder(real=False)                  # FakeEncoder, CPU, instant
"""

from typing import Optional


def load_encoder(
    real: bool,
    device: str = "cuda",
    model_name: Optional[str] = None,
    quantize: bool = True,
    batch_size: int = 32,
    max_length: int = 512,
):
    if not real:
        from jepa.encoder.fake_encoder import FakeEncoder
        print("[Encoder] Using FakeEncoder (random vectors — for code testing only)")
        return FakeEncoder()

    from jepa.encoder.text_encoder import TextEncoder  # needs transformers
    return TextEncoder(
        model_name=model_name or "nvidia/NV-Embed-v2",
        quantize=quantize,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
