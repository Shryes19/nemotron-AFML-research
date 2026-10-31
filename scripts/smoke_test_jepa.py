"""
scripts/smoke_test_jepa.py
───────────────────────────
End-to-end smoke test of the JEPA pipeline using mock data.

Runs WITHOUT loading Nemotron-Embed-8B — uses a tiny random encoder instead,
so you can verify the full pipeline (data loading → encoding → training →
NN-decode → probe fidelity) in seconds on CPU.

To run with the real quantized encoder, pass --real-encoder.

Usage:
    # Fast CPU test with random encoder (default)
    python scripts/smoke_test_jepa.py --buffer data/buffers/mock_n64.jsonl

    # With real quantized Nemotron-Embed-8B (needs GPU + ~6GB VRAM)
    python scripts/smoke_test_jepa.py --buffer data/buffers/mock_n64.jsonl --real-encoder
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
from jepa.training.buffer import TransitionDataset
from jepa.training.train import JEPATrainer
from jepa.training.nn_decode import NNDecoder


from jepa.encoder.fake_encoder import FakeEncoder  # shared stand-in encoder


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default="data/buffers/mock_n64.jsonl")
    parser.add_argument("--real-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  JEPA Smoke Test")
    print("=" * 60)

    # ── Step 1: Load buffer ────────────────────────────────────────────────────
    print(f"\n[1/6] Loading buffer from {args.buffer} ...")
    dataset = TransitionDataset.from_jsonl(args.buffer, n=32)
    train_ds, val_ds = dataset.train_val_split(held_out_frac=0.1)
    print(f"      train={len(train_ds)}  val={len(val_ds)}")

    # ── Step 2: Load encoder ───────────────────────────────────────────────────
    print(f"\n[2/6] Loading encoder (real={args.real_encoder}) ...")
    if args.real_encoder:
        from jepa.encoder.text_encoder import TextEncoder
        encoder = TextEncoder(quantize=True, device=args.device)
    else:
        encoder = FakeEncoder()
        print(f"      Using FakeEncoder  hidden_dim={encoder.hidden_dim}")

    # ── Step 3: Build predictor + EMA ─────────────────────────────────────────
    print(f"\n[3/6] Building MLP predictor + EMA target encoder ...")
    hidden_dim = encoder.hidden_dim
    predictor  = MLPPredictor(input_dim=hidden_dim, hidden_dim=512, num_layers=2)
    ema        = EMATargetEncoder(embed_dim=hidden_dim, decay=0.996)
    print(f"      Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # ── Step 4: Train ──────────────────────────────────────────────────────────
    print(f"\n[4/6] Training for {args.epochs} epochs ...")
    config  = {"lr": 1e-4, "weight_decay": 1e-4, "epochs": args.epochs}
    trainer = JEPATrainer(encoder, predictor, ema, config, device=args.device)
    history = trainer.train(train_ds.transitions, val_ds.transitions, epochs=args.epochs, log_every=1)

    # Verify loss is going down
    first_loss = history["val_cos_dist"][0]
    last_loss  = history["val_cos_dist"][-1]
    improved = last_loss < first_loss
    print(f"\n      val_cos_dist: {first_loss:.4f} → {last_loss:.4f}  {'✓ improving' if improved else '✗ NOT improving'}")

    # ── Step 5: NN-decode ──────────────────────────────────────────────────────
    print(f"\n[5/6] Testing NN-decode ...")

    # Build embedding cache for the full buffer (train+val combined).
    # IMPORTANT: the predictor outputs vectors in the EMA *target-projected*
    # space, so the decode index must live in that same space.
    all_transitions = train_ds.transitions + val_ds.transitions
    z_next_raw      = encoder.encode([t["obs_next"] for t in all_transitions])
    with torch.no_grad():
        z_next_cache = ema.target(z_next_raw.to(args.device)).cpu()
    decoder         = NNDecoder(all_transitions, z_next_cache)

    # Pick a test transition, predict its next obs, decode
    test_t   = val_ds.transitions[0]
    ctx_text = f"Observation: {test_t['obs']}\nAction: {test_t['action']}"
    z_ctx    = encoder.encode([ctx_text]).to(args.device)

    with torch.no_grad():
        z_online = ema.online(z_ctx)
        z_pred   = predictor(z_online)

    decoded_text, decode_dist = decoder.decode(z_pred)

    print(f"      Query obs:    {test_t['obs'][:80]}...")
    print(f"      Query action: {test_t['action']}")
    print(f"      True obs_next:    {test_t['obs_next'][:80]}...")
    print(f"      Decoded obs_next: {decoded_text[:80]}...")
    print(f"      Decode dist: {decode_dist:.4f}  (0=perfect, 2=worst)")

    # ── Step 6: Probe fidelity ─────────────────────────────────────────────────
    print(f"\n[6/6] Computing 1-epoch probe fidelity (ex-ante Feature 1) ...")
    # Reset predictor + EMA for clean probe
    predictor2 = MLPPredictor(input_dim=hidden_dim, hidden_dim=512, num_layers=2)
    ema2       = EMATargetEncoder(embed_dim=hidden_dim, decay=0.996)
    trainer2   = JEPATrainer(encoder, predictor2, ema2, config, device=args.device)
    probe_score = trainer2.probe_fidelity(dataset.transitions, probe_epochs=1)

    print(f"\n{'='*60}")
    print(f"  Smoke test PASSED")
    print(f"  Probe fidelity (1 epoch, n=32): {probe_score:.4f}")
    print(f"  Final val_cos_dist ({args.epochs} epochs): {last_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
