"""
scripts/probe_fidelity.py
─────────────────────────
Ex-ante Feature 1 for RQ3: the "probe fidelity" score, computed separately for
every GAME in a buffer (the per-game unit agreed with the team).

The question this score answers
───────────────────────────────
"If we train the JEPA very briefly (1 epoch) on just 32 transitions from this
game, how well does it already predict held-out next states from the SAME
game?"  It is cheap, and RQ3 asks whether cheap scores like this can predict
which world-model arm wins after full training.

How it works
────────────
For each game:
  1. Take n (default 32) of its transitions for training and up to 64 OTHER
     transitions from the same game for scoring.
  2. Train a fresh predictor for 1 epoch, measure the held-out cosine distance.
  3. Repeat with 3 different random draws (seeds) and average — with only 32
     examples a single draw is noisy, so the spread (std) is reported too.

Games with fewer than n + 4 transitions are probed on whatever they have and
flagged "short" in the notes column, so you can decide whether to keep them.

Usage
─────
    # Real run on GPU
    python scripts/probe_fidelity.py --buffer data/buffers/alfworld.jsonl --real-encoder

    # Quick code check with the fake encoder
    python scripts/probe_fidelity.py --buffer data/buffers/mock_n64.jsonl

Output (default folder: outputs/probe/<buffer name>/)
─────────────────────────────────────────────────────
    probe_fidelity.csv   one row per game — this feeds the RQ3 feature matrix
    probe_fidelity.json  same data plus run settings
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa.encoder.load import load_encoder
from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
from jepa.training.buffer import TransitionDataset
from jepa.training.embed_cache import encode_buffer
from jepa.training.train import JEPATrainer

MIN_VAL = 4        # never score on fewer held-out examples than this
MIN_TRAIN = 8      # never probe a game with fewer training examples than this


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def split_for_game(idx, n, max_val, rng):
    """Return (train_idx, val_idx, note) for one game's transition indices."""
    idx = idx[:]
    rng.shuffle(idx)
    if len(idx) >= n + MIN_VAL:
        return idx[:n], idx[n : n + max_val], ""
    n_val = max(MIN_VAL, len(idx) // 5)
    n_train = len(idx) - n_val
    if n_train < MIN_TRAIN:
        return None, None, f"skipped: only {len(idx)} transitions"
    return idx[:n_train], idx[n_train:], f"short: trained on {n_train} not {n}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--n", type=int, default=32, help="training transitions per game")
    p.add_argument("--probe-epochs", type=int, default=None, help="default: training.probe_epochs")
    p.add_argument("--seeds", type=int, default=3, help="repeats per game (averaged)")
    p.add_argument("--max-val", type=int, default=64, help="max held-out transitions per game")
    p.add_argument("--real-encoder", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    jcfg, tcfg, ecfg = (cfg.get(k, {}) for k in ("jepa", "training", "encoder"))
    probe_epochs = args.probe_epochs or int(tcfg.get("probe_epochs", 1))
    base_seed = int(cfg.get("seed", 42))
    out_dir = Path(args.out or f"outputs/probe/{Path(args.buffer).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = TransitionDataset.from_jsonl(args.buffer).transitions
    groups = defaultdict(list)
    for i, t in enumerate(items):
        groups[(t.get("env", "unknown"), t.get("game", "unknown"))].append(i)
    print(f"[Probe] {len(items)} transitions across {len(groups)} games  "
          f"n={args.n}  epochs={probe_epochs}  seeds={args.seeds}")

    encoder = load_encoder(
        real=args.real_encoder,
        device=args.device,
        model_name=ecfg.get("model_name"),
        quantize=bool(ecfg.get("quantize", True)),
        batch_size=int(ecfg.get("batch_size", 32)),
        max_length=int(ecfg.get("max_length", 512)),
    )
    z_ctx, z_next = encode_buffer(encoder, items)
    dim = z_ctx.shape[1]

    train_cfg = {"lr": float(tcfg.get("lr", 1e-4)),
                 "weight_decay": float(tcfg.get("weight_decay", 1e-4)),
                 "batch_size": 64}

    rows = []
    for (env, game), idx in sorted(groups.items()):
        dists, top1s, n_train, n_val, note = [], [], 0, 0, ""
        for s in range(args.seeds):
            seed = base_seed + s
            tr, va, note = split_for_game(idx, args.n, args.max_val, random.Random(seed))
            if tr is None:
                break
            torch.manual_seed(seed)
            predictor = MLPPredictor(dim, hidden_dim=int(jcfg.get("hidden_dim", 2048)),
                                     num_layers=int(jcfg.get("num_layers", 2)),
                                     dropout=float(jcfg.get("dropout", 0.1)))
            ema = EMATargetEncoder(dim, decay=float(jcfg.get("ema_decay", 0.996)))
            trainer = JEPATrainer(None, predictor, ema, train_cfg, device=args.device)
            hist = trainer.train_on_embeddings(z_ctx[tr], z_next[tr], z_ctx[va], z_next[va],
                                               epochs=probe_epochs, verbose=False)
            dists.append(hist["val_cos_dist"][-1])
            top1s.append(hist["val_top1"][-1])
            n_train, n_val = len(tr), len(va)

        row = {
            "env": env,
            "game": game,
            "n_total": len(idx),
            "n_train": n_train,
            "n_val": n_val,
            "probe_cos_dist_mean": round(mean(dists), 5) if dists else None,
            "probe_cos_dist_std": round(pstdev(dists), 5) if len(dists) > 1 else None,
            "probe_top1_mean": round(mean(top1s), 4) if top1s else None,
            "seeds_run": len(dists),
            "note": note,
        }
        rows.append(row)
        shown = f"{row['probe_cos_dist_mean']:.4f} ± {row['probe_cos_dist_std'] or 0:.4f}" if dists else "—"
        print(f"  {env:>12} / {game:<16} n={len(idx):>4}  probe={shown}  {note}")

    csv_path = out_dir / "probe_fidelity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "probe_fidelity.json", "w") as f:
        json.dump({"buffer": args.buffer, "encoder": getattr(encoder, "model_name", "unknown"),
                   "n": args.n, "probe_epochs": probe_epochs, "seeds": args.seeds,
                   "games": rows}, f, indent=2)

    scored = [r for r in rows if r["seeds_run"]]
    print(f"\n[Probe] Scored {len(scored)}/{len(rows)} games  →  {csv_path}")


if __name__ == "__main__":
    main()
