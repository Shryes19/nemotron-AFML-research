"""
scripts/train_jepa.py
─────────────────────
Train the JEPA predictor at every data budget (n = 32, 64, ... 1024) on one
buffer, and save the loss curves. This is the Phase 1 deliverable.

How it works, in plain words
────────────────────────────
1.  Load the buffer and set aside a FIXED held-out set (10% by default).
    Every budget is scored on the same held-out examples, so the numbers are
    comparable across budgets.
2.  Encode every transition once (cached in outputs/cache/), because the 8B
    encoder is the slow part.
3.  For each budget n, take the first n transitions of the (shuffled) training
    pool — so the 64 set contains the 32 set, and so on — and train a fresh
    predictor on them.
4.  Save, per budget: a checkpoint and its per-epoch history. Save overall:
    results.json, a printed summary table and loss_curves.png.

Usage
─────
    # Real run on GPU (Siddhant's machine, inside .venv-enc)
    python scripts/train_jepa.py --buffer data/buffers/alfworld.jsonl --real-encoder

    # Quick code check on CPU with the fake encoder
    python scripts/train_jepa.py --buffer data/buffers/mock_n64.jsonl --epochs 3

Outputs (default folder: outputs/jepa/<buffer name>/)
─────────────────────────────────────────────────────
    results.json         settings + final metrics + full history per budget
    loss_curves.png      4 panels: train_mse, val_cos_dist, val_top1, spread
    budget_<n>.pt        trained predictor + EMA weights for that budget
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa.encoder.load import load_encoder
from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
from jepa.training.buffer import TransitionDataset
from jepa.training.embed_cache import encode_buffer
from jepa.training.nn_decode import NNDecoder
from jepa.training.train import JEPATrainer


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[Config] {path} not found — using built-in defaults")
        return {}


def decode_metrics(trainer, train_items, z_next_train, val_items, z_ctx_val):
    """
    NN-decode every held-out example against the TRAINING buffer
    (JEPA can only ever answer with a state it has seen).

    decode_exact:  decoded text == true next-state text
    reachable:     the true next-state text exists in the training buffer at all
                   (decode_exact can never be higher than this)
    decode_dist:   mean cosine distance to the chosen neighbour
    """
    index = trainer.project_targets(z_next_train)
    decoder = NNDecoder(train_items, index)
    texts, dists = decoder.decode_batch(trainer.predict(z_ctx_val))

    seen = {t["obs_next"] for t in train_items}
    truth = [t["obs_next"] for t in val_items]
    n = len(truth)
    return {
        "decode_exact": sum(a == b for a, b in zip(texts, truth)) / n,
        "reachable": sum(t in seen for t in truth) / n,
        "decode_dist": sum(dists) / n,
    }


def plot_curves(results: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("[Plot] matplotlib not installed — skipping loss_curves.png "
              "(pip install matplotlib)")
        return

    panels = [
        ("train_mse", "Train loss (MSE)", "lower is better"),
        ("val_cos_dist", "Held-out cosine distance", "lower is better"),
        ("val_top1", "Held-out top-1 match", "higher is better"),
        ("target_spread", "Target spread (collapse check)", "near 0 = collapse"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, hint) in zip(axes.flat, panels):
        for b, r in results["budgets"].items():
            hist = r["history"][key]
            ax.plot(range(1, len(hist) + 1), hist, marker=".", label=f"n={b}")
        ax.set_title(f"{title}\n({hint})", fontsize=10)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        if key == "target_spread":
            ax.set_ylim(0, 1.1)   # fixed scale so a drop toward 0 is obvious
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"JEPA training — {results['buffer']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True, help="JSONL transition buffer")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--budgets", type=int, nargs="+", default=None,
                   help="data budgets (default: data.budgets in config)")
    p.add_argument("--epochs", type=int, default=None, help="default: training.epochs")
    p.add_argument("--val-frac", type=float, default=None, help="default: training.held_out_frac")
    p.add_argument("--real-encoder", action="store_true", help="use NV-Embed-v2 (GPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    jcfg, tcfg, dcfg, ecfg = (cfg.get(k, {}) for k in ("jepa", "training", "data", "encoder"))

    budgets = sorted(args.budgets or dcfg.get("budgets", [32, 64, 128, 256, 512, 1024]))
    epochs = args.epochs or int(tcfg.get("epochs", 20))
    val_frac = args.val_frac if args.val_frac is not None else float(tcfg.get("held_out_frac", 0.1))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    out_dir = Path(args.out or f"outputs/jepa/{Path(args.buffer).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"  JEPA training  buffer={args.buffer}")
    print(f"  budgets={budgets}  epochs={epochs}  device={args.device}")
    print("=" * 64)

    # ── 1. Fixed held-out set + nested training pool ───────────────────────────
    items = TransitionDataset.from_jsonl(args.buffer).transitions
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(items) * val_frac))
    val_idx, pool_idx = order[:n_val], order[n_val:]
    print(f"[Split] held-out={n_val}  training pool={len(pool_idx)}")

    # ── 2. Encode everything once ──────────────────────────────────────────────
    encoder = load_encoder(
        real=args.real_encoder,
        device=args.device,
        model_name=ecfg.get("model_name"),
        quantize=bool(ecfg.get("quantize", True)),
        batch_size=int(ecfg.get("batch_size", 32)),
        max_length=int(ecfg.get("max_length", 512)),
    )
    t0 = time.time()
    z_ctx, z_next = encode_buffer(encoder, items)
    print(f"[Encode] done in {time.time() - t0:.1f}s  dim={z_ctx.shape[1]}")
    z_ctx_val, z_next_val = z_ctx[val_idx], z_next[val_idx]
    val_items = [items[i] for i in val_idx]

    # ── 3. Train one fresh predictor per budget ────────────────────────────────
    results = {
        "buffer": args.buffer,
        "encoder": getattr(encoder, "model_name", "unknown"),
        "n_transitions": len(items),
        "n_val": n_val,
        "epochs": epochs,
        "seed": seed,
        "budgets": {},
        "skipped_budgets": [],
    }
    train_cfg = {
        "lr": float(tcfg.get("lr", 1e-4)),
        "weight_decay": float(tcfg.get("weight_decay", 1e-4)),
        "batch_size": 64,
    }

    for b in budgets:
        if b > len(pool_idx):
            print(f"\n[Budget n={b}] SKIPPED — only {len(pool_idx)} training transitions available")
            results["skipped_budgets"].append(b)
            continue

        print(f"\n[Budget n={b}] training {epochs} epochs ...")
        torch.manual_seed(seed)
        tr_idx = pool_idx[:b]
        predictor = MLPPredictor(
            input_dim=z_ctx.shape[1],
            hidden_dim=int(jcfg.get("hidden_dim", 2048)),
            num_layers=int(jcfg.get("num_layers", 2)),
            dropout=float(jcfg.get("dropout", 0.1)),
        )
        ema = EMATargetEncoder(embed_dim=z_ctx.shape[1], decay=float(jcfg.get("ema_decay", 0.996)))
        trainer = JEPATrainer(None, predictor, ema, train_cfg, device=args.device)

        t0 = time.time()
        hist = trainer.train_on_embeddings(
            z_ctx[tr_idx], z_next[tr_idx], z_ctx_val, z_next_val,
            epochs=epochs, log_every=max(1, epochs // 4),
        )
        train_items = [items[i] for i in tr_idx]
        dec = decode_metrics(trainer, train_items, z_next[tr_idx], val_items, z_ctx_val)

        final = {k: v[-1] for k, v in hist.items()}
        final.update(dec)
        final["seconds"] = round(time.time() - t0, 1)
        results["budgets"][b] = {"final": final, "history": hist}
        trainer.save(str(out_dir / f"budget_{b}.pt"), extra={"budget": b, "final": final})

    # ── 4. Save + summarise ────────────────────────────────────────────────────
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  Summary (held-out set of {n_val}; random top-1 = {1 / n_val:.3f})")
    print("=" * 64)
    print(f"  {'n':>5} {'cos_dist':>9} {'top1':>6} {'spread':>7} {'decode':>7} {'reach':>6}")
    for b, r in results["budgets"].items():
        f_ = r["final"]
        print(f"  {b:>5} {f_['val_cos_dist']:>9.4f} {f_['val_top1']:>6.3f} "
              f"{f_['target_spread']:>7.4f} {f_['decode_exact']:>7.3f} {f_['reachable']:>6.3f}")
    if results["skipped_budgets"]:
        print(f"  skipped (not enough data): {results['skipped_budgets']}")

    plot_curves(results, out_dir / "loss_curves.png")
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
