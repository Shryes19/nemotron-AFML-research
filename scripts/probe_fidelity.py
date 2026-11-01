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
  3. Repeat with 5 different random draws (seeds) and average — with only 32
     examples a single draw is noisy, so the spread (std) is reported too.
     5 draws matches Feature 2 (capability mix) in Siddhant's pipeline.

The probe uses the SAME model as the JEPA arm (configs/config.yaml: input
mode, admissible-action feature, training target, loss).

Games with fewer than n + 4 transitions are probed on whatever they have and
flagged "short" in the notes column, so you can decide whether to keep them.

Usage
─────
    # Real run on GPU
    python scripts/probe_fidelity.py --buffer data/buffers/alfworld.jsonl --real-encoder

    # Quick code check with the fake encoder
    python scripts/probe_fidelity.py --buffer data/buffers/mock_n64.jsonl

Feature-matrix output
─────────────────────
    results/jepa_phase1/probe_scores.csv   columns env, game, probe_fidelity, ...
    read by scripts/build_feature_matrix.py --probe. Each run REPLACES the rows
    of the games it probed and keeps everything else, so run it once per
    buffer (ALFWorld, ScienceWorld, ByteSized32) to build one combined file.

Detailed output (default folder: outputs/probe/<buffer name>/)
─────────────────────────────────────────────────────
    probe_fidelity.csv   one row per game — this feeds the RQ3 feature matrix
    probe_fidelity.json  same data plus run settings
"""

import argparse
import csv
import json
import math
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
from jepa.training.features import with_features
from jepa.training.train import JEPATrainer
from jepa.compute import track

SCORES_CSV = "results/jepa_phase1/probe_scores.csv"   # read by scripts/build_feature_matrix.py

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


SCORE_COLS = ["env", "game", "probe_fidelity", "probe_fidelity_std", "probe_n_train",
              "probe_seeds", "probe_schedule", "probe_model"]


def write_scores(path: Path, rows, meta):
    """
    Upsert (env, game) rows into the feature-matrix CSV. Games that could not
    be probed are left out, so the matrix shows them as missing, not as 0.
    """
    keep = {}
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                keep[(r["env"], r["game"])] = r
    for r in rows:
        key = (r["env"], r["game"])
        keep.pop(key, None)
        if r["probe_fidelity"] is None:
            continue
        keep[key] = {"env": r["env"], "game": r["game"], "probe_fidelity": r["probe_fidelity"],
                     "probe_fidelity_std": r["probe_cos_dist_std"] if r["probe_cos_dist_std"] is not None else "",
                     "probe_n_train": r["n_train"], "probe_seeds": r["seeds_run"],
                     "probe_schedule": meta["probe_schedule"], "probe_model": meta["probe_model"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(keep):
            w.writerow(keep[key])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--n", type=int, default=32, help="training transitions per game")
    p.add_argument("--probe-epochs", type=int, default=None,
                   help="train for this many epochs instead of the default step count")
    p.add_argument("--probe-steps", type=int, default=None,
                   help="optimiser steps per probe (default: training.probe_steps = 50; "
                        "1 epoch at n=32 is a single step and does not separate games)")
    p.add_argument("--seeds", type=int, default=5,
                   help="random 32-transition draws per game, averaged (5 = same as Feature 2)")
    p.add_argument("--scores-out", default=SCORES_CSV,
                   help="feature-matrix CSV (env, game, probe_fidelity); rows for these games are "
                        "replaced, rows for other envs/games are kept")
    p.add_argument("--max-val", type=int, default=64, help="max held-out transitions per game")
    p.add_argument("--real-encoder", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    jcfg, tcfg, ecfg = (cfg.get(k, {}) for k in ("jepa", "training", "encoder"))
    probe_epochs = args.probe_epochs or int(tcfg.get("probe_epochs", 1))
    # Default schedule = training.probe_steps (50, agreed 2026-11-01);
    # an explicit --probe-epochs overrides it.
    if args.probe_steps is None and args.probe_epochs is None and tcfg.get("probe_steps"):
        args.probe_steps = int(tcfg["probe_steps"])
    batch_size = 64
    # Same model as the JEPA arm (configs/config.yaml): input mode, admissible
    # feature, training target and loss — so Feature 1 probes the arm we report.
    mode = jcfg.get("input_mode", "joint")
    use_adm = bool(jcfg.get("admissible_feature", False))
    target_mode = jcfg.get("target_mode", "ema")
    loss_name = jcfg.get("loss", "mse")
    base_seed = int(cfg.get("seed", 42))
    out_dir = Path(args.out or f"outputs/probe/{Path(args.buffer).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = TransitionDataset.from_jsonl(args.buffer).transitions
    groups = defaultdict(list)
    for i, t in enumerate(items):
        groups[(t.get("env", "unknown"), t.get("game", "unknown"))].append(i)
    print(f"[Probe] {len(items)} transitions across {len(groups)} games  "
          f"n={args.n}  {f'steps={args.probe_steps}' if args.probe_steps else f'epochs={probe_epochs}'}  "
          f"seeds={args.seeds}\n[Probe] model: input={mode}{' + admissible' if use_adm else ''}  "
          f"target={target_mode}  loss={loss_name}")

    encoder = load_encoder(
        real=args.real_encoder,
        device=args.device,
        model_name=ecfg.get("model_name"),
        quantize=bool(ecfg.get("quantize", True)),
        batch_size=int(ecfg.get("batch_size", 32)),
        max_length=int(ecfg.get("max_length", 512)),
    )
    z_ctx, z_next = encode_buffer(encoder, items, mode=mode)
    if use_adm:
        z_ctx = with_features(z_ctx, items)

    train_cfg = {"lr": float(tcfg.get("lr", 1e-4)),
                 "weight_decay": float(tcfg.get("weight_decay", 1e-4)),
                 "batch_size": batch_size, "target_mode": target_mode, "loss": loss_name}

    rows = []
    for (env, game), idx in sorted(groups.items()):
        dists, top1s, n_train, n_val, note = [], [], 0, 0, ""
        game_track = track("probe", arm="D", env=env, game=game, n=args.n, seeds=args.seeds)
        game_track.__enter__()
        for s in range(args.seeds):
            seed = base_seed + s
            tr, va, note = split_for_game(idx, args.n, args.max_val, random.Random(seed))
            if tr is None:
                break
            torch.manual_seed(seed)
            predictor = MLPPredictor(z_ctx.shape[1], hidden_dim=int(jcfg.get("hidden_dim", 2048)),
                                     num_layers=int(jcfg.get("num_layers", 2)),
                                     dropout=float(jcfg.get("dropout", 0.1)),
                                     output_dim=z_next.shape[1])
            ema = EMATargetEncoder(z_next.shape[1], decay=float(jcfg.get("ema_decay", 0.996)))
            trainer = JEPATrainer(None, predictor, ema, train_cfg, device=args.device)
            epochs = (max(1, round(args.probe_steps / math.ceil(len(tr) / batch_size)))
                      if args.probe_steps else probe_epochs)
            hist = trainer.train_on_embeddings(z_ctx[tr], z_next[tr], z_ctx[va], z_next[va],
                                               epochs=epochs, verbose=False)
            dists.append(hist["val_cos_dist"][-1])
            top1s.append(hist["val_top1"][-1])
            n_train, n_val = len(tr), len(va)
        game_track.__exit__(None, None, None)

        row = {
            "env": env,
            "game": game,
            "n_total": len(idx),
            "n_train": n_train,
            "n_val": n_val,
            "probe_fidelity": round(mean(dists), 5) if dists else None,   # = cos-dist mean
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
                   "n": args.n, "probe_epochs": None if args.probe_steps else probe_epochs,
                   "probe_steps": args.probe_steps, "seeds": args.seeds,
                   "input_mode": mode, "admissible_feature": use_adm,
                   "target_mode": target_mode, "loss": loss_name, "games": rows}, f, indent=2)

    # Feature-matrix CSV: merge into one file across buffers/envs
    write_scores(Path(args.scores_out), rows, {
        "probe_n": args.n, "probe_seeds": args.seeds,
        "probe_schedule": f"{args.probe_steps} steps" if args.probe_steps else f"{probe_epochs} epoch(s)",
        "probe_model": f"{mode}{'+adm' if use_adm else ''}/{target_mode}/{loss_name}"})

    scored = [r for r in rows if r["seeds_run"]]
    print(f"\n[Probe] Scored {len(scored)}/{len(rows)} games  →  {csv_path}")
    print(f"[Probe] Feature 1 for the RQ3 matrix  →  {args.scores_out}")


if __name__ == "__main__":
    main()
