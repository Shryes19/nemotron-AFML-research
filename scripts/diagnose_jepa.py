"""
scripts/diagnose_jepa.py
────────────────────────
Three diagnostics that answer: "Does the JEPA predictor really generalise to
unseen games, or is it overfitting / held back by the decode step?"

All three run on the UNSEEN-GAMES test (whole games held out of training) and
are averaged over several held-out game pairs (--seeds).

Diagnostic 1 — compare without the buffer ceiling
    NN-decode can only answer with a state already in the training buffer, so
    exact-match accuracy is capped by "reach". Here JEPA's prediction may also
    pick from the unseen games' true next states ("open-set"), which shows what
    the predictor itself gets right. The action lookup (naive retrieval) can
    never name a state it has not seen, so its score is the same either way.
    Also reported: "soft" similarity — how close (cosine, in the frozen
    NV-Embed space) each method's answer text is to the true text, so a
    near-miss ("cabinet 2" vs "cabinet 3") counts for something.

Diagnostic 2 — split the errors by action type
      reveals hidden info : go / open / examine that succeed — the answer shows
                            what is on or in a receptacle, which no model can
                            know in a game it has never seen
      predictable         : take, move/put, close, use, heat, cool, clean,
                            look, inventory ... that succeed
      failed action       : the true next state is "Nothing happens."
    If JEPA beats the lookup on the predictable group, that is real
    generalisation the overall average hides.

Diagnostic 3 — early stopping
    One more whole game is held out as a VALIDATION game. Training runs the
    usual 2,000 steps, but the checkpoint that scored best on the validation
    game is also kept. Comparing "best" with "final" on the test games shows
    how much the fixed 2,000 steps overfit. The gap between training error and
    test error is reported too.

Usage
─────
    python scripts/diagnose_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
    python scripts/diagnose_jepa.py --buffer data/buffers/mock_n1200.jsonl            # fake encoder, code check

Output: printed tables + outputs/diagnose/<buffer name>/diagnostics.json
"""

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa.encoder.load import load_encoder
from jepa.predictor.mlp_predictor import MLPPredictor, EMATargetEncoder
from jepa.training.buffer import TransitionDataset
from jepa.training.embed_cache import encode_buffer
from jepa.training.train import JEPATrainer

FAIL_TEXT = "Nothing happens."
HIDDEN_VERBS = {"go", "open", "examine"}
GROUPS = ["reveals hidden info", "predictable", "failed action"]


def load_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def action_group(t):
    if t["obs_next"].strip() == FAIL_TEXT:
        return "failed action"
    verb = t["action"].split()[0].lower() if t["action"].split() else ""
    return "reveals hidden info" if verb in HIDDEN_VERBS else "predictable"


# ──────────────────────────────────────────────────────────────────────────────
# Splits
# ──────────────────────────────────────────────────────────────────────────────

def split_games(items, n_test_games, seed):
    """Return (pool_idx, val_idx, test_idx, test_games, val_game)."""
    games = sorted({t.get("game", "unknown") for t in items})
    if len(games) < n_test_games + 2:
        raise SystemExit(f"need at least {n_test_games + 2} games, buffer has {len(games)}")
    rng = random.Random(seed)
    rng.shuffle(games)
    test_games = sorted(games[:n_test_games])          # same pick as train_jepa.py --holdout-games
    val_game = games[n_test_games]
    test_idx = [i for i, t in enumerate(items) if t.get("game", "unknown") in test_games]
    val_idx = [i for i, t in enumerate(items) if t.get("game", "unknown") == val_game]
    pool_idx = [i for i, t in enumerate(items) if t.get("game", "unknown") not in test_games
                and t.get("game", "unknown") != val_game]
    rng.shuffle(pool_idx)
    return pool_idx, val_idx, test_idx, test_games, val_game


# ──────────────────────────────────────────────────────────────────────────────
# Metrics on the test set
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics(trainer, items, z_next, tr_idx, te_idx, pred):
    """
    pred: JEPA predictions for the test transitions (projected space), (T, D).
    Returns overall metrics + per-group metrics.
    """
    train_items = [items[i] for i in tr_idx]
    test_items = [items[i] for i in te_idx]
    truth = [t["obs_next"] for t in test_items]

    # one row per distinct next-state text (first occurrence)
    def unique_rows(idxs):
        seen, out = set(), []
        for i in idxs:
            txt = items[i]["obs_next"]
            if txt not in seen:
                seen.add(txt)
                out.append(i)
        return out

    buf_rows = unique_rows(tr_idx)
    open_rows = unique_rows(tr_idx + te_idx)          # buffer + the unseen games' true states
    buf_txt = [items[i]["obs_next"] for i in buf_rows]
    open_txt = [items[i]["obs_next"] for i in open_rows]
    buf_index = trainer.project_targets(z_next[buf_rows])
    open_index = trainer.project_targets(z_next[open_rows])

    jepa_buf = [buf_txt[k] for k in (pred @ buf_index.T).argmax(1).tolist()]
    jepa_open = [open_txt[k] for k in (pred @ open_index.T).argmax(1).tolist()]

    # action lookup = naive retrieval
    majority = Counter(t["obs_next"] for t in train_items).most_common(1)[0][0]
    by_action = defaultdict(Counter)
    for t in train_items:
        by_action[t["action"]][t["obs_next"]] += 1
    lookup = [by_action[t["action"]].most_common(1)[0][0] if t["action"] in by_action else majority
              for t in test_items]

    # soft similarity in the frozen encoder space (neutral to both methods)
    text_row = {}
    for i in tr_idx + te_idx:
        text_row.setdefault(items[i]["obs_next"], i)
    raw = F.normalize(z_next.float(), dim=-1)

    def soft(answers):
        return [float(raw[text_row[a]] @ raw[text_row[b]]) for a, b in zip(answers, truth)]

    seen_buf = set(buf_txt)
    per = {
        "jepa_decode": [a == b for a, b in zip(jepa_buf, truth)],
        "jepa_open": [a == b for a, b in zip(jepa_open, truth)],
        "lookup": [a == b for a, b in zip(lookup, truth)],
        "reach": [b in seen_buf for b in truth],
        "soft_jepa": soft(jepa_buf),
        "soft_lookup": soft(lookup),
    }
    overall = {k: float(mean(v)) for k, v in per.items()}
    overall["n_test"] = len(truth)

    groups = {}
    labels = [action_group(t) for t in test_items]
    for g in GROUPS:
        ix = [k for k, lab in enumerate(labels) if lab == g]
        if not ix:
            continue
        groups[g] = {k: float(mean([v[j] for j in ix])) for k, v in per.items()}
        groups[g]["count"] = len(ix)
    return overall, groups


# ──────────────────────────────────────────────────────────────────────────────
# One budget, one seed
# ──────────────────────────────────────────────────────────────────────────────

def run_budget(b, seed, z_ctx, z_next, items, pool_idx, val_idx, test_idx, cfg, device, steps):
    jcfg, tcfg = cfg.get("jepa", {}), cfg.get("training", {})
    batch_size = 64
    tr_idx = pool_idx[:b]
    torch.manual_seed(seed)
    predictor = MLPPredictor(z_ctx.shape[1], hidden_dim=int(jcfg.get("hidden_dim", 2048)),
                             num_layers=int(jcfg.get("num_layers", 2)),
                             dropout=float(jcfg.get("dropout", 0.1)), output_dim=z_next.shape[1])
    ema = EMATargetEncoder(z_next.shape[1], decay=float(jcfg.get("ema_decay", 0.996)))
    trainer = JEPATrainer(None, predictor, ema,
                          {"lr": float(tcfg.get("lr", 1e-4)),
                           "weight_decay": float(tcfg.get("weight_decay", 1e-4)),
                           "batch_size": batch_size}, device=device)

    spe = math.ceil(b / batch_size)
    epochs = max(1, round(steps / spe))
    best = {"val": float("inf"), "step": 0, "state": None}
    for e in range(1, epochs + 1):
        hist = trainer.train_on_embeddings(z_ctx[tr_idx], z_next[tr_idx],
                                           z_ctx[val_idx], z_next[val_idx],
                                           epochs=1, verbose=False)
        v = hist["val_cos_dist"][-1]
        if v < best["val"]:
            best = {"val": v, "step": e * spe, "state": (
                copy.deepcopy(trainer.predictor.state_dict()),
                copy.deepcopy(trainer.ema.online_proj.state_dict()),
                copy.deepcopy(trainer.ema.target_proj.state_dict()))}

    def evaluate_now():
        pred = trainer.predict(z_ctx[test_idx])
        overall, groups = test_metrics(trainer, items, z_next, tr_idx, test_idx, pred)
        overall["train_cos_dist"] = trainer.evaluate(z_ctx[tr_idx], z_next[tr_idx])["val_cos_dist"]
        overall["test_cos_dist"] = trainer.evaluate(z_ctx[test_idx], z_next[test_idx])["val_cos_dist"]
        return overall, groups

    final_overall, final_groups = evaluate_now()
    p_sd, on_sd, tg_sd = best["state"]
    trainer.predictor.load_state_dict(p_sd)
    trainer.ema.online_proj.load_state_dict(on_sd)
    trainer.ema.target_proj.load_state_dict(tg_sd)
    best_overall, _ = evaluate_now()

    return {"final": final_overall, "groups": final_groups, "early_stopped": best_overall,
            "best_step": best["step"], "total_steps": epochs * spe}


# ──────────────────────────────────────────────────────────────────────────────

def avg(dicts, key):
    vals = [d[key] for d in dicts if key in d]
    return mean(vals) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--budgets", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="one held-out game pair per seed (same pairs as train_jepa.py --holdout-games 2 --seed S)")
    p.add_argument("--test-games", type=int, default=2)
    p.add_argument("--steps", type=int, default=None, help="default: training.steps in config")
    p.add_argument("--joint", action="store_true", help="use the old joint text input")
    p.add_argument("--real-encoder", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ecfg, tcfg, dcfg = cfg.get("encoder", {}), cfg.get("training", {}), cfg.get("data", {})
    budgets = sorted(args.budgets or dcfg.get("budgets", [32, 64, 128, 256, 512, 1024]))
    steps = args.steps or int(tcfg.get("steps") or 2000)
    mode = "joint" if args.joint else cfg.get("jepa", {}).get("input_mode", "separate")
    out_dir = Path(args.out or f"outputs/diagnose/{Path(args.buffer).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = TransitionDataset.from_jsonl(args.buffer).transitions
    encoder = load_encoder(real=args.real_encoder, device=args.device,
                           model_name=ecfg.get("model_name"), quantize=bool(ecfg.get("quantize", True)),
                           batch_size=int(ecfg.get("batch_size", 16)),
                           max_length=int(ecfg.get("max_length", 512)))
    z_ctx, z_next = encode_buffer(encoder, items, mode=mode)
    print(f"[Diag] {len(items)} transitions  input={mode}  steps={steps}  seeds={args.seeds}")

    runs = defaultdict(list)          # budget -> list of per-seed results
    splits = {}
    for seed in args.seeds:
        pool_idx, val_idx, test_idx, test_games, val_game = split_games(items, args.test_games, seed)
        splits[seed] = {"test_games": test_games, "val_game": val_game,
                        "n_pool": len(pool_idx), "n_val": len(val_idx), "n_test": len(test_idx)}
        print(f"\n[Seed {seed}] test={test_games}\n           val={val_game}  pool={len(pool_idx)}")
        for b in budgets:
            if b > len(pool_idx):
                print(f"  n={b}: skipped (pool has {len(pool_idx)})")
                continue
            r = run_budget(b, seed, z_ctx, z_next, items, pool_idx, val_idx, test_idx,
                           cfg, args.device, steps)
            runs[b].append(r)
            f_ = r["final"]
            print(f"  n={b:>4}  decode={f_['jepa_decode']:.3f}  open={f_['jepa_open']:.3f}  "
                  f"lookup={f_['lookup']:.3f}  best_step={r['best_step']}/{r['total_steps']}")

    # ── Summaries (means over seeds) ──────────────────────────────────────────
    line = "=" * 92
    print(f"\n{line}\n  DIAGNOSTIC 1 — without the buffer ceiling   (unseen games, mean of {len(args.seeds)} pairs)\n{line}")
    print(f"  {'n':>5} {'JEPA decode':>12} {'JEPA open-set':>14} {'lookup':>8} {'reach':>7} "
          f"{'soft JEPA':>10} {'soft lookup':>12}")
    for b in sorted(runs):
        fs = [r["final"] for r in runs[b]]
        print(f"  {b:>5} {avg(fs,'jepa_decode'):>12.3f} {avg(fs,'jepa_open'):>14.3f} {avg(fs,'lookup'):>8.3f} "
              f"{avg(fs,'reach'):>7.3f} {avg(fs,'soft_jepa'):>10.3f} {avg(fs,'soft_lookup'):>12.3f}")
    print("  open-set = JEPA may also pick the unseen games' true states (no buffer ceiling).")
    print("  soft = cosine similarity of the answer text to the true text (1 = identical).")

    print(f"\n{line}\n  DIAGNOSTIC 2 — by action type   (unseen games, mean of pairs; count = test transitions per pair)\n{line}")
    print(f"  {'n':>5} {'group':<21} {'count':>6} {'JEPA decode':>12} {'JEPA open':>10} {'lookup':>8} {'reach':>7}")
    for b in sorted(runs):
        for g in GROUPS:
            gs = [r["groups"][g] for r in runs[b] if g in r["groups"]]
            if not gs:
                continue
            print(f"  {b:>5} {g:<21} {avg(gs,'count'):>6.0f} {avg(gs,'jepa_decode'):>12.3f} "
                  f"{avg(gs,'jepa_open'):>10.3f} {avg(gs,'lookup'):>8.3f} {avg(gs,'reach'):>7.3f}")

    print(f"\n{line}\n  DIAGNOSTIC 3 — early stopping on a validation game   (unseen test games, mean of pairs)\n{line}")
    print(f"  {'n':>5} {'best step':>10} {'decode final':>13} {'decode early':>13} "
          f"{'open final':>11} {'open early':>11} {'train dist':>11} {'test dist':>10}")
    for b in sorted(runs):
        fs = [r["final"] for r in runs[b]]
        es = [r["early_stopped"] for r in runs[b]]
        bs = mean(r["best_step"] for r in runs[b])
        print(f"  {b:>5} {bs:>10.0f} {avg(fs,'jepa_decode'):>13.3f} {avg(es,'jepa_decode'):>13.3f} "
              f"{avg(fs,'jepa_open'):>11.3f} {avg(es,'jepa_open'):>11.3f} "
              f"{avg(fs,'train_cos_dist'):>11.3f} {avg(fs,'test_cos_dist'):>10.3f}")
    print(f"  best step = where the validation game scored best (out of ~{steps}).")
    print("  train/test dist = prediction error on training data vs unseen test games (big gap = overfitting).")

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump({"buffer": args.buffer, "input_mode": mode, "steps": steps, "seeds": args.seeds,
                   "splits": {str(k): v for k, v in splits.items()},
                   "runs": {str(b): v for b, v in runs.items()}}, f, indent=2)
    print(f"\nSaved {out_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
