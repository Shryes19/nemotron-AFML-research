"""
scripts/diagnose_jepa.py
────────────────────────
Diagnostics that answer: "Does the JEPA predictor really generalise to unseen
games, or is it overfitting / held back by the decode step?"

All run on the UNSEEN-GAMES test (whole games held out of training), averaged
over several held-out game pairs (--seeds). Defaults follow configs/config.yaml
(separate action input, admissible-action feature, 2,000 steps).

Diagnostic 1 — compare without the buffer ceiling
    NN-decode can only answer with a state already in the training buffer, so
    exact-match accuracy is capped by "reach". "open-set" lets JEPA's prediction
    also pick the unseen games' true next states, showing what the predictor
    itself gets right (a measure of the predictor, not a usable decoder).
    "top-5" = the true state is among the 5 nearest buffer states — what an
    LLM scorer reading 5 candidates would have to work with. The lookup's top-5
    = the 5 outcomes most often seen after that action string.
    "soft" = cosine similarity of the answer text to the true text in the
    frozen NV-Embed space (biased toward JEPA, which picks answers by closeness
    in a related space — report with care).

Diagnostic 2 — split the errors by action type
      reveals hidden info : go / open / examine that succeed
      predictable         : take, move, close, use, heat, cool, clean, look ...
      failed action       : the true next state is "Nothing happens."

Diagnostic 3 — choosing the action, no decoding (closest stand-in for decision utility)
    For each test step: the action actually taken + K-1 other actions from the
    same game. Each method predicts the consequence of every candidate; it
    "chooses right" if the actual action's predicted consequence is the one
    closest to what really happened. JEPA compares in embedding space with no
    decoding, so there is no buffer ceiling. The lookup compares its looked-up
    answer texts (in the frozen encoder space). Chance = 1/K.

Diagnostic 4 (optional, --early-stop-check) — early stopping
    Tested 2026-11-01: stopping at the best validation-game checkpoint scored
    worse than the full 2,000 steps at every budget, so it is off by default.

Usage
─────
    python scripts/diagnose_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder
    python scripts/diagnose_jepa.py --buffer data/buffers/alfworld_games.jsonl --real-encoder --no-admissible

Output: printed tables + outputs/diagnose/<buffer name><suffix>/diagnostics.json
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
from jepa.training.features import action_features, coverage, with_features
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


class Lookup:
    """Naive retrieval: the outcome most often seen after the same action string."""

    def __init__(self, train_items):
        self.majority = Counter(t["obs_next"] for t in train_items).most_common()
        self.by_action = defaultdict(Counter)
        for t in train_items:
            self.by_action[t["action"]][t["obs_next"]] += 1

    def answer(self, action):
        c = self.by_action.get(action)
        return c.most_common(1)[0][0] if c else self.majority[0][0]

    def topk(self, action, k):
        out = [txt for txt, _ in self.by_action.get(action, Counter()).most_common(k)]
        for txt, _ in self.majority:                    # fill with globally common outcomes
            if len(out) >= k:
                break
            if txt not in out:
                out.append(txt)
        return out


def group_means(per, labels):
    out = {}
    for g in GROUPS:
        ix = [k for k, lab in enumerate(labels) if lab == g]
        if ix:
            out[g] = {k: float(mean([v[j] for j in ix])) for k, v in per.items()}
            out[g]["count"] = len(ix)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics 1 + 2: prediction quality on the test set
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics(trainer, items, z_next, tr_idx, te_idx, pred, k_top=5):
    train_items = [items[i] for i in tr_idx]
    test_items = [items[i] for i in te_idx]
    truth = [t["obs_next"] for t in test_items]

    def unique_rows(idxs):
        seen, out = set(), []
        for i in idxs:
            if items[i]["obs_next"] not in seen:
                seen.add(items[i]["obs_next"])
                out.append(i)
        return out

    buf_rows, open_rows = unique_rows(tr_idx), unique_rows(tr_idx + te_idx)
    buf_txt = [items[i]["obs_next"] for i in buf_rows]
    open_txt = [items[i]["obs_next"] for i in open_rows]
    buf_sims = pred @ trainer.project_targets(z_next[buf_rows]).T
    open_sims = pred @ trainer.project_targets(z_next[open_rows]).T

    jepa_buf = [buf_txt[j] for j in buf_sims.argmax(1).tolist()]
    jepa_open = [open_txt[j] for j in open_sims.argmax(1).tolist()]
    k = min(k_top, len(buf_txt))
    jepa_top = [{buf_txt[j] for j in row} for row in buf_sims.topk(k, dim=1).indices.tolist()]

    lk = Lookup(train_items)
    lookup = [lk.answer(t["action"]) for t in test_items]
    lookup_top = [set(lk.topk(t["action"], k_top)) for t in test_items]

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
        "jepa_top5": [b in s for s, b in zip(jepa_top, truth)],
        "lookup": [a == b for a, b in zip(lookup, truth)],
        "lookup_top5": [b in s for s, b in zip(lookup_top, truth)],
        "reach": [b in seen_buf for b in truth],
        "soft_jepa": soft(jepa_buf),
        "soft_lookup": soft(lookup),
    }
    overall = {k_: float(mean(v)) for k_, v in per.items()}
    overall["n_test"] = len(truth)
    return overall, group_means(per, [action_group(t) for t in test_items])


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic 3: choosing the action, decode-free
# ──────────────────────────────────────────────────────────────────────────────

def action_choice(trainer, items, base_ctx, z_next, tr_idx, te_idx, use_adm, K, seed):
    """
    base_ctx: separate-mode embeddings WITHOUT features, (N, 2D) = [z_obs | z_act].
    """
    D = z_next.shape[1]
    rng = random.Random(seed)
    act_row = {}
    for i in range(len(items)):
        act_row.setdefault(items[i]["action"], i)
    by_game = defaultdict(set)
    for i in te_idx:
        by_game[items[i].get("game", "unknown")].add(items[i]["action"])

    rows, owners, sizes = [], [], []
    for n_, i in enumerate(te_idx):
        t = items[i]
        others = sorted(by_game[t.get("game", "unknown")] - {t["action"]})
        alts = rng.sample(others, min(K - 1, len(others)))
        cands = [t["action"]] + alts
        for a in cands:
            vec = torch.cat([base_ctx[i, :D], base_ctx[act_row[a], D:2 * D]])
            if use_adm:
                vec = torch.cat([vec, torch.tensor(action_features(t["obs"], a), dtype=vec.dtype)])
            rows.append(vec)
            owners.append((n_, a))
        sizes.append(len(cands))

    pred = trainer.predict(torch.stack(rows))
    target = trainer.project_targets(z_next[te_idx])

    lk = Lookup([items[j] for j in tr_idx])
    text_row = {}
    for j in tr_idx + te_idx:
        text_row.setdefault(items[j]["obs_next"], j)
    raw = F.normalize(z_next.float(), dim=-1)

    jepa_ok, look_ok, chance = [], [], []
    pos = 0
    for n_, i in enumerate(te_idx):
        m = sizes[n_]
        js = (pred[pos:pos + m] @ target[n_]).tolist()
        jepa_ok.append(1.0 if max(range(m), key=lambda c: js[c]) == 0 else 0.0)
        ls = [float(raw[text_row[lk.answer(owners[pos + c][1])]] @ raw[i]) for c in range(m)]
        best = max(ls)
        tied = [c for c in range(m) if abs(ls[c] - best) < 1e-6]
        look_ok.append(1.0 / len(tied) if 0 in tied else 0.0)   # fair tie-break
        chance.append(1.0 / m)
        pos += m

    per = {"jepa": jepa_ok, "lookup": look_ok, "chance": chance}
    overall = {k: float(mean(v)) for k, v in per.items()}
    return overall, group_means(per, [action_group(items[i]) for i in te_idx])


# ──────────────────────────────────────────────────────────────────────────────
# One budget, one seed
# ──────────────────────────────────────────────────────────────────────────────

def run_budget(b, seed, z_ctx, base_ctx, z_next, items, pool_idx, val_idx, test_idx,
               cfg, device, steps, use_adm, K, early_stop_check):
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

    if early_stop_check:
        for e in range(1, epochs + 1):
            hist = trainer.train_on_embeddings(z_ctx[tr_idx], z_next[tr_idx], z_ctx[val_idx],
                                               z_next[val_idx], epochs=1, verbose=False)
            if hist["val_cos_dist"][-1] < best["val"]:
                best = {"val": hist["val_cos_dist"][-1], "step": e * spe, "state": (
                    copy.deepcopy(trainer.predictor.state_dict()),
                    copy.deepcopy(trainer.ema.online_proj.state_dict()),
                    copy.deepcopy(trainer.ema.target_proj.state_dict()))}
    else:
        trainer.train_on_embeddings(z_ctx[tr_idx], z_next[tr_idx], z_ctx[val_idx], z_next[val_idx],
                                    epochs=epochs, verbose=False)

    def evaluate_now():
        pred = trainer.predict(z_ctx[test_idx])
        overall, groups = test_metrics(trainer, items, z_next, tr_idx, test_idx, pred)
        overall["train_cos_dist"] = trainer.evaluate(z_ctx[tr_idx], z_next[tr_idx])["val_cos_dist"]
        overall["test_cos_dist"] = trainer.evaluate(z_ctx[test_idx], z_next[test_idx])["val_cos_dist"]
        return overall, groups

    final_overall, final_groups = evaluate_now()
    out = {"final": final_overall, "groups": final_groups, "total_steps": epochs * spe}
    if base_ctx is not None:
        out["choice"], out["choice_groups"] = action_choice(
            trainer, items, base_ctx, z_next, tr_idx, test_idx, use_adm, K, seed)

    if early_stop_check:
        p_sd, on_sd, tg_sd = best["state"]
        trainer.predictor.load_state_dict(p_sd)
        trainer.ema.online_proj.load_state_dict(on_sd)
        trainer.ema.target_proj.load_state_dict(tg_sd)
        out["early_stopped"], _ = evaluate_now()
        out["best_step"] = best["step"]
    return out


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
    p.add_argument("--candidates", type=int, default=5, help="K for the action-choice test")
    p.add_argument("--joint", action="store_true", help="old joint text input (skips the action-choice test)")
    p.add_argument("--no-admissible", action="store_true", help="leave out the admissible-action input")
    p.add_argument("--early-stop-check", action="store_true",
                   help="also report early stopping on a validation game (slower)")
    p.add_argument("--real-encoder", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ecfg, tcfg, dcfg, jcfg = (cfg.get(k, {}) for k in ("encoder", "training", "data", "jepa"))
    budgets = sorted(args.budgets or dcfg.get("budgets", [32, 64, 128, 256, 512, 1024]))
    steps = args.steps or int(tcfg.get("steps") or 2000)
    mode = "joint" if args.joint else jcfg.get("input_mode", "separate")
    use_adm = bool(jcfg.get("admissible_feature", False)) and not args.no_admissible
    suffix = ("_joint" if mode == "joint" else "") + ("" if use_adm else "_noadm")
    out_dir = Path(args.out or f"outputs/diagnose/{Path(args.buffer).stem}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = TransitionDataset.from_jsonl(args.buffer).transitions
    encoder = load_encoder(real=args.real_encoder, device=args.device,
                           model_name=ecfg.get("model_name"), quantize=bool(ecfg.get("quantize", True)),
                           batch_size=int(ecfg.get("batch_size", 16)),
                           max_length=int(ecfg.get("max_length", 512)))
    z_ctx, z_next = encode_buffer(encoder, items, mode=mode)
    base_ctx = z_ctx if mode == "separate" else None
    if use_adm:
        cov = coverage(items)
        print(f"[Features] admissible-action input ON  (obs with a list: {cov['has_list']:.0%})")
        z_ctx = with_features(z_ctx, items)
    print(f"[Diag] {len(items)} transitions  input={mode}{' + admissible' if use_adm else ''}  "
          f"steps={steps}  seeds={args.seeds}")

    runs, splits = defaultdict(list), {}
    for seed in args.seeds:
        pool_idx, val_idx, test_idx, test_games, val_game = split_games(items, args.test_games, seed)
        splits[seed] = {"test_games": test_games, "val_game": val_game,
                        "n_pool": len(pool_idx), "n_val": len(val_idx), "n_test": len(test_idx)}
        print(f"\n[Seed {seed}] test={test_games}\n           val={val_game}  pool={len(pool_idx)}")
        for b in budgets:
            if b > len(pool_idx):
                print(f"  n={b}: skipped (pool has {len(pool_idx)})")
                continue
            r = run_budget(b, seed, z_ctx, base_ctx, z_next, items, pool_idx, val_idx, test_idx,
                           cfg, args.device, steps, use_adm, args.candidates, args.early_stop_check)
            runs[b].append(r)
            f_ = r["final"]
            ch = f"  choice={r['choice']['jepa']:.3f}/{r['choice']['lookup']:.3f}" if "choice" in r else ""
            print(f"  n={b:>4}  decode={f_['jepa_decode']:.3f}  open={f_['jepa_open']:.3f}  "
                  f"lookup={f_['lookup']:.3f}{ch}")

    line = "=" * 100
    npairs = len(args.seeds)
    print(f"\n{line}\n  DIAGNOSTIC 1 — prediction quality on unseen games   (mean of {npairs} held-out pairs)\n{line}")
    print(f"  {'n':>5} {'JEPA decode':>12} {'JEPA open':>10} {'JEPA top5':>10} {'lookup':>8} "
          f"{'lookup top5':>12} {'reach':>7} {'soft JEPA':>10} {'soft lookup':>12}")
    for b in sorted(runs):
        fs = [r["final"] for r in runs[b]]
        print(f"  {b:>5} {avg(fs,'jepa_decode'):>12.3f} {avg(fs,'jepa_open'):>10.3f} {avg(fs,'jepa_top5'):>10.3f} "
              f"{avg(fs,'lookup'):>8.3f} {avg(fs,'lookup_top5'):>12.3f} {avg(fs,'reach'):>7.3f} "
              f"{avg(fs,'soft_jepa'):>10.3f} {avg(fs,'soft_lookup'):>12.3f}")
    print("  open = JEPA may also pick the unseen games' true states (measures the predictor, no ceiling)")
    print("  top5 = true state among 5 candidates handed to an LLM scorer (still capped by reach)")

    print(f"\n{line}\n  DIAGNOSTIC 2 — by action type   (unseen games; count = test steps per pair)\n{line}")
    print(f"  {'n':>5} {'group':<21} {'count':>6} {'JEPA decode':>12} {'JEPA open':>10} {'JEPA top5':>10} "
          f"{'lookup':>8} {'lookup top5':>12} {'reach':>7}")
    for b in sorted(runs):
        for g in GROUPS:
            gs = [r["groups"][g] for r in runs[b] if g in r["groups"]]
            if gs:
                print(f"  {b:>5} {g:<21} {avg(gs,'count'):>6.0f} {avg(gs,'jepa_decode'):>12.3f} "
                      f"{avg(gs,'jepa_open'):>10.3f} {avg(gs,'jepa_top5'):>10.3f} {avg(gs,'lookup'):>8.3f} "
                      f"{avg(gs,'lookup_top5'):>12.3f} {avg(gs,'reach'):>7.3f}")

    if any("choice" in r for rs in runs.values() for r in rs):
        print(f"\n{line}\n  DIAGNOSTIC 3 — choosing the action, no decoding   "
              f"(actual action vs {args.candidates - 1} others from the same game)\n{line}")
        print(f"  {'n':>5} {'group':<21} {'JEPA':>8} {'lookup':>8} {'chance':>8}")
        for b in sorted(runs):
            cs = [r["choice"] for r in runs[b]]
            print(f"  {b:>5} {'ALL':<21} {avg(cs,'jepa'):>8.3f} {avg(cs,'lookup'):>8.3f} {avg(cs,'chance'):>8.3f}")
            for g in GROUPS:
                gs = [r["choice_groups"][g] for r in runs[b] if g in r["choice_groups"]]
                if gs:
                    print(f"  {'':>5} {g:<21} {avg(gs,'jepa'):>8.3f} {avg(gs,'lookup'):>8.3f} {avg(gs,'chance'):>8.3f}")
        print("  JEPA compares predicted consequences in embedding space (no buffer ceiling);")
        print("  the lookup compares its looked-up answer texts. Higher = better at telling actions apart.")

    if args.early_stop_check:
        print(f"\n{line}\n  DIAGNOSTIC 4 — early stopping on a validation game\n{line}")
        print(f"  {'n':>5} {'best step':>10} {'decode final':>13} {'decode early':>13} {'train dist':>11} {'test dist':>10}")
        for b in sorted(runs):
            fs = [r["final"] for r in runs[b]]
            es = [r["early_stopped"] for r in runs[b]]
            print(f"  {b:>5} {mean(r['best_step'] for r in runs[b]):>10.0f} {avg(fs,'jepa_decode'):>13.3f} "
                  f"{avg(es,'jepa_decode'):>13.3f} {avg(fs,'train_cos_dist'):>11.3f} {avg(fs,'test_cos_dist'):>10.3f}")

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump({"buffer": args.buffer, "input_mode": mode, "admissible_feature": use_adm,
                   "steps": steps, "seeds": args.seeds, "candidates": args.candidates,
                   "splits": {str(k): v for k, v in splits.items()},
                   "runs": {str(b): v for b, v in runs.items()}}, f, indent=2)
    print(f"\nSaved {out_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
