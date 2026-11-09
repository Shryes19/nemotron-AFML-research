"""
scripts/sft_probe.py
────────────────────
SFT version of ex-ante Feature 1, for the Phase 3 ablation
"JEPA probe fidelity vs SFT text fidelity as the predictor's Feature 1".

Same question as scripts/probe_fidelity.py, asked of the SFT arm (F):
"If we fine-tune Qwen2.5-1.5B + LoRA very briefly on just 32 transitions from
this game, how well does it already write held-out next states of the SAME game?"

Same protocol as the JEPA probe, so the two features are comparable:
  - groups = (env, game) in the buffer; the same split rule (split_for_game below is
    copied unchanged from probe_fidelity.py): n = 32 training transitions and up to
    64 OTHER transitions of the game for scoring
  - 5 random draws (seeds 42..46), averaged; std reported
  - a short fixed training budget: 20 steps (the SFT floor from the pilot), against
    JEPA's 50 probe steps

Score = word F1 and exact match on the held-out transitions (higher = easier game;
note the JEPA probe is a distance, where lower = easier).

The base model is loaded once; the LoRA weights are reset to the same starting values
before every draw. Results are written after every game, and finished games are skipped
on a rerun, so a crash only loses the game in progress.

    # GPU box, inside .venv-sft (all three buffers, about 3 h)
    python scripts/sft_probe.py --buffer data/buffers/alfworld_games.jsonl \
        data/buffers/sciworld_games.jsonl data/buffers/bytesized32_games.jsonl

Output: results/sft/sft_probe_scores.csv
        env, game, sft_probe_f1, sft_probe_f1_std, sft_probe_exact, sft_probe_exact_std,
        sft_probe_n_train, sft_probe_n_val, sft_probe_seeds, sft_probe_steps
"""

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from jepa.compute import track  # noqa: E402
from sft.data import fidelity, read_jsonl  # noqa: E402
from sft.model import (generate, load_lora_model, load_tokenizer, tokenize_example,  # noqa: E402
                       tokenize_prompt, train_lora)

MIN_VAL = 4        # same constants as scripts/probe_fidelity.py
MIN_TRAIN = 8
COLS = ["env", "game", "sft_probe_f1", "sft_probe_f1_std", "sft_probe_exact", "sft_probe_exact_std",
        "sft_probe_n_train", "sft_probe_n_val", "sft_probe_seeds", "sft_probe_steps"]


def split_for_game(idx, n, max_val, rng):
    """Return (train_idx, val_idx, note) for one game's transition indices.
    (Copied unchanged from scripts/probe_fidelity.py so both probes see the same split.)"""
    idx = idx[:]
    rng.shuffle(idx)
    if len(idx) >= n + MIN_VAL:
        return idx[:n], idx[n: n + max_val], ""
    n_val = max(MIN_VAL, len(idx) // 5)
    n_train = len(idx) - n_val
    if n_train < MIN_TRAIN:
        return None, None, f"skipped: only {len(idx)} transitions"
    return idx[:n_train], idx[n_train:], f"short: trained on {n_train} not {n}"


def load_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def read_scores(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {(r["env"], r["game"]): r for r in csv.DictReader(f)}


def write_scores(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", nargs="+", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-val", type=int, default=64)
    p.add_argument("--max-new", type=int, default=96, help="answer length cap while scoring")
    p.add_argument("--out", default="results/sft/sft_probe_scores.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="redo games already in the output file")
    p.add_argument("--limit-games", type=int, default=None, help="code check: first N games only")
    args = p.parse_args()

    cfg_all = load_config(args.config)
    cfg = dict(cfg_all.get("sft", {}))
    base_seed = int(cfg_all.get("seed", 42))
    model_name = args.model or cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_len = int(cfg.get("max_seq_len", 1024))
    gen_bs = int(cfg.get("gen_batch", 32))
    out = Path(args.out)
    scores = read_scores(out)

    tok = load_tokenizer(model_name)
    model = load_lora_model(model_name, cfg, args.device, seed=base_seed)
    start = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    print(f"[SFT probe] model={model_name}  n={args.n}  steps={args.steps}  seeds={args.seeds}  "
          f"max_val={args.max_val}  → {out}")

    for buf in args.buffer:
        items = read_jsonl(buf)
        groups = defaultdict(list)
        for i, t in enumerate(items):
            groups[(t.get("env", "unknown"), t.get("game", "unknown"))].append(i)
        todo = sorted(groups.items())[: args.limit_games]
        print(f"\n[SFT probe] {buf}: {len(items)} transitions, {len(groups)} games")
        for (env, game), idx in todo:
            if (env, game) in scores and not args.force:
                print(f"  {env:>12} / {game[:40]:<40} done — skipping")
                continue
            t0 = time.time()
            f1s, ems, n_tr, n_va, note = [], [], 0, 0, ""
            with track("probe", arm="F", env=env, game=game, n=args.n, seeds=args.seeds):
                for s in range(args.seeds):
                    seed = base_seed + s
                    tr, va, note = split_for_game(idx, args.n, args.max_val, random.Random(seed))
                    if tr is None:
                        break
                    with torch.no_grad():                         # fresh LoRA, same start
                        for k, v in model.named_parameters():
                            if k in start:
                                v.copy_(start[k])
                    ex = [tokenize_example(tok, items[i]["obs"], items[i]["action"],
                                           items[i]["obs_next"], max_len) for i in tr]
                    train_lora(model, ex, args.steps, cfg, args.device, seed=seed, log_every=10**9)
                    prompts = [tokenize_prompt(tok, items[i]["obs"], items[i]["action"], max_len,
                                               args.max_new) for i in va]
                    preds = generate(model, tok, prompts, args.max_new, gen_bs, args.device)
                    r = fidelity(preds, [items[i]["obs_next"] for i in va])
                    f1s.append(r["f1"])
                    ems.append(r["exact"])
                    n_tr, n_va = len(tr), len(va)
            if not f1s:
                print(f"  {env:>12} / {game[:40]:<40} {note}")
                continue
            scores[(env, game)] = {
                "env": env, "game": game,
                "sft_probe_f1": round(mean(f1s), 5), "sft_probe_f1_std": round(pstdev(f1s), 5),
                "sft_probe_exact": round(mean(ems), 5), "sft_probe_exact_std": round(pstdev(ems), 5),
                "sft_probe_n_train": n_tr, "sft_probe_n_val": n_va, "sft_probe_seeds": len(f1s),
                "sft_probe_steps": args.steps,
            }
            write_scores(out, scores)
            print(f"  {env:>12} / {game[:40]:<40} F1 {mean(f1s):.3f} ± {pstdev(f1s):.3f}  "
                  f"exact {mean(ems):.3f}  ({n_tr} train / {n_va} val, {time.time() - t0:.0f}s) {note}")

    print(f"\n[SFT probe] {len(scores)} games in {out}")


if __name__ == "__main__":
    main()
