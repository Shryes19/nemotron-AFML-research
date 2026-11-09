"""
scripts/train_sft.py
────────────────────
SFT arm (F): train Qwen2.5-1.5B-Instruct + LoRA at every budget for one
environment, then write everything the team needs from each budget:

  1. fidelity (exact match + word F1) on the held-out buffers, next to JEPA's numbers
  2. predictions for the decision-utility branched states, in Saanvi's generic format:
       {"id", "arm": "F", "budget", "actions": [...5], "pred_next": [...5]}
     -> outputs/sft_candidates/<set>_b<budget>.jsonl

F trains on exactly JEPA's rows for each budget (sft/rows/<env>.json, made by
scripts/export_train_rows.py). See docs/SFT.md for the full guide.

Usage (GPU box, inside .venv-sft)
    # pilot: picks the training length on 2 held-out TRAINING games (never the test sets)
    python scripts/train_sft.py --env alfworld --pilot

    # full run, all budgets (resumes: finished budgets are skipped)
    python scripts/train_sft.py --env alfworld

    # quick CPU code check with a tiny local model
    python scripts/train_sft.py --env alfworld --model /tmp/tiny-qwen --budgets 32 --device cpu ...
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from jepa.compute import track  # noqa: E402
from sft.data import (check_rows, fidelity, holdout_games_split, load_rows,  # noqa: E402
                      pick_prediction_fields, read_jsonl)
from sft.model import (generate, load_lora_model, load_tokenizer, n_trainable,  # noqa: E402
                       steps_for, tokenize_example, tokenize_prompt, train_lora)

# Which files belong to each environment (same buffers and branched states as arm D).
ENVS = {
    "alfworld": {
        "buffer": "data/buffers/alfworld_games.jsonl",
        "rows": "sft/rows/alfworld.json",
        "tests": {"alfworld_heldout": "data/eval/alfworld_heldout_buffer.jsonl"},
        "branched": {"alfworld": "data/eval/alfworld_branched_states.jsonl"},
    },
    "sciworld": {
        "buffer": "data/buffers/sciworld_games.jsonl",
        "rows": "sft/rows/sciworld.json",
        "tests": {"sciworld_heldout": "data/eval/sciworld_heldout_buffer.jsonl",
                  "sciworld_seen_heldout": "data/eval/sciworld_seen_heldout_buffer.jsonl"},
        "branched": {"sciworld": "data/eval/sciworld_branched_states.jsonl",
                     "sciworld_seen": "data/eval/sciworld_seen_branched_states.jsonl"},
    },
    "bytesized32": {
        "buffer": "data/buffers/bytesized32_games.jsonl",
        "rows": "sft/rows/bytesized32.json",
        "tests": {},            # held-out = JEPA's random 10% (val_rows in the rows file)
        "branched": {"bytesized32": "data/eval/bytesized32_branched_states.jsonl"},
    },
}


def load_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=sorted(ENVS))
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", default=None, help="default: sft.model_name in config")
    p.add_argument("--budgets", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=float, default=None, help="override sft.epochs")
    p.add_argument("--min-steps", type=int, default=None, help="override sft.min_steps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="default: outputs/sft/<env>")
    p.add_argument("--pred-dir", default="outputs/sft_candidates")
    p.add_argument("--force", action="store_true", help="redo budgets that are already finished")
    p.add_argument("--no-predict", action="store_true", help="train + fidelity only")
    p.add_argument("--limit", type=int, default=None, help="code check: use only N test rows/states")
    # pilot
    p.add_argument("--pilot", action="store_true",
                   help="pick the training length: hold out 2 whole TRAINING games, train at "
                        "--budgets (default 32 1024) for --pilot-steps, score at several steps")
    p.add_argument("--pilot-steps", type=int, default=480)
    p.add_argument("--pilot-val", type=int, default=200, help="held-out rows scored in the pilot")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg_all = load_config(args.config)
    cfg = dict(cfg_all.get("sft", {}))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.min_steps is not None:
        cfg["min_steps"] = args.min_steps
    seed = args.seed if args.seed is not None else int(cfg_all.get("seed", 42))
    model_name = args.model or cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_len = int(cfg.get("max_seq_len", 1024))
    max_new = int(cfg.get("max_new_tokens", 192))
    gen_bs = int(cfg.get("gen_batch", 32))
    E = ENVS[args.env]

    items = read_jsonl(E["buffer"])
    tok = load_tokenizer(model_name)
    print(f"[SFT] env={args.env}  model={model_name}  buffer={E['buffer']} ({len(items)} rows)  "
          f"device={args.device}")

    def examples_for(idx):
        ex = [tokenize_example(tok, items[i]["obs"], items[i]["action"], items[i]["obs_next"], max_len)
              for i in idx]
        cut = sum(e["cut"] for e in ex)
        if cut:
            print(f"[SFT] {cut}/{len(ex)} training examples were longer than {max_len} tokens and were cut")
        return ex

    def prompts_for(rows):
        return [tokenize_prompt(tok, r["obs"], r["action"], max_len, max_new) for r in rows]

    if args.pilot:
        run_pilot(args, cfg, items, model_name, seed, examples_for, prompts_for, max_new, gen_bs, tok)
        return

    # ── Full run ───────────────────────────────────────────────────────────────
    rows = load_rows(E["rows"])
    check_rows(rows, len(items))
    budgets = sorted(args.budgets or rows["budgets"])
    out_dir = Path(args.out or f"outputs/sft/{args.env}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = {name: read_jsonl(path) for name, path in E["tests"].items()}
    if rows.get("val_rows"):
        tests["random10_heldout"] = [items[i] for i in rows["val_rows"]]
    if args.limit:
        tests = {k: v[: args.limit] for k, v in tests.items()}
    test_prompts = {k: prompts_for(v) for k, v in tests.items()}

    branched = {}
    for name, path in ({} if args.no_predict else E["branched"]).items():
        states = [s for s in (pick_prediction_fields(x) for x in read_jsonl(path)) if s]
        if args.limit:
            states = states[: args.limit]
        flat = [{"obs": s["obs"], "action": a} for s in states for a in s["actions"]]
        branched[name] = (states, prompts_for(flat))
    print(f"[SFT] budgets {budgets} | held-out sets "
          + ", ".join(f"{k} ({len(v)})" for k, v in tests.items())
          + (" | branched " + ", ".join(f"{k} ({len(v[0])} states)" for k, v in branched.items())
             if branched else ""))

    summary = {}
    for b in budgets:
        bdir = out_dir / f"budget_{b}"
        done = bdir / "done.json"
        if done.exists() and not args.force:
            print(f"\n[Budget n={b}] already finished — skipping (use --force to redo)")
            summary[b] = json.loads(done.read_text())
            continue
        steps = steps_for(b, cfg)
        print(f"\n[Budget n={b}] {steps} steps (epochs={cfg.get('epochs')}, "
              f"min_steps={cfg.get('min_steps')}, batch={cfg.get('batch_size')})")
        ex = examples_for(rows["budgets"][b])
        model = load_lora_model(model_name, cfg, args.device, seed=seed)
        print(f"[SFT] LoRA trainable params: {n_trainable(model) / 1e6:.1f}M")
        with track("sft_train", arm="F", env=args.env, game="*", n=b, steps=steps):
            hist = train_lora(model, ex, steps, cfg, args.device, seed=seed)
        model.save_pretrained(str(bdir / "adapter"))

        res = {"budget": b, "steps": steps, "train_seconds": hist["seconds"],
               "final_loss": hist["loss"][-1], "fidelity": {}}
        with track("sft_predict", arm="F", env=args.env, game="*", n=b):
            for name, rws in tests.items():
                t0 = time.time()
                preds = generate(model, tok, test_prompts[name], max_new, gen_bs, args.device)
                fid = fidelity(preds, [r["obs_next"] for r in rws])
                res["fidelity"][name] = fid
                write_jsonl(bdir / f"heldout_{name}.jsonl",
                            [{"action": r["action"], "true": r["obs_next"], "pred": pr}
                             for r, pr in zip(rws, preds)])
                print(f"    {name}: exact {fid['exact']:.3f}  F1 {fid['f1']:.3f}  "
                      f"(n={fid['n']}, {time.time() - t0:.0f}s)")
            for name, (states, prompts) in branched.items():
                t0 = time.time()
                preds = generate(model, tok, prompts, max_new, gen_bs, args.device)
                out, k = [], 0
                for s in states:
                    m = len(s["actions"])
                    out.append({"id": s["id"], "arm": "F", "budget": b, "actions": s["actions"],
                                "pred_next": preds[k:k + m]})
                    k += m
                path = Path(args.pred_dir) / f"{name}_b{b}.jsonl"
                write_jsonl(path, out)
                print(f"    {name}: {len(out)} states → {path} ({time.time() - t0:.0f}s)")
        res["history"] = hist
        done.write_text(json.dumps(res, indent=2))
        summary[b] = res
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    with open(out_dir / "results.json", "w") as f:
        json.dump({"env": args.env, "model": model_name, "sft_config": cfg, "seed": seed,
                   "budgets": {str(k): {kk: vv for kk, vv in v.items() if kk != "history"}
                               for k, v in summary.items()}}, f, indent=2)
    names = sorted({n for v in summary.values() for n in v["fidelity"]})
    print("\n" + "=" * 72)
    print(f"  SFT (arm F) summary — {args.env}   exact match / word F1 on held-out sets")
    print("=" * 72)
    print(f"  {'n':>5} {'steps':>6} " + " ".join(f"{n[:22]:>24}" for n in names))
    for b in sorted(summary):
        f_ = summary[b]["fidelity"]
        cells = [f"{f_[n]['exact']:.3f} / {f_[n]['f1']:.3f}" if n in f_ else "-" for n in names]
        print(f"  {b:>5} {summary[b]['steps']:>6} " + " ".join(f"{c:>24}" for c in cells))
    print(f"\nAll outputs in {out_dir}/ and {args.pred_dir}/")


def run_pilot(args, cfg, items, model_name, seed, examples_for, prompts_for, max_new, gen_bs, tok):
    """Train at a small and a large budget on rows from 10 training games, score on rows
    from 2 other training games at several step counts. Test buffers are never touched."""
    val_idx, pool, held = holdout_games_split(items, 2, seed)
    rng = random.Random(seed)
    if len(val_idx) > args.pilot_val:
        val_idx = sorted(rng.sample(val_idx, args.pilot_val))
    val = [items[i] for i in val_idx]
    if args.limit:
        val = val[: args.limit]
    vprompts = prompts_for(val)
    truths = [r["obs_next"] for r in val]
    budgets = sorted(args.budgets or [32, 1024])
    marks = [s for s in (10, 20, 40, 80, 160, 320, 480, 640, 960) if s <= args.pilot_steps]
    if args.pilot_steps not in marks:
        marks.append(args.pilot_steps)
    print(f"[Pilot] held-out games {held} ({len(val)} rows scored); budgets {budgets}; "
          f"{args.pilot_steps} steps; scored at steps {marks}")
    out_dir = Path(args.out or f"outputs/sft/pilot_{args.env}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"held_out_games": held, "n_val": len(val), "budgets": {}}

    for b in budgets:
        if b > len(pool):
            print(f"[Pilot] n={b} skipped — only {len(pool)} pool rows")
            continue
        ex = examples_for(pool[:b])
        model = load_lora_model(model_name, cfg, args.device, seed=seed)

        def score(step):
            preds = generate(model, tok, vprompts, max_new, gen_bs, args.device)
            r = fidelity(preds, truths)
            print(f"    [score] n={b} step {step:>4}: exact {r['exact']:.3f}  F1 {r['f1']:.3f}")
            return r

        print(f"\n[Pilot n={b}] ({b / int(cfg.get('batch_size', 16)):.0f} steps per epoch)")
        model.eval()
        zero = score(0)
        zero["step"] = 0
        with track("sft_train", arm="F", env=args.env, game="*", n=b, steps=args.pilot_steps,
                   pilot=True):
            hist = train_lora(model, ex, args.pilot_steps, cfg, args.device, seed=seed,
                              eval_fn=score, eval_at=marks)
        result["budgets"][b] = {"evals": [zero] + hist["evals"], "loss": hist["loss"],
                                "seconds": hist["seconds"]}
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    (out_dir / "pilot.json").write_text(json.dumps(result, indent=2))
    bs = int(cfg.get("batch_size", 16))
    print("\n" + "=" * 64)
    print(f"  Pilot — {args.env}, held-out training games {held}  (exact / F1)")
    print("=" * 64)
    for b, r in result["budgets"].items():
        print(f"  n={b}  ({b / bs:.1f} steps per epoch)")
        for e in r["evals"]:
            ep = e["step"] * bs / b
            print(f"    step {e['step']:>4}  (~{ep:6.1f} epochs)   exact {e['exact']:.3f}   F1 {e['f1']:.3f}")
    print(f"\nSaved {out_dir / 'pilot.json'}")


if __name__ == "__main__":
    main()
