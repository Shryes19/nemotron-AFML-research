"""
scripts/export_train_rows.py
────────────────────────────
Copy the training rows of every JEPA checkpoint (saved as "train_rows" by
train_jepa.py) into a small JSON file, so the SFT arm (F) trains on EXACTLY the
same rows as JEPA (arm D) at every budget, without loading 250 MB checkpoints.

For a run with a random held-out split (no --test-buffer, e.g. ByteSized32) it
also rebuilds the held-out rows (same seed and rule as train_jepa.py) so F's
fidelity is measured on the same held-out rows as D's.

    # on the GPU box, inside .venv-enc (CPU only, ~1 min)
    python scripts/export_train_rows.py \
        --ckpt-dir outputs/jepa/bytesized32_games_steps2000_sepact_adm_frozen \
        --buffer data/buffers/bytesized32_games.jsonl --out sft/rows/bytesized32.json

Output: {"buffer", "n_buffer_rows", "checkpoint_dir", "split", "seed",
         "budgets": {"32": [row, ...], ...}, "val_rows": [...] | null}
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.buffer, encoding="utf-8") as f:
        n_items = sum(1 for line in f if line.strip())
    res_path = Path(args.ckpt_dir) / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    seed = int(res.get("seed", 42))
    split = res.get("split", "")

    budgets = {}
    for ck in sorted(glob.glob(str(Path(args.ckpt_dir) / "budget_*.pt"))):
        extra = torch.load(ck, map_location="cpu", weights_only=False).get("extra", {})
        rows = extra.get("train_rows")
        if rows is None:
            sys.exit(f"[Error] {ck} has no train_rows — retrain it with the current train_jepa.py")
        if extra.get("buffer") and Path(extra["buffer"]).name != Path(args.buffer).name:
            sys.exit(f"[Error] {ck} was trained on {extra['buffer']}, not {args.buffer}")
        budgets[int(extra["budget"])] = [int(i) for i in rows]
    if not budgets:
        sys.exit(f"[Error] no budget_*.pt in {args.ckpt_dir}")

    # Checks: rows inside the buffer, and nested (the 64 set contains the 32 set, ...)
    bs = sorted(budgets)
    for b in bs:
        assert len(budgets[b]) == b and max(budgets[b]) < n_items, f"bad rows for budget {b}"
    for a, b in zip(bs, bs[1:]):
        assert budgets[b][:a] == budgets[a], f"budgets {a} and {b} are not nested"

    val_rows = None
    if split.startswith("random"):
        # train_jepa.py: rng = Random(seed); order = shuffle(range(N)); val = first int(N*frac)
        frac = float(split.split("%")[0].split()[-1]) / 100 if "%" in split else 0.1
        order = list(range(n_items))
        random.Random(seed).shuffle(order)
        n_val = max(1, int(n_items * frac))
        val_rows, pool = order[:n_val], order[n_val:]
        assert budgets[bs[-1]] == pool[:bs[-1]], "could not rebuild train_jepa.py's split"
        assert not set(val_rows) & set(budgets[bs[-1]])

    out = {"buffer": args.buffer, "n_buffer_rows": n_items, "checkpoint_dir": args.ckpt_dir,
           "split": split, "seed": seed, "budgets": {str(b): budgets[b] for b in bs},
           "val_rows": val_rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"[Rows] {args.out}: budgets {bs}, buffer {n_items} rows, split: {split or '?'}"
          + (f", {len(val_rows)} held-out rows rebuilt" if val_rows else ""))


if __name__ == "__main__":
    main()
