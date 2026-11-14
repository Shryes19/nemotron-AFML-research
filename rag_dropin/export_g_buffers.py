#!/usr/bin/env python3
"""Write G's retrieval buffer for every budget: exactly the rows arms D and F trained on.

train_jepa.py saves each budget's training rows (indices into the buffer) in the checkpoint as
"train_rows"; F trains on the same rows (scripts/export_train_rows.py). This writes them as
<out-dir>/<set>_b<n>.jsonl with obs / action / obs_next, the input generate_predictions.py needs.

    python rag_dropin/export_g_buffers.py --set alfworld \
        --ckpt-dir outputs/jepa/alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen \
        --buffer data/buffers/alfworld_games.tagged.jsonl --out-dir g_buffers \
        --check outputs/jepa_candidates

--check (optional, recommended) confirms the rows are right: every nearest state JEPA returned in
<check>/<set>_b<n>.jsonl must be the obs_next of one of the exported rows. This also proves a
".tagged" copy of the buffer has the same rows in the same order as the file JEPA trained on.
Needs torch (CPU is enough).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_against_candidates(path, rows):
    known = {r["obs_next"] for r in rows}
    texts = [t["text"] for line in read_jsonl(path) for c in line["candidates"] for t in c["topk"]]
    found = sum(t in known for t in texts)
    return found, len(texts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--check", default=None, help="folder of JEPA candidate files (<set>_b<n>.jsonl)")
    args = p.parse_args()

    items = read_jsonl(args.buffer)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(glob.glob(str(Path(args.ckpt_dir) / "budget_*.pt")))
    if not ckpts:
        sys.exit(f"[Error] no budget_*.pt in {args.ckpt_dir}")
    for ck in ckpts:
        extra = torch.load(ck, map_location="cpu", weights_only=False).get("extra", {})
        idx = extra.get("train_rows")
        if idx is None:
            sys.exit(f"[Error] {ck} has no train_rows")
        trained_on = Path(extra.get("buffer", "")).name.replace(".tagged", "")
        if trained_on and trained_on != Path(args.buffer).name.replace(".tagged", ""):
            sys.exit(f"[Error] {ck} was trained on {extra['buffer']}, not {args.buffer}")
        if max(idx) >= len(items):
            sys.exit(f"[Error] {ck}: row {max(idx)} is past the end of {args.buffer} ({len(items)} rows)")
        n = int(extra.get("budget", len(idx)))
        rows = [{"obs": items[i]["obs"], "action": items[i]["action"], "obs_next": items[i]["obs_next"],
                 "row": int(i)} for i in idx]
        assert len(rows) == n, f"{ck}: {len(rows)} rows for budget {n}"
        out = out_dir / f"{args.set}_b{n}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        msg = f"[Rows] {out}: {n} rows"
        if args.check:
            found, total = check_against_candidates(Path(args.check) / f"{args.set}_b{n}.jsonl", rows)
            msg += f" | JEPA nearest states found in these rows: {found}/{total}"
            if found < total:
                sys.exit(msg + "\n[Error] rows do not match what JEPA used; do not run G on them")
        print(msg)


if __name__ == "__main__":
    main()
