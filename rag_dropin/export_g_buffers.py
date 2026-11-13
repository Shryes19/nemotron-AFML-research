#!/usr/bin/env python3
"""Export the exact training rows used by D/F for each G retrieval budget."""
import argparse, glob, json, sys
from pathlib import Path
import torch

def rows(path):
    with open(path,encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--set",required=True); p.add_argument("--ckpt-dir",required=True)
    p.add_argument("--buffer",required=True); p.add_argument("--out-dir",required=True); a=p.parse_args()
    source=rows(a.buffer); out=Path(a.out_dir).expanduser(); out.mkdir(parents=True,exist_ok=True)
    checkpoints=sorted(glob.glob(str(Path(a.ckpt_dir)/"budget_*.pt")))
    if not checkpoints: sys.exit("No budget_*.pt checkpoint found")
    for path in checkpoints:
        extra=torch.load(path,map_location="cpu",weights_only=False).get("extra",{}); indices=extra.get("train_rows")
        if indices is None: sys.exit(f"{path} has no train_rows")
        if max(indices)>=len(source): sys.exit(f"{path} references a row beyond {a.buffer}")
        budget=int(extra.get("budget",len(indices)))
        if len(indices)!=budget: sys.exit(f"{path}: {len(indices)} rows, expected budget {budget}")
        target=out/f"{a.set}_b{budget}.jsonl"
        with target.open("w",encoding="utf-8") as f:
            for index in indices:
                row=source[index]
                f.write(json.dumps({"obs":row["obs"],"action":row["action"],"obs_next":row["obs_next"],"row":index})+"\n")
        print(target)
if __name__=="__main__": main()
