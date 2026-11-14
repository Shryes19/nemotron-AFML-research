#!/usr/bin/env python3
"""Run Recall@5 at all required budgets and plot Recall@5 versus n."""
import argparse
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import csv

BUDGETS = (32, 64, 128, 256, 512, 1024)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--env", required=True); p.add_argument("--buffer", required=True); p.add_argument("--queries", required=True); p.add_argument("--outdir", default="metrics"); a=p.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True); means=[]
    for n in BUDGETS:
        csv_path=out/f"{a.env}_recall_b{n}.csv"
        subprocess.run(["python", "measure_recall.py", "--env",a.env,"--buffer",a.buffer,"--queries",a.queries,"--budget",str(n),"--out",str(csv_path)], check=True)
        with csv_path.open() as f: means.append(sum(float(r["recall_at5"]) for r in csv.DictReader(f)))
        with csv_path.open() as f: count=sum(1 for _ in csv.DictReader(f))
        means[-1] /= max(count,1)
    plt.plot(BUDGETS,means,marker="o",label=a.env); plt.xscale("log",base=2); plt.ylim(0,1); plt.xlabel("Buffer size n"); plt.ylabel("Recall@5"); plt.legend(); plt.tight_layout(); plt.savefig(out/f"{a.env}_recall_vs_n.png",dpi=200)
if __name__=="__main__": main()
