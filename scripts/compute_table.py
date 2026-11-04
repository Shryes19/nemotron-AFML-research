#!/usr/bin/env python3
"""
Summarise results/compute/compute_log.jsonl into the GPU-hour cost table (arm x env x n).

    python scripts/compute_table.py --log results/compute/compute_log.jsonl --out results/compute/cost_table.csv

GPU names are kept as a column: an RTX 5070 hour and a 4090 hour are not the same cost, so do not add
them without saying so.
"""
import argparse
import collections
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.environ.get("COMPUTE_LOG", "results/compute/compute_log.jsonl"),
                    help="defaults to $COMPUTE_LOG, the same file track() writes to")
    ap.add_argument("--out", default="results/compute/cost_table.csv")
    args = ap.parse_args()
    agg = collections.defaultdict(lambda: {"runs": 0, "wall_s": 0.0, "gpu_s": 0.0, "peak_mem_gb": 0.0})
    with open(args.log) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r.get("arm", ""), r["stage"], r.get("env", ""), str(r.get("n", "")), r.get("gpu_name", ""))
            a = agg[k]
            a["runs"] += 1
            a["wall_s"] += r["wall_s"]
            a["gpu_s"] += r["gpu_s"]
            a["peak_mem_gb"] = max(a["peak_mem_gb"], r.get("peak_mem_gb", 0.0))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "stage", "env", "n", "gpu_name", "runs", "wall_hours", "gpu_hours", "peak_mem_gb"])
        for (arm, stage, env, n, gpu), a in sorted(agg.items()):
            w.writerow([arm, stage, env, n, gpu, a["runs"], round(a["wall_s"] / 3600, 4),
                        round(a["gpu_s"] / 3600, 4), a["peak_mem_gb"]])
            print(f"{arm:7s} {stage:18s} {env:12s} n={n:>5s} {gpu[:22]:22s} gpu_h={a['gpu_s'] / 3600:8.4f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
