#!/usr/bin/env python3
"""
Summarise compute logs (written by capability/compute_log.track) into the GPU-hour cost table.

    python scripts/compute_table.py \
        --log ~/nemotron-shryes/results/compute/compute_log.jsonl results/compute_saanvi_gpu.jsonl \
        --exclude "stage=nv_embed_encode,env=sciworld,time=2026-11-08" \
        --http-gpu "NVIDIA GeForce RTX 5070" \
        --out results/compute/cost_table.csv

Every logged step is put in one of three categories:
  arm       cost of an arm (G / F / D): training, prediction, encoding, scoring
  feature   cost of an ex-ante feature (stage "probe": JEPA probe = Feature 1, SFT probe = ablation)
  excluded  pilot runs (pilot: true), rows matching --exclude, and older duplicates of the same run
Only "arm" belongs in the compute-vs-regret comparison (RQ2); "feature" is reported next to the RQ3
predictor; "excluded" is listed so nothing disappears silently.
Duplicates: with --dedupe, if the same configuration (all fields except timings, host and time) was logged
more than once, only the latest run counts. Off by default: the logs do not record every setting (e.g. JEPA
loss, scoring mode), so identical-looking rows can be different real runs.
--http-gpu: the utility scorer calls vLLM over HTTP, so its rows say gpu_name "cpu" with gpu_s 0 although
the time was spent on the GPU; this counts their wall time as GPU time on the named GPU.
--shared-to D: assign arm "shared" (NV-Embed encoding) to one arm, once it is known who uses it.
GPU names stay a column: an RTX 5070 hour and a 4090 hour are not the same cost.
"""
import argparse
import collections
import csv
import json
import os

TIMING = {"wall_s", "gpu_s", "peak_mem_gb", "host", "time", "gpu_name"}
HTTP_STAGES = {"utility_scoring"}


def load(paths):
    rows = []
    for p in paths:
        with open(os.path.expanduser(p)) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_src"] = os.path.basename(p)
                    rows.append(r)
    return rows


def parse_excludes(specs):
    return [dict(kv.split("=", 1) for kv in spec.split(",")) for spec in specs or []]


def matches(r, rule):
    # time matches by prefix (a date like 2026-11-08 selects that day); other fields exactly
    return all(str(r.get(k, "")).startswith(v) if k == "time" else str(r.get(k, "")) == v
               for k, v in rule.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+",
                    default=[os.environ.get("COMPUTE_LOG", "results/compute/compute_log.jsonl")],
                    help="one or more logs; defaults to $COMPUTE_LOG, the file track() writes to")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help='rules like "stage=nv_embed_encode,env=sciworld,time=2026-11-08"')
    ap.add_argument("--http-gpu", default="",
                    help="GPU name to use for utility_scoring rows logged as cpu (vLLM over HTTP)")
    ap.add_argument("--shared-to", default="", help="assign arm 'shared' to this arm (e.g. D)")
    ap.add_argument("--dedupe", action="store_true",
                    help="drop older runs of identical logged configurations (only if the log records "
                         "every setting that differs between runs, e.g. loss and scoring mode)")
    ap.add_argument("--out", default="results/compute/cost_table.csv")
    args = ap.parse_args()

    rows, rules = load(args.log), parse_excludes(args.exclude)
    for r in rows:
        if args.http_gpu and r.get("stage") in HTTP_STAGES and r.get("gpu_name") == "cpu":
            r["gpu_name"], r["gpu_s"] = args.http_gpu, r["wall_s"]
        if args.shared_to and r.get("arm") == "shared":
            r["arm"] = args.shared_to
        if r.get("pilot"):
            r["_cat"], r["_why"] = "excluded", "pilot"
        elif any(matches(r, rule) for rule in rules):
            r["_cat"], r["_why"] = "excluded", "--exclude"
        elif r.get("stage") == "probe":
            r["_cat"], r["_why"] = "feature", ""
        else:
            r["_cat"], r["_why"] = "arm", ""

    latest = {}
    for r in (rows if args.dedupe else []):
        if r["_cat"] == "excluded":
            continue
        key = tuple(sorted((k, str(v)) for k, v in r.items() if k not in TIMING and not k.startswith("_")))
        if key in latest and latest[key].get("time", "") > r.get("time", ""):
            r["_cat"], r["_why"] = "excluded", "older duplicate"
        else:
            if key in latest:
                latest[key]["_cat"], latest[key]["_why"] = "excluded", "older duplicate"
            latest[key] = r

    agg = collections.defaultdict(lambda: {"runs": 0, "wall_s": 0.0, "gpu_s": 0.0, "peak_mem_gb": 0.0})
    for r in rows:
        k = (r["_cat"], r["_why"], r.get("arm", ""), r["stage"], r.get("env", ""), str(r.get("n", "")),
             r.get("gpu_name", ""))
        a = agg[k]
        a["runs"] += 1
        a["wall_s"] += r["wall_s"]
        a["gpu_s"] += r["gpu_s"]
        a["peak_mem_gb"] = max(a["peak_mem_gb"], r.get("peak_mem_gb", 0.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    order = {"arm": 0, "feature": 1, "excluded": 2}

    def sort_key(k):
        n = k[5]
        return (order[k[0]], k[2], k[3], k[4], int(n) if n.isdigit() else 10 ** 9, n)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "reason", "arm", "stage", "env", "n", "gpu_name", "runs", "wall_hours",
                    "gpu_hours", "peak_mem_gb"])
        for k in sorted(agg, key=sort_key):
            a = agg[k]
            w.writerow([*k[:6], k[6], a["runs"], round(a["wall_s"] / 3600, 4), round(a["gpu_s"] / 3600, 4),
                        a["peak_mem_gb"]])

    tot = collections.defaultdict(float)
    for k, a in agg.items():
        tot[(k[0], k[2], k[4], k[6])] += a["gpu_s"] / 3600
    print(f"{'category':9s} {'arm':7s} {'env':12s} {'gpu':24s} gpu_hours")
    for (cat, arm, env, gpu), h in sorted(tot.items(), key=lambda x: (order[x[0][0]], x[0][1:])):
        print(f"{cat:9s} {arm:7s} {env:12s} {gpu[:24]:24s} {h:8.3f}")
    n_ex = sum(1 for r in rows if r["_cat"] == "excluded")
    print(f"-> {args.out}   ({len(rows)} logged steps, {n_ex} excluded: see the 'excluded' rows)")


if __name__ == "__main__":
    main()
