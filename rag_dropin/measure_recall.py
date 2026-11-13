#!/usr/bin/env python3
"""Create the required per-game Recall@5 CSV for one budget."""
import argparse
import csv
from collections import defaultdict

from rag_arm import HierarchicalRetriever, first, read_jsonl, recall_at5

UNIT_KEY = {"alfworld": "task_id", "sciworld": "task_type", "sciworld_seen": "task_type", "bytesized32": "game"}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=UNIT_KEY)
    p.add_argument("--buffer", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--budget", required=True, type=int)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    retriever = HierarchicalRetriever(read_jsonl(a.buffer)[:a.budget])
    groups = defaultdict(list)
    for row in read_jsonl(a.queries): groups[first(row, UNIT_KEY[a.env])].append(row)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "game", "recall_at5"])
        writer.writeheader()
        for game, rows in sorted(groups.items()):
            writer.writerow({"env": a.env, "game": game, "recall_at5": recall_at5(rows, retriever)})

if __name__ == "__main__": main()
