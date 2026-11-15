#!/usr/bin/env python3
"""Export the historical three-column Recall@5 contract from the rich CSV.

The G prompt-preparation artifact keeps budget and state-count provenance in
``results/utility/g_recall_at5.csv``.  Older consumers expect one fixed budget
with exactly ``env,game,recall_at5``.  This exporter deliberately produces a
budget-scoped compatibility file instead of discarding the richer artifact.
"""
import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/utility/g_recall_at5.csv")
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--out", default="results/utility/g_recall_at5_b256.csv")
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["budget"]) == args.budget]
    if not rows:
        raise SystemExit(f"No Recall@5 rows for budget {args.budget} in {args.input}")

    keys = [(row["env"], row["game"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"Duplicate env/game rows at budget {args.budget}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["env", "game", "recall_at5"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    print(f"{out}: {len(rows)} rows at budget {args.budget}")


if __name__ == "__main__":
    main()
