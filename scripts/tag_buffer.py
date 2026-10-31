#!/usr/bin/env python3
"""
Fill the capability tags of a transition buffer.

    python scripts/tag_buffer.py --buffer data/buffers/alfworld_games.jsonl \
                                 --out data/buffers/alfworld_games.tagged.jsonl

Accepts .jsonl or .jsonl.gz, in either the JEPA buffer format (obs/action/obs_next/obs_next_full) or the
exporter format (o_t/a_t/o_next). Writes the same records with:
    cap_tag     single primary tag (string)  - what train_jepa.py / probe_fidelity.py read
    cap_tags    all tags (list)
    capability_tag  same list, for the exporter format's field name
Records that already carry ground-truth tags (ByteSized32) keep them; the text tagger's output is stored
next to them as text_cap_tag / text_cap_tags.
"""
import argparse
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from capability.tagger import TAGS, tag_record  # noqa: E402


def _open(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode.replace("t", "") or "r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-ground-truth", action="store_true", default=True,
                    help="do not overwrite cap_tags of env=bytesized32 records (ground truth from JSON state)")
    args = ap.parse_args()

    counts, n = collections.Counter(), 0
    with _open(args.buffer) as fin, _open(args.out, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = tag_record(rec)
            if rec.get("env") == "bytesized32" and args.keep_ground_truth and rec.get("cap_tags") is not None:
                rec["text_cap_tag"], rec["text_cap_tags"] = t["primary"], t["tags"]
            else:
                rec["cap_tag"], rec["cap_tags"] = t["primary"], t["tags"]
                if "capability_tag" in rec:
                    rec["capability_tag"] = t["tags"]
            counts[rec["cap_tag"]] += 1
            n += 1
            fout.write(json.dumps(rec) + "\n")

    print(f"tagged {n} transitions -> {args.out}")
    for tag in ["none"] + TAGS:
        print(f"  {tag:22s} {counts[tag]:6d}  {counts[tag] / max(n, 1):6.1%}  (primary)")


if __name__ == "__main__":
    main()
