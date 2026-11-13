#!/usr/bin/env python3
"""Launch a G job under the repository's existing experiment tracker.

Set TRACK_IMPORT to the import path that exports `track`, e.g.
TRACK_IMPORT=src.tracking.  Keeping this adapter separate avoids guessing or
modifying the host repository's tracking API.
"""
import argparse
import importlib
import os
import subprocess

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("generator_args", nargs=argparse.REMAINDER,
                   help="arguments after -- are passed to generate_predictions.py")
    a = p.parse_args()
    module_name = os.environ.get("TRACK_IMPORT")
    if not module_name:
        raise SystemExit("Set TRACK_IMPORT to the module exporting track (for example src.tracking).")
    track = getattr(importlib.import_module(module_name), "track")
    with track(arm="G"):
        subprocess.run(["python", "generate_predictions.py", *a.generator_args], check=True)

if __name__ == "__main__":
    main()
