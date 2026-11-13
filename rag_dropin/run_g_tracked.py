#!/usr/bin/env python3
"""Compatibility launcher; generation now tracks itself via wm.utility.score.track."""
import argparse, subprocess

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("generator_args", nargs=argparse.REMAINDER,
                   help="arguments after -- are passed to generate_predictions.py")
    a = p.parse_args()
    subprocess.run(["python", "rag_dropin/generate_predictions.py", *a.generator_args], check=True)

if __name__ == "__main__":
    main()
