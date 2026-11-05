#!/bin/bash
# Check collected data, write budgets, and gzip + checksum files for the team Drive.
#   bash scripts/package_data.sh             provisional budgets (expert/random only)
#   bash scripts/package_data.sh final       final budgets (after LLM rows are collected)
set -e
cd "$(dirname "$0")/.."
PY=~/w2w/uv_alfworld/bin/python
FLAG="--provisional"; [ "$1" = "final" ] && FLAG=""

for env in alfworld sciworld; do
  $PY -m wm.stats data/raw/${env}_train.jsonl data/raw/${env}_heldout.jsonl
  $PY -m wm.budgets --env $env --data data/raw/${env}_train.jsonl $FLAG
done

mkdir -p data/upload
for f in data/raw/*.jsonl; do gzip -c "$f" > data/upload/$(basename "$f").gz; done
(cd data/upload && sha256sum *.jsonl.gz > SHA256SUMS)
ls -la data/upload
echo "Upload everything in data/upload/ to the team Drive folder."
