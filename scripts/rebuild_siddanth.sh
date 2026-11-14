#!/usr/bin/env bash
# Rebuild everything in Siddanth's part from inputs, in order. Run from the repo root.
#   bash scripts/rebuild_siddanth.sh            # CPU steps only (seconds)
#   RECALL=1 bash scripts/rebuild_siddanth.sh   # also Feature 3 (needs the GPU and the encoder venv)
# Inputs it expects (not in git, see capability/README.md for how to get them):
#   data/buffers/alfworld_games.jsonl  data/buffers/sciworld_games.jsonl  data/buffers/bytesized32_games.jsonl
#   data/raw/bytesized32/test.jsonl
#   the compute logs: $SHRYES_LOG (default ~/nemotron-shryes/results/compute/compute_log.jsonl),
#   results/compute_saanvi_gpu.jsonl, results/compute_g_gpu.jsonl
set -euo pipefail
PY=${PY:-python3}
SHRYES_LOG=${SHRYES_LOG:-$HOME/nemotron-shryes/results/compute/compute_log.jsonl}
GPU="NVIDIA GeForce RTX 5070"

echo "== tests";                  $PY tests/test_capability_tagger.py
echo "== capability tags";        $PY scripts/tag_buffer.py --buffer data/buffers/alfworld_games.jsonl --out data/buffers/alfworld_games.tagged.jsonl
                                  $PY scripts/tag_buffer.py --buffer data/buffers/sciworld_games.jsonl --out data/buffers/sciworld_games.tagged.jsonl
echo "== ByteSized32-SP";         $PY scripts/setup_bytesized32.py --raw data/raw/bytesized32/test.jsonl --out data/buffers/bytesized32_games.jsonl > /dev/null
echo "== Feature 2";              $PY scripts/capability_mix.py --text-tags --buffers data/buffers/alfworld_games.tagged.jsonl data/buffers/sciworld_games.tagged.jsonl data/buffers/bytesized32_games.jsonl --out results/capability/capability_mix.csv > /dev/null
echo "== Feature 4";              $PY scripts/env_stats.py --buffers data/buffers/alfworld_games.jsonl data/buffers/sciworld_games.jsonl data/buffers/bytesized32_games.jsonl --out results/features/env_stats.csv > /dev/null
if [ "${RECALL:-0}" = "1" ]; then
  echo "== Feature 3 (GPU)";      $PY scripts/recall_feature.py --buffers data/buffers/alfworld_games.jsonl data/buffers/sciworld_games.jsonl data/buffers/bytesized32_games.jsonl --out results/features/recall_at5.csv
fi
echo "== cost table"
$PY scripts/compute_table.py --log "$SHRYES_LOG" results/compute_saanvi_gpu.jsonl results/compute_g_gpu.jsonl \
    --exclude "stage=nv_embed_encode,env=sciworld,n=3900" "stage=jepa_train,time=2026-11-12T18" \
              "stage=jepa_decoder_train" "stage=jepa_decoder_predict" \
    --http-gpu "$GPU" --shared-to D --out results/compute/cost_table.csv
echo "== compute crossover (RQ2)"; $PY scripts/compute_crossover.py --cost results/compute/cost_table.csv --summary results/utility/summary.csv --d-mode top1 --out-dir results/crossover > /dev/null
echo "== RQ1 by capability";       $PY scripts/capability_regret.py --budget 256 --out-dir results/capability_regret > /dev/null
echo "== RQ4 router bound";         $PY scripts/router_bound.py --out-dir results/router > /dev/null
REC=""; [ -f results/features/recall_at5.csv ] && REC="--recall results/features/recall_at5.csv"
echo "== feature matrix";         $PY scripts/build_feature_matrix.py --mix results/capability/capability_mix.csv --stats results/features/env_stats.csv --probe results/jepa_phase1/probe_scores.csv --labels results/utility/labels_Dtop1.csv $REC --out results/features/feature_matrix.csv
echo "done: git status shows what changed"
