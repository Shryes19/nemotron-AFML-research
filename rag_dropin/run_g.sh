#!/usr/bin/env bash
# Arm G on the GPU box: export each budget's training rows (the rows D and F used), then generate
# G's predictions for all four branched sets, n=256 first. Resumable: finished files are skipped.
# Needs the vLLM server running (LLM_BASE_URL / LLM_MODEL set). Run from the repo root:
#
#   SID=/home/siddanth_anil/nemotron-AFML-research TORCH_PY=~/vllm-venv/bin/python G_PY=~/g-venv/bin/python \
#       bash rag_dropin/run_g.sh
#
# Then score G with:  G_DIR=$G_OUT PRED_ROOT=$SID/outputs bash scripts/run_utility.sh
set -euo pipefail

SID=${SID:?set SID to the repo folder that holds outputs/jepa and data/buffers}
TORCH_PY=${TORCH_PY:-python}      # any python with torch (reads the JEPA checkpoints)
G_PY=${G_PY:-python}              # python with sentence-transformers + requests
ROWS=${ROWS:-$HOME/g_buffers}
G_OUT=${G_OUT:-$HOME/g_candidates}
SETS=${SETS:-"alfworld sciworld sciworld_seen bytesized32"}
BUDGETS=${BUDGETS:-"256 32 64 128 512 1024"}
WORKERS=${WORKERS:-16}

ckpt_of() {
    case $1 in
        alfworld)      echo "$SID/outputs/jepa/alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen" ;;
        sciworld)      echo "$SID/outputs/jepa/sciworld_games_steps2000_test-sciworld_heldout_buffer_sepact_adm_frozen" ;;
        sciworld_seen) echo "$SID/outputs/jepa/sciworld_games_steps2000_test-sciworld_seen_heldout_buffer_sepact_adm_frozen" ;;
        bytesized32)   echo "$SID/outputs/jepa/bytesized32_games_steps2000_sepact_adm_frozen" ;;
    esac
}
buffer_of() {
    case $1 in
        alfworld)               echo "$SID/data/buffers/alfworld_games.tagged.jsonl" ;;  # same rows; --check proves it
        sciworld|sciworld_seen) echo "$SID/data/buffers/sciworld_games.jsonl" ;;
        bytesized32)            echo "$SID/data/buffers/bytesized32_games.jsonl" ;;
    esac
}

# 1) training rows per budget (CPU, a few minutes)
for set in $SETS; do
    [ -s "$ROWS/${set}_b1024.jsonl" ] && continue
    $TORCH_PY rag_dropin/export_g_buffers.py --set "$set" --ckpt-dir "$(ckpt_of "$set")" \
        --buffer "$(buffer_of "$set")" --out-dir "$ROWS" --check "$SID/outputs/jepa_candidates"
done

# 2) predictions, budget by budget (256 first: the labels need it)
$G_PY -m wm.llm_client --ping
for n in $BUDGETS; do
    for set in $SETS; do
        echo "== G $set n=$n  $(date +%H:%M)"
        $G_PY rag_dropin/generate_predictions.py --buffer "$ROWS/${set}_b${n}.jsonl" \
            --states "data/eval/${set}_branched_states.jsonl" --set "$set" --budget "$n" \
            --outdir "$G_OUT" --workers "$WORKERS"
    done
done
echo "done: $G_OUT"
