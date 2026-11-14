#!/usr/bin/env bash
# Everything left on the GPU, in one go (run from the repo root, inside tmux):
#   1) G's training rows per budget (= D/F's rows)              CPU, skipped if done
#   2) G stage 1: NV-Embed retrieval -> prompts + Recall@5        GPU, ~15-20 min
#   3) start vLLM (Qwen2.5-7B-Instruct-AWQ)
#   4) G stage 2: predictions, n=256 first, then 32 ... 1024      GPU, ~4 h (rough)
#   5) decision-utility scoring of G + labels + figures           GPU, ~1 h
#   6) stop vLLM, pack the results into ~/g_results.tgz
# Resumable: rerun the same command after a crash and finished steps are skipped.
#
#   bash rag_dropin/overnight.sh 2>&1 | tee overnight.log
set -euo pipefail

SID=${SID:-/home/siddanth_anil/nemotron-AFML-research}   # Siddhant's repo: JEPA checkpoints, buffers, D/F predictions
ENC_PY=${ENC_PY:-$SID/.venv-enc/bin/python}              # NV-Embed venv (transformers 4.42, bitsandbytes)
ENC_HF_HOME=${ENC_HF_HOME:-/home/siddanth_anil/.cache/huggingface}   # NV-Embed weights already there
TORCH_PY=${TORCH_PY:-$HOME/vllm-venv/bin/python}
WM_PY=${WM_PY:-$HOME/wm-venv/bin/python}
VLLM=${VLLM:-$HOME/vllm-venv/bin/vllm}
ROWS=${ROWS:-$HOME/g_buffers}
PROMPTS=${PROMPTS:-$HOME/g_prompts}
G_OUT=${G_OUT:-$HOME/g_candidates}
SETS="alfworld sciworld sciworld_seen bytesized32"
BUDGETS="256 32 64 128 512 1024"
MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export PATH=$PATH:/usr/lib/wsl/lib
export COMPUTE_LOG=${COMPUTE_LOG:-results/compute/compute_log_g.jsonl}
stamp() { echo "=== $(date '+%d %b %H:%M') $*"; }

VPID=""
cleanup() { [ -n "$VPID" ] && kill "$VPID" 2>/dev/null && wait "$VPID" 2>/dev/null; true; }
trap cleanup EXIT

stamp "start (branch $(git branch --show-current))"
while pgrep -f prepare_g_prompts.py > /dev/null; do   # an NV-Embed stage started earlier: let it finish
    stamp "waiting for a running prepare_g_prompts.py"; sleep 60
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ]; then
    echo "[Error] the GPU already has ${used} MiB in use (another vLLM or job?). Stop it first."; exit 1
fi

# 1) training rows
ckpt_of() { case $1 in
    alfworld)      echo "$SID/outputs/jepa/alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen" ;;
    sciworld)      echo "$SID/outputs/jepa/sciworld_games_steps2000_test-sciworld_heldout_buffer_sepact_adm_frozen" ;;
    sciworld_seen) echo "$SID/outputs/jepa/sciworld_games_steps2000_test-sciworld_seen_heldout_buffer_sepact_adm_frozen" ;;
    bytesized32)   echo "$SID/outputs/jepa/bytesized32_games_steps2000_sepact_adm_frozen" ;; esac; }
buffer_of() { case $1 in
    alfworld)               echo "$SID/data/buffers/alfworld_games.tagged.jsonl" ;;  # same rows; --check proves it
    sciworld|sciworld_seen) echo "$SID/data/buffers/sciworld_games.jsonl" ;;
    bytesized32)            echo "$SID/data/buffers/bytesized32_games.jsonl" ;; esac; }
stamp "1/6 training rows"
for set in $SETS; do
    [ -s "$ROWS/${set}_b1024.jsonl" ] && continue
    $TORCH_PY rag_dropin/export_g_buffers.py --set "$set" --ckpt-dir "$(ckpt_of "$set")" \
        --buffer "$(buffer_of "$set")" --out-dir "$ROWS" --check "$SID/outputs/jepa_candidates"
done

# 2) NV-Embed stage (vLLM is not running yet, so NV-Embed has the GPU)
stamp "2/6 NV-Embed retrieval -> prompts"
if [ ! -s "$PROMPTS/recall_at5.csv" ]; then
    HF_HOME=$ENC_HF_HOME HF_HUB_OFFLINE=1 $ENC_PY rag_dropin/prepare_g_prompts.py --rows "$ROWS" --out-dir "$PROMPTS"
fi

# 3) vLLM
stamp "3/6 starting vLLM (log: vllm_overnight.log)"
HF_HOME=$HOME/hf VLLM_USE_FLASHINFER_SAMPLER=0 $VLLM serve $MODEL --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --seed 0 > vllm_overnight.log 2>&1 &
VPID=$!
for i in $(seq 1 90); do
    curl -sf http://localhost:8000/v1/models > /dev/null && break
    kill -0 "$VPID" 2>/dev/null || { echo "[Error] vLLM exited; see vllm_overnight.log"; tail -20 vllm_overnight.log; exit 1; }
    sleep 10
done
curl -sf http://localhost:8000/v1/models > /dev/null || { echo "[Error] vLLM not up after 15 min"; exit 1; }
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=$MODEL
$WM_PY -m wm.llm_client --ping

# 4) G predictions
for n in $BUDGETS; do
    for set in $SETS; do
        stamp "4/6 G predictions $set n=$n"
        $WM_PY rag_dropin/generate_predictions.py --prompts "$PROMPTS/${set}_b${n}.jsonl" \
            --states "data/eval/${set}_branched_states.jsonl" --set "$set" --budget "$n" --outdir "$G_OUT"
    done
done

# 5) score G (D and F are already scored in results/utility and are skipped), labels, figures
stamp "5/6 decision utility: score G, summary, labels, figures"
PY=$WM_PY G_DIR=$G_OUT PRED_ROOT=$SID/outputs bash scripts/run_utility.sh

# 6) stop vLLM, pack
stamp "6/6 packing"
cleanup; VPID=""
cp "$PROMPTS/recall_at5.csv" results/utility/g_recall_at5.csv
TEST_LOG=$HOME/g_test/results/compute/compute_log_g.jsonl   # NV-Embed stage run before this script
if [ -s "$TEST_LOG" ] && ! grep -q rag_index_build "$COMPUTE_LOG"; then cat "$TEST_LOG" >> "$COMPUTE_LOG"; fi
tar czf "$HOME/g_results.tgz" results/utility results/figures "$COMPUTE_LOG" -C "$HOME" "$(basename "$G_OUT")"
stamp "ALL DONE -> $HOME/g_results.tgz"
