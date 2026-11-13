#!/usr/bin/env bash
# Run G reproducibly in two GPU stages: NV-Embed prompts, then vLLM Qwen predictions.
set -euo pipefail
SID=${SID:?Set SID to the repository with data and outputs}
PY=${PY:-python}; ROWS=${ROWS:-$HOME/g_buffers}; PROMPTS=${PROMPTS:-$HOME/g_prompts}; G_OUT=${G_OUT:-$HOME/g_candidates}
SETS=${SETS:-"alfworld sciworld sciworld_seen bytesized32"}; BUDGETS=${BUDGETS:-"256 32 64 128 512 1024"}
ckpt_of() { case "$1" in
  alfworld) echo "$SID/outputs/jepa/alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen";;
  sciworld) echo "$SID/outputs/jepa/sciworld_games_steps2000_test-sciworld_heldout_buffer_sepact_adm_frozen";;
  sciworld_seen) echo "$SID/outputs/jepa/sciworld_games_steps2000_test-sciworld_seen_heldout_buffer_sepact_adm_frozen";;
  bytesized32) echo "$SID/outputs/jepa/bytesized32_games_steps2000_sepact_adm_frozen";; esac; }
buffer_of() { case "$1" in
  alfworld) echo "$SID/data/buffers/alfworld_games.tagged.jsonl";;
  sciworld|sciworld_seen) echo "$SID/data/buffers/sciworld_games.jsonl";;
  bytesized32) echo "$SID/data/buffers/bytesized32_games.jsonl";; esac; }
for set in $SETS; do
  [ -s "$ROWS/${set}_b1024.jsonl" ] || $PY rag_dropin/export_g_buffers.py --set "$set" --ckpt-dir "$(ckpt_of "$set")" --buffer "$(buffer_of "$set")" --out-dir "$ROWS"
done
for set in $SETS; do for n in $BUDGETS; do
  $PY rag_dropin/prepare_g_prompts.py --buffer "$ROWS/${set}_b${n}.jsonl" --states "data/eval/${set}_branched_states.jsonl" --set "$set" --budget "$n" --outdir "$PROMPTS"
done; done
echo "NV-Embed stage complete. Unload NV-Embed, start vLLM, then rerun with GENERATE=1."
if [ "${GENERATE:-0}" = 1 ]; then
  $PY -m wm.llm_client --ping
  for set in $SETS; do for n in $BUDGETS; do
    $PY rag_dropin/generate_predictions.py --prompt-cache "$PROMPTS/${set}_b${n}.jsonl" --states "data/eval/${set}_branched_states.jsonl" --set "$set" --budget "$n" --outdir "$G_OUT"
  done; done
fi
