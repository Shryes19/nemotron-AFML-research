#!/usr/bin/env bash
# Decision-utility run for all branched sets: dry run, true values, score every arm file, summary, labels.
# Resumable: true values append to their cache, scored files that already exist are skipped.
#
#   PRED_ROOT=~/nemotron-AFML-research/outputs bash scripts/run_utility.sh            # full run (needs vLLM)
#   PRED_ROOT=... DRY=1 bash scripts/run_utility.sh                                     # CPU only, overlap judge
#
# Prediction files: $PRED_ROOT/jepa_candidates/<set>_b<n>.jsonl (arm D, 3 modes),
# $PRED_ROOT/sft_candidates/<set>_b<n>.jsonl (arm F) and, once they exist,
# ${G_DIR:-$PRED_ROOT/generic_candidates}/<set>_b<n>.jsonl (arm G). Missing arms are skipped.
set -euo pipefail

PRED_ROOT=${PRED_ROOT:?set PRED_ROOT to the folder holding jepa_candidates/ and sft_candidates/}
G_DIR=${G_DIR:-$PRED_ROOT/generic_candidates}
WORKERS=${WORKERS:-32}
SETS=${SETS:-"alfworld sciworld sciworld_seen bytesized32"}
PY=${PY:-python}

if [ "${DRY:-0}" = 1 ]; then
    JUDGE=overlap; OUT=${OUT:-/tmp/utility_dry}
    export WM_NO_COMPUTE_LOG=1          # dry runs stay out of the GPU-hour log
else
    JUDGE=llm; OUT=${OUT:-results/utility}
    $PY -m wm.llm_client --ping
fi
mkdir -p "$OUT/scored"

states_of() { echo "data/eval/$1_branched_states.jsonl"; }

score() {   # set arm format mode pred_file
    local set=$1 arm=$2 fmt=$3 mode=$4 pred=$5 n
    n=$(basename "$pred" .jsonl | sed 's/.*_b//')
    local out="$OUT/scored/${set}_${arm}_${mode}_b${n}.jsonl"
    [ -s "$out" ] && return 0
    echo "== $set $arm $mode n=$n"
    $PY -m wm.utility.score score --states "$(states_of "$set")" --true-values "$OUT/${set}_true_values.jsonl" \
        --pred "$pred" --format "$fmt" --mode "$mode" --arm "$arm" --judge "$JUDGE" --workers "$WORKERS" \
        --out "$out.tmp" && mv "$out.tmp" "$out"
}

preds() {   # dir set -> prediction files for that set: n=256 first (labels need it), then the rest
    ls "$1/$2"_b*.jsonl 2>/dev/null | sed 's/.*_b\([0-9]*\)\.jsonl/\1 &/' \
        | awk '{print ($1 == 256 ? 0 : $1), $2}' | sort -n | cut -d' ' -f2 || true
}

# 1) true values (one judge pass per set)
for set in $SETS; do
    $PY -m wm.utility.score true-values --states "$(states_of "$set")" \
        --out "$OUT/${set}_true_values.jsonl" --judge "$JUDGE" --workers "$WORKERS"
done

# 2) cheapest first: D native (no judge), then F, G, D top1, D topk (longest prompts)
for set in $SETS; do for f in $(preds "$PRED_ROOT/jepa_candidates" "$set"); do score "$set" D jepa native "$f"; done; done
for set in $SETS; do for f in $(preds "$PRED_ROOT/sft_candidates" "$set"); do score "$set" F generic top1 "$f"; done; done
for set in $SETS; do for f in $(preds "$G_DIR" "$set"); do score "$set" G generic top1 "$f"; done; done
for set in $SETS; do for f in $(preds "$PRED_ROOT/jepa_candidates" "$set"); do score "$set" D jepa top1 "$f"; done; done
for set in $SETS; do for f in $(preds "$PRED_ROOT/jepa_candidates" "$set"); do score "$set" D jepa topk "$f"; done; done

# 3) tables
STATES=$(for s in $SETS; do states_of "$s"; done)
TV=$(for s in $SETS; do echo "$OUT/${s}_true_values.jsonl"; done)
$PY -m wm.utility.score summarize --states $STATES --true-values $TV \
    --scored "$OUT/scored/*.jsonl" --out-dir "$OUT"

# 4) labels at n=256 (families), one file per D mode; the team picks which D mode counts
for dmode in top1 topk native; do
    $PY -m wm.utility.labels --scored "$OUT/scored/*.jsonl" --budget 256 --arms G,F,D --key family \
        --arm-mode D=$dmode --out "$OUT/labels_D${dmode}.csv"
done
echo "done: $OUT/summary.csv, fidelity_vs_utility.csv, judge_check.csv, labels_D*.csv"
