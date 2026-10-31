#!/bin/bash
# Phase 1 data collection for both envs (train + held-out), run in parallel.
#   bash scripts/collect_all.sh          expert + random only (no GPU needed)
#   bash scripts/collect_all.sh llm      adds the LLM rows (needs LLM_BASE_URL in .env)
# Safe to rerun: each collector resumes where it stopped. Run detached so SSH drops don't kill it:
#   nohup bash scripts/collect_all.sh > logs/collect_all.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw logs
ALF=~/w2w/uv_alfworld/bin/python
SCI=~/w2w/uv_sciworld/bin/python

if [ "$1" = "llm" ]; then
  $ALF -m wm.llm_client --ping   # fail fast if the GPU server is unreachable
  TRAIN="expert=800,random=480,llm=320"
  HELD="expert=150,random=90,llm=60"
else
  TRAIN="expert=800,random=480"
  HELD="expert=150,random=90"
fi
# fewer rows per episode on held-out so it covers more distinct tasks
OPTS="--seed 0"
CAP_train=10
CAP_heldout=4

run() {  # python env split target
  cap_var="CAP_$3"
  $1 -m wm.collect --env $2 --split $3 --target $4 $OPTS --max-rows-per-episode ${!cap_var}     --out data/raw/$2_$3.jsonl >> logs/collect_$2_$3.log 2>&1
  echo "done: $2 $3"
}

(run $ALF alfworld train $TRAIN && run $ALF alfworld heldout $HELD) &
(run $SCI sciworld train $TRAIN && run $SCI sciworld heldout $HELD) &
wait
echo "all collection finished"
