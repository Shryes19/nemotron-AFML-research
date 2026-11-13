# Rohini RAG drop-in

Standalone files only: no repository code is changed. The G arm is frozen `Qwen/Qwen2.5-7B-Instruct`; retrieval is `BAAI/bge-base-en-v1.5`, **not NV-Embed**. It uses top-`min(50, n//2)` observation similarity, action-similarity re-ranking, and `k=5` returned transitions.

## Required input and outputs

Use the exact shared branched states, preserving their candidate-action order:

```bash
python generate_predictions.py --buffer data/<set>_transitions.jsonl \
  --states data/eval/<set>_branched_states.jsonl --set <set> --budget 256 \
  --outdir /shared/siddanth/rohini_g
```

This produces `/shared/siddanth/rohini_g/<set>_b256.jsonl` with exactly `{"id", "arm":"G", "budget", "actions", "pred_next"}`. Repeat for `32 64 128 256 512 1024` and sets `alfworld`, `sciworld`, `sciworld_seen`, `bytesized32`. Validate any project-specific decision-utility constraints against `docs/DECISION_UTILITY.md` before scoring.

Generate the requested n=32 feature-matrix column (one row per unit):

```bash
python measure_recall.py --env alfworld --budget 32 --buffer data/alfworld_transitions.jsonl \
  --queries data/alfworld_queries.jsonl --out metrics/alfworld_recall_b32.csv
```

The CSV columns are `env,game,recall_at5`; units are task ID (ALFWorld), task type (ScienceWorld), and game (ByteSized32). Relevance is intentionally model-free: same normalized action template and primary action object.

## Team handoff

Use the tracking adapter so G GPU-hours reach the cost table (set the actual module, without guessing a repository layout):

```bash
TRACK_IMPORT=your_project.tracking python run_g_tracked.py -- --buffer ... --states ... --set alfworld --budget 256
```

It invokes `track(arm="G")`. Compare G decision utility against Shryes’s action-lookup baseline and push the Recall CSV on the review branch.

## WebShop checklist

WebShop is excluded from the current feature matrix and G output-set list. The `webshop/` scripts implement the remaining work: install via AgentGym, collect/tag 128 mixed-policy transitions, and validate the tagger contract. Supply the environment factory required by the installed AgentGym release:

```bash
bash webshop/setup_agentgym_webshop.sh
python webshop/collect_mixed_policy.py --factory your_agentgym_webshop_factory
python webshop/validate_observation_format.py webshop/webshop_mixed_n128.jsonl
```

Compare scored G and action-lookup outputs after the common decision-utility labeler has added `decision_utility`:

```bash
python compare_action_lookup.py --g scored_g.jsonl --lookup scored_action_lookup.jsonl
```
