# Rohini RAG drop-in

Standalone files only: no repository code is changed. The G arm is frozen `Qwen2.5-7B-Instruct`; retrieval is shared `nvidia/NV-Embed-v2`. It uses top-`min(50, n//2)` observation similarity, action-similarity re-ranking, and `k=5` returned transitions.

## Required input and outputs

Use the exact shared branched states and the exact D/F training rows. On the 12 GB GPU, first prepare retrieval prompts with NV-Embed, then unload it and generate through vLLM Qwen:

```bash
SID=/path/to/repo bash rag_dropin/run_g_nv.sh
# after NV-Embed is unloaded and vLLM is running:
SID=/path/to/repo GENERATE=1 bash rag_dropin/run_g_nv.sh
```

This produces `/shared/siddanth/rohini_g/<set>_b256.jsonl` with exactly `{"id", "arm":"G", "budget", "actions", "pred_next"}`. Repeat for `32 64 128 256 512 1024` and sets `alfworld`, `sciworld`, `sciworld_seen`, `bytesized32`. Validate any project-specific decision-utility constraints against `docs/DECISION_UTILITY.md` before scoring.

Generate the requested n=32 feature-matrix column (one row per unit):

```bash
python measure_recall.py --env alfworld --budget 32 --buffer data/alfworld_transitions.jsonl \
  --queries data/alfworld_queries.jsonl --out metrics/alfworld_recall_b32.csv
```

The CSV columns are `env,game,recall_at5`; units are task ID (ALFWorld), task type (ScienceWorld), and game (ByteSized32). Relevance is intentionally model-free: same normalized action template and primary action object.

## Team handoff

Generation invokes the repository's `wm.utility.score.track("g_generation", arm="G", ...)`, so GPU-hours land in the shared cost table. Compare G decision utility against Shryes’s action-lookup baseline and push the Recall CSV on the review branch.

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
