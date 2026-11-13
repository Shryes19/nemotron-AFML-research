# Rohini RAG drop-in

The G arm is frozen Qwen2.5-7B-Instruct; retrieval is `BAAI/bge-base-en-v1.5`, **not NV-Embed**. It uses top-`min(50, n//2)` observation similarity, action-similarity re-ranking, and `k=5` returned transitions.

## Producing G's predictions (GPU box)

`run_g.sh` does everything, resumably, from the repo root, with the team's vLLM server running:

1. `export_g_buffers.py` writes each budget's retrieval buffer: **exactly the rows D and F trained on**
   (the `train_rows` saved in the JEPA checkpoints), and checks them against JEPA's own candidate files.
2. `generate_predictions.py` writes `<set>_b<n>.jsonl` = `{"id", "arm":"G", "budget", "actions", "pred_next"}`
   for `alfworld`, `sciworld`, `sciworld_seen`, `bytesized32`, n = 256 first, then 32 … 1024.
   Its GPU time is logged as stage `g_generation` in the shared compute log.

```bash
SID=/home/siddanth_anil/nemotron-AFML-research TORCH_PY=~/vllm-venv/bin/python G_PY=~/g-venv/bin/python \
    bash rag_dropin/run_g.sh
G_DIR=~/g_candidates PRED_ROOT=$SID/outputs bash scripts/run_utility.sh   # score G
```

**Deviations from the plan (report in the paper):**
- The LLM is **Qwen2.5-7B-Instruct-AWQ** served by vLLM (`--backend vllm`, the default), the same model
  as the decision-utility judge: full-precision 7B does not fit the 12 GB RTX 5070.
  `--backend hf` keeps the original `transformers` path for a bigger GPU.
- Each text in the prompt is cut at 3000 characters (rarely hit; keeps prompts inside 8192 tokens).
- Greedy decoding, at most 128 new tokens, one seed, as for the other arms.

Generate the requested n=32 feature-matrix column (one row per unit):

```bash
python measure_recall.py --env alfworld --budget 32 --buffer data/alfworld_transitions.jsonl \
  --queries data/alfworld_queries.jsonl --out metrics/alfworld_recall_b32.csv
```

The CSV columns are `env,game,recall_at5`; units are task ID (ALFWorld), task type (ScienceWorld), and game (ByteSized32). Relevance is intentionally model-free: same normalized action template and primary action object.

## Team handoff

Compare G decision utility against Shryes’s action-lookup baseline and push the Recall CSV on the review branch.

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
