# Rohini RAG drop-in

The G arm is frozen Qwen2.5-7B-Instruct with retrieval by `nvidia/NV-Embed-v2` (4-bit, the same
encoder as JEPA, arm D). It uses top-`min(50, n//2)` observation similarity, action-similarity
re-ranking, and `k=5` returned transitions.

## Producing G's predictions (GPU box)

NV-Embed and the vLLM Qwen server cannot share the 12 GB GPU, so G runs in two stages.
`overnight.sh` runs everything left on the GPU in one go, resumably, from the repo root:

1. `export_g_buffers.py`: each budget's retrieval buffer = **exactly the rows D and F trained on**
   (the `train_rows` saved in the JEPA checkpoints), checked against JEPA's own candidate files.
2. `prepare_g_prompts.py` (NV-Embed, loaded once): encodes every text of a set in one batched pass,
   writes one prompt per (state, candidate action) for every budget, and `recall_at5.csv`.
3. vLLM starts; `generate_predictions.py` writes `<set>_b<n>.jsonl` =
   `{"id", "arm":"G", "budget", "actions", "pred_next"}`, n = 256 first, then 32 ... 1024.
4. `scripts/run_utility.sh` scores G (D and F are already scored), then labels and figures.

```bash
bash rag_dropin/overnight.sh 2>&1 | tee overnight.log
```

GPU time is logged as stages `rag_index_build` and `g_generation` (`results/compute/compute_log_g.jsonl`).

**Deviations from the plan (report in the paper):**
- The LLM is **Qwen2.5-7B-Instruct-AWQ** served by vLLM (`--backend vllm`, the default), the same model
  as the decision-utility judge: full-precision 7B does not fit the 12 GB RTX 5070.
  `--backend hf` keeps the original `transformers` path for a bigger GPU.
- Each example text in the prompt is cut at 1500 characters and the current state at 4000
  (keeps prompts inside vLLM's 8192 tokens; most texts are shorter).
- Greedy decoding, at most 128 new tokens, one seed, as for the other arms.

**Recall@5 (feature-matrix column):** `prepare_g_prompts.py` writes `recall_at5.csv`
(copied to `results/utility/g_recall_at5.csv`): `env, game, budget, n_states, recall_at5` for every set
and budget; the matrix uses `budget = 32`. Query = each branched state with its gold action; a hit is
a retrieved example with the same normalized action template and object (model-free). Units are
task ID (ALFWorld), task type (ScienceWorld), and game (ByteSized32).
`measure_recall.py` does the same for one budget and a custom query file.

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
