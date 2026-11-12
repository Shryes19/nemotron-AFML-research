# Task 3 sanity check — pre-registered acceptance rule

**Fixed on 30 Oct 2026, before any run.** Do not change the thresholds after seeing results. If the check fails, debug the pipeline; don't move the bar.

## Purpose

Confirm that the collected ALFWorld transitions are valid and that the prompt/observation format works, by checking that a 3-shot retrieval world model behaves sensibly on held-out transitions. It is **not** a reproduction of Word2World's number.

## Why we don't expect Word2World's 57–77%

Word2World's 3-shot numbers (57–77% on ALFWorld/ScienceWorld) use **frontier models** (GPT-4o/4.1/5, Gemini 2.5 Flash, Claude Sonnet 4.5) **and give the world model the hidden initial state** (e.g. what is inside every closed receptacle). Their metric is exact match on the full next observation.

We use open Qwen2.5-7B-Instruct and **no hidden state**. Without the hidden state, `go to X` and `open X` reveal contents the model cannot know, so those transitions should be much harder. Landing below 57% is therefore expected.

## Setup

| Item | Value |
|---|---|
| Env | ALFWorld |
| Held-out set | 200 transitions from `alfworld_heldout` (Word2World test task ids), stratified by policy in the 50/30/20 mix, seed 0 |
| Retrieval pool | training budget `alfworld_seed0_n1024` (train task ids only; zero task overlap is asserted) |
| Retrieval | BM25 over `goal + last feedback + action`; top 3 examples |
| Model | Qwen2.5-7B-Instruct via `LLM_BASE_URL`, temperature 0, max 128 tokens |
| Target | `feedback_next` (the raw environment text) |
| Metric | normalized exact match (lowercase, collapse whitespace, strip trailing period), 95% bootstrap CI (1000 resamples) |
| Baselines | (a) copy the top-1 retrieved example's `feedback_next`; (b) always predict `Nothing happens.` |
| Breakdown | per action verb (go to / open / close / take / move / clean-heat-cool / examine / use / look / inventory) and failure transitions (true feedback is `Nothing happens.`) |

## Acceptance rule

**PASS only if all three hold:**

1. Model exact match (point estimate) is **between 25% and 80%**.
2. It beats **both** baselines by **at least 10 points**.
3. EM on `go to` transitions is **lower** than EM on failure transitions.

Reading a failure:

- **Below 25%** → likely a pipeline bug: prompt/format mismatch, wrong field, broken observations.
- **Above 80%** → likely leakage: held-out tasks in the pool, or hidden state leaking into `o_t`.
- **Fails 2** → the model isn't using the examples or the observation; check prompts.
- **Fails 3** → `go to` should be hard without hidden state; if it isn't, check that `o_t` doesn't reveal receptacle contents early.

## How to run

```bash
# today, no GPU: baselines only
~/w2w/uv_alfworld/bin/python -m sanity.three_shot --baselines-only
# once LLM_BASE_URL is set
~/w2w/uv_alfworld/bin/python -m sanity.three_shot
```

Output: `results/sanity_alfworld.json` (numbers + rule verdict) and `results/sanity_alfworld_predictions.jsonl` (every prediction, for inspection).

## Baselines on provisional data (31 Oct 2026, no LLM)

Expert/random data only, provisional budget `alfworld_seed0_n1024_provisional`, 200 held-out rows (125 expert / 75 random):

| Baseline | EM | 95% CI |
|---|---|---|
| copy top-1 retrieved `feedback_next` | 28.5% | 22.5–35.0% |
| always `Nothing happens.` | 12.5% | 8.5–17.0% |

So criterion 2 currently requires the model to reach at least 38.5%. These numbers will be recomputed on the final data (with LLM rows) in the real run; the rule itself does not change.

## Result

**Run 12 Nov 2026, RTX 5070, Qwen2.5-7B-Instruct-AWQ (4-bit; deviation from the full-precision
model, needed to fit 12 GB). Pool: `alfworld_seed0_n1024_provisional` (the final budget: every arm
trained on these buffers). Held-out: 200 transitions (125 expert, 75 random; no LLM rows, see below).
Files: `results/sanity_alfworld.json`, `results/sanity_alfworld_predictions.jsonl`.**

| | Exact match | 95% CI |
|---|---|---|
| Qwen 3-shot (BM25 top 3) | **34.5%** | 27.5–41.0 |
| Baseline (a): copy top-1 retrieved | 28.5% | 22.5–35.0 |
| Baseline (b): always `Nothing happens.` | 12.5% | 8.5–17.0 |

| Rule | Result |
|---|---|
| 1. EM between 25% and 80% | pass (34.5%) |
| 2. Beats both baselines by >= 10 points | **fail** (+6.0 over copy-top-1; +22.0 over `Nothing happens.`) |
| 3. EM on `go to` below EM on failures | pass (18.6% vs 24.0%) |

**Verdict: FAIL** (rule 2). Thresholds unchanged, as pre-registered.

Per category (EM, n): go to 18.6% (113), look 71.8% (39), failure 24.0% (25), clean/heat/cool
33.3% (12), open 0% (10), take 100% (10), close 0% (4), inventory 100% (4), examine 0% (3),
use 0% (2), other 50% (2), move 100% (1).

**Pipeline debugging (as the rule requires).** All 131 misses were inspected for format or field
bugs; none found. No prediction is the right answer with extra text around it, and rules 1 and 3
pass, so there is no sign of broken observations or leakage. The misses are model errors on
information the observation does not contain: the contents of receptacles reached by `go to` or
`open` (wrong or invented objects), and 12 `look` predictions that list room objects where the
true answer is "you see nothing".

**Reading.** The data and prompt format work, but at 7B (4-bit) with three retrieved examples the
model is only slightly better than copying the most similar example's outcome. This bears on arm
G, which is also a retrieval-augmented LLM.
