# When the GPU is available — runbook

Everything below is already written and tested offline (fake LLM server). It needs only a reachable vLLM server.

## 1. Network: VM ↔ teammate's RTX 4090 (≈15 min, once)

Tailscale is free and needs no router changes. On **both** the 4090 machine and the VM:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```
```bash
sudo tailscale up
```

Log in to the same Tailscale account, or have the teammate share their machine with you from the Tailscale admin console. Note the 4090 machine's Tailscale IP (`tailscale ip -4`, looks like `100.x.y.z`). The VM has no `curl`; use `wget -qO- https://tailscale.com/install.sh | sh` there instead.

## 2. Start the model (teammate, on the 4090)

Pick a random string as the API key and share it privately (not in git or the team chat):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --host 0.0.0.0 --api-key <random-string> --seed 0
```

## 3. Point the VM at it (you)

Create `~/wm/.env` (never committed; `.gitignore` blocks it):

```
LLM_BASE_URL=http://<teammate-tailscale-ip>:8000/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_KEY=<random-string>
```

Check it works:

```bash
cd ~/wm && ~/w2w/uv_alfworld/bin/python -m wm.llm_client --ping
```

You should see `reply: ready` (or similar) and a latency.

## 4. Collect the LLM rows (≈1–3 h)

```bash
cd ~/wm && nohup bash scripts/collect_all.sh llm > logs/collect_llm.log 2>&1 &
```

This adds `llm` rows (320 train + 60 held-out per env) to the existing files. Expert and random targets are already met, so only LLM-driven episodes run. Progress: `tail -f logs/collect_alfworld_train.log`. Every reply is cached in `cache/llm_cache.sqlite`, so an interrupted run resumes without repeating calls.

## 5. Final budgets + upload

```bash
cd ~/wm && bash scripts/package_data.sh final
```

This replaces the provisional budgets with final 50/30/20 ones. Upload `data/upload/*` to Drive and tell the team the provisional budgets are retired.

## 6. Sanity check (Task 3)

```bash
cd ~/wm && ~/w2w/uv_alfworld/bin/python -m sanity.three_shot
```

Compare against the rule in [SANITY_CHECK.md](SANITY_CHECK.md) and fill in its Result section. 200 calls; well under an hour.

## If something fails

- `--ping` hangs or times out: Tailscale isn't connected on one side (`tailscale status`), or vLLM isn't using `--host 0.0.0.0`.
- `HTTP 401`: the `LLM_API_KEY` doesn't match the `--api-key`.
- Many `llm_fallback` rows (check with `python -m wm.stats`): the model isn't replying with a number; look at `llm_meta.reply` in those rows.
