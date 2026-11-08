"""
scripts/sft_gpu_check.py
────────────────────────
Small step-by-step GPU test for the SFT arm, to find where a CUDA crash happens.
Each stage prints OK (with GPU memory) before the next one starts.

    python scripts/sft_gpu_check.py                    # default attention
    python scripts/sft_gpu_check.py --attn eager       # plain attention kernels
    python scripts/sft_gpu_check.py --dtype float16
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


def ok(stage, t0):
    torch.cuda.synchronize()
    print(f"[OK] {stage:<34} {time.time() - t0:6.1f}s   GPU mem {torch.cuda.memory_allocated() / 2**30:.2f} GB "
          f"(peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--attn", default=None, help="eager | sdpa")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--max-new", type=int, default=192)
    args = p.parse_args()

    import transformers, peft
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda})  transformers {transformers.__version__}  "
          f"peft {peft.__version__}")
    print(f"GPU {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    t0 = time.time()
    x = torch.randn(2048, 2048, device="cuda", dtype=getattr(torch, args.dtype))
    (x @ x).sum().item()
    ok("matmul", t0)

    from sft.data import read_jsonl
    from sft.model import (generate, load_lora_model, load_tokenizer, tokenize_example,
                           tokenize_prompt, train_lora)
    # the LONGEST observations: the hardest case for memory
    items = sorted(read_jsonl("data/buffers/alfworld_games.jsonl"), key=lambda r: -len(r["obs"]))[:64]
    tok = load_tokenizer(args.model)
    cfg = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05, "batch_size": 16, "micro_batch": 4,
           "lr": 2e-4, "warmup_steps": 1, "attn_implementation": args.attn}
    t0 = time.time()
    if args.dtype != "bfloat16":
        torch.set_default_dtype(getattr(torch, args.dtype))
    model = load_lora_model(args.model, cfg, "cuda")
    if args.dtype != "bfloat16":
        model = model.to(getattr(torch, args.dtype))
    print(f"     attention: {model.config._attn_implementation}  dtype: {next(model.parameters()).dtype}")
    ok("load model + LoRA", t0)

    t0 = time.time()
    prompts = [tokenize_prompt(tok, r["obs"], r["action"], 1024, args.max_new) for r in items]
    print(f"     prompt lengths: {min(map(len, prompts))}-{max(map(len, prompts))} tokens")
    with torch.no_grad():
        ids = torch.tensor([prompts[0]], device="cuda")
        model(input_ids=ids)
    ok("one forward pass (batch 1)", t0)

    t0 = time.time()
    out = generate(model, tok, prompts[:1], 32, 1, "cuda")
    ok("generate, batch 1", t0)
    print("     sample:", repr(out[0][:80]))

    t0 = time.time()
    generate(model, tok, prompts[: args.gen_batch], args.max_new, args.gen_batch, "cuda")
    ok(f"generate, batch {args.gen_batch}, longest prompts", t0)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    ex = [tokenize_example(tok, r["obs"], r["action"], r["obs_next"], 1024) for r in items[:16]]
    print(f"     training examples: {min(len(e['input_ids']) for e in ex)}-{max(len(e['input_ids']) for e in ex)} tokens")
    train_lora(model, ex, 3, cfg, "cuda", log_every=1)
    ok("3 training steps, longest examples", t0)
    print("\nALL STAGES OK")


if __name__ == "__main__":
    main()
