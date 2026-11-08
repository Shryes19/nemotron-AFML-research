"""
sft/model.py
────────────
The model side of the SFT arm (F): Qwen2.5-1.5B-Instruct + LoRA.

    tokenize_example()   prompt ids + answer ids (loss only on the answer)
    load_lora_model()    base model in bf16 + a fresh LoRA adapter
    train_lora()         short supervised run, next-token loss on the answer tokens
    generate()           batched greedy answers

Settings and where they come from
─────────────────────────────────
- Loss only on the answer (next observation) tokens: Ashok et al. 2026, Eq. 3
  (the sum runs over the target observation's tokens); ExpRAG-LoRA (2026) also
  scores assistant tokens only.
- Input = current observation + action, no history: Ashok et al. Eq. 3.
- bf16, constant learning rate after a short warmup: Word2World (Li et al. 2026,
  Table 7: constant with warmup, 10 warmup steps, BF16).
- LoRA rank 16 / alpha 32 / dropout 0.05 on all linear layers (configs/config.yaml;
  all linear layers is the QLoRA recommendation). lr 2e-4 is the usual LoRA rate;
  Word2World's 1e-5 is for full fine-tuning.
- Greedy decoding, so predictions are deterministic.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sft.data import messages

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ── Tokenizing ───────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def prompt_ids(tok, obs: str, action: str) -> List[int]:
    ids = tok.apply_chat_template(messages(obs, action), add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict):            # some tokenizer versions return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def answer_ids(tok, text: str) -> List[int]:
    """Answer tokens + end-of-turn token (Qwen's eos is <|im_end|>), so the model learns to stop."""
    return tok(text.strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]


def fit_length(p: List[int], a: List[int], max_len: int, keep_head: int = 64,
               max_answer: int = 256) -> Tuple[List[int], List[int], bool]:
    """Make prompt + answer fit max_len. Long answers are cut first (to max_answer),
    then tokens are dropped from the MIDDLE of the prompt (keeps the system text +
    goal at the start and the action lines at the end). Returns (p, a, was_cut)."""
    cut = False
    if len(a) > max_answer:
        a, cut = a[: max_answer - 1] + a[-1:], True
    room = max_len - len(a)
    if len(p) > room:
        tail = room - keep_head
        p, cut = p[:keep_head] + p[-tail:], True
    return p, a, cut


def tokenize_example(tok, obs: str, action: str, target: str, max_len: int):
    p, a, cut = fit_length(prompt_ids(tok, obs, action), answer_ids(tok, target), max_len)
    return {"input_ids": p + a, "labels": [-100] * len(p) + a, "cut": cut}


def tokenize_prompt(tok, obs: str, action: str, max_len: int, reserve: int) -> List[int]:
    """Prompt for generation, cut the same way as in training if too long."""
    p, _, _ = fit_length(prompt_ids(tok, obs, action), [], max_len - reserve)
    return p


# ── Model ────────────────────────────────────────────────────────────────────

def load_lora_model(model_name: str, cfg: Dict, device: str, seed: int = 42):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    kw = {}
    if cfg.get("attn_implementation"):    # "eager" | "sdpa" (default: transformers' choice)
        kw["attn_implementation"] = cfg["attn_implementation"]
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kw)
    except TypeError:                     # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kw)
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=int(cfg.get("lora_rank", 16)), lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("lora_targets") or DEFAULT_TARGETS),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model.to(device)


def n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def steps_for(n: int, cfg: Dict) -> int:
    """Training steps for a budget of n rows: `epochs` passes over the data, but at least
    `min_steps` (set by the pilot), so tiny budgets are not left with ~2 updates."""
    bs = int(cfg.get("batch_size", 16))
    by_epochs = math.ceil(float(cfg.get("epochs", 5)) * n / bs)
    return max(by_epochs, int(cfg.get("min_steps") or 0))


def _pad(seqs: List[List[int]], value: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs])


def train_lora(model, examples: List[Dict], steps: int, cfg: Dict, device: str, seed: int = 42,
               eval_fn: Optional[Callable[[int], Dict]] = None, eval_at: Sequence[int] = (),
               log_every: int = 10) -> Dict:
    """
    Plain training loop (no Trainer, fewer moving parts):
    each optimiser step uses `batch_size` examples, split into micro-batches of
    `micro_batch` (grouped by length so little padding is wasted). The loss is the
    mean cross-entropy over all answer tokens in the step. Data is reshuffled every epoch.
    eval_fn(step) is called after the steps listed in eval_at (pilot only).
    """
    bs = min(int(cfg.get("batch_size", 16)), len(examples))
    mb = int(cfg.get("micro_batch", 4))
    warmup = int(cfg.get("warmup_steps", 10))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 2e-4)),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    rng = random.Random(seed)
    order, pos = [], 0
    hist = {"step": [], "loss": [], "evals": []}
    eval_at = set(eval_at)
    model.train()
    t0 = time.time()

    for step in range(1, steps + 1):
        batch = []
        while len(batch) < bs:
            if pos >= len(order):
                order, pos = list(range(len(examples))), 0
                rng.shuffle(order)
            batch.append(examples[order[pos]])
            pos += 1
        batch.sort(key=lambda e: len(e["input_ids"]))
        n_tok = sum(sum(l != -100 for l in e["labels"]) for e in batch)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, len(batch), mb):
            chunk = batch[i:i + mb]
            ids = _pad([e["input_ids"] for e in chunk], 0).to(device)
            labels = _pad([e["labels"] for e in chunk], -100).to(device)
            mask = _pad([[1] * len(e["input_ids"]) for e in chunk], 0).to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                ignore_index=-100, reduction="sum") / n_tok
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
        opt.step()
        sched.step()
        hist["step"].append(step)
        hist["loss"].append(round(total, 4))
        if step % log_every == 0 or step == steps:
            print(f"    step {step:>5}/{steps}  loss {total:.4f}  ({time.time() - t0:.0f}s)")
        if eval_fn is not None and step in eval_at:
            model.eval()
            r = eval_fn(step)
            model.train()
            r["step"] = step
            hist["evals"].append(r)
    model.eval()
    hist["seconds"] = round(time.time() - t0, 1)
    return hist


@torch.no_grad()
def generate(model, tok, prompts: List[List[int]], max_new_tokens: int, batch_size: int,
             device: str) -> List[str]:
    """Greedy answers for many prompts. Longest prompts first, left padding."""
    model.eval()
    was_cache = model.config.use_cache
    model.config.use_cache = True
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: List[Optional[str]] = [None] * len(prompts)
    pad = tok.pad_token_id
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs = [prompts[i] for i in idx]
        width = max(len(x) for x in seqs)
        ids = torch.tensor([[pad] * (width - len(x)) + x for x in seqs], device=device)
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in seqs], device=device)
        gen = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                             max_new_tokens=max_new_tokens, pad_token_id=pad,
                             eos_token_id=tok.eos_token_id)
        for j, i in enumerate(idx):
            out[i] = tok.decode(gen[j, width:], skip_special_tokens=True).strip()
    model.config.use_cache = was_cache
    return out  # type: ignore[return-value]
