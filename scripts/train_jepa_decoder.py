"""
scripts/train_jepa_decoder.py
─────────────────────────────
Step 2 of the "JEPA + text decoder" experiment (VL-JEPA style y-decoder, 13 Nov 2026).

A light text decoder turns a JEPA next-state VECTOR into the next-state TEXT, so JEPA can
write sentences it never saw instead of picking the nearest training sentence.

    decoder input  = only the vector (4096 numbers), mapped by a small trainable layer to
                     k "soft prompt" tokens for Qwen2.5-1.5B-Instruct (+ LoRA)
    decoder output = the next observation text

It sees NO observation or action text, so the answer still comes entirely from JEPA's
prediction (otherwise it would turn into the SFT arm). Like VL-JEPA's decoder, it is
trained on TRUE target vectors (the checkpoint's own training rows only, so no extra
data), then given JEPA's PREDICTED vectors at test time.

Reported per held-out set (exact match / word F1, Saanvi's metrics):
  decoder(predicted vector)   JEPA + text decoder (the new arm)
  decoder(true vector)        oracle control: how much of the text survives the vector
  nearest training sentence   today's JEPA decode, on the same predicted vectors

    # GPU box, inside .venv-sft
    python scripts/train_jepa_decoder.py --data outputs/jepa_decoder/alfworld_b256.pt

The training length is fixed in advance (300 steps); scores at steps 100 and 200 are
printed too, for transparency only.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from jepa.compute import track  # noqa: E402
from sft.data import fidelity  # noqa: E402
from sft.model import answer_ids, load_lora_model, load_tokenizer  # noqa: E402

PRE = "<|im_start|>system\nTurn the state embedding into the text-game feedback it describes.<|im_end|>\n<|im_start|>user\nState embedding: "
POST = "<|im_end|>\n<|im_start|>assistant\n"


class SoftPrompt(nn.Module):
    """vector (D) -> k soft tokens (k, H)."""

    def __init__(self, d_in, hidden, k):
        super().__init__()
        self.k, self.hidden = k, hidden
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, k * hidden))

    def forward(self, z):
        return self.net(z).view(z.shape[0], self.k, self.hidden)


def build_batch(model, soft, tok, z, answers, device, dtype):
    """inputs_embeds = [PRE tokens][k soft tokens][POST tokens][answer], left-padded.
    Returns embeds, mask, labels (labels only for the answer when answers is not None)."""
    emb = model.get_input_embeddings()
    pre = torch.tensor(tok(PRE, add_special_tokens=False)["input_ids"], device=device)
    post = torch.tensor(tok(POST, add_special_tokens=False)["input_ids"], device=device)
    sp = soft(z.to(device=device, dtype=torch.float32)).to(dtype)
    seqs, labs = [], []
    for i in range(z.shape[0]):
        parts = [emb(pre), sp[i], emb(post)]
        lab = [-100] * (len(pre) + soft.k + len(post))
        if answers is not None:
            a = torch.tensor(answers[i], device=device)
            parts.append(emb(a))
            lab += answers[i]
        seqs.append(torch.cat(parts, 0))
        labs.append(lab)
    width = max(s.shape[0] for s in seqs)
    H = seqs[0].shape[1]
    embeds = torch.zeros(len(seqs), width, H, device=device, dtype=dtype)
    mask = torch.zeros(len(seqs), width, device=device, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, device=device, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        embeds[i, width - s.shape[0]:] = s
        mask[i, width - s.shape[0]:] = 1
        labels[i, width - len(l):] = torch.tensor(l, device=device)
    return embeds, mask, labels


@torch.no_grad()
def decode(model, soft, tok, z, device, dtype, max_new=96, batch=32):
    model.eval()
    model.config.use_cache = True
    out = []
    for s in range(0, z.shape[0], batch):
        e, m, _ = build_batch(model, soft, tok, z[s:s + batch], None, device, dtype)
        g = model.generate(inputs_embeds=e, attention_mask=m, do_sample=False, max_new_tokens=max_new,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        out += [tok.decode(row, skip_special_tokens=True).strip() for row in g]
    model.config.use_cache = False
    return out


def nearest(z_pred, train_z, train_text):
    sims = torch.nn.functional.normalize(z_pred.float(), dim=-1) @ \
        torch.nn.functional.normalize(train_z.float(), dim=-1).T
    return [train_text[j] for j in sims.argmax(1).tolist()]


def load_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="outputs/jepa_decoder/<env>_b<n>.pt")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--k", type=int, default=8, help="soft prompt tokens")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-at", type=int, nargs="*", default=[100, 200])
    p.add_argument("--lr-soft", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=None, help="code check: N test rows per set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg_all = load_config(args.config)
    cfg = dict(cfg_all.get("sft", {}))
    seed = int(cfg_all.get("seed", 42))
    model_name = args.model or cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    env, budget = d["env"], d["budget"]
    tests = d["tests"]
    if args.limit:
        tests = {k: {kk: vv[: args.limit] for kk, vv in v.items()} for k, v in tests.items()}
    random.seed(seed)
    torch.manual_seed(seed)

    tok = load_tokenizer(model_name)
    model = load_lora_model(model_name, cfg, args.device, seed=seed)
    dtype = next(model.parameters()).dtype
    H = model.get_input_embeddings().weight.shape[1]
    soft = SoftPrompt(d["train_z"].shape[1], H, args.k).to(args.device)
    lora_params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": float(cfg.get("lr", 2e-4))},
                             {"params": soft.parameters(), "lr": args.lr_soft}], weight_decay=0.0)
    warm = int(cfg.get("warmup_steps", 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    Z = d["train_z"].float()
    A = [answer_ids(tok, t) for t in d["train_text"]]
    n, bs, mb = len(A), int(cfg.get("batch_size", 16)), int(cfg.get("micro_batch", 4)) * 2
    print(f"[Decoder] env={env} budget={budget} train={n} k={args.k} steps={args.steps}  "
          f"sets: " + ", ".join(f"{k} ({len(v['text'])})" for k, v in tests.items()))

    results = {"env": env, "budget": budget, "steps": args.steps, "k": args.k, "evals": []}

    def evaluate(step, with_controls):
        row = {"step": step}
        for name, t in tests.items():
            r = {"decoder_pred": fidelity(decode(model, soft, tok, t["z_pred"].float(), args.device, dtype),
                                          t["text"])}
            if with_controls:
                preds_true = decode(model, soft, tok, t["z_true"].float(), args.device, dtype)
                r["decoder_true"] = fidelity(preds_true, t["text"])
                r["nearest_pred"] = fidelity(nearest(t["z_pred"], d["train_z"], d["train_text"]), t["text"])
            row[name] = r
            print(f"    [step {step}] {name}: " + "  ".join(
                f"{k} {v['exact']:.3f}/{v['f1']:.3f}" for k, v in r.items()))
        results["evals"].append(row)
        soft.train()
        model.train()

    order, pos = [], 0
    rng = random.Random(seed)
    t0 = time.time()
    model.train()
    with track("jepa_decoder_train", arm="D", env=env, game="*", n=budget, steps=args.steps):
        for step in range(1, args.steps + 1):
            idx = []
            while len(idx) < min(bs, n):
                if pos >= len(order):
                    order, pos = list(range(n)), 0
                    rng.shuffle(order)
                idx.append(order[pos])
                pos += 1
            n_tok = sum(len(A[i]) for i in idx)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in range(0, len(idx), mb):
                ch = idx[s:s + mb]
                e, m, lab = build_batch(model, soft, tok, Z[ch], [A[i] for i in ch], args.device, dtype)
                logits = model(inputs_embeds=e, attention_mask=m).logits[:, :-1].float()
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                   lab[:, 1:].reshape(-1), ignore_index=-100,
                                                   reduction="sum") / n_tok
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(lora_params + list(soft.parameters()), 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == args.steps:
                print(f"    step {step:>4}/{args.steps}  loss {total:.4f}  ({time.time() - t0:.0f}s)")
            if step in args.eval_at and step != args.steps:
                evaluate(step, with_controls=False)
    with track("jepa_decoder_predict", arm="D", env=env, game="*", n=budget):
        evaluate(args.steps, with_controls=True)

    out = Path(args.out or str(Path(args.data).with_suffix("")) + "_results.json")
    out.write_text(json.dumps(results, indent=2))
    final = results["evals"][-1]
    print("\n" + "=" * 76)
    print(f"  JEPA + text decoder — {env}, n={budget}, step {args.steps}  (exact / F1)")
    print("=" * 76)
    print(f"  {'held-out set':<24} {'decoder(pred)':>16} {'nearest(pred)':>16} {'decoder(true)':>16}")
    for name in tests:
        r = final[name]
        cell = lambda k: f"{r[k]['exact']:.3f} / {r[k]['f1']:.3f}"  # noqa: E731
        print(f"  {name:<24} {cell('decoder_pred'):>16} {cell('nearest_pred'):>16} {cell('decoder_true'):>16}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
