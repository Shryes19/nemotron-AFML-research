"""
scripts/export_jepa_vectors.py
──────────────────────────────
Step 1 of the "JEPA + text decoder" experiment (VL-JEPA style, 13 Nov 2026).

VL-JEPA predicts the embedding of the target text, then uses a light text decoder to
turn the predicted embedding into words. Our JEPA arm (D) instead answers with the
nearest training sentence. This script saves the vectors a decoder needs, from ONE
trained JEPA checkpoint (default budget 256):

  train : the true next-state vectors of the checkpoint's own training rows + their
          text (the decoder learns "vector → text" on these, so it sees no extra data)
  tests : for each held-out set, JEPA's PREDICTED next-state vector, the TRUE next-state
          vector (an "oracle" control: how well can the decoder read a perfect vector?)
          and the true text

All vectors are in JEPA's target space (frozen NV-Embed, L2-normalised), the same
space nearest-state decoding uses.

    # GPU box, inside .venv-enc (loads NV-Embed, ~5 min per env)
    python scripts/export_jepa_vectors.py --env alfworld
    python scripts/export_jepa_vectors.py --env sciworld
    python scripts/export_jepa_vectors.py --env bytesized32

Output: outputs/jepa_decoder/<env>_b<budget>.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from jepa.encoder.load import load_encoder  # noqa: E402
from jepa.planning.latent_world_model import LatentWorldModel  # noqa: E402
from jepa.training.embed_cache import action_text, make_context, obs_text  # noqa: E402
from jepa.training.features import feature_matrix  # noqa: E402

ENVS = {
    "alfworld": {"buffer": "data/buffers/alfworld_games.jsonl",
                 "ckpt": "outputs/jepa/alfworld_games_steps2000_test-alfworld_heldout_buffer_sepact_adm_frozen",
                 "tests": {"alfworld_heldout": "data/eval/alfworld_heldout_buffer.jsonl"}},
    "sciworld": {"buffer": "data/buffers/sciworld_games.jsonl",
                 "ckpt": "outputs/jepa/sciworld_games_steps2000_test-sciworld_heldout_buffer_sepact_adm_frozen",
                 "tests": {"sciworld_heldout": "data/eval/sciworld_heldout_buffer.jsonl",
                           "sciworld_seen_heldout": "data/eval/sciworld_seen_heldout_buffer.jsonl"}},
    "bytesized32": {"buffer": "data/buffers/bytesized32_games.jsonl",
                    "ckpt": "outputs/jepa/bytesized32_games_steps2000_sepact_adm_frozen",
                    "tests": {}, "val_rows": "sft/rows/bytesized32.json"},
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def predict_rows(wm, rows, batch=256):
    """Batched version of LatentWorldModel.predict for (obs, action) rows."""
    out = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        if wm.input_mode == "separate":
            z = torch.cat([wm.encoder.encode([obs_text(r["obs"]) for r in chunk]),
                           wm.encoder.encode([action_text(r["action"]) for r in chunk])], dim=1)
        else:
            z = wm.encoder.encode([make_context(r["obs"], r["action"]) for r in chunk])
        if wm.admissible_feature:
            z = torch.cat([z, feature_matrix(chunk).to(z.dtype)], dim=1)
        out.append(wm.predictor(wm._online(z.to(wm.device))).float().cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=sorted(ENVS))
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--fake-encoder", action="store_true", help="CPU code check")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    E = ENVS[args.env]
    ckpt = Path(args.ckpt_dir or E["ckpt"]) / f"budget_{args.budget}.pt"
    items = read_jsonl(E["buffer"])
    extra = torch.load(ckpt, map_location="cpu", weights_only=False).get("extra", {})
    rows = extra.get("train_rows")
    if rows is None:
        sys.exit(f"[Error] {ckpt} has no train_rows")
    train = [items[i] for i in rows]

    tests = {name: read_jsonl(path) for name, path in E["tests"].items()}
    if E.get("val_rows"):
        val = json.loads(Path(E["val_rows"]).read_text())["val_rows"]
        tests["random10_heldout"] = [items[i] for i in val]

    import yaml
    ecfg = (yaml.safe_load(open("configs/config.yaml")) or {}).get("encoder", {})
    encoder = load_encoder(real=not args.fake_encoder, device=args.device,
                           model_name=ecfg.get("model_name"), quantize=bool(ecfg.get("quantize", True)),
                           batch_size=int(ecfg.get("batch_size", 32)),
                           max_length=int(ecfg.get("max_length", 512)))
    wm = LatentWorldModel.from_checkpoint(str(ckpt), encoder, device=args.device)

    def targets(texts):
        return wm._project_targets(encoder.encode(texts)).float().cpu()

    out = {"env": args.env, "budget": args.budget, "checkpoint": str(ckpt),
           "train_text": [t["obs_next"] for t in train],
           "train_z": targets([t["obs_next"] for t in train]).half(), "tests": {}}
    print(f"[Export] {ckpt}: {len(train)} training rows")
    for name, rws in tests.items():
        z_pred = predict_rows(wm, rws)
        z_true = targets([r["obs_next"] for r in rws])
        cos = torch.nn.functional.cosine_similarity(z_pred, z_true).mean().item()
        out["tests"][name] = {"text": [r["obs_next"] for r in rws], "action": [r["action"] for r in rws],
                              "z_pred": z_pred.half(), "z_true": z_true.half()}
        print(f"[Export] {name}: {len(rws)} rows, mean cos(predicted, true) = {cos:.3f}")
    path = Path(args.out or f"outputs/jepa_decoder/{args.env}_b{args.budget}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"[Export] saved {path}")


if __name__ == "__main__":
    main()
