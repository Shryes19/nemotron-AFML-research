# nemotron-AFML-research

**Team Nemotron** — AFML Course Project, PES University

## Research Question (RQ3)

> Can cheap ex-ante features computed at **n=32 transitions** predict which world model arm will win in decision utility — *before* any full training?

We compare three world model arms across four text environments:

| Arm | Description |
|-----|-------------|
| **G — General RAG** | Frozen Qwen2.5-7B + hierarchical Nemotron-Embed retrieval. No training. |
| **F — Fine-tuned SFT** | Qwen2.5-1.5B + LoRA, trained with next-token prediction loss (Word2World approach). |
| **D — JEPA Specialist** | Learned retrieval: predicts next-obs *embedding*, then NN-decodes to nearest buffer state. |

## Environments

- **ALFWorld** — household navigation (6 task types)
- **ScienceWorld** — science experiments (numeric/physical dynamics)
- **WebShop** — e-commerce product search (high partial observability)
- **ByteSized32-SP** — 31 hand-crafted text games with ground-truth JSON states

## Repo Structure

```
nemotron-afml-research/
├── configs/              # Shared config files (model paths, seeds, env versions)
├── data/
│   ├── buffers/          # Saved transition buffers per (env, n)
│   └── raw/              # Raw environment outputs
├── jepa/
│   ├── encoder/          # Nemotron-Embed-8B wrapper (NF4 quantized)
│   ├── predictor/        # MLP / Transformer predictor + EMA target
│   └── training/         # Training loop, probe fidelity, NN-decode
├── sft/                  # Fine-tuned SFT arm (Qwen2.5-1.5B + LoRA)
├── rag/                  # RAG arm — hierarchical retrieval
├── evaluation/
│   ├── fidelity/         # LM-judge + cosine fidelity metrics
│   └── utility/          # Decision utility / regret scorer
├── rq3/
│   ├── features/         # Ex-ante feature extraction (capability mix, probe, stats)
│   └── predictor/        # Ridge + GBT predictor, LOEO cross-validation
├── scripts/              # End-to-end run scripts
├── notebooks/            # Analysis and figures
└── tests/                # Unit tests
```

## Team

| Member | Role |
|--------|------|
| **Shryes** | JEPA architecture, SFT training loop, probe fidelity, NN-decode |
| **Saanvi** | Environment setup, data collection pipeline, decision utility scorer |
| **Siddhant** | Capability tagger, ByteSized32 setup, feature matrix, compute accounting |
| **Rohini** | RAG arm, Recall@5, WebShop setup, paper assembly |

## Setup

```bash
# Create environment
conda create -n nemotron python=3.10 -y
conda activate nemotron

# Install dependencies
pip install -r requirements.txt
```

### Encoder environment (NV-Embed-v2 / JEPA arm)

NV-Embed-v2's custom model code only works with `transformers==4.42.4`, which
clashes with the newer versions the SFT arm needs. So the JEPA/encoder work
runs in its **own** environment with exact pinned versions:

```bash
python3 -m venv .venv-enc
.venv-enc/bin/pip install -r requirements-encoder.txt

# Real-encoder smoke test (GPU, ~6GB VRAM)
.venv-enc/bin/python scripts/smoke_test_jepa.py \
    --buffer data/buffers/mock_n64.jsonl --real-encoder --device cuda
```

## Quick Start

```bash
# Generate mock data (no env needed)
python scripts/gen_mock_data.py --n 64 --out data/buffers/mock_n64.jsonl

# Smoke test JEPA pipeline end-to-end
python scripts/smoke_test_jepa.py --buffer data/buffers/mock_n64.jsonl
```
