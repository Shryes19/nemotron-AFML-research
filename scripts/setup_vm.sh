#!/bin/bash
# Install ALFWorld and ScienceWorld the way Word2World does (their vendored AgentGym),
# minus the GPU-only agentgym_rl env. Needs: git, wget, unzip, Java 17 on PATH.
# Usage: bash scripts/setup_vm.sh   (run from the repo root)
set -ex

W2W_COMMIT=e8dc240269fb222de3c242b09c70bfbf0b4ac1c9
W2W_DIR="$HOME/w2w"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null; then
  if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"
  else wget -qO- https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
fi
java -version

if [ ! -d "$W2W_DIR/.git" ]; then git clone https://github.com/X1AOX1A/Word2World "$W2W_DIR"; fi
git -C "$W2W_DIR" checkout -q "$W2W_COMMIT"

# ALFWorld (mirrors Word2World scripts/env_setup/uv_alfworld.sh)
cd "$W2W_DIR"
[ -d uv_alfworld ] || uv venv uv_alfworld --python 3.9
VIRTUAL_ENV="$W2W_DIR/uv_alfworld" uv pip install alfworld==0.3.3
VIRTUAL_ENV="$W2W_DIR/uv_alfworld" uv pip uninstall opencv-python || true
VIRTUAL_ENV="$W2W_DIR/uv_alfworld" uv pip install -e AgentGym/agentenv-alfworld
VIRTUAL_ENV="$W2W_DIR/uv_alfworld" uv pip install -e "$REPO_DIR"

# ScienceWorld (mirrors uv_sciworld.sh; Java installed separately)
[ -d uv_sciworld ] || uv venv uv_sciworld --python 3.8
VIRTUAL_ENV="$W2W_DIR/uv_sciworld" uv pip install -e AgentGym/agentenv-sciworld
VIRTUAL_ENV="$W2W_DIR/uv_sciworld" uv pip install -e "$REPO_DIR"

echo "Setup done. Next: $W2W_DIR/uv_alfworld/bin/python scripts/fetch_data.py"
