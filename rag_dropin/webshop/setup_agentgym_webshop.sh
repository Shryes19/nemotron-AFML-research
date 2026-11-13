#!/usr/bin/env bash
set -euo pipefail
# Run in an isolated environment; this installs no files into the host repository.
python -m pip install -r webshop/requirements.txt
python - <<'PY'
try:
    import agentgym  # noqa: F401
except ImportError as exc:
    raise SystemExit("AgentGym import failed; check its current WebShop extra/package name.") from exc
print("AgentGym is installed. Configure the WebShop endpoint per your AgentGym version before collection.")
PY
