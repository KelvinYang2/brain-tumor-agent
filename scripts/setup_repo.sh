#!/usr/bin/env bash
# One-shot bootstrap helper. Run from inside an empty git repo.
set -euo pipefail

echo "==> Creating venv with uv (python 3.11)"
uv venv --python 3.11

echo "==> Installing project (editable, with dev extras)"
# shellcheck source=/dev/null
source .venv/bin/activate
uv pip install -e ".[dev]"

echo "==> Verifying imports"
python -c "import src; print('src version:', src.__version__)"

echo "==> Running smoke tests"
pytest tests/ -q

echo "==> Setting up .env (if missing)"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    Created .env from .env.example — edit it to point at your data."
else
    echo "    .env already exists — leaving it alone."
fi

echo ""
echo "DONE. Next: edit .env, then proceed to Step 1.2 (tool wrapping)."
