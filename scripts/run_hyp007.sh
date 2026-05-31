#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NUMPY_SPEC="${NUMPY_SPEC:-numpy}"

if command -v uv >/dev/null 2>&1; then
  uv run --with "$NUMPY_SPEC" python scripts/repair_aware_checkpoint.py "$@"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install "$NUMPY_SPEC"
  python scripts/repair_aware_checkpoint.py "$@"
fi
