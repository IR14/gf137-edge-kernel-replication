#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NUMPY_SPEC="${NUMPY_SPEC:-numpy}"
ONNX_SPEC="${ONNX_SPEC:-onnx}"
ONNXRUNTIME_SPEC="${ONNXRUNTIME_SPEC:-onnxruntime}"

if command -v uv >/dev/null 2>&1; then
  uv run --with "$NUMPY_SPEC" --with "$ONNX_SPEC" --with "$ONNXRUNTIME_SPEC" \
    python scripts/onnxruntime_int8_baseline.py "$@"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install "$NUMPY_SPEC" "$ONNX_SPEC" "$ONNXRUNTIME_SPEC"
  python scripts/onnxruntime_int8_baseline.py "$@"
fi
