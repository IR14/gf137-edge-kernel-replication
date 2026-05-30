#!/usr/bin/env python3
"""Fail CI unless the HYP-003 pre-freeze engineering gate passes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path("outputs/quantized_baseline_sweep.json")
    if not path.exists():
        print(f"missing result file: {path}", file=sys.stderr)
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    required = ["pass_storage", "pass_agreement", "pass_speed_proxy"]
    failed = [key for key in required if not summary.get(key)]

    print(json.dumps(summary, indent=2))
    if failed:
        print(f"HYP-003 gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
