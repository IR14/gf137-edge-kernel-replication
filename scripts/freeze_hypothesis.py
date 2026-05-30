#!/usr/bin/env python3
"""Write a SHA-256 freeze manifest for hypothesis files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Files to freeze.")
    parser.add_argument(
        "--out",
        default="outputs/freeze_manifest.json",
        help="Output manifest path.",
    )
    args = parser.parse_args()

    files = [Path(item).resolve() for item in args.files]
    rows = []
    combined = hashlib.sha256()

    for path in files:
        digest = sha256_file(path)
        rows.append({"path": str(path), "sha256": digest})
        combined.update(str(path).encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": rows,
        "combined_sha256": combined.hexdigest(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
