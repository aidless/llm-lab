#!/usr/bin/env python3
"""Write a deterministic checksum manifest for Kubernetes smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {"SHA256SUMS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(ROOT / "evidence"))
    args = parser.parse_args()
    evidence = Path(args.evidence)
    files = [
        path for path in evidence.rglob("*") if path.is_file() and path.relative_to(evidence).as_posix() not in EXCLUDED
    ]
    lines = [f"{sha256(path)}  {path.relative_to(evidence).as_posix()}" for path in sorted(files)]
    (evidence / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums")


if __name__ == "__main__":
    main()
