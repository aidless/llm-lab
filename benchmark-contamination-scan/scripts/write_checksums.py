#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for contamination evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
EXCLUDED = {"SHA256SUMS", "verification-result.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = [
        path
        for path in EVIDENCE.rglob("*")
        if path.is_file() and path.relative_to(EVIDENCE).as_posix() not in EXCLUDED
    ]
    lines = [f"{sha256(path)}  {path.relative_to(EVIDENCE).as_posix()}" for path in sorted(files)]
    (EVIDENCE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {EVIDENCE / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
