#!/usr/bin/env python3
"""Check local Markdown links without executing code examples or needing network."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    ROOT / "README.md",
    ROOT / "llm_lab" / "README.md",
    *sorted((ROOT / "knowledge").glob("*.md")),
]
LINK_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))"
    r"(?:\s+[\"'][^)]*[\"'])?\s*\)"
)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def without_fenced_code(text: str) -> str:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = without_fenced_code(path.read_text(encoding="utf-8", errors="replace"))
    text = INLINE_CODE_RE.sub("", text)
    for match in LINK_RE.finditer(text):
        raw = (match.group(1) or match.group(2) or "").strip()
        if not raw or raw.startswith("#"):
            continue
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc:
            continue
        relative = unquote(parsed.path)
        if not relative:
            continue
        target = (path.parent / relative).resolve()
        try:
            target.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"{path.relative_to(ROOT)} -> {raw} (escapes repository)")
            continue
        if not target.exists():
            errors.append(f"{path.relative_to(ROOT)} -> {raw} (missing)")
    return errors


def main() -> int:
    errors: list[str] = []
    for source in SOURCE_FILES:
        if not source.is_file():
            errors.append(f"missing source file: {source.relative_to(ROOT)}")
            continue
        errors.extend(check_file(source))
    if errors:
        print("\n".join(f"FAIL  {error}" for error in errors))
        return 1
    print(
        f"PASS  checked {len(SOURCE_FILES)} Markdown files; "
        "all local file links resolve"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
