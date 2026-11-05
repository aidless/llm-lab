#!/usr/bin/env python3
"""Reject stale status claims and author-machine paths on the public surface.

Current-status documents must not hard-code a pass count. The checked count
lives in docs/test-inventory.json. Historical notes (changelog, old reviews,
blog drafts) are allowlisted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Documents a reviewer treats as the current status of the project.
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "llm_lab" / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "llm_lab" / "docs" / "ARCHITECTURE.md",
    ROOT / "ADOPTERS.md",
)

FORBIDDEN_BADGES = (
    "img.shields.io/badge/tests",
    "actions/workflows/codeql.yml/badge",
    "img.shields.io/pypi",
    "badge.fury.io/py",
    "img.shields.io/github/v/release",
)

AUTHOR_PATHS = (
    re.compile(r"[A-Za-z]:\\Users", re.IGNORECASE),
    re.compile(r"[A-Za-z]:/Users", re.IGNORECASE),
    re.compile(r"/c/Users/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\TMLR", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\fortune", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\test\\", re.IGNORECASE),
)

PASS_COUNT = re.compile(r"\b\d+\s+passed\b", re.IGNORECASE)


def problem_paragraph(text: str) -> list[str]:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("# "):
            start = index + 1
            break
    if start is None:
        return []
    while start < len(lines) and not lines[start].strip():
        start += 1
    paragraph: list[str] = []
    while start < len(lines) and lines[start].strip():
        paragraph.append(lines[start])
        start += 1
    return paragraph


def tracked_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(".git/") or relative == "tools/check_honesty_surface.py":
            continue
        if path.suffix.lower() not in {".md", ".py", ".ps1", ".sh", ".yml", ".yaml", ".txt", ".toml", ".json"}:
            continue
        files.append(path)
    return files


def main() -> int:
    errors: list[str] = []
    for path in (ROOT / "README.md", ROOT / "llm_lab" / "README.md"):
        paragraph = problem_paragraph(path.read_text(encoding="utf-8"))
        if not paragraph or len(paragraph) > 3:
            errors.append(f"{path.relative_to(ROOT)} problem statement is {len(paragraph)} lines; maximum is 3")
    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        for badge in FORBIDDEN_BADGES:
            if badge in text:
                errors.append(f"{path.relative_to(ROOT)} contains badge fragment {badge!r}")
        if path.name != "ADOPTERS.md" and PASS_COUNT.search(text):
            errors.append(f"{path.relative_to(ROOT)} hard-codes a pass count; point at docs/test-inventory.json")
    for path in tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in AUTHOR_PATHS:
            if pattern.search(text):
                errors.append(f"{path.relative_to(ROOT)} contains author path {pattern.pattern}")
                break
    if errors:
        print("\n".join(errors))
        return 1
    print("honesty surface ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
