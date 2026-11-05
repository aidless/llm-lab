#!/usr/bin/env python3
"""Fail if pytest --collect-only disagrees with docs/test-inventory.json.

The checked field is ``collected``. Pass/skip totals depend on optional
extras and API keys, so they are recorded for humans and are not compared.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "test-inventory.json"
COLLECTED_RE = re.compile(r"(\d+) tests collected")


def collect(python: str, cwd: Path, pytest_args: list[str]) -> int:
    cmd = [python, "-m", "pytest", *pytest_args, "--collect-only", "-q"]
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        sys.stderr.write(f"pytest failed in {cwd}:\n{output}\n")
        raise SystemExit(1)
    matches = COLLECTED_RE.findall(output)
    if not matches:
        sys.stderr.write(f"no collection summary in {cwd}:\n{output}\n")
        raise SystemExit(1)
    return int(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated suite ids. Default: every suite in the inventory.",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    seen: set[str] = set()
    failed = False
    for suite in inventory["suites"]:
        suite_id = suite["id"]
        if wanted and suite_id not in wanted:
            continue
        seen.add(suite_id)
        cwd = (ROOT / suite["cwd"]).resolve()
        observed = collect(args.python, cwd, list(suite["pytest_args"]))
        expected = int(suite["collected"])
        if observed != expected:
            failed = True
            print(f"DRIFT {suite_id}: inventory={expected} collected={observed} cwd={cwd}")
        else:
            print(f"ok {suite_id}: collected={observed}")
    missing = wanted - seen
    if missing:
        print(f"unknown suite ids: {', '.join(sorted(missing))}")
        return 1
    if not seen:
        print("no suites checked")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
