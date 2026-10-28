#!/usr/bin/env python3
"""Prove that the EvalOps gate exits non-zero for the current report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-report", default=str(ROOT / "evidence" / "report.json"))
    args = parser.parse_args()
    report_path = Path(args.expected_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {"PASS": 0, "HOLD": 2, "BLOCK": 3}.get(str(report.get("decision")))
    if expected is None:
        raise SystemExit(f"unknown report decision: {report.get('decision')!r}")
    if expected == 0:
        raise SystemExit("current report is PASS; this script only proves a non-PASS gate fails closed")
    with tempfile.TemporaryDirectory(prefix="contamination-gate-") as temporary:
        output = Path(temporary) / "preflight.json"
        process = subprocess.run(  # noqa: S603 - fixed module and local report path
            [
                sys.executable,
                "-m",
                "contamination_scan",
                "gate",
                "--report",
                str(report_path),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != expected:
            sys.stderr.write(process.stdout)
            sys.stderr.write(process.stderr)
            raise SystemExit(f"gate exit {process.returncode}, expected {expected}")
        preflight = json.loads(output.read_text(encoding="utf-8"))
        if preflight.get("decision") != report.get("decision"):
            raise SystemExit("gate preflight decision does not match the report")
    print(f"PASS: {report.get('decision')} gate exited {expected} and wrote a fail-closed preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
