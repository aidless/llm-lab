#!/usr/bin/env python3
"""Stop before EvalOps execution when the contamination preflight is not PASS."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLM_LAB_ROOT = ROOT.parent
PREFLIGHT = ROOT / "evidence" / "evalops-preflight.json"


def main() -> int:
    if not PREFLIGHT.exists():
        print(f"missing preflight: {PREFLIGHT}", file=sys.stderr)
        return 2
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if report.get("decision") != "PASS":
        print(
            f"EvalOps preflight is {report.get('decision')}: "
            f"{report.get('reasons')}",
            file=sys.stderr,
        )
        return 2
    process = subprocess.run(  # noqa: S603 - fixed local command
        [
            sys.executable,
            "-m",
            "llm_lab.cli",
            "evalops",
            "verify",
            str(LLM_LAB_ROOT / "evalops" / "ceval-201"),
        ],
        cwd=LLM_LAB_ROOT,
        check=False,
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
