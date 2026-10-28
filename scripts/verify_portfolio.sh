#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

cd "$ROOT"

"$PY" agent-safety-eval/verify_evidence.py --allow-pending-human
"$PY" safety-judge-calibration/verify_evidence.py --allow-pending-human
"$PY" benchmark-contamination-scan/verify_evidence.py --allow-pending-human
(
  cd benchmark-contamination-scan
  "$PY" scripts/assert_gate_fails_closed.py --expected-report evidence/report.json
)
"$PY" eval-infra-adapters/verify_evidence.py
"$PY" kube-eval-service/verify_evidence.py

echo "All checked-in portfolio evidence verifiers passed."
echo "Human-review and GPU gates remain explicit and may still be pending."
