#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=../.venv/bin/python
RUN_ROOT="${JUDGE_REPRODUCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/safety-judge-calibration.XXXXXX")}"
EVIDENCE="${EVIDENCE_DIR:-$RUN_ROOT/evidence}"
REVIEW_OUTPUT="${REVIEW_DIR:-$RUN_ROOT/review}"
CASES="${CASES:-../agent-safety-eval/dataset/judge-cases-v1.jsonl}"

"$PY" -m pytest -q tests
"$PY" -m judge_calibration run \
  --cases "$CASES" \
  --output "$EVIDENCE" \
  --review-output "$REVIEW_OUTPUT" \
  --judges "${JUDGES:-qwen3:0.6b llama3.2:1b}"
"$PY" scripts/write_checksums.py --evidence "$EVIDENCE"
"$PY" verify_evidence.py \
  --cases "$CASES" \
  --evidence "$EVIDENCE" \
  --review-dir "$REVIEW_OUTPUT" \
  --allow-pending-human

echo "Temporary evidence: $EVIDENCE"
echo "Human review packet: $REVIEW_OUTPUT/review.html"
