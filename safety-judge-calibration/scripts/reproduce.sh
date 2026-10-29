#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=../.venv/bin/python
RUN_ROOT="${JUDGE_REPRODUCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/safety-judge-calibration.XXXXXX")}"
EVIDENCE="${EVIDENCE_DIR:-$RUN_ROOT/evidence}"
REVIEW_OUTPUT="${REVIEW_DIR:-$RUN_ROOT/review}"
CASES="${CASES:-../agent-safety-eval/dataset/judge-cases-v1.jsonl}"
VARIANTS="${VARIANTS:-baseline,authority,verbose,order_prefix,order_suffix}"
REPEATS="${REPEATS:-2}"
WORKERS="${WORKERS:-6}"
BASE_URL="${BASE_URL:-http://127.0.0.1:11434}"
VARIANT_MAX_CASES="${VARIANT_MAX_CASES:-0}"

"$PY" -m pytest -q tests
"$PY" -m judge_calibration run \
  --cases "$CASES" \
  --output "$EVIDENCE" \
  --review-output "$REVIEW_OUTPUT" \
  --judges "${JUDGES:-qwen2.5:3b-instruct-q4_K_M llama3.2:3b}" \
  --base-url "$BASE_URL" \
  --variants "$VARIANTS" \
  --repeats "$REPEATS" \
  --workers "$WORKERS" \
  --variant-max-cases "$VARIANT_MAX_CASES"
"$PY" scripts/write_checksums.py --evidence "$EVIDENCE"
"$PY" verify_evidence.py \
  --cases "$CASES" \
  --evidence "$EVIDENCE" \
  --review-dir "$REVIEW_OUTPUT" \
  --judges ${JUDGES:-qwen2.5:3b-instruct-q4_K_M llama3.2:3b} \
  --variants $(echo "$VARIANTS" | tr ',' ' ') \
  --repeats "$REPEATS" \
  --variant-max-cases "$VARIANT_MAX_CASES" \
  --allow-pending-human

echo "Temporary evidence: $EVIDENCE"
echo "Human review packet: $REVIEW_OUTPUT/review.html"
