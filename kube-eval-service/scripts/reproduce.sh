#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
CLUSTER="eval-lab-smoke-$(date +%s)-$$"
EVIDENCE="$(mktemp -d "${TMPDIR:-/tmp}/kube-eval-evidence.XXXXXX")"
TOOLS="$REPO_ROOT/.tools/bin"

cleanup() {
  if [[ -x "$TOOLS/kind" ]]; then
    PATH="$TOOLS:$PATH" "$TOOLS/kind" delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "$ROOT"
KIND_CLUSTER_NAME="$CLUSTER" EVIDENCE_DIR="$EVIDENCE" ./scripts/kind-smoke.sh
"$REPO_ROOT/.venv/bin/python" verify_evidence.py --evidence "$EVIDENCE"

echo "Temporary evidence: $EVIDENCE"
echo "The checked-in evidence directory was not modified."
