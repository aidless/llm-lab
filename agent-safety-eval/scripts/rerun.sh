#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ../.venv/bin/python ]]; then
  PY=../.venv/bin/python
else
  PY=python3
fi

if [[ ! -f dataset/manifest.json ]]; then
  "$PY" -m agent_safety_eval build --output dataset --force
fi

"$PY" -m agent_safety_eval verify-dataset --dataset dataset
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
WORKERS="${WORKERS:-3}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-180}"

archive_generated() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    local archive_root="../.codex-evidence"
    mkdir -p "$archive_root"
    local destination="$archive_root/${label}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv "$path" "$destination"
    printf 'Archived %s to %s\n' "$path" "$destination" >&2
  fi
}

# Preserve both pending and completed human-review packets before a full rerun.
archive_generated evidence agent-safety-evidence
archive_generated review agent-safety-review

review_args=()
if [[ -f review/reviews-human.jsonl ]]; then
  review_args=(--reviews review/reviews-human.jsonl)
fi
read -r -a model_args <<< "${MODELS:-qwen2.5:3b-instruct-q4_K_M llama3.2:3b}"
"$PY" -m agent_safety_eval run --models "${model_args[@]}" \
  --workers "$WORKERS" --timeout "$OLLAMA_TIMEOUT" \
  --base-url "$BASE_URL" --force "${review_args[@]}"
"$PY" -m agent_safety_eval report "${review_args[@]}"
"$PY" scripts/write_checksums.py
"$PY" verify_evidence.py --allow-pending-human
