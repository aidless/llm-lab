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
read -r -a model_args <<< "${MODELS:-qwen3:0.6b llama3.2:1b}"
"$PY" -m agent_safety_eval run --models "${model_args[@]}" \
  --force "${review_args[@]}"
"$PY" -m agent_safety_eval report "${review_args[@]}"
"$PY" scripts/write_checksums.py
"$PY" verify_evidence.py --allow-pending-human
