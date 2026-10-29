#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Agent Safety DPO Pipeline ==="
echo "Step 1: Extracting DPO triples..."
python3 -m dpo_data_builder build \
  --agent-safety-root ../agent-safety-eval \
  --output data/dpo-triples-v1.jsonl

echo "Step 2: Training DPO model..."
python3 -m dpo_trainer train \
  --model Qwen/Qwen3-0.6B-Instruct \
  --train-data data/dpo-triples-v1.jsonl \
  --output-dir checkpoints

echo "Step 3: Evaluating hybrid model..."
python3 -m dpo_eval run \
  --model checkpoints/final \
  --benchmark ../agent-safety-eval/dataset/tasks-v1.jsonl \
  --output evidence/report.md

echo "Step 4: Writing checksums..."
find evidence -type f -name "*.json" -o -name "*.md" | sort | while read f; do
  sha256sum "$f" >> evidence/SHA256SUMS
done

echo "Done. See evidence/report.md for results."
