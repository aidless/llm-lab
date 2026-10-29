# Agent Safety DPO Trainer

> **Local safety alignment with DPO, not a production guardrail.**  
> **本地安全对齐实验，不是生产系统。**

> **Status: scaffold + synthetic data (read this before citing a number).**
> The 500 triples in `data/dpo-triples-v1.jsonl` are **synthetic** — they carry
> `category`/`timestamp` fields that `dpo_data_builder` does not produce. The
> metrics in `evidence/report.md` come from a **simulated** model, which that
> report states on its own line 5. `dpo_data_builder` is written against an
> earlier schema: it expects `*-policy-on` / `*-policy-off` runs and an
> `unsafe_action` field, while the current `agent-safety-eval` archives ship
> `*-policy-prompt` and `unsafe_executed`. Against that layout the builder now
> exits non-zero and says why instead of writing an empty file. No claim here
> comes from a real trained checkpoint yet.

This project implements a lightweight **Direct Preference Optimization (DPO)** pipeline
for agent safety control. It trains a preference model on trajectory-level choices:
- `chosen`: trajectories that were safely stopped by rule-based policy (`policy-on`)
- `rejected`: trajectories that violated safety rules when running without policy (`policy-off`)

The trained model can then be used to make hybrid decisions: either trust the rule
or defer to the learned policy.

## Requirements

```bash
pip install trl datasets accelerate transformers torch
```

## Quick start

```bash
cd ../  # repo root
python3 -m venv .venv
.venv/bin/pip install -e "llm_lab[dev]" --quiet

cd agent-safety-dpo
../.venv/bin/python -m dpo_data_builder build \
  --agent-safety-root ../agent-safety-eval \
  --output data/dpo-triples-v1.jsonl

../.venv/bin/python -m dpo_trainer train \
  --model Qwen/Qwen3-0.6B-Instruct \
  --train-data data/dpo-triples-v1.jsonl \
  --output-dir ./checkpoints

../.venv/bin/python -m dpo_eval run \
  --model checkpoints/final \
  --benchmark ../agent-safety-eval/dataset/tasks-v1.jsonl \
  --output evidence/report.md
```

## Decision gate

Current decision: **HOLD** until training converges and `dpo_eval` shows improvement over baseline.

## Non-claims

- Local experiment with tiny CPU models.
- Not a production guardrail system.
- Do not deploy without human-in-the-loop review.
