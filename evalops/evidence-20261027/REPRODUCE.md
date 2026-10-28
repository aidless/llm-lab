# Reproducing the EvalOps evidence

The frozen task file, run metadata, raw records, reports, and environment
snapshot are included in this directory.  The main result is a local,
non-production evaluation of a public C-Eval validation subset.  It is not an
official leaderboard submission.

## 1. Verify the packaged evidence

```bash
cd evalops/evidence-20261027
python3 verify_evidence.py
```

The verifier checks the frozen task hash, per-run config and prompt hashes,
record counts, answer hashes, aggregate metrics, repeat agreement, fault
classes, and the optional `SHA256SUMS` file.  It exits non-zero on any
mismatch.

## 2. Recompute a run from the 201-task subset

```bash
cd "$(git rev-parse --show-toplevel)"
.venv/bin/llm-lab evalops run \
  evalops/ceval-201/tasks-v1.jsonl \
  --output /tmp/evalops-replay-c8 \
  --adapter ollama \
  --model qwen25-3b-16k:latest \
  --concurrency 8 \
  --timeout-seconds 90 \
  --max-retries 1 \
  --backoff-seconds 0.5 \
  --temperature 0 \
  --max-tokens 8 \
  --label evalops-replay-c8
```

The run metadata captures the exact config hash.  A compatible resume requires
matching config, task-set, and task-count hashes:

```bash
.venv/bin/llm-lab evalops run \
  evalops/ceval-201/tasks-v1.jsonl \
  --output /tmp/evalops-replay-c8 \
  --adapter ollama \
  --model qwen25-3b-16k:latest \
  --concurrency 8 \
  --timeout-seconds 90 \
  --max-retries 1 \
  --backoff-seconds 0.5 \
  --temperature 0 \
  --max-tokens 8 \
  --label evalops-replay-c8 \
  --resume
```

Sampling remains stochastic even at temperature zero because backend
execution is not bitwise deterministic.  A replay should reproduce the
contract and report shape, not necessarily every answer byte.

## 3. Run the contract tests

```bash
cd llm_lab
../.venv/bin/pytest --timeout=30 -q tests/test_evalops.py
```

This covers strict answer parsing, config drift rejection, retry/resume
semantics, atomic record replacement, JSON CLI output, concurrency bounds, and
all four injected failure classes.

## OpenCompass boundary

`dataset/opencompass/` is a compatibility export only.  OpenCompass was not
installed or executed for this evidence, so no external-runner validation is
claimed.  The same boundary applies to vLLM.
