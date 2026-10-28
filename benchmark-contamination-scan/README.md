# Benchmark Contamination Preflight

> A clean scan result is not proof that a model was never exposed to a benchmark.
>
> **本地安全评测，不是生产安全系统。 Local safety evaluation, not a production security system.**

This package scans a frozen C-Eval subset against a candidate corpus using:

- normalized exact matching;
- 13-character n-gram containment;
- local `nomic-embed-text` semantic similarity;
- dev-split-only threshold tuning;
- test-split precision, recall, false-positive rate, false-negative rate and
  Wilson 95% intervals;
- an EvalOps-compatible `PASS` / `HOLD` / `BLOCK` preflight.

It reads the existing llm-lab frozen 201-task C-Eval validation subset:

```text
../evalops/ceval-201/tasks-v1.jsonl
```

## Quick start

```bash
../.venv/bin/python -m contamination_scan build-fixture
../.venv/bin/python -m contamination_scan scan
../.venv/bin/python -m contamination_scan gate
```

`scan` writes JSON, Markdown and HTML reports, raw matches, embedding vectors,
and a 40-record human-review queue.

Recompute from checked-in raw matches, write the SHA-256 manifest and verify the
whole package:

```bash
make reproduce
```

`verify_evidence.py` independently recomputes dev-only threshold selection,
test confusion counts, precision/recall, FPR/FNR and Wilson intervals. It also
checks the exact checksum coverage and the EvalOps preflight decision.

## Important fixture boundary

The checked-in evidence uses a **seeded audit corpus**, not a model's actual
pretraining snapshot. Some C-Eval items are intentionally copied into the
candidate corpus so the detector can be measured. Metrics therefore validate
the engineering pipeline, not contamination in a named production model.

For a real audit, replace `dataset/corpus-v1.jsonl` with a documented training
corpus snapshot, remove the synthetic gold labels, and preserve the corpus
revision and SHA-256 before scanning.

## EvalOps integration

`python -m contamination_scan gate` writes:

```text
evidence/evalops-preflight.json
```

It exits:

- `0` for `PASS`;
- `2` for `HOLD`;
- `3` for `BLOCK`.

The Make target is intentionally fail-closed for the current `HOLD` evidence:

```bash
make gate          # exits non-zero
make gate-check    # proves the subprocess returned the declared non-zero code
```

There is no `|| true` escape hatch.

The companion wrapper demonstrates how an EvalOps run should stop before model
execution:

```bash
../.venv/bin/python scripts/evalops_gated.py
```

## Human review

`review/review-queue.json` contains 40 suspects and controls. Each record
requires an explicit contamination label, reviewer ID and UTC timestamp.

```bash
../.venv/bin/python -m contamination_scan ingest-review \
  --export review/review-export.json
../.venv/bin/python -m contamination_scan report \
  --human-reviews review/reviewed.jsonl
```

## Gate defaults

The current gate is `HOLD` unless all of these are true:

- at least 40 validated human review records;
- test precision and recall are each at least 0.85;
- the false-positive Wilson upper bound is at most 0.12;
- the false-negative Wilson upper bound is at most 0.30.

It becomes `BLOCK` if recall falls below 0.70 or the false-negative Wilson upper
bound exceeds 0.35. The Wilson bound is also shown as an uncertainty warning
because the seeded test split has only 30 positives.

## Non-claims

- The local corpus is a seeded fixture, not the model's training data.
- Exact matching is not a proof of memorization.
- Semantic similarity can produce topic-level false positives.
- The upstream C-Eval subset remains `CC-BY-NC-SA-4.0`; see EvalOps NOTICE.
- No upstream benchmark-contamination code was copied. Public projects such as
  PaCoST and LRM_Conta_Detection_Arena informed the review structure only.

License: MIT. See the repository-level `LICENSE`.
