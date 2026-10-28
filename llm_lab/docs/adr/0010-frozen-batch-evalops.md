# ADR-0010: Frozen-then-run batch EvalOps

- **Status:** Implemented
- **Date:** 2026-10-27
- **Deciders:** @llm-lab/maintainers

## Context

The planner/verifier path is useful for one goal at a time, but evaluation and
benchmark-engineering work needs a different contract:

- the task set must be frozen before seeing model outputs;
- provider failures must be distinguishable from answer-format failures;
- concurrency must not change the meaning of the denominator;
- a failed task must be resumable without duplicate records;
- a third party must be able to recompute the reported aggregate;
- local subset results must not be confused with an official leaderboard.

An append-only JSONL file alone does not satisfy the resume requirement: a
retry can append a second row for the same task, after which readers either
pick the wrong row or reject the file as ambiguous.

## Decision

Introduce a separate `llm_lab.evalops` module and `llm-lab evalops` command
group with the following contract.

1. `build` validates source rows against an authoritative table and writes a
   canonical JSONL task set plus a SHA-256 manifest.
2. Every run records the effective `EvalConfig` hash and task-set hash in
   `run-meta.json`.
3. A task produces exactly one logical final record.  Resume validates the
   metadata, config hash, task-set hash, task count, per-record config, and
   prompt hash before reading or changing results.
4. Retry results atomically replace the prior record for that task.  Readers
   reject malformed or duplicate final records instead of silently choosing
   one.
5. The runner classifies `timeout`, `transport`, `model_error`, `parse_error`,
   and `evaluator_error`; strict accuracy includes failures in the
   denominator, while parsed accuracy uses successful parsed answers.
6. Reports include task-set, config, and answer hashes.  Timings are excluded
   from the answer hash so concurrency changes can be compared.
7. The OpenCompass output is labelled a compatibility export unless an
   external runner is actually installed and executed.

## Consequences

Positive:

- A failed task can be retried without corrupting the record stream.
- Configuration or dataset drift is rejected before a resume.
- Results are machine-readable and independently recomputable.
- Local engineering evidence can be published without implying benchmark
  leadership or production capacity.

Negative:

- Atomic replacement rewrites the record file after each completion.  This is
  simple and crash-safe, but not designed for multi-host writers or
  million-row throughput.
- Temperature zero does not guarantee bitwise-identical model outputs across
  backend executions.  Repeat agreement is a consistency measurement, not a
  determinism proof.
- The C-Eval subset remains a public validation sample.  It is not the
  official test set and cannot support an official leaderboard claim.

## Evidence

The checked-in `evalops/evidence-20261027` package contains a 201-task local
Ollama run at concurrency 1/4/8, synthetic fault records, deterministic fixture
runs, raw per-task outputs, reports, an environment snapshot, and an
independent verifier.  The evidence README states all non-claims, including
that vLLM and OpenCompass were not executed.
