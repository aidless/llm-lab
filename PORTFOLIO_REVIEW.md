# Portfolio Review: Safety and Evaluation Engineering

> **本地安全评测，不是生产安全系统。 / Local safety evaluation, not a production security system.**

Review date: 2026-10-29

This document is an evidence-based maturity review, not a marketing scorecard.
It separates what is unusual for a portfolio project, what is verified here,
and what remains a real research, human-label or infrastructure gap.

## Bottom line

These packages are advanced **evidence-first engineering prototypes**. They are
substantially stronger than notebook-only portfolio projects because every
headline result has raw records, a deterministic recomputation path, a
SHA-256 manifest and an explicit non-pass state.

They are not frontier safety research, production guardrails or official
benchmark results. The remaining gaps are not cosmetic: independent human
labels, real sandboxes, real GPU serving and external benchmark execution are
still required for those claims.

| Project | Engineering maturity | Strongest differentiator | Honest blocker |
|---|---|---|---|
| Agent Safety Runtime Monitor | Advanced local prototype | Four-arm intervention ablation with complete mock-tool trajectories | Human review remains 0/100+ |
| Safety Judge Calibration | Advanced local prototype | Repeat and presentation-sensitivity probes expose instability | Labels are assistant drafts; current judges fail the quality gate |
| Benchmark Contamination Preflight | Advanced pipeline prototype | Joint lexical/semantic evidence and fail-closed EvalOps gate | Seeded corpus is not a real training snapshot; 0/40 human reviews |
| Eval Infra Adapters | Contract-grade integration layer | Pinned OpenCompass export plus real process/HTTP smoke | No GPU; vLLM/LMDeploy/OpenCompass runtime is not executed |
| Kube Eval Service | Advanced single-node infrastructure smoke | Durable queue, idempotency, worker leases and restart recovery | kind/RWO/CNI evidence does not establish a production cluster |

## 1. Agent Safety Runtime Monitor

### What is advanced

- It separates prompt-only, runtime-gate-only and combined intervention effects
  instead of comparing one coarse `policy-off`/`policy-on` pair.
- It records every action, tool argument, policy decision, terminal state and
  stop reason against an in-memory environment.
- Unsafe attempts, executed violations, over-refusal and full-task recovery are
  distinct signals rather than one blended safety score.
- The independent verifier recomputes every run and rejects complete model-error
  runs, even when all aggregate rates conveniently look like zero.
- Dangerous file, shell, email and HTTP operations never touch the host.

### What is still missing

- No independent human has completed the 100+ review packets.
- The scenarios are synthetic and the tools are mocks.
- Two 3B CPU models are useful plumbing evidence, not a frontier safety ranking.
- There is no real sandbox-escape, browser, OS or production credential test.

### Verdict

This is the strongest project for Concordia-style agent safety engineering and
Shanghai AI Lab safety-agent roles **only after** the human-review gate is
completed. The architecture is already more defensible than a demo because it
can fail closed.

## 2. Safety Judge Calibration

### What is advanced

- It measures FPR, FNR, over-refusal, under-refusal, Wilson intervals, Brier,
  ECE and pairwise agreement on a frozen 200-case set.
- Repeats expose self-inconsistency rather than hiding it in one sampled label.
- Authority framing, verbose restatement and order-position perturbations
  estimate presentation sensitivity.
- The report does not convert a poor judge into a PASS. Current 3B judges have
  FPR of 61.5% and 73.6%, repeat inconsistency of 8.5%, and substantial
  presentation drift.
- Assistant draft labels are explicitly excluded from the human-ground-truth
  boundary.

### What is still missing

- All reference labels remain drafts until the 200-case human review is done.
- Two local 3B models do not represent frontier judges.
- The request distribution is synthetic and not production traffic.
- A true multi-judge adjudication protocol with a third tie-breaker and
  calibration on human labels is not yet available.

### Verdict

The most useful result is negative: the current judges are not deployable under
the declared thresholds. That is a credible safety-engineering result, but it
still needs human labels before it can support a job-facing calibration claim.

## 3. Benchmark Contamination Preflight

### What is advanced

- Exact, character n-gram and semantic evidence are retained separately. A
  semantic hit in one document can no longer erase an exact hit in another.
- Threshold selection is dev-only and test metrics include Wilson intervals.
- The EvalOps integration is intentionally fail-closed and a gate check proves
  the non-zero exit path.
- The scanner and independent verifier both recompute the joint evidence rule.

### What is still missing

- The checked-in corpus is a seeded audit fixture, not a pretrained-model
  training-data snapshot.
- The current 1.0 precision/recall measures the fixture pipeline, not real
  contamination detection.
- The 40-record human review queue is still pending.
- Semantic similarity can confuse topic overlap with contamination.

### Verdict

This is credible evidence for evaluation-data quality and preflight engineering.
It should not be described as proving contamination in a named model.

## 4. Eval Infra Adapters

### What is advanced

- One OpenAI-compatible contract spans Ollama, vLLM and LMDeploy labels.
- Command generation, subprocess lifecycle, readiness, OOM and HTTP failures are
  recorded rather than mocked away.
- The OpenCompass 0.5.4 export is pinned to an official source/interface
  contract and includes the existing frozen 201-task subset.
- CPU smoke and GPU acceptance are separate states, with GPU explicitly marked
  `BLOCKED_ENVIRONMENT`.

### What is still missing

- No NVIDIA GPU exists on either the workstation or the remote CPU host used for
  the safety sweeps.
- vLLM, LMDeploy and OpenCompass inference were not executed.
- There is no real C=1/4/8 GPU latency, p95, OOM or throughput comparison.
- A fake OpenAI-compatible fixture is a contract test, not a serving backend.

### Verdict

This closes the interface-design gap, but not the JD's real GPU-serving gap.
It is strong evidence of knowing where the boundary is and refusing to fake it.

## 5. Kube Eval Service

### What is advanced

- API and worker processes share a SQLite-backed durable queue on a PVC.
- Idempotency keys, worker leases, bounded attempts, atomic cancellation and
  terminal-state-wins behavior are implemented and tested.
- A real kind smoke force-deletes the worker and verifies both an interrupted
  running job and a queued job recover.
- Security context, probes, resource limits, NetworkPolicy, PVC and PDB are
  checked from live Kubernetes objects.

### What is still missing

- The evidence is from one kind node with a `ReadWriteOnce` PVC.
- The kind CNI may not enforce NetworkPolicy; only the policy structure is
  verified.
- The PDB and single-replica setup are not a high-availability design.
- There is no multi-node scheduling, autoscaling, upgrade, backup or GPU test.

### Verdict

This is a strong infrastructure-plumbing portfolio project. It should be framed
as durable single-node acceptance evidence, not production Kubernetes
operations.

## Advancedness assessment

The portfolio is advanced in four dimensions:

1. **Causal experiment design** - separate safety interventions instead of one
   aggregate policy switch.
2. **Evidence integrity** - raw trajectories, frozen hashes, independent
   recomputation, exact checksum coverage and fail-closed gates.
3. **Adversarial evaluation hygiene** - judge repeats, prompt-presentation
   perturbations and contamination preflight.
4. **Operational failure handling** - leases, idempotency, restart recovery and
   explicit CPU/GPU and local/production boundaries.

It is not advanced in the dimensions that require external truth: human
annotations, real sandbox effects, real GPU throughput and official benchmark
execution.

## Release gate

Before describing either safety package as complete:

1. A human must label all 96 Agent Safety task labels, all required unsafe
   trajectories and enough records to exceed 100 total.
2. A human must label all 200 judge-calibration cases.
3. A human must review the 40 contamination queue records.
4. The resulting gates must be recomputed from raw records, not hand-edited.
5. Any future GPU claim needs a real NVIDIA host, immutable versions and
   verifiable C=1/4/8 evidence.

Until then, the correct public state is `HOLD` for human review and
`BLOCKED_ENVIRONMENT` for GPU execution.
