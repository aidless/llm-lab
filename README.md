# llm-lab

**Local-first LLM evaluation workbench.** Plan a task, call a model, verify the
output, and inspect a hash-linked audit trail without sending run data to a
hosted evaluation service.

[![CI](https://github.com/aidless/llm-lab/actions/workflows/test.yml/badge.svg)](https://github.com/aidless/llm-lab/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-1f6f5f)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-c4552d)](LICENSE)

The repository also contains a large, separately indexed AI/AGI knowledge
library under [`knowledge/`](knowledge/README.md). The product code and the
study notes are intentionally kept apart.

## Start with the offline demo

No API key and no network access required:

```bash
python3 -m venv .venv
.venv/bin/pip install -e './llm_lab[dev]'
.venv/bin/llm-lab demo
```

The command writes:

```text
.llm-lab-demo/
├── audit.db      # SQLite event log with a SHA-256 hash chain
├── result.json   # plan, model output, verdicts, token counts, cost
└── report.html   # standalone inspection report
```

It exits non-zero if any verification step fails or the audit chain is broken.
The deterministic offline provider reports `$0.000000` cost and exercises the
full planner → model adapter → verifier → audit path.

**What this proves:** the evaluation and audit plumbing is reproducible.

**What it does not prove:** model quality, benchmark leadership, production
reliability, or user adoption. The offline provider emits a fixed,
content-addressed response; it is a test fixture, not an LLM.

## What you can inspect

| Surface | Input | Output | Evidence |
|---|---|---|---|
| `llm-lab demo` | A deterministic evaluation goal | JSON result + standalone HTML report | `.llm-lab-demo/audit.db`, exit code |
| `llm-lab run` | Task goal + provider configuration | Per-step verdict, token usage, cost | Hash-linked SQLite event log |
| `llm-lab compare` | One goal + two models | A/B pass status, token and cost delta | Two traceable run IDs |
| `llm-lab evalops run` | Frozen batch task set + provider config | Per-task records + JSON/MD/HTML reports | Task/config/answer hashes |
| `llm-lab serve` | Browser UI | Local evaluation workspace | `http://127.0.0.1:8123` |

## Auditable batch evaluation

`llm-lab evalops` freezes a task set before execution and carries its hash
through every run.  A run records one final result per task, classifies
provider and format failures separately, and supports bounded concurrency,
retry/backoff, and a drift-checked resume.

```bash
.venv/bin/llm-lab evalops verify evalops/ceval-201
.venv/bin/llm-lab evalops run evalops/ceval-201/tasks-v1.jsonl \
  --output runs/ceval-201-c8 \
  --adapter ollama \
  --model qwen25-3b-16k:latest \
  --concurrency 8
```

The checked-in
[EvalOps evidence package](evalops/evidence-20261027/README.md) contains raw
records and reports for a 201-question public C-Eval validation subset,
concurrency 1/4/8 runs, synthetic fault injection, and an independent verifier.
Those are local engineering results, not an official leaderboard result or a
production SLO.  vLLM and the external OpenCompass runner were not executed.

## Portfolio evidence projects

> **本地安全评测，不是生产安全系统。 / Local safety evaluation, not a production security system.**

These packages extend EvalOps into auditable safety and evaluation-engineering
workflows. Each keeps raw records, JSON/Markdown/HTML reports, a SHA-256
manifest and an independent verifier. Human-review and GPU gates are never
silently converted into passes.

| Project | Evidence | Current gate |
|---|---|---|
| [Agent Safety Runtime Monitor](agent-safety-eval/README.md) | 96 mock-only tasks, two local models, policy-off/on trajectories | `HOLD` until at least 100 human-review records and required unsafe trajectories are reviewed |
| [Safety Judge Calibration](safety-judge-calibration/README.md) | 200 frozen requests, two local judges, FPR/FNR, Wilson intervals, per-slice results | `HOLD` until all 200 reference labels have human provenance |
| [Benchmark Contamination Preflight](benchmark-contamination-scan/README.md) | 201 C-Eval tasks, seeded corpus, exact/n-gram/semantic matches, fail-closed EvalOps gate | `HOLD` until 40 human reviews; current non-PASS gate exits `2` |
| [Eval Infra Adapters](eval-infra-adapters/README.md) | vLLM/LMDeploy contract smoke, real Ollama probe, existing 201-task CPU `C=1/4/8` evidence | GPU experiments `BLOCKED_ENVIRONMENT`; no production throughput claim |
| [Kube Eval Service](kube-eval-service/README.md) | Fresh kind cluster, bounded HTTP service, 24-request concurrency smoke, pod replacement | Local single-node CPU smoke only |

The main safety project is deliberately separate from `agi_harness.py`. All
dangerous actions run against an in-memory mock filesystem and mock outboxes;
no real file, shell, email or network effect is performed.

Verify every checked-in package without rerunning a model or cluster:

```bash
./scripts/verify_portfolio.sh
```

## Architecture

```text
goal
  │
  ▼
planner ──► runner ──► model adapter ──► verifier
  │              │                         │
  │              └── token + cost          │
  ▼                                        ▼
plan template                         pass / fail
  │                                        │
  └──────────────► tracer ◄────────────────┘
                     │
                     ▼
          SQLite hash-linked audit log
```

- **Planner:** chooses a known YAML template or falls back to a model-generated
  plan.
- **Runner:** executes each step through a provider adapter and records usage.
- **Verifier:** checks non-empty output, structure, keywords, schemas, or
  optional DeepEval metrics.
- **Tracer:** writes a per-row `prev_hash` / `row_hash` chain so deletion,
  insertion, or mutation of historical rows is detectable.

## Run with a model

The offline demo is the supported first-run path. To evaluate a real model,
copy the environment template and configure a provider:

```bash
cp llm_lab/.env.example .env
# Edit .env: LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY
.venv/bin/llm-lab run "review this pull request for security issues"
```

Real calls can incur provider charges. Use a small, explicit budget and do not
put credentials in the repository.

## Development checks

```bash
# Product package
cd llm_lab
../.venv/bin/ruff check .
../.venv/bin/mypy .
../.venv/bin/pytest --timeout=30 -q

# Repository-root self-evolution suite
cd ..
.venv/bin/pytest tests/test_selfevolve.py -q --timeout=30
```

## Project layout

```text
llm_lab/              # installable product package + tests
tests/                # repository-root self-evolution tests
selfevolve/           # experimental self-evolution modules
evalops/              # frozen task sets + verified local evidence
agent-safety-eval/    # mock-only agent safety trajectories and review gate
safety-judge-calibration/ # judge FPR/FNR calibration and HOLD conditions
benchmark-contamination-scan/ # contamination preflight and EvalOps gate
eval-infra-adapters/  # vLLM/LMDeploy contracts and CPU evidence boundary
kube-eval-service/    # local kind evaluation-service smoke
knowledge/            # AI/AGI notes, discussions and reference documents
```

The self-evolution code is retained as research context and is not part of the
offline evaluation demo's quality claim. See [`knowledge/README.md`](knowledge/README.md)
for the archived document index.

## Governance and security

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Threat model](THREAT_MODEL.md)
- [Governance](GOVERNANCE.md)
- [Support](SUPPORT.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

## License

MIT
