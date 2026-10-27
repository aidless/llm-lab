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
| `llm-lab serve` | Browser UI | Local evaluation workspace | `http://127.0.0.1:8123` |

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
