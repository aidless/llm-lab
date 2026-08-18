# llm-lab

> **Local-first LLM evaluation framework** with tamper-evident audit logs and security-first design.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-18%20suites-brightgreen)

---

## What is this?

A CLI + web tool for **running, verifying, and comparing LLM outputs** — entirely on your own machine.

```bash
# Run a single LLM task with automatic verification
llm-lab run "Write a haiku about Rust"

# Compare two models head-to-head
llm-lab compare "Write a haiku about Rust" --model-a gpt-4o --model-b claude-3-5-sonnet

# Launch the web UI
llm-lab serve  # → http://localhost:8123
```

Every run is traced, verified, and stored in a local tamper-evident audit log. You get pass/fail verdicts, cost tracking, and full reproducibility — without sending your data to anyone.

---

## Why llm-lab?

| Problem | llm-lab's answer |
|---------|-------------------|
| "Is my LLM actually doing the task correctly?" | Built-in Verifier checks output against criteria → pass/fail verdict |
| "Which model should I use?" | `compare` command runs A/B tests with cost + quality metrics |
| "Can I prove what happened in each run?" | Tamper-evident audit chain (SHA-16 hash-linked logs) |
| "Will my data stay private?" | Local-first: runs entirely on your machine, SQLite storage |
| "Can I reproduce a result?" | Full trace + seed + config stored per run |

---

## Architecture

```
┌─────────────┐     ┌───────────┐     ┌───────────┐     ┌──────────┐
│   Planner    │────→│  Runner   │────→│ Verifier  │────→│  Tracer  │
│ (task → plan)│     │ (LLM call)│     │ (check)   │     │ (audit)  │
└─────────────┘     └───────────┘     └───────────┘     └──────────┘
                          │                                   │
                    ┌─────┴─────┐                       ┌──────┴──────┐
                    │  Pricing  │                       │  SQLite DB  │
                    │ (cost)    │                       │ (history)   │
                    └───────────┘                       └─────────────┘
```

- **Planner**: Converts task descriptions into structured execution plans (YAML templates for code review, A/B compare, model eval, summarization, translation)
- **Runner**: Executes LLM calls with retry, timeout, and cost tracking
- **Verifier**: Checks LLM output against acceptance criteria → pass/fail
- **Tracer**: Records every step in a hash-linked audit chain (tamper-evident)
- **DB**: SQLite storage for full run history with reproducibility metadata

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env  # Add your LLM_API_KEY

# Run
llm-lab run "Summarize the key points of clean code"

# Compare models
llm-lab compare "Summarize the key points of clean code" \
  --model-a gpt-4o --model-b deepseek-chat

# Web UI
llm-lab serve

# View history
llm-lab history

# Export a result
llm-lab export <run-id>
```

### Docker

```bash
docker-compose up -d  # Starts llm-lab + web UI on :8123
```

---

## Key Features

- **📋 Task Templates**: Pre-built YAML templates for common LLM tasks (code review, A/B testing, model evaluation, summarization, translation)
- **🔍 Verification Engine**: Rule-based + DeepEval automated scoring with configurable thresholds
- **📝 Audit Chain**: SHA-16 hash-linked event log — every run is tamper-evident and reproducible
- **💰 Cost Tracking**: Per-run token count + pricing for cost-aware model selection
- **🖥️ Web UI**: Browser-based interface at `localhost:8123`
- **🐳 Docker Ready**: Full docker-compose deployment
- **📊 Observability**: Structured logging + Prometheus metrics (ADR-0008)
- **🔒 Security**: Threat model, security policy, SBOM (CycloneDX), auth support
- **🧪 Tests**: 18 test suites covering API, audit chain, auth, CLI, DB, e2e, export, LLM contract, observability, planner, runner, tracer, verifier, worker

---

## Project Structure

```
llm-lab/
├── llm_lab/              # Main package
│   ├── cli.py            # CLI entry point (run/compare/serve/history/export)
│   ├── runner.py         # LLM execution engine
│   ├── verifier.py       # Output verification
│   ├── tracer.py         # Audit trail
│   ├── db.py             # SQLite storage
│   ├── planner/          # Task planning + YAML templates
│   ├── tests/            # 18 test suites
│   └── docs/             # ADRs, blog posts, roadmap
├── selfevolve/           # Self-evolution modules
│   ├── core.py           # Evolution loop
│   ├── multi_agent.py    # Multi-agent coordination
│   └── self_modification.py  # Safe self-modification
├── agi_harness.py        # AGI harness integration
├── Dockerfile            # Container build
├── docker-compose.yml    # Full deployment
└── .github/              # CI workflows, issue templates, PR template
```

---

## Governance

This project follows open-source governance best practices:

- [ADOPTERS.md](ADOPTERS.md) — who uses llm-lab
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute
- [GOVERNANCE.md](GOVERNANCE.md) — decision-making process
- [SECURITY.md](SECURITY.md) — security policy
- [THREAT_MODEL.md](THREAT_MODEL.md) — threat model
- [CODEOWNERS](CODEOWNERS) — code ownership
- 9 Architecture Decision Records in `llm_lab/docs/adr/`

---

## License

MIT
