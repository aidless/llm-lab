# llm-lab

Local-first LLM orchestration and evaluation lab. Run and compare LLM outputs across providers (OpenAI, Anthropic, Gemini, Ollama, vLLM, promptfoo), verify results with structural/heuristic checks or DeepEval, and trace everything to Langfuse or local SQLite.

## Quickstart

```bash
pip install -e ".[dev]"

# Copy .env.example → .env, set LLM_API_KEY, then:
llm-lab run "write a haiku about Rust"
llm-lab compare "write a haiku about Rust" --model-a gpt-4o --model-b claude-3-5-sonnet-20241022
llm-lab serve
```

## Architecture

```
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│  CLI     │   │  Web API     │   │  Compare     │
│ (typer)  │──▶│  (FastAPI)   │──▶│  Report HTML │
└──────────┘   └──────┬───────┘   └──────────────┘
                      │
               ┌──────▼───────┐   ┌──────────────┐
               │   Planner    │──▶│  Verifier    │
               │ (promptfoo)  │   │ (deepeval)   │
               └──────┬───────┘   └──────────────┘
                      │
               ┌──────▼───────┐   ┌──────────────┐
               │   Tracer     │──▶│  SQLite /    │
               │ (langfuse)   │   │  Langfuse    │
               └──────────────┘   └──────────────┘
```

**Modules** (all at `llm_lab/`):

| Module | Role |
|--------|------|
| `cli.py` | Typer CLI: `run`, `compare`, `serve`, `history`, `export` |
| `main.py` | FastAPI app with async submit, compare, batch, history endpoints |
| `planner/` | Eval-plan package (`engine.py`): generates plan for a goal, dispatches to LLM providers |
| `verifier.py` | Checks outputs with heuristics + optional DeepEval semantic eval |
| `tracer.py` | Records runs and traces to local SQLite or Langfuse |
| `db.py` | SQLite layer: intents, events, history queries |
| `runner.py` | Run orchestration for single / compare / batch eval plans |
| `worker.py` | LLM provider calls (OpenAI / Anthropic / Gemini) + background tasks |
| `promptfoo_provider.py` | Provider wrapper using promptfoo for local model inference |
| `settings.py` | Environment-based configuration and provider presets |
| `pricing.py` | Token / cost pricing helpers |
| `models.py` | Pydantic request / response models |

## CLI

| Command | Description |
|---------|-------------|
| `run <goal>` | Run a single eval plan |
| `compare <goal>` | A/B compare two models |
| `serve` | Start web UI on port 8123 |
| `history` | List recent runs |
| `export <intent_id>` | Export run as JSON |

Options: `--model`, `--verifier`, `--json`, `--dry-run`.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/submit` | Run sync eval |
| POST | `/submit/async` | Run async (poll `/status/{id}`) |
| POST | `/compare` | A/B compare sync |
| POST | `/batch` | Multi-model eval |
| GET | `/history` | Recent runs |
| GET | `/result/{id}` | Run summary |
| GET | `/trace/{id}` | Full trace |
| GET | `/export/json/{id}` | JSON download |
| GET | `/export/csv` | CSV download |

## Configuration

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | Default API key |
| `LLM_PROVIDER` | `openai` | Provider: openai, anthropic, gemini, ollama, vllm, llamacpp, localai, tgi, promptfoo |
| `LLM_MODEL` | `gpt-4o` | Default model name |
| `LLM_BASE_URL` | per-provider | API base URL |
| `ANTHROPIC_API_KEY` | — | Anthropic-specific key |
| `GEMINI_API_KEY` | — | Gemini-specific key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse tracing (optional) |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse tracing (optional) |
| `DEEPEVAL_ENABLED` | `false` | Enable DeepEval metrics |
| `DEEPEVAL_THRESHOLD` | `0.5` | Pass threshold |

Suffixed vars (`LLM_API_KEY_2`, `LLM_BASE_URL_2`) access a second provider for comparisons.

## Docker

```bash
docker compose up --build
```

Opens at `http://localhost:8123`. The web UI runs at `/static/ui.html`.

## Testing

```bash
# Full test suite with coverage
pytest --cov --cov-report=term-missing

# Lint & type check
ruff check .
mypy .
```

## Project Structure

```
llm_lab/
├── cli.py              # Typer CLI commands
├── main.py             # FastAPI application
├── db.py               # SQLite layer
├── tracer.py           # Run tracing (SQLite / Langfuse)
├── runner.py           # Eval plan run orchestration
├── worker.py           # LLM provider calls (OpenAI / Anthropic / Gemini)
├── verifier.py         # Output verification
├── settings.py         # Configuration & provider presets
├── pricing.py          # Cost / pricing helpers
├── models.py           # Pydantic models
├── promptfoo_provider.py  # promptfoo wrapper
├── planner/            # Eval-plan package
│   ├── engine.py       # Plan generation & dispatch
│   └── templates/      # Plan prompt templates
├── templates/          # HTML templates
│   ├── ui.html         # Single-file web dashboard
│   └── compare_report.html  # A/B compare report
├── static/             # Static assets (web UI)
└── tests/              # Test suite (pytest, CI-gated)
    ├── conftest.py
    ├── test_api.py
    ├── test_cli.py
    ├── test_db.py
    ├── test_e2e_smoke.py
    ├── test_export.py
    ├── test_planner.py
    ├── test_promptfoo_provider.py
    ├── test_runner.py
    ├── test_settings.py
    ├── test_tracer.py
    ├── test_verifier.py
    └── test_worker.py
```
