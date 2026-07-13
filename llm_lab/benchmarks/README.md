# Running real benchmarks for `llm-lab`

The `self_bench.py` harness runs three scenarios:

1. **`smoke`** — fast offline benchmark (CI-gated, ~2 seconds). Uses an
   embedded stub LLM that produces deterministic responses. Verifies the
   harness is wired up correctly.
2. **`perf`** — latency / throughput measurement. By default, uses the
   same stub. With `--real`, hits a real LLM provider.
3. **`fault`** — graceful-degradation scenarios (provider timeout,
   missing API key, optional SDK missing, SQLite lock contention, audit
   chain clean / tamper / concurrent writers, first-time-POC cold start).

## Quick start

The smoke scenario requires **no LLM credentials** and is what CI runs:

```bash
cd F:\TMLR\llm_lab
python benchmarks/self_bench.py --mode all --steps 50 --real \
    --output benchmarks/v2-results.json
```

Takes about 2 seconds. Verifies the full benchmark pipeline including
JSON rendering, fault-scenario dispatch, and per-scenario assertions.

## Real-provider benchmarks

To run against a real LLM (for measuring **actual** latency, throughput,
cost — the numbers that go in your README or blog post):

```bash
export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, GEMINI_API_KEY

cd F:\TMLR\llm_lab
python llm_lab/benchmarks/self_bench.py --mode all --steps 50 --real \
    --output llm_lab/benchmarks/v2-results.json
```

This runs `perf` (50 steps against your provider), `fault` (always
offline), and writes a complete report to `v2-results.json`.

### What `--real` does

| Mode | LLM source | Token accounting | Cost |
|---|---|---|---|
| `smoke` (default) | embedded stub | approximate, not real | always $0.00 |
| `perf` without `--real` | embedded stub | approximate, not real | always $0.00 |
| `perf` with `--real` | **the configured LLM provider** | **real tokens** | **real cost** |

`--real` switches the provider from the local stub to whatever
`LLM_PROVIDER` (or `OPENAI_API_KEY`) configures. The benchmark then makes
50 sequential `chat.completions.create` calls and measures wall-clock
latency + token counts + cost.

### Cost expectations

- **`gpt-4o-mini`** at `--steps 50`: ~$0.10 - $0.30
- **`gpt-4o`** at `--steps 50`: ~$1.50 - $5.00
- **`claude-haiku-3.5`** at `--steps 50`: ~$0.20 - $0.50
- **`claude-opus`** at `--steps 50`: ~$3.00 - $15.00

> **Do not run with Opus for benchmark runs.** The numbers won't be
> better; the cost will be. Use the cheapest model in the family.

### What to do with the output

The output JSON has this shape:

```json
{
  "scenarios": [
    {
      "name": "perf",
      "steps": 50,
      "completed": 50,
      "throughput_steps_per_sec": 23.4,
      "latency_ms": {
        "p50": 850.2,
        "p95": 1823.5,
        "max": 2010.1
      },
      "tokens": {"total": 2847},
      "cost_usd": 0.12,
      "mode": "real"
    },
    ...
  ]
}
```

Three reasonable things to do with it:

1. **Update the project README** with your real numbers (see
   `llm_lab/README.md` — the "Self-benchmark numbers" section).
2. **Add to a blog post** showing real-world numbers. The data is
   reproducible — anyone running the same command against the same model
   at a similar load will get similar numbers.
3. **Track over time.** The output is git-friendly JSON; commit
   `v2-results.json` with a date in the filename like
   `v2-results-2026-07-13.json` if you want a historical record.

### Re-running for comparison

Each run produces a new `v2-results.json`. To compare runs, save the
output with a timestamp:

```bash
python benchmarks/self_bench.py --mode all --steps 50 --real \
    --output benchmarks/v2-results-$(date +%Y-%m-%d).json
```

## What's NOT in `self_bench.py`

- **Cost in different currencies** — only USD.
- **Concurrent request rate** — the benchmark is sequential, not
  parallel. For load testing, use a separate tool (e.g. `locust`).
- **Streaming responses** — all calls are non-streaming.
- **Other providers** — `openai` only by default. To benchmark
  `anthropic` or `gemini`, set the appropriate env var and edit
  `self_bench.py` to call the relevant function.

## Troubleshooting

- **`AuthenticationError`**: `OPENAI_API_KEY` is set but invalid.
  Generate a new one at https://platform.openai.com/api-keys.
- **`ModuleNotFoundError: No module named 'openai'`**: `pip install -e ".[dev]"`
  (or just `pip install openai`). The harness imports `openai` lazily
  only when `--real` is set.
- **`429 Too Many Requests`**: your account is rate-limited. Lower
  `--steps` or add a sleep between calls.
- **Tests pass locally but fail in CI**: the sandbox has no outbound
  network. The harness automatically falls back to the stub when
  `OPENAI_API_KEY` is `sk-test` (the default in CI). Don't set a real
  key in CI unless you also accept the cost.

## See also

- `llm_lab/observability.py` — the metrics store the benchmark writes to
- `docs/MASTER-PLAN.md` — phase 1 trigger: "real benchmark report replacing
  the stub numbers with measurements from a real LLM provider"
- `docs/ROADMAP-v0.10.0.md` — v0.10.0 bar: "Real benchmark report
  (`benchmarks/v2-results.json`) replacing the stub numbers with
  measurements from a real LLM provider"
