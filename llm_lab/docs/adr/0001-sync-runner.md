# ADR-0001: Sync Runner + ThreadPoolExecutor, not full asyncio

- **Status:** Accepted
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

The core orchestration in `llm_lab/runner.py` runs a planner → LLM → verifier
loop over multiple steps and (optionally) multiple plan variants. Each LLM
call is a network round-trip to a remote provider, so concurrency matters.

Two natural shapes:

1. **Sync orchestration with a worker thread pool** (what we have).
2. **Full asyncio** with `asyncio.gather` over async LLM clients.

## Decision drivers

- **Library maturity.** At the time of writing, the OpenAI / Anthropic /
  google-genai Python SDKs are first-class sync APIs with async wrappers
  that are still labelled experimental or partially supported. The FastAPI
  ecosystem happily runs sync handlers in a thread pool.
- **Local SQLite.** `llm_lab/db.py` uses `aiosqlite` for the few async
  callers (FastAPI request handlers), but the bulk of the event log writes
  happen from the runner, which is sync. A sync runner lets us avoid bridging
  asyncio and sync code in the hot path.
- **Operational simplicity.** Sync code is easier to profile, easier to
  interrupt (Ctrl-C actually stops at a clear boundary), and easier to
  reason about under `pytest`.
- **Single-machine scope.** The runner does not need to coordinate across
  processes or hosts. ThreadPoolExecutor's GIL constraint is not on the
  critical path because the GIL is released during network I/O.

## Considered options

1. **Sync Runner + ThreadPoolExecutor** — orchestration is a plain `for` loop;
   concurrent steps run via `concurrent.futures.ThreadPoolExecutor`. Network
   I/O releases the GIL, so concurrency actually works.
2. **Full asyncio** — runner is `async def`, uses `httpx.AsyncClient` or async
   SDK variants, DB writes via `aiosqlite`. Cleaner in theory; pays the
   "async all the way down" tax everywhere.
3. **Multiprocessing** — bypass the GIL. Pays a serialisation cost on every
   step boundary; overkill for a single-machine eval tool.

## Decision outcome

**Option 1 — sync Runner + ThreadPoolExecutor.**

We get concurrency where it matters (network I/O), avoid the asyncio tax, and
keep the runner trivially testable. The price — no parallelism for CPU-bound
verifier work — is acceptable because verifier work is bounded and fast in the
common case.

### Consequences

**Positive:**

- `run_plan` is a plain function; trivial to call from CLI, tests, or a
  notebook.
- Network concurrency works via the executor's thread pool.
- Ctrl-C cleanly stops the runner.
- `pytest` works without any async-plugin gymnastics for the runner itself.

**Negative:**

- CPU-bound verifier work does not parallelise across cores.
- If we ever need 1000+ concurrent steps, we will hit the executor's default
  thread limit (32). Workaround: explicit `max_workers` argument.

**Neutral:**

- `db.py` remains async (`aiosqlite`) for FastAPI request handlers; the
  runner does not interact with the async DB path.

## Validation

- The `tests/test_runner.py` suite exercises concurrency explicitly. If we
  ever regress to serial execution, those tests will catch it.
- The `benchmarks/self_bench.py --mode perf` run produces a concurrency
  throughput number; we expect at least 4× speedup over serial for 8-step
  plans with a 1-second-per-step LLM.

## Links / references

- `llm_lab/runner.py:35` — ThreadPoolExecutor import and use.
- `tests/test_runner.py` — concurrency tests.
- Python docs: [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor).