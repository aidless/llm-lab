# ADR-0002: Two parallel LLM call paths — `worker.py` and `promptfoo_provider.py`

- **Status:** Accepted
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

`llm-lab` exposes two LLM-calling surfaces:

1. `llm_lab/worker.py` — the main pipeline. Multi-provider (OpenAI / Anthropic
   / Gemini / local), lazy SDK imports, used by `runner.py` and the FastAPI
   service.
2. `llm_lab/promptfoo_provider.py` — a YAML-config-driven path with SQLite
   caching and exponential-backoff retry, exposed at
   `POST /promptfoo/run` and used to **simulate how a promptfoo-style tool
   would invoke the same model**.

The duplication has been flagged repeatedly as a code-smell. This ADR records
why we keep the two paths separate despite the duplication.

## Decision drivers

- **Semantic purpose.** `promptfoo_provider` exists to behave the way
  promptfoo behaves — read `PROMPTFOO_CONFIG`, cache hits, retry on transient
  failures, surface a promptfoo-shaped result dict. Merging it into
  `worker.py` would require `worker.py` to gain cache + retry knobs that
  the main pipeline does not need.
- **Backwards compatibility.** External integrations (including a partner's
  internal tooling) call `POST /promptfoo/run` and rely on the exact response
  shape. Convergent refactors risk subtle shape changes that are hard to
  detect in tests.
- **Surface area.** `worker.py` handles 8 providers with provider-specific
  error paths. `promptfoo_provider` is OpenAI-compatible only. Sharing the
  provider matrix would expand `promptfoo_provider`'s scope and re-open
  compatibility risks.

## Considered options

1. **Keep two paths, extract a tiny shared core.** What we are doing now.
   Both paths go through `worker.build_openai_client(base_url, api_key)`
   for OpenAI client construction (single source of truth for that one
   primitive). Behaviour is identical; only call shape differs.
2. **Merge into a single configurable client.** One entry point with cache,
   retry, provider, and config flags. Cleaner code; loses the
   promptfoo-shaped response contract.
3. **Delete `promptfoo_provider`.** Simpler codebase. Loses the
   promptfoo-simulation use case and the partner integration that depends on
   it.

## Decision outcome

**Option 1 — keep the two paths; minimise duplication at the
client-construction primitive.**

We extract `build_openai_client(base_url, api_key)` as the single source of
truth for OpenAI-compatible client setup (already done in M1 governance push).
Both `worker._build_client` and `promptfoo_provider._build_client_from_config`
go through it. The rest of each path stays independent.

Future work (M2) is to add a **contract test suite** that runs the same
prompt through both paths and asserts the response *shape* matches on the
shared fields (model, finish_reason, token_usage structure), so that we can
evolve the two paths without diverging in user-visible ways.

### Consequences

**Positive:**

- Promptfoo simulation use case preserved.
- Backwards-compatible response shape on `POST /promptfoo/run`.
- Single source of truth for OpenAI client construction.

**Negative:**

- Some duplication remains (the `chat.completions.create` call, token
  parsing, cost estimation). We accept this in exchange for clarity.

**Neutral:**

- New providers added to `worker.py` are not automatically available in
  `promptfoo_provider` (intentional — the promptfoo path stays
  OpenAI-compatible to match promptfoo semantics).

## Validation

- `tests/test_promptfoo_provider.py` covers cache hits, retry, and config
  loading.
- M2 deliverable: `tests/test_llm_contract.py` — same-input, both paths,
  assert matching response shape on shared fields.

## Links / references

- `llm_lab/worker.py:14` — `build_openai_client` definition.
- `llm_lab/promptfoo_provider.py:79` — `_build_client_from_config` now
  delegates to the shared client builder.
- `llm_lab/main.py` — the `POST /promptfoo/run` route is the public surface.