# Threat Model — `llm-lab` v0.x

This document describes **what `llm-lab` defends against**, **what it does not**,
and **the assumptions we make about the deployment environment**.

It is intentionally explicit about non-goals. Saying "we don't defend against X"
is more useful than implying we do.

---

## System overview

```
                         ┌──────────────────────────────┐
                         │       Trust boundary         │
   Operator ────────────►│  ┌────────────────────────┐  │
   (CLI / HTTP)          │  │   llm-lab process      │  │
                         │  │  (FastAPI + runner +   │  │
   Tenant A ─────────────┼─►│   SQLite + workers)    │  │
   Tenant B ─────────────┼─►│                        │  │
                         │  └────────────────────────┘  │
                         │             │                │
                         │             ▼                │
                         │      ┌──────────────┐        │
                         │      │ LLM provider │        │
                         │      │  (OpenAI /   │        │
                         │      │  Anthropic / │        │
                         │      │  local)      │        │
                         │      └──────────────┘        │
                         └──────────────────────────────┘
```

`llm-lab` runs as a **single-process service** (FastAPI + CLI). It uses local
SQLite for state and calls external LLM providers over HTTPS.

---

## In scope — what we defend against

### S1. Path traversal in user-supplied identifiers

- **Threat:** attacker supplies `..`-laden or otherwise malicious IDs to read
  / write files outside the intended directory.
- **Defence:** `intent_id` is matched against a strict regex (`_ID_RE`) in
  `main.py`; path parameters go through `validate_path_param`.
- **Tests:** `tests/test_auth.py` covers invalid inputs.
- **Status:** ✅ Hardened (audit fix #2).

### S2. Reflected XSS in HTML reports

- **Threat:** attacker injects HTML / JS into prompts that gets reflected
  verbatim into the exported HTML report.
- **Defence:** all HTML export goes through `_esc()` = `html.escape(...)`.
- **Status:** ✅ Hardened (audit fix #1).

### S3. Auth bypass / missing auth on sensitive endpoints

- **Threat:** attacker hits an internal endpoint without credentials.
- **Defence:** 8 endpoints require auth; `/health` and `/promptfoo/health`
  are explicitly exempted. API key comparison uses `hmac.compare_digest`.
- **Status:** ✅ Hardened (audit fixes #3, #4).

### S4. Template path traversal (`planner`)

- **Threat:** attacker loads a planner template from outside the templates
  directory.
- **Defence:** `_safe_template_path` blocks `..`, absolute paths, and resolves
  through `Path.resolve()` with a strict prefix check.
- **Status:** ✅ Hardened (audit fix #6).

### S5. SQL injection

- **Threat:** attacker injects SQL via unsanitised inputs.
- **Defence:** all DB calls use parameterised queries (`?` placeholders); no
  string interpolation into SQL anywhere.
- **Status:** ✅ By construction.

### S6. Missing security response headers

- **Threat:** browser-side attacks via missing CSP / X-Frame-Options / etc.
- **Defence:** security headers middleware sets the standard set on every
  response.
- **Status:** ✅ Hardened (audit fix #5).

### S7. Dependency CVEs

- **Threat:** known vulnerabilities in upstream packages.
- **Defence:** `pip-audit --strict` in CI on every push / PR. Fail the build
  on any known CVE at or above the configured threshold.
- **Status:** ✅ CI-enforced.

### S8. Code-level security smells

- **Threat:** hard-coded secrets, weak crypto, `eval` / `exec`, etc.
- **Defence:** `bandit -r llm_lab -ll` in CI on every push / PR.
- **Status:** ✅ CI-enforced.

---

## Partial coverage — what we mitigate, not eliminate

### P1. Concurrent writes to SQLite

- **Threat:** under heavy concurrent write load, SQLite can return
  `SQLITE_BUSY` or, in edge cases, report corruption.
- **Mitigation (current):** `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=5000`
  in `db._connect()`. WAL allows concurrent readers + one writer; busy_timeout
  makes the writer wait up to 5 seconds. Single-host deployments are safe.
- **Not defended against:** multi-host deployments where each host has its
  own SQLite file (would need a real RDBMS — out of scope for v0.x).
- **Status:** ⚠️ Adequate for single-host; see `docs/adr/` for upgrade path.

### P2. LLM provider compromise / prompt injection from provider responses

- **Threat:** a compromised or malicious LLM provider returns crafted content
  that exploits the parsing in `worker.py` / `verifier.py` / `export.py`.
- **Mitigation (current):** outputs are treated as untrusted strings; HTML
  output is escaped; verifier results are bounded to known labels.
- **Not defended against:** semantic prompt injection that influences the
  *verifier's verdict* (e.g., a model that says "verifier: pass" in its output).
  This is a fundamental eval-tool problem; we document it and recommend
  human-in-the-loop for high-stakes decisions.
- **Status:** ⚠️ Documented limitation. See §"Out of scope" below.

### P3. Audit-log tampering

- **Threat:** operator with file-system access edits the SQLite event_log
  retroactively to cover tracks.
- **Mitigation (current):** every row in `event_log` carries a
  `prev_hash` (the previous row's hash) and `row_hash` (`sha256(prev_hash
  || canonical_json(row_content))`). `llm-lab verify` walks the chain in
  `id` order and detects any insertion, deletion, or modification of a
  historical row, including the row's `verdict` and `output` payload.
  See `docs/adr/0006-audit-trail-integrity.md` for the design.
- **Implementation outline:** `db.append_event` wraps the read-insert-
  update sequence in a `BEGIN IMMEDIATE` transaction so concurrent
  writers (across processes, not just threads) serialize on the SQLite
  reserved lock; `db.verify_log()` recomputes and compares; `llm-lab
  verify` is the CLI surface.
- **Status:** ⚠️ **Detected, not prevented.** An operator with file-system
  write access can still tamper — but `llm-lab verify` will catch it on
  the next run. This raises the cost of undetected tampering from
  "trivial" to "must also defeat the chain".
- **Recommended external mitigation:** for high-stakes deployments, ship
  a periodic hash snapshot of the SQLite file (or the file itself) to a
  write-once / append-only store. S3 Object Lock, an immutable syslog
  forwarder, or a write-only NFS share are all viable. Compare the
  stored hash to the current hash on every verify. Without this, the
  chain alone is "tampering is detectable after the fact", not "tampering
  is impossible".
- **Not defended against:** an attacker who modifies a row *and*
  recomputes the chain for all subsequent rows (requires reading every
  row in order to compute the new chain — high cost but not impossible).
- **Not defended against:** multi-host deployments where each host has
  its own SQLite file on a network filesystem. SQLite is a single-host
  library. For multi-host, use a real RDBMS or a write-once external
  store.

---

## Out of scope — what we explicitly do not defend against

### O1. Compromised host

If the operator's host is compromised, all bets are off. `llm-lab` does not
defend against an attacker with code-execution on the host. Use OS-level
controls (least-privilege users, mandatory access control, full-disk encryption).

### O2. Compromised LLM provider

If your LLM provider is compromised at the API level, promptfoo-style tools
including `llm-lab` inherit the trust boundary. Verify providers via TLS
certificate pinning in your reverse proxy if this matters.

### O3. Side-channel / timing attacks on the SQLite layer

We use `hmac.compare_digest` for API key comparison (constant-time) but do not
make strong claims about side-channel resistance in the SQLite layer. Deploy
behind a trusted network boundary.

### O4. Denial of service against the LLM provider

`llm-lab` does not rate-limit outbound LLM calls. If you call OpenAI at high
QPS and exhaust your quota, that's on you. We do, however, support a retry
budget (`PROMPTFOO_MAX_RETRIES`) to avoid amplifying transient failures.

### O5. Prompt injection as a semantic attack

We escape HTML output and bound verifier labels. We do **not** defend against
a model that returns crafted text designed to manipulate downstream consumers
(humans reading the report, CI scripts parsing JSON output). Treat LLM
outputs as untrusted input — same posture as any other eval tool.

### O6. Multi-tenant data leakage through shared SQLite

We support multi-tenant via `intent_id` scoping, but the underlying SQLite
file is shared. Tenants can in principle read each other's `event_log` rows
through path-traversal or SQL bugs we haven't caught. **If you need strict
multi-tenant isolation, run separate `llm-lab` instances per tenant.**

---

## Assumptions about the deployment

We assume the operator:

1. Runs `llm-lab` behind a network boundary they control (reverse proxy,
   VPN, or localhost).
2. Sets API keys via env vars, not in source.
3. Restricts file-system access to the SQLite database and templates directory.
4. Trusts the LLM provider to behave non-maliciously at the API layer (O2).
5. Reviews verifier verdicts before acting on them (P2, O5).

If any of these don't hold, additional controls (reverse proxy auth, separate
DB host, output sandboxing) are the operator's responsibility.

---

## Update cadence

This document is reviewed:

- Before every minor release (0.Y.0).
- After any change to the auth / DB / export / planner modules.
- When a new attack class becomes broadly known.

Last reviewed: at v0.9.0 governance push (this commit).