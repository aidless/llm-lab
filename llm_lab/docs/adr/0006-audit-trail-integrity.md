# ADR-0006: Audit-trail integrity — current limitations and upgrade roadmap

- **Status:** Implemented (M3) + concurrency fix (M3.5)
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

The "security tool you can show your security team" wedge for `llm-lab`
requires the `event_log` to be a **real audit trail**, not just a history
table. "Real audit trail" means a third party can verify that:

1. No row has been deleted.
2. No row has been modified after insertion.
3. The order of events has not been rearranged.
4. Timestamps are within an acceptable bound of real time.

`llm-lab`'s current `event_log` (see `db.py:14`) satisfies none of these:

- It is a plain SQLite table. Anyone with file-system write access can
  `DELETE FROM event_log WHERE ...` or `UPDATE event_log SET verdict='pass' ...`
  with no detection.
- Timestamps are local Python `datetime.now(timezone.utc).isoformat()` —
  trivially forgeable by anyone with code execution.
- The two `_sha16` columns (`input_hash`, `output_hash`) are *fingerprints
  of the row's content*, not *fingerprints chained to previous rows*. They
  detect content changes only as long as the surrounding row is also
  re-hashed, which an attacker editing both columns in tandem can defeat.

This ADR records the gap and proposes the M3 upgrade. **Until the upgrade
ships, the audit log is best-effort and `THREAT_MODEL.md` P3 ("audit-log
tampering") explicitly says so.**

## Decision drivers

- **Wedge credibility.** The plan puts "the eval tool you can show your
  security team" at the centre of our positioning. Without an actual
  tamper-evident log, the wedge is marketing.
- **Backward compatibility.** Existing `event_log` consumers (the FastAPI
  read endpoints, the export pipelines) must continue to work.
- **Operational simplicity.** We do not want to introduce a separate
  database, an external timestamping service, or a heavy dependency. The
  upgrade should be a small change to `db.py` plus a `verify` CLI command.

## Considered options

1. **SQLite + hash-chain column + `verify` command** (chosen). Add a
   `prev_hash` column. Each new row stores `sha256(prev_row_hash + this_row_canonical_json)`.
   Ship a `llm-lab verify` command that re-walks the log and reports the
   first inconsistency.
2. **Append-only WORM-style storage** (e.g., write to a filesystem path
   with `chattr +a` on Linux). Requires operator-side configuration we
   cannot rely on.
3. **External timestamping / transparency log.** Out of scope — adds an
   external dependency and a network call, which breaks our offline-first
   posture.
4. **Replace SQLite with an immutable store.** Too large a change for the
   value delivered.

## Decision outcome

**Option 1 — hash-chain + `verify` command.**

We add to `event_log`:

- `prev_hash TEXT` — hash of the previous row's canonical content (empty
  string for the first row in a chain).
- `row_hash TEXT` — `sha256(prev_hash + canonical_json(this_row))`. This
  binds each row to all preceding rows.

We ship `llm-lab verify` that:

- Walks `event_log` in `id` order (the autoincrement primary key is itself
  part of the canonical content).
- Recomputes `row_hash` for each row from `prev_hash` and the canonical
  content.
- Reports `(row_id, expected_hash, found_hash)` for the first mismatch.
- Reports "OK: N rows verified, chain intact" on success.

We also add a `canonical_json(row)` helper that produces a deterministic
serialisation (sorted keys, no whitespace, fixed timestamp format) so the
hash is reproducible across runs.

### Implementation outline (for the M3 PR)

1. Migration in `init_db()` — idempotent `ALTER TABLE event_log ADD COLUMN
   prev_hash TEXT` / `ADD COLUMN row_hash TEXT`.
2. Update `append_event()` to compute and store the hash chain.
3. Add `verify_log()` function in `db.py`.
4. Add `llm-lab verify` subcommand in `cli.py`.
5. Tests: tamper with one row's `verdict`, assert `verify_log()` reports
   the right `row_id`. Add multiple rows, assert chain advances correctly.

### Consequences

**Positive:**

- Audit log becomes tamper-evident at the row level. Detects deletion
  (broken chain) and modification (mismatched hash).
- No new external dependencies.
- Backwards-compatible: old `event_log` rows can be migrated by computing
  their hashes on first access (treating the empty `prev_hash` row as the
  genesis).

**Negative:**

- Does not protect against clock skew or timestamp forgery within a row —
  we still trust the insertion-time `timestamp`. Forging a row's timestamp
  is detectable if and only if the operator trusts the system clock at
  insert time. Documented limitation.
- Chain verification cost is O(N) over the log. Acceptable for single-tenant
  logs in the millions; for billion-row logs we would need an index of
  checkpoints.

**Neutral:**

- `THREAT_MODEL.md` P3 moves from "❌ not yet defended" to "⚠️ best-effort,
  detect tampering after the fact, not prevent it".

## Known limitations (M3.5 honest list)

The audit chain is a *detection* mechanism, not a *prevention* one. Read
this section before claiming tamper-evidence to a buyer or auditor.

1. **Multi-host deployments are unsupported.** SQLite is a single-host
   library. If two hosts point at the same SQLite file on a network
   filesystem (NFS, SMB, CephFS), the file will corrupt. The chain is
   only correct within one host. For multi-host, use a real RDBMS or
   keep `event_log` local and ship a hash snapshot to an external store.

2. **Concurrent multi-writer safety is achieved via `BEGIN IMMEDIATE`.**
   As of M3.5, `append_event` opens a `BEGIN IMMEDIATE` transaction
   before reading `prev_hash` and holds it through the `INSERT` and
   `row_hash` `UPDATE`. Combined with `PRAGMA busy_timeout=5000`, this
   serialises concurrent writers across processes; a writer that
   cannot acquire the lock within 5 s raises a `sqlite3.OperationalError`.
   This is sufficient for typical use (one process per host, a few
   concurrent writers). It is not designed for thousands of writers
   per second.

3. **JSON canonicalisation is platform-sensitive.** `canonical_json`
   uses CPython's `json.dumps` with `sort_keys=True`,
   `separators=(",", ":")`, `ensure_ascii=False`. The output is
   deterministic on CPython 3.7+ for finite floats and standard JSON
   types. **PyPy is not tested.** If you verify a log on a different
   Python implementation than the one that wrote it, hashes may
   differ. Document the writer implementation in the audit policy.

4. **The chain does not protect timestamps.** The `timestamp` column is
   populated from the local clock at insert time. An attacker with
   code-execution on the host can lie about the timestamp. If the
   operator trusts the host clock, this is fine; if not, the chain
   won't catch it. Mitigations: NTP-attested clock, monotonic
   incrementing sequence in addition to wall-clock, or external
   timestamping (RFC 3161).

5. **The chain does not prevent tamper-and-rewrite.** An attacker with
   file-system write access can read the chain, modify any row, and
   recompute the chain forward — defeating detection. The cost is
   "read every row in order, then rewrite them all", which is high
   but not impossible. **The chain raises the cost of *undetected*
   tampering; it does not make tampering impossible.** For
   tamper-resistance (vs. tamper-evidence), operators should also ship
   the SQLite file or its hash to a write-once store (S3 Object Lock,
   immutable syslog, etc.) on a schedule.

6. **`verify_log` is O(N).** It walks every row. For a million-row log
   this is a few seconds; for a billion-row log, minutes. We do not yet
   have a checkpointing scheme (e.g., periodic `checkpoint_hash` rows
   to enable incremental verification). Acceptable for the v0.x
   scale we expect; revisit before v2.0.

7. **Legacy genesis rows are accepted.** Pre-M3 rows (NULL
   `prev_hash` / `row_hash`) are treated as the start of a sub-chain.
   `verify_log` reports `legacy_genesis_count` so operators know the
   post-M3 fraction. **A legacy row followed by a tampered row can
   hide the tamper in the gap** (the verify treats the legacy row as
   the start of a fresh chain). This is acceptable for the
   upgrade-in-place scenario, but operations expecting a fully
   tamper-evident history should treat the entire log as
   "trusted-from-M3-onward" rather than "trusted-from-the-beginning".

## Validation (updated)

- Unit tests (`tests/test_audit_chain.py`).
- **M3.5 regression test:** `test_concurrent_appends_produce_valid_chain`
  runs 4 concurrent `append_event` tasks × 25 rows each (100 rows
  total) and asserts `verify_log` reports `ok=True`. Without the
  `BEGIN IMMEDIATE` fix, the chain would fail at id=5 with
  `prev_hash` mismatch. (Verified by temporarily disabling the fix
  in M3.5 development — the test correctly flagged the regression.)
- Benchmark: `benchmarks/self_bench.py --mode fault` includes
  `audit_chain_concurrent_writers` (4 × 25 rows, hermetic temp DB).

## Validation

- Unit tests (above).
- The benchmark suite grows a `--mode tamper` that intentionally modifies
  one row in a copy of the log, then asserts `verify_log()` flags it.

## Links / references

- `llm_lab/db.py:14` — current `event_log` schema.
- `docs/adr/0003-sha16-hash.md` — content-hash (per-row) primitive that
  the chain builds on.
- `THREAT_MODEL.md` §P3 — the limitation this ADR addresses.