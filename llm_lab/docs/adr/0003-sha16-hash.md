# ADR-0003: `_sha16` — 16-hex-char SHA-256 truncation in the event log

- **Status:** Accepted
- **Date:** 2026-07-13
- **Deciders:** @llm-lab/maintainers

## Context and problem statement

`llm_lab/db.py` defines `_sha16(text)`:

```python
def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
```

This is used to fingerprint `input_text` and `output_text` rows in
`event_log`. Two questions come up:

1. Why truncate to 16 hex chars (64 bits) instead of the full 256?
2. Why SHA-256 and not, say, BLAKE2 or a non-cryptographic hash?

## Decision drivers

- **Collision probability.** 64 bits gives ~1.8 × 10¹⁹ distinct values.
  Birthday-paradox collision at 50% probability requires ~4.3 × 10⁹ entries.
  For an event log scoped to a single tenant / single evaluation, the
  effective row count is in the millions at worst. Collision risk is
  negligible.
- **Storage.** Each hash column is 16 bytes instead of 64. Across millions of
  rows this saves real disk.
- **Index size.** SQLite indexes on the hash column are 4× smaller.
- **Readability.** 16 hex chars fits in one visual cluster in logs and
  reports; 64 hex chars don't.

## Considered options

1. **Full SHA-256 (64 hex chars)** — cryptographic purity, no truncation
   concern. Doubles (or quadruples) storage cost.
2. **Truncated SHA-256 (16 hex chars)** — what we do. Acceptable collision
   probability at our scale; cheaper storage and indexes.
3. **Non-cryptographic hash (xxhash, blake2b-128)** — faster, smaller.
   Loses the property that the hash is "obviously" a SHA-256 family value,
   which matters for ops people eyeballing logs.
4. **No hash at all** — saves space entirely. Removes any audit-trail
   integrity check (see ADR-0006 for the upgrade path).

## Decision outcome

**Option 2 — `_sha16` truncated SHA-256.**

The truncation is for storage efficiency; the algorithm choice is for
familiarity (ops people recognise a SHA-256-family hex string at a glance).
The collision probability at our scale is acceptable.

### Consequences

**Positive:**

- 4× smaller index + column footprint vs full SHA-256.
- "Obviously a SHA-256 hex value" property helps log-readability.
- Collision risk acceptable for single-tenant / single-eval log sizes.

**Negative:**

- A determined attacker who controls both the input and the log could in
  principle engineer a collision (the truncation is the weakness). For our
  threat model (operator-trusted local SQLite), this is not on the critical
  path. ADR-0006 plans a hash chain on top.

**Neutral:**

- The function name `_sha16` documents the truncation at the call site.

## Validation

- `tests/test_db.py` exercises `_sha16` and verifies hash determinism.
- ADR-0006 plans an upgrade: each event row will also carry a `prev_hash`
  forming a chain, making any tampering (including collisions) detectable
  via a `verify` command.

## Links / references

- `llm_lab/db.py:38` — `_sha16` definition.
- `docs/adr/0006-audit-trail-integrity.md` — planned upgrade.