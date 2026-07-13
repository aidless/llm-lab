---
title: "Adding a tamper-evident event log to an LLM evaluation framework (and the multi-process race my regression test caught)"
published: true
date: 2026-07-13
canonical_url: https://dev.to/aidless/llm-lab-audit-chain
tags: python, sqlite, audit, security, testing
---

# Adding a tamper-evident event log to an LLM evaluation framework (and the multi-process race my regression test caught)

I maintain [llm-lab](https://github.com/aidless/llm-lab), a small Python framework for running and comparing LLM evaluations. The thing I am proudest of is also the thing I almost shipped broken: the **tamper-evident event log**.

Here's the story.

## What I was trying to solve

When you run an LLM evaluation, the output alone is not enough. You also need to know: which prompt was sent, which model answered, what the verifier said, what the cost was, and in what order. If your security team asks "show me the last 30 days of evals", the answer can't be "I have a JSON file somewhere".

The naive solution is a SQL table. The problem is: **anyone with database access can edit a row**. There's no way to tell whether a row was inserted by the system or retroactively modified by a human.

I wanted a tamper-evident log: a SQL table where the database itself proves that no row has been altered, inserted out of order, or removed.

## The chain

The trick is a hash chain. Every row carries a `row_hash` computed from the previous row's `row_hash` plus a canonical serialisation of its own content:

```
row_hash(N) = sha256(prev_hash(N) || canonical_json(row(N)))
```

The first row in the log has `prev_hash = ""` (empty string). Each subsequent row links to its predecessor. Walking the chain in `id` order and recomputing each row's hash detects any insertion, deletion, or modification: a tampered row's recomputed hash won't match the stored one, and every later row's `prev_hash` will be wrong too.

The canonicalisation is the important part. I use Python's `json.dumps` with `sort_keys=True`, `separators=(",", ":")`, and `ensure_ascii=False`. That gives a deterministic byte string on CPython 3.10+ regardless of insertion order, whitespace, or non-ASCII content.

The walking code is 25 lines:

```python
async def verify_log(*, limit: int | None = None) -> dict[str, Any]:
    ...
    prev_hash = ""
    for row in rows_in_id_order:
        if row["prev_hash"] is None and row["row_hash"] is None:
            # legacy genesis row, pre-M3
            continue
        if row["prev_hash"] != prev_hash:
            return {"ok": False, "first_break": {...}}
        if compute_row_hash(row["prev_hash"], row) != row["row_hash"]:
            return {"ok": False, "first_break": {...}}
        prev_hash = row["row_hash"]
    return {"ok": True, ...}
```

That's the whole thing. The `llm-lab verify` CLI subcommand runs this and exits 0 on success, 1 on the first break.

## What the test caught

Here is the bug I almost shipped. The chain works fine for a single process. It also works fine for multiple processes that each only do reads. **It breaks the moment two processes write concurrently.**

The race:

1. Process A reads the previous row's `row_hash`, gets `"abc"`.
2. Process B also reads the previous row's `row_hash`, also gets `"abc"`.
3. Process A inserts its row with `prev_hash = "abc"`, computes its own `row_hash = X`, commits.
4. Process B inserts its row with `prev_hash = "abc"` (stale!), computes its own `row_hash = Y`, commits.

Now row B's stored `prev_hash` says `"abc"`, but the previous row is row A with `row_hash = "X"`. The chain reports a break at row B even though neither writer did anything wrong.

The fix is to wrap the read-insert-update in a `BEGIN IMMEDIATE` transaction. SQLite's reserved lock serialises writers; `PRAGMA busy_timeout=5000` makes contending writers wait instead of erroring. With the fix, Process B blocks until Process A's transaction commits, then reads row A's `row_hash` (now `"X"`) as its `prev_hash`. Chain holds.

I would not have caught this race on my own. The tests I wrote before the fix only exercised **a single writer process** (one `append_event` after another in the same `asyncio.run`). They passed. The CI was green for that.

What caught it: I deliberately disabled the fix, re-ran the test, and watched the test fail at `id=5` with a `prev_hash` mismatch. That is the only reason I trust the fix. Without the test, the race would have shipped and would have failed in production the first time someone ran two `llm-lab run` instances simultaneously.

This is the test I wrote:

```python
async def test_concurrent_appends_produce_valid_chain():
    await database.init_db()

    async def writer(writer_id: int, n: int) -> None:
        for i in range(n):
            await database.append_event(
                intent_id=f"conc-{writer_id}", seq=i + 1, action="call", model="m"
            )

    await asyncio.gather(*(writer(w, 25) for w in range(4)))
    report = await database.verify_log()
    assert report["ok"] is True
    assert report["rows_checked"] == 100
```

Four concurrent `append_event` tasks, 25 rows each, 100 rows total. With the `BEGIN IMMEDIATE` fix: green. Without: fails at `id=5` because of the race I described above.

## The lesson

The lesson is not "use `BEGIN IMMEDIATE` in SQLite". The lesson is **don't trust a test that passes for the wrong reason**.

A test that runs one writer, in one process, in sequence, will pass whether or not the code is concurrency-safe. It tells you the code works **for that scenario**. It does not tell you the code is correct.

To know your test actually catches the bug it's supposed to catch, you have to demonstrate the bug exists in the absence of the fix. The simplest way:

1. Write the test for the new behavior.
2. Temporarily disable the fix.
3. Verify the test fails.
4. Restore the fix.
5. Verify the test passes.
6. Commit.

If you skip step 2-3, you have a test that passes today. Tomorrow's bug is unguarded.

I have skipped this step in past projects. Every time, I have shipped a bug the test should have caught. This is the first time I caught myself doing it, by deliberately breaking the fix. The discipline is: **never commit a test you haven't seen fail**.

## What it does NOT do

Honest limitations. The chain detects tampering after the fact. It does not prevent it. An attacker with file-system write access can rewrite a row and recompute the chain forward; the cost is "read every row in order, then rewrite them all", which is high but not impossible.

Multi-host deployments are unsupported. SQLite is a single-host library. Two hosts with a shared SQLite file on a network filesystem will corrupt the file.

For high-stakes deployments, ship a periodic hash snapshot of the SQLite file to a write-once store (S3 Object Lock, append-only syslog, etc.) and compare the stored hash to the current hash on every verify. Then the attacker has to compromise both the SQLite file and the immutable snapshot.

These are documented in `THREAT_MODEL.md`. We are explicit that the chain is not tamper-proof. We are selling honest detection, not magic.

## The repo

`llm-lab` is at [github.com/aidless/llm-lab](https://github.com/aidless/llm-lab). 381 tests pass, ruff + mypy + bandit clean, CI on macOS + 3 Python versions, CycloneDX SBOM per release.

If you have ever had a security team block an eval tool because "we can't audit it", the audit chain is for you. If you have ideas on a stronger primitive (append-only S3, an actual blockchain, whatever), I would genuinely like to hear them — the comment thread is open.

---

*About the author: maintains llm-lab, a Python LLM evaluation framework. Previously broken a test on purpose, on purpose.*
