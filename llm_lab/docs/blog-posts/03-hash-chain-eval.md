# Blog post M3 — "Adding a SHA-256 hash chain to an audit log, and the multi-process race I caught the day I wrote the test"

> **Status:** Skeleton. Story-driven. ~1200-1500 words when fleshed out.
> Target: dev.to + r/Python + Hacker News (timed with v0.9.0 release).

## Hook

> I shipped the feature. I shipped a regression test. Then I
> temporarily disabled the fix to see if the test would catch a
> regression. It caught it on the first try, at row id=5, with
> the exact error message I would have spent a day debugging in
> production. This is the story of how that happened.

## What I wanted

My tool has a `verify_log` command that walks the `event_log`
table and tells you if anyone has tampered with it. The chain
works like this:

[diagram: row 1 → row 2 → row 3, each linked by SHA-256]

```
row_hash(N) = sha256(prev_hash(N) || canonical_json(row(N)))
```

If anyone edits a row in the middle, the next row's stored
`prev_hash` no longer matches. Tampering detected.

## The first version (broken under contention)

[code block: the original `append_event`]

I read the previous row's hash, inserted a new row with that as
`prev_hash`, computed the new row's hash, and UPDATEd.

The bug: this works in single-process. It breaks the chain
under multi-process. Two writers both read the same
`prev_hash`, both INSERT rows with the same `prev_hash`. The
second writer's row is now in the chain with a `prev_hash`
that points to "the row before the one I just inserted" — but
that's not what `prev_hash` is supposed to mean. The chain is
broken, even though neither writer did anything wrong.

## The fix

Wrap the read-insert-update in a `BEGIN IMMEDIATE` transaction.
SQLite serializes writers; the busy_timeout=5000 makes
contending writers wait instead of erroring.

[code block: the fixed `append_event`]

Two lines of code, but the kind of two lines that would have
been a month of "why does the chain break in production but
passes in dev?" without a regression test.

## The test

```python
async def test_concurrent_appends_produce_valid_chain():
    await database.init_db()
    async def writer(writer_id, n):
        for i in range(n):
            await database.append_event(
                intent_id=f"conc-{writer_id}", seq=i + 1, ...
            )
    await asyncio.gather(*(writer(w, 25) for w in range(4)))
    report = await database.verify_log()
    assert report["ok"] is True
    assert report["rows_checked"] == 100
```

4 concurrent writers × 25 rows each = 100 rows. The chain
should verify clean.

## The test I wrote and the test I should have written

I wrote the test, ran it, it passed. **Then I did the thing that
separates "tests that pass" from "tests that catch bugs":**

1. Commented out the `BEGIN IMMEDIATE` line.
2. Re-ran the test.
3. It failed at id=5 with `prev_hash` mismatch.

That's the moment. The test isn't a checkbox — it's a
demonstration that the bug exists in the absence of the fix. If
the test passed with or without the fix, it would be a
tautology, not a test.

This is what test-driven development is supposed to mean. Most
of us skip the "demonstrate the bug exists" step and end up with
tests that pass on green and fail on red for reasons unrelated
to the fix.

## The honest limit

The chain detects *after-the-fact* tampering. An attacker with
file-system write access can read the chain, edit a row, and
recompute the chain forward. The cost is "read every row in
order, then rewrite them all" — high but not impossible.

For high-stakes deployments, ship a periodic hash snapshot of
the SQLite file to a write-once store (S3 Object Lock, immutable
syslog). Compare the stored hash to the current hash on every
verify. Then the attacker has to compromise both the SQLite
file AND the immutable snapshot.

We document this in `THREAT_MODEL.md §P3` and `README.md`. We
don't pretend we can prevent tampering — only detect it.

## Why this post matters more than the feature

The hash chain is 50 lines of code. The thing I want you to
remember is the test discipline:

1. Write the test for the new behavior.
2. **Temporarily disable the fix.**
3. Verify the test fails.
4. Restore the fix.
5. Verify the test passes.
6. Commit.

If you skip step 2, you don't actually know your test catches
the bug. You have a test that passes today. Tomorrow's bug is
unguarded.

## What's next

Next month: a different audit-log problem. (Blog post M4 — the
"wedge" post.)

---

**Tags:** `python` `sqlite` `concurrency` `audit` `testing`
**Length target:** 1200-1500 words
**Read time target:** 7-9 minutes
**CTA:** "When did you last run your tests against the bug, not
against the fix?"