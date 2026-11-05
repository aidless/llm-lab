# Failure modes

`llm-lab verify` and a green CI run are easy to over-read. This page lists
the cases where the tool exits 0 without proving the thing a reader might
think it proved. The tests named below are regression locks for those limits.
They assert the weak outcome on purpose.

## Audit-log forgery

The `event_log` chain is SHA-256 over `prev_hash` plus a canonical JSON
projection of the row (`llm_lab/db.py`). `llm-lab verify` walks rows in `id`
order. A mismatch exits 1. That detects an in-place edit, insert, or delete
that does not recompute the stored hashes
(`tests/test_audit_chain.py::test_chain_detects_modification_of_existing_row`,
`test_chain_detects_deletion`).

It does not authenticate the log.

| Failure | What an attacker does | What `verify` returns |
|---|---|---|
| Forward recompute | Change a row, then recompute `row_hash` and every later `prev_hash` / `row_hash` | Exit 0. `test_forward_recompute_hides_a_verdict_edit` |
| File replacement | Replace `audit.db` with a new database that has its own valid chain | Exit 0. Nothing outside the file is anchored |
| Legacy genesis rows | Leave `prev_hash` and `row_hash` NULL (pre-chain rows, or rows inserted that way) and edit their payload | Exit 0. Those rows are skipped. The CLI prints a warning and still exits 0. `test_legacy_row_mutation_still_verifies` |
| Truncated walk | `llm-lab verify --limit N` | Exit 0 if the first N rows are consistent, even when a later row is not |
| Clock | Set `timestamp` to any string at insert time, then hash it | Exit 0. The chain binds the lie; it does not know the time |
| Unchained table | Edit `tasks` | Not checked. That table is scratch state, not an audit surface |
| Another host | Point two hosts at one SQLite file on a network filesystem | The chain is not a replication protocol. The file can corrupt |

There is no signed checkpoint, transparency log, or write-once copy in this
repository. Until an operator stores a hash of the database somewhere the
writer of the database cannot change, a green `verify` means "this file is
internally consistent," not "this file is the file I saw yesterday."

## False greens

A passing command here is not model quality, a safety acceptance, or adoption.

1. **Offline demo.** `llm-lab demo` forces `LLM_PROVIDER=offline`. The model
   id is `offline-demo-v1`, a fixed string. Exit 0 means the planner, verifier,
   and hash chain ran on that fixture. CI runs this command with no API key.
2. **Package tests skip the live provider.** CI sets dummy provider keys.
   `test_real_gpt4o_mini_call_succeeds` skips unless `OPENAI_API_KEY` is set
   and is not one of those dummy values. A green `test` job did not call a
   model. A real key changes the pass/skip tally, which is why that tally is
   not the CI contract.
3. **Collected counts, not a badge.** [`docs/test-inventory.json`](docs/test-inventory.json)
   is the count `pytest --collect-only` must match.
   `tools/check_test_inventory.py` fails CI on drift. A shields.io image that
   says "N passed" is not used; it would stay green after the suite changed.
4. **Portfolio verifiers and `--allow-pending-human`.**
   `scripts/verify_portfolio.sh` and the `portfolio-test` job pass
   `--allow-pending-human`. The verifier can exit 0 while
   `human_gate_pending` is true. That is an integrity check of checked-in
   files, not a human-review PASS. The safety projects stay `HOLD` until the
   review counts in [`PORTFOLIO_REVIEW.md`](PORTFOLIO_REVIEW.md) are met.
5. **GPU simulation.** `evalops-gpu-sim/verify_evidence.py` exits 0 when
   checksums match even if the decision is `HOLD` because no GPU ran. That
   script is not in the portfolio CI job.
6. **Suites CI does not run.** `continual_learning/tests` imports PyTorch and
   is research code. `agent-safety-dpo/` has no pytest suite. Neither is a
   green job.
7. **Benchmark smoke.** The `benchmark-smoke` job runs
   `benchmarks/self_bench.py --mode smoke` and uploads the JSON. It does not
   re-execute [`llm_lab/benchmarks/v2-results.json`](llm_lab/benchmarks/v2-results.json).
   Those figures are one recorded Windows run.
8. **Coverage upload.** The Codecov step sets `fail_ci_if_error: false`. A
   failed upload does not fail the job.
9. **Dependency audit.** `pip-audit --strict` scans a requirements list built
   from `pyproject.toml`. It does not look up the `llm-lab` distribution.
   There is no PyPI release to look up. A green security job is not a scan of
   this repository as a published package, and it is not a CodeQL result.
   This repo has no `.github/workflows/codeql.yml`; a badge pointing at that
   path is a 404.
10. **`knowledge/`.** CI runs `tools/check_knowledge_links.py` on the notes
    index. That checks local links. It does not run the code samples or
    confirm any number in those notes. The notes are not evaluated science.
    See [`SCOPE.md`](SCOPE.md).
11. **OpenCompass export tests.** Two `eval-infra-adapters` tests skip when
    official OpenCompass is not installed. CI does not install it. A full
    pass count there depends on a machine that this workflow is not.
