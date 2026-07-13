# Contributing to llm-lab

Thanks for your interest. `llm-lab` is a small project with a high quality bar;
we'd rather merge one good PR a month than twenty sloppy ones.

---

## Ground rules

1. **Open an issue first for non-trivial changes.** A 5-line typo fix can go
   straight to PR; anything that touches architecture, public API surface,
   security, or schema needs a short proposal. This avoids wasted work.
2. **One concern per PR.** Don't bundle a refactor + a feature + a docs fix.
   Each deserves its own review and its own changelog entry.
3. **Tests are mandatory.** New code without tests will not be merged. Bug fixes
   should ship with a regression test that fails before the fix.
4. **Security-sensitive changes** (anything in `auth`, `db`, `main.py`,
   `planner/`, `export.py`) require an explicit `SECURITY.md` cross-check
   in the PR description.

---

## Dev setup

```bash
git clone <your-fork-url>
cd <repo>
pip install -e ".[dev,xlsx,sbom]"
```

The install pulls in pytest, pytest-cov, ruff, mypy, plus the optional `openpyxl`
needed for XLSX export tests.

### Run the test suite

From the `llm_lab/` directory (where `pyproject.toml` lives):

```bash
pytest --cov --cov-report=term-missing -q
```

Expected: **349 passed / 1 skipped** on the current main. The one skipped test
(`test_export.py::TestExportXlsx::test_requires_openpyxl_skip`) is conditional
on `openpyxl` being absent.

### Lint and type-check

```bash
ruff check .
mypy .
```

Both must pass before a PR is mergeable. CI runs them on every push.

### Optional security scan

```bash
pip install bandit pip-audit
bandit -r llm_lab -ll
pip-audit --strict
```

CI runs these on every push to `main` and every PR.

### Optional: SBOM generation

The CI `sbom` job generates a CycloneDX SBOM per push. To run it locally:

```bash
pip install -e ".[sbom]"
python -m cyclonedx_py environment \
    --output-file sbom.cdx.json \
    --spec-version 1.6 \
    --output-format JSON \
    --pyproject pyproject.toml
```

The resulting `sbom.cdx.json` is in `.gitignore` — it is a build artifact,
not source.

---

## Code style

- Python ≥ 3.10. Type hints on public functions.
- `ruff` enforces style (see `pyproject.toml [tool.ruff]`). Line length 120.
- `mypy` enforces type safety (loose-strict: `ignore_missing_imports=true`).
- Prefer the standard library over third-party deps. New dependencies need an
  ADR (or a clear case in the PR description) justifying them.

### Things we care about

- **Security first.** Any PR touching `main.py`, `auth`, `db.py`, `export.py`,
  or `planner/_safe_template_path` needs extra scrutiny.
- **Determinism.** Tests should be hermetic — no network calls, no real LLM
  endpoints. Use the stub pattern already in `tests/`.
- **Audit-trail friendliness.** If your change touches the event log schema,
  update `db._CREATE_SQL` and the migration in `init_db()`. Don't break
  existing databases.

### Things we don't enforce

- 100% test coverage on every PR. Coverage is tracked but the bar is "no
  meaningful branch untested", not "every line covered".
- Docstrings on private helpers. Docstrings on public functions only.

---

## Pull request process

1. **Fork** the repo and create a branch off `main`. Branch names like
   `fix/sha16-collision-test` or `feat/prometheus-endpoint` are welcome.
2. **Run the full check locally** before pushing:
   ```bash
   ruff check . && mypy . && pytest --cov -q
   ```
3. **Fill in the PR template** (`.github/PULL_REQUEST_TEMPLATE.md`). It will
   ask for:
   - What problem this solves (link an issue).
   - How you tested it.
   - Any breaking changes and a migration note.
4. **One reviewer required.** For security-sensitive changes, the maintainer
   will do a second-pass review. For others, the maintainer may merge after one
   approval if changes are trivial.
5. **CI must pass.** The PR will not be merged until `test`, `security`, and
   `benchmark-smoke` jobs are green.

### Commit message style

We do not enforce a strict format, but a useful convention:

```
<area>: <imperative summary>

<optional body explaining why>
```

Examples:

- `db: extend event_log with prev_hash column for tamper-evident chain`
- `worker: lazy-import google.generativeai to avoid hard SDK dep`
- `docs: add ADR-0001 sync runner rationale`

---

## Issue labels

We use roughly this label taxonomy (not exhaustive):

- `bug` — something is broken.
- `feature` — new capability proposal.
- `docs` — docs-only change.
- `good first issue` — small, well-scoped, newcomer-friendly.
- `help wanted` — maintainer would like a contributor to pick this up.
- `security` — review by maintainer required.
- `breaking` — would require a major-version bump.

If you're filing an issue, apply one of the above where it fits.

---

## Areas where we'd especially love help

Even at v0.x we have real, well-scoped work for newcomers:

- Adding more providers to `worker._LOCAL_DEFAULTS`.
- Adding more `verifier.py` strategies (e.g., regex, JSON-schema).
- Writing more docstrings on public APIs.
- Translating the README into other languages.
- Triaging open issues and reproducing bugs.

If you don't see your area listed, open an issue and propose it — we'd rather
you scope a 50-line PR than guess.

---

## Code of Conduct

TBD. We will adopt a Code of Conduct (likely the CNCF CoC) before the project
gets its first non-trivial community conflict. Until then, treat each other
with the same directness and respect you'd want for your own work.