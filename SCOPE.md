# Scope

The product is a local evaluation tool with a hash-linked SQLite log. Most of
the checkout is not that product. Nothing in the non-product trees is a result
of the CI jobs unless a row below says so.

## Product

| Path | Role | CI |
|---|---|---|
| `llm_lab/` | Installable package, CLI, audit log | `test` job: ruff, mypy, pytest, offline `llm-lab demo` |
| `tests/test_selfevolve.py`, `tests/test_repair_bench.py` | Repo-root tests for `selfevolve/` | Same `test` job, after the package suite |
| `selfevolve/` | Code those two test modules import | Covered only by those tests |
| `evalops/` | Frozen task sets and one checked-in evidence package | Evidence text is not re-run as a model eval in CI |
| `agent-safety-eval/`, `safety-judge-calibration/`, `benchmark-contamination-scan/`, `eval-infra-adapters/`, `kube-eval-service/` | Separate packages with their own tests and verifiers | `portfolio-test` job |

Commands and collection counts: [`docs/test-inventory.json`](docs/test-inventory.json).
What a green run does not prove: [`FAILURE_MODES.md`](FAILURE_MODES.md).

There is no PyPI project. Install from a checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install -e './llm_lab[dev]'
.venv/bin/llm-lab demo
```

## Not the product

| Path | What it actually is |
|---|---|
| `knowledge/` | Reading notes and drafts. Counts, citations, and sample code in those files were not produced by this repo's tests or evidence verifiers. |
| `计划/` | An outline for writing more of those notes. |
| `phase8-deploy/` | Deployment notes for a different application. Kept in the tree so the history stays intact. Paths are environment variables, not a developer machine. Not run by CI. |
| `continual_learning/` | Research code. Its tests import PyTorch and are not a CI job. |
| `agent-safety-dpo/`, `evalops-gpu-sim/` | Extra packages. No pytest suite in the portfolio job. The GPU simulator's verifier exits 0 on a `HOLD` decision. |
| `llm_lab/docs/PERSONAS.md`, `FINDING-FIRST-USER.md`, `OUTREACH-TEMPLATES.md`, `outreach-tracking.csv`, `MARKETING-COPY.md`, `CONTENT-CALENDAR.md`, `MASTER-PLAN.md`, `blog-posts/`, `talks/` | Draft planning. [`ADOPTERS.md`](ADOPTERS.md) lists zero organisations. Personas are hypothetical. The tracking sheet has no sent mail. |

`Microsoft/Windows/PowerShell/ModuleAnalysisCache` was an accidental
PowerShell module cache, not research. This change deletes it from the tree.
Git history still has the blob. `.gitignore` ignores `Microsoft/` so a later
local run does not add it back.

## Review without the notes

Sparse checkout leaves the notes in git and out of the working tree:

```bash
git clone --filter=blob:none --sparse https://github.com/aidless/llm-lab.git
cd llm-lab
git sparse-checkout set \
  llm_lab tests evalops scripts tools .github selfevolve \
  agent-safety-eval safety-judge-calibration \
  benchmark-contamination-scan eval-infra-adapters kube-eval-service
```

`git archive` and GitHub's source download omit the paths marked
`export-ignore` in [`.gitattributes`](.gitattributes). A normal `git clone`
still contains them. That is deliberate: the notes are the author's material,
and deleting them would only hide them from the latest commit.
