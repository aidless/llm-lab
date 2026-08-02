# v0.10.0 scope â€? what it is, and what it isn't

> **Status:** Draft. Targets the **v0.10.0 release**. No fixed date â€?
> the cutoff is "6 months of stable operation in real deployments
> with at least 2 external users" per `GOVERNANCE.md`.

## What v0.10.0 IS

v0.10.0 is the **0.x â†? 1.0 transition commit**. It commits to:

1. **Semantic Versioning from this version forward.** v0.10.0 will
   not contain breaking changes. v0.11.0 may. v1.0.0 will be the next
   committed-1.0 cut, gated on real external adoption.
2. **A real-world data point.** Not "we ran the benchmark on stub";
   "we ran llm-lab on 50 real prompts across 3 providers and the
   results are in `benchmarks/v1-results.json` with timestamps". This
   is the deliverable that turns the v0.9 README claims from
   "should work" to "did work, here's the data".
3. **A documented external user.** v0.10.0 ships with at least one
   `ADOPTERS.md` entry (a Pilot user) and a case study
   (in `docs/case-studies/`). Even one is enough to prove the
   workflow. Per `GOVERNANCE.md`, "first external user" is the trigger
   for the "core team" promotion.
   > **Deferred to v0.11.0** (2026-07-14): no external user
   > documented yet, so v0.10.0 ships without an ADOPTERS.md entry
   > or case study; the README will state this. First-POC outreach
   > (per docs/FINDING-FIRST-USER.md) stays on this checklist.
4. **Third-party security audit results, addressed.** If a real
   audit is done (budget pending), all Critical and High findings
   have a fix PR. If no audit is done by the cutoff, v0.10.0 ships
   as a "no third-party audit yet, see v0.11.0" release and the
   README says so.
5. **An LF AI & Data Sandbox application** (if the GOVERANCE
   trigger conditions are met: 2+ external users, 5+ stars,
   1+ merged external PR). If not met, no application, no problem.

## What v0.10.0 is NOT

It is **not** a feature release. Specifically, v0.10.0 will **not** add:

- New provider integrations (more LLMs, more verifiers, etc.)
- New commands in the CLI
- New metrics in the audit chain
- New schema migrations that change existing column types
- New ADR-prefix topics (the current 7 ADRs cover the design space)

This is **deliberate restraint.** The v0.9 â†? v1.0 transition's job is
to prove that what we have is solid, not to add scope. New features
go in v0.11+.

## Specific work that needs to happen between v0.9.7 and v0.10.0

### Content (M5 per `docs/CONTENT-CALENDAR.md`)

- [ ] **Month 1** â€? Publish the multi-process race post (skeleton
  already written at `docs/blog-posts/03-hash-chain-eval.md`; needs
  personalisation of the intro and the about-the-author block).
- [ ] **Month 2** â€? Publish the stdlib-observability post
  (`docs/blog-posts/02-*.md`).
- [ ] **Month 3** â€? Publish the wedge / positioning post
  (`docs/blog-posts/04-*.md`).
- [ ] **Month 4-12** â€? One piece per month from the 12-month calendar.

### Real-world validation (M6 per `NEXT-STEPS.md`)

- [ ] **Real benchmark run.** Use a real OpenAI key (or Anthropic) to
  replace the stub benchmark numbers. Save as
  `benchmarks/v2-results.json`. Use `gpt-4o-mini` (cheap) or
  `claude-haiku-3.5` (also cheap). **Not Opus.** Publish the
  numbers.
- [ ] **First POC with a real user.** 30-min screen-share. Per
  `docs/FINDING-FIRST-USER.md`. Don't push for commitment, push
  for honest feedback.
- [ ] **First meetup talk.** Per `docs/talks/2026-local-ml-meetup.md`.
  Local Python/ML meetup. Practice twice. Video if offered.

### Stability

- [ ] **Run the full test suite under real-provider CI.** Not stub.
  This means adding a `gpt-4o-mini` smoke test gated on `OPENAI_API_KEY`
  in CI secrets. Don't commit the key. Don't fail the build if the
  key is missing â€? just skip the test.
- [ ] **Three months of zero CI failures.** (v0.9.7 got there; the
  bar for v0.10.0 is "stays there for a quarter".)
- [ ] **Update THREAT_MODEL.md** with anything that came out of
  real-world usage that wasn't visible from local development.

### Process (per the lessons from v0.9.0 â†? v0.9.7)

- [ ] **Release checklist before every tag** (prevent the v0.9.0
  ship-without-CI failure):
  - [ ] `python -X utf8 -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"`
  - [ ] `python -X utf8 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
  - [ ] `pytest --cov --cov-report=xml --cov-report=term-missing --timeout=30 -q`
  - [ ] `mypy .` and `ruff check .` and `bandit -r llm_lab -ll -q`
  - [ ] Visual check of `git status` for accidental `coverage.xml`
- [ ] **Don't release from a feature branch.** All releases cut
  from `main` after the PR-merge cycle.

## What v0.10.0 is gated on

Per `GOVERNANCE.md`:

> The "core team" trigger requires 2+ rows with `response = yes`
> AND at least 1 POC completed.

If by the cutoff date (6 months from v0.9.7) we have:
- **â‰? 2 external users** and **â‰? 1 merged external PR** â†? cut v0.10.0
- **< 2 external users** â†? extend v0.9.7 maintenance, re-evaluate at
  12 months. Document why in `CHANGELOG.md` and move on.

This is not a sad outcome. Maintenance is a valid state. Most
open-source projects have long maintenance windows. The v0.9.7 â†?
v0.10.0 transition is about **proving the project has users**, not
about adding features.

## After v0.10.0

Once v0.10.0 ships with â‰? 2 external users, the next targets
(per `NEXT-STEPS.md`) are:

- **M7** â€? LF AI & Data Sandbox application
- **M8** â€? Third-party security audit (budget pending)
- **M9** â€? Real sustainability question: GitHub Sponsors? Consulting?
  Pure OSS? Decision depends on whether anyone is actually willing
  to pay, not on philosophical preference.
- **M12** â€? v1.0.0 cut. The bar is "â‰? 5 external users, â‰? 1 POC
  completed in production, â‰? 1 security audit, â‰? 10 merged
  external PRs". Not "we shipped every feature in the wishlist".
