# Blog post M1 — "Why we wrote 7 governance docs before tagging v0.9.0"

> **Status:** Skeleton. Personal essay tone. ~600-800 words when fleshed out.
> Target: dev.to + r/Python + Hacker News (Show HN with v0.9.0).

## Hook (3-5 lines)

> I shipped 7 governance files — CHANGELOG, GOVERNANCE, CONTRIBUTING,
> SECURITY, THREAT_MODEL, CODEOWNERS, ADOPTERS — and zero new lines
> of model code. We went from 349 tests to 381 tests, but
> `git diff --stat` shows almost everything is markdown. Here's why
> I think that's the right trade for a project at this stage.

## The "but I should be shipping features" guilt

Every OSS maintainer has felt it. Someone files an issue: "could you
add OAuth?" or "the X feature would be great". Meanwhile, you're
spending your best hours writing `GOVERNANCE.md`.

The temptation is to do the features first. **Don't.**

The features nobody uses don't compound. The docs do.

## What we actually shipped in M1

[table]

- **CHANGELOG.md** — first formal release log; sets the
  expectation that releases are documented.
- **GOVERNANCE.md** — one maintainer today, criteria for promoting
  to a 3-person core team. Most projects skip this and panic
  when a contributor lands.
- **CONTRIBUTING.md** — dev setup, test command, security-sensitive
  review checklist. Stops the "I don't know how to run the tests"
  Slack DMs.
- **SECURITY.md** — disclosure process, response SLA. Operators
  ask "what's your disclosure process?" before adopting. We now
  have an answer.
- **THREAT_MODEL.md** — explicit list of what we defend against and
  what we don't. The "what we don't" list is the honest part.
- **CODEOWNERS** — even with one owner, the file documents which
  modules are security-sensitive.
- **ADOPTERS.md** — empty file with a header, waiting for the
  first entry. Embarrassing but correct: it's a real signal.

## The cost / benefit

Cost: ~6 hours of writing.

Benefit, all downstream:

- New contributors know where to look.
- Security teams have a document to evaluate.
- Down-the-road me has a record of *why* each decision was made.
- The first security audit (when it happens) has a starting point.
- The first external user has a CONTRIBUTING.md to read.

## The 80/20 of OSS sustainability

[link to M1 plan / ADR-0001 / ADR-0002]

[insert: 1-2 paragraphs about how the docs enabled the M2 + M3
work to go faster — because decisions were already recorded]

## What I'd do differently

The `SECURITY.md` email is a placeholder. I should set up a real
`security@` address before tagging v0.9.0 publicly. I didn't.

## What's next

Next month: zero-dep structured logging + Prometheus metrics.
Why `structlog` was overkill. (Blog post M2.)

---

**Tags:** `oss` `python` `governance` `documentation`
**Length target:** 600-800 words
**Read time target:** 4-5 minutes
**CTA:** "If you've done this differently, I'd love to hear
what worked. Open an issue on the repo."