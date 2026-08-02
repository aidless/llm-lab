---
title: "Why I wrote 7 governance docs before tagging v0.9.0"
published: false
date: 2026-07-13
tags: oss, python, governance, documentation
---

# Why I wrote 7 governance docs before tagging v0.9.0

I shipped 7 governance files — `CHANGELOG.md`, `GOVERNANCE.md`, `CONTRIBUTING.md`, `SECURITY.md`, `THREAT_MODEL.md`, `CODEOWNERS`, `ADOPTERS.md` — and zero new lines of model code. The test count went from 349 to 381, but `git diff --stat` shows almost everything is markdown. Here's why I think that's the right trade for a project at this stage.

## The "but I should be shipping features" guilt

Every OSS maintainer has felt it. Someone files an issue: "could you add OAuth?" or "this feature would be great". Meanwhile, you're spending your best hours writing `GOVERNANCE.md`.

The temptation is to do the features first. **Don't.**

Features are a trade: you spend effort now to serve users you don't have yet. Documentation is an investment: you spend effort now so every future contributor, reviewer, and auditor starts from a known state. The features nobody uses don't compound. The docs do — every new contributor reads them, every release consumes them, every security review evaluates them.

## What I actually shipped

| File | What it does | Why it matters |
|------|--------------|----------------|
| **CHANGELOG.md** | First formal release log | Sets the expectation that releases are documented, not just tagged |
| **GOVERNANCE.md** | One maintainer today, criteria for promoting to a 3-person core team | Most projects skip this and panic when a contributor lands |
| **CONTRIBUTING.md** | Dev setup, test command, security-sensitive review checklist | Stops the "I don't know how to run the tests" Slack DMs |
| **SECURITY.md** | Disclosure process, response SLA | Operators ask "what's your disclosure process?" before adopting. Now we have an answer |
| **THREAT_MODEL.md** | Explicit list of what we defend against and what we don't | The "what we don't" list is the honest part |
| **CODEOWNERS** | Documents which modules are security-sensitive | Even with one owner, the file records the map |
| **ADOPTERS.md** | Empty file with a header, waiting for the first entry | Embarrassing but correct: it's a real signal |

## The cost / benefit

Cost: roughly 6 hours of writing.

Benefit, all downstream:

- New contributors know where to look instead of guessing.
- Security teams have a document to evaluate before the first conversation.
- Down-the-road me has a record of *why* each decision was made.
- The first security audit (when it happens) has a starting point.
- The first external user has a `CONTRIBUTING.md` to read.

## The 80/20 of OSS sustainability

Here's the part I didn't expect: the docs didn't just prepare the ground for M2 and M3 — they made those months *faster*.

M2 was zero-dep observability: structured JSON logging and Prometheus metrics, 280 lines of `observability.py`. M3 was a hash-chained audit trail: every eval event logged, every row linked to the previous one by hash, tamper detection built in. Neither required a single governance decision to be made on the fly — the threat model already told me what to defend against, the ADRs already recorded why. I wrote feature code instead of writing position papers mid-stream.

That's the compounding. The 6 hours spent in M1 were recovered in M2 and M3 because decisions were already made, already written down, and already agreed.

## What I'd do differently

The `SECURITY.md` disclosure email is a placeholder. I should have set up a real `security@` address before tagging v0.9.0 publicly. I didn't. If you're doing this, set up the mailbox first — it's five minutes in your DNS provider and it's embarrassing to fix after someone actually discloses something.

## What's next

This is the first post in the series. Next up: zero-dep structured logging + Prometheus metrics — why `structlog` was overkill for a single-process project, and the 280 lines I wrote instead.

---

*About the author: maintains llm-lab, a Python LLM evaluation framework. This is the first post in a series about the boring engineering that makes an LLM tool actually ship. If you've done governance-first differently, I'd love to hear what worked — open an issue on the repo.*
