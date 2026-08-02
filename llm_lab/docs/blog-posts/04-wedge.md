---
title: "The three eval tools I considered for my LLM project, and the one I built instead"
published: false
date: 2026-07-13
tags: llm, evaluation, security, positioning
---

# The three eval tools I considered for my LLM project, and the one I built instead

promptfoo is fast and easy. ragas is great for RAG. deepeval has every metric you can name. I needed something none of them offered: an eval tool that would survive my security team's review. So I built a fourth.

## The eval-tool landscape (mid-2026)

| Tool | Strong at | Weak at |
|------|----------|---------|
| **promptfoo** | Quick A/B, easy setup, large community | No audit trail, no threat model, no compliance story |
| **ragas** | RAG-specific metrics (faithfulness, context recall) | Tied to RAG; not a general LLM eval tool |
| **deepeval** | 14+ metrics, academic rigor | Heavy deps, opaque scoring, opinionated |
| **llm-lab** (this project) | Audit trail, threat model, security hardening | Smaller community, younger codebase |

I'm not claiming `llm-lab` is *better* than the others. I'm claiming it occupies a different cell in the matrix — and that the cell it occupies is the one my project actually needed.

## The cell I needed

When my security team asked "what happens to the eval data?", the honest answer for any of the three tools was: "It's stored in SQLite. Anyone with file-system access can edit it. There's no audit log of who changed what."

That's not a "no" — but it is a "we need to talk to you more before we approve this for production". And "we need to talk to you more" from a security team means two weeks of meetings, a spreadsheet of follow-up questions, and a review slot on a calendar that's booked three weeks out. The tool didn't fail a test; it failed a conversation.

That's the part the benchmark numbers don't capture. promptfoo's 20 ms eval time is real, but it's meaningless if the eval never runs because security won't sign off. In a regulated environment, the blocker isn't throughput — it's evidence. Who ran the eval, against which model, with what prompt, when, and did anyone modify the results afterward. Three of those four questions have no answer in the mainstream tools.

The cell I needed: a tool that answered "what happens to the eval data?" with "here's a tamper-evident audit trail, here's our threat model, here's our security disclosure process, and yes you can run `llm-lab verify` to detect tampering."

That's a small cell. It's a real cell. It's underserved.

## What I didn't do

I didn't try to compete on speed (promptfoo wins). I didn't try to compete on metric count (deepeval wins). I didn't try to compete on RAG-specific features (ragas wins).

I tried to be the *only* tool that:

- Has a published threat model
- Has a hash-chained event log
- Has a security disclosure process with an SLA
- Ships a SBOM per release
- Has CI-gated security scanning (bandit + pip-audit)

If you don't care about those, use promptfoo. If you care about those, you have one option.

## The cost of "the security cell"

I traded features for hardening:

- ❌ No vector database / RAG (use ragas or a dedicated tool)
- ❌ No cloud SaaS (open-core only)
- ❌ No fine-tuning / RLHF (out of scope)
- ❌ No prompt IDE / playground (use promptfoo / PromptLayer)

What I kept:

- ✅ Multi-provider LLM calls (OpenAI, Anthropic, Gemini, Ollama, vLLM, llama.cpp, TGI, LocalAI)
- ✅ 8 audit-fixed security posture
- ✅ 381 tests, all green
- ✅ Observability built in (JSON logs + Prometheus)
- ✅ Threat model + SBOM + hash-chained audit log

The audit-fixed posture deserves unpacking, because it's the cheapest thing I did that had the biggest security payoff. I ran the security scanner, and for each finding I didn't just patch it — I wrote a test that would have caught the vulnerability, then fixed the code, then verified the test goes red on the unfixed code and green on the fix. Eight findings, eight tests, eight commits. That's the difference between "we ran a scanner" and "we can prove the fixes hold" — and it's the difference that survives an auditor's third question.

The hash-chained audit log is the other pillar. Every eval event is a row linked to the previous row by hash; `llm-lab verify` re-walks the chain and reports any row whose hash doesn't match. If someone edits a result in the database, verification fails with the exact row and kind of tamper. That single feature is what turned "we need to talk to you more" into "we need to talk to you about how you did that".

## What "occupies a cell" actually means

In OSS, "we built a tool" is not a moat. "We built a tool that serves a real, narrow, undersupplied user" is. The user is "an ML team at a regulated company that has been asked to use LLM evals and has been blocked by security review". There are maybe 5,000 such teams in the world. We need maybe 100 of them to find the project, 10 to use it in production, 1 to sponsor it.

That's the bet. It's a bet on the *undersupplied* part — on the fact that promptfoo's community size and deepeval's metric count don't help anyone whose blocker is a security review. The people in that cell are not looking for more metrics; they're looking for permission to run evals at all.

And I'd rather be 10th in downloads with a defensible security posture than 3rd in downloads with a "we'll get to the audit trail later" roadmap. Later never happens in OSS — the feature that's missing at launch is the feature that's missing forever, because the backlog fills with user requests for things that already exist. By shipping the threat model and the audit log first, I fixed the ordering problem: the security work got done when the project was small enough that 6 hours of writing mattered, instead of when it was big enough that a rewrite was unthinkable.

## If you're not that team

Use promptfoo. It's faster and the community is bigger. Seriously — if your eval data is a throwaway CSV and your only question is "which prompt wins?", the security-cell tool is overkill. The whole point of the cell is that it's narrow.

## If you are that team

You'll want the threat model, the hash-chained audit log, and the verify command. Everything is in the repo: [`THREAT_MODEL.md`](https://github.com/aidless/llm-lab/blob/main/docs/THREAT_MODEL.md), [`README.md`](https://github.com/aidless/llm-lab), and `ADOPTERS.md` is open — leave your info if you're running it in production.

## What's next

This post is slightly out of order: the M3 post about catching my own bug by deliberately disabling the fix and re-running the test is already out. The wedge post matters more for positioning, so it goes here — right where the series explains why the security posture exists.

---

*About the author: maintains llm-lab, a Python LLM evaluation framework. The previous post in this series is about catching a regression by deliberately disabling the fix; this one is about why the security cell exists. If you work at a company that needs to defend its evals, I'd love a 30-minute conversation.*
