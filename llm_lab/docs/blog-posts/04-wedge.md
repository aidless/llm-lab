# Blog post M4 — "The three eval tools I considered for my LLM project, and the one I built instead"

> **Status:** Skeleton. Positioning essay. ~1000-1200 words when fleshed out.
> Target: r/MachineLearning + ML-focused Slack channels.

## Hook

> promptfoo is fast and easy. ragas is great for RAG. deepeval
> has every metric you can name. I needed something none of
> them offered: an eval tool that would survive my security
> team's review. So I built a fourth.

## The eval-tool landscape (mid-2026)

| Tool | Strong at | Weak at |
|------|----------|---------|
| **promptfoo** | Quick A/B, easy setup, large community | No audit trail, no threat model, no compliance |
| **ragas** | RAG-specific metrics (faithfulness, context recall) | Tied to RAG; not a general LLM eval tool |
| **deepeval** | 14+ metrics, academic rigor | Heavy deps, opaque scoring, opinionated |
| **llm-lab** (this project) | Audit trail, threat model, security hardening | Smaller community, younger codebase |

I'm not claiming `llm-lab` is *better* than the others. I'm
claiming it occupies a different cell in the matrix.

## The cell I needed

When my security team asked "what happens to the eval data?"
the honest answer for any of the three was: "It's stored in
SQLite. Anyone with file-system access can edit it. There's no
audit log of who changed what."

That's not a "no" — but it is a "we need to talk to you more
before we approve this for production".

The cell I needed: a tool that answered "what happens to the
eval data?" with "here's a tamper-evident audit trail, here's
our threat model, here's our security disclosure process, and
yes you can run `llm-lab verify` to detect tampering."

That's a small cell. It's a real cell. It's underserved.

## What I didn't do

I didn't try to compete on speed (promptfoo wins). I didn't
try to compete on metric count (deepeval wins). I didn't try
to compete on RAG-specific features (ragas wins).

I tried to be the *only* tool that:
- Has a published threat model
- Has a hash-chained event log
- Has a security disclosure process with an SLA
- Ships a SBOM per release
- Has CI-gated security scanning (bandit + pip-audit)

If you don't care about those, use promptfoo. If you care about
those, you have one option.

## The cost of "the security cell"

I traded features for hardening:

- ❌ No vector database / RAG (use ragas or a dedicated tool)
- ❌ No cloud SaaS (open-core only)
- ❌ No fine-tuning / RLHF (out of scope)
- ❌ No prompt IDE / playground (use promptfoo / PromptLayer)

What I kept:
- ✅ Multi-provider LLM calls (OpenAI, Anthropic, Gemini, Ollama,
  vLLM, llama.cpp, TGI, LocalAI)
- ✅ 8 audit-fixed security posture
- ✅ 381 tests, all green
- ✅ Observability built in (JSON logs + Prometheus)
- ✅ Threat model + SBOM + hash-chained audit log

## What "occupies a cell" actually means

In OSS, "we built a tool" is not a moat. "We built a tool that
serves a real, narrow, undersupplied user" is. The user is "an
ML team at a regulated company that has been asked to use LLM
evals and has been blocked by security review". There are maybe
5,000 such teams in the world. We need maybe 100 of them to
find the project, 10 to use it in production, 1 to sponsor it.

That's the bet.

## If you're not that team

Use promptfoo. It's faster and the community is bigger.

## If you are that team

[link to THREAT_MODEL, README, ADOPTERS — leave your info]

## What's next

Next month: I caught my own bug by deliberately disabling the
fix and re-running the test. (Blog post M3 — slightly out of
order because the wedge post is more important for positioning.)

---

**Tags:** `llm` `evaluation` `security` `positioning`
**Length target:** 1000-1200 words
**Read time target:** 6-7 minutes
**CTA:** "If you work at a company that needs to defend its
evals, I'd love a 30-min conversation."