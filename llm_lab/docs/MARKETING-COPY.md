# Marketing copy-paste pack (v0.9.7)

> **What this is:** 4 ready-to-paste text blocks for the 4 most useful
> distribution channels. Each is sized for the channel's norms (HN
> terse, Reddit slightly more technical, dev.to long-form, outreach
> personal). Personalize the placeholders, do not rewrite the structure.
>
> **Why all in one file:** when you're ready to publish, open this
> file, copy the channel you want, paste into a new tab. Don't
> search across files. Don't rewrite.

---

## 1. Show HN (Hacker News)

**Where:** `news.ycombinator.com` → submit → fill in title and URL.
**Tone:** terse, technical, no marketing language. HN's audience is
skeptical; overclaiming kills the post.

**Title (pick one — A is the safer default):**

- A. `Show HN: llm-lab – LLM eval framework with tamper-evident audit log`
- B. `Show HN: Auditable LLM evaluation for security-conscious teams`

**URL:** `https://github.com/aidless/llm-lab` (the repo root, not a
specific commit or file)

**Body:**

```
Hi HN,

llm-lab is a local-first Python framework for running and comparing
LLM evaluations, with a focus on the audit trail being defensible.

Three things it does that promptfoo / ragas / deepeval don't:

1. A tamper-evident event log: every row carries a SHA-256 hash
   chained to the previous. `llm-lab verify` walks the chain and
   reports the first break.

2. A published threat model (`THREAT_MODEL.md`) with explicit
   "we defend against X, we don't defend against Y" sections.
   The chain detects tampering after the fact; we document that
   openly rather than implying it's tamper-proof.

3. A CycloneDX SBOM per release, attached as a CI artifact.
   Plus Dependabot alerts on every PR.

Use case: teams that need to use an LLM eval tool but have been
blocked by their security team because the existing tools don't
produce an audit trail.

381 tests pass. Zero new runtime deps for the observability layer
(uses stdlib). MIT license. PRs welcome.

After 7 release candidates, v0.9.7 is the first one that passes
CI end-to-end (previous ones had test failures or workflow
syntax errors that local dev didn't surface).

Repo: https://github.com/aidless/llm-lab
Quickstart: https://github.com/aidless/llm-lab#5-minute-quickstart
Threat model: https://github.com/aidless/llm-lab/blob/main/THREAT_MODEL.md
```

**Posting rules:**
- Submit **once**. Reposting after it falls off the front page is OK
  (after 24h) but never on the same day.
- **Reply to every comment for the first 6 hours.** HN's algorithm
  weights recent activity; if you reply fast, the post stays
  visible longer. Don't argue, don't sell — answer technical
  questions.
- If a comment is critical, **acknowledge the truth** ("yes, you're
  right that X is a limitation; here's the trade-off in the threat
  model"). HN respects honest self-correction.
- Don't link to your personal Twitter / LinkedIn. Ever.

---

## 2. r/MachineLearning (Reddit)

**Where:** `reddit.com/r/MachineLearning/submit` → "Text" post.
**Tone:** more technical than HN. Assume readers know what an
"eval framework" is. No "Show HN" framing — Reddit punishes
anything that smells like promotion.

**Title (max 300 chars):**

```
I built a tamper-evident event log for LLM evals — looking for
feedback on the audit-chain approach (Python, MIT)
```

**Body:**

```
I've been working on `llm-lab`, a Python framework for running and
comparing LLM evaluations. The thing I'm proudest of is the audit
log: every event row in the SQLite store carries a SHA-256 hash
chained to the previous row, and `llm-lab verify` walks the chain
and reports the first break.

```
row_hash(N) = sha256(prev_hash(N) || canonical_json(row(N)))
```

The canonicalisation is `sort_keys=True`, no whitespace, UTF-8
(so non-ASCII intent_ids hash the same across locales). The
`append_event` path uses `BEGIN IMMEDIATE` + `busy_timeout=5000`
so concurrent writers across processes serialise cleanly (the
test I wrote by deliberately disabling the fix caught a multi-
process race on first run — interesting story, blog post soon).

What it does NOT do (and what we say so in the threat model):
- Prevent an attacker with file-system write access from rewriting
  the chain forward (it can; the cost is "read every row in
  order and rewrite them all")
- Work on multi-host deployments (SQLite is single-host)
- Run on PyPy (the canonicalisation depends on CPython's json)

The repo is here: https://github.com/aidless/llm-lab

Three questions I have for this sub:
1. Is the hash chain approach even worth it, or is append-only
   audit to S3 Object Lock the right primitive?
2. For the multi-host story, would you use a real RDBMS or a
   write-once external store?
3. Anyone using `promptfoo` or `ragas` in production who has
   hit the same audit-trail wall?

I have ~380 tests, ruff + mypy + bandit clean, 9 benchmark fault
scenarios. PRs welcome.
```

**Posting rules:**
- Pick the right subreddit: r/MachineLearning is for ML research
  / production. r/Python is for language-level discussion. **Post
  to r/MachineLearning first**; cross-post to r/Python in a
  follow-up comment, not a new post (Reddit bans cross-posts).
- Don't add a "Show HN" link.
- Be ready for 2-3 "why not just use X" comments. Have answers.
  If you don't have answers, say "good question, I'll look into
  it" — that's the right answer for v0.x.

---

## 3. dev.to (long-form blog)

**Where:** `dev.to/new` → "Markdown" editor.
**Tone:** First-person technical narrative. Longer is fine. dev.to
SEO is good for `pip install llm-lab`-style searches. 800-1500 words
target.

**Title (one of these):**

- A. "Adding a tamper-evident event log to an LLM evaluation framework"
- B. "How I shipped structured logging + Prometheus metrics with zero new dependencies"
- C. "The multi-process race my regression test caught the day I shipped"

**Body (use skeleton `docs/blog-posts/03-hash-chain-eval.md` as the spine):**

```
[First-person intro — what you were trying to do, what you built,
what hook this post will deliver. ~150 words. Make it concrete:
"I was writing a small Python tool for running LLM evals and I
needed an audit trail that wouldn't lie to me."]

## What "tamper-evident" actually means

[Explain hash chain in 200 words. Diagram: row 1 → row 2 → row 3
with each link carrying the hash of the previous. Show the
canonical_json formula.]

## What I shipped

[Walk through `db.append_event` and `db.verify_log`. Show the
10-line `compute_row_hash` function. Show the BEGIN IMMEDIATE
wrapper. Show `llm-lab verify` exit codes.]

## The bug my regression test caught

[This is the strongest part. Walk through the multi-process race:
1. I wrote the test
2. I temporarily disabled the fix
3. The test failed at id=5
4. I re-enabled the fix
5. The test passed

Show the actual error. ~200 words.]

## What it does NOT do (and what I say so in the threat model)

[The honest-limit section. This is the most credible part of the
post. ~250 words. List the things the chain does NOT defend
against and what we recommend instead.]

## The repo

[Link + quickstart. 50 words.]

## What I'd love feedback on

[1-2 specific questions. Not "what do you think?" — that's
useless. Specific things: hash chain vs S3 Object Lock, multi-host
story, your eval workflow.]
```

**Posting rules:**
- Use 1-2 code blocks. Max. dev.to is for humans, not docs.
- Add a cover image (use a real screenshot, not stock). 1280×640.
- Tag: `python`, `llm`, `security`, `audit-log`, `devops`.
- Don't add "Show HN" link. The two communities are different.

---

## 4. Outreach (5 hooks, one per archetype)

These are personalised first-touch messages. **Pick one archetype**
that matches someone you actually know. Don't send all 5.

### Hook A — Alice (ML engineer at a 50-500 person SaaS)

You have a specific person in mind: someone who's posted on
r/MachineLearning, written a blog post about prompt versioning, or
shipped an open-source LLM project. Replace `[specific thing]`.

```
Subject: 30 min — would you try llm-lab on one real prompt?

Hi [first name],

Saw your [specific thing — pick one:
  - comment on r/MachineLearning about prompt-versioning
  - blog post "[specific title]"
  - open-source project [specific name]
  - talk at [PyCon / MLConf / local meetup] on [topic]
].

The way you described [their specific problem in one sentence] is
exactly the failure mode I've been working around in llm-lab.

Quick pitch: it's an LLM evaluation framework with a tamper-
evident audit log — the kind of thing a security team would
approve. 30 seconds to install, 5 minutes to run a comparison.

Repo: https://github.com/aidless/llm-lab
Quickstart: https://github.com/aidless/llm-lab#5-minute-quickstart

Would you be willing to try it on one of your real prompts and
tell me what breaks? 30 min of your time buys you a year of free
updates and a credit in the case study.

If the answer is no, that's fine too — would you be willing to
suggest someone else who might be interested?

[your name]
```

### Hook B — Bob (security architect at a regulated company)

You have a specific person: someone who's posted on r/netsec,
written about SOC 2 / ISO 27001, or spoken at OWASP / SANS. Replace
`[specific thing]`.

```
Subject: Threat-model review of an open-source LLM eval tool — 20 min?

Hi [first name],

I maintain `llm-lab`, an open-source LLM evaluation framework. The
thing I want a security reviewer's take on is the threat model:
I've published it
(https://github.com/aidless/llm-lab/blob/main/THREAT_MODEL.md)
and the audit log has a SHA-256 hash chain, but I have no idea
whether either of those would survive your last LLM-tool approval
process.

The honest question I want answered: would you block this tool
from being deployed in a regulated environment, and if yes,
what's the first thing that would change your mind?

20 min phone call. No slides, no demo — just a threat model
review.

Repo: https://github.com/aidless/llm-lab
Threat model: https://github.com/aidless/llm-lab/blob/main/THREAT_MODEL.md

[your name]
```

### Hook C — Carol (independent ML consultant)

You have a specific person: someone who does ML consulting. LinkedIn
"ML Consultant" / freelancer marketplaces.

```
Subject: Cut your "how do we eval this model" ramp-up from 2 weeks
to 30 minutes?

Hi [first name],

I see you do [specific type of consulting — e.g. RAG
implementation for fintech startups]. I maintain `llm-lab`, an
LLM eval framework that I'd love a consultant's take on.
Specifically: would it save you time on client engagements if
you could drop in a working eval pipeline on day one?

The pitch: install, point at one client prompt, get a structured
HTML report with token + cost + verifier verdicts. End of eval.
Cite `llm-lab` in your deliverable if it helps; if it doesn't,
ignore it.

Repo: https://github.com/aidless/llm-lab

Worth 30 min of your time?

[your name]
```

### Hook D — Influencer (security blogger / ML newsletter)

You have a specific author who writes about ML or security.

```
Subject: Your post on [their post topic] made me write this —
would you look at it?

Hi [first name],

Your [post / newsletter issue / paper] on [topic] made the case
for [their specific point]. I just shipped a thing that takes
the [point] seriously:
https://github.com/aidless/llm-lab

Specifically, the threat model is published, the audit log has
a hash chain, and the SBOM ships per release. No "trust me, it's
secure".

If you think it's worth 15 minutes, I'd love a critique. Not a
review, not a quote — a "this is broken because X" is more
useful to me than a star.

[your name]
```

### Hook E — Friend-of-friend (referral chain)

Someone you know introduced you, OR someone Alice / Bob / Carol
referred you to.

```
Subject: [Referrer's name] suggested I reach out

Hi [first name],

[Referrer] mentioned you as someone who might be interested in
an LLM eval framework with a tamper-evident audit log (the kind
that survives a security review). I just shipped v0.9.7 of
`llm-lab` — first CI-green release after a 7-iteration debugging
saga that's well-documented in the changelog.

The thing I most want feedback on: does the audit chain actually
answer the "show me what ran, in what order, without lying"
question that your team would ask? Or is it overengineered for
your use case?

Repo: https://github.com/aidless/llm-lab

Worth 20 min on a call?

[your name]
```

---

## 5. Tracking

After you send 1 outreach message per archetype, track in a
**private** spreadsheet:

| name | archetype | date_sent | response | next_action |
| ---- | --------- | --------- | -------- | ------------ |
| ... | Alice     | 2026-07-14 | (none yet) | follow up 2026-07-19 |

**Do not share this spreadsheet publicly.** It's a sales tool, the
people in it did not consent to be tracked.

**Per `GOVERNANCE.md`**: the "first external user" milestone needs
1+ row with `response = yes`. The "core team" trigger needs 2+ rows
with `response = yes` AND at least 1 POC completed.

---

## Don't do

- ❌ Post the same text to HN, Reddit, dev.to, LinkedIn, Twitter.
  Each channel has its own norms.
- ❌ "We built this amazing tool that..." — read the post text
  again. No "amazing". No "we're excited to announce". No "the
  future of X". Marketing language kills credibility.
- ❌ Outreach to 10 people at once. **5 is the max per week.**
  More than that and you're spamming.
- ❌ Outreach to someone who has nothing to do with LLM / eval /
  security. PERSONAS.md is the list. Stick to it.
- ❌ Argue with critical comments. Acknowledge, then move on.
