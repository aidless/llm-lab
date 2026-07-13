# Finding the first external user

> **What this is:** A concrete, week-by-week playbook for getting the first
> 1–3 humans (other than you) to actually run `llm-lab` on something they
> care about. Not a marketing plan. A *recruiting* plan.
>
> **Why now:** The M1 governance push is in place, M2 observability is
> shipping. Without external users, the project is a portfolio piece —
> valuable to you but invisible to anyone who could pay for it or
> contribute code. The single biggest leverage you have over the next
> 90 days is *who runs this against their own workload*.

## The hard constraints (read first)

1. **You will not get a stranger to install a 0-star tool to do real work.**
   Every external user you find will be **someone you already know**, or
   **someone one degree of separation from you**. This is not a marketing
   problem; it's a relationship problem.
2. **They will not invest more than 30 minutes.** If `pip install -e .[dev]`
   plus `llm-lab run "..."` plus looking at one report doesn't show them
   something useful in 30 minutes, you lose them.
3. **They need a reason that is not "I built this".** They need
   *a problem they already have* that this solves. The wedge
   ("security-auditable eval") is the most concrete reason — most ML
   engineers have been asked "why did we pick this model?" and had no
   good answer.

## The four persona archetypes

Rank these by *who you know personally*. Start at the top of whichever
archetype is closest to you; don't waste time on the rest until you have
one user from your closest archetype.

### A. "Alice" — ML engineer at a 50–500 person SaaS

- **Team:** 2–10 ML engineers, often reporting to Eng / Product.
- **Problem they have:** "We changed the prompt three weeks ago and
  nobody can tell you why it changed, or what it used to do."
- **What you offer:** "Run `llm-lab compare` against your old prompt and
  the new one — you get a structured HTML report with token counts,
  cost, and verifier verdicts, plus a JSON event log with hashes you
  can show your security team."
- **Decision authority:** themselves (budget < $5k/year).
- **Where to find them:** LinkedIn (search "ML engineer" + your city),
  r/MachineLearning, the AI Engineer Summit / MLSys Discord,
  local PyData / ML meetups.
- **Outreach template:** "Saw your [paper / talk / comment] on
  [specific topic]. I've been working on an open-source eval tool that
  focuses on auditable eval trails — would you be willing to try it on
  one of your real prompts and tell me what breaks? 30 min of your time
  buys you a year of free updates."

### B. "Bob" — security architect at a fintech / healthtech

- **Team:** security / platform, often reporting to CTO or CISO.
- **Problem they have:** "We approved an LLM tool last quarter and now
  nobody can prove it didn't exfiltrate data — we need an audit trail
  we can show the regulator."
- **What you offer:** `llm-lab`'s `event_log` with `_sha16` hashes,
  audit-trail upgrade (ADR-0006) in flight, structured logging
  (ADR-0007), and the `THREAT_MODEL.md` showing exactly what we defend
  against.
- **Decision authority:** recommends tools; CISO signs off.
- **Where to find them:** SANS / OWASP / Cloud Security Alliance
  meetups; the `#security` channel of Hacker News; the CISO Series
  podcast community.
- **Outreach template:** "I'm building an open-source LLM eval tool
  with explicit threat-model documentation and an audit-trail upgrade
  in flight (hash-chained event log). I'd love a security reviewer's
  take on whether this would survive your last LLM-tool approval
  process. 20 min call?"

### C. "Carol" — independent ML consultant

- **Team:** solo or 2–3 person consultancy.
- **Problem they have:** "Every client engagement starts with two weeks
  of 'how do we evaluate this model' that I re-do from scratch."
- **What you offer:** A pre-built, opinionated eval harness that
  consultants can drop into a client engagement in 30 minutes.
- **Decision authority:** themselves.
- **Where to find them:** Upwork top-rated ML freelancers; the
  Toptal AI practice; the r/MachineLearning weekly "Who's hiring" thread.
- **Outreach template:** "I'm the maintainer of llm-lab. Looking for a
  consultant who'd try it on one paid engagement and (if it works)
  cite it in the client report. I'd give you priority support and a
  credit in the case study."

### D. "Dan" — researcher / academic in NLP / ML evaluation

- **Team:** lab group.
- **Problem they have:** "I want my benchmark results to be reproducible
  and cited."
- **What you offer:** A tool that emits machine-readable eval reports.
- **Decision authority:** themselves; their lab may pay you to extend it.
- **Where to find them:** arXiv cs.CL; ACL / EMNLP / NeurIPS author lists;
  the `paperswithcode` ecosystem.
- **Outreach template:** "I saw your paper on [specific topic]. I built
  an LLM eval tool that produces reproducible JSON+HTML reports — would
  you be interested in running your [specific benchmark] through it
  and comparing the output format against your current setup?"

## The 30-day calendar

### Week 1 — list 20 names

In a spreadsheet, write down 20 people who match one of the archetypes
above. The list is private to you. Rank by *how directly you know them*:

- **Tier 1 (know personally, can DM today):** 5 names.
- **Tier 2 (one degree of separation):** 10 names.
- **Tier 3 (cold outreach, will need a warm intro):** 5 names.

Don't aim for 20 perfect names. Aim for 20 *plausible* names. Many will
ghost; that's fine.

### Week 2 — message all 5 Tier 1

Use the outreach template for whichever archetype fits. **Don't mass-
customise** — the template is good. One short follow-up after 4 days
if no reply. **Stop after that.**

Goal: at least 1 "yes, I'll try it" by end of week 2.

If 0/5 say yes, the problem is not outreach. It's either (a) the tool
has a real onboarding friction you didn't see, or (b) your archetype
is wrong. **Talk to one Tier-1 no-reply person** (a different one) and
ask what stopped them.

### Week 3 — run a 30-min POC with whoever said yes

The POC has **exactly one goal**: get them to run `llm-lab` against one
real prompt and have a reaction. You be on the call, share screen, do
the typing. After the call, write up what happened in a private note
(even one paragraph).

Hard rule: **do not** push them to file an issue, write a PR, or join
Discord. Just run the tool with them. Everything else is a follow-up.

### Week 4 — Tier 2 (one-degree) outreach, ask the POC for a referral

If the POC went well, ask: "Is there anyone else who'd want to see
this?" That referral list is the highest-signal list you'll ever get.
Message all 5 Tier-2 names **and** the referrals, total 10–15 messages.

Goal: end of month with **2–3 people who have run `llm-lab` on real
work** and one public credit (a quote in `ADOPTERS.md`, or a tweet, or
a blog comment — their choice).

## The failure modes to avoid

| Failure mode | What it looks like | The fix |
| --- | --- | --- |
| **The free-tool salesman** | You spend 4 hours per person explaining the tool | Cap POC at 30 minutes; if they don't get it, they're not the buyer |
| **The infinite-customisation trap** | "Can you add OAuth / a UI / a database?" | "Yes, in exchange for $X or a PR. Which do you want?" |
| **The never-publishes user** | They love it, run it for 3 months, but won't say so publicly | Don't push. Move on. The testimonial will come when they're ready. |
| **The wrong buyer** | They're a hobbyist who'll never pay or contribute | Fine as a fan; don't mistake enthusiasm for commitment. |
| **The competitor's employee** | They want to see your code "for inspiration" | Show the public surface, never the issue tracker until they're a real contributor. |

## What success looks like at month 3

- **3 humans** (not you) have run `llm-lab` on a real workload.
- **1 of them** has publicly said so (ADOPTERS.md row, blog comment, or tweet).
- **1 of them** has filed a real issue that you didn't know about.
- **0** of them have paid you anything yet.

That last one is correct. M2 is not about money. It's about
*evidence*. Every external user is an experiment that falsifies or
confirms your assumptions about who the wedge serves.

The first paying customer is at month 6+ if you do this right. Trying
to skip to it earlier is how you spend four months talking to people
who never needed what you're building.