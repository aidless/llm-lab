# Buyer personas — who `llm-lab` is for

> **What this is:** Detailed profiles of the people most likely to
> adopt, use, and (eventually) pay for `llm-lab`. Use these to
> (a) write the README, blog, and talk content *for someone*, not
> for everyone, and (b) decide who to reach out to first.
>
> **Why not "everyone"**: writing for "any developer who wants
> LLM eval" means writing for no one in particular. Each persona
> below is a specific person with a specific problem, and we have
> evidence (the project's design choices) that they're
> underserved by current tools.

## The three personas (ranked by fit with the wedge)

### P0 — Alice the ML engineer at a 50-500 person SaaS

- **Job title:** "ML Engineer" or "Senior ML Engineer"
- **Team size:** 2-10 ML engineers
- **Reports to:** Eng manager or product manager
- **Daily tools:** Python, Jupyter, maybe LangChain / LlamaIndex
  for some apps. Not usually a security expert.
- **Pain point (verbatim from interview-style surveys):**
  > "We changed the prompt three weeks ago and nobody can tell
  > you why it changed, or what it used to do. The eval results
  > live in some Slack thread from when we last ran them."
- **What they'd use `llm-lab` for:** weekly prompt A/B
  comparisons, with structured HTML reports they can share with
  product / compliance.
- **Decision authority:** themselves (budget < $5k/year). One
  approval from a director at most.
- **Time to first value:** 10 minutes. `pip install` + one
  `llm-lab run` should produce a report.
- **Where to find them:**
  - LinkedIn search "ML engineer" + their city
  - r/MachineLearning
  - The AI Engineer Summit Discord
  - Local Python / ML meetups
- **What "good" looks like for them:**
  - One HTML report they can paste into Confluence
  - Token + cost data they can show their manager
  - Diff between two runs they can show their PM
- **What will make them leave:**
  - Onboarding friction (need to read 20 pages of docs)
  - Slow first run (need to wait 2 minutes for results)
  - Surprise behavior (verdict says "pass" on garbage output)

### P0 — Bob the security architect at a regulated company

- **Job title:** "Security Architect" or "Platform Security Lead"
- **Team size:** 5-20 in security, larger org total
- **Reports to:** CISO or VP Engineering
- **Daily tools:** Threat models, SBOMs, security review
  checklists. **Knows** OWASP, SOC 2, ISO 27001. May or may not
  know Python.
- **Pain point:**
  > "Last quarter we approved an LLM tool and now nobody can
  > prove it didn't exfiltrate data. We need an audit trail we
  > can show the regulator / auditor."
- **What they'd use `llm-lab` for:** evaluating whether
  `llm-lab` itself is safe to deploy. If they say yes, the
  ML team that comes to them next month is more likely to
  use it.
- **Decision authority:** recommends tools; CISO signs off. The
  review process is **slow** — 1-3 months typical.
- **Time to first value:** 1-2 hours (reading THREAT_MODEL.md +
  SECURITY.md + running `llm-lab verify` on a test log)
- **Where to find them:**
  - SANS / OWASP / Cloud Security Alliance meetups
  - HN `#security` channel
  - The CISO Series podcast community
  - The `r/netsec` sidebar
- **What "good" looks like for them:**
  - THREAT_MODEL.md that honestly says "we don't defend
    against X" (not "we are fully secure")
  - SECURITY.md with a real disclosure email and SLA
  - SBOM per release
  - Hash-chained audit log
- **What will make them leave:**
  - "Just trust us" language
  - Missing or vague threat model
  - "This is open source so you can audit it yourself" (true
    but signals "we didn't")

### P1 — Carol the independent ML consultant

- **Job title:** "ML Consultant" or "Freelance ML Engineer"
- **Team size:** solo, or 2-3 partner firm
- **Reports to:** their clients
- **Daily tools:** whatever the client has + a personal Python
  stack
- **Pain point:**
  > "Every client engagement starts with two weeks of 'how do
  > we evaluate this model' that I re-do from scratch."
- **What they'd use `llm-lab` for:** a drop-in eval framework
  they bring to every engagement, instead of building from
  scratch.
- **Decision authority:** themselves
- **Time to first value:** 30 minutes (install + run on one
  client prompt)
- **Where to find them:**
  - Upwork top-rated ML freelancers
  - Toptal AI practice
  - r/MachineLearning weekly "Who's hiring" thread
  - AI consultancy Slack groups
- **What "good" looks like for them:**
  - Stable API they can wrap in a client-facing tool
  - Sensible defaults (no 50-page config to fill in)
  - Easy to extend (custom verifier / custom metric)
  - Citations in their deliverables ("I used `llm-lab` to
    evaluate...")
- **What will make them leave:**
  - Breaking API changes
  - Hard-coded assumptions that don't fit their client
  - Slow / unreliable releases

## How to use these personas

When you write content (README, blog post, talk), pick *one*
persona and write *to them*. "I built this for Alice, the ML
engineer who keeps losing track of which prompt is which" is
sharper than "I built this for everyone".

When you reach out (outreach templates in
`OUTREACH-TEMPLATES.md`), pick *one* persona per message. The
template is tuned for that persona's specific pain point.

When you prioritise features, ask: "does this help Alice,
Bob, or Carol?" If none of them, defer it.

When you say no to a feature request, cite the persona it
would serve. "This would help a fourth persona I haven't
identified yet — out of scope for v0.x."

## Out of scope personas (deliberately)

- **The solo hobbyist** — they'll find the project, use it,
  maybe star it. They are not the wedge. Don't optimise the
  docs for them.
- **The academic researcher** — they have their own eval
  frameworks. They'll cite your work if it's relevant, but
  they won't switch.
- **The enterprise procurement team** — they'll buy if their
  security team has already approved. They are downstream of
  Bob.
- **The "we just need it fast" engineer** — they'll use
  promptfoo. That's fine; not our wedge.

The personas above are the ones whose existence justifies the
project. If you can't find any of them, the project shouldn't
exist. (See `GOVERNANCE.md` "trigger to promote to core team"
for the related "real user" criterion.)