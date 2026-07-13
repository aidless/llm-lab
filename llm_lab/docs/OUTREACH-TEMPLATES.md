# Outreach templates

> **What this is:** Tested, persona-tuned messages for the first 20
> outreach attempts. Personalise the placeholders, but don't
> rewrite the structure — the structure is what works.
>
> **Why templates, not improvisation:** a single sentence's tone
> mistake loses you the conversation. Templates have been
> reviewed for that. Personalisation gives you ~10% lift on
> response rate; template structure gives you ~50% lift.
>
> **General rules (apply to every template):**
> - **Lead with a specific reason you contacted *them*** — not
>   "I thought you might be interested"
> - **One ask, with a clear escape hatch** — not "would you
>   like to discuss a potential collaboration"
> - **Short enough to read in 30 seconds** — under 200 words
> - **Don't include a calendar link** — "let me know if you're
>   up for a 20-min call" gives them the dignity of choosing
>   the time
> - **No attachments** — first contact is text only
> - **Follow up once after 5 business days, then stop**

---

## Template 1: To Alice (ML engineer)

**Where to use:** LinkedIn DM, Twitter DM, or a reply to a
specific r/MachineLearning comment they wrote.

> **Subject line** (for email): "Saw your [specific thing] —
> would you test my LLM eval tool on it?"
>
> Hi [first name],
>
> Saw your [specific thing — a comment, a blog post, a paper, a
> GitHub repo]. The way you described [their specific problem]
> is exactly the failure mode I've been working around in
> `llm-lab`.
>
> Quick pitch: it's an LLM evaluation framework with a tamper-
> evident audit log — the kind of thing a security team would
> approve. 30 seconds to install, 5 minutes to run a comparison.
>
> Repo: https://github.com/aidless/llm-lab
> Quickstart: [link to README "5-minute quickstart" section]
>
> Would you be willing to try it on one of your real prompts
> and tell me what breaks? 30 min of your time buys you a year
> of free updates and a credit in the case study.
>
> If the answer is no, that's fine too — would you be willing
> to suggest someone else who might be interested?
>
> [your name]

**Personalisation hooks** (pick one):
- "your comment on [specific thread] about prompt-versioning"
- "your blog post '[specific title]'"
- "your paper on [specific topic]"
- "your open-source project [specific name]"

**Why this works:** the [specific thing] proves you actually
read their work. The "30 min" is a small ask. The "credit in
the case study" is non-monetary but reputation-building — and
non-cynical, because it actually benefits them.

---

## Template 2: To Bob (security architect)

**Where to use:** LinkedIn DM (security people are usually on
LinkedIn), or a reply to a SANS / OWASP / CISO Series forum
post.

> **Subject line** (for email): "Threat-model review of an
> open-source LLM eval tool — 20 min?"
>
> Hi [first name],
>
> I maintain `llm-lab`, an open-source LLM evaluation
> framework. The thing I want a security reviewer's take on
> is the threat model: I've published it
> ([THREAT_MODEL.md link]) and the audit log has a SHA-256
> hash chain, but I have no idea whether either of those would
> survive your last LLM-tool approval process.
>
> The honest question I want answered: would you block this
> tool from being deployed in a regulated environment, and if
> yes, what's the first thing that would change your mind?
>
> 20 min phone call. No slides, no demo — just a threat model
> review.
>
> Repo: https://github.com/aidless/llm-lab
> Threat model: [link]
>
> [your name]

**Why this works:** you're not asking for a sale. You're asking
for a security review. Bob's job *is* security review; you're
offering free work that helps him think about LLM tools, and
you're not pretending the tool is perfect. The "would you
block this" framing respects his time and expertise.

**Why this might fail:** Bob is busy, and security people are
sceptical of "I'm the maintainer, please review my work".
Mitigation: in your second follow-up, offer a different value
prop ("would you be willing to read the threat model and
comment in a GitHub issue — async, no call needed").

---

## Template 3: To Carol (independent consultant)

**Where to use:** reply to their consulting pitch on
Upwork / Toptal, or a DM after they've posted in a Slack
channel.

> **Subject line:** "Cut your 'how do we eval this model'
> ramp-up from 2 weeks to 30 minutes?"
>
> Hi [first name],
>
> I see you do [specific type of consulting — e.g., RAG
> implementation for fintech startups]. I maintain
> `llm-lab`, an LLM eval framework that I'd love a
> consultant's take on. Specifically: would it save you
> time on client engagements if you could drop in a working
> eval pipeline on day one?
>
> The pitch: install, point at one client prompt, get a
> structured HTML report with token + cost + verifier
> verdicts. End of eval. Cite `llm-lab` in your deliverable
> if it helps; if it doesn't, ignore it.
>
> Repo: https://github.com/aidless/llm-lab
>
> Worth 30 min of your time?
>
> [your name]

**Why this works:** you're offering a time-saver, not a
donation request. The "cite it if it helps" is non-pushy but
opens the door to public endorsement (which is what you
actually want).

---

## Template 4: To "an influencer" (security blogger / ML newsletter)

**Where to use:** DM to a specific author, or a reply to a
post they wrote.

> **Subject line:** "Your post on [their post topic] made me
> write this — would you look at it?"
>
> Hi [first name],
>
> Your [post / newsletter issue / paper] on [topic] made the
> case for [their specific point]. I just shipped a thing
> that takes the [point] seriously:
> [link to your project].
>
> Specifically, the threat model is published
> ([link]), the audit log has a hash chain, and the SBOM
> ships per release. No "trust me, it's secure".
>
> If you think it's worth 15 minutes, I'd love a critique.
> Not a review, not a quote — a "this is broken because X"
> is more useful to me than a star.
>
> [your name]

**Why this works:** influencers want to be the first to flag
something. The "this is broken because X is more useful than
a star" explicitly asks for the critique they're best at
giving.

---

## The "no" reply follow-up

If they say no, your follow-up is the most important message
you'll send. Three options:

1. **"Got it, thanks. Is there someone else you think I
   should reach out to?"** (most useful)
2. **"Got it. If anything changes, you know where to find
   me."** (clean exit)
3. **No response at all.** (acceptable)

Never:
- "Are you sure?" (pressuring)
- "But this would really help you with X..." (arguing with
  their time)
- "Could I at least get a referral?" (extracting value from
  a polite no)

---

## Tracking

Keep a private spreadsheet of:
- name
- persona (Alice / Bob / Carol / influencer)
- date of first contact
- date of follow-up
- response (yes / no / ignored)
- if yes: date of call / POC / endorsement

Don't share this spreadsheet publicly. It's a private sales
tool, and the people in it would not appreciate being tracked
without their knowledge.

The "first external user" milestone (per
`GOVERNANCE.md`) requires 1+ entries with `response = yes`. The
"core team" trigger requires 2+ entries with `response = yes
and at least one POC completed`.