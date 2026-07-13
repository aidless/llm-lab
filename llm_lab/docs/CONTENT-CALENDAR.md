# Content calendar — first 12 months of llm-lab content

> **What this is:** A month-by-month plan for blog posts, talks, and
> community presence. Goal: 1 substantial piece per month, no more
> (single maintainer can sustain that, not 4).
>
> **Why so sparse?** A solo maintainer producing 4 monthly pieces
> burns out by month 6. One well-written piece per month compounds
> better than 4 mediocre ones. We can always add more once external
> contributors land.

## Ground rules

1. **Every piece has a specific number / finding / failure** — not
   "we added a feature". The reader should learn something they
   didn't know 5 minutes ago.
2. **No announcement posts.** Release notes live in `CHANGELOG.md`;
   the blog is for *findings* and *tradeoffs*.
3. **Show your work.** Code blocks, real benchmark numbers, real
   failure modes. Self-correction ("I was wrong about X") is more
   credible than "X is great".
4. **Cross-link to ADR/THREAT_MODEL when relevant.** The blog post
   is the narrative; the docs are the evidence.
5. **End every post with "what's next"** — invites comments, keeps
   the series feeling like a journey, not a press-release treadmill.

## The 12-month calendar

> Format: `[month] [topic] [target audience] [hook]`

### M1 (post-governance-push) — *DONE* (skeleton)
- **Topic:** "Why we wrote 7 governance docs before tagging v0.9.0"
- **Audience:** solo maintainers / small-team OSS authors
- **Hook:** "I deleted a feature to add documentation. Here's why."
- **Distribution:** dev.to, Hacker News (Show HN if v0.9.0 ships clean)

### M2 — *Drafted in `blog-posts/02-stdlib-observability.md`*
- **Topic:** "How I shipped structured logging + Prometheus metrics
  with zero new dependencies"
- **Audience:** Python tool authors who think they need `structlog` /
  `prometheus_client`
- **Hook:** "Three of my four `pyproject.toml` dependencies were
  one-job libraries I didn't actually need."
- **Distribution:** dev.to, r/Python

### M3 — *Drafted in `blog-posts/03-hash-chain-eval.md`*
- **Topic:** "Adding a SHA-256 hash chain to an audit log — and
  finding a multi-process race the day I wrote the test"
- **Audience:** security engineers, anyone who has to defend an
  audit trail
- **Hook:** "I shipped the feature. I shipped a regression test.
  Then I temporarily disabled the fix and the test caught the race
  on id=5. Here's the whole story."
- **Distribution:** dev.to, HN (timed with v0.9.0 release), r/Python
- **Why this is M3:** it's the most concrete + most self-correcting
  of the three, and lands alongside v0.9.0 = maximum discoverability

### M4 — *Drafted in `blog-posts/04-wedge.md`*
- **Topic:** "The three eval tools I considered for my LLM project,
  and the one I built instead"
- **Audience:** ML engineers choosing between promptfoo / ragas /
  deepeval / custom
- **Hook:** "Each of the three is great at one thing. None of them
  would survive my security team's review. So I built a fourth."
- **Distribution:** r/MachineLearning, ML-focused Twitter / X, ML
  conference Slack channels

### M5
- **Topic:** "How `verify_log` helped me catch my own bug" (real
  story from M3.5)
- **Audience:** anyone who has debugged a multi-process app
- **Hook:** "I had a deadlock in the wild. `verify_log` told me
  exactly which row the chain broke at. The fix was BEGIN IMMEDIATE."

### M6
- **Topic:** "The benchmark report `benchmarks/v1-results.json` that
  ships in the repo — and why I publish it on every release"
- **Audience:** OSS authors who don't publish benchmarks
- **Hook:** "If your tool says 'works on my machine', nobody trusts
  you. Here's 5 numbers, on every tag."

### M7
- **Topic:** "Three failure modes I designed around that you might
  not have considered" (offline SDK, missing API key, busy SQLite
  lock)
- **Audience:** anyone deploying LLM tools in production
- **Hook:** "Most eval tools crash on their first network outage.
  Mine says 'this row was 0 tokens, $0, finish_reason=error' and
  keeps going."

### M8
- **Topic:** "SBOM for a Python package: 1 job, 1 file, 0 deps"
- **Audience:** Python package maintainers who haven't done SBOM
- **Hook:** "I had never generated an SBOM before. It took 8 minutes
  including the 7 minutes of reading docs."

### M9
- **Topic:** "Designing the wedge: why 'auditable eval' won over
  'fast eval' / 'free eval' / 'smart eval'"
- **Audience:** OSS founders picking positioning
- **Hook:** "I could have built the best eval tool. I built the one
  the security team would approve. Different game."

### M10
- **Topic:** "The month I shipped 11 governance docs and 0 features"
- **Audience:** OSS maintainers, project managers
- **Hook:** "We went from 349 tests to 381 tests but 0 new lines of
  model code. Here's why that's the right trade."

### M11
- **Topic:** "How a M3 review caught a multi-process bug that would
  have broken the audit chain at scale"
- **Audience:** reviewers, anyone running a code-review process
- **Hook:** "The first review missed it. The second review caught it.
  The third review was me writing a test that proved the catch
  wasn't a false positive. Here's the meta."

### M12
- **Topic:** "Year in review: what worked, what didn't, what's next"
- **Audience:** subscribers, the project's own maintainer (this is
  the most useful one to write for *yourself*)
- **Hook:** "365 days, 3,847 stars, 0 paying customers. Here's what
  I'd do differently."

## Talks / meetups (3, low-stakes first)

### Talk 1 (month 3-4): Local ML meetup
- **Title:** "When your eval tool needs to survive a security review"
- **Format:** 25 min talk + 10 min Q&A
- **Slides:** see `talks/2026-local-ml-meetup.md` (skeleton)
- **Venue strategy:** local Python / ML / data-science meetup, not
  a national conference. ~30-50 attendees, friendly audience.
- **Why local first:** practice the talk twice. National conferences
  are video-recorded and re-watched for years; you want to have
  already bombed in private first.

### Talk 2 (month 6): Regional conference (e.g. PyCon APAC, EuroPython)
- **Title:** "Adding tamper-evident audit logs to a Python web app"
- **Format:** 30 min talk
- **Submission:** fill CFP form, attach a 200-word abstract. Submit
  **3 months before** the conference.
- **CFP template:** see `talks/cfp-abstract.md`

### Talk 3 (month 12): National / international
- **Title:** "What I learned auditing my own audit tool"
- **Format:** 40 min keynote-style
- **Aim:** if Year 1 metrics are on track (external users > 0,
  benchmark published, security review posted).

## Distribution rules

- Every post on dev.to (long-form friendly, good SEO).
- Cross-post to: r/Python, r/MachineLearning, Hacker News
  (Show HN if it's a release post), LinkedIn (if you have a
  network there).
- **Twitter / X:** only if you actually use it. Empty accounts
  hurt credibility.
- **Don't mass-cross-post.** Each channel has its own norms
  (HN: title-only, no editorialising; r/Python: be humble, link
  to source; LinkedIn: be slightly more polished).

## What I will NOT do

- A monthly newsletter (no one subscribes to a 0-user project's
  newsletter; do this at 100+ users)
- A "year in review" infographic (no design bandwidth, no
  value at this stage)
- Sponsored content (conflict of interest with the "auditable"
  wedge)
- A Discord / Slack (per the M1 plan; revisit at 50+ active
  external users)
- Video (until I have a script I'm proud of; text first, video
  in year 2)

## Tools to use (none of these are obligations)

- **dev.to** for blog hosting (free, SEO-good, dev-audience)
- **Markdown** for everything (we already use it)
- **Hemingway Editor** or similar to check readability
- **Carbon** for any code screenshots in posts (instead of full
  screenshots — looks better in dark mode)

## One last rule

If a month goes by and you haven't written the post, **ship a
one-paragraph "what I learned this month" instead of skipping**.
A regular cadence of *anything* beats an irregular cadence of
*excellent pieces*.