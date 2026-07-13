# Master plan — llm-lab from v0.9.7 to v1.0.0

> **What this is:** The single document you read to know "where are
> we, where are we going, what do I do this week." Every other plan
> document is a child of this one. If something in this plan
> contradicts `ROADMAP-v0.10.0.md` or `NEXT-STEPS.md`, this plan
> wins (it was written later, after those).
>
> **Last updated:** 2026-07-13, after v0.9.7 (first CI-green release)
> shipped, 8 release tags on the project, 0 external users, 0 stars.

---

## Where we are

| | |
|---|---|
| Public release | v0.9.7 |
| CI status | All green (6 test jobs + security + benchmark + SBOM + CodeQL) |
| Tests | 381 passed, 1 skipped, 9/9 benchmark fault scenarios |
| Static | ruff + mypy + bandit clean |
| CodeQL alerts | 0 open |
| Releases page | v0.9.0 - v0.9.6 deleted (they were never CI-green); v0.9.7 is Latest |
| Stars | 0 |
| Watchers | 0 |
| External users | 0 |
| External contributors | 0 |
| Content shipped | 4 blog post skeletons, 1 ready-to-publish first post (`03-hash-chain-eval.md`) |
| Outreach messages sent | 0 |
| Time elapsed since first public push | ~2 hours |

**Honest characterisation:** we have a working, well-tested,
well-documented codebase that has been pushed to a public GitHub
repo and tagged once. We have not yet had a single external human
look at it, run it, or judge whether the wedge is real.

That is the only metric that matters from here on.

---

## Where we're going

**Goal:** a v1.0.0 release that we can stand behind.

**v1.0.0 bar** (all must be true):

1. ≥ 5 external users who have run `llm-lab` against real prompts
2. ≥ 1 POC completed in a real production-like environment
3. ≥ 1 third-party security audit, all Critical/High findings
   addressed
4. ≥ 10 stars (modest; not a goal, a side effect)
5. ≥ 2 merged external PRs (someone other than the maintainer)
6. v0.9.x → v0.10.x → v1.0 with **no breaking changes** in v0.10.x
   (SemVer commit)
7. Real benchmark report (`benchmarks/v2-results.json`) replacing
   the stub numbers with measurements from a real LLM provider

**We do not need:**
- More features
- More providers
- More commands
- More ADR topics
- More architecture changes

**v1.0 is a proof of usefulness, not a feature milestone.**

If we don't hit the v1.0 bar within 12 months, that's a valid
signal: the wedge isn't working, and we should publish a "year in
review" post-mortem and archive.

---

## Phase 0 — Today (Day 0)

**Status: substantially complete.** v0.9.7-v0.9.10 (commits
`07f207d` through `bafcc9e`) are pushed. v0.9.10 adds the dev.to
body for the M2 post and the frontmatter flip to `published: true`.

**Done so far (7 tasks):**

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | `git push origin main` (v0.9.7 release) | done | commit `07f207d` |
| 2 | `published: true` frontmatter on M2 + M3 | done | M2 published to dev.to; M3 still draft |
| 3 | dev.to publish M2 post | **done** | [link](https://dev.to/aidless/how-i-shipped-structured-json-logging-prometheus-metrics-with-zero-new-dependencies-1fmj) — 1,083 words, 5 code blocks |
| 4 | HN Show HN post | **TODO** | use `docs/MARKETING-COPY.md` §1 with dev.to URL not GitHub |
| 5 | Reddit r/MachineLearning | **TODO** | §2; **wait 24-48h after HN** |
| 6 | 1 outreach email | **TODO** | §4; pick one archetype, send to one person you know |
| 7 | Real benchmark | **TODO** | need `OPENAI_API_KEY`, run `self_bench.py --real` |

**Decision point at end of Phase 0:** do the messages read as
written by a real person, or do they read as marketing copy? If
the former, continue. If the latter, rewrite the intros before
sending.

---

## Phase 1 — First month (Day 1-30)

**Trigger to enter:** Phase 0 done (post + outreach sent).

**Goals:**
- 1-5 inbound responses (stars, comments, issues, replies)
- 1 second blog post published
- 0-1 real-world POC scheduled

**You do (this month):**

- **Day 1-3:** Run real benchmark. Set `OPENAI_API_KEY=sk-...` in
  your shell. Run `python llm_lab/benchmarks/self_bench.py --mode all
  --steps 50 --real --output llm_lab/benchmarks/v2-results.json`.
  Use `gpt-4o-mini` ($0.10-0.50 for 50 steps). This **replaces the
  stub numbers** in `v1-results.json` with real measurements.
- **Day 3-5:** Publish the second blog post (M2 stdlib
  observability). Flesh out `docs/blog-posts/02-stdlib-observability.md`
  from skeleton to 1000-1500 words. Publish to dev.to.
- **Day 7-14:** Reply to any HN/Reddit/outreach responses. **Don't
  pitch**; just answer questions and ask if they want to try it.
- **Day 14:** If you have a "yes, I'll try it" → schedule a 30-min
  screen-share. Per `docs/FINDING-FIRST-USER.md`. Don't push, don't
  ask for commitment.
- **Day 21-30:** Write the third blog post (M4 wedge) from
  `docs/blog-posts/04-wedge.md` skeleton. Publish.
- **Day 30:** Review. Count real interactions (not impressions —
  emails replied to, issues filed, PRs opened).

**Exit criteria for Phase 1:**
- Real benchmark numbers in `benchmarks/v2-results.json`
- 2 blog posts published on dev.to
- At least 1 inbound conversation with someone other than yourself
- (If 0 inbound conversations, see decision point below.)

**Decision point at end of Phase 1 (CRITICAL):**

| Signal | Read it as | Action |
|---|---|---|
| ≥ 1 "yes, I'll try it" | The wedge resonates with someone | Continue to Phase 2 |
| 0 "yes", but ≥ 3 polite replies | The wedge is unclear in messaging | Rewrite the README's first 200 words, re-pitch |
| 0 replies at all | Either: (a) wedge is wrong, (b) message is wrong, (c) wrong channel | **STOP** and re-read `docs/PERSONAS.md` and `NEXT-STEPS.md` Decision Point section |

If 0 replies after 30 days, do not send more. Re-evaluate the
wedge. The problem is not "we need more outreach"; the problem is
"we built a thing nobody wants yet."

---

## Phase 2 — Second month (Day 31-60)

**Trigger to enter:** Phase 1 had at least 1 inbound "yes, I'll try it".

**Goals:**
- 1 completed POC (30 min screen-share with a real user)
- 1 ADOPTERS.md Pilot entry (or near-entry — "we tried it, here's
  what we found")
- 1-5 stars (modest target)
- 1 real issue filed (something the user finds that we didn't)

**You do (this month):**

- **Day 31-35:** Run the POC. 30 minutes. Per
  `docs/FINDING-FIRST-USER.md`. **You drive the screen-share.**
  Take notes. After, write 1 paragraph in `docs/case-studies/`
  (create the directory) about what surprised you.
- **Day 35-40:** Write 1 issue from the POC. **Self-file** if no
  external user filed anything yet. The point is to demonstrate the
  issue → fix → PR cycle works.
- **Day 40-50:** First merged external PR. Either: (a) the POC
  user sends a small fix, (b) someone in your network who has
  used `llm-lab` files a small PR. If neither happens, accept it
  and move on — the maintainer is a valid first contributor.
- **Day 50-60:** If the POC went well, ask if the user will be a
  Listed Adopter (full entry in `ADOPTERS.md`). If not, ask if
  they'll be a Pilot (no public attribution; just "we tried it").
  Either is a win.

**Exit criteria for Phase 2:**
- 1 completed POC with notes
- 1 issue filed + 1 PR merged (or accepted that the maintainer
  is a valid first contributor)
- 1 entry in `ADOPTERS.md` (Listed or Pilot)
- 1-5 stars (real metric, not vanity)

**Decision point at end of Phase 2:**

| Signal | Read it as | Action |
|---|---|---|
| 1+ completed POC + ADOPTERS entry | **v1.0 is reachable.** Go to Phase 3. |
| 0 POC but 1+ "I'm thinking about it" | The wedge resonates but timing is off | Schedule 1 more POC in Phase 3 before scaling outreach |
| 0 conversations at all | The wedge isn't working yet | **Pivot.** Re-read `docs/PERSONAS.md`. Consider Plan B (archive). |

---

## Phase 3 — Third month (Day 61-90)

**Trigger to enter:** Phase 2 had a completed POC.

**Goals:**
- 2-3 external users (Listed or Pilot)
- 1-3 GitHub issues filed by external users
- Decide on third-party security audit
- First meetup talk (local, not regional)

**You do (this month):**

- **Day 60-65:** Decide on third-party security audit. Options:
  - **(a)** Self-audit only, using Bandit + pip-audit + CodeQL (what we
    already have). Free, but lower credibility.
  - **(b)** Third-party audit. Trail of Bits / Cure53 / NCC Group.
    $5k-15k for a small Python project. Worth it only if you have
    > 5 users who care.
  - **(c)** Friendly-security-reviewer (someone in your network who
    knows Python and can review for an hour). $0, mid credibility.

  Default: **(c)** if you can find someone. **(a)** is the default
  if you can't. **(b)** is the goal for the v1.0 release, not now.

- **Day 70-80:** Local meetup talk. Per
  `docs/talks/2026-local-ml-meetup.md`. Local Python/ML meetup.
  Practice twice. Video if offered. **Not a regional conference.**
- **Day 80-90:** Write a case study (`docs/case-studies/<org>.md`)
  based on the POC notes from Phase 2. With the user's permission,
  even a sanitised version is valuable.

**Exit criteria for Phase 3:**
- 2+ external users
- 1+ external issue or PR
- Decision on security audit (a, b, or c)
- 1 local meetup talk delivered
- 1 case study written

**Decision point at end of Phase 3 (this is the v0.10.0 gate):**

| Signal | v0.10.0 status |
|---|---|
| 2+ external users + 1+ merged external PR | **Cut v0.10.0.** Add the SemVer commit. Add LF AI & Data Sandbox application (if conditions met). |
| 1 external user, no external PR | Cut v0.10.0 anyway, but extend outreach for 2 more months before considering LF AI |
| 0 external users | **Do not cut v0.10.0.** Extend v0.9.7 maintenance. Continue outreach but cap at 1 email per week. |

---

## Phase 4 — Fourth to sixth month (Day 91-180)

**Trigger to enter:** Phase 3 had ≥ 2 external users.

**Goals:**
- v0.10.0 cut (if not already) or maintenance
- 5+ external users
- 1+ POC in production-like environment
- 1+ third-party security audit (if budget allows)

**You do (this month):**

- **Day 90-100:** Cut v0.10.0 if conditions are met. Per
  `docs/ROADMAP-v0.10.0.md`. **v0.10.0 is the SemVer commit; no
  new features.**
- **Day 100-120:** The third-party security audit. Even if it's
  the "friendly security reviewer" option.
- **Day 120-150:** Address all Critical/High findings from the
  audit. **This is the real v0.10.0 work.** Anything that emerges
  here is a real v0.10.x bug fix.
- **Day 150-180:** First regional conference talk. Per
  `docs/talks/cfp-abstract.md`. Submit to PyCon APAC / EuroPython /
  PyCon US (3 months lead time).

**Exit criteria for Phase 4:**
- v0.10.0 cut
- 5+ external users
- 1+ POC in production-like env
- 1+ security audit, all findings addressed

---

## Phase 5 — Sixth to twelfth month (Day 181-365)

**Trigger to enter:** Phase 4 met all exit criteria.

**Goals:**
- Decide on sustainability (GitHub Sponsors? Consulting? Pure OSS?)
- Apply to LF AI & Data Sandbox (if conditions met)
- v1.0.0 commit

**You do (this month):**

- **Day 180-200:** Sustainability question. **Do not** start this
  earlier — it's premature. Options:
  - **(a)** GitHub Sponsors + Open Collective. Realistic monthly
    income at 5+ users: $50-200/mo. Mostly for donations from
    grateful users, not significant revenue.
  - **(b)** Consulting ($200-500/hr) for "help us deploy llm-lab
    in our environment". Requires 2+ paying clients to be worth
    the time investment.
  - **(c)** Pure OSS, no monetisation. Most realistic option. The
    project is the portfolio piece.

  Default: **(c)**. Switch to (a) if/when a user offers. Switch
  to (b) if/when a real client engagement comes.

- **Day 200-250:** LF AI & Data Sandbox application. **Required
  conditions** (per LF AI charter): 2+ maintainers, 5+ external
  contributors, defined governance, public roadmap, public
  meetings. If you don't have all of these, the application will
  be rejected. Don't apply until you do.
- **Day 250-300:** If accepted to LF: do the LF incubation. If not:
  re-evaluate in 6 months.
- **Day 300-365:** v1.0.0 cut. Per the bar above.

**Exit criteria for Phase 5 (= v1.0 release):**
- All v1.0 bar items met (5+ users, 1+ POC, 1+ audit, 10+ stars,
  2+ merged external PRs, no breaking changes in v0.10.x,
  real benchmark report)

---

## What this plan does NOT promise

- **Stars.** Don't set star targets. Set user-count targets. Stars
  are a lagging indicator of users.
- **VC funding.** This plan has no funding step. If you want
  funding, that's a different plan (different post).
- **"Success" by any measure other than "people use this on real
  work."** The wedge either resonates with real ML/security teams
  or it doesn't. No content marketing, no clever positioning, no
  polished demo will change that.

## Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| 0 inbound responses after 60 days | Medium | High | Stop outreach. Re-evaluate wedge. Don't send more messages. |
| Third-party audit finds real bugs | High (any first audit does) | Medium | Fix Critical/High immediately; defer Low/Medium. Document in CHANGELOG. |
| User submits PR with low quality | Medium | Low | Review politely, request changes, merge after revisions. Don't be precious about your codebase. |
| Friendly-user asks for feature that doesn't fit the wedge | High | Medium | Listen, take notes, say "we'll consider it after v1.0". Don't build features pre-v1.0. |
| You burn out | High (1+ year of OSS solo) | High | Plan B (archive) is a valid outcome. Don't guilt yourself. |
| Show HN gets 0 votes | High | Low | Not a failure. Re-post after 24h if it falls off. |
| Real LLM API has a CVE mid-project | Low | Medium | Dependabot handles it. |

## The plan's biggest assumption

**Someone other than the maintainer will find this useful.**

If by month 6 that hasn't happened, the wedge is wrong. Don't
spend month 7-12 polishing. Spend month 7 writing a "what I
learned" post-mortem and archiving. Then build something else
with the lessons.

Maintenance is a valid state. So is failure. The only invalid
state is "kept going for 3 years hoping it would catch on."

---

## What I (the maintainer AI) can do from here

- **More blog post drafts** when you ask (1 per week, not 10)
- **CHANGELOG / ROADMAP / ADOPTERS / case study** updates as
  real things happen
- **Bug fixes** if you paste a real error from a real user
- **Real benchmark run** if you give me a working `OPENAI_API_KEY`
  in a way that doesn't go to chat history
- **Process help** (release checklists, YAML validation,
  anything that prevents the v0.9.0-v0.9.7 disaster from repeating)

What I **can't** do: any action that requires a real person with
real account credentials or real relationships. No HN posts, no
outreach emails, no real LLM API calls with your key, no
third-party audit sign-off.

The project lives or dies on whether **you** do those things.
