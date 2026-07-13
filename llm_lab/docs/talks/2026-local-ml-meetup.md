# Talk: "When your eval tool needs to survive a security review"

> **Format:** 25 min talk + 10 min Q&A
> **Audience:** local ML / Python / data-science meetup (~30-80 attendees)
> **Slides:** ~30, mostly diagrams + 2 code blocks + 1 demo
> **Time target:** first delivery 3-4 months after project start

## Abstract (200 words, for CFP)

> Your team has picked an LLM evaluation tool. Your security team
> has questions. "Where does the eval data live? Who can read it?
> What happens if someone edits a row in the audit log?"
> Most eval tools have no good answer.
>
> This talk is the story of building one that does. It's a
> small project (`llm-lab`) with an unusual shape: more
> governance docs than features, a published threat model, a
> SHA-256 hash chain on the audit log, and zero new
> dependencies for structured logging + Prometheus metrics.
>
> We'll cover:
> - Why "fast and easy" was the wrong wedge (and what to choose instead)
> - The 80-line `JsonFormatter` that replaced `structlog`
> - The 200-line Prometheus store that replaced `prometheus_client`
> - The 50-line hash chain that detects tampering — and the
>   multi-process race I caught by deliberately disabling the fix
> - The benchmark report that ships with every release
>
> No prior security knowledge assumed. Familiarity with Python
> and a vague sense of "LLM eval" is enough.

## Slide deck (skeleton)

1. **Title slide** — name, project link, your handle
2. **The eval-tool landscape** — promptfoo / ragas / deepeval /
   llm-lab at a glance
3. **The question that killed the conversation** — "what happens
   to the eval data?" (the one security team asks)
4. **The cell I picked** — auditable, not fast / not pretty / not
   the most metrics
5. **Demo** — `pip install -e ".[dev]"` + `llm-lab run` (1 min)
6. **The first audit fix** — XSS, path traversal, auth bypass
   (2 min, no code, just the bug + fix)
7. **The threat model** — what's in scope, what's out (skim)
8. **Observability in 250 lines** — JSON formatter (10 lines on
   screen) + Prometheus store (15 lines on screen)
9. **The hash chain** — diagram of row N → row N+1
10. **The bug I caught myself** — the test I wrote, the fix I
    shipped, the test I deliberately broke
11. **The honest limits** — multi-host unsupported, tamper
    detection not prevention
12. **What I learned** — three lessons (governance > features
    for OSS, audit by self-correction, position narrow)
13. **Q&A** + your contact info

## Delivery notes (for the speaker)

- **Practice twice.** First delivery should be a *local* meetup.
- **Time the demo.** `pip install -e ".[dev]"` takes 30s on a cold
  cache; `llm-lab run` takes 5s. If the demo fails, skip and
  show a screenshot. Don't apologise for 2 minutes.
- **Don't badmouth competitors.** Frame as "different cell, not
  better tool". Attendees use them too.
- **The self-correction story (slide 10) is the most memorable
  part.** Lead with it in the talk if you're nervous.

## Promotion path

1. Local meetup (month 3-4): practice, video if offered.
2. Regional conference (month 6): PyCon APAC, EuroPython,
   PyCon DE, etc. Submit CFP **3 months early**.
3. National / international (month 12): PyCon US, KubeCon,
   LF AI & Data, or ML-focused conference.

If you bomb the local meetup, don't submit to the regional
conference. Practice more, deliver again, then submit.