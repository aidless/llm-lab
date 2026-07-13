# CFP abstract — "Adding tamper-evident audit logs to a Python web app"

> **Format:** 30 min talk
> **Target:** regional Python conference (PyCon APAC, EuroPython,
> PyCon DE, PyCon AU, etc.)
> **Submission window:** 3 months before the conference

## 200-word abstract (CFP form length)

> Most web apps store their audit log as a regular SQL table.
> "Someone with database access edited a row" is undetectable.
> This talk is about making it detectable.
>
> We'll build a tamper-evident audit log from scratch in 50 lines
> of Python + SQLite. Each row carries a SHA-256 hash of the
> previous row's hash plus a canonical JSON serialisation of the
> row's content. Walking the chain in insertion order detects
> any modification, deletion, or insertion.
>
> Along the way, we'll see:
> - Why "wrap it in a transaction" is not enough for multi-writer
>   safety, and the one-line fix (`BEGIN IMMEDIATE`) that is
> - How to test for race conditions you don't fully understand
>   (hint: temporarily disable the fix)
> - The honest limits: tamper *detection* is not tamper
>   *prevention*; the chain raises the cost of undetectable
>   tampering, it doesn't make tampering impossible
> - The recommended external mitigation (write-once snapshot
>   store) for high-stakes deployments
>
> Suitable for Python web developers and security engineers.
> Familiarity with SQL is helpful but not required.

## 50-word short abstract (CFP "elevator pitch" field)

> 50 lines of Python + SQLite can make your audit log
> tamper-evident. We'll build it, test it, find a multi-process
> race the test catches, and discuss the honest limits of
> tamper-detection (vs tamper-prevention).

## What the reviewers will want to know

- **Is this beginner-friendly?** Yes — the core idea is "hash of
  previous + canonical row content", and the code fits on a
  screen.
- **Is this novel?** The hash-chain technique is well-known in
  blockchain / git / log systems. The novelty is in applying it
  to a Python web app's audit log specifically, with the
  multi-process gotcha made explicit.
- **Is this relevant to the audience?** Every Python team with
  a web app has an audit log. Most don't have this protection.
  High relevance.
- **Is the speaker credible?** The talk is based on a real
  feature in a real project, with a real test that catches a real
  bug. The honesty about limits ("detection, not prevention")
  makes the speaker credible, not less.

## Pre-submission checklist

- [ ] Abstract fits in 200 words (cut adjectives, not nouns).
- [ ] Short abstract is 50 words.
- [ ] Talk has at least 2 concrete code blocks, 1 diagram, 1
      demo OR screenshot.
- [ ] Speaker bio is 2-3 sentences, links to project.
- [ ] Submitted at least 3 months before the conference date.
- [ ] If accepted: 1 practice run before the conference.

## If rejected

Don't resubmit the same abstract to the same conference the
next year. Iterate:
- Was the talk too narrow for the audience? Try a broader
  version at a different venue.
- Was the abstract too vague? Get a reviewer to read it (a
  friend, a colleague) and tell you what they think the talk
  is about.
- Did the talk get one specific criticism? Fix that one thing
  and resubmit.