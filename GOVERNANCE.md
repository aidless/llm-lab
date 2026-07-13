# Governance

`llm-lab` is currently maintained by a single primary maintainer with plans to
evolve into a small core team as external contributors land.

This document describes **how decisions are made today** and **how we plan to evolve**
once the project has 2+ active external contributors.

---

## Current state (single-maintainer phase)

- **Decision authority:** primary maintainer.
- **Release authority:** primary maintainer signs off on every release tag.
- **Code review:** primary maintainer reviews all PRs; for now external PRs may
  be merged after one approving review from the maintainer.
- **Security disclosures:** see `SECURITY.md` — handled by the maintainer.

### Why single-maintainer for now

A single decision-maker removes coordination overhead during the 0.x → 1.0 push,
which is the right tradeoff when external contributors are 0–1. We accept the
bus-factor risk in exchange for shipping speed. The roadmap (see CHANGELOG and
`docs/adr/`) is the public record of what we intend to build; the maintainer
checks in there when intent changes.

### Trigger to promote to "core team"

When **any two** of the following are true, we promote to a 3-person core team:

- 2 or more external contributors with merged PRs.
- 1+ paid / production deployment confirmed (per `ADOPTERS.md`).
- Sustainably-resourced funding (consulting retainer, grant, or sponsorship ≥ $1k/mo).

---

## Core team phase (target: month 12)

When triggered, governance becomes:

- **3 core maintainers**, drawn from the most active contributors and the founder.
- **Lazy consensus** on routine changes (PR merges after 1 approval + 48 h
  without veto).
- **Explicit vote** on breaking changes, new public API surface, or scope changes
  to `llm-lab`'s mission. Vote outcome decided by simple majority of core team;
  the founder has a tie-breaking vote.
- **Maintainers** (wider ring, can merge non-breaking PRs in their area): added
  per `CODEOWNERS` once we have 5+ active external contributors.

### Adding a maintainer

Nominated by an existing core maintainer, seconded by another, with 7 days'
public comment period on a GitHub Discussion. Approval by simple majority of the
core team.

### Removing a maintainer

Maintainer steps down voluntarily → easy. Inactivity > 6 months → core team can
reassign their `CODEOWNERS` areas by majority vote. Misconduct → see Code of
Conduct (TBD; will mirror CNCF CoC once we adopt one).

---

## Decision records

Significant technical and scope decisions are recorded as Architecture Decision
Records in `docs/adr/`. The five ADRs we ship at v0.9.0 are:

- `0001-sync-runner.md` — why a sync Runner + ThreadPoolExecutor, not full asyncio.
- `0002-two-llm-paths.md` — why `promptfoo_provider` and `worker` stay separate.
- `0003-sha16-hash.md` — why `_sha16` truncates SHA-256 to 16 hex chars.
- `0006-audit-trail-integrity.md` — current audit-log limitations + upgrade roadmap.

ADRs use the template in `docs/adr/template.md`. New ADRs are proposed via PR;
existing ADRs are amended via a follow-up ADR that supersedes the old one (link
back from the old ADR's header).

---

## Scope discipline (a hard rule)

`llm-lab` stays focused on **LLM evaluation & orchestration for security-conscious teams**.

Things we explicitly do **not** do:

- Cloud SaaS hosting (out of scope — open core only).
- Fine-tuning / RLHF / model training (use dedicated tools).
- Vector databases / RAG pipelines (use dedicated tools).
- A prompt IDE / playground (use promptfoo / PromptLayer / LangSmith).

If a feature fits another tool better, we will say so rather than build a
worse version. See "放弃清单" in the project plan (internal doc).

---

## Contact

- General questions: open a GitHub Discussion.
- Security: see `SECURITY.md`.
- Conduct issues: TBD (will add a Code of Conduct before adopting CNCF-style governance).