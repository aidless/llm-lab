# Security

`llm-lab` is designed to be safe to deploy in security-conscious environments.
This document covers **how to report vulnerabilities** and **how we handle them**.

If you are looking for *what* llm-lab defends against, see [`THREAT_MODEL.md`](./THREAT_MODEL.md).

---

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | ✅ Active          |
| older   | ⚠️ Best-effort only |

`llm-lab` is pre-1.0. We backport security fixes to the latest minor release.
We do **not** commit to maintaining old minor versions indefinitely.

---

## Reporting a vulnerability

**Please do not file a public GitHub issue for security bugs.**

Use **GitHub's private vulnerability reporting** (the "Security" tab on
this repository → "Report a vulnerability"). This keeps the conversation
private between you and the maintainer, and gives you a permanent URL
for the report.

Include:

1. A clear description of the issue and impact.
2. Steps to reproduce, or a proof-of-concept.
3. The affected version(s) and commit(s), if known.
4. Whether you want public credit.

If the GitHub reporting flow is unavailable for any reason, open a
GitHub issue with the `security` label and a minimal title (e.g.,
"XSS in HTML export under Y condition") — the maintainer will move
the conversation private.

---

## Response timeline

We aim for:

- **Acknowledgement** within **3 business days**.
- **Triage & impact assessment** within **7 days**.
- **Fix or mitigation** for critical issues within **30 days**, others within
  **90 days**.

We follow **coordinated disclosure**: we ask reporters to give us a reasonable
window (typically 90 days) before publishing details. We will credit reporters
in the release notes unless asked otherwise.

---

## Severity rating

We use a four-level scale, loosely aligned with CVSS:

| Severity      | Examples                                               |
| ------------- | ------------------------------------------------------ |
| Critical      | RCE, auth bypass, silent data corruption              |
| High          | Privilege escalation, SQLi/XSS in trusted contexts     |
| Medium        | Information disclosure, DoS in normal use              |
| Low           | Minor info leak, hardening opportunity                 |

---

## Security audit history

| Date       | Scope                  | Auditor           | Report |
| ---------- | ---------------------- | ----------------- | ------ |
| _planned_  | v0.9.0 release         | self-audit + Bandit + pip-audit in CI | this changelog |
| _planned_  | v1.0.0 release         | third-party (TBD) | TBD    |

Until a third-party audit lands, the `security` job in CI runs `Bandit` (code-level
static analysis) and `pip-audit` (dependency CVE scan) on every push to `main`
and every PR.

---

## Out of scope

- Issues in **upstream dependencies** (OpenAI SDK, FastAPI, etc.) — please
  report those to the upstream project.
- Issues that require the operator to have already compromised the host.
- Theoretical concerns without a realistic attack scenario.

---

## Acknowledgements

We thank the following reporters (none yet — this section grows as it should):

- _your name could be here_