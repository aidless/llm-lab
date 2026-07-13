## Summary

<!-- One or two sentences: what does this PR do? -->

## Linked issue

<!-- Link the issue this PR closes: "Closes #123" or "Refs #456". -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation / governance / CI only

## How I tested it

<!-- Describe how you verified your change. For new tests, list them. -->

## Security review (mandatory if touching these)

If your PR modifies any of the following, fill this section out. Otherwise delete it.

- [ ] `llm_lab/main.py`
- [ ] `llm_lab/db.py`
- [ ] `llm_lab/planner/` (template path handling)
- [ ] `llm_lab/export.py` (HTML output)
- [ ] `llm_lab/auth` or any endpoint's auth dependency

Reviewer check: does the change preserve the security posture described in
`THREAT_MODEL.md`? Any new attack surface?

## Breaking change / migration note

<!-- If applicable: what users must change to upgrade. -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `mypy .` passes
- [ ] `pytest --cov -q` passes locally
- [ ] CHANGELOG.md updated (under [Unreleased])
- [ ] New / changed public API documented in docstrings