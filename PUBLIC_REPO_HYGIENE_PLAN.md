# axiompy-agents: public repo hygiene plan

This document summarizes hygiene applied to this repository: CI, workflows, Makefile, pre-commit, metadata, and removal of disallowed legacy identifiers from the tree and defaults.

## Scope

- **`pyproject.toml`:** Public authors; no private Poetry index; `[project.urls]` under `varonusmaximus/axiompy-agents`.
- **Defaults:** Code review rules repo, GHCR registry paths, Docker/Terraform examples aligned with `varonusmaximus` and `ghcr.io/${{ github.repository_owner }}/code-review-agent`.
- **`python-ci.yml`:** Ruff on 3.11; tests on 3.12 only; Bandit + pip-audit; no masked security; no optional mypy job.
- **`code-review-agent-deploy.yml`:** No private pip index; Ruff 0.14.14; tests on 3.12 with `make ci-test` under `axiompy/agents/code_review`.
- **Root `Makefile` / `.pre-commit-config.yaml`:** Local parity with main CI; pre-push pip-audit and pytest.

## Verification

- Full-tree search per team checklist: no disallowed vendor or legacy monorepo package substrings.
- Both workflows green on `main` when paths trigger them.
- Optional: single orphan root commit and force-push if history must contain no old blobs.
