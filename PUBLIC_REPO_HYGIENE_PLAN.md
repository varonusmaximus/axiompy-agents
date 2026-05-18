# Public repo hygiene

CI, workflows, Makefile, and metadata cleanup for the public `axiompy-agents` repository.

**Superseded by [PRODUCTION_READINESS_PLAN.md](PRODUCTION_READINESS_PLAN.md)** for architecture, 3.0 migration, and package layout.

## Scope (completed in 3.0)

- `pyproject.toml`: public authors; `[project.urls]`; version 3.0.0; extras `kernel`, `io-rag`, etc.
- `python-ci.yml`: Ruff on 3.11; tests on 3.12; Bandit + pip-audit; 80% coverage
- Root `Makefile` / `.pre-commit-config.yaml`: lint, test, coverage, security
- Removed code review agent deploy workflow and `code_review` tree

## Verification

```bash
make lint && make test && make coverage && make security
```
