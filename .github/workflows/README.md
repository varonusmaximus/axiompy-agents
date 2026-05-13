# GitHub Actions (axiompy-agents)

| Workflow | When | Purpose |
|----------|------|---------|
| [`python-ci.yml`](python-ci.yml) | Push/PR to `main` or `develop`, or **workflow_dispatch** | **Ruff** (Python 3.11), **pytest + coverage** on **Python 3.12** (80% gate), **Bandit** + **pip-audit** |
| [`code-review-agent-deploy.yml`](code-review-agent-deploy.yml) | Push/PR to `main` (paths under `axiompy/agents/code_review/`), or **workflow_dispatch** | Code review agent **tests** (`make ci-test`), **Ruff** on that subtree, **Docker build + push** to GHCR (`ghcr.io/<owner>/code-review-agent`) |

No Artifactory or private PyPI index in these workflows. Install **`axiompy`** from PyPI before this package, or from Git if core is not published (see root `README.md` if documented).

## Secrets

| Secret | Used by | Notes |
|--------|---------|--------|
| `CODECOV_TOKEN` | `python-ci.yml` Codecov step | Optional; upload is non-blocking if unset. |
| `GITHUB_TOKEN` | Docker login / push | Default token with `packages: write` on push. |

## Local parity

```bash
make lint
make test
make coverage
make security
make code-review-ci-test
```

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
pre-commit run --all-files
```
