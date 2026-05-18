# GitHub Actions (axiompy-agents)

| Workflow | When | Purpose |
|----------|------|---------|
| [`python-ci.yml`](python-ci.yml) | Push/PR to `main` or `develop`, or **workflow_dispatch** | **Ruff** (Python 3.11), **pytest + coverage** and **Bandit + pip-audit** on **Python 3.12** (80% gate) |

**`axiompy` 2.x** is installed from `https://github.com/varonusmaximus/axiompy.git` (`main`) before this package (PyPI only has 0.2.x).

## Secrets

| Secret | Used by | Notes |
|--------|---------|--------|
| `CODECOV_TOKEN` | `python-ci.yml` Codecov step | Optional; upload is non-blocking if unset. |

## Local parity

```bash
make lint
make test
make coverage
make security
```

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
pre-commit run --all-files
```
