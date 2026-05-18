# Production readiness (axiompy-agents 3.0)

## Package layout

- `axiompy.kernel` — hexagonal agent runtime (native loop; optional LangGraph/LangChain adapters)
- `axiompy.agents.io` — embeddings, vector stores, document sources/chunkers
- `axiompy.reasoning` — LLM clients; `ReasoningLLMAdapter` implements kernel `LLMPort`

## Install

CI and local dev install **axiompy 2.x from GitHub** (`varonusmaximus/axiompy@main`) because PyPI only has 0.2.x:

```bash
pip install "axiompy[servers,http,http-async,databases,storage] @ git+https://github.com/varonusmaximus/axiompy.git@main"
pip install -e ".[kernel,io-rag,test-all]"
```

Extras: `kernel`, `io-rag` (alias `rag`), `memory-redis`, `openai`, `anthropic`, `kernel-langgraph`, `kernel-langchain`, `test-all`.

## Breaking changes (3.0.0)

- Removed `axiompy.agents.code_review` and deploy assets
- Removed `RAGService` / `RAGServiceFactory`; use `axiompy.agents.io` primitives + `axiompy.kernel`
- `ReasoningFactory.create()` uses `ReasoningSettings` instead of `**kwargs`

## CI parity

```bash
make lint && make test && make coverage && make security
```

Workflow: `.github/workflows/python-ci.yml` (Ruff 3.11; pytest, coverage, Bandit, pip-audit on 3.12). Requires **Python 3.12+** (matches axiompy core).

## Namespace note

This repo extends the `axiompy` namespace alongside core. Submodules `axiompy.agents.io.embeddings`, `axiompy.agents.io.vector`, and `axiompy.agents.io.documents` are owned by **axiompy-agents**; core continues to provide `axiompy.io.http`, database, object storage, etc.

## Optional history scrub

See legacy `PUBLIC_REPO_HYGIENE_PLAN.md` for orphan-root commit guidance if git history must not contain old blobs.
