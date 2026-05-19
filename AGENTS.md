# axiompy-agents — Cursor rules

Scoped to this repository (`axiompy.kernel`, `axiompy.agents.io`, `axiompy.reasoning`). Core axiompy patterns (HTTP, databases, validators) live in the **`axiompy`** dependency.

## Layout

- `axiompy/kernel/` — domain ports, `NativeAgentRuntime`, `KernelFactory`, in-memory adapters
- `axiompy/agents/io/` — embeddings, vector stores, documents (no product `RAGService`)
- `axiompy/reasoning/` — LLM providers; `ReasoningLLMAdapter` bridges to kernel `LLMPort`

## Conventions

- Python **3.12+**, type hints on all public APIs, `match/case` for type dispatch
- **Factory + Settings dataclass** for new components; explicit DI (no `create_from_env()` on factories)
- **Enum-based** factory type selection (no `create_for_postgres()`-style helpers)
- Validate at boundaries with `axiompy.validators`; let `ValidationError` propagate
- HTTP via `axiompy.io.http.HTTPClientFactory` when calling external APIs
- Logging via `axiompy.loggers.LoggerFactory`
- Format/lint: `ruff` (line length 100), `make lint` / `make test`

## Namespace

Use **`axiompy.agents.io`**, not `axiompy.io`, for agent embeddings/vector/documents (avoids collision with core `axiompy.io`).

## Testing

- `tests/test_kernel.py`, `tests/io/test_io.py`, `tests/test_reasoning_*.py`
- Mocks: `KernelFactory` / io factories `create_mock()` where available
- Target **80%+** coverage (`make coverage`)

## CI / security

- `make security` — Bandit (`-s B608` for validated SQL identifiers) + pip-audit (`--ignore-vuln CVE-2026-1839` until transformers 5.x is on PyPI)
- Install core: `axiompy @ git+https://github.com/varonusmaximus/axiompy.git@main`

## Do not reintroduce

- `code_review` agent tree or deploy workflows
- Monolithic `RAGService` — compose io primitives with the kernel instead
