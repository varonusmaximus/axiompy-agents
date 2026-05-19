# axiompy-agents

Distribution **`axiompy-agents`** extends the `axiompy` namespace with:

| Module | Purpose |
|--------|---------|
| **`axiompy.kernel`** | Hexagonal agent runtime (native loop, tools, memory, events) |
| **`axiompy.agents.io`** | Embeddings, vector stores, document sources (compose with kernel) |
| **`axiompy.reasoning`** | Provider-agnostic LLM clients; `ReasoningLLMAdapter` implements kernel `LLMPort` |

Requires **Python 3.12+** and **`axiompy` 2.x** with server and HTTP-related extras.

## Install

`axiompy` 2.x is not on PyPI yet. Install core from GitHub, then this package:

```bash
pip install "axiompy[servers,http,http-async,databases,storage] @ git+https://github.com/varonusmaximus/axiompy.git@main"
pip install -e ".[kernel,io-rag,test-all]"
```

When `axiompy>=2` is published to PyPI:

```bash
pip install "axiompy[servers,http,http-async,databases,storage]>=2.0.0,<3.0.0"
pip install -e ".[kernel,io-rag,test-all]"
```

**Extras:** `kernel`, `io-rag` (alias `rag`), `memory-redis`, `openai`, `anthropic`, `kernel-langgraph`, `kernel-langchain`, `test-all`.

## Quick start

```python
from axiompy.kernel import KernelFactory, KernelSettings, RuntimeType
from axiompy.reasoning import ReasoningFactory, ReasoningProvider
from axiompy.reasoning.adapter import ReasoningLLMAdapter

llm = ReasoningFactory.create(ReasoningProvider.OLLAMA)
kernel = KernelFactory.create(
    RuntimeType.NATIVE,
    KernelSettings(llm=ReasoningLLMAdapter(llm)),
)
result = kernel.run("Summarize the project goals.")
print(result.output)
```

See [`axiompy/agents/examples/minimal_kernel.py`](axiompy/agents/examples/minimal_kernel.py) and module READMEs under [`axiompy/agents/`](axiompy/agents/), [`axiompy/kernel/`](axiompy/kernel/), [`axiompy/reasoning/`](axiompy/reasoning/).

## Development

```bash
make lint       # ruff check + format
make test       # pytest
make coverage   # pytest + 80% gate
make security   # bandit + pip-audit
```

CI: [`.github/workflows/python-ci.yml`](.github/workflows/python-ci.yml) — Ruff on 3.11; pytest, coverage, Bandit, and pip-audit on 3.12.

## Cursor skills

After installing sibling [axiompy](https://github.com/varonusmaximus/axiompy) (see its README, “Cursor skills”):

```bash
pip install -e ../axiompy
axiompy-skills --project   # syncs into ./.cursor/skills/ (gitignored; re-run after axiompy upgrades)
```

## 3.1 migration (agents.io renames)

| 3.0.x | 3.1.0 |
|-------|--------|
| `RAGError` | `AgentIOError` |
| `RAGConfigurationError` | `AgentIOConfigurationError` |
| `RAGIngestionError` | `AgentIOIngestionError` |
| `RAGEmbeddingError` | `AgentIOEmbeddingError` |
| `RAGVectorStoreError` | `AgentIOVectorStoreError` |
| `RAGQueryError` | `AgentIOQueryError` |
| `RAGLLMError` | `AgentIOLLMError` |
| `RAGResponse` | `RetrievalResponse` |
| `DEFAULT_RAG_PROMPT` | `DEFAULT_RETRIEVAL_PROMPT` |
| `axiompy.reasoning.rag_llm_adapter` | `axiompy.reasoning.llm_provider_adapter` |

## 3.0 migration

- **Removed:** `axiompy.agents.code_review`, deploy/Docker/Terraform assets, `RAGService` / `RAGServiceFactory`
- **Use instead:** `axiompy.agents.io` primitives + `axiompy.kernel` for agent loops
- **Changed:** `ReasoningFactory.create()` takes `ReasoningSettings` (no `**kwargs`)

## Namespace

`axiompy.agents.io.*` lives in this repo. Core continues to own `axiompy.io.http`, databases, object storage, etc. — do not add a top-level `axiompy.io` package here.
