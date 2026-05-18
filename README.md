# axiompy-agents

Distribution **`axiompy-agents`** extends the `axiompy` namespace with:

- **`axiompy.kernel`** — hexagonal agent runtime (native loop, tools, memory, events)
- **`axiompy.agents.io`** — embeddings, vector stores, document sources (compose with kernel)
- **`axiompy.reasoning`** — provider-agnostic LLM clients and data-query helpers

Requires **`axiompy`>=2** with server and HTTP-related extras.

## Install

```bash
pip install "axiompy[servers,http,http-async,databases,storage]>=2.0.0,<3.0.0"
pip install -e ".[kernel,io-rag,test-all]"
```

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

See [PRODUCTION_READINESS_PLAN.md](PRODUCTION_READINESS_PLAN.md) for CI, packaging extras, and migration notes (3.0.0).
