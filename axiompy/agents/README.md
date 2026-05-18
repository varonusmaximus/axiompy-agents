# AxiomPy Agents

Documentation and examples for building agentic applications on **axiompy-agents**.

## Platform modules

| Module | Purpose |
|--------|---------|
| [`axiompy.kernel`](../kernel/) | Hexagonal agent runtime (native loop, tools, memory, events) |
| [`axiompy.agents.io`](io/) | Embeddings, vector stores, document sources (compose with kernel) |
| [`axiompy.reasoning`](../reasoning/) | Provider-agnostic LLM clients; use `ReasoningLLMAdapter` with kernel |

## Quick example

```python
from axiompy.kernel import KernelFactory, KernelSettings, RuntimeType
from axiompy.reasoning import ReasoningFactory, ReasoningProvider
from axiompy.reasoning.adapter import ReasoningLLMAdapter

llm = ReasoningFactory.create(ReasoningProvider.OLLAMA)
kernel = KernelFactory.create(
    RuntimeType.NATIVE,
    KernelSettings(llm=ReasoningLLMAdapter(llm)),
)
print(kernel.run("Hello").output)
```

## Examples

See [`examples/`](examples/) for minimal CLIs and composition patterns.
