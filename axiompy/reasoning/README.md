# axiompy.reasoning

Provider-agnostic LLM clients and data-query helpers. Bridge to the kernel via `ReasoningLLMAdapter` (`LLMPort`).

## Quick start

```python
from axiompy.reasoning import ReasoningFactory, ReasoningProvider, ReasoningSettings
from axiompy.reasoning.adapter import ReasoningLLMAdapter
from axiompy.kernel import KernelFactory, KernelSettings, RuntimeType

client = ReasoningFactory.create(
    ReasoningProvider.OLLAMA,
    settings=ReasoningSettings(model="mistral"),
)
kernel = KernelFactory.create(
    RuntimeType.NATIVE,
    KernelSettings(llm=ReasoningLLMAdapter(client)),
)
```

## Core types

| Symbol | Purpose |
|--------|---------|
| `ReasoningFactory` | `create(provider, settings)` — enum + `match/case` |
| `ReasoningSettings` | model, endpoint, api_key, cache_size |
| `AIClient` | HTTP-backed completions (`axiompy.io.http`) |
| `ReasoningLLMAdapter` | Kernel `LLMPort` |
| `ReasoningAdapter` | agents.io `LLMProvider` (retrieval prompts) |
| `QueryAgent` | Metadata-driven NL → SQL routing |

## Providers

Built-in: `ReasoningProvider.OLLAMA`, `OPENAI`, `ANTHROPIC`. Provider-specific prompt formatting lives under `providers/`.

## Testing

```bash
pytest tests/test_reasoning_*.py -v
```

`ReasoningFactory.create_mock(responses=[...])` returns an `AIClient` with canned completions.

## Optional extras

```bash
pip install -e ".[openai,anthropic]"
```

Ollama uses local HTTP; no extra package required beyond core `axiompy`.
