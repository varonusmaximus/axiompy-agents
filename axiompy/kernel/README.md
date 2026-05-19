# axiompy.kernel

Hexagonal agent runtime: domain ports, native plan-act loop, and optional framework adapters.

## Quick start

```python
from axiompy.kernel import KernelFactory, KernelSettings, RuntimeType
from axiompy.kernel.adapters.llm.mock import MockLLMPort

kernel = KernelFactory.create(
    RuntimeType.NATIVE,
    KernelSettings(llm=MockLLMPort(responses=["Done."])),
)
result = kernel.run("Summarize the goals.")
print(result.output)
```

## Components

| Piece | Role |
|-------|------|
| `KernelFactory` | Enum-based runtime selection (`RuntimeType`) |
| `KernelSettings` | Explicit ports: LLM, tools, memory, events, checkpoints |
| `NativeAgentRuntime` | Default loop: LLM → tools → observe |
| `LangGraphRuntime` / `LangChainRuntime` | Optional extras; **currently fall back to native** with a per-instance warning on first `run()` |

## Ports

- `LLMPort` — completions and tool calls
- `ToolRegistry` — register and invoke tools
- `MemoryStore` — session message history
- `EventPublisher` — run/step events
- `CheckpointStore` — persist run state
- `AgentCoordinator` — multi-agent orchestration (sequential adapter included)

## Testing

```bash
pytest tests/test_kernel.py -v
```

Use `KernelFactory.create_mock()` for canned LLM responses.
