# axiompy.agents.io

IO primitives for retrieval workflows: embeddings, vector stores, document sources, chunkers. Compose with `axiompy.kernel` — there is no monolithic `RAGService`.

## Sub-factories

| Factory | Enum | Module |
|---------|------|--------|
| `EmbedderFactory` | `EmbedderType` | `embeddings/` |
| `VectorStoreFactory` | `VectorStoreType` | `vector/` |
| `SourceFactory` | `SourceType` | `documents/` |
| `ChunkerFactory` | `ChunkerType` | `documents/chunker.py` |

Each factory provides `create(type, settings)` and `create_mock()`. `SourceSettings` validates encoding and HTTP timeout in `__post_init__`.

## Compose with kernel

```python
from axiompy.agents.io import EmbedderFactory, VectorStoreFactory, EmbedderType, VectorStoreType
from axiompy.agents.io.settings import EmbedderSettings, VectorStoreSettings

embedder = EmbedderFactory.create(EmbedderType.MOCK, EmbedderSettings())
store = VectorStoreFactory.create(VectorStoreType.MEMORY, VectorStoreSettings())
# Wire into tools / agent loop via kernel ToolRegistry
```

## SourceFactory exception

`SourceType.OBJECT_STORE` and `SourceType.DATABASE` require core axiompy settings. Use:

- `SourceFactory.create_object_store(StorageType, StorageSettings)`
- `SourceFactory.create_database(DatabaseType, DatabaseSettings)`

## Errors

All io errors inherit from `AgentIOError`:

- `AgentIOConfigurationError`, `AgentIOIngestionError`, `AgentIOEmbeddingError`
- `AgentIOVectorStoreError`, `AgentIOQueryError`, `AgentIOLLMError`

## LLM bridge

`axiompy.reasoning.llm_provider_adapter.ReasoningAdapter` implements `LLMProvider` using `AIClient`.

## Testing

```bash
pytest tests/io/test_io.py -v
```

Install extras: `pip install -e ".[io-rag,test-all]"`.
