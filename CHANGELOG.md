# Changelog

## 3.0.0

### Added

- `axiompy.kernel` — native agent runtime with tools, memory, events, checkpoints, coordinator ports
- `axiompy.io` — embeddings, vector stores, document sources (migrated from `agents.rag` adapters)
- `ReasoningLLMAdapter` — kernel `LLMPort` bridge for `AIClient`
- Optional extras: `kernel`, `io-rag`, `memory-redis`, `kernel-langgraph`, `kernel-langchain`

### Removed

- `axiompy.agents.code_review` (including Docker, Terraform, deploy workflow)
- `RAGService`, `RAGServiceFactory`, and RAG CLI/API applications

### Changed

- `ReasoningFactory.create()` accepts `ReasoningSettings` (no `**kwargs`)
- Tests reorganized under `tests/io/` and `tests/test_kernel.py`
