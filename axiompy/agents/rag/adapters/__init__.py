"""RAG Adapters - Infrastructure implementations.

Contains implementations for:
- embedders/: Embedding adapters (SentenceTransformer, FastEmbed, etc.)
- vector_stores/: Vector store adapters (InMemory, Chroma, etc.)
- llm/: LLM adapters (ReasoningAdapter)
- mocks.py: Mock implementations for testing

Local-first adapters (no external services):
- SentenceTransformerEmbedder: PyTorch-based (sentence-transformers)
- FastEmbedEmbedder: ONNX-based (fastembed)
- InMemoryVectorStore: NumPy-based vector store
- ReasoningAdapter: Wraps axiompy.reasoning for any LLM
"""

from axiompy.agents.rag.adapters.mocks import (
    MockDocumentSource,
    MockEmbedder,
    MockLLMProvider,
    MockVectorStore,
)

# Lazy imports for optional dependencies
# These are imported explicitly to avoid loading large libraries unnecessarily

__all__ = [
    # Mocks (always available)
    "MockDocumentSource",
    "MockEmbedder",
    "MockLLMProvider",
    "MockVectorStore",
    # Real adapters (import from submodules)
    # from axiompy.agents.rag.adapters.embedders import SentenceTransformerEmbedder
    # from axiompy.agents.rag.adapters.embedders import FastEmbedEmbedder
    # from axiompy.agents.rag.adapters.vector_stores import InMemoryVectorStore
    # from axiompy.agents.rag.adapters.llm import ReasoningAdapter
]
