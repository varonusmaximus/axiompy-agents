"""RAG Domain Layer - Core business logic.

Contains:
- models.py: Domain models (Document, DocumentChunk, Query, etc.)
- ports.py: Protocol definitions (interfaces)
- chunker.py: Document chunking implementations
- service.py: RAGService orchestrating the workflow
"""

from axiompy.agents.rag.domain.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    Query,
    RAGResponse,
    SearchResult,
)
from axiompy.agents.rag.domain.ports import (
    DocumentChunker,
    DocumentSource,
    Embedder,
    LLMProvider,
    VectorStore,
)
from axiompy.agents.rag.domain.service import RAGService

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "Query",
    "RAGResponse",
    "SearchResult",
    "DocumentChunker",
    "DocumentSource",
    "Embedder",
    "LLMProvider",
    "VectorStore",
    "RAGService",
]
