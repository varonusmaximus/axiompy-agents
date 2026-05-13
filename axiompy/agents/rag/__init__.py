"""RAG Agent - Retrieval-Augmented Generation for axiompy.

Provides a generalized RAG implementation following axiompy patterns:
- Clean Architecture with domain/adapters/applications
- Factory pattern with enum-based type selection
- Protocol-based ports for dependency injection
- Comprehensive testing with mocks

Quick Start (Local-first, no external services):
    from axiompy.agents.rag import (
        RAGServiceFactory,
        EmbedderType,
        VectorStoreType,
    )

    # Create RAG service with local embeddings and in-memory store
    rag = RAGServiceFactory.create(
        embedder_type=EmbedderType.SENTENCE_TRANSFORMERS,  # or FASTEMBED
        vector_store_type=VectorStoreType.MEMORY,
        llm_provider="ollama",  # Local Ollama
    )

    # Ingest and query
    rag.ingest_documents(["./docs/"])
    response = rag.query("What is the main feature?")

Component Factories (axiompy convention):
    - RAGServiceFactory: Main factory for RAGService
    - EmbedderFactory: Create embedders (from adapters.embedders)
    - VectorStoreFactory: Create vector stores (from adapters.vector_stores)
    - ChunkerFactory: Create chunkers (from factory)

Available Embedders (local):
    - EmbedderType.SENTENCE_TRANSFORMERS: PyTorch-based (pip install sentence-transformers)
    - EmbedderType.FASTEMBED: ONNX-based, lighter (pip install fastembed)
    - EmbedderType.OLLAMA: Via Ollama server
    - EmbedderType.OPENAI: OpenAI API

Available Vector Stores:
    - VectorStoreType.MEMORY: In-memory numpy store (no persistence)
    - VectorStoreType.CHROMA: ChromaDB (pip install chromadb)
    - VectorStoreType.PINECONE: Pinecone cloud (pip install pinecone-client)
    - VectorStoreType.PGVECTOR: PostgreSQL + pgvector (pip install psycopg2-binary)
    - VectorStoreType.MOCK: For testing

Available LLM Providers:
    - "ollama": Local Ollama server
    - "openai": OpenAI API (requires api_key)
    - "anthropic": Anthropic API (requires api_key)
    - "mock": For testing

For comprehensive documentation, see:
    - axiompy/agents/rag/README.md
"""

# Domain - Models
# Sub-Factories (axiompy convention: each adapter module has its own factory)
from axiompy.agents.rag.adapters.embedders import EmbedderFactory

# Adapters - Mocks (for testing)
from axiompy.agents.rag.adapters.mocks import (
    MockDocumentSource,
    MockEmbedder,
    MockLLMProvider,
    MockVectorStore,
)
from axiompy.agents.rag.adapters.vector_stores import VectorStoreFactory
from axiompy.agents.rag.domain.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    Query,
    RAGResponse,
    SearchResult,
)

# Domain - Ports (Protocols)
from axiompy.agents.rag.domain.ports import (
    DocumentChunker,
    DocumentSource,
    Embedder,
    LLMProvider,
    VectorStore,
)

# Domain - Service
from axiompy.agents.rag.domain.service import RAGService

# Errors
from axiompy.agents.rag.errors import (
    RAGConfigurationError,
    RAGEmbeddingError,
    RAGError,
    RAGIngestionError,
    RAGLLMError,
    RAGQueryError,
    RAGVectorStoreError,
)

# Factory - Types and Settings
from axiompy.agents.rag.factory import (
    ChunkerFactory,
    ChunkerSettings,
    ChunkerType,
    EmbedderSettings,
    EmbedderType,
    LLMSettings,
    RAGServiceFactory,
    VectorStoreSettings,
    VectorStoreType,
)

__all__ = [
    # Domain - Models
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "Query",
    "RAGResponse",
    "SearchResult",
    # Domain - Ports
    "DocumentChunker",
    "DocumentSource",
    "Embedder",
    "LLMProvider",
    "VectorStore",
    # Domain - Service
    "RAGService",
    # Factory - Main
    "RAGServiceFactory",
    # Factory - Types and Settings
    "ChunkerSettings",
    "ChunkerType",
    "EmbedderSettings",
    "EmbedderType",
    "LLMSettings",
    "VectorStoreSettings",
    "VectorStoreType",
    # Factory - Sub-factories (axiompy convention)
    "ChunkerFactory",
    "EmbedderFactory",
    "VectorStoreFactory",
    # Mocks
    "MockDocumentSource",
    "MockEmbedder",
    "MockLLMProvider",
    "MockVectorStore",
    # Errors
    "RAGError",
    "RAGConfigurationError",
    "RAGEmbeddingError",
    "RAGIngestionError",
    "RAGLLMError",
    "RAGQueryError",
    "RAGVectorStoreError",
]
