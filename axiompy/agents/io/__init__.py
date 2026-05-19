"""IO primitives for agentic applications.

Embeddings, vector stores, and document loading/chunking — compose with axiompy.kernel.
"""

from axiompy.agents.io.embeddings import EmbedderFactory
from axiompy.agents.io.errors import (
    AgentIOConfigurationError,
    AgentIOEmbeddingError,
    AgentIOError,
    AgentIOIngestionError,
    AgentIOLLMError,
    AgentIOQueryError,
    AgentIOVectorStoreError,
)
from axiompy.agents.io.mocks import (
    MockDocumentSource,
    MockEmbedder,
    MockLLMProvider,
    MockVectorStore,
)
from axiompy.agents.io.ports import (
    DocumentChunker,
    DocumentSource,
    Embedder,
    LLMProvider,
    VectorStore,
)
from axiompy.agents.io.settings import (
    ChunkerFactory,
    ChunkerSettings,
    ChunkerType,
    EmbedderSettings,
    EmbedderType,
    LLMSettings,
    VectorStoreSettings,
    VectorStoreType,
)
from axiompy.agents.io.types import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    Query,
    RetrievalResponse,
    SearchResult,
)
from axiompy.agents.io.vector import VectorStoreFactory

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "Query",
    "RetrievalResponse",
    "SearchResult",
    "DocumentChunker",
    "DocumentSource",
    "Embedder",
    "LLMProvider",
    "VectorStore",
    "EmbedderType",
    "EmbedderSettings",
    "VectorStoreType",
    "VectorStoreSettings",
    "ChunkerType",
    "ChunkerSettings",
    "LLMSettings",
    "ChunkerFactory",
    "EmbedderFactory",
    "VectorStoreFactory",
    "MockDocumentSource",
    "MockEmbedder",
    "MockLLMProvider",
    "MockVectorStore",
    "AgentIOError",
    "AgentIOConfigurationError",
    "AgentIOIngestionError",
    "AgentIOEmbeddingError",
    "AgentIOVectorStoreError",
    "AgentIOQueryError",
    "AgentIOLLMError",
]
