"""RAG Factory - Dependency injection for RAGService.

Provides factory method to create properly configured RAGService instances
using enum-based type selection for embedders, vector stores, and LLM providers.

Example:
    from axiompy.agents.rag import (
        RAGServiceFactory,
        EmbedderType,
        VectorStoreType,
        EmbedderSettings,
        VectorStoreSettings,
    )
    from axiompy.reasoning import ReasoningProvider

    # Create RAG service with explicit configuration
    rag = RAGServiceFactory.create(
        embedder_type=EmbedderType.OPENAI,
        vector_store_type=VectorStoreType.CHROMA,
        llm_provider=ReasoningProvider.OPENAI,
        embedder_settings=EmbedderSettings(api_key="sk-..."),
        vector_store_settings=VectorStoreSettings(persist_path="./data"),
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from axiompy.agents.rag.adapters.mocks import (
    MockDocumentSource,
    MockEmbedder,
    MockLLMProvider,
    MockVectorStore,
)
from axiompy.agents.rag.defaults import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RAG_PROMPT,
    DEFAULT_TEMPERATURE,
)
from axiompy.agents.rag.domain.chunker import (
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
)
from axiompy.agents.rag.domain.service import RAGService
from axiompy.agents.rag.errors import RAGConfigurationError
from axiompy.loggers import LoggerFactory
from axiompy.validators import ensure_in_range, ensure_positive

# Lazy imports for optional dependencies (done in _create_* functions):
# - SentenceTransformerEmbedder (requires sentence-transformers)
# - FastEmbedEmbedder (requires fastembed)
# - InMemoryVectorStore (requires numpy)
# - ReasoningAdapter (requires axiompy.reasoning)

logger = LoggerFactory.create_logger(__name__)


# =============================================================================
# Type Enums
# =============================================================================


class EmbedderType(str, Enum):
    """Supported embedding providers."""

    # Local, in-process (no external services)
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    FASTEMBED = "fastembed"

    # Local server
    OLLAMA = "ollama"

    # Cloud APIs
    OPENAI = "openai"

    # Testing
    MOCK = "mock"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"
    MEMORY = "memory"
    MOCK = "mock"


class ChunkerType(str, Enum):
    """Supported chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


# =============================================================================
# Settings Dataclasses
# =============================================================================


@dataclass
class EmbedderSettings:
    """
    Configuration for embedders.

    Attributes:
        model: Provider-specific model name (uses provider default if not set)
        api_key: API key for cloud providers (required for OpenAI)
        endpoint: Custom endpoint URL (for Ollama or custom deployments)
        cache_dir: Local directory for cached/downloaded models
        batch_size: Batch size for embedding multiple texts
        local_files_only: If True, only use local cached models (default: True)
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    cache_dir: Optional[str] = None
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    local_files_only: bool = True  # Default to local-only (no network calls)

    def __post_init__(self) -> None:
        """Validate embedder settings."""
        ensure_positive(self.batch_size, "batch_size must be positive")


@dataclass
class VectorStoreSettings:
    """
    Configuration for vector stores.

    Attributes:
        collection_name: Name of the collection/index
        persist_path: Path for local persistence (Chroma)
        host: Host for remote stores
        port: Port for remote stores
        api_key: API key for cloud stores (Pinecone)
        database_url: Connection URL for database stores (pgvector)
    """

    collection_name: str = "default"
    persist_path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    database_url: Optional[str] = None


@dataclass
class LLMSettings:
    """
    Configuration for LLM providers.

    Attributes:
        model: Model name (gpt-4o, claude-3, llama3, etc.)
        api_key: API key for cloud providers
        endpoint: Custom endpoint URL
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        """Validate LLM settings."""
        ensure_in_range(self.temperature, 0.0, 1.0, "temperature must be between 0.0 and 1.0")
        ensure_positive(self.max_tokens, "max_tokens must be positive")


@dataclass
class ChunkerSettings:
    """
    Configuration for document chunking.

    Attributes:
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        """Validate chunker settings."""
        ensure_positive(self.chunk_size, "chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


# =============================================================================
# Factory
# =============================================================================


class RAGServiceFactory:
    """
    Factory for creating RAGService instances.

    Uses enum-based type selection consistent with axiompy patterns
    (DatabaseFactory, ObjectStorageFactory, ReasoningFactory).

    Example:
        # Full configuration
        rag = RAGServiceFactory.create(
            embedder_type=EmbedderType.OPENAI,
            vector_store_type=VectorStoreType.CHROMA,
            llm_provider=ReasoningProvider.OPENAI,
            embedder_settings=EmbedderSettings(api_key="sk-..."),
            vector_store_settings=VectorStoreSettings(persist_path="./data"),
        )

        # Mock for testing
        rag = RAGServiceFactory.create_mock()
    """

    @staticmethod
    def create(
        embedder_type: EmbedderType,
        vector_store_type: VectorStoreType,
        llm_provider: str,  # ReasoningProvider enum value
        chunker_type: ChunkerType = ChunkerType.FIXED_SIZE,
        embedder_settings: Optional[EmbedderSettings] = None,
        vector_store_settings: Optional[VectorStoreSettings] = None,
        llm_settings: Optional[LLMSettings] = None,
        chunker_settings: Optional[ChunkerSettings] = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
    ) -> RAGService:
        """
        Create a RAGService with specified component types.

        Args:
            embedder_type: Embedding provider enum
            vector_store_type: Vector store backend enum
            llm_provider: LLM provider (use ReasoningProvider enum value)
            chunker_type: Chunking strategy enum (default: FIXED_SIZE)
            embedder_settings: Embedder configuration
            vector_store_settings: Vector store configuration
            llm_settings: LLM configuration
            chunker_settings: Chunker configuration
            prompt_template: RAG prompt template

        Returns:
            Configured RAGService instance

        Raises:
            RAGConfigurationError: If configuration is invalid
            ValueError: If required settings are missing
        """
        logger.info(
            f"Creating RAGService: embedder={embedder_type.value}, "
            f"vector_store={vector_store_type.value}, llm={llm_provider}"
        )

        # Apply defaults
        emb_settings = embedder_settings or EmbedderSettings()
        vs_settings = vector_store_settings or VectorStoreSettings()
        llm_cfg = llm_settings or LLMSettings()
        chunk_settings = chunker_settings or ChunkerSettings()

        try:
            # Create components using sub-factories (axiompy convention)
            from axiompy.agents.rag.adapters.embedders import EmbedderFactory
            from axiompy.agents.rag.adapters.vector_stores import VectorStoreFactory

            embedder = EmbedderFactory.create(embedder_type, emb_settings)
            vector_store = VectorStoreFactory.create(vector_store_type, vs_settings)
            llm = _create_llm_provider(llm_provider, llm_cfg)
            chunker = ChunkerFactory.create(chunker_type, chunk_settings)

            # Document source - use FileSystemSource by default
            from axiompy.agents.rag.adapters.sources import FileSystemSource

            document_source = FileSystemSource()

            return RAGService(
                document_source=document_source,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                llm_provider=llm,
                prompt_template=prompt_template,
            )

        except Exception as e:
            logger.error(f"Failed to create RAGService: {e}")
            raise RAGConfigurationError(f"Failed to create RAGService: {e}") from e

    @staticmethod
    def create_mock(
        mock_response: str = "Mock response",
        mock_embedding_dimension: int = 384,
    ) -> RAGService:
        """
        Create mock RAGService for testing.

        All components are mocked with controllable responses.

        Args:
            mock_response: Default LLM response
            mock_embedding_dimension: Embedding dimension for mock embedder

        Returns:
            RAGService with all mock components
        """
        logger.info("Creating mock RAGService")

        return RAGService(
            document_source=MockDocumentSource(),
            chunker=FixedSizeChunker(),
            embedder=MockEmbedder(dimension=mock_embedding_dimension),
            vector_store=MockVectorStore(),
            llm_provider=MockLLMProvider(response=mock_response),
        )


class ChunkerFactory:
    """
    Factory for creating document chunkers.

    Uses enum-based type selection consistent with axiompy conventions
    (DatabaseFactory, ServerFactory, etc.).

    Example:
        from axiompy.agents.rag import ChunkerType, ChunkerSettings

        chunker = ChunkerFactory.create(
            ChunkerType.FIXED_SIZE,
            ChunkerSettings(chunk_size=500, chunk_overlap=50)
        )
    """

    @staticmethod
    def create(
        chunker_type: ChunkerType,
        settings: ChunkerSettings,
    ) -> FixedSizeChunker | SentenceChunker | ParagraphChunker:
        """
        Create a chunker based on type.

        Args:
            chunker_type: Type of chunker to create
            settings: Chunker configuration

        Returns:
            Configured chunker instance

        Raises:
            ValueError: If chunker_type is unknown
        """
        match chunker_type:
            case ChunkerType.FIXED_SIZE:
                return FixedSizeChunker(
                    _chunk_size=settings.chunk_size,
                    _chunk_overlap=settings.chunk_overlap,
                )
            case ChunkerType.SENTENCE:
                return SentenceChunker(
                    _target_size=settings.chunk_size,
                    _overlap_sentences=1,  # Default overlap
                )
            case ChunkerType.PARAGRAPH:
                return ParagraphChunker(
                    _target_size=settings.chunk_size,
                    _merge_small=True,  # Default merge behavior
                )
            case _:
                raise ValueError(f"Unknown chunker type: {chunker_type}")

    @staticmethod
    def create_mock() -> FixedSizeChunker:
        """
        Create a mock chunker for testing.

        Returns:
            Default FixedSizeChunker with standard settings
        """
        return FixedSizeChunker()


# =============================================================================
# LLM Provider Creation (uses axiompy.reasoning factory)
# =============================================================================


def _create_llm_provider(llm_provider: str, settings: LLMSettings):
    """Create LLM provider based on type."""
    from axiompy.agents.rag.adapters.llm import ReasoningAdapter
    from axiompy.reasoning import ReasoningFactory, ReasoningProvider

    # Normalize provider string
    provider = llm_provider.lower() if isinstance(llm_provider, str) else str(llm_provider).lower()

    match provider:
        case "mock":
            return MockLLMProvider()

        case "ollama":
            ai_client = ReasoningFactory.create(
                ReasoningProvider.OLLAMA,
                model=settings.model or "mistral",
                endpoint=settings.endpoint or "http://localhost:11434/api/generate",
            )
            return ReasoningAdapter(ai_client)

        case "openai":
            if not settings.api_key:
                raise ValueError("api_key required for OpenAI")
            ai_client = ReasoningFactory.create(
                ReasoningProvider.OPENAI,
                model=settings.model or "gpt-3.5-turbo",
                api_key=settings.api_key,
            )
            return ReasoningAdapter(ai_client)

        case "anthropic":
            if not settings.api_key:
                raise ValueError("api_key required for Anthropic")
            ai_client = ReasoningFactory.create(
                ReasoningProvider.ANTHROPIC,
                model=settings.model or "claude-2",
                api_key=settings.api_key,
            )
            return ReasoningAdapter(ai_client)

        case _:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
