"""IO component settings and type enums.

Configuration types for embeddings, vector stores, and document chunkers.
Compose these with axiompy.kernel for agentic retrieval workflows.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from axiompy.agents.io.defaults import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from axiompy.agents.io.documents.chunker import FixedSizeChunker, ParagraphChunker, SentenceChunker
from axiompy.validators import ensure_in_range, ensure_positive


class EmbedderType(str, Enum):
    """Supported embedding providers."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"
    FASTEMBED = "fastembed"
    OLLAMA = "ollama"
    OPENAI = "openai"
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


@dataclass
class EmbedderSettings:
    """Configuration for embedders."""

    model: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    cache_dir: Optional[str] = None
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    local_files_only: bool = True

    def __post_init__(self) -> None:
        ensure_positive(self.batch_size, "batch_size must be positive")


@dataclass
class VectorStoreSettings:
    """Configuration for vector stores."""

    collection_name: str = "default"
    persist_path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    database_url: Optional[str] = None


@dataclass
class LLMSettings:
    """Configuration for LLM providers when wiring reasoning adapters."""

    model: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        ensure_in_range(self.temperature, 0.0, 1.0, "temperature must be between 0.0 and 1.0")
        ensure_positive(self.max_tokens, "max_tokens must be positive")


@dataclass
class ChunkerSettings:
    """Configuration for document chunking."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        ensure_positive(self.chunk_size, "chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


class ChunkerFactory:
    """Factory for creating document chunkers."""

    @staticmethod
    def create(
        chunker_type: ChunkerType,
        settings: ChunkerSettings,
    ) -> FixedSizeChunker | SentenceChunker | ParagraphChunker:
        match chunker_type:
            case ChunkerType.FIXED_SIZE:
                return FixedSizeChunker(
                    _chunk_size=settings.chunk_size,
                    _chunk_overlap=settings.chunk_overlap,
                )
            case ChunkerType.SENTENCE:
                return SentenceChunker(
                    _target_size=settings.chunk_size,
                    _overlap_sentences=1,
                )
            case ChunkerType.PARAGRAPH:
                return ParagraphChunker(
                    _target_size=settings.chunk_size,
                    _merge_small=True,
                )
            case _:
                raise ValueError(f"Unknown chunker type: {chunker_type}")

    @staticmethod
    def create_mock() -> FixedSizeChunker:
        return FixedSizeChunker()
