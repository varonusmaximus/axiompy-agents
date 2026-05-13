"""Embedder Adapters.

Implementations of the Embedder port for different providers.

Available (local, in-process):
- SentenceTransformerEmbedder: Uses sentence-transformers (PyTorch-based)
- FastEmbedEmbedder: Uses fastembed (ONNX-based, lighter)

Local server:
- OllamaEmbedder: Local Ollama server

Cloud:
- OpenAIEmbedder: OpenAI text-embedding API

Testing:
- MockEmbedder: In adapters/mocks.py

Factory:
- EmbedderFactory: Enum-based creation (consistent with axiompy patterns)
"""

from typing import TYPE_CHECKING

from axiompy.agents.rag.adapters.embedders.fastembed_embedder import FastEmbedEmbedder
from axiompy.agents.rag.adapters.embedders.ollama import OllamaEmbedder
from axiompy.agents.rag.adapters.embedders.openai import OpenAIEmbedder
from axiompy.agents.rag.adapters.embedders.sentence_transformer import (
    SentenceTransformerEmbedder,
)

if TYPE_CHECKING:
    from axiompy.agents.rag.domain.ports import Embedder
    from axiompy.agents.rag.factory import EmbedderSettings, EmbedderType


class EmbedderFactory:
    """
    Factory for creating Embedder instances.

    Uses enum-based type selection consistent with axiompy conventions
    (DatabaseFactory, ServerFactory, ObjectStorageFactory).

    Example:
        from axiompy.agents.rag import EmbedderType, EmbedderSettings
        from axiompy.agents.rag.adapters.embedders import EmbedderFactory

        embedder = EmbedderFactory.create(
            EmbedderType.FASTEMBED,
            EmbedderSettings(model="BAAI/bge-small-en-v1.5", cache_dir="./models")
        )
    """

    @staticmethod
    def create(
        embedder_type: "EmbedderType",
        settings: "EmbedderSettings",
    ) -> "Embedder":
        """
        Create an Embedder based on type.

        Args:
            embedder_type: Type of embedder to create
            settings: Embedder configuration

        Returns:
            Configured Embedder instance

        Raises:
            ValueError: If embedder_type is unknown or required settings missing
        """
        # Import here to avoid circular imports
        from axiompy.agents.rag.adapters.mocks import MockEmbedder
        from axiompy.agents.rag.factory import EmbedderType

        match embedder_type:
            case EmbedderType.MOCK:
                return MockEmbedder()

            case EmbedderType.SENTENCE_TRANSFORMERS:
                return SentenceTransformerEmbedder(
                    model_name=settings.model or "all-MiniLM-L6-v2",
                    cache_dir=settings.cache_dir,
                    local_files_only=settings.local_files_only,
                )

            case EmbedderType.FASTEMBED:
                return FastEmbedEmbedder(
                    model_name=settings.model or "BAAI/bge-small-en-v1.5",
                    cache_dir=settings.cache_dir,
                    local_files_only=settings.local_files_only,
                )

            case EmbedderType.OLLAMA:
                return OllamaEmbedder(
                    model=settings.model or "nomic-embed-text",
                    host=settings.endpoint or "http://localhost:11434",
                )

            case EmbedderType.OPENAI:
                if not settings.api_key:
                    raise ValueError("api_key required for OpenAI embedder")
                return OpenAIEmbedder(
                    api_key=settings.api_key,
                    model=settings.model or "text-embedding-3-small",
                    endpoint=settings.endpoint,
                )

            case _:
                raise ValueError(f"Unknown embedder type: {embedder_type}")

    @staticmethod
    def create_mock(dimension: int = 384) -> "Embedder":
        """
        Create a mock embedder for testing.

        Args:
            dimension: Embedding dimension

        Returns:
            MockEmbedder instance
        """
        from axiompy.agents.rag.adapters.mocks import MockEmbedder

        return MockEmbedder(dimension=dimension)


__all__ = [
    "SentenceTransformerEmbedder",
    "FastEmbedEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "EmbedderFactory",
]
