"""Embedding provider adapters and factory."""

from typing import TYPE_CHECKING

from axiompy.agents.io.embeddings.fastembed_embedder import FastEmbedEmbedder
from axiompy.agents.io.embeddings.ollama import OllamaEmbedder
from axiompy.agents.io.embeddings.openai import OpenAIEmbedder
from axiompy.agents.io.embeddings.sentence_transformer import SentenceTransformerEmbedder

if TYPE_CHECKING:
    from axiompy.agents.io.ports import Embedder
    from axiompy.agents.io.settings import EmbedderSettings, EmbedderType


class EmbedderFactory:
    """Factory for creating Embedder port implementations."""

    @staticmethod
    def create(
        embedder_type: "EmbedderType",
        settings: "EmbedderSettings",
    ) -> "Embedder":
        from axiompy.agents.io.mocks import MockEmbedder
        from axiompy.agents.io.settings import EmbedderType

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
        from axiompy.agents.io.mocks import MockEmbedder

        return MockEmbedder(dimension=dimension)


__all__ = [
    "SentenceTransformerEmbedder",
    "FastEmbedEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "EmbedderFactory",
]
