"""Vector Store Adapters.

Implementations of the VectorStore port for different backends.

Available (local):
- InMemoryVectorStore: Simple numpy-based store, no persistence
- ChromaVectorStore: ChromaDB with optional persistence

Cloud:
- PineconeVectorStore: Cloud Pinecone vector database

Database:
- PGVectorStore: PostgreSQL + pgvector

Testing:
- MockVectorStore: In adapters/mocks.py

Factory:
- VectorStoreFactory: Enum-based creation (consistent with axiompy patterns)
"""

from typing import TYPE_CHECKING

from axiompy.agents.io.vector.memory import InMemoryVectorStore
from axiompy.agents.io.vector.pgvector import PGVectorStore
from axiompy.agents.io.vector.pinecone import PineconeVectorStore

# ChromaDB is optional - only import if installed
_HAS_CHROMA = False
try:
    from axiompy.agents.io.vector.chroma import ChromaVectorStore

    _HAS_CHROMA = True
except ImportError:
    ChromaVectorStore = None  # type: ignore

if TYPE_CHECKING:
    from axiompy.agents.io.ports import VectorStore
    from axiompy.agents.io.settings import VectorStoreSettings, VectorStoreType


class VectorStoreFactory:
    """
    Factory for creating VectorStore instances.

    Uses enum-based type selection consistent with axiompy conventions
    (DatabaseFactory, ServerFactory, ObjectStorageFactory).

    Example:
        from axiompy.agents.io import VectorStoreType, VectorStoreSettings
        from axiompy.agents.io.vector import VectorStoreFactory

        store = VectorStoreFactory.create(
            VectorStoreType.CHROMA,
            VectorStoreSettings(persist_path="./data", collection_name="docs")
        )
    """

    @staticmethod
    def create(
        vector_store_type: "VectorStoreType",
        settings: "VectorStoreSettings",
    ) -> "VectorStore":
        """
        Create a VectorStore based on type.

        Args:
            vector_store_type: Type of vector store to create
            settings: Vector store configuration

        Returns:
            Configured VectorStore instance

        Raises:
            ValueError: If vector_store_type is unknown or required settings missing
            ImportError: If required dependency is not installed
        """
        # Import here to avoid circular imports
        from axiompy.agents.io.mocks import MockVectorStore
        from axiompy.agents.io.settings import VectorStoreType

        match vector_store_type:
            case VectorStoreType.MOCK:
                return MockVectorStore()

            case VectorStoreType.MEMORY:
                return InMemoryVectorStore()

            case VectorStoreType.CHROMA:
                if not _HAS_CHROMA:
                    raise ImportError(
                        "chromadb is required for CHROMA vector store. "
                        "Install with: pip install chromadb"
                    )
                return ChromaVectorStore(
                    persist_path=settings.persist_path,
                    collection_name=settings.collection_name,
                )

            case VectorStoreType.PINECONE:
                if not settings.api_key:
                    raise ValueError("api_key required for Pinecone")
                if not settings.host:
                    raise ValueError("host (Pinecone index URL) required for Pinecone")
                return PineconeVectorStore(
                    api_key=settings.api_key,
                    index_name=settings.collection_name,
                    host=settings.host,
                )

            case VectorStoreType.PGVECTOR:
                if not settings.database_url:
                    raise ValueError("database_url required for pgvector")
                return PGVectorStore(
                    database_url=settings.database_url,
                    table_name=settings.collection_name,
                )

            case _:
                raise ValueError(f"Unknown vector store type: {vector_store_type}")

    @staticmethod
    def create_mock() -> "VectorStore":
        """
        Create a mock vector store for testing.

        Returns:
            MockVectorStore instance
        """
        from axiompy.agents.io.mocks import MockVectorStore

        return MockVectorStore()


# Build __all__ based on what's available
__all__ = [
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "PGVectorStore",
    "VectorStoreFactory",
]
if _HAS_CHROMA:
    __all__.append("ChromaVectorStore")
