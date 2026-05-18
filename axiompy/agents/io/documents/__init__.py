"""Document Source Adapters.

Implementations of the DocumentSource port for loading documents.

Available Sources:
- FileSystemSource: Load from local filesystem (txt, md, py, etc.)
- URLSource: Fetch from web URLs (HTML, text, JSON)
- ObjectStoreSource: Load from cloud object storage (S3, GCS, Azure)
- DatabaseSource: Load from database tables/queries
- PDFSource: Extract text from PDF files

Testing:
- MockDocumentSource: In adapters/mocks.py

Example:
    from axiompy.agents.io.documents import (
        SourceFactory,
        SourceType,
        SourceSettings,
        FileSystemSource,
    )

    # Using factory
    source = SourceFactory.create(SourceType.FILESYSTEM, SourceSettings())
    docs = source.load_documents(["./docs/"])

    # Direct instantiation
    source = FileSystemSource()
    doc = source.load_document("./README.md")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Always available
from axiompy.agents.io.documents.filesystem import FileSystemSource
from axiompy.agents.io.ports import DocumentSource
from axiompy.agents.io.errors import RAGConfigurationError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class SourceType(str, Enum):
    """Supported document source types."""

    FILESYSTEM = "filesystem"
    URL = "url"
    OBJECT_STORE = "object_store"
    DATABASE = "database"
    PDF = "pdf"
    MOCK = "mock"


@dataclass
class SourceSettings:
    """
    Configuration for document sources.

    Attributes:
        # FileSystem settings
        extensions: Set of file extensions to include
        encoding: Text encoding (default: utf-8)
        ignore_patterns: Glob patterns to ignore

        # URL settings
        timeout_secs: HTTP request timeout
        user_agent: User-Agent header
        headers: Additional HTTP headers

        # S3 settings (requires StorageSettings passed separately)
        # storage_settings: StorageSettings object

        # Database settings (requires DatabaseSettings passed separately)
        # database_type: DatabaseType enum
        # database_settings: DatabaseSettings object

        # PDF settings
        pages_as_documents: If True, each page is a separate document

        # General
        extra: Additional source-specific parameters
    """

    # FileSystem
    extensions: Optional[Set[str]] = None
    encoding: str = "utf-8"
    ignore_patterns: Optional[List[str]] = None

    # URL
    timeout_secs: int = 30
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    # PDF
    pages_as_documents: bool = False
    include_metadata: bool = True

    # General
    extra: Dict[str, Any] = field(default_factory=dict)


class SourceFactory:
    """
    Factory for creating DocumentSource instances.

    Uses enum-based type selection consistent with axiompy patterns.

    Example:
        # Simple filesystem source
        source = SourceFactory.create(SourceType.FILESYSTEM, SourceSettings())

        # URL source with custom timeout
        source = SourceFactory.create(
            SourceType.URL,
            SourceSettings(timeout_secs=60, user_agent="MyBot/1.0")
        )

        # For Object Store and Database, use dedicated factory methods
        source = SourceFactory.create_object_store(StorageType.S3, storage_settings)
        source = SourceFactory.create_database(DatabaseType.POSTGRES, db_settings)
    """

    @staticmethod
    def create(
        source_type: SourceType,
        settings: Optional[SourceSettings] = None,
    ) -> DocumentSource:
        """
        Create a DocumentSource instance.

        Args:
            source_type: Type of source to create
            settings: Source configuration

        Returns:
            Configured DocumentSource instance

        Raises:
            RAGConfigurationError: If configuration is invalid

        Note:
            For S3 and Database sources, use create_s3() or create_database()
            methods which accept the required provider-specific settings.
        """
        settings = settings or SourceSettings()

        match source_type:
            case SourceType.FILESYSTEM:
                return FileSystemSource(
                    extensions=settings.extensions,
                    encoding=settings.encoding,
                    ignore_patterns=settings.ignore_patterns,
                )

            case SourceType.URL:
                from axiompy.agents.io.documents.url import URLSource

                return URLSource(
                    timeout_secs=settings.timeout_secs,
                    user_agent=settings.user_agent,
                    headers=settings.headers,
                )

            case SourceType.PDF:
                from axiompy.agents.io.documents.pdf import PDFSource

                return PDFSource(
                    pages_as_documents=settings.pages_as_documents,
                    include_metadata=settings.include_metadata,
                )

            case SourceType.OBJECT_STORE:
                raise RAGConfigurationError(
                    "ObjectStoreSource requires StorageSettings. "
                    "Use SourceFactory.create_object_store() instead."
                )

            case SourceType.DATABASE:
                raise RAGConfigurationError(
                    "DatabaseSource requires DatabaseSettings. "
                    "Use SourceFactory.create_database() instead."
                )

            case SourceType.MOCK:
                from axiompy.agents.io.mocks import MockDocumentSource

                return MockDocumentSource()

            case _:
                raise RAGConfigurationError(f"Unknown source type: {source_type}")

    @staticmethod
    def create_object_store(
        storage_type: Any,  # StorageType from axiompy.io.object
        storage_settings: Any,  # StorageSettings from axiompy.io.object
        extensions: Optional[Set[str]] = None,
        encoding: str = "utf-8",
    ) -> DocumentSource:
        """
        Create an ObjectStoreSource.

        Args:
            storage_type: axiompy.io.object.StorageType (S3, GCS, AZURE)
            storage_settings: axiompy.io.object.StorageSettings with config
            extensions: File extensions to include (default: common text/code)
            encoding: Text encoding

        Returns:
            Configured ObjectStoreSource

        Example:
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(
                bucket="my-docs",
                region="us-east-1",
                access_key_id="...",
                secret_access_key="...",
            )
            source = SourceFactory.create_object_store(StorageType.S3, settings)
        """
        from axiompy.agents.io.documents.object_store import ObjectStoreSource

        return ObjectStoreSource(
            storage_type=storage_type,
            settings=storage_settings,
            extensions=extensions,
            encoding=encoding,
        )

    @staticmethod
    def create_database(
        database_type: Any,  # DatabaseType from axiompy.io.database
        database_settings: Any,  # DatabaseSettings from axiompy.io.database
    ) -> DocumentSource:
        """
        Create a DatabaseSource.

        Args:
            database_type: axiompy.io.database.DatabaseType enum
            database_settings: axiompy.io.database.DatabaseSettings

        Returns:
            Configured DatabaseSource

        Example:
            from axiompy.io.database import DatabaseType, DatabaseSettings

            settings = DatabaseSettings(
                host="localhost",
                port=5432,
                database="mydb",
                username="user",
                password="pass",
            )
            source = SourceFactory.create_database(
                DatabaseType.POSTGRES,
                settings
            )
        """
        from axiompy.agents.io.documents.database import DatabaseSource

        return DatabaseSource(
            database_type=database_type,
            settings=database_settings,
        )

    @staticmethod
    def create_mock() -> DocumentSource:
        """Create a mock source for testing."""
        from axiompy.agents.io.mocks import MockDocumentSource

        return MockDocumentSource()


# Convenience exports
__all__ = [
    # Factory
    "SourceFactory",
    "SourceType",
    "SourceSettings",
    # Direct imports
    "FileSystemSource",
]


# Lazy imports for optional sources (to avoid import errors if deps missing)
def __getattr__(name: str):
    """Lazy load optional source classes."""
    if name == "URLSource":
        from axiompy.agents.io.documents.url import URLSource

        return URLSource
    elif name == "ObjectStoreSource":
        from axiompy.agents.io.documents.object_store import ObjectStoreSource

        return ObjectStoreSource
    elif name == "DatabaseSource":
        from axiompy.agents.io.documents.database import DatabaseSource

        return DatabaseSource
    elif name == "PDFSource":
        from axiompy.agents.io.documents.pdf import PDFSource

        return PDFSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
