"""Object Store Document Source.

Load documents from cloud object storage using axiompy ObjectStorageFactory.

Supports:
- AWS S3
- Google Cloud Storage (GCS)
- Azure Blob Storage
- S3-compatible services (MinIO, etc.)

Example:
    from axiompy.agents.io.documents import ObjectStoreSource
    from axiompy.io.object import StorageSettings, StorageType

    settings = StorageSettings(
        bucket="my-docs",
        region="us-east-1",
        access_key_id="...",
        secret_access_key="...",
    )
    source = ObjectStoreSource(StorageType.S3, settings)

    # Load single object
    doc = source.load_document("docs/readme.md")

    # Load all documents under a prefix
    docs = source.load_documents(["docs/"])
"""

from typing import List, Optional, Set

from axiompy.agents.io.types import Document, DocumentMetadata
from axiompy.agents.io.errors import RAGIngestionError
from axiompy.io.object import ObjectStorageFactory, StorageSettings, StorageType
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# Default supported extensions for object storage
DEFAULT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".css",
    ".sql",
}


class ObjectStoreSource:
    """
    Document source for loading content from cloud object storage.

    Uses axiompy ObjectStorageFactory for storage operations.
    Supports S3, GCS, and Azure Blob Storage.

    Attributes:
        bucket: Bucket/container name
        storage_type: Type of storage (S3, GCS, AZURE)
        extensions: Set of supported file extensions

    Example:
        from axiompy.io.object import StorageSettings, StorageType

        settings = StorageSettings(bucket="my-bucket", region="us-east-1")
        source = ObjectStoreSource(StorageType.S3, settings)
        docs = source.load_documents(["docs/"])
    """

    def __init__(
        self,
        storage_type: StorageType,
        settings: StorageSettings,
        extensions: Optional[Set[str]] = None,
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize object store source.

        Args:
            storage_type: Type of storage (S3, GCS, AZURE)
            settings: Storage settings (bucket, credentials, etc.)
            extensions: Set of file extensions to load (default: common text/code)
            encoding: Text encoding for reading files (default: utf-8)
        """
        self._storage_type = storage_type
        self._settings = settings
        self._extensions = extensions or DEFAULT_EXTENSIONS
        self._encoding = encoding

        # Create storage client
        self._storage = ObjectStorageFactory.create(storage_type, settings)

        logger.debug(
            f"ObjectStoreSource initialized: {storage_type.value} bucket={settings.bucket}"
        )

    def load_document(self, key: str) -> Document:
        """
        Load a single document from object storage.

        Args:
            key: Object key (path)

        Returns:
            Document with content and metadata

        Raises:
            RAGIngestionError: If load fails
        """
        try:
            # Get object content
            content_bytes = self._storage.get_object(key)
            content = content_bytes.decode(self._encoding)

            # Get object metadata
            obj_meta = self._storage.head_object(key)

            # Build source URI based on storage type
            match self._storage_type:
                case StorageType.S3:
                    source_uri = f"s3://{self._settings.bucket}/{key}"
                case StorageType.GCS:
                    source_uri = f"gs://{self._settings.bucket}/{key}"
                case StorageType.AZURE:
                    source_uri = f"azure://{self._settings.bucket}/{key}"
                case _:
                    source_uri = f"{self._storage_type.value}://{self._settings.bucket}/{key}"

            # Generate document ID
            doc_id = f"{self._storage_type.value}:{self._settings.bucket}/{key}".replace("/", "_")

            metadata = DocumentMetadata(
                source=source_uri,
                title=key.split("/")[-1],  # filename
                content_type=obj_meta.content_type or "text/plain",
                created_at=obj_meta.last_modified,
                extra={
                    "storage_type": self._storage_type.value,
                    "bucket": self._settings.bucket,
                    "key": key,
                    "size": obj_meta.size,
                    "etag": obj_meta.etag,
                },
            )

            logger.debug(f"Loaded document from {source_uri} ({len(content)} chars)")

            return Document(id=doc_id, content=content, metadata=metadata)

        except Exception as e:
            raise RAGIngestionError(f"Failed to load object {key}: {e}") from e

    def load_documents(self, prefixes: List[str]) -> List[Document]:
        """
        Load documents from object storage prefixes.

        Args:
            prefixes: List of prefixes to scan

        Returns:
            List of Document objects

        Note:
            Only loads objects matching supported extensions.
            Failed objects are logged and skipped.
        """
        documents = []
        keys_processed = set()

        for prefix in prefixes:
            try:
                # List objects under prefix
                objects = self._storage.list_objects(prefix=prefix)

                for obj in objects:
                    key = obj.key

                    # Skip if already processed
                    if key in keys_processed:
                        continue
                    keys_processed.add(key)

                    # Check extension
                    ext = "." + key.split(".")[-1].lower() if "." in key else ""
                    if ext not in self._extensions:
                        logger.debug(f"Skipping unsupported extension: {key}")
                        continue

                    # Skip directories (keys ending with /)
                    if key.endswith("/"):
                        continue

                    try:
                        doc = self.load_document(key)
                        documents.append(doc)
                    except RAGIngestionError as e:
                        logger.warning(f"Skipping object {key}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to list prefix {prefix}: {e}")
                continue

        logger.info(
            f"Loaded {len(documents)} documents from "
            f"{self._storage_type.value}://{self._settings.bucket}"
        )
        return documents

    def __repr__(self) -> str:
        return f"ObjectStoreSource({self._storage_type.value}, bucket={self._settings.bucket})"
