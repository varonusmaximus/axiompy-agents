"""ChromaDB Vector Store.

Persistent vector store using ChromaDB for similarity search.
Supports both in-memory and persistent (on-disk) storage.

Features:
- Cosine similarity search (default)
- Metadata filtering
- Document-level deletion
- Persistence to disk
- Collections for data organization

Good for:
- Development with persistence
- Medium datasets (<1M chunks)
- Local deployments

Example:
    from axiompy.agents.io.vector import ChromaVectorStore

    # In-memory (ephemeral)
    store = ChromaVectorStore()

    # Persistent (survives restarts)
    store = ChromaVectorStore(
        persist_path="./chroma_data",
        collection_name="my_docs",
    )

    store.add_chunks(chunks_with_embeddings)
    results = store.search(query_embedding, top_k=5)

Requires:
    pip install chromadb
"""

from typing import Any, Dict, List, Optional

from axiompy.agents.io.types import DocumentChunk, SearchResult
from axiompy.agents.io.errors import AgentIOVectorStoreError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class ChromaVectorStore:
    """
    Vector store backed by ChromaDB.

    Supports both ephemeral (in-memory) and persistent storage.
    Uses cosine similarity by default.

    Attributes:
        chunk_count: Number of chunks in the store
        collection_name: Name of the ChromaDB collection

    Example:
        # Ephemeral
        store = ChromaVectorStore()

        # Persistent
        store = ChromaVectorStore(persist_path="./data")
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: str = "rag_chunks",
    ) -> None:
        """
        Initialize ChromaDB vector store.

        Args:
            persist_path: Path for persistent storage (None for in-memory)
            collection_name: Name of the collection to use

        Raises:
            AgentIOVectorStoreError: If ChromaDB is not installed
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise AgentIOVectorStoreError(
                "ChromaDB not installed. Install with: pip install chromadb"
            ) from e

        self._persist_path = persist_path
        self._collection_name = collection_name

        # Create client
        if persist_path:
            logger.info(f"Initializing persistent ChromaDB at: {persist_path}")
            self._client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            logger.info("Initializing ephemeral ChromaDB (in-memory)")
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaVectorStore initialized: collection={collection_name}, "
            f"chunks={self._collection.count()}"
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of document chunks (must have embeddings)

        Returns:
            Number of chunks added

        Raises:
            AgentIOVectorStoreError: If chunks missing embeddings
        """
        if not chunks:
            return 0

        # Validate and prepare data
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise AgentIOVectorStoreError(
                    f"Chunk {i} missing embedding. Embed chunks before adding."
                )

            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)

            # Build metadata dict
            meta = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            # Add chunk metadata (flatten simple values)
            for k, v in chunk.metadata.items():
                if isinstance(v, str | int | float | bool):
                    meta[k] = v
            metadatas.append(meta)

        # Upsert to ChromaDB (handles duplicates)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Added {len(chunks)} chunks (total: {self._collection.count()})")
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)
            filters: Optional metadata filters (ChromaDB where clause)

        Returns:
            List of search results ordered by similarity (descending)
        """
        if self._collection.count() == 0:
            return []

        # Build where clause for filters
        where = None
        if filters:
            # ChromaDB uses specific filter syntax
            where = self._build_where_clause(filters)

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []

        if not results["ids"] or not results["ids"][0]:
            return []

        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances, convert to similarity
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            score = 1.0 - distance

            # Apply min_score filter
            if score < min_score:
                continue

            # Reconstruct DocumentChunk
            metadata = results["metadatas"][0][i]
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=metadata.get("document_id", "unknown"),
                content=results["documents"][0][i],
                chunk_index=metadata.get("chunk_index", 0),
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0),
                embedding=results["embeddings"][0][i] if results["embeddings"] else None,
                metadata={
                    k: v
                    for k, v in metadata.items()
                    if k not in ("document_id", "chunk_index", "start_char", "end_char")
                },
            )

            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        # Get chunks for this document
        results = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if not results["ids"]:
            return 0

        # Delete by IDs
        count = len(results["ids"])
        self._collection.delete(ids=results["ids"])

        logger.debug(f"Deleted {count} chunks for document: {document_id}")
        return count

    def clear(self) -> None:
        """Clear all chunks from the store."""
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all chunks from store")

    @property
    def chunk_count(self) -> int:
        """Get total number of chunks in the store."""
        return self._collection.count()

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from simple filters.

        Args:
            filters: Simple key-value filters

        Returns:
            ChromaDB where clause
        """
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: {"$eq": value}}

        # Multiple filters use $and
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        return {"$and": conditions}

    def __repr__(self) -> str:
        return (
            f"ChromaVectorStore(collection={self._collection_name}, "
            f"chunks={self.chunk_count}, persist={self._persist_path})"
        )
