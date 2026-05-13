"""In-Memory Vector Store.

Simple, fast vector store using numpy for similarity search.
No external dependencies beyond numpy.

Features:
- Cosine similarity search
- Metadata filtering
- Document-level deletion
- No persistence (data lost on restart)

Good for:
- Development and testing
- Small datasets (<100k chunks)
- Quick prototyping

Example:
    from axiompy.agents.rag.adapters.vector_stores import InMemoryVectorStore

    store = InMemoryVectorStore()
    store.add_chunks(chunks_with_embeddings)
    results = store.search(query_embedding, top_k=5)
"""

from typing import Any, Dict, List, Optional

import numpy as np

from axiompy.agents.rag.domain.models import DocumentChunk, SearchResult
from axiompy.agents.rag.errors import RAGVectorStoreError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class InMemoryVectorStore:
    """
    Simple in-memory vector store using numpy.

    Uses cosine similarity for search. All data is kept in memory
    and lost when the process exits.

    Attributes:
        chunk_count: Number of chunks in the store

    Example:
        store = InMemoryVectorStore()
        store.add_chunks(chunks)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self) -> None:
        """Initialize empty vector store."""
        self._chunks: List[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        logger.info("Initialized InMemoryVectorStore")

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of document chunks (must have embeddings)

        Returns:
            Number of chunks added

        Raises:
            RAGVectorStoreError: If chunks missing embeddings or dimension mismatch
        """
        if not chunks:
            return 0

        # Validate all chunks have embeddings
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise RAGVectorStoreError(
                    f"Chunk {i} missing embedding. Embed chunks before adding."
                )

        # Get embedding dimension from first chunk
        new_dim = len(chunks[0].embedding)

        # Check dimension consistency with existing embeddings
        if self._embeddings is not None:
            existing_dim = self._embeddings.shape[1]
            if new_dim != existing_dim:
                raise RAGVectorStoreError(
                    f"Embedding dimension mismatch: store has {existing_dim}, got {new_dim}"
                )

        # Add chunks to list
        self._chunks.extend(chunks)

        # Build embedding matrix
        new_embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        logger.debug(f"Added {len(chunks)} chunks (total: {len(self._chunks)})")
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
            filters: Optional metadata filters (supports exact match)

        Returns:
            List of search results ordered by similarity (highest first)
        """
        if not self._chunks or self._embeddings is None:
            return []

        # Convert query to numpy
        query = np.array(query_embedding, dtype=np.float32)

        # Normalize query for cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-9)

        # Normalize stored embeddings
        emb_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9
        normalized_emb = self._embeddings / emb_norms

        # Compute cosine similarity
        similarities = np.dot(normalized_emb, query_norm)

        # Build results with filtering
        results: List[SearchResult] = []

        for idx in np.argsort(similarities)[::-1]:  # Sort descending
            score = float(similarities[idx])

            # Apply min_score filter
            if score < min_score:
                break  # Sorted, so no more results above threshold

            chunk = self._chunks[idx]

            # Apply metadata filters
            if filters and not self._matches_filters(chunk, filters):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        logger.debug(f"Search returned {len(results)} results (top_k={top_k})")
        return results

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        if not self._chunks:
            return 0

        # Find indices to keep
        keep_indices = [
            i for i, chunk in enumerate(self._chunks) if chunk.document_id != document_id
        ]

        deleted = len(self._chunks) - len(keep_indices)

        if deleted > 0:
            # Rebuild chunks list and embeddings matrix
            self._chunks = [self._chunks[i] for i in keep_indices]

            if self._embeddings is not None and keep_indices:
                self._embeddings = self._embeddings[keep_indices]
            elif not keep_indices:
                self._embeddings = None

            logger.debug(f"Deleted {deleted} chunks for document {document_id}")

        return deleted

    def clear(self) -> None:
        """Remove all chunks from the store."""
        count = len(self._chunks)
        self._chunks = []
        self._embeddings = None
        logger.info(f"Cleared {count} chunks from store")

    @property
    def chunk_count(self) -> int:
        """Get total number of chunks in the store."""
        return len(self._chunks)

    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """
        Check if a chunk matches all filters.

        Supports:
        - Exact match on metadata fields
        - document_id filter

        Args:
            chunk: Chunk to check
            filters: Filter dictionary

        Returns:
            True if chunk matches all filters
        """
        for key, value in filters.items():
            if key == "document_id":
                if chunk.document_id != value:
                    return False
            elif chunk.metadata:
                chunk_value = chunk.metadata.get(key)
                if chunk_value != value:
                    return False
            else:
                return False

        return True

    def __repr__(self) -> str:
        return f"InMemoryVectorStore(chunks={len(self._chunks)})"
