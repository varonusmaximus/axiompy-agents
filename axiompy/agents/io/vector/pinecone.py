"""Pinecone Vector Store.

Cloud vector store using Pinecone for similarity search.

Features:
- Serverless or pod-based deployment
- High performance at scale
- Metadata filtering
- Uses axiompy HTTPClient for API calls

Example:
    from axiompy.agents.io.vector import PineconeVectorStore

    store = PineconeVectorStore(
        api_key="pk-...",
        index_name="my-index",
        environment="us-east-1",  # For pod-based
    )

    store.add_chunks(chunks_with_embeddings)
    results = store.search(query_embedding, top_k=5)

Requirements:
    - Pinecone account and API key
    - Index created in Pinecone console

Note:
    This adapter uses Pinecone's REST API directly via HTTPClient.
    For production use, consider using the official pinecone-client package.
"""

from typing import Any, Dict, List, Optional

from axiompy.agents.io.types import DocumentChunk, SearchResult
from axiompy.agents.io.errors import AgentIOVectorStoreError
from axiompy.io.http import HTTPClientFactory
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class PineconeVectorStore:
    """
    Pinecone cloud vector store.

    Uses Pinecone's serverless or pod-based vector database.

    Attributes:
        chunk_count: Approximate number of chunks in the store
        index_name: Name of the Pinecone index

    Example:
        store = PineconeVectorStore(api_key="pk-...", index_name="docs")
        store.add_chunks(chunks)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        host: Optional[str] = None,
        namespace: str = "",
        timeout_secs: int = 30,
    ) -> None:
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the index
            host: Index host URL (from Pinecone console)
            namespace: Namespace within the index (default: "")
            timeout_secs: Request timeout in seconds

        Raises:
            AgentIOVectorStoreError: If API key or index name not provided
        """
        if not api_key:
            raise AgentIOVectorStoreError("Pinecone API key is required")
        if not index_name:
            raise AgentIOVectorStoreError("Pinecone index name is required")

        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._timeout = timeout_secs
        self._chunk_count = 0

        # Host is required for direct API access
        if not host:
            raise AgentIOVectorStoreError(
                "Pinecone host URL is required. "
                "Find it in Pinecone console: https://app.pinecone.io"
            )

        self._host = host.rstrip("/")

        # Create HTTP client
        self._client = (
            HTTPClientFactory.create(timeout_secs=timeout_secs)
            .add_header("Api-Key", api_key)
            .add_header("Content-Type", "application/json")
        )

        logger.info(f"PineconeVectorStore initialized: index={index_name}, namespace={namespace}")

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of document chunks (must have embeddings)

        Returns:
            Number of chunks added

        Raises:
            AgentIOVectorStoreError: If chunks missing embeddings or API error
        """
        if not chunks:
            return 0

        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise AgentIOVectorStoreError(
                    f"Chunk {i} missing embedding. Embed chunks before adding."
                )

            # Build metadata
            metadata = {
                "document_id": chunk.document_id,
                "content": chunk.content[:1000],  # Pinecone metadata limit
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            # Add chunk metadata
            for k, v in chunk.metadata.items():
                if isinstance(v, str | int | float | bool):
                    metadata[k] = v

            vectors.append(
                {
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": metadata,
                }
            )

        # Upsert in batches (Pinecone limit: 100 vectors per request)
        batch_size = 100
        total_added = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]

            try:
                response = self._client.post(
                    f"{self._host}/vectors/upsert",
                    json={
                        "vectors": batch,
                        "namespace": self._namespace,
                    },
                )

                if response.status_code != 200:
                    raise AgentIOVectorStoreError(
                        f"Pinecone upsert failed ({response.status_code}): {response.text}"
                    )

                data = response.json()
                total_added += data.get("upsertedCount", len(batch))

            except AgentIOVectorStoreError:
                raise
            except Exception as e:
                raise AgentIOVectorStoreError(f"Pinecone upsert failed: {e}") from e

        self._chunk_count += total_added
        logger.debug(f"Added {total_added} chunks to Pinecone")
        return total_added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)
            filters: Optional metadata filters (Pinecone filter format)

        Returns:
            List of search results ordered by similarity (descending)
        """
        try:
            payload = {
                "vector": query_embedding,
                "topK": top_k,
                "includeMetadata": True,
                "includeValues": True,
                "namespace": self._namespace,
            }

            if filters:
                payload["filter"] = filters

            response = self._client.post(
                f"{self._host}/query",
                json=payload,
            )

            if response.status_code != 200:
                raise AgentIOVectorStoreError(
                    f"Pinecone query failed ({response.status_code}): {response.text}"
                )

            data = response.json()
            matches = data.get("matches", [])

            # Convert to SearchResult objects
            results = []
            for match in matches:
                score = match.get("score", 0.0)

                # Apply min_score filter
                if score < min_score:
                    continue

                metadata = match.get("metadata", {})
                chunk = DocumentChunk(
                    id=match["id"],
                    document_id=metadata.get("document_id", "unknown"),
                    content=metadata.get("content", ""),
                    chunk_index=metadata.get("chunk_index", 0),
                    start_char=metadata.get("start_char", 0),
                    end_char=metadata.get("end_char", 0),
                    embedding=match.get("values"),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k
                        not in ("document_id", "content", "chunk_index", "start_char", "end_char")
                    },
                )
                results.append(SearchResult(chunk=chunk, score=score))

            return results

        except AgentIOVectorStoreError:
            raise
        except Exception as e:
            raise AgentIOVectorStoreError(f"Pinecone search failed: {e}") from e

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted (approximate)
        """
        try:
            # Pinecone delete by filter
            response = self._client.post(
                f"{self._host}/vectors/delete",
                json={
                    "filter": {"document_id": document_id},
                    "namespace": self._namespace,
                },
            )

            if response.status_code != 200:
                raise AgentIOVectorStoreError(
                    f"Pinecone delete failed ({response.status_code}): {response.text}"
                )

            logger.debug(f"Deleted chunks for document: {document_id}")
            # Pinecone doesn't return count, estimate based on cached count
            return 1  # Return non-zero to indicate success

        except AgentIOVectorStoreError:
            raise
        except Exception as e:
            raise AgentIOVectorStoreError(f"Pinecone delete failed: {e}") from e

    @property
    def chunk_count(self) -> int:
        """Get approximate number of chunks in the store."""
        try:
            response = self._client.post(
                f"{self._host}/describe_index_stats",
                json={"filter": {}},
            )

            if response.status_code == 200:
                data = response.json()
                namespaces = data.get("namespaces", {})
                if self._namespace in namespaces:
                    return namespaces[self._namespace].get("vectorCount", 0)
                return data.get("totalVectorCount", self._chunk_count)
            return self._chunk_count

        except Exception:
            return self._chunk_count

    @property
    def index_name(self) -> str:
        """Get the index name."""
        return self._index_name

    def __repr__(self) -> str:
        return f"PineconeVectorStore(index={self._index_name}, namespace={self._namespace})"
