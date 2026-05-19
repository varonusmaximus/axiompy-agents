"""PostgreSQL pgvector Vector Store.

Vector store using PostgreSQL with the pgvector extension.

Features:
- Persistent storage in PostgreSQL
- Cosine similarity search (default)
- Metadata filtering via SQL
- Efficient indexing with HNSW or IVFFlat

Example:
    from axiompy.agents.io.vector import PGVectorStore

    store = PGVectorStore(
        database_url="postgresql://user:pass@localhost/db",
        table_name="embeddings",
    )

    store.add_chunks(chunks_with_embeddings)
    results = store.search(query_embedding, top_k=5)

Requirements:
    - PostgreSQL with pgvector extension installed
    - psycopg2-binary package

Setup SQL:
    CREATE EXTENSION IF NOT EXISTS vector;
"""

import contextlib
import json
from typing import Any, Dict, List, Optional

from axiompy.agents.io.errors import AgentIOVectorStoreError
from axiompy.agents.io.sql_identifiers import validate_sql_identifier
from axiompy.agents.io.types import DocumentChunk, SearchResult
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class PGVectorStore:
    """
    PostgreSQL pgvector vector store.

    Stores embeddings in PostgreSQL using the pgvector extension.

    Attributes:
        chunk_count: Number of chunks in the store
        table_name: Name of the database table

    Example:
        store = PGVectorStore(database_url="postgresql://...")
        store.add_chunks(chunks)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        database_url: str,
        table_name: str = "rag_chunks",
        embedding_dimension: int = 384,
        create_table: bool = True,
    ) -> None:
        """
        Initialize pgvector store.

        Args:
            database_url: PostgreSQL connection URL
            table_name: Name of table to store embeddings
            embedding_dimension: Dimension of embedding vectors
            create_table: Whether to create the table if not exists

        Raises:
            AgentIOVectorStoreError: If cannot connect or psycopg2 not installed
        """
        if not database_url:
            raise AgentIOVectorStoreError("PostgreSQL database_url is required")

        try:
            import psycopg2
            from psycopg2.extras import execute_values
        except ImportError as e:
            raise AgentIOVectorStoreError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            ) from e

        self._database_url = database_url
        self._table_name = validate_sql_identifier(table_name, "table_name")
        self._embedding_dim = embedding_dimension

        # Store module reference
        self._psycopg2 = psycopg2
        self._execute_values = execute_values

        try:
            self._conn = psycopg2.connect(database_url)
            self._conn.autocommit = True

            if create_table:
                self._create_table()

            logger.info(
                f"PGVectorStore initialized: table={table_name}, dimension={embedding_dimension}"
            )

        except Exception as e:
            raise AgentIOVectorStoreError(f"Failed to connect to PostgreSQL: {e}") from e

    def _create_table(self) -> None:
        """Create the embeddings table if not exists."""
        with self._conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            cur.execute(f"""  # nosec B608
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    embedding vector({self._embedding_dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for vector similarity search
            cur.execute(f"""  # nosec B608
                CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
                ON {self._table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Create index for document_id lookups
            cur.execute(f"""  # nosec B608
                CREATE INDEX IF NOT EXISTS {self._table_name}_document_id_idx
                ON {self._table_name} (document_id)
            """)

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of document chunks (must have embeddings)

        Returns:
            Number of chunks added

        Raises:
            AgentIOVectorStoreError: If chunks missing embeddings or insert fails
        """
        if not chunks:
            return 0

        # Prepare data for insert
        values = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise AgentIOVectorStoreError(
                    f"Chunk {i} missing embedding. Embed chunks before adding."
                )

            # Validate embedding dimension
            if len(chunk.embedding) != self._embedding_dim:
                raise AgentIOVectorStoreError(
                    f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                    f"got {len(chunk.embedding)}"
                )

            values.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    str(chunk.embedding),  # pgvector expects string format
                    json.dumps(chunk.metadata),
                )
            )

        try:
            with self._conn.cursor() as cur:
                self._execute_values(
                    cur,
                    f"""
                    INSERT INTO {self._table_name}  # nosec B608
                    (id, document_id, content, chunk_index, start_char, end_char,
                     embedding, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        content = EXCLUDED.content,
                        chunk_index = EXCLUDED.chunk_index,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    values,
                )

            logger.debug(f"Added {len(chunks)} chunks to pgvector")
            return len(chunks)

        except Exception as e:
            raise AgentIOVectorStoreError(f"Failed to insert chunks: {e}") from e

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
            filters: Optional metadata filters (simple key=value)

        Returns:
            List of search results ordered by similarity (descending)
        """
        # Build query
        embedding_str = str(query_embedding)

        # Build WHERE clause for filters
        where_clauses = []
        params = [embedding_str]

        if filters:
            for key, value in filters.items():
                where_clauses.append(f"metadata->>'{key}' = %s")
                params.append(str(value))

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Score is 1 - cosine_distance (cosine_distance ranges from 0 to 2)
        query = f"""
            SELECT
                id,
                document_id,
                content,
                chunk_index,
                start_char,
                end_char,
                embedding::text,
                metadata,
                1 - (embedding <=> %s::vector) as score
            FROM {self._table_name}  # nosec B608
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT {top_k}
        """

        params.append(embedding_str)

        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            results = []
            for row in rows:
                score = float(row[8])

                # Apply min_score filter
                if score < min_score:
                    continue

                # Parse embedding from string
                embedding = None
                if row[6]:
                    # Remove brackets and parse
                    emb_str = row[6].strip("[]")
                    embedding = [float(x) for x in emb_str.split(",")]

                chunk = DocumentChunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    chunk_index=row[3],
                    start_char=row[4],
                    end_char=row[5],
                    embedding=embedding,
                    metadata=row[7] if row[7] else {},
                )
                results.append(SearchResult(chunk=chunk, score=score))

            return results

        except Exception as e:
            raise AgentIOVectorStoreError(f"Search failed: {e}") from e

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE document_id = %s",  # nosec B608
                    (document_id,),
                )
                deleted = cur.rowcount

            logger.debug(f"Deleted {deleted} chunks for document: {document_id}")
            return deleted

        except Exception as e:
            raise AgentIOVectorStoreError(f"Delete failed: {e}") from e

    def clear(self) -> None:
        """Clear all chunks from the store."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self._table_name}")  # nosec B608

            logger.info("Cleared all chunks from pgvector store")

        except Exception as e:
            raise AgentIOVectorStoreError(f"Clear failed: {e}") from e

    @property
    def chunk_count(self) -> int:
        """Get total number of chunks in the store."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")  # nosec B608
                result = cur.fetchone()
                return result[0] if result else 0

        except Exception:
            return 0

    @property
    def table_name(self) -> str:
        """Get the table name."""
        return self._table_name

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.debug("Closed pgvector connection")

    def __del__(self) -> None:
        """Cleanup on destruction."""
        with contextlib.suppress(Exception):
            self.close()

    def __repr__(self) -> str:
        return f"PGVectorStore(table={self._table_name}, dimension={self._embedding_dim})"
