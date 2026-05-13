"""RAG Service - Core domain service.

Orchestrates the RAG workflow using injected dependencies.
All dependencies are ports (protocols), not concrete implementations.

Example:
    service = RAGService(
        document_source=FileSystemSource(),
        chunker=FixedSizeChunker(),
        embedder=OpenAIEmbedder(...),
        vector_store=ChromaVectorStore(...),
        llm_provider=ReasoningAdapter(...),
    )

    # Ingest documents
    count = service.ingest_documents(["./docs/"])

    # Query
    response = service.query("What is the main feature?")
    print(response.answer)
"""

# pylint: disable=assignment-from-no-return,not-an-iterable
# Note: The above pylint disables are necessary because pylint doesn't understand
# that Protocol methods are abstract interfaces implemented by concrete classes.
# The actual implementations DO return values/iterables.

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from axiompy.agents.rag.defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_SCORE,
    DEFAULT_RAG_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
)
from axiompy.agents.rag.domain.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    Query,
    RAGResponse,
    SearchResult,
)
from axiompy.agents.rag.domain.ports import (
    DocumentChunker,
    DocumentSource,
    Embedder,
    LLMProvider,
    VectorStore,
)
from axiompy.agents.rag.errors import (
    RAGEmbeddingError,
    RAGIngestionError,
    RAGQueryError,
)
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


@dataclass
class RAGService:
    """
    Core service for Retrieval-Augmented Generation.

    Orchestrates the RAG workflow using injected dependencies:
    1. Ingestion: Load → Chunk → Embed → Store
    2. Query: Embed query → Search → Build context → Generate

    All dependencies are ports (protocols), enabling easy testing
    and swapping of implementations.

    Attributes:
        document_source: Port for loading documents
        chunker: Port for splitting documents into chunks
        embedder: Port for generating embeddings
        vector_store: Port for storing and searching embeddings
        llm_provider: Port for generating responses
        prompt_template: RAG prompt template with {context} and {question} placeholders
    """

    document_source: DocumentSource
    chunker: DocumentChunker
    embedder: Embedder
    vector_store: VectorStore
    llm_provider: LLMProvider
    prompt_template: str = DEFAULT_RAG_PROMPT

    def __post_init__(self) -> None:
        """Validate service configuration."""
        if "{context}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {context} placeholder")
        if "{question}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {question} placeholder")
        logger.info("RAGService initialized")

    # -------------------------------------------------------------------------
    # Ingestion Methods
    # -------------------------------------------------------------------------

    def ingest_documents(self, paths: List[str]) -> int:
        """
        Ingest documents from paths.

        Loads documents, chunks them, generates embeddings, and stores in vector store.

        Args:
            paths: List of file paths, URLs, or identifiers

        Returns:
            Total number of chunks ingested

        Raises:
            RAGIngestionError: If ingestion fails
        """
        logger.info(f"Ingesting {len(paths)} document paths")
        total_chunks = 0

        try:
            documents = self.document_source.load_documents(paths)
            logger.debug(f"Loaded {len(documents)} documents")

            for doc in documents:
                count = self._ingest_document(doc)
                total_chunks += count

            logger.info(
                f"Ingestion complete: {total_chunks} chunks from {len(documents)} documents"
            )
            return total_chunks

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise RAGIngestionError(f"Failed to ingest documents: {e}") from e

    def ingest_document(self, path: str) -> int:
        """
        Ingest a single document.

        Args:
            path: File path, URL, or identifier

        Returns:
            Number of chunks ingested

        Raises:
            RAGIngestionError: If ingestion fails
        """
        logger.info(f"Ingesting document: {path}")

        try:
            document = self.document_source.load_document(path)
            return self._ingest_document(document)

        except Exception as e:
            logger.error(f"Ingestion failed for {path}: {e}")
            raise RAGIngestionError(f"Failed to ingest {path}: {e}") from e

    def ingest_text(
        self,
        text: str,
        source: str = "inline",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Ingest raw text directly.

        Args:
            text: Text content to ingest
            source: Source identifier
            metadata: Optional metadata

        Returns:
            Number of chunks ingested

        Raises:
            RAGIngestionError: If ingestion fails
        """
        logger.info(f"Ingesting inline text ({len(text)} chars)")

        try:
            document = Document(
                id=f"text_{hash(text) % 10000}",
                content=text,
                metadata=DocumentMetadata(
                    source=source,
                    extra=metadata or {},
                ),
            )
            return self._ingest_document(document)

        except Exception as e:
            logger.error(f"Text ingestion failed: {e}")
            raise RAGIngestionError(f"Failed to ingest text: {e}") from e

    def ingest(self, documents: List[Document]) -> Dict[str, int]:
        """
        Ingest pre-loaded Document objects.

        Use this when you have already loaded documents (e.g., from CLI).

        Args:
            documents: List of Document objects to ingest

        Returns:
            Dictionary with ingestion stats:
            - documents_processed: Number of documents processed
            - chunks_created: Total chunks created

        Raises:
            RAGIngestionError: If ingestion fails
        """
        logger.info(f"Ingesting {len(documents)} pre-loaded documents")
        total_chunks = 0

        try:
            for doc in documents:
                count = self._ingest_document(doc)
                total_chunks += count

            logger.info(
                f"Ingestion complete: {total_chunks} chunks from {len(documents)} documents"
            )
            return {
                "documents_processed": len(documents),
                "chunks_created": total_chunks,
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise RAGIngestionError(f"Failed to ingest documents: {e}") from e

    def _ingest_document(self, document: Document) -> int:
        """
        Internal: Ingest a single document.

        Args:
            document: Document to ingest

        Returns:
            Number of chunks created
        """
        # Chunk the document
        chunks = self.chunker.chunk_document(document)
        if not chunks:
            logger.debug(f"No chunks created for document {document.id}")
            return 0

        logger.debug(f"Created {len(chunks)} chunks for document {document.id}")

        # Generate embeddings
        try:
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_texts(texts)

            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                chunk.embedding = embedding

        except Exception as e:
            raise RAGEmbeddingError(f"Failed to embed chunks: {e}") from e

        # Store in vector store
        count = self.vector_store.add_chunks(chunks)
        logger.debug(f"Stored {count} chunks for document {document.id}")

        return count

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> RAGResponse:
        """
        Query the RAG system.

        Embeds the question, searches for relevant chunks, builds context,
        and generates a response using the LLM.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens for response

        Returns:
            RAGResponse with answer and sources

        Raises:
            RAGQueryError: If query fails
        """
        start_time = time.time()
        logger.info(f"Processing query: {question[:50]}...")

        try:
            # Create query object
            query = Query(text=question, top_k=top_k, min_score=min_score)

            # Embed the question
            query.embedding = self.embedder.embed_text(question)

            # Search for relevant chunks
            results = self.vector_store.search(
                query_embedding=query.embedding,
                top_k=top_k,
                min_score=min_score,
            )

            logger.debug(f"Found {len(results)} matching chunks")

            # Build context from results
            context_text = self._build_context(results)

            # Generate response
            answer = self.llm_provider.generate(
                prompt=question,
                context=context_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            response = RAGResponse(
                query=query,
                answer=answer,
                sources=results,
                context_text=context_text,
                model=self.llm_provider.model_name,
                latency_ms=latency_ms,
            )

            logger.info(f"Query complete in {latency_ms:.0f}ms")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RAGQueryError(f"Failed to process query: {e}") from e

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks without generating a response.

        Useful for debugging or when you only need retrieval.

        Args:
            question: Search query
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of search results

        Raises:
            RAGQueryError: If search fails
        """
        logger.info(f"Searching: {question[:50]}...")

        try:
            # Embed the question
            embedding = self.embedder.embed_text(question)

            # Search
            results = self.vector_store.search(
                query_embedding=embedding,
                top_k=top_k,
                min_score=min_score,
            )

            logger.debug(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RAGQueryError(f"Failed to search: {e}") from e

    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Build context string from search results.

        Args:
            results: Search results to include

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] (Source: {source}, Score: {result.score:.2f})\n{result.content}"
            )

        return "\n\n".join(context_parts)

    # -------------------------------------------------------------------------
    # Management Methods
    # -------------------------------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        """
        Delete a document and its chunks from the store.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        logger.info(f"Deleting document: {document_id}")
        count = self.vector_store.delete_document(document_id)
        logger.debug(f"Deleted {count} chunks")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "chunk_count": self.vector_store.chunk_count,
            "embedding_dimension": self.embedder.embedding_dimension,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "model": self.llm_provider.model_name,
        }
