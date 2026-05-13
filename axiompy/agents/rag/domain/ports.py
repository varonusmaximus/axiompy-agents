"""RAG Port Definitions (Interfaces).

Protocol-based interfaces for dependency injection following axiompy patterns.

Ports define the contracts that adapters must implement:
- DocumentSource: How we load documents
- Embedder: How we generate embeddings
- VectorStore: How we store and search embeddings
- LLMProvider: How we generate responses
- DocumentChunker: How we split documents into chunks

Example:
    # Ports are implemented by adapters
    class OpenAIEmbedder:
        def embed_text(self, text: str) -> List[float]:
            # OpenAI-specific implementation
            ...

    # RAGService depends on ports, not concrete implementations
    service = RAGService(
        embedder=OpenAIEmbedder(...),  # Any Embedder implementation
        ...
    )
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from axiompy.agents.rag.domain.models import Document, DocumentChunk, SearchResult


@runtime_checkable
class DocumentSource(Protocol):
    """
    Port: How we load documents.

    Implementations:
    - FileSystemSource: Load from local filesystem
    - URLSource: Fetch from URLs
    - S3Source: Load from S3
    - MockDocumentSource: For testing
    """

    def load_documents(self, paths: List[str]) -> List[Document]:
        """
        Load multiple documents from paths.

        Args:
            paths: List of file paths, URLs, or identifiers

        Returns:
            List of loaded documents
        """
        ...

    def load_document(self, path: str) -> Document:
        """
        Load a single document.

        Args:
            path: File path, URL, or identifier

        Returns:
            Loaded document
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """
    Port: How we generate embeddings.

    Implementations:
    - OpenAIEmbedder: OpenAI text-embedding-3-small/large
    - OllamaEmbedder: Local Ollama embeddings
    - HuggingFaceEmbedder: sentence-transformers models
    - MockEmbedder: For testing
    """

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """
    Port: How we store and search embeddings.

    Implementations:
    - ChromaVectorStore: Local ChromaDB
    - PineconeVectorStore: Pinecone cloud
    - PGVectorStore: PostgreSQL + pgvector
    - InMemoryVectorStore: Simple in-memory store
    - MockVectorStore: For testing
    """

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of document chunks (must have embeddings)

        Returns:
            Number of chunks added
        """
        ...

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
            min_score: Minimum similarity score threshold
            filters: Optional metadata filters

        Returns:
            List of search results ordered by similarity
        """
        ...

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        ...

    @property
    def chunk_count(self) -> int:
        """Get total number of chunks in the store."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """
    Port: How we generate responses.

    Implementations:
    - ReasoningAdapter: Wraps axiompy.reasoning.AIClient
    - MockLLMProvider: For testing
    """

    def generate(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate a response given prompt and context.

        Args:
            prompt: User question/prompt
            context: Retrieved context from vector store
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        ...

    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        ...


@runtime_checkable
class DocumentChunker(Protocol):
    """
    Port: How we split documents into chunks.

    Implementations:
    - FixedSizeChunker: Fixed character-based chunking
    - SentenceChunker: Sentence-aware chunking
    - ParagraphChunker: Paragraph-aware chunking
    """

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        ...

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size."""
        ...

    @property
    def chunk_overlap(self) -> int:
        """Get the overlap between chunks."""
        ...
