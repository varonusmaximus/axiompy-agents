"""Mock Adapters for Testing.

Provides mock implementations of all RAG ports for unit testing.
Following axiompy patterns with calls tracking and set_response methods.

Example:
    # Create mock service for testing
    rag = RAGServiceFactory.create_mock()
    rag.embedder.set_dimension(384)
    rag.llm_provider.set_response("Mocked answer")

    response = rag.query("test question")
    assert response.answer == "Mocked answer"
"""

from typing import Any, Dict, List, Optional, Tuple

from axiompy.agents.rag.domain.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    SearchResult,
)


class MockDocumentSource:
    """
    Mock document source for testing.

    Attributes:
        calls: List of recorded method calls
    """

    def __init__(self) -> None:
        """Initialize mock document source."""
        self.calls: List[Tuple[str, Any]] = []
        self._documents: Dict[str, Document] = {}

    def set_document(self, path: str, content: str) -> "MockDocumentSource":
        """
        Set a mock document for a path.

        Args:
            path: Path to associate with document
            content: Document content

        Returns:
            Self for method chaining
        """
        doc = Document(
            id=f"doc_{len(self._documents)}",
            content=content,
            metadata=DocumentMetadata(source=path),
        )
        self._documents[path] = doc
        return self

    def load_documents(self, paths: List[str]) -> List[Document]:
        """Load mock documents."""
        self.calls.append(("load_documents", paths))
        return [self._documents.get(p, self._create_default(p)) for p in paths]

    def load_document(self, path: str) -> Document:
        """Load a mock document."""
        self.calls.append(("load_document", path))
        return self._documents.get(path, self._create_default(path))

    def _create_default(self, path: str) -> Document:
        """Create default document if not set."""
        return Document(
            id=f"doc_{path}",
            content=f"Default content for {path}",
            metadata=DocumentMetadata(source=path),
        )

    def reset(self) -> None:
        """Reset recorded calls."""
        self.calls.clear()


class MockEmbedder:
    """
    Mock embedder for testing.

    Returns deterministic embeddings based on text hash.

    Attributes:
        calls: List of recorded method calls
    """

    DEFAULT_DIMENSION = 384

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        """
        Initialize mock embedder.

        Args:
            dimension: Embedding dimension to use
        """
        self._dimension = dimension
        self.calls: List[Tuple[str, Any]] = []

    def set_dimension(self, dimension: int) -> "MockEmbedder":
        """
        Set embedding dimension.

        Args:
            dimension: New dimension

        Returns:
            Self for method chaining
        """
        self._dimension = dimension
        return self

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate mock embedding for text.

        Uses hash-based deterministic values for reproducibility.
        """
        self.calls.append(("embed_text", text))
        return self._generate_embedding(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        self.calls.append(("embed_texts", texts))
        return [self._generate_embedding(t) for t in texts]

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding based on text content.

        Uses word-level hashing to create pseudo-semantic embeddings.
        Similar texts with shared words will have similar embeddings.
        """
        import hashlib

        words = text.lower().split()
        embedding = [0.0] * self._dimension

        for word in words:
            # Hash each word deterministically
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            for j in range(self._dimension):
                # Spread word influence across embedding
                embedding[j] += ((word_hash >> (j % 32)) & 1) * 0.1

        # Normalize the embedding
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        else:
            embedding = [1.0 / (self._dimension**0.5)] * self._dimension

        return embedding

    def reset(self) -> None:
        """Reset recorded calls."""
        self.calls.clear()


class MockVectorStore:
    """
    Mock vector store for testing.

    Stores chunks in memory with simple similarity matching.

    Attributes:
        calls: List of recorded method calls
    """

    def __init__(self) -> None:
        """Initialize mock vector store."""
        self.calls: List[Tuple[str, Any]] = []
        self._chunks: Dict[str, DocumentChunk] = {}
        self._responses: List[SearchResult] = []

    def set_search_results(self, results: List[SearchResult]) -> "MockVectorStore":
        """
        Set predefined search results.

        Args:
            results: Results to return from search

        Returns:
            Self for method chaining
        """
        self._responses = results
        return self

    @property
    def chunk_count(self) -> int:
        """Get total chunk count."""
        return len(self._chunks)

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add chunks to mock store."""
        self.calls.append(("add_chunks", chunks))
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Return mock search results.

        If results were set via set_search_results, returns those.
        Otherwise returns stored chunks with mock scores.
        """
        self.calls.append(("search", (query_embedding, top_k, min_score, filters)))

        if self._responses:
            return self._responses[:top_k]

        # Return stored chunks with mock scores
        results = []
        for i, chunk in enumerate(list(self._chunks.values())[:top_k]):
            score = 1.0 - (i * 0.1)  # Decreasing scores
            if score >= min_score:
                results.append(SearchResult(chunk=chunk, score=score))
        return results

    def delete_document(self, document_id: str) -> int:
        """Delete chunks for a document."""
        self.calls.append(("delete_document", document_id))
        to_delete = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
        for cid in to_delete:
            del self._chunks[cid]
        return len(to_delete)

    def reset(self) -> None:
        """Reset recorded calls and stored chunks."""
        self.calls.clear()
        self._chunks.clear()
        self._responses.clear()


class MockLLMProvider:
    """
    Mock LLM provider for testing.

    Returns configurable responses.

    Attributes:
        calls: List of recorded method calls
    """

    DEFAULT_RESPONSE = "This is a mock response."

    def __init__(self, response: str = DEFAULT_RESPONSE) -> None:
        """
        Initialize mock LLM provider.

        Args:
            response: Default response to return
        """
        self._response = response
        self._model_name = "mock-model"
        self.calls: List[Tuple[str, Any]] = []

    def set_response(self, response: str) -> "MockLLMProvider":
        """
        Set the response to return.

        Args:
            response: Response text

        Returns:
            Self for method chaining
        """
        self._response = response
        return self

    def set_model_name(self, name: str) -> "MockLLMProvider":
        """
        Set the model name.

        Args:
            name: Model name

        Returns:
            Self for method chaining
        """
        self._model_name = name
        return self

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def generate(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate mock response."""
        self.calls.append(("generate", (prompt, context, temperature, max_tokens)))
        return self._response

    def reset(self) -> None:
        """Reset recorded calls."""
        self.calls.clear()
