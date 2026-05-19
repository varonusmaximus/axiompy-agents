"""agents.io domain models.

Core data structures for retrieval workflows following axiompy dataclass patterns.

Models:
- Document: A document to be indexed
- DocumentMetadata: Metadata about a document
- DocumentChunk: A chunk of a document with embedding
- Query: A user query for RAG
- SearchResult: A matched chunk with similarity score
- RetrievalResponse: Final response with context and answer
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from axiompy.agents.io.defaults import DEFAULT_MIN_SCORE, DEFAULT_TOP_K
from axiompy.validators import ensure_in_range, ensure_not_empty, ensure_positive


@dataclass
class DocumentMetadata:
    """
    Metadata about a document.

    Attributes:
        source: File path, URL, or identifier
        title: Document title
        author: Document author
        created_at: Creation timestamp
        content_type: MIME type (text/plain, text/markdown, etc.)
        extra: Arbitrary additional metadata
    """

    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    content_type: str = "text/plain"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    A document to be indexed and searched.

    Attributes:
        id: Unique document identifier
        content: Full text content of the document
        metadata: Document metadata
    """

    id: str
    content: str
    metadata: DocumentMetadata

    @property
    def source(self) -> str:
        """Get document source from metadata."""
        return self.metadata.source

    @property
    def char_count(self) -> int:
        """Get character count of content."""
        return len(self.content)


@dataclass
class DocumentChunk:
    """
    A chunk of a document with optional embedding.

    Attributes:
        id: Unique chunk identifier
        document_id: Parent document ID
        content: Text content of the chunk
        embedding: Vector embedding (None until embedded)
        chunk_index: Index of this chunk in the document
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        metadata: Additional chunk metadata
    """

    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Get character count of chunk content."""
        return len(self.content)

    @property
    def has_embedding(self) -> bool:
        """Check if chunk has been embedded."""
        return self.embedding is not None


@dataclass
class Query:
    """
    A user query for RAG retrieval.

    Attributes:
        text: Query text
        embedding: Query embedding (None until embedded)
        top_k: Number of results to retrieve
        min_score: Minimum similarity score threshold
        filters: Metadata filters for search
    """

    text: str
    embedding: Optional[List[float]] = None
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate query parameters."""
        ensure_not_empty(self.text.strip(), "Query text cannot be empty")
        ensure_positive(self.top_k, "top_k must be at least 1")
        ensure_in_range(self.min_score, 0.0, 1.0, "min_score must be between 0.0 and 1.0")


@dataclass
class SearchResult:
    """
    A matched chunk with similarity score.

    Attributes:
        chunk: The matched document chunk
        score: Similarity score (0.0 to 1.0)
        document_metadata: Optional metadata from parent document
    """

    chunk: DocumentChunk
    score: float
    document_metadata: Optional[DocumentMetadata] = None

    def __post_init__(self) -> None:
        """Validate search result."""
        ensure_in_range(self.score, 0.0, 1.0, "Score must be between 0.0 and 1.0")

    @property
    def content(self) -> str:
        """Get chunk content for convenience."""
        return self.chunk.content


@dataclass
class RetrievalResponse:
    """
    Final RAG response with context and answer.

    Attributes:
        query: Original query
        answer: Generated answer
        sources: List of search results used as context
        context_text: Concatenated context text sent to LLM
        model: Model name used for generation
        tokens_used: Token count (if available)
        latency_ms: Response latency in milliseconds
    """

    query: Query
    answer: str
    sources: List[SearchResult]
    context_text: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None

    @property
    def source_count(self) -> int:
        """Get number of sources used."""
        return len(self.sources)

    @property
    def has_sources(self) -> bool:
        """Check if response has any sources."""
        return len(self.sources) > 0

    @property
    def top_score(self) -> float:
        """Get highest similarity score from sources."""
        if not self.sources:
            return 0.0
        return max(s.score for s in self.sources)
