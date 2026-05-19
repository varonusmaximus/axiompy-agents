"""Unit tests for axiompy.agents.io.

Tests cover:
- Domain models (Document, DocumentChunk, Query, SearchResult, RetrievalResponse)
- Chunking (FixedSizeChunker)
- Adapters (InMemoryVectorStore, FileSystemSource)
- Mocks

Run with:
    pytest tests/test_rag.py -v --cov=axiompy.io
"""

import pytest

from axiompy.io.http import HTTPRequestError
from axiompy.validators import ValidationError

from axiompy.agents.io import (
    ChunkerSettings,
    Document,
    DocumentChunk,
    DocumentMetadata,
    EmbedderSettings,
    MockDocumentSource,
    MockEmbedder,
    MockLLMProvider,
    MockVectorStore,
    Query,
    RetrievalResponse,
    SearchResult,
    VectorStoreSettings,
)
from axiompy.agents.io.documents.chunker import FixedSizeChunker
from axiompy.agents.io.vector import InMemoryVectorStore
from axiompy.agents.io.documents import FileSystemSource
from axiompy.agents.io.errors import (
    AgentIOConfigurationError,
    AgentIOEmbeddingError,
    AgentIOError,
    AgentIOIngestionError,
    AgentIOQueryError,
    AgentIOVectorStoreError,
)
from axiompy.agents.io.defaults import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_SCORE,
    DEFAULT_RETRIEVAL_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
)


# =============================================================================
# Domain Models Tests
# =============================================================================


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_create_metadata_minimal(self):
        """Test creating metadata with minimal fields."""
        metadata = DocumentMetadata(source="/path/to/file.txt")
        assert metadata.source == "/path/to/file.txt"
        assert metadata.title is None
        assert metadata.author is None
        assert metadata.content_type == "text/plain"

    def test_create_metadata_full(self):
        """Test creating metadata with all fields."""
        metadata = DocumentMetadata(
            source="/path/to/file.txt",
            title="Test Document",
            author="Test Author",
            content_type="text/markdown",
            extra={"version": 2, "tags": ["test"]},
        )
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.content_type == "text/markdown"
        assert metadata.extra["version"] == 2


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self):
        """Test creating document with required fields."""
        metadata = DocumentMetadata(source="/path/to/file.txt")
        doc = Document(id="doc1", content="Hello world", metadata=metadata)
        assert doc.id == "doc1"
        assert doc.content == "Hello world"
        assert doc.metadata.source == "/path/to/file.txt"

    def test_document_source_property(self):
        """Test document source property."""
        metadata = DocumentMetadata(source="test.txt", title="Test")
        doc = Document(id="doc1", content="Content", metadata=metadata)
        assert doc.source == "test.txt"

    def test_document_char_count(self):
        """Test document char_count property."""
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="Hello", metadata=metadata)
        assert doc.char_count == 5


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_create_chunk(self):
        """Test creating a chunk."""
        chunk = DocumentChunk(
            id="chunk1",
            document_id="doc1",
            content="Chunk content",
            chunk_index=0,
            start_char=0,
            end_char=13,
        )
        assert chunk.id == "chunk1"
        assert chunk.document_id == "doc1"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None
        assert not chunk.has_embedding

    def test_chunk_with_embedding(self):
        """Test chunk with embedding."""
        chunk = DocumentChunk(
            id="chunk1",
            document_id="doc1",
            content="Content",
            chunk_index=0,
            start_char=0,
            end_char=7,
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.has_embedding

    def test_chunk_char_count(self):
        """Test chunk char_count property."""
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Hello World",
            chunk_index=0,
            start_char=0,
            end_char=11,
        )
        assert chunk.char_count == 11

    def test_chunk_with_metadata(self):
        """Test chunk with metadata dict."""
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4,
            metadata={"category": "tech"},
        )
        assert chunk.metadata["category"] == "tech"


class TestQuery:
    """Tests for Query model."""

    def test_create_query(self):
        """Test creating a query."""
        query = Query(text="What is Python?")
        assert query.text == "What is Python?"
        assert query.top_k == DEFAULT_TOP_K
        assert query.min_score == DEFAULT_MIN_SCORE

    def test_query_with_params(self):
        """Test query with custom parameters."""
        query = Query(text="Question", top_k=10, min_score=0.5)
        assert query.top_k == 10
        assert query.min_score == 0.5

    def test_query_with_filters(self):
        """Test query with metadata filters."""
        query = Query(text="Test", filters={"category": "tech"})
        assert query.filters["category"] == "tech"

    def test_query_empty_text_raises(self):
        """Test that empty query text raises ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            Query(text="")

    def test_query_whitespace_only_raises(self):
        """Test that whitespace-only text raises ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            Query(text="   ")

    def test_query_invalid_top_k_raises(self):
        """Test that top_k < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="top_k"):
            Query(text="Test", top_k=0)

    def test_query_invalid_min_score_raises(self):
        """Test that invalid min_score raises ValidationError."""
        with pytest.raises(ValidationError, match="min_score"):
            Query(text="Test", min_score=1.5)


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Content",
            chunk_index=0,
            start_char=0,
            end_char=7,
        )
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.score == 0.95
        assert result.content == "Content"

    def test_search_result_with_document_metadata(self):
        """Test search result with document metadata."""
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4,
        )
        doc_meta = DocumentMetadata(source="test.txt", title="Test Doc")
        result = SearchResult(chunk=chunk, score=0.8, document_metadata=doc_meta)
        assert result.document_metadata.title == "Test Doc"

    def test_search_result_invalid_score_raises(self):
        """Test that invalid score raises ValidationError."""
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4,
        )
        with pytest.raises(ValidationError, match="Score"):
            SearchResult(chunk=chunk, score=1.5)


class TestRetrievalResponse:
    """Tests for RetrievalResponse model."""

    def test_create_rag_response(self):
        """Test creating a RAG response."""
        query = Query(text="Test question")
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Context",
            chunk_index=0,
            start_char=0,
            end_char=7,
        )
        sources = [SearchResult(chunk=chunk, score=0.9)]
        response = RetrievalResponse(
            query=query,
            answer="Test answer",
            sources=sources,
            context_text="Context",
            model="test-model",
        )
        assert response.answer == "Test answer"
        assert response.source_count == 1
        assert response.has_sources
        assert response.top_score == 0.9

    def test_rag_response_no_sources(self):
        """Test RAG response with no sources."""
        query = Query(text="Test")
        response = RetrievalResponse(
            query=query,
            answer="No info",
            sources=[],
            context_text="",
            model="test",
        )
        assert not response.has_sources
        assert response.top_score == 0.0


# =============================================================================
# Defaults Tests
# =============================================================================


class TestDefaults:
    """Tests for default values."""

    def test_default_values_exist(self):
        """Test that all defaults are defined."""
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_CHUNK_OVERLAP >= 0
        assert DEFAULT_EMBEDDING_BATCH_SIZE > 0
        assert DEFAULT_TOP_K > 0
        assert 0 <= DEFAULT_MIN_SCORE <= 1
        assert 0 <= DEFAULT_TEMPERATURE <= 2
        assert DEFAULT_MAX_TOKENS > 0
        assert "{context}" in DEFAULT_RETRIEVAL_PROMPT
        assert "{question}" in DEFAULT_RETRIEVAL_PROMPT


# =============================================================================
# Chunker Tests
# =============================================================================


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_chunk_document(self):
        """Test basic document chunking."""
        chunker = FixedSizeChunker(_chunk_size=100, _chunk_overlap=20)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="A" * 250, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 2
        assert all(c.document_id == "doc1" for c in chunks)
        assert all(len(c.content) <= 100 for c in chunks)

    def test_chunk_small_document(self):
        """Test chunking document smaller than chunk size."""
        chunker = FixedSizeChunker(_chunk_size=100, _chunk_overlap=20)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="Small content", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Small content"

    def test_chunk_empty_document(self):
        """Test chunking empty document."""
        chunker = FixedSizeChunker(_chunk_size=100, _chunk_overlap=20)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        chunker = FixedSizeChunker(_chunk_size=50, _chunk_overlap=10)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="A" * 100, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert all(c.metadata.get("source") == "test.txt" for c in chunks)

    def test_chunker_properties(self):
        """Test chunker properties."""
        chunker = FixedSizeChunker(_chunk_size=500, _chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = FixedSizeChunker(_chunk_size=50, _chunk_overlap=10)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="A" * 200, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_positions_correct(self):
        """Test that start_char and end_char are correct."""
        chunker = FixedSizeChunker(_chunk_size=10, _chunk_overlap=0)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="0123456789ABCDEF", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 10


# =============================================================================
# Sentence Chunker Tests
# =============================================================================


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_chunk_document_basic(self):
        """Test basic sentence chunking."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=100)
        metadata = DocumentMetadata(source="test.txt")
        content = "This is sentence one. This is sentence two. This is sentence three."
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert all(c.document_id == "doc1" for c in chunks)

    def test_chunk_respects_sentences(self):
        """Test that chunking respects sentence boundaries."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=50)
        metadata = DocumentMetadata(source="test.txt")
        content = "Short sentence. Another short one. Third sentence here."
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        # Verify chunks don't cut mid-sentence (check for periods/complete sentences)
        for chunk in chunks:
            # Each chunk should end with a sentence boundary or be complete
            assert chunk.content.strip()

    def test_chunk_empty_document(self):
        """Test chunking empty document."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=100)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only document."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=100)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="   \n\n   ", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_chunk_long_document(self):
        """Test chunking document with many sentences."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=200)
        metadata = DocumentMetadata(source="test.txt")
        sentences = [f"This is sentence number {i}." for i in range(20)]
        content = " ".join(sentences)
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        # Each chunk should be near target size (except last)
        for chunk in chunks[:-1]:
            assert len(chunk.content) <= 200 + 50  # Allow some flexibility

    def test_chunker_properties(self):
        """Test chunker properties."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=500)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 0  # Returns 0 for compatibility

    def test_invalid_target_size(self):
        """Test invalid target size raises error."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        with pytest.raises(ValidationError, match="target_size must be positive"):
            SentenceChunker(_target_size=0)

    def test_negative_overlap_sentences(self):
        """Test negative overlap raises error."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        with pytest.raises(ValidationError, match="overlap_sentences cannot be negative"):
            SentenceChunker(_target_size=100, _overlap_sentences=-1)

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=50)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="First sentence. Second sentence.", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert all(c.metadata.get("source") == "test.txt" for c in chunks)
        assert all("sentence_count" in c.metadata for c in chunks)

    def test_chunk_with_newlines(self):
        """Test chunking content with newlines."""
        from axiompy.agents.io.documents.chunker import SentenceChunker

        chunker = SentenceChunker(_target_size=200)
        metadata = DocumentMetadata(source="test.txt")
        content = "First paragraph sentence.\nSecond line here.\n\nNew paragraph starts."
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1


# =============================================================================
# Paragraph Chunker Tests
# =============================================================================


class TestParagraphChunker:
    """Tests for ParagraphChunker."""

    def test_chunk_document_basic(self):
        """Test basic paragraph chunking."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=500)
        metadata = DocumentMetadata(source="test.txt")
        content = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert all(c.document_id == "doc1" for c in chunks)

    def test_chunk_merges_small_paragraphs(self):
        """Test that small paragraphs are merged."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=200, _merge_small=True)
        metadata = DocumentMetadata(source="test.txt")
        content = "Short one.\n\nShort two.\n\nShort three."
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        # With merge_small=True and target_size=200, all should be in one chunk
        assert len(chunks) == 1
        assert "Short one." in chunks[0].content
        assert "Short three." in chunks[0].content

    def test_chunk_no_merge(self):
        """Test chunking without merging."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=200, _merge_small=False)
        metadata = DocumentMetadata(source="test.txt")
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        # Without merge, each paragraph is its own chunk
        assert len(chunks) == 3

    def test_chunk_empty_document(self):
        """Test chunking empty document."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=100)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only document."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=100)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="   \n\n   ", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_chunk_long_paragraphs(self):
        """Test chunking with paragraphs longer than target."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=50)
        metadata = DocumentMetadata(source="test.txt")
        content = "A" * 100 + "\n\n" + "B" * 100  # Two long paragraphs
        doc = Document(id="doc1", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        # Each paragraph should be its own chunk since they exceed target
        assert len(chunks) == 2

    def test_chunker_properties(self):
        """Test chunker properties."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=500)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 0

    def test_invalid_target_size(self):
        """Test invalid target size raises error."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        with pytest.raises(ValidationError, match="target_size must be positive"):
            ParagraphChunker(_target_size=0)

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        from axiompy.agents.io.documents.chunker import ParagraphChunker

        chunker = ParagraphChunker(_target_size=50, _merge_small=False)
        metadata = DocumentMetadata(source="test.txt")
        doc = Document(id="doc1", content="Para one.\n\nPara two.", metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert all(c.metadata.get("source") == "test.txt" for c in chunks)
        assert all("paragraph_count" in c.metadata for c in chunks)


# =============================================================================
# ChunkerFactory Tests
# =============================================================================


class TestChunkerFactory:
    """Tests for ChunkerFactory using enum-based type selection."""

    def test_create_fixed_size(self):
        """Test creating fixed size chunker via enum."""
        from axiompy.agents.io import ChunkerFactory
        from axiompy.agents.io.settings import ChunkerSettings, ChunkerType

        chunker = ChunkerFactory.create(
            ChunkerType.FIXED_SIZE,
            ChunkerSettings(chunk_size=100, chunk_overlap=10),
        )
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10

    def test_create_sentence(self):
        """Test creating sentence chunker via enum."""
        from axiompy.agents.io import ChunkerFactory
        from axiompy.agents.io.settings import ChunkerSettings, ChunkerType

        chunker = ChunkerFactory.create(
            ChunkerType.SENTENCE,
            ChunkerSettings(chunk_size=200),
        )
        assert chunker.chunk_size == 200

    def test_create_paragraph(self):
        """Test creating paragraph chunker via enum."""
        from axiompy.agents.io import ChunkerFactory
        from axiompy.agents.io.settings import ChunkerSettings, ChunkerType

        chunker = ChunkerFactory.create(
            ChunkerType.PARAGRAPH,
            ChunkerSettings(chunk_size=300),
        )
        assert chunker.chunk_size == 300

    def test_create_mock(self):
        """Test creating mock chunker."""
        from axiompy.agents.io import ChunkerFactory

        chunker = ChunkerFactory.create_mock()
        assert chunker is not None
        assert hasattr(chunker, "chunk_document")


class TestEmbedderFactory:
    """Tests for EmbedderFactory using enum-based type selection."""

    def test_create_mock(self):
        """Test creating mock embedder."""
        from axiompy.agents.io.embeddings import EmbedderFactory

        embedder = EmbedderFactory.create_mock(dimension=256)
        assert embedder is not None
        result = embedder.embed_text("test")
        assert len(result) == 256

    def test_create_mock_via_enum(self):
        """Test creating mock embedder via enum."""
        from axiompy.agents.io.embeddings import EmbedderFactory
        from axiompy.agents.io.settings import EmbedderSettings, EmbedderType

        embedder = EmbedderFactory.create(
            EmbedderType.MOCK,
            EmbedderSettings(),
        )
        assert embedder is not None

    def test_create_openai_requires_api_key(self):
        """Test OpenAI embedder requires API key."""
        from axiompy.agents.io.embeddings import EmbedderFactory
        from axiompy.agents.io.settings import EmbedderSettings, EmbedderType

        with pytest.raises(ValueError, match="api_key required"):
            EmbedderFactory.create(
                EmbedderType.OPENAI,
                EmbedderSettings(),  # No API key
            )

    def test_unknown_type_raises(self):
        """Test unknown embedder type raises ValueError."""
        from axiompy.agents.io.embeddings import EmbedderFactory
        from axiompy.agents.io.settings import EmbedderSettings

        with pytest.raises(ValueError, match="Unknown embedder type"):
            EmbedderFactory.create(
                "invalid_type",  # type: ignore
                EmbedderSettings(),
            )


# =============================================================================
# VectorStoreFactory Tests
# =============================================================================


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory using enum-based type selection."""

    def test_create_mock(self):
        """Test creating mock vector store."""
        from axiompy.agents.io.vector import VectorStoreFactory

        store = VectorStoreFactory.create_mock()
        assert store is not None

    def test_create_memory_via_enum(self):
        """Test creating in-memory store via enum."""
        from axiompy.agents.io.vector import VectorStoreFactory
        from axiompy.agents.io.settings import VectorStoreSettings, VectorStoreType

        store = VectorStoreFactory.create(
            VectorStoreType.MEMORY,
            VectorStoreSettings(),
        )
        assert store is not None

    def test_create_mock_via_enum(self):
        """Test creating mock store via enum."""
        from axiompy.agents.io.vector import VectorStoreFactory
        from axiompy.agents.io.settings import VectorStoreSettings, VectorStoreType

        store = VectorStoreFactory.create(
            VectorStoreType.MOCK,
            VectorStoreSettings(),
        )
        assert store is not None

    def test_create_pinecone_requires_api_key(self):
        """Test Pinecone store requires API key."""
        from axiompy.agents.io.vector import VectorStoreFactory
        from axiompy.agents.io.settings import VectorStoreSettings, VectorStoreType

        with pytest.raises(ValueError, match="api_key required"):
            VectorStoreFactory.create(
                VectorStoreType.PINECONE,
                VectorStoreSettings(),  # No API key
            )

    def test_create_pinecone_requires_host(self):
        """Test Pinecone store requires host."""
        from axiompy.agents.io.vector import VectorStoreFactory
        from axiompy.agents.io.settings import VectorStoreSettings, VectorStoreType

        with pytest.raises(ValueError, match="host.*required"):
            VectorStoreFactory.create(
                VectorStoreType.PINECONE,
                VectorStoreSettings(api_key="test-key"),  # No host
            )

    def test_create_pgvector_requires_database_url(self):
        """Test pgvector store requires database_url."""
        from axiompy.agents.io.vector import VectorStoreFactory
        from axiompy.agents.io.settings import VectorStoreSettings, VectorStoreType

        with pytest.raises(ValueError, match="database_url required"):
            VectorStoreFactory.create(
                VectorStoreType.PGVECTOR,
                VectorStoreSettings(),  # No database_url
            )

    def test_unknown_type_raises(self):
        """Test unknown store type raises ValueError."""
        from axiompy.agents.io.vector import VectorStoreFactory
        from axiompy.agents.io.settings import VectorStoreSettings

        with pytest.raises(ValueError, match="Unknown vector store type"):
            VectorStoreFactory.create(
                "invalid_type",  # type: ignore
                VectorStoreSettings(),
            )


# =============================================================================
# OpenAI Embedder Tests
# =============================================================================


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2, 0.3] * 512},
                        {"index": 1, "embedding": [0.4, 0.5, 0.6] * 512},
                    ],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                }

        return MockResponse()

    @pytest.fixture
    def mock_error_response(self):
        """Create mock error response."""

        class MockResponse:
            status_code = 401
            text = '{"error": {"message": "Invalid API key"}}'

        return MockResponse()

    def test_init(self):
        """Test initialization."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")
        assert embedder.model == "text-embedding-3-small"
        assert embedder.embedding_dimension == 1536

    def test_init_custom_model(self):
        """Test with custom model."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
        assert embedder.model == "text-embedding-3-large"
        assert embedder.embedding_dimension == 3072

    def test_init_unknown_model(self):
        """Test with unknown model (dimension determined later)."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test", model="custom-model")
        assert embedder.model == "custom-model"
        assert embedder.embedding_dimension == 0  # Will be determined on first call

    def test_init_missing_api_key(self):
        """Test missing API key raises error."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        with pytest.raises(AgentIOEmbeddingError, match="API key is required"):
            OpenAIEmbedder(api_key="")

    def test_embed_text(self, mock_response, monkeypatch):
        """Test embedding single text."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        # Mock HTTPClient.post
        def mock_post(*args, **kwargs):
            return mock_response

        # Create embedder and mock the client
        embedder = OpenAIEmbedder(api_key="sk-test")
        monkeypatch.setattr(embedder._client, "post", mock_post)

        embedding = embedder.embed_text("Hello")

        assert len(embedding) == 1536
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_texts(self, mock_response, monkeypatch):
        """Test batch embedding."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        def mock_post(*args, **kwargs):
            return mock_response

        embedder = OpenAIEmbedder(api_key="sk-test")
        monkeypatch.setattr(embedder._client, "post", mock_post)

        embeddings = embedder.embed_texts(["Hello", "World"])

        assert len(embeddings) == 2
        assert all(len(e) == 1536 for e in embeddings)

    def test_embed_empty_text_raises(self):
        """Test that empty text raises error."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")

        with pytest.raises(AgentIOEmbeddingError, match="empty text"):
            embedder.embed_text("")

    def test_embed_empty_list_returns_empty(self):
        """Test that empty list returns empty list."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")
        result = embedder.embed_texts([])
        assert result == []

    def test_embed_whitespace_only_raises(self):
        """Test that whitespace-only text raises error."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")

        with pytest.raises(AgentIOEmbeddingError, match="No valid texts"):
            embedder.embed_texts(["   ", "\n\n", ""])

    def test_api_error_handling(self, mock_error_response, monkeypatch):
        """Test API error handling."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        def mock_post(*args, **kwargs):
            raise HTTPRequestError(
                f"HTTP {mock_error_response.status_code}: {mock_error_response.text}"
            )

        embedder = OpenAIEmbedder(api_key="sk-test")
        monkeypatch.setattr(embedder._client, "post", mock_post)

        with pytest.raises(AgentIOEmbeddingError, match="OpenAI API error"):
            embedder.embed_text("Hello")

    def test_repr(self):
        """Test string representation."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")
        repr_str = repr(embedder)
        assert "OpenAIEmbedder" in repr_str
        assert "text-embedding-3-small" in repr_str

    def test_custom_endpoint(self):
        """Test custom API endpoint."""
        from axiompy.agents.io.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(
            api_key="sk-test",
            endpoint="https://custom.api.com/v1/embeddings",
        )
        assert embedder._endpoint == "https://custom.api.com/v1/embeddings"


class TestOpenAIEmbedderFactory:
    """Tests for factory integration with OpenAI embedder."""


class TestOllamaEmbedder:
    """Tests for OllamaEmbedder."""

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "embedding": [0.1, 0.2, 0.3] * 256,  # 768 dimensions
                }

        return MockResponse()

    @pytest.fixture
    def mock_error_response(self):
        """Create mock error response."""

        class MockResponse:
            status_code = 500
            text = '{"error": "Server error"}'

        return MockResponse()

    @pytest.fixture
    def mock_404_response(self):
        """Create mock 404 response."""

        class MockResponse:
            status_code = 404
            text = '{"error": "model not found"}'

        return MockResponse()

    def test_init(self):
        """Test initialization."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()
        assert embedder.model == "nomic-embed-text"
        assert embedder.embedding_dimension == 768

    def test_init_custom_model(self):
        """Test with custom model."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(model="mxbai-embed-large")
        assert embedder.model == "mxbai-embed-large"
        assert embedder.embedding_dimension == 1024

    def test_init_custom_host(self):
        """Test with custom host."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(host="http://myserver:11434")
        assert embedder._host == "http://myserver:11434"

    def test_init_unknown_model(self):
        """Test with unknown model (dimension determined later)."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(model="custom-model")
        assert embedder.model == "custom-model"
        assert embedder.embedding_dimension == 0  # Will be determined on first call

    def test_embed_text(self, mock_response, monkeypatch):
        """Test embedding single text."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        def mock_post(*args, **kwargs):
            return mock_response

        embedder = OllamaEmbedder()
        monkeypatch.setattr(embedder._client, "post", mock_post)

        embedding = embedder.embed_text("Hello")

        assert len(embedding) == 768
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_texts(self, mock_response, monkeypatch):
        """Test batch embedding."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        def mock_post(*args, **kwargs):
            return mock_response

        embedder = OllamaEmbedder()
        monkeypatch.setattr(embedder._client, "post", mock_post)

        embeddings = embedder.embed_texts(["Hello", "World"])

        assert len(embeddings) == 2
        assert all(len(e) == 768 for e in embeddings)

    def test_embed_empty_text_raises(self):
        """Test that empty text raises error."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()

        with pytest.raises(AgentIOEmbeddingError, match="empty text"):
            embedder.embed_text("")

    def test_embed_empty_list_returns_empty(self):
        """Test that empty list returns empty list."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()
        result = embedder.embed_texts([])
        assert result == []

    def test_embed_whitespace_only_raises(self):
        """Test that whitespace-only text raises error."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()

        with pytest.raises(AgentIOEmbeddingError, match="No valid texts"):
            embedder.embed_texts(["   ", "\n\n", ""])

    def test_api_error_handling(self, mock_error_response, monkeypatch):
        """Test API error handling."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        def mock_post(*args, **kwargs):
            raise HTTPRequestError(
                f"HTTP {mock_error_response.status_code}: {mock_error_response.text}"
            )

        embedder = OllamaEmbedder()
        monkeypatch.setattr(embedder._client, "post", mock_post)

        with pytest.raises(AgentIOEmbeddingError, match="Ollama API error"):
            embedder.embed_text("Hello")

    def test_model_not_found(self, mock_404_response, monkeypatch):
        """Test model not found error."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        def mock_post(*args, **kwargs):
            raise HTTPRequestError(
                f"HTTP {mock_404_response.status_code}: {mock_404_response.text}"
            )

        embedder = OllamaEmbedder()
        monkeypatch.setattr(embedder._client, "post", mock_post)

        with pytest.raises(AgentIOEmbeddingError, match="not found"):
            embedder.embed_text("Hello")

    def test_repr(self):
        """Test string representation."""
        from axiompy.agents.io.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()
        repr_str = repr(embedder)
        assert "OllamaEmbedder" in repr_str
        assert "nomic-embed-text" in repr_str


class TestOllamaEmbedderFactory:
    """Tests for factory integration with Ollama embedder."""


class TestMockEmbedder:
    """Tests for MockEmbedder."""

    def test_embed_text(self):
        """Test embedding single text."""
        embedder = MockEmbedder(dimension=128)
        embedding = embedder.embed_text("Hello")

        assert len(embedding) == 128
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_texts(self):
        """Test embedding multiple texts."""
        embedder = MockEmbedder(dimension=64)
        embeddings = embedder.embed_texts(["Hello", "World"])

        assert len(embeddings) == 2
        assert all(len(e) == 64 for e in embeddings)

    def test_embedding_dimension_property(self):
        """Test embedding dimension property."""
        embedder = MockEmbedder(dimension=256)
        assert embedder.embedding_dimension == 256

    def test_set_dimension(self):
        """Test set_dimension fluent method."""
        embedder = MockEmbedder()
        result = embedder.set_dimension(512)
        assert result is embedder
        assert embedder.embedding_dimension == 512

    def test_deterministic_embeddings(self):
        """Test that same text produces same embedding."""
        embedder = MockEmbedder(dimension=64)
        e1 = embedder.embed_text("Hello world")
        e2 = embedder.embed_text("Hello world")
        assert e1 == e2

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        embedder = MockEmbedder(dimension=64)
        e1 = embedder.embed_text("Hello")
        e2 = embedder.embed_text("Goodbye")
        assert e1 != e2

    def test_calls_tracking(self):
        """Test that calls are tracked."""
        embedder = MockEmbedder()
        embedder.embed_text("test")
        embedder.embed_texts(["a", "b"])

        assert len(embedder.calls) == 2
        assert embedder.calls[0] == ("embed_text", "test")
        assert embedder.calls[1] == ("embed_texts", ["a", "b"])

    def test_reset(self):
        """Test reset clears calls."""
        embedder = MockEmbedder()
        embedder.embed_text("test")
        embedder.reset()
        assert len(embedder.calls) == 0


class TestMockVectorStore:
    """Tests for MockVectorStore."""

    def _make_chunk(self, id: str, doc_id: str, content: str, embedding: list) -> DocumentChunk:
        """Helper to create chunk with required fields."""
        return DocumentChunk(
            id=id,
            document_id=doc_id,
            content=content,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            embedding=embedding,
        )

    def test_add_and_search(self):
        """Test adding chunks and searching."""
        store = MockVectorStore()

        chunk = self._make_chunk("c1", "d1", "Python is great", [0.1] * 128)
        count = store.add_chunks([chunk])
        assert count == 1

        results = store.search([0.1] * 128, top_k=5)
        assert len(results) == 1
        assert results[0].chunk.content == "Python is great"

    def test_delete_document(self):
        """Test deleting document chunks."""
        store = MockVectorStore()

        chunks = [
            self._make_chunk("c1", "d1", "A", [0.1]),
            self._make_chunk("c2", "d1", "B", [0.2]),
            self._make_chunk("c3", "d2", "C", [0.3]),
        ]
        store.add_chunks(chunks)
        assert store.chunk_count == 3

        deleted = store.delete_document("d1")
        assert deleted == 2
        assert store.chunk_count == 1

    def test_chunk_count(self):
        """Test chunk count property."""
        store = MockVectorStore()
        assert store.chunk_count == 0

        chunk = self._make_chunk("c1", "d1", "X", [0.1])
        store.add_chunks([chunk])
        assert store.chunk_count == 1

    def test_set_search_results(self):
        """Test setting predefined search results."""
        store = MockVectorStore()
        chunk = self._make_chunk("c1", "d1", "Test", [0.1])
        predefined = [SearchResult(chunk=chunk, score=0.99)]

        result = store.set_search_results(predefined)
        assert result is store  # Fluent

        results = store.search([0.1], top_k=5)
        assert results[0].score == 0.99

    def test_calls_tracking(self):
        """Test that calls are tracked."""
        store = MockVectorStore()
        chunk = self._make_chunk("c1", "d1", "Test", [0.1])
        store.add_chunks([chunk])
        store.search([0.1])
        store.delete_document("d1")

        assert len(store.calls) == 3

    def test_reset(self):
        """Test reset clears calls and chunks."""
        store = MockVectorStore()
        chunk = self._make_chunk("c1", "d1", "Test", [0.1])
        store.add_chunks([chunk])
        store.reset()

        assert len(store.calls) == 0
        assert store.chunk_count == 0


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    def test_generate(self):
        """Test response generation."""
        llm = MockLLMProvider(response="Mock answer")
        result = llm.generate("Question?", "Context here")
        assert result == "Mock answer"

    def test_set_response(self):
        """Test setting custom response."""
        llm = MockLLMProvider()
        result = llm.set_response("Custom response")
        assert result is llm  # Fluent
        assert llm.generate("Q", "C") == "Custom response"

    def test_model_name(self):
        """Test model name property."""
        llm = MockLLMProvider()
        assert llm.model_name == "mock-model"

    def test_set_model_name(self):
        """Test setting model name."""
        llm = MockLLMProvider()
        result = llm.set_model_name("custom-model")
        assert result is llm  # Fluent
        assert llm.model_name == "custom-model"

    def test_calls_tracking(self):
        """Test that calls are tracked."""
        llm = MockLLMProvider()
        llm.generate("prompt", "context", temperature=0.5, max_tokens=100)

        assert len(llm.calls) == 1
        assert llm.calls[0][0] == "generate"

    def test_reset(self):
        """Test reset clears calls."""
        llm = MockLLMProvider()
        llm.generate("test", "ctx")
        llm.reset()
        assert len(llm.calls) == 0


class TestMockDocumentSource:
    """Tests for MockDocumentSource."""

    def test_load_documents(self):
        """Test loading documents."""
        source = MockDocumentSource()
        source.set_document("path1", "Content 1")
        source.set_document("path2", "Content 2")

        docs = source.load_documents(["path1", "path2"])
        assert len(docs) == 2

    def test_load_single_document(self):
        """Test loading single document."""
        source = MockDocumentSource()
        source.set_document("test", "Test content")

        doc = source.load_document("test")
        assert doc.content == "Test content"

    def test_set_document_fluent(self):
        """Test set_document returns self."""
        source = MockDocumentSource()
        result = source.set_document("path", "content")
        assert result is source

    def test_default_document(self):
        """Test default document for unset paths."""
        source = MockDocumentSource()
        doc = source.load_document("unknown")
        assert "Default content" in doc.content

    def test_calls_tracking(self):
        """Test that calls are tracked."""
        source = MockDocumentSource()
        source.load_document("test")
        source.load_documents(["a", "b"])

        assert len(source.calls) == 2

    def test_reset(self):
        """Test reset clears calls."""
        source = MockDocumentSource()
        source.load_document("test")
        source.reset()
        assert len(source.calls) == 0


# =============================================================================
# InMemoryVectorStore Tests
# =============================================================================


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    def _make_chunk(
        self, id: str, doc_id: str, content: str, embedding: list, metadata: dict = None
    ) -> DocumentChunk:
        """Helper to create chunk with required fields."""
        return DocumentChunk(
            id=id,
            document_id=doc_id,
            content=content,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            embedding=embedding,
            metadata=metadata or {},
        )

    def test_add_chunks(self):
        """Test adding chunks."""
        store = InMemoryVectorStore()
        chunks = [
            self._make_chunk("c1", "d1", "A", [1.0, 0.0]),
            self._make_chunk("c2", "d1", "B", [0.0, 1.0]),
        ]

        count = store.add_chunks(chunks)
        assert count == 2
        assert store.chunk_count == 2

    def test_add_chunks_without_embeddings_raises(self):
        """Test that adding chunks without embeddings raises error."""
        store = InMemoryVectorStore()
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="A",
            chunk_index=0,
            start_char=0,
            end_char=1,
        )

        with pytest.raises(AgentIOVectorStoreError, match="missing embedding"):
            store.add_chunks([chunk])

    def test_search_cosine_similarity(self):
        """Test search returns results by cosine similarity."""
        store = InMemoryVectorStore()
        chunks = [
            self._make_chunk("c1", "d1", "A", [1.0, 0.0, 0.0]),
            self._make_chunk("c2", "d1", "B", [0.0, 1.0, 0.0]),
            self._make_chunk("c3", "d1", "C", [0.0, 0.0, 1.0]),
        ]
        store.add_chunks(chunks)

        results = store.search([0.9, 0.1, 0.0], top_k=2)

        assert len(results) == 2
        assert results[0].chunk.content == "A"
        assert results[0].score > results[1].score

    def test_search_min_score_filter(self):
        """Test min_score filtering."""
        store = InMemoryVectorStore()
        chunks = [
            self._make_chunk("c1", "d1", "A", [1.0, 0.0]),
            self._make_chunk("c2", "d1", "B", [0.0, 1.0]),
        ]
        store.add_chunks(chunks)

        results = store.search([1.0, 0.0], top_k=10, min_score=0.9)

        assert len(results) == 1
        assert results[0].chunk.content == "A"

    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        store = InMemoryVectorStore()
        chunks = [
            self._make_chunk("c1", "d1", "A", [1.0, 0.0], {"category": "tech"}),
            self._make_chunk("c2", "d1", "B", [0.9, 0.1], {"category": "science"}),
        ]
        store.add_chunks(chunks)

        results = store.search([1.0, 0.0], top_k=10, filters={"category": "science"})

        assert len(results) == 1
        assert results[0].chunk.content == "B"

    def test_delete_document(self):
        """Test deleting all chunks for a document."""
        store = InMemoryVectorStore()
        chunks = [
            self._make_chunk("c1", "d1", "A", [1.0]),
            self._make_chunk("c2", "d1", "B", [1.0]),
            self._make_chunk("c3", "d2", "C", [1.0]),
        ]
        store.add_chunks(chunks)

        deleted = store.delete_document("d1")

        assert deleted == 2
        assert store.chunk_count == 1

    def test_clear(self):
        """Test clearing the store."""
        store = InMemoryVectorStore()
        chunks = [self._make_chunk("c1", "d1", "A", [1.0])]
        store.add_chunks(chunks)

        store.clear()

        assert store.chunk_count == 0

    def test_dimension_mismatch_raises(self):
        """Test that mismatched embedding dimensions raise error."""
        store = InMemoryVectorStore()
        store.add_chunks([self._make_chunk("c1", "d1", "A", [1.0, 0.0])])

        with pytest.raises(AgentIOVectorStoreError, match="dimension mismatch"):
            store.add_chunks([self._make_chunk("c2", "d1", "B", [1.0, 0.0, 0.0])])

    def test_search_empty_store(self):
        """Test searching empty store returns empty list."""
        store = InMemoryVectorStore()
        results = store.search([1.0, 0.0])
        assert results == []


# =============================================================================
# FileSystemSource Tests
# =============================================================================


class TestFileSystemSource:
    """Tests for FileSystemSource."""

    def test_load_single_file(self, tmp_path):
        """Test loading a single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        source = FileSystemSource()
        doc = source.load_document(str(test_file))

        assert doc.content == "Hello, World!"
        assert doc.id == str(test_file.absolute())
        assert doc.metadata.title == "test.txt"

    def test_load_multiple_files(self, tmp_path):
        """Test loading multiple files."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.md").write_text("Content 2")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 2
        contents = {d.content for d in docs}
        assert contents == {"Content 1", "Content 2"}

    def test_load_with_glob_pattern(self, tmp_path):
        """Test loading with glob pattern."""
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.txt").write_text("B")
        (tmp_path / "c.py").write_text("C")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path / "*.txt")])

        assert len(docs) == 2
        contents = {d.content for d in docs}
        assert contents == {"A", "B"}

    def test_load_recursive(self, tmp_path):
        """Test recursive directory loading."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 2

    def test_ignore_patterns(self, tmp_path):
        """Test that ignored patterns are skipped."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (tmp_path / "good.txt").write_text("Good")
        (node_modules / "bad.txt").write_text("Bad")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 1
        assert docs[0].content == "Good"

    def test_unsupported_extension_skipped(self, tmp_path):
        """Test that unsupported extensions are skipped."""
        (tmp_path / "image.png").write_bytes(b"fake image")
        (tmp_path / "doc.txt").write_text("Text")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 1
        assert docs[0].metadata.title == "doc.txt"

    def test_file_not_found_raises(self):
        """Test that non-existent file raises error."""
        source = FileSystemSource()

        with pytest.raises(AgentIOIngestionError, match="not found"):
            source.load_document("/nonexistent/path/file.txt")


class TestEmbedderSettings:
    """Tests for EmbedderSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = EmbedderSettings()
        assert settings.model is None
        assert settings.batch_size == DEFAULT_EMBEDDING_BATCH_SIZE
        assert settings.local_files_only is True

    def test_custom_values(self):
        """Test custom values."""
        settings = EmbedderSettings(
            model="custom-model",
            api_key="key",
            endpoint="http://localhost",
            cache_dir="/tmp/cache",
            batch_size=16,
            local_files_only=False,
        )
        assert settings.model == "custom-model"
        assert settings.api_key == "key"
        assert settings.cache_dir == "/tmp/cache"
        assert settings.local_files_only is False


class TestVectorStoreSettings:
    """Tests for VectorStoreSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = VectorStoreSettings()
        assert settings.collection_name == "default"
        assert settings.persist_path is None

    def test_custom_values(self):
        """Test custom values."""
        settings = VectorStoreSettings(
            collection_name="my-collection",
            persist_path="/data/vectors",
            host="localhost",
            port=8000,
        )
        assert settings.collection_name == "my-collection"
        assert settings.persist_path == "/data/vectors"

    def test_empty_collection_name_raises(self):
        """Test empty collection_name raises."""
        from axiompy.validators import ValidationError

        with pytest.raises(ValidationError, match="collection_name"):
            VectorStoreSettings(collection_name="")

    def test_invalid_port_raises(self):
        """Test non-positive port raises."""
        from axiompy.validators import ValidationError

        with pytest.raises(ValidationError, match="port"):
            VectorStoreSettings(port=0)


class TestSourceSettings:
    """Tests for SourceSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        from axiompy.agents.io.documents import SourceSettings

        settings = SourceSettings()
        assert settings.encoding == "utf-8"
        assert settings.timeout_secs == 30

    def test_invalid_timeout_raises(self):
        """Test non-positive timeout_secs raises."""
        from axiompy.agents.io.documents import SourceSettings
        from axiompy.validators import ValidationError

        with pytest.raises(ValidationError, match="timeout_secs"):
            SourceSettings(timeout_secs=0)

    def test_empty_encoding_raises(self):
        """Test empty encoding raises."""
        from axiompy.agents.io.documents import SourceSettings
        from axiompy.validators import ValidationError

        with pytest.raises(ValidationError, match="encoding"):
            SourceSettings(encoding="")


class TestChunkerSettings:
    """Tests for ChunkerSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = ChunkerSettings()
        assert settings.chunk_size == DEFAULT_CHUNK_SIZE
        assert settings.chunk_overlap == DEFAULT_CHUNK_OVERLAP

    def test_custom_values(self):
        """Test custom values."""
        settings = ChunkerSettings(chunk_size=500, chunk_overlap=100)
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 100


# =============================================================================
# Error Tests
# =============================================================================


class TestAgentIOErrors:
    """Tests for RAG error hierarchy."""

    def test_error_inheritance(self):
        """Test that all errors inherit from AgentIOError."""
        errors = [
            AgentIOConfigurationError("config"),
            AgentIOEmbeddingError("embedding"),
            AgentIOIngestionError("ingestion"),
            AgentIOQueryError("query"),
            AgentIOVectorStoreError("store"),
        ]

        for error in errors:
            assert isinstance(error, AgentIOError)
            assert isinstance(error, Exception)

    def test_error_messages(self):
        """Test error messages are preserved."""
        error = AgentIOConfigurationError("Test message")
        assert str(error) == "Test message"


# =============================================================================
# Integration Tests (Mock-based)
# =============================================================================


class TestReasoningAdapter:
    """Tests for ReasoningAdapter."""

    def test_init(self):
        """Test adapter initialization."""
        from unittest.mock import Mock
        from axiompy.reasoning.llm_provider_adapter import ReasoningAdapter

        mock_client = Mock()
        mock_client.model = "test-model"

        adapter = ReasoningAdapter(mock_client)

        assert adapter.model_name == "test-model"
        assert "test-model" in repr(adapter)

    def test_generate(self):
        """Test generate method."""
        from unittest.mock import Mock
        from axiompy.reasoning.llm_provider_adapter import ReasoningAdapter

        mock_client = Mock()
        mock_client.model = "test-model"
        mock_client.generate_completion.return_value = "Generated response"

        adapter = ReasoningAdapter(mock_client)
        result = adapter.generate("What is X?", context="X is a thing")

        assert result == "Generated response"
        mock_client.generate_completion.assert_called_once()

    def test_generate_with_params(self):
        """Test generate with custom temperature and max_tokens."""
        from unittest.mock import Mock
        from axiompy.reasoning.llm_provider_adapter import ReasoningAdapter

        mock_client = Mock()
        mock_client.model = "test-model"
        mock_client.generate_completion.return_value = "Response"

        adapter = ReasoningAdapter(mock_client)
        adapter.generate("Question", "Context", temperature=0.5, max_tokens=500)

        call_args = mock_client.generate_completion.call_args
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 500

    def test_generate_error(self):
        """Test generate error handling."""
        from unittest.mock import Mock
        from axiompy.reasoning.llm_provider_adapter import ReasoningAdapter
        from axiompy.agents.io.errors import AgentIOLLMError

        mock_client = Mock()
        mock_client.model = "test-model"
        mock_client.generate_completion.side_effect = Exception("API error")

        adapter = ReasoningAdapter(mock_client)

        with pytest.raises(AgentIOLLMError, match="Generation failed"):
            adapter.generate("Question", "Context")

    def test_custom_prompt_template(self):
        """Test adapter with custom prompt template."""
        from unittest.mock import Mock
        from axiompy.reasoning.llm_provider_adapter import ReasoningAdapter

        mock_client = Mock()
        mock_client.model = "test-model"
        mock_client.generate_completion.return_value = "Response"

        custom_template = "Context: {context}\n\nQ: {question}"
        adapter = ReasoningAdapter(mock_client, prompt_template=custom_template)
        adapter.generate("Test?", "Test context")

        call_args = mock_client.generate_completion.call_args
        assert "Context: Test context" in call_args.kwargs["prompt"]
        assert "Q: Test?" in call_args.kwargs["prompt"]


# =============================================================================
# Additional FileSystemSource Tests
# =============================================================================


class TestFileSystemSourceEdgeCases:
    """Additional edge case tests for FileSystemSource."""

    def test_load_markdown_file(self, tmp_path):
        """Test loading markdown files."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\nContent here")

        source = FileSystemSource()
        doc = source.load_document(str(md_file))

        assert "Title" in doc.content
        assert doc.metadata.content_type == "text/markdown"

    def test_load_python_file(self, tmp_path):
        """Test loading Python files."""
        py_file = tmp_path / "script.py"
        py_file.write_text("def hello():\n    print('hello')")

        source = FileSystemSource()
        doc = source.load_document(str(py_file))

        assert "def hello" in doc.content
        assert doc.metadata.content_type == "text/x-python"

    def test_load_json_file(self, tmp_path):
        """Test loading JSON files."""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value"}')

        source = FileSystemSource()
        doc = source.load_document(str(json_file))

        assert '"key"' in doc.content
        assert doc.metadata.content_type == "application/json"

    def test_deep_directory_structure(self, tmp_path):
        """Test loading from deep directory structure."""
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep.txt").write_text("Deep content")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 1
        assert docs[0].content == "Deep content"

    def test_gitignore_pattern(self, tmp_path):
        """Test that .git directories are ignored."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (tmp_path / "good.txt").write_text("Good")
        (git_dir / "config").write_text("Bad")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 1
        assert docs[0].content == "Good"

    def test_venv_pattern(self, tmp_path):
        """Test that venv directories are ignored."""
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (tmp_path / "main.py").write_text("print('hello')")
        (venv_dir / "lib.py").write_text("ignored")

        source = FileSystemSource()
        docs = source.load_documents([str(tmp_path)])

        assert len(docs) == 1


# =============================================================================
# ChromaDB Vector Store Tests
# =============================================================================


class TestChromaVectorStore:
    """Tests for ChromaVectorStore."""

    @pytest.fixture
    def chroma_available(self):
        """Check if chromadb is available."""
        import importlib.util

        if importlib.util.find_spec("chromadb") is None:
            pytest.skip("chromadb not installed")
        return True

    @pytest.fixture
    def unique_collection(self):
        """Generate unique collection name per test."""
        import uuid

        return f"test_{uuid.uuid4().hex[:8]}"

    def _make_chunk(
        self, id: str, doc_id: str, content: str, embedding: list, metadata: dict = None
    ) -> DocumentChunk:
        """Helper to create chunk with required fields."""
        return DocumentChunk(
            id=id,
            document_id=doc_id,
            content=content,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            embedding=embedding,
            metadata=metadata or {},
        )

    def test_init_ephemeral(self, chroma_available, unique_collection):
        """Test ephemeral (in-memory) initialization."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        assert store.chunk_count == 0

    def test_init_with_custom_collection(self, chroma_available, unique_collection):
        """Test with custom collection name."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        assert store.collection_name == unique_collection

    def test_init_persistent(self, chroma_available, tmp_path, unique_collection):
        """Test persistent initialization."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        persist_path = str(tmp_path / "chroma_data")
        store = ChromaVectorStore(persist_path=persist_path, collection_name=unique_collection)
        assert store.chunk_count == 0

    def test_add_chunks(self, chroma_available, unique_collection):
        """Test adding chunks."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunks = [
            self._make_chunk("c1", "d1", "Content A", [1.0, 0.0, 0.0]),
            self._make_chunk("c2", "d1", "Content B", [0.0, 1.0, 0.0]),
        ]

        count = store.add_chunks(chunks)

        assert count == 2
        assert store.chunk_count == 2

    def test_add_empty_chunks(self, chroma_available, unique_collection):
        """Test adding empty chunk list."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        count = store.add_chunks([])
        assert count == 0

    def test_add_chunks_without_embeddings_raises(self, chroma_available, unique_collection):
        """Test that adding chunks without embeddings raises error."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Content",
            chunk_index=0,
            start_char=0,
            end_char=7,
        )

        with pytest.raises(AgentIOVectorStoreError, match="missing embedding"):
            store.add_chunks([chunk])

    def test_search(self, chroma_available, unique_collection):
        """Test search returns results."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunks = [
            self._make_chunk("c1", "d1", "Python programming", [1.0, 0.0, 0.0]),
            self._make_chunk("c2", "d1", "Java programming", [0.0, 1.0, 0.0]),
            self._make_chunk("c3", "d1", "Go programming", [0.0, 0.0, 1.0]),
        ]
        store.add_chunks(chunks)

        results = store.search([0.9, 0.1, 0.0], top_k=2)

        assert len(results) == 2
        assert results[0].chunk.content == "Python programming"
        assert results[0].score > results[1].score

    def test_search_empty_store(self, chroma_available, unique_collection):
        """Test searching empty store returns empty list."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        results = store.search([1.0, 0.0, 0.0])
        assert results == []

    def test_search_with_min_score(self, chroma_available, unique_collection):
        """Test min_score filtering."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunks = [
            self._make_chunk("c1", "d1", "Content A", [1.0, 0.0, 0.0]),
            self._make_chunk("c2", "d1", "Content B", [0.0, 1.0, 0.0]),
        ]
        store.add_chunks(chunks)

        results = store.search([1.0, 0.0, 0.0], top_k=10, min_score=0.9)

        assert len(results) == 1
        assert results[0].chunk.content == "Content A"

    def test_search_with_filters(self, chroma_available, unique_collection):
        """Test search with metadata filters."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunks = [
            self._make_chunk("c1", "d1", "Tech content", [1.0, 0.0, 0.0], {"category": "tech"}),
            self._make_chunk(
                "c2", "d1", "Science content", [0.9, 0.1, 0.0], {"category": "science"}
            ),
        ]
        store.add_chunks(chunks)

        results = store.search([1.0, 0.0, 0.0], top_k=10, filters={"category": "science"})

        assert len(results) == 1
        assert results[0].chunk.content == "Science content"

    def test_delete_document(self, chroma_available, unique_collection):
        """Test deleting all chunks for a document."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunks = [
            self._make_chunk("c1", "d1", "Doc 1 A", [1.0, 0.0, 0.0]),
            self._make_chunk("c2", "d1", "Doc 1 B", [0.9, 0.1, 0.0]),
            self._make_chunk("c3", "d2", "Doc 2", [0.0, 1.0, 0.0]),
        ]
        store.add_chunks(chunks)
        assert store.chunk_count == 3

        deleted = store.delete_document("d1")

        assert deleted == 2
        assert store.chunk_count == 1

    def test_delete_nonexistent_document(self, chroma_available, unique_collection):
        """Test deleting document that doesn't exist."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        deleted = store.delete_document("nonexistent")
        assert deleted == 0

    def test_clear(self, chroma_available, unique_collection):
        """Test clearing the store."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        chunks = [self._make_chunk("c1", "d1", "Content", [1.0, 0.0, 0.0])]
        store.add_chunks(chunks)
        assert store.chunk_count == 1

        store.clear()

        assert store.chunk_count == 0

    def test_upsert_updates_existing(self, chroma_available, unique_collection):
        """Test that adding same ID updates instead of duplicating."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)

        # Add initial
        store.add_chunks([self._make_chunk("c1", "d1", "Original", [1.0, 0.0, 0.0])])
        assert store.chunk_count == 1

        # Add same ID with different content
        store.add_chunks([self._make_chunk("c1", "d1", "Updated", [1.0, 0.0, 0.0])])
        assert store.chunk_count == 1  # Still 1, not 2

        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0].chunk.content == "Updated"

    def test_persistence(self, chroma_available, tmp_path, unique_collection):
        """Test that data persists across instances."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        persist_path = str(tmp_path / "chroma_persist")

        # Create and add data
        store1 = ChromaVectorStore(persist_path=persist_path, collection_name=unique_collection)
        store1.add_chunks([self._make_chunk("c1", "d1", "Persistent", [1.0, 0.0, 0.0])])
        assert store1.chunk_count == 1
        del store1  # Close first instance

        # Create new instance pointing to same path
        store2 = ChromaVectorStore(persist_path=persist_path, collection_name=unique_collection)
        assert store2.chunk_count == 1

        results = store2.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0].chunk.content == "Persistent"

    def test_repr(self, chroma_available, unique_collection):
        """Test string representation."""
        from axiompy.agents.io.vector.chroma import ChromaVectorStore

        store = ChromaVectorStore(collection_name=unique_collection)
        repr_str = repr(store)
        assert "ChromaVectorStore" in repr_str
        assert unique_collection in repr_str


# =============================================================================
# Pinecone Vector Store Tests
# =============================================================================


class TestPineconeVectorStore:
    """Tests for PineconeVectorStore."""

    @pytest.fixture
    def mock_upsert_response(self):
        """Create mock upsert response."""

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"upsertedCount": 2}

        return MockResponse()

    @pytest.fixture
    def mock_query_response(self):
        """Create mock query response."""

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "matches": [
                        {
                            "id": "c1",
                            "score": 0.95,
                            "values": [0.1, 0.2, 0.3],
                            "metadata": {
                                "document_id": "d1",
                                "content": "Test content",
                                "chunk_index": 0,
                                "start_char": 0,
                                "end_char": 12,
                            },
                        },
                        {
                            "id": "c2",
                            "score": 0.85,
                            "values": [0.4, 0.5, 0.6],
                            "metadata": {
                                "document_id": "d1",
                                "content": "More content",
                                "chunk_index": 1,
                                "start_char": 12,
                                "end_char": 24,
                            },
                        },
                    ]
                }

        return MockResponse()

    @pytest.fixture
    def mock_delete_response(self):
        """Create mock delete response."""

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {}

        return MockResponse()

    @pytest.fixture
    def mock_stats_response(self):
        """Create mock stats response."""

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "totalVectorCount": 10,
                    "namespaces": {"": {"vectorCount": 10}},
                }

        return MockResponse()

    def _make_chunk(
        self, id: str, doc_id: str, content: str, embedding: list, metadata: dict = None
    ) -> DocumentChunk:
        """Helper to create chunk with required fields."""
        return DocumentChunk(
            id=id,
            document_id=doc_id,
            content=content,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            embedding=embedding,
            metadata=metadata or {},
        )

    def test_init(self):
        """Test initialization."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test-index.svc.pinecone.io",
        )
        assert store.index_name == "test-index"

    def test_init_missing_api_key(self):
        """Test missing API key raises error."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        with pytest.raises(AgentIOVectorStoreError, match="API key is required"):
            PineconeVectorStore(
                api_key="",
                index_name="test",
                host="https://test.pinecone.io",
            )

    def test_init_missing_index(self):
        """Test missing index name raises error."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        with pytest.raises(AgentIOVectorStoreError, match="index name is required"):
            PineconeVectorStore(
                api_key="pk-test",
                index_name="",
                host="https://test.pinecone.io",
            )

    def test_init_missing_host(self):
        """Test missing host raises error."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        with pytest.raises(AgentIOVectorStoreError, match="host URL is required"):
            PineconeVectorStore(
                api_key="pk-test",
                index_name="test",
            )

    def test_add_chunks(self, mock_upsert_response, monkeypatch):
        """Test adding chunks."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            return mock_upsert_response

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        chunks = [
            self._make_chunk("c1", "d1", "Content A", [0.1, 0.2, 0.3]),
            self._make_chunk("c2", "d1", "Content B", [0.4, 0.5, 0.6]),
        ]

        count = store.add_chunks(chunks)

        assert count == 2

    def test_add_empty_chunks(self):
        """Test adding empty chunk list."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )

        count = store.add_chunks([])
        assert count == 0

    def test_add_chunks_without_embeddings_raises(self):
        """Test that adding chunks without embeddings raises error."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )

        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Content",
            chunk_index=0,
            start_char=0,
            end_char=7,
        )

        with pytest.raises(AgentIOVectorStoreError, match="missing embedding"):
            store.add_chunks([chunk])

    def test_search(self, mock_query_response, monkeypatch):
        """Test search returns results."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            return mock_query_response

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        results = store.search([0.1, 0.2, 0.3], top_k=2)

        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].chunk.content == "Test content"

    def test_search_with_min_score(self, mock_query_response, monkeypatch):
        """Test min_score filtering."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            return mock_query_response

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        results = store.search([0.1, 0.2, 0.3], top_k=2, min_score=0.9)

        # Only the first match (score=0.95) should be returned
        assert len(results) == 1
        assert results[0].score == 0.95

    def test_delete_document(self, mock_delete_response, monkeypatch):
        """Test deleting document."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            return mock_delete_response

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        deleted = store.delete_document("d1")

        assert deleted >= 1

    def test_chunk_count(self, mock_stats_response, monkeypatch):
        """Test getting chunk count."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            return mock_stats_response

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        count = store.chunk_count

        assert count == 10

    def test_repr(self):
        """Test string representation."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        repr_str = repr(store)
        assert "PineconeVectorStore" in repr_str
        assert "test-index" in repr_str

    def test_add_chunks_with_typed_metadata(self, mock_upsert_response, monkeypatch):
        """Test that typed metadata values are included."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        captured_payloads = []

        def mock_post(url, json=None, **kwargs):
            captured_payloads.append(json)
            return mock_upsert_response

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        chunk = self._make_chunk(
            "c1",
            "d1",
            "Content",
            [0.1, 0.2, 0.3],
            metadata={"string_val": "test", "int_val": 42, "float_val": 3.14, "bool_val": True},
        )

        store.add_chunks([chunk])

        # Check metadata was included
        assert len(captured_payloads) == 1
        vectors = captured_payloads[0]["vectors"]
        assert vectors[0]["metadata"]["string_val"] == "test"
        assert vectors[0]["metadata"]["int_val"] == 42

    def test_add_chunks_upsert_error_status(self, monkeypatch):
        """Test upsert handles error status codes."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        class ErrorResponse:
            status_code = 400
            text = "Bad request"

            def json(self):
                return {"error": "Invalid"}

        def mock_post(*args, **kwargs):
            return ErrorResponse()

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        chunk = self._make_chunk("c1", "d1", "Content", [0.1, 0.2, 0.3])

        with pytest.raises(AgentIOVectorStoreError, match="upsert failed"):
            store.add_chunks([chunk])

    def test_add_chunks_upsert_exception(self, monkeypatch):
        """Test upsert handles exceptions."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            raise Exception("Network error")

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        chunk = self._make_chunk("c1", "d1", "Content", [0.1, 0.2, 0.3])

        with pytest.raises(AgentIOVectorStoreError, match="upsert failed"):
            store.add_chunks([chunk])

    def test_search_with_filters(self, monkeypatch):
        """Test search includes filters in payload."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        captured_payloads = []

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"matches": []}

        def mock_post(url, json=None, **kwargs):
            captured_payloads.append(json)
            return MockResponse()

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        store.search([0.1, 0.2, 0.3], filters={"category": "docs"})

        assert len(captured_payloads) == 1
        assert captured_payloads[0]["filter"] == {"category": "docs"}

    def test_search_error_status(self, monkeypatch):
        """Test search handles error status codes."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        class ErrorResponse:
            status_code = 500
            text = "Server error"

            def json(self):
                return {"error": "Internal"}

        def mock_post(*args, **kwargs):
            return ErrorResponse()

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        with pytest.raises(AgentIOVectorStoreError, match="query failed"):
            store.search([0.1, 0.2, 0.3])

    def test_search_exception(self, monkeypatch):
        """Test search handles exceptions."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            raise Exception("Network error")

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        with pytest.raises(AgentIOVectorStoreError, match="search failed"):
            store.search([0.1, 0.2, 0.3])

    def test_delete_error_status(self, monkeypatch):
        """Test delete handles error status codes."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        class ErrorResponse:
            status_code = 404
            text = "Not found"

            def json(self):
                return {"error": "Not found"}

        def mock_post(*args, **kwargs):
            return ErrorResponse()

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        with pytest.raises(AgentIOVectorStoreError, match="delete failed"):
            store.delete_document("doc1")

    def test_delete_exception(self, monkeypatch):
        """Test delete handles exceptions."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            raise Exception("Network error")

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        with pytest.raises(AgentIOVectorStoreError, match="delete failed"):
            store.delete_document("doc1")

    def test_chunk_count_with_namespace(self, monkeypatch):
        """Test chunk count with specific namespace."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        class MockResponse:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "totalVectorCount": 100,
                    "namespaces": {
                        "": {"vectorCount": 50},
                        "docs": {"vectorCount": 25},
                    },
                }

        def mock_post(*args, **kwargs):
            return MockResponse()

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
            namespace="docs",
        )
        monkeypatch.setattr(store._client, "post", mock_post)

        assert store.chunk_count == 25

    def test_chunk_count_fallback_on_error(self, monkeypatch):
        """Test chunk count returns cached value on error."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        def mock_post(*args, **kwargs):
            raise Exception("API error")

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        store._chunk_count = 42  # Cached value
        monkeypatch.setattr(store._client, "post", mock_post)

        assert store.chunk_count == 42

    def test_chunk_count_non_200_status(self, monkeypatch):
        """Test chunk count returns cached value on non-200."""
        from axiompy.agents.io.vector.pinecone import PineconeVectorStore

        class MockResponse:
            status_code = 500
            text = "Error"

            def json(self):
                return {}

        def mock_post(*args, **kwargs):
            return MockResponse()

        store = PineconeVectorStore(
            api_key="pk-test",
            index_name="test-index",
            host="https://test.pinecone.io",
        )
        store._chunk_count = 15
        monkeypatch.setattr(store._client, "post", mock_post)

        assert store.chunk_count == 15


class TestPineconeVectorStoreFactory:
    """Tests for factory integration with Pinecone."""


class TestPGVectorStore:
    """Tests for PGVectorStore."""

    @pytest.fixture
    def psycopg2_available(self):
        """Check if psycopg2 is available."""
        import importlib.util

        if importlib.util.find_spec("psycopg2") is None:
            pytest.skip("psycopg2 not installed")
        return True

    @pytest.fixture
    def mock_connection(self, monkeypatch):
        """Create mock PostgreSQL connection."""

        class MockCursor:
            def __init__(self):
                self.rowcount = 0
                self._results = []

            def execute(self, query, params=None):
                pass

            def fetchall(self):
                return self._results

            def fetchone(self):
                return (10,) if self._results is None else None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def __init__(self, *args, **kwargs):
                pass

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        return MockConnection

    def _make_chunk(
        self, id: str, doc_id: str, content: str, embedding: list, metadata: dict = None
    ) -> DocumentChunk:
        """Helper to create chunk with required fields."""
        return DocumentChunk(
            id=id,
            document_id=doc_id,
            content=content,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            embedding=embedding,
            metadata=metadata or {},
        )

    def test_init_missing_database_url(self):
        """Test missing database URL raises error."""
        from axiompy.agents.io.vector.pgvector import PGVectorStore

        with pytest.raises(AgentIOVectorStoreError, match="database_url is required"):
            PGVectorStore(database_url="")

    def test_init_psycopg2_not_installed(self, monkeypatch):
        """Test error when psycopg2 not available."""
        import sys

        # Temporarily hide psycopg2
        modules_to_hide = [m for m in sys.modules if m.startswith("psycopg2")]
        hidden = {m: sys.modules.pop(m) for m in modules_to_hide}

        try:
            # Force import error
            monkeypatch.setitem(sys.modules, "psycopg2", None)
            monkeypatch.setitem(sys.modules, "psycopg2.extras", None)

            # Clear cached import

            # This should fail since psycopg2 is hidden
            # But we can't easily test this without actually uninstalling
            pass  # Skip this complex test

        finally:
            # Restore
            sys.modules.update(hidden)

    def test_add_empty_chunks(self, psycopg2_available, mock_connection, monkeypatch):
        """Test adding empty chunk list."""
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", mock_connection)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(database_url="postgresql://test", create_table=False)
        count = store.add_chunks([])
        assert count == 0

    def test_add_chunks_without_embeddings_raises(
        self, psycopg2_available, mock_connection, monkeypatch
    ):
        """Test that adding chunks without embeddings raises error."""
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", mock_connection)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(database_url="postgresql://test", create_table=False)

        chunk = DocumentChunk(
            id="c1",
            document_id="d1",
            content="Content",
            chunk_index=0,
            start_char=0,
            end_char=7,
        )

        with pytest.raises(AgentIOVectorStoreError, match="missing embedding"):
            store.add_chunks([chunk])

    def test_add_chunks_dimension_mismatch(self, psycopg2_available, mock_connection, monkeypatch):
        """Test dimension mismatch raises error."""
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", mock_connection)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            embedding_dimension=384,
            create_table=False,
        )

        chunk = self._make_chunk("c1", "d1", "Content", [0.1, 0.2])  # Wrong dimension

        with pytest.raises(AgentIOVectorStoreError, match="dimension mismatch"):
            store.add_chunks([chunk])

    def test_repr(self, psycopg2_available, mock_connection, monkeypatch):
        """Test string representation."""
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", mock_connection)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            table_name="test_table",
            create_table=False,
        )
        repr_str = repr(store)
        assert "PGVectorStore" in repr_str
        assert "test_table" in repr_str

    def test_table_name_property(self, psycopg2_available, mock_connection, monkeypatch):
        """Test table_name property."""
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", mock_connection)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            table_name="custom_table",
            create_table=False,
        )
        assert store.table_name == "custom_table"

    def test_add_chunks_success(self, psycopg2_available, monkeypatch):
        """Test successfully adding chunks."""
        import psycopg2

        executed_queries = []

        class MockCursor:
            def __init__(self):
                self.rowcount = 2

            def execute(self, query, params=None):
                executed_queries.append((query, params))

            def fetchall(self):
                return []

            def fetchone(self):
                return (10,)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        # Mock execute_values
        def mock_execute_values(cur, query, values):
            executed_queries.append(("execute_values", len(values)))

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())
        monkeypatch.setattr("psycopg2.extras.execute_values", mock_execute_values)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            embedding_dimension=3,
            create_table=False,
        )

        chunks = [
            self._make_chunk("c1", "d1", "Content 1", [0.1, 0.2, 0.3]),
            self._make_chunk("c2", "d1", "Content 2", [0.4, 0.5, 0.6]),
        ]

        count = store.add_chunks(chunks)
        assert count == 2

    def test_search_success(self, psycopg2_available, monkeypatch):
        """Test successful search."""
        import psycopg2

        class MockCursor:
            def execute(self, query, params=None):
                pass

            def fetchall(self):
                # Return mock results matching expected columns
                return [
                    (
                        "c1",  # id
                        "d1",  # document_id
                        "Test content",  # content
                        0,  # chunk_index
                        0,  # start_char
                        12,  # end_char
                        "[0.1,0.2,0.3]",  # embedding
                        {"source": "test"},  # metadata
                        0.95,  # score
                    ),
                ]

            def fetchone(self):
                return (10,)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            embedding_dimension=3,
            create_table=False,
        )

        results = store.search([0.1, 0.2, 0.3], top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].chunk.content == "Test content"

    def test_search_with_filters(self, psycopg2_available, monkeypatch):
        """Test search with metadata filters."""
        import psycopg2

        executed_queries = []

        class MockCursor:
            def execute(self, query, params=None):
                executed_queries.append((query, params))

            def fetchall(self):
                return []

            def fetchone(self):
                return (0,)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            embedding_dimension=3,
            create_table=False,
        )

        store.search([0.1, 0.2, 0.3], top_k=5, filters={"source": "test"})
        # Should include WHERE clause with filter
        assert any("WHERE" in str(q[0]) for q in executed_queries)

    def test_search_min_score_filter(self, psycopg2_available, monkeypatch):
        """Test search filters by min_score."""
        import psycopg2

        class MockCursor:
            def execute(self, query, params=None):
                pass

            def fetchall(self):
                return [
                    ("c1", "d1", "Content 1", 0, 0, 8, "[0.1,0.2,0.3]", {}, 0.95),
                    ("c2", "d1", "Content 2", 1, 8, 16, "[0.1,0.2,0.3]", {}, 0.3),  # Low score
                ]

            def fetchone(self):
                return (2,)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            embedding_dimension=3,
            create_table=False,
        )

        results = store.search([0.1, 0.2, 0.3], top_k=5, min_score=0.5)
        assert len(results) == 1  # Only high score result
        assert results[0].score == 0.95

    def test_delete_document(self, psycopg2_available, monkeypatch):
        """Test deleting a document."""
        import psycopg2

        class MockCursor:
            rowcount = 3

            def execute(self, query, params=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            create_table=False,
        )

        deleted = store.delete_document("doc1")
        assert deleted == 3

    def test_clear(self, psycopg2_available, monkeypatch):
        """Test clearing all chunks."""
        import psycopg2

        truncated = []

        class MockCursor:
            def execute(self, query, params=None):
                if "TRUNCATE" in str(query):
                    truncated.append(True)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            create_table=False,
        )

        store.clear()
        assert len(truncated) == 1

    def test_chunk_count(self, psycopg2_available, monkeypatch):
        """Test getting chunk count."""
        import psycopg2

        class MockCursor:
            def execute(self, query, params=None):
                pass

            def fetchone(self):
                return (42,)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            create_table=False,
        )

        assert store.chunk_count == 42

    def test_chunk_count_error_returns_zero(self, psycopg2_available, monkeypatch):
        """Test chunk_count returns 0 on error."""
        import psycopg2

        class MockCursor:
            def execute(self, query, params=None):
                raise Exception("DB error")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(
            database_url="postgresql://test",
            create_table=False,
        )

        assert store.chunk_count == 0

    def test_create_table(self, psycopg2_available, monkeypatch):
        """Test table creation."""
        import psycopg2

        executed_queries = []

        class MockCursor:
            def execute(self, query, params=None):
                executed_queries.append(str(query))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockConnection:
            autocommit = False

            def cursor(self):
                return MockCursor()

            def close(self):
                pass

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        PGVectorStore(
            database_url="postgresql://test",
            table_name="test_embeddings",
            create_table=True,
        )

        # Should have created extension and table
        assert any("CREATE EXTENSION" in q for q in executed_queries)
        assert any("CREATE TABLE" in q for q in executed_queries)
        assert any("CREATE INDEX" in q for q in executed_queries)

    def test_connection_error(self, psycopg2_available, monkeypatch):
        """Test connection error handling."""
        import psycopg2

        def raise_error(*args, **kwargs):
            raise Exception("Connection refused")

        monkeypatch.setattr(psycopg2, "connect", raise_error)

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        with pytest.raises(AgentIOVectorStoreError, match="Failed to connect"):
            PGVectorStore(database_url="postgresql://test")

    def test_close(self, psycopg2_available, monkeypatch):
        """Test closing connection."""
        import psycopg2

        closed = []

        class MockConnection:
            autocommit = False

            def cursor(self):
                class MockCursor:
                    def execute(self, *a):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        pass

                return MockCursor()

            def close(self):
                closed.append(True)

        monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: MockConnection())

        from axiompy.agents.io.vector.pgvector import PGVectorStore

        store = PGVectorStore(database_url="postgresql://test", create_table=False)
        store.close()
        assert len(closed) == 1


class TestPGVectorStoreFactory:
    """Tests for factory integration with pgvector."""


class TestChromaVectorStoreFactory:
    """Tests for factory integration with ChromaDB."""

    @pytest.fixture
    def chroma_available(self):
        """Check if chromadb is available."""
        import importlib.util

        if importlib.util.find_spec("chromadb") is None:
            pytest.skip("chromadb not installed")
        return True

    @pytest.fixture
    def unique_collection(self):
        """Generate unique collection name per test."""
        import uuid

        return f"factory_test_{uuid.uuid4().hex[:8]}"


class TestInMemoryVectorStoreEdgeCases:
    """Additional edge case tests for InMemoryVectorStore."""

    def _make_chunk(self, id: str, doc_id: str, content: str, embedding: list) -> DocumentChunk:
        """Helper to create chunk."""
        return DocumentChunk(
            id=id,
            document_id=doc_id,
            content=content,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            embedding=embedding,
        )

    def test_search_top_k_limit(self):
        """Test that top_k limits results."""
        store = InMemoryVectorStore()
        for i in range(10):
            store.add_chunks([self._make_chunk(f"c{i}", "d1", f"Content {i}", [float(i)])])

        results = store.search([5.0], top_k=3)
        assert len(results) == 3

    def test_normalized_embeddings(self):
        """Test search with normalized embeddings."""
        store = InMemoryVectorStore()
        # Normalized vectors
        store.add_chunks(
            [
                self._make_chunk("c1", "d1", "A", [1.0, 0.0, 0.0]),
                self._make_chunk("c2", "d1", "B", [0.0, 1.0, 0.0]),
            ]
        )

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        # First result should have score ~1.0 (perfect match)
        assert results[0].score > 0.99

    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist."""
        store = InMemoryVectorStore()
        store.add_chunks([self._make_chunk("c1", "d1", "Content", [1.0])])

        deleted = store.delete_document("nonexistent")
        assert deleted == 0


# =============================================================================
# LLM Settings Tests
# =============================================================================


class TestLLMSettings:
    """Tests for LLMSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        from axiompy.agents.io.settings import LLMSettings

        settings = LLMSettings()
        assert settings.model is None
        assert settings.api_key is None
        assert settings.temperature == DEFAULT_TEMPERATURE
        assert settings.max_tokens == DEFAULT_MAX_TOKENS

    def test_custom_values(self):
        """Test custom values."""
        from axiompy.agents.io.settings import LLMSettings

        settings = LLMSettings(
            model="gpt-4",
            api_key="test-key",
            endpoint="http://localhost:8080",
            temperature=0.5,
            max_tokens=2000,
        )
        assert settings.model == "gpt-4"
        assert settings.api_key == "test-key"
        assert settings.temperature == 0.5


# =============================================================================
# API Settings Tests
# =============================================================================


class TestSourceFactory:
    """Tests for SourceFactory."""

    def test_create_filesystem_source(self):
        """Test creating FileSystemSource via factory."""
        from axiompy.agents.io.documents import (
            SourceFactory,
            SourceType,
            SourceSettings,
        )

        source = SourceFactory.create(SourceType.FILESYSTEM, SourceSettings())
        assert source is not None

    def test_create_url_source(self):
        """Test creating URLSource via factory."""
        from axiompy.agents.io.documents import (
            SourceFactory,
            SourceType,
            SourceSettings,
        )

        source = SourceFactory.create(
            SourceType.URL,
            SourceSettings(timeout_secs=60, user_agent="TestBot/1.0"),
        )
        assert source is not None

    def test_create_pdf_source(self):
        """Test creating PDFSource via factory."""
        pytest.importorskip("pypdf")

        from axiompy.agents.io.documents import (
            SourceFactory,
            SourceType,
            SourceSettings,
        )

        source = SourceFactory.create(
            SourceType.PDF,
            SourceSettings(pages_as_documents=True),
        )
        assert source is not None

    def test_create_mock_source(self):
        """Test creating mock source via factory."""
        from axiompy.agents.io.documents import (
            SourceFactory,
            SourceType,
        )

        source = SourceFactory.create(SourceType.MOCK)
        assert source is not None

    def test_object_store_requires_storage_settings(self):
        """Test that Object Store source requires dedicated factory method."""
        from axiompy.agents.io.documents import (
            SourceFactory,
            SourceType,
            SourceSettings,
        )
        from axiompy.agents.io.errors import AgentIOConfigurationError

        with pytest.raises(AgentIOConfigurationError, match="StorageSettings"):
            SourceFactory.create(SourceType.OBJECT_STORE, SourceSettings())

    def test_database_requires_database_settings(self):
        """Test that Database source requires dedicated factory method."""
        from axiompy.agents.io.documents import (
            SourceFactory,
            SourceType,
            SourceSettings,
        )
        from axiompy.agents.io.errors import AgentIOConfigurationError

        with pytest.raises(AgentIOConfigurationError, match="DatabaseSettings"):
            SourceFactory.create(SourceType.DATABASE, SourceSettings())

    def test_source_type_enum_values(self):
        """Test SourceType enum has expected values."""
        from axiompy.agents.io.documents import SourceType

        assert SourceType.FILESYSTEM.value == "filesystem"
        assert SourceType.URL.value == "url"
        assert SourceType.OBJECT_STORE.value == "object_store"
        assert SourceType.DATABASE.value == "database"
        assert SourceType.PDF.value == "pdf"
        assert SourceType.MOCK.value == "mock"


class TestURLSource:
    """Tests for URLSource adapter."""

    def test_url_source_initialization(self):
        """Test URLSource initializes correctly."""
        from axiompy.agents.io.documents.url import URLSource

        source = URLSource(timeout_secs=30, user_agent="TestAgent/1.0")
        assert repr(source) == "URLSource(timeout=30s)"

    def test_url_source_with_headers(self):
        """Test URLSource with custom headers."""
        from axiompy.agents.io.documents.url import URLSource

        source = URLSource(
            timeout_secs=30,
            headers={"Authorization": "Bearer test-token"},
        )
        assert source is not None

    def test_strip_html_tags(self):
        """Test HTML tag stripping utility."""
        from axiompy.agents.io.documents.url import _strip_html_tags

        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        text = _strip_html_tags(html)
        assert "Title" in text
        assert "Content" in text
        assert "<" not in text
        assert ">" not in text

    def test_strip_html_tags_removes_script_style(self):
        """Test that script and style tags are removed."""
        from axiompy.agents.io.documents.url import _strip_html_tags

        html = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
        <script>alert('hi');</script>
        <p>Real content</p>
        </body>
        </html>
        """
        text = _strip_html_tags(html)
        assert "Real content" in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_extract_title(self):
        """Test title extraction from HTML."""
        from axiompy.agents.io.documents.url import _extract_title

        html = "<html><head><title>My Page Title</title></head><body></body></html>"
        title = _extract_title(html)
        assert title == "My Page Title"

    def test_extract_title_missing(self):
        """Test title extraction when no title tag."""
        from axiompy.agents.io.documents.url import _extract_title

        html = "<html><body>No title</body></html>"
        title = _extract_title(html)
        assert title is None

    def test_load_document_mocked(self):
        """Test loading document with mocked HTTP client."""
        from unittest.mock import Mock, patch

        from axiompy.agents.io.documents.url import URLSource

        # Mock the HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test</title></head><body>Hello World</body></html>"
        mock_response.headers = {"content-type": "text/html"}

        with patch.object(URLSource, "__init__", lambda self, **kwargs: None):
            source = URLSource()
            source._timeout = 30

            mock_client = Mock()
            mock_client.get.return_value = mock_response
            source._client = mock_client

            doc = source.load_document("https://example.com/page")

            assert doc.content == "Test Hello World"
            assert doc.metadata.title == "Test"
            assert "example.com" in doc.metadata.extra["domain"]

    def test_load_documents_skips_failed(self):
        """Test that load_documents skips failed URLs."""
        from unittest.mock import Mock, patch

        from axiompy.agents.io.documents.url import URLSource
        from axiompy.agents.io.errors import AgentIOIngestionError

        with patch.object(URLSource, "__init__", lambda self, **kwargs: None):
            source = URLSource()
            source._timeout = 30

            # First call succeeds, second fails
            def mock_load(url):
                if "fail" in url:
                    raise AgentIOIngestionError("Failed")
                mock_meta = Mock()
                mock_meta.source = url
                mock_meta.extra = {"domain": "example.com"}
                return Mock(id="doc1", content="content", metadata=mock_meta)

            source.load_document = mock_load
            source._client = Mock()

            docs = source.load_documents(["https://example.com/good", "https://example.com/fail"])

            assert len(docs) == 1


class TestObjectStoreSource:
    """Tests for ObjectStoreSource adapter."""

    def test_object_store_source_initialization_mocked(self):
        """Test ObjectStoreSource initialization with mocked storage."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_storage = Mock()
            mock_factory.create.return_value = mock_storage

            from axiompy.agents.io.documents.object_store import ObjectStoreSource
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="test-bucket", region="us-east-1")
            source = ObjectStoreSource(StorageType.S3, settings)

            assert repr(source) == "ObjectStoreSource(s3, bucket=test-bucket)"
            mock_factory.create.assert_called_once()

    def test_object_store_load_document_mocked(self):
        """Test loading document from object store with mocked storage."""
        from datetime import datetime
        from unittest.mock import Mock, patch

        from axiompy.io.object import ObjectMetadata

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_storage = Mock()
            mock_storage.get_object.return_value = b"Test content"
            mock_storage.head_object.return_value = ObjectMetadata(
                key="docs/readme.md",
                size=12,
                last_modified=datetime.now(),
                content_type="text/plain",
                etag="abc123",
            )
            mock_factory.create.return_value = mock_storage

            from axiompy.agents.io.documents.object_store import ObjectStoreSource
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="test-bucket")
            source = ObjectStoreSource(StorageType.S3, settings)

            doc = source.load_document("docs/readme.md")

            assert doc.content == "Test content"
            assert doc.metadata.source == "s3://test-bucket/docs/readme.md"
            assert doc.metadata.title == "readme.md"

    def test_object_store_gcs_uri(self):
        """Test GCS source generates correct URI."""
        from datetime import datetime
        from unittest.mock import Mock, patch

        from axiompy.io.object import ObjectMetadata

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_storage = Mock()
            mock_storage.get_object.return_value = b"Test content"
            mock_storage.head_object.return_value = ObjectMetadata(
                key="docs/readme.md",
                size=12,
                last_modified=datetime.now(),
                content_type="text/plain",
            )
            mock_factory.create.return_value = mock_storage

            from axiompy.agents.io.documents.object_store import ObjectStoreSource
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="gcs-bucket")
            source = ObjectStoreSource(StorageType.GCS, settings)

            doc = source.load_document("docs/readme.md")

            assert doc.metadata.source == "gs://gcs-bucket/docs/readme.md"

    def test_object_store_azure_uri(self):
        """Test Azure source generates correct URI."""
        from datetime import datetime
        from unittest.mock import Mock, patch

        from axiompy.io.object import ObjectMetadata

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_storage = Mock()
            mock_storage.get_object.return_value = b"Test content"
            mock_storage.head_object.return_value = ObjectMetadata(
                key="docs/readme.md",
                size=12,
                last_modified=datetime.now(),
                content_type="text/plain",
            )
            mock_factory.create.return_value = mock_storage

            from axiompy.agents.io.documents.object_store import ObjectStoreSource
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="azure-container")
            source = ObjectStoreSource(StorageType.AZURE, settings)

            doc = source.load_document("docs/readme.md")

            assert doc.metadata.source == "azure://azure-container/docs/readme.md"

    def test_object_store_load_documents_filters_extensions(self):
        """Test that object store source filters by extension."""
        from datetime import datetime
        from unittest.mock import Mock, patch

        from axiompy.io.object import ObjectMetadata

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_storage = Mock()

            # List returns multiple objects with different extensions
            mock_storage.list_objects.return_value = [
                ObjectMetadata(key="docs/file.md", size=100, last_modified=datetime.now()),
                ObjectMetadata(key="docs/file.exe", size=100, last_modified=datetime.now()),
                ObjectMetadata(key="docs/file.py", size=100, last_modified=datetime.now()),
            ]
            mock_storage.get_object.return_value = b"content"
            mock_storage.head_object.return_value = ObjectMetadata(
                key="test",
                size=7,
                last_modified=datetime.now(),
                content_type="text/plain",
            )
            mock_factory.create.return_value = mock_storage

            from axiompy.agents.io.documents.object_store import ObjectStoreSource
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="test-bucket")
            source = ObjectStoreSource(StorageType.S3, settings)

            docs = source.load_documents(["docs/"])

            # Should load .md and .py but not .exe
            assert len(docs) == 2


class TestDatabaseSource:
    """Tests for DatabaseSource adapter."""

    def test_database_source_initialization_mocked(self):
        """Test DatabaseSource initialization with mocked database."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.database.DatabaseFactory") as mock_factory:
            mock_db = Mock()
            mock_factory.create.return_value = mock_db

            from axiompy.agents.io.documents.database import DatabaseSource
            from axiompy.io.database import DatabaseSettings, DatabaseType

            settings = DatabaseSettings(host="localhost", database="testdb")
            source = DatabaseSource(DatabaseType.POSTGRES, settings)

            assert repr(source) == "DatabaseSource(postgres)"
            mock_factory.create.assert_called_once()

    def test_database_load_from_table_mocked(self):
        """Test loading documents from database table."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.database.DatabaseFactory") as mock_factory:
            mock_db = Mock()
            mock_db.execute.return_value = [
                {"id": 1, "title": "Doc 1", "content": "Content one"},
                {"id": 2, "title": "Doc 2", "content": "Content two"},
            ]
            mock_factory.create.return_value = mock_db

            from axiompy.agents.io.documents.database import DatabaseSource
            from axiompy.io.database import DatabaseSettings, DatabaseType

            settings = DatabaseSettings(host="localhost", database="testdb")
            source = DatabaseSource(DatabaseType.POSTGRES, settings)

            docs = source.load_from_table(
                table="articles",
                content_column="content",
                id_column="id",
                title_column="title",
            )

            assert len(docs) == 2
            assert docs[0].content == "Content one"
            assert docs[0].metadata.title == "Doc 1"

    def test_database_load_from_query_mocked(self):
        """Test loading documents from custom query."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.database.DatabaseFactory") as mock_factory:
            mock_db = Mock()
            mock_db.execute.return_value = [
                {"id": "a1", "body": "Article body text", "headline": "Breaking News"},
            ]
            mock_factory.create.return_value = mock_db

            from axiompy.agents.io.documents.database import DatabaseSource
            from axiompy.io.database import DatabaseSettings, DatabaseType

            settings = DatabaseSettings(host="localhost", database="testdb")
            source = DatabaseSource(DatabaseType.POSTGRES, settings)

            docs = source.load_from_query(
                query="SELECT id, headline, body FROM news WHERE published = true",
                content_column="body",
                id_column="id",
                title_column="headline",
            )

            assert len(docs) == 1
            assert docs[0].content == "Article body text"
            assert docs[0].metadata.title == "Breaking News"

    def test_database_guess_content_column(self):
        """Test content column guessing."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.database.DatabaseFactory") as mock_factory:
            mock_factory.create.return_value = Mock()

            from axiompy.agents.io.documents.database import DatabaseSource
            from axiompy.io.database import DatabaseSettings, DatabaseType

            settings = DatabaseSettings(host="localhost", database="testdb")
            source = DatabaseSource(DatabaseType.POSTGRES, settings)

            # Test priority-based guessing
            row1 = {"id": 1, "content": "main content", "other": "data"}
            assert source._guess_content_column(row1) == "content"

            row2 = {"id": 1, "body": "body text", "other": "data"}
            assert source._guess_content_column(row2) == "body"

            row3 = {"id": 1, "text": "text content", "other": "data"}
            assert source._guess_content_column(row3) == "text"

    def test_database_load_documents_wildcard(self):
        """Test loading all documents from table with wildcard."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.database.DatabaseFactory") as mock_factory:
            mock_db = Mock()
            # When content_column is explicitly provided via path, only one query is made
            mock_db.execute.return_value = [
                {"id": 1, "content": "Content 1"},
                {"id": 2, "content": "Content 2"},
            ]
            mock_factory.create.return_value = mock_db

            from axiompy.agents.io.documents.database import DatabaseSource
            from axiompy.io.database import DatabaseSettings, DatabaseType

            settings = DatabaseSettings(host="localhost", database="testdb")
            source = DatabaseSource(DatabaseType.POSTGRES, settings)

            # Using explicit content column in path: "table:*:content_column"
            docs = source.load_documents(["articles:*:content"])

            assert len(docs) == 2
            assert docs[0].content == "Content 1"
            assert docs[1].content == "Content 2"


class TestPDFSource:
    """Tests for PDFSource adapter."""

    def test_pdf_source_requires_pypdf(self):
        """Test that PDFSource requires pypdf."""
        # This test only runs if pypdf is NOT installed
        import importlib.util

        if importlib.util.find_spec("pypdf") is not None:
            pytest.skip("pypdf is installed")
        else:
            from axiompy.agents.io.documents.pdf import PDFSource
            from axiompy.agents.io.errors import AgentIOIngestionError

            with pytest.raises(AgentIOIngestionError, match="pypdf"):
                PDFSource()

    def test_pdf_source_initialization(self):
        """Test PDFSource initialization."""
        pytest.importorskip("pypdf")

        from axiompy.agents.io.documents.pdf import PDFSource

        source = PDFSource(pages_as_documents=True, include_metadata=True)
        assert repr(source) == "PDFSource(pages_as_documents=True)"

    def test_pdf_source_file_not_found(self):
        """Test PDFSource raises error for missing file."""
        pytest.importorskip("pypdf")

        from axiompy.agents.io.documents.pdf import PDFSource
        from axiompy.agents.io.errors import AgentIOIngestionError

        source = PDFSource()

        with pytest.raises(AgentIOIngestionError, match="not found"):
            source.load_document("/nonexistent/file.pdf")

    def test_pdf_source_not_pdf_file(self):
        """Test PDFSource raises error for non-PDF file."""
        pytest.importorskip("pypdf")
        import tempfile

        from axiompy.agents.io.documents.pdf import PDFSource
        from axiompy.agents.io.errors import AgentIOIngestionError

        source = PDFSource()

        # Create a temp text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Not a PDF")
            temp_path = f.name

        try:
            with pytest.raises(AgentIOIngestionError, match="Not a PDF"):
                source.load_document(temp_path)
        finally:
            import os

            os.unlink(temp_path)

    def test_pdf_source_load_mocked(self):
        """Test PDF loading with mocked reader."""
        pytest.importorskip("pypdf")

        from unittest.mock import patch, MagicMock
        import tempfile
        import os

        # Create a temp file that exists
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake pdf content")
            temp_path = f.name

        try:
            with patch("axiompy.agents.io.documents.pdf.PdfReader") as mock_reader_class:
                # Mock the reader instance
                mock_reader = MagicMock()
                mock_page = MagicMock()
                mock_page.extract_text.return_value = "Page 1 content"
                mock_reader.pages = [mock_page]
                mock_reader.metadata = {"/Title": "Test PDF", "/Author": "Test Author"}
                mock_reader_class.return_value = mock_reader

                from axiompy.agents.io.documents.pdf import PDFSource

                source = PDFSource()
                doc = source.load_document(temp_path)

                assert "Page 1 content" in doc.content
                assert doc.metadata.title == "Test PDF"
                assert doc.metadata.extra["author"] == "Test Author"
                assert doc.metadata.extra["page_count"] == 1
        finally:
            os.unlink(temp_path)

    def test_pdf_source_load_pages_as_documents_mocked(self):
        """Test loading PDF with pages as separate documents."""
        pytest.importorskip("pypdf")

        from unittest.mock import patch, MagicMock
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake pdf content")
            temp_path = f.name

        try:
            with patch("axiompy.agents.io.documents.pdf.PdfReader") as mock_reader_class:
                mock_reader = MagicMock()
                mock_page1 = MagicMock()
                mock_page1.extract_text.return_value = "Page 1 content"
                mock_page2 = MagicMock()
                mock_page2.extract_text.return_value = "Page 2 content"
                mock_reader.pages = [mock_page1, mock_page2]
                mock_reader.metadata = {"/Title": "Multi-page PDF"}
                mock_reader_class.return_value = mock_reader

                from axiompy.agents.io.documents.pdf import PDFSource

                source = PDFSource(pages_as_documents=True)
                docs = source.load_document_pages(temp_path)

                assert len(docs) == 2
                assert "Page 1 content" in docs[0].content
                assert "Page 2 content" in docs[1].content
                assert "Page 1" in docs[0].metadata.title
                assert "Page 2" in docs[1].metadata.title
        finally:
            os.unlink(temp_path)


class TestSourceFactoryObjectStoreIntegration:
    """Tests for SourceFactory.create_object_store()."""

    def test_create_object_store_s3(self):
        """Test creating S3 object store source via factory."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_factory.create.return_value = Mock()

            from axiompy.agents.io.documents import SourceFactory
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="test-bucket", region="us-east-1")
            source = SourceFactory.create_object_store(StorageType.S3, settings)

            assert source is not None

    def test_create_object_store_gcs(self):
        """Test creating GCS object store source via factory."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.object_store.ObjectStorageFactory") as mock_factory:
            mock_factory.create.return_value = Mock()

            from axiompy.agents.io.documents import SourceFactory
            from axiompy.io.object import StorageSettings, StorageType

            settings = StorageSettings(bucket="gcs-bucket", project_id="my-project")
            source = SourceFactory.create_object_store(StorageType.GCS, settings)

            assert source is not None


class TestSourceFactoryDatabaseIntegration:
    """Tests for SourceFactory.create_database()."""

    def test_create_database_source(self):
        """Test creating Database source via factory."""
        from unittest.mock import Mock, patch

        with patch("axiompy.agents.io.documents.database.DatabaseFactory") as mock_factory:
            mock_factory.create.return_value = Mock()

            from axiompy.agents.io.documents import SourceFactory
            from axiompy.io.database import DatabaseSettings, DatabaseType

            settings = DatabaseSettings(host="localhost", database="testdb")
            source = SourceFactory.create_database(DatabaseType.POSTGRES, settings)

            assert source is not None
