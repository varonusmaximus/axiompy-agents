# RAG Agent

A local-first Retrieval-Augmented Generation (RAG) agent for building knowledge-grounded AI applications.

## Overview

The `axiompy.agents.rag` module provides a complete RAG pipeline with support for:

- ✅ **Local-First Embeddings**: FastEmbed (ONNX) and sentence-transformers run in-process
- ✅ **Multiple Vector Stores**: In-memory, ChromaDB, Pinecone, pgvector
- ✅ **Pluggable LLM Providers**: Integrates with `axiompy.reasoning` for Ollama, OpenAI, Anthropic
- ✅ **Document Sources**: FileSystem, URL, Object Store (S3/GCS/Azure), Database, PDF
- ✅ **Clean Architecture**: Protocol-based ports for testability and flexibility
- ✅ **Factory Pattern**: Enum-based type selection consistent with axiompy conventions
- 🔐 **Offline Mode**: `local_files_only=True` by default (no network calls)

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [CLI Usage](#cli-usage)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Extending the RAG Module](#extending-the-rag-module)

## Quick Start

```python
from axiompy.agents.rag import (
    RAGServiceFactory,
    EmbedderType,
    VectorStoreType,
    EmbedderSettings,
)
from axiompy.agents.rag.adapters.sources import FileSystemSource

# Create RAG service with local embeddings
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.MEMORY,
    llm_provider="mock",  # Or "ollama", "openai", etc.
    embedder_settings=EmbedderSettings(
        model="BAAI/bge-small-en-v1.5",
        cache_dir="./models/fastembed",
    ),
)

# Ingest documents
source = FileSystemSource()
docs = source.load_documents(["./documents/"])
rag.ingest(docs)

# Search
results = rag.search("What is the return policy?", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:100]}...")

# Query with LLM generation
response = rag.query("What is the return policy?")
print(response.answer)
```

## Architecture

```
axiompy/agents/rag/
│
├── domain/                     # Core business logic
│   ├── models.py               # Document, Chunk, Query, SearchResult, RAGResponse
│   ├── ports.py                # Protocol interfaces (Embedder, VectorStore, etc.)
│   ├── chunker.py              # FixedSize, Sentence, Paragraph chunkers
│   └── service.py              # RAGService orchestrator
│
├── adapters/                   # Port implementations
│   ├── embedders/
│   │   ├── fastembed_embedder.py      # ✅ ONNX-based (recommended, local)
│   │   ├── sentence_transformer.py    # ✅ PyTorch-based (local)
│   │   ├── ollama.py                  # ✅ Ollama server
│   │   └── openai.py                  # ✅ OpenAI API
│   ├── vector_stores/
│   │   ├── memory.py                  # ✅ In-memory NumPy store
│   │   ├── chroma.py                  # ✅ ChromaDB (persistent/ephemeral)
│   │   ├── pinecone.py                # ✅ Pinecone cloud
│   │   └── pgvector.py                # ✅ PostgreSQL + pgvector
│   ├── sources/
│   │   ├── filesystem.py              # ✅ Local files with globs
│   │   ├── url.py                     # ✅ Web pages via HTTPClient
│   │   ├── object_store.py            # ✅ S3/GCS/Azure via axiompy.io.object
│   │   ├── database.py                # ✅ Database tables/queries
│   │   └── pdf.py                     # ✅ PDF text extraction (pypdf)
│   ├── llm/
│   │   └── reasoning_adapter.py       # Wraps axiompy.reasoning
│   └── mocks.py                       # Mock implementations for testing
│
├── applications/
│   └── cli.py                  # Command-line interface
│
├── factory.py                  # RAGServiceFactory
├── defaults.py                 # Default configurations
└── errors.py                   # Error hierarchy
```

### Data Flow

```
Ingestion:
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│   Chunker     │────▶│   Embedder   │────▶│ VectorStore │
│   (Source)  │     │               │     │              │     │             │
└─────────────┘     └───────────────┘     └──────────────┘     └─────────────┘

Query:
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query     │────▶│   Embedder    │────▶│   Search     │────▶│   Context   │
└─────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
                                                                      │
                                                                      ▼
                                          ┌──────────────┐     ┌─────────────┐
                                          │   Response   │◀────│ LLMProvider │
                                          └──────────────┘     └─────────────┘
```

## API Reference

### RAGServiceFactory

| Method | Description | Returns |
|--------|-------------|---------|
| `create(...)` | Create RAGService with specified components | `RAGService` |
| `create_mock(...)` | Create mock service for testing | `RAGService` |

### RAGService Methods

| Method | Description |
|--------|-------------|
| `ingest(documents)` | Ingest pre-loaded Document objects |
| `ingest_documents(paths)` | Load and ingest documents from paths |
| `ingest_text(text, source)` | Ingest raw text |
| `query(question, top_k, ...)` | Search + LLM generation |
| `search(question, top_k, ...)` | Search only (no LLM) |
| `delete_document(document_id)` | Remove document from store |
| `get_stats()` | Get service statistics |

### Type Enums

```python
class EmbedderType(Enum):
    FASTEMBED = "fastembed"              # ✅ Recommended: ONNX, fast, local
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # ✅ PyTorch-based
    OLLAMA = "ollama"                    # ✅ Via Ollama server
    OPENAI = "openai"                    # ✅ Cloud API
    MOCK = "mock"                        # ✅ Testing

class VectorStoreType(Enum):
    MEMORY = "memory"                    # ✅ In-memory NumPy
    CHROMA = "chroma"                    # ✅ ChromaDB (persistent)
    PINECONE = "pinecone"                # ✅ Cloud Pinecone
    PGVECTOR = "pgvector"                # ✅ PostgreSQL + pgvector
    MOCK = "mock"                        # ✅ Testing

class ChunkerType(Enum):
    FIXED_SIZE = "fixed_size"            # ✅ Character-based chunks
    SENTENCE = "sentence"                # ✅ Sentence-aware
    PARAGRAPH = "paragraph"              # ✅ Paragraph-aware
```

## Configuration

### EmbedderSettings

```python
@dataclass
class EmbedderSettings:
    model: Optional[str] = None           # Model name
    api_key: Optional[str] = None         # For cloud providers
    endpoint: Optional[str] = None        # Custom endpoint
    cache_dir: Optional[str] = None       # Local model cache
    batch_size: int = 100                 # Embedding batch size
    local_files_only: bool = True         # No network calls (default)
```

### VectorStoreSettings

```python
@dataclass
class VectorStoreSettings:
    collection_name: str = "default"
    persist_path: Optional[str] = None    # For persistent stores
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    database_url: Optional[str] = None
```

### ChunkerSettings

```python
@dataclass
class ChunkerSettings:
    chunk_size: int = 500                 # Characters per chunk
    chunk_overlap: int = 50               # Overlap between chunks
```

### LLMSettings

```python
@dataclass
class LLMSettings:
    model: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
```

## Examples

### Local Development with FastEmbed

```python
from axiompy.agents.rag import (
    RAGServiceFactory,
    EmbedderType,
    VectorStoreType,
    EmbedderSettings,
)

# FastEmbed runs entirely in-process (no server needed)
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.MEMORY,
    llm_provider="ollama",  # Requires local Ollama server
    embedder_settings=EmbedderSettings(
        model="BAAI/bge-small-en-v1.5",
        cache_dir="./models/fastembed",
        local_files_only=True,  # Default: no network calls
    ),
)

# Ingest markdown files
rag.ingest_documents(["./docs/"])

# Query
response = rag.query("How do I configure logging?")
print(response.answer)
```

### Using Ollama for Embeddings

```python
# Ollama can provide both embeddings AND LLM generation
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.OLLAMA,
    vector_store_type=VectorStoreType.MEMORY,
    llm_provider="ollama",
    embedder_settings=EmbedderSettings(
        model="nomic-embed-text",  # Ollama embedding model
        endpoint="http://localhost:11434",
    ),
)
```

### Production with OpenAI

```python
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.OPENAI,
    vector_store_type=VectorStoreType.CHROMA,
    llm_provider="openai",
    embedder_settings=EmbedderSettings(
        model="text-embedding-3-small",
        api_key="sk-...",
        local_files_only=False,  # Allow API calls
    ),
    vector_store_settings=VectorStoreSettings(
        persist_path="./rag_data",
        collection_name="knowledge_base",
    ),
)
```

### Production with ChromaDB (Persistent)

```python
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.CHROMA,
    llm_provider="ollama",
    embedder_settings=EmbedderSettings(
        model="BAAI/bge-small-en-v1.5",
        cache_dir="./models/fastembed",
    ),
    vector_store_settings=VectorStoreSettings(
        persist_path="./rag_data",
        collection_name="knowledge_base",
    ),
)
```

### PostgreSQL with pgvector

```python
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.PGVECTOR,
    llm_provider="openai",
    vector_store_settings=VectorStoreSettings(
        database_url="postgresql://user:pass@localhost:5432/vectors",
        collection_name="documents",
    ),
)
```

### Pinecone (Cloud)

```python
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.OPENAI,
    vector_store_type=VectorStoreType.PINECONE,
    llm_provider="openai",
    embedder_settings=EmbedderSettings(
        api_key="sk-...",
        local_files_only=False,
    ),
    vector_store_settings=VectorStoreSettings(
        api_key="pinecone-api-key",
        host="your-index.svc.region.pinecone.io",
        collection_name="knowledge_base",
    ),
)
```

### Testing with Mocks

```python
# All components mocked - no external dependencies
rag = RAGServiceFactory.create_mock(
    mock_response="This is a test answer",
    embedding_dimension=384,
)

rag.ingest_text("Test document content", source="test.txt")
response = rag.query("Test question")
assert response.answer == "This is a test answer"
```

### Document Sources

The RAG module provides multiple source adapters for loading documents from various locations.

#### FileSystemSource - Local Files

```python
from axiompy.agents.rag.adapters.sources import FileSystemSource

source = FileSystemSource(
    extensions=[".md", ".txt", ".py"],  # Only these file types
    encoding="utf-8",
)

# Load from multiple paths (supports globs)
docs = source.load_documents([
    "./docs/",
    "./src/**/*.py",
    "./README.md",
])

# Auto-ignores: node_modules, __pycache__, .git, venv, etc.
```

#### URLSource - Web Pages

```python
from axiompy.agents.rag.adapters.sources import URLSource

source = URLSource(timeout_secs=30, user_agent="MyBot/1.0")

# Load single URL (HTML converted to text)
doc = source.load_document("https://example.com/docs/guide")

# Load multiple URLs
docs = source.load_documents([
    "https://example.com/docs/intro",
    "https://example.com/docs/api",
])
```

#### ObjectStoreSource - Cloud Storage (S3, GCS, Azure)

```python
from axiompy.agents.rag.adapters.sources import SourceFactory
from axiompy.io.object import StorageSettings, StorageType

# AWS S3
s3_settings = StorageSettings(
    bucket="my-docs",
    region="us-east-1",
    access_key_id="...",
    secret_access_key="...",
)
source = SourceFactory.create_object_store(StorageType.S3, s3_settings)

# Google Cloud Storage
gcs_settings = StorageSettings(
    bucket="my-gcs-bucket",
    project_id="my-project",
    credentials_path="/path/to/creds.json",
)
source = SourceFactory.create_object_store(StorageType.GCS, gcs_settings)

# Azure Blob Storage
azure_settings = StorageSettings(
    bucket="my-container",  # Azure container name
    account_name="mystorageaccount",
    account_key="...",
)
source = SourceFactory.create_object_store(StorageType.AZURE, azure_settings)

# Load documents from prefix
docs = source.load_documents(["docs/"])
```

#### DatabaseSource - SQL Databases

```python
from axiompy.agents.rag.adapters.sources import SourceFactory
from axiompy.io.database import DatabaseType, DatabaseSettings

db_settings = DatabaseSettings(
    host="localhost",
    port=5432,
    database="mydb",
    username="user",
    password="pass",
)

source = SourceFactory.create_database(DatabaseType.POSTGRES, db_settings)

# Load from table
docs = source.load_from_table(
    table="articles",
    content_column="body",
    id_column="id",
    title_column="title",
)

# Or custom query
docs = source.load_from_query(
    query="SELECT id, title, body FROM posts WHERE status = 'published'",
    content_column="body",
    id_column="id",
)
```

#### PDFSource - PDF Files

```python
from axiompy.agents.rag.adapters.sources import PDFSource

# Requires: pip install pypdf
source = PDFSource(
    pages_as_documents=False,  # Entire PDF as one document
    include_metadata=True,     # Extract PDF metadata
)

# Load single PDF
doc = source.load_document("./report.pdf")

# Load directory of PDFs
docs = source.load_documents(["./reports/", "./papers/*.pdf"])

# Each page as separate document
page_source = PDFSource(pages_as_documents=True)
pages = page_source.load_document_pages("./large-manual.pdf")
```

#### SourceFactory - Unified Interface

```python
from axiompy.agents.rag.adapters.sources import (
    SourceFactory,
    SourceType,
    SourceSettings,
)

# FileSystem
fs_source = SourceFactory.create(SourceType.FILESYSTEM, SourceSettings())

# URL with custom settings
url_source = SourceFactory.create(
    SourceType.URL,
    SourceSettings(timeout_secs=60, user_agent="MyBot/1.0"),
)

# PDF
pdf_source = SourceFactory.create(
    SourceType.PDF,
    SourceSettings(pages_as_documents=True),
)

# Object Store (use dedicated method - supports S3, GCS, Azure)
obj_source = SourceFactory.create_object_store(StorageType.S3, storage_settings)

# Database (use dedicated method)
db_source = SourceFactory.create_database(DatabaseType.POSTGRES, db_settings)

# Mock (for testing)
mock_source = SourceFactory.create_mock()
```

## CLI Usage

The RAG CLI is available via Makefile targets:

```bash
# Ingest documents
make rag-ingest PATHS="./docs/" EMBEDDER=fastembed

# Query
make rag-query QUESTION="What is the return policy?" TOP_K=3

# Interactive chat
make rag-chat

# Show statistics
make rag-stats
```

Or run directly:

```bash
python -m axiompy.agents.rag.applications.cli ingest ./docs/ \
    --embedder fastembed \
    --model-cache-dir ./models/fastembed

python -m axiompy.agents.rag.applications.cli query "What is X?" \
    --top-k 5
```

## Error Handling

```python
from axiompy.agents.rag.errors import (
    RAGError,                  # Base error
    RAGConfigurationError,     # Invalid settings
    RAGIngestionError,         # Document loading/chunking failed
    RAGEmbeddingError,         # Embedding generation failed
    RAGVectorStoreError,       # Vector store operations failed
    RAGQueryError,             # Query/generation failed
)

try:
    rag = RAGServiceFactory.create(...)
    response = rag.query("Question")
except RAGConfigurationError as e:
    print(f"Configuration error: {e}")
except RAGEmbeddingError as e:
    print(f"Embedding failed: {e}")
except RAGQueryError as e:
    print(f"Query failed: {e}")
```

## Testing

```bash
# Run RAG tests
pytest tests/test_rag.py -v

# With coverage
pytest tests/test_rag.py -v --cov=axiompy.agents.rag
```

### Mock Testing Pattern

```python
import pytest
from axiompy.agents.rag import RAGServiceFactory

class TestMyRAGFeature:
    @pytest.fixture
    def rag_service(self):
        """Create mock RAG service for testing."""
        return RAGServiceFactory.create_mock(
            mock_response="Test answer",
            embedding_dimension=384,
        )
    
    def test_query_returns_answer(self, rag_service):
        rag_service.ingest_text("Some content", source="test.txt")
        response = rag_service.query("What is the content?")
        assert response.answer == "Test answer"
        assert len(response.sources) > 0
```

## Best Practices

### 1. Use Local-First Embeddings

```python
# Recommended: FastEmbed with local models
embedder_settings = EmbedderSettings(
    model="BAAI/bge-small-en-v1.5",
    cache_dir="./models/fastembed",
    local_files_only=True,  # Default
)
```

### 2. Pre-download Models

Download models before deployment to avoid network dependencies:

```
models/
├── fastembed/
│   └── bge-small-en-v1.5/         # Symlink or actual model
└── huggingface/
    └── hub/
        └── models--sentence-transformers--all-MiniLM-L6-v2/
```

### 3. Use Settings Dataclasses

```python
# Good: Explicit settings
settings = EmbedderSettings(
    model="BAAI/bge-small-en-v1.5",
    cache_dir="./models",
)
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    embedder_settings=settings,
    ...
)

# Bad: Hidden configuration
rag = create_rag_from_env()  # Don't do this
```

### 4. Configure Chunking for Your Content

```python
# Technical documentation: larger chunks
chunker_settings = ChunkerSettings(chunk_size=1000, chunk_overlap=100)

# Conversational content: smaller chunks  
chunker_settings = ChunkerSettings(chunk_size=300, chunk_overlap=50)
```

### 5. Use Mocks in Tests

```python
# Always use mocks in unit tests
rag = RAGServiceFactory.create_mock()

# Only use real embedders in integration tests
# and mark them appropriately
@pytest.mark.integration
def test_with_real_embedder():
    ...
```

## Extending the RAG Module

The RAG module uses protocol-based ports for easy extension. Add custom embedders, vector stores, or chunkers by implementing the appropriate protocol and registering with the factory.

### Adding a Custom Embedder

**Step 1: Create a Settings dataclass (if custom settings needed)**

```python
# axiompy/agents/rag/adapters/embedders/my_embedder.py
"""Custom embedder implementation."""

from dataclasses import dataclass
from typing import List, Optional

from axiompy.agents.rag.domain.ports import Embedder
from axiompy.agents.rag.errors import RAGConfigurationError
from axiompy.loggers import LoggerFactory
from axiompy.validators import ensure_not_empty, ensure_positive

logger = LoggerFactory.create_logger(__name__)


@dataclass
class MyEmbedderSettings:
    """
    Configuration for MyCustomEmbedder.
    
    Attributes:
        api_key: API key for MyService (required)
        model: Model name to use
        dimension: Embedding dimension
        batch_size: Texts per API call
    """
    api_key: str
    model: str = "my-embed-model"
    dimension: int = 768
    batch_size: int = 100
    
    def __post_init__(self) -> None:
        """Validate settings - let validators throw directly."""
        ensure_not_empty(self.api_key, "api_key is required")
        ensure_not_empty(self.model, "model is required")
        ensure_positive(self.dimension, "dimension must be positive")
        ensure_positive(self.batch_size, "batch_size must be positive")
```

**Step 2: Implement the `Embedder` protocol**

```python
class MyCustomEmbedder:
    """
    Custom embedder using MyService API.
    
    Implements the Embedder protocol for RAG integration.
    """
    
    def __init__(self, settings: MyEmbedderSettings) -> None:
        """
        Initialize custom embedder.
        
        Args:
            settings: Embedder configuration
        """
        self._settings = settings
        self._dimension = settings.dimension
        # Initialize your client here
        logger.info(f"MyCustomEmbedder initialized: model={settings.model}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self._call_api(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
    
    def _call_api(self, text: str) -> List[float]:
        """Call embedding API (implement your logic)."""
        pass
```

**Step 3: Add to the `EmbedderType` enum**

```python
# axiompy/agents/rag/factory.py
class EmbedderType(str, Enum):
    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"
    OPENAI = "openai"
    MY_CUSTOM = "my_custom"  # Add your type
    MOCK = "mock"
```

**Step 4: Register in `EmbedderFactory`**

```python
# axiompy/agents/rag/adapters/embedders/__init__.py
from axiompy.agents.rag.factory import EmbedderType, EmbedderSettings

class EmbedderFactory:
    @staticmethod
    def create(
        embedder_type: EmbedderType,
        settings: EmbedderSettings,  # Factory receives standard settings
    ) -> Embedder:
        match embedder_type:
            # ... existing cases ...
            
            case EmbedderType.MY_CUSTOM:
                from axiompy.agents.rag.adapters.embedders.my_embedder import (
                    MyCustomEmbedder,
                    MyEmbedderSettings,
                )
                # Map standard settings to custom settings
                custom_settings = MyEmbedderSettings(
                    api_key=settings.api_key,
                    model=settings.model or "my-embed-model",
                )
                return MyCustomEmbedder(custom_settings)
            
            case _:
                raise RAGConfigurationError(f"Unknown embedder: {embedder_type}")
```

**Step 5: Use your embedder**

```python
from axiompy.agents.rag import RAGServiceFactory, EmbedderType, EmbedderSettings

# Settings dataclass - explicit configuration
embedder_settings = EmbedderSettings(
    api_key="my-api-key",
    model="my-embed-model",
)

rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.MY_CUSTOM,
    vector_store_type=VectorStoreType.MEMORY,
    llm_provider="ollama",
    embedder_settings=embedder_settings,
)
```

### Adding a Custom Vector Store

**Step 1: Create a Settings dataclass**

```python
# axiompy/agents/rag/adapters/vector_stores/my_store.py
"""Custom vector store implementation."""

from dataclasses import dataclass
from typing import List

from axiompy.agents.rag.domain.models import DocumentChunk, SearchResult
from axiompy.agents.rag.domain.ports import VectorStore
from axiompy.loggers import LoggerFactory
from axiompy.validators import ensure_not_empty, ensure_positive

logger = LoggerFactory.create_logger(__name__)


@dataclass
class MyVectorStoreSettings:
    """
    Configuration for MyVectorStore.
    
    Attributes:
        connection_string: Database connection URL (required)
        collection_name: Name of the collection/table
        embedding_dimension: Vector dimension for index
    """
    connection_string: str
    collection_name: str = "documents"
    embedding_dimension: int = 384
    
    def __post_init__(self) -> None:
        """Validate settings - let validators throw directly."""
        ensure_not_empty(self.connection_string, "connection_string is required")
        ensure_not_empty(self.collection_name, "collection_name is required")
        ensure_positive(self.embedding_dimension, "embedding_dimension must be positive")
```

**Step 2: Implement the `VectorStore` protocol**

```python
class MyVectorStore:
    """
    Custom vector store using MyDatabase.
    
    Implements the VectorStore protocol for RAG integration.
    """
    
    def __init__(self, settings: MyVectorStoreSettings) -> None:
        """
        Initialize vector store.
        
        Args:
            settings: Vector store configuration
        """
        self._settings = settings
        # Initialize your database connection
        logger.info(f"MyVectorStore initialized: collection={settings.collection_name}")
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the store."""
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            self._insert(chunk)
        logger.debug(f"Added {len(chunks)} chunks to store")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search for similar chunks."""
        results = self._similarity_search(query_embedding, top_k)
        return [
            SearchResult(chunk=chunk, score=score)
            for chunk, score in results
            if score >= min_score
        ]
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return self._delete_by_document(document_id)
    
    def clear(self) -> None:
        """Clear all chunks from the store."""
        self._truncate()
    
    def chunk_count(self) -> int:
        """Return total number of chunks."""
        return self._count()
    
    # Implement your database methods
    def _insert(self, chunk: DocumentChunk) -> None: ...
    def _similarity_search(self, embedding: List[float], k: int) -> List: ...
    def _delete_by_document(self, doc_id: str) -> int: ...
    def _truncate(self) -> None: ...
    def _count(self) -> int: ...
```

**Step 3: Add to `VectorStoreType` enum and factory**

```python
# axiompy/agents/rag/factory.py
class VectorStoreType(str, Enum):
    MEMORY = "memory"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"
    MY_STORE = "my_store"  # Add your type
    MOCK = "mock"

# axiompy/agents/rag/adapters/vector_stores/__init__.py
class VectorStoreFactory:
    @staticmethod
    def create(
        store_type: VectorStoreType,
        settings: VectorStoreSettings,  # Standard settings from factory.py
    ) -> VectorStore:
        match store_type:
            # ... existing cases ...
            
            case VectorStoreType.MY_STORE:
                from axiompy.agents.rag.adapters.vector_stores.my_store import (
                    MyVectorStore,
                    MyVectorStoreSettings,
                )
                # Map standard settings to custom settings
                custom_settings = MyVectorStoreSettings(
                    connection_string=settings.database_url,
                    collection_name=settings.collection_name,
                )
                return MyVectorStore(custom_settings)
```

**Step 4: Use your vector store**

```python
from axiompy.agents.rag import RAGServiceFactory, VectorStoreType, VectorStoreSettings

vector_store_settings = VectorStoreSettings(
    database_url="mydb://localhost:5000/vectors",
    collection_name="knowledge_base",
)

rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.MY_STORE,
    llm_provider="ollama",
    vector_store_settings=vector_store_settings,
)
```

### Adding a Custom Chunker

**Step 1: Create a Settings dataclass**

```python
# axiompy/agents/rag/domain/chunker.py (or new file)
"""Custom chunker implementation."""

from dataclasses import dataclass
from typing import List

from axiompy.agents.rag.domain.models import Document, DocumentChunk
from axiompy.agents.rag.domain.ports import DocumentChunker
from axiompy.loggers import LoggerFactory
from axiompy.validators import ensure_positive, ensure_in_range

logger = LoggerFactory.create_logger(__name__)


@dataclass
class SemanticChunkerSettings:
    """
    Configuration for SemanticChunker.
    
    Attributes:
        target_chunk_size: Target characters per chunk
        similarity_threshold: Threshold for topic boundary detection (0.0-1.0)
        min_chunk_size: Minimum chunk size to avoid tiny chunks
    """
    target_chunk_size: int = 500
    similarity_threshold: float = 0.5
    min_chunk_size: int = 100
    
    def __post_init__(self) -> None:
        """Validate settings - let validators throw directly."""
        ensure_positive(self.target_chunk_size, "target_chunk_size must be positive")
        ensure_in_range(
            self.similarity_threshold, 0.0, 1.0,
            "similarity_threshold must be between 0.0 and 1.0"
        )
        ensure_positive(self.min_chunk_size, "min_chunk_size must be positive")
```

**Step 2: Implement the `DocumentChunker` protocol**

```python
class SemanticChunker:
    """
    Semantic chunker that splits on topic boundaries.
    
    Uses sentence embeddings to detect topic shifts.
    """
    
    def __init__(self, settings: SemanticChunkerSettings) -> None:
        """
        Initialize semantic chunker.
        
        Args:
            settings: Chunker configuration
        """
        self._settings = settings
        logger.debug(f"SemanticChunker: target={settings.target_chunk_size}")
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split document into semantic chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        boundaries = self._find_boundaries(document.content)
        
        for i, (start, end) in enumerate(boundaries):
            chunk_content = document.content[start:end]
            chunks.append(
                DocumentChunk(
                    id=f"{document.id}:chunk_{i}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                    metadata={"chunker": "semantic"},
                )
            )
        
        return chunks
    
    def _find_boundaries(self, text: str) -> List[tuple]:
        """Find semantic boundaries in text."""
        # Your boundary detection logic
        pass
```

**Step 3: Add to `ChunkerType` enum and factory**

```python
# axiompy/agents/rag/factory.py
class ChunkerType(str, Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"  # Add your type

# axiompy/agents/rag/domain/chunker.py
class ChunkerFactory:
    @staticmethod
    def create(
        chunker_type: ChunkerType,
        settings: ChunkerSettings,  # Standard settings from factory.py
    ) -> DocumentChunker:
        match chunker_type:
            # ... existing cases ...
            
            case ChunkerType.SEMANTIC:
                # Map standard settings to custom settings
                custom_settings = SemanticChunkerSettings(
                    target_chunk_size=settings.chunk_size,
                )
                return SemanticChunker(custom_settings)
```

**Step 4: Use your chunker**

```python
from axiompy.agents.rag import RAGServiceFactory, ChunkerType, ChunkerSettings

chunker_settings = ChunkerSettings(
    chunk_size=500,
    chunk_overlap=50,
)

rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.MEMORY,
    llm_provider="ollama",
    chunker_type=ChunkerType.SEMANTIC,
    chunker_settings=chunker_settings,
)
```

### Extension Checklist

When adding a new adapter:

- [ ] **Create Settings dataclass** with `__post_init__` validation (use `axiompy.validators`)
- [ ] **Implement the protocol** (`Embedder`, `VectorStore`, `DocumentChunker`, `DocumentSource`)
- [ ] **Constructor takes Settings** - not individual parameters
- [ ] **Add enum value** to the type enum (e.g., `EmbedderType.MY_CUSTOM`)
- [ ] **Register in sub-factory** using `match/case` - map standard settings to custom settings
- [ ] **Write unit tests** with mocked dependencies
- [ ] **Add documentation** to the README
- [ ] Update `__all__` exports if needed

**Key axiompy conventions:**
- Factories accept **Settings dataclasses**, not individual parameters
- Let validators throw directly in `__post_init__` - don't catch and re-raise
- Use `match/case` for type dispatch (not if/elif chains)
- Follow composition over inheritance

### Protocol Reference

```python
# axiompy/agents/rag/domain/ports.py

class Embedder(Protocol):
    """Protocol for text embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
    @property
    def dimension(self) -> int: ...

class VectorStore(Protocol):
    """Protocol for vector storage backends."""
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None: ...
    def search(self, query_embedding: List[float], top_k: int, min_score: float) -> List[SearchResult]: ...
    def delete_document(self, document_id: str) -> int: ...
    def clear(self) -> None: ...
    def chunk_count(self) -> int: ...

class DocumentChunker(Protocol):
    """Protocol for document chunking strategies."""
    
    def chunk(self, document: Document) -> List[DocumentChunk]: ...

class DocumentSource(Protocol):
    """Protocol for document loading."""
    
    def load_document(self, path: str) -> Document: ...
    def load_documents(self, paths: List[str]) -> List[Document]: ...
```

## Supported Models

### FastEmbed (Recommended)

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | ~130MB | Fast, good quality |
| `BAAI/bge-base-en-v1.5` | 768 | ~440MB | Better quality |

### sentence-transformers

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | ~90MB | Classic, well-tested |
| `all-mpnet-base-v2` | 768 | ~420MB | Higher quality |

## Dependencies

Core:
```
numpy>=1.24.0,<2.0       # Required for torch compatibility
```

Optional (install as needed):
```bash
# Embedders
pip install fastembed              # For EmbedderType.FASTEMBED (ONNX, lightweight)
pip install sentence-transformers  # For EmbedderType.SENTENCE_TRANSFORMERS
pip install 'transformers<4.50'    # Required for sentence-transformers with torch<2.6
pip install openai                 # For EmbedderType.OPENAI

# Vector Stores
pip install chromadb               # For VectorStoreType.CHROMA
pip install pinecone-client        # For VectorStoreType.PINECONE
pip install psycopg2-binary        # For VectorStoreType.PGVECTOR

# Document Sources
pip install pypdf                  # For PDFSource (PDF text extraction)
# Note: S3Source and DatabaseSource use axiompy.io.object and axiompy.io.database
```

### Dependency Notes

**For both local embedders to work:**
```bash
# FastEmbed works out of the box
pip install fastembed

# sentence-transformers requires specific versions due to security checks
pip install sentence-transformers 'transformers<4.50' 'numpy<2'
```

**Why these constraints?**
- `numpy<2` - PyTorch 2.2.x was compiled with NumPy 1.x
- `transformers<4.50` - Newer versions require torch>=2.6 for `.bin` model files
- FastEmbed uses ONNX and doesn't have these constraints

**Vector Store Dependencies:**
- `chromadb` - For persistent and ephemeral vector stores with SQLite backend
- `pinecone-client` - For Pinecone cloud vector database
- `psycopg2-binary` - For PostgreSQL with pgvector extension

---

> **See Also**: 
> - [`axiompy.reasoning`](../../reasoning/README.md) - LLM integration
> - [`axiompy.io.http`](../../io/README.md) - HTTP client for API calls
> - [`tests/test_rag.py`](../../../tests/test_rag.py) - Test examples

