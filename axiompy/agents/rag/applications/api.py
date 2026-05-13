"""RAG API Adapter - REST API for RAG service.

Uses axiompy.servers.ServerFactory for server creation.

Provides REST endpoints for document ingestion and querying.

Usage:
    # Start API server
    python -m axiompy.agents.rag.applications.api

    # Or via Makefile
    make rag-api

Endpoints:
    GET  /health          Health check
    POST /ingest          Ingest documents
    POST /ingest/text     Ingest raw text
    POST /query           Query with LLM generation
    POST /search          Search only (no LLM)
    GET  /stats           Index statistics
    DELETE /documents/{id} Delete a document
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from axiompy.loggers import LoggerFactory
from axiompy.servers import ServerFactory, ServerSettings, ServerType

from ..domain.service import RAGService
from ..factory import (
    ChunkerSettings,
    EmbedderSettings,
    EmbedderType,
    LLMSettings,
    RAGServiceFactory,
    VectorStoreSettings,
    VectorStoreType,
)

logger = LoggerFactory.create_logger(__name__)


# Default configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_EMBEDDER = EmbedderType.SENTENCE_TRANSFORMERS
DEFAULT_VECTOR_STORE = VectorStoreType.MEMORY
DEFAULT_LLM = "ollama"
DEFAULT_MODEL = "mistral"
DEFAULT_PERSIST_PATH = "./rag_data"
DEFAULT_COLLECTION = "rag_documents"


@dataclass
class APISettings:
    """Settings for RAG API service."""

    # Server settings
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # RAG configuration
    embedder_type: EmbedderType = DEFAULT_EMBEDDER
    vector_store_type: VectorStoreType = DEFAULT_VECTOR_STORE
    llm_provider: str = DEFAULT_LLM

    # Embedder settings
    embed_model: Optional[str] = None
    model_cache_dir: Optional[str] = None

    # Vector store settings
    persist_path: Optional[str] = None
    collection_name: str = DEFAULT_COLLECTION

    # LLM settings
    llm_model: str = DEFAULT_MODEL
    ollama_host: str = "http://localhost:11434"

    def __post_init__(self):
        # For chroma, use default persist path if not specified
        if self.vector_store_type == VectorStoreType.CHROMA and not self.persist_path:
            self.persist_path = DEFAULT_PERSIST_PATH

    @classmethod
    def from_env(cls) -> "APISettings":
        """Create settings from environment variables."""
        import os

        # Map string to enum
        embedder_map = {
            "sentence_transformers": EmbedderType.SENTENCE_TRANSFORMERS,
            "fastembed": EmbedderType.FASTEMBED,
            "ollama": EmbedderType.OLLAMA,
            "openai": EmbedderType.OPENAI,
            "mock": EmbedderType.MOCK,
        }

        store_map = {
            "memory": VectorStoreType.MEMORY,
            "chroma": VectorStoreType.CHROMA,
            "pinecone": VectorStoreType.PINECONE,
            "pgvector": VectorStoreType.PGVECTOR,
            "mock": VectorStoreType.MOCK,
        }

        embedder_str = os.environ.get("RAG_EMBEDDER", "sentence_transformers")
        store_str = os.environ.get("RAG_STORE", "memory")

        return cls(
            host=os.environ.get("RAG_HOST", DEFAULT_HOST),
            port=int(os.environ.get("RAG_PORT", str(DEFAULT_PORT))),
            embedder_type=embedder_map.get(embedder_str, DEFAULT_EMBEDDER),
            vector_store_type=store_map.get(store_str, DEFAULT_VECTOR_STORE),
            llm_provider=os.environ.get("RAG_LLM", DEFAULT_LLM),
            embed_model=os.environ.get("RAG_EMBED_MODEL"),
            model_cache_dir=os.environ.get("RAG_MODEL_CACHE_DIR"),
            persist_path=os.environ.get("RAG_PERSIST_PATH"),
            collection_name=os.environ.get("RAG_COLLECTION", DEFAULT_COLLECTION),
            llm_model=os.environ.get("RAG_LLM_MODEL", DEFAULT_MODEL),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )


class RAGAPIService:
    """
    RAG API service using axiompy.servers.ServerFactory.

    Provides REST endpoints for RAG operations following axiompy patterns.

    Example:
        # Create with explicit settings
        settings = APISettings(
            vector_store_type=VectorStoreType.CHROMA,
            persist_path="./rag_data",
        )
        service = RAGAPIService.create(settings)
        service.run()

        # Create from environment
        service = RAGAPIService.create()  # Uses APISettings.from_env()
        service.run()
    """

    def __init__(
        self,
        server,
        settings: APISettings,
        rag_service: RAGService,
    ):
        """
        Initialize RAG API service.

        Args:
            server: Server from ServerFactory
            settings: API configuration
            rag_service: RAGService instance
        """
        self._server = server
        self._settings = settings
        self._rag_service = rag_service
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register API routes."""

        @self._server.route("/health", methods=["GET"])
        def health() -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "embedder": self._settings.embedder_type.value,
                "vector_store": self._settings.vector_store_type.value,
                "llm_provider": self._settings.llm_provider,
                "persistent": self._settings.vector_store_type == VectorStoreType.CHROMA,
                "persist_path": self._settings.persist_path,
            }

        @self._server.route("/ingest", methods=["POST"])
        def ingest(data: dict) -> Dict[str, Any]:
            """
            Ingest documents from file paths.

            Request body:
                {
                    "paths": ["./docs/", "./README.md"],
                    "glob_pattern": "**/*.md"  // optional
                }

            Returns:
                {"status": "ok", "documents_processed": N, "chunks_created": M}
            """
            paths = data.get("paths", [])
            if not paths:
                return {"status": "error", "message": "No paths provided"}

            try:
                # Use FileSystemSource for proper document loading
                from ..adapters.sources import FileSystemSource

                source = FileSystemSource()
                documents = source.load_documents(paths)

                if not documents:
                    return {"status": "error", "message": "No documents found"}

                stats = self._rag_service.ingest(documents)

                logger.info(
                    f"Ingested {stats['chunks_created']} chunks "
                    f"from {stats['documents_processed']} documents"
                )

                return {
                    "status": "ok",
                    "documents_processed": stats["documents_processed"],
                    "chunks_created": stats["chunks_created"],
                }
            except Exception as e:
                logger.exception(f"Ingest failed: {e}")
                return {"status": "error", "message": str(e)}

        @self._server.route("/ingest/text", methods=["POST"])
        def ingest_text(data: dict) -> Dict[str, Any]:
            """
            Ingest raw text.

            Request body:
                {
                    "text": "Content to ingest...",
                    "source": "document_name",  // optional
                    "metadata": {}  // optional
                }

            Returns:
                {"status": "ok", "chunks_created": N}
            """
            text = data.get("text")
            if not text:
                return {"status": "error", "message": "No text provided"}

            try:
                source = data.get("source", "inline")
                metadata = data.get("metadata")

                chunks_created = self._rag_service.ingest_text(
                    text=text,
                    source=source,
                    metadata=metadata,
                )

                logger.info(f"Ingested text: {chunks_created} chunks from '{source}'")

                return {"status": "ok", "chunks_created": chunks_created}
            except Exception as e:
                logger.exception(f"Ingest text failed: {e}")
                return {"status": "error", "message": str(e)}

        @self._server.route("/query", methods=["POST"])
        def query(data: dict) -> Dict[str, Any]:
            """
            Query with LLM generation.

            Request body:
                {
                    "question": "What is...",
                    "top_k": 5,  // optional
                    "min_score": 0.0,  // optional
                    "temperature": 0.7,  // optional
                    "max_tokens": 500  // optional
                }

            Returns:
                {
                    "status": "ok",
                    "answer": "...",
                    "sources": [{"document_id": "...", "content": "...", "score": 0.95}]
                }
            """
            question = data.get("question")
            if not question:
                return {"status": "error", "message": "No question provided"}

            try:
                response = self._rag_service.query(
                    question=question,
                    top_k=data.get("top_k", 5),
                    min_score=data.get("min_score", 0.0),
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens", 500),
                )

                sources = [
                    {
                        "document_id": s.chunk.document_id,
                        "content": s.chunk.content[:500],  # Truncate for response
                        "score": s.score,
                    }
                    for s in response.sources
                ]

                return {
                    "status": "ok",
                    "answer": response.answer,
                    "sources": sources,
                }
            except Exception as e:
                logger.exception(f"Query failed: {e}")
                return {"status": "error", "message": str(e)}

        @self._server.route("/search", methods=["POST"])
        def search(data: dict) -> Dict[str, Any]:
            """
            Search only (no LLM generation).

            Request body:
                {
                    "question": "What is...",
                    "top_k": 5,  // optional
                    "min_score": 0.0  // optional
                }

            Returns:
                {
                    "status": "ok",
                    "results": [{"document_id": "...", "content": "...", "score": 0.95}]
                }
            """
            question = data.get("question")
            if not question:
                return {"status": "error", "message": "No question provided"}

            try:
                results = self._rag_service.search(
                    question=question,
                    top_k=data.get("top_k", 5),
                    min_score=data.get("min_score", 0.0),
                )

                formatted = [
                    {
                        "document_id": r.chunk.document_id,
                        "chunk_id": r.chunk.id,
                        "content": r.chunk.content,
                        "score": r.score,
                        "metadata": r.chunk.metadata,
                    }
                    for r in results
                ]

                return {"status": "ok", "results": formatted}
            except Exception as e:
                logger.exception(f"Search failed: {e}")
                return {"status": "error", "message": str(e)}

        @self._server.route("/stats", methods=["GET"])
        def stats() -> Dict[str, Any]:
            """Get index statistics."""
            try:
                service_stats = self._rag_service.get_stats()
                return {
                    "status": "ok",
                    "stats": service_stats,
                    "config": {
                        "embedder": self._settings.embedder_type.value,
                        "vector_store": self._settings.vector_store_type.value,
                        "persistent": self._settings.vector_store_type == VectorStoreType.CHROMA,
                        "persist_path": self._settings.persist_path,
                    },
                }
            except Exception as e:
                logger.exception(f"Stats failed: {e}")
                return {"status": "error", "message": str(e)}

        @self._server.route("/documents/{document_id}", methods=["DELETE"])
        def delete_document(document_id: str) -> Dict[str, Any]:
            """Delete a document by ID."""
            try:
                deleted = self._rag_service.delete_document(document_id)
                return {
                    "status": "ok",
                    "deleted": deleted,
                    "document_id": document_id,
                }
            except Exception as e:
                logger.exception(f"Delete failed: {e}")
                return {"status": "error", "message": str(e)}

    def run(self) -> None:
        """Start the API server."""
        logger.info(f"Starting RAG API on {self._settings.host}:{self._settings.port}")
        logger.info(f"Vector store: {self._settings.vector_store_type.value}")
        if self._settings.persist_path:
            logger.info(f"Persist path: {self._settings.persist_path}")
        self._server.run(host=self._settings.host, port=self._settings.port)

    def get_app(self):
        """Get the underlying app (for uvicorn)."""
        return self._server.get_app()

    @staticmethod
    def _create_rag_service(settings: APISettings) -> RAGService:
        """Create RAGService from API settings."""
        embedder_settings = EmbedderSettings(
            model=settings.embed_model,
            cache_dir=settings.model_cache_dir,
        )

        vector_store_settings = VectorStoreSettings(
            collection_name=settings.collection_name,
            persist_path=settings.persist_path,
        )

        llm_endpoint = None
        if settings.llm_provider == "ollama":
            llm_endpoint = f"{settings.ollama_host}/api/generate"

        llm_settings = LLMSettings(
            model=settings.llm_model,
            endpoint=llm_endpoint,
        )

        return RAGServiceFactory.create(
            embedder_type=settings.embedder_type,
            vector_store_type=settings.vector_store_type,
            llm_provider=settings.llm_provider,
            embedder_settings=embedder_settings,
            vector_store_settings=vector_store_settings,
            llm_settings=llm_settings,
        )

    @staticmethod
    def create(settings: Optional[APISettings] = None) -> "RAGAPIService":
        """
        Create RAGAPIService from settings.

        Args:
            settings: API settings (uses env vars if None)

        Returns:
            Configured RAGAPIService
        """
        if settings is None:
            settings = APISettings.from_env()

        # Create server using axiompy ServerFactory
        server_settings = ServerSettings(
            host=settings.host,
            port=settings.port,
            extra_params={
                "title": "RAG API",
                "description": "REST API for Retrieval-Augmented Generation",
                "version": "1.0.0",
            },
        )
        server = ServerFactory.create(ServerType.FASTAPI, server_settings)

        # Create RAG service
        rag_service = RAGAPIService._create_rag_service(settings)

        return RAGAPIService(
            server=server,
            settings=settings,
            rag_service=rag_service,
        )

    @staticmethod
    def create_mock() -> "RAGAPIService":
        """Create mock service for testing."""
        settings = APISettings(
            embedder_type=EmbedderType.MOCK,
            vector_store_type=VectorStoreType.MOCK,
            llm_provider="mock",
        )

        server_settings = ServerSettings(host="127.0.0.1", port=8080)
        server = ServerFactory.create(ServerType.FASTAPI, server_settings)

        rag_service = RAGServiceFactory.create_mock()

        return RAGAPIService(
            server=server,
            settings=settings,
            rag_service=rag_service,
        )


if __name__ == "__main__":
    RAGAPIService.create().run()
