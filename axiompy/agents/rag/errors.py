"""RAG Error Hierarchy.

Defines exception hierarchy for RAG operations following axiompy patterns.

Example:
    try:
        response = rag.query("What is X?")
    except RAGEmbeddingError as e:
        logger.error(f"Embedding failed: {e}")
    except RAGVectorStoreError as e:
        logger.error(f"Vector store error: {e}")
    except RAGError as e:
        logger.error(f"RAG error: {e}")
"""


class RAGError(Exception):
    """Base exception for all RAG errors."""

    pass


class RAGConfigurationError(RAGError):
    """Invalid configuration or settings."""

    pass


class RAGIngestionError(RAGError):
    """Error during document ingestion."""

    pass


class RAGEmbeddingError(RAGError):
    """Error generating embeddings."""

    pass


class RAGVectorStoreError(RAGError):
    """Error interacting with vector store."""

    pass


class RAGQueryError(RAGError):
    """Error during query execution."""

    pass


class RAGLLMError(RAGError):
    """Error during LLM generation."""

    pass
