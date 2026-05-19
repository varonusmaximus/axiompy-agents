"""Agent IO error hierarchy.

Defines exception hierarchy for agents.io operations following axiompy patterns.

Example:
    try:
        embedder.embed(["text"])
    except AgentIOEmbeddingError as e:
        logger.error(f"Embedding failed: {e}")
    except AgentIOVectorStoreError as e:
        logger.error(f"Vector store error: {e}")
    except AgentIOError as e:
        logger.error(f"IO error: {e}")
"""


class AgentIOError(Exception):
    """Base exception for all agents.io errors."""

    pass


class AgentIOConfigurationError(AgentIOError):
    """Invalid configuration or settings."""

    pass


class AgentIOIngestionError(AgentIOError):
    """Error during document ingestion."""

    pass


class AgentIOEmbeddingError(AgentIOError):
    """Error generating embeddings."""

    pass


class AgentIOVectorStoreError(AgentIOError):
    """Error interacting with vector store."""

    pass


class AgentIOQueryError(AgentIOError):
    """Error during query execution."""

    pass


class AgentIOLLMError(AgentIOError):
    """Error during LLM generation."""

    pass
