"""Ollama Embedder Adapter.

Generates embeddings using a local Ollama server.

Features:
- Local embedding generation (no external API calls)
- Supports any model that Ollama supports
- Uses nomic-embed-text by default
- Uses axiompy HTTPClient for API calls

Example:
    from axiompy.agents.io.embeddings import OllamaEmbedder

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        host="http://localhost:11434",
    )

    embedding = embedder.embed_text("Hello, world!")
    embeddings = embedder.embed_texts(["Hello", "World"])

Common Models:
- nomic-embed-text: 768 dimensions, general purpose
- mxbai-embed-large: 1024 dimensions, high quality
- all-minilm: 384 dimensions, lightweight

Requirements:
- Ollama server running locally (ollama serve)
- Model pulled (ollama pull nomic-embed-text)
"""

from typing import List

from axiompy.agents.io.errors import RAGEmbeddingError
from axiompy.io.http import HTTPClientFactory
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


# Model dimension mapping (common models)
MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"


class OllamaEmbedder:
    """
    Ollama-based embedder.

    Uses a local Ollama server to generate embeddings.

    Attributes:
        model: Model name being used
        embedding_dimension: Dimension of embeddings produced

    Example:
        embedder = OllamaEmbedder()
        embedding = embedder.embed_text("Hello")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout_secs: int = 60,
    ) -> None:
        """
        Initialize Ollama embedder.

        Args:
            model: Model to use (default: nomic-embed-text)
            host: Ollama server URL (default: http://localhost:11434)
            timeout_secs: Request timeout in seconds

        Raises:
            RAGEmbeddingError: If cannot connect to Ollama
        """
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout_secs
        self._endpoint = f"{self._host}/api/embeddings"

        # Determine embedding dimension
        self._dimension = MODEL_DIMENSIONS.get(model)
        if self._dimension is None:
            # For unknown models, we'll determine dimension on first call
            logger.warning(f"Unknown model '{model}', dimension will be determined on first call")
            self._dimension = 0  # Will be set on first embedding

        # Create HTTP client
        self._client = HTTPClientFactory.create(timeout_secs=timeout_secs).add_header(
            "Content-Type", "application/json"
        )

        logger.info(
            f"OllamaEmbedder initialized: model={model}, host={host}, dimension={self._dimension}"
        )

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        return self._dimension

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            RAGEmbeddingError: If API call fails
        """
        if not text or not text.strip():
            raise RAGEmbeddingError("Cannot embed empty text")

        try:
            response = self._client.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "prompt": text,
                },
            )

            if response.status_code != 200:
                error_msg = response.text
                if response.status_code == 404:
                    raise RAGEmbeddingError(
                        f"Model '{self._model}' not found. Run: ollama pull {self._model}"
                    )
                raise RAGEmbeddingError(f"Ollama API error ({response.status_code}): {error_msg}")

            data = response.json()
            embedding = data.get("embedding", [])

            # Update dimension if not set
            if self._dimension == 0 and embedding:
                self._dimension = len(embedding)
                logger.info(f"Determined embedding dimension: {self._dimension}")

            return embedding

        except RAGEmbeddingError:
            raise
        except Exception as e:
            # Check for connection errors
            if "Connection" in str(e) or "refused" in str(e).lower():
                raise RAGEmbeddingError(
                    f"Cannot connect to Ollama at {self._host}. "
                    "Ensure Ollama is running: ollama serve"
                ) from e
            raise RAGEmbeddingError(f"Ollama embedding failed: {e}") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Note: Ollama API doesn't support batch embedding,
        so this calls embed_text for each text.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RAGEmbeddingError: If API call fails
        """
        if not texts:
            return []

        # Filter and validate texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise RAGEmbeddingError("No valid texts to embed")

        # Embed each text (Ollama doesn't support batch)
        embeddings = []
        for text in valid_texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return embeddings

    def __repr__(self) -> str:
        return (
            f"OllamaEmbedder(model={self._model}, host={self._host}, dimension={self._dimension})"
        )
