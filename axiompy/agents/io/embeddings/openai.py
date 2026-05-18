"""OpenAI Embedder Adapter.

Generates embeddings using OpenAI's text-embedding API.

Features:
- Uses text-embedding-3-small by default (cost-effective, good quality)
- Supports text-embedding-3-large for higher quality
- Batch embedding support
- Uses axiompy HTTPClient for API calls

Example:
    from axiompy.agents.io.embeddings import OpenAIEmbedder

    embedder = OpenAIEmbedder(
        api_key="sk-...",
        model="text-embedding-3-small",
    )

    embedding = embedder.embed_text("Hello, world!")
    embeddings = embedder.embed_texts(["Hello", "World"])

Models:
- text-embedding-3-small: 1536 dimensions, 62,500 tokens/min, $0.02/1M tokens
- text-embedding-3-large: 3072 dimensions, 62,500 tokens/min, $0.13/1M tokens
- text-embedding-ada-002: 1536 dimensions (legacy)
"""

from typing import List, Optional

from axiompy.agents.io.errors import RAGEmbeddingError
from axiompy.io.http import HTTPClientFactory
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


# Model dimension mapping
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbedder:
    """
    OpenAI API-based embedder.

    Uses OpenAI's text-embedding API to generate embeddings.

    Attributes:
        model: Model name being used
        embedding_dimension: Dimension of embeddings produced

    Example:
        embedder = OpenAIEmbedder(api_key="sk-...")
        embedding = embedder.embed_text("Hello")
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: Optional[str] = None,
        timeout_secs: int = 30,
    ) -> None:
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: text-embedding-3-small)
            endpoint: Custom API endpoint (default: OpenAI API)
            timeout_secs: Request timeout in seconds

        Raises:
            RAGEmbeddingError: If API key is not provided
        """
        if not api_key:
            raise RAGEmbeddingError("OpenAI API key is required")

        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint or OPENAI_API_URL
        self._timeout = timeout_secs

        # Determine embedding dimension
        self._dimension = MODEL_DIMENSIONS.get(model)
        if self._dimension is None:
            # For custom/unknown models, we'll determine dimension on first call
            logger.warning(f"Unknown model '{model}', dimension will be determined on first call")
            self._dimension = 0  # Will be set on first embedding

        # Create HTTP client
        self._client = (
            HTTPClientFactory.create(timeout_secs=timeout_secs)
            .bearer_token(api_key)
            .add_header("Content-Type", "application/json")
        )

        logger.info(f"OpenAIEmbedder initialized: model={model}, dimension={self._dimension}")

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

        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

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
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            raise RAGEmbeddingError("No valid texts to embed")

        try:
            response = self._client.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "input": valid_texts,
                },
            )

            if response.status_code != 200:
                error_msg = response.text
                raise RAGEmbeddingError(f"OpenAI API error ({response.status_code}): {error_msg}")

            data = response.json()

            # Extract embeddings from response
            embeddings = []
            for item in sorted(data["data"], key=lambda x: x["index"]):
                embeddings.append(item["embedding"])

            # Update dimension if not set
            if self._dimension == 0 and embeddings:
                self._dimension = len(embeddings[0])
                logger.info(f"Determined embedding dimension: {self._dimension}")

            return embeddings

        except RAGEmbeddingError:
            raise
        except Exception as e:
            raise RAGEmbeddingError(f"OpenAI embedding failed: {e}") from e

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self._model}, dimension={self._dimension})"
