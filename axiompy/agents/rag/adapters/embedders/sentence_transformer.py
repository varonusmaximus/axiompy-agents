"""Sentence Transformer Embedder.

Local, in-process embedding using sentence-transformers library.

This is the industry standard for local embeddings, supporting hundreds
of pre-trained models from HuggingFace.

Requirements:
    pip install sentence-transformers

Recommended models:
    - all-MiniLM-L6-v2: Fast, 384 dimensions, good quality (~80MB)
    - all-mpnet-base-v2: Better quality, 768 dimensions (~420MB)
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual (~420MB)

Example:
    from axiompy.agents.rag.adapters.embedders import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    embedding = embedder.embed_text("Hello, world!")
    embeddings = embedder.embed_texts(["Hello", "World"])
"""

from typing import List, Optional

from axiompy.agents.rag.errors import RAGEmbeddingError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# Default model - small, fast, good quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """
    Local embedder using sentence-transformers library.

    Runs entirely in-process without external API calls.
    Supports CPU and GPU (if available).

    Attributes:
        model_name: HuggingFace model name
        embedding_dimension: Dimension of output embeddings

    Example:
        embedder = SentenceTransformerEmbedder()
        vectors = embedder.embed_texts(["doc1", "doc2"])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        """
        Initialize sentence transformer embedder.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            normalize_embeddings: Whether to L2-normalize embeddings (recommended)
            cache_dir: Directory for cached models (default: ~/.cache/huggingface)
            local_files_only: If True, only use local cached models (no network)

        Raises:
            RAGEmbeddingError: If sentence-transformers not installed or model fails
        """
        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._device = device
        self._cache_dir = cache_dir

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading sentence-transformer model: {model_name} (local_only={local_files_only})"
            )
            self._model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_dir,
                local_files_only=local_files_only,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded model {model_name}: dimension={self._dimension}, "
                f"device={self._model.device}"
            )

        except ImportError as e:
            raise RAGEmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise RAGEmbeddingError(f"Failed to load model '{model_name}': {e}") from e

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RAGEmbeddingError: If embedding fails
        """
        if not texts:
            return []

        try:
            logger.debug(f"Embedding {len(texts)} texts")
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            raise RAGEmbeddingError(f"Embedding failed: {e}") from e

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(model={self._model_name}, dim={self._dimension})"
