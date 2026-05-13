"""FastEmbed Embedder.

Lightweight, in-process embedding using fastembed (ONNX-based).

Faster and lighter than sentence-transformers because:
- Uses ONNX runtime instead of PyTorch
- Smaller memory footprint
- Optimized for CPU inference

Requirements:
    pip install fastembed

Recommended models:
    - BAAI/bge-small-en-v1.5: Fast, 384 dimensions, good quality (~130MB)
    - BAAI/bge-base-en-v1.5: Better quality, 768 dimensions (~440MB)
    - sentence-transformers/all-MiniLM-L6-v2: Classic model (~90MB)

Example:
    from axiompy.agents.rag.adapters.embedders import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
    embedding = embedder.embed_text("Hello, world!")
    embeddings = embedder.embed_texts(["Hello", "World"])
"""

from typing import List, Optional

from axiompy.agents.rag.errors import RAGEmbeddingError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# Default model - small, fast, good quality
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Model dimension mapping (fastembed doesn't expose this easily)
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


class FastEmbedEmbedder:
    """
    Lightweight local embedder using fastembed (ONNX).

    Runs entirely in-process without external API calls.
    Faster and lighter than PyTorch-based alternatives.

    Attributes:
        model_name: Model identifier
        embedding_dimension: Dimension of output embeddings

    Example:
        embedder = FastEmbedEmbedder()
        vectors = embedder.embed_texts(["doc1", "doc2"])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        local_files_only: bool = False,
    ) -> None:
        """
        Initialize FastEmbed embedder.

        Args:
            model_name: Model identifier (default: BAAI/bge-small-en-v1.5)
            cache_dir: Optional directory to cache downloaded models
            threads: Number of threads for ONNX runtime (None for auto)
            local_files_only: If True, only use local cached models (no network)

        Raises:
            RAGEmbeddingError: If fastembed not installed or model fails
        """
        self._model_name = model_name
        self._cache_dir = cache_dir

        try:
            from fastembed import TextEmbedding

            logger.info(f"Loading fastembed model: {model_name} (local_only={local_files_only})")

            # Build kwargs
            kwargs = {"local_files_only": local_files_only}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            if threads:
                kwargs["threads"] = threads

            self._model = TextEmbedding(model_name=model_name, **kwargs)

            # Get dimension from mapping or detect it
            if model_name in MODEL_DIMENSIONS:
                self._dimension = MODEL_DIMENSIONS[model_name]
            else:
                # Detect dimension by embedding a test text
                test_emb = list(self._model.embed(["test"]))[0]
                self._dimension = len(test_emb)

            logger.info(f"Loaded model {model_name}: dimension={self._dimension}")

        except ImportError as e:
            raise RAGEmbeddingError(
                "fastembed not installed. Install with: pip install fastembed"
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
            # fastembed returns a generator, convert to list
            embeddings = list(self._model.embed(texts))
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
        return f"FastEmbedEmbedder(model={self._model_name}, dim={self._dimension})"
