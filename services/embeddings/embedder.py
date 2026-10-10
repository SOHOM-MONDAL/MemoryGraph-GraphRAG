"""Singleton embedding service using sentence-transformers."""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Embedder using BAAI/bge-small-en-v1.5 or configured sentence-transformers model."""

    _instance: Optional["Embedder"] = None
    _model = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", embedding_dim: int = 384) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model lazily if not already loaded."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load SentenceTransformer model %s: %s", self.model_name, str(e))
                self._model = None

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized embedding vector for single text."""
        if not text:
            return [0.0] * self.embedding_dim

        if self._model is not None:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return [float(x) for x in embedding]

        # Graceful fallback in case model failed to load
        logger.warning("Embedding model unavailable; generating deterministic fallback vector.")
        return self._fallback_embedding(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for batch of texts."""
        if not texts:
            return []

        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [[float(x) for x in emb] for emb in embeddings]

        return [self._fallback_embedding(t) for t in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Generate a deterministic normalized pseudo-embedding for testing or offline mode."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31 - 1))
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def health(self) -> bool:
        """Check if embedder model is loaded and ready."""
        return self._model is not None


_cached_embedder: Optional[Embedder] = None


def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5", embedding_dim: int = 384) -> Embedder:
    """Return singleton Embedder instance."""
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = Embedder(model_name=model_name, embedding_dim=embedding_dim)
    return _cached_embedder
