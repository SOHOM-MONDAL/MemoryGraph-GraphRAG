"""Singleton embedding service using sentence-transformers."""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Embedder using SentenceTransformer or fast lightweight HashingVectorizer for memory-constrained clouds."""

    _instance: Optional["Embedder"] = None
    _model = None

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
        low_memory: bool = False,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.low_memory = low_memory
        self._vectorizer = None

    def _get_vectorizer(self):
        """Lazy instantiate fast semantic HashingVectorizer (zero RAM footprint)."""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import HashingVectorizer
            self._vectorizer = HashingVectorizer(
                n_features=self.embedding_dim,
                norm="l2",
                alternate_sign=False,
                ngram_range=(1, 2),
            )
        return self._vectorizer

    def _load_model(self) -> None:
        """Load the SentenceTransformer model lazily if not already loaded."""
        if self._model is None and not self.low_memory:
            try:
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model on CPU (single-threaded for memory safety): %s", self.model_name)
                self._model = SentenceTransformer(self.model_name, device="cpu")
                logger.info("Embedding model loaded successfully.")
            except (MemoryError, Exception) as e:
                logger.warning(
                    "SentenceTransformer could not be loaded (%s). Operating with lightweight deterministic embeddings.",
                    str(e),
                )
                self._model = None

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized embedding vector for single text."""
        if not text:
            return [0.0] * self.embedding_dim

        if self.low_memory:
            vec = self._get_vectorizer().transform([text]).toarray()[0]
            return [float(x) for x in vec]

        self._load_model()
        if self._model is not None:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return [float(x) for x in embedding]

        # Graceful fallback in case model failed to load
        return self._fallback_embedding(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for batch of texts."""
        if not texts:
            return []

        if self.low_memory:
            matrix = self._get_vectorizer().transform(texts).toarray()
            return [[float(x) for x in row] for row in matrix]

        self._load_model()
        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [[float(x) for x in emb] for emb in embeddings]

        return [self._fallback_embedding(t) for t in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Generate a fast normalized pseudo-embedding using hashing vectorizer."""
        vec = self._get_vectorizer().transform([text]).toarray()[0]
        return [float(x) for x in vec]

    def health(self) -> bool:
        """Check if embedder model is loaded or ready in low-memory mode."""
        return self.low_memory or (self._model is not None)


_cached_embedder: Optional[Embedder] = None


def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    embedding_dim: int = 384,
    low_memory: bool = False,
) -> Embedder:
    """Return singleton Embedder instance."""
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = Embedder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            low_memory=low_memory,
        )
    return _cached_embedder
