"""Tests for embedding service."""

import numpy as np
from services.embeddings.embedder import Embedder


def test_embedder_dimension_and_normalization():
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=384)
    vec = embedder.embed_text("Graph retrieval augmented generation with Neo4j.")

    assert len(vec) == 384
    # Check that vector is normalized (unit norm ~ 1.0)
    norm = np.linalg.norm(np.array(vec))
    assert pytest_approx_equal(norm, 1.0, tolerance=1e-3)


def test_embedder_batch():
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=384)
    texts = [
        "First document paragraph.",
        "Second document paragraph with different content.",
    ]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def pytest_approx_equal(val: float, target: float, tolerance: float = 1e-3) -> bool:
    return abs(val - target) <= tolerance
