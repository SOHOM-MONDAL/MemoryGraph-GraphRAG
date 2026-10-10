"""Integration tests for GraphRAG: Triplet Extraction, Graph Storage, Multi-hop Traversal, and Hybrid Retrieval."""

import pytest
from starlette.testclient import TestClient

from services.api.main import app
from services.graph.extractor import extract_triplets
from services.graph.graph_store import GraphStore, get_graph_store
from services.graph.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from services.graph.models import GraphTriplet
from services.pdf.chunker import PDFChunk
from services.retrieval.vector_retriever import get_vector_store


def test_extractor_extracts_triplets():
    """Verify entity-relation triplet extractor identifies relationships."""
    sample_text = (
        "FastAPI is a modern web framework for Python. "
        "MemoryGraph uses Kafka for event streaming and Redis for caching."
    )
    triplets = extract_triplets(sample_text, document_id="doc-test", chunk_id="chk-1")
    assert len(triplets) >= 1
    # Check triplet structure
    first = triplets[0]
    assert isinstance(first, GraphTriplet)
    assert len(first.subject) > 0
    assert len(first.predicate) > 0
    assert len(first.object) > 0


def test_graph_store_multihop_traversal():
    """Verify multi-hop graph BFS traversal across connected entities."""
    store = GraphStore()
    user_id = "test_graph_user"
    store.clear(user_id)

    # Build a 2-hop chain: MemoryGraph -> Kafka -> DocumentWorker -> Embeddings
    triplets = [
        GraphTriplet(subject="MemoryGraph", predicate="publishes events to", object="Kafka", document_id="doc1"),
        GraphTriplet(subject="Kafka", predicate="delivers tasks to", object="DocumentWorker", document_id="doc1"),
        GraphTriplet(subject="DocumentWorker", predicate="computes", object="Embeddings", document_id="doc1"),
    ]
    store.add_triplets(triplets, user_id=user_id)

    stats = store.get_stats(user_id)
    assert stats["node_count"] == 4
    assert stats["edge_count"] == 3

    # 1-hop traversal starting from MemoryGraph
    hop1 = store.traverse_subgraph(["MemoryGraph"], user_id=user_id, max_hops=1)
    hop1_objects = [t.object for t in hop1]
    assert "Kafka" in hop1_objects
    assert "DocumentWorker" not in hop1_objects

    # 2-hop traversal starting from MemoryGraph
    hop2 = store.traverse_subgraph(["MemoryGraph"], user_id=user_id, max_hops=2)
    hop2_objects = [t.object for t in hop2]
    assert "Kafka" in hop2_objects
    assert "DocumentWorker" in hop2_objects

    # Cleanup
    store.clear(user_id)


def test_hybrid_retriever_fuses_graph_and_vector():
    """Verify HybridRetriever fuses vector similarity chunks and knowledge graph subgraphs."""
    vstore = get_vector_store()
    gstore = get_graph_store()
    user_id = "hybrid_test_user"

    vstore.clear()
    gstore.clear(user_id)

    # 1. Add vector chunk
    embedder = retriever = get_hybrid_retriever().embedder
    chunk_text = "Redis serves as high-speed memory for chat sessions."
    chunk = PDFChunk(
        chunk_id="chk-001",
        document_id="doc-001",
        user_id=user_id,
        page_number=1,
        chunk_index=0,
        text=chunk_text,
        embedding=embedder.embed_text(chunk_text),
    )
    vstore.add_document(
        document_id="doc-001",
        user_id=user_id,
        filename="redis_guide.pdf",
        stored_path="/tmp/redis.pdf",
        size_bytes=500,
        chunks=[chunk],
    )

    # 2. Add knowledge graph relationship
    gstore.add_triplet(
        GraphTriplet(
            subject="Redis",
            predicate="stores",
            object="Session Memory",
            document_id="doc-001",
        ),
        user_id=user_id,
    )

    retriever = HybridRetriever(vector_store=vstore, graph_store=gstore)
    result = retriever.retrieve(query="What does Redis do with session memory?", user_id=user_id)

    # Check fusion
    assert len(result.chunks) >= 1
    assert len(result.triplets) >= 1
    assert "Redis" in result.seed_entities
    assert "Relevant Knowledge Graph Relationships" in result.fused_context
    assert "(Redis) -[stores]-> (Session Memory)" in result.fused_context

    # Cleanup
    vstore.clear()
    gstore.clear(user_id)


def test_graph_api_endpoints():
    """Verify /documents/graph/stats and /documents/graph/triplets endpoints."""
    gstore = get_graph_store()
    user_id = "api_graph_user"
    gstore.clear(user_id)

    gstore.add_triplet(
        GraphTriplet(subject="PyMuPDF", predicate="extracts text from", object="PDF", document_id="doc-api"),
        user_id=user_id,
    )

    with TestClient(app) as client:
        # Check stats
        res_stats = client.get("/documents/graph/stats", headers={"X-User-Id": user_id})
        assert res_stats.status_code == 200
        data_stats = res_stats.json()
        assert data_stats["node_count"] == 2
        assert data_stats["edge_count"] == 1

        # Check triplets
        res_trip = client.get("/documents/graph/triplets", headers={"X-User-Id": user_id})
        assert res_trip.status_code == 200
        data_trip = res_trip.json()
        assert len(data_trip) == 1
        assert data_trip[0]["subject"] == "PyMuPDF"
        assert data_trip[0]["predicate"] == "extracts text from"
        assert data_trip[0]["object"] == "PDF"

    gstore.clear(user_id)
