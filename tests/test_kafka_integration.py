"""Integration tests for Kafka EventProducer and document consumer."""

import asyncio
import uuid
import fitz
import pytest
from starlette.testclient import TestClient

from config import get_settings
from services.api.main import app
from services.events.producer import EventProducer
from services.retrieval.vector_retriever import get_vector_store
from services.workers.document_worker import process_document_event


def create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), text, fontsize=11)
    content = doc.tobytes()
    doc.close()
    return content


@pytest.mark.asyncio
async def test_kafka_producer_publish_direct():
    """Verify EventProducer can publish an event to local Kafka."""
    settings = get_settings()
    if not settings.KAFKA_ENABLED:
        pytest.skip("Kafka is not enabled in settings.")

    producer = EventProducer()
    await producer.start()
    assert producer.is_connected

    test_payload = {
        "event_id": str(uuid.uuid4()),
        "test": "sample_message",
    }
    success = await producer.publish("chat.events", test_payload, key="test-key")
    assert success is True

    await producer.stop()


@pytest.mark.asyncio
async def test_process_document_event():
    """Verify process_document_event correctly parses, chunks, and indexes a PDF."""
    store = get_vector_store()
    store.clear()

    doc_id = str(uuid.uuid4())
    user_id = "test_kafka_user"
    pdf_bytes = create_sample_pdf("Kafka event-driven document processing in MemoryGraph.")

    temp_path = f"./data/uploads/{doc_id}.pdf"
    import os
    os.makedirs("./data/uploads", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    event_data = {
        "document_id": doc_id,
        "user_id": user_id,
        "filename": "kafka_sample.pdf",
        "stored_path": temp_path,
        "size_bytes": len(pdf_bytes),
    }

    result = await process_document_event(event_data)
    assert result is True

    # Verify document exists in vector store
    doc = store.get_document(doc_id, user_id)
    assert doc is not None
    assert doc["filename"] == "kafka_sample.pdf"
    assert doc["chunk_count"] >= 1
    assert doc["status"] == "READY"

    # Cleanup
    store.clear()
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_health_endpoint_reports_kafka():
    """Verify /health endpoint reports Kafka, Redis, and MySQL configuration."""
    settings = get_settings()
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "kafka" in data
        assert data["kafka"]["enabled"] == settings.KAFKA_ENABLED
        assert "redis" in data
        assert data["redis"]["enabled"] == settings.REDIS_ENABLED
        assert "mysql" in data
        assert data["mysql"]["enabled"] == settings.MYSQL_ENABLED
