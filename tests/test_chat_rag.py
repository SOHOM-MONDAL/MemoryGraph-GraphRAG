"""Tests for Chat RAG retrieval and source attribution."""

import fitz
import pytest
from starlette.testclient import TestClient

from services.api.main import app
from services.retrieval.vector_retriever import get_vector_store


def create_test_pdf(text: str) -> bytes:
    """Create in-memory PDF bytes for API tests."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), text, fontsize=11)
    content = doc.tobytes()
    doc.close()
    return content


from services.api.deps import get_llm
from services.llm.base import LLMProvider


class MockLLM(LLMProvider):
    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        return f"Mock answer generated based on prompt of length {len(prompt)}."

    def health(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Reset vector store and override LLM for tests."""
    store = get_vector_store()
    store.clear()
    app.dependency_overrides[get_llm] = lambda: MockLLM()
    yield
    store.clear()
    app.dependency_overrides.clear()


def test_chat_without_documents_returns_empty_sources():
    client = TestClient(app)

    res = client.post(
        "/chat",
        headers={"X-User-Id": "alice"},
        json={"session_id": "sess_1", "message": "Hello, how are you today?"},
    )

    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert data["sources"] == []
    assert data["latency_ms"] >= 0
    assert "request_id" in data


def test_chat_with_document_populates_sources():
    client = TestClient(app)
    pdf_text = "Project Nebula is an autonomous space exploration drone built with titanium alloys."
    pdf_bytes = create_test_pdf(pdf_text)

    # 1. Upload PDF
    upload_res = client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("nebula.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201

    # 2. Query chat
    chat_res = client.post(
        "/chat",
        headers={"X-User-Id": "alice"},
        json={"session_id": "sess_1", "message": "What is Project Nebula built with?"},
    )

    assert chat_res.status_code == 200
    data = chat_res.json()
    assert len(data["sources"]) > 0

    first_source = data["sources"][0]
    assert first_source["filename"] == "nebula.pdf"
    assert first_source["page_number"] == 1
    assert "nebula.pdf" in first_source["chunk_id"] or "p1" in first_source["chunk_id"]
    assert "Project Nebula" in first_source["snippet"]


def test_chat_user_isolation_cannot_access_other_users_sources():
    client = TestClient(app)
    pdf_text = "Top secret formula: 42-ALPHA-OMEGA."
    pdf_bytes = create_test_pdf(pdf_text)

    # Alice uploads secret document
    client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("secret.pdf", pdf_bytes, "application/pdf")},
    )

    # Bob queries about top secret formula
    bob_chat = client.post(
        "/chat",
        headers={"X-User-Id": "bob"},
        json={"session_id": "bob_sess", "message": "What is the top secret formula?"},
    )

    assert bob_chat.status_code == 200
    bob_data = bob_chat.json()
    # Bob has no documents; sources must be empty
    assert bob_data["sources"] == []
