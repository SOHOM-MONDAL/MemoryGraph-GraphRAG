"""Tests for Document API endpoints and user isolation."""

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


@pytest.fixture(autouse=True)
def clean_store():
    """Clear in-memory store before each test run."""
    store = get_vector_store()
    store.clear()
    yield
    store.clear()


def test_upload_valid_pdf_endpoint():
    client = TestClient(app)
    pdf_bytes = create_test_pdf("MemoryGraph is a persistent graph memory system.")

    response = client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("architecture.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "architecture.pdf"
    assert data["chunk_count"] >= 1
    assert data["status"] == "READY"
    assert "document_id" in data


def test_upload_invalid_file_rejected():
    client = TestClient(app)
    fake_txt = b"This is not a PDF file."

    response = client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("notes.txt", fake_txt, "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF documents" in response.json()["detail"]


def test_list_and_get_document():
    client = TestClient(app)
    pdf_bytes = create_test_pdf("Deep learning neural networks and embeddings.")

    upload_res = client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("neural_nets.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    # List documents
    list_res = client.get("/documents", headers={"X-User-Id": "alice"})
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) == 1
    assert docs[0]["document_id"] == doc_id

    # Get single document detail
    detail_res = client.get(f"/documents/{doc_id}", headers={"X-User-Id": "alice"})
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["document_id"] == doc_id
    assert len(detail["chunks"]) >= 1


def test_user_isolation_document_access():
    client = TestClient(app)
    pdf_bytes = create_test_pdf("Confidential document belonging to Alice.")

    upload_res = client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("confidential.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    # User Bob tries to list documents -> should see 0 documents
    bob_list = client.get("/documents", headers={"X-User-Id": "bob"})
    assert bob_list.status_code == 200
    assert len(bob_list.json()) == 0

    # User Bob tries to get Alice's document -> 404 Not Found
    bob_get = client.get(f"/documents/{doc_id}", headers={"X-User-Id": "bob"})
    assert bob_get.status_code == 404

    # User Bob tries to delete Alice's document -> 404 Not Found
    bob_del = client.delete(f"/documents/{doc_id}", headers={"X-User-Id": "bob"})
    assert bob_del.status_code == 404


def test_delete_document():
    client = TestClient(app)
    pdf_bytes = create_test_pdf("Temporary document to be deleted.")

    upload_res = client.post(
        "/documents/upload",
        headers={"X-User-Id": "alice"},
        files={"file": ("temp.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    del_res = client.delete(f"/documents/{doc_id}", headers={"X-User-Id": "alice"})
    assert del_res.status_code == 200

    # Verify document is gone
    get_res = client.get(f"/documents/{doc_id}", headers={"X-User-Id": "alice"})
    assert get_res.status_code == 404
