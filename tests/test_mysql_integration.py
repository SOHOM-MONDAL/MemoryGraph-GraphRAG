"""Tests for MySQL database repository and models."""

import pytest
from services.database.connection import is_database_connected
from services.database.repository import ChatRepository, DocumentRepository


def test_database_connected_when_disabled():
    """Verify is_database_connected returns False when MySQL is disabled."""
    assert is_database_connected() is False


def test_document_repository_graceful_fallback_when_disabled():
    """Verify DocumentRepository methods return safe fallbacks without exceptions."""
    # When MySQL is disabled, repository operations safely return None or []
    doc = DocumentRepository.create_or_update(
        document_id="test-doc-fallback",
        user_id="user-1",
        filename="test.pdf",
        stored_path="/tmp/test.pdf",
    )
    assert doc is None

    docs = DocumentRepository.list_by_user("user-1")
    assert docs == []

    updated = DocumentRepository.update_status("test-doc-fallback", "READY", "READY")
    assert updated is False

    deleted = DocumentRepository.delete("test-doc-fallback", "user-1")
    assert deleted is False


def test_chat_repository_graceful_fallback_when_disabled():
    """Verify ChatRepository methods return safe fallbacks without exceptions."""
    saved = ChatRepository.save_message("session-1", "user-1", "user", "Hello database")
    assert saved is False
