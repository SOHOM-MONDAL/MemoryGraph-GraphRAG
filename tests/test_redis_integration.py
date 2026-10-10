"""Tests for Redis cache and conversation memory service."""

import pytest
from services.cache.redis_client import RedisService, get_redis_service


def test_redis_service_singleton():
    """Verify get_redis_service returns singleton instance."""
    s1 = get_redis_service()
    s2 = get_redis_service()
    assert s1 is s2


def test_redis_service_graceful_fallback_when_disabled(monkeypatch):
    """Ensure operations do not crash when Redis is disabled or unreachable."""
    service = RedisService()
    # Force client to None (simulating offline/disabled)
    service._client = None

    assert service.is_connected() is False

    # Should not raise any exceptions
    service.add_chat_message("test_sess", "user", "Hello Redis")
    history = service.get_chat_history("test_sess")
    assert history == []

    service.set_cached_document_status("doc123", {"status": "READY"})
    cached = service.get_cached_document_status("doc123")
    assert cached is None
