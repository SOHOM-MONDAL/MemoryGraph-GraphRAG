"""Redis client and caching service for MemoryGraph."""

import json
import logging
from typing import Any, Dict, List, Optional
import redis

from config import get_settings

logger = logging.getLogger("memorygraph.cache")


class RedisService:
    """Manages Redis connection, session history, and cached states."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Optional[redis.Redis] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Redis connection if enabled."""
        if not self.settings.REDIS_ENABLED:
            logger.info("Redis is disabled in settings.")
            self._client = None
            return

        try:
            self._client = redis.Redis.from_url(
                self.settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            # Test connectivity
            self._client.ping()
            logger.info("Connected to Redis at %s", self.settings.REDIS_URL)
        except Exception as e:
            logger.warning("Failed to connect to Redis (%s): %s. Operating in cache-disabled fallback.", self.settings.REDIS_URL, str(e))
            self._client = None

    def is_connected(self) -> bool:
        """Check if Redis connection is active and responsive."""
        if not self.settings.REDIS_ENABLED or self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def add_chat_message(self, session_id: str, role: str, content: str) -> None:
        """Store chat message in Redis conversation list.
        
        Args:
            session_id: Unique chat session ID.
            role: 'user' or 'assistant'.
            content: Message string.
        """
        if not self.is_connected() or not self._client:
            return

        key = f"chat:{session_id}:history"
        item = json.dumps({"role": role, "content": content})
        try:
            self._client.rpush(key, item)
            # Trim list to recent history limit
            limit = self.settings.REDIS_HISTORY_LIMIT
            if limit > 0:
                self._client.ltrim(key, -limit, -1)
            # 7-day expiration for idle sessions
            self._client.expire(key, 604800)
        except Exception as e:
            logger.error("Failed to append chat message to Redis key %s: %s", key, str(e))

    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve recent chat history for a session.
        
        Args:
            session_id: Chat session ID.
            limit: Optional override for max messages.
            
        Returns:
            List of {"role": str, "content": str} dictionaries.
        """
        if not self.is_connected() or not self._client:
            return []

        key = f"chat:{session_id}:history"
        try:
            max_msgs = limit or self.settings.REDIS_HISTORY_LIMIT
            raw_items = self._client.lrange(key, -max_msgs, -1)
            history = []
            for item in raw_items:
                try:
                    history.append(json.loads(item))
                except json.JSONDecodeError:
                    continue
            return history
        except Exception as e:
            logger.error("Failed to read chat history from Redis key %s: %s", key, str(e))
            return []

    def set_cached_document_status(self, document_id: str, status_payload: dict, ttl: int = 86400) -> None:
        """Cache document processing status for fast retrieval."""
        if not self.is_connected() or not self._client:
            return
        key = f"doc:{document_id}:status"
        try:
            self._client.setex(key, ttl, json.dumps(status_payload))
        except Exception as e:
            logger.error("Failed to cache document status for %s: %s", document_id, str(e))

    def get_cached_document_status(self, document_id: str) -> Optional[dict]:
        """Fetch cached document status if present."""
        if not self.is_connected() or not self._client:
            return None
        key = f"doc:{document_id}:status"
        try:
            data = self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error("Failed to read cached document status for %s: %s", document_id, str(e))
            return None


_redis_service_instance: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Return singleton RedisService instance."""
    global _redis_service_instance
    if _redis_service_instance is None:
        _redis_service_instance = RedisService()
    return _redis_service_instance
