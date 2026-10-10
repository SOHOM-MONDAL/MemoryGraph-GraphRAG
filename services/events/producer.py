"""Kafka event producer module for MemoryGraph."""

import json
import logging
from typing import Any, Dict, Optional
from aiokafka import AIOKafkaProducer

from config import get_settings

logger = logging.getLogger("memorygraph.events")


class EventProducer:
    """AIOKafka Producer wrapper supporting asynchronous event publishing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        """Start Kafka producer if Kafka is enabled."""
        if not self.settings.KAFKA_ENABLED:
            logger.info("Kafka is disabled in settings. EventProducer will not start.")
            return

        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            await self._producer.start()
            logger.info(
                "Kafka EventProducer connected to %s",
                self.settings.KAFKA_BOOTSTRAP_SERVERS,
            )
        except Exception as e:
            logger.error("Failed to start Kafka EventProducer: %s", str(e))
            self._producer = None

    async def stop(self) -> None:
        """Stop Kafka producer."""
        if self._producer:
            try:
                await self._producer.stop()
                logger.info("Kafka EventProducer stopped.")
            except Exception as e:
                logger.warning("Error stopping Kafka EventProducer: %s", str(e))
            finally:
                self._producer = None

    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
    ) -> bool:
        """Publish a message to a Kafka topic.

        Args:
            topic: Target Kafka topic name.
            message: Dictionary payload to serialize as JSON.
            key: Optional message key for partitioning.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._producer:
            logger.debug(
                "EventProducer is inactive; skipping publish to topic '%s'",
                topic,
            )
            return False

        try:
            await self._producer.send_and_wait(topic, value=message, key=key)
            logger.info("Published event to '%s' (key=%s)", topic, key)
            return True
        except Exception as e:
            logger.error("Failed to publish event to '%s': %s", topic, str(e))
            return False

    @property
    def is_connected(self) -> bool:
        """Check if producer is currently running."""
        return self._producer is not None


_producer_instance: Optional[EventProducer] = None


def get_producer() -> EventProducer:
    """Return singleton EventProducer instance."""
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = EventProducer()
    return _producer_instance
