"""Events package exposing Kafka event producer."""

from services.events.producer import EventProducer, get_producer

__all__ = ["EventProducer", "get_producer"]
