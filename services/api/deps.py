"""FastAPI dependency injection module."""

from typing import Optional
from fastapi import Depends, Header

from config import Settings, get_settings
from services.embeddings.embedder import Embedder, get_embedder
from services.events.producer import EventProducer, get_producer
from services.llm import LLMProvider, get_llm_provider
from services.retrieval.vector_retriever import InMemoryVectorStore, get_vector_store


def get_current_user(x_user_id: Optional[str] = Header(default="default_user", alias="X-User-Id")) -> str:
    """Extract authenticated user ID from X-User-Id header with fallback to default_user."""
    if not x_user_id or not x_user_id.strip():
        return "default_user"
    return x_user_id.strip()


def get_llm(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Inject LLM provider instance."""
    return get_llm_provider(settings)


def get_embedder_dep(settings: Settings = Depends(get_settings)) -> Embedder:
    """Inject singleton Embedder instance."""
    return get_embedder(
        model_name=settings.EMBEDDING_MODEL,
        embedding_dim=settings.EMBEDDING_DIM,
    )


def get_store_dep() -> InMemoryVectorStore:
    """Inject singleton in-memory vector store."""
    return get_vector_store()


def get_producer_dep() -> EventProducer:
    """Inject singleton Kafka EventProducer instance."""
    return get_producer()
