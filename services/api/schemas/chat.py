"""Pydantic schemas for Chat endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat query payload."""
    session_id: str = Field(default="default_session", description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User question or input")


class SourceItem(BaseModel):
    """Retrieved document chunk source metadata."""
    filename: str
    page_number: int
    chunk_id: str
    snippet: str


class ChatResponse(BaseModel):
    """Chat endpoint response with grounded sources and graph knowledge relationships."""
    response: str
    sources: List[SourceItem] = Field(default_factory=list)
    graph_triplets: List[str] = Field(default_factory=list, description="Knowledge graph facts used in retrieval")
    request_id: str
    latency_ms: float
