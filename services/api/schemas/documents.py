"""Pydantic schemas for Document endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful PDF upload and processing."""
    document_id: str
    filename: str
    chunk_count: int
    status: str
    message: str


class DocumentItem(BaseModel):
    """Document summary item in list."""
    document_id: str
    filename: str
    chunk_count: int
    status: str
    size_bytes: int
    created_at: str


class DocumentDetail(BaseModel):
    """Detailed document information including chunk snippets."""
    document_id: str
    filename: str
    chunk_count: int
    status: str
    size_bytes: int
    created_at: str
    chunks: List[dict] = Field(default_factory=list)


class DocumentStatusResponse(BaseModel):
    """Document processing status response."""
    document_id: str
    status: str
    stage: str
    error: Optional[str] = None
