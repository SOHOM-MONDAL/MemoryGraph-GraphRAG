"""SQLAlchemy data models for MySQL storage in MemoryGraph."""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DocumentRecord(Base):
    """Metadata and processing state for uploaded documents."""

    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    size_bytes = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(32), default="PROCESSING", index=True)
    stage = Column(String(32), default="UPLOADED")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ChatMessageRecord(Base):
    """Permanent audit log of chat interactions."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class GraphNodeRecord(Base):
    """Knowledge Graph entity node persisted in MySQL."""

    __tablename__ = "graph_nodes"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    name = Column(String(128), nullable=False, index=True)
    entity_type = Column(String(64), default="CONCEPT", index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class GraphEdgeRecord(Base):
    """Knowledge Graph relationship edge persisted in MySQL."""

    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    source = Column(String(128), nullable=False, index=True)
    target = Column(String(128), nullable=False, index=True)
    relation = Column(String(128), nullable=False, index=True)
    document_id = Column(String(64), nullable=False, index=True)
    chunk_id = Column(String(64), nullable=True)
    weight = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
