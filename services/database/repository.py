"""Repository module for storing and querying documents and chat messages in MySQL."""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from services.database.connection import get_db_session
from services.database.models import ChatMessageRecord, DocumentRecord

logger = logging.getLogger("memorygraph.repository")


class DocumentRepository:
    """Handles CRUD operations for DocumentRecord in MySQL."""

    @staticmethod
    def create_or_update(
        document_id: str,
        user_id: str,
        filename: str,
        stored_path: str,
        size_bytes: int = 0,
        chunk_count: int = 0,
        status: str = "PROCESSING",
        stage: str = "UPLOADED",
        error: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """Save or update document metadata in MySQL."""
        for session in get_db_session():
            if not session:
                return None
            try:
                record = session.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
                if not record:
                    record = DocumentRecord(
                        id=document_id,
                        user_id=user_id,
                        filename=filename,
                        stored_path=stored_path,
                        size_bytes=size_bytes,
                        chunk_count=chunk_count,
                        status=status,
                        stage=stage,
                        error_message=error,
                    )
                    session.add(record)
                else:
                    record.chunk_count = chunk_count
                    record.status = status
                    record.stage = stage
                    record.error_message = error
                session.commit()
                return record
            except Exception as e:
                logger.error("Failed to save DocumentRecord in MySQL: %s", str(e))
                return None

    @staticmethod
    def update_status(
        document_id: str,
        status: str,
        stage: str,
        chunk_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update processing stage and status in MySQL."""
        for session in get_db_session():
            if not session:
                return False
            try:
                record = session.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
                if record:
                    record.status = status
                    record.stage = stage
                    if chunk_count is not None:
                        record.chunk_count = chunk_count
                    if error is not None:
                        record.error_message = error
                    session.commit()
                    return True
            except Exception as e:
                logger.error("Failed to update status for document %s in MySQL: %s", document_id, str(e))
                return False
        return False

    @staticmethod
    def get_by_id(document_id: str, user_id: str) -> Optional[dict]:
        """Fetch document by ID for specified user."""
        for session in get_db_session():
            if not session:
                return None
            try:
                record = (
                    session.query(DocumentRecord)
                    .filter(DocumentRecord.id == document_id, DocumentRecord.user_id == user_id)
                    .first()
                )
                if record:
                    return {
                        "document_id": record.id,
                        "user_id": record.user_id,
                        "filename": record.filename,
                        "stored_path": record.stored_path,
                        "size_bytes": record.size_bytes,
                        "chunk_count": record.chunk_count,
                        "status": record.status,
                        "stage": record.stage,
                        "error": record.error_message,
                        "created_at": record.created_at.isoformat() if record.created_at else "",
                    }
            except Exception as e:
                logger.error("Failed to get document %s from MySQL: %s", document_id, str(e))
        return None

    @staticmethod
    def list_by_user(user_id: str) -> List[dict]:
        """List all documents for a user from MySQL."""
        for session in get_db_session():
            if not session:
                return []
            try:
                records = (
                    session.query(DocumentRecord)
                    .filter(DocumentRecord.user_id == user_id)
                    .order_by(DocumentRecord.created_at.desc())
                    .all()
                )
                return [
                    {
                        "document_id": r.id,
                        "user_id": r.user_id,
                        "filename": r.filename,
                        "stored_path": r.stored_path,
                        "size_bytes": r.size_bytes,
                        "chunk_count": r.chunk_count,
                        "status": r.status,
                        "stage": r.stage,
                        "created_at": r.created_at.isoformat() if r.created_at else "",
                    }
                    for r in records
                ]
            except Exception as e:
                logger.error("Failed to list documents for user %s from MySQL: %s", user_id, str(e))
        return []

    @staticmethod
    def delete(document_id: str, user_id: str) -> bool:
        """Delete document record from MySQL."""
        for session in get_db_session():
            if not session:
                return False
            try:
                record = (
                    session.query(DocumentRecord)
                    .filter(DocumentRecord.id == document_id, DocumentRecord.user_id == user_id)
                    .first()
                )
                if record:
                    session.delete(record)
                    session.commit()
                    return True
            except Exception as e:
                logger.error("Failed to delete document %s from MySQL: %s", document_id, str(e))
        return False


class ChatRepository:
    """Handles logging and querying chat messages in MySQL."""

    @staticmethod
    def save_message(
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        latency_ms: Optional[int] = None,
    ) -> bool:
        """Save a chat message turn to MySQL audit log."""
        for session in get_db_session():
            if not session:
                return False
            try:
                msg = ChatMessageRecord(
                    session_id=session_id,
                    user_id=user_id,
                    role=role,
                    content=content,
                    latency_ms=latency_ms,
                )
                session.add(msg)
                session.commit()
                return True
            except Exception as e:
                logger.error("Failed to save chat message in MySQL: %s", str(e))
                return False
        return False


class GraphRepository:
    """Handles CRUD operations for Knowledge Graph nodes and edges in MySQL."""

    @staticmethod
    def save_triplet(
        user_id: str,
        source: str,
        target: str,
        relation: str,
        document_id: str,
        chunk_id: Optional[str] = None,
    ) -> bool:
        """Persist a single extracted graph triplet into MySQL."""
        for session in get_db_session():
            if not session:
                return False
            try:
                from services.database.models import GraphEdgeRecord, GraphNodeRecord

                # Ensure source and target nodes exist
                src_id = f"{user_id}:{source}"
                tgt_id = f"{user_id}:{target}"

                if not session.query(GraphNodeRecord).filter(GraphNodeRecord.id == src_id).first():
                    session.add(GraphNodeRecord(id=src_id, user_id=user_id, name=source))
                if not session.query(GraphNodeRecord).filter(GraphNodeRecord.id == tgt_id).first():
                    session.add(GraphNodeRecord(id=tgt_id, user_id=user_id, name=target))

                # Add edge
                edge = GraphEdgeRecord(
                    user_id=user_id,
                    source=source,
                    target=target,
                    relation=relation,
                    document_id=document_id,
                    chunk_id=chunk_id,
                )
                session.add(edge)
                session.commit()
                return True
            except Exception as e:
                logger.error("Failed to save graph triplet in MySQL: %s", str(e))
                return False
        return False

    @staticmethod
    def get_stats(user_id: str) -> dict:
        """Return node and edge counts for specified user."""
        for session in get_db_session():
            if not session:
                return {"nodes": 0, "edges": 0}
            try:
                from services.database.models import GraphEdgeRecord, GraphNodeRecord
                node_cnt = session.query(GraphNodeRecord).filter(GraphNodeRecord.user_id == user_id).count()
                edge_cnt = session.query(GraphEdgeRecord).filter(GraphEdgeRecord.user_id == user_id).count()
                return {"nodes": node_cnt, "edges": edge_cnt}
            except Exception as e:
                logger.error("Failed to query graph stats in MySQL: %s", str(e))
                return {"nodes": 0, "edges": 0}
        return {"nodes": 0, "edges": 0}

    @staticmethod
    def delete_document_graph(document_id: str, user_id: str) -> bool:
        """Delete all graph edges associated with a document."""
        for session in get_db_session():
            if not session:
                return False
            try:
                from services.database.models import GraphEdgeRecord
                session.query(GraphEdgeRecord).filter(
                    GraphEdgeRecord.document_id == document_id,
                    GraphEdgeRecord.user_id == user_id,
                ).delete()
                session.commit()
                return True
            except Exception as e:
                logger.error("Failed to delete document graph edges in MySQL: %s", str(e))
                return False
        return False
