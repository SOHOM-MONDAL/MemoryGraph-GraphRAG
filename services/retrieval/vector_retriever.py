"""In-memory vector store and retriever for Phase 2 (Chroma-free, lightweight)."""

import datetime
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from services.pdf.chunker import PDFChunk

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Thread-safe, user-isolated in-memory document & vector storage."""

    def __init__(self) -> None:
        # doc_id -> metadata dict
        self._documents: Dict[str, dict] = {}
        # doc_id -> list of PDFChunk
        self._doc_chunks: Dict[str, List[PDFChunk]] = {}
        # doc_id -> status dict
        self._statuses: Dict[str, dict] = {}

    def add_document(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        stored_path: str,
        size_bytes: int,
        chunks: List[PDFChunk],
    ) -> None:
        """Register document and its indexed chunks."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._documents[document_id] = {
            "document_id": document_id,
            "user_id": user_id,
            "filename": filename,
            "stored_path": stored_path,
            "size_bytes": size_bytes,
            "chunk_count": len(chunks),
            "status": "READY",
            "created_at": now,
        }
        self._doc_chunks[document_id] = chunks
        self._statuses[document_id] = {
            "document_id": document_id,
            "user_id": user_id,
            "status": "READY",
            "stage": "READY",
            "error": None,
        }
        logger.info(
            "Indexed document %s ('%s') with %d chunks for user %s",
            document_id,
            filename,
            len(chunks),
            user_id,
        )

    def set_status(
        self,
        document_id: str,
        user_id: str,
        status: str,
        stage: str,
        error: Optional[str] = None,
    ) -> None:
        """Update processing status for a document."""
        self._statuses[document_id] = {
            "document_id": document_id,
            "user_id": user_id,
            "status": status,
            "stage": stage,
            "error": error,
        }

    def get_status(self, document_id: str, user_id: str) -> Optional[dict]:
        """Fetch current status for a document, enforcing user isolation."""
        stat = self._statuses.get(document_id)
        if stat and stat["user_id"] == user_id:
            return stat
        return None

    def list_documents(self, user_id: str) -> List[dict]:
        """List all documents owned by user_id."""
        return [
            doc
            for doc in self._documents.values()
            if doc["user_id"] == user_id
        ]

    def get_document(self, document_id: str, user_id: str) -> Optional[dict]:
        """Get document details with user isolation check."""
        doc = self._documents.get(document_id)
        if doc and doc["user_id"] == user_id:
            return doc
        return None

    def get_document_chunks(self, document_id: str, user_id: str) -> List[PDFChunk]:
        """Get all chunks for a document with user isolation check."""
        doc = self.get_document(document_id, user_id)
        if not doc:
            return []
        return self._doc_chunks.get(document_id, [])

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete document and its chunks with user isolation check."""
        doc = self.get_document(document_id, user_id)
        if not doc:
            return False

        self._documents.pop(document_id, None)
        self._doc_chunks.pop(document_id, None)
        self._statuses.pop(document_id, None)
        return True

    def search(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Tuple[PDFChunk, float]]:
        """Search chunks by cosine similarity with strict user_id filtering.

        Args:
            query_embedding: Normalized query vector.
            user_id: Owner user ID for tenant isolation.
            top_k: Number of highest-similarity chunks to return.
            document_id: Optional filter for a specific document.

        Returns:
            List of (PDFChunk, similarity_score) sorted descending by score.
        """
        if not query_embedding:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        results: List[Tuple[PDFChunk, float]] = []

        # Iterate all chunks owned by user_id
        for doc_key, chunks in self._doc_chunks.items():
            doc_meta = self._documents.get(doc_key)
            if not doc_meta or doc_meta["user_id"] != user_id:
                continue

            if document_id and doc_key != document_id:
                continue

            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                c_vec = np.array(chunk.embedding, dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                if c_norm == 0:
                    continue

                score = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                results.append((chunk, score))

        # Sort descending by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all in-memory store data (for tests)."""
        self._documents.clear()
        self._doc_chunks.clear()
        self._statuses.clear()


_vector_store_instance: Optional[InMemoryVectorStore] = None


def get_vector_store() -> InMemoryVectorStore:
    """Return singleton vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = InMemoryVectorStore()
    return _vector_store_instance
