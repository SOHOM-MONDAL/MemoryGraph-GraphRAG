"""Document management routes for PDF upload, processing, and retrieval."""

import logging
import os
from pathlib import Path
import re
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from config import Settings, get_settings
from services.api.deps import (
    get_current_user,
    get_embedder_dep,
    get_producer_dep,
    get_store_dep,
)
from services.api.schemas.documents import (
    DocumentDetail,
    DocumentItem,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from services.cache.redis_client import get_redis_service
from services.database.repository import DocumentRepository
from services.embeddings.embedder import Embedder
from services.events.producer import EventProducer
from services.graph.extractor import extract_triplets
from services.graph.graph_store import get_graph_store
from services.pdf.chunker import chunk_document
from services.pdf.parser import PDFParseError, parse_pdf
from services.retrieval.vector_retriever import InMemoryVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def sanitize_filename(name: str) -> str:
    """Sanitize original filename to prevent path traversal."""
    cleaned = re.sub(r"[^\w\s\.-]", "_", name).strip()
    return cleaned if cleaned else "document.pdf"


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder_dep),
    store: InMemoryVectorStore = Depends(get_store_dep),
    producer: EventProducer = Depends(get_producer_dep),
) -> DocumentUploadResponse:
    """Upload a PDF file, validate, extract text, chunk, embed, and index into the store."""
    original_filename = file.filename or "uploaded.pdf"
    clean_name = sanitize_filename(original_filename)

    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported.",
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content.",
        )

    doc_id = str(uuid.uuid4())
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{doc_id}.pdf"

    # Save to disk
    try:
        stored_path.write_bytes(content)
    except Exception as e:
        logger.error("Failed to persist uploaded PDF to disk: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist file to storage disk.",
        )

    # Persist initial record in MySQL if enabled
    if settings.MYSQL_ENABLED:
        DocumentRepository.create_or_update(
            document_id=doc_id,
            user_id=user_id,
            filename=clean_name,
            stored_path=str(stored_path),
            size_bytes=len(content),
            status="PROCESSING",
            stage="QUEUED" if (settings.KAFKA_ENABLED and producer.is_connected) else "EXTRACTING",
        )

    # Cache status in Redis if enabled
    if settings.REDIS_ENABLED:
        get_redis_service().set_cached_document_status(
            doc_id,
            {"document_id": doc_id, "user_id": user_id, "status": "PROCESSING", "stage": "QUEUED"},
        )

    # If Kafka event-driven pipeline is enabled, publish upload event and return immediately
    if settings.KAFKA_ENABLED and producer.is_connected:
        store.set_status(doc_id, user_id, status="PROCESSING", stage="QUEUED")
        published = await producer.publish(
            topic="document.uploaded",
            key=user_id,
            message={
                "document_id": doc_id,
                "user_id": user_id,
                "filename": clean_name,
                "stored_path": str(stored_path),
                "size_bytes": len(content),
            },
        )
        if published:
            return DocumentUploadResponse(
                document_id=doc_id,
                filename=clean_name,
                chunk_count=0,
                status="QUEUED",
                message="Document uploaded and queued for asynchronous Kafka processing.",
            )
        logger.warning("Kafka publish failed; falling back to synchronous processing.")

    # 1. Parsing & Extraction stage (synchronous fallback)
    store.set_status(doc_id, user_id, status="PROCESSING", stage="EXTRACTING")
    try:
        pages = parse_pdf(content, max_mb=settings.MAX_UPLOAD_MB, max_pages=settings.MAX_PARSE_PAGES)
    except PDFParseError as pe:
        logger.warning("PDF parsing validation failed for doc %s: %s", doc_id, str(pe))
        store.set_status(doc_id, user_id, status="FAILED", stage="EXTRACTING", error=str(pe))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(pe))
    except Exception as e:
        logger.error("Unexpected error during PDF parsing: %s", str(e))
        store.set_status(doc_id, user_id, status="FAILED", stage="EXTRACTING", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse PDF document.")

    # 2. Chunking stage
    store.set_status(doc_id, user_id, status="PROCESSING", stage="CHUNKING")
    chunks = chunk_document(
        pages=pages,
        document_id=doc_id,
        user_id=user_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        msg = "No readable text content could be extracted from the document."
        store.set_status(doc_id, user_id, status="FAILED", stage="CHUNKING", error=msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=msg)

    # 3. Embedding stage
    store.set_status(doc_id, user_id, status="PROCESSING", stage="EMBEDDING")
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # 4. Knowledge Graph Extraction stage (GraphRAG)
    if settings.GRAPH_ENABLED:
        store.set_status(doc_id, user_id, status="PROCESSING", stage="GRAPH_INDEXING")
        graph_store = get_graph_store()
        all_triplets = []
        for chunk in chunks:
            triplets = extract_triplets(chunk.text, document_id=doc_id, chunk_id=chunk.chunk_id)
            all_triplets.extend(triplets)
        if all_triplets:
            graph_store.add_triplets(all_triplets, user_id=user_id)

    # 5. Storage & Indexing stage
    store.add_document(
        document_id=doc_id,
        user_id=user_id,
        filename=clean_name,
        stored_path=str(stored_path),
        size_bytes=len(content),
        chunks=chunks,
    )

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=clean_name,
        chunk_count=len(chunks),
        status="READY",
        message=f"Document processed and indexed successfully into {len(chunks)} chunks.",
    )


@router.get("", response_model=List[DocumentItem])
async def list_documents(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    store: InMemoryVectorStore = Depends(get_store_dep),
) -> List[DocumentItem]:
    """List all documents for the authenticated user."""
    docs = []
    if settings.MYSQL_ENABLED:
        docs = DocumentRepository.list_by_user(user_id)
    if not docs:
        docs = store.list_documents(user_id)

    return [
        DocumentItem(
            document_id=d["document_id"],
            filename=d["filename"],
            chunk_count=d["chunk_count"],
            status=d["status"],
            size_bytes=d["size_bytes"],
            created_at=d["created_at"],
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
    store: InMemoryVectorStore = Depends(get_store_dep),
) -> DocumentDetail:
    """Fetch details and chunk list for a specific document."""
    doc = store.get_document(document_id, user_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    chunks = store.get_document_chunks(document_id, user_id)
    serialized_chunks = [
        {
            "chunk_id": c.chunk_id,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "text": c.text,
        }
        for c in chunks
    ]

    return DocumentDetail(
        document_id=doc["document_id"],
        filename=doc["filename"],
        chunk_count=doc["chunk_count"],
        status=doc["status"],
        size_bytes=doc["size_bytes"],
        created_at=doc["created_at"],
        chunks=serialized_chunks,
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    store: InMemoryVectorStore = Depends(get_store_dep),
) -> DocumentStatusResponse:
    """Fetch processing status and stage for a document with Redis caching and MySQL fallback."""
    # 1. Check Redis cache first
    if settings.REDIS_ENABLED:
        cached = get_redis_service().get_cached_document_status(document_id)
        if cached and cached.get("user_id") == user_id:
            return DocumentStatusResponse(
                document_id=cached["document_id"],
                status=cached["status"],
                stage=cached["stage"],
                error=cached.get("error"),
            )

    # 2. Check in-memory store
    stat = store.get_status(document_id, user_id)
    if not stat and settings.MYSQL_ENABLED:
        # 3. Check MySQL repository fallback
        db_doc = DocumentRepository.get_by_id(document_id, user_id)
        if db_doc:
            stat = {
                "document_id": db_doc["document_id"],
                "status": db_doc["status"],
                "stage": db_doc["stage"],
                "error": db_doc.get("error"),
            }

    if not stat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document status not found.",
        )

    return DocumentStatusResponse(
        document_id=stat["document_id"],
        status=stat["status"],
        stage=stat["stage"],
        error=stat.get("error"),
    )


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    store: InMemoryVectorStore = Depends(get_store_dep),
) -> dict:
    """Delete a document, its stored chunks, and knowledge graph triplets."""
    deleted_store = store.delete_document(document_id, user_id)
    deleted_db = False
    if settings.MYSQL_ENABLED:
        deleted_db = DocumentRepository.delete(document_id, user_id)

    # Clean up knowledge graph
    if settings.GRAPH_ENABLED:
        get_graph_store().delete_document_triplets(document_id, user_id)

    if not deleted_store and not deleted_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to user.",
        )
    return {"deleted": True, "document_id": document_id}


@router.get("/graph/stats", tags=["graph"])
async def get_graph_stats(
    user_id: str = Depends(get_current_user),
) -> dict:
    """Fetch Knowledge Graph node and edge count statistics."""
    graph_store = get_graph_store()
    return graph_store.get_stats(user_id)


@router.get("/graph/triplets", tags=["graph"])
async def get_graph_triplets(
    user_id: str = Depends(get_current_user),
    document_id: Optional[str] = None,
) -> List[dict]:
    """Fetch extracted Knowledge Graph triplets."""
    graph_store = get_graph_store()
    triplets = graph_store.list_all_triplets(user_id, document_id=document_id)
    return [
        {
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.object,
            "document_id": t.document_id,
            "chunk_id": t.chunk_id,
            "fact": t.to_readable_fact(),
        }
        for t in triplets
    ]
