"""Kafka document processing consumer worker for MemoryGraph."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from aiokafka import AIOKafkaConsumer

from config import get_settings
from services.cache.redis_client import get_redis_service
from services.database.repository import DocumentRepository
from services.embeddings.embedder import get_embedder
from services.graph.extractor import extract_triplets
from services.graph.graph_store import get_graph_store
from services.pdf.chunker import chunk_document
from services.pdf.parser import PDFParseError, parse_pdf
from services.retrieval.vector_retriever import get_vector_store

logger = logging.getLogger("memorygraph.worker")


async def process_document_event(data: dict) -> bool:
    """Process a single document.uploaded event.

    Args:
        data: Dict containing document_id, user_id, filename, stored_path, size_bytes.

    Returns:
        True if successfully parsed and indexed, False otherwise.
    """
    settings = get_settings()
    store = get_vector_store()
    embedder = get_embedder(
        model_name=settings.EMBEDDING_MODEL,
        embedding_dim=settings.EMBEDDING_DIM,
    )

    doc_id = data.get("document_id")
    user_id = data.get("user_id", "default_user")
    filename = data.get("filename", "document.pdf")
    stored_path_str = data.get("stored_path")
    size_bytes = data.get("size_bytes", 0)

    if not doc_id or not stored_path_str:
        logger.error("Invalid document event payload: %s", data)
        return False

    def update_state(status: str, stage: str, chunk_count: Optional[int] = None, error: Optional[str] = None):
        store.set_status(doc_id, user_id, status=status, stage=stage, error=error)
        if settings.REDIS_ENABLED:
            get_redis_service().set_cached_document_status(
                doc_id,
                {"document_id": doc_id, "user_id": user_id, "status": status, "stage": stage, "error": error},
            )
        if settings.MYSQL_ENABLED:
            DocumentRepository.update_status(
                doc_id, status=status, stage=stage, chunk_count=chunk_count, error=error
            )

    stored_path = Path(stored_path_str)
    if not stored_path.exists():
        msg = f"Stored PDF file not found at: {stored_path}"
        logger.error(msg)
        update_state(status="FAILED", stage="EXTRACTING", error=msg)
        return False

    try:
        content = stored_path.read_bytes()
    except Exception as e:
        msg = f"Failed to read file from disk: {str(e)}"
        logger.error(msg)
        update_state(status="FAILED", stage="EXTRACTING", error=msg)
        return False

    # 1. Parsing & Extraction stage
    update_state(status="PROCESSING", stage="EXTRACTING")
    try:
        pages = parse_pdf(content, max_mb=settings.MAX_UPLOAD_MB)
    except PDFParseError as pe:
        logger.warning("PDF parsing validation failed for %s: %s", doc_id, str(pe))
        update_state(status="FAILED", stage="EXTRACTING", error=str(pe))
        return False
    except Exception as e:
        logger.error("Unexpected error during PDF parsing for %s: %s", doc_id, str(e))
        update_state(status="FAILED", stage="EXTRACTING", error=str(e))
        return False

    # 2. Chunking stage
    update_state(status="PROCESSING", stage="CHUNKING")
    chunks = chunk_document(
        pages=pages,
        document_id=doc_id,
        user_id=user_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        msg = "No readable text content could be extracted from the document."
        logger.warning("No chunks generated for %s: %s", doc_id, msg)
        update_state(status="FAILED", stage="CHUNKING", error=msg)
        return False

    # 3. Embedding stage
    update_state(status="PROCESSING", stage="EMBEDDING", chunk_count=len(chunks))
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # 4. Knowledge Graph Extraction stage (GraphRAG)
    if settings.GRAPH_ENABLED:
        update_state(status="PROCESSING", stage="GRAPH_INDEXING", chunk_count=len(chunks))
        graph_store = get_graph_store()
        all_triplets = []
        for chunk in chunks:
            triplets = extract_triplets(chunk.text, document_id=doc_id, chunk_id=chunk.chunk_id)
            all_triplets.extend(triplets)
        if all_triplets:
            graph_store.add_triplets(all_triplets, user_id=user_id)
            logger.info("Worker extracted %d knowledge graph triplets for document %s", len(all_triplets), doc_id)

    # 5. Storage & Indexing stage
    store.add_document(
        document_id=doc_id,
        user_id=user_id,
        filename=filename,
        stored_path=str(stored_path),
        size_bytes=size_bytes,
        chunks=chunks,
    )
    update_state(status="READY", stage="READY", chunk_count=len(chunks))

    logger.info("Worker finished indexing doc %s with %d chunks.", doc_id, len(chunks))
    return True


async def run_document_consumer(
    topic: str = "document.uploaded",
    group_id: str = "memorygraph-doc-workers",
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Run Kafka consumer loop processing document upload events.

    Args:
        topic: Kafka topic to consume.
        group_id: Consumer group ID.
        stop_event: Optional asyncio Event to cleanly stop the loop.
    """
    settings = get_settings()
    if not settings.KAFKA_ENABLED:
        logger.info("Kafka is disabled in configuration. Worker consumer will not start.")
        return

    logger.info("Initializing Kafka Consumer on %s for topic '%s'...", settings.KAFKA_BOOTSTRAP_SERVERS, topic)

    consumer = None
    try:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        await consumer.start()
        logger.info("Kafka Consumer started and listening on topic '%s'.", topic)

        while stop_event is None or not stop_event.is_set():
            try:
                # Poll with short timeout to allow responsive shutdown
                msg = await asyncio.wait_for(consumer.getone(), timeout=1.0)
                logger.info("Received event on '%s': key=%s", topic, msg.key)
                await process_document_event(msg.value)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing Kafka message: %s", str(e), exc_info=True)
                await asyncio.sleep(1.0)

    except Exception as e:
        logger.error("Failed to run Kafka document consumer: %s", str(e), exc_info=True)
    finally:
        if consumer:
            await consumer.stop()
            logger.info("Kafka Consumer cleanly stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("Starting MemoryGraph Document Kafka Worker...")
    try:
        asyncio.run(run_document_consumer())
    except KeyboardInterrupt:
        print("Worker stopped by user.")
