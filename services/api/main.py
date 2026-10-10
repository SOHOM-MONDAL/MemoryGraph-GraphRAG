import asyncio
from contextlib import asynccontextmanager
import logging
import uuid
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from services.api.deps import get_embedder_dep, get_llm, get_store_dep
from services.api.routes.chat import router as chat_router
from services.api.routes.documents import router as documents_router
from services.cache.redis_client import get_redis_service
from services.database.connection import init_db, is_database_connected
from services.events.producer import get_producer
from services.workers.document_worker import run_document_consumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (req_id=%(request_id)s): %(message)s"
    if hasattr(logging.LogRecord, "request_id")
    else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("memorygraph.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle tasks."""
    settings = get_settings()
    producer = get_producer()
    stop_event = asyncio.Event()
    worker_task = None

    if settings.MYSQL_ENABLED:
        init_db()

    if settings.REDIS_ENABLED:
        redis_svc = get_redis_service()
        logger.info("Redis cache initialized (connected=%s).", redis_svc.is_connected())

    if settings.KAFKA_ENABLED:
        await producer.start()
        worker_task = asyncio.create_task(run_document_consumer(stop_event=stop_event))
        logger.info("Kafka producer and background document worker started.")

    yield

    stop_event.set()
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    if settings.KAFKA_ENABLED:
        await producer.stop()


app = FastAPI(
    title="MemoryGraph API",
    description="Event-Driven GraphRAG Chatbot with Persistent Memory",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign or propagate unique request ID across logs and headers."""
    req_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = req_id

    response = await call_next(request)
    response.headers["X-Request-Id"] = req_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Format uniform error responses for HTTP exceptions."""
    req_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "detail": exc.detail,
            "request_id": req_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Format uniform error responses for unhandled server exceptions."""
    req_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled server error: %s (request_id=%s)", str(exc), req_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "detail": "An unexpected error occurred during request processing.",
            "request_id": req_id,
        },
    )


@app.api_route("/", methods=["GET", "HEAD"], tags=["system"])
async def root() -> Dict[str, Any]:
    """Root endpoint providing service discovery links."""
    return {
        "service": "MemoryGraph API",
        "description": "Event-Driven GraphRAG Engine with Hybrid Dense + Multi-Hop Graph Retrieval",
        "version": "0.2.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "ui_hint": "Run 'streamlit run frontend/streamlit_app.py' to launch the interactive UI on http://localhost:8501",
    }


@app.api_route("/health", methods=["GET", "HEAD"], tags=["system"])
async def health_check() -> Dict[str, Any]:
    """Check health of all active system dependencies in a fast, non-blocking manner."""
    settings = get_settings()
    producer = get_producer()
    redis_svc = get_redis_service()
    mysql_connected = is_database_connected()

    # Fast non-blocking checks to avoid hanging Render liveness probes
    from services.embeddings.embedder import _cached_embedder
    embedder_healthy = _cached_embedder.health() if _cached_embedder is not None else True
    llm_healthy = bool(settings.GROQ_API_KEY) if settings.LLM_PROVIDER == "groq" else True

    return {
        "status": "ok",
        "llm": {
            "provider": settings.LLM_PROVIDER,
            "healthy": llm_healthy,
        },
        "embedder": {
            "model": settings.EMBEDDING_MODEL,
            "healthy": embedder_healthy,
        },
        "storage": {
            "type": "in_memory",
            "healthy": True,
        },
        "kafka": {
            "enabled": settings.KAFKA_ENABLED,
            "connected": producer.is_connected,
        },
        "redis": {
            "enabled": settings.REDIS_ENABLED,
            "connected": redis_svc.is_connected(),
        },
        "mysql": {
            "enabled": settings.MYSQL_ENABLED,
            "connected": mysql_connected,
        },
    }


# Include Routers
app.include_router(chat_router)
app.include_router(documents_router)
