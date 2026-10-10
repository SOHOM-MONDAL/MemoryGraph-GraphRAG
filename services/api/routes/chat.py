"""Chat routes with hybrid RAG retrieval and source attribution."""

import logging
import time
from typing import List
from fastapi import APIRouter, Depends, Request

from config import Settings, get_settings
from services.api.deps import get_current_user, get_embedder_dep, get_llm, get_store_dep
from services.api.schemas.chat import ChatRequest, ChatResponse, SourceItem
from services.cache.redis_client import get_redis_service
from services.database.repository import ChatRepository
from services.embeddings.embedder import Embedder
from services.graph.hybrid_retriever import get_hybrid_retriever
from services.llm.base import LLMProvider
from services.retrieval.vector_retriever import InMemoryVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

GROUNDED_SYSTEM_PROMPT = (
    "You are MemoryGraph, an intelligent GraphRAG assistant. Answer the user's inquiry accurately "
    "using the provided Knowledge Graph relationships and document context below. "
    "Synthesize both the structured graph facts and the document excerpts to answer. "
    "If the question cannot be answered from the provided facts and context, state clearly "
    "that the provided documents and knowledge graph do not contain enough information. "
    "Do NOT fabricate information or citations."
)


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm),
    embedder: Embedder = Depends(get_embedder_dep),
    store: InMemoryVectorStore = Depends(get_store_dep),
) -> ChatResponse:
    """Chat endpoint supporting Hybrid GraphRAG retrieval (Knowledge Graph + Vector Grounding)."""
    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", "req-unknown")

    # Fetch conversation memory from Redis
    redis_svc = get_redis_service()
    history = redis_svc.get_chat_history(req.session_id)
    history_context = ""
    if history:
        history_lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]]
        history_context = "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    sources: List[SourceItem] = []
    graph_triplets: List[str] = []
    fused_context = ""

    # Check if user has any documents indexed
    user_docs = store.list_documents(user_id)

    if user_docs:
        # Execute Hybrid GraphRAG Retrieval (Dense Vectors + Multi-hop Graph Traversal)
        retriever = get_hybrid_retriever()
        retrieval_result = retriever.retrieve(
            query=req.message,
            user_id=user_id,
            top_k=settings.TOP_K,
            max_hops=settings.GRAPH_HOPS,
        )

        fused_context = retrieval_result.fused_context
        graph_triplets = [t.to_readable_fact() for t in retrieval_result.triplets]

        for chunk, score in retrieval_result.chunks:
            doc_meta = store.get_document(chunk.document_id, user_id)
            filename = doc_meta["filename"] if doc_meta else "document.pdf"
            snippet = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")

            sources.append(
                SourceItem(
                    filename=filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    snippet=snippet,
                )
            )

    if fused_context:
        augmented_prompt = (
            f"{history_context}"
            f"{fused_context}\n\n"
            f"User Question: {req.message}\n\n"
            f"Answer based on the Knowledge Graph facts and document context above:"
        )
        llm_response = llm.generate(prompt=augmented_prompt, system=GROUNDED_SYSTEM_PROMPT)
    else:
        # Chit-chat or no documents uploaded
        general_system = "You are MemoryGraph, a knowledgeable and concise assistant."
        chit_prompt = f"{history_context}User: {req.message}" if history_context else req.message
        llm_response = llm.generate(prompt=chit_prompt, system=general_system)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Persist turns into Redis memory and MySQL audit log
    if settings.REDIS_ENABLED:
        redis_svc.add_chat_message(req.session_id, "user", req.message)
        redis_svc.add_chat_message(req.session_id, "assistant", llm_response)

    if settings.MYSQL_ENABLED:
        ChatRepository.save_message(req.session_id, user_id, "user", req.message)
        ChatRepository.save_message(req.session_id, user_id, "assistant", llm_response, latency_ms=int(latency_ms))

    return ChatResponse(
        response=llm_response,
        sources=sources,
        graph_triplets=graph_triplets,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
    )
