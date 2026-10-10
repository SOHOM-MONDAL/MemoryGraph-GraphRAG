"""Hybrid GraphRAG retriever combining Dense Vector Search with Multi-Hop Knowledge Graph Traversal."""

import logging
import re
from typing import List, Optional, Tuple

from config import get_settings
from services.embeddings.embedder import Embedder, get_embedder
from services.graph.graph_store import GraphStore, get_graph_store
from services.graph.models import GraphTriplet, HybridRetrievalResult
from services.pdf.chunker import PDFChunk
from services.retrieval.vector_retriever import InMemoryVectorStore, get_vector_store

logger = logging.getLogger("memorygraph.graph.hybrid")


class HybridRetriever:
    """Orchestrates Hybrid GraphRAG retrieval: Vector Search + Knowledge Graph Expansion."""

    def __init__(
        self,
        vector_store: Optional[InMemoryVectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.graph_store = graph_store or get_graph_store()
        self.embedder = embedder or get_embedder(
            model_name=self.settings.EMBEDDING_MODEL,
            embedding_dim=self.settings.EMBEDDING_DIM,
        )

    def extract_query_entities(self, query: str) -> List[str]:
        """Extract candidate entity tokens from query string."""
        # Find capitalized words or significant nouns/acronyms
        tokens = re.findall(r"\b[A-Za-z0-9_-]{2,}\b", query)
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "can", "could", "should", "would",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down", "in",
            "tell", "me", "show", "explain", "describe", "summary", "summarize",
        }
        return [t for t in tokens if t.lower() not in stopwords]

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> HybridRetrievalResult:
        """Execute hybrid vector + graph retrieval for a query.

        Args:
            query: User's natural language question.
            user_id: Tenant user ID for data isolation.
            top_k: Number of vector chunks to retrieve.
            max_hops: Maximum graph traversal hops.

        Returns:
            HybridRetrievalResult containing fused context, chunks, and graph triplets.
        """
        k = top_k or self.settings.TOP_K
        hops = max_hops or self.settings.GRAPH_HOPS

        # 1. Vector Dense Search
        query_embedding = self.embedder.embed_text(query)
        vector_results: List[Tuple[PDFChunk, float]] = self.vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=k,
        )

        # 2. Extract Seed Entities from Query & Top Vector Chunks
        query_candidates = self.extract_query_entities(query)
        seed_nodes = self.graph_store.search_seed_nodes(query_candidates, user_id=user_id)

        # Expand seed nodes with terms present in the top-ranked vector chunk
        if vector_results:
            top_chunk_text = vector_results[0][0].text
            top_terms = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", top_chunk_text)
            additional_seeds = self.graph_store.search_seed_nodes(top_terms[:5], user_id=user_id)
            for s in additional_seeds:
                if s not in seed_nodes:
                    seed_nodes.append(s)

        # 3. Multi-Hop Knowledge Graph Traversal
        graph_triplets: List[GraphTriplet] = []
        if seed_nodes and self.settings.GRAPH_ENABLED:
            graph_triplets = self.graph_store.traverse_subgraph(
                seed_entities=seed_nodes,
                user_id=user_id,
                max_hops=hops,
                max_triplets=self.settings.GRAPH_MAX_SUBGRAPH_NODES,
            )

        # 4. Context Assembly & Fusion
        graph_context_lines = []
        if graph_triplets:
            graph_context_lines.append("### Relevant Knowledge Graph Relationships:")
            for t in graph_triplets:
                graph_context_lines.append(f"• {t.to_readable_fact()}")

        vector_context_blocks = []
        for chunk, score in vector_results:
            doc_meta = self.vector_store.get_document(chunk.document_id, user_id)
            filename = doc_meta["filename"] if doc_meta else "document.pdf"
            vector_context_blocks.append(
                f"[Document: {filename} | Page: {chunk.page_number} | Chunk: {chunk.chunk_id}]\n"
                f"{chunk.text}"
            )

        graph_context = "\n".join(graph_context_lines)
        vector_context = "\n---\n".join(vector_context_blocks)

        parts = []
        if graph_context:
            parts.append(graph_context)
        if vector_context:
            parts.append("### Relevant Document Excerpts:\n" + vector_context)

        fused_context = "\n\n".join(parts)

        return HybridRetrievalResult(
            chunks=vector_results,
            triplets=graph_triplets,
            seed_entities=seed_nodes,
            fused_context=fused_context,
            graph_context=graph_context,
            vector_context=vector_context,
        )


_hybrid_retriever_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Return singleton HybridRetriever instance."""
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRetriever()
    return _hybrid_retriever_instance
