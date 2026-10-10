"""Data models for Knowledge Graph entities, relationships, and hybrid retrieval."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.pdf.chunker import PDFChunk


@dataclass
class GraphTriplet:
    """A directed semantic relationship statement (Subject -[Predicate]-> Object)."""

    subject: str
    predicate: str
    object: str
    document_id: str = ""
    chunk_id: Optional[str] = None
    confidence: float = 1.0

    def to_readable_fact(self) -> str:
        """Format triplet as a readable proposition."""
        return f"({self.subject}) -[{self.predicate}]-> ({self.object})"


@dataclass
class GraphNode:
    """An entity or concept node in the Knowledge Graph."""

    id: str
    name: str
    entity_type: str = "CONCEPT"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed edge between two entity nodes."""

    source: str
    target: str
    relation: str
    document_id: str
    chunk_id: Optional[str] = None
    weight: int = 1


@dataclass
class HybridRetrievalResult:
    """Unified result containing both semantic vector chunks and knowledge graph subgraphs."""

    chunks: List[Tuple[PDFChunk, float]] = field(default_factory=list)
    triplets: List[GraphTriplet] = field(default_factory=list)
    seed_entities: List[str] = field(default_factory=list)
    fused_context: str = ""
    graph_context: str = ""
    vector_context: str = ""
