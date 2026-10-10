"""In-memory and persistent Knowledge Graph storage and multi-hop traversal engine."""

import logging
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx

from config import get_settings
from services.database.repository import GraphRepository
from services.graph.models import GraphEdge, GraphTriplet

logger = logging.getLogger("memorygraph.graph.store")


class GraphStore:
    """Thread-safe, multi-tenant Knowledge Graph store using NetworkX with MySQL synchronization."""

    def __init__(self) -> None:
        self.settings = get_settings()
        # user_id -> nx.MultiDiGraph
        self._user_graphs: Dict[str, nx.MultiDiGraph] = {}

    def _get_graph(self, user_id: str) -> nx.MultiDiGraph:
        """Retrieve or create NetworkX graph for user."""
        if user_id not in self._user_graphs:
            self._user_graphs[user_id] = nx.MultiDiGraph()
        return self._user_graphs[user_id]

    def add_triplet(self, triplet: GraphTriplet, user_id: str) -> None:
        """Add a single triplet to the in-memory graph and persist to MySQL if enabled."""
        graph = self._get_graph(user_id)
        s, p, o = triplet.subject.strip(), triplet.predicate.strip(), triplet.object.strip()

        if not s or not o:
            return

        # Add nodes with metadata
        if not graph.has_node(s):
            graph.add_node(s, label=s, entity_type="CONCEPT")
        if not graph.has_node(o):
            graph.add_node(o, label=o, entity_type="CONCEPT")

        # Add directed edge
        graph.add_edge(
            s,
            o,
            relation=p,
            document_id=triplet.document_id,
            chunk_id=triplet.chunk_id,
            confidence=triplet.confidence,
        )

        # Sync to MySQL if enabled
        if self.settings.MYSQL_ENABLED:
            GraphRepository.save_triplet(
                user_id=user_id,
                source=s,
                target=o,
                relation=p,
                document_id=triplet.document_id,
                chunk_id=triplet.chunk_id,
            )

    def add_triplets(self, triplets: List[GraphTriplet], user_id: str) -> int:
        """Batch insert triplets for a user."""
        count = 0
        for t in triplets:
            self.add_triplet(t, user_id)
            count += 1
        logger.info("Indexed %d triplets into Knowledge Graph for user %s", count, user_id)
        return count

    def search_seed_nodes(self, query_tokens: List[str], user_id: str) -> List[str]:
        """Find matching node names in graph corresponding to query keywords."""
        graph = self._get_graph(user_id)
        matches: List[str] = []

        query_lower = [q.lower().strip() for q in query_tokens if len(q) >= 2]
        if not query_lower:
            return []

        for node in graph.nodes():
            node_lower = node.lower()
            for token in query_lower:
                if token in node_lower or node_lower in token:
                    if node not in matches:
                        matches.append(node)
                    break
        return matches

    def traverse_subgraph(
        self,
        seed_entities: List[str],
        user_id: str,
        max_hops: int = 2,
        max_triplets: int = 25,
    ) -> List[GraphTriplet]:
        """Perform multi-hop breadth-first traversal starting from seed entities.

        Args:
            seed_entities: Initial entity node names.
            user_id: Multi-tenant user scope.
            max_hops: Maximum traversal depth along graph edges.
            max_triplets: Safety limit on total relations returned.

        Returns:
            List of unique GraphTriplets discovered along the paths.
        """
        graph = self._get_graph(user_id)
        discovered_triplets: List[GraphTriplet] = []
        visited_nodes: Set[str] = set()
        seen_triplets: Set[Tuple[str, str, str]] = set()

        current_frontier = [s for s in seed_entities if graph.has_node(s)]

        for hop in range(max_hops):
            if not current_frontier or len(discovered_triplets) >= max_triplets:
                break

            next_frontier: Set[str] = set()

            for node in current_frontier:
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)

                # 1. Outgoing edges: node -> target
                for neighbor in graph.successors(node):
                    edge_data_dict = graph.get_edge_data(node, neighbor)
                    for key, edge_data in edge_data_dict.items():
                        pred = edge_data.get("relation", "relates to")
                        triplet_key = (node.lower(), pred.lower(), neighbor.lower())
                        if triplet_key not in seen_triplets:
                            seen_triplets.add(triplet_key)
                            discovered_triplets.append(
                                GraphTriplet(
                                    subject=node,
                                    predicate=pred,
                                    object=neighbor,
                                    document_id=edge_data.get("document_id", ""),
                                    chunk_id=edge_data.get("chunk_id"),
                                    confidence=edge_data.get("confidence", 1.0),
                                )
                            )
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)

                # 2. Incoming edges: predecessor -> node
                for pred_node in graph.predecessors(node):
                    edge_data_dict = graph.get_edge_data(pred_node, node)
                    for key, edge_data in edge_data_dict.items():
                        pred = edge_data.get("relation", "relates to")
                        triplet_key = (pred_node.lower(), pred.lower(), node.lower())
                        if triplet_key not in seen_triplets:
                            seen_triplets.add(triplet_key)
                            discovered_triplets.append(
                                GraphTriplet(
                                    subject=pred_node,
                                    predicate=pred,
                                    object=node,
                                    document_id=edge_data.get("document_id", ""),
                                    chunk_id=edge_data.get("chunk_id"),
                                    confidence=edge_data.get("confidence", 1.0),
                                )
                            )
                    if pred_node not in visited_nodes:
                        next_frontier.add(pred_node)

                if len(discovered_triplets) >= max_triplets:
                    break

            current_frontier = list(next_frontier)

        return discovered_triplets[:max_triplets]

    def list_all_triplets(self, user_id: str, document_id: Optional[str] = None) -> List[GraphTriplet]:
        """Fetch all triplets registered in the user's graph."""
        graph = self._get_graph(user_id)
        triplets: List[GraphTriplet] = []

        for u, v, key, data in graph.edges(keys=True, data=True):
            doc_id = data.get("document_id", "")
            if document_id and doc_id != document_id:
                continue
            triplets.append(
                GraphTriplet(
                    subject=u,
                    predicate=data.get("relation", "related to"),
                    object=v,
                    document_id=doc_id,
                    chunk_id=data.get("chunk_id"),
                    confidence=data.get("confidence", 1.0),
                )
            )
        return triplets

    def get_stats(self, user_id: str) -> Dict[str, int]:
        """Return counts of entities and relations in the Knowledge Graph."""
        graph = self._get_graph(user_id)
        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }

    def delete_document_triplets(self, document_id: str, user_id: str) -> int:
        """Remove all edges associated with a deleted document."""
        graph = self._get_graph(user_id)
        edges_to_remove = []

        for u, v, key, data in graph.edges(keys=True, data=True):
            if data.get("document_id") == document_id:
                edges_to_remove.append((u, v, key))

        for u, v, key in edges_to_remove:
            graph.remove_edge(u, v, key)

        # Remove isolated nodes
        isolated = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated)

        if self.settings.MYSQL_ENABLED:
            GraphRepository.delete_document_graph(document_id, user_id)

        logger.info("Removed %d graph edges for document %s (user %s)", len(edges_to_remove), document_id, user_id)
        return len(edges_to_remove)

    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear graph data (for testing purposes)."""
        if user_id:
            if user_id in self._user_graphs:
                self._user_graphs[user_id].clear()
        else:
            self._user_graphs.clear()


_graph_store_instance: Optional[GraphStore] = None


def get_graph_store() -> GraphStore:
    """Return singleton GraphStore instance."""
    global _graph_store_instance
    if _graph_store_instance is None:
        _graph_store_instance = GraphStore()
    return _graph_store_instance
