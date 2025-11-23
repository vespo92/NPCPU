"""
Dynamic Social Graph for NPCPU

Provides advanced graph-based social network representation with:
- Weighted directed graph structure
- Community detection algorithms
- Network centrality metrics
- Dynamic graph evolution
- Temporal relationship tracking

Example:
    from social.social_graph import SocialGraph, GraphMetrics

    # Create social graph
    graph = SocialGraph()

    # Add organisms and connections
    graph.add_organism("org_1", attributes={"age": 5, "species": "herbivore"})
    graph.add_organism("org_2", attributes={"age": 3, "species": "herbivore"})
    graph.add_edge("org_1", "org_2", weight=0.8, edge_type="ally")

    # Calculate centrality
    metrics = graph.calculate_centrality("org_1")
    print(f"Betweenness: {metrics.betweenness}")

    # Detect communities
    communities = graph.detect_communities()
"""

from typing import Dict, Any, List, Optional, Set, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
import math
import uuid
import heapq

from core.events import get_event_bus


class EdgeType(Enum):
    """Types of edges in the social graph"""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    KINSHIP = "kinship"
    ALLIANCE = "alliance"
    RIVALRY = "rivalry"
    COMMUNICATION = "communication"
    RESOURCE_SHARING = "resource_sharing"


@dataclass
class GraphNode:
    """
    Represents a node (organism) in the social graph.

    Attributes:
        organism_id: Unique identifier
        attributes: Node attributes (age, species, etc.)
        creation_time: When node was added
        metadata: Additional node data
    """
    organism_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached metrics
    _degree: int = 0
    _in_degree: int = 0
    _out_degree: int = 0

    def update_attribute(self, key: str, value: Any) -> None:
        """Update a node attribute"""
        self.attributes[key] = value


@dataclass
class GraphEdge:
    """
    Represents a weighted, directed edge in the social graph.

    Attributes:
        source_id: Source organism ID
        target_id: Target organism ID
        weight: Edge weight (0.0 to 1.0)
        edge_type: Type of relationship
        creation_time: When edge was created
        last_update: Last modification time
        interaction_count: Number of interactions along this edge
        metadata: Additional edge data
    """
    source_id: str
    target_id: str
    weight: float = 0.5
    edge_type: EdgeType = EdgeType.NEUTRAL
    creation_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        self.weight = max(0.0, min(1.0, self.weight))

    def update_weight(self, delta: float) -> None:
        """Update edge weight with bounds checking"""
        self.weight = max(0.0, min(1.0, self.weight + delta))
        self.last_update = datetime.now()

    def record_interaction(self) -> None:
        """Record an interaction along this edge"""
        self.interaction_count += 1
        self.last_update = datetime.now()


@dataclass
class GraphMetrics:
    """
    Centrality and connectivity metrics for a node.

    Attributes:
        degree_centrality: Proportion of nodes connected to
        betweenness_centrality: Fraction of shortest paths through node
        closeness_centrality: Inverse sum of distances to all nodes
        eigenvector_centrality: Influence based on neighbor importance
        clustering_coefficient: Degree of neighbor interconnection
        pagerank: Importance based on incoming edges
    """
    organism_id: str
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank: float = 0.0
    in_degree: int = 0
    out_degree: int = 0


@dataclass
class Community:
    """
    Represents a detected community in the graph.

    Attributes:
        id: Unique community identifier
        members: Set of organism IDs in the community
        modularity_contribution: How much this community contributes to modularity
        density: Internal edge density
        metadata: Additional community data
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    members: Set[str] = field(default_factory=set)
    modularity_contribution: float = 0.0
    density: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.members)


class SocialGraph:
    """
    Advanced graph-based social network representation.

    Provides:
    - Efficient adjacency list representation
    - Centrality calculations
    - Community detection
    - Shortest path algorithms
    - Temporal evolution tracking

    Example:
        graph = SocialGraph()
        graph.add_organism("alice")
        graph.add_organism("bob")
        graph.add_edge("alice", "bob", weight=0.9)

        # Get metrics
        metrics = graph.calculate_centrality("alice")
        communities = graph.detect_communities()
    """

    def __init__(self, emit_events: bool = True):
        """
        Initialize the social graph.

        Args:
            emit_events: Whether to emit events on graph changes
        """
        # Nodes indexed by organism ID
        self._nodes: Dict[str, GraphNode] = {}

        # Adjacency list: source -> {target -> edge}
        self._edges: Dict[str, Dict[str, GraphEdge]] = {}

        # Reverse adjacency for incoming edges
        self._reverse_edges: Dict[str, Set[str]] = {}

        # Cached metrics
        self._metrics_cache: Dict[str, GraphMetrics] = {}
        self._cache_valid: bool = False

        # Community assignments
        self._communities: Dict[str, Community] = {}
        self._node_community: Dict[str, str] = {}

        # Event emission
        self._emit_events = emit_events

        # Graph statistics
        self._edge_count: int = 0

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_organism(
        self,
        organism_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphNode:
        """
        Add an organism node to the graph.

        Args:
            organism_id: Unique organism identifier
            attributes: Node attributes
            metadata: Additional data

        Returns:
            The created GraphNode
        """
        if organism_id in self._nodes:
            # Update existing node
            node = self._nodes[organism_id]
            if attributes:
                node.attributes.update(attributes)
            if metadata:
                node.metadata.update(metadata)
            return node

        node = GraphNode(
            organism_id=organism_id,
            attributes=attributes or {},
            metadata=metadata or {}
        )

        self._nodes[organism_id] = node
        self._edges[organism_id] = {}
        self._reverse_edges[organism_id] = set()
        self._invalidate_cache()

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("graph.node_added", {
                "organism_id": organism_id,
                "attributes": attributes or {}
            })

        return node

    def remove_organism(self, organism_id: str) -> bool:
        """
        Remove an organism and all its edges from the graph.

        Returns:
            True if organism was removed
        """
        if organism_id not in self._nodes:
            return False

        # Remove outgoing edges
        if organism_id in self._edges:
            for target_id in list(self._edges[organism_id].keys()):
                self.remove_edge(organism_id, target_id)

        # Remove incoming edges
        for source_id in list(self._reverse_edges.get(organism_id, set())):
            self.remove_edge(source_id, organism_id)

        # Clean up node data
        del self._nodes[organism_id]
        if organism_id in self._edges:
            del self._edges[organism_id]
        if organism_id in self._reverse_edges:
            del self._reverse_edges[organism_id]

        # Remove from community
        if organism_id in self._node_community:
            community_id = self._node_community[organism_id]
            if community_id in self._communities:
                self._communities[community_id].members.discard(organism_id)
            del self._node_community[organism_id]

        self._invalidate_cache()

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("graph.node_removed", {"organism_id": organism_id})

        return True

    def get_organism(self, organism_id: str) -> Optional[GraphNode]:
        """Get a node by organism ID"""
        return self._nodes.get(organism_id)

    def has_organism(self, organism_id: str) -> bool:
        """Check if organism exists in graph"""
        return organism_id in self._nodes

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.5,
        edge_type: EdgeType = EdgeType.NEUTRAL,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphEdge:
        """
        Add or update an edge between two organisms.

        Args:
            source_id: Source organism ID
            target_id: Target organism ID
            weight: Edge weight (0.0 to 1.0)
            edge_type: Type of relationship
            bidirectional: Create edge in both directions
            metadata: Additional edge data

        Returns:
            The created/updated GraphEdge
        """
        # Ensure nodes exist
        if source_id not in self._nodes:
            self.add_organism(source_id)
        if target_id not in self._nodes:
            self.add_organism(target_id)

        # Create or update edge
        existing = self._edges[source_id].get(target_id)
        if existing:
            existing.weight = weight
            existing.edge_type = edge_type
            existing.last_update = datetime.now()
            if metadata:
                existing.metadata.update(metadata)
            edge = existing
        else:
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                weight=weight,
                edge_type=edge_type,
                metadata=metadata or {}
            )
            self._edges[source_id][target_id] = edge
            self._reverse_edges[target_id].add(source_id)
            self._edge_count += 1

        self._invalidate_cache()

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("graph.edge_added", {
                "source_id": source_id,
                "target_id": target_id,
                "weight": weight,
                "edge_type": edge_type.value
            })

        # Handle bidirectional
        if bidirectional and source_id != target_id:
            self.add_edge(
                target_id, source_id, weight, edge_type,
                bidirectional=False, metadata=metadata
            )

        return edge

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """
        Remove an edge from the graph.

        Returns:
            True if edge was removed
        """
        if source_id in self._edges and target_id in self._edges[source_id]:
            del self._edges[source_id][target_id]
            self._reverse_edges[target_id].discard(source_id)
            self._edge_count -= 1
            self._invalidate_cache()

            if self._emit_events:
                bus = get_event_bus()
                bus.emit("graph.edge_removed", {
                    "source_id": source_id,
                    "target_id": target_id
                })

            return True
        return False

    def get_edge(self, source_id: str, target_id: str) -> Optional[GraphEdge]:
        """Get an edge between two organisms"""
        return self._edges.get(source_id, {}).get(target_id)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if edge exists"""
        return target_id in self._edges.get(source_id, {})

    def get_neighbors(self, organism_id: str, direction: str = "out") -> Set[str]:
        """
        Get neighboring organisms.

        Args:
            organism_id: Target organism
            direction: "out" for outgoing, "in" for incoming, "all" for both
        """
        neighbors = set()

        if direction in ("out", "all"):
            neighbors.update(self._edges.get(organism_id, {}).keys())

        if direction in ("in", "all"):
            neighbors.update(self._reverse_edges.get(organism_id, set()))

        return neighbors

    def get_outgoing_edges(self, organism_id: str) -> List[GraphEdge]:
        """Get all outgoing edges from an organism"""
        return list(self._edges.get(organism_id, {}).values())

    def get_incoming_edges(self, organism_id: str) -> List[GraphEdge]:
        """Get all incoming edges to an organism"""
        edges = []
        for source_id in self._reverse_edges.get(organism_id, set()):
            edge = self.get_edge(source_id, organism_id)
            if edge:
                edges.append(edge)
        return edges

    # =========================================================================
    # Centrality Calculations
    # =========================================================================

    def calculate_centrality(self, organism_id: str) -> GraphMetrics:
        """
        Calculate centrality metrics for an organism.

        Args:
            organism_id: Target organism

        Returns:
            GraphMetrics with centrality scores
        """
        if organism_id not in self._nodes:
            raise ValueError(f"Organism {organism_id} not in graph")

        # Use cache if valid
        if self._cache_valid and organism_id in self._metrics_cache:
            return self._metrics_cache[organism_id]

        n = len(self._nodes)
        if n <= 1:
            return GraphMetrics(organism_id=organism_id)

        metrics = GraphMetrics(organism_id=organism_id)

        # Degree centrality
        out_degree = len(self._edges.get(organism_id, {}))
        in_degree = len(self._reverse_edges.get(organism_id, set()))
        metrics.out_degree = out_degree
        metrics.in_degree = in_degree
        metrics.degree_centrality = (out_degree + in_degree) / (2 * (n - 1))

        # Closeness centrality (inverse sum of distances)
        distances = self._dijkstra_distances(organism_id)
        reachable = [d for d in distances.values() if d < float('inf')]
        if reachable:
            metrics.closeness_centrality = len(reachable) / sum(reachable) if sum(reachable) > 0 else 0

        # Clustering coefficient
        neighbors = self.get_neighbors(organism_id, direction="all")
        if len(neighbors) >= 2:
            neighbor_edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and self.has_edge(n1, n2):
                        neighbor_edges += 1
            possible_edges = len(neighbors) * (len(neighbors) - 1)
            metrics.clustering_coefficient = neighbor_edges / possible_edges

        self._metrics_cache[organism_id] = metrics
        return metrics

    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all nodes.

        Returns:
            Dictionary mapping organism IDs to betweenness scores
        """
        betweenness = defaultdict(float)
        nodes = list(self._nodes.keys())

        for source in nodes:
            # BFS for shortest paths
            distances = {source: 0}
            num_paths = {source: 1}
            predecessors: Dict[str, List[str]] = defaultdict(list)
            queue = [source]
            order = []

            while queue:
                current = queue.pop(0)
                order.append(current)

                for neighbor in self.get_neighbors(current, "out"):
                    # First visit
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)

                    # Shortest path
                    if distances[neighbor] == distances[current] + 1:
                        num_paths[neighbor] = num_paths.get(neighbor, 0) + num_paths[current]
                        predecessors[neighbor].append(current)

            # Accumulate dependencies
            dependencies = defaultdict(float)
            for node in reversed(order):
                for pred in predecessors[node]:
                    if num_paths[node] > 0:
                        dependencies[pred] += (num_paths[pred] / num_paths[node]) * (1 + dependencies[node])
                if node != source:
                    betweenness[node] += dependencies[node]

        # Normalize
        n = len(nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            betweenness = {k: v * norm for k, v in betweenness.items()}

        return dict(betweenness)

    def calculate_pagerank(self, damping: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """
        Calculate PageRank for all nodes.

        Args:
            damping: Damping factor (0.85 default)
            max_iter: Maximum iterations

        Returns:
            Dictionary mapping organism IDs to PageRank scores
        """
        nodes = list(self._nodes.keys())
        n = len(nodes)

        if n == 0:
            return {}

        # Initialize
        pagerank = {node: 1.0 / n for node in nodes}

        for _ in range(max_iter):
            new_pagerank = {}

            for node in nodes:
                rank = (1 - damping) / n

                for source in self._reverse_edges.get(node, set()):
                    out_degree = len(self._edges.get(source, {}))
                    if out_degree > 0:
                        rank += damping * pagerank[source] / out_degree

                new_pagerank[node] = rank

            # Check convergence
            diff = sum(abs(new_pagerank[n] - pagerank[n]) for n in nodes)
            pagerank = new_pagerank

            if diff < 1e-6:
                break

        return pagerank

    def _dijkstra_distances(self, source: str) -> Dict[str, float]:
        """Calculate shortest distances from source to all nodes"""
        distances = {node: float('inf') for node in self._nodes}
        distances[source] = 0

        heap = [(0, source)]
        visited = set()

        while heap:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            for neighbor, edge in self._edges.get(current, {}).items():
                # Use inverse weight as distance
                new_dist = dist + (1 - edge.weight + 0.1)
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

        return distances

    # =========================================================================
    # Community Detection
    # =========================================================================

    def detect_communities(self, resolution: float = 1.0) -> List[Community]:
        """
        Detect communities using a greedy modularity optimization.

        Args:
            resolution: Resolution parameter (higher = smaller communities)

        Returns:
            List of detected communities
        """
        if not self._nodes:
            return []

        # Initialize: each node is its own community
        node_to_community: Dict[str, int] = {
            node: i for i, node in enumerate(self._nodes.keys())
        }

        # Total edge weight
        total_weight = sum(
            edge.weight
            for edges in self._edges.values()
            for edge in edges.values()
        )

        if total_weight == 0:
            # No edges, each node is its own community
            communities = []
            for node_id in self._nodes:
                comm = Community(members={node_id})
                communities.append(comm)
            return communities

        # Greedy optimization
        improved = True
        while improved:
            improved = False

            for node in self._nodes:
                current_comm = node_to_community[node]

                # Calculate neighbors' communities
                neighbor_comms: Dict[int, float] = defaultdict(float)
                for neighbor, edge in self._edges.get(node, {}).items():
                    neighbor_comms[node_to_community[neighbor]] += edge.weight

                for source in self._reverse_edges.get(node, set()):
                    edge = self.get_edge(source, node)
                    if edge:
                        neighbor_comms[node_to_community[source]] += edge.weight

                # Find best community
                best_comm = current_comm
                best_gain = 0.0

                for comm, weight in neighbor_comms.items():
                    if comm != current_comm:
                        gain = weight - resolution * self._community_degree_product(
                            node, comm, node_to_community, total_weight
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_comm = comm

                if best_comm != current_comm:
                    node_to_community[node] = best_comm
                    improved = True

        # Convert to Community objects
        comm_members: Dict[int, Set[str]] = defaultdict(set)
        for node, comm_id in node_to_community.items():
            comm_members[comm_id].add(node)

        communities = []
        for comm_id, members in comm_members.items():
            community = Community(members=members)
            community.density = self._calculate_community_density(members)
            communities.append(community)

            # Update mappings
            self._communities[community.id] = community
            for member in members:
                self._node_community[member] = community.id

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("graph.communities_detected", {
                "num_communities": len(communities),
                "sizes": [c.size for c in communities]
            })

        return communities

    def _community_degree_product(
        self,
        node: str,
        community: int,
        node_to_community: Dict[str, int],
        total_weight: float
    ) -> float:
        """Calculate degree product term for modularity"""
        node_degree = (
            len(self._edges.get(node, {})) +
            len(self._reverse_edges.get(node, set()))
        )

        comm_degree = sum(
            len(self._edges.get(n, {})) + len(self._reverse_edges.get(n, set()))
            for n, c in node_to_community.items()
            if c == community
        )

        return (node_degree * comm_degree) / (2 * total_weight) if total_weight > 0 else 0

    def _calculate_community_density(self, members: Set[str]) -> float:
        """Calculate internal edge density of a community"""
        if len(members) < 2:
            return 1.0

        internal_edges = 0
        for member in members:
            for neighbor in self._edges.get(member, {}).keys():
                if neighbor in members:
                    internal_edges += 1

        max_edges = len(members) * (len(members) - 1)
        return internal_edges / max_edges if max_edges > 0 else 0

    def get_organism_community(self, organism_id: str) -> Optional[Community]:
        """Get the community an organism belongs to"""
        comm_id = self._node_community.get(organism_id)
        if comm_id:
            return self._communities.get(comm_id)
        return None

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between two organisms.

        Returns:
            List of organism IDs in path, or None if no path exists
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        if source_id == target_id:
            return [source_id]

        # BFS
        queue = [(source_id, [source_id])]
        visited = {source_id}

        while queue:
            current, path = queue.pop(0)

            for neighbor in self.get_neighbors(current, "out"):
                if neighbor == target_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def calculate_network_density(self) -> float:
        """Calculate overall network density"""
        n = len(self._nodes)
        if n < 2:
            return 0.0

        max_edges = n * (n - 1)
        return self._edge_count / max_edges

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if not self._nodes:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "num_communities": 0,
                "avg_degree": 0.0
            }

        total_degree = sum(
            len(self._edges.get(n, {})) + len(self._reverse_edges.get(n, set()))
            for n in self._nodes
        )

        return {
            "num_nodes": len(self._nodes),
            "num_edges": self._edge_count,
            "density": self.calculate_network_density(),
            "num_communities": len(self._communities),
            "avg_degree": total_degree / len(self._nodes) if self._nodes else 0
        }

    def _invalidate_cache(self) -> None:
        """Invalidate cached metrics"""
        self._cache_valid = False
        self._metrics_cache.clear()

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        nodes = [
            {
                "organism_id": node.organism_id,
                "attributes": node.attributes,
                "metadata": node.metadata
            }
            for node in self._nodes.values()
        ]

        edges = [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "weight": edge.weight,
                "edge_type": edge.edge_type.value,
                "interaction_count": edge.interaction_count,
                "metadata": edge.metadata
            }
            for source_edges in self._edges.values()
            for edge in source_edges.values()
        ]

        return {
            "nodes": nodes,
            "edges": edges
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'SocialGraph':
        """Deserialize graph from dictionary"""
        graph = cls(emit_events=emit_events)

        for node_data in data.get("nodes", []):
            graph.add_organism(
                node_data["organism_id"],
                attributes=node_data.get("attributes", {}),
                metadata=node_data.get("metadata", {})
            )

        for edge_data in data.get("edges", []):
            edge_type = EdgeType(edge_data.get("edge_type", "neutral"))
            graph.add_edge(
                edge_data["source_id"],
                edge_data["target_id"],
                weight=edge_data.get("weight", 0.5),
                edge_type=edge_type,
                metadata=edge_data.get("metadata", {})
            )

        return graph
