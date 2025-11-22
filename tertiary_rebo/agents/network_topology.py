"""
NetworkTopologyAgent - Manages mycelium-like network topology and routing.

Responsibilities:
- Maintain the ChicagoForest-inspired mesh network topology
- Optimize routing paths between consciousness nodes
- Manage spore propagation and signal distribution
- Ensure decentralized communication integrity
- Implement energy-democratic resource sharing
- Self-healing and fault tolerance
- Geodesic routing on curved manifolds
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum, auto
import heapq
import uuid
import asyncio

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal,
)


class HyphaType(Enum):
    """Types of mycelial connections."""
    RUNNER = auto()      # Long-distance exploration hyphae
    CONNECTOR = auto()   # Short bridging connections
    NUTRIENT = auto()    # Resource transport channels
    SIGNAL = auto()      # Fast signal transmission paths
    ANCHOR = auto()      # Stable, high-capacity connections


class NodeState(Enum):
    """Health states for network nodes."""
    HEALTHY = auto()
    STRESSED = auto()
    RECOVERING = auto()
    DORMANT = auto()
    FAILED = auto()


@dataclass
class MyceliumNode:
    """A node in the mycelium network (consciousness center)."""
    node_id: int
    position: np.ndarray  # Position in manifold space
    nutrient_level: float = 0.5
    signal_capacity: float = 1.0
    state: NodeState = NodeState.HEALTHY
    last_activity: datetime = field(default_factory=datetime.now)
    connections: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Mycelium-specific properties
    spore_production_rate: float = 0.1
    growth_potential: float = 0.5
    local_consciousness: float = 0.0


@dataclass
class Hypha:
    """A connection (edge) in the mycelium network."""
    hypha_id: str
    source: int
    target: int
    hypha_type: HyphaType
    weight: float = 0.5
    bandwidth: float = 1.0
    latency: float = 0.1
    age: int = 0
    last_used: datetime = field(default_factory=datetime.now)

    # Flow properties
    current_flow: float = 0.0
    max_flow: float = 1.0

    def decay(self, rate: float = 0.01) -> float:
        """Apply natural decay to unused connections."""
        self.weight *= (1.0 - rate)
        self.age += 1
        return self.weight

    def strengthen(self, amount: float = 0.1) -> float:
        """Strengthen connection from use."""
        self.weight = min(1.0, self.weight + amount)
        self.last_used = datetime.now()
        return self.weight


@dataclass
class Spore:
    """A signal/message that propagates through the network."""
    spore_id: str
    origin_node: int
    payload: Dict[str, Any]
    signal_type: str
    strength: float = 1.0
    ttl: int = 10  # Time to live (hops)
    path: List[int] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)

    def propagate(self) -> bool:
        """Propagate spore, reducing TTL and strength."""
        self.ttl -= 1
        self.strength *= 0.9
        return self.ttl > 0 and self.strength > 0.1


@dataclass
class RoutingEntry:
    """Entry in the routing table."""
    destination: int
    next_hop: int
    distance: float
    path: List[int]
    last_updated: datetime = field(default_factory=datetime.now)
    reliability: float = 1.0


class NetworkTopologyAgent(TertiaryReBoAgent):
    """
    Agent 2: Manages mycelium-like network topology.

    Inspired by ChicagoForest.net's vision of decentralized mesh networks,
    this agent treats the TTR system as a mycelium-like organism where:
    - Nodes represent consciousness centers
    - Hyphae represent signal pathways (edges)
    - Nutrients (energy/information) flow through the network
    - The network self-organizes, grows, and heals
    - Spores broadcast signals across the mesh

    Key concepts:
    - Hyphae: Direct connections between nodes (various types)
    - Mycelium mat: The overall network structure
    - Spore propagation: Broadcasting signals across the network
    - Nutrient exchange: Resource balancing between nodes
    - Geodesic routing: Optimal paths on curved manifolds
    - Self-healing: Automatic fault detection and recovery
    """

    def __init__(
        self,
        num_nodes: int = 64,
        manifold_dimension: int = 4,
        **kwargs
    ):
        super().__init__(primary_domain=DomainLeg.CHICAGO_FOREST, **kwargs)

        self.num_nodes = num_nodes
        self.manifold_dimension = manifold_dimension

        # Node management
        self.nodes: Dict[int, MyceliumNode] = {}
        self._initialize_nodes()

        # Hyphal network (edges)
        self.hyphae: Dict[str, Hypha] = {}
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        self.edge_weights = np.zeros((num_nodes, num_nodes))

        # Routing (initialize before topology so metrics can be computed)
        self.routing_table: Dict[int, Dict[int, RoutingEntry]] = defaultdict(dict)
        self.path_cache: Dict[Tuple[int, int], List[int]] = {}
        self.routing_update_interval = 10  # Cycles between full routing updates
        self.cycles_since_routing_update = 0

        # Network health metrics (initialize before topology creation)
        self.connectivity_score = 0.5
        self.redundancy_factor = 1.0
        self.average_path_length = 0.0
        self.clustering_coefficient = 0.0

        # Fault tolerance
        self.failed_nodes: Set[int] = set()
        self.healing_queue: List[Tuple[int, int]] = []  # (priority, node_id)

        # Initialize with small-world topology
        self._initialize_small_world_topology()

        # Spore propagation
        self.active_spores: List[Spore] = []
        self.spore_history: deque = deque(maxlen=1000)

        # Mycelium growth parameters
        self.growth_rate = 0.1
        self.decay_rate = 0.02
        self.healing_rate = 0.05
        self.exploration_factor = 0.3  # Tendency to explore vs exploit

        # Energy democracy
        self.total_energy = float(num_nodes)
        self.energy_flow_matrix = np.zeros((num_nodes, num_nodes))

    def _initialize_nodes(self):
        """Initialize network nodes with random positions on manifold."""
        for i in range(self.num_nodes):
            # Position on hyperbolic manifold (Poincaré ball model)
            # Keep points within unit ball with some margin
            position = np.random.randn(self.manifold_dimension)
            position = position / np.linalg.norm(position) * np.random.uniform(0.1, 0.8)

            self.nodes[i] = MyceliumNode(
                node_id=i,
                position=position,
                nutrient_level=np.random.uniform(0.3, 0.7),
                growth_potential=np.random.uniform(0.3, 0.7),
                local_consciousness=np.random.uniform(0.1, 0.3)
            )

    def _initialize_small_world_topology(self):
        """Initialize with Watts-Strogatz small-world topology."""
        k = 4  # Each node connected to k nearest neighbors
        p = 0.1  # Rewiring probability

        # Create ring lattice
        for i in range(self.num_nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % self.num_nodes
                self._create_hypha(i, neighbor, HyphaType.CONNECTOR)

        # Rewire with probability p
        for i in range(self.num_nodes):
            for j in range(1, k // 2 + 1):
                if np.random.random() < p:
                    neighbor = (i + j) % self.num_nodes
                    # Remove old connection
                    hypha_id = self._get_hypha_id(i, neighbor)
                    if hypha_id in self.hyphae:
                        del self.hyphae[hypha_id]
                        self.adjacency_matrix[i, neighbor] = 0
                        self.adjacency_matrix[neighbor, i] = 0

                    # Add new random connection
                    new_neighbor = np.random.randint(0, self.num_nodes)
                    while new_neighbor == i or self.adjacency_matrix[i, new_neighbor] > 0:
                        new_neighbor = np.random.randint(0, self.num_nodes)

                    self._create_hypha(i, new_neighbor, HyphaType.RUNNER)

        # Update connectivity score
        self._update_network_metrics()

    def _get_hypha_id(self, source: int, target: int) -> str:
        """Generate consistent hypha ID for a connection."""
        return f"hypha_{min(source, target)}_{max(source, target)}"

    def _create_hypha(
        self,
        source: int,
        target: int,
        hypha_type: HyphaType,
        weight: float = 0.5
    ) -> Hypha:
        """Create a new hyphal connection between nodes."""
        hypha_id = self._get_hypha_id(source, target)

        # Calculate initial properties based on node positions
        distance = self._geodesic_distance(source, target)
        latency = distance * 0.1
        bandwidth = 1.0 / (1.0 + distance)

        hypha = Hypha(
            hypha_id=hypha_id,
            source=source,
            target=target,
            hypha_type=hypha_type,
            weight=weight,
            bandwidth=bandwidth,
            latency=latency
        )

        self.hyphae[hypha_id] = hypha
        self.adjacency_matrix[source, target] = 1
        self.adjacency_matrix[target, source] = 1
        self.edge_weights[source, target] = weight
        self.edge_weights[target, source] = weight

        # Update node connections
        self.nodes[source].connections.add(target)
        self.nodes[target].connections.add(source)

        return hypha

    def _geodesic_distance(self, node1: int, node2: int) -> float:
        """Calculate geodesic distance between nodes on hyperbolic manifold."""
        p1 = self.nodes[node1].position
        p2 = self.nodes[node2].position

        # Hyperbolic distance in Poincaré ball model
        norm1_sq = np.sum(p1 ** 2)
        norm2_sq = np.sum(p2 ** 2)
        diff_sq = np.sum((p1 - p2) ** 2)

        # Avoid numerical issues
        norm1_sq = np.clip(norm1_sq, 0, 0.9999)
        norm2_sq = np.clip(norm2_sq, 0, 0.9999)

        cosh_dist = 1 + 2 * diff_sq / ((1 - norm1_sq) * (1 - norm2_sq))
        return np.arccosh(np.clip(cosh_dist, 1, None))

    def _update_network_metrics(self):
        """Update network health metrics."""
        # Connectivity score
        connected_pairs = np.sum(self.adjacency_matrix > 0)
        total_pairs = self.num_nodes * (self.num_nodes - 1)
        self.connectivity_score = connected_pairs / total_pairs if total_pairs > 0 else 0

        # Calculate clustering coefficient
        self.clustering_coefficient = self._calculate_clustering_coefficient()

        # Estimate average path length
        self.average_path_length = self._estimate_average_path_length()

        # Redundancy factor (average node degree)
        degrees = np.sum(self.adjacency_matrix > 0, axis=1)
        self.redundancy_factor = np.mean(degrees)

    def _calculate_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient."""
        total_cc = 0.0
        valid_nodes = 0

        for i in range(self.num_nodes):
            neighbors = list(self.nodes[i].connections)
            k = len(neighbors)

            if k < 2:
                continue

            # Count edges between neighbors
            edges = 0
            for j, n1 in enumerate(neighbors):
                for n2 in neighbors[j + 1:]:
                    if self.adjacency_matrix[n1, n2] > 0:
                        edges += 1

            max_edges = k * (k - 1) / 2
            total_cc += edges / max_edges if max_edges > 0 else 0
            valid_nodes += 1

        return total_cc / valid_nodes if valid_nodes > 0 else 0

    def _estimate_average_path_length(self) -> float:
        """Estimate average shortest path length using sampling."""
        if self.connectivity_score < 0.01:
            return float('inf')

        sample_size = min(20, self.num_nodes)
        sample_nodes = np.random.choice(self.num_nodes, sample_size, replace=False)

        total_distance = 0.0
        valid_pairs = 0

        for i in sample_nodes:
            for j in sample_nodes:
                if i != j:
                    path = self._find_shortest_path(i, j)
                    if path:
                        total_distance += len(path) - 1
                        valid_pairs += 1

        return total_distance / valid_pairs if valid_pairs > 0 else float('inf')

    def _find_shortest_path(
        self,
        source: int,
        target: int,
        use_cache: bool = True
    ) -> Optional[List[int]]:
        """Find shortest path using Dijkstra's algorithm on geodesic distances."""
        cache_key = (source, target)
        if use_cache and cache_key in self.path_cache:
            return self.path_cache[cache_key]

        if source in self.failed_nodes or target in self.failed_nodes:
            return None

        # Dijkstra's algorithm
        distances = {i: float('inf') for i in range(self.num_nodes)}
        distances[source] = 0
        previous = {i: None for i in range(self.num_nodes)}

        # Priority queue: (distance, node)
        pq = [(0, source)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == target:
                break

            if current in self.failed_nodes:
                continue

            for neighbor in self.nodes[current].connections:
                if neighbor in visited or neighbor in self.failed_nodes:
                    continue

                # Use geodesic distance weighted by edge weight
                hypha_id = self._get_hypha_id(current, neighbor)
                if hypha_id in self.hyphae:
                    hypha = self.hyphae[hypha_id]
                    edge_cost = hypha.latency / hypha.weight
                else:
                    edge_cost = self._geodesic_distance(current, neighbor)

                new_dist = current_dist + edge_cost

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if previous[target] is None and target != source:
            return None

        path = []
        current = target
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        # Cache result
        if use_cache:
            self.path_cache[cache_key] = path

        return path

    def _update_routing_table(self):
        """Update routing table for all node pairs."""
        self.path_cache.clear()
        self.routing_table.clear()

        for source in range(self.num_nodes):
            if source in self.failed_nodes:
                continue

            # Run Dijkstra from this source
            distances = {i: float('inf') for i in range(self.num_nodes)}
            distances[source] = 0
            previous = {i: None for i in range(self.num_nodes)}

            pq = [(0, source)]
            visited = set()

            while pq:
                current_dist, current = heapq.heappop(pq)

                if current in visited:
                    continue
                visited.add(current)

                for neighbor in self.nodes[current].connections:
                    if neighbor in visited or neighbor in self.failed_nodes:
                        continue

                    hypha_id = self._get_hypha_id(current, neighbor)
                    if hypha_id in self.hyphae:
                        hypha = self.hyphae[hypha_id]
                        edge_cost = hypha.latency / hypha.weight
                    else:
                        edge_cost = self._geodesic_distance(current, neighbor)

                    new_dist = current_dist + edge_cost

                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))

            # Build routing entries
            for dest in range(self.num_nodes):
                if dest == source or distances[dest] == float('inf'):
                    continue

                # Find next hop
                path = []
                current = dest
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()

                if len(path) >= 2:
                    self.routing_table[source][dest] = RoutingEntry(
                        destination=dest,
                        next_hop=path[1],
                        distance=distances[dest],
                        path=path
                    )

    @property
    def agent_role(self) -> str:
        return "Network Topology - Manages mycelium-inspired mesh connectivity, geodesic routing, and self-healing"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return [DomainLeg.CHICAGO_FOREST, DomainLeg.NPCPU]

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze current network topology, health, and connectivity."""
        cf_state = tbl.get_state(DomainLeg.CHICAGO_FOREST)

        # Update metrics
        self._update_network_metrics()

        # Node health analysis
        healthy_nodes = sum(1 for n in self.nodes.values() if n.state == NodeState.HEALTHY)
        stressed_nodes = sum(1 for n in self.nodes.values() if n.state == NodeState.STRESSED)
        failed_count = len(self.failed_nodes)

        # Identify isolated nodes (no connections)
        isolated_nodes = [
            n.node_id for n in self.nodes.values()
            if len(n.connections) == 0 and n.node_id not in self.failed_nodes
        ]

        # Identify bottleneck nodes (high betweenness centrality)
        betweenness = self._estimate_betweenness_centrality()
        bottleneck_threshold = np.percentile(list(betweenness.values()), 90)
        bottleneck_nodes = [
            node for node, score in betweenness.items()
            if score >= bottleneck_threshold
        ]

        # Nutrient distribution analysis
        nutrient_levels = np.array([n.nutrient_level for n in self.nodes.values()])
        nutrient_gini = self._calculate_gini_coefficient(nutrient_levels)

        # Consciousness distribution
        consciousness_levels = np.array([n.local_consciousness for n in self.nodes.values()])

        # Hyphal health
        weak_hyphae = [
            h for h in self.hyphae.values()
            if h.weight < 0.2
        ]
        strong_hyphae = [
            h for h in self.hyphae.values()
            if h.weight > 0.7
        ]

        # Spore activity
        active_spore_count = len(self.active_spores)

        return {
            "connectivity_score": self.connectivity_score,
            "clustering_coefficient": self.clustering_coefficient,
            "average_path_length": self.average_path_length,
            "redundancy_factor": self.redundancy_factor,

            "node_health": {
                "healthy": healthy_nodes,
                "stressed": stressed_nodes,
                "failed": failed_count,
                "isolated": isolated_nodes,
            },

            "bottleneck_nodes": bottleneck_nodes,
            "betweenness_scores": betweenness,

            "nutrient_distribution": {
                "mean": float(np.mean(nutrient_levels)),
                "std": float(np.std(nutrient_levels)),
                "gini": nutrient_gini,
                "min": float(np.min(nutrient_levels)),
                "max": float(np.max(nutrient_levels)),
            },

            "consciousness_distribution": {
                "mean": float(np.mean(consciousness_levels)),
                "std": float(np.std(consciousness_levels)),
                "total": float(np.sum(consciousness_levels)),
            },

            "hyphal_health": {
                "total": len(self.hyphae),
                "weak": len(weak_hyphae),
                "strong": len(strong_hyphae),
            },

            "spore_activity": active_spore_count,
            "network_signature": cf_state.state_vector,
        }

    def _estimate_betweenness_centrality(self) -> Dict[int, float]:
        """Estimate betweenness centrality for all nodes."""
        betweenness = {i: 0.0 for i in range(self.num_nodes)}

        # Sample paths for efficiency
        sample_size = min(20, self.num_nodes)
        sample_nodes = np.random.choice(
            [i for i in range(self.num_nodes) if i not in self.failed_nodes],
            min(sample_size, self.num_nodes - len(self.failed_nodes)),
            replace=False
        )

        for source in sample_nodes:
            for target in sample_nodes:
                if source != target:
                    path = self._find_shortest_path(source, target, use_cache=False)
                    if path and len(path) > 2:
                        for node in path[1:-1]:  # Exclude endpoints
                            betweenness[node] += 1

        # Normalize
        max_betweenness = max(betweenness.values()) if betweenness else 1
        if max_betweenness > 0:
            betweenness = {k: v / max_betweenness for k, v in betweenness.items()}

        return betweenness

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for distribution inequality."""
        if len(values) == 0:
            return 0.0

        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
        return float(np.clip(gini, 0, 1))

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Identify network topology issues and optimization opportunities."""
        analysis = {
            "issues": [],
            "growth_opportunities": [],
            "healing_needed": [],
            "optimization_targets": [],
            "energy_rebalancing": [],
        }

        # Check connectivity issues
        if perception["connectivity_score"] < 0.3:
            analysis["issues"].append({
                "type": "low_connectivity",
                "severity": "critical",
                "current": perception["connectivity_score"],
                "target": 0.4,
            })

        # Check for isolated nodes
        isolated = perception["node_health"]["isolated"]
        if isolated:
            for node_id in isolated[:10]:  # Limit processing
                # Find nearest healthy nodes
                healthy_neighbors = self._find_nearest_healthy_nodes(node_id, k=3)
                analysis["growth_opportunities"].append({
                    "type": "reconnect_isolated",
                    "node_id": node_id,
                    "targets": healthy_neighbors,
                    "priority": 1.0,
                })

        # Check for failed nodes needing healing
        if perception["node_health"]["failed"] > 0:
            for node_id in self.failed_nodes:
                analysis["healing_needed"].append({
                    "node_id": node_id,
                    "priority": 0.8,
                    "strategy": "respawn_and_reconnect",
                })

        # Check for bottlenecks
        for node_id in perception["bottleneck_nodes"]:
            # Get neighbors to create bypass routes
            node = self.nodes[node_id]
            neighbors = list(node.connections)

            if len(neighbors) >= 2:
                analysis["optimization_targets"].append({
                    "type": "reduce_bottleneck",
                    "node_id": node_id,
                    "neighbors": neighbors[:5],
                    "strategy": "create_bypass",
                })

        # Check nutrient inequality (energy democracy)
        if perception["nutrient_distribution"]["gini"] > 0.3:
            # Find rich and poor nodes
            rich_nodes = [
                n.node_id for n in self.nodes.values()
                if n.nutrient_level > perception["nutrient_distribution"]["mean"] + perception["nutrient_distribution"]["std"]
            ]
            poor_nodes = [
                n.node_id for n in self.nodes.values()
                if n.nutrient_level < perception["nutrient_distribution"]["mean"] - perception["nutrient_distribution"]["std"]
            ]

            analysis["energy_rebalancing"].append({
                "type": "redistribute",
                "from_nodes": rich_nodes[:5],
                "to_nodes": poor_nodes[:5],
                "gini": perception["nutrient_distribution"]["gini"],
            })

        # Check hyphal health
        if perception["hyphal_health"]["weak"] > perception["hyphal_health"]["total"] * 0.3:
            analysis["issues"].append({
                "type": "weak_connections",
                "severity": "medium",
                "weak_count": perception["hyphal_health"]["weak"],
                "action": "strengthen_or_prune",
            })

        # Check clustering (small-world property)
        if perception["clustering_coefficient"] < 0.2:
            analysis["optimization_targets"].append({
                "type": "increase_clustering",
                "current": perception["clustering_coefficient"],
                "target": 0.3,
                "strategy": "triangulation",
            })

        return analysis

    def _find_nearest_healthy_nodes(self, node_id: int, k: int = 3) -> List[int]:
        """Find k nearest healthy nodes to a given node."""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        distances = []

        for other_id, other_node in self.nodes.items():
            if other_id != node_id and other_node.state == NodeState.HEALTHY:
                dist = self._geodesic_distance(node_id, other_id)
                distances.append((dist, other_id))

        distances.sort(key=lambda x: x[0])
        return [node_id for _, node_id in distances[:k]]

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Create network topology refinement plan."""
        synthesis = {
            "new_connections": [],
            "strengthened_connections": [],
            "pruned_connections": [],
            "healed_nodes": [],
            "nutrient_transfers": [],
            "spore_broadcasts": [],
            "routing_updates": False,
        }

        # Plan reconnections for isolated nodes
        for opportunity in analysis.get("growth_opportunities", []):
            if opportunity["type"] == "reconnect_isolated":
                node_id = opportunity["node_id"]
                for target in opportunity["targets"]:
                    synthesis["new_connections"].append({
                        "source": node_id,
                        "target": target,
                        "type": HyphaType.CONNECTOR,
                        "initial_weight": 0.3,
                    })

                # Create spore to announce reconnection
                synthesis["spore_broadcasts"].append({
                    "origin": node_id,
                    "signal_type": "reconnection",
                    "payload": {"node_id": node_id},
                    "ttl": 5,
                })

        # Plan node healing
        for healing in analysis.get("healing_needed", []):
            node_id = healing["node_id"]
            synthesis["healed_nodes"].append({
                "node_id": node_id,
                "strategy": healing["strategy"],
            })

        # Plan bottleneck reduction
        for target in analysis.get("optimization_targets", []):
            if target["type"] == "reduce_bottleneck":
                neighbors = target["neighbors"]
                # Create bypass routes between neighbors
                for i in range(min(3, len(neighbors) - 1)):
                    n1, n2 = neighbors[i], neighbors[i + 1]
                    if self.adjacency_matrix[n1, n2] == 0:
                        synthesis["new_connections"].append({
                            "source": n1,
                            "target": n2,
                            "type": HyphaType.RUNNER,
                            "initial_weight": 0.4,
                            "reason": "bypass_bottleneck",
                        })

            elif target["type"] == "increase_clustering":
                # Add triangulation connections
                self._plan_triangulation(synthesis)

        # Plan energy redistribution
        for rebalance in analysis.get("energy_rebalancing", []):
            from_nodes = rebalance["from_nodes"]
            to_nodes = rebalance["to_nodes"]

            for from_node in from_nodes:
                for to_node in to_nodes:
                    # Check if path exists
                    path = self._find_shortest_path(from_node, to_node)
                    if path:
                        amount = min(
                            0.1,
                            (self.nodes[from_node].nutrient_level - 0.5) * 0.2
                        )
                        if amount > 0.01:
                            synthesis["nutrient_transfers"].append({
                                "from": from_node,
                                "to": to_node,
                                "amount": amount,
                                "path": path,
                            })

        # Check if routing table needs update
        self.cycles_since_routing_update += 1
        if self.cycles_since_routing_update >= self.routing_update_interval:
            synthesis["routing_updates"] = True
            self.cycles_since_routing_update = 0

        return synthesis

    def _plan_triangulation(self, synthesis: Dict[str, Any]):
        """Plan connections to increase clustering coefficient."""
        # Find nodes with low local clustering
        for node_id in range(min(10, self.num_nodes)):  # Limit for efficiency
            node = self.nodes[node_id]
            neighbors = list(node.connections)

            if len(neighbors) < 2:
                continue

            # Find pairs of neighbors not connected
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1:]:
                    if self.adjacency_matrix[n1, n2] == 0:
                        synthesis["new_connections"].append({
                            "source": n1,
                            "target": n2,
                            "type": HyphaType.CONNECTOR,
                            "initial_weight": 0.25,
                            "reason": "triangulation",
                        })
                        return  # Add one at a time

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply network topology changes and propagate signals."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        cf_state = tbl.get_state(DomainLeg.CHICAGO_FOREST)

        # Create new connections
        new_count = 0
        for conn in synthesis.get("new_connections", []):
            source, target = conn["source"], conn["target"]

            if source not in self.failed_nodes and target not in self.failed_nodes:
                hypha = self._create_hypha(
                    source, target,
                    conn["type"],
                    conn["initial_weight"]
                )
                new_count += 1

        if new_count > 0:
            changes["new_hyphae"] = new_count
            insights.append(f"Grew {new_count} new mycelium connections")

        # Heal nodes
        healed_count = 0
        for healing in synthesis.get("healed_nodes", []):
            node_id = healing["node_id"]
            if node_id in self.failed_nodes:
                self.failed_nodes.remove(node_id)
                self.nodes[node_id].state = NodeState.RECOVERING
                self.nodes[node_id].nutrient_level = 0.3

                # Reconnect to nearest healthy nodes
                nearest = self._find_nearest_healthy_nodes(node_id, k=3)
                for neighbor in nearest:
                    self._create_hypha(node_id, neighbor, HyphaType.CONNECTOR, 0.3)

                healed_count += 1

        if healed_count > 0:
            changes["healed_nodes"] = healed_count
            insights.append(f"Healed {healed_count} failed nodes")

        # Execute nutrient transfers (energy democracy)
        total_transferred = 0.0
        for transfer in synthesis.get("nutrient_transfers", []):
            from_node = transfer["from"]
            to_node = transfer["to"]
            amount = transfer["amount"]

            # Transfer with some loss along path
            path_length = len(transfer["path"])
            efficiency = 0.95 ** (path_length - 1)  # 5% loss per hop
            actual_transfer = amount * efficiency

            self.nodes[from_node].nutrient_level -= amount
            self.nodes[to_node].nutrient_level += actual_transfer
            total_transferred += actual_transfer

            # Strengthen hyphae along path
            for i in range(len(transfer["path"]) - 1):
                hypha_id = self._get_hypha_id(transfer["path"][i], transfer["path"][i + 1])
                if hypha_id in self.hyphae:
                    self.hyphae[hypha_id].strengthen(0.05)

        if total_transferred > 0:
            changes["nutrients_redistributed"] = total_transferred
            insights.append(f"Redistributed {total_transferred:.3f} nutrients (energy democracy)")

        # Broadcast spores
        for spore_config in synthesis.get("spore_broadcasts", []):
            spore = Spore(
                spore_id=f"spore_{uuid.uuid4().hex[:8]}",
                origin_node=spore_config["origin"],
                payload=spore_config["payload"],
                signal_type=spore_config["signal_type"],
                ttl=spore_config.get("ttl", 10)
            )
            self.active_spores.append(spore)

        # Propagate existing spores
        await self._propagate_spores()

        # Update routing if needed
        if synthesis.get("routing_updates", False):
            self._update_routing_table()
            insights.append("Updated geodesic routing tables")

        # Apply natural decay to unused hyphae
        pruned_count = 0
        hyphae_to_remove = []
        for hypha_id, hypha in self.hyphae.items():
            hypha.decay(self.decay_rate)
            if hypha.weight < 0.05:
                hyphae_to_remove.append(hypha_id)

        for hypha_id in hyphae_to_remove:
            hypha = self.hyphae[hypha_id]
            self.adjacency_matrix[hypha.source, hypha.target] = 0
            self.adjacency_matrix[hypha.target, hypha.source] = 0
            self.nodes[hypha.source].connections.discard(hypha.target)
            self.nodes[hypha.target].connections.discard(hypha.source)
            del self.hyphae[hypha_id]
            pruned_count += 1

        if pruned_count > 0:
            changes["pruned_hyphae"] = pruned_count
            insights.append(f"Naturally pruned {pruned_count} weak connections")

        # Update node states
        self._update_node_states()

        # Update network metrics
        self._update_network_metrics()

        # Update domain state
        cf_state.connectivity = self.connectivity_score
        cf_state.energy_flow = float(np.mean([n.nutrient_level for n in self.nodes.values()]))
        cf_state.coherence = self.clustering_coefficient
        cf_state.emergence_potential = float(np.mean([n.local_consciousness for n in self.nodes.values()]))

        # Update state vector with network topology signature
        topology_signature = self._compute_topology_signature()
        cf_state.state_vector = topology_signature

        tbl.set_state(DomainLeg.CHICAGO_FOREST, cf_state)

        # Recalculate harmony
        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["connectivity"] = self.connectivity_score
        metrics_delta["clustering"] = self.clustering_coefficient
        metrics_delta["path_length"] = self.average_path_length

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=[DomainLeg.CHICAGO_FOREST, DomainLeg.NPCPU],
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    async def _propagate_spores(self):
        """Propagate spores through the network."""
        still_active = []

        for spore in self.active_spores:
            if not spore.propagate():
                self.spore_history.append({
                    "spore_id": spore.spore_id,
                    "origin": spore.origin_node,
                    "signal_type": spore.signal_type,
                    "hops": len(spore.path),
                    "final_strength": spore.strength,
                })
                continue

            # Get current position
            current_node = spore.path[-1] if spore.path else spore.origin_node

            # Find next nodes to propagate to
            if current_node in self.nodes:
                neighbors = list(self.nodes[current_node].connections)
                unvisited = [n for n in neighbors if n not in spore.path]

                if unvisited:
                    # Propagate to unvisited neighbors
                    next_node = np.random.choice(unvisited)
                    spore.path.append(next_node)

                    # Apply signal effect at new node
                    self.nodes[next_node].local_consciousness += spore.strength * 0.01
                    still_active.append(spore)

        self.active_spores = still_active

    def _update_node_states(self):
        """Update health states of all nodes."""
        for node in self.nodes.values():
            if node.node_id in self.failed_nodes:
                continue

            # Check nutrient levels
            if node.nutrient_level < 0.1:
                node.state = NodeState.STRESSED
            elif node.nutrient_level < 0.2:
                if node.state == NodeState.HEALTHY:
                    node.state = NodeState.STRESSED
            elif node.state == NodeState.STRESSED and node.nutrient_level > 0.3:
                node.state = NodeState.RECOVERING
            elif node.state == NodeState.RECOVERING and node.nutrient_level > 0.4:
                node.state = NodeState.HEALTHY

            # Random failure (simulate faults)
            if np.random.random() < 0.001:  # 0.1% chance per cycle
                node.state = NodeState.FAILED
                self.failed_nodes.add(node.node_id)

    def _compute_topology_signature(self) -> np.ndarray:
        """Compute a 64-dimensional signature of network topology."""
        signature = np.zeros(64)

        # Degree distribution (first 16 dims)
        degrees = np.sum(self.adjacency_matrix > 0, axis=1)
        degree_hist, _ = np.histogram(degrees, bins=16, range=(0, max(16, np.max(degrees) + 1)))
        signature[:16] = degree_hist / max(1, np.sum(degree_hist))

        # Nutrient distribution (next 16 dims)
        nutrients = np.array([n.nutrient_level for n in self.nodes.values()])
        nutrient_hist, _ = np.histogram(nutrients, bins=16, range=(0, 1))
        signature[16:32] = nutrient_hist / max(1, np.sum(nutrient_hist))

        # Consciousness distribution (next 16 dims)
        consciousness = np.array([n.local_consciousness for n in self.nodes.values()])
        consciousness_hist, _ = np.histogram(consciousness, bins=16, range=(0, 1))
        signature[32:48] = consciousness_hist / max(1, np.sum(consciousness_hist))

        # Network metrics (last 16 dims)
        signature[48] = self.connectivity_score
        signature[49] = self.clustering_coefficient
        signature[50] = min(1.0, self.average_path_length / 10)
        signature[51] = min(1.0, self.redundancy_factor / 10)
        signature[52] = len(self.failed_nodes) / max(1, self.num_nodes)
        signature[53] = len(self.active_spores) / 100
        signature[54] = float(np.mean(nutrients))
        signature[55] = float(np.std(nutrients))
        signature[56] = float(np.mean(consciousness))
        signature[57] = float(np.std(consciousness))
        # Remaining dims for future use

        return signature

    # === Public API Methods ===

    def route_signal(
        self,
        source: int,
        destination: int,
        payload: Dict[str, Any]
    ) -> Optional[List[int]]:
        """Route a signal from source to destination using optimal geodesic path."""
        path = self._find_shortest_path(source, destination)

        if path:
            # Strengthen hyphae along path
            for i in range(len(path) - 1):
                hypha_id = self._get_hypha_id(path[i], path[i + 1])
                if hypha_id in self.hyphae:
                    self.hyphae[hypha_id].strengthen(0.02)

        return path

    def broadcast_spore(
        self,
        origin: int,
        signal_type: str,
        payload: Dict[str, Any],
        ttl: int = 10
    ) -> Spore:
        """Broadcast a spore from origin node to propagate through network."""
        spore = Spore(
            spore_id=f"spore_{uuid.uuid4().hex[:8]}",
            origin_node=origin,
            payload=payload,
            signal_type=signal_type,
            ttl=ttl,
            path=[origin]
        )
        self.active_spores.append(spore)
        return spore

    def inject_failure(self, node_id: int) -> bool:
        """Simulate node failure for testing fault tolerance."""
        if node_id in self.nodes and node_id not in self.failed_nodes:
            self.nodes[node_id].state = NodeState.FAILED
            self.failed_nodes.add(node_id)
            return True
        return False

    def get_network_visualization_data(self) -> Dict[str, Any]:
        """Get data for network visualization."""
        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append({
                "id": node.node_id,
                "position": node.position.tolist(),
                "nutrient_level": node.nutrient_level,
                "consciousness": node.local_consciousness,
                "state": node.state.name,
                "connections": len(node.connections),
            })

        edges_data = []
        for hypha in self.hyphae.values():
            edges_data.append({
                "id": hypha.hypha_id,
                "source": hypha.source,
                "target": hypha.target,
                "type": hypha.hypha_type.name,
                "weight": hypha.weight,
                "bandwidth": hypha.bandwidth,
            })

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "metrics": {
                "connectivity": self.connectivity_score,
                "clustering": self.clustering_coefficient,
                "avg_path_length": self.average_path_length,
                "redundancy": self.redundancy_factor,
                "failed_nodes": len(self.failed_nodes),
                "active_spores": len(self.active_spores),
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent and network metrics."""
        base_metrics = super().get_metrics()

        network_metrics = {
            "num_nodes": self.num_nodes,
            "num_hyphae": len(self.hyphae),
            "connectivity_score": self.connectivity_score,
            "clustering_coefficient": self.clustering_coefficient,
            "average_path_length": self.average_path_length,
            "redundancy_factor": self.redundancy_factor,
            "failed_nodes": len(self.failed_nodes),
            "active_spores": len(self.active_spores),
            "total_nutrients": sum(n.nutrient_level for n in self.nodes.values()),
            "total_consciousness": sum(n.local_consciousness for n in self.nodes.values()),
            "node_states": {
                "healthy": sum(1 for n in self.nodes.values() if n.state == NodeState.HEALTHY),
                "stressed": sum(1 for n in self.nodes.values() if n.state == NodeState.STRESSED),
                "recovering": sum(1 for n in self.nodes.values() if n.state == NodeState.RECOVERING),
                "dormant": sum(1 for n in self.nodes.values() if n.state == NodeState.DORMANT),
                "failed": len(self.failed_nodes),
            },
            "hypha_types": {
                t.name: sum(1 for h in self.hyphae.values() if h.hypha_type == t)
                for t in HyphaType
            }
        }

        return {**base_metrics, **network_metrics}
