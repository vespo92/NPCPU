"""
NetworkTopologyAgent - Manages mycelium-like network topology and routing.

Responsibilities:
- Maintain the ChicagoForest-inspired mesh network topology
- Optimize routing paths between consciousness nodes
- Manage spore propagation and signal distribution
- Ensure decentralized communication integrity
- Implement energy-democratic resource sharing
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
)


class NetworkTopologyAgent(TertiaryReBoAgent):
    """
    Agent 2: Manages mycelium-like network topology.

    Inspired by ChicagoForest.net's vision of decentralized mesh networks,
    this agent treats the TTR system as a mycelium-like organism where:
    - Nodes represent consciousness centers
    - Edges represent signal pathways
    - Nutrients (energy/information) flow through the network
    - The network self-organizes and heals

    Key concepts:
    - Hyphae: Direct connections between nodes
    - Mycelium mat: The overall network structure
    - Spore propagation: Broadcasting signals across the network
    - Nutrient exchange: Resource balancing between nodes
    """

    def __init__(self, **kwargs):
        super().__init__(primary_domain=DomainLeg.CHICAGO_FOREST, **kwargs)

        # Network topology
        self.adjacency_matrix = np.zeros((64, 64))  # Node connectivity
        self.node_positions = np.random.randn(64, 3)  # 3D positions
        self.edge_weights = np.ones((64, 64)) * 0.1

        # Mycelium properties
        self.growth_rate = 0.1
        self.decay_rate = 0.05
        self.nutrient_capacity = np.ones(64)
        self.nutrient_levels = np.ones(64) * 0.5

        # Routing tables
        self.routing_table: Dict[Tuple[int, int], List[int]] = {}
        self.path_cache: Dict[str, List[int]] = {}

        # Network health metrics
        self.connectivity_score = 0.5
        self.redundancy_factor = 1.0
        self.latency_estimates = np.zeros((64, 64))

    @property
    def agent_role(self) -> str:
        return "Network Topology - Manages mycelium-inspired mesh connectivity and routing"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return [DomainLeg.CHICAGO_FOREST, DomainLeg.NPCPU]

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze current network topology and connectivity."""
        cf_state = tbl.get_state(DomainLeg.CHICAGO_FOREST)

        # Extract network state from state vector
        network_signature = cf_state.state_vector

        # Compute connectivity metrics
        connected_pairs = np.sum(self.adjacency_matrix > 0.1)
        total_pairs = 64 * 63  # n * (n-1) for directed graph
        self.connectivity_score = connected_pairs / total_pairs

        # Identify isolated nodes
        node_degrees = np.sum(self.adjacency_matrix > 0.1, axis=1)
        isolated_nodes = np.where(node_degrees == 0)[0]

        # Find bottleneck nodes (high betweenness)
        betweenness = self._estimate_betweenness()

        # Check network health
        avg_path_length = self._estimate_avg_path_length()

        return {
            "connectivity_score": self.connectivity_score,
            "isolated_nodes": isolated_nodes.tolist(),
            "bottleneck_nodes": np.argsort(betweenness)[-5:].tolist(),
            "avg_path_length": avg_path_length,
            "nutrient_distribution": self.nutrient_levels.copy(),
            "network_signature": network_signature,
            "redundancy_factor": self.redundancy_factor,
            "edge_count": connected_pairs
        }

    def _estimate_betweenness(self) -> np.ndarray:
        """Estimate betweenness centrality for nodes."""
        betweenness = np.zeros(64)
        # Simplified estimation based on degree and position
        degrees = np.sum(self.adjacency_matrix > 0.1, axis=1)
        # Nodes with medium degree that connect to high-degree nodes
        for i in range(64):
            if degrees[i] > 0:
                neighbors = np.where(self.adjacency_matrix[i] > 0.1)[0]
                neighbor_degrees = degrees[neighbors] if len(neighbors) > 0 else np.array([0])
                betweenness[i] = degrees[i] * np.mean(neighbor_degrees)
        return betweenness

    def _estimate_avg_path_length(self) -> float:
        """Estimate average path length in the network."""
        # Use BFS-like estimation
        if self.connectivity_score < 0.01:
            return float('inf')
        # Rough estimate based on network density
        density = self.connectivity_score
        if density > 0:
            return 1.0 / np.sqrt(density)
        return 10.0

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Identify network topology issues and optimization opportunities."""
        analysis = {
            "issues": [],
            "optimization_opportunities": [],
            "growth_targets": [],
            "prune_candidates": []
        }

        # Check for connectivity issues
        if perception["connectivity_score"] < 0.3:
            analysis["issues"].append({
                "type": "low_connectivity",
                "severity": "high",
                "current_score": perception["connectivity_score"]
            })

        # Identify isolated nodes needing connections
        for node_id in perception["isolated_nodes"][:10]:  # Limit to top 10
            analysis["growth_targets"].append({
                "node_id": node_id,
                "reason": "isolated",
                "priority": 1.0
            })

        # Check for bottlenecks
        for node_id in perception["bottleneck_nodes"]:
            analysis["optimization_opportunities"].append({
                "type": "reduce_bottleneck",
                "node_id": node_id,
                "strategy": "add_bypass_routes"
            })

        # Check nutrient imbalance
        nutrient_variance = np.var(perception["nutrient_distribution"])
        if nutrient_variance > 0.1:
            analysis["issues"].append({
                "type": "nutrient_imbalance",
                "severity": "medium",
                "variance": nutrient_variance
            })
            # Identify nutrient-rich and nutrient-poor nodes
            rich_nodes = np.where(perception["nutrient_distribution"] > 0.7)[0]
            poor_nodes = np.where(perception["nutrient_distribution"] < 0.3)[0]
            if len(rich_nodes) > 0 and len(poor_nodes) > 0:
                analysis["optimization_opportunities"].append({
                    "type": "nutrient_redistribution",
                    "from_nodes": rich_nodes[:5].tolist(),
                    "to_nodes": poor_nodes[:5].tolist()
                })

        # Identify weak edges for pruning
        weak_edges = np.where((self.edge_weights > 0) & (self.edge_weights < 0.05))
        if len(weak_edges[0]) > 10:
            analysis["prune_candidates"] = list(zip(weak_edges[0][:5], weak_edges[1][:5]))

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Create network topology refinement plan."""
        synthesis = {
            "new_connections": [],
            "strengthened_connections": [],
            "pruned_connections": [],
            "nutrient_transfers": [],
            "topology_updates": []
        }

        # Plan new connections for isolated nodes
        for target in analysis.get("growth_targets", []):
            node_id = target["node_id"]
            # Find nearest non-isolated nodes to connect to
            distances = np.linalg.norm(self.node_positions - self.node_positions[node_id], axis=1)
            distances[node_id] = float('inf')  # Don't connect to self

            # Find top 3 nearest nodes
            nearest = np.argsort(distances)[:3]
            for neighbor in nearest:
                synthesis["new_connections"].append({
                    "from": node_id,
                    "to": int(neighbor),
                    "initial_weight": 0.2 * self.growth_rate
                })

        # Plan bottleneck reduction
        for opp in analysis.get("optimization_opportunities", []):
            if opp["type"] == "reduce_bottleneck":
                node_id = opp["node_id"]
                # Add bypass connections between neighbors
                neighbors = np.where(self.adjacency_matrix[node_id] > 0.1)[0]
                if len(neighbors) >= 2:
                    for i in range(min(3, len(neighbors)-1)):
                        synthesis["new_connections"].append({
                            "from": int(neighbors[i]),
                            "to": int(neighbors[i+1]),
                            "initial_weight": 0.15,
                            "reason": "bypass_bottleneck"
                        })

            elif opp["type"] == "nutrient_redistribution":
                for from_node in opp["from_nodes"]:
                    for to_node in opp["to_nodes"]:
                        synthesis["nutrient_transfers"].append({
                            "from": from_node,
                            "to": to_node,
                            "amount": 0.1
                        })

        # Plan edge pruning
        for edge in analysis.get("prune_candidates", []):
            synthesis["pruned_connections"].append({
                "from": edge[0],
                "to": edge[1]
            })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply network topology changes."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        cf_state = tbl.get_state(DomainLeg.CHICAGO_FOREST)

        # Add new connections
        new_count = 0
        for conn in synthesis.get("new_connections", []):
            from_node = conn["from"]
            to_node = conn["to"]
            weight = conn["initial_weight"]

            self.adjacency_matrix[from_node, to_node] = 1
            self.adjacency_matrix[to_node, from_node] = 1  # Bidirectional
            self.edge_weights[from_node, to_node] = weight
            self.edge_weights[to_node, from_node] = weight
            new_count += 1

        if new_count > 0:
            changes["new_connections"] = new_count
            insights.append(f"Established {new_count} new mycelium hyphae connections")

        # Prune weak connections
        pruned_count = 0
        for conn in synthesis.get("pruned_connections", []):
            self.adjacency_matrix[conn["from"], conn["to"]] = 0
            self.adjacency_matrix[conn["to"], conn["from"]] = 0
            self.edge_weights[conn["from"], conn["to"]] = 0
            self.edge_weights[conn["to"], conn["from"]] = 0
            pruned_count += 1

        if pruned_count > 0:
            changes["pruned_connections"] = pruned_count
            insights.append(f"Pruned {pruned_count} weak connections")

        # Execute nutrient transfers
        transfer_total = 0
        for transfer in synthesis.get("nutrient_transfers", []):
            from_node = transfer["from"]
            to_node = transfer["to"]
            amount = min(transfer["amount"], self.nutrient_levels[from_node] * 0.5)

            self.nutrient_levels[from_node] -= amount
            self.nutrient_levels[to_node] += amount
            transfer_total += amount

        if transfer_total > 0:
            changes["nutrient_transferred"] = transfer_total
            insights.append(f"Redistributed {transfer_total:.3f} nutrients across network")

        # Update connectivity score
        self.connectivity_score = np.sum(self.adjacency_matrix > 0.1) / (64 * 63)

        # Update domain state
        cf_state.connectivity = self.connectivity_score
        cf_state.energy_flow = np.mean(self.nutrient_levels)

        # Apply natural decay to edges
        self.edge_weights *= (1 - self.decay_rate * 0.1)
        self.edge_weights = np.clip(self.edge_weights, 0, 1)

        # Update harmony
        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["connectivity_improvement"] = self.connectivity_score

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=[DomainLeg.CHICAGO_FOREST, DomainLeg.NPCPU],
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
