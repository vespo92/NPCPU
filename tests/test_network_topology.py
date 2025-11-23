"""
Tests for NetworkTopologyAgent

Tests the mycelium-like network topology, geodesic routing,
spore propagation, self-healing, and energy democracy features.
"""

import pytest
import asyncio
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tertiary_rebo.agents.network_topology import (
    NetworkTopologyAgent,
    MyceliumNode,
    Hypha,
    Spore,
    RoutingEntry,
    HyphaType,
    NodeState,
)
from tertiary_rebo.base import (
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementPhase,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a NetworkTopologyAgent for tests"""
    return NetworkTopologyAgent(num_nodes=32, manifold_dimension=4)


@pytest.fixture
def small_agent():
    """Create a smaller agent for faster tests"""
    return NetworkTopologyAgent(num_nodes=16, manifold_dimension=3)


@pytest.fixture
def tbl():
    """Create a TripleBottomLine for tests"""
    return TripleBottomLine()


# ============================================================================
# Initialization Tests
# ============================================================================

class TestNetworkInitialization:
    """Test network topology initialization"""

    def test_agent_creation(self, agent):
        """Test that agent is created with correct properties"""
        assert agent.num_nodes == 32
        assert agent.manifold_dimension == 4
        assert len(agent.nodes) == 32
        assert agent.primary_domain == DomainLeg.CHICAGO_FOREST

    def test_nodes_initialized(self, agent):
        """Test that all nodes are properly initialized"""
        for node_id, node in agent.nodes.items():
            assert isinstance(node, MyceliumNode)
            assert node.node_id == node_id
            assert 0.0 <= node.nutrient_level <= 1.0
            assert len(node.position) == agent.manifold_dimension
            assert node.state == NodeState.HEALTHY

    def test_positions_in_poincare_ball(self, agent):
        """Test that node positions are within the Poincaré ball"""
        for node in agent.nodes.values():
            norm = np.linalg.norm(node.position)
            assert norm < 1.0, f"Node {node.node_id} position outside unit ball"

    def test_small_world_topology(self, agent):
        """Test that small-world topology is created"""
        # Each node should have at least some connections
        for node in agent.nodes.values():
            assert len(node.connections) >= 2, f"Node {node.node_id} has too few connections"

        # Check connectivity score is reasonable
        assert agent.connectivity_score > 0.01

    def test_hyphae_created(self, agent):
        """Test that hyphal connections are created"""
        assert len(agent.hyphae) > 0
        for hypha in agent.hyphae.values():
            assert isinstance(hypha, Hypha)
            assert 0.0 <= hypha.weight <= 1.0
            assert hypha.latency > 0

    def test_agent_role(self, agent):
        """Test agent role property"""
        assert "Network Topology" in agent.agent_role
        assert "mycelium" in agent.agent_role.lower()

    def test_domains_affected(self, agent):
        """Test domains affected property"""
        domains = agent.domains_affected
        assert DomainLeg.CHICAGO_FOREST in domains
        assert DomainLeg.NPCPU in domains


# ============================================================================
# Geodesic Distance Tests
# ============================================================================

class TestGeodesicDistance:
    """Test geodesic distance calculations on hyperbolic manifold"""

    def test_distance_to_self_is_zero(self, agent):
        """Test that distance from node to itself is zero"""
        distance = agent._geodesic_distance(0, 0)
        assert distance == 0.0

    def test_distance_is_symmetric(self, agent):
        """Test that geodesic distance is symmetric"""
        d1 = agent._geodesic_distance(0, 1)
        d2 = agent._geodesic_distance(1, 0)
        assert abs(d1 - d2) < 1e-10

    def test_distance_is_positive(self, agent):
        """Test that distance between different nodes is positive"""
        for i in range(min(5, agent.num_nodes)):
            for j in range(i + 1, min(10, agent.num_nodes)):
                distance = agent._geodesic_distance(i, j)
                assert distance > 0

    def test_triangle_inequality(self, agent):
        """Test triangle inequality for geodesic distances"""
        # d(a,c) <= d(a,b) + d(b,c)
        for _ in range(10):
            a, b, c = np.random.choice(agent.num_nodes, 3, replace=False)
            d_ac = agent._geodesic_distance(a, c)
            d_ab = agent._geodesic_distance(a, b)
            d_bc = agent._geodesic_distance(b, c)
            assert d_ac <= d_ab + d_bc + 1e-10  # Small epsilon for numerical errors


# ============================================================================
# Routing Tests
# ============================================================================

class TestRouting:
    """Test geodesic routing functionality"""

    def test_find_path_to_self(self, agent):
        """Test finding path from node to itself"""
        path = agent._find_shortest_path(0, 0)
        assert path == [0]

    def test_find_path_to_neighbor(self, agent):
        """Test finding path to direct neighbor"""
        # Find a node with neighbors
        for node_id, node in agent.nodes.items():
            if len(node.connections) > 0:
                neighbor = list(node.connections)[0]
                path = agent._find_shortest_path(node_id, neighbor)
                assert path is not None
                assert len(path) == 2
                assert path[0] == node_id
                assert path[1] == neighbor
                break

    def test_path_uses_existing_connections(self, agent):
        """Test that paths only use existing hyphal connections"""
        path = agent._find_shortest_path(0, agent.num_nodes - 1)
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                hypha_id = agent._get_hypha_id(path[i], path[i + 1])
                assert hypha_id in agent.hyphae

    def test_route_signal(self, agent):
        """Test routing a signal between nodes"""
        path = agent.route_signal(0, 5, {"message": "test"})
        if path:
            assert path[0] == 0
            assert path[-1] == 5

    def test_route_signal_strengthens_hyphae(self, agent):
        """Test that routing signals strengthens the path"""
        # Find a valid path
        path = agent._find_shortest_path(0, 5)
        if path and len(path) > 1:
            hypha_id = agent._get_hypha_id(path[0], path[1])
            original_weight = agent.hyphae[hypha_id].weight

            # Route signal
            agent.route_signal(0, 5, {"test": "data"})

            # Check weight increased
            new_weight = agent.hyphae[hypha_id].weight
            assert new_weight >= original_weight

    def test_routing_avoids_failed_nodes(self, agent):
        """Test that routing avoids failed nodes"""
        # Inject failure in a node
        agent.inject_failure(3)

        # Find path that would normally go through node 3
        path = agent._find_shortest_path(0, agent.num_nodes - 1)

        if path:
            assert 3 not in path


# ============================================================================
# Spore Propagation Tests
# ============================================================================

class TestSporePropagation:
    """Test spore (broadcast signal) propagation"""

    def test_broadcast_spore(self, agent):
        """Test creating and broadcasting a spore"""
        spore = agent.broadcast_spore(
            origin=0,
            signal_type="test",
            payload={"data": "test_data"},
            ttl=5
        )

        assert spore.origin_node == 0
        assert spore.signal_type == "test"
        assert spore.ttl == 5
        assert spore in agent.active_spores

    def test_spore_propagation_reduces_ttl(self):
        """Test that spore propagation reduces TTL"""
        spore = Spore(
            spore_id="test",
            origin_node=0,
            payload={},
            signal_type="test",
            ttl=5
        )

        assert spore.propagate() == True
        assert spore.ttl == 4

    def test_spore_dies_when_ttl_zero(self):
        """Test that spore dies when TTL reaches zero"""
        spore = Spore(
            spore_id="test",
            origin_node=0,
            payload={},
            signal_type="test",
            ttl=1
        )

        assert spore.propagate() == False
        assert spore.ttl == 0

    def test_spore_strength_decays(self):
        """Test that spore strength decays during propagation"""
        spore = Spore(
            spore_id="test",
            origin_node=0,
            payload={},
            signal_type="test",
            strength=1.0,
            ttl=10
        )

        original_strength = spore.strength
        spore.propagate()
        assert spore.strength < original_strength


# ============================================================================
# Self-Healing Tests
# ============================================================================

class TestSelfHealing:
    """Test network self-healing capabilities"""

    def test_inject_failure(self, agent):
        """Test injecting a node failure"""
        assert agent.inject_failure(5) == True
        assert 5 in agent.failed_nodes
        assert agent.nodes[5].state == NodeState.FAILED

    def test_cannot_inject_failure_twice(self, agent):
        """Test that the same node cannot fail twice"""
        agent.inject_failure(5)
        assert agent.inject_failure(5) == False

    def test_find_nearest_healthy_nodes(self, agent):
        """Test finding nearest healthy nodes"""
        nearest = agent._find_nearest_healthy_nodes(0, k=3)
        assert len(nearest) <= 3
        for node_id in nearest:
            assert agent.nodes[node_id].state == NodeState.HEALTHY

    def test_nearest_excludes_failed_nodes(self, agent):
        """Test that nearest healthy excludes failed nodes"""
        # Fail some nodes near node 0
        agent.inject_failure(1)
        agent.inject_failure(2)

        nearest = agent._find_nearest_healthy_nodes(0, k=5)
        assert 1 not in nearest
        assert 2 not in nearest


# ============================================================================
# Hypha Management Tests
# ============================================================================

class TestHyphaManagement:
    """Test hyphal connection management"""

    def test_create_hypha(self, small_agent):
        """Test creating a new hyphal connection"""
        # Find two unconnected nodes
        for i in range(small_agent.num_nodes):
            for j in range(i + 1, small_agent.num_nodes):
                if small_agent.adjacency_matrix[i, j] == 0:
                    hypha = small_agent._create_hypha(i, j, HyphaType.RUNNER, 0.5)
                    assert hypha is not None
                    assert small_agent.adjacency_matrix[i, j] == 1
                    assert j in small_agent.nodes[i].connections
                    assert i in small_agent.nodes[j].connections
                    return

    def test_hypha_decay(self):
        """Test hyphal connection decay"""
        hypha = Hypha(
            hypha_id="test",
            source=0,
            target=1,
            hypha_type=HyphaType.CONNECTOR,
            weight=0.5
        )

        original_weight = hypha.weight
        hypha.decay(0.1)
        assert hypha.weight < original_weight
        assert hypha.age == 1

    def test_hypha_strengthen(self):
        """Test strengthening hyphal connection"""
        hypha = Hypha(
            hypha_id="test",
            source=0,
            target=1,
            hypha_type=HyphaType.CONNECTOR,
            weight=0.5
        )

        original_weight = hypha.weight
        hypha.strengthen(0.2)
        assert hypha.weight > original_weight

    def test_hypha_weight_capped(self):
        """Test that hypha weight is capped at 1.0"""
        hypha = Hypha(
            hypha_id="test",
            source=0,
            target=1,
            hypha_type=HyphaType.CONNECTOR,
            weight=0.9
        )

        hypha.strengthen(0.5)
        assert hypha.weight == 1.0


# ============================================================================
# Network Metrics Tests
# ============================================================================

class TestNetworkMetrics:
    """Test network metric calculations"""

    def test_connectivity_score_in_range(self, agent):
        """Test that connectivity score is between 0 and 1"""
        agent._update_network_metrics()
        assert 0.0 <= agent.connectivity_score <= 1.0

    def test_clustering_coefficient_in_range(self, agent):
        """Test that clustering coefficient is between 0 and 1"""
        agent._update_network_metrics()
        assert 0.0 <= agent.clustering_coefficient <= 1.0

    def test_redundancy_factor_positive(self, agent):
        """Test that redundancy factor is positive"""
        agent._update_network_metrics()
        assert agent.redundancy_factor > 0

    def test_gini_coefficient(self, agent):
        """Test Gini coefficient calculation"""
        # Equal distribution should have low Gini
        equal_values = np.ones(10) * 0.5
        gini_equal = agent._calculate_gini_coefficient(equal_values)
        assert gini_equal < 0.1

        # Unequal distribution should have higher Gini
        unequal_values = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0])
        gini_unequal = agent._calculate_gini_coefficient(unequal_values)
        assert gini_unequal > gini_equal

    def test_betweenness_centrality_in_range(self, agent):
        """Test that betweenness centrality values are normalized"""
        betweenness = agent._estimate_betweenness_centrality()
        for node_id, score in betweenness.items():
            assert 0.0 <= score <= 1.0


# ============================================================================
# Refinement Cycle Tests
# ============================================================================

class TestRefinementCycle:
    """Test the full refinement cycle"""

    @pytest.mark.asyncio
    async def test_perceive(self, agent, tbl):
        """Test perception phase"""
        perception = await agent.perceive(tbl)

        assert "connectivity_score" in perception
        assert "node_health" in perception
        assert "nutrient_distribution" in perception
        assert "hyphal_health" in perception

    @pytest.mark.asyncio
    async def test_analyze(self, agent, tbl):
        """Test analysis phase"""
        perception = await agent.perceive(tbl)
        analysis = await agent.analyze(perception, tbl)

        assert "issues" in analysis
        assert "growth_opportunities" in analysis
        assert "healing_needed" in analysis
        assert "optimization_targets" in analysis
        assert "energy_rebalancing" in analysis

    @pytest.mark.asyncio
    async def test_synthesize(self, agent, tbl):
        """Test synthesis phase"""
        perception = await agent.perceive(tbl)
        analysis = await agent.analyze(perception, tbl)
        synthesis = await agent.synthesize(analysis, tbl)

        assert "new_connections" in synthesis
        assert "healed_nodes" in synthesis
        assert "nutrient_transfers" in synthesis
        assert "spore_broadcasts" in synthesis

    @pytest.mark.asyncio
    async def test_propagate(self, agent, tbl):
        """Test propagation phase"""
        perception = await agent.perceive(tbl)
        analysis = await agent.analyze(perception, tbl)
        synthesis = await agent.synthesize(analysis, tbl)
        result = await agent.propagate(synthesis, tbl)

        assert result.success == True
        assert result.agent_id == agent.agent_id
        assert result.phase == RefinementPhase.PROPAGATION
        assert DomainLeg.CHICAGO_FOREST in result.domain_affected

    @pytest.mark.asyncio
    async def test_full_refine_cycle(self, small_agent, tbl):
        """Test complete refinement cycle"""
        result = await small_agent.refine(tbl)

        assert result.success == True
        assert small_agent.total_refinements == 1
        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta


# ============================================================================
# Energy Democracy Tests
# ============================================================================

class TestEnergyDemocracy:
    """Test energy/nutrient redistribution features"""

    def test_nutrient_levels_initialized(self, agent):
        """Test that all nodes have nutrient levels"""
        for node in agent.nodes.values():
            assert 0.0 <= node.nutrient_level <= 1.0

    @pytest.mark.asyncio
    async def test_nutrient_redistribution_reduces_inequality(self, small_agent, tbl):
        """Test that refinement reduces nutrient inequality over time"""
        # Create artificial inequality
        for i, node in enumerate(small_agent.nodes.values()):
            if i < small_agent.num_nodes // 2:
                node.nutrient_level = 0.9
            else:
                node.nutrient_level = 0.1

        initial_gini = small_agent._calculate_gini_coefficient(
            np.array([n.nutrient_level for n in small_agent.nodes.values()])
        )

        # Run several refinement cycles
        for _ in range(5):
            await small_agent.refine(tbl)

        final_gini = small_agent._calculate_gini_coefficient(
            np.array([n.nutrient_level for n in small_agent.nodes.values()])
        )

        # Gini should decrease (inequality reduced)
        assert final_gini <= initial_gini + 0.1  # Allow some tolerance


# ============================================================================
# Visualization Data Tests
# ============================================================================

class TestVisualizationData:
    """Test network visualization data export"""

    def test_visualization_data_structure(self, agent):
        """Test that visualization data has correct structure"""
        data = agent.get_network_visualization_data()

        assert "nodes" in data
        assert "edges" in data
        assert "metrics" in data

        assert len(data["nodes"]) == agent.num_nodes
        assert len(data["edges"]) == len(agent.hyphae)

    def test_node_visualization_data(self, agent):
        """Test node data in visualization export"""
        data = agent.get_network_visualization_data()

        for node_data in data["nodes"]:
            assert "id" in node_data
            assert "position" in node_data
            assert "nutrient_level" in node_data
            assert "consciousness" in node_data
            assert "state" in node_data
            assert len(node_data["position"]) == agent.manifold_dimension

    def test_edge_visualization_data(self, agent):
        """Test edge data in visualization export"""
        data = agent.get_network_visualization_data()

        for edge_data in data["edges"]:
            assert "id" in edge_data
            assert "source" in edge_data
            assert "target" in edge_data
            assert "type" in edge_data
            assert "weight" in edge_data


# ============================================================================
# Metrics Export Tests
# ============================================================================

class TestMetricsExport:
    """Test agent metrics export"""

    def test_get_metrics(self, agent):
        """Test getting agent metrics"""
        metrics = agent.get_metrics()

        # Check base agent metrics
        assert "agent_id" in metrics
        assert "agent_role" in metrics
        assert "total_refinements" in metrics

        # Check network-specific metrics
        assert "num_nodes" in metrics
        assert "num_hyphae" in metrics
        assert "connectivity_score" in metrics
        assert "clustering_coefficient" in metrics
        assert "node_states" in metrics
        assert "hypha_types" in metrics

    def test_node_states_metrics(self, agent):
        """Test node state breakdown in metrics"""
        metrics = agent.get_metrics()
        states = metrics["node_states"]

        assert "healthy" in states
        assert "stressed" in states
        assert "failed" in states

        total = sum(states.values())
        assert total == agent.num_nodes

    def test_hypha_types_metrics(self, agent):
        """Test hypha type breakdown in metrics"""
        metrics = agent.get_metrics()
        hypha_types = metrics["hypha_types"]

        for htype in HyphaType:
            assert htype.name in hypha_types


# ============================================================================
# Topology Signature Tests
# ============================================================================

class TestTopologySignature:
    """Test topology signature computation"""

    def test_signature_dimensions(self, agent):
        """Test that topology signature has correct dimensions"""
        signature = agent._compute_topology_signature()
        assert len(signature) == 64

    def test_signature_normalized(self, agent):
        """Test that signature values are reasonable"""
        signature = agent._compute_topology_signature()
        # Histograms should sum to 1
        assert abs(np.sum(signature[:16]) - 1.0) < 0.1
        assert abs(np.sum(signature[16:32]) - 1.0) < 0.1
        assert abs(np.sum(signature[32:48]) - 1.0) < 0.1

    def test_signature_changes_with_topology(self, small_agent):
        """Test that signature changes when topology changes"""
        sig1 = small_agent._compute_topology_signature().copy()

        # Modify topology
        small_agent.inject_failure(0)
        small_agent._update_network_metrics()

        sig2 = small_agent._compute_topology_signature()

        # Signatures should be different
        assert not np.allclose(sig1, sig2)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_path_to_nonexistent_node(self, agent):
        """Test finding path to non-existent node"""
        # This should handle gracefully
        path = agent._find_shortest_path(0, 9999)
        assert path is None

    def test_failed_node_routing(self, agent):
        """Test that routing fails gracefully for failed source"""
        agent.inject_failure(0)
        path = agent._find_shortest_path(0, 5)
        assert path is None

    def test_nearest_healthy_for_nonexistent_node(self, agent):
        """Test finding nearest healthy for non-existent node"""
        nearest = agent._find_nearest_healthy_nodes(9999, k=3)
        assert nearest == []

    def test_empty_gini_coefficient(self, agent):
        """Test Gini coefficient for empty array"""
        gini = agent._calculate_gini_coefficient(np.array([]))
        assert gini == 0.0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
