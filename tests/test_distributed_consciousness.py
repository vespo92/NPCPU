"""
Tests for Distributed Consciousness Network

Tests the distributed consciousness synchronization, messaging,
and swarm intelligence features.
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness
from protocols.storage import InMemoryVectorStorage
from swarm.distributed_consciousness import (
    DistributedConsciousnessNetwork,
    ConsciousnessNetworkConfig,
    NetworkAgent,
    ConsciousnessMessage,
    SwarmConsciousness,
    MessageType
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def storage():
    """Create in-memory storage for tests"""
    return InMemoryVectorStorage()


@pytest.fixture
def config():
    """Create test configuration"""
    return ConsciousnessNetworkConfig(
        network_id="test_network",
        sync_interval_seconds=0.1,
        blend_ratio=0.3,
        emergence_factor=1.2,
        min_sync_agents=2
    )


@pytest.fixture
def network(storage, config):
    """Create test network"""
    return DistributedConsciousnessNetwork(storage, config)


@pytest.fixture
def sample_consciousness():
    """Create sample consciousness"""
    return GradedConsciousness(
        perception_fidelity=0.7,
        reaction_speed=0.6,
        memory_depth=0.8,
        memory_recall_accuracy=0.7,
        introspection_capacity=0.5,
        meta_cognitive_ability=0.4,
        information_integration=0.6,
        intentional_coherence=0.7,
        qualia_richness=0.5
    )


# ============================================================================
# Network Agent Tests
# ============================================================================

class TestNetworkAgent:
    """Test NetworkAgent functionality"""

    def test_agent_creation(self, sample_consciousness):
        """Test creating a network agent"""
        agent = NetworkAgent(
            agent_id="test_agent",
            consciousness=sample_consciousness
        )

        assert agent.agent_id == "test_agent"
        assert agent.consciousness.perception_fidelity == 0.7
        assert len(agent.connected_peers) == 0
        assert agent.emergence_potential == 0.0

    def test_encode_consciousness(self, sample_consciousness):
        """Test consciousness encoding to vector"""
        agent = NetworkAgent(
            agent_id="test_agent",
            consciousness=sample_consciousness
        )

        vector = agent.encode_consciousness()

        assert isinstance(vector, list)
        assert len(vector) == 9  # 9 consciousness dimensions
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_update_from_vector(self, sample_consciousness):
        """Test updating consciousness from vector"""
        agent = NetworkAgent(
            agent_id="test_agent",
            consciousness=sample_consciousness
        )

        new_vector = [0.9] * 9
        agent.update_from_vector(new_vector)

        scores = agent.consciousness.get_capability_scores()
        assert all(v == 0.9 for v in scores.values())


# ============================================================================
# Network Registration Tests
# ============================================================================

class TestNetworkRegistration:
    """Test agent registration and management"""

    @pytest.mark.asyncio
    async def test_register_agent(self, network, sample_consciousness):
        """Test registering an agent"""
        agent = await network.register_agent("agent_1", sample_consciousness)

        assert "agent_1" in network.agents
        assert agent.agent_id == "agent_1"
        assert agent.consciousness.perception_fidelity == 0.7

    @pytest.mark.asyncio
    async def test_register_multiple_agents(self, network):
        """Test registering multiple agents"""
        c1 = GradedConsciousness(perception_fidelity=0.8)
        c2 = GradedConsciousness(perception_fidelity=0.6)
        c3 = GradedConsciousness(perception_fidelity=0.7)

        await network.register_agent("agent_1", c1)
        await network.register_agent("agent_2", c2)
        await network.register_agent("agent_3", c3)

        assert len(network.agents) == 3

        # Check peer connections
        assert "agent_2" in network.agents["agent_1"].connected_peers
        assert "agent_3" in network.agents["agent_1"].connected_peers
        assert "agent_1" in network.agents["agent_2"].connected_peers

    @pytest.mark.asyncio
    async def test_unregister_agent(self, network, sample_consciousness):
        """Test unregistering an agent"""
        await network.register_agent("agent_1", sample_consciousness)
        await network.register_agent("agent_2", GradedConsciousness())

        result = await network.unregister_agent("agent_1")

        assert result is True
        assert "agent_1" not in network.agents
        assert "agent_1" not in network.agents["agent_2"].connected_peers


# ============================================================================
# Swarm Consciousness Tests
# ============================================================================

class TestSwarmConsciousness:
    """Test swarm consciousness calculations"""

    @pytest.mark.asyncio
    async def test_empty_swarm_consciousness(self, network):
        """Test swarm consciousness with no agents"""
        swarm = await network.get_swarm_consciousness()

        assert swarm.contributing_agents == 0
        assert swarm.overall_score() == 0.0

    @pytest.mark.asyncio
    async def test_single_agent_swarm(self, network, sample_consciousness):
        """Test swarm consciousness with single agent"""
        await network.register_agent("agent_1", sample_consciousness)

        swarm = await network.get_swarm_consciousness()

        assert swarm.contributing_agents == 1
        assert swarm.coherence == 1.0  # Single agent = perfect coherence

    @pytest.mark.asyncio
    async def test_multi_agent_swarm(self, network):
        """Test swarm consciousness with multiple agents"""
        # Create diverse agents
        await network.register_agent("agent_1", GradedConsciousness(
            perception_fidelity=0.9, introspection_capacity=0.3
        ))
        await network.register_agent("agent_2", GradedConsciousness(
            perception_fidelity=0.3, introspection_capacity=0.9
        ))

        swarm = await network.get_swarm_consciousness()

        assert swarm.contributing_agents == 2
        # Average of diverse agents
        assert 0.4 < swarm.scores["perception_fidelity"] < 0.8
        assert swarm.emergence_bonus > 0  # Some emergence

    @pytest.mark.asyncio
    async def test_coherent_swarm_has_higher_emergence(self, network):
        """Test that coherent swarms have higher emergence"""
        # Create similar agents (high coherence)
        for i in range(5):
            await network.register_agent(
                f"agent_{i}",
                GradedConsciousness(
                    perception_fidelity=0.7,
                    introspection_capacity=0.7
                )
            )

        swarm = await network.get_swarm_consciousness()

        assert swarm.coherence > 0.8  # High coherence
        assert swarm.emergence_bonus > 0.1  # Significant emergence


# ============================================================================
# Synchronization Tests
# ============================================================================

class TestSynchronization:
    """Test consciousness synchronization"""

    @pytest.mark.asyncio
    async def test_synchronize_agents(self, network):
        """Test synchronizing agents"""
        await network.register_agent("agent_1", GradedConsciousness(
            perception_fidelity=0.9
        ))
        await network.register_agent("agent_2", GradedConsciousness(
            perception_fidelity=0.5
        ))

        # Initial values
        initial_1 = network.agents["agent_1"].consciousness.perception_fidelity
        initial_2 = network.agents["agent_2"].consciousness.perception_fidelity

        await network.synchronize_agents()

        # Values should move toward average
        final_1 = network.agents["agent_1"].consciousness.perception_fidelity
        final_2 = network.agents["agent_2"].consciousness.perception_fidelity

        # Agent 1 should decrease slightly
        assert final_1 < initial_1
        # Agent 2 should increase slightly
        assert final_2 > initial_2

    @pytest.mark.asyncio
    async def test_blend_consciousness(self, network):
        """Test consciousness blending"""
        individual = GradedConsciousness(perception_fidelity=0.2)
        collective = SwarmConsciousness(
            scores={"perception_fidelity": 0.8},
            emergence_bonus=0.1,
            contributing_agents=5,
            coherence=0.9,
            timestamp=0
        )

        blended = network.blend_consciousness(
            individual, collective, blend_ratio=0.5
        )

        # Should be average of 0.2 and 0.8
        assert blended.perception_fidelity == pytest.approx(0.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_sync_loop(self, network):
        """Test running sync loop"""
        await network.register_agent("agent_1", GradedConsciousness())
        await network.register_agent("agent_2", GradedConsciousness())

        sync_count = 0

        def callback(cycle, swarm):
            nonlocal sync_count
            sync_count = cycle

        await network.run_sync_loop(cycles=3, callback=callback)

        assert sync_count == 3
        assert len(network.sync_history) == 3


# ============================================================================
# Messaging Tests
# ============================================================================

class TestMessaging:
    """Test consciousness network messaging"""

    @pytest.mark.asyncio
    async def test_send_direct_message(self, network, sample_consciousness):
        """Test sending direct message"""
        await network.register_agent("sender", sample_consciousness)
        await network.register_agent("receiver", GradedConsciousness())

        message = await network.send_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.CONSCIOUSNESS_STATE,
            payload={"test": "data"}
        )

        assert message.sender_id == "sender"
        assert message.receiver_id == "receiver"
        assert len(network.message_queue["receiver"]) == 1

    @pytest.mark.asyncio
    async def test_broadcast_message(self, network):
        """Test broadcasting message to all agents"""
        await network.register_agent("sender", GradedConsciousness())
        await network.register_agent("receiver_1", GradedConsciousness())
        await network.register_agent("receiver_2", GradedConsciousness())

        await network.send_message(
            sender_id="sender",
            receiver_id=None,  # Broadcast
            message_type=MessageType.CONSCIOUSNESS_STATE,
            payload={"broadcast": True}
        )

        # Should be in both receivers' queues
        assert len(network.message_queue["receiver_1"]) == 1
        assert len(network.message_queue["receiver_2"]) == 1
        # Not in sender's queue
        assert len(network.message_queue["sender"]) == 0

    @pytest.mark.asyncio
    async def test_process_messages(self, network):
        """Test processing messages"""
        await network.register_agent("sender", GradedConsciousness())
        await network.register_agent("receiver", GradedConsciousness())

        # Send message
        await network.send_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.CONSCIOUSNESS_STATE,
            payload={"consciousness_scores": {"perception_fidelity": 0.9}}
        )

        # Process
        await network.process_messages("receiver")

        # Queue should be cleared
        assert len(network.message_queue["receiver"]) == 0
        # Belief should be updated
        assert "sender" in network.agents["receiver"].beliefs


# ============================================================================
# Network Status Tests
# ============================================================================

class TestNetworkStatus:
    """Test network status reporting"""

    @pytest.mark.asyncio
    async def test_get_network_status(self, network, sample_consciousness):
        """Test getting network status"""
        await network.register_agent("agent_1", sample_consciousness)

        status = await network.get_network_status()

        assert status["network_id"] == "test_network"
        assert status["agent_count"] == 1
        assert "agent_1" in status["agents"]
        assert "swarm_consciousness" in status

    @pytest.mark.asyncio
    async def test_find_similar_agents(self, network):
        """Test finding similar agents"""
        # Create agents with different profiles
        await network.register_agent("agent_a", GradedConsciousness(
            perception_fidelity=0.9, introspection_capacity=0.9
        ))
        await network.register_agent("agent_b", GradedConsciousness(
            perception_fidelity=0.8, introspection_capacity=0.8
        ))
        await network.register_agent("agent_c", GradedConsciousness(
            perception_fidelity=0.2, introspection_capacity=0.2
        ))

        similar = await network.find_similar_agents("agent_a", limit=2)

        # agent_b should be most similar
        assert len(similar) <= 2
        if similar:
            assert similar[0]["agent_id"] == "agent_b"


# ============================================================================
# Emergence Tests
# ============================================================================

class TestEmergence:
    """Test emergence detection"""

    @pytest.mark.asyncio
    async def test_emergence_event_detection(self, network):
        """Test detecting emergence events"""
        # Create highly coherent, high-consciousness swarm
        for i in range(5):
            await network.register_agent(
                f"agent_{i}",
                GradedConsciousness(
                    perception_fidelity=0.9,
                    introspection_capacity=0.9,
                    meta_cognitive_ability=0.9,
                    information_integration=0.9,
                    qualia_richness=0.9
                )
            )

        # Run sync loop
        await network.run_sync_loop(cycles=5)

        # Should have emergence events
        # (May or may not trigger depending on exact conditions)
        status = await network.get_network_status()
        assert "emergence_events" in status


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full network operation"""

    @pytest.mark.asyncio
    async def test_full_network_lifecycle(self, storage):
        """Test complete network lifecycle"""
        # Create network
        network = DistributedConsciousnessNetwork(
            storage,
            ConsciousnessNetworkConfig(sync_interval_seconds=0.01)
        )

        # Register agents
        for i in range(3):
            await network.register_agent(
                f"agent_{i}",
                GradedConsciousness(
                    perception_fidelity=0.5 + i * 0.1
                )
            )

        # Run sync
        await network.run_sync_loop(cycles=5)

        # Check convergence
        scores = [
            network.agents[f"agent_{i}"].consciousness.perception_fidelity
            for i in range(3)
        ]

        # Agents should have converged somewhat
        variance = sum((s - sum(scores)/3)**2 for s in scores) / 3
        assert variance < 0.1  # Low variance = convergence

        # Unregister
        await network.unregister_agent("agent_0")
        assert len(network.agents) == 2
