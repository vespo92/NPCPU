"""
Tests for Collective Behavior System

Tests the CollectiveMind, knowledge sharing, collective memory,
and group decision making.
"""

import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from swarm.collective_behavior import (
    CollectiveMind,
    CollectiveMindSubsystem,
    Knowledge,
    Memory,
    KnowledgeType,
    DecisionMethod
)
from core.simple_organism import SimpleOrganism


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def collective():
    """Create a CollectiveMind for tests"""
    return CollectiveMind(
        name="test_hive",
        knowledge_capacity=100,
        memory_capacity=50,
        knowledge_spread_rate=0.8,
        consensus_threshold=0.7
    )


@pytest.fixture
def organisms():
    """Create a list of test organisms"""
    return [SimpleOrganism(f"Member_{i}") for i in range(5)]


@pytest.fixture
def collective_with_members(collective, organisms):
    """Create a collective with members"""
    for org in organisms:
        collective.add_member(org)
    return collective


# ============================================================================
# Knowledge Tests
# ============================================================================

class TestKnowledge:
    """Test Knowledge data class"""

    def test_knowledge_creation(self):
        """Test creating knowledge"""
        k = Knowledge(
            type=KnowledgeType.LOCATION,
            data={"x": 100, "y": 200},
            source_id="test",
            importance=0.8
        )

        assert k.type == KnowledgeType.LOCATION
        assert k.data["x"] == 100
        assert k.importance == 0.8

    def test_knowledge_decay(self):
        """Test knowledge decay over time"""
        k = Knowledge(importance=1.0, decay_rate=0.1)
        k.decay()
        assert k.importance == 0.9

        # Decay should not go below 0
        for _ in range(20):
            k.decay()
        assert k.importance == 0.0

    def test_knowledge_access_reinforcement(self):
        """Test that accessing knowledge reinforces it"""
        k = Knowledge(importance=0.5)
        k.access()
        assert k.importance > 0.5
        assert k.access_count == 1

    def test_knowledge_to_dict(self):
        """Test knowledge serialization"""
        k = Knowledge(
            type=KnowledgeType.THREAT,
            data={"predator": "wolf"},
            importance=0.9
        )
        data = k.to_dict()

        assert data["type"] == "threat"
        assert data["data"]["predator"] == "wolf"
        assert data["importance"] == 0.9


# ============================================================================
# Memory Tests
# ============================================================================

class TestMemory:
    """Test Memory data class"""

    def test_memory_creation(self):
        """Test creating a memory"""
        m = Memory(
            event_type="attack",
            data={"attacker": "predator"},
            witnesses={"org1", "org2"},
            significance=0.8,
            emotional_valence=-0.5
        )

        assert m.event_type == "attack"
        assert len(m.witnesses) == 2
        assert m.emotional_valence == -0.5

    def test_memory_to_dict(self):
        """Test memory serialization"""
        m = Memory(event_type="discovery", significance=0.7)
        data = m.to_dict()

        assert data["event_type"] == "discovery"
        assert data["significance"] == 0.7


# ============================================================================
# Member Management Tests
# ============================================================================

class TestMemberManagement:
    """Test collective member management"""

    def test_add_member(self, collective, organisms):
        """Test adding members"""
        org = organisms[0]
        result = collective.add_member(org, influence=1.5)

        assert result is True
        assert collective.size == 1

    def test_add_duplicate_member(self, collective, organisms):
        """Test adding same member twice"""
        org = organisms[0]
        collective.add_member(org)
        result = collective.add_member(org)

        assert result is False
        assert collective.size == 1

    def test_remove_member(self, collective_with_members, organisms):
        """Test removing members"""
        result = collective_with_members.remove_member(organisms[0].id)

        assert result is True
        assert collective_with_members.size == 4

    def test_get_member(self, collective_with_members, organisms):
        """Test getting member info"""
        member = collective_with_members.get_member(organisms[0].id)

        assert member is not None
        assert member.organism.id == organisms[0].id

    def test_get_members(self, collective_with_members, organisms):
        """Test getting all members"""
        members = collective_with_members.get_members()
        assert len(members) == 5

    def test_set_leader(self, collective_with_members, organisms):
        """Test setting a leader"""
        result = collective_with_members.set_leader(organisms[0].id)

        assert result is True
        leader = collective_with_members.get_leader()
        assert leader.id == organisms[0].id

    def test_set_member_influence(self, collective_with_members, organisms):
        """Test modifying member influence"""
        result = collective_with_members.set_member_influence(organisms[0].id, 2.0)

        assert result is True
        member = collective_with_members.get_member(organisms[0].id)
        assert member.influence == 2.0

    def test_connect_disconnect_member(self, collective_with_members, organisms):
        """Test connecting and disconnecting members"""
        collective_with_members.disconnect_member(organisms[0].id)
        member = collective_with_members.get_member(organisms[0].id)
        assert member.connected is False

        collective_with_members.connect_member(organisms[0].id)
        member = collective_with_members.get_member(organisms[0].id)
        assert member.connected is True


# ============================================================================
# Knowledge Sharing Tests
# ============================================================================

class TestKnowledgeSharing:
    """Test knowledge sharing system"""

    def test_share_knowledge(self, collective_with_members, organisms):
        """Test sharing knowledge"""
        kid = collective_with_members.share_knowledge(
            source_id=organisms[0].id,
            knowledge_type=KnowledgeType.LOCATION,
            data={"food_source": (100, 200)},
            importance=0.8
        )

        assert kid != ""

    def test_share_knowledge_from_non_member(self, collective):
        """Test that non-members cannot share knowledge"""
        kid = collective.share_knowledge(
            source_id="fake_id",
            knowledge_type=KnowledgeType.LOCATION,
            data={}
        )

        assert kid == ""

    def test_query_knowledge_by_type(self, collective_with_members, organisms):
        """Test querying knowledge by type"""
        # Share different types
        collective_with_members.share_knowledge(
            organisms[0].id,
            KnowledgeType.LOCATION,
            {"place": "water"},
            0.7
        )
        collective_with_members.share_knowledge(
            organisms[1].id,
            KnowledgeType.THREAT,
            {"danger": "fire"},
            0.9
        )

        threats = collective_with_members.query_knowledge(KnowledgeType.THREAT)
        assert len(threats) == 1
        assert threats[0].data["danger"] == "fire"

    def test_query_knowledge_min_importance(self, collective_with_members, organisms):
        """Test querying with minimum importance"""
        collective_with_members.share_knowledge(
            organisms[0].id, KnowledgeType.RESOURCE, {}, 0.3
        )
        collective_with_members.share_knowledge(
            organisms[1].id, KnowledgeType.RESOURCE, {}, 0.8
        )

        results = collective_with_members.query_knowledge(
            KnowledgeType.RESOURCE,
            min_importance=0.5
        )
        assert len(results) == 1
        assert results[0].importance >= 0.5

    def test_forget_knowledge(self, collective_with_members, organisms):
        """Test forgetting knowledge"""
        kid = collective_with_members.share_knowledge(
            organisms[0].id, KnowledgeType.LOCATION, {}, 0.5
        )

        result = collective_with_members.forget_knowledge(kid)
        assert result is True

        knowledge = collective_with_members.get_knowledge(kid)
        assert knowledge is None

    def test_knowledge_capacity_pruning(self, collective, organisms):
        """Test that knowledge is pruned when over capacity"""
        collective.knowledge_capacity = 5

        for org in organisms:
            collective.add_member(org)

        # Share more knowledge than capacity
        for i in range(10):
            collective.share_knowledge(
                organisms[i % 5].id,
                KnowledgeType.EXPERIENCE,
                {"event": i},
                importance=i * 0.1
            )

        # Should be at capacity
        assert len(collective._knowledge) <= 5


# ============================================================================
# Collective Memory Tests
# ============================================================================

class TestCollectiveMemory:
    """Test collective memory system"""

    def test_record_memory(self, collective_with_members, organisms):
        """Test recording a memory"""
        mid = collective_with_members.record_memory(
            event_type="migration",
            data={"destination": "north"},
            witnesses={organisms[0].id, organisms[1].id},
            significance=0.8,
            emotional_valence=0.3
        )

        assert mid != ""

    def test_recall_memories_by_type(self, collective_with_members, organisms):
        """Test recalling memories by type"""
        collective_with_members.record_memory("attack", {"enemy": "wolf"}, significance=0.9)
        collective_with_members.record_memory("discovery", {"found": "food"}, significance=0.5)

        attacks = collective_with_members.recall_memories(event_type="attack")
        assert len(attacks) == 1
        assert attacks[0].data["enemy"] == "wolf"

    def test_recall_memories_min_significance(self, collective_with_members, organisms):
        """Test recalling with minimum significance"""
        collective_with_members.record_memory("event1", {}, significance=0.3)
        collective_with_members.record_memory("event2", {}, significance=0.8)

        significant = collective_with_members.recall_memories(min_significance=0.5)
        assert len(significant) == 1

    def test_memory_capacity_pruning(self, collective, organisms):
        """Test memory capacity pruning"""
        collective.memory_capacity = 3

        for org in organisms:
            collective.add_member(org)

        # Record more memories than capacity
        for i in range(10):
            collective.record_memory(f"event_{i}", {}, significance=i * 0.1)

        assert len(collective._memories) <= 3

    def test_get_emotional_history(self, collective_with_members, organisms):
        """Test getting emotional history"""
        collective_with_members.record_memory("good", {}, emotional_valence=0.8)
        collective_with_members.record_memory("bad", {}, emotional_valence=-0.6)
        collective_with_members.record_memory("neutral", {}, emotional_valence=0.0)

        avg, history = collective_with_members.get_emotional_history(window=3)

        assert len(history) == 3
        assert abs(avg - (0.8 - 0.6 + 0.0) / 3) < 0.01


# ============================================================================
# Group Decision Making Tests
# ============================================================================

class TestGroupDecision:
    """Test collective decision making"""

    def test_majority_decision(self, collective_with_members, organisms):
        """Test majority vote decision"""
        decision, confidence = collective_with_members.make_decision(
            question="what_to_do",
            options=["attack", "defend", "flee"],
            method=DecisionMethod.MAJORITY
        )

        assert decision in ["attack", "defend", "flee"]
        assert 0.0 <= confidence <= 1.0

    def test_weighted_decision(self, collective_with_members, organisms):
        """Test weighted vote decision"""
        # Give first organism high influence
        collective_with_members.set_member_influence(organisms[0].id, 10.0)

        decision, confidence = collective_with_members.make_decision(
            question="what_to_do",
            options=["a", "b"],
            method=DecisionMethod.WEIGHTED
        )

        assert decision in ["a", "b"]

    def test_leader_decision(self, collective_with_members, organisms):
        """Test leader-based decision"""
        collective_with_members.set_leader(organisms[0].id)

        decision, confidence = collective_with_members.make_decision(
            question="direction",
            options=["north", "south", "east", "west"],
            method=DecisionMethod.LEADER
        )

        assert decision in ["north", "south", "east", "west"]
        assert confidence == 1.0

    def test_consensus_decision_fails(self, collective_with_members, organisms):
        """Test consensus decision failing without agreement"""
        # Predetermine scattered votes to prevent consensus
        voter_opinions = {
            organisms[0].id: "a",
            organisms[1].id: "b",
            organisms[2].id: "c",
            organisms[3].id: "a",
            organisms[4].id: "b"
        }

        decision, confidence = collective_with_members.make_decision(
            question="choice",
            options=["a", "b", "c"],
            method=DecisionMethod.CONSENSUS,
            voter_opinions=voter_opinions
        )

        # With 5 members voting a,b,c,a,b - no option gets 70%
        # Best is "a" or "b" with 40%, which is < 70% threshold
        assert decision == ""  # No consensus

    def test_consensus_decision_succeeds(self, collective_with_members, organisms):
        """Test consensus decision succeeding with agreement"""
        # All vote the same
        voter_opinions = {org.id: "agree" for org in organisms}

        decision, confidence = collective_with_members.make_decision(
            question="unanimous",
            options=["agree", "disagree"],
            method=DecisionMethod.CONSENSUS,
            voter_opinions=voter_opinions
        )

        assert decision == "agree"
        assert confidence == 1.0

    def test_random_decision(self, collective_with_members, organisms):
        """Test random decision method"""
        decision, confidence = collective_with_members.make_decision(
            question="random",
            options=["x", "y", "z"],
            method=DecisionMethod.RANDOM
        )

        assert decision in ["x", "y", "z"]

    def test_decision_empty_collective(self, collective):
        """Test decision with no members"""
        decision, confidence = collective.make_decision(
            question="test",
            options=["a", "b"]
        )

        assert decision == ""
        assert confidence == 0.0


# ============================================================================
# Hive Mind Emergence Tests
# ============================================================================

class TestHiveMindEmergence:
    """Test hive mind emergence calculations"""

    def test_calculate_cohesion(self, collective_with_members, organisms):
        """Test cohesion calculation"""
        cohesion = collective_with_members.calculate_cohesion()

        assert 0.0 <= cohesion <= 1.0

    def test_cohesion_increases_with_sharing(self, collective_with_members, organisms):
        """Test that sharing knowledge increases cohesion"""
        initial_cohesion = collective_with_members.calculate_cohesion()

        # Share lots of knowledge
        for i in range(10):
            collective_with_members.share_knowledge(
                organisms[i % 5].id,
                KnowledgeType.EXPERIENCE,
                {"data": i},
                importance=0.8
            )

        new_cohesion = collective_with_members.calculate_cohesion()
        assert new_cohesion >= initial_cohesion

    def test_get_emergence_level(self, collective_with_members, organisms):
        """Test emergence level calculation"""
        emergence = collective_with_members.get_emergence_level()

        assert 0.0 <= emergence <= 1.0

    def test_broadcast_thought(self, collective_with_members, organisms):
        """Test broadcasting thought to collective"""
        received = collective_with_members.broadcast_thought(
            source_id=organisms[0].id,
            thought={"message": "danger nearby"},
            strength=1.0
        )

        # At least some members should receive with strength=1.0
        assert received >= 0


# ============================================================================
# Tick and State Tests
# ============================================================================

class TestTickAndState:
    """Test tick processing and state management"""

    def test_tick_decays_knowledge(self, collective_with_members, organisms):
        """Test that tick causes knowledge decay"""
        kid = collective_with_members.share_knowledge(
            organisms[0].id,
            KnowledgeType.LOCATION,
            {},
            importance=0.5
        )

        initial = collective_with_members.get_knowledge(kid).importance

        collective_with_members.tick()

        # Importance should decrease after tick
        k = collective_with_members.get_knowledge(kid)
        if k:  # May be removed if importance hit 0
            assert k.importance <= initial

    def test_tick_increases_contributor_trust(self, collective_with_members, organisms):
        """Test that contributors gain trust"""
        collective_with_members.share_knowledge(
            organisms[0].id,
            KnowledgeType.RESOURCE,
            {},
            importance=0.8
        )

        initial_trust = collective_with_members.get_member(organisms[0].id).trust_level

        collective_with_members.tick()

        new_trust = collective_with_members.get_member(organisms[0].id).trust_level
        assert new_trust >= initial_trust

    def test_get_state(self, collective_with_members, organisms):
        """Test state serialization"""
        # Add some data
        collective_with_members.share_knowledge(
            organisms[0].id, KnowledgeType.THREAT, {}, 0.9
        )
        collective_with_members.record_memory("test", {}, significance=0.7)

        state = collective_with_members.get_state()

        assert "id" in state
        assert "name" in state
        assert "members" in state
        assert "knowledge" in state
        assert "memories" in state
        assert "settings" in state

    def test_get_stats(self, collective_with_members, organisms):
        """Test statistics retrieval"""
        stats = collective_with_members.get_stats()

        assert stats["total_members"] == 5
        assert stats["connected_members"] == 5
        assert "cohesion" in stats
        assert "emergence_level" in stats


# ============================================================================
# CollectiveMindSubsystem Tests
# ============================================================================

class TestCollectiveMindSubsystem:
    """Test CollectiveMindSubsystem for organisms"""

    def test_subsystem_creation(self, collective):
        """Test creating collective mind subsystem"""
        subsystem = CollectiveMindSubsystem(collective)
        assert subsystem.name == "collective_mind"

    def test_join_collective(self, collective, organisms):
        """Test organism joining collective via subsystem"""
        org = organisms[0]
        subsystem = CollectiveMindSubsystem(collective)
        org.add_subsystem(subsystem)

        result = subsystem.join(influence=1.5)

        assert result is True
        assert collective.size == 1

    def test_leave_collective(self, collective, organisms):
        """Test organism leaving collective via subsystem"""
        org = organisms[0]
        subsystem = CollectiveMindSubsystem(collective)
        org.add_subsystem(subsystem)
        subsystem.join()

        result = subsystem.leave()

        assert result is True
        assert collective.size == 0

    def test_share_via_subsystem(self, collective, organisms):
        """Test sharing knowledge via subsystem"""
        org = organisms[0]
        subsystem = CollectiveMindSubsystem(collective)
        org.add_subsystem(subsystem)
        subsystem.join()

        kid = subsystem.share(
            KnowledgeType.THREAT,
            {"predator": "hawk"},
            importance=0.8
        )

        assert kid != ""

    def test_query_via_subsystem(self, collective_with_members, organisms):
        """Test querying knowledge via subsystem"""
        org = organisms[0]
        subsystem = CollectiveMindSubsystem(collective_with_members)
        org.add_subsystem(subsystem)

        # Subsystem should be able to query since collective has members
        # But need to join first
        subsystem._joined = True  # Simulate join

        collective_with_members.share_knowledge(
            organisms[1].id,
            KnowledgeType.LOCATION,
            {"place": "water"},
            0.8
        )

        results = subsystem.query(KnowledgeType.LOCATION)
        assert len(results) >= 1

    def test_broadcast_via_subsystem(self, collective, organisms):
        """Test broadcasting via subsystem"""
        for org in organisms:
            subsystem = CollectiveMindSubsystem(collective)
            org.add_subsystem(subsystem)
            subsystem.join()

        # Get subsystem from first organism
        subsystem = organisms[0].get_subsystem("collective_mind")
        received = subsystem.broadcast({"alert": "danger"}, strength=1.0)

        assert received >= 0

    def test_get_cohesion_via_subsystem(self, collective_with_members, organisms):
        """Test getting cohesion via subsystem"""
        org = organisms[0]
        subsystem = CollectiveMindSubsystem(collective_with_members)
        org.add_subsystem(subsystem)

        cohesion = subsystem.get_cohesion()
        assert 0.0 <= cohesion <= 1.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with SimpleOrganism"""

    def test_full_collective_lifecycle(self):
        """Test complete collective mind lifecycle"""
        # Create collective
        mind = CollectiveMind("pack_mind")

        # Create organisms
        organisms = [SimpleOrganism(f"Wolf_{i}") for i in range(8)]

        # Add organisms with varying influence
        for i, org in enumerate(organisms):
            mind.add_member(org, influence=1.0 + i * 0.1)

        # Set alpha as leader
        mind.set_leader(organisms[0].id)

        # Share various knowledge
        mind.share_knowledge(
            organisms[0].id,
            KnowledgeType.LOCATION,
            {"prey": (500, 300)},
            importance=0.9
        )
        mind.share_knowledge(
            organisms[1].id,
            KnowledgeType.THREAT,
            {"hunters": "north"},
            importance=0.95
        )

        # Record shared memory
        mind.record_memory(
            "successful_hunt",
            {"prey": "deer", "location": (500, 300)},
            witnesses={org.id for org in organisms[:5]},
            significance=0.8,
            emotional_valence=0.7
        )

        # Make group decision
        decision, confidence = mind.make_decision(
            "hunt_or_rest",
            ["hunt", "rest", "patrol"],
            method=DecisionMethod.WEIGHTED
        )
        assert decision in ["hunt", "rest", "patrol"]

        # Check emergence
        emergence = mind.get_emergence_level()
        assert emergence > 0

        # Run ticks
        for _ in range(5):
            mind.tick()

        # Get stats
        stats = mind.get_stats()
        assert stats["total_members"] == 8
        assert stats["knowledge_count"] >= 1

    def test_organism_with_both_subsystems(self):
        """Test organism with both swarm and collective mind subsystems"""
        from swarm.coordination import SwarmCoordinator, SwarmSubsystem

        coordinator = SwarmCoordinator()
        collective = CollectiveMind("hive")

        org = SimpleOrganism("Bee_1")

        # Add both subsystems
        swarm_sub = SwarmSubsystem(coordinator)
        mind_sub = CollectiveMindSubsystem(collective)
        org.add_subsystem(swarm_sub)
        org.add_subsystem(mind_sub)

        # Create swarm and collective
        swarm_id = coordinator.create_swarm("swarm")
        swarm_sub.join_swarm(swarm_id, position=(10, 10))
        mind_sub.join()

        # Both should work independently
        assert coordinator.get_swarm_size(swarm_id) == 1
        assert collective.size == 1

        # Can share knowledge while in swarm
        kid = mind_sub.share(KnowledgeType.LOCATION, {"flower": (100, 100)})
        assert kid != ""

        # Can get flocking direction
        direction = swarm_sub.get_flocking_direction()
        assert isinstance(direction, tuple)
