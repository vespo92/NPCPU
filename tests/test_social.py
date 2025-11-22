"""
Tests for Social Module

Tests for:
- SocialNetwork: relationship management, groups, influence
- CommunicationSystem: signals, messages, deception detection
- Integration with SimpleOrganism
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from social.relationships import (
    SocialNetwork,
    RelationshipType,
    Relationship,
    Group,
)
from social.communication import (
    CommunicationSystem,
    CommunicationChannel,
    SignalType,
    MessageType,
    Signal,
    Message,
)
from core.simple_organism import SimpleOrganism
from core.events import EventBus, set_event_bus, get_event_bus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def fresh_event_bus():
    """Ensure each test has a fresh event bus"""
    bus = EventBus()
    set_event_bus(bus)
    yield bus
    bus.shutdown()


@pytest.fixture
def social_network():
    """Create a social network without event emission for cleaner tests"""
    return SocialNetwork(emit_events=False)


@pytest.fixture
def network_with_events():
    """Create a social network with event emission"""
    return SocialNetwork(emit_events=True)


@pytest.fixture
def organism():
    """Create a simple organism for testing"""
    return SimpleOrganism(name="TestOrganism")


# =============================================================================
# Relationship Tests
# =============================================================================

class TestRelationship:
    """Tests for the Relationship dataclass"""

    def test_relationship_creation(self):
        """Test creating a relationship"""
        rel = Relationship(
            source_id="org1",
            target_id="org2",
            type=RelationshipType.ALLY,
            strength=0.8,
            trust=0.5
        )
        assert rel.source_id == "org1"
        assert rel.target_id == "org2"
        assert rel.type == RelationshipType.ALLY
        assert rel.strength == 0.8
        assert rel.trust == 0.5
        assert rel.interactions == 0

    def test_relationship_bounds(self):
        """Test that values are bounded correctly"""
        rel = Relationship(
            source_id="org1",
            target_id="org2",
            strength=1.5,  # Should be clamped to 1.0
            trust=-2.0     # Should be clamped to -1.0
        )
        assert rel.strength == 1.0
        assert rel.trust == -1.0

    def test_record_positive_interaction(self):
        """Test recording a positive interaction"""
        rel = Relationship(source_id="org1", target_id="org2", trust=0.0)
        rel.record_interaction(positive=True, weight=0.2)

        assert rel.interactions == 1
        assert rel.trust > 0  # Trust increased
        assert rel.familiarity > 0  # Familiarity increased
        assert rel.last_interaction is not None

    def test_record_negative_interaction(self):
        """Test recording a negative interaction"""
        rel = Relationship(source_id="org1", target_id="org2", trust=0.5, strength=0.5)
        rel.record_interaction(positive=False, weight=0.2)

        assert rel.trust < 0.5  # Trust decreased
        assert rel.strength < 0.5  # Strength decreased

    def test_decay(self):
        """Test relationship decay over time"""
        rel = Relationship(source_id="org1", target_id="org2", strength=0.8, trust=0.5)
        rel.decay(rate=0.1)

        assert rel.strength < 0.8
        assert rel.trust < 0.5

    def test_serialization(self):
        """Test relationship serialization/deserialization"""
        original = Relationship(
            source_id="org1",
            target_id="org2",
            type=RelationshipType.MATE,
            strength=0.9,
            trust=0.8
        )
        original.record_interaction(positive=True)

        data = original.to_dict()
        restored = Relationship.from_dict(data)

        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.type == original.type
        assert restored.strength == original.strength
        assert restored.trust == original.trust
        assert restored.interactions == original.interactions


# =============================================================================
# Group Tests
# =============================================================================

class TestGroup:
    """Tests for the Group dataclass"""

    def test_group_creation(self):
        """Test creating a group"""
        group = Group(name="hunters", members={"org1", "org2", "org3"})
        assert group.name == "hunters"
        assert group.size == 3
        assert "org1" in group.members

    def test_add_member(self):
        """Test adding members to a group"""
        group = Group(name="test")
        assert group.add_member("org1") is True
        assert group.add_member("org1") is False  # Already exists
        assert group.size == 1

    def test_remove_member(self):
        """Test removing members from a group"""
        group = Group(name="test", members={"org1", "org2"}, leader_id="org1")
        assert group.remove_member("org1") is True
        assert group.leader_id is None  # Leader removed
        assert group.size == 1

    def test_set_leader(self):
        """Test setting group leader"""
        group = Group(name="test", members={"org1", "org2"})
        assert group.set_leader("org1") is True
        assert group.set_leader("org3") is False  # Not a member
        assert group.leader_id == "org1"

    def test_serialization(self):
        """Test group serialization/deserialization"""
        original = Group(
            name="tribe",
            members={"org1", "org2"},
            leader_id="org1",
            cohesion=0.7
        )

        data = original.to_dict()
        restored = Group.from_dict(data)

        assert restored.name == original.name
        assert restored.members == original.members
        assert restored.leader_id == original.leader_id
        assert restored.cohesion == original.cohesion


# =============================================================================
# SocialNetwork Tests
# =============================================================================

class TestSocialNetwork:
    """Tests for the SocialNetwork class"""

    def test_form_relationship(self, social_network):
        """Test forming a relationship"""
        rel = social_network.form_relationship("org1", "org2", RelationshipType.ALLY)

        assert rel.source_id == "org1"
        assert rel.target_id == "org2"
        assert social_network.has_relationship("org1", "org2")

    def test_form_bidirectional_relationship(self, social_network):
        """Test forming a bidirectional relationship"""
        social_network.form_relationship(
            "org1", "org2",
            RelationshipType.ALLY,
            bidirectional=True
        )

        assert social_network.has_relationship("org1", "org2")
        assert social_network.has_relationship("org2", "org1")

    def test_form_parent_child_relationship(self, social_network):
        """Test that parent/child relationships get correct reciprocals"""
        social_network.form_relationship(
            "parent", "child",
            RelationshipType.PARENT,
            bidirectional=True
        )

        parent_rel = social_network.get_relationship("parent", "child")
        child_rel = social_network.get_relationship("child", "parent")

        assert parent_rel.type == RelationshipType.PARENT
        assert child_rel.type == RelationshipType.CHILD

    def test_get_relationship(self, social_network):
        """Test getting a relationship"""
        social_network.form_relationship("org1", "org2", RelationshipType.RIVAL)
        rel = social_network.get_relationship("org1", "org2")

        assert rel is not None
        assert rel.type == RelationshipType.RIVAL

    def test_remove_relationship(self, social_network):
        """Test removing a relationship"""
        social_network.form_relationship("org1", "org2")
        removed = social_network.remove_relationship("org1", "org2")

        assert removed is not None
        assert not social_network.has_relationship("org1", "org2")

    def test_get_all_relationships(self, social_network):
        """Test getting all outgoing relationships"""
        social_network.form_relationship("org1", "org2")
        social_network.form_relationship("org1", "org3")
        social_network.form_relationship("org2", "org1")

        rels = social_network.get_all_relationships("org1")
        assert len(rels) == 2

    def test_get_incoming_relationships(self, social_network):
        """Test getting all incoming relationships"""
        social_network.form_relationship("org2", "org1")
        social_network.form_relationship("org3", "org1")

        incoming = social_network.get_incoming_relationships("org1")
        assert len(incoming) == 2

    def test_get_relationships_by_type(self, social_network):
        """Test filtering relationships by type"""
        social_network.form_relationship("org1", "org2", RelationshipType.ALLY)
        social_network.form_relationship("org1", "org3", RelationshipType.RIVAL)
        social_network.form_relationship("org1", "org4", RelationshipType.ALLY)

        allies = social_network.get_relationships_by_type("org1", RelationshipType.ALLY)
        assert len(allies) == 2

    def test_get_connected_organisms(self, social_network):
        """Test getting all connected organisms"""
        social_network.form_relationship("org1", "org2")
        social_network.form_relationship("org3", "org1")

        connected = social_network.get_connected_organisms("org1")
        assert connected == {"org2", "org3"}

    def test_record_interaction(self, social_network):
        """Test recording an interaction"""
        social_network.form_relationship("org1", "org2")
        rel = social_network.record_interaction("org1", "org2", positive=True)

        assert rel.interactions == 1
        assert rel.trust > 0

    def test_record_interaction_creates_relationship(self, social_network):
        """Test that recording interaction creates relationship if none exists"""
        rel = social_network.record_interaction("org1", "org2")

        assert rel is not None
        assert social_network.has_relationship("org1", "org2")

    def test_decay_relationships(self, social_network):
        """Test decaying all relationships"""
        social_network.form_relationship("org1", "org2", strength=0.8, trust=0.5)
        social_network.decay_relationships(rate=0.1)

        rel = social_network.get_relationship("org1", "org2")
        assert rel.strength < 0.8
        assert rel.trust < 0.5

    def test_get_influence(self, social_network):
        """Test calculating influence"""
        # org1 has many positive relationships pointing to them
        for i in range(5):
            social_network.form_relationship(
                f"org{i+2}", "org1",
                RelationshipType.FOLLOWER,
                trust=0.8,
                strength=0.7
            )

        influence = social_network.get_influence("org1")
        assert influence > 0.5  # Should have significant influence

    def test_get_reputation(self, social_network):
        """Test calculating reputation"""
        # Give org1 good reputation from multiple sources
        for i in range(3):
            rel = social_network.form_relationship(
                f"org{i+2}", "org1",
                trust=0.8
            )
            rel.familiarity = 0.9

        reputation = social_network.get_reputation("org1")
        assert reputation > 0.5

    def test_propagate_influence(self, social_network):
        """Test influence propagation through network"""
        # Create chain: org1 -> org2 -> org3
        social_network.form_relationship("org1", "org2", strength=0.9, trust=0.8)
        social_network.form_relationship("org2", "org3", strength=0.9, trust=0.8)

        reached = social_network.propagate_influence(
            "org1",
            "news",
            {"info": "test"},
            max_hops=2
        )

        assert "org2" in reached
        assert "org3" in reached

    def test_form_group(self, social_network):
        """Test forming a group"""
        group_id = social_network.form_group("hunters", ["org1", "org2", "org3"])

        group = social_network.get_group(group_id)
        assert group is not None
        assert group.name == "hunters"
        assert group.size == 3

    def test_dissolve_group(self, social_network):
        """Test dissolving a group"""
        group_id = social_network.form_group("temp", ["org1", "org2"])
        assert social_network.dissolve_group(group_id) is True
        assert social_network.get_group(group_id) is None

    def test_add_to_group(self, social_network):
        """Test adding organism to group"""
        group_id = social_network.form_group("team", ["org1"])
        assert social_network.add_to_group(group_id, "org2") is True

        group = social_network.get_group(group_id)
        assert "org2" in group.members

    def test_remove_from_group(self, social_network):
        """Test removing organism from group"""
        group_id = social_network.form_group("team", ["org1", "org2"])
        assert social_network.remove_from_group(group_id, "org1") is True

        group = social_network.get_group(group_id)
        assert "org1" not in group.members

    def test_set_group_leader(self, social_network):
        """Test setting group leader"""
        group_id = social_network.form_group("team", ["org1", "org2"])
        assert social_network.set_group_leader(group_id, "org1") is True

        group = social_network.get_group(group_id)
        assert group.leader_id == "org1"

        # Check leader/follower relationships created
        assert social_network.has_relationship("org1", "org2")

    def test_get_organism_groups(self, social_network):
        """Test getting groups for an organism"""
        social_network.form_group("group1", ["org1", "org2"])
        social_network.form_group("group2", ["org1", "org3"])

        groups = social_network.get_organism_groups("org1")
        assert len(groups) == 2

    def test_calculate_group_cohesion(self, social_network):
        """Test calculating group cohesion"""
        group_id = social_network.form_group("team", ["org1", "org2", "org3"])

        # Create strong relationships between members
        social_network.form_relationship("org1", "org2", strength=0.9, trust=0.8)
        social_network.form_relationship("org2", "org1", strength=0.9, trust=0.8)
        social_network.form_relationship("org1", "org3", strength=0.9, trust=0.8)
        social_network.form_relationship("org3", "org1", strength=0.9, trust=0.8)
        social_network.form_relationship("org2", "org3", strength=0.9, trust=0.8)
        social_network.form_relationship("org3", "org2", strength=0.9, trust=0.8)

        cohesion = social_network.calculate_group_cohesion(group_id)
        assert cohesion > 0.5  # Should be cohesive

    def test_find_path(self, social_network):
        """Test finding path between organisms"""
        social_network.form_relationship("org1", "org2")
        social_network.form_relationship("org2", "org3")
        social_network.form_relationship("org3", "org4")

        path = social_network.find_path("org1", "org4")
        assert path is not None
        assert path[0] == "org1"
        assert path[-1] == "org4"

    def test_find_path_no_connection(self, social_network):
        """Test finding path when no connection exists"""
        social_network.form_relationship("org1", "org2")
        # org3 is isolated

        path = social_network.find_path("org1", "org3")
        assert path is None

    def test_get_common_connections(self, social_network):
        """Test getting common connections"""
        social_network.form_relationship("org1", "common1")
        social_network.form_relationship("org1", "common2")
        social_network.form_relationship("org2", "common1")
        social_network.form_relationship("org2", "common2")
        social_network.form_relationship("org1", "only1")

        common = social_network.get_common_connections("org1", "org2")
        assert "common1" in common
        assert "common2" in common
        assert "only1" not in common

    def test_remove_organism(self, social_network):
        """Test removing an organism from the network entirely"""
        social_network.form_relationship("org1", "org2")
        social_network.form_relationship("org2", "org1")
        group_id = social_network.form_group("team", ["org1", "org2"])

        social_network.remove_organism("org1")

        assert not social_network.has_relationship("org1", "org2")
        assert not social_network.has_relationship("org2", "org1")
        group = social_network.get_group(group_id)
        assert "org1" not in group.members

    def test_network_stats(self, social_network):
        """Test getting network statistics"""
        social_network.form_relationship("org1", "org2", RelationshipType.ALLY)
        social_network.form_relationship("org1", "org3", RelationshipType.RIVAL)
        social_network.form_group("team", ["org1", "org2"])

        stats = social_network.get_network_stats()
        assert stats["total_relationships"] == 2
        assert stats["total_groups"] == 1
        assert "ally" in stats["relationship_types"]

    def test_serialization(self, social_network):
        """Test network serialization/deserialization"""
        social_network.form_relationship("org1", "org2", RelationshipType.ALLY, trust=0.7)
        social_network.form_group("team", ["org1", "org2"])

        data = social_network.to_dict()
        restored = SocialNetwork.from_dict(data, emit_events=False)

        assert restored.has_relationship("org1", "org2")
        rel = restored.get_relationship("org1", "org2")
        assert rel.type == RelationshipType.ALLY

    def test_events_emitted(self, network_with_events, fresh_event_bus):
        """Test that events are emitted correctly"""
        events = []
        fresh_event_bus.subscribe("relationship.formed", lambda e: events.append(e))

        network_with_events.form_relationship("org1", "org2")

        assert len(events) == 1
        assert events[0].data["source_id"] == "org1"


# =============================================================================
# Signal Tests
# =============================================================================

class TestSignal:
    """Tests for the Signal dataclass"""

    def test_signal_creation(self):
        """Test creating a signal"""
        signal = Signal(
            type=SignalType.WARNING,
            source_id="org1",
            source_position=(10.0, 20.0),
            data={"threat": "predator"},
            strength=0.9,
            range=15.0
        )

        assert signal.type == SignalType.WARNING
        assert signal.source_id == "org1"
        assert signal.data["threat"] == "predator"

    def test_signal_strength_at_position(self):
        """Test signal strength calculation"""
        signal = Signal(
            type=SignalType.WARNING,
            source_id="org1",
            source_position=(0.0, 0.0),
            strength=1.0,
            range=10.0,
            attenuation=0.1
        )

        # At source position
        assert signal.get_strength_at((0.0, 0.0)) == 1.0

        # At distance 5
        strength_at_5 = signal.get_strength_at((5.0, 0.0))
        assert strength_at_5 == pytest.approx(0.5, rel=0.1)

        # Out of range
        assert signal.get_strength_at((15.0, 0.0)) == 0.0


# =============================================================================
# CommunicationSystem Tests
# =============================================================================

class TestCommunicationSystem:
    """Tests for the CommunicationSystem class"""

    def test_creation(self, organism):
        """Test creating a communication system"""
        comm = CommunicationSystem("comm", owner=organism)
        assert comm.name == "comm"
        assert comm.owner == organism

    def test_set_position_callback(self, organism):
        """Test setting position callback"""
        comm = CommunicationSystem("comm", owner=organism)
        comm.set_position_callback(lambda: (10.0, 20.0))

        # Verify callback works (internal test)
        assert comm._position_callback() == (10.0, 20.0)

    def test_broadcast_signal(self, organism, fresh_event_bus):
        """Test broadcasting a signal"""
        comm = CommunicationSystem("comm", owner=organism)
        comm.set_position_callback(lambda: (0.0, 0.0))

        events = []
        fresh_event_bus.subscribe("signal.broadcast", lambda e: events.append(e))

        signal = comm.broadcast_signal(
            SignalType.WARNING,
            {"threat": "fire"},
            strength=0.8
        )

        assert signal.type == SignalType.WARNING
        assert signal.source_id == organism.id
        assert len(events) == 1

    def test_send_message(self, organism, fresh_event_bus):
        """Test sending a direct message"""
        comm = CommunicationSystem("comm", owner=organism)

        events = []
        fresh_event_bus.subscribe("message.sent", lambda e: events.append(e))

        message = comm.send_message(
            "target_org",
            {"info": "food at (10, 20)"},
            MessageType.INFORMATION
        )

        assert message.sender_id == organism.id
        assert message.recipient_id == "target_org"
        assert len(events) == 1

    def test_send_message_with_response(self, organism):
        """Test sending a message that requires response"""
        comm = CommunicationSystem("comm", owner=organism)

        message = comm.send_message(
            "target",
            {"request": "help"},
            MessageType.REQUEST,
            requires_response=True
        )

        assert message.requires_response is True
        assert message.type == MessageType.REQUEST

    def test_reply_to_message(self, organism):
        """Test replying to a message"""
        comm = CommunicationSystem("comm", owner=organism)

        original = Message(
            sender_id="other",
            recipient_id=organism.id,
            type=MessageType.REQUEST,
            content={"request": "help"}
        )

        reply = comm.reply_to_message(original, {"answer": "yes"}, accept=True)

        assert reply.recipient_id == "other"
        assert reply.type == MessageType.ACCEPTANCE
        assert "reply_to" in reply.content

    def test_assess_credibility_unknown(self, organism):
        """Test credibility assessment for unknown organism"""
        comm = CommunicationSystem("comm", owner=organism)

        credibility = comm.assess_credibility("unknown_org")
        assert credibility == 0.5  # Neutral for unknown

    def test_assess_credibility_with_history(self, organism):
        """Test credibility assessment with history"""
        comm = CommunicationSystem("comm", owner=organism)

        # Record truthful communications
        for _ in range(5):
            comm.record_verification("org2", was_truthful=True)

        credibility = comm.assess_credibility("org2")
        assert credibility > 0.8  # Should be high

        # Now record some lies
        for _ in range(5):
            comm.record_verification("org2", was_truthful=False)

        credibility = comm.assess_credibility("org2")
        assert credibility < 0.8  # Should have decreased

    def test_get_known_deceivers(self, organism):
        """Test identifying known deceivers"""
        comm = CommunicationSystem("comm", owner=organism)

        # Make org2 a deceiver
        for _ in range(10):
            comm.record_verification("deceiver", was_truthful=False)

        # Make org3 trustworthy
        for _ in range(10):
            comm.record_verification("honest", was_truthful=True)

        deceivers = comm.get_known_deceivers(threshold=0.3)
        assert "deceiver" in deceivers
        assert "honest" not in deceivers

    def test_register_signal_handler(self, organism):
        """Test registering signal handlers"""
        comm = CommunicationSystem("comm", owner=organism)

        handled = []
        def handler(signal):
            handled.append(signal)

        comm.register_signal_handler(SignalType.WARNING, handler)

        # Manually add a received signal and tick
        signal = Signal(
            type=SignalType.WARNING,
            source_id="other",
            data={"test": True}
        )
        comm._received_signals.append(signal)
        comm.tick()

        assert len(handled) == 1

    def test_get_state(self, organism):
        """Test getting subsystem state"""
        comm = CommunicationSystem("comm", owner=organism, base_range=15.0)
        comm.record_verification("org2", was_truthful=True)

        state = comm.get_state()
        assert state["base_range"] == 15.0
        assert state["known_organisms"] == 1

    def test_set_state(self, organism):
        """Test restoring subsystem state"""
        comm = CommunicationSystem("comm", owner=organism)

        state = {
            "base_range": 20.0,
            "communication_history": {}
        }
        comm.set_state(state)

        assert comm._base_range == 20.0

    def test_reset(self, organism):
        """Test resetting the subsystem"""
        comm = CommunicationSystem("comm", owner=organism)
        comm.record_verification("org2", was_truthful=True)
        comm._received_signals.append(Signal(type=SignalType.WARNING, source_id="x"))

        comm.reset()

        assert len(comm._received_signals) == 0
        assert len(comm._communication_history) == 0


# =============================================================================
# CommunicationChannel Tests
# =============================================================================

class TestCommunicationChannel:
    """Tests for the CommunicationChannel class"""

    def test_creation(self):
        """Test creating a channel"""
        channel = CommunicationChannel("main", default_range=50.0)
        assert channel.name == "main"

    def test_subscribe(self):
        """Test subscribing to channel"""
        channel = CommunicationChannel("main")
        channel.subscribe("org1", (10.0, 20.0))
        channel.subscribe("org2", (30.0, 40.0))

        stats = channel.get_stats()
        assert stats["subscribers"] == 2

    def test_unsubscribe(self):
        """Test unsubscribing from channel"""
        channel = CommunicationChannel("main")
        channel.subscribe("org1", (10.0, 20.0))
        channel.unsubscribe("org1")

        stats = channel.get_stats()
        assert stats["subscribers"] == 0

    def test_update_position(self):
        """Test updating subscriber position"""
        channel = CommunicationChannel("main")
        channel.subscribe("org1", (0.0, 0.0))
        channel.update_position("org1", (10.0, 10.0))

        assert channel._subscribers["org1"] == (10.0, 10.0)

    def test_broadcast(self):
        """Test broadcasting on channel"""
        channel = CommunicationChannel("main")
        channel.subscribe("org1", (0.0, 0.0))
        channel.subscribe("org2", (5.0, 0.0))
        channel.subscribe("org3", (100.0, 0.0))  # Out of range

        signal = Signal(
            type=SignalType.WARNING,
            source_id="org1",
            source_position=(0.0, 0.0),
            range=10.0
        )

        in_range = channel.broadcast(signal)
        assert in_range == 1  # Only org2 in range (org1 is source)

    def test_get_signals_at(self):
        """Test getting signals at position"""
        channel = CommunicationChannel("main")

        signal = Signal(
            type=SignalType.WARNING,
            source_id="org1",
            source_position=(0.0, 0.0),
            range=10.0,
            strength=1.0
        )
        channel.broadcast(signal)

        # Get signals at nearby position
        signals = channel.get_signals_at((5.0, 0.0))
        assert len(signals) == 1
        assert signals[0][0].type == SignalType.WARNING
        assert signals[0][1] > 0  # Has some strength

        # Get signals at far position
        signals = channel.get_signals_at((100.0, 0.0))
        assert len(signals) == 0

    def test_signal_ttl(self):
        """Test signal time-to-live"""
        channel = CommunicationChannel("main", signal_ttl=2.0)

        signal = Signal(type=SignalType.WARNING, source_id="org1")
        channel.broadcast(signal)

        assert len(channel._active_signals) == 1

        # Tick twice to expire
        channel.tick()
        channel.tick()

        assert len(channel._active_signals) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with SimpleOrganism"""

    def test_organism_with_communication(self, fresh_event_bus):
        """Test adding CommunicationSystem to SimpleOrganism"""
        org = SimpleOrganism(name="Communicator")
        comm = CommunicationSystem("communication", owner=org)
        org.add_subsystem(comm)

        assert org.get_subsystem("communication") is not None
        assert comm.owner == org

    def test_organism_broadcast(self, fresh_event_bus):
        """Test organism broadcasting signals"""
        org = SimpleOrganism(name="Broadcaster")
        comm = CommunicationSystem("communication", owner=org)
        comm.set_position_callback(lambda: (50.0, 50.0))
        org.add_subsystem(comm)

        signal = comm.broadcast_signal(SignalType.MATING, {"fitness": 0.9})

        assert signal.source_id == org.id
        assert signal.source_position == (50.0, 50.0)

    def test_social_network_with_organisms(self, fresh_event_bus):
        """Test social network with actual organisms"""
        org1 = SimpleOrganism(name="Alpha")
        org2 = SimpleOrganism(name="Beta")
        org3 = SimpleOrganism(name="Gamma")

        network = SocialNetwork(emit_events=False)

        # Form relationships using organism IDs
        network.form_relationship(org1.id, org2.id, RelationshipType.ALLY)
        network.form_relationship(org1.id, org3.id, RelationshipType.MENTOR)
        network.form_relationship(org2.id, org3.id, RelationshipType.SIBLING)

        # Form a group
        group_id = network.form_group("pack", [org1.id, org2.id, org3.id])
        network.set_group_leader(group_id, org1.id)

        # Verify relationships
        assert network.has_relationship(org1.id, org2.id)
        influence = network.get_influence(org1.id)
        assert influence > 0

    def test_communication_between_organisms(self, fresh_event_bus):
        """Test organisms communicating with each other"""
        org1 = SimpleOrganism(name="Sender")
        org2 = SimpleOrganism(name="Receiver")

        comm1 = CommunicationSystem("communication", owner=org1)
        comm2 = CommunicationSystem("communication", owner=org2)

        org1.add_subsystem(comm1)
        org2.add_subsystem(comm2)

        # Send direct message
        msg = comm1.send_message(
            org2.id,
            {"info": "food location", "coords": (10, 20)}
        )

        assert msg.sender_id == org1.id
        assert msg.recipient_id == org2.id
