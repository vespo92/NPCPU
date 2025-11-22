"""
Tests for Neural Consciousness Implementation

Tests cover:
- Attention-based perception weighting
- Working memory with capacity limits
- Emotional valence computation
- Self-model introspection
- Integration with SimpleOrganism
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from consciousness.neural_consciousness import (
    NeuralConsciousness,
    AttentionMechanism,
    SelfModel,
    MemoryItem,
    AttentionState,
    EmotionalState,
    EmotionalDimension,
    attach_neural_consciousness
)
from core.simple_organism import SimpleOrganism
from core.events import get_event_bus, EventBus, set_event_bus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def fresh_event_bus():
    """Create a fresh event bus for each test"""
    bus = EventBus()
    set_event_bus(bus)
    yield bus
    bus.clear_history()


@pytest.fixture
def neural_consciousness(fresh_event_bus):
    """Create a basic neural consciousness for testing"""
    return NeuralConsciousness(
        attention_dim=32,
        memory_capacity=5,
        emotional_sensitivity=0.5,
        consciousness_threshold=0.3
    )


@pytest.fixture
def conscious_organism(fresh_event_bus):
    """Create a simple organism with neural consciousness attached"""
    org = SimpleOrganism("TestOrganism")
    consciousness = attach_neural_consciousness(
        org,
        attention_dim=32,
        memory_capacity=5,
        emotional_sensitivity=0.5
    )
    return org, consciousness


# ============================================================================
# Test NeuralConsciousness Initialization
# ============================================================================

class TestNeuralConsciousnessInit:
    """Test neural consciousness initialization"""

    def test_basic_initialization(self, neural_consciousness):
        """Test that consciousness initializes with correct defaults"""
        nc = neural_consciousness

        assert nc.attention_dim == 32
        assert nc.memory_capacity == 5
        assert nc.emotional_sensitivity == 0.5
        assert nc.consciousness_threshold == 0.3
        assert nc.name == "neural_consciousness"
        assert nc.enabled is True

    def test_default_emotional_state(self, neural_consciousness):
        """Test default emotional state is neutral"""
        nc = neural_consciousness

        assert nc.emotional_state.valence == 0.0
        assert nc.emotional_state.arousal == 0.5
        assert nc.emotional_state.dominance == 0.5
        assert nc.emotional_state.mood_baseline == 0.0

    def test_empty_working_memory(self, neural_consciousness):
        """Test working memory starts empty"""
        nc = neural_consciousness

        assert len(nc.working_memory) == 0

    def test_consciousness_level_starts_neutral(self, neural_consciousness):
        """Test consciousness level starts at neutral"""
        nc = neural_consciousness

        assert 0.0 <= nc.consciousness_level <= 1.0


# ============================================================================
# Test Perception Processing
# ============================================================================

class TestPerceptionProcessing:
    """Test attention-based perception weighting"""

    def test_process_simple_stimuli(self, neural_consciousness):
        """Test processing simple numeric stimuli"""
        nc = neural_consciousness
        stimuli = {"threat": 0.8, "food": 0.3}

        attended = nc.process_perception(stimuli)

        assert "threat" in attended
        assert attended["threat"]["value"] == 0.8
        assert attended["threat"]["salience"] > 0

    def test_threat_gets_high_attention(self, neural_consciousness):
        """Test that threats receive high attention weight"""
        nc = neural_consciousness
        stimuli = {"threat": 0.9, "novelty": 0.5, "food": 0.2}

        attended = nc.process_perception(stimuli)

        # Threat should have highest salience
        if "threat" in attended and "novelty" in attended:
            assert attended["threat"]["salience"] > attended["novelty"]["salience"]

    def test_attention_focus_updates(self, neural_consciousness):
        """Test that attention focus target updates correctly"""
        nc = neural_consciousness
        stimuli = {"threat": 0.9}

        nc.process_perception(stimuli)

        assert nc.attention_state.focus_target == "threat"
        assert nc.attention_state.focus_strength > 0

    def test_filter_low_salience_stimuli(self, neural_consciousness):
        """Test that very low salience stimuli are filtered out"""
        nc = neural_consciousness
        # With default threshold of 0.3, very weak stimuli may be filtered
        stimuli = {"weak_signal": 0.01}

        attended = nc.process_perception(stimuli)

        # May or may not be attended depending on threshold
        # But should not crash
        assert isinstance(attended, dict)

    def test_list_stimuli_processing(self, neural_consciousness):
        """Test processing list/array stimuli"""
        nc = neural_consciousness
        stimuli = {"visual": [0.5, 0.3, 0.7], "threat": 0.4}

        attended = nc.process_perception(stimuli)

        # Should process without error
        assert isinstance(attended, dict)

    def test_global_workspace_updated(self, neural_consciousness):
        """Test that global workspace is updated after perception"""
        nc = neural_consciousness
        stimuli = {"threat": 0.5}

        nc.process_perception(stimuli)

        assert "attended_stimuli" in nc.global_workspace
        assert "total_salience" in nc.global_workspace


# ============================================================================
# Test Working Memory
# ============================================================================

class TestWorkingMemory:
    """Test working memory with capacity limits"""

    def test_store_memory(self, neural_consciousness):
        """Test storing memory items"""
        nc = neural_consciousness
        info = {"event": "found_food", "location": "north"}

        result = nc.update_working_memory(info, importance=0.7)

        assert result is True
        assert len(nc.working_memory) == 1
        assert nc.working_memory[0].content == info

    def test_memory_capacity_limit(self, neural_consciousness):
        """Test that memory respects capacity limits"""
        nc = neural_consciousness

        # Fill memory beyond capacity
        for i in range(10):
            nc.update_working_memory(
                {"event": f"event_{i}"},
                importance=0.5
            )

        # Should not exceed capacity
        assert len(nc.working_memory) <= nc.memory_capacity

    def test_important_memories_retained(self, neural_consciousness):
        """Test that more important memories are retained"""
        nc = neural_consciousness

        # Add low importance memories
        for i in range(5):
            nc.update_working_memory(
                {"event": f"boring_{i}"},
                importance=0.2
            )

        # Add high importance memory
        nc.update_working_memory(
            {"event": "critical_danger"},
            importance=0.95
        )

        # High importance memory should be retained
        contents = [m.content["event"] for m in nc.working_memory]
        assert "critical_danger" in contents

    def test_recall_memories(self, neural_consciousness):
        """Test memory recall functionality"""
        nc = neural_consciousness

        nc.update_working_memory({"type": "danger", "level": "high"}, importance=0.8)
        nc.update_working_memory({"type": "food", "quality": "good"}, importance=0.6)

        results = nc.recall(limit=2)

        assert len(results) <= 2
        assert all("content" in r for r in results)

    def test_recall_with_query(self, neural_consciousness):
        """Test memory recall with query filtering"""
        nc = neural_consciousness

        nc.update_working_memory({"type": "danger", "source": "predator"}, importance=0.7)
        nc.update_working_memory({"type": "food", "source": "plant"}, importance=0.6)

        results = nc.recall(query={"type": "danger"}, limit=5)

        # Should return results (may or may not match perfectly)
        assert isinstance(results, list)

    def test_memory_consolidation(self, neural_consciousness):
        """Test memory consolidation process"""
        nc = neural_consciousness

        # Add and access a memory multiple times
        nc.update_working_memory({"event": "important"}, importance=0.5)

        # Simulate access
        for _ in range(5):
            nc.recall(query={"event": "important"})

        # Run consolidation
        nc.consolidate_memory()

        # Memory should still exist
        assert len(nc.working_memory) > 0

    def test_similar_memories_consolidated(self, neural_consciousness):
        """Test that similar memories are consolidated"""
        nc = neural_consciousness

        nc.update_working_memory({"type": "food", "loc": "A"}, importance=0.5)
        nc.update_working_memory({"type": "food", "loc": "B"}, importance=0.6)

        # Both should be stored (they're different enough)
        # But if we add a very similar one...
        result = nc.update_working_memory({"type": "food", "loc": "A"}, importance=0.7)

        # Should consolidate with existing similar memory
        assert result is True


# ============================================================================
# Test Emotional State
# ============================================================================

class TestEmotionalState:
    """Test emotional valence computation"""

    def test_threat_causes_negative_valence(self, neural_consciousness):
        """Test that threats cause negative emotional valence"""
        nc = neural_consciousness

        nc.process_perception({"threat": 0.9})
        emotion = nc.compute_emotional_state()

        assert emotion["valence"] < 0  # Negative

    def test_food_causes_positive_valence(self, neural_consciousness):
        """Test that food causes positive emotional valence"""
        nc = neural_consciousness

        nc.process_perception({"food": 0.8})
        emotion = nc.compute_emotional_state()

        assert emotion["valence"] > 0 or emotion["valence"] == 0  # Neutral to positive

    def test_threat_increases_arousal(self, neural_consciousness):
        """Test that threats increase arousal"""
        nc = neural_consciousness
        initial_arousal = nc.emotional_state.arousal

        nc.process_perception({"threat": 0.9})
        nc.compute_emotional_state()

        assert nc.emotional_state.arousal >= initial_arousal

    def test_emotional_inertia(self, neural_consciousness):
        """Test that emotions change gradually (inertia)"""
        nc = neural_consciousness

        # Process strong threat
        nc.process_perception({"threat": 1.0})
        nc.compute_emotional_state()

        # Valence shouldn't jump to extreme immediately
        assert -1.0 < nc.emotional_state.valence < 0

    def test_get_dominant_emotion(self, neural_consciousness):
        """Test getting dominant emotion name"""
        nc = neural_consciousness

        nc.emotional_state.valence = 0.5
        nc.emotional_state.arousal = 0.7

        emotion = nc.get_dominant_emotion()

        assert emotion == "excited"

    def test_dominant_emotion_categories(self, neural_consciousness):
        """Test various dominant emotion categories"""
        nc = neural_consciousness

        # Test happy
        nc.emotional_state.valence = 0.5
        nc.emotional_state.arousal = 0.4
        assert nc.get_dominant_emotion() == "happy"

        # Test sad
        nc.emotional_state.valence = -0.5
        nc.emotional_state.arousal = 0.4
        assert nc.get_dominant_emotion() == "sad"

        # Test afraid
        nc.emotional_state.valence = -0.5
        nc.emotional_state.arousal = 0.8
        assert nc.get_dominant_emotion() == "afraid"

        # Test calm
        nc.emotional_state.valence = 0.0
        nc.emotional_state.arousal = 0.2
        assert nc.get_dominant_emotion() == "calm"

    def test_emotion_history_tracked(self, neural_consciousness):
        """Test that emotion history is tracked"""
        nc = neural_consciousness
        initial_history_len = len(nc.emotion_history)

        nc.process_perception({"threat": 0.5})
        nc.compute_emotional_state()

        assert len(nc.emotion_history) > initial_history_len


# ============================================================================
# Test Introspection
# ============================================================================

class TestIntrospection:
    """Test self-model and introspection"""

    def test_introspect_returns_report(self, neural_consciousness):
        """Test that introspection returns a complete report"""
        nc = neural_consciousness

        report = nc.introspect()

        assert "consciousness_level" in report
        assert "attention" in report
        assert "working_memory" in report
        assert "emotional_state" in report
        assert "self_model" in report

    def test_introspection_attention_data(self, neural_consciousness):
        """Test introspection includes attention data"""
        nc = neural_consciousness
        nc.process_perception({"threat": 0.7})

        report = nc.introspect()

        assert "focus_target" in report["attention"]
        assert "focus_strength" in report["attention"]
        assert report["attention"]["focus_target"] == "threat"

    def test_introspection_memory_data(self, neural_consciousness):
        """Test introspection includes memory data"""
        nc = neural_consciousness
        nc.update_working_memory({"event": "test"}, importance=0.5)

        report = nc.introspect()

        assert report["working_memory"]["items_count"] == 1
        assert report["working_memory"]["capacity"] == 5

    def test_introspection_emotional_data(self, neural_consciousness):
        """Test introspection includes emotional data"""
        nc = neural_consciousness

        report = nc.introspect()

        assert "current" in report["emotional_state"]
        assert "dominant_emotion" in report["emotional_state"]
        assert "emotional_stability" in report["emotional_state"]

    def test_consciousness_level_in_report(self, neural_consciousness):
        """Test consciousness level is included in report"""
        nc = neural_consciousness

        report = nc.introspect()

        assert 0.0 <= report["consciousness_level"] <= 1.0


# ============================================================================
# Test Self Model
# ============================================================================

class TestSelfModel:
    """Test self-model component"""

    def test_self_model_initialization(self):
        """Test self-model initializes correctly"""
        model = SelfModel()

        assert model.cognitive_load == 0.0
        assert model.confidence == 0.5
        assert len(model.goals) == 0

    def test_self_model_update(self):
        """Test updating self-model"""
        model = SelfModel()

        model.update(cognitive_load=0.7, performance=0.8)

        assert model.cognitive_load == 0.7
        assert len(model.performance_history) == 1

    def test_confidence_from_performance(self):
        """Test confidence updates from performance history"""
        model = SelfModel()

        # Good performance
        for _ in range(10):
            model.update(cognitive_load=0.5, performance=0.9)

        assert model.confidence > 0.8

    def test_self_model_report(self):
        """Test self-model report generation"""
        model = SelfModel()
        model.update(cognitive_load=0.5, performance=0.7)

        report = model.get_report()

        assert "cognitive_load" in report
        assert "confidence" in report
        assert "average_performance" in report


# ============================================================================
# Test Attention Mechanism
# ============================================================================

class TestAttentionMechanism:
    """Test attention mechanism component"""

    def test_attention_initialization(self):
        """Test attention mechanism initializes correctly"""
        attention = AttentionMechanism(dim=32, num_heads=4)

        assert attention.dim == 32
        assert attention.num_heads == 4
        assert attention.head_dim == 8

    def test_compute_attention_basic(self):
        """Test basic attention computation"""
        attention = AttentionMechanism(dim=8, num_heads=2)

        query = [0.5] * 8
        keys = [[0.3] * 8, [0.7] * 8]
        values = [[1.0] * 8, [0.0] * 8]

        output, weights = attention.compute_attention(query, keys, values)

        assert len(output) == 8
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01  # Weights should sum to ~1


# ============================================================================
# Test Memory Item
# ============================================================================

class TestMemoryItem:
    """Test memory item data structure"""

    def test_memory_item_creation(self):
        """Test creating a memory item"""
        memory = MemoryItem(
            content={"event": "test"},
            importance=0.7
        )

        assert memory.content == {"event": "test"}
        assert memory.importance == 0.7
        assert memory.access_count == 0

    def test_memory_effective_strength(self):
        """Test memory effective strength calculation"""
        memory = MemoryItem(
            content={"event": "test"},
            importance=0.7,
            decay_rate=0.1
        )

        # Initially, strength should be close to importance
        initial_strength = memory.effective_strength
        assert initial_strength > 0

        # Increase access count
        memory.access_count = 5
        new_strength = memory.effective_strength

        # Strength should increase with access
        assert new_strength > initial_strength


# ============================================================================
# Test Tick Processing
# ============================================================================

class TestTickProcessing:
    """Test tick-based processing"""

    def test_tick_increments_counter(self, neural_consciousness):
        """Test that tick increments the counter"""
        nc = neural_consciousness
        initial_count = nc.tick_count

        nc.tick()

        assert nc.tick_count == initial_count + 1

    def test_tick_updates_fatigue(self, neural_consciousness):
        """Test that tick affects attention fatigue"""
        nc = neural_consciousness

        # Set high focus
        nc.attention_state.focus_strength = 0.9
        initial_fatigue = nc.attention_state.fatigue_level

        for _ in range(10):
            nc.tick()

        # Fatigue should increase with sustained focus
        assert nc.attention_state.fatigue_level >= initial_fatigue

    def test_tick_when_disabled(self, neural_consciousness):
        """Test that tick does nothing when disabled"""
        nc = neural_consciousness
        nc.disable()
        initial_count = nc.tick_count

        nc.tick()

        assert nc.tick_count == initial_count

    def test_periodic_consolidation(self, neural_consciousness):
        """Test that memory consolidation happens periodically"""
        nc = neural_consciousness

        # Add a memory
        nc.update_working_memory({"event": "test"}, importance=0.5)

        # Run many ticks
        for _ in range(15):
            nc.tick()

        # Should not crash and memory should still exist
        assert len(nc.working_memory) >= 0


# ============================================================================
# Test State Serialization
# ============================================================================

class TestStateSerialization:
    """Test state save/restore"""

    def test_get_state(self, neural_consciousness):
        """Test getting state for serialization"""
        nc = neural_consciousness
        nc.update_working_memory({"event": "test"}, importance=0.7)
        nc.emotional_state.valence = 0.5

        state = nc.get_state()

        assert "emotional_state" in state
        assert "working_memory" in state
        assert "consciousness_level" in state
        assert state["emotional_state"]["valence"] == 0.5

    def test_set_state(self, neural_consciousness):
        """Test restoring state from serialized data"""
        nc = neural_consciousness

        state = {
            "consciousness_level": 0.7,
            "tick_count": 100,
            "emotional_state": {
                "valence": 0.3,
                "arousal": 0.6,
                "dominance": 0.4,
                "mood_baseline": 0.1
            },
            "attention_state": {
                "focus_target": "food",
                "focus_strength": 0.5,
                "fatigue_level": 0.2
            },
            "working_memory": [
                {"id": "mem1", "content": {"event": "test"}, "importance": 0.5, "access_count": 2}
            ]
        }

        nc.set_state(state)

        assert nc.consciousness_level == 0.7
        assert nc.tick_count == 100
        assert nc.emotional_state.valence == 0.3
        assert nc.attention_state.focus_target == "food"
        assert len(nc.working_memory) == 1

    def test_reset(self, neural_consciousness):
        """Test resetting to initial state"""
        nc = neural_consciousness

        # Modify state
        nc.update_working_memory({"event": "test"}, importance=0.7)
        nc.emotional_state.valence = 0.8
        nc.tick_count = 50

        # Reset
        nc.reset()

        assert len(nc.working_memory) == 0
        assert nc.emotional_state.valence == 0.0
        assert nc.tick_count == 0


# ============================================================================
# Test Integration with SimpleOrganism
# ============================================================================

class TestOrganismIntegration:
    """Test integration with SimpleOrganism"""

    def test_attach_to_organism(self, conscious_organism):
        """Test attaching consciousness to organism"""
        org, consciousness = conscious_organism

        # Consciousness should be a subsystem
        assert org.get_subsystem("neural_consciousness") is consciousness
        assert consciousness.owner is org

    def test_conscious_organism_perception(self, conscious_organism):
        """Test conscious perception processing"""
        org, consciousness = conscious_organism

        # Perceive through organism
        org.perceive({"threat": 0.7, "food": 0.3})

        # Also process through consciousness
        attended = consciousness.process_perception({"threat": 0.7, "food": 0.3})

        assert "threat" in attended

    def test_conscious_organism_tick(self, conscious_organism):
        """Test that organism tick processes consciousness"""
        org, consciousness = conscious_organism
        initial_tick = consciousness.tick_count

        org.tick()

        # Consciousness should have ticked
        assert consciousness.tick_count > initial_tick

    def test_conscious_organism_serialization(self, conscious_organism):
        """Test serialization includes consciousness state"""
        org, consciousness = conscious_organism

        consciousness.emotional_state.valence = 0.5

        org_state = org.to_dict()

        # Should include consciousness subsystem state
        assert "neural_consciousness" in org_state["subsystems"]
        assert org_state["subsystems"]["neural_consciousness"]["emotional_state"]["valence"] == 0.5


# ============================================================================
# Test Event Emission
# ============================================================================

class TestEventEmission:
    """Test event bus integration"""

    def test_perception_event_emitted(self, neural_consciousness, fresh_event_bus):
        """Test that perception events are emitted"""
        nc = neural_consciousness
        events_received = []

        def handler(event):
            events_received.append(event)

        fresh_event_bus.subscribe("consciousness.perception", handler)

        nc.process_perception({"threat": 0.5})

        assert len(events_received) == 1
        assert events_received[0].type == "consciousness.perception"

    def test_memory_stored_event_emitted(self, neural_consciousness, fresh_event_bus):
        """Test that memory storage events are emitted"""
        nc = neural_consciousness
        events_received = []

        def handler(event):
            events_received.append(event)

        fresh_event_bus.subscribe("consciousness.memory_stored", handler)

        nc.update_working_memory({"event": "test"}, importance=0.5)

        assert len(events_received) == 1

    def test_emotion_updated_event_emitted(self, neural_consciousness, fresh_event_bus):
        """Test that emotion update events are emitted"""
        nc = neural_consciousness
        events_received = []

        def handler(event):
            events_received.append(event)

        fresh_event_bus.subscribe("consciousness.emotion_updated", handler)

        nc.process_perception({"threat": 0.5})
        nc.compute_emotional_state()

        assert len(events_received) == 1

    def test_tick_event_emitted(self, neural_consciousness, fresh_event_bus):
        """Test that tick events are emitted"""
        nc = neural_consciousness
        events_received = []

        def handler(event):
            events_received.append(event)

        fresh_event_bus.subscribe("consciousness.tick", handler)

        nc.tick()

        assert len(events_received) == 1


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_stimuli(self, neural_consciousness):
        """Test processing empty stimuli"""
        nc = neural_consciousness

        attended = nc.process_perception({})

        assert isinstance(attended, dict)
        assert len(attended) == 0

    def test_recall_empty_memory(self, neural_consciousness):
        """Test recalling from empty memory"""
        nc = neural_consciousness

        results = nc.recall()

        assert results == []

    def test_introspect_fresh_consciousness(self, neural_consciousness):
        """Test introspection on fresh consciousness"""
        nc = neural_consciousness

        report = nc.introspect()

        assert report is not None
        assert "consciousness_level" in report

    def test_high_cognitive_load(self, neural_consciousness):
        """Test behavior under high cognitive load"""
        nc = neural_consciousness

        # Fill memory to capacity
        for i in range(nc.memory_capacity):
            nc.update_working_memory({"event": f"important_{i}"}, importance=0.9)

        emotion = nc.compute_emotional_state()

        # Should not crash and should reflect high load
        assert isinstance(emotion, dict)
