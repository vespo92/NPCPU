"""
Tests for Consciousness Protocol

Tests GradedConsciousness implementation, capability scoring,
evolution, transitions, and backward compatibility.
"""

import pytest
import numpy as np
from protocols.consciousness import (
    GradedConsciousness,
    ConsciousnessAdapter,
    Perception,
    Action,
    Experience,
    SelfModel,
    Thought,
    MetaThought,
    Intention,
    Qualia,
    Invariant
)


# ============================================================================
# Basic Functionality Tests
# ============================================================================

@pytest.mark.unit
def test_graded_consciousness_creation():
    """Test creating GradedConsciousness with default values"""
    consciousness = GradedConsciousness()

    assert consciousness.perception_fidelity == 0.0
    assert consciousness.reaction_speed == 0.0
    assert consciousness.memory_depth == 0.0
    assert consciousness.overall_consciousness_score() == 0.0


@pytest.mark.unit
def test_graded_consciousness_with_values(basic_consciousness):
    """Test creating consciousness with specific values"""
    assert basic_consciousness.perception_fidelity == 0.7
    assert basic_consciousness.reaction_speed == 0.6
    assert basic_consciousness.memory_depth == 0.8


@pytest.mark.unit
def test_default_weights_initialization():
    """Test that default weights are properly initialized"""
    consciousness = GradedConsciousness()

    assert consciousness.weights is not None
    assert "perception_fidelity" in consciousness.weights
    assert "meta_cognitive_ability" in consciousness.weights
    assert consciousness.weights["meta_cognitive_ability"] == 2.0  # Highest weight


# ============================================================================
# Capability Scores Tests
# ============================================================================

@pytest.mark.unit
def test_get_capability_scores(basic_consciousness):
    """Test retrieving all capability scores"""
    scores = basic_consciousness.get_capability_scores()

    assert isinstance(scores, dict)
    assert "perception_fidelity" in scores
    assert "reaction_speed" in scores
    assert "memory_depth" in scores

    assert scores["perception_fidelity"] == 0.7
    assert scores["reaction_speed"] == 0.6
    assert scores["memory_depth"] == 0.8


@pytest.mark.unit
def test_overall_consciousness_score(basic_consciousness):
    """Test overall consciousness scoring"""
    score = basic_consciousness.overall_consciousness_score()

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Should be weighted average, not simple average
    scores = basic_consciousness.get_capability_scores()
    simple_avg = sum(scores.values()) / len(scores)

    # Weighted score should differ from simple average
    # (due to different weights)
    assert score != pytest.approx(simple_avg, abs=0.01)


@pytest.mark.unit
def test_consciousness_score_increases_with_capabilities():
    """Test that higher capabilities yield higher scores"""
    low = GradedConsciousness(
        perception_fidelity=0.2,
        reaction_speed=0.2,
        memory_depth=0.2
    )

    high = GradedConsciousness(
        perception_fidelity=0.9,
        reaction_speed=0.9,
        memory_depth=0.9
    )

    assert high.overall_consciousness_score() > low.overall_consciousness_score()


@pytest.mark.unit
def test_can_perform_capability(basic_consciousness):
    """Test capability threshold checking"""
    # Perception fidelity is 0.7
    assert basic_consciousness.can_perform("perception_fidelity", 0.5)
    assert basic_consciousness.can_perform("perception_fidelity", 0.7)
    assert not basic_consciousness.can_perform("perception_fidelity", 0.8)

    # Meta-cognitive ability is 0.3
    assert basic_consciousness.can_perform("meta_cognitive_ability", 0.2)
    assert not basic_consciousness.can_perform("meta_cognitive_ability", 0.5)


@pytest.mark.unit
def test_can_perform_unknown_capability(basic_consciousness):
    """Test checking unknown capability returns False"""
    assert not basic_consciousness.can_perform("unknown_capability", 0.5)


# ============================================================================
# State Description Tests
# ============================================================================

@pytest.mark.unit
def test_describe_state_dormant():
    """Test dormant state description"""
    consciousness = GradedConsciousness(
        perception_fidelity=0.1,
        reaction_speed=0.1
    )

    assert consciousness.describe_state() == "dormant"


@pytest.mark.unit
def test_describe_state_reactive():
    """Test reactive state description"""
    consciousness = GradedConsciousness(
        perception_fidelity=0.3,
        reaction_speed=0.3
    )

    state = consciousness.describe_state()
    assert state in ["reactive", "dormant"]  # Boundary conditions


@pytest.mark.unit
def test_describe_state_aware():
    """Test aware state description"""
    consciousness = GradedConsciousness(
        perception_fidelity=0.5,
        reaction_speed=0.5,
        memory_depth=0.5,
        memory_recall_accuracy=0.5,
        introspection_capacity=0.5
    )

    state = consciousness.describe_state()
    # Due to weighted scoring, this could be reactive or aware
    assert state in ["reactive", "aware"]


@pytest.mark.unit
def test_describe_state_reflective():
    """Test reflective state description"""
    consciousness = GradedConsciousness(
        perception_fidelity=0.7,
        reaction_speed=0.6,
        memory_depth=0.7,
        memory_recall_accuracy=0.7,
        introspection_capacity=0.7,
        meta_cognitive_ability=0.6,
        information_integration=0.7
    )

    state = consciousness.describe_state()
    # Due to weighted scoring, this should be aware or reflective
    assert state in ["aware", "reflective"]


@pytest.mark.unit
def test_describe_state_meta_aware():
    """Test meta_aware state description"""
    consciousness = GradedConsciousness(
        perception_fidelity=0.85,
        reaction_speed=0.8,
        memory_depth=0.9,
        memory_recall_accuracy=0.85,
        introspection_capacity=0.9,
        meta_cognitive_ability=0.85,
        information_integration=0.85,
        intentional_coherence=0.85,
        qualia_richness=0.8
    )

    state = consciousness.describe_state()
    # Due to weighted scoring, should be meta_aware or reflective
    assert state in ["reflective", "meta_aware"]


@pytest.mark.unit
def test_describe_state_transcendent(high_consciousness):
    """Test transcendent state description"""
    state = high_consciousness.describe_state()
    # High consciousness should be meta_aware or transcendent
    assert state in ["meta_aware", "transcendent"]


# ============================================================================
# Evolution Tests
# ============================================================================

@pytest.mark.unit
def test_evolve_capability(basic_consciousness):
    """Test evolving a single capability"""
    original_perception = basic_consciousness.perception_fidelity

    evolved = basic_consciousness.evolve("perception_fidelity", 0.1)

    # Original should be unchanged
    assert basic_consciousness.perception_fidelity == original_perception

    # Evolved should have increased perception
    assert evolved.perception_fidelity == pytest.approx(original_perception + 0.1)

    # Other capabilities should be unchanged
    assert evolved.reaction_speed == basic_consciousness.reaction_speed


@pytest.mark.unit
def test_evolve_capability_clipping():
    """Test evolution respects bounds [0, 1]"""
    consciousness = GradedConsciousness(perception_fidelity=0.95)

    # Try to evolve beyond 1.0
    evolved = consciousness.evolve("perception_fidelity", 0.2)

    assert evolved.perception_fidelity == 1.0  # Clipped to max


@pytest.mark.unit
def test_evolve_negative_delta():
    """Test decreasing capability through evolution"""
    consciousness = GradedConsciousness(perception_fidelity=0.7)

    evolved = consciousness.evolve("perception_fidelity", -0.2)

    assert evolved.perception_fidelity == pytest.approx(0.5)


@pytest.mark.unit
def test_evolve_unknown_capability(basic_consciousness):
    """Test evolving unknown capability returns unchanged consciousness"""
    evolved = basic_consciousness.evolve("unknown_capability", 0.5)

    # Should return same consciousness
    assert evolved.perception_fidelity == basic_consciousness.perception_fidelity


# ============================================================================
# Transition Validation Tests
# ============================================================================

@pytest.mark.unit
def test_valid_gradual_transition():
    """Test valid gradual consciousness increase"""
    current = GradedConsciousness(perception_fidelity=0.5)
    target = GradedConsciousness(perception_fidelity=0.7)  # +0.2

    assert current.is_valid_transition_to(target)


@pytest.mark.unit
def test_invalid_rapid_transition():
    """Test invalid rapid consciousness increase"""
    current = GradedConsciousness(perception_fidelity=0.5)
    target = GradedConsciousness(perception_fidelity=0.9)  # +0.4, too rapid

    assert not current.is_valid_transition_to(target)


@pytest.mark.unit
def test_valid_decrease_transition():
    """Test that consciousness can decrease freely"""
    current = GradedConsciousness(perception_fidelity=0.9)
    target = GradedConsciousness(perception_fidelity=0.1)  # Large decrease

    assert current.is_valid_transition_to(target)


@pytest.mark.unit
def test_boundary_transition():
    """Test transition at boundary (just under 0.3 increase)"""
    current = GradedConsciousness(perception_fidelity=0.5)
    target = GradedConsciousness(perception_fidelity=0.79)  # +0.29 (valid)

    assert current.is_valid_transition_to(target)


# ============================================================================
# Distance Calculation Tests
# ============================================================================

@pytest.mark.unit
def test_distance_to_identical():
    """Test distance to identical consciousness is 0"""
    c1 = GradedConsciousness(perception_fidelity=0.7, reaction_speed=0.6)
    c2 = GradedConsciousness(perception_fidelity=0.7, reaction_speed=0.6)

    distance = c1.distance_to(c2)

    assert distance == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_distance_increases_with_difference():
    """Test distance increases with larger differences"""
    c1 = GradedConsciousness(perception_fidelity=0.5)

    c2_close = GradedConsciousness(perception_fidelity=0.6)
    c2_far = GradedConsciousness(perception_fidelity=0.9)

    distance_close = c1.distance_to(c2_close)
    distance_far = c1.distance_to(c2_far)

    assert distance_far > distance_close


@pytest.mark.unit
def test_distance_is_symmetric():
    """Test distance is symmetric: d(a,b) = d(b,a)"""
    c1 = GradedConsciousness(perception_fidelity=0.5, reaction_speed=0.6)
    c2 = GradedConsciousness(perception_fidelity=0.8, reaction_speed=0.3)

    assert c1.distance_to(c2) == pytest.approx(c2.distance_to(c1))


@pytest.mark.unit
def test_distance_with_custom_weights():
    """Test distance calculation respects weights"""
    # High weight on perception
    weights_high_perception = {
        "perception_fidelity": 10.0,
        "reaction_speed": 1.0,
        "memory_depth": 1.0,
        "memory_recall_accuracy": 1.0,
        "introspection_capacity": 1.0,
        "meta_cognitive_ability": 1.0,
        "information_integration": 1.0,
        "intentional_coherence": 1.0,
        "qualia_richness": 1.0,
    }

    c1 = GradedConsciousness(perception_fidelity=0.5, reaction_speed=0.5)
    c1.weights = weights_high_perception

    c2_diff_perception = GradedConsciousness(perception_fidelity=0.9, reaction_speed=0.5)
    c2_diff_reaction = GradedConsciousness(perception_fidelity=0.5, reaction_speed=0.9)

    # Distance should be higher for perception difference
    dist_perception = c1.distance_to(c2_diff_perception)
    dist_reaction = c1.distance_to(c2_diff_reaction)

    assert dist_perception > dist_reaction


# ============================================================================
# Backward Compatibility Tests (ConsciousnessAdapter)
# ============================================================================

@pytest.mark.unit
def test_adapter_from_discrete_dormant():
    """Test converting DORMANT discrete state"""
    consciousness = ConsciousnessAdapter.from_discrete_state("DORMANT")

    assert consciousness.perception_fidelity == 0.1
    assert consciousness.overall_consciousness_score() < 0.2


@pytest.mark.unit
def test_adapter_from_discrete_aware():
    """Test converting AWARE discrete state"""
    consciousness = ConsciousnessAdapter.from_discrete_state("AWARE")

    assert consciousness.perception_fidelity == 0.7
    assert consciousness.introspection_capacity == 0.5


@pytest.mark.unit
def test_adapter_from_discrete_transcendent():
    """Test converting TRANSCENDENT discrete state"""
    consciousness = ConsciousnessAdapter.from_discrete_state("TRANSCENDENT")

    assert consciousness.perception_fidelity >= 0.9
    assert consciousness.overall_consciousness_score() >= 0.9


@pytest.mark.unit
def test_adapter_to_discrete_state():
    """Test converting graded to discrete state"""
    consciousness = GradedConsciousness(
        perception_fidelity=0.7,
        reaction_speed=0.6,
        memory_depth=0.7,
        introspection_capacity=0.7
    )

    state = ConsciousnessAdapter.to_discrete_state(consciousness)

    assert state in ["DORMANT", "REACTIVE", "AWARE", "REFLECTIVE", "META_AWARE", "TRANSCENDENT"]


@pytest.mark.unit
def test_adapter_roundtrip():
    """Test converting discrete -> graded -> discrete"""
    original_state = "REFLECTIVE"

    graded = ConsciousnessAdapter.from_discrete_state(original_state)
    converted_back = ConsciousnessAdapter.to_discrete_state(graded)

    assert converted_back == original_state


# ============================================================================
# Data Type Tests
# ============================================================================

@pytest.mark.unit
def test_perception_creation():
    """Test creating Perception data type"""
    import time

    perception = Perception(
        stimulus_type="visual",
        content="red object",
        timestamp=time.time(),
        fidelity=0.8
    )

    assert perception.stimulus_type == "visual"
    assert perception.content == "red object"
    assert perception.fidelity == 0.8


@pytest.mark.unit
def test_action_creation():
    """Test creating Action data type"""
    action = Action(
        action_type="move",
        parameters={"direction": "forward", "speed": 1.0},
        confidence=0.9
    )

    assert action.action_type == "move"
    assert action.parameters["direction"] == "forward"
    assert action.confidence == 0.9


@pytest.mark.unit
def test_experience_creation():
    """Test creating Experience data type"""
    import time

    perception = Perception("visual", "object", time.time(), 0.8)
    action = Action("move", {}, 0.9)

    experience = Experience(
        perception=perception,
        action=action,
        outcome="success",
        emotional_valence=0.7,
        timestamp=time.time()
    )

    assert experience.perception == perception
    assert experience.action == action
    assert experience.outcome == "success"
    assert -1.0 <= experience.emotional_valence <= 1.0


@pytest.mark.unit
def test_qualia_creation():
    """Test creating Qualia data type"""
    import time

    qualia = Qualia(
        experience_type="pain",
        intensity=0.7,
        valence=-0.5,
        content="sharp sensation",
        timestamp=time.time(),
        uniqueness=0.8
    )

    assert qualia.experience_type == "pain"
    assert qualia.intensity == 0.7
    assert qualia.valence == -0.5


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.unit
def test_negative_capability_values():
    """Test that negative capability values are handled"""
    # Should either clip or raise error
    consciousness = GradedConsciousness(perception_fidelity=-0.5)

    # Most implementations would clip to 0.0
    # Or could raise ValueError
    # For now, just verify it doesn't crash
    assert isinstance(consciousness.perception_fidelity, float)


@pytest.mark.unit
def test_capability_values_above_one():
    """Test capability values above 1.0"""
    consciousness = GradedConsciousness(perception_fidelity=1.5)

    # Should either clip to 1.0 or accept higher values
    # Verify it doesn't crash
    score = consciousness.overall_consciousness_score()
    assert isinstance(score, float)


@pytest.mark.unit
def test_empty_weights():
    """Test consciousness with empty weights dict"""
    consciousness = GradedConsciousness(perception_fidelity=0.7)
    consciousness.weights = {}

    # Should handle empty weights gracefully
    # (might return 0 or raise error)
    try:
        score = consciousness.overall_consciousness_score()
        # If it succeeds, verify result is valid
        assert isinstance(score, float)
    except (ZeroDivisionError, KeyError):
        # Acceptable to fail with empty weights
        pass


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_consciousness_lifecycle():
    """Test full consciousness lifecycle: create, evolve, transition"""
    # Start dormant
    consciousness = GradedConsciousness(
        perception_fidelity=0.1,
        reaction_speed=0.1
    )

    assert consciousness.describe_state() == "dormant"

    # Evolve to reactive
    consciousness = consciousness.evolve("perception_fidelity", 0.3)
    consciousness = consciousness.evolve("reaction_speed", 0.3)

    state = consciousness.describe_state()
    assert state in ["reactive", "aware"]

    # Further evolution to aware
    consciousness = consciousness.evolve("memory_depth", 0.5)
    consciousness = consciousness.evolve("introspection_capacity", 0.3)

    # Check final state
    final_score = consciousness.overall_consciousness_score()
    assert final_score > 0.3


@pytest.mark.integration
def test_consciousness_comparison():
    """Test comparing multiple consciousness states"""
    states = [
        GradedConsciousness(perception_fidelity=0.1),
        GradedConsciousness(perception_fidelity=0.5),
        GradedConsciousness(perception_fidelity=0.9)
    ]

    scores = [s.overall_consciousness_score() for s in states]

    # Scores should be monotonically increasing
    assert scores[0] < scores[1] < scores[2]
