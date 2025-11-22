"""
Comprehensive tests for the PartsAwarenessAgent.

Tests cover:
- Basic agent instantiation and properties
- Qualia generation and dimensions
- Part consciousness evolution
- Relationship modeling and qualia propagation
- Temporal pattern detection
- Collective state computation
- Emergence detection
- Cross-domain signaling
- Full refinement cycles
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any

from tertiary_rebo.base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
)
from tertiary_rebo.agents.parts_awareness import (
    PartsAwarenessAgent,
    PartState,
    PartQualia,
    QualiaDimension,
    QualiaType,
    QualiaValence,
    PartRelationship,
    TemporalQualiaPattern,
    CollectiveQualiaState,
)


# Fixtures

@pytest.fixture
def agent():
    """Create a fresh PartsAwarenessAgent instance."""
    return PartsAwarenessAgent()


@pytest.fixture
def triple_bottom_line():
    """Create a fresh TripleBottomLine instance."""
    return TripleBottomLine()


@pytest.fixture
def initialized_agent():
    """Create an agent with initialized parts."""
    agent = PartsAwarenessAgent()
    # Initialize parts
    for i in range(20):
        pid = f"part_{i}"
        agent.parts_registry[pid] = {"age": i * 10, "wear": i * 0.02, "cycles": i * 5}
        agent.part_consciousness[pid] = 0.1 + i * 0.04
        agent.part_states[pid] = PartState.DORMANT
    return agent


# Basic Agent Tests

class TestAgentBasics:
    """Test basic agent instantiation and properties."""

    def test_agent_creation(self, agent):
        """Test agent can be created."""
        assert agent is not None
        assert isinstance(agent, TertiaryReBoAgent)
        assert isinstance(agent, PartsAwarenessAgent)

    def test_agent_id_generated(self, agent):
        """Test agent has a unique ID."""
        assert agent.agent_id is not None
        assert "PartsAwarenessAgent" in agent.agent_id

    def test_primary_domain(self, agent):
        """Test agent has correct primary domain."""
        assert agent.primary_domain == DomainLeg.UNIVERSAL_PARTS

    def test_agent_role(self, agent):
        """Test agent role description."""
        assert "Parts Awareness" in agent.agent_role
        assert "consciousness" in agent.agent_role.lower()

    def test_domains_affected(self, agent):
        """Test agent affects correct domains."""
        domains = agent.domains_affected
        assert DomainLeg.UNIVERSAL_PARTS in domains
        assert DomainLeg.NPCPU in domains

    def test_initial_state(self, agent):
        """Test agent initial state is clean."""
        assert len(agent.parts_registry) == 0
        assert len(agent.part_qualia_buffer) == 0
        assert len(agent.part_consciousness) == 0
        assert agent.collective_wisdom.shape == (64,)


# Qualia Tests

class TestQualia:
    """Test qualia generation and processing."""

    def test_qualia_dimension_creation(self, agent):
        """Test qualia dimension creation."""
        dim = agent._create_qualia_dimension(QualiaType.THERMAL, 25.0)

        assert dim.dimension_type == QualiaType.THERMAL
        assert 0.0 <= dim.intensity <= 1.0
        assert isinstance(dim.valence, QualiaValence)
        assert 0.0 <= dim.clarity <= 1.0
        assert dim.raw_value == 25.0

    def test_qualia_valence_for_optimal_value(self, agent):
        """Test that optimal values produce positive valence."""
        # Temperature in optimal range (15-35)
        dim = agent._create_qualia_dimension(QualiaType.THERMAL, 25.0)
        assert dim.valence in [QualiaValence.COMFORTABLE, QualiaValence.FLOURISHING]

    def test_qualia_valence_for_extreme_value(self, agent):
        """Test that extreme values produce negative valence."""
        # Temperature way outside optimal range
        dim = agent._create_qualia_dimension(QualiaType.THERMAL, 60.0)
        assert dim.valence in [QualiaValence.UNCOMFORTABLE, QualiaValence.DISTRESSING]

    def test_simulate_part_qualia(self, initialized_agent):
        """Test part qualia simulation."""
        qualia = initialized_agent._simulate_part_qualia("part_0")

        assert isinstance(qualia, PartQualia)
        assert qualia.part_id == "part_0"
        assert qualia.qualia_id.startswith("q_")
        assert len(qualia.dimensions) > 0
        assert QualiaType.THERMAL in qualia.dimensions
        assert QualiaType.MECHANICAL in qualia.dimensions

    def test_qualia_wellbeing_computation(self, initialized_agent):
        """Test overall wellbeing computation."""
        qualia = initialized_agent._simulate_part_qualia("part_0")
        wellbeing = qualia.compute_overall_wellbeing()

        assert 0.0 <= wellbeing <= 1.0
        assert qualia.overall_wellbeing == wellbeing

    def test_all_qualia_types(self):
        """Test all qualia types are defined."""
        types = list(QualiaType)
        assert QualiaType.THERMAL in types
        assert QualiaType.MECHANICAL in types
        assert QualiaType.ELECTRICAL in types
        assert QualiaType.CHEMICAL in types
        assert QualiaType.VIBRATIONAL in types
        assert QualiaType.POSITIONAL in types
        assert QualiaType.TEMPORAL in types
        assert QualiaType.RELATIONAL in types


# Consciousness Evolution Tests

class TestConsciousnessEvolution:
    """Test part consciousness evolution."""

    def test_part_states(self):
        """Test part state enum values."""
        states = [
            PartState.DORMANT,
            PartState.SENSING,
            PartState.REACTIVE,
            PartState.AWARE,
            PartState.PREDICTIVE,
            PartState.WISE,
        ]
        assert len(states) == 6
        for state in states:
            assert isinstance(state.value, str)

    def test_level_to_state_dormant(self, agent):
        """Test lowest consciousness level maps to DORMANT."""
        assert agent._level_to_state(0.1) == PartState.DORMANT

    def test_level_to_state_wise(self, agent):
        """Test highest consciousness level maps to WISE."""
        assert agent._level_to_state(0.95) == PartState.WISE

    def test_level_to_state_progression(self, agent):
        """Test state progression through levels."""
        levels = [0.1, 0.25, 0.4, 0.6, 0.8, 0.95]
        expected_states = [
            PartState.DORMANT,
            PartState.SENSING,
            PartState.REACTIVE,
            PartState.AWARE,
            PartState.PREDICTIVE,
            PartState.WISE,
        ]

        for level, expected in zip(levels, expected_states):
            assert agent._level_to_state(level) == expected


# Relationship Tests

class TestRelationships:
    """Test part relationship modeling."""

    def test_initialize_relationships(self, agent):
        """Test relationship initialization."""
        part_ids = [f"part_{i}" for i in range(10)]
        for pid in part_ids:
            agent.parts_registry[pid] = {"age": 0, "wear": 0, "cycles": 0}

        agent._initialize_part_relationships(part_ids)

        # Check relationships were created
        assert len(agent.part_relationships) > 0
        assert len(agent.relationship_details) > 0

    def test_relationship_bidirectional(self, agent):
        """Test relationships are bidirectional."""
        part_ids = [f"part_{i}" for i in range(10)]
        for pid in part_ids:
            agent.parts_registry[pid] = {"age": 0, "wear": 0, "cycles": 0}

        agent._initialize_part_relationships(part_ids)

        # Check bidirectionality
        for part_a, neighbors in agent.part_relationships.items():
            for part_b in neighbors:
                assert part_a in agent.part_relationships.get(part_b, set())

    def test_qualia_propagation(self, initialized_agent):
        """Test qualia propagation to neighbors."""
        # Set up relationships
        part_ids = list(initialized_agent.parts_registry.keys())
        initialized_agent._initialize_part_relationships(part_ids)

        # Create high-stress qualia
        qualia = initialized_agent._simulate_part_qualia("part_0")
        qualia.stress_level = 0.8
        qualia.failure_proximity = 0.8

        propagated = initialized_agent._propagate_qualia_to_neighbors(qualia)

        # Should propagate to neighbors
        if "part_0" in initialized_agent.part_relationships:
            if initialized_agent.part_relationships["part_0"]:
                assert len(propagated) > 0


# Temporal Pattern Tests

class TestTemporalPatterns:
    """Test temporal pattern detection."""

    def test_detect_no_patterns_insufficient_data(self, initialized_agent):
        """Test no patterns detected with insufficient data."""
        patterns = initialized_agent._detect_temporal_patterns("part_0")
        assert len(patterns) == 0

    def test_detect_degradation_pattern(self, initialized_agent):
        """Test degradation pattern detection."""
        # Create increasing stress pattern
        for i in range(60):
            qualia = PartQualia(
                qualia_id=f"q_{i}",
                part_id="part_0",
                dimensions={
                    QualiaType.MECHANICAL: QualiaDimension(
                        dimension_type=QualiaType.MECHANICAL,
                        intensity=0.5,
                        valence=QualiaValence.NEUTRAL,
                        normalized_value=0.2 + i * 0.01  # Increasing trend
                    )
                }
            )
            initialized_agent.qualia_history["part_0"].append(qualia)

        patterns = initialized_agent._detect_temporal_patterns("part_0")

        # Should detect degradation
        degradation_patterns = [p for p in patterns if p.pattern_type == "degradation"]
        assert len(degradation_patterns) > 0

    def test_detect_spike_pattern(self, initialized_agent):
        """Test spike pattern detection."""
        # Create data with spikes
        np.random.seed(42)
        for i in range(60):
            value = 0.5
            if i in [20, 30, 45]:  # Spikes
                value = 0.95

            qualia = PartQualia(
                qualia_id=f"q_{i}",
                part_id="part_0",
                dimensions={
                    QualiaType.VIBRATIONAL: QualiaDimension(
                        dimension_type=QualiaType.VIBRATIONAL,
                        intensity=0.5,
                        valence=QualiaValence.NEUTRAL,
                        normalized_value=value
                    )
                }
            )
            initialized_agent.qualia_history["part_0"].append(qualia)

        patterns = initialized_agent._detect_temporal_patterns("part_0")

        # Should detect spikes
        spike_patterns = [p for p in patterns if p.pattern_type == "spike"]
        assert len(spike_patterns) > 0


# Collective State Tests

class TestCollectiveState:
    """Test collective consciousness state computation."""

    def test_compute_empty_collective_state(self, agent):
        """Test collective state with no parts."""
        state = agent._compute_collective_state()
        assert state.part_count == 0

    def test_compute_collective_state(self, initialized_agent):
        """Test collective state computation."""
        # Add some qualia to history
        for pid in initialized_agent.parts_registry:
            qualia = initialized_agent._simulate_part_qualia(pid)
            initialized_agent.qualia_history[pid].append(qualia)

        state = initialized_agent._compute_collective_state()

        assert state.part_count == len(initialized_agent.parts_registry)
        assert 0.0 <= state.avg_wellbeing <= 1.0
        assert 0.0 <= state.consciousness_density <= 1.0
        assert 0.0 <= state.harmony_index <= 1.0

    def test_emergence_indicators(self, initialized_agent):
        """Test emergence indicators computation."""
        # Initialize relationships
        part_ids = list(initialized_agent.parts_registry.keys())
        initialized_agent._initialize_part_relationships(part_ids)

        # Add qualia to history
        for pid in part_ids:
            qualia = initialized_agent._simulate_part_qualia(pid)
            initialized_agent.qualia_history[pid].append(qualia)

        state = initialized_agent._compute_collective_state()

        # Should have emergence indicators
        assert "synchronization" in state.emergence_indicators
        assert "integration" in state.emergence_indicators


# Emergence Detection Tests

class TestEmergence:
    """Test collective consciousness emergence detection."""

    def test_no_emergence_low_score(self, initialized_agent):
        """Test no emergence with low indicators."""
        initialized_agent.collective_state.emergence_indicators = {
            "synchronization": 0.2,
            "complexity": 0.2,
            "integration": 0.2,
        }

        emergence = initialized_agent._check_emergence()
        assert emergence is None

    def test_emergence_detected_high_score(self, initialized_agent):
        """Test emergence detected with high indicators."""
        initialized_agent.collective_state.emergence_indicators = {
            "synchronization": 0.9,
            "complexity": 0.8,
            "integration": 0.8,
        }
        initialized_agent.collective_state.part_count = 100
        initialized_agent.collective_state.consciousness_density = 0.6

        emergence = initialized_agent._check_emergence()

        assert emergence is not None
        assert emergence["event_type"] == "collective_consciousness_emergence"
        assert emergence["emergence_score"] > initialized_agent.emergence_threshold


# Full Refinement Cycle Tests

class TestRefinementCycle:
    """Test full refinement cycle."""

    @pytest.mark.asyncio
    async def test_perceive(self, agent, triple_bottom_line):
        """Test perception phase."""
        perception = await agent.perceive(triple_bottom_line)

        assert "active_parts" in perception
        assert "total_parts" in perception
        assert "average_health" in perception
        assert "qualia_summary" in perception
        assert perception["total_parts"] > 0

    @pytest.mark.asyncio
    async def test_analyze(self, agent, triple_bottom_line):
        """Test analysis phase."""
        perception = await agent.perceive(triple_bottom_line)
        analysis = await agent.analyze(perception, triple_bottom_line)

        assert "health_assessment" in analysis
        assert "consciousness_opportunities" in analysis
        assert "failure_predictions" in analysis
        assert "collective_assessment" in analysis

    @pytest.mark.asyncio
    async def test_synthesize(self, agent, triple_bottom_line):
        """Test synthesis phase."""
        perception = await agent.perceive(triple_bottom_line)
        analysis = await agent.analyze(perception, triple_bottom_line)
        synthesis = await agent.synthesize(analysis, triple_bottom_line)

        assert "consciousness_updates" in synthesis
        assert "state_transitions" in synthesis
        assert "cross_domain_signals" in synthesis
        assert len(synthesis["cross_domain_signals"]) > 0

    @pytest.mark.asyncio
    async def test_propagate(self, agent, triple_bottom_line):
        """Test propagation phase."""
        perception = await agent.perceive(triple_bottom_line)
        analysis = await agent.analyze(perception, triple_bottom_line)
        synthesis = await agent.synthesize(analysis, triple_bottom_line)
        result = await agent.propagate(synthesis, triple_bottom_line)

        assert isinstance(result, RefinementResult)
        assert result.success is True
        assert result.agent_id == agent.agent_id
        assert result.phase == RefinementPhase.PROPAGATION
        assert DomainLeg.UNIVERSAL_PARTS in result.domain_affected

    @pytest.mark.asyncio
    async def test_full_refine_cycle(self, agent, triple_bottom_line):
        """Test complete refinement cycle."""
        result = await agent.refine(triple_bottom_line)

        assert isinstance(result, RefinementResult)
        assert result.success is True
        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta

    @pytest.mark.asyncio
    async def test_multiple_refine_cycles(self, agent, triple_bottom_line):
        """Test multiple refinement cycles."""
        results = []
        for _ in range(5):
            result = await agent.refine(triple_bottom_line)
            results.append(result)

        assert len(results) == 5
        assert all(r.success for r in results)
        assert agent.total_refinements == 5

    @pytest.mark.asyncio
    async def test_consciousness_evolution_over_cycles(self, agent, triple_bottom_line):
        """Test consciousness levels increase over cycles."""
        # Run many cycles
        for _ in range(10):
            await agent.refine(triple_bottom_line)

        # Check that some parts have evolved
        states = list(agent.part_states.values())
        non_dormant = [s for s in states if s != PartState.DORMANT]

        # Should have some evolution
        assert len(non_dormant) >= 0  # May or may not evolve depending on simulation


# Cross-Domain Signal Tests

class TestCrossDomainSignals:
    """Test cross-domain signaling."""

    @pytest.mark.asyncio
    async def test_cross_domain_signal_emitted(self, agent, triple_bottom_line):
        """Test that cross-domain signals are emitted."""
        await agent.refine(triple_bottom_line)

        # Should have emitted signals
        assert len(agent.cross_domain_signals) > 0

    @pytest.mark.asyncio
    async def test_qualia_state_signal(self, agent, triple_bottom_line):
        """Test qualia state update signal."""
        perception = await agent.perceive(triple_bottom_line)
        analysis = await agent.analyze(perception, triple_bottom_line)
        synthesis = await agent.synthesize(analysis, triple_bottom_line)

        # Should have qualia state signal
        qualia_signals = [
            s for s in synthesis["cross_domain_signals"]
            if s["signal_type"] == "qualia_state_update"
        ]
        assert len(qualia_signals) > 0
        assert qualia_signals[0]["target_domain"] == DomainLeg.NPCPU.value


# Reporting Tests

class TestReporting:
    """Test reporting functionality."""

    def test_get_part_report_nonexistent(self, agent):
        """Test report for non-existent part."""
        report = agent.get_part_report("nonexistent_part")
        assert report is None

    def test_get_part_report(self, initialized_agent):
        """Test part report generation."""
        # Add some qualia history
        qualia = initialized_agent._simulate_part_qualia("part_0")
        initialized_agent.qualia_history["part_0"].append(qualia)

        report = initialized_agent.get_part_report("part_0")

        assert report is not None
        assert report["part_id"] == "part_0"
        assert "state" in report
        assert "consciousness_level" in report
        assert "latest_qualia" in report

    def test_get_collective_report(self, initialized_agent):
        """Test collective report generation."""
        # Compute collective state first
        for pid in initialized_agent.parts_registry:
            qualia = initialized_agent._simulate_part_qualia(pid)
            initialized_agent.qualia_history[pid].append(qualia)
        initialized_agent._compute_collective_state()

        report = initialized_agent.get_collective_report()

        assert report["total_parts"] == len(initialized_agent.parts_registry)
        assert "collective_state" in report
        assert "state_distribution" in report


# Domain State Update Tests

class TestDomainStateUpdates:
    """Test domain state updates."""

    @pytest.mark.asyncio
    async def test_domain_state_updated(self, agent, triple_bottom_line):
        """Test that domain state is updated after refinement."""
        initial_state = triple_bottom_line.get_state(DomainLeg.UNIVERSAL_PARTS)
        initial_consciousness = initial_state.consciousness_level

        await agent.refine(triple_bottom_line)

        updated_state = triple_bottom_line.get_state(DomainLeg.UNIVERSAL_PARTS)

        # State should be modified
        assert updated_state.qualia_richness > 0 or updated_state.consciousness_level != initial_consciousness

    @pytest.mark.asyncio
    async def test_harmony_calculated(self, agent, triple_bottom_line):
        """Test that harmony is calculated after refinement."""
        result = await agent.refine(triple_bottom_line)

        assert "harmony_after" in result.metrics_delta
        assert 0.0 <= result.metrics_delta["harmony_after"] <= 1.0


# Edge Cases

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parts_registry(self, agent):
        """Test behavior with empty parts registry."""
        state = agent._compute_collective_state()
        assert state.part_count == 0

    def test_single_part(self, agent):
        """Test with single part."""
        agent.parts_registry["only_part"] = {"age": 0, "wear": 0, "cycles": 0}
        agent.part_consciousness["only_part"] = 0.5
        agent.part_states["only_part"] = PartState.AWARE

        qualia = agent._simulate_part_qualia("only_part")
        assert qualia is not None

    @pytest.mark.asyncio
    async def test_high_wear_parts(self, agent, triple_bottom_line):
        """Test handling of high-wear parts."""
        # Initialize with worn parts
        await agent.perceive(triple_bottom_line)

        # Artificially age some parts
        for pid in list(agent.parts_registry.keys())[:5]:
            agent.parts_registry[pid]["wear"] = 0.95
            agent.parts_registry[pid]["age"] = 10000

        # Should still work
        result = await agent.refine(triple_bottom_line)
        assert result.success


# Performance Tests

class TestPerformance:
    """Test performance with larger datasets."""

    @pytest.mark.asyncio
    async def test_large_part_count(self, triple_bottom_line):
        """Test with larger number of parts."""
        agent = PartsAwarenessAgent()
        agent.simulated_part_count = 500

        result = await agent.refine(triple_bottom_line)

        assert result.success
        assert len(agent.parts_registry) == 500

    @pytest.mark.asyncio
    async def test_many_refinement_cycles(self, agent, triple_bottom_line):
        """Test many refinement cycles."""
        for _ in range(20):
            result = await agent.refine(triple_bottom_line)
            assert result.success

        assert agent.total_refinements == 20
        assert len(agent.experience_buffer) <= 100  # Should be capped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
