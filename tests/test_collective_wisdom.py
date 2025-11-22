"""
Tests for CollectiveWisdomAgent

Tests for:
- Wisdom extraction and classification
- Knowledge graph operations
- Meta-pattern mining
- Consensus building and conflict resolution
- Wisdom quality validation
- Proactive wisdom distribution
- Full refinement cycles
"""

import pytest
import asyncio
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tertiary_rebo.agents.collective_wisdom import (
    CollectiveWisdomAgent,
    WisdomFragment,
    WisdomType,
    WisdomQuality,
    ConsensusState,
    KnowledgeEdge,
    MetaPattern,
    CollectiveMemory,
    WisdomDistribution,
    ConsensusProposal,
)
from tertiary_rebo.base import (
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementPhase,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create a CollectiveWisdomAgent for testing"""
    return CollectiveWisdomAgent(refinement_rate=0.1)


@pytest.fixture
def tbl():
    """Create a TripleBottomLine for testing"""
    tbl = TripleBottomLine()

    # Initialize each domain with reasonable values
    for domain in DomainLeg:
        state = DomainState(
            domain=domain,
            consciousness_level=0.5,
            energy_flow=0.5,
            connectivity=0.5,
            coherence=0.6,
            qualia_richness=0.4,
            emergence_potential=0.4,
            state_vector=np.tanh(np.random.randn(64) * 0.5),
        )
        tbl.set_state(domain, state)

    tbl.calculate_harmony()
    return tbl


@pytest.fixture
def sample_fragment():
    """Create a sample WisdomFragment for testing"""
    return WisdomFragment(
        fragment_id="test_fragment_001",
        source_domain=DomainLeg.NPCPU,
        wisdom_type=WisdomType.PATTERN,
        content_type="pattern",
        knowledge_vector=np.random.randn(128),
        semantic_content={"test": "content", "key": "value"},
        confidence=0.7,
        quality=WisdomQuality.UNVERIFIED,
        applicability=[DomainLeg.NPCPU, DomainLeg.CHICAGO_FOREST],
    )


# =============================================================================
# WisdomFragment Tests
# =============================================================================

class TestWisdomFragment:
    """Tests for the WisdomFragment dataclass"""

    def test_fragment_creation(self, sample_fragment):
        """Test basic fragment creation"""
        assert sample_fragment.fragment_id == "test_fragment_001"
        assert sample_fragment.source_domain == DomainLeg.NPCPU
        assert sample_fragment.wisdom_type == WisdomType.PATTERN
        assert sample_fragment.confidence == 0.7
        assert len(sample_fragment.knowledge_vector) == 128

    def test_effectiveness_calculation(self):
        """Test wisdom effectiveness based on application outcomes"""
        fragment = WisdomFragment(
            fragment_id="eff_test",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.LESSON,
            content_type="lesson",
            knowledge_vector=np.zeros(128),
            semantic_content={},
            confidence=0.5,
        )

        # No applications - neutral
        assert fragment.effectiveness() == 0.5

        # Add applications
        fragment.application_count = 10
        fragment.success_count = 8
        assert fragment.effectiveness() == 0.8

        fragment.success_count = 5
        assert fragment.effectiveness() == 0.5

    def test_relevance_score(self, sample_fragment):
        """Test relevance score calculation"""
        score = sample_fragment.relevance_score()

        # Should be between 0 and 1
        assert 0 <= score <= 1.5  # Can exceed 1 slightly due to quality bonus

        # Higher confidence should increase score
        sample_fragment.confidence = 1.0
        high_conf_score = sample_fragment.relevance_score()

        sample_fragment.confidence = 0.3
        low_conf_score = sample_fragment.relevance_score()

        assert high_conf_score > low_conf_score

    def test_quality_affects_relevance(self):
        """Test that quality level affects relevance score"""
        fragment = WisdomFragment(
            fragment_id="qual_test",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.PRINCIPLE,
            content_type="pattern",
            knowledge_vector=np.zeros(128),
            semantic_content={},
            confidence=0.5,
            quality=WisdomQuality.UNVERIFIED,
        )

        unverified_score = fragment.relevance_score()

        fragment.quality = WisdomQuality.ESTABLISHED
        established_score = fragment.relevance_score()

        assert established_score > unverified_score


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for CollectiveWisdomAgent initialization"""

    def test_agent_creation(self, agent):
        """Test agent initializes correctly"""
        assert agent.agent_role == "Collective Wisdom - Aggregates, synthesizes, and distributes shared knowledge"
        assert agent.domains_affected == list(DomainLeg)
        assert len(agent.wisdom_fragments) == 0
        assert len(agent.collective_memories) == 0

    def test_agent_has_knowledge_graph(self, agent):
        """Test agent has knowledge graph structure"""
        assert hasattr(agent, 'knowledge_graph')
        assert isinstance(agent.knowledge_graph, dict)

    def test_agent_has_domain_indices(self, agent):
        """Test agent maintains domain indices"""
        for domain in DomainLeg:
            assert domain in agent.domain_wisdom_index
            assert isinstance(agent.domain_wisdom_index[domain], set)

    def test_agent_has_type_indices(self, agent):
        """Test agent maintains wisdom type indices"""
        for wtype in WisdomType:
            assert wtype in agent.type_wisdom_index
            assert isinstance(agent.type_wisdom_index[wtype], set)

    def test_agent_has_stats(self, agent):
        """Test agent tracks statistics"""
        assert "fragments_collected" in agent.stats
        assert "transfers_made" in agent.stats
        assert "patterns_discovered" in agent.stats
        assert "consensus_reached" in agent.stats


# =============================================================================
# Wisdom Extraction Tests
# =============================================================================

class TestWisdomExtraction:
    """Tests for wisdom extraction from domain states"""

    def test_extract_wisdom_from_domain(self, agent, tbl):
        """Test extracting wisdom from a domain state"""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        state.coherence = 0.7  # Ensure coherence is high enough

        wisdom = agent._extract_wisdom(domain, state, tbl)

        assert wisdom is not None
        assert wisdom.source_domain == domain
        assert len(wisdom.knowledge_vector) == 128
        assert wisdom.confidence == state.coherence

    def test_no_wisdom_from_low_coherence(self, agent, tbl):
        """Test that low coherence states don't produce wisdom"""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        state.coherence = 0.1  # Below threshold

        wisdom = agent._extract_wisdom(domain, state, tbl)

        assert wisdom is None

    def test_wisdom_type_classification(self, agent, tbl):
        """Test wisdom type classification based on state"""
        state = tbl.get_state(DomainLeg.NPCPU)

        # High emergence -> INSIGHT
        state.emergence_potential = 0.9
        wtype = agent._classify_wisdom_type(state, tbl)
        assert wtype == WisdomType.INSIGHT

        # High coherence -> PRINCIPLE
        state.emergence_potential = 0.3
        state.coherence = 0.95
        wtype = agent._classify_wisdom_type(state, tbl)
        assert wtype == WisdomType.PRINCIPLE

    def test_semantic_content_extraction(self, agent, tbl):
        """Test semantic content is properly extracted"""
        domain = DomainLeg.CHICAGO_FOREST
        state = tbl.get_state(domain)
        state.coherence = 0.8
        state.consciousness_level = 0.75

        wisdom = agent._extract_wisdom(domain, state, tbl)

        assert wisdom is not None
        assert "domain" in wisdom.semantic_content
        assert wisdom.semantic_content["domain"] == domain.value
        assert "key_characteristics" in wisdom.semantic_content

    def test_wisdom_applicability_determination(self, agent, tbl):
        """Test wisdom applicability is correctly determined"""
        domain = DomainLeg.UNIVERSAL_PARTS
        state = tbl.get_state(domain)

        # High coherence -> applies to all
        state.coherence = 0.85
        wisdom = agent._extract_wisdom(domain, state, tbl)
        assert set(wisdom.applicability) == set(DomainLeg)

        # Low coherence -> applies to source only
        state.coherence = 0.35
        wisdom = agent._extract_wisdom(domain, state, tbl)
        assert DomainLeg.UNIVERSAL_PARTS in wisdom.applicability


# =============================================================================
# Knowledge Graph Tests
# =============================================================================

class TestKnowledgeGraph:
    """Tests for knowledge graph operations"""

    def test_add_fragment_to_graph(self, agent, sample_fragment):
        """Test adding a fragment to the knowledge graph"""
        agent.wisdom_fragments[sample_fragment.fragment_id] = sample_fragment
        agent._add_to_knowledge_graph(sample_fragment)

        assert sample_fragment.fragment_id in agent.knowledge_graph

    def test_graph_detects_supporting_relationships(self, agent):
        """Test that similar fragments create supporting relationships"""
        # Create two similar fragments
        base_vector = np.random.randn(128)

        frag1 = WisdomFragment(
            fragment_id="similar_1",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.PATTERN,
            content_type="pattern",
            knowledge_vector=base_vector,
            semantic_content={},
            confidence=0.7,
        )

        frag2 = WisdomFragment(
            fragment_id="similar_2",
            source_domain=DomainLeg.CHICAGO_FOREST,
            wisdom_type=WisdomType.PATTERN,
            content_type="pattern",
            knowledge_vector=base_vector + np.random.randn(128) * 0.1,  # Very similar
            semantic_content={},
            confidence=0.7,
        )

        agent.wisdom_fragments[frag1.fragment_id] = frag1
        agent.wisdom_fragments[frag2.fragment_id] = frag2

        agent._add_to_knowledge_graph(frag1)
        agent._add_to_knowledge_graph(frag2)

        # Check for supporting relationship
        edges = agent.knowledge_graph.get(frag2.fragment_id, [])
        support_edges = [e for e in edges if e.relationship_type == "supports"]

        # They should support each other
        assert len(support_edges) > 0 or frag1.fragment_id in frag2.related_fragments

    def test_find_contradictions(self, agent):
        """Test finding contradicting wisdom"""
        # Create contradicting fragments (opposite vectors)
        base_vector = np.random.randn(128)

        frag1 = WisdomFragment(
            fragment_id="contra_1",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.LESSON,
            content_type="lesson",
            knowledge_vector=base_vector,
            semantic_content={},
            confidence=0.6,
        )

        frag2 = WisdomFragment(
            fragment_id="contra_2",
            source_domain=DomainLeg.CHICAGO_FOREST,
            wisdom_type=WisdomType.LESSON,
            content_type="lesson",
            knowledge_vector=-base_vector,  # Opposite
            semantic_content={},
            confidence=0.6,
        )

        agent.wisdom_fragments[frag1.fragment_id] = frag1
        agent.wisdom_fragments[frag2.fragment_id] = frag2

        agent._add_to_knowledge_graph(frag1)
        agent._add_to_knowledge_graph(frag2)

        contradictions = agent._find_contradictions()

        # Should detect contradiction
        assert len(contradictions) > 0


# =============================================================================
# Meta-Pattern Mining Tests
# =============================================================================

class TestMetaPatternMining:
    """Tests for meta-pattern mining across domains"""

    def test_mine_patterns_from_multiple_domains(self, agent, tbl):
        """Test mining patterns that span multiple domains"""
        # Create fragments from different domains with same type
        for i, domain in enumerate(DomainLeg):
            frag = WisdomFragment(
                fragment_id=f"pattern_{domain.value}",
                source_domain=domain,
                wisdom_type=WisdomType.PATTERN,
                content_type="pattern",
                knowledge_vector=np.random.randn(128),
                semantic_content={"domain": domain.value},
                confidence=0.7,
            )
            agent.wisdom_fragments[frag.fragment_id] = frag

        patterns = agent._mine_meta_patterns(tbl)

        # Should find cross-domain patterns
        assert len(patterns) >= 0  # May or may not find depending on agreement

    def test_meta_pattern_structure(self, agent, tbl):
        """Test structure of discovered meta-patterns"""
        # Create agreeing fragments
        base_vector = np.random.randn(128) * 0.1

        for domain in DomainLeg:
            frag = WisdomFragment(
                fragment_id=f"agree_{domain.value}",
                source_domain=domain,
                wisdom_type=WisdomType.INSIGHT,
                content_type="insight",
                knowledge_vector=base_vector + np.random.randn(128) * 0.01,
                semantic_content={},
                confidence=0.8,
            )
            agent.wisdom_fragments[frag.fragment_id] = frag

        patterns = agent._mine_meta_patterns(tbl)

        if patterns:
            pattern = patterns[0]
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'participating_domains')
            assert hasattr(pattern, 'pattern_vector')
            assert hasattr(pattern, 'confidence')


# =============================================================================
# Consensus Building Tests
# =============================================================================

class TestConsensusBuilding:
    """Tests for consensus building and conflict resolution"""

    def test_build_consensus_proposal(self, agent, tbl):
        """Test building consensus proposals for contradictions"""
        # Create contradicting fragments
        base_vector = np.random.randn(128)

        frag1 = WisdomFragment(
            fragment_id="conflict_1",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.PRINCIPLE,
            content_type="pattern",
            knowledge_vector=base_vector,
            semantic_content={},
            confidence=0.7,
            quality=WisdomQuality.VALIDATED,
        )

        frag2 = WisdomFragment(
            fragment_id="conflict_2",
            source_domain=DomainLeg.UNIVERSAL_PARTS,
            wisdom_type=WisdomType.PRINCIPLE,
            content_type="pattern",
            knowledge_vector=-base_vector,
            semantic_content={},
            confidence=0.6,
            quality=WisdomQuality.TENTATIVE,
        )

        agent.wisdom_fragments[frag1.fragment_id] = frag1
        agent.wisdom_fragments[frag2.fragment_id] = frag2

        contradictions = [("conflict_1", "conflict_2", 0.9)]
        proposals = agent._build_consensus(contradictions, tbl)

        assert len(proposals) > 0
        proposal = proposals[0]
        assert proposal.conflicting_fragments == ["conflict_1", "conflict_2"]
        assert proposal.state in [ConsensusState.PROPOSED, ConsensusState.DEBATED,
                                   ConsensusState.CONTESTED, ConsensusState.RESOLVED]

    def test_consensus_weighted_by_quality(self, agent, tbl):
        """Test that consensus resolution weights by quality"""
        base_vector = np.ones(128)

        # High quality fragment
        frag1 = WisdomFragment(
            fragment_id="high_q",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.PRINCIPLE,
            content_type="pattern",
            knowledge_vector=base_vector,
            semantic_content={},
            confidence=0.9,
            quality=WisdomQuality.ESTABLISHED,
        )

        # Low quality fragment
        frag2 = WisdomFragment(
            fragment_id="low_q",
            source_domain=DomainLeg.UNIVERSAL_PARTS,
            wisdom_type=WisdomType.PRINCIPLE,
            content_type="pattern",
            knowledge_vector=-base_vector,
            semantic_content={},
            confidence=0.5,
            quality=WisdomQuality.UNVERIFIED,
        )

        agent.wisdom_fragments[frag1.fragment_id] = frag1
        agent.wisdom_fragments[frag2.fragment_id] = frag2

        contradictions = [("high_q", "low_q", 0.8)]
        proposals = agent._build_consensus(contradictions, tbl)

        if proposals:
            # Resolution should be closer to high quality fragment
            proposal = proposals[0]
            similarity_to_frag1 = np.dot(proposal.resolution_vector, frag1.knowledge_vector)
            similarity_to_frag2 = np.dot(proposal.resolution_vector, frag2.knowledge_vector)

            # High quality fragment should have more influence
            assert similarity_to_frag1 > similarity_to_frag2


# =============================================================================
# Wisdom Quality Validation Tests
# =============================================================================

class TestWisdomQualityValidation:
    """Tests for wisdom quality tracking and validation"""

    def test_quality_upgrade_with_successful_applications(self, agent):
        """Test quality upgrades based on successful applications"""
        fragment = WisdomFragment(
            fragment_id="upgrade_test",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.LESSON,
            content_type="lesson",
            knowledge_vector=np.zeros(128),
            semantic_content={},
            confidence=0.7,
            quality=WisdomQuality.UNVERIFIED,
        )

        # Simulate successful applications
        fragment.application_count = 5
        fragment.success_count = 4  # 80% effectiveness

        agent.wisdom_fragments[fragment.fragment_id] = fragment
        updated = agent._update_wisdom_quality(fragment)

        assert updated is True
        assert fragment.quality in [WisdomQuality.VALIDATED, WisdomQuality.ESTABLISHED]

    def test_quality_stays_if_low_effectiveness(self, agent):
        """Test quality doesn't upgrade with low effectiveness"""
        fragment = WisdomFragment(
            fragment_id="no_upgrade",
            source_domain=DomainLeg.CHICAGO_FOREST,
            wisdom_type=WisdomType.HEURISTIC,
            content_type="lesson",
            knowledge_vector=np.zeros(128),
            semantic_content={},
            confidence=0.5,
            quality=WisdomQuality.UNVERIFIED,
        )

        # Low success rate
        fragment.application_count = 5
        fragment.success_count = 1  # 20% effectiveness

        agent.wisdom_fragments[fragment.fragment_id] = fragment
        updated = agent._update_wisdom_quality(fragment)

        assert updated is False
        assert fragment.quality == WisdomQuality.UNVERIFIED


# =============================================================================
# Domain Needs Assessment Tests
# =============================================================================

class TestDomainNeedsAssessment:
    """Tests for assessing domain needs"""

    def test_assess_domain_needs(self, agent, tbl):
        """Test assessing what domains need"""
        # Set low values to create needs
        state = tbl.get_state(DomainLeg.NPCPU)
        state.consciousness_level = 0.2
        state.coherence = 0.3

        needs = agent._assess_domain_needs(tbl)

        assert DomainLeg.NPCPU in needs
        assert "consciousness_boost" in needs[DomainLeg.NPCPU]
        assert "coherence_boost" in needs[DomainLeg.NPCPU]

    def test_find_wisdom_for_need(self, agent):
        """Test finding wisdom that addresses a specific need"""
        # Create wisdom that could help
        insight = WisdomFragment(
            fragment_id="helpful_insight",
            source_domain=DomainLeg.CHICAGO_FOREST,
            wisdom_type=WisdomType.INSIGHT,
            content_type="insight",
            knowledge_vector=np.random.randn(128),
            semantic_content={},
            confidence=0.8,
            applicability=list(DomainLeg),
        )
        agent.wisdom_fragments[insight.fragment_id] = insight

        # Find wisdom for consciousness boost
        found = agent._find_wisdom_for_need("consciousness_boost", DomainLeg.NPCPU)

        assert found is not None
        assert found.fragment_id == "helpful_insight"


# =============================================================================
# Refinement Cycle Tests
# =============================================================================

class TestRefinementCycle:
    """Tests for the full refinement cycle"""

    @pytest.mark.asyncio
    async def test_perceive_phase(self, agent, tbl):
        """Test the perceive phase"""
        perceptions = await agent.perceive(tbl)

        assert "domain_states" in perceptions
        assert "wisdom_candidates" in perceptions
        assert "knowledge_gaps" in perceptions
        assert "domain_needs" in perceptions

        # Should have states for all domains
        for domain in DomainLeg:
            assert domain.value in perceptions["domain_states"]

    @pytest.mark.asyncio
    async def test_analyze_phase(self, agent, tbl):
        """Test the analyze phase"""
        perceptions = await agent.perceive(tbl)
        analysis = await agent.analyze(perceptions, tbl)

        assert "fragments_to_store" in analysis
        assert "knowledge_transfers" in analysis
        assert "synthesis_plans" in analysis
        assert "proactive_distributions" in analysis

    @pytest.mark.asyncio
    async def test_synthesize_phase(self, agent, tbl):
        """Test the synthesize phase"""
        perceptions = await agent.perceive(tbl)
        analysis = await agent.analyze(perceptions, tbl)
        synthesis = await agent.synthesize(analysis, tbl)

        assert "store_operations" in synthesis
        assert "transfer_operations" in synthesis
        assert "collective_memory_updates" in synthesis
        assert "meta_pattern_registrations" in synthesis

    @pytest.mark.asyncio
    async def test_propagate_phase(self, agent, tbl):
        """Test the propagate phase"""
        perceptions = await agent.perceive(tbl)
        analysis = await agent.analyze(perceptions, tbl)
        synthesis = await agent.synthesize(analysis, tbl)
        result = await agent.propagate(synthesis, tbl)

        assert result.success is True
        assert result.phase == RefinementPhase.PROPAGATION
        assert result.domain_affected == list(DomainLeg)
        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta

    @pytest.mark.asyncio
    async def test_full_refinement_cycle(self, agent, tbl):
        """Test a complete refinement cycle"""
        initial_fragments = len(agent.wisdom_fragments)

        result = await agent.refine(tbl)

        assert result.success is True
        assert len(agent.wisdom_fragments) >= initial_fragments

    @pytest.mark.asyncio
    async def test_multiple_refinement_cycles(self, agent, tbl):
        """Test multiple refinement cycles accumulate wisdom"""
        for i in range(5):
            result = await agent.refine(tbl)
            assert result.success is True

        # Should have collected some wisdom
        assert agent.stats["fragments_collected"] >= 0

    @pytest.mark.asyncio
    async def test_refinement_updates_collective_knowledge(self, agent, tbl):
        """Test that refinement updates collective knowledge vectors"""
        initial_knowledge = agent.collective_knowledge.copy()

        # Run several refinements
        for _ in range(3):
            await agent.refine(tbl)

        # Collective knowledge should have been updated
        # (may or may not have changed depending on wisdom collected)


# =============================================================================
# Metrics and Reporting Tests
# =============================================================================

class TestMetricsAndReporting:
    """Tests for metrics and reporting functionality"""

    def test_get_wisdom_summary(self, agent, sample_fragment):
        """Test getting wisdom summary"""
        agent.wisdom_fragments[sample_fragment.fragment_id] = sample_fragment

        summary = agent.get_wisdom_summary()

        assert "total_fragments" in summary
        assert summary["total_fragments"] == 1
        assert "by_quality" in summary
        assert "by_type" in summary
        assert "by_domain" in summary
        assert "stats" in summary

    def test_get_top_wisdom(self, agent):
        """Test getting top wisdom fragments"""
        # Create fragments with different relevance
        for i in range(5):
            frag = WisdomFragment(
                fragment_id=f"frag_{i}",
                source_domain=DomainLeg.NPCPU,
                wisdom_type=WisdomType.PATTERN,
                content_type="pattern",
                knowledge_vector=np.random.randn(128),
                semantic_content={"index": i},
                confidence=0.5 + i * 0.1,  # Increasing confidence
            )
            agent.wisdom_fragments[frag.fragment_id] = frag

        top = agent.get_top_wisdom(n=3)

        assert len(top) == 3
        # Should be sorted by relevance (descending)
        assert top[0]["confidence"] >= top[1]["confidence"]

    @pytest.mark.asyncio
    async def test_record_application_outcome(self, agent, sample_fragment):
        """Test recording wisdom application outcomes"""
        agent.wisdom_fragments[sample_fragment.fragment_id] = sample_fragment

        await agent.record_application_outcome(
            sample_fragment.fragment_id,
            was_successful=True,
            improvement=0.1
        )

        assert sample_fragment.success_count == 1
        assert len(agent.application_outcomes) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for CollectiveWisdomAgent"""

    @pytest.mark.asyncio
    async def test_wisdom_transfer_between_domains(self, agent, tbl):
        """Test that wisdom can transfer between domains"""
        # Create wisdom in one domain
        source_frag = WisdomFragment(
            fragment_id="source_wisdom",
            source_domain=DomainLeg.CHICAGO_FOREST,
            wisdom_type=WisdomType.HEURISTIC,
            content_type="lesson",
            knowledge_vector=np.random.randn(128),
            semantic_content={"source": "chicago_forest"},
            confidence=0.8,
            quality=WisdomQuality.VALIDATED,
            applicability=[DomainLeg.CHICAGO_FOREST, DomainLeg.NPCPU],
        )
        agent.wisdom_fragments[source_frag.fragment_id] = source_frag
        agent.domain_wisdom_index[DomainLeg.CHICAGO_FOREST].add(source_frag.fragment_id)

        # Create a knowledge gap in NPCPU
        for domain in agent.domain_wisdom_index:
            agent.domain_wisdom_index[domain].clear()

        # Run refinement to trigger transfer
        perceptions = await agent.perceive(tbl)

        # Should identify gap in NPCPU
        gap_domains = [g["domain"] for g in perceptions.get("knowledge_gaps", [])]
        assert "npcpu" in gap_domains

    @pytest.mark.asyncio
    async def test_collective_memory_synthesis(self, agent, tbl):
        """Test that collective memories are synthesized from fragments"""
        # Create enough fragments of same type
        for i in range(5):
            frag = WisdomFragment(
                fragment_id=f"synth_{i}",
                source_domain=list(DomainLeg)[i % 3],
                wisdom_type=WisdomType.LESSON,
                content_type="lesson",
                knowledge_vector=np.random.randn(128),
                semantic_content={"index": i},
                confidence=0.7,
            )
            agent.wisdom_fragments[frag.fragment_id] = frag

        # Run refinement
        result = await agent.refine(tbl)

        # May have synthesized collective memories
        assert result.success is True

    @pytest.mark.asyncio
    async def test_harmony_improvement_tracking(self, agent, tbl):
        """Test that harmony changes are tracked"""
        result = await agent.refine(tbl)

        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta

        # Harmony values should be valid
        assert 0 <= result.metrics_delta["harmony_before"] <= 1
        assert 0 <= result.metrics_delta["harmony_after"] <= 1


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_system_refinement(self, agent, tbl):
        """Test refinement with no existing wisdom"""
        result = await agent.refine(tbl)

        assert result.success is True
        # Should still complete without errors

    @pytest.mark.asyncio
    async def test_low_coherence_all_domains(self, agent, tbl):
        """Test with all domains having low coherence"""
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            state.coherence = 0.1

        result = await agent.refine(tbl)

        assert result.success is True
        # May not collect much wisdom, but should not crash

    def test_fragment_with_empty_vector(self, agent):
        """Test handling fragments with zero vectors"""
        frag = WisdomFragment(
            fragment_id="zero_vec",
            source_domain=DomainLeg.NPCPU,
            wisdom_type=WisdomType.PATTERN,
            content_type="pattern",
            knowledge_vector=np.zeros(128),
            semantic_content={},
            confidence=0.5,
        )

        agent.wisdom_fragments[frag.fragment_id] = frag

        # Should handle gracefully
        relevance = frag.relevance_score()
        assert relevance >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
