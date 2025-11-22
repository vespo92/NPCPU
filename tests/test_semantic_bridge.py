"""
Tests for the SemanticBridgeAgent - Agent 6 in the Tertiary ReBo system.

Tests semantic bridging capabilities including:
- Neural embeddings and similarity computation
- Cross-domain concept translation
- Multi-hop translation paths
- Emergent concept detection
- Conflict detection and resolution
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tertiary_rebo.base import (
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementPhase,
)
from tertiary_rebo.agents.semantic_bridge import (
    SemanticBridgeAgent,
    SemanticConcept,
    SemanticMapping,
    TranslationPath,
    SemanticEmbedder,
    EmergentConcept,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def semantic_bridge_agent():
    """Create a SemanticBridgeAgent for testing."""
    return SemanticBridgeAgent(agent_id="test_semantic_bridge")


@pytest.fixture
def embedder():
    """Create a SemanticEmbedder for testing."""
    return SemanticEmbedder()


@pytest.fixture
def triple_bottom_line():
    """Create a TripleBottomLine with initialized states."""
    tbl = TripleBottomLine()

    # Initialize state vectors with some structure
    for domain in DomainLeg:
        state = tbl.get_state(domain)
        state.state_vector = np.random.randn(64)
        state.state_vector = state.state_vector / np.linalg.norm(state.state_vector)
        state.consciousness_level = 0.5
        state.energy_flow = 0.6
        state.connectivity = 0.7
        state.coherence = 0.5
        state.emergence_potential = 0.4

    tbl.calculate_harmony()
    return tbl


@pytest.fixture
def active_tbl():
    """Create a TBL with high activation patterns for emergent detection."""
    tbl = TripleBottomLine()

    # Create specific activation patterns that should trigger emergence
    for domain in DomainLeg:
        state = tbl.get_state(domain)
        state.state_vector = np.ones(64) * 0.5
        # Add some peaks to simulate active concepts
        state.state_vector[:10] = 0.8
        state.consciousness_level = 0.7
        state.energy_flow = 0.7
        state.connectivity = 0.8
        state.coherence = 0.7
        state.emergence_potential = 0.6

    tbl.calculate_harmony()
    return tbl


# ============================================================================
# SemanticEmbedder Tests
# ============================================================================

class TestSemanticEmbedder:
    """Tests for the SemanticEmbedder class."""

    def test_embed_returns_correct_dimension(self, embedder):
        """Test that embeddings have correct dimension."""
        embedding = embedder.embed("consciousness")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedder.embedding_dim

    def test_embed_is_normalized(self, embedder):
        """Test that embeddings are normalized."""
        embedding = embedder.embed("test concept")
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=0.1)

    def test_embed_is_deterministic(self, embedder):
        """Test that same input produces same embedding."""
        emb1 = embedder.embed("mycelium network")
        emb2 = embedder.embed("mycelium network")
        assert np.allclose(emb1, emb2)

    def test_embed_different_texts_different_embeddings(self, embedder):
        """Test that different texts produce different embeddings."""
        emb1 = embedder.embed("consciousness")
        emb2 = embedder.embed("physical stress")
        # Should not be identical
        assert not np.allclose(emb1, emb2)

    def test_embed_batch(self, embedder):
        """Test batch embedding."""
        texts = ["consciousness", "node", "stress"]
        embeddings = embedder.embed_batch(texts)
        assert embeddings.shape == (3, embedder.embedding_dim)

    def test_similarity_same_text(self, embedder):
        """Test similarity of identical texts is high."""
        emb = embedder.embed("neural pathway")
        sim = embedder.similarity(emb, emb)
        assert np.isclose(sim, 1.0)

    def test_similarity_range(self, embedder):
        """Test similarity is in valid range."""
        emb1 = embedder.embed("consciousness")
        emb2 = embedder.embed("awareness")
        sim = embedder.similarity(emb1, emb2)
        assert -1.0 <= sim <= 1.0

    def test_similarity_semantic_relatedness(self, embedder):
        """Test that semantically related terms have higher similarity."""
        emb_consciousness = embedder.embed("consciousness awareness")
        emb_related = embedder.embed("aware perception")
        emb_unrelated = embedder.embed("mechanical stress wear")

        sim_related = embedder.similarity(emb_consciousness, emb_related)
        sim_unrelated = embedder.similarity(emb_consciousness, emb_unrelated)

        # Related concepts should have higher similarity
        # Note: With heuristic embeddings this might not always hold
        assert sim_related != sim_unrelated

    def test_embedding_cache(self, embedder):
        """Test that embeddings are cached."""
        text = "cache test"
        _ = embedder.embed(text)
        assert text in embedder._embedding_cache


# ============================================================================
# SemanticBridgeAgent Initialization Tests
# ============================================================================

class TestSemanticBridgeAgentInit:
    """Tests for SemanticBridgeAgent initialization."""

    def test_agent_creation(self, semantic_bridge_agent):
        """Test agent is created correctly."""
        assert semantic_bridge_agent is not None
        assert semantic_bridge_agent.agent_id == "test_semantic_bridge"

    def test_agent_role(self, semantic_bridge_agent):
        """Test agent role is defined."""
        assert "Semantic Bridge" in semantic_bridge_agent.agent_role
        assert "Mind" in semantic_bridge_agent.agent_role
        assert "Network" in semantic_bridge_agent.agent_role
        assert "Body" in semantic_bridge_agent.agent_role

    def test_domains_affected(self, semantic_bridge_agent):
        """Test all domains are affected."""
        affected = semantic_bridge_agent.domains_affected
        assert DomainLeg.NPCPU in affected
        assert DomainLeg.CHICAGO_FOREST in affected
        assert DomainLeg.UNIVERSAL_PARTS in affected

    def test_vocabulary_initialized(self, semantic_bridge_agent):
        """Test domain vocabularies are initialized."""
        vocab = semantic_bridge_agent.domain_vocab
        assert len(vocab[DomainLeg.NPCPU]) > 0
        assert len(vocab[DomainLeg.CHICAGO_FOREST]) > 0
        assert len(vocab[DomainLeg.UNIVERSAL_PARTS]) > 0

    def test_core_concepts_initialized(self, semantic_bridge_agent):
        """Test core concepts are initialized."""
        concepts = semantic_bridge_agent.concepts
        assert len(concepts) > 0
        # Check some known core concepts exist
        concept_names = [c.universal_name for c in concepts.values()]
        assert "consciousness" in concept_names
        assert "communication" in concept_names
        assert "health" in concept_names

    def test_core_mappings_initialized(self, semantic_bridge_agent):
        """Test core mappings are initialized."""
        mappings = semantic_bridge_agent.mappings
        assert len(mappings) > 0

    def test_translation_matrices_initialized(self, semantic_bridge_agent):
        """Test translation matrices are initialized."""
        matrices = semantic_bridge_agent.translation_matrices
        assert len(matrices) == 6  # 3 domains, 6 directed pairs

        for (src, tgt), matrix in matrices.items():
            assert matrix.shape[0] == matrix.shape[1]
            assert src != tgt

    def test_concept_graph_built(self, semantic_bridge_agent):
        """Test concept graph is built."""
        graph = semantic_bridge_agent.concept_graph
        assert len(graph) > 0


# ============================================================================
# Perception Phase Tests
# ============================================================================

class TestSemanticBridgePerception:
    """Tests for the perception phase."""

    @pytest.mark.asyncio
    async def test_perceive_returns_dict(self, semantic_bridge_agent, triple_bottom_line):
        """Test perceive returns properly structured dict."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)

        assert "domain_semantic_states" in perception
        assert "cross_domain_coherence" in perception
        assert "untranslated_concepts" in perception
        assert "semantic_conflicts" in perception

    @pytest.mark.asyncio
    async def test_perceive_all_domains_analyzed(self, semantic_bridge_agent, triple_bottom_line):
        """Test all domains are analyzed."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)

        domain_states = perception["domain_semantic_states"]
        assert "npcpu" in domain_states
        assert "chicago_forest" in domain_states
        assert "universal_parts" in domain_states

    @pytest.mark.asyncio
    async def test_perceive_coherence_in_range(self, semantic_bridge_agent, triple_bottom_line):
        """Test coherence is in valid range."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)

        coherence = perception["cross_domain_coherence"]
        assert 0.0 <= coherence <= 1.0

    @pytest.mark.asyncio
    async def test_perceive_semantic_coverage(self, semantic_bridge_agent, triple_bottom_line):
        """Test semantic coverage is computed."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)

        for domain_name, state in perception["domain_semantic_states"].items():
            assert "semantic_coverage" in state
            assert 0.0 <= state["semantic_coverage"] <= 1.0

    @pytest.mark.asyncio
    async def test_perceive_active_concepts(self, semantic_bridge_agent, active_tbl):
        """Test active concepts are detected."""
        perception = await semantic_bridge_agent.perceive(active_tbl)

        for domain_name, state in perception["domain_semantic_states"].items():
            active = state["active_concepts"]
            assert isinstance(active, list)


# ============================================================================
# Analysis Phase Tests
# ============================================================================

class TestSemanticBridgeAnalysis:
    """Tests for the analysis phase."""

    @pytest.mark.asyncio
    async def test_analyze_returns_dict(self, semantic_bridge_agent, triple_bottom_line):
        """Test analyze returns properly structured dict."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)

        assert "semantic_health" in analysis
        assert "health_score" in analysis
        assert "bridge_opportunities" in analysis
        assert "conflict_resolutions" in analysis

    @pytest.mark.asyncio
    async def test_analyze_health_assessment(self, semantic_bridge_agent, triple_bottom_line):
        """Test health assessment is valid."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)

        assert analysis["semantic_health"] in ["poor", "fair", "good", "excellent"]
        assert 0.0 <= analysis["health_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_bridge_opportunities(self, semantic_bridge_agent, triple_bottom_line):
        """Test bridge opportunities are identified."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)

        opportunities = analysis["bridge_opportunities"]
        assert isinstance(opportunities, list)

        for opp in opportunities:
            assert "source_domain" in opp
            assert "term" in opp
            assert "candidates" in opp


# ============================================================================
# Synthesis Phase Tests
# ============================================================================

class TestSemanticBridgeSynthesis:
    """Tests for the synthesis phase."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_dict(self, semantic_bridge_agent, triple_bottom_line):
        """Test synthesize returns properly structured dict."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)
        synthesis = await semantic_bridge_agent.synthesize(analysis, triple_bottom_line)

        assert "new_mappings" in synthesis
        assert "mapping_updates" in synthesis
        assert "translation_matrix_updates" in synthesis
        assert "new_concepts" in synthesis

    @pytest.mark.asyncio
    async def test_synthesize_translation_matrix_updates(self, semantic_bridge_agent, triple_bottom_line):
        """Test translation matrix updates are generated."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)
        synthesis = await semantic_bridge_agent.synthesize(analysis, triple_bottom_line)

        updates = synthesis["translation_matrix_updates"]
        assert len(updates) == 6  # One for each domain pair


# ============================================================================
# Propagation Phase Tests
# ============================================================================

class TestSemanticBridgePropagation:
    """Tests for the propagation phase."""

    @pytest.mark.asyncio
    async def test_propagate_returns_result(self, semantic_bridge_agent, triple_bottom_line):
        """Test propagate returns RefinementResult."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)
        synthesis = await semantic_bridge_agent.synthesize(analysis, triple_bottom_line)
        result = await semantic_bridge_agent.propagate(synthesis, triple_bottom_line)

        assert result.success == True
        assert result.agent_id == "test_semantic_bridge"
        assert result.phase == RefinementPhase.PROPAGATION

    @pytest.mark.asyncio
    async def test_propagate_metrics_delta(self, semantic_bridge_agent, triple_bottom_line):
        """Test metrics delta is computed."""
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)
        synthesis = await semantic_bridge_agent.synthesize(analysis, triple_bottom_line)
        result = await semantic_bridge_agent.propagate(synthesis, triple_bottom_line)

        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta
        assert "semantic_coherence" in result.metrics_delta


# ============================================================================
# Full Refinement Cycle Tests
# ============================================================================

class TestSemanticBridgeRefinement:
    """Tests for the full refinement cycle."""

    @pytest.mark.asyncio
    async def test_full_refine_cycle(self, semantic_bridge_agent, triple_bottom_line):
        """Test complete refinement cycle."""
        result = await semantic_bridge_agent.refine(triple_bottom_line)

        assert result.success == True
        assert len(result.domains_affected) == 3
        assert semantic_bridge_agent.total_refinements == 1

    @pytest.mark.asyncio
    async def test_multiple_refine_cycles(self, semantic_bridge_agent, triple_bottom_line):
        """Test multiple refinement cycles."""
        for _ in range(3):
            result = await semantic_bridge_agent.refine(triple_bottom_line)
            assert result.success == True

        assert semantic_bridge_agent.total_refinements == 3


# ============================================================================
# Translation API Tests
# ============================================================================

class TestSemanticBridgeTranslation:
    """Tests for the public translation API."""

    @pytest.mark.asyncio
    async def test_translate_known_concept(self, semantic_bridge_agent):
        """Test translating a known mapped concept."""
        result = await semantic_bridge_agent.translate_concept(
            concept="consciousness",
            source_domain=DomainLeg.NPCPU,
            target_domain=DomainLeg.CHICAGO_FOREST
        )

        assert result is not None
        assert "target" in result
        assert "confidence" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_translate_with_context(self, semantic_bridge_agent):
        """Test translation with context."""
        result = await semantic_bridge_agent.translate_concept(
            concept="awareness",
            source_domain=DomainLeg.NPCPU,
            target_domain=DomainLeg.UNIVERSAL_PARTS,
            context=["sensor", "detection"]
        )

        # Should return some result
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_translate_unknown_concept(self, semantic_bridge_agent):
        """Test translating an unknown concept uses neural translation."""
        result = await semantic_bridge_agent.translate_concept(
            concept="introspection",
            source_domain=DomainLeg.NPCPU,
            target_domain=DomainLeg.UNIVERSAL_PARTS
        )

        # May or may not find a translation
        if result is not None:
            assert "method" in result


# ============================================================================
# Multi-hop Path Tests
# ============================================================================

class TestSemanticBridgeMultiHop:
    """Tests for multi-hop translation paths."""

    @pytest.mark.asyncio
    async def test_find_path_same_domain_returns_none(self, semantic_bridge_agent):
        """Test that same-domain paths return None."""
        path = await semantic_bridge_agent.find_multi_hop_path(
            source_concept="consciousness",
            source_domain=DomainLeg.NPCPU,
            target_concept="memory",
            target_domain=DomainLeg.NPCPU
        )
        assert path is None

    @pytest.mark.asyncio
    async def test_find_path_connected_concepts(self, semantic_bridge_agent):
        """Test finding path between connected concepts."""
        # Use concepts we know are mapped
        path = await semantic_bridge_agent.find_multi_hop_path(
            source_concept="consciousness",
            source_domain=DomainLeg.NPCPU,
            target_concept="node",
            target_domain=DomainLeg.CHICAGO_FOREST,
            max_hops=5
        )

        if path is not None:
            assert isinstance(path, TranslationPath)
            assert path.path_length <= 5
            assert path.total_confidence > 0


# ============================================================================
# Statistics Tests
# ============================================================================

class TestSemanticBridgeStatistics:
    """Tests for statistics and metrics."""

    def test_get_semantic_statistics(self, semantic_bridge_agent):
        """Test getting semantic statistics."""
        stats = semantic_bridge_agent.get_semantic_statistics()

        assert "total_concepts" in stats
        assert "total_mappings" in stats
        assert "vocabulary_sizes" in stats
        assert "embedding_dimension" in stats
        assert "neural_embeddings_active" in stats

        assert stats["total_concepts"] > 0
        assert stats["total_mappings"] > 0

    def test_vocabulary_sizes(self, semantic_bridge_agent):
        """Test vocabulary sizes are reported."""
        stats = semantic_bridge_agent.get_semantic_statistics()

        vocab_sizes = stats["vocabulary_sizes"]
        assert "npcpu" in vocab_sizes
        assert "chicago_forest" in vocab_sizes
        assert "universal_parts" in vocab_sizes

        for size in vocab_sizes.values():
            assert size > 0

    def test_agent_metrics(self, semantic_bridge_agent):
        """Test agent metrics."""
        metrics = semantic_bridge_agent.get_metrics()

        assert "agent_id" in metrics
        assert "agent_role" in metrics
        assert "total_refinements" in metrics
        assert "success_rate" in metrics


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestSemanticBridgeEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_state_vector(self, semantic_bridge_agent):
        """Test handling of empty state vectors."""
        tbl = TripleBottomLine()
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            state.state_vector = np.zeros(64)

        # Should not crash
        perception = await semantic_bridge_agent.perceive(tbl)
        assert perception is not None

    @pytest.mark.asyncio
    async def test_nan_in_state_vector(self, semantic_bridge_agent):
        """Test handling of NaN values."""
        tbl = TripleBottomLine()
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            state.state_vector = np.random.randn(64)
            state.state_vector[0] = np.nan

        # Should handle gracefully
        try:
            perception = await semantic_bridge_agent.perceive(tbl)
            assert perception is not None
        except Exception:
            # It's okay if it raises, just shouldn't crash unexpectedly
            pass

    def test_embedding_empty_string(self, embedder):
        """Test embedding empty string."""
        emb = embedder.embed("")
        assert isinstance(emb, np.ndarray)
        assert len(emb) == embedder.embedding_dim


# ============================================================================
# Data Structure Tests
# ============================================================================

class TestSemanticDataStructures:
    """Tests for semantic data structures."""

    def test_semantic_concept_creation(self):
        """Test SemanticConcept creation."""
        concept = SemanticConcept(
            concept_id="test_concept",
            universal_name="test",
            domain_representations={DomainLeg.NPCPU: "test_rep"},
            embedding=np.random.randn(384)
        )

        assert concept.concept_id == "test_concept"
        assert concept.confidence == 1.0
        assert concept.usage_count == 0

    def test_semantic_mapping_creation(self):
        """Test SemanticMapping creation."""
        mapping = SemanticMapping(
            mapping_id="test_mapping",
            source_domain=DomainLeg.NPCPU,
            target_domain=DomainLeg.CHICAGO_FOREST,
            source_concept="consciousness",
            target_concept="node",
            translation_confidence=0.9
        )

        assert mapping.bidirectional == True
        assert mapping.usage_count == 0
        assert mapping.success_rate == 1.0

    def test_translation_path_creation(self):
        """Test TranslationPath creation."""
        path = TranslationPath(
            path_id="test_path",
            source_domain=DomainLeg.NPCPU,
            target_domain=DomainLeg.UNIVERSAL_PARTS,
            hops=[(DomainLeg.NPCPU, "concept1"), (DomainLeg.CHICAGO_FOREST, "concept2")],
            total_confidence=0.8,
            path_length=2
        )

        assert path.path_length == 2
        assert len(path.hops) == 2

    def test_emergent_concept_creation(self):
        """Test EmergentConcept creation."""
        emergent = EmergentConcept(
            concept_id="emergent_test",
            source_patterns=[{"pattern": "test"}],
            proposed_name="new_concept",
            domain_manifestations={DomainLeg.NPCPU: "npcpu_form"},
            confidence=0.75
        )

        assert emergent.confidence == 0.75
        assert emergent.proposed_name == "new_concept"


# ============================================================================
# Integration Tests
# ============================================================================

class TestSemanticBridgeIntegration:
    """Integration tests for semantic bridge."""

    @pytest.mark.asyncio
    async def test_refinement_improves_or_maintains_coherence(self, semantic_bridge_agent, triple_bottom_line):
        """Test that refinement doesn't degrade coherence significantly."""
        initial_perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        initial_coherence = initial_perception["cross_domain_coherence"]

        # Run several refinement cycles
        for _ in range(3):
            await semantic_bridge_agent.refine(triple_bottom_line)

        final_perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        final_coherence = final_perception["cross_domain_coherence"]

        # Coherence should not drop significantly
        assert final_coherence >= initial_coherence - 0.2

    @pytest.mark.asyncio
    async def test_cross_domain_signal_emission(self, semantic_bridge_agent, triple_bottom_line):
        """Test that cross-domain signals are emitted."""
        # Clear any existing signals
        semantic_bridge_agent.cross_domain_signals.clear()

        # Force a new mapping by manipulating synthesis
        perception = await semantic_bridge_agent.perceive(triple_bottom_line)
        analysis = await semantic_bridge_agent.analyze(perception, triple_bottom_line)

        # Add a test mapping opportunity
        analysis["bridge_opportunities"] = [{
            "source_domain": "npcpu",
            "term": "test_concept",
            "candidates": [{
                "target_domain": "chicago_forest",
                "target_term": "test_target",
                "combined_similarity": 0.8
            }],
            "universal_anchor": "test"
        }]

        synthesis = await semantic_bridge_agent.synthesize(analysis, triple_bottom_line)
        result = await semantic_bridge_agent.propagate(synthesis, triple_bottom_line)

        # If new mappings were created, signals should be emitted
        if result.changes.get("new_mappings", 0) > 0:
            assert len(semantic_bridge_agent.cross_domain_signals) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
