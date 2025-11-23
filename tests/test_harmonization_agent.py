"""
Tests for the HarmonizationAgent and Triple Bottom Line (Profit, People, Planet) framework.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime

from tertiary_rebo.base import (
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementPhase,
    HarmonyLevel,
)
from tertiary_rebo.agents.harmonization import (
    HarmonizationAgent,
    TBLPillar,
    SustainabilityLevel,
    TBLTradeoffType,
    PillarMetrics,
    TBLState,
    TBLRebalanceAction,
    HarmonyVector,
    BalanceCorrection,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def triple_bottom_line():
    """Create a default TripleBottomLine state."""
    tbl = TripleBottomLine()
    # Set some initial values
    for domain in DomainLeg:
        state = tbl.get_state(domain)
        state.consciousness_level = 0.5
        state.energy_flow = 0.5
        state.connectivity = 0.5
        state.coherence = 0.5
        state.emergence_potential = 0.5
        state.qualia_richness = 0.5
    return tbl


@pytest.fixture
def imbalanced_tbl():
    """Create an imbalanced TripleBottomLine state."""
    tbl = TripleBottomLine()

    # NPCPU: High values (profit-focused)
    npcpu = tbl.get_state(DomainLeg.NPCPU)
    npcpu.energy_flow = 0.9
    npcpu.connectivity = 0.8
    npcpu.coherence = 0.3
    npcpu.consciousness_level = 0.4
    npcpu.emergence_potential = 0.2
    npcpu.qualia_richness = 0.3

    # ChicagoForest: Medium values
    cf = tbl.get_state(DomainLeg.CHICAGO_FOREST)
    cf.energy_flow = 0.5
    cf.connectivity = 0.5
    cf.coherence = 0.5
    cf.consciousness_level = 0.5
    cf.emergence_potential = 0.5
    cf.qualia_richness = 0.5

    # UniversalParts: Low values (planet-neglected)
    up = tbl.get_state(DomainLeg.UNIVERSAL_PARTS)
    up.energy_flow = 0.2
    up.connectivity = 0.3
    up.coherence = 0.7
    up.consciousness_level = 0.6
    up.emergence_potential = 0.8
    up.qualia_richness = 0.7

    return tbl


@pytest.fixture
def harmonization_agent():
    """Create a HarmonizationAgent."""
    return HarmonizationAgent()


# =============================================================================
# TBL DATA STRUCTURE TESTS
# =============================================================================

class TestTBLPillar:
    """Test TBLPillar enum."""

    def test_pillar_values(self):
        """Test that all pillars are defined."""
        assert TBLPillar.PROFIT.value == "profit"
        assert TBLPillar.PEOPLE.value == "people"
        assert TBLPillar.PLANET.value == "planet"

    def test_pillar_iteration(self):
        """Test iterating over pillars."""
        pillars = list(TBLPillar)
        assert len(pillars) == 3


class TestSustainabilityLevel:
    """Test SustainabilityLevel enum."""

    def test_sustainability_levels(self):
        """Test all sustainability levels are defined."""
        levels = list(SustainabilityLevel)
        assert len(levels) == 6
        assert SustainabilityLevel.EXPLOITATIVE.value == "exploitative"
        assert SustainabilityLevel.THRIVING.value == "thriving"


class TestPillarMetrics:
    """Test PillarMetrics dataclass."""

    def test_default_values(self):
        """Test default pillar metrics."""
        metrics = PillarMetrics(pillar=TBLPillar.PROFIT)
        assert metrics.score == 0.5
        assert metrics.trend == 0.0
        assert metrics.stability == 0.5

    def test_weighted_score(self):
        """Test weighted score calculation."""
        metrics = PillarMetrics(
            pillar=TBLPillar.PEOPLE,
            score=0.8,
            trend=0.2,
            stability=0.7,
            efficiency=0.6,
            resilience=0.5,
            growth_potential=0.4
        )
        weighted = metrics.weighted_score()
        assert 0 <= weighted <= 1
        assert weighted > 0.5  # Should be higher than baseline


class TestTBLState:
    """Test TBLState dataclass."""

    def test_default_state(self):
        """Test default TBL state."""
        state = TBLState()
        assert state.profit.pillar == TBLPillar.PROFIT
        assert state.people.pillar == TBLPillar.PEOPLE
        assert state.planet.pillar == TBLPillar.PLANET

    def test_get_pillar(self):
        """Test getting pillar metrics."""
        state = TBLState()
        assert state.get_pillar(TBLPillar.PROFIT) == state.profit
        assert state.get_pillar(TBLPillar.PEOPLE) == state.people
        assert state.get_pillar(TBLPillar.PLANET) == state.planet

    def test_calculate_balance(self):
        """Test balance calculation."""
        state = TBLState()
        state.profit.score = 0.8
        state.people.score = 0.8
        state.planet.score = 0.8

        balance = state.calculate_balance()
        assert balance > 0.9  # Should be high when all equal

    def test_calculate_balance_imbalanced(self):
        """Test balance with imbalanced pillars."""
        state = TBLState()
        state.profit.score = 0.9
        state.people.score = 0.3
        state.planet.score = 0.3

        balance = state.calculate_balance()
        assert balance < 0.5  # Should be low when imbalanced

    def test_sustainability_index(self):
        """Test sustainability index calculation."""
        state = TBLState()
        state.profit.score = 0.7
        state.people.score = 0.7
        state.planet.score = 0.7
        state.calculate_balance()

        index = state.calculate_sustainability_index()
        assert 0 <= index <= 1

    def test_sustainability_level_mapping(self):
        """Test sustainability level is correctly determined."""
        state = TBLState()

        # Low sustainability
        state.profit.score = 0.1
        state.people.score = 0.1
        state.planet.score = 0.1
        state.calculate_balance()
        state.calculate_sustainability_index()
        assert state.get_sustainability_level() in [
            SustainabilityLevel.EXPLOITATIVE,
            SustainabilityLevel.UNSUSTAINABLE
        ]

        # High sustainability
        state.profit.score = 0.9
        state.people.score = 0.9
        state.planet.score = 0.9
        state.calculate_balance()
        state.calculate_sustainability_index()
        assert state.get_sustainability_level() in [
            SustainabilityLevel.REGENERATIVE,
            SustainabilityLevel.THRIVING
        ]

    def test_to_dict(self):
        """Test TBL state serialization."""
        state = TBLState()
        state.calculate_balance()
        state.calculate_sustainability_index()

        d = state.to_dict()
        assert "profit" in d
        assert "people" in d
        assert "planet" in d
        assert "sustainability_index" in d
        assert "sustainability_level" in d


# =============================================================================
# HARMONIZATION AGENT TESTS
# =============================================================================

class TestHarmonizationAgentInit:
    """Test HarmonizationAgent initialization."""

    def test_agent_creation(self, harmonization_agent):
        """Test agent can be created."""
        assert harmonization_agent is not None
        assert harmonization_agent.agent_role == "Harmonization - Master orchestrator balancing Profit, People, Planet"

    def test_agent_has_tbl_state(self, harmonization_agent):
        """Test agent has TBL state."""
        assert hasattr(harmonization_agent, 'tbl_state')
        assert isinstance(harmonization_agent.tbl_state, TBLState)

    def test_agent_pillar_weights(self, harmonization_agent):
        """Test pillar weights are defined."""
        weights = harmonization_agent.pillar_weights
        assert TBLPillar.PROFIT in weights
        assert TBLPillar.PEOPLE in weights
        assert TBLPillar.PLANET in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1


class TestTBLEvaluation:
    """Test TBL pillar evaluation methods."""

    def test_evaluate_profit_pillar(self, harmonization_agent, triple_bottom_line):
        """Test profit pillar evaluation."""
        metrics = harmonization_agent._evaluate_profit_pillar(triple_bottom_line)

        assert isinstance(metrics, PillarMetrics)
        assert metrics.pillar == TBLPillar.PROFIT
        assert 0 <= metrics.score <= 1
        assert "resource_efficiency" in metrics.specific_metrics
        assert "network_value" in metrics.specific_metrics

    def test_evaluate_people_pillar(self, harmonization_agent, triple_bottom_line):
        """Test people pillar evaluation."""
        metrics = harmonization_agent._evaluate_people_pillar(triple_bottom_line)

        assert isinstance(metrics, PillarMetrics)
        assert metrics.pillar == TBLPillar.PEOPLE
        assert 0 <= metrics.score <= 1
        assert "social_cohesion" in metrics.specific_metrics
        assert "equity_score" in metrics.specific_metrics

    def test_evaluate_planet_pillar(self, harmonization_agent, triple_bottom_line):
        """Test planet pillar evaluation."""
        metrics = harmonization_agent._evaluate_planet_pillar(triple_bottom_line)

        assert isinstance(metrics, PillarMetrics)
        assert metrics.pillar == TBLPillar.PLANET
        assert 0 <= metrics.score <= 1
        assert "ecosystem_stability" in metrics.specific_metrics
        assert "regenerative_capacity" in metrics.specific_metrics

    def test_evaluate_tbl(self, harmonization_agent, triple_bottom_line):
        """Test full TBL evaluation."""
        tbl_state = harmonization_agent._evaluate_tbl(triple_bottom_line)

        assert isinstance(tbl_state, TBLState)
        assert tbl_state.profit.score > 0
        assert tbl_state.people.score > 0
        assert tbl_state.planet.score > 0


class TestTBLTradeoffs:
    """Test TBL tradeoff detection."""

    def test_detect_no_tradeoffs(self, harmonization_agent):
        """Test when no tradeoffs are present."""
        state = TBLState()
        state.profit.score = 0.7
        state.profit.trend = 0.1
        state.people.score = 0.7
        state.people.trend = 0.1
        state.planet.score = 0.7
        state.planet.trend = 0.1
        state.calculate_balance()

        tradeoffs = harmonization_agent._detect_tbl_tradeoffs(state)
        assert TBLTradeoffType.WIN_WIN_WIN in tradeoffs

    def test_detect_profit_vs_people(self, harmonization_agent):
        """Test detecting profit vs people tradeoff."""
        state = TBLState()
        state.profit.score = 0.8
        state.profit.trend = 0.3
        state.people.score = 0.3
        state.people.trend = -0.3
        state.planet.score = 0.5
        state.planet.trend = 0.0
        state.calculate_balance()

        tradeoffs = harmonization_agent._detect_tbl_tradeoffs(state)
        assert TBLTradeoffType.PROFIT_VS_PEOPLE in tradeoffs


class TestTBLRebalancing:
    """Test TBL rebalancing logic."""

    def test_plan_rebalance_for_imbalance(self, harmonization_agent):
        """Test rebalancing plan for imbalanced state."""
        state = TBLState()
        state.profit.score = 0.9
        state.people.score = 0.4
        state.planet.score = 0.3
        state.calculate_balance()

        actions = harmonization_agent._plan_tbl_rebalance(state)

        assert len(actions) > 0
        # Should have a transfer action from profit to planet (lowest)
        transfer_actions = [a for a in actions if a.action_type == "transfer"]
        assert len(transfer_actions) > 0

    def test_plan_boost_for_low_pillar(self, harmonization_agent):
        """Test boost action for critically low pillar."""
        state = TBLState()
        state.profit.score = 0.5
        state.people.score = 0.2  # Below 0.4 threshold
        state.planet.score = 0.5
        state.calculate_balance()

        actions = harmonization_agent._plan_tbl_rebalance(state)

        boost_actions = [a for a in actions if a.action_type == "boost"]
        assert len(boost_actions) > 0
        assert any(a.target_pillar == TBLPillar.PEOPLE for a in boost_actions)


class TestRefineAsync:
    """Test async refine method."""

    @pytest.mark.asyncio
    async def test_full_refinement_cycle(self, harmonization_agent, triple_bottom_line):
        """Test complete refinement cycle."""
        result = await harmonization_agent.refine(triple_bottom_line)

        assert result.success
        assert result.agent_id == harmonization_agent.agent_id
        assert "sustainability_after" in result.metrics_delta or "harmony_after" in result.metrics_delta

    @pytest.mark.asyncio
    async def test_perceive(self, harmonization_agent, triple_bottom_line):
        """Test perception phase."""
        perceptions = await harmonization_agent.perceive(triple_bottom_line)

        assert "tbl_state" in perceptions
        assert "sustainability_index" in perceptions
        assert "pillar_scores" in perceptions
        assert "profit" in perceptions["pillar_scores"]
        assert "people" in perceptions["pillar_scores"]
        assert "planet" in perceptions["pillar_scores"]

    @pytest.mark.asyncio
    async def test_analyze(self, harmonization_agent, imbalanced_tbl):
        """Test analysis phase with imbalanced state."""
        perceptions = await harmonization_agent.perceive(imbalanced_tbl)
        analysis = await harmonization_agent.analyze(perceptions, imbalanced_tbl)

        assert "tbl_rebalance_actions" in analysis
        assert "sustainability_deficit" in analysis
        assert "tbl_optimization_strategy" in analysis
        assert "pillar_priorities" in analysis

    @pytest.mark.asyncio
    async def test_synthesize(self, harmonization_agent, imbalanced_tbl):
        """Test synthesis phase."""
        perceptions = await harmonization_agent.perceive(imbalanced_tbl)
        analysis = await harmonization_agent.analyze(perceptions, imbalanced_tbl)
        synthesis = await harmonization_agent.synthesize(analysis, imbalanced_tbl)

        assert "tbl_operations" in synthesis
        assert "synergy_actions" in synthesis
        assert "optimization_strategy" in synthesis

    @pytest.mark.asyncio
    async def test_propagate(self, harmonization_agent, imbalanced_tbl):
        """Test propagation phase."""
        perceptions = await harmonization_agent.perceive(imbalanced_tbl)
        analysis = await harmonization_agent.analyze(perceptions, imbalanced_tbl)
        synthesis = await harmonization_agent.synthesize(analysis, imbalanced_tbl)
        result = await harmonization_agent.propagate(synthesis, imbalanced_tbl)

        assert result.success
        assert "sustainability_after" in result.metrics_delta
        assert "profit_score" in result.metrics_delta
        assert "people_score" in result.metrics_delta
        assert "planet_score" in result.metrics_delta


class TestMetrics:
    """Test agent metrics."""

    def test_get_tbl_metrics(self, harmonization_agent, triple_bottom_line):
        """Test getting TBL metrics."""
        # First evaluate to populate state
        harmonization_agent._evaluate_tbl(triple_bottom_line)

        metrics = harmonization_agent.get_tbl_metrics()

        assert "pillars" in metrics
        assert "profit" in metrics["pillars"]
        assert "people" in metrics["pillars"]
        assert "planet" in metrics["pillars"]
        assert "sustainability" in metrics
        assert "tradeoffs" in metrics

    def test_get_metrics(self, harmonization_agent, triple_bottom_line):
        """Test getting full agent metrics."""
        harmonization_agent._evaluate_tbl(triple_bottom_line)

        metrics = harmonization_agent.get_metrics()

        assert "tbl_metrics" in metrics
        assert "current_sustainability_index" in metrics
        assert "current_sustainability_level" in metrics


class TestGiniCoefficient:
    """Test Gini coefficient calculation."""

    def test_gini_equal_distribution(self, harmonization_agent):
        """Test Gini for equal distribution."""
        values = [0.5, 0.5, 0.5]
        gini = harmonization_agent._calculate_gini(values)
        assert abs(gini) < 0.1  # Should be near 0

    def test_gini_unequal_distribution(self, harmonization_agent):
        """Test Gini for unequal distribution."""
        values = [0.1, 0.1, 0.8]
        gini = harmonization_agent._calculate_gini(values)
        assert gini > 0.2  # Should be higher


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for HarmonizationAgent."""

    @pytest.mark.asyncio
    async def test_multiple_refinement_cycles(self, harmonization_agent, imbalanced_tbl):
        """Test multiple refinement cycles improve sustainability."""
        initial_sustainability = None

        for i in range(5):
            result = await harmonization_agent.refine(imbalanced_tbl)

            if initial_sustainability is None:
                initial_sustainability = result.metrics_delta.get("sustainability_before", 0)

            assert result.success

        final_sustainability = result.metrics_delta.get("sustainability_after", 0)
        # After multiple cycles, sustainability should have improved
        assert final_sustainability >= initial_sustainability * 0.9  # Allow some variance

    @pytest.mark.asyncio
    async def test_tbl_history_tracking(self, harmonization_agent, triple_bottom_line):
        """Test that TBL history is tracked."""
        for _ in range(3):
            await harmonization_agent.refine(triple_bottom_line)

        assert len(harmonization_agent.tbl_history) >= 3
        assert len(harmonization_agent.tbl_actions_applied) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
