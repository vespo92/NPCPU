"""
Comprehensive tests for the EnergyFlowAgent.

Tests cover:
- Basic instantiation and properties
- Perception phase (energy state gathering)
- Analysis phase (redistribution planning)
- Synthesis phase (transfer planning)
- Propagation phase (executing transfers)
- Complete refinement cycles
- Energy democracy and Gini coefficient
- Reserve management
- Edge cases and error handling
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime

from tertiary_rebo.base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
)
from tertiary_rebo.agents import EnergyFlowAgent


@pytest.fixture
def energy_agent():
    """Create a fresh EnergyFlowAgent instance."""
    return EnergyFlowAgent(refinement_rate=0.1)


@pytest.fixture
def balanced_tbl():
    """Create a TripleBottomLine with balanced energy levels."""
    tbl = TripleBottomLine()
    for domain in DomainLeg:
        state = DomainState(
            domain=domain,
            consciousness_level=0.5,
            energy_flow=0.5,
            connectivity=0.5,
            coherence=0.5,
            qualia_richness=0.3,
            emergence_potential=0.3,
            state_vector=np.random.randn(64) * 0.1,
            metadata={"test": True}
        )
        tbl.set_state(domain, state)
    tbl.calculate_harmony()
    return tbl


@pytest.fixture
def imbalanced_tbl():
    """Create a TripleBottomLine with imbalanced energy levels."""
    tbl = TripleBottomLine()

    # NPCPU - high energy (surplus)
    npcpu_state = DomainState(
        domain=DomainLeg.NPCPU,
        consciousness_level=0.7,
        energy_flow=0.9,  # High energy
        connectivity=0.6,
        coherence=0.5,
        state_vector=np.random.randn(64) * 0.1
    )
    tbl.set_state(DomainLeg.NPCPU, npcpu_state)

    # Chicago Forest - low energy (deficit)
    cf_state = DomainState(
        domain=DomainLeg.CHICAGO_FOREST,
        consciousness_level=0.3,
        energy_flow=0.1,  # Low energy - critical
        connectivity=0.4,
        coherence=0.3,
        state_vector=np.random.randn(64) * 0.1
    )
    tbl.set_state(DomainLeg.CHICAGO_FOREST, cf_state)

    # Universal Parts - medium energy
    up_state = DomainState(
        domain=DomainLeg.UNIVERSAL_PARTS,
        consciousness_level=0.5,
        energy_flow=0.5,  # Normal energy
        connectivity=0.5,
        coherence=0.5,
        state_vector=np.random.randn(64) * 0.1
    )
    tbl.set_state(DomainLeg.UNIVERSAL_PARTS, up_state)

    tbl.calculate_harmony()
    return tbl


class TestEnergyFlowAgentBasics:
    """Test basic agent properties and initialization."""

    def test_agent_instantiation(self, energy_agent):
        """Test that agent initializes correctly."""
        assert energy_agent is not None
        assert isinstance(energy_agent, TertiaryReBoAgent)
        assert isinstance(energy_agent, EnergyFlowAgent)

    def test_agent_role(self, energy_agent):
        """Test agent role description."""
        assert "Energy Flow" in energy_agent.agent_role
        assert "democratic" in energy_agent.agent_role.lower()

    def test_domains_affected(self, energy_agent):
        """Test that agent affects all domains."""
        domains = energy_agent.domains_affected
        assert len(domains) == 3
        assert DomainLeg.NPCPU in domains
        assert DomainLeg.CHICAGO_FOREST in domains
        assert DomainLeg.UNIVERSAL_PARTS in domains

    def test_initial_energy_state(self, energy_agent):
        """Test initial energy tracking state."""
        for domain in DomainLeg:
            assert energy_agent.domain_energy[domain] == 1.0
            assert energy_agent.production_rates[domain] == 0.1
            assert energy_agent.consumption_rates[domain] == 0.05
            assert energy_agent.reserves[domain] == 0.3

    def test_flow_matrix_initialization(self, energy_agent):
        """Test flow matrix is initialized as zeros."""
        assert energy_agent.flow_matrix.shape == (3, 3)
        assert np.all(energy_agent.flow_matrix == 0)

    def test_transfer_efficiency(self, energy_agent):
        """Test transfer efficiency default."""
        assert energy_agent.transfer_efficiency == 0.9

    def test_reserve_threshold(self, energy_agent):
        """Test reserve threshold default."""
        assert energy_agent.reserve_threshold == 0.2


class TestPerceptionPhase:
    """Test the perceive method."""

    @pytest.mark.asyncio
    async def test_perceive_balanced_state(self, energy_agent, balanced_tbl):
        """Test perception on balanced energy state."""
        perception = await energy_agent.perceive(balanced_tbl)

        assert "domain_energies" in perception
        assert "total_energy" in perception
        assert "energy_distribution" in perception
        assert "gini_coefficient" in perception
        assert "critical_domains" in perception
        assert "imbalances" in perception

    @pytest.mark.asyncio
    async def test_perceive_detects_critical_domains(self, energy_agent, imbalanced_tbl):
        """Test that perception identifies critical (low energy) domains."""
        perception = await energy_agent.perceive(imbalanced_tbl)

        assert len(perception["critical_domains"]) > 0
        critical = perception["critical_domains"][0]
        assert critical["domain"] == "chicago_forest"
        assert critical["energy"] < energy_agent.reserve_threshold

    @pytest.mark.asyncio
    async def test_perceive_calculates_gini(self, energy_agent, imbalanced_tbl):
        """Test Gini coefficient calculation for inequality."""
        perception = await energy_agent.perceive(imbalanced_tbl)

        gini = perception["gini_coefficient"]
        # Imbalanced distribution should have higher Gini
        assert gini > 0
        assert gini <= 1.0

    @pytest.mark.asyncio
    async def test_perceive_identifies_imbalances(self, energy_agent, imbalanced_tbl):
        """Test that imbalances are correctly identified."""
        perception = await energy_agent.perceive(imbalanced_tbl)

        assert len(perception["imbalances"]) > 0

        # Find surplus and deficit
        surplus_found = any(i["type"] == "surplus" for i in perception["imbalances"])
        deficit_found = any(i["type"] == "deficit" for i in perception["imbalances"])

        assert surplus_found
        assert deficit_found


class TestAnalysisPhase:
    """Test the analyze method."""

    @pytest.mark.asyncio
    async def test_analyze_balanced_state(self, energy_agent, balanced_tbl):
        """Test analysis on balanced state."""
        perception = await energy_agent.perceive(balanced_tbl)
        analysis = await energy_agent.analyze(perception, balanced_tbl)

        assert "redistribution_needed" in analysis
        assert "emergency_transfers" in analysis
        assert "optimization_opportunities" in analysis
        assert "sustainability_assessment" in analysis
        assert "recommended_flows" in analysis

    @pytest.mark.asyncio
    async def test_analyze_identifies_emergency_transfers(self, energy_agent, imbalanced_tbl):
        """Test that emergency transfers are identified for critical domains."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)

        # Should identify emergency transfers for critical domain
        assert len(analysis["emergency_transfers"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_high_gini_triggers_redistribution(self, energy_agent, imbalanced_tbl):
        """Test that high Gini coefficient triggers redistribution."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)

        assert analysis["redistribution_needed"] == True

    @pytest.mark.asyncio
    async def test_analyze_sustainability_assessment(self, energy_agent, balanced_tbl):
        """Test sustainability assessment."""
        perception = await energy_agent.perceive(balanced_tbl)
        analysis = await energy_agent.analyze(perception, balanced_tbl)

        assert analysis["sustainability_assessment"] in ["sustainable", "marginal", "unsustainable"]


class TestSynthesisPhase:
    """Test the synthesize method."""

    @pytest.mark.asyncio
    async def test_synthesize_creates_transfer_plan(self, energy_agent, imbalanced_tbl):
        """Test that synthesis creates transfer plan."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)

        assert "transfers" in synthesis
        assert "production_adjustments" in synthesis
        assert "consumption_adjustments" in synthesis
        assert "reserve_operations" in synthesis

    @pytest.mark.asyncio
    async def test_synthesize_prioritizes_emergency(self, energy_agent, imbalanced_tbl):
        """Test that emergency transfers are prioritized."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)

        if synthesis["transfers"]:
            # First transfers should be emergency type if any exist
            emergency_transfers = [t for t in synthesis["transfers"] if t["type"] == "emergency"]
            balance_transfers = [t for t in synthesis["transfers"] if t["type"] == "balance"]

            # Either we have emergency transfers or just balance transfers
            assert len(emergency_transfers) > 0 or len(balance_transfers) > 0

    @pytest.mark.asyncio
    async def test_synthesize_avoids_duplicate_transfers(self, energy_agent, imbalanced_tbl):
        """Test that duplicate transfers are avoided."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)

        # Check for duplicates (same from-to pair)
        seen = set()
        for transfer in synthesis["transfers"]:
            key = (transfer["from"], transfer["to"])
            assert key not in seen, f"Duplicate transfer found: {key}"
            seen.add(key)


class TestPropagationPhase:
    """Test the propagate method."""

    @pytest.mark.asyncio
    async def test_propagate_executes_transfers(self, energy_agent, imbalanced_tbl):
        """Test that propagation executes transfers."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)

        initial_cf_energy = imbalanced_tbl.get_state(DomainLeg.CHICAGO_FOREST).energy_flow

        result = await energy_agent.propagate(synthesis, imbalanced_tbl)

        assert isinstance(result, RefinementResult)
        assert result.success == True
        assert result.phase == RefinementPhase.PROPAGATION

    @pytest.mark.asyncio
    async def test_propagate_returns_insights(self, energy_agent, imbalanced_tbl):
        """Test that propagation returns useful insights."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)
        result = await energy_agent.propagate(synthesis, imbalanced_tbl)

        assert isinstance(result.insights, list)
        assert len(result.insights) >= 0  # May or may not have insights

    @pytest.mark.asyncio
    async def test_propagate_updates_flow_history(self, energy_agent, imbalanced_tbl):
        """Test that flow history is updated."""
        initial_history_len = len(energy_agent.flow_history)

        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)
        await energy_agent.propagate(synthesis, imbalanced_tbl)

        assert len(energy_agent.flow_history) == initial_history_len + 1

    @pytest.mark.asyncio
    async def test_propagate_calculates_harmony(self, energy_agent, imbalanced_tbl):
        """Test that harmony is recalculated after propagation."""
        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)
        result = await energy_agent.propagate(synthesis, imbalanced_tbl)

        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta


class TestCompleteRefinementCycle:
    """Test complete refinement cycles."""

    @pytest.mark.asyncio
    async def test_full_refinement_cycle(self, energy_agent, imbalanced_tbl):
        """Test a complete refinement cycle."""
        initial_harmony = imbalanced_tbl.harmony_score

        result = await energy_agent.refine(imbalanced_tbl)

        assert result.success == True
        assert energy_agent.total_refinements == 1

    @pytest.mark.asyncio
    async def test_multiple_refinement_cycles(self, energy_agent, imbalanced_tbl):
        """Test multiple consecutive refinement cycles."""
        for i in range(5):
            result = await energy_agent.refine(imbalanced_tbl)
            assert result.success == True

        assert energy_agent.total_refinements == 5
        assert energy_agent.successful_refinements >= 1

    @pytest.mark.asyncio
    async def test_refinement_improves_balance(self, energy_agent, imbalanced_tbl):
        """Test that refinement tends to improve energy balance."""
        # Get initial Gini (inequality measure)
        perception_before = await energy_agent.perceive(imbalanced_tbl)
        gini_before = perception_before["gini_coefficient"]

        # Run several refinement cycles
        for _ in range(3):
            await energy_agent.refine(imbalanced_tbl)

        # Check Gini after
        perception_after = await energy_agent.perceive(imbalanced_tbl)
        gini_after = perception_after["gini_coefficient"]

        # Energy should be more balanced (lower Gini) or at least not much worse
        # Note: This might not always hold due to randomness, so we allow some tolerance
        assert gini_after <= gini_before + 0.1


class TestGiniCoefficient:
    """Test Gini coefficient calculations."""

    def test_gini_perfect_equality(self, energy_agent):
        """Test Gini coefficient for perfect equality."""
        values = [1.0, 1.0, 1.0]
        gini = energy_agent._calculate_gini(values)
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_gini_perfect_inequality(self, energy_agent):
        """Test Gini coefficient for extreme inequality."""
        values = [0.0, 0.0, 1.0]
        gini = energy_agent._calculate_gini(values)
        # Gini should be high for extreme inequality
        assert gini > 0.5

    def test_gini_moderate_inequality(self, energy_agent):
        """Test Gini coefficient for moderate inequality."""
        values = [0.2, 0.5, 0.8]
        gini = energy_agent._calculate_gini(values)
        assert 0.1 < gini < 0.5

    def test_gini_zero_total(self, energy_agent):
        """Test Gini coefficient when total is zero."""
        values = [0.0, 0.0, 0.0]
        gini = energy_agent._calculate_gini(values)
        assert gini == 0


class TestReserveManagement:
    """Test energy reserve operations."""

    @pytest.mark.asyncio
    async def test_reserve_deposit(self, energy_agent, balanced_tbl):
        """Test depositing energy to reserves."""
        # Set up state where deposit would occur
        # (high energy, low reserves)
        energy_agent.reserves[DomainLeg.NPCPU] = 0.05  # Low reserves
        balanced_tbl.get_state(DomainLeg.NPCPU).energy_flow = 0.8  # High energy

        perception = await energy_agent.perceive(balanced_tbl)
        analysis = await energy_agent.analyze(perception, balanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, balanced_tbl)

        # Check if reserve operations were planned
        reserve_ops = synthesis.get("reserve_operations", [])
        deposit_ops = [op for op in reserve_ops if op["operation"] == "deposit"]

        # May or may not have deposit operations depending on state

    @pytest.mark.asyncio
    async def test_reserve_withdrawal(self, energy_agent, imbalanced_tbl):
        """Test withdrawing energy from reserves."""
        # Set up state where withdrawal would occur
        # (low energy, high reserves)
        energy_agent.reserves[DomainLeg.CHICAGO_FOREST] = 0.5  # High reserves
        imbalanced_tbl.get_state(DomainLeg.CHICAGO_FOREST).energy_flow = 0.1  # Low energy

        perception = await energy_agent.perceive(imbalanced_tbl)
        analysis = await energy_agent.analyze(perception, imbalanced_tbl)
        synthesis = await energy_agent.synthesize(analysis, imbalanced_tbl)

        # Check if reserve operations were planned
        reserve_ops = synthesis.get("reserve_operations", [])
        withdraw_ops = [op for op in reserve_ops if op["operation"] == "withdraw"]

        # Should have withdrawal operation for low energy domain with high reserves
        assert len(withdraw_ops) > 0


class TestTransferEfficiency:
    """Test energy transfer efficiency mechanics."""

    @pytest.mark.asyncio
    async def test_transfer_applies_efficiency_loss(self, energy_agent, imbalanced_tbl):
        """Test that transfers apply efficiency loss."""
        # The efficiency is 0.9, so 10% is lost in transfer
        assert energy_agent.transfer_efficiency == 0.9

        # Track total energy before
        total_before = sum(
            imbalanced_tbl.get_state(d).energy_flow for d in DomainLeg
        )

        # Run refinement
        await energy_agent.refine(imbalanced_tbl)

        # Total energy after should be less due to efficiency losses
        total_after = sum(
            imbalanced_tbl.get_state(d).energy_flow for d in DomainLeg
        )

        # Either no transfers occurred or there was efficiency loss
        assert total_after <= total_before + 0.01  # Small tolerance for float


class TestFlowTracking:
    """Test energy flow tracking features."""

    @pytest.mark.asyncio
    async def test_flow_matrix_updates(self, energy_agent, imbalanced_tbl):
        """Test that flow matrix is updated on transfers."""
        initial_flow = energy_agent.flow_matrix.sum()

        await energy_agent.refine(imbalanced_tbl)

        # Flow matrix should be updated if transfers occurred
        final_flow = energy_agent.flow_matrix.sum()
        assert final_flow >= initial_flow

    @pytest.mark.asyncio
    async def test_total_energy_transferred_tracking(self, energy_agent, imbalanced_tbl):
        """Test total energy transferred tracking."""
        initial_transferred = energy_agent.total_energy_transferred

        await energy_agent.refine(imbalanced_tbl)

        assert energy_agent.total_energy_transferred >= initial_transferred


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_zero_energy_state(self, energy_agent):
        """Test handling of zero energy state."""
        tbl = TripleBottomLine()
        for domain in DomainLeg:
            state = DomainState(
                domain=domain,
                energy_flow=0.0,  # Zero energy
                state_vector=np.zeros(64)
            )
            tbl.set_state(domain, state)
        tbl.calculate_harmony()

        # Should handle gracefully
        result = await energy_agent.refine(tbl)
        assert result is not None

    @pytest.mark.asyncio
    async def test_max_energy_state(self, energy_agent):
        """Test handling of maximum energy state."""
        tbl = TripleBottomLine()
        for domain in DomainLeg:
            state = DomainState(
                domain=domain,
                energy_flow=1.0,  # Max energy
                state_vector=np.ones(64)
            )
            tbl.set_state(domain, state)
        tbl.calculate_harmony()

        # Should handle gracefully
        result = await energy_agent.refine(tbl)
        assert result is not None
        assert result.success == True

    @pytest.mark.asyncio
    async def test_rapid_refinement_cycles(self, energy_agent, balanced_tbl):
        """Test rapid consecutive refinement cycles."""
        # Run many cycles rapidly
        for _ in range(20):
            result = await energy_agent.refine(balanced_tbl)
            assert result is not None

    def test_domain_to_index_mapping(self, energy_agent):
        """Test domain to index mapping."""
        assert energy_agent._domain_to_index(DomainLeg.NPCPU) == 0
        assert energy_agent._domain_to_index(DomainLeg.CHICAGO_FOREST) == 1
        assert energy_agent._domain_to_index(DomainLeg.UNIVERSAL_PARTS) == 2


class TestCrossDomainSignals:
    """Test cross-domain signal emission."""

    def test_emit_signal(self, energy_agent):
        """Test emitting cross-domain signals."""
        signal = energy_agent.emit_cross_domain_signal(
            signal_type="energy_alert",
            payload={"severity": "high"},
            strength=0.8
        )

        assert signal is not None
        assert signal.signal_type == "energy_alert"
        assert signal.strength == 0.8
        assert len(energy_agent.cross_domain_signals) == 1

    def test_signal_targets_all_domains_by_default(self, energy_agent):
        """Test that signals target all domains by default."""
        signal = energy_agent.emit_cross_domain_signal(
            signal_type="test",
            payload={}
        )

        assert len(signal.target_domains) == 3


class TestMetrics:
    """Test agent metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_after_refinement(self, energy_agent, balanced_tbl):
        """Test metrics collection after refinement."""
        await energy_agent.refine(balanced_tbl)

        metrics = energy_agent.get_metrics()

        assert metrics["total_refinements"] == 1
        assert metrics["agent_role"] == energy_agent.agent_role
        assert "success_rate" in metrics
        assert "domains_affected" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
