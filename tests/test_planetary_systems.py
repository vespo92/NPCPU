"""
Comprehensive Test Suite for GAIA-7 Planetary Systems

Tests all planetary-scale consciousness and environmental systems:
- Biosphere simulation
- Gaia consciousness emergence
- Resource cycles
- Climate-consciousness feedback
- Tipping point detection
- Migration patterns
- Planetary memory
- Extinction events
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from planetary.biosphere import (
    Biosphere, BiosphereConfig, BiomeRegion, BiomeType,
    ClimateZone, AtmosphericComposition, OceanState
)
from planetary.gaia_consciousness import (
    GaiaConsciousness, GaiaConfig, GaiaAwarenessLevel,
    PlanetaryIntention, PlanetaryPerception
)
from planetary.resource_cycles import (
    CarbonCycle, NitrogenCycle, WaterCycle, PlanetaryCycles,
    CycleConfig, Reservoir, Flux
)
from planetary.climate_feedback import (
    ClimateFeedbackSystem, FeedbackConfig, FeedbackLoop,
    FeedbackType, FeedbackDirection
)
from planetary.tipping_points import (
    TippingPointDetector, TippingPointConfig, TippingElement,
    TippingElementType, TippingState, WarningSignal
)
from planetary.migration_patterns import (
    MigrationSystem, MigrationConfig, MigrationRegion,
    MigrationDriver, MigrationPattern
)
from planetary.planetary_memory import (
    PlanetaryMemory, MemoryConfig, PlanetaryMemoryItem,
    MemoryType, MemoryPersistence, GeologicalEra
)
from planetary.extinction_events import (
    ExtinctionSystem, ExtinctionConfig, ExtinctionEvent,
    ExtinctionCause, ExtinctionSeverity, Species
)


# ============================================================================
# Biosphere Tests
# ============================================================================

class TestBiosphere:
    """Tests for Biosphere simulation"""

    def test_biosphere_initialization(self):
        """Test biosphere can be initialized"""
        biosphere = Biosphere()
        assert biosphere is not None
        assert biosphere.tick_count == 0

    def test_biosphere_default_biomes(self):
        """Test default biome creation"""
        biosphere = Biosphere()
        biosphere.initialize_default_biomes()

        assert len(biosphere.biomes) > 0
        assert biosphere.global_temperature == 15.0

    def test_biosphere_tick(self):
        """Test biosphere simulation tick"""
        biosphere = Biosphere()
        biosphere.initialize_default_biomes()

        initial_tick = biosphere.tick_count
        biosphere.tick()

        assert biosphere.tick_count == initial_tick + 1

    def test_biosphere_metrics(self):
        """Test global metrics calculation"""
        biosphere = Biosphere()
        biosphere.initialize_default_biomes()

        for _ in range(10):
            biosphere.tick()

        metrics = biosphere.get_global_metrics()

        assert "global_temperature" in metrics
        assert "atmosphere" in metrics
        assert "ocean" in metrics
        assert "biodiversity" in metrics

    def test_atmospheric_composition(self):
        """Test atmospheric composition"""
        atmosphere = AtmosphericComposition()

        assert atmosphere.co2 == 400.0
        assert atmosphere.o2 > 20.0
        assert atmosphere.get_breathability() > 0

    def test_ocean_state(self):
        """Test ocean state"""
        ocean = OceanState()

        assert ocean.temperature == 17.0
        assert ocean.get_health() > 0


# ============================================================================
# Gaia Consciousness Tests
# ============================================================================

class TestGaiaConsciousness:
    """Tests for Gaia consciousness emergence"""

    def test_gaia_initialization(self):
        """Test Gaia consciousness initialization"""
        gaia = GaiaConsciousness()

        assert gaia is not None
        assert gaia.awareness_level == GaiaAwarenessLevel.DORMANT
        assert gaia.awareness_score == 0.0

    def test_gaia_consciousness_cycle(self):
        """Test consciousness cycle execution"""
        gaia = GaiaConsciousness()

        initial_tick = gaia.tick_count
        gaia.conscious_cycle()

        assert gaia.tick_count == initial_tick + 1

    def test_organism_registration(self):
        """Test organism consciousness registration"""
        gaia = GaiaConsciousness()

        gaia.register_organism_consciousness("org_1", 0.5)
        gaia.register_organism_consciousness("org_2", 0.7)

        assert len(gaia.organism_consciousnesses) == 2

    def test_awareness_emergence(self):
        """Test awareness emergence from organisms"""
        gaia = GaiaConsciousness()

        # Register many organisms
        for i in range(100):
            gaia.register_organism_consciousness(f"org_{i}", 0.6)

        # Run consciousness cycles
        for _ in range(50):
            gaia.conscious_cycle()

        # Should show some emergence
        assert gaia.awareness_score > 0

    def test_gaia_status(self):
        """Test status reporting"""
        gaia = GaiaConsciousness()
        status = gaia.get_status()

        assert "awareness_level" in status
        assert "awareness_score" in status
        assert "current_intention" in status


# ============================================================================
# Resource Cycles Tests
# ============================================================================

class TestResourceCycles:
    """Tests for biogeochemical cycles"""

    def test_carbon_cycle_initialization(self):
        """Test carbon cycle initialization"""
        cycle = CarbonCycle()
        cycle.initialize()

        assert len(cycle.reservoirs) > 0
        assert len(cycle.fluxes) > 0

    def test_carbon_cycle_tick(self):
        """Test carbon cycle simulation"""
        cycle = CarbonCycle()
        cycle.initialize()

        initial_co2 = cycle.get_atmospheric_co2()
        cycle.tick()

        # CO2 should still be reasonable
        assert 100 < cycle.get_atmospheric_co2() < 1000

    def test_anthropogenic_emissions(self):
        """Test anthropogenic carbon emissions"""
        cycle = CarbonCycle()
        cycle.initialize()

        initial_co2 = cycle.get_atmospheric_co2()
        cycle.emit_anthropogenic(10)  # 10 Gt C
        cycle.tick()

        assert cycle.get_atmospheric_co2() > initial_co2

    def test_nitrogen_cycle(self):
        """Test nitrogen cycle"""
        cycle = NitrogenCycle()
        cycle.initialize()

        assert len(cycle.reservoirs) > 0
        cycle.tick()

    def test_water_cycle(self):
        """Test water cycle"""
        cycle = WaterCycle()
        cycle.initialize()

        assert len(cycle.reservoirs) > 0
        cycle.tick()

    def test_integrated_cycles(self):
        """Test integrated planetary cycles"""
        cycles = PlanetaryCycles()
        cycles.initialize()

        for _ in range(10):
            cycles.tick()

        status = cycles.get_global_status()
        assert "carbon" in status
        assert "nitrogen" in status
        assert "water" in status


# ============================================================================
# Climate Feedback Tests
# ============================================================================

class TestClimateFeedback:
    """Tests for climate-consciousness feedback"""

    def test_feedback_initialization(self):
        """Test feedback system initialization"""
        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()

        assert len(feedback.feedback_loops) > 0

    def test_feedback_processing(self):
        """Test feedback processing"""
        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()

        feedback.climate_state.temperature_anomaly = 2.0
        feedback.consciousness_state.awareness_level = 0.5

        feedback.process_tick()

        assert feedback.tick_count == 1

    def test_stressor_detection(self):
        """Test climate stressor detection"""
        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()

        feedback.climate_state.temperature_anomaly = 3.0
        feedback.climate_state.co2_level = 500

        feedback.process_tick()

        assert len(feedback.active_stressors) > 0

    def test_feedback_analysis(self):
        """Test feedback analysis"""
        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()

        for _ in range(10):
            feedback.process_tick()

        analysis = feedback.get_feedback_analysis()
        assert "climate_state" in analysis
        assert "consciousness_state" in analysis


# ============================================================================
# Tipping Points Tests
# ============================================================================

class TestTippingPoints:
    """Tests for tipping point detection"""

    def test_tipping_detector_initialization(self):
        """Test tipping point detector initialization"""
        detector = TippingPointDetector()
        detector.initialize_elements()

        assert len(detector.elements) > 0

    def test_element_update(self):
        """Test element value updates"""
        detector = TippingPointDetector()
        detector.initialize_elements()

        element = list(detector.elements.values())[0]
        initial_value = element.current_value

        detector.update_element(element.id, initial_value + 0.1)

        # Value should change (with inertia)
        assert element.current_value != initial_value

    def test_warning_detection(self):
        """Test early warning signal detection"""
        detector = TippingPointDetector()
        detector.initialize_elements()

        element = list(detector.elements.values())[0]

        # Push toward threshold
        for i in range(60):
            detector.update_element(element.id, element.current_value + 0.01)
            detector.analyze()

        # Should detect approach
        assert element.state != TippingState.STABLE or len(element.warning_signals) > 0

    def test_global_stability(self):
        """Test global stability calculation"""
        detector = TippingPointDetector()
        detector.initialize_elements()

        detector.analyze()

        assert 0 <= detector.global_stability <= 1


# ============================================================================
# Migration Patterns Tests
# ============================================================================

class TestMigrationPatterns:
    """Tests for migration patterns"""

    def test_migration_initialization(self):
        """Test migration system initialization"""
        migration = MigrationSystem()
        migration.create_region_network(5)

        assert len(migration.regions) == 5

    def test_region_connectivity(self):
        """Test region connections"""
        migration = MigrationSystem()
        migration.create_region_network(5)

        # At least some regions should be connected
        connected_count = sum(
            len(r.connected_regions) for r in migration.regions.values()
        )
        assert connected_count > 0

    def test_migration_tick(self):
        """Test migration simulation tick"""
        migration = MigrationSystem()
        migration.create_region_network(5)

        migration.simulate_tick()

        assert migration.tick_count == 1

    def test_population_distribution(self):
        """Test population distribution tracking"""
        migration = MigrationSystem()
        migration.create_region_network(5)

        for _ in range(10):
            migration.simulate_tick()

        dist = migration.get_global_distribution()
        assert "total_population" in dist
        assert "distribution" in dist


# ============================================================================
# Planetary Memory Tests
# ============================================================================

class TestPlanetaryMemory:
    """Tests for planetary memory"""

    def test_memory_initialization(self):
        """Test memory system initialization"""
        memory = PlanetaryMemory()

        assert memory is not None
        assert memory.current_epoch is not None

    def test_memory_storage(self):
        """Test memory storage"""
        memory = PlanetaryMemory()

        memory_id = memory.store(
            content={"event": "test"},
            memory_type=MemoryType.EVENT,
            importance=0.5
        )

        assert memory_id is not None
        assert memory.total_memories_stored == 1

    def test_memory_recall(self):
        """Test memory recall"""
        memory = PlanetaryMemory()

        memory_id = memory.store(
            content={"event": "test"},
            memory_type=MemoryType.EVENT,
            importance=0.5
        )

        recalled = memory.recall(memory_id)
        assert recalled is not None
        assert recalled.content["event"] == "test"

    def test_memory_query(self):
        """Test memory querying"""
        memory = PlanetaryMemory()

        # Store multiple memories
        for i in range(10):
            memory.store(
                content={"index": i},
                memory_type=MemoryType.EVENT,
                importance=0.5 + i * 0.05,
                tags=["test"]
            )

        results = memory.query(
            memory_type=MemoryType.EVENT,
            min_importance=0.6
        )

        assert len(results) > 0

    def test_epoch_tracking(self):
        """Test epoch tracking"""
        config = MemoryConfig(epoch_length=100)
        memory = PlanetaryMemory(config)

        for _ in range(150):
            memory.tick()

        assert len(memory.epochs) > 0


# ============================================================================
# Extinction Events Tests
# ============================================================================

class TestExtinctionEvents:
    """Tests for extinction event modeling"""

    def test_extinction_initialization(self):
        """Test extinction system initialization"""
        extinction = ExtinctionSystem()
        extinction.create_species(20)

        assert len(extinction.species) == 20

    def test_species_creation(self):
        """Test species creation"""
        extinction = ExtinctionSystem()
        extinction.create_species(10)

        # All species should start alive
        living = [s for s in extinction.species.values() if not s.extinct]
        assert len(living) == 10

    def test_extinction_trigger(self):
        """Test extinction event triggering"""
        extinction = ExtinctionSystem()
        extinction.create_species(50)

        event = extinction.trigger_extinction(
            cause=ExtinctionCause.CLIMATE_CHANGE,
            severity=ExtinctionSeverity.MODERATE
        )

        assert event is not None
        assert event.active

    def test_extinction_simulation(self):
        """Test extinction event simulation"""
        extinction = ExtinctionSystem()
        extinction.create_species(50)

        event = extinction.trigger_extinction(
            cause=ExtinctionCause.CLIMATE_CHANGE,
            severity=ExtinctionSeverity.MAJOR,
            duration=50
        )

        initial_living = sum(1 for s in extinction.species.values() if not s.extinct)

        for _ in range(60):
            extinction.tick()

        final_living = sum(1 for s in extinction.species.values() if not s.extinct)

        # Some species should be lost
        assert final_living < initial_living

    def test_extinction_risk(self):
        """Test extinction risk calculation"""
        extinction = ExtinctionSystem()
        extinction.create_species(10)

        species_id = list(extinction.species.keys())[0]
        risk = extinction.get_extinction_risk(species_id)

        assert 0 <= risk <= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestPlanetaryIntegration:
    """Integration tests for planetary systems"""

    def test_biosphere_gaia_integration(self):
        """Test biosphere-Gaia consciousness integration"""
        biosphere = Biosphere()
        biosphere.initialize_default_biomes()

        gaia = GaiaConsciousness()
        gaia.connect_biosphere(biosphere)

        for _ in range(10):
            biosphere.tick()
            gaia.conscious_cycle()

        # Gaia should perceive biosphere
        status = gaia.get_status()
        assert status is not None

    def test_cycles_feedback_integration(self):
        """Test resource cycles and feedback integration"""
        cycles = PlanetaryCycles()
        cycles.initialize()

        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()
        feedback.set_cycles(cycles)

        for _ in range(10):
            cycles.tick()
            feedback.process_tick()

        analysis = feedback.get_feedback_analysis()
        assert analysis is not None

    def test_full_system_simulation(self):
        """Test full planetary system simulation"""
        # Initialize all systems
        biosphere = Biosphere()
        biosphere.initialize_default_biomes()

        gaia = GaiaConsciousness()
        gaia.connect_biosphere(biosphere)

        cycles = PlanetaryCycles()
        cycles.initialize()

        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()

        detector = TippingPointDetector()
        detector.initialize_elements()

        memory = PlanetaryMemory()

        # Run simulation
        for i in range(50):
            biosphere.tick()
            gaia.conscious_cycle()
            cycles.tick()
            feedback.process_tick()
            detector.analyze()
            memory.tick()

            # Store periodic state in memory
            if i % 10 == 0:
                memory.store(
                    content={
                        "tick": i,
                        "biodiversity": biosphere.biodiversity_global,
                        "awareness": gaia.awareness_score
                    },
                    memory_type=MemoryType.STATE,
                    importance=0.5
                )

        # Verify all systems ran
        assert biosphere.tick_count == 50
        assert gaia.tick_count == 50
        assert cycles.tick_count == 50
        assert feedback.tick_count == 50


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
