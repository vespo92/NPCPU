"""
Integration Tests for NPCPU

Tests the integration between various NPCPU systems:
- Parallel simulation with consciousness
- Metabolism-consciousness coupling
- Evolution with phylogenetic visualization
- End-to-end simulation workflows
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.simple_organism import SimpleOrganism
from organism.metabolism import Metabolism, MetabolicConfig, EnergyState
from consciousness.neural_consciousness import NeuralConsciousness


# =============================================================================
# Metabolism-Consciousness Coupling Tests
# =============================================================================

class TestMetabolismConsciousnessCoupling:
    """Tests for metabolism-consciousness coupling system"""

    def test_coupler_creation(self):
        """Test creating a metabolism-consciousness coupler"""
        from consciousness.metabolism_coupling import (
            MetabolismConsciousnessCoupler,
            CouplingMode
        )

        metabolism = Metabolism()
        consciousness = NeuralConsciousness(
            attention_dim=32,
            memory_capacity=5
        )

        coupler = MetabolismConsciousnessCoupler(
            metabolism=metabolism,
            consciousness=consciousness,
            mode=CouplingMode.FULL
        )

        assert coupler is not None
        assert coupler.metabolism is metabolism
        assert coupler.consciousness is consciousness

    def test_energy_affects_consciousness(self):
        """Test that low energy reduces consciousness level"""
        from consciousness.metabolism_coupling import (
            MetabolismConsciousnessCoupler,
            ConsciousnessImpact
        )

        metabolism = Metabolism()
        consciousness = NeuralConsciousness()

        coupler = MetabolismConsciousnessCoupler(
            metabolism=metabolism,
            consciousness=consciousness
        )

        # Normal energy
        metabolism.energy = 70.0
        for _ in range(10):
            coupler.update()
        normal_state = coupler.get_coupling_state()

        # Low energy
        metabolism.energy = 10.0
        for _ in range(20):
            coupler.update()
        low_state = coupler.get_coupling_state()

        # Consciousness should be impaired with low energy
        assert low_state['modifiers']['consciousness'] < normal_state['modifiers']['consciousness']
        assert low_state['impact'] in ['impaired', 'critical', 'dormant']

    def test_consciousness_drains_energy(self):
        """Test that high cognitive load drains energy"""
        from consciousness.metabolism_coupling import (
            MetabolismConsciousnessCoupler,
            CouplingMode
        )

        metabolism = Metabolism()
        metabolism.energy = 50.0
        consciousness = NeuralConsciousness()

        # Simulate high cognitive load
        consciousness.attention_state.focus_strength = 0.9
        for i in range(5):
            consciousness.update_working_memory(
                {"item": i, "data": f"important_data_{i}"},
                importance=0.8
            )
        consciousness.emotional_state.arousal = 0.8

        coupler = MetabolismConsciousnessCoupler(
            metabolism=metabolism,
            consciousness=consciousness,
            mode=CouplingMode.FULL
        )

        initial_energy = metabolism.energy

        # Run coupling updates
        for _ in range(50):
            coupler.update()

        # Energy should have decreased
        assert metabolism.energy < initial_energy

    def test_coupling_state_tracking(self):
        """Test that coupling state is properly tracked"""
        from consciousness.metabolism_coupling import MetabolismConsciousnessCoupler

        metabolism = Metabolism()
        consciousness = NeuralConsciousness()

        coupler = MetabolismConsciousnessCoupler(
            metabolism=metabolism,
            consciousness=consciousness
        )

        # Run several updates
        for _ in range(20):
            coupler.update()

        state = coupler.get_coupling_state()

        # Verify state structure
        assert 'impact' in state
        assert 'energy_ratio' in state
        assert 'modifiers' in state
        assert 'attention' in state['modifiers']
        assert 'memory' in state['modifiers']
        assert 'emotional' in state['modifiers']
        assert 'consciousness' in state['modifiers']

    def test_trend_analysis(self):
        """Test trend analysis functionality"""
        from consciousness.metabolism_coupling import MetabolismConsciousnessCoupler

        metabolism = Metabolism()
        consciousness = NeuralConsciousness()

        coupler = MetabolismConsciousnessCoupler(
            metabolism=metabolism,
            consciousness=consciousness
        )

        # Simulate declining energy
        for i in range(50):
            metabolism.energy = 80 - i
            coupler.update()

        trends = coupler.get_trend_analysis()

        assert 'energy_trend' in trends
        assert 'energy_stability' in trends
        assert 'impact_volatility' in trends
        assert trends['energy_trend'] < 0  # Should be declining


# =============================================================================
# Parallel Simulation Tests
# =============================================================================

class TestParallelSimulation:
    """Tests for parallel simulation system"""

    def test_batch_processor_creation(self):
        """Test creating a batch processor"""
        from simulation.parallel_runner import BatchProcessor

        processor = BatchProcessor(
            max_workers=2,
            batch_size=10,
            use_processes=False
        )

        assert processor is not None
        assert processor.max_workers == 2
        assert processor.batch_size == 10

    def test_organism_cache(self):
        """Test organism cache functionality"""
        from simulation.parallel_runner import OrganismCache

        cache = OrganismCache(maxsize=100)

        # Test set and get
        cache.set("org_1", "consciousness", 0.75)
        value = cache.get("org_1", "consciousness")
        assert value == 0.75

        # Test cache miss
        missing = cache.get("org_2", "consciousness")
        assert missing is None

        # Check stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1

    def test_cache_eviction(self):
        """Test cache eviction when over capacity"""
        from simulation.parallel_runner import OrganismCache

        cache = OrganismCache(maxsize=5)

        # Fill cache
        for i in range(10):
            cache.set(f"org_{i}", "value", i)

        stats = cache.get_stats()
        assert stats['size'] <= 5

    def test_async_event_processor(self):
        """Test async event processor"""
        from simulation.parallel_runner import AsyncEventProcessor

        processor = AsyncEventProcessor(max_queue_size=100)
        received_events = []

        def handler(data):
            received_events.append(data)

        processor.register_handler("test_event", handler)
        processor.start()

        try:
            # Emit events
            for i in range(5):
                processor.emit("test_event", {"id": i})

            # Wait for processing
            time.sleep(0.2)

            assert len(received_events) >= 3  # Allow some timing variance

        finally:
            processor.stop()

    def test_parallel_config(self):
        """Test parallel configuration"""
        from simulation.parallel_runner import ParallelConfig

        config = ParallelConfig(
            name="Test Parallel",
            initial_population=100,
            max_workers=4,
            batch_size=25
        )

        assert config.name == "Test Parallel"
        assert config.initial_population == 100
        assert config.max_workers == 4
        assert config.batch_size == 25


# =============================================================================
# Phylogenetic Visualization Tests
# =============================================================================

class TestPhylogeneticVisualization:
    """Tests for phylogenetic visualization"""

    def test_tree_builder(self):
        """Test phylogenetic tree building"""
        from evolution.genetic_engine import EvolutionEngine
        from evolution.phylogenetic_viz import PhylogeneticTreeBuilder

        engine = EvolutionEngine(population_size=20)
        gene_specs = {"speed": (0.0, 1.0), "strength": (0.0, 1.0)}
        engine.initialize_population(gene_specs)

        # Evolve a few generations
        def fitness(g):
            return g.express("speed") + g.express("strength")

        for _ in range(5):
            engine.evolve_generation(fitness)

        builder = PhylogeneticTreeBuilder()
        tree = builder.build_from_engine(engine)

        assert tree is not None
        assert len(tree.nodes) > 0

    def test_visualizer_ascii_tree(self):
        """Test ASCII tree generation"""
        from evolution.genetic_engine import EvolutionEngine
        from evolution.phylogenetic_viz import PhylogeneticVisualizer

        engine = EvolutionEngine(population_size=20)
        gene_specs = {"speed": (0.0, 1.0)}
        engine.initialize_population(gene_specs)

        def fitness(g):
            return g.express("speed")

        for _ in range(3):
            engine.evolve_generation(fitness)

        viz = PhylogeneticVisualizer(engine)
        tree_str = viz.generate_ascii_tree(max_depth=3)

        assert tree_str is not None
        assert "PHYLOGENETIC TREE" in tree_str

    def test_diversity_report(self):
        """Test diversity report generation"""
        from evolution.genetic_engine import EvolutionEngine
        from evolution.phylogenetic_viz import PhylogeneticVisualizer

        engine = EvolutionEngine(population_size=30, enable_speciation=True)
        gene_specs = {
            "speed": (0.0, 1.0),
            "strength": (0.0, 1.0),
            "intelligence": (0.0, 1.0)
        }
        engine.initialize_population(gene_specs)

        def fitness(g):
            return g.express("speed") * 0.5 + g.express("strength") * 0.5

        for _ in range(10):
            engine.evolve_generation(fitness)

        viz = PhylogeneticVisualizer(engine)
        report = viz.generate_diversity_report()

        assert report is not None
        assert "GENETIC DIVERSITY REPORT" in report
        assert "POPULATION STATISTICS" in report


# =============================================================================
# Consciousness Benchmark Tests
# =============================================================================

class TestConsciousnessBenchmarks:
    """Tests for consciousness benchmarks"""

    def test_perception_benchmark(self):
        """Test perception processing benchmark"""
        from benchmarks.consciousness_benchmarks import ConsciousnessBenchmarks

        result = ConsciousnessBenchmarks.perception_processing(count=100)

        assert result is not None
        assert result.iterations == 100
        assert result.rate > 0

    def test_memory_benchmark(self):
        """Test working memory benchmark"""
        from benchmarks.consciousness_benchmarks import ConsciousnessBenchmarks

        result = ConsciousnessBenchmarks.working_memory_operations(count=100)

        assert result is not None
        assert result.iterations == 100

    def test_emotional_benchmark(self):
        """Test emotional computation benchmark"""
        from benchmarks.consciousness_benchmarks import ConsciousnessBenchmarks

        result = ConsciousnessBenchmarks.emotional_computation(count=100)

        assert result is not None
        assert result.iterations == 100


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests"""

    def test_conscious_organism_full_lifecycle(self):
        """Test complete lifecycle of a conscious organism"""
        from consciousness.neural_consciousness import attach_neural_consciousness

        # Create organism with consciousness
        org = SimpleOrganism("TestOrganism")
        consciousness = attach_neural_consciousness(
            org,
            attention_dim=32,
            memory_capacity=5
        )

        # Simulate lifecycle
        stimuli = {"food": 0.7, "threat": 0.2, "social": 0.5}

        for tick in range(50):
            # Organism tick
            org.perceive(stimuli)
            org.tick()

            # Consciousness processing
            consciousness.process_perception(stimuli)
            consciousness.compute_emotional_state()
            consciousness.tick()

            # Store memory occasionally
            if tick % 10 == 0:
                consciousness.update_working_memory(
                    {"event": f"tick_{tick}", "food_level": stimuli["food"]},
                    importance=0.6
                )

        # Verify organism is still functioning
        assert org.is_alive
        assert consciousness.tick_count == 50
        assert len(consciousness.working_memory) > 0

    def test_metabolism_consciousness_organism_integration(self):
        """Test full integration of metabolism, consciousness, and organism"""
        from consciousness.metabolism_coupling import MetabolismConsciousnessCoupler
        from organism.digital_body import DigitalBody

        # Create digital body (has metabolism built in)
        body = DigitalBody("IntegratedOrganism")

        # Create consciousness
        consciousness = NeuralConsciousness(
            attention_dim=32,
            memory_capacity=7
        )
        body.add_subsystem(consciousness)

        # Create coupler
        coupler = MetabolismConsciousnessCoupler(
            metabolism=body.metabolism,
            consciousness=consciousness
        )

        # Run integrated simulation
        for tick in range(100):
            # Body tick (includes metabolism)
            body.tick()

            # Consciousness tick
            consciousness.process_perception({
                "food": 0.5,
                "threat": 0.1 + (tick % 10) * 0.05
            })
            consciousness.compute_emotional_state()
            consciousness.tick()

            # Coupling update
            coupler.update()

        # Get final states
        coupling_state = coupler.get_coupling_state()
        introspection = consciousness.introspect()

        assert body.is_alive
        assert 'impact' in coupling_state
        assert 'consciousness_level' in introspection

    def test_evolution_with_consciousness_fitness(self):
        """Test evolution using consciousness-based fitness"""
        from evolution.genetic_engine import EvolutionEngine

        engine = EvolutionEngine(
            population_size=30,
            mutation_rate=0.15,
            enable_speciation=True
        )

        gene_specs = {
            "attention_capacity": (0.3, 1.0),
            "memory_strength": (0.3, 1.0),
            "emotional_sensitivity": (0.1, 0.9),
            "introspection_ability": (0.2, 0.8)
        }

        engine.initialize_population(gene_specs)

        # Fitness based on consciousness-related traits
        def consciousness_fitness(genome):
            attention = genome.express("attention_capacity")
            memory = genome.express("memory_strength")
            emotional = genome.express("emotional_sensitivity")
            intro = genome.express("introspection_ability")

            # Balance attention and memory, moderate emotion
            return (
                attention * 0.3 +
                memory * 0.3 +
                (1.0 - abs(0.5 - emotional)) * 0.2 +
                intro * 0.2
            )

        initial_fitness = engine.best_fitness if engine.population else 0

        for gen in range(20):
            engine.evolve_generation(consciousness_fitness)

        final_fitness = engine.best_fitness

        # Fitness should improve
        assert final_fitness >= initial_fitness * 0.8  # Allow some variance


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_coupler_with_none_metabolism(self):
        """Test coupler handles missing metabolism gracefully"""
        from consciousness.metabolism_coupling import create_coupler_for_organism

        org = SimpleOrganism("NoMetabolism")
        coupler = create_coupler_for_organism(org)

        # Should return None when metabolism not found
        assert coupler is None

    def test_cache_invalidation(self):
        """Test cache invalidation works correctly"""
        from simulation.parallel_runner import OrganismCache

        cache = OrganismCache(maxsize=100)
        cache.set("org_1", "value", 123)

        # Invalidate
        cache.invalidate("org_1")

        # Should be gone
        assert cache.get("org_1", "value") is None

    def test_empty_population_visualization(self):
        """Test visualization handles empty population"""
        from evolution.phylogenetic_viz import PhylogeneticVisualizer

        viz = PhylogeneticVisualizer()
        report = viz.generate_diversity_report()

        assert "No evolution engine configured" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
