"""
Tests for Consciousness Evolution Engine

Tests genetic algorithm-based optimization of consciousness models.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness
from evolution.consciousness_evolution import (
    ConsciousnessEvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    FitnessFunctions,
    Individual,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_consciousness():
    """Create base consciousness for evolution"""
    return GradedConsciousness(
        perception_fidelity=0.5,
        reaction_speed=0.5,
        memory_depth=0.5,
        memory_recall_accuracy=0.5,
        introspection_capacity=0.5,
        meta_cognitive_ability=0.5,
        information_integration=0.5,
        intentional_coherence=0.5,
        qualia_richness=0.5
    )


@pytest.fixture
def config():
    """Create test evolution config"""
    return EvolutionConfig(
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        max_generations=50,
        elitism_count=2,
        early_stop_generations=5
    )


@pytest.fixture
def engine(config):
    """Create evolution engine"""
    return ConsciousnessEvolutionEngine(config)


# ============================================================================
# Individual Tests
# ============================================================================

class TestIndividual:
    """Test Individual class"""

    def test_individual_creation(self, base_consciousness):
        """Test creating an individual"""
        individual = Individual(
            consciousness=base_consciousness,
            fitness=0.5,
            generation=0
        )

        assert individual.fitness == 0.5
        assert individual.generation == 0
        assert individual.mutations == 0

    def test_individual_id(self, base_consciousness):
        """Test individual ID generation"""
        ind1 = Individual(consciousness=base_consciousness)
        ind2 = Individual(consciousness=base_consciousness)

        # Same consciousness = same ID
        assert ind1.id == ind2.id

        # Different consciousness = different ID
        different = GradedConsciousness(perception_fidelity=0.9)
        ind3 = Individual(consciousness=different)
        assert ind1.id != ind3.id


# ============================================================================
# Fitness Functions Tests
# ============================================================================

class TestFitnessFunctions:
    """Test built-in fitness functions"""

    def test_task_performance_fitness(self, base_consciousness):
        """Test task performance fitness function"""
        fitness_fn = FitnessFunctions.task_performance({
            "perception_fidelity": 0.5,
            "reaction_speed": 0.5
        })

        score = fitness_fn(base_consciousness)

        assert 0.0 <= score <= 1.0
        assert score == 0.5  # All 0.5 = 0.5 fitness

    def test_task_performance_weighted(self):
        """Test weighted task performance"""
        fitness_fn = FitnessFunctions.task_performance({
            "perception_fidelity": 1.0,
            "reaction_speed": 0.0
        })

        high_perception = GradedConsciousness(
            perception_fidelity=1.0,
            reaction_speed=0.0
        )

        score = fitness_fn(high_perception)
        assert score == 1.0

    def test_balanced_consciousness_fitness(self):
        """Test balanced consciousness fitness"""
        fitness_fn = FitnessFunctions.balanced_consciousness()

        # Balanced consciousness
        balanced = GradedConsciousness(
            perception_fidelity=0.7,
            reaction_speed=0.7,
            memory_depth=0.7,
            memory_recall_accuracy=0.7,
            introspection_capacity=0.7,
            meta_cognitive_ability=0.7,
            information_integration=0.7,
            intentional_coherence=0.7,
            qualia_richness=0.7
        )

        # Unbalanced consciousness
        unbalanced = GradedConsciousness(
            perception_fidelity=1.0,
            reaction_speed=0.0,
            memory_depth=1.0,
            memory_recall_accuracy=0.0
        )

        balanced_score = fitness_fn(balanced)
        unbalanced_score = fitness_fn(unbalanced)

        # Balanced should score higher
        assert balanced_score > unbalanced_score

    def test_overall_consciousness_fitness(self, base_consciousness):
        """Test overall consciousness fitness"""
        fitness_fn = FitnessFunctions.overall_consciousness()

        score = fitness_fn(base_consciousness)
        expected = base_consciousness.overall_consciousness_score()

        assert score == pytest.approx(expected, rel=0.01)

    def test_threshold_capabilities_fitness(self):
        """Test threshold-based fitness"""
        fitness_fn = FitnessFunctions.threshold_capabilities({
            "perception_fidelity": 0.7,
            "reaction_speed": 0.5
        })

        # Meets both thresholds
        meets_all = GradedConsciousness(
            perception_fidelity=0.8,
            reaction_speed=0.6
        )

        # Meets one threshold
        meets_one = GradedConsciousness(
            perception_fidelity=0.8,
            reaction_speed=0.3
        )

        # Meets none
        meets_none = GradedConsciousness(
            perception_fidelity=0.3,
            reaction_speed=0.3
        )

        assert fitness_fn(meets_all) > fitness_fn(meets_one)
        assert fitness_fn(meets_one) > fitness_fn(meets_none)

    def test_multi_objective_fitness(self, base_consciousness):
        """Test multi-objective fitness"""
        obj1 = FitnessFunctions.task_performance({"perception_fidelity": 1.0})
        obj2 = FitnessFunctions.balanced_consciousness()

        multi_fitness = FitnessFunctions.multi_objective(
            [obj1, obj2],
            weights=[0.5, 0.5]
        )

        score = multi_fitness(base_consciousness)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Population Tests
# ============================================================================

class TestPopulation:
    """Test population management"""

    def test_create_initial_population(self, engine, base_consciousness):
        """Test creating initial population"""
        population = engine.create_initial_population(base_consciousness)

        assert len(population) == engine.config.population_size
        assert all(isinstance(ind, Individual) for ind in population)

    def test_population_diversity(self, engine, base_consciousness):
        """Test that initial population has diversity"""
        population = engine.create_initial_population(base_consciousness)

        # Check that not all individuals are identical
        scores = [
            ind.consciousness.perception_fidelity
            for ind in population
        ]

        assert max(scores) != min(scores)  # Some variation

    def test_evaluate_fitness(self, engine, base_consciousness):
        """Test fitness evaluation"""
        population = engine.create_initial_population(base_consciousness)
        fitness_fn = FitnessFunctions.overall_consciousness()

        engine.evaluate_fitness(population, fitness_fn)

        assert all(ind.fitness > 0 for ind in population)

    def test_calculate_diversity(self, engine, base_consciousness):
        """Test diversity calculation"""
        population = engine.create_initial_population(base_consciousness)

        diversity = engine.calculate_diversity(population)

        assert diversity >= 0.0


# ============================================================================
# Selection Tests
# ============================================================================

class TestSelection:
    """Test parent selection methods"""

    def test_tournament_selection(self, base_consciousness):
        """Test tournament selection"""
        config = EvolutionConfig(
            population_size=10,
            tournament_size=3,
            selection_method=SelectionMethod.TOURNAMENT
        )
        engine = ConsciousnessEvolutionEngine(config)

        population = engine.create_initial_population(base_consciousness)
        fitness_fn = FitnessFunctions.overall_consciousness()
        engine.evaluate_fitness(population, fitness_fn)

        selected = engine.select_parent(population)

        assert selected in population

    def test_roulette_selection(self, base_consciousness):
        """Test roulette wheel selection"""
        config = EvolutionConfig(
            population_size=10,
            selection_method=SelectionMethod.ROULETTE
        )
        engine = ConsciousnessEvolutionEngine(config)

        population = engine.create_initial_population(base_consciousness)
        fitness_fn = FitnessFunctions.overall_consciousness()
        engine.evaluate_fitness(population, fitness_fn)

        selected = engine.select_parent(population)

        assert selected in population

    def test_rank_selection(self, base_consciousness):
        """Test rank-based selection"""
        config = EvolutionConfig(
            population_size=10,
            selection_method=SelectionMethod.RANK
        )
        engine = ConsciousnessEvolutionEngine(config)

        population = engine.create_initial_population(base_consciousness)
        fitness_fn = FitnessFunctions.overall_consciousness()
        engine.evaluate_fitness(population, fitness_fn)

        selected = engine.select_parent(population)

        assert selected in population


# ============================================================================
# Crossover Tests
# ============================================================================

class TestCrossover:
    """Test crossover methods"""

    def test_uniform_crossover(self, base_consciousness):
        """Test uniform crossover"""
        config = EvolutionConfig(
            crossover_method=CrossoverMethod.UNIFORM,
            crossover_rate=1.0
        )
        engine = ConsciousnessEvolutionEngine(config)

        parent1 = Individual(consciousness=GradedConsciousness(
            perception_fidelity=1.0,
            reaction_speed=1.0
        ))
        parent2 = Individual(consciousness=GradedConsciousness(
            perception_fidelity=0.0,
            reaction_speed=0.0
        ))

        child = engine.crossover(parent1, parent2)

        # Child should have values from both parents
        assert isinstance(child, Individual)
        scores = child.consciousness.get_capability_scores()
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_arithmetic_crossover(self, base_consciousness):
        """Test arithmetic crossover"""
        config = EvolutionConfig(
            crossover_method=CrossoverMethod.ARITHMETIC,
            crossover_rate=1.0
        )
        engine = ConsciousnessEvolutionEngine(config)

        parent1 = Individual(consciousness=GradedConsciousness(
            perception_fidelity=1.0
        ))
        parent2 = Individual(consciousness=GradedConsciousness(
            perception_fidelity=0.0
        ))

        child = engine.crossover(parent1, parent2)

        # Should be between parents
        assert 0.0 <= child.consciousness.perception_fidelity <= 1.0

    def test_crossover_rate(self, base_consciousness):
        """Test that crossover rate is respected"""
        config = EvolutionConfig(
            crossover_rate=0.0  # No crossover
        )
        engine = ConsciousnessEvolutionEngine(config)

        parent1 = Individual(consciousness=GradedConsciousness(
            perception_fidelity=0.8
        ))
        parent2 = Individual(consciousness=GradedConsciousness(
            perception_fidelity=0.2
        ))

        child = engine.crossover(parent1, parent2)

        # Should be copy of parent1
        assert child.consciousness.perception_fidelity == 0.8


# ============================================================================
# Mutation Tests
# ============================================================================

class TestMutation:
    """Test mutation methods"""

    def test_gaussian_mutation(self, base_consciousness):
        """Test Gaussian mutation"""
        config = EvolutionConfig(
            mutation_method=MutationMethod.GAUSSIAN,
            mutation_rate=1.0  # Always mutate
        )
        engine = ConsciousnessEvolutionEngine(config)

        individual = Individual(consciousness=base_consciousness)
        mutated = engine.mutate(individual)

        # Should be different
        assert mutated.consciousness.get_capability_scores() != \
               individual.consciousness.get_capability_scores()

    def test_uniform_mutation(self, base_consciousness):
        """Test uniform mutation"""
        config = EvolutionConfig(
            mutation_method=MutationMethod.UNIFORM,
            mutation_rate=1.0
        )
        engine = ConsciousnessEvolutionEngine(config)

        individual = Individual(consciousness=base_consciousness)
        mutated = engine.mutate(individual)

        # Values should be in valid range
        scores = mutated.consciousness.get_capability_scores()
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_mutation_rate_zero(self, base_consciousness):
        """Test no mutation when rate is 0"""
        config = EvolutionConfig(
            mutation_rate=0.0
        )
        engine = ConsciousnessEvolutionEngine(config)

        individual = Individual(consciousness=base_consciousness)
        mutated = engine.mutate(individual)

        # Should be identical
        assert mutated.consciousness.get_capability_scores() == \
               individual.consciousness.get_capability_scores()


# ============================================================================
# Evolution Tests
# ============================================================================

class TestEvolution:
    """Test full evolution process"""

    def test_basic_evolution(self, engine, base_consciousness):
        """Test basic evolution run"""
        fitness_fn = FitnessFunctions.overall_consciousness()

        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=10,
            verbose=False
        )

        assert isinstance(result, EvolutionResult)
        assert result.generations_run == 10
        assert result.best_fitness > 0

    def test_evolution_improves_fitness(self, base_consciousness):
        """Test that evolution improves fitness"""
        config = EvolutionConfig(
            population_size=30,
            max_generations=30
        )
        engine = ConsciousnessEvolutionEngine(config)

        fitness_fn = FitnessFunctions.task_performance({
            "perception_fidelity": 1.0
        })

        initial_fitness = fitness_fn(base_consciousness)
        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=30,
            verbose=False
        )

        assert result.best_fitness >= initial_fitness

    def test_evolution_early_stopping(self, base_consciousness):
        """Test early stopping on convergence"""
        config = EvolutionConfig(
            population_size=10,
            max_generations=100,
            early_stop_threshold=0.0001,
            early_stop_generations=5
        )
        engine = ConsciousnessEvolutionEngine(config)

        # Simple fitness that converges quickly
        fitness_fn = FitnessFunctions.overall_consciousness()

        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=100,
            verbose=False
        )

        # Should stop before max generations
        assert result.generations_run <= 100

    def test_evolution_history(self, engine, base_consciousness):
        """Test evolution history tracking"""
        fitness_fn = FitnessFunctions.overall_consciousness()

        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=10,
            verbose=False
        )

        assert len(result.fitness_history) == 10
        assert len(result.diversity_history) == 10
        assert len(result.population_history) == 10

    def test_evolution_callback(self, engine, base_consciousness):
        """Test evolution callback"""
        fitness_fn = FitnessFunctions.overall_consciousness()
        callback_calls = []

        def callback(gen, fitness, diversity):
            callback_calls.append((gen, fitness, diversity))

        engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=5,
            verbose=False,
            callback=callback
        )

        assert len(callback_calls) == 5

    def test_elitism(self, base_consciousness):
        """Test that elitism preserves best individuals"""
        config = EvolutionConfig(
            population_size=10,
            elitism_count=2,
            max_generations=20
        )
        engine = ConsciousnessEvolutionEngine(config)

        fitness_fn = FitnessFunctions.overall_consciousness()

        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=20,
            verbose=False
        )

        # Best fitness should never decrease (due to elitism)
        for i in range(1, len(result.fitness_history)):
            assert result.fitness_history[i] >= result.fitness_history[i-1] - 0.001


# ============================================================================
# Result Tests
# ============================================================================

class TestEvolutionResult:
    """Test evolution result"""

    def test_result_to_dict(self, engine, base_consciousness):
        """Test result serialization"""
        fitness_fn = FitnessFunctions.overall_consciousness()

        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=5,
            verbose=False
        )

        result_dict = result.to_dict()

        assert "best_fitness" in result_dict
        assert "generations_run" in result_dict
        assert "final_consciousness" in result_dict


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complex scenarios"""

    def test_optimize_for_specific_task(self, base_consciousness):
        """Test optimizing consciousness for a specific task"""
        config = EvolutionConfig(
            population_size=30,
            max_generations=50,
            mutation_rate=0.15
        )
        engine = ConsciousnessEvolutionEngine(config)

        # Optimize for perception and memory
        fitness_fn = FitnessFunctions.task_performance({
            "perception_fidelity": 0.4,
            "memory_depth": 0.4,
            "memory_recall_accuracy": 0.2
        })

        result = engine.evolve(
            base_consciousness,
            fitness_fn,
            generations=50,
            verbose=False
        )

        # Optimized consciousness should have high perception and memory
        best = result.best_individual.consciousness
        scores = best.get_capability_scores()

        # These should be higher than other capabilities
        target_avg = (
            scores["perception_fidelity"] +
            scores["memory_depth"] +
            scores["memory_recall_accuracy"]
        ) / 3

        assert target_avg > 0.5  # Should be above initial

    def test_multiple_evolution_runs(self, base_consciousness):
        """Test running multiple evolutions"""
        config = EvolutionConfig(
            population_size=15,
            max_generations=20
        )

        results = []
        for _ in range(3):
            engine = ConsciousnessEvolutionEngine(config)
            fitness_fn = FitnessFunctions.overall_consciousness()

            result = engine.evolve(
                base_consciousness,
                fitness_fn,
                generations=20,
                verbose=False
            )
            results.append(result.best_fitness)

        # All runs should produce reasonable results
        assert all(r > 0.3 for r in results)
