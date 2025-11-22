"""
Consciousness Evolution Engine

Evolves optimal consciousness configurations through genetic algorithms.

Based on Week 3 roadmap: Consciousness Evolution - Genetic Algorithm Optimization

Use cases:
- Task-specific consciousness optimization
- Multi-objective optimization
- Adaptive agent design
- Discovering novel consciousness configurations
"""

import random
import copy
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


class SelectionMethod(Enum):
    """Selection methods for parent selection"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITIST = "elitist"


class CrossoverMethod(Enum):
    """Crossover methods for reproduction"""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    ARITHMETIC = "arithmetic"


class MutationMethod(Enum):
    """Mutation methods for genetic variation"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine"""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    mutation_sigma: float = 0.1
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    max_generations: int = 100
    early_stop_threshold: float = 0.001  # Stop if improvement < threshold
    early_stop_generations: int = 10  # Consecutive generations to check
    diversity_threshold: float = 0.1  # Maintain minimum diversity


@dataclass
class Individual:
    """An individual in the population"""
    consciousness: GradedConsciousness
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: int = 0

    @property
    def id(self) -> str:
        """Unique ID based on capability hash"""
        scores = self.consciousness.get_capability_scores()
        return f"ind_{hash(tuple(sorted(scores.items()))) % 100000:05d}"


@dataclass
class EvolutionResult:
    """Result of evolution run"""
    best_individual: Individual
    best_fitness: float
    generations_run: int
    population_history: List[Dict[str, float]]
    fitness_history: List[float]
    diversity_history: List[float]
    early_stopped: bool = False
    convergence_generation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_fitness": self.best_fitness,
            "generations_run": self.generations_run,
            "early_stopped": self.early_stopped,
            "convergence_generation": self.convergence_generation,
            "final_consciousness": self.best_individual.consciousness.get_capability_scores(),
            "fitness_history": self.fitness_history[-10:],  # Last 10
            "diversity_history": self.diversity_history[-10:]
        }


class FitnessFunctions:
    """Collection of pre-built fitness functions"""

    @staticmethod
    def task_performance(weights: Dict[str, float]) -> Callable[[GradedConsciousness], float]:
        """
        Create task-specific fitness function.

        Args:
            weights: Dictionary mapping capabilities to their importance

        Example:
            fitness = FitnessFunctions.task_performance({
                "perception_fidelity": 0.4,
                "reaction_speed": 0.4,
                "memory_depth": 0.2
            })
        """
        def fitness_fn(consciousness: GradedConsciousness) -> float:
            scores = consciousness.get_capability_scores()
            total = 0.0
            total_weight = sum(weights.values())

            for capability, weight in weights.items():
                if capability in scores:
                    total += scores[capability] * weight

            return total / total_weight if total_weight > 0 else 0.0

        return fitness_fn

    @staticmethod
    def balanced_consciousness() -> Callable[[GradedConsciousness], float]:
        """
        Fitness function favoring balanced capabilities.
        Penalizes extreme specialization.
        """
        def fitness_fn(consciousness: GradedConsciousness) -> float:
            scores = list(consciousness.get_capability_scores().values())
            mean_score = np.mean(scores)
            variance = np.var(scores)

            # High mean + low variance = balanced and capable
            return mean_score * (1 - variance)

        return fitness_fn

    @staticmethod
    def overall_consciousness() -> Callable[[GradedConsciousness], float]:
        """Simple fitness: overall consciousness score"""
        def fitness_fn(consciousness: GradedConsciousness) -> float:
            return consciousness.overall_consciousness_score()

        return fitness_fn

    @staticmethod
    def threshold_capabilities(
        thresholds: Dict[str, float]
    ) -> Callable[[GradedConsciousness], float]:
        """
        Fitness based on meeting capability thresholds.

        Args:
            thresholds: Minimum required score for each capability
        """
        def fitness_fn(consciousness: GradedConsciousness) -> float:
            scores = consciousness.get_capability_scores()
            met = 0
            total = len(thresholds)

            for capability, threshold in thresholds.items():
                if capability in scores and scores[capability] >= threshold:
                    met += 1
                    # Bonus for exceeding threshold
                    met += (scores[capability] - threshold) * 0.5

            return met / total if total > 0 else 0.0

        return fitness_fn

    @staticmethod
    def multi_objective(
        objectives: List[Callable[[GradedConsciousness], float]],
        weights: Optional[List[float]] = None
    ) -> Callable[[GradedConsciousness], float]:
        """
        Combine multiple objectives with weights.

        Args:
            objectives: List of fitness functions
            weights: Optional weights for each objective
        """
        if weights is None:
            weights = [1.0] * len(objectives)

        def fitness_fn(consciousness: GradedConsciousness) -> float:
            total = 0.0
            for obj, weight in zip(objectives, weights):
                total += obj(consciousness) * weight
            return total / sum(weights)

        return fitness_fn


class ConsciousnessEvolutionEngine:
    """
    Evolve optimal consciousness through genetic algorithms.

    Features:
    - Multiple selection methods (tournament, roulette, rank)
    - Multiple crossover methods (uniform, single-point, arithmetic)
    - Multiple mutation methods (gaussian, uniform, adaptive)
    - Elitism to preserve best individuals
    - Diversity maintenance
    - Early stopping on convergence
    - Detailed evolution history

    Example:
        engine = ConsciousnessEvolutionEngine()

        # Define fitness function
        fitness = FitnessFunctions.task_performance({
            "perception_fidelity": 0.4,
            "reaction_speed": 0.4,
            "memory_depth": 0.2
        })

        # Evolve from base consciousness
        base = GradedConsciousness(
            perception_fidelity=0.5,
            reaction_speed=0.5,
            memory_depth=0.5
        )

        result = engine.evolve(base, fitness, generations=100)
        print(f"Best fitness: {result.best_fitness:.4f}")
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def create_initial_population(
        self,
        base_consciousness: GradedConsciousness
    ) -> List[Individual]:
        """Create initial population with random variations"""
        population = []

        for i in range(self.config.population_size):
            # Clone base
            scores = base_consciousness.get_capability_scores()

            # Add random variation (±20% of base value or ±0.2)
            varied_scores = {}
            for capability, value in scores.items():
                variation = random.uniform(-0.2, 0.2)
                new_value = np.clip(value + variation, 0.0, 1.0)
                varied_scores[capability] = new_value

            individual = Individual(
                consciousness=GradedConsciousness(**varied_scores),
                generation=0
            )
            population.append(individual)

        return population

    def evaluate_fitness(
        self,
        population: List[Individual],
        fitness_function: Callable[[GradedConsciousness], float]
    ):
        """Evaluate fitness for entire population"""
        for individual in population:
            individual.fitness = fitness_function(individual.consciousness)

    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity (average distance between individuals)"""
        if len(population) < 2:
            return 0.0

        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = population[i].consciousness.distance_to(
                    population[j].consciousness
                )
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Select parent using tournament selection"""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _roulette_selection(self, population: List[Individual]) -> Individual:
        """Select parent using roulette wheel selection"""
        total_fitness = sum(max(0.001, ind.fitness) for ind in population)
        pick = random.uniform(0, total_fitness)
        current = 0.0

        for individual in population:
            current += max(0.001, individual.fitness)
            if current >= pick:
                return individual

        return population[-1]

    def _rank_selection(self, population: List[Individual]) -> Individual:
        """Select parent using rank-based selection"""
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0.0

        for ind, rank in zip(sorted_pop, ranks):
            current += rank
            if current >= pick:
                return ind

        return sorted_pop[-1]

    def select_parent(self, population: List[Individual]) -> Individual:
        """Select a parent based on configured selection method"""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population)
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection(population)
        else:
            return self._tournament_selection(population)

    def _uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> GradedConsciousness:
        """Uniform crossover: randomly select each gene from either parent"""
        scores1 = parent1.consciousness.get_capability_scores()
        scores2 = parent2.consciousness.get_capability_scores()
        child_scores = {}

        for capability in scores1.keys():
            child_scores[capability] = (
                scores1[capability] if random.random() < 0.5
                else scores2[capability]
            )

        return GradedConsciousness(**child_scores)

    def _single_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> GradedConsciousness:
        """Single-point crossover"""
        scores1 = parent1.consciousness.get_capability_scores()
        scores2 = parent2.consciousness.get_capability_scores()
        capabilities = list(scores1.keys())

        point = random.randint(1, len(capabilities) - 1)
        child_scores = {}

        for i, capability in enumerate(capabilities):
            child_scores[capability] = (
                scores1[capability] if i < point else scores2[capability]
            )

        return GradedConsciousness(**child_scores)

    def _arithmetic_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> GradedConsciousness:
        """Arithmetic crossover: weighted average of parents"""
        scores1 = parent1.consciousness.get_capability_scores()
        scores2 = parent2.consciousness.get_capability_scores()
        alpha = random.random()
        child_scores = {}

        for capability in scores1.keys():
            child_scores[capability] = (
                alpha * scores1[capability] +
                (1 - alpha) * scores2[capability]
            )

        return GradedConsciousness(**child_scores)

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Individual:
        """Perform crossover between two parents"""
        if random.random() > self.config.crossover_rate:
            return Individual(
                consciousness=copy.deepcopy(parent1.consciousness),
                generation=self.generation,
                parent_ids=[parent1.id]
            )

        if self.config.crossover_method == CrossoverMethod.UNIFORM:
            child_consciousness = self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            child_consciousness = self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.ARITHMETIC:
            child_consciousness = self._arithmetic_crossover(parent1, parent2)
        else:
            child_consciousness = self._uniform_crossover(parent1, parent2)

        return Individual(
            consciousness=child_consciousness,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id]
        )

    def _gaussian_mutation(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Gaussian mutation: add normal distributed noise"""
        mutated = {}
        for capability, value in scores.items():
            if random.random() < self.config.mutation_rate:
                mutation = random.gauss(0, self.config.mutation_sigma)
                mutated[capability] = np.clip(value + mutation, 0.0, 1.0)
            else:
                mutated[capability] = value
        return mutated

    def _uniform_mutation(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Uniform mutation: replace with random value"""
        mutated = {}
        for capability, value in scores.items():
            if random.random() < self.config.mutation_rate:
                mutated[capability] = random.random()
            else:
                mutated[capability] = value
        return mutated

    def _adaptive_mutation(
        self,
        scores: Dict[str, float],
        individual: Individual
    ) -> Dict[str, float]:
        """Adaptive mutation: rate based on fitness"""
        # Lower fitness = higher mutation rate
        adaptive_rate = self.config.mutation_rate * (1.5 - individual.fitness)
        adaptive_rate = np.clip(adaptive_rate, 0.01, 0.5)

        mutated = {}
        for capability, value in scores.items():
            if random.random() < adaptive_rate:
                mutation = random.gauss(0, self.config.mutation_sigma)
                mutated[capability] = np.clip(value + mutation, 0.0, 1.0)
            else:
                mutated[capability] = value
        return mutated

    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to individual"""
        scores = individual.consciousness.get_capability_scores()

        if self.config.mutation_method == MutationMethod.GAUSSIAN:
            mutated_scores = self._gaussian_mutation(scores)
        elif self.config.mutation_method == MutationMethod.UNIFORM:
            mutated_scores = self._uniform_mutation(scores)
        elif self.config.mutation_method == MutationMethod.ADAPTIVE:
            mutated_scores = self._adaptive_mutation(scores, individual)
        else:
            mutated_scores = self._gaussian_mutation(scores)

        # Count actual mutations
        mutations = sum(
            1 for k in scores.keys()
            if abs(scores[k] - mutated_scores[k]) > 0.001
        )

        return Individual(
            consciousness=GradedConsciousness(**mutated_scores),
            generation=individual.generation,
            parent_ids=individual.parent_ids,
            mutations=individual.mutations + mutations
        )

    def evolve(
        self,
        initial_consciousness: GradedConsciousness,
        fitness_function: Callable[[GradedConsciousness], float],
        generations: Optional[int] = None,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, float], None]] = None
    ) -> EvolutionResult:
        """
        Evolve consciousness to maximize fitness.

        Args:
            initial_consciousness: Starting consciousness to evolve from
            fitness_function: Function that scores consciousness (higher = better)
            generations: Number of generations (uses config default if None)
            verbose: Print progress every 10 generations
            callback: Optional callback(generation, best_fitness, diversity)

        Returns:
            EvolutionResult with best individual and statistics
        """
        generations = generations or self.config.max_generations

        # Initialize
        self.population = self.create_initial_population(initial_consciousness)
        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []
        population_history = []

        # Track for early stopping
        no_improvement_count = 0
        prev_best_fitness = 0.0
        convergence_gen = None

        for gen in range(generations):
            self.generation = gen

            # Evaluate fitness
            self.evaluate_fitness(self.population, fitness_function)

            # Find best
            gen_best = max(self.population, key=lambda x: x.fitness)
            gen_best_fitness = gen_best.fitness

            # Update best ever
            if self.best_ever is None or gen_best_fitness > self.best_ever.fitness:
                self.best_ever = copy.deepcopy(gen_best)

            # Calculate diversity
            diversity = self.calculate_diversity(self.population)

            # Record history
            self.fitness_history.append(gen_best_fitness)
            self.diversity_history.append(diversity)
            population_history.append({
                "generation": gen,
                "best_fitness": gen_best_fitness,
                "avg_fitness": np.mean([ind.fitness for ind in self.population]),
                "diversity": diversity
            })

            # Callback
            if callback:
                callback(gen, gen_best_fitness, diversity)

            # Verbose output
            if verbose and gen % 10 == 0:
                print(
                    f"Generation {gen}: "
                    f"Best fitness = {gen_best_fitness:.4f}, "
                    f"Diversity = {diversity:.4f}"
                )

            # Early stopping check
            improvement = gen_best_fitness - prev_best_fitness
            if improvement < self.config.early_stop_threshold:
                no_improvement_count += 1
                if convergence_gen is None:
                    convergence_gen = gen
            else:
                no_improvement_count = 0
                convergence_gen = None

            if no_improvement_count >= self.config.early_stop_generations:
                if verbose:
                    print(f"Early stopping at generation {gen}")
                return EvolutionResult(
                    best_individual=self.best_ever,
                    best_fitness=self.best_ever.fitness,
                    generations_run=gen + 1,
                    population_history=population_history,
                    fitness_history=self.fitness_history,
                    diversity_history=self.diversity_history,
                    early_stopped=True,
                    convergence_generation=convergence_gen
                )

            prev_best_fitness = gen_best_fitness

            # Create next generation
            next_population = []

            # Elitism: keep best individuals
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            for i in range(self.config.elitism_count):
                elite = copy.deepcopy(sorted_pop[i])
                elite.generation = gen + 1
                next_population.append(elite)

            # Create offspring
            while len(next_population) < self.config.population_size:
                parent1 = self.select_parent(self.population)
                parent2 = self.select_parent(self.population)

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child.generation = gen + 1

                next_population.append(child)

            # Diversity maintenance: if diversity too low, inject random individuals
            if diversity < self.config.diversity_threshold:
                inject_count = max(1, self.config.population_size // 10)
                for _ in range(inject_count):
                    random_scores = {
                        k: random.random()
                        for k in initial_consciousness.get_capability_scores().keys()
                    }
                    random_ind = Individual(
                        consciousness=GradedConsciousness(**random_scores),
                        generation=gen + 1
                    )
                    # Replace weakest individuals
                    next_population = sorted(
                        next_population, key=lambda x: x.fitness, reverse=True
                    )
                    next_population[-1] = random_ind

            self.population = next_population

        return EvolutionResult(
            best_individual=self.best_ever,
            best_fitness=self.best_ever.fitness,
            generations_run=generations,
            population_history=population_history,
            fitness_history=self.fitness_history,
            diversity_history=self.diversity_history,
            early_stopped=False,
            convergence_generation=convergence_gen
        )

    def get_population_stats(self) -> Dict[str, Any]:
        """Get current population statistics"""
        if not self.population:
            return {}

        fitnesses = [ind.fitness for ind in self.population]

        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(fitnesses),
            "avg_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "min_fitness": min(fitnesses),
            "diversity": self.calculate_diversity(self.population),
            "best_ever_fitness": self.best_ever.fitness if self.best_ever else None
        }


# Example usage
if __name__ == "__main__":
    # Create evolution engine
    config = EvolutionConfig(
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.8,
        max_generations=50
    )
    engine = ConsciousnessEvolutionEngine(config)

    # Define fitness: optimize for fast perception and reaction
    fitness = FitnessFunctions.task_performance({
        "perception_fidelity": 0.4,
        "reaction_speed": 0.4,
        "memory_depth": 0.2
    })

    # Start with balanced consciousness
    base_consciousness = GradedConsciousness(
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

    print("Starting evolution...")
    print(f"Initial overall score: {base_consciousness.overall_consciousness_score():.4f}")

    result = engine.evolve(base_consciousness, fitness, verbose=True)

    print(f"\n=== Evolution Complete ===")
    print(f"Best fitness: {result.best_fitness:.4f}")
    print(f"Generations: {result.generations_run}")
    print(f"Early stopped: {result.early_stopped}")

    print(f"\nOptimized consciousness:")
    for cap, score in result.best_individual.consciousness.get_capability_scores().items():
        print(f"  {cap}: {score:.3f}")

    print(f"\nOverall score: {result.best_individual.consciousness.overall_consciousness_score():.4f}")
