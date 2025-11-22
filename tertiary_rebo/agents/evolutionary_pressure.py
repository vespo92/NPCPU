"""
EvolutionaryPressureAgent - Applies evolutionary pressure for system optimization.

Responsibilities:
- Apply selective pressure across all domains
- Manage multi-objective fitness functions for triple bottom line optimization
- Coordinate adaptive mutation and crossover operations
- Track evolutionary lineages and speciation with fitness sharing
- Balance exploration vs exploitation via novelty search
- Co-evolve populations across domain boundaries
- Self-tune evolutionary parameters based on fitness landscape dynamics
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
)


class SelectionStrategy(Enum):
    """Selection strategies for evolutionary pressure."""
    TOURNAMENT = auto()
    ROULETTE = auto()
    RANK = auto()
    NSGA2 = auto()  # Non-dominated Sorting Genetic Algorithm II
    NOVELTY = auto()


class MutationStrategy(Enum):
    """Mutation strategies."""
    GAUSSIAN = auto()
    POLYNOMIAL = auto()
    ADAPTIVE = auto()
    CAUCHY = auto()  # Heavy-tailed for escaping local optima


@dataclass
class NoveltyArchiveEntry:
    """Entry in the novelty archive for behavioral diversity."""
    genome_id: str
    behavior_vector: np.ndarray
    novelty_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Species:
    """Species for fitness sharing and niche preservation."""
    species_id: str
    representative: 'EvolutionaryGenome'
    members: List[str] = field(default_factory=list)
    age: int = 0
    best_fitness: float = 0.0
    stagnation_count: int = 0


@dataclass
class EvolutionaryGenome:
    """Genome representation for TTR system components."""
    genome_id: str
    domain: DomainLeg
    genes: np.ndarray
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Multi-objective fitness components
    fitness_objectives: Dict[str, float] = field(default_factory=dict)
    pareto_rank: int = 0
    crowding_distance: float = 0.0

    # Novelty search
    behavior_vector: np.ndarray = field(default_factory=lambda: np.zeros(16))
    novelty_score: float = 0.0

    # Species membership
    species_id: Optional[str] = None
    shared_fitness: float = 0.0

    # Self-adaptive parameters
    mutation_sigma: float = 0.2
    crossover_preference: float = 0.5

    # Lineage tracking
    lineage_depth: int = 0
    successful_offspring: int = 0


class EvolutionaryPressureAgent(TertiaryReBoAgent):
    """
    Agent 7: Applies evolutionary pressure for system optimization.

    The TTR system evolves through natural selection principles:
    - States that increase harmony have higher fitness
    - Fitter configurations propagate their patterns
    - Mutations introduce variation with self-adaptive rates
    - Crossover combines successful patterns
    - Multi-objective Pareto optimization balances competing goals
    - Novelty search prevents premature convergence
    - Species-based fitness sharing maintains diversity

    Each domain maintains a population of potential states, with the
    best ones surviving to influence future generations. Populations
    can co-evolve through cross-domain genetic exchange.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Core evolutionary parameters (adaptive)
        self.population_size = 30
        self.base_mutation_rate = 0.15
        self.base_crossover_rate = 0.4
        self.selection_pressure = 2.0

        # Adaptive parameter bounds
        self.min_mutation_rate = 0.01
        self.max_mutation_rate = 0.5
        self.mutation_rate = self.base_mutation_rate
        self.crossover_rate = self.base_crossover_rate

        # Strategy selection
        self.selection_strategy = SelectionStrategy.NSGA2
        self.mutation_strategy = MutationStrategy.ADAPTIVE

        # Populations for each domain
        self.populations: Dict[DomainLeg, List[EvolutionaryGenome]] = {
            leg: [] for leg in DomainLeg
        }

        # Fitness history with detailed tracking
        self.fitness_history: deque = deque(maxlen=200)
        self.generation_count = 0
        self.stagnation_counter = 0
        self.best_fitness_ever = 0.0

        # Elite preservation
        self.elite_count = 5
        self.elites: Dict[DomainLeg, List[EvolutionaryGenome]] = {
            leg: [] for leg in DomainLeg
        }

        # Species management
        self.species_registry: Dict[str, Species] = {}
        self.speciation_threshold = 0.4
        self.species_stagnation_limit = 15
        self.species_age_bonus = 0.1

        # Novelty search
        self.novelty_archive: List[NoveltyArchiveEntry] = []
        self.novelty_archive_size = 100
        self.novelty_k_neighbors = 15
        self.novelty_weight = 0.3  # Balance fitness vs novelty

        # Multi-objective optimization
        self.objectives = [
            "consciousness", "coherence", "connectivity",
            "energy_flow", "harmony", "emergence"
        ]
        self.objective_weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}

        # Co-evolution parameters
        self.migration_rate = 0.05
        self.migration_interval = 5  # generations
        self.cross_domain_mating_prob = 0.1

        # Pressure scheduling
        self.pressure_schedule = {
            "initial": 1.5,
            "exploration": 1.0,
            "exploitation": 3.0,
            "current": 1.5
        }
        self.exploration_phase_generations = 50

        # Landscape analysis
        self.fitness_landscape_samples: deque = deque(maxlen=50)
        self.estimated_ruggedness = 0.5

        # Performance tracking
        self.pareto_fronts: Dict[DomainLeg, List[EvolutionaryGenome]] = {
            leg: [] for leg in DomainLeg
        }

    @property
    def agent_role(self) -> str:
        return "Evolutionary Pressure - Applies selection and mutation for optimization"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def _create_genome(self, domain: DomainLeg, state_vector: np.ndarray) -> EvolutionaryGenome:
        """Create a genome from a domain state with full initialization."""
        # Initialize behavior vector from state characteristics
        behavior_vec = self._compute_behavior_vector(state_vector)

        return EvolutionaryGenome(
            genome_id=f"gen_{domain.value}_{self.generation_count}_{np.random.randint(10000)}",
            domain=domain,
            genes=state_vector.copy(),
            generation=self.generation_count,
            behavior_vector=behavior_vec,
            mutation_sigma=self.base_mutation_rate + np.random.randn() * 0.05,
            crossover_preference=0.5 + np.random.randn() * 0.1
        )

    def _compute_behavior_vector(self, genes: np.ndarray) -> np.ndarray:
        """Compute behavioral characterization for novelty search."""
        # Compress genes into behavior space using statistical features
        n_segments = 16
        segment_size = len(genes) // n_segments
        behavior = np.zeros(n_segments)

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size
            segment = genes[start:end]
            # Combine mean and variance as behavior characterization
            behavior[i] = np.mean(segment) * 0.7 + np.std(segment) * 0.3

        return behavior

    def _calculate_multi_objective_fitness(
        self, genome: EvolutionaryGenome, tbl: TripleBottomLine
    ) -> Dict[str, float]:
        """Calculate multi-objective fitness components."""
        domain_state = tbl.get_state(genome.domain)

        objectives = {
            "consciousness": float(domain_state.consciousness_level),
            "coherence": float(domain_state.coherence),
            "connectivity": float(domain_state.connectivity),
            "energy_flow": float(domain_state.energy_flow),
            "harmony": float(tbl.harmony_score),
            "emergence": float(domain_state.emergence_potential)
        }

        # Store objectives in genome
        genome.fitness_objectives = objectives

        return objectives

    def _calculate_fitness(self, genome: EvolutionaryGenome, tbl: TripleBottomLine) -> float:
        """Calculate weighted aggregate fitness with novelty bonus."""
        # Get multi-objective components
        objectives = self._calculate_multi_objective_fitness(genome, tbl)

        # Weighted sum for base fitness
        base_fitness = sum(
            objectives[obj] * self.objective_weights.get(obj, 1.0 / len(objectives))
            for obj in objectives
        )

        # Compute behavior vector for novelty
        genome.behavior_vector = self._compute_behavior_vector(genome.genes)

        # Calculate novelty score
        novelty = self._calculate_novelty(genome)
        genome.novelty_score = novelty

        # Combined fitness: objective + novelty
        combined_fitness = (1 - self.novelty_weight) * base_fitness + self.novelty_weight * novelty

        # Apply species-based fitness sharing if assigned
        if genome.species_id and genome.species_id in self.species_registry:
            species = self.species_registry[genome.species_id]
            niche_count = max(1, len(species.members))
            genome.shared_fitness = combined_fitness / np.sqrt(niche_count)
        else:
            genome.shared_fitness = combined_fitness

        # Penalty for extreme gene values
        gene_variance = np.var(genome.genes)
        if gene_variance > 2.0:
            combined_fitness *= 0.9

        return float(np.clip(combined_fitness, 0, 1))

    def _calculate_novelty(self, genome: EvolutionaryGenome) -> float:
        """Calculate novelty score based on behavioral distance to archive."""
        if not self.novelty_archive:
            return 1.0  # Maximum novelty for first individual

        # Calculate distances to archive
        distances = []
        for entry in self.novelty_archive:
            dist = np.linalg.norm(genome.behavior_vector - entry.behavior_vector)
            distances.append(dist)

        # Add distances to current population
        for domain in DomainLeg:
            for other in self.populations[domain]:
                if other.genome_id != genome.genome_id:
                    dist = np.linalg.norm(genome.behavior_vector - other.behavior_vector)
                    distances.append(dist)

        # Average distance to k nearest neighbors
        if len(distances) >= self.novelty_k_neighbors:
            distances.sort()
            novelty = np.mean(distances[:self.novelty_k_neighbors])
        else:
            novelty = np.mean(distances) if distances else 1.0

        # Normalize novelty to [0, 1]
        return float(np.clip(novelty / 2.0, 0, 1))

    def _update_novelty_archive(self, genome: EvolutionaryGenome):
        """Add genome to novelty archive if novel enough."""
        novelty_threshold = 0.3

        if genome.novelty_score > novelty_threshold or len(self.novelty_archive) < 10:
            entry = NoveltyArchiveEntry(
                genome_id=genome.genome_id,
                behavior_vector=genome.behavior_vector.copy(),
                novelty_score=genome.novelty_score
            )
            self.novelty_archive.append(entry)

            # Keep archive size bounded
            if len(self.novelty_archive) > self.novelty_archive_size:
                # Remove least novel entries
                self.novelty_archive.sort(key=lambda e: e.novelty_score)
                self.novelty_archive = self.novelty_archive[-self.novelty_archive_size:]

    def _non_dominated_sort(self, population: List[EvolutionaryGenome]) -> List[List[EvolutionaryGenome]]:
        """NSGA-II style non-dominated sorting."""
        fronts = [[]]
        domination_count = {g.genome_id: 0 for g in population}
        dominated_set = {g.genome_id: [] for g in population}

        for p in population:
            for q in population:
                if p.genome_id != q.genome_id:
                    if self._dominates(p, q):
                        dominated_set[p.genome_id].append(q.genome_id)
                    elif self._dominates(q, p):
                        domination_count[p.genome_id] += 1

            if domination_count[p.genome_id] == 0:
                p.pareto_rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q_id in dominated_set[p.genome_id]:
                    domination_count[q_id] -= 1
                    if domination_count[q_id] == 0:
                        q = next(g for g in population if g.genome_id == q_id)
                        q.pareto_rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _dominates(self, p: EvolutionaryGenome, q: EvolutionaryGenome) -> bool:
        """Check if genome p Pareto-dominates genome q."""
        if not p.fitness_objectives or not q.fitness_objectives:
            return p.fitness > q.fitness

        at_least_one_better = False
        for obj in self.objectives:
            p_val = p.fitness_objectives.get(obj, 0)
            q_val = q.fitness_objectives.get(obj, 0)
            if p_val < q_val:
                return False
            if p_val > q_val:
                at_least_one_better = True

        return at_least_one_better

    def _calculate_crowding_distance(self, front: List[EvolutionaryGenome]):
        """Calculate crowding distance for a Pareto front."""
        if len(front) <= 2:
            for g in front:
                g.crowding_distance = float('inf')
            return

        for g in front:
            g.crowding_distance = 0.0

        for obj in self.objectives:
            # Sort by objective
            sorted_front = sorted(front, key=lambda g: g.fitness_objectives.get(obj, 0))

            # Boundary points get infinite distance
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')

            # Calculate range
            obj_min = sorted_front[0].fitness_objectives.get(obj, 0)
            obj_max = sorted_front[-1].fitness_objectives.get(obj, 0)
            obj_range = obj_max - obj_min if obj_max > obj_min else 1.0

            # Calculate distances for intermediate points
            for i in range(1, len(sorted_front) - 1):
                prev_val = sorted_front[i - 1].fitness_objectives.get(obj, 0)
                next_val = sorted_front[i + 1].fitness_objectives.get(obj, 0)
                sorted_front[i].crowding_distance += (next_val - prev_val) / obj_range

    # ==================== SPECIES MANAGEMENT ====================

    def _genetic_distance(self, g1: EvolutionaryGenome, g2: EvolutionaryGenome) -> float:
        """Calculate genetic distance between two genomes."""
        gene_diff = np.linalg.norm(g1.genes - g2.genes) / len(g1.genes)
        behavior_diff = np.linalg.norm(g1.behavior_vector - g2.behavior_vector) / len(g1.behavior_vector)
        return 0.7 * gene_diff + 0.3 * behavior_diff

    def _speciate_population(self, domain: DomainLeg):
        """Assign genomes to species based on genetic distance."""
        population = self.populations[domain]
        if not population:
            return

        # Clear previous assignments
        for genome in population:
            genome.species_id = None

        # Age existing species
        for species in self.species_registry.values():
            species.age += 1
            species.members = []

        # Assign each genome to a species
        for genome in population:
            assigned = False

            # Try to assign to existing species
            for species_id, species in self.species_registry.items():
                if species.representative.domain != domain:
                    continue
                distance = self._genetic_distance(genome, species.representative)
                if distance < self.speciation_threshold:
                    genome.species_id = species_id
                    species.members.append(genome.genome_id)
                    assigned = True
                    break

            # Create new species if no match
            if not assigned:
                new_species_id = f"species_{domain.value}_{len(self.species_registry)}_{np.random.randint(1000)}"
                new_species = Species(
                    species_id=new_species_id,
                    representative=genome,
                    members=[genome.genome_id]
                )
                self.species_registry[new_species_id] = new_species
                genome.species_id = new_species_id

        # Update species representatives and check stagnation
        for species_id, species in list(self.species_registry.items()):
            if not species.members:
                # Remove empty species
                del self.species_registry[species_id]
                continue

            # Find best member for new representative
            members = [g for g in population if g.genome_id in species.members]
            if members:
                best_member = max(members, key=lambda g: g.fitness)
                if best_member.fitness > species.best_fitness:
                    species.best_fitness = best_member.fitness
                    species.stagnation_count = 0
                    species.representative = best_member
                else:
                    species.stagnation_count += 1

    def _apply_fitness_sharing(self, domain: DomainLeg):
        """Apply fitness sharing within species."""
        population = self.populations[domain]

        for genome in population:
            if genome.species_id in self.species_registry:
                species = self.species_registry[genome.species_id]
                niche_count = max(1, len(species.members))

                # Boost young species
                age_factor = 1.0
                if species.age < 10:
                    age_factor = 1.0 + self.species_age_bonus * (10 - species.age)

                # Penalize stagnant species
                stagnation_factor = 1.0
                if species.stagnation_count > self.species_stagnation_limit // 2:
                    stagnation_factor = 0.8
                if species.stagnation_count > self.species_stagnation_limit:
                    stagnation_factor = 0.5

                genome.shared_fitness = (genome.fitness * age_factor * stagnation_factor) / np.sqrt(niche_count)

    # ==================== ADAPTIVE PARAMETERS ====================

    def _adapt_mutation_rate(self):
        """Adapt mutation rate based on fitness landscape dynamics."""
        if len(self.fitness_history) < 10:
            return

        recent = list(self.fitness_history)[-10:]
        fitness_trend = [h["mean_fitness"] for h in recent]

        # Calculate fitness improvement
        improvement = fitness_trend[-1] - fitness_trend[0]

        # If stagnating, increase mutation
        if abs(improvement) < 0.01:
            self.stagnation_counter += 1
            self.mutation_rate = min(
                self.max_mutation_rate,
                self.mutation_rate * 1.1
            )
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
            # If improving, can reduce mutation slightly
            if improvement > 0.02:
                self.mutation_rate = max(
                    self.min_mutation_rate,
                    self.mutation_rate * 0.95
                )

        # Adapt novelty weight based on diversity
        total_diversity = 0.0
        for domain in DomainLeg:
            if self.populations[domain]:
                genes_matrix = np.stack([g.genes for g in self.populations[domain]])
                total_diversity += np.mean(np.var(genes_matrix, axis=0))

        avg_diversity = total_diversity / 3
        if avg_diversity < 0.1:
            self.novelty_weight = min(0.5, self.novelty_weight + 0.05)
        elif avg_diversity > 0.5:
            self.novelty_weight = max(0.1, self.novelty_weight - 0.05)

    def _adapt_selection_pressure(self):
        """Adapt selection pressure based on evolutionary phase."""
        if self.generation_count < self.exploration_phase_generations:
            # Exploration phase: lower pressure, more diversity
            self.pressure_schedule["current"] = self.pressure_schedule["exploration"]
        else:
            # Exploitation phase: higher pressure, focus on best
            progress = min(1.0, (self.generation_count - self.exploration_phase_generations) / 100)
            self.pressure_schedule["current"] = (
                self.pressure_schedule["exploration"] +
                progress * (self.pressure_schedule["exploitation"] - self.pressure_schedule["exploration"])
            )

    def _estimate_landscape_ruggedness(self, domain: DomainLeg):
        """Estimate fitness landscape ruggedness for adaptive strategies."""
        population = self.populations[domain]
        if len(population) < 5:
            return

        # Sample pairs and compute fitness-distance correlation
        samples = []
        for _ in range(min(20, len(population) * 2)):
            g1, g2 = np.random.choice(population, 2, replace=False)
            distance = self._genetic_distance(g1, g2)
            fitness_diff = abs(g1.fitness - g2.fitness)
            samples.append((distance, fitness_diff))

        if samples:
            distances, fitness_diffs = zip(*samples)
            # High correlation = smooth landscape, low = rugged
            if np.std(distances) > 0 and np.std(fitness_diffs) > 0:
                correlation = np.corrcoef(distances, fitness_diffs)[0, 1]
                self.estimated_ruggedness = 1.0 - abs(correlation)
                self.fitness_landscape_samples.append(self.estimated_ruggedness)

    # ==================== CO-EVOLUTION ====================

    def _cross_domain_migration(self, tbl: TripleBottomLine):
        """Perform migration between domain populations for co-evolution."""
        if self.generation_count % self.migration_interval != 0:
            return

        domains = list(DomainLeg)
        migrations = []

        for source_domain in domains:
            source_pop = self.populations[source_domain]
            if not source_pop:
                continue

            # Select migrants (best performers)
            migrants_count = max(1, int(len(source_pop) * self.migration_rate))
            sorted_pop = sorted(source_pop, key=lambda g: g.fitness, reverse=True)
            migrants = sorted_pop[:migrants_count]

            for target_domain in domains:
                if target_domain == source_domain:
                    continue

                for migrant in migrants:
                    # Adapt migrant genes to target domain
                    target_state = tbl.get_state(target_domain)
                    adapted_genes = 0.7 * migrant.genes + 0.3 * target_state.state_vector

                    migrations.append({
                        "source": source_domain,
                        "target": target_domain,
                        "genes": adapted_genes,
                        "original_fitness": migrant.fitness
                    })

        # Apply migrations
        for migration in migrations:
            adapted_genome = self._create_genome(migration["target"], migration["genes"])
            adapted_genome.lineage_depth = 1  # Mark as migrant
            self.populations[migration["target"]].append(adapted_genome)

        return len(migrations)

    def _cross_domain_mating(self, parent1: EvolutionaryGenome, parent2: EvolutionaryGenome) -> EvolutionaryGenome:
        """Create offspring from parents in different domains."""
        # Use parent1's domain for the offspring
        target_domain = parent1.domain

        # Weighted blend based on fitness
        total_fitness = parent1.fitness + parent2.fitness
        if total_fitness > 0:
            w1 = parent1.fitness / total_fitness
            w2 = parent2.fitness / total_fitness
        else:
            w1 = w2 = 0.5

        child_genes = w1 * parent1.genes + w2 * parent2.genes

        # Add some noise for diversity
        child_genes += np.random.randn(len(child_genes)) * 0.05

        offspring = EvolutionaryGenome(
            genome_id=f"gen_{target_domain.value}_{self.generation_count}_xd{np.random.randint(10000)}",
            domain=target_domain,
            genes=child_genes,
            generation=self.generation_count,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            lineage_depth=max(parent1.lineage_depth, parent2.lineage_depth) + 1,
            mutation_sigma=(parent1.mutation_sigma + parent2.mutation_sigma) / 2,
            crossover_preference=(parent1.crossover_preference + parent2.crossover_preference) / 2
        )

        return offspring

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather evolutionary state from all domains with advanced metrics."""
        perceptions = {
            "domain_fitness": {},
            "population_stats": {},
            "evolutionary_pressure": 0.0,
            "diversity_scores": {},
            "pareto_fronts": {},
            "species_stats": {},
            "novelty_stats": {},
            "landscape_analysis": {}
        }

        # Initialize populations if empty
        for domain in DomainLeg:
            if not self.populations[domain]:
                state = tbl.get_state(domain)
                for _ in range(self.population_size):
                    # Create variants of current state
                    variant = state.state_vector + np.random.randn(64) * 0.1
                    genome = self._create_genome(domain, variant)
                    self.populations[domain].append(genome)

            # Calculate fitness for each genome (includes novelty and multi-objective)
            for genome in self.populations[domain]:
                genome.fitness = self._calculate_fitness(genome, tbl)

            # Perform speciation
            self._speciate_population(domain)
            self._apply_fitness_sharing(domain)

            # Compute Pareto fronts using NSGA-II
            if self.selection_strategy == SelectionStrategy.NSGA2:
                fronts = self._non_dominated_sort(self.populations[domain])
                self.pareto_fronts[domain] = fronts[0] if fronts and fronts[0] else []

                # Calculate crowding distance for each front
                for front in fronts:
                    if front:
                        self._calculate_crowding_distance(front)

                perceptions["pareto_fronts"][domain.value] = {
                    "num_fronts": len(fronts),
                    "front_0_size": len(fronts[0]) if fronts and fronts[0] else 0
                }

            # Estimate landscape ruggedness
            self._estimate_landscape_ruggedness(domain)

            # Compute population statistics
            fitnesses = [g.fitness for g in self.populations[domain]]
            shared_fitnesses = [g.shared_fitness for g in self.populations[domain]]
            novelties = [g.novelty_score for g in self.populations[domain]]

            perceptions["domain_fitness"][domain.value] = {
                "mean": float(np.mean(fitnesses)),
                "max": float(np.max(fitnesses)),
                "min": float(np.min(fitnesses)),
                "std": float(np.std(fitnesses)),
                "mean_shared": float(np.mean(shared_fitnesses)),
                "mean_novelty": float(np.mean(novelties))
            }

            # Diversity score (genetic variance)
            genes_matrix = np.stack([g.genes for g in self.populations[domain]])
            perceptions["diversity_scores"][domain.value] = float(np.mean(np.var(genes_matrix, axis=0)))

            # Species statistics
            domain_species = [s for s in self.species_registry.values()
                            if s.representative.domain == domain]
            perceptions["species_stats"][domain.value] = {
                "num_species": len(domain_species),
                "avg_species_size": np.mean([len(s.members) for s in domain_species]) if domain_species else 0,
                "stagnant_species": sum(1 for s in domain_species if s.stagnation_count > self.species_stagnation_limit // 2)
            }

        # Calculate evolutionary pressure (fitness gradient)
        all_fitnesses = [perceptions["domain_fitness"][d.value]["mean"] for d in DomainLeg]
        if len(self.fitness_history) > 5:
            recent_fitness = [f["mean_fitness"] for f in list(self.fitness_history)[-5:]]
            perceptions["evolutionary_pressure"] = all_fitnesses[0] - recent_fitness[0]
        else:
            perceptions["evolutionary_pressure"] = 0.0

        # Novelty archive stats
        perceptions["novelty_stats"] = {
            "archive_size": len(self.novelty_archive),
            "mean_novelty": np.mean([e.novelty_score for e in self.novelty_archive]) if self.novelty_archive else 0.0
        }

        # Landscape analysis
        perceptions["landscape_analysis"] = {
            "estimated_ruggedness": self.estimated_ruggedness,
            "current_mutation_rate": self.mutation_rate,
            "current_novelty_weight": self.novelty_weight,
            "stagnation_counter": self.stagnation_counter,
            "selection_pressure": self.pressure_schedule["current"]
        }

        # Adapt parameters based on perception
        self._adapt_mutation_rate()
        self._adapt_selection_pressure()

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze evolutionary dynamics and opportunities with advanced selection."""
        analysis = {
            "selection_candidates": {},
            "crossover_pairs": [],
            "cross_domain_pairs": [],
            "mutation_targets": [],
            "diversity_interventions": [],
            "species_actions": [],
            "pareto_selection": {}
        }

        for domain in DomainLeg:
            fitness_stats = perception["domain_fitness"][domain.value]
            diversity = perception["diversity_scores"][domain.value]
            species_stats = perception.get("species_stats", {}).get(domain.value, {})

            population = self.populations[domain]

            # Selection based on strategy
            if self.selection_strategy == SelectionStrategy.NSGA2:
                # Use Pareto rank and crowding distance
                sorted_pop = sorted(
                    population,
                    key=lambda g: (g.pareto_rank, -g.crowding_distance)
                )
            elif self.selection_strategy == SelectionStrategy.NOVELTY:
                # Blend fitness and novelty
                sorted_pop = sorted(
                    population,
                    key=lambda g: 0.5 * g.fitness + 0.5 * g.novelty_score,
                    reverse=True
                )
            else:
                # Tournament selection based on shared fitness
                sorted_pop = sorted(population, key=lambda g: g.shared_fitness, reverse=True)

            analysis["selection_candidates"][domain.value] = [
                g.genome_id for g in sorted_pop[:self.elite_count]
            ]

            # Pareto front selection
            if domain in self.pareto_fronts:
                analysis["pareto_selection"][domain.value] = [
                    g.genome_id for g in self.pareto_fronts[domain][:5]
                ]

            # Identify crossover pairs with tournament selection
            if len(sorted_pop) >= 4:
                tournament_size = int(self.pressure_schedule["current"])
                for _ in range(min(5, len(sorted_pop) // 2)):
                    # Tournament selection for parent 1
                    tournament1 = np.random.choice(population, min(tournament_size, len(population)), replace=False)
                    parent1 = max(tournament1, key=lambda g: g.shared_fitness)

                    # Tournament selection for parent 2
                    tournament2 = np.random.choice(population, min(tournament_size, len(population)), replace=False)
                    parent2 = max(tournament2, key=lambda g: g.shared_fitness)

                    if parent1.genome_id != parent2.genome_id:
                        analysis["crossover_pairs"].append({
                            "domain": domain.value,
                            "parent1": parent1.genome_id,
                            "parent2": parent2.genome_id,
                            "parent1_fitness": parent1.fitness,
                            "parent2_fitness": parent2.fitness
                        })

            # Identify mutation targets based on adaptive rate
            for genome in sorted_pop:
                # Higher chance for mid-fitness genomes, lower for elites and worst
                rank = sorted_pop.index(genome)
                position_factor = 1.0 - abs(rank / len(sorted_pop) - 0.5) * 2
                adjusted_rate = self.mutation_rate * position_factor

                if np.random.random() < adjusted_rate:
                    analysis["mutation_targets"].append({
                        "domain": domain.value,
                        "genome_id": genome.genome_id,
                        "current_fitness": genome.fitness,
                        "mutation_sigma": genome.mutation_sigma
                    })

            # Check for diversity collapse
            if diversity < 0.1:
                analysis["diversity_interventions"].append({
                    "domain": domain.value,
                    "current_diversity": diversity,
                    "action": "inject_randomness"
                })

            # Species-level actions
            domain_species = [s for s in self.species_registry.values()
                            if s.representative.domain == domain]
            for species in domain_species:
                if species.stagnation_count > self.species_stagnation_limit:
                    analysis["species_actions"].append({
                        "species_id": species.species_id,
                        "action": "extinction",
                        "reason": "stagnation"
                    })
                elif len(species.members) == 0:
                    analysis["species_actions"].append({
                        "species_id": species.species_id,
                        "action": "extinction",
                        "reason": "empty"
                    })

        # Cross-domain mating opportunities
        if np.random.random() < self.cross_domain_mating_prob:
            domains = list(DomainLeg)
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i+1:]:
                    pop1 = self.populations[domain1]
                    pop2 = self.populations[domain2]
                    if pop1 and pop2:
                        # Select best from each domain
                        best1 = max(pop1, key=lambda g: g.fitness)
                        best2 = max(pop2, key=lambda g: g.fitness)
                        analysis["cross_domain_pairs"].append({
                            "parent1_domain": domain1.value,
                            "parent2_domain": domain2.value,
                            "parent1": best1.genome_id,
                            "parent2": best2.genome_id
                        })

        return analysis

    def _crossover(self, parent1: EvolutionaryGenome, parent2: EvolutionaryGenome) -> EvolutionaryGenome:
        """Perform adaptive crossover between two genomes."""
        # Choose crossover type based on parent preferences
        avg_preference = (parent1.crossover_preference + parent2.crossover_preference) / 2

        if avg_preference < 0.3:
            # Uniform crossover
            mask = np.random.random(len(parent1.genes)) > 0.5
            child_genes = np.where(mask, parent1.genes, parent2.genes)
        elif avg_preference < 0.6:
            # Two-point crossover
            points = sorted(np.random.choice(len(parent1.genes), 2, replace=False))
            child_genes = parent1.genes.copy()
            child_genes[points[0]:points[1]] = parent2.genes[points[0]:points[1]]
        else:
            # Simulated Binary Crossover (SBX)
            eta = 2.0  # Distribution index
            child_genes = np.zeros_like(parent1.genes)
            for i in range(len(parent1.genes)):
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child_genes[i] = 0.5 * ((1 + beta) * parent1.genes[i] + (1 - beta) * parent2.genes[i])

        # Inherit and slightly mutate self-adaptive parameters
        child_mutation_sigma = (parent1.mutation_sigma + parent2.mutation_sigma) / 2
        child_mutation_sigma *= np.exp(np.random.randn() * 0.1)
        child_mutation_sigma = np.clip(child_mutation_sigma, 0.01, 0.5)

        child_crossover_pref = (parent1.crossover_preference + parent2.crossover_preference) / 2
        child_crossover_pref += np.random.randn() * 0.05
        child_crossover_pref = np.clip(child_crossover_pref, 0, 1)

        offspring = EvolutionaryGenome(
            genome_id=f"gen_{parent1.domain.value}_{self.generation_count}_c{np.random.randint(10000)}",
            domain=parent1.domain,
            genes=child_genes,
            generation=self.generation_count,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            lineage_depth=max(parent1.lineage_depth, parent2.lineage_depth) + 1,
            mutation_sigma=child_mutation_sigma,
            crossover_preference=child_crossover_pref
        )

        # Track successful parent
        parent1.successful_offspring += 1
        parent2.successful_offspring += 1

        return offspring

    def _mutate(self, genome: EvolutionaryGenome) -> EvolutionaryGenome:
        """Apply adaptive mutation to a genome."""
        mutated_genes = genome.genes.copy()

        # Use genome's self-adaptive mutation sigma
        sigma = genome.mutation_sigma

        # Choose mutation strategy based on landscape ruggedness
        if self.mutation_strategy == MutationStrategy.ADAPTIVE:
            if self.estimated_ruggedness > 0.7:
                # Rugged landscape: use Cauchy (heavy-tailed) for large jumps
                mutation_mask = np.random.random(len(mutated_genes)) < self.mutation_rate
                mutated_genes[mutation_mask] += np.random.standard_cauchy(np.sum(mutation_mask)) * sigma * 0.5
            elif self.estimated_ruggedness > 0.4:
                # Moderate: polynomial mutation
                mutation_mask = np.random.random(len(mutated_genes)) < self.mutation_rate
                eta_m = 20.0  # Distribution index
                for i in np.where(mutation_mask)[0]:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    mutated_genes[i] += delta * sigma
            else:
                # Smooth landscape: Gaussian mutation
                mutation_mask = np.random.random(len(mutated_genes)) < self.mutation_rate
                mutated_genes[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * sigma
        else:
            # Default Gaussian
            mutation_mask = np.random.random(len(mutated_genes)) < self.mutation_rate
            mutated_genes[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * sigma

        # Apply bounds
        mutated_genes = np.tanh(mutated_genes)

        # Self-adapt mutation sigma
        new_sigma = sigma * np.exp(np.random.randn() * 0.1)
        new_sigma = np.clip(new_sigma, 0.01, 0.5)

        return EvolutionaryGenome(
            genome_id=f"gen_{genome.domain.value}_{self.generation_count}_m{np.random.randint(10000)}",
            domain=genome.domain,
            genes=mutated_genes,
            generation=self.generation_count,
            parent_ids=[genome.genome_id],
            mutations=genome.mutations + 1,
            lineage_depth=genome.lineage_depth + 1,
            mutation_sigma=new_sigma,
            crossover_preference=genome.crossover_preference
        )

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Generate evolutionary operations with cross-domain and species support."""
        synthesis = {
            "offspring": [],
            "cross_domain_offspring": [],
            "mutations": [],
            "selection_results": {},
            "diversity_injections": [],
            "species_extinctions": [],
            "novelty_candidates": []
        }

        # Create offspring through crossover
        for pair in analysis.get("crossover_pairs", []):
            domain = DomainLeg(pair["domain"])
            parent1 = next((g for g in self.populations[domain] if g.genome_id == pair["parent1"]), None)
            parent2 = next((g for g in self.populations[domain] if g.genome_id == pair["parent2"]), None)

            if parent1 and parent2:
                offspring = self._crossover(parent1, parent2)
                synthesis["offspring"].append(offspring)

        # Create cross-domain offspring
        for pair in analysis.get("cross_domain_pairs", []):
            domain1 = DomainLeg(pair["parent1_domain"])
            domain2 = DomainLeg(pair["parent2_domain"])
            parent1 = next((g for g in self.populations[domain1] if g.genome_id == pair["parent1"]), None)
            parent2 = next((g for g in self.populations[domain2] if g.genome_id == pair["parent2"]), None)

            if parent1 and parent2:
                # Create offspring for both domains
                offspring1 = self._cross_domain_mating(parent1, parent2)
                offspring2 = self._cross_domain_mating(parent2, parent1)
                synthesis["cross_domain_offspring"].extend([offspring1, offspring2])

        # Apply mutations
        for target in analysis.get("mutation_targets", []):
            domain = DomainLeg(target["domain"])
            genome = next((g for g in self.populations[domain] if g.genome_id == target["genome_id"]), None)

            if genome:
                mutant = self._mutate(genome)
                synthesis["mutations"].append(mutant)

                # Track highly novel mutants
                if mutant.novelty_score > 0.5:
                    synthesis["novelty_candidates"].append(mutant)

        # Handle diversity interventions
        for intervention in analysis.get("diversity_interventions", []):
            domain = DomainLeg(intervention["domain"])
            state = tbl.get_state(domain)

            # Inject multiple random genomes for diversity
            for _ in range(3):
                random_genome = self._create_genome(
                    domain,
                    state.state_vector + np.random.randn(64) * 0.5
                )
                synthesis["diversity_injections"].append(random_genome)

        # Handle species actions
        for action in analysis.get("species_actions", []):
            if action["action"] == "extinction":
                synthesis["species_extinctions"].append(action["species_id"])

        # Record selection results
        for domain_name, elite_ids in analysis.get("selection_candidates", {}).items():
            synthesis["selection_results"][domain_name] = elite_ids

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply evolutionary changes with advanced population management."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        self.generation_count += 1

        # Handle species extinctions first
        for species_id in synthesis.get("species_extinctions", []):
            if species_id in self.species_registry:
                del self.species_registry[species_id]
                changes["species_extinct"] = changes.get("species_extinct", 0) + 1
                insights.append(f"Species {species_id} went extinct")

        # Add offspring to populations
        offspring_count = 0
        for offspring in synthesis.get("offspring", []):
            domain = offspring.domain
            offspring.fitness = self._calculate_fitness(offspring, tbl)
            self._update_novelty_archive(offspring)
            self.populations[domain].append(offspring)
            offspring_count += 1

        if offspring_count > 0:
            changes["offspring_created"] = offspring_count
            insights.append(f"Created {offspring_count} offspring through crossover")

        # Add cross-domain offspring
        xd_offspring_count = 0
        for offspring in synthesis.get("cross_domain_offspring", []):
            domain = offspring.domain
            offspring.fitness = self._calculate_fitness(offspring, tbl)
            self._update_novelty_archive(offspring)
            self.populations[domain].append(offspring)
            xd_offspring_count += 1

        if xd_offspring_count > 0:
            changes["cross_domain_offspring"] = xd_offspring_count
            insights.append(f"Created {xd_offspring_count} cross-domain offspring")

        # Add mutations
        mutation_count = 0
        for mutant in synthesis.get("mutations", []):
            domain = mutant.domain
            mutant.fitness = self._calculate_fitness(mutant, tbl)
            self._update_novelty_archive(mutant)
            self.populations[domain].append(mutant)
            mutation_count += 1

        if mutation_count > 0:
            changes["mutations_applied"] = mutation_count
            insights.append(f"Applied {mutation_count} adaptive mutations")

        # Add diversity injections
        injection_count = 0
        for injection in synthesis.get("diversity_injections", []):
            domain = injection.domain
            injection.fitness = self._calculate_fitness(injection, tbl)
            self._update_novelty_archive(injection)
            self.populations[domain].append(injection)
            injection_count += 1

        if injection_count > 0:
            changes["diversity_injections"] = injection_count
            insights.append(f"Injected {injection_count} diversity genomes")

        # Perform cross-domain migration
        migration_count = self._cross_domain_migration(tbl)
        if migration_count:
            changes["migrations"] = migration_count
            insights.append(f"Performed {migration_count} cross-domain migrations")

        # Selection: NSGA-II style with crowding distance
        selected_count = 0
        for domain in DomainLeg:
            population = self.populations[domain]
            if len(population) > self.population_size:
                if self.selection_strategy == SelectionStrategy.NSGA2:
                    # Sort by Pareto rank first, then crowding distance
                    sorted_pop = sorted(
                        population,
                        key=lambda g: (g.pareto_rank, -g.crowding_distance)
                    )
                else:
                    # Sort by shared fitness for niche preservation
                    sorted_pop = sorted(population, key=lambda g: g.shared_fitness, reverse=True)

                self.populations[domain] = sorted_pop[:self.population_size]
                selected_count += len(population) - self.population_size

                # Update elites (always use raw fitness)
                fitness_sorted = sorted(population, key=lambda g: g.fitness, reverse=True)
                self.elites[domain] = fitness_sorted[:self.elite_count]

                # Apply best genome to domain state
                best_genome = fitness_sorted[0]
                state = tbl.get_state(domain)

                # Track if this is a new best
                if best_genome.fitness > self.best_fitness_ever:
                    self.best_fitness_ever = best_genome.fitness
                    insights.append(f"New best fitness: {best_genome.fitness:.4f} in {domain.value}")

                # Blend current state with best genome (gradual evolution)
                blend_factor = 0.2 if self.stagnation_counter < 10 else 0.3
                state.state_vector = (1 - blend_factor) * state.state_vector + blend_factor * best_genome.genes

        if selected_count > 0:
            changes["individuals_selected_out"] = selected_count
            insights.append(f"Selection removed {selected_count} less fit individuals")

        # Record detailed fitness history
        domain_fitness = {}
        for d in DomainLeg:
            fitnesses = [g.fitness for g in self.populations[d]]
            domain_fitness[d.value] = {
                "mean": float(np.mean(fitnesses)),
                "max": float(np.max(fitnesses)),
                "std": float(np.std(fitnesses))
            }

        mean_fitness = np.mean([domain_fitness[d.value]["mean"] for d in DomainLeg])
        max_fitness = max([domain_fitness[d.value]["max"] for d in DomainLeg])

        self.fitness_history.append({
            "generation": self.generation_count,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "mutation_rate": self.mutation_rate,
            "novelty_weight": self.novelty_weight,
            "num_species": len(self.species_registry),
            "timestamp": datetime.now().isoformat()
        })

        # Emit cross-domain signal about evolutionary progress
        if mean_fitness > 0.7:
            self.emit_cross_domain_signal(
                signal_type="high_fitness_achieved",
                payload={
                    "mean_fitness": mean_fitness,
                    "generation": self.generation_count
                },
                strength=mean_fitness
            )

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["harmony_delta"] = tbl.harmony_score - metrics_delta["harmony_before"]
        metrics_delta["generation"] = self.generation_count
        metrics_delta["mean_fitness"] = mean_fitness
        metrics_delta["max_fitness"] = max_fitness
        metrics_delta["mutation_rate"] = self.mutation_rate
        metrics_delta["novelty_weight"] = self.novelty_weight
        metrics_delta["num_species"] = len(self.species_registry)
        metrics_delta["novelty_archive_size"] = len(self.novelty_archive)
        metrics_delta["stagnation_counter"] = self.stagnation_counter

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
