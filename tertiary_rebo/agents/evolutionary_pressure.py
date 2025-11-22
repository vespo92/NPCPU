"""
EvolutionaryPressureAgent - Applies evolutionary pressure for system optimization.

Responsibilities:
- Apply selective pressure across all domains
- Manage fitness functions for triple bottom line optimization
- Coordinate mutation and crossover operations
- Track evolutionary lineages and speciation
- Balance exploration vs exploitation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
)


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


class EvolutionaryPressureAgent(TertiaryReBoAgent):
    """
    Agent 7: Applies evolutionary pressure for system optimization.

    The TTR system evolves through natural selection principles:
    - States that increase harmony have higher fitness
    - Fitter configurations propagate their patterns
    - Mutations introduce variation
    - Crossover combines successful patterns

    Each domain maintains a population of potential states, with the
    best ones surviving to influence future generations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Evolutionary parameters
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        self.selection_pressure = 2.0  # Tournament size factor

        # Populations for each domain
        self.populations: Dict[DomainLeg, List[EvolutionaryGenome]] = {
            leg: [] for leg in DomainLeg
        }

        # Fitness history
        self.fitness_history: deque = deque(maxlen=100)
        self.generation_count = 0

        # Elite preservation
        self.elite_count = 3
        self.elites: Dict[DomainLeg, List[EvolutionaryGenome]] = {
            leg: [] for leg in DomainLeg
        }

        # Speciation tracking
        self.species: Dict[str, List[str]] = {}  # species_id -> genome_ids
        self.speciation_threshold = 0.5

    @property
    def agent_role(self) -> str:
        return "Evolutionary Pressure - Applies selection and mutation for optimization"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def _create_genome(self, domain: DomainLeg, state_vector: np.ndarray) -> EvolutionaryGenome:
        """Create a genome from a domain state."""
        return EvolutionaryGenome(
            genome_id=f"gen_{domain.value}_{self.generation_count}_{np.random.randint(10000)}",
            domain=domain,
            genes=state_vector.copy(),
            generation=self.generation_count
        )

    def _calculate_fitness(self, genome: EvolutionaryGenome, tbl: TripleBottomLine) -> float:
        """Calculate fitness of a genome based on TTR objectives."""
        domain_state = tbl.get_state(genome.domain)

        # Multi-objective fitness
        fitness_components = [
            domain_state.consciousness_level * 0.2,
            domain_state.coherence * 0.2,
            domain_state.connectivity * 0.15,
            domain_state.energy_flow * 0.15,
            tbl.harmony_score * 0.3  # Global harmony matters most
        ]

        base_fitness = sum(fitness_components)

        # Bonus for emergence potential
        base_fitness += domain_state.emergence_potential * 0.1

        # Penalty for extreme values
        gene_variance = np.var(genome.genes)
        if gene_variance > 2.0:
            base_fitness *= 0.9

        return float(np.clip(base_fitness, 0, 1))

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather evolutionary state from all domains."""
        perceptions = {
            "domain_fitness": {},
            "population_stats": {},
            "evolutionary_pressure": 0.0,
            "diversity_scores": {}
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

            # Calculate fitness for each genome
            for genome in self.populations[domain]:
                genome.fitness = self._calculate_fitness(genome, tbl)

            # Compute population statistics
            fitnesses = [g.fitness for g in self.populations[domain]]
            perceptions["domain_fitness"][domain.value] = {
                "mean": np.mean(fitnesses),
                "max": np.max(fitnesses),
                "min": np.min(fitnesses),
                "std": np.std(fitnesses)
            }

            # Diversity score (genetic variance)
            genes_matrix = np.stack([g.genes for g in self.populations[domain]])
            perceptions["diversity_scores"][domain.value] = float(np.mean(np.var(genes_matrix, axis=0)))

        # Calculate evolutionary pressure (fitness gradient)
        all_fitnesses = [perceptions["domain_fitness"][d.value]["mean"] for d in DomainLeg]
        if len(self.fitness_history) > 5:
            recent_fitness = [f["mean_fitness"] for f in list(self.fitness_history)[-5:]]
            perceptions["evolutionary_pressure"] = all_fitnesses[0] - recent_fitness[0]
        else:
            perceptions["evolutionary_pressure"] = 0.0

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze evolutionary dynamics and opportunities."""
        analysis = {
            "selection_candidates": {},
            "crossover_pairs": [],
            "mutation_targets": [],
            "diversity_interventions": []
        }

        for domain in DomainLeg:
            fitness_stats = perception["domain_fitness"][domain.value]
            diversity = perception["diversity_scores"][domain.value]

            # Select candidates for reproduction
            population = self.populations[domain]
            sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)

            analysis["selection_candidates"][domain.value] = [
                g.genome_id for g in sorted_pop[:self.elite_count]
            ]

            # Identify crossover pairs (between high-fitness genomes)
            if len(sorted_pop) >= 4:
                for i in range(min(3, len(sorted_pop) // 2)):
                    analysis["crossover_pairs"].append({
                        "domain": domain.value,
                        "parent1": sorted_pop[i].genome_id,
                        "parent2": sorted_pop[i + 1].genome_id
                    })

            # Identify mutation targets (medium fitness genomes)
            mid_start = len(sorted_pop) // 3
            mid_end = 2 * len(sorted_pop) // 3
            for genome in sorted_pop[mid_start:mid_end]:
                if np.random.random() < self.mutation_rate:
                    analysis["mutation_targets"].append({
                        "domain": domain.value,
                        "genome_id": genome.genome_id,
                        "current_fitness": genome.fitness
                    })

            # Check for diversity collapse
            if diversity < 0.1:
                analysis["diversity_interventions"].append({
                    "domain": domain.value,
                    "current_diversity": diversity,
                    "action": "inject_randomness"
                })

        return analysis

    def _crossover(self, parent1: EvolutionaryGenome, parent2: EvolutionaryGenome) -> EvolutionaryGenome:
        """Perform crossover between two genomes."""
        # Two-point crossover
        points = sorted(np.random.choice(len(parent1.genes), 2, replace=False))
        child_genes = parent1.genes.copy()
        child_genes[points[0]:points[1]] = parent2.genes[points[0]:points[1]]

        return EvolutionaryGenome(
            genome_id=f"gen_{parent1.domain.value}_{self.generation_count}_c{np.random.randint(10000)}",
            domain=parent1.domain,
            genes=child_genes,
            generation=self.generation_count,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

    def _mutate(self, genome: EvolutionaryGenome) -> EvolutionaryGenome:
        """Apply mutation to a genome."""
        mutated_genes = genome.genes.copy()

        # Gaussian mutation
        mutation_mask = np.random.random(len(mutated_genes)) < self.mutation_rate
        mutated_genes[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.2

        # Clip to reasonable range
        mutated_genes = np.tanh(mutated_genes)

        return EvolutionaryGenome(
            genome_id=f"gen_{genome.domain.value}_{self.generation_count}_m{np.random.randint(10000)}",
            domain=genome.domain,
            genes=mutated_genes,
            generation=self.generation_count,
            parent_ids=[genome.genome_id],
            mutations=genome.mutations + 1
        )

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Generate evolutionary operations."""
        synthesis = {
            "offspring": [],
            "mutations": [],
            "selection_results": {},
            "diversity_injections": []
        }

        # Create offspring through crossover
        for pair in analysis.get("crossover_pairs", []):
            domain = DomainLeg(pair["domain"])
            parent1 = next((g for g in self.populations[domain] if g.genome_id == pair["parent1"]), None)
            parent2 = next((g for g in self.populations[domain] if g.genome_id == pair["parent2"]), None)

            if parent1 and parent2:
                offspring = self._crossover(parent1, parent2)
                synthesis["offspring"].append(offspring)

        # Apply mutations
        for target in analysis.get("mutation_targets", []):
            domain = DomainLeg(target["domain"])
            genome = next((g for g in self.populations[domain] if g.genome_id == target["genome_id"]), None)

            if genome:
                mutant = self._mutate(genome)
                synthesis["mutations"].append(mutant)

        # Handle diversity interventions
        for intervention in analysis.get("diversity_interventions", []):
            domain = DomainLeg(intervention["domain"])
            state = tbl.get_state(domain)

            # Inject random genome
            random_genome = self._create_genome(
                domain,
                state.state_vector + np.random.randn(64) * 0.5
            )
            synthesis["diversity_injections"].append(random_genome)

        # Record selection results
        for domain_name, elite_ids in analysis.get("selection_candidates", {}).items():
            synthesis["selection_results"][domain_name] = elite_ids

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply evolutionary changes."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        self.generation_count += 1

        # Add offspring to populations
        offspring_count = 0
        for offspring in synthesis.get("offspring", []):
            domain = offspring.domain
            offspring.fitness = self._calculate_fitness(offspring, tbl)
            self.populations[domain].append(offspring)
            offspring_count += 1

        if offspring_count > 0:
            changes["offspring_created"] = offspring_count
            insights.append(f"Created {offspring_count} offspring through crossover")

        # Add mutations
        mutation_count = 0
        for mutant in synthesis.get("mutations", []):
            domain = mutant.domain
            mutant.fitness = self._calculate_fitness(mutant, tbl)
            self.populations[domain].append(mutant)
            mutation_count += 1

        if mutation_count > 0:
            changes["mutations_applied"] = mutation_count
            insights.append(f"Applied {mutation_count} mutations")

        # Add diversity injections
        for injection in synthesis.get("diversity_injections", []):
            domain = injection.domain
            injection.fitness = self._calculate_fitness(injection, tbl)
            self.populations[domain].append(injection)
            insights.append(f"Injected diversity into {domain.value}")

        # Selection: keep only top genomes
        selected_count = 0
        for domain in DomainLeg:
            population = self.populations[domain]
            if len(population) > self.population_size:
                # Sort by fitness and keep best
                sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
                self.populations[domain] = sorted_pop[:self.population_size]
                selected_count += len(population) - self.population_size

                # Update elites
                self.elites[domain] = sorted_pop[:self.elite_count]

                # Apply best genome to domain state
                best_genome = sorted_pop[0]
                state = tbl.get_state(domain)
                # Blend current state with best genome (gradual evolution)
                state.state_vector = 0.8 * state.state_vector + 0.2 * best_genome.genes

        if selected_count > 0:
            changes["individuals_selected_out"] = selected_count
            insights.append(f"Selection removed {selected_count} less fit individuals")

        # Record fitness history
        mean_fitness = np.mean([
            np.mean([g.fitness for g in self.populations[d]])
            for d in DomainLeg
        ])
        self.fitness_history.append({
            "generation": self.generation_count,
            "mean_fitness": mean_fitness,
            "timestamp": datetime.now().isoformat()
        })

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["generation"] = self.generation_count
        metrics_delta["mean_fitness"] = mean_fitness

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
