"""
Mutation Operators for Genetic Evolution

Provides various mutation strategies for evolving organism genomes.

Features:
- Point mutations (single gene changes)
- Segment mutations (multiple contiguous genes)
- Gene duplication and deletion
- Adaptive mutation rates
- Mutation tracking and history
"""

from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import math
import copy

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.events import get_event_bus


class MutationType(Enum):
    """Types of mutations that can occur"""
    POINT = auto()          # Single gene value change
    SWAP = auto()           # Swap two genes
    INVERSION = auto()      # Reverse a segment
    DUPLICATION = auto()    # Duplicate a gene
    DELETION = auto()       # Delete a gene
    INSERTION = auto()      # Insert new gene
    GAUSSIAN = auto()       # Add Gaussian noise
    UNIFORM = auto()        # Uniform random replacement
    BOUNDARY = auto()       # Push to boundary values
    POLYNOMIAL = auto()     # Polynomial mutation


@dataclass
class MutationRecord:
    """Record of a mutation event"""
    mutation_type: MutationType
    gene_name: str
    old_value: Any
    new_value: Any
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.mutation_type.name,
            "gene": self.gene_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "generation": self.generation
        }


class BaseMutator(ABC):
    """
    Abstract base class for mutation operators.

    Mutators modify genome values to introduce genetic variation.

    Example:
        mutator = GaussianMutator(sigma=0.1)
        new_value = mutator.mutate(0.5, "speed")
    """

    @abstractmethod
    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        """
        Apply mutation to a gene value.

        Args:
            value: Current gene value (typically 0.0-1.0)
            gene_name: Name of the gene being mutated

        Returns:
            Tuple of (new_value, mutation_type)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this mutator"""
        pass


class GaussianMutator(BaseMutator):
    """
    Gaussian (normal distribution) mutation.

    Adds normally distributed noise to gene values.
    Good for fine-tuning continuous traits.
    """

    def __init__(self, sigma: float = 0.1, adaptive: bool = False):
        """
        Args:
            sigma: Standard deviation of Gaussian noise
            adaptive: If True, sigma adapts based on gene value distance from bounds
        """
        self.sigma = sigma
        self.adaptive = adaptive

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        if self.adaptive:
            # Reduce sigma near boundaries to avoid getting stuck
            distance_from_boundary = min(value, 1.0 - value)
            effective_sigma = self.sigma * (0.5 + distance_from_boundary)
        else:
            effective_sigma = self.sigma

        noise = random.gauss(0, effective_sigma)
        new_value = max(0.0, min(1.0, value + noise))
        return new_value, MutationType.GAUSSIAN

    def get_name(self) -> str:
        return f"Gaussian(σ={self.sigma})"


class UniformMutator(BaseMutator):
    """
    Uniform random mutation.

    Replaces gene value with random value from uniform distribution.
    Good for escaping local optima.
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        new_value = random.uniform(self.min_val, self.max_val)
        return new_value, MutationType.UNIFORM

    def get_name(self) -> str:
        return f"Uniform({self.min_val}-{self.max_val})"


class BoundaryMutator(BaseMutator):
    """
    Boundary mutation.

    Pushes gene value to either minimum or maximum boundary.
    Useful for exploring extreme phenotypes.
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        # Choose closer boundary with 70% probability, farther with 30%
        dist_to_min = abs(value - self.min_val)
        dist_to_max = abs(value - self.max_val)

        if dist_to_min < dist_to_max:
            new_value = self.max_val if random.random() < 0.7 else self.min_val
        else:
            new_value = self.min_val if random.random() < 0.7 else self.max_val

        return new_value, MutationType.BOUNDARY

    def get_name(self) -> str:
        return "Boundary"


class PolynomialMutator(BaseMutator):
    """
    Polynomial mutation (commonly used in NSGA-II).

    Creates a non-uniform perturbation that favors small changes
    but allows occasional large changes.
    """

    def __init__(self, eta: float = 20.0):
        """
        Args:
            eta: Distribution index (larger = smaller mutations)
        """
        self.eta = eta

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        u = random.random()

        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (self.eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta + 1.0))

        new_value = max(0.0, min(1.0, value + delta))
        return new_value, MutationType.POLYNOMIAL

    def get_name(self) -> str:
        return f"Polynomial(η={self.eta})"


class PointMutator(BaseMutator):
    """
    Point mutation with configurable step size.

    Makes discrete step changes to gene values.
    """

    def __init__(self, step_size: float = 0.1):
        self.step_size = step_size

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        direction = random.choice([-1, 1])
        new_value = max(0.0, min(1.0, value + direction * self.step_size))
        return new_value, MutationType.POINT

    def get_name(self) -> str:
        return f"Point(step={self.step_size})"


class CompositeMutator(BaseMutator):
    """
    Combines multiple mutators with configurable weights.

    Randomly selects a mutator based on weights for each mutation.

    Example:
        mutator = CompositeMutator([
            (GaussianMutator(0.1), 0.6),
            (UniformMutator(), 0.3),
            (BoundaryMutator(), 0.1)
        ])
    """

    def __init__(self, mutators: List[Tuple[BaseMutator, float]]):
        """
        Args:
            mutators: List of (mutator, weight) tuples
        """
        self.mutators = mutators
        total_weight = sum(w for _, w in mutators)
        self.weights = [w / total_weight for _, w in mutators]

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        mutator = random.choices(
            [m for m, _ in self.mutators],
            weights=self.weights,
            k=1
        )[0]
        return mutator.mutate(value, gene_name)

    def get_name(self) -> str:
        return "Composite"


class AdaptiveMutator(BaseMutator):
    """
    Adaptive mutation that adjusts behavior based on evolution progress.

    - Early generations: More exploration (larger mutations)
    - Later generations: More exploitation (smaller mutations)
    """

    def __init__(
        self,
        initial_sigma: float = 0.3,
        final_sigma: float = 0.05,
        decay_generations: int = 100
    ):
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_generations = decay_generations
        self.current_generation = 0

    def set_generation(self, generation: int) -> None:
        """Update current generation for adaptive behavior"""
        self.current_generation = generation

    def mutate(self, value: float, gene_name: str) -> Tuple[float, MutationType]:
        # Calculate adaptive sigma using exponential decay
        progress = min(1.0, self.current_generation / self.decay_generations)
        sigma = self.initial_sigma * (1 - progress) + self.final_sigma * progress

        noise = random.gauss(0, sigma)
        new_value = max(0.0, min(1.0, value + noise))
        return new_value, MutationType.GAUSSIAN

    def get_name(self) -> str:
        return f"Adaptive({self.initial_sigma}→{self.final_sigma})"


class MutationEngine:
    """
    Engine for applying mutations to genomes.

    Features:
    - Configurable mutation rate per gene
    - Multiple mutation strategies
    - Mutation tracking and history
    - Gene-specific mutation rules

    Example:
        engine = MutationEngine(
            base_rate=0.1,
            mutator=GaussianMutator(sigma=0.15)
        )

        genome = {"speed": 0.5, "strength": 0.7, "intelligence": 0.3}
        mutated, records = engine.mutate_genome(genome, generation=10)
    """

    def __init__(
        self,
        base_rate: float = 0.1,
        mutator: Optional[BaseMutator] = None,
        gene_rates: Optional[Dict[str, float]] = None,
        gene_mutators: Optional[Dict[str, BaseMutator]] = None
    ):
        """
        Args:
            base_rate: Default mutation probability per gene
            mutator: Default mutator to use
            gene_rates: Override mutation rates for specific genes
            gene_mutators: Override mutators for specific genes
        """
        self.base_rate = base_rate
        self.default_mutator = mutator or GaussianMutator(sigma=0.1)
        self.gene_rates = gene_rates or {}
        self.gene_mutators = gene_mutators or {}
        self.mutation_history: List[MutationRecord] = []
        self.total_mutations = 0

    def get_mutation_rate(self, gene_name: str) -> float:
        """Get mutation rate for a specific gene"""
        return self.gene_rates.get(gene_name, self.base_rate)

    def get_mutator(self, gene_name: str) -> BaseMutator:
        """Get mutator for a specific gene"""
        return self.gene_mutators.get(gene_name, self.default_mutator)

    def mutate_gene(
        self,
        gene_name: str,
        value: float,
        generation: int = 0
    ) -> Tuple[float, Optional[MutationRecord]]:
        """
        Potentially mutate a single gene.

        Args:
            gene_name: Name of the gene
            value: Current gene value
            generation: Current generation number

        Returns:
            Tuple of (new_value, mutation_record or None)
        """
        rate = self.get_mutation_rate(gene_name)

        if random.random() < rate:
            mutator = self.get_mutator(gene_name)
            new_value, mutation_type = mutator.mutate(value, gene_name)

            record = MutationRecord(
                mutation_type=mutation_type,
                gene_name=gene_name,
                old_value=value,
                new_value=new_value,
                generation=generation
            )

            self.mutation_history.append(record)
            self.total_mutations += 1

            # Emit mutation event
            bus = get_event_bus()
            bus.emit("evolution.mutation", {
                "gene": gene_name,
                "type": mutation_type.name,
                "old_value": value,
                "new_value": new_value,
                "generation": generation
            })

            return new_value, record

        return value, None

    def mutate_genome(
        self,
        genome: Dict[str, float],
        generation: int = 0
    ) -> Tuple[Dict[str, float], List[MutationRecord]]:
        """
        Apply mutations to an entire genome.

        Args:
            genome: Dictionary mapping gene names to values
            generation: Current generation number

        Returns:
            Tuple of (mutated_genome, list of mutation records)
        """
        mutated = {}
        records = []

        for gene_name, value in genome.items():
            new_value, record = self.mutate_gene(gene_name, value, generation)
            mutated[gene_name] = new_value
            if record:
                records.append(record)

        return mutated, records

    def forced_mutate(
        self,
        genome: Dict[str, float],
        num_mutations: int = 1,
        generation: int = 0
    ) -> Tuple[Dict[str, float], List[MutationRecord]]:
        """
        Force a specific number of mutations on the genome.

        Args:
            genome: Dictionary mapping gene names to values
            num_mutations: Number of mutations to apply
            generation: Current generation number

        Returns:
            Tuple of (mutated_genome, list of mutation records)
        """
        mutated = genome.copy()
        records = []
        gene_names = list(genome.keys())

        # Select genes to mutate
        genes_to_mutate = random.sample(
            gene_names,
            min(num_mutations, len(gene_names))
        )

        for gene_name in genes_to_mutate:
            mutator = self.get_mutator(gene_name)
            new_value, mutation_type = mutator.mutate(mutated[gene_name], gene_name)

            record = MutationRecord(
                mutation_type=mutation_type,
                gene_name=gene_name,
                old_value=mutated[gene_name],
                new_value=new_value,
                generation=generation
            )

            mutated[gene_name] = new_value
            records.append(record)
            self.mutation_history.append(record)
            self.total_mutations += 1

        return mutated, records

    def get_mutation_stats(self) -> Dict[str, Any]:
        """Get statistics about mutations applied"""
        if not self.mutation_history:
            return {
                "total_mutations": 0,
                "by_type": {},
                "by_gene": {},
                "avg_magnitude": 0
            }

        by_type: Dict[str, int] = {}
        by_gene: Dict[str, int] = {}
        magnitudes = []

        for record in self.mutation_history:
            type_name = record.mutation_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1
            by_gene[record.gene_name] = by_gene.get(record.gene_name, 0) + 1

            if isinstance(record.old_value, (int, float)) and isinstance(record.new_value, (int, float)):
                magnitudes.append(abs(record.new_value - record.old_value))

        return {
            "total_mutations": self.total_mutations,
            "by_type": by_type,
            "by_gene": by_gene,
            "avg_magnitude": sum(magnitudes) / len(magnitudes) if magnitudes else 0
        }

    def clear_history(self) -> None:
        """Clear mutation history"""
        self.mutation_history.clear()


# Convenience functions for common mutation operations

def apply_gaussian_mutation(
    genome: Dict[str, float],
    rate: float = 0.1,
    sigma: float = 0.1
) -> Dict[str, float]:
    """
    Apply Gaussian mutation to a genome.

    Example:
        genome = {"speed": 0.5, "strength": 0.7}
        mutated = apply_gaussian_mutation(genome, rate=0.15, sigma=0.1)
    """
    engine = MutationEngine(base_rate=rate, mutator=GaussianMutator(sigma=sigma))
    mutated, _ = engine.mutate_genome(genome)
    return mutated


def apply_uniform_mutation(
    genome: Dict[str, float],
    rate: float = 0.05
) -> Dict[str, float]:
    """
    Apply uniform random mutation to a genome.

    Example:
        genome = {"speed": 0.5, "strength": 0.7}
        mutated = apply_uniform_mutation(genome, rate=0.1)
    """
    engine = MutationEngine(base_rate=rate, mutator=UniformMutator())
    mutated, _ = engine.mutate_genome(genome)
    return mutated


def create_adaptive_mutator(
    exploration_rate: float = 0.3,
    exploitation_rate: float = 0.05,
    transition_generations: int = 100
) -> AdaptiveMutator:
    """
    Create an adaptive mutator that transitions from exploration to exploitation.

    Example:
        mutator = create_adaptive_mutator(
            exploration_rate=0.3,
            exploitation_rate=0.05,
            transition_generations=50
        )
        mutator.set_generation(25)  # Halfway through transition
    """
    return AdaptiveMutator(
        initial_sigma=exploration_rate,
        final_sigma=exploitation_rate,
        decay_generations=transition_generations
    )
