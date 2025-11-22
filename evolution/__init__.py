"""
NPCPU Evolution Module

Provides genetic algorithm-based optimization for consciousness models
and organism traits. Enables task-specific optimization and adaptive agent design.

Includes:
- ConsciousnessEvolutionEngine: Evolve consciousness parameters
- EvolutionEngine: Evolve organism traits via genetic algorithms
- Genome: Genetic representation with dominant/recessive alleles
- Mutation operators: Gaussian, uniform, adaptive, polynomial, etc.
"""

from .consciousness_evolution import (
    ConsciousnessEvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    FitnessFunctions
)

from .genetic_engine import (
    Genome,
    Gene,
    Allele,
    Species,
    Individual,
    EvolutionEngine,
    SelectionStrategy,
    CrossoverMethod,
    compose_fitness,
    trait_fitness,
    balanced_fitness
)

from .mutations import (
    MutationType,
    MutationRecord,
    MutationEngine,
    BaseMutator,
    GaussianMutator,
    UniformMutator,
    BoundaryMutator,
    PolynomialMutator,
    PointMutator,
    CompositeMutator,
    AdaptiveMutator,
    apply_gaussian_mutation,
    apply_uniform_mutation,
    create_adaptive_mutator
)

__all__ = [
    # Consciousness evolution
    "ConsciousnessEvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "FitnessFunctions",

    # Genetic evolution
    "Genome",
    "Gene",
    "Allele",
    "Species",
    "Individual",
    "EvolutionEngine",
    "SelectionStrategy",
    "CrossoverMethod",
    "compose_fitness",
    "trait_fitness",
    "balanced_fitness",

    # Mutations
    "MutationType",
    "MutationRecord",
    "MutationEngine",
    "BaseMutator",
    "GaussianMutator",
    "UniformMutator",
    "BoundaryMutator",
    "PolynomialMutator",
    "PointMutator",
    "CompositeMutator",
    "AdaptiveMutator",
    "apply_gaussian_mutation",
    "apply_uniform_mutation",
    "create_adaptive_mutator"
]
