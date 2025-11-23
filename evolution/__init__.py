"""
NPCPU Evolution Module

Provides genetic algorithm-based optimization for consciousness models
and organism traits. Enables task-specific optimization and adaptive agent design.

Core Components:
- ConsciousnessEvolutionEngine: Evolve consciousness parameters
- EvolutionEngine: Evolve organism traits via genetic algorithms
- Genome: Genetic representation with dominant/recessive alleles
- Mutation operators: Gaussian, uniform, adaptive, polynomial, etc.

GENESIS-E Advanced Evolution Components:
- HorizontalTransferEngine: Non-vertical gene inheritance
- EpigeneticLayer: Epigenetic inheritance and environmental imprinting
- SpeciationEngine: Species formation and divergence
- SexualSelectionEngine: Mate choice and competition
- ArmsRaceEngine: Co-evolutionary dynamics
- NicheConstructionEngine: Organism-environment feedback
- InnovationEngine: Major evolutionary transitions
- ExtinctionEngine: Extinction and recovery dynamics
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

# GENESIS-E: Advanced Evolution Components

from .horizontal_transfer import (
    HorizontalTransferEngine,
    TransferType,
    TransferResult,
    GeneticElement,
    GenePool,
    TransferRecord,
    CompatibilityChecker,
    create_hgt_engine
)

from .epigenetics import (
    EpigeneticLayer,
    EpigeneticMark,
    EpigeneticMarkType,
    ExpressionModifier,
    EnvironmentalStressor,
    Epigenome,
    create_common_stressors
)

from .speciation import (
    SpeciationEngine,
    SpeciationType,
    IsolationType,
    IsolationBarrier,
    SpeciesRecord,
    HybridZone
)

from .sexual_selection import (
    SexualSelectionEngine,
    SelectionType,
    PreferenceType,
    MatePreference,
    Ornament,
    CompetitionResult,
    create_peacock_selection,
    create_combat_selection
)

from .evolutionary_arms_race import (
    ArmsRaceEngine,
    CompetitorRole,
    InteractionType,
    TraitPair,
    ArmsRaceRelationship,
    InteractionRecord
)

from .niche_construction import (
    NicheConstructionEngine,
    ConstructionType,
    EffectPersistence,
    EnvironmentalVariable,
    ConstructionBehavior,
    ConstructedNiche,
    SelectionFeedback,
    Environment,
    create_beaver_model
)

from .evolutionary_innovation import (
    InnovationEngine,
    InnovationType,
    TransitionType,
    InnovationPathway,
    InnovationEvent,
    AdaptiveRadiation,
    create_standard_pathways
)

from .extinction_dynamics import (
    ExtinctionEngine,
    ExtinctionType,
    RecoveryPhase,
    ExtinctionEvent,
    PopulationBottleneck,
    RecoveryState,
    create_stable_world,
    create_volatile_world
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
    "create_adaptive_mutator",

    # GENESIS-E: Horizontal Gene Transfer
    "HorizontalTransferEngine",
    "TransferType",
    "TransferResult",
    "GeneticElement",
    "GenePool",
    "TransferRecord",
    "CompatibilityChecker",
    "create_hgt_engine",

    # GENESIS-E: Epigenetics
    "EpigeneticLayer",
    "EpigeneticMark",
    "EpigeneticMarkType",
    "ExpressionModifier",
    "EnvironmentalStressor",
    "Epigenome",
    "create_common_stressors",

    # GENESIS-E: Speciation
    "SpeciationEngine",
    "SpeciationType",
    "IsolationType",
    "IsolationBarrier",
    "SpeciesRecord",
    "HybridZone",

    # GENESIS-E: Sexual Selection
    "SexualSelectionEngine",
    "SelectionType",
    "PreferenceType",
    "MatePreference",
    "Ornament",
    "CompetitionResult",
    "create_peacock_selection",
    "create_combat_selection",

    # GENESIS-E: Evolutionary Arms Race
    "ArmsRaceEngine",
    "CompetitorRole",
    "InteractionType",
    "TraitPair",
    "ArmsRaceRelationship",
    "InteractionRecord",

    # GENESIS-E: Niche Construction
    "NicheConstructionEngine",
    "ConstructionType",
    "EffectPersistence",
    "EnvironmentalVariable",
    "ConstructionBehavior",
    "ConstructedNiche",
    "SelectionFeedback",
    "Environment",
    "create_beaver_model",

    # GENESIS-E: Evolutionary Innovation
    "InnovationEngine",
    "InnovationType",
    "TransitionType",
    "InnovationPathway",
    "InnovationEvent",
    "AdaptiveRadiation",
    "create_standard_pathways",

    # GENESIS-E: Extinction Dynamics
    "ExtinctionEngine",
    "ExtinctionType",
    "RecoveryPhase",
    "ExtinctionEvent",
    "PopulationBottleneck",
    "RecoveryState",
    "create_stable_world",
    "create_volatile_world"
]
