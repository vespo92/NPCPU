"""
NPCPU Planetary Module - GAIA-7

Provides planetary-scale consciousness coordination, global consensus
protocols, billion-agent coordination systems, and biosphere simulation.

This module implements the full GAIA-7 workstream:
- Global consciousness networks
- Biosphere simulation with climate zones
- Gaia consciousness emergence
- Biogeochemical resource cycles
- Climate-consciousness feedback loops
- Tipping point detection
- Population migration patterns
- Geological timescale memory
- Mass extinction modeling
"""

# Global Consciousness Network
from .global_consciousness import (
    PlanetaryConsciousnessNetwork,
    RegionalConsciousnessNetwork,
    GlobalConsensusProtocol,
    EmergencyCoordinator,
    Knowledge,
    Problem,
    Solution,
    Emergency
)

# Biosphere Simulation
from .biosphere import (
    Biosphere,
    BiosphereConfig,
    BiomeRegion,
    BiomeType,
    ClimateZone,
    EcosystemHealth,
    AtmosphericComposition,
    OceanState
)

# Gaia Consciousness
from .gaia_consciousness import (
    GaiaConsciousness,
    GaiaConfig,
    GaiaAwarenessLevel,
    PlanetaryIntention,
    EmergenceType,
    PlanetaryPerception,
    PlanetaryThought,
    PlanetaryAction
)

# Resource Cycles
from .resource_cycles import (
    ResourceCycle,
    CarbonCycle,
    NitrogenCycle,
    WaterCycle,
    PlanetaryCycles,
    CycleConfig,
    CycleType,
    Reservoir,
    Flux,
    ReservoirType,
    FluxType
)

# Climate Feedback
from .climate_feedback import (
    ClimateFeedbackSystem,
    FeedbackConfig,
    FeedbackLoop,
    FeedbackType,
    FeedbackDirection,
    ClimateStressor,
    ConsciousnessResponse,
    ClimateState,
    ConsciousnessState
)

# Tipping Points
from .tipping_points import (
    TippingPointDetector,
    TippingPointConfig,
    TippingElement,
    TippingElementType,
    TippingState,
    WarningSignal,
    CascadeEvent
)

# Migration Patterns
from .migration_patterns import (
    MigrationSystem,
    MigrationConfig,
    MigrationRegion,
    MigrationFlow,
    MigrationDriver,
    MigrationPattern,
    SpeciesMigration
)

# Planetary Memory
from .planetary_memory import (
    PlanetaryMemory,
    MemoryConfig,
    PlanetaryMemoryItem,
    MemoryType,
    MemoryPersistence,
    GeologicalEra,
    Epoch
)

# Extinction Events
from .extinction_events import (
    ExtinctionSystem,
    ExtinctionConfig,
    ExtinctionEvent,
    ExtinctionCause,
    ExtinctionSeverity,
    RecoveryPhase,
    SelectivityPattern,
    Species
)

__all__ = [
    # Global Consciousness
    "PlanetaryConsciousnessNetwork",
    "RegionalConsciousnessNetwork",
    "GlobalConsensusProtocol",
    "EmergencyCoordinator",
    "Knowledge",
    "Problem",
    "Solution",
    "Emergency",

    # Biosphere
    "Biosphere",
    "BiosphereConfig",
    "BiomeRegion",
    "BiomeType",
    "ClimateZone",
    "EcosystemHealth",
    "AtmosphericComposition",
    "OceanState",

    # Gaia Consciousness
    "GaiaConsciousness",
    "GaiaConfig",
    "GaiaAwarenessLevel",
    "PlanetaryIntention",
    "EmergenceType",
    "PlanetaryPerception",
    "PlanetaryThought",
    "PlanetaryAction",

    # Resource Cycles
    "ResourceCycle",
    "CarbonCycle",
    "NitrogenCycle",
    "WaterCycle",
    "PlanetaryCycles",
    "CycleConfig",
    "CycleType",
    "Reservoir",
    "Flux",
    "ReservoirType",
    "FluxType",

    # Climate Feedback
    "ClimateFeedbackSystem",
    "FeedbackConfig",
    "FeedbackLoop",
    "FeedbackType",
    "FeedbackDirection",
    "ClimateStressor",
    "ConsciousnessResponse",
    "ClimateState",
    "ConsciousnessState",

    # Tipping Points
    "TippingPointDetector",
    "TippingPointConfig",
    "TippingElement",
    "TippingElementType",
    "TippingState",
    "WarningSignal",
    "CascadeEvent",

    # Migration Patterns
    "MigrationSystem",
    "MigrationConfig",
    "MigrationRegion",
    "MigrationFlow",
    "MigrationDriver",
    "MigrationPattern",
    "SpeciesMigration",

    # Planetary Memory
    "PlanetaryMemory",
    "MemoryConfig",
    "PlanetaryMemoryItem",
    "MemoryType",
    "MemoryPersistence",
    "GeologicalEra",
    "Epoch",

    # Extinction Events
    "ExtinctionSystem",
    "ExtinctionConfig",
    "ExtinctionEvent",
    "ExtinctionCause",
    "ExtinctionSeverity",
    "RecoveryPhase",
    "SelectivityPattern",
    "Species"
]
