"""
NPCPU Ecosystem Module

Multi-organism ecosystem simulation enabling populations of
digital organisms to interact, compete, cooperate, and evolve.
"""

from .population import (
    Population,
    PopulationDynamics,
    InteractionType,
    RelationshipType,
    SocialNetwork
)

from .world import (
    World,
    WorldConfig,
    Region,
    ResourcePool,
    WorldEvent
)

from .world_generator import (
    WorldGenerator,
    ProceduralWorld,
    BiomeType,
    Biome,
    HazardType,
    Hazard,
    ResourceType,
    Resource,
    TerrainCell,
    SimplexNoise,
    BIOME_DEFINITIONS,
    BIOME_RESOURCES,
    BIOME_HAZARDS,
)

__all__ = [
    # Population
    "Population",
    "PopulationDynamics",
    "InteractionType",
    "RelationshipType",
    "SocialNetwork",
    # World
    "World",
    "WorldConfig",
    "Region",
    "ResourcePool",
    "WorldEvent",
    # Procedural Generation
    "WorldGenerator",
    "ProceduralWorld",
    "BiomeType",
    "Biome",
    "HazardType",
    "Hazard",
    "ResourceType",
    "Resource",
    "TerrainCell",
    "SimplexNoise",
    "BIOME_DEFINITIONS",
    "BIOME_RESOURCES",
    "BIOME_HAZARDS",
]
