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

__all__ = [
    "Population",
    "PopulationDynamics",
    "InteractionType",
    "RelationshipType",
    "SocialNetwork",
    "World",
    "WorldConfig",
    "Region",
    "ResourcePool",
    "WorldEvent"
]
