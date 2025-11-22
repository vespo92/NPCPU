"""
NPCPU Organism Module

Core systems for digital living organisms including lifecycle management,
metabolism, homeostasis, and organic growth patterns.

This module transforms NPCPU from a collection of components into
a unified living digital organism.
"""

from .lifecycle import (
    OrganismLifecycle,
    LifecycleStage,
    LifecycleEvent,
    GrowthPattern,
    DeathCause
)
from .metabolism import (
    Metabolism,
    Resource,
    ResourceType,
    EnergyState,
    MetabolicProcess
)
from .homeostasis import (
    HomeostasisController,
    VitalSign,
    Setpoint,
    RegulationResponse,
    StressResponse
)

__all__ = [
    "OrganismLifecycle",
    "LifecycleStage",
    "LifecycleEvent",
    "GrowthPattern",
    "DeathCause",
    "Metabolism",
    "Resource",
    "ResourceType",
    "EnergyState",
    "MetabolicProcess",
    "HomeostasisController",
    "VitalSign",
    "Setpoint",
    "RegulationResponse",
    "StressResponse"
]
