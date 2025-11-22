"""
NPCPU Organism Module

Core systems for digital living organisms including lifecycle management,
metabolism, homeostasis, and the complete integrated digital body.

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
from .digital_body import (
    DigitalBody,
    OrganismState,
    ConsciousnessLevel,
    OrganismIdentity,
    VitalSigns
)

__all__ = [
    # Lifecycle
    "OrganismLifecycle",
    "LifecycleStage",
    "LifecycleEvent",
    "GrowthPattern",
    "DeathCause",
    # Metabolism
    "Metabolism",
    "Resource",
    "ResourceType",
    "EnergyState",
    "MetabolicProcess",
    # Homeostasis
    "HomeostasisController",
    "VitalSign",
    "Setpoint",
    "RegulationResponse",
    "StressResponse",
    # Digital Body (complete organism)
    "DigitalBody",
    "OrganismState",
    "ConsciousnessLevel",
    "OrganismIdentity",
    "VitalSigns"
]
