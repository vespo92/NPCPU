"""
NPCPU Immune Module

Digital immune system for threat detection, defense, and self-repair.
Protects the organism from corruption, attacks, and degradation.
"""

from .defense import (
    ImmuneSystem,
    Threat,
    ThreatType,
    ThreatSeverity,
    DefenseResponse,
    AntibodyPattern,
    QuarantineZone
)
from .repair import (
    RepairSystem,
    Damage,
    DamageType,
    RepairStrategy,
    HealingProcess,
    IntegrityCheck
)

__all__ = [
    "ImmuneSystem",
    "Threat",
    "ThreatType",
    "ThreatSeverity",
    "DefenseResponse",
    "AntibodyPattern",
    "QuarantineZone",
    "RepairSystem",
    "Damage",
    "DamageType",
    "RepairStrategy",
    "HealingProcess",
    "IntegrityCheck"
]
