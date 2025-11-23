"""
NPCPU Immune Module - SENTINEL-K

Comprehensive digital immune system for threat detection, defense, and self-repair.
Protects the organism from corruption, attacks, and degradation.

Components:
- Defense: Core immune system for threat detection and response
- Repair: Self-repair system with regeneration cascades
- Pathogen Detection: Advanced anomaly detection
- Antibody System: Adaptive response generation
- Innate Immunity: Fast, non-specific defense
- Adaptive Immunity: Learning-based threat response
- Memory Cells: Long-term threat memory storage
- Autoimmune Prevention: Self-attack prevention
- Inflammation: System-wide alert propagation
"""

# Core Defense
from .defense import (
    ImmuneSystem,
    Threat,
    ThreatType,
    ThreatSeverity,
    DefenseAction,
    DefenseResponse,
    AntibodyPattern,
    QuarantineZone
)

# Repair System
from .repair import (
    RepairSystem,
    Damage,
    DamageType,
    RepairStrategy,
    HealingPhase,
    HealingProcess,
    IntegrityCheck,
    BackupState,
    RegenerationCascade,
    HealingCascade,
    TissueRegenerator,
    TissueState
)

# Pathogen Detection
from .pathogen_detection import (
    PathogenDetector,
    PathogenType,
    DetectionMethod,
    RiskLevel,
    PathogenSignature,
    DetectedPathogen,
    ScanResult,
    AnomalyProfile
)

# Antibody System
from .antibody_system import (
    AntibodyGenerator,
    Antibody,
    AntibodyState,
    AntibodyResponse,
    Antigen,
    ResponseType,
    BindingStrength,
    ClonalExpansion
)

# Innate Immunity
from .innate_immunity import (
    InnateImmuneSystem,
    Barrier,
    BarrierType,
    PatternRecognitionReceptor,
    PRRType,
    NaturalKillerCell,
    NKCellState,
    InnateResponse,
    InnateResponseType
)

# Adaptive Immunity
from .adaptive_immunity import (
    AdaptiveImmuneSystem,
    TCell,
    TCellType,
    BCell,
    BCellType,
    AdaptiveResponse,
    ImmunePhase,
    ResponseStrength,
    ImmuneMemory
)

# Memory Cells
from .memory_cells import (
    MemoryCellSystem,
    ThreatMemory,
    MemoryType,
    MemoryStrength,
    ConsolidationState,
    MemoryCluster,
    RecallResult,
    MemoryTransfer
)

# Autoimmune Prevention
from .autoimmune import (
    AutoimmunePreventionSystem,
    SelfMarker,
    SelfMarkerType,
    ToleranceRecord,
    ToleranceType,
    AutoimmuneEvent,
    AutoimmuneRisk,
    SuppressionMethod,
    RegulatoryCell
)

# Inflammation
from .inflammation import (
    InflammationSystem,
    InflammationLevel,
    InflammationPhase,
    InflammationZone,
    InflammatorySignal,
    Cytokine,
    CytokineType,
    AlertPriority,
    ResolutionEvent
)

__all__ = [
    # Defense
    "ImmuneSystem",
    "Threat",
    "ThreatType",
    "ThreatSeverity",
    "DefenseAction",
    "DefenseResponse",
    "AntibodyPattern",
    "QuarantineZone",

    # Repair
    "RepairSystem",
    "Damage",
    "DamageType",
    "RepairStrategy",
    "HealingPhase",
    "HealingProcess",
    "IntegrityCheck",
    "BackupState",
    "RegenerationCascade",
    "HealingCascade",
    "TissueRegenerator",
    "TissueState",

    # Pathogen Detection
    "PathogenDetector",
    "PathogenType",
    "DetectionMethod",
    "RiskLevel",
    "PathogenSignature",
    "DetectedPathogen",
    "ScanResult",
    "AnomalyProfile",

    # Antibody System
    "AntibodyGenerator",
    "Antibody",
    "AntibodyState",
    "AntibodyResponse",
    "Antigen",
    "ResponseType",
    "BindingStrength",
    "ClonalExpansion",

    # Innate Immunity
    "InnateImmuneSystem",
    "Barrier",
    "BarrierType",
    "PatternRecognitionReceptor",
    "PRRType",
    "NaturalKillerCell",
    "NKCellState",
    "InnateResponse",
    "InnateResponseType",

    # Adaptive Immunity
    "AdaptiveImmuneSystem",
    "TCell",
    "TCellType",
    "BCell",
    "BCellType",
    "AdaptiveResponse",
    "ImmunePhase",
    "ResponseStrength",
    "ImmuneMemory",

    # Memory Cells
    "MemoryCellSystem",
    "ThreatMemory",
    "MemoryType",
    "MemoryStrength",
    "ConsolidationState",
    "MemoryCluster",
    "RecallResult",
    "MemoryTransfer",

    # Autoimmune Prevention
    "AutoimmunePreventionSystem",
    "SelfMarker",
    "SelfMarkerType",
    "ToleranceRecord",
    "ToleranceType",
    "AutoimmuneEvent",
    "AutoimmuneRisk",
    "SuppressionMethod",
    "RegulatoryCell",

    # Inflammation
    "InflammationSystem",
    "InflammationLevel",
    "InflammationPhase",
    "InflammationZone",
    "InflammatorySignal",
    "Cytokine",
    "CytokineType",
    "AlertPriority",
    "ResolutionEvent",
]
