"""
SYNAPSE-M: Modality Types

Extended sensory modality definitions for multi-modal perception.
Defines 12+ sensory modalities with their characteristics, processing
parameters, and integration weights.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod


# ============================================================================
# Extended Modality Enumeration
# ============================================================================

class ExtendedModality(Enum):
    """
    Extended sensory modalities covering full perceptual space.

    Traditional senses + digital/computational extensions.
    """
    # Traditional senses
    VISION = "vision"                     # Visual perception
    AUDITION = "audition"                 # Auditory perception
    OLFACTION = "olfaction"               # Smell
    GUSTATION = "gustation"               # Taste
    TACTILE = "tactile"                   # Touch/pressure

    # Body awareness
    PROPRIOCEPTION = "proprioception"     # Body position/movement
    VESTIBULAR = "vestibular"             # Balance/spatial orientation
    KINESTHESIA = "kinesthesia"           # Movement sense

    # Internal states
    INTEROCEPTION = "interoception"       # Internal organ states
    NOCICEPTION = "nociception"           # Pain perception
    THERMOCEPTION = "thermoception"       # Temperature perception

    # Digital/Extended senses
    ELECTROMAGNETIC = "electromagnetic"   # EM field detection
    TEMPORAL = "temporal"                 # Time perception
    LINGUISTIC = "linguistic"             # Language processing
    SOCIAL = "social"                     # Social signal detection
    METACOGNITIVE = "metacognitive"       # Self-awareness sensing


class ModalityDomain(Enum):
    """Classification of modality domains"""
    EXTEROCEPTIVE = auto()   # External environment
    INTEROCEPTIVE = auto()   # Internal body states
    PROPRIOCEPTIVE = auto()  # Body position/movement
    COGNITIVE = auto()       # Mental/computational


# ============================================================================
# Modality Characteristics
# ============================================================================

@dataclass
class ModalityCharacteristics:
    """
    Defines the processing characteristics of a sensory modality.
    """
    modality: ExtendedModality
    domain: ModalityDomain

    # Feature dimensions
    feature_dim: int = 128              # Dimension of feature vector
    semantic_dim: int = 64              # Dimension of semantic embedding

    # Temporal properties
    latency_ms: float = 50.0            # Processing latency
    temporal_resolution_ms: float = 20.0  # Minimum temporal discrimination
    persistence_ms: float = 250.0       # Sensory memory duration

    # Sensitivity
    base_sensitivity: float = 1.0       # Baseline sensitivity
    adaptation_rate: float = 0.01       # Sensory adaptation speed
    saturation_threshold: float = 1.0   # Maximum signal before saturation

    # Integration
    binding_weight: float = 1.0         # Weight in cross-modal binding
    semantic_strength: float = 1.0      # Contribution to semantic content

    # Metadata
    description: str = ""

    def __post_init__(self):
        """Set description if not provided"""
        if not self.description:
            self.description = f"{self.modality.value} modality ({self.domain.name})"


# Default characteristics for each modality
MODALITY_DEFAULTS: Dict[ExtendedModality, ModalityCharacteristics] = {
    ExtendedModality.VISION: ModalityCharacteristics(
        modality=ExtendedModality.VISION,
        domain=ModalityDomain.EXTEROCEPTIVE,
        feature_dim=256,
        semantic_dim=128,
        latency_ms=60.0,
        temporal_resolution_ms=16.7,  # ~60 Hz
        persistence_ms=100.0,         # Iconic memory
        binding_weight=1.5,           # High binding weight
        description="Visual perception with high spatial resolution"
    ),
    ExtendedModality.AUDITION: ModalityCharacteristics(
        modality=ExtendedModality.AUDITION,
        domain=ModalityDomain.EXTEROCEPTIVE,
        feature_dim=128,
        semantic_dim=64,
        latency_ms=20.0,              # Fast processing
        temporal_resolution_ms=2.0,   # High temporal resolution
        persistence_ms=2000.0,        # Echoic memory (longer)
        binding_weight=1.2,
        description="Auditory perception with high temporal resolution"
    ),
    ExtendedModality.OLFACTION: ModalityCharacteristics(
        modality=ExtendedModality.OLFACTION,
        domain=ModalityDomain.EXTEROCEPTIVE,
        feature_dim=64,
        semantic_dim=32,
        latency_ms=300.0,             # Slow processing
        temporal_resolution_ms=500.0,
        persistence_ms=5000.0,        # Long persistence
        binding_weight=0.8,
        description="Olfactory perception linked to emotion/memory"
    ),
    ExtendedModality.GUSTATION: ModalityCharacteristics(
        modality=ExtendedModality.GUSTATION,
        domain=ModalityDomain.EXTEROCEPTIVE,
        feature_dim=32,
        semantic_dim=16,
        latency_ms=200.0,
        temporal_resolution_ms=1000.0,
        persistence_ms=3000.0,
        binding_weight=0.6,
        description="Gustatory perception with emotional valence"
    ),
    ExtendedModality.TACTILE: ModalityCharacteristics(
        modality=ExtendedModality.TACTILE,
        domain=ModalityDomain.EXTEROCEPTIVE,
        feature_dim=128,
        semantic_dim=64,
        latency_ms=40.0,
        temporal_resolution_ms=10.0,
        persistence_ms=500.0,
        binding_weight=1.1,
        description="Tactile perception with spatial discrimination"
    ),
    ExtendedModality.PROPRIOCEPTION: ModalityCharacteristics(
        modality=ExtendedModality.PROPRIOCEPTION,
        domain=ModalityDomain.PROPRIOCEPTIVE,
        feature_dim=64,
        semantic_dim=32,
        latency_ms=15.0,              # Very fast
        temporal_resolution_ms=5.0,
        persistence_ms=100.0,
        binding_weight=1.0,
        description="Body position and joint angle awareness"
    ),
    ExtendedModality.VESTIBULAR: ModalityCharacteristics(
        modality=ExtendedModality.VESTIBULAR,
        domain=ModalityDomain.PROPRIOCEPTIVE,
        feature_dim=32,
        semantic_dim=16,
        latency_ms=10.0,              # Fastest
        temporal_resolution_ms=3.0,
        persistence_ms=200.0,
        binding_weight=0.9,
        description="Balance and spatial orientation"
    ),
    ExtendedModality.KINESTHESIA: ModalityCharacteristics(
        modality=ExtendedModality.KINESTHESIA,
        domain=ModalityDomain.PROPRIOCEPTIVE,
        feature_dim=64,
        semantic_dim=32,
        latency_ms=20.0,
        temporal_resolution_ms=8.0,
        persistence_ms=150.0,
        binding_weight=0.9,
        description="Sense of movement and velocity"
    ),
    ExtendedModality.INTEROCEPTION: ModalityCharacteristics(
        modality=ExtendedModality.INTEROCEPTION,
        domain=ModalityDomain.INTEROCEPTIVE,
        feature_dim=48,
        semantic_dim=24,
        latency_ms=500.0,             # Slow, diffuse
        temporal_resolution_ms=1000.0,
        persistence_ms=10000.0,
        binding_weight=0.7,
        description="Internal organ state awareness"
    ),
    ExtendedModality.NOCICEPTION: ModalityCharacteristics(
        modality=ExtendedModality.NOCICEPTION,
        domain=ModalityDomain.INTEROCEPTIVE,
        feature_dim=32,
        semantic_dim=16,
        latency_ms=25.0,              # Fast for danger
        temporal_resolution_ms=50.0,
        persistence_ms=2000.0,
        binding_weight=2.0,           # High priority
        description="Pain and tissue damage detection"
    ),
    ExtendedModality.THERMOCEPTION: ModalityCharacteristics(
        modality=ExtendedModality.THERMOCEPTION,
        domain=ModalityDomain.INTEROCEPTIVE,
        feature_dim=16,
        semantic_dim=8,
        latency_ms=100.0,
        temporal_resolution_ms=200.0,
        persistence_ms=5000.0,
        binding_weight=0.6,
        description="Temperature perception"
    ),
    ExtendedModality.ELECTROMAGNETIC: ModalityCharacteristics(
        modality=ExtendedModality.ELECTROMAGNETIC,
        domain=ModalityDomain.EXTEROCEPTIVE,
        feature_dim=64,
        semantic_dim=32,
        latency_ms=5.0,               # Digital speed
        temporal_resolution_ms=1.0,
        persistence_ms=100.0,
        binding_weight=0.8,
        description="EM field and signal detection"
    ),
    ExtendedModality.TEMPORAL: ModalityCharacteristics(
        modality=ExtendedModality.TEMPORAL,
        domain=ModalityDomain.COGNITIVE,
        feature_dim=32,
        semantic_dim=16,
        latency_ms=0.0,               # Immediate
        temporal_resolution_ms=1.0,
        persistence_ms=60000.0,       # Long memory
        binding_weight=1.0,
        description="Time perception and temporal reasoning"
    ),
    ExtendedModality.LINGUISTIC: ModalityCharacteristics(
        modality=ExtendedModality.LINGUISTIC,
        domain=ModalityDomain.COGNITIVE,
        feature_dim=512,
        semantic_dim=256,
        latency_ms=100.0,
        temporal_resolution_ms=50.0,
        persistence_ms=30000.0,
        binding_weight=1.5,           # High semantic content
        semantic_strength=2.0,
        description="Language processing and semantic understanding"
    ),
    ExtendedModality.SOCIAL: ModalityCharacteristics(
        modality=ExtendedModality.SOCIAL,
        domain=ModalityDomain.COGNITIVE,
        feature_dim=128,
        semantic_dim=64,
        latency_ms=150.0,
        temporal_resolution_ms=100.0,
        persistence_ms=20000.0,
        binding_weight=1.2,
        description="Social cue detection and interpretation"
    ),
    ExtendedModality.METACOGNITIVE: ModalityCharacteristics(
        modality=ExtendedModality.METACOGNITIVE,
        domain=ModalityDomain.COGNITIVE,
        feature_dim=256,
        semantic_dim=128,
        latency_ms=200.0,
        temporal_resolution_ms=500.0,
        persistence_ms=60000.0,
        binding_weight=1.0,
        description="Self-awareness and metacognitive monitoring"
    ),
}


def get_modality_characteristics(modality: ExtendedModality) -> ModalityCharacteristics:
    """Get characteristics for a modality"""
    return MODALITY_DEFAULTS.get(modality, ModalityCharacteristics(
        modality=modality,
        domain=ModalityDomain.EXTEROCEPTIVE
    ))


# ============================================================================
# Modality Input
# ============================================================================

@dataclass
class ModalityInput:
    """
    Raw input from a sensory modality.
    """
    modality: ExtendedModality
    raw_data: Any
    intensity: float = 1.0                    # Signal intensity
    confidence: float = 1.0                   # Signal reliability
    timestamp: float = field(default_factory=time.time)
    source_location: Optional[np.ndarray] = None  # Spatial source
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_age_ms(self) -> float:
        """Get age of input in milliseconds"""
        return (time.time() - self.timestamp) * 1000


@dataclass
class ProcessedModality:
    """
    Processed output from a modality processor.
    """
    modality: ExtendedModality
    raw_input: ModalityInput
    features: np.ndarray                      # Extracted features
    semantic_embedding: np.ndarray            # Semantic representation
    salience: float = 0.5                     # Attention-grabbing score
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Modality Registry
# ============================================================================

class ModalityRegistry:
    """
    Registry for managing available modalities and their processors.
    """

    def __init__(self):
        self._modalities: Dict[ExtendedModality, ModalityCharacteristics] = {}
        self._active: set[ExtendedModality] = set()

        # Initialize with defaults
        for modality, chars in MODALITY_DEFAULTS.items():
            self.register(modality, chars)

    def register(
        self,
        modality: ExtendedModality,
        characteristics: Optional[ModalityCharacteristics] = None
    ):
        """Register a modality with its characteristics"""
        if characteristics is None:
            characteristics = get_modality_characteristics(modality)
        self._modalities[modality] = characteristics

    def activate(self, modality: ExtendedModality):
        """Activate a modality for perception"""
        if modality in self._modalities:
            self._active.add(modality)

    def deactivate(self, modality: ExtendedModality):
        """Deactivate a modality"""
        self._active.discard(modality)

    def is_active(self, modality: ExtendedModality) -> bool:
        """Check if modality is active"""
        return modality in self._active

    def get_characteristics(self, modality: ExtendedModality) -> Optional[ModalityCharacteristics]:
        """Get characteristics for a modality"""
        return self._modalities.get(modality)

    def get_all_modalities(self) -> List[ExtendedModality]:
        """Get all registered modalities"""
        return list(self._modalities.keys())

    def get_active_modalities(self) -> List[ExtendedModality]:
        """Get active modalities"""
        return list(self._active)

    def get_by_domain(self, domain: ModalityDomain) -> List[ExtendedModality]:
        """Get modalities by domain"""
        return [
            m for m, c in self._modalities.items()
            if c.domain == domain
        ]

    def get_total_feature_dim(self) -> int:
        """Get total feature dimensions across active modalities"""
        return sum(
            self._modalities[m].feature_dim
            for m in self._active
            if m in self._modalities
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of modality registry"""
        return {
            "total_registered": len(self._modalities),
            "active_count": len(self._active),
            "active_modalities": [m.value for m in self._active],
            "domains": {
                d.name: len(self.get_by_domain(d))
                for d in ModalityDomain
            }
        }


# ============================================================================
# Modality Groups
# ============================================================================

class ModalityGroup(Enum):
    """Predefined groups of related modalities"""
    DISTANCE_SENSES = auto()    # Vision, audition
    CHEMICAL_SENSES = auto()    # Olfaction, gustation
    BODY_SENSES = auto()        # Proprioception, vestibular, kinesthesia
    INTERNAL_SENSES = auto()    # Interoception, nociception, thermoception
    DIGITAL_SENSES = auto()     # Electromagnetic, temporal
    COGNITIVE_SENSES = auto()   # Linguistic, social, metacognitive


MODALITY_GROUPS: Dict[ModalityGroup, List[ExtendedModality]] = {
    ModalityGroup.DISTANCE_SENSES: [
        ExtendedModality.VISION,
        ExtendedModality.AUDITION
    ],
    ModalityGroup.CHEMICAL_SENSES: [
        ExtendedModality.OLFACTION,
        ExtendedModality.GUSTATION
    ],
    ModalityGroup.BODY_SENSES: [
        ExtendedModality.PROPRIOCEPTION,
        ExtendedModality.VESTIBULAR,
        ExtendedModality.KINESTHESIA,
        ExtendedModality.TACTILE
    ],
    ModalityGroup.INTERNAL_SENSES: [
        ExtendedModality.INTEROCEPTION,
        ExtendedModality.NOCICEPTION,
        ExtendedModality.THERMOCEPTION
    ],
    ModalityGroup.DIGITAL_SENSES: [
        ExtendedModality.ELECTROMAGNETIC,
        ExtendedModality.TEMPORAL
    ],
    ModalityGroup.COGNITIVE_SENSES: [
        ExtendedModality.LINGUISTIC,
        ExtendedModality.SOCIAL,
        ExtendedModality.METACOGNITIVE
    ]
}


def get_group_modalities(group: ModalityGroup) -> List[ExtendedModality]:
    """Get modalities in a group"""
    return MODALITY_GROUPS.get(group, [])


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Modality Types Demo")
    print("=" * 50)

    # Create registry
    registry = ModalityRegistry()

    # Activate some modalities
    for modality in [
        ExtendedModality.VISION,
        ExtendedModality.AUDITION,
        ExtendedModality.PROPRIOCEPTION,
        ExtendedModality.LINGUISTIC
    ]:
        registry.activate(modality)

    print("\n1. Registry Summary:")
    summary = registry.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    print("\n2. Modality Characteristics:")
    for modality in registry.get_active_modalities():
        chars = registry.get_characteristics(modality)
        if chars:
            print(f"\n   {modality.value}:")
            print(f"      Feature dim: {chars.feature_dim}")
            print(f"      Latency: {chars.latency_ms}ms")
            print(f"      Persistence: {chars.persistence_ms}ms")
            print(f"      Binding weight: {chars.binding_weight}")

    print("\n3. Modality Groups:")
    for group in ModalityGroup:
        modalities = get_group_modalities(group)
        print(f"   {group.name}: {[m.value for m in modalities]}")

    print("\n4. Domain Classification:")
    for domain in ModalityDomain:
        modalities = registry.get_by_domain(domain)
        print(f"   {domain.name}: {len(modalities)} modalities")
