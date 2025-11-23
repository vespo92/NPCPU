"""
NPCPU Sensory System - SYNAPSE-M Multi-Modal Perception Engine

Comprehensive perception and environment sensing for digital organisms.
Enables organisms to sense their environment across multiple modalities
and integrate information for unified conscious experience.

Components:
- modality_types: Extended sensory modality definitions (12+)
- perception: Core sensory system and attention mechanisms
- cross_modal: Cross-modal binding and integration
- attention_filter: Selective attention mechanisms
- predictive_perception: Prediction-based perception
- sensory_memory: Iconic/echoic memory buffers
- gestalt_processing: Pattern completion and grouping
- proprioception: Body state sensing
- interoception: Internal organ awareness
- environment: Environment and stimulus handling
"""

# Core perception system
from .perception import (
    SensorySystem,
    Sensor,
    SensorType,
    SensoryInput,
    PerceptionFilter,
    AttentionMechanism,
    SensoryModality,
    AttentionFocus,
    SensoryPriority,
)

# Environment
from .environment import (
    Environment,
    EnvironmentState,
    Stimulus,
    StimulusType,
    EnvironmentEvent
)

# SYNAPSE-M: Modality Types
from .modality_types import (
    ExtendedModality,
    ModalityDomain,
    ModalityCharacteristics,
    ModalityRegistry,
    ModalityInput,
    ProcessedModality,
    ModalityGroup,
    get_modality_characteristics,
    get_group_modalities,
    MODALITY_DEFAULTS,
    MODALITY_GROUPS,
)

# SYNAPSE-M: Cross-Modal Binding
from .cross_modal import (
    CrossModalIntegrator,
    SensoryFusion,
    BoundPercept,
    BindingPair,
    BindingType,
    BindingStrength,
    TemporalBindingMechanism,
    SpatialBindingMechanism,
    SemanticBindingMechanism,
)

# SYNAPSE-M: Attention Filtering
from .attention_filter import (
    AttentionFilter,
    MultiModalAttentionManager,
    SaliencyComputer,
    SaliencyMap,
    AttentionWeight,
    AttentionType,
    AttentionPriority,
)

# SYNAPSE-M: Predictive Perception
from .predictive_perception import (
    PredictivePerceptionSystem,
    Prediction,
    PredictionError,
    PredictiveState,
    PredictiveModel,
    TemporalPredictiveModel,
    CrossModalPredictiveModel,
    PredictionType,
    PredictionConfidence,
)

# SYNAPSE-M: Sensory Memory
from .sensory_memory import (
    SensoryMemorySystem,
    SensoryMemoryBuffer,
    IconicMemory,
    EchoicMemory,
    HapticMemory,
    MemoryTrace,
    SensorySnapshot,
    MemoryType,
    DecayModel,
)

# SYNAPSE-M: Gestalt Processing
from .gestalt_processing import (
    GestaltProcessor,
    GestaltPrinciple,
    PerceptualGroup,
    CompletedPattern,
    FigureGroundSegmentation,
    ProximityProcessor,
    SimilarityProcessor,
    ClosureProcessor,
    FigureGroundProcessor,
)

# SYNAPSE-M: Proprioception
from .proprioception import (
    ProprioceptionSystem,
    BodyPartType,
    MovementType,
    PostureType,
    BodySchema,
    BodyPartState,
    JointState,
    ProprioceptivePerception,
    MuscleSpindleSensor,
    GolgiTendonOrgan,
    JointReceptor,
)

# SYNAPSE-M: Interoception
from .interoception import (
    InteroceptionSystem,
    OrganSystem,
    HomeostasisState,
    ArousalLevel,
    OrganSystemState,
    VitalSign,
    InternalStateSnapshot,
    InteroceptivePerception,
    InteroceptiveSensor,
    BaroreceptorSensor,
    ChemoreceptorSensor,
    ThermoreceptorSensor,
    NociceptorSensor,
)


__all__ = [
    # Core perception
    "SensorySystem",
    "Sensor",
    "SensorType",
    "SensoryInput",
    "PerceptionFilter",
    "AttentionMechanism",
    "SensoryModality",
    "AttentionFocus",
    "SensoryPriority",

    # Environment
    "Environment",
    "EnvironmentState",
    "Stimulus",
    "StimulusType",
    "EnvironmentEvent",

    # Modality Types
    "ExtendedModality",
    "ModalityDomain",
    "ModalityCharacteristics",
    "ModalityRegistry",
    "ModalityInput",
    "ProcessedModality",
    "ModalityGroup",
    "get_modality_characteristics",
    "get_group_modalities",
    "MODALITY_DEFAULTS",
    "MODALITY_GROUPS",

    # Cross-Modal
    "CrossModalIntegrator",
    "SensoryFusion",
    "BoundPercept",
    "BindingPair",
    "BindingType",
    "BindingStrength",
    "TemporalBindingMechanism",
    "SpatialBindingMechanism",
    "SemanticBindingMechanism",

    # Attention
    "AttentionFilter",
    "MultiModalAttentionManager",
    "SaliencyComputer",
    "SaliencyMap",
    "AttentionWeight",
    "AttentionType",
    "AttentionPriority",

    # Predictive Perception
    "PredictivePerceptionSystem",
    "Prediction",
    "PredictionError",
    "PredictiveState",
    "PredictiveModel",
    "TemporalPredictiveModel",
    "CrossModalPredictiveModel",
    "PredictionType",
    "PredictionConfidence",

    # Sensory Memory
    "SensoryMemorySystem",
    "SensoryMemoryBuffer",
    "IconicMemory",
    "EchoicMemory",
    "HapticMemory",
    "MemoryTrace",
    "SensorySnapshot",
    "MemoryType",
    "DecayModel",

    # Gestalt Processing
    "GestaltProcessor",
    "GestaltPrinciple",
    "PerceptualGroup",
    "CompletedPattern",
    "FigureGroundSegmentation",
    "ProximityProcessor",
    "SimilarityProcessor",
    "ClosureProcessor",
    "FigureGroundProcessor",

    # Proprioception
    "ProprioceptionSystem",
    "BodyPartType",
    "MovementType",
    "PostureType",
    "BodySchema",
    "BodyPartState",
    "JointState",
    "ProprioceptivePerception",
    "MuscleSpindleSensor",
    "GolgiTendonOrgan",
    "JointReceptor",

    # Interoception
    "InteroceptionSystem",
    "OrganSystem",
    "HomeostasisState",
    "ArousalLevel",
    "OrganSystemState",
    "VitalSign",
    "InternalStateSnapshot",
    "InteroceptivePerception",
    "InteroceptiveSensor",
    "BaroreceptorSensor",
    "ChemoreceptorSensor",
    "ThermoreceptorSensor",
    "NociceptorSensor",
]
