"""
NPCPU Consciousness Module

Advanced consciousness models including adaptive, multi-modal,
neural network-based, and transferable consciousness.
"""

from .adaptive_consciousness import (
    AdaptiveConsciousness,
    EnvironmentContext,
    TaskContext,
    AdaptationRecord
)
from .multimodal_consciousness import (
    MultiModalConsciousness,
    ModalityProcessor,
    CrossModalIntegrator,
    MultiModalPerception,
    IntegratedPerception
)
from .neural_consciousness import (
    NeuralConsciousness,
    AttentionMechanism,
    SelfModel,
    MemoryItem,
    AttentionState,
    EmotionalState,
    EmotionalDimension,
    attach_neural_consciousness
)
from .transfer import (
    ConsciousnessTransferProtocol,
    DistributedConsciousnessManager,
    TransferMode,
    TransferResult,
    ConsciousnessSnapshot,
    Agent
)

__all__ = [
    # Adaptive consciousness
    "AdaptiveConsciousness",
    "EnvironmentContext",
    "TaskContext",
    "AdaptationRecord",
    # Multi-modal consciousness
    "MultiModalConsciousness",
    "ModalityProcessor",
    "CrossModalIntegrator",
    "MultiModalPerception",
    "IntegratedPerception",
    # Neural consciousness
    "NeuralConsciousness",
    "AttentionMechanism",
    "SelfModel",
    "MemoryItem",
    "AttentionState",
    "EmotionalState",
    "EmotionalDimension",
    "attach_neural_consciousness",
    # Consciousness transfer
    "ConsciousnessTransferProtocol",
    "DistributedConsciousnessManager",
    "TransferMode",
    "TransferResult",
    "ConsciousnessSnapshot",
    "Agent"
]
