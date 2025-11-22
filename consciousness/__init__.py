"""
NPCPU Consciousness Module

Advanced consciousness models including adaptive, multi-modal,
and transferable consciousness.
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
from .transfer import (
    ConsciousnessTransferProtocol,
    DistributedConsciousnessManager,
    TransferMode,
    TransferResult,
    ConsciousnessSnapshot,
    Agent
)

__all__ = [
    "AdaptiveConsciousness",
    "EnvironmentContext",
    "TaskContext",
    "AdaptationRecord",
    "MultiModalConsciousness",
    "ModalityProcessor",
    "CrossModalIntegrator",
    "MultiModalPerception",
    "IntegratedPerception",
    "ConsciousnessTransferProtocol",
    "DistributedConsciousnessManager",
    "TransferMode",
    "TransferResult",
    "ConsciousnessSnapshot",
    "Agent"
]
