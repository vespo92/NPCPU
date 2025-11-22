"""
NPCPU Neural Module

Neural network-based consciousness models with architecture search.
"""

from .neural_consciousness import (
    NeuralConsciousnessModel,
    ConsciousnessNAS,
    NASConfig,
    Architecture,
    TrainingConfig
)

__all__ = [
    "NeuralConsciousnessModel",
    "ConsciousnessNAS",
    "NASConfig",
    "Architecture",
    "TrainingConfig"
]
