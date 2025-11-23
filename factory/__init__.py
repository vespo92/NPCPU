"""
NPCPU Factory Module

Provides factory classes for creating consciousness instances
from configuration files.
"""

from .consciousness_factory import (
    ConsciousnessFactory,
    ConsciousnessModelConfig,
    load_consciousness_model,
)

from .quantum_consciousness_factory import (
    QuantumConsciousnessFactory,
    QuantumConsciousnessConfig,
    create_quantum_consciousness,
)

__all__ = [
    # Standard consciousness factory
    "ConsciousnessFactory",
    "ConsciousnessModelConfig",
    "load_consciousness_model",

    # Quantum consciousness factory (NEXUS-Q)
    "QuantumConsciousnessFactory",
    "QuantumConsciousnessConfig",
    "create_quantum_consciousness",
]
