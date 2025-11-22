"""
NPCPU Nervous System Module

Central coordination system for digital organisms. Handles signal
processing, reflex responses, and inter-system coordination.
"""

from .coordination import (
    NervousSystem,
    Signal,
    SignalType,
    Reflex,
    NeuralPathway,
    CentralCoordinator
)

__all__ = [
    "NervousSystem",
    "Signal",
    "SignalType",
    "Reflex",
    "NeuralPathway",
    "CentralCoordinator"
]
