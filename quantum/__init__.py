"""
NPCPU Quantum Module - NEXUS-Q

Provides quantum-inspired consciousness models including:
- Superposition of thoughts
- Entangled concepts
- Quantum parallel reasoning
- Quantum memory with Grover search
- Quantum tunneling for decisions
- Coherence and decoherence modeling
- Measurement and observer effects

Part of the NEXUS-Q agent workstream.
"""

# Core quantum state management
from .quantum_state import (
    StateType,
    QuantumAmplitude,
    BasisState,
    QuantumStateVector,
    DensityMatrix,
    SuperpositionState,
)

# Main quantum consciousness
from .quantum_consciousness import (
    QuantumConsciousness,
    QuantumThought,
    EntangledConcepts,
    QuantumState,
    QuantumGate,
    QuantumCircuit,
    Thought,
    Concept,
    HybridQuantumConsciousness,
)

# Entanglement module
from .entanglement import (
    EntanglementType,
    EntanglementPair,
    EntanglementCluster,
    EntanglementManager,
    create_bell_state,
    concurrence,
)

# Coherence and decoherence
from .coherence import (
    DecoherenceChannel,
    CoherenceMetrics,
    CoherenceCalculator,
    DecoherenceModel,
    CoherenceTracker,
)

# Measurement and observation
from .measurement import (
    MeasurementBasis,
    CollapseInterpretation,
    MeasurementOutcome,
    MeasurementOperator,
    QuantumMeasurement,
    ConsciousnessObserver,
)

# Quantum tunneling
from .tunneling import (
    BarrierType,
    DecisionBarrier,
    TunnelingEvent,
    TunnelingCalculator,
    QuantumTunneling,
    DecisionTunneling,
)

# Quantum memory
from .quantum_memory import (
    MemoryType,
    QuantumMemoryItem,
    MemoryRecall,
    QuantumMemoryRegister,
    AssociativeQuantumMemory,
    WorkingQuantumMemory,
)

__all__ = [
    # Quantum State
    "StateType",
    "QuantumAmplitude",
    "BasisState",
    "QuantumStateVector",
    "DensityMatrix",
    "SuperpositionState",

    # Quantum Consciousness (main)
    "QuantumConsciousness",
    "QuantumThought",
    "EntangledConcepts",
    "QuantumState",
    "QuantumGate",
    "QuantumCircuit",
    "Thought",
    "Concept",
    "HybridQuantumConsciousness",

    # Entanglement
    "EntanglementType",
    "EntanglementPair",
    "EntanglementCluster",
    "EntanglementManager",
    "create_bell_state",
    "concurrence",

    # Coherence
    "DecoherenceChannel",
    "CoherenceMetrics",
    "CoherenceCalculator",
    "DecoherenceModel",
    "CoherenceTracker",

    # Measurement
    "MeasurementBasis",
    "CollapseInterpretation",
    "MeasurementOutcome",
    "MeasurementOperator",
    "QuantumMeasurement",
    "ConsciousnessObserver",

    # Tunneling
    "BarrierType",
    "DecisionBarrier",
    "TunnelingEvent",
    "TunnelingCalculator",
    "QuantumTunneling",
    "DecisionTunneling",

    # Memory
    "MemoryType",
    "QuantumMemoryItem",
    "MemoryRecall",
    "QuantumMemoryRegister",
    "AssociativeQuantumMemory",
    "WorkingQuantumMemory",
]

# Version
__version__ = "1.0.0"
__agent__ = "NEXUS-Q"
