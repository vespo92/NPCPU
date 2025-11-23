"""
NPCPU Neural Module - DENDRITE-X Advanced Neural Architecture

Neural network-based consciousness models with advanced plasticity,
oscillations, and learning mechanisms.

This module is part of AGENT DENDRITE-X (Agent #3) in the NPCPU framework.

Components:
-----------
- attention_layers: Multi-head attention mechanisms for consciousness
- transformer_consciousness: Transformer-based consciousness model
- neuroplasticity: Dynamic network restructuring
- synaptic_pruning: Experience-based connection pruning
- long_term_potentiation: LTP memory consolidation
- spike_timing: Spike-timing-dependent plasticity (STDP)
- neural_oscillations: Brain wave patterns (alpha, beta, theta, gamma)
- dream_states: Dream states and offline learning consolidation
- neural_consciousness: Neural architecture search for consciousness

Expected Output Interfaces (DENDRITE-X):
- TransformerConsciousness class
- NeuroplasticityEngine for network adaptation
- OscillationManager for brain state management
- AttentionMechanism reusable component
"""

# Original exports
from .neural_consciousness import (
    NeuralConsciousnessModel,
    ConsciousnessNAS,
    NASConfig,
    Architecture,
    TrainingConfig
)

# DENDRITE-X: Attention Layers
from .attention_layers import (
    AttentionMechanism,
    ConsciousnessAttention,
    SelfAttentionLayer,
    CrossAttentionLayer,
    GlobalWorkspaceAttention,
    PositionalEncoding,
    AttentionConfig,
    AttentionType
)

# DENDRITE-X: Transformer Consciousness
from .transformer_consciousness import (
    TransformerConsciousness,
    TransformerConfig,
    ConsciousnessPhase,
    FeedForwardNetwork,
    TransformerBlock,
    ConsciousnessIntegrationModule,
    create_transformer_consciousness
)

# DENDRITE-X: Neuroplasticity
from .neuroplasticity import (
    NeuroplasticityEngine,
    PlasticityConfig,
    PlasticityType,
    PlasticityPhase,
    SynapticConnection,
    AdaptivePlasticityRate,
    ExperienceDependentPlasticity,
    CriticalPeriodManager
)

# DENDRITE-X: Synaptic Pruning
from .synaptic_pruning import (
    SynapticPruningEngine,
    PruningConfig,
    PruningStrategy,
    PruningPhase,
    SynapseInfo,
    CompetitiveElimination
)

# DENDRITE-X: Long-Term Potentiation
from .long_term_potentiation import (
    LTPEngine,
    LTPConfig,
    LTPPhase,
    LTDPhase,
    SynapticTag,
    MemoryReconsolidation
)

# DENDRITE-X: Spike-Timing-Dependent Plasticity
from .spike_timing import (
    STDPEngine,
    STDPConfig,
    STDPType,
    SpikeEvent,
    SpikeEventType,
    RewardModulatedSTDP,
    HomeostaticSTDP
)

# DENDRITE-X: Neural Oscillations
from .neural_oscillations import (
    OscillationManager,
    OscillationConfig,
    OscillatorState,
    BrainWave,
    ConsciousnessMode,
    NeuralSynchrony,
    SleepWakeDynamics
)

# DENDRITE-X: Dream States
from .dream_states import (
    DreamStateEngine,
    DreamConfig,
    SleepStage,
    DreamType,
    Memory,
    OfflineLearningSystem
)


__all__ = [
    # Original Neural Consciousness
    "NeuralConsciousnessModel",
    "ConsciousnessNAS",
    "NASConfig",
    "Architecture",
    "TrainingConfig",

    # Attention Layers
    "AttentionMechanism",
    "ConsciousnessAttention",
    "SelfAttentionLayer",
    "CrossAttentionLayer",
    "GlobalWorkspaceAttention",
    "PositionalEncoding",
    "AttentionConfig",
    "AttentionType",

    # Transformer Consciousness
    "TransformerConsciousness",
    "TransformerConfig",
    "ConsciousnessPhase",
    "FeedForwardNetwork",
    "TransformerBlock",
    "ConsciousnessIntegrationModule",
    "create_transformer_consciousness",

    # Neuroplasticity
    "NeuroplasticityEngine",
    "PlasticityConfig",
    "PlasticityType",
    "PlasticityPhase",
    "SynapticConnection",
    "AdaptivePlasticityRate",
    "ExperienceDependentPlasticity",
    "CriticalPeriodManager",

    # Synaptic Pruning
    "SynapticPruningEngine",
    "PruningConfig",
    "PruningStrategy",
    "PruningPhase",
    "SynapseInfo",
    "CompetitiveElimination",

    # Long-Term Potentiation
    "LTPEngine",
    "LTPConfig",
    "LTPPhase",
    "LTDPhase",
    "SynapticTag",
    "MemoryReconsolidation",

    # Spike-Timing-Dependent Plasticity
    "STDPEngine",
    "STDPConfig",
    "STDPType",
    "SpikeEvent",
    "SpikeEventType",
    "RewardModulatedSTDP",
    "HomeostaticSTDP",

    # Neural Oscillations
    "OscillationManager",
    "OscillationConfig",
    "OscillatorState",
    "BrainWave",
    "ConsciousnessMode",
    "NeuralSynchrony",
    "SleepWakeDynamics",

    # Dream States
    "DreamStateEngine",
    "DreamConfig",
    "SleepStage",
    "DreamType",
    "Memory",
    "OfflineLearningSystem",
]
