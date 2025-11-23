"""
NPCPU Endocrine System - HORMONE-Y

Comprehensive hormonal regulation for digital organisms.
Manages long-term state, behavioral modulation, and physiological integration.

Components:
- hormones: Core hormone types, glands, and endocrine system
- receptor: Receptor binding and sensitivity dynamics
- feedback_loops: Negative/positive feedback regulation
- circadian_rhythm: Day/night hormone cycling
- stress_response: Comprehensive stress cascade (HPA axis)
- reward_system: Dopamine-based reward signaling
- social_bonding: Oxytocin-like social bonding
- metabolism_coupling: Endocrine-metabolism integration
- neuroendocrine: Neural-endocrine interface
"""

# Core hormone system
from .hormones import (
    EndocrineSystem,
    Hormone,
    HormoneType,
    Gland,
    GlandType,
    HormoneEffect,
    HormoneSignal,
    MoodState,
    DriveState
)

# Receptor system
from .receptor import (
    ReceptorSystem,
    Receptor,
    ReceptorType,
    ReceptorState,
    ReceptorBinding,
    SignalCascade,
    SignalType
)

# Feedback loops
from .feedback_loops import (
    FeedbackSystem,
    FeedbackLoop,
    FeedbackType,
    FeedbackState,
    NeuroendocrineAxis,
    AxisType,
    SetpointShift
)

# Circadian rhythm
from .circadian_rhythm import (
    CircadianSystem,
    CircadianOscillator,
    CircadianPhase,
    SleepState,
    ChronotypeBias,
    HormoneRhythm,
    ZeitgeberEvent
)

# Stress response
from .stress_response import (
    StressSystem,
    Stressor,
    StressorType,
    StressPhase,
    StressResponse,
    AllostaticLoad,
    CopingStyle
)

# Reward system
from .reward_system import (
    RewardSystem,
    RewardEvent,
    RewardType,
    RewardPrediction,
    MotivationalDrive,
    MotivationalState,
    HedonicTone
)

# Social bonding
from .social_bonding import (
    SocialBondingSystem,
    SocialBond,
    BondType,
    TrustLevel,
    SocialState,
    SocialInteraction,
    InteractionType,
    SocialNeed
)

# Metabolism coupling
from .metabolism_coupling import (
    MetabolismCouplingSystem,
    GlucoseState,
    EnergyReserves,
    MetabolicState,
    EnergyState,
    HungerLevel,
    MetabolicRate,
    MetabolicSignal
)

# Neuroendocrine interface
from .neuroendocrine import (
    NeuroendocrineSystem,
    NeuralSignal,
    NeuralSignalType,
    ReleasingHormone,
    HypothalamicOutput,
    EmotionalState,
    SensoryInput,
    NeuroendocrineEvent
)

__all__ = [
    # Core
    "EndocrineSystem",
    "Hormone",
    "HormoneType",
    "Gland",
    "GlandType",
    "HormoneEffect",
    "HormoneSignal",
    "MoodState",
    "DriveState",

    # Receptor
    "ReceptorSystem",
    "Receptor",
    "ReceptorType",
    "ReceptorState",
    "ReceptorBinding",
    "SignalCascade",
    "SignalType",

    # Feedback
    "FeedbackSystem",
    "FeedbackLoop",
    "FeedbackType",
    "FeedbackState",
    "NeuroendocrineAxis",
    "AxisType",
    "SetpointShift",

    # Circadian
    "CircadianSystem",
    "CircadianOscillator",
    "CircadianPhase",
    "SleepState",
    "ChronotypeBias",
    "HormoneRhythm",
    "ZeitgeberEvent",

    # Stress
    "StressSystem",
    "Stressor",
    "StressorType",
    "StressPhase",
    "StressResponse",
    "AllostaticLoad",
    "CopingStyle",

    # Reward
    "RewardSystem",
    "RewardEvent",
    "RewardType",
    "RewardPrediction",
    "MotivationalDrive",
    "MotivationalState",
    "HedonicTone",

    # Social
    "SocialBondingSystem",
    "SocialBond",
    "BondType",
    "TrustLevel",
    "SocialState",
    "SocialInteraction",
    "InteractionType",
    "SocialNeed",

    # Metabolism
    "MetabolismCouplingSystem",
    "GlucoseState",
    "EnergyReserves",
    "MetabolicState",
    "EnergyState",
    "HungerLevel",
    "MetabolicRate",
    "MetabolicSignal",

    # Neuroendocrine
    "NeuroendocrineSystem",
    "NeuralSignal",
    "NeuralSignalType",
    "ReleasingHormone",
    "HypothalamicOutput",
    "EmotionalState",
    "SensoryInput",
    "NeuroendocrineEvent"
]
