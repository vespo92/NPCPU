"""
NPCPU Sensory System

Perception and environment sensing for digital organisms.
Enables organisms to sense their environment and process inputs.
"""

from .perception import (
    SensorySystem,
    Sensor,
    SensorType,
    SensoryInput,
    PerceptionFilter,
    AttentionMechanism,
    SensoryModality
)

from .environment import (
    Environment,
    EnvironmentState,
    Stimulus,
    StimulusType,
    EnvironmentEvent
)

__all__ = [
    "SensorySystem",
    "Sensor",
    "SensorType",
    "SensoryInput",
    "PerceptionFilter",
    "AttentionMechanism",
    "SensoryModality",
    "Environment",
    "EnvironmentState",
    "Stimulus",
    "StimulusType",
    "EnvironmentEvent"
]
