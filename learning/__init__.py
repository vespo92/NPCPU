"""
Learning & Memory Module for NPCPU

This module provides cognitive learning and memory systems for organisms:
- Multi-tier memory system (sensory, working, long-term)
- Episodic and semantic memory types
- Reinforcement learning with Q-learning
- Experience replay and policy updates

Example:
    from learning import MemorySystem, ReinforcementLearner

    # Create memory system for an organism
    memory = MemorySystem(owner=organism)
    memory.store(MemoryType.EPISODIC, {"event": "found_food", "location": (10, 20)}, importance=0.8)

    # Create reinforcement learner
    learner = ReinforcementLearner(owner=organism, learning_rate=0.1)
    learner.learn_from_experience(state, action, reward, next_state)
"""

from .memory_systems import (
    MemorySystem,
    MemoryType,
    Memory,
    MemoryTier,
)

from .reinforcement import (
    ReinforcementLearner,
    ExplorationStrategy,
    Experience,
)

__all__ = [
    # Memory systems
    'MemorySystem',
    'MemoryType',
    'Memory',
    'MemoryTier',
    # Reinforcement learning
    'ReinforcementLearner',
    'ExplorationStrategy',
    'Experience',
]
