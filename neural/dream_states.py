"""
DENDRITE-X: Dream States and Offline Learning

Implements dream-like states for memory consolidation and offline learning.
Simulates REM/NREM sleep cycles and their cognitive functions.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Sleep Stages
# ============================================================================

class SleepStage(Enum):
    """Stages of sleep"""
    WAKE = "wake"
    N1 = "n1"  # Light sleep, transition
    N2 = "n2"  # Light sleep, spindles
    N3 = "n3"  # Deep sleep, slow waves
    REM = "rem"  # Rapid eye movement, dreams


class DreamType(Enum):
    """Types of dream-like processing"""
    REPLAY = "replay"  # Memory replay
    CREATIVE = "creative"  # Novel combinations
    EMOTIONAL = "emotional"  # Emotional processing
    PREDICTIVE = "predictive"  # Future simulation
    CONSOLIDATION = "consolidation"  # Memory strengthening


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DreamConfig:
    """Configuration for dream states"""
    # Sleep cycle timing (in steps)
    cycle_duration: int = 9000  # ~90 minutes at 100 steps/minute
    rem_ratio: float = 0.25  # Fraction of cycle in REM
    n3_ratio: float = 0.20  # Fraction in deep sleep

    # Replay parameters
    replay_speed: float = 20.0  # Times faster than real-time
    replay_noise: float = 0.1  # Noise added during replay
    replay_selection: str = "importance"  # "importance", "recency", "random"

    # Consolidation parameters
    consolidation_strength: float = 0.1
    pruning_threshold: float = 0.05

    # Creative recombination
    recombination_rate: float = 0.3


@dataclass
class Memory:
    """A memory trace for replay"""
    content: np.ndarray
    importance: float
    emotional_valence: float
    timestamp: float
    replay_count: int = 0
    consolidated: bool = False


# ============================================================================
# Dream State Engine
# ============================================================================

class DreamStateEngine:
    """
    Engine for dream states and offline learning.

    Implements:
    1. Sleep stage transitions
    2. Memory replay during sleep
    3. Creative recombination in REM
    4. Memory consolidation in NREM
    5. Emotional processing

    Based on:
    - Two-stage memory model
    - Active Systems Consolidation theory
    - REM sleep hypothesis for creativity

    Example:
        engine = DreamStateEngine()

        # Store experiences during wake
        engine.store_memory(experience, importance=0.8)

        # Enter sleep
        engine.enter_sleep()

        # Run sleep cycles
        for step in range(9000):
            engine.step()

        # Wake up with consolidated memories
        engine.wake_up()
    """

    def __init__(self, config: Optional[DreamConfig] = None):
        self.config = config or DreamConfig()

        # Current state
        self.stage = SleepStage.WAKE
        self.cycle_position = 0
        self.cycles_completed = 0

        # Memory systems
        self.episodic_buffer: List[Memory] = []  # Recent experiences
        self.consolidated_memories: List[Memory] = []  # Long-term storage
        self.max_buffer_size = 1000
        self.max_consolidated_size = 10000

        # Dream content
        self.current_dream: Optional[Dict[str, Any]] = None
        self.dream_log: List[Dict[str, Any]] = []

        # Processing state
        self.replay_queue: List[Memory] = []
        self.current_replay_idx = 0

        # Statistics
        self.total_replays = 0
        self.total_consolidations = 0
        self.total_creative_events = 0

    def store_memory(
        self,
        content: np.ndarray,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        timestamp: Optional[float] = None
    ):
        """Store a memory in the episodic buffer"""
        if timestamp is None:
            timestamp = float(len(self.episodic_buffer))

        memory = Memory(
            content=content.copy(),
            importance=importance,
            emotional_valence=emotional_valence,
            timestamp=timestamp
        )

        self.episodic_buffer.append(memory)

        # Trim buffer if needed
        if len(self.episodic_buffer) > self.max_buffer_size:
            # Remove least important
            self.episodic_buffer.sort(key=lambda m: m.importance, reverse=True)
            self.episodic_buffer = self.episodic_buffer[:self.max_buffer_size]

    def enter_sleep(self):
        """Transition from wake to sleep"""
        if self.stage != SleepStage.WAKE:
            return

        self.stage = SleepStage.N1
        self.cycle_position = 0

        # Prepare replay queue
        self._prepare_replay_queue()

    def wake_up(self):
        """Transition from sleep to wake"""
        self.stage = SleepStage.WAKE
        self.cycle_position = 0
        self.current_dream = None

    def _prepare_replay_queue(self):
        """Prepare memories for replay based on selection strategy"""
        if self.config.replay_selection == "importance":
            # Sort by importance (highest first)
            self.replay_queue = sorted(
                self.episodic_buffer,
                key=lambda m: m.importance,
                reverse=True
            )
        elif self.config.replay_selection == "recency":
            # Sort by timestamp (most recent first)
            self.replay_queue = sorted(
                self.episodic_buffer,
                key=lambda m: m.timestamp,
                reverse=True
            )
        else:
            # Random
            self.replay_queue = list(self.episodic_buffer)
            np.random.shuffle(self.replay_queue)

        self.current_replay_idx = 0

    def _get_current_stage(self) -> SleepStage:
        """Determine sleep stage based on cycle position"""
        if self.stage == SleepStage.WAKE:
            return SleepStage.WAKE

        cycle_frac = self.cycle_position / self.config.cycle_duration

        # Sleep architecture: N1 -> N2 -> N3 -> N2 -> REM
        if cycle_frac < 0.05:
            return SleepStage.N1
        elif cycle_frac < 0.25:
            return SleepStage.N2
        elif cycle_frac < 0.25 + self.config.n3_ratio:
            return SleepStage.N3
        elif cycle_frac < 0.75:
            return SleepStage.N2
        else:
            return SleepStage.REM

    def step(self) -> Dict[str, Any]:
        """
        Advance dream state by one step.

        Returns:
            Dict with current processing state
        """
        if self.stage == SleepStage.WAKE:
            return {"stage": "wake", "processing": None}

        # Update cycle position
        self.cycle_position += 1
        if self.cycle_position >= self.config.cycle_duration:
            self.cycle_position = 0
            self.cycles_completed += 1
            self._prepare_replay_queue()  # New queue for new cycle

        # Determine current stage
        self.stage = self._get_current_stage()

        # Stage-specific processing
        if self.stage == SleepStage.N3:
            result = self._process_slow_wave_sleep()
        elif self.stage == SleepStage.REM:
            result = self._process_rem_sleep()
        elif self.stage == SleepStage.N2:
            result = self._process_light_sleep()
        else:
            result = {"processing": "transition"}

        result["stage"] = self.stage.value
        result["cycle_position"] = self.cycle_position
        result["cycles_completed"] = self.cycles_completed

        return result

    def _process_slow_wave_sleep(self) -> Dict[str, Any]:
        """Process during slow-wave (N3) sleep - memory consolidation"""
        if not self.replay_queue:
            return {"processing": "consolidation", "replayed": False}

        # Replay memory (compressed time)
        if self.current_replay_idx < len(self.replay_queue):
            memory = self.replay_queue[self.current_replay_idx]

            # Consolidation strengthening
            self._consolidate_memory(memory)

            self.current_replay_idx += 1
            self.total_replays += 1

            return {
                "processing": "consolidation",
                "replayed": True,
                "memory_importance": memory.importance
            }

        return {"processing": "consolidation", "replayed": False}

    def _consolidate_memory(self, memory: Memory):
        """Consolidate a single memory"""
        # Increase importance (strengthening)
        memory.importance = min(1.0, memory.importance + self.config.consolidation_strength)
        memory.replay_count += 1

        # Check for full consolidation
        if memory.replay_count >= 3 and memory.importance > 0.7:
            if not memory.consolidated:
                memory.consolidated = True
                self.consolidated_memories.append(memory)
                self.total_consolidations += 1

                # Trim consolidated if needed
                if len(self.consolidated_memories) > self.max_consolidated_size:
                    self.consolidated_memories.sort(key=lambda m: m.importance, reverse=True)
                    self.consolidated_memories = self.consolidated_memories[:self.max_consolidated_size]

    def _process_rem_sleep(self) -> Dict[str, Any]:
        """Process during REM sleep - dreams and creative recombination"""
        # Generate dream content
        dream = self._generate_dream()

        self.current_dream = dream
        self.dream_log.append(dream)

        # Trim dream log
        if len(self.dream_log) > 100:
            self.dream_log = self.dream_log[-50:]

        return {
            "processing": "dreaming",
            "dream_type": dream["type"],
            "emotional_intensity": dream.get("emotional_intensity", 0)
        }

    def _generate_dream(self) -> Dict[str, Any]:
        """Generate dream content"""
        if not self.episodic_buffer:
            return {"type": DreamType.CREATIVE.value, "content": None}

        # Determine dream type based on emotional content
        emotional_memories = [m for m in self.episodic_buffer if abs(m.emotional_valence) > 0.5]

        if emotional_memories and np.random.random() < 0.4:
            # Emotional processing dream
            memory = np.random.choice(emotional_memories)
            dream_type = DreamType.EMOTIONAL
            content = self._process_emotional_dream(memory)
        elif np.random.random() < self.config.recombination_rate:
            # Creative recombination
            dream_type = DreamType.CREATIVE
            content = self._creative_recombination()
            self.total_creative_events += 1
        else:
            # Memory replay with distortion
            dream_type = DreamType.REPLAY
            if self.replay_queue:
                memory = self.replay_queue[self.current_replay_idx % len(self.replay_queue)]
                content = self._replay_with_noise(memory)
            else:
                content = None

        return {
            "type": dream_type.value,
            "content": content,
            "emotional_intensity": np.random.random() * 0.5
        }

    def _process_emotional_dream(self, memory: Memory) -> np.ndarray:
        """Process emotional memory in dream"""
        # Add noise proportional to emotional intensity
        noise_level = abs(memory.emotional_valence) * 0.2
        noisy = memory.content + np.random.randn(*memory.content.shape) * noise_level

        # Emotional processing can reduce valence intensity
        memory.emotional_valence *= 0.9

        return noisy

    def _creative_recombination(self) -> Optional[np.ndarray]:
        """Combine multiple memories creatively"""
        if len(self.episodic_buffer) < 2:
            return None

        # Select 2-3 random memories
        num_combine = min(3, len(self.episodic_buffer))
        selected = np.random.choice(self.episodic_buffer, num_combine, replace=False)

        # Average with random weights
        weights = np.random.dirichlet(np.ones(num_combine))
        combined = sum(w * m.content for w, m in zip(weights, selected))

        # Add creative noise
        combined += np.random.randn(*combined.shape) * 0.1

        return combined

    def _replay_with_noise(self, memory: Memory) -> np.ndarray:
        """Replay memory with dream-like distortion"""
        return memory.content + np.random.randn(*memory.content.shape) * self.config.replay_noise

    def _process_light_sleep(self) -> Dict[str, Any]:
        """Process during light (N2) sleep - sleep spindles"""
        # Sleep spindles help with motor learning and memory
        return {"processing": "spindles", "memory_stabilization": True}

    def run_full_cycle(self) -> Dict[str, Any]:
        """Run a complete sleep cycle"""
        results = []

        for _ in range(self.config.cycle_duration):
            result = self.step()
            results.append(result)

        # Summarize cycle
        n3_steps = sum(1 for r in results if r["stage"] == "n3")
        rem_steps = sum(1 for r in results if r["stage"] == "rem")

        return {
            "cycle_completed": True,
            "n3_steps": n3_steps,
            "rem_steps": rem_steps,
            "total_replays": self.total_replays,
            "total_consolidations": self.total_consolidations,
            "total_creative": self.total_creative_events
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get dream state statistics"""
        return {
            "current_stage": self.stage.value,
            "cycles_completed": self.cycles_completed,
            "episodic_buffer_size": len(self.episodic_buffer),
            "consolidated_count": len(self.consolidated_memories),
            "total_replays": self.total_replays,
            "total_consolidations": self.total_consolidations,
            "total_creative_events": self.total_creative_events,
            "dream_log_size": len(self.dream_log)
        }

    def get_consolidated_memories(self) -> List[Memory]:
        """Get list of consolidated memories"""
        return self.consolidated_memories.copy()


# ============================================================================
# Offline Learning System
# ============================================================================

class OfflineLearningSystem:
    """
    System for offline learning during sleep-like states.

    Implements:
    - Memory replay for experience reinforcement
    - Gradient computation on replayed experiences
    - Weight updates during offline periods
    """

    def __init__(
        self,
        dream_engine: DreamStateEngine,
        learning_rate: float = 0.001
    ):
        self.dream_engine = dream_engine
        self.learning_rate = learning_rate

        # Weight updates accumulated during sleep
        self.accumulated_updates: List[np.ndarray] = []

        # Model state (simplified)
        self.model_weights: Optional[np.ndarray] = None

    def set_model_weights(self, weights: np.ndarray):
        """Set the model weights to update during sleep"""
        self.model_weights = weights.copy()

    def compute_replay_gradient(
        self,
        memory_content: np.ndarray,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """
        Compute pseudo-gradient from memory replay.

        In a real system, this would involve actual gradient computation.
        Here we simulate the effect of replay.
        """
        # Add slight noise (representing uncertainty)
        noisy_content = memory_content + np.random.randn(*memory_content.shape) * noise_level

        # Pseudo-gradient: direction of reinforcement
        if self.model_weights is not None:
            # Simple correlation-based update
            gradient = np.outer(noisy_content.flatten(), noisy_content.flatten())
            # Truncate to match weights shape if needed
            if gradient.shape != self.model_weights.shape:
                min_size = min(gradient.shape[0], self.model_weights.shape[0])
                gradient = gradient[:min_size, :min_size]
        else:
            gradient = noisy_content

        return gradient * self.learning_rate

    def sleep_training_step(self) -> Dict[str, Any]:
        """
        Perform one offline learning step during sleep.

        Returns:
            Dict with training information
        """
        # Get dream state
        dream_result = self.dream_engine.step()

        if self.dream_engine.stage == SleepStage.WAKE:
            return {"training": False, "reason": "awake"}

        # During N3, perform replay-based learning
        if self.dream_engine.stage == SleepStage.N3:
            if self.dream_engine.current_dream and self.dream_engine.current_dream.get("content") is not None:
                content = self.dream_engine.current_dream["content"]
                gradient = self.compute_replay_gradient(content)

                self.accumulated_updates.append(gradient)

                return {
                    "training": True,
                    "stage": "n3",
                    "update_computed": True
                }

        # During REM, creative exploration
        elif self.dream_engine.stage == SleepStage.REM:
            if self.dream_engine.current_dream and self.dream_engine.current_dream.get("content") is not None:
                content = self.dream_engine.current_dream["content"]
                # More exploratory gradient
                gradient = self.compute_replay_gradient(content, noise_level=0.2)

                self.accumulated_updates.append(gradient)

                return {
                    "training": True,
                    "stage": "rem",
                    "update_computed": True
                }

        return {"training": True, "stage": dream_result["stage"], "update_computed": False}

    def apply_accumulated_updates(self) -> int:
        """
        Apply accumulated weight updates after sleep.

        Returns:
            Number of updates applied
        """
        if self.model_weights is None or not self.accumulated_updates:
            return 0

        # Average all updates
        total_updates = len(self.accumulated_updates)
        mean_update = np.mean(self.accumulated_updates, axis=0)

        # Apply with momentum-like decay
        self.model_weights += mean_update * 0.1

        # Clear accumulator
        self.accumulated_updates = []

        return total_updates

    def run_sleep_learning(self, num_cycles: int = 1) -> Dict[str, Any]:
        """
        Run complete sleep learning session.

        Args:
            num_cycles: Number of sleep cycles to run

        Returns:
            Summary of learning session
        """
        self.dream_engine.enter_sleep()

        total_steps = num_cycles * self.dream_engine.config.cycle_duration
        training_steps = 0

        for _ in range(total_steps):
            result = self.sleep_training_step()
            if result.get("update_computed"):
                training_steps += 1

        self.dream_engine.wake_up()

        updates_applied = self.apply_accumulated_updates()

        return {
            "cycles_completed": num_cycles,
            "training_steps": training_steps,
            "updates_applied": updates_applied,
            "consolidations": self.dream_engine.total_consolidations
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Dream States Demo")
    print("=" * 50)

    # Create dream engine
    engine = DreamStateEngine()

    print("\n1. Storing Memories:")
    for i in range(50):
        content = np.random.randn(64)
        importance = np.random.random()
        valence = np.random.randn() * 0.5
        engine.store_memory(content, importance, valence)

    print(f"   Stored {len(engine.episodic_buffer)} memories")

    # Enter sleep
    print("\n2. Entering Sleep:")
    engine.enter_sleep()
    print(f"   Current stage: {engine.stage.value}")

    # Run one sleep cycle
    print("\n3. Running Sleep Cycle:")
    result = engine.run_full_cycle()
    print(f"   N3 steps: {result['n3_steps']}")
    print(f"   REM steps: {result['rem_steps']}")
    print(f"   Total replays: {result['total_replays']}")
    print(f"   Consolidations: {result['total_consolidations']}")

    # Check statistics
    print("\n4. Statistics:")
    stats = engine.get_statistics()
    print(f"   Cycles completed: {stats['cycles_completed']}")
    print(f"   Consolidated: {stats['consolidated_count']}")
    print(f"   Creative events: {stats['total_creative_events']}")

    # Wake up
    print("\n5. Waking Up:")
    engine.wake_up()
    print(f"   Current stage: {engine.stage.value}")

    # Offline learning
    print("\n6. Offline Learning System:")
    engine2 = DreamStateEngine()
    offline = OfflineLearningSystem(engine2, learning_rate=0.01)

    # Store memories
    for i in range(30):
        engine2.store_memory(np.random.randn(32), importance=0.5 + np.random.random() * 0.5)

    # Set model weights
    offline.set_model_weights(np.random.randn(32, 32) * 0.1)

    # Run sleep learning
    result = offline.run_sleep_learning(num_cycles=1)
    print(f"   Training steps: {result['training_steps']}")
    print(f"   Updates applied: {result['updates_applied']}")

    print("\n" + "=" * 50)
    print("Dream States Engine ready!")
