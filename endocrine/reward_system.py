"""
Reward System

Implements dopamine-based reward signaling for digital organisms including:
- Reward prediction and error
- Motivation and drive
- Pleasure and hedonic response
- Reward learning and adaptation
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .hormones import HormoneType


# ============================================================================
# Enums
# ============================================================================

class RewardType(Enum):
    """Types of rewards"""
    PRIMARY = "primary"             # Basic needs (energy, survival)
    SOCIAL = "social"               # Social connection, approval
    ACHIEVEMENT = "achievement"     # Goal completion, mastery
    NOVELTY = "novelty"             # New information, discovery
    ANTICIPATORY = "anticipatory"   # Expected future reward
    RELIEF = "relief"               # Removal of aversive state


class MotivationalState(Enum):
    """Motivational drive states"""
    SEEKING = "seeking"             # Actively pursuing reward
    ENGAGED = "engaged"             # Currently experiencing reward
    SATIATED = "satiated"           # Temporarily satisfied
    DEPLETED = "depleted"           # Low motivation, anhedonia
    CRAVING = "craving"             # Strong desire for reward


class HedonicTone(Enum):
    """Hedonic (pleasure) states"""
    EUPHORIC = "euphoric"
    PLEASURABLE = "pleasurable"
    NEUTRAL = "neutral"
    UNCOMFORTABLE = "uncomfortable"
    ANHEDONIC = "anhedonic"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RewardEvent:
    """A reward-related event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: RewardType = RewardType.PRIMARY
    magnitude: float = 0.5          # Reward size (0-1)
    expected: float = 0.0           # Expected reward
    actual: float = 0.0             # Actual received
    prediction_error: float = 0.0   # Actual - Expected
    source: str = ""                # What caused the reward
    timestamp: float = field(default_factory=time.time)


@dataclass
class RewardPrediction:
    """A learned reward prediction"""
    cue: str = ""                   # What predicts the reward
    reward_type: RewardType = RewardType.PRIMARY
    expected_value: float = 0.5     # Predicted reward magnitude
    confidence: float = 0.5         # Prediction confidence
    occurrences: int = 0            # Times experienced
    last_error: float = 0.0         # Most recent prediction error
    learning_rate: float = 0.1      # How fast prediction updates


@dataclass
class MotivationalDrive:
    """A motivational drive toward a reward"""
    target: str = ""
    reward_type: RewardType = RewardType.PRIMARY
    intensity: float = 0.5          # Drive strength
    urgency: float = 0.0            # Time pressure
    deprivation: float = 0.0        # Time since last satisfaction
    approach_tendency: float = 0.5  # Tendency to approach


# ============================================================================
# Reward System
# ============================================================================

class RewardSystem:
    """
    Complete dopamine-based reward system for digital organisms.

    Features:
    - Reward prediction error computation
    - Motivation and drive management
    - Hedonic tone tracking
    - Reward learning and adaptation
    - Tolerance and sensitization

    Example:
        reward_system = RewardSystem()

        # Deliver a reward
        reward_system.deliver_reward(RewardType.ACHIEVEMENT, 0.8)

        # Check motivation
        motivation = reward_system.get_motivation()

        # Update system
        reward_system.tick()
    """

    def __init__(
        self,
        baseline_dopamine: float = 0.4,
        learning_rate: float = 0.1,
        decay_rate: float = 0.05
    ):
        self.baseline_dopamine = baseline_dopamine
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        # Dopamine state
        self.dopamine_level: float = baseline_dopamine
        self.phasic_dopamine: float = 0.0      # Transient spikes
        self.tonic_dopamine: float = baseline_dopamine  # Baseline level

        # Motivational state
        self.motivational_state = MotivationalState.SEEKING
        self.hedonic_tone = HedonicTone.NEUTRAL
        self.overall_motivation: float = 0.5

        # Drives
        self.active_drives: Dict[str, MotivationalDrive] = {}

        # Reward predictions
        self.predictions: Dict[str, RewardPrediction] = {}

        # Reward history
        self.reward_history: List[RewardEvent] = []
        self.max_history = 300

        # Tolerance/sensitization
        self.tolerance: Dict[RewardType, float] = {rt: 0.0 for rt in RewardType}
        self.sensitization: Dict[RewardType, float] = {rt: 0.0 for rt in RewardType}

        # Cycle tracking
        self.cycle_count = 0
        self.last_reward_cycle = 0

        # Hormone effects
        self._hormone_effects: Dict[HormoneType, float] = {}

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "reward_received": [],
            "prediction_error": [],
            "motivation_changed": [],
            "hedonic_changed": [],
            "craving_triggered": []
        }

    def deliver_reward(
        self,
        reward_type: RewardType,
        magnitude: float,
        source: str = "",
        predicted: bool = False,
        prediction_cue: str = ""
    ) -> RewardEvent:
        """
        Deliver a reward and compute prediction error.

        Returns the reward event with computed prediction error.
        """
        # Get prediction if available
        expected = 0.0
        if prediction_cue and prediction_cue in self.predictions:
            pred = self.predictions[prediction_cue]
            expected = pred.expected_value

        # Apply tolerance (reduces felt magnitude)
        effective_magnitude = magnitude * (1.0 - self.tolerance[reward_type] * 0.5)

        # Compute prediction error
        prediction_error = effective_magnitude - expected

        # Create reward event
        event = RewardEvent(
            type=reward_type,
            magnitude=magnitude,
            expected=expected,
            actual=effective_magnitude,
            prediction_error=prediction_error,
            source=source
        )

        # Update dopamine based on prediction error
        self._update_dopamine(prediction_error, reward_type)

        # Update prediction if cue exists
        if prediction_cue:
            self._update_prediction(prediction_cue, reward_type, effective_magnitude)

        # Update tolerance
        self.tolerance[reward_type] = min(1.0, self.tolerance[reward_type] + magnitude * 0.02)

        # Record event
        self.reward_history.append(event)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

        self.last_reward_cycle = self.cycle_count

        # Trigger callbacks
        for callback in self._callbacks["reward_received"]:
            callback(event)

        if abs(prediction_error) > 0.2:
            for callback in self._callbacks["prediction_error"]:
                callback(prediction_cue, prediction_error)

        return event

    def _update_dopamine(self, prediction_error: float, reward_type: RewardType):
        """Update dopamine levels based on prediction error"""
        # Phasic dopamine responds to prediction error
        if prediction_error > 0:
            # Positive error: dopamine burst
            burst = prediction_error * (1.0 + self.sensitization[reward_type])
            self.phasic_dopamine = min(1.0, self.phasic_dopamine + burst)
        elif prediction_error < 0:
            # Negative error: dopamine dip
            dip = abs(prediction_error) * 0.5
            self.phasic_dopamine = max(-0.5, self.phasic_dopamine - dip)

        # Update total dopamine
        self.dopamine_level = self.tonic_dopamine + self.phasic_dopamine
        self.dopamine_level = max(0.0, min(1.0, self.dopamine_level))

        # Generate hormone effects
        self._generate_hormone_effects()

    def _update_prediction(
        self,
        cue: str,
        reward_type: RewardType,
        actual_value: float
    ):
        """Update reward prediction based on experience"""
        if cue not in self.predictions:
            self.predictions[cue] = RewardPrediction(
                cue=cue,
                reward_type=reward_type,
                expected_value=actual_value,
                learning_rate=self.learning_rate
            )
            return

        pred = self.predictions[cue]
        error = actual_value - pred.expected_value

        # Update prediction (Rescorla-Wagner style)
        pred.expected_value += pred.learning_rate * error
        pred.expected_value = max(0.0, min(1.0, pred.expected_value))
        pred.last_error = error
        pred.occurrences += 1

        # Increase confidence with experience
        pred.confidence = min(1.0, pred.confidence + 0.01)

    def add_drive(
        self,
        target: str,
        reward_type: RewardType,
        intensity: float
    ):
        """Add a motivational drive"""
        self.active_drives[target] = MotivationalDrive(
            target=target,
            reward_type=reward_type,
            intensity=intensity
        )

    def satisfy_drive(self, target: str, amount: float = 1.0):
        """Satisfy (reduce) a drive"""
        if target in self.active_drives:
            drive = self.active_drives[target]
            drive.intensity = max(0.0, drive.intensity - amount)
            drive.deprivation = 0.0

            if drive.intensity < 0.1:
                del self.active_drives[target]

    def tick(self):
        """Process one reward cycle"""
        self.cycle_count += 1

        # Decay phasic dopamine
        self.phasic_dopamine *= (1.0 - self.decay_rate)
        if abs(self.phasic_dopamine) < 0.01:
            self.phasic_dopamine = 0.0

        # Update dopamine level
        self.dopamine_level = self.tonic_dopamine + self.phasic_dopamine
        self.dopamine_level = max(0.0, min(1.0, self.dopamine_level))

        # Decay tolerance
        for rt in RewardType:
            self.tolerance[rt] = max(0.0, self.tolerance[rt] - 0.005)

        # Update drives
        self._update_drives()

        # Update motivational state
        self._update_motivational_state()

        # Update hedonic tone
        self._update_hedonic_tone()

        # Generate hormone effects
        self._generate_hormone_effects()

    def _update_drives(self):
        """Update active drives"""
        for drive in self.active_drives.values():
            # Increase deprivation over time
            drive.deprivation += 0.01

            # Increase urgency with deprivation
            drive.urgency = min(1.0, drive.deprivation * 0.1)

            # Update intensity based on deprivation
            if drive.deprivation > 10:
                drive.intensity = min(1.0, drive.intensity + 0.01)

            # Update approach tendency
            drive.approach_tendency = drive.intensity * (1.0 + drive.urgency)

    def _update_motivational_state(self):
        """Update overall motivational state"""
        old_state = self.motivational_state

        # Calculate overall motivation
        self.overall_motivation = self.dopamine_level * 0.6

        if self.active_drives:
            max_drive = max(d.intensity for d in self.active_drives.values())
            self.overall_motivation += max_drive * 0.4

        # Time since last reward
        cycles_since_reward = self.cycle_count - self.last_reward_cycle

        # Determine state
        if self.dopamine_level > 0.7:
            if cycles_since_reward < 5:
                self.motivational_state = MotivationalState.ENGAGED
            else:
                self.motivational_state = MotivationalState.SATIATED
        elif self.dopamine_level > 0.4:
            self.motivational_state = MotivationalState.SEEKING
        elif self.dopamine_level > 0.2:
            if cycles_since_reward > 50 and self.active_drives:
                self.motivational_state = MotivationalState.CRAVING
                for callback in self._callbacks["craving_triggered"]:
                    callback()
            else:
                self.motivational_state = MotivationalState.SEEKING
        else:
            self.motivational_state = MotivationalState.DEPLETED

        if old_state != self.motivational_state:
            for callback in self._callbacks["motivation_changed"]:
                callback(self.motivational_state)

    def _update_hedonic_tone(self):
        """Update hedonic (pleasure) state"""
        old_tone = self.hedonic_tone

        # Recent reward events
        recent_rewards = [
            r for r in self.reward_history[-10:]
            if time.time() - r.timestamp < 60
        ]

        if recent_rewards:
            avg_magnitude = np.mean([r.actual for r in recent_rewards])
            avg_error = np.mean([r.prediction_error for r in recent_rewards])
        else:
            avg_magnitude = 0.0
            avg_error = 0.0

        # Calculate hedonic value
        hedonic_value = (
            self.dopamine_level * 0.4 +
            avg_magnitude * 0.3 +
            (avg_error + 0.5) * 0.3  # Positive errors feel good
        )

        if hedonic_value > 0.8:
            self.hedonic_tone = HedonicTone.EUPHORIC
        elif hedonic_value > 0.6:
            self.hedonic_tone = HedonicTone.PLEASURABLE
        elif hedonic_value > 0.35:
            self.hedonic_tone = HedonicTone.NEUTRAL
        elif hedonic_value > 0.2:
            self.hedonic_tone = HedonicTone.UNCOMFORTABLE
        else:
            self.hedonic_tone = HedonicTone.ANHEDONIC

        if old_tone != self.hedonic_tone:
            for callback in self._callbacks["hedonic_changed"]:
                callback(self.hedonic_tone)

    def _generate_hormone_effects(self):
        """Generate hormone effects based on reward state"""
        self._hormone_effects = {}

        # Dopamine effect (directly)
        dopamine_delta = self.dopamine_level - self.baseline_dopamine
        self._hormone_effects[HormoneType.DOPAMINE] = dopamine_delta

        # Serotonin (satisfaction, well-being)
        if self.hedonic_tone in [HedonicTone.PLEASURABLE, HedonicTone.EUPHORIC]:
            self._hormone_effects[HormoneType.SEROTONIN] = 0.2
        elif self.hedonic_tone == HedonicTone.ANHEDONIC:
            self._hormone_effects[HormoneType.SEROTONIN] = -0.3

        # Endorphin (pleasure, reward)
        if self.phasic_dopamine > 0.3:
            self._hormone_effects[HormoneType.ENDORPHIN] = self.phasic_dopamine * 0.5

        # Cortisol (craving, frustration)
        if self.motivational_state == MotivationalState.CRAVING:
            self._hormone_effects[HormoneType.CORTISOL] = 0.2
        elif self.motivational_state == MotivationalState.DEPLETED:
            self._hormone_effects[HormoneType.CORTISOL] = 0.1

    def get_hormone_effects(self) -> Dict[HormoneType, float]:
        """Get current hormone effects from reward system"""
        return self._hormone_effects.copy()

    def get_motivation(self) -> float:
        """Get overall motivation level"""
        return self.overall_motivation

    def get_approach_tendency(self, target: str) -> float:
        """Get approach tendency for a specific target"""
        if target in self.active_drives:
            return self.active_drives[target].approach_tendency
        return 0.0

    def predict_reward(self, cue: str) -> Tuple[float, float]:
        """Get predicted reward for a cue (value, confidence)"""
        if cue in self.predictions:
            pred = self.predictions[cue]
            return (pred.expected_value, pred.confidence)
        return (0.0, 0.0)

    def on_reward_received(self, callback: Callable):
        """Register callback for reward delivery"""
        self._callbacks["reward_received"].append(callback)

    def on_prediction_error(self, callback: Callable):
        """Register callback for prediction errors"""
        self._callbacks["prediction_error"].append(callback)

    def on_motivation_changed(self, callback: Callable):
        """Register callback for motivation changes"""
        self._callbacks["motivation_changed"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get reward system status"""
        return {
            "dopamine_level": round(self.dopamine_level, 3),
            "phasic_dopamine": round(self.phasic_dopamine, 3),
            "tonic_dopamine": round(self.tonic_dopamine, 3),
            "motivational_state": self.motivational_state.value,
            "hedonic_tone": self.hedonic_tone.value,
            "overall_motivation": round(self.overall_motivation, 3),
            "active_drives": len(self.active_drives),
            "predictions": len(self.predictions),
            "recent_rewards": len(self.reward_history),
            "cycle_count": self.cycle_count,
            "tolerance": {
                rt.value: round(v, 3) for rt, v in self.tolerance.items()
            },
            "hormone_effects": {
                h.value: round(v, 3) for h, v in self._hormone_effects.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Reward System Demo")
    print("=" * 50)

    # Create reward system
    reward_system = RewardSystem()

    print(f"\n1. Initial state:")
    print(f"   Dopamine: {reward_system.dopamine_level:.2f}")
    print(f"   Motivation: {reward_system.motivational_state.value}")

    # Deliver unexpected reward
    print("\n2. Delivering unexpected reward (0.8 magnitude)...")
    event = reward_system.deliver_reward(RewardType.ACHIEVEMENT, 0.8, source="task_complete")
    print(f"   Prediction error: {event.prediction_error:+.2f}")
    print(f"   Dopamine after: {reward_system.dopamine_level:.2f}")

    # Process cycles
    print("\n3. Processing cycles...")
    for i in range(10):
        reward_system.tick()

    print(f"   Dopamine: {reward_system.dopamine_level:.2f}")
    print(f"   Hedonic tone: {reward_system.hedonic_tone.value}")

    # Add a drive
    print("\n4. Adding motivational drive...")
    reward_system.add_drive("goal_x", RewardType.ACHIEVEMENT, 0.6)

    for i in range(20):
        reward_system.tick()

    print(f"   Motivation state: {reward_system.motivational_state.value}")
    print(f"   Drive intensity: {reward_system.active_drives['goal_x'].intensity:.2f}")

    # Satisfy drive with reward
    print("\n5. Satisfying drive...")
    reward_system.deliver_reward(RewardType.ACHIEVEMENT, 0.7, source="goal_x")
    reward_system.satisfy_drive("goal_x", 0.8)
    reward_system.tick()

    print(f"   Hedonic tone: {reward_system.hedonic_tone.value}")

    # Create learned prediction
    print("\n6. Learning reward prediction...")
    for i in range(10):
        reward_system.deliver_reward(
            RewardType.PRIMARY,
            0.5 + np.random.uniform(-0.1, 0.1),
            prediction_cue="cue_a"
        )
        reward_system.tick()

    pred_value, pred_conf = reward_system.predict_reward("cue_a")
    print(f"   Prediction for 'cue_a': {pred_value:.2f} (confidence: {pred_conf:.2f})")

    # Violate prediction
    print("\n7. Violating prediction (larger than expected)...")
    event = reward_system.deliver_reward(
        RewardType.PRIMARY,
        0.9,
        prediction_cue="cue_a"
    )
    print(f"   Expected: {event.expected:.2f}, Actual: {event.actual:.2f}")
    print(f"   Prediction error: {event.prediction_error:+.2f}")
    print(f"   Dopamine burst: {reward_system.phasic_dopamine:.2f}")

    # Final status
    print("\n8. Final status:")
    status = reward_system.get_status()
    print(f"   Dopamine: {status['dopamine_level']:.2f}")
    print(f"   Motivation: {status['motivational_state']}")
    print(f"   Hedonic tone: {status['hedonic_tone']}")
