"""
Stress Response System

Implements comprehensive stress cascade for digital organisms including:
- Acute stress response (fight-or-flight)
- Chronic stress adaptation
- Allostatic load tracking
- Stress resilience and recovery
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

class StressorType(Enum):
    """Types of stressors"""
    PHYSICAL = "physical"           # Physical demands, exertion
    COGNITIVE = "cognitive"         # Mental load, complexity
    EMOTIONAL = "emotional"         # Emotional challenges
    SOCIAL = "social"               # Social conflicts, isolation
    ENVIRONMENTAL = "environmental" # Resource scarcity, threats
    METABOLIC = "metabolic"         # Energy depletion, hunger


class StressPhase(Enum):
    """Phases of stress response"""
    BASELINE = "baseline"           # Normal, unstressed
    ALARM = "alarm"                 # Initial stress detection
    RESISTANCE = "resistance"       # Active coping
    EXHAUSTION = "exhaustion"       # Resource depletion
    RECOVERY = "recovery"           # Post-stress recovery


class CopingStyle(Enum):
    """Stress coping strategies"""
    ACTIVE = "active"               # Fight, problem-solving
    AVOIDANT = "avoidant"           # Flight, withdrawal
    PASSIVE = "passive"             # Freeze, wait
    SOCIAL = "social"               # Seek support
    ADAPTIVE = "adaptive"           # Flexible coping


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Stressor:
    """A stress-inducing event or condition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: StressorType = StressorType.PHYSICAL
    intensity: float = 0.5          # 0-1 severity
    duration: float = 0.0           # How long it's been active
    controllability: float = 0.5    # How controllable (0=none, 1=full)
    predictability: float = 0.5     # How predictable (0=none, 1=full)
    novelty: float = 0.5            # How novel (0=familiar, 1=new)
    active: bool = True
    timestamp: float = field(default_factory=time.time)


@dataclass
class StressResponse:
    """Record of a stress response"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stressor_id: str = ""
    phase: StressPhase = StressPhase.ALARM
    intensity: float = 0.0
    hormone_release: Dict[str, float] = field(default_factory=dict)
    coping_style: CopingStyle = CopingStyle.ADAPTIVE
    effectiveness: float = 0.5      # How well coping worked
    timestamp: float = field(default_factory=time.time)


@dataclass
class AllostaticLoad:
    """Cumulative stress burden"""
    total_load: float = 0.0         # Overall burden (0-1)
    physical_load: float = 0.0      # Physical stress accumulation
    cognitive_load: float = 0.0     # Cognitive stress accumulation
    emotional_load: float = 0.0     # Emotional stress accumulation
    recovery_capacity: float = 1.0  # Ability to recover
    resilience: float = 0.5         # Stress resilience factor


# ============================================================================
# Stress System
# ============================================================================

class StressSystem:
    """
    Complete stress response system for digital organisms.

    Features:
    - Multi-stage stress response (alarm, resistance, exhaustion)
    - Multiple stressor types with different effects
    - Allostatic load tracking
    - Stress resilience and adaptation
    - Coping strategy selection

    Example:
        stress_system = StressSystem()

        # Apply a stressor
        stress_system.add_stressor(StressorType.COGNITIVE, 0.7)

        # Process stress response
        stress_system.tick()

        # Get hormone effects
        effects = stress_system.get_hormone_effects()
    """

    def __init__(
        self,
        base_resilience: float = 0.5,
        recovery_rate: float = 0.02,
        coping_style: CopingStyle = CopingStyle.ADAPTIVE
    ):
        self.base_resilience = base_resilience
        self.recovery_rate = recovery_rate
        self.default_coping = coping_style

        # Current stressors
        self.active_stressors: Dict[str, Stressor] = {}

        # Stress state
        self.phase = StressPhase.BASELINE
        self.stress_level: float = 0.0         # Current acute stress
        self.chronic_stress: float = 0.0       # Long-term stress

        # Allostatic load
        self.allostatic_load = AllostaticLoad(
            resilience=base_resilience
        )

        # Response history
        self.response_history: List[StressResponse] = []
        self.max_history = 200

        # Hormone effects
        self._hormone_effects: Dict[HormoneType, float] = {}

        # Timing
        self.cycle_count = 0
        self.phase_start_cycle = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "phase_changed": [],
            "stressor_added": [],
            "high_stress": [],
            "exhaustion_reached": [],
            "recovery_complete": []
        }

    def add_stressor(
        self,
        stressor_type: StressorType,
        intensity: float,
        controllability: float = 0.5,
        predictability: float = 0.5,
        novelty: float = 0.5
    ) -> str:
        """Add a new stressor"""
        stressor = Stressor(
            type=stressor_type,
            intensity=intensity,
            controllability=controllability,
            predictability=predictability,
            novelty=novelty
        )

        self.active_stressors[stressor.id] = stressor

        # Trigger callback
        for callback in self._callbacks["stressor_added"]:
            callback(stressor)

        return stressor.id

    def remove_stressor(self, stressor_id: str):
        """Remove an active stressor"""
        if stressor_id in self.active_stressors:
            self.active_stressors[stressor_id].active = False
            del self.active_stressors[stressor_id]

    def tick(self):
        """Process one stress cycle"""
        self.cycle_count += 1

        # Update stressor durations
        for stressor in self.active_stressors.values():
            stressor.duration += 1

        # Calculate total stress load
        self._calculate_stress_level()

        # Update phase
        self._update_phase()

        # Generate hormone effects
        self._generate_hormone_effects()

        # Update allostatic load
        self._update_allostatic_load()

        # Natural recovery
        self._apply_recovery()

    def _calculate_stress_level(self):
        """Calculate current stress level from all stressors"""
        if not self.active_stressors:
            self.stress_level *= 0.95  # Decay if no stressors
            return

        total_stress = 0.0
        weights = {
            StressorType.PHYSICAL: 1.0,
            StressorType.COGNITIVE: 0.9,
            StressorType.EMOTIONAL: 1.1,
            StressorType.SOCIAL: 0.8,
            StressorType.ENVIRONMENTAL: 1.2,
            StressorType.METABOLIC: 0.7
        }

        for stressor in self.active_stressors.values():
            if not stressor.active:
                continue

            # Base stress from intensity
            stress = stressor.intensity * weights.get(stressor.type, 1.0)

            # Modulate by controllability (less control = more stress)
            stress *= (1.0 + 0.5 * (1.0 - stressor.controllability))

            # Modulate by predictability (less predictable = more stress)
            stress *= (1.0 + 0.3 * (1.0 - stressor.predictability))

            # Modulate by novelty (novel = more stress initially)
            novelty_factor = stressor.novelty * max(0, 1.0 - stressor.duration / 20.0)
            stress *= (1.0 + 0.2 * novelty_factor)

            # Modulate by resilience
            stress /= (1.0 + self.allostatic_load.resilience)

            total_stress += stress

        # Average and cap
        self.stress_level = min(1.0, total_stress / max(1, len(self.active_stressors)))

        # Check for high stress
        if self.stress_level > 0.8:
            for callback in self._callbacks["high_stress"]:
                callback(self.stress_level)

    def _update_phase(self):
        """Update stress response phase"""
        old_phase = self.phase
        cycles_in_phase = self.cycle_count - self.phase_start_cycle

        if self.phase == StressPhase.BASELINE:
            if self.stress_level > 0.3:
                self.phase = StressPhase.ALARM
                self.phase_start_cycle = self.cycle_count

        elif self.phase == StressPhase.ALARM:
            if self.stress_level < 0.2:
                self.phase = StressPhase.RECOVERY
                self.phase_start_cycle = self.cycle_count
            elif cycles_in_phase > 10:
                self.phase = StressPhase.RESISTANCE
                self.phase_start_cycle = self.cycle_count

        elif self.phase == StressPhase.RESISTANCE:
            if self.stress_level < 0.2:
                self.phase = StressPhase.RECOVERY
                self.phase_start_cycle = self.cycle_count
            elif self.chronic_stress > 0.8 or cycles_in_phase > 100:
                self.phase = StressPhase.EXHAUSTION
                self.phase_start_cycle = self.cycle_count
                for callback in self._callbacks["exhaustion_reached"]:
                    callback(self.chronic_stress)

        elif self.phase == StressPhase.EXHAUSTION:
            if self.stress_level < 0.3 and cycles_in_phase > 20:
                self.phase = StressPhase.RECOVERY
                self.phase_start_cycle = self.cycle_count

        elif self.phase == StressPhase.RECOVERY:
            if self.stress_level > 0.5:
                self.phase = StressPhase.ALARM
                self.phase_start_cycle = self.cycle_count
            elif self.chronic_stress < 0.1 and cycles_in_phase > 30:
                self.phase = StressPhase.BASELINE
                self.phase_start_cycle = self.cycle_count
                for callback in self._callbacks["recovery_complete"]:
                    callback()

        if old_phase != self.phase:
            for callback in self._callbacks["phase_changed"]:
                callback(self.phase)

    def _generate_hormone_effects(self):
        """Generate hormone release based on stress state"""
        self._hormone_effects = {}

        if self.phase == StressPhase.BASELINE:
            return

        elif self.phase == StressPhase.ALARM:
            # Acute stress: immediate catecholamine release
            self._hormone_effects[HormoneType.ADRENALINE] = self.stress_level * 0.8
            self._hormone_effects[HormoneType.CORTISOL] = self.stress_level * 0.4
            self._hormone_effects[HormoneType.DOPAMINE] = -self.stress_level * 0.2

        elif self.phase == StressPhase.RESISTANCE:
            # Sustained stress: cortisol dominates
            self._hormone_effects[HormoneType.ADRENALINE] = self.stress_level * 0.3
            self._hormone_effects[HormoneType.CORTISOL] = self.stress_level * 0.7
            self._hormone_effects[HormoneType.SEROTONIN] = -self.stress_level * 0.3
            self._hormone_effects[HormoneType.DOPAMINE] = -self.stress_level * 0.2

        elif self.phase == StressPhase.EXHAUSTION:
            # Depletion: reduced hormone output
            self._hormone_effects[HormoneType.ADRENALINE] = 0.1
            self._hormone_effects[HormoneType.CORTISOL] = self.chronic_stress * 0.5
            self._hormone_effects[HormoneType.SEROTONIN] = -0.4
            self._hormone_effects[HormoneType.DOPAMINE] = -0.3
            self._hormone_effects[HormoneType.ENDORPHIN] = -0.2

        elif self.phase == StressPhase.RECOVERY:
            # Recovery: restorative hormones
            recovery_factor = 1.0 - self.chronic_stress
            self._hormone_effects[HormoneType.SEROTONIN] = recovery_factor * 0.3
            self._hormone_effects[HormoneType.OXYTOCIN] = recovery_factor * 0.2
            self._hormone_effects[HormoneType.GROWTH_HORMONE] = recovery_factor * 0.2
            self._hormone_effects[HormoneType.CORTISOL] = -recovery_factor * 0.2

    def _update_allostatic_load(self):
        """Update cumulative stress burden"""
        load = self.allostatic_load

        # Accumulate load by stressor type
        for stressor in self.active_stressors.values():
            if not stressor.active:
                continue

            increment = stressor.intensity * 0.001

            if stressor.type == StressorType.PHYSICAL:
                load.physical_load = min(1.0, load.physical_load + increment)
            elif stressor.type == StressorType.COGNITIVE:
                load.cognitive_load = min(1.0, load.cognitive_load + increment)
            elif stressor.type in [StressorType.EMOTIONAL, StressorType.SOCIAL]:
                load.emotional_load = min(1.0, load.emotional_load + increment)

        # Calculate total load
        load.total_load = (
            load.physical_load * 0.3 +
            load.cognitive_load * 0.3 +
            load.emotional_load * 0.4
        )

        # Update chronic stress
        self.chronic_stress = (self.chronic_stress * 0.99 + load.total_load * 0.01)

        # High load reduces recovery capacity
        load.recovery_capacity = max(0.2, 1.0 - load.total_load * 0.5)

        # Chronic stress can reduce resilience
        if self.chronic_stress > 0.7:
            load.resilience = max(0.1, load.resilience - 0.001)
        elif self.chronic_stress < 0.2 and self.phase == StressPhase.RECOVERY:
            load.resilience = min(1.0, load.resilience + 0.002)

    def _apply_recovery(self):
        """Apply natural recovery processes"""
        if self.phase in [StressPhase.BASELINE, StressPhase.RECOVERY]:
            load = self.allostatic_load
            recovery = self.recovery_rate * load.recovery_capacity

            load.physical_load = max(0.0, load.physical_load - recovery)
            load.cognitive_load = max(0.0, load.cognitive_load - recovery)
            load.emotional_load = max(0.0, load.emotional_load - recovery)

    def get_hormone_effects(self) -> Dict[HormoneType, float]:
        """Get current hormone effects from stress"""
        return self._hormone_effects.copy()

    def get_coping_effectiveness(self, style: CopingStyle) -> float:
        """Calculate effectiveness of a coping style for current stressors"""
        if not self.active_stressors:
            return 1.0

        effectiveness = 0.0
        count = 0

        for stressor in self.active_stressors.values():
            if not stressor.active:
                continue

            # Different coping styles work better for different situations
            if style == CopingStyle.ACTIVE:
                # Active coping works for controllable stressors
                eff = stressor.controllability * 0.8 + 0.2
            elif style == CopingStyle.AVOIDANT:
                # Avoidance works for uncontrollable, short-term
                eff = (1.0 - stressor.controllability) * 0.6
                if stressor.duration > 20:
                    eff *= 0.5  # Less effective over time
            elif style == CopingStyle.PASSIVE:
                # Passive works for unpredictable, uncontrollable
                eff = (1.0 - stressor.controllability) * (1.0 - stressor.predictability) * 0.5
            elif style == CopingStyle.SOCIAL:
                # Social coping works for emotional/social stressors
                if stressor.type in [StressorType.EMOTIONAL, StressorType.SOCIAL]:
                    eff = 0.8
                else:
                    eff = 0.4
            else:  # ADAPTIVE
                # Adaptive balances based on situation
                eff = 0.5 + 0.3 * stressor.controllability

            effectiveness += eff
            count += 1

        return effectiveness / max(1, count)

    def apply_coping(self, style: CopingStyle) -> float:
        """Apply a coping strategy and get stress reduction"""
        effectiveness = self.get_coping_effectiveness(style)

        # Reduce stress based on effectiveness
        reduction = effectiveness * 0.1
        self.stress_level = max(0.0, self.stress_level - reduction)

        # Record response
        response = StressResponse(
            phase=self.phase,
            intensity=self.stress_level,
            coping_style=style,
            effectiveness=effectiveness
        )
        self.response_history.append(response)
        if len(self.response_history) > self.max_history:
            self.response_history.pop(0)

        return reduction

    def on_phase_changed(self, callback: Callable):
        """Register callback for phase changes"""
        self._callbacks["phase_changed"].append(callback)

    def on_high_stress(self, callback: Callable):
        """Register callback for high stress"""
        self._callbacks["high_stress"].append(callback)

    def on_exhaustion_reached(self, callback: Callable):
        """Register callback for exhaustion"""
        self._callbacks["exhaustion_reached"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get stress system status"""
        return {
            "phase": self.phase.value,
            "stress_level": round(self.stress_level, 3),
            "chronic_stress": round(self.chronic_stress, 3),
            "active_stressors": len(self.active_stressors),
            "cycle_count": self.cycle_count,
            "allostatic_load": {
                "total": round(self.allostatic_load.total_load, 3),
                "physical": round(self.allostatic_load.physical_load, 3),
                "cognitive": round(self.allostatic_load.cognitive_load, 3),
                "emotional": round(self.allostatic_load.emotional_load, 3),
                "resilience": round(self.allostatic_load.resilience, 3),
                "recovery_capacity": round(self.allostatic_load.recovery_capacity, 3)
            },
            "hormone_effects": {
                h.value: round(v, 3) for h, v in self._hormone_effects.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Stress Response System Demo")
    print("=" * 50)

    # Create stress system
    stress_system = StressSystem(base_resilience=0.6)

    print(f"\n1. Initial state:")
    print(f"   Phase: {stress_system.phase.value}")
    print(f"   Stress level: {stress_system.stress_level}")

    # Add a cognitive stressor
    print("\n2. Adding cognitive stressor (0.6 intensity)...")
    stress_system.add_stressor(StressorType.COGNITIVE, 0.6)
    stress_system.tick()

    print(f"   Phase: {stress_system.phase.value}")
    print(f"   Stress level: {stress_system.stress_level:.2f}")

    # Add more stressors
    print("\n3. Adding additional stressors...")
    stress_system.add_stressor(StressorType.EMOTIONAL, 0.5)
    stress_system.add_stressor(StressorType.SOCIAL, 0.4)

    for i in range(20):
        stress_system.tick()

        if i % 5 == 0:
            status = stress_system.get_status()
            print(f"   Cycle {i}: phase={status['phase']}, "
                  f"stress={status['stress_level']:.2f}, "
                  f"chronic={status['chronic_stress']:.2f}")

    # Check hormone effects
    print("\n4. Hormone effects:")
    effects = stress_system.get_hormone_effects()
    for hormone, effect in effects.items():
        if effect != 0:
            print(f"   {hormone.value}: {effect:+.2f}")

    # Try coping strategies
    print("\n5. Coping strategy effectiveness:")
    for style in CopingStyle:
        eff = stress_system.get_coping_effectiveness(style)
        print(f"   {style.value}: {eff:.2f}")

    # Apply coping
    print("\n6. Applying adaptive coping...")
    for i in range(10):
        reduction = stress_system.apply_coping(CopingStyle.ADAPTIVE)
        stress_system.tick()

    print(f"   Stress after coping: {stress_system.stress_level:.2f}")

    # Remove stressors for recovery
    print("\n7. Removing stressors for recovery...")
    stress_system.active_stressors.clear()

    for i in range(50):
        stress_system.tick()

        if i % 10 == 0:
            status = stress_system.get_status()
            print(f"   Cycle {i}: phase={status['phase']}, "
                  f"chronic={status['chronic_stress']:.2f}")

    # Final status
    print("\n8. Final status:")
    status = stress_system.get_status()
    print(f"   Phase: {status['phase']}")
    print(f"   Resilience: {status['allostatic_load']['resilience']:.2f}")
    print(f"   Recovery capacity: {status['allostatic_load']['recovery_capacity']:.2f}")
