"""
Endocrine Feedback Loop System

Implements hormonal feedback regulation for digital organisms including:
- Negative feedback (homeostatic regulation)
- Positive feedback (amplification cascades)
- Multi-level feedback hierarchies (HPA, HPT, HPG axes)
- Dynamic setpoint adjustment
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

from .hormones import HormoneType, GlandType


# ============================================================================
# Enums
# ============================================================================

class FeedbackType(Enum):
    """Types of feedback mechanisms"""
    NEGATIVE = "negative"              # Inhibitory, homeostatic
    POSITIVE = "positive"              # Amplifying, cascade
    ULTRA_SHORT = "ultra_short"        # Same-gland immediate
    SHORT = "short"                    # Direct gland-to-gland
    LONG = "long"                      # Full axis feedback


class AxisType(Enum):
    """Major neuroendocrine axes"""
    HPA = "hpa"    # Hypothalamic-Pituitary-Adrenal (stress)
    HPT = "hpt"    # Hypothalamic-Pituitary-Thyroid (metabolism)
    HPG = "hpg"    # Hypothalamic-Pituitary-Gonadal (not implemented)
    REWARD = "reward"  # Dopamine reward axis
    SOCIAL = "social"  # Oxytocin social axis


class FeedbackState(Enum):
    """Current state of a feedback loop"""
    INACTIVE = "inactive"
    MONITORING = "monitoring"
    INHIBITING = "inhibiting"
    AMPLIFYING = "amplifying"
    SATURATED = "saturated"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FeedbackLoop:
    """A single feedback loop connection"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Connection points
    sensor_hormone: HormoneType = HormoneType.CORTISOL
    target_gland: GlandType = GlandType.ADRENAL
    target_hormone: Optional[HormoneType] = None

    # Feedback properties
    feedback_type: FeedbackType = FeedbackType.NEGATIVE
    gain: float = 1.0                     # Strength of feedback
    threshold: float = 0.5                # Activation threshold
    saturation: float = 0.9               # Saturation point

    # Dynamics
    delay: float = 0.0                    # Response delay (cycles)
    time_constant: float = 10.0           # How fast effect builds

    # State
    state: FeedbackState = FeedbackState.INACTIVE
    current_effect: float = 0.0           # Current feedback effect (-1 to 1)
    activation_time: float = 0.0

    # Setpoint
    setpoint: float = 0.5                 # Target hormone level
    setpoint_plasticity: float = 0.01     # How fast setpoint can shift


@dataclass
class NeuroendocrineAxis:
    """A complete neuroendocrine axis with multiple feedback levels"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: AxisType = AxisType.HPA
    name: str = ""

    # Axis components
    hypothalamic_signal: float = 0.0      # Releasing hormone level
    pituitary_signal: float = 0.0         # Tropic hormone level
    peripheral_signal: float = 0.0        # Target organ hormone

    # Feedback loops in this axis
    loops: List[str] = field(default_factory=list)  # Loop IDs

    # Axis state
    active: bool = True
    activation_level: float = 0.0
    tone: float = 0.5                     # Basal activity level

    # History
    activation_history: List[float] = field(default_factory=list)


@dataclass
class SetpointShift:
    """Record of a setpoint adjustment"""
    loop_id: str = ""
    old_setpoint: float = 0.0
    new_setpoint: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Feedback System
# ============================================================================

class FeedbackSystem:
    """
    Complete endocrine feedback regulation system.

    Features:
    - Negative feedback for homeostasis
    - Positive feedback for amplification
    - Multi-level axis regulation (HPA, HPT)
    - Adaptive setpoint adjustment
    - Feedback loop dynamics with delays

    Example:
        feedback_system = FeedbackSystem()

        # Provide hormone levels
        hormone_levels = {HormoneType.CORTISOL: 0.8}

        # Calculate feedback effects
        effects = feedback_system.calculate_feedback(hormone_levels)

        # Apply to glands
        for gland_type, effect in effects.items():
            # Modify gland production by effect amount
            pass
    """

    def __init__(
        self,
        enable_plasticity: bool = True,
        feedback_strength: float = 1.0
    ):
        self.enable_plasticity = enable_plasticity
        self.feedback_strength = feedback_strength

        # Feedback loops
        self.loops: Dict[str, FeedbackLoop] = self._initialize_loops()

        # Neuroendocrine axes
        self.axes: Dict[AxisType, NeuroendocrineAxis] = self._initialize_axes()

        # Setpoint history
        self.setpoint_history: List[SetpointShift] = []
        self.max_history = 100

        # Cycle tracking
        self.cycle_count = 0

        # Pending feedback effects (for delayed feedback)
        self._pending_effects: List[Tuple[int, GlandType, float]] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "feedback_activated": [],
            "setpoint_shifted": [],
            "axis_activated": [],
            "saturation_reached": []
        }

    def _initialize_loops(self) -> Dict[str, FeedbackLoop]:
        """Initialize all feedback loops"""
        loops = {}

        # HPA Axis loops (stress response)
        loops["hpa_long_negative"] = FeedbackLoop(
            name="HPA Long Negative",
            sensor_hormone=HormoneType.CORTISOL,
            target_gland=GlandType.HYPOTHALAMUS,
            target_hormone=None,
            feedback_type=FeedbackType.LONG,
            gain=0.8,
            threshold=0.4,
            setpoint=0.35,
            delay=2.0,
            time_constant=15.0
        )

        loops["hpa_short_negative"] = FeedbackLoop(
            name="HPA Short Negative",
            sensor_hormone=HormoneType.CORTISOL,
            target_gland=GlandType.PITUITARY,
            feedback_type=FeedbackType.SHORT,
            gain=0.6,
            threshold=0.5,
            setpoint=0.4,
            delay=1.0,
            time_constant=10.0
        )

        loops["adrenal_ultrashort"] = FeedbackLoop(
            name="Adrenal Ultra-short",
            sensor_hormone=HormoneType.ADRENALINE,
            target_gland=GlandType.ADRENAL,
            feedback_type=FeedbackType.ULTRA_SHORT,
            gain=0.4,
            threshold=0.6,
            setpoint=0.3,
            delay=0.0,
            time_constant=5.0
        )

        # HPT Axis loops (metabolism)
        loops["hpt_negative"] = FeedbackLoop(
            name="HPT Negative",
            sensor_hormone=HormoneType.THYROXINE,
            target_gland=GlandType.PITUITARY,
            feedback_type=FeedbackType.LONG,
            gain=0.7,
            threshold=0.4,
            setpoint=0.5,
            delay=3.0,
            time_constant=20.0
        )

        loops["thyroid_autoregulation"] = FeedbackLoop(
            name="Thyroid Autoregulation",
            sensor_hormone=HormoneType.THYROXINE,
            target_gland=GlandType.THYROID,
            feedback_type=FeedbackType.ULTRA_SHORT,
            gain=0.3,
            threshold=0.7,
            setpoint=0.5,
            delay=0.0,
            time_constant=8.0
        )

        # Reward system loops
        loops["dopamine_negative"] = FeedbackLoop(
            name="Dopamine Negative",
            sensor_hormone=HormoneType.DOPAMINE,
            target_gland=GlandType.REWARD_CENTER,
            feedback_type=FeedbackType.SHORT,
            gain=0.5,
            threshold=0.6,
            setpoint=0.45,
            delay=1.0,
            time_constant=8.0
        )

        loops["serotonin_negative"] = FeedbackLoop(
            name="Serotonin Negative",
            sensor_hormone=HormoneType.SEROTONIN,
            target_gland=GlandType.REWARD_CENTER,
            feedback_type=FeedbackType.SHORT,
            gain=0.4,
            threshold=0.5,
            setpoint=0.5,
            delay=2.0,
            time_constant=12.0
        )

        # Positive feedback loop (rare, amplification)
        loops["endorphin_positive"] = FeedbackLoop(
            name="Endorphin Positive",
            sensor_hormone=HormoneType.ENDORPHIN,
            target_gland=GlandType.PITUITARY,
            target_hormone=HormoneType.ENDORPHIN,
            feedback_type=FeedbackType.POSITIVE,
            gain=0.3,
            threshold=0.7,
            saturation=0.95,
            delay=0.0,
            time_constant=5.0
        )

        # Growth regulation
        loops["growth_negative"] = FeedbackLoop(
            name="Growth Hormone Negative",
            sensor_hormone=HormoneType.GROWTH_HORMONE,
            target_gland=GlandType.PITUITARY,
            feedback_type=FeedbackType.SHORT,
            gain=0.5,
            threshold=0.6,
            setpoint=0.4,
            delay=1.0,
            time_constant=15.0
        )

        # Metabolic regulation
        loops["insulin_negative"] = FeedbackLoop(
            name="Insulin Negative",
            sensor_hormone=HormoneType.INSULIN,
            target_gland=GlandType.PITUITARY,
            feedback_type=FeedbackType.LONG,
            gain=0.6,
            threshold=0.6,
            setpoint=0.5,
            delay=2.0,
            time_constant=10.0
        )

        # Circadian regulation
        loops["melatonin_negative"] = FeedbackLoop(
            name="Melatonin Negative",
            sensor_hormone=HormoneType.MELATONIN,
            target_gland=GlandType.HYPOTHALAMUS,
            feedback_type=FeedbackType.LONG,
            gain=0.4,
            threshold=0.5,
            setpoint=0.35,
            delay=4.0,
            time_constant=25.0
        )

        return loops

    def _initialize_axes(self) -> Dict[AxisType, NeuroendocrineAxis]:
        """Initialize neuroendocrine axes"""
        return {
            AxisType.HPA: NeuroendocrineAxis(
                type=AxisType.HPA,
                name="Hypothalamic-Pituitary-Adrenal Axis",
                loops=["hpa_long_negative", "hpa_short_negative", "adrenal_ultrashort"],
                tone=0.3
            ),
            AxisType.HPT: NeuroendocrineAxis(
                type=AxisType.HPT,
                name="Hypothalamic-Pituitary-Thyroid Axis",
                loops=["hpt_negative", "thyroid_autoregulation"],
                tone=0.5
            ),
            AxisType.REWARD: NeuroendocrineAxis(
                type=AxisType.REWARD,
                name="Reward Processing Axis",
                loops=["dopamine_negative", "serotonin_negative", "endorphin_positive"],
                tone=0.4
            ),
            AxisType.SOCIAL: NeuroendocrineAxis(
                type=AxisType.SOCIAL,
                name="Social Bonding Axis",
                loops=[],  # To be expanded
                tone=0.3
            )
        }

    def calculate_feedback(
        self,
        hormone_levels: Dict[HormoneType, float]
    ) -> Dict[GlandType, float]:
        """
        Calculate feedback effects on gland production.

        Returns dict mapping glands to production modifiers (-1 to 1).
        Negative = inhibition, Positive = stimulation
        """
        effects: Dict[GlandType, float] = {gland: 0.0 for gland in GlandType}

        for loop_id, loop in self.loops.items():
            if loop.sensor_hormone not in hormone_levels:
                continue

            hormone_level = hormone_levels[loop.sensor_hormone]
            effect = self._calculate_loop_effect(loop, hormone_level)

            if effect != 0.0:
                # Handle delay
                if loop.delay > 0:
                    apply_cycle = self.cycle_count + int(loop.delay)
                    self._pending_effects.append((apply_cycle, loop.target_gland, effect))
                else:
                    effects[loop.target_gland] += effect

        # Apply pending delayed effects
        current_effects = [
            (gland, eff) for cycle, gland, eff in self._pending_effects
            if cycle <= self.cycle_count
        ]
        for gland, eff in current_effects:
            effects[gland] += eff

        # Clean up applied pending effects
        self._pending_effects = [
            (cycle, gland, eff) for cycle, gland, eff in self._pending_effects
            if cycle > self.cycle_count
        ]

        # Clamp effects
        for gland in effects:
            effects[gland] = max(-1.0, min(1.0, effects[gland] * self.feedback_strength))

        return effects

    def _calculate_loop_effect(
        self,
        loop: FeedbackLoop,
        hormone_level: float
    ) -> float:
        """Calculate effect of a single feedback loop"""
        old_state = loop.state

        # Check threshold
        if hormone_level < loop.threshold and loop.feedback_type != FeedbackType.POSITIVE:
            loop.state = FeedbackState.MONITORING
            loop.current_effect *= 0.9  # Decay effect
            return 0.0

        # Calculate deviation from setpoint
        deviation = hormone_level - loop.setpoint

        if loop.feedback_type == FeedbackType.POSITIVE:
            # Positive feedback: amplify when above threshold
            if hormone_level >= loop.threshold:
                if hormone_level >= loop.saturation:
                    loop.state = FeedbackState.SATURATED
                    for callback in self._callbacks["saturation_reached"]:
                        callback(loop.id)
                    return 0.0  # Saturated, no more amplification
                else:
                    loop.state = FeedbackState.AMPLIFYING
                    effect = deviation * loop.gain
                    effect = min(effect, loop.saturation - hormone_level)
            else:
                loop.state = FeedbackState.MONITORING
                effect = 0.0
        else:
            # Negative feedback: inhibit when above setpoint
            loop.state = FeedbackState.INHIBITING if deviation > 0 else FeedbackState.MONITORING
            effect = -deviation * loop.gain

        # Apply time constant (smoothing)
        loop.current_effect += (effect - loop.current_effect) / loop.time_constant

        # Trigger callbacks on state change
        if old_state != loop.state and loop.state in [FeedbackState.INHIBITING, FeedbackState.AMPLIFYING]:
            for callback in self._callbacks["feedback_activated"]:
                callback(loop.id, loop.feedback_type, loop.current_effect)

        return loop.current_effect

    def update_axis(
        self,
        axis_type: AxisType,
        hormone_levels: Dict[HormoneType, float]
    ):
        """Update a neuroendocrine axis based on hormone levels"""
        if axis_type not in self.axes:
            return

        axis = self.axes[axis_type]
        if not axis.active:
            return

        # Calculate axis activation based on component loops
        total_activation = 0.0
        active_loops = 0

        for loop_id in axis.loops:
            if loop_id in self.loops:
                loop = self.loops[loop_id]
                if loop.sensor_hormone in hormone_levels:
                    level = hormone_levels[loop.sensor_hormone]
                    total_activation += level
                    active_loops += 1

        if active_loops > 0:
            axis.activation_level = total_activation / active_loops
        else:
            axis.activation_level = axis.tone

        # Track history
        axis.activation_history.append(axis.activation_level)
        if len(axis.activation_history) > 100:
            axis.activation_history.pop(0)

        # Trigger callback
        if axis.activation_level > 0.7:
            for callback in self._callbacks["axis_activated"]:
                callback(axis_type, axis.activation_level)

    def shift_setpoint(
        self,
        loop_id: str,
        direction: float,
        reason: str = ""
    ):
        """Shift a feedback loop's setpoint (allostasis)"""
        if not self.enable_plasticity:
            return

        if loop_id not in self.loops:
            return

        loop = self.loops[loop_id]
        old_setpoint = loop.setpoint

        # Shift setpoint
        shift = direction * loop.setpoint_plasticity
        loop.setpoint += shift
        loop.setpoint = max(0.1, min(0.9, loop.setpoint))

        # Record shift
        self.setpoint_history.append(SetpointShift(
            loop_id=loop_id,
            old_setpoint=old_setpoint,
            new_setpoint=loop.setpoint,
            reason=reason
        ))

        if len(self.setpoint_history) > self.max_history:
            self.setpoint_history.pop(0)

        # Trigger callback
        for callback in self._callbacks["setpoint_shifted"]:
            callback(loop_id, old_setpoint, loop.setpoint)

    def apply_chronic_stress(self, intensity: float, duration: int):
        """Simulate chronic stress effects on feedback loops"""
        if not self.enable_plasticity:
            return

        # Chronic stress shifts HPA setpoints upward (allostatic load)
        for loop_id in ["hpa_long_negative", "hpa_short_negative"]:
            if loop_id in self.loops:
                shift = intensity * 0.01 * duration
                self.shift_setpoint(loop_id, shift, "chronic_stress")

    def tick(self, hormone_levels: Optional[Dict[HormoneType, float]] = None):
        """Process one feedback cycle"""
        self.cycle_count += 1

        if hormone_levels:
            # Update all axes
            for axis_type in self.axes:
                self.update_axis(axis_type, hormone_levels)

        # Natural setpoint drift toward baseline
        if self.enable_plasticity:
            for loop in self.loops.values():
                baseline = 0.5  # Default baseline
                diff = baseline - loop.setpoint
                loop.setpoint += diff * 0.001  # Very slow drift

    def get_loop_state(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed state of a feedback loop"""
        if loop_id not in self.loops:
            return None

        loop = self.loops[loop_id]
        return {
            "name": loop.name,
            "type": loop.feedback_type.value,
            "state": loop.state.value,
            "setpoint": loop.setpoint,
            "current_effect": loop.current_effect,
            "sensor": loop.sensor_hormone.value,
            "target": loop.target_gland.value,
            "gain": loop.gain
        }

    def get_axis_state(self, axis_type: AxisType) -> Optional[Dict[str, Any]]:
        """Get detailed state of an axis"""
        if axis_type not in self.axes:
            return None

        axis = self.axes[axis_type]
        return {
            "name": axis.name,
            "type": axis.type.value,
            "active": axis.active,
            "activation_level": axis.activation_level,
            "tone": axis.tone,
            "loop_count": len(axis.loops),
            "recent_history": axis.activation_history[-10:] if axis.activation_history else []
        }

    def on_feedback_activated(self, callback: Callable):
        """Register callback for feedback activation"""
        self._callbacks["feedback_activated"].append(callback)

    def on_setpoint_shifted(self, callback: Callable):
        """Register callback for setpoint changes"""
        self._callbacks["setpoint_shifted"].append(callback)

    def on_axis_activated(self, callback: Callable):
        """Register callback for axis activation"""
        self._callbacks["axis_activated"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get feedback system status"""
        loop_summary = {}
        for loop_id, loop in self.loops.items():
            loop_summary[loop_id] = {
                "state": loop.state.value,
                "setpoint": round(loop.setpoint, 3),
                "effect": round(loop.current_effect, 3)
            }

        axis_summary = {}
        for axis_type, axis in self.axes.items():
            axis_summary[axis_type.value] = {
                "activation": round(axis.activation_level, 3),
                "active": axis.active
            }

        return {
            "cycle_count": self.cycle_count,
            "total_loops": len(self.loops),
            "pending_effects": len(self._pending_effects),
            "setpoint_shifts": len(self.setpoint_history),
            "loops": loop_summary,
            "axes": axis_summary,
            "plasticity_enabled": self.enable_plasticity
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Feedback Loop System Demo")
    print("=" * 50)

    # Create feedback system
    feedback_system = FeedbackSystem()

    print(f"\n1. Initialized {len(feedback_system.loops)} feedback loops")
    print(f"   Axes: {list(feedback_system.axes.keys())}")

    # Simulate normal hormone levels
    print("\n2. Normal hormone levels:")
    normal_levels = {
        HormoneType.CORTISOL: 0.35,
        HormoneType.DOPAMINE: 0.45,
        HormoneType.THYROXINE: 0.5
    }

    effects = feedback_system.calculate_feedback(normal_levels)
    print(f"   Gland effects: {effects}")

    # Simulate stress response (high cortisol)
    print("\n3. Stress response (high cortisol):")
    stress_levels = {
        HormoneType.CORTISOL: 0.8,
        HormoneType.ADRENALINE: 0.7,
        HormoneType.DOPAMINE: 0.3
    }

    for i in range(10):
        effects = feedback_system.calculate_feedback(stress_levels)
        feedback_system.tick(stress_levels)

        if i % 3 == 0:
            hpa_effect = effects[GlandType.ADRENAL]
            print(f"   Cycle {i}: Adrenal effect = {hpa_effect:.3f}")

    # Check HPA axis
    print("\n4. HPA Axis state:")
    hpa_state = feedback_system.get_axis_state(AxisType.HPA)
    print(f"   Activation: {hpa_state['activation_level']:.3f}")

    # Simulate chronic stress
    print("\n5. Simulating chronic stress...")
    feedback_system.apply_chronic_stress(0.5, 20)

    # Check setpoint shifts
    print(f"   Setpoint shifts: {len(feedback_system.setpoint_history)}")
    for shift in feedback_system.setpoint_history[-3:]:
        print(f"   - {shift.loop_id}: {shift.old_setpoint:.3f} -> {shift.new_setpoint:.3f}")

    # Recovery period
    print("\n6. Recovery period (low cortisol):")
    recovery_levels = {
        HormoneType.CORTISOL: 0.25,
        HormoneType.DOPAMINE: 0.5
    }

    for i in range(10):
        effects = feedback_system.calculate_feedback(recovery_levels)
        feedback_system.tick(recovery_levels)

        if i % 3 == 0:
            hpa_effect = effects[GlandType.ADRENAL]
            print(f"   Cycle {i}: Adrenal effect = {hpa_effect:.3f}")

    # Final status
    print("\n7. Final status:")
    status = feedback_system.get_status()
    print(f"   Total cycles: {status['cycle_count']}")
    print(f"   Pending effects: {status['pending_effects']}")
    print(f"   HPA axis: {status['axes']['hpa']}")
