"""
Homeostatic Regulation System

Maintains internal stability and balance in digital organisms.
Implements feedback control loops to regulate vital parameters
within optimal ranges.

Key features:
- Vital sign monitoring
- Setpoint regulation
- Stress response
- Adaptive regulation
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Enums
# ============================================================================

class VitalSignType(Enum):
    """Types of vital signs to monitor"""
    ENERGY_LEVEL = "energy_level"
    CONSCIOUSNESS_SCORE = "consciousness_score"
    MEMORY_USAGE = "memory_usage"
    PROCESSING_LOAD = "processing_load"
    COMMUNICATION_RATE = "communication_rate"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    COHERENCE = "coherence"


class RegulationType(Enum):
    """Types of regulation"""
    PROPORTIONAL = "proportional"    # Simple proportional control
    PID = "pid"                      # PID controller
    ADAPTIVE = "adaptive"            # Self-adjusting
    PREDICTIVE = "predictive"        # Anticipatory control


class StressLevel(Enum):
    """Stress levels"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class VitalSign:
    """A vital sign to be monitored"""
    type: VitalSignType
    current_value: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    min_safe: float = 0.0
    max_safe: float = 1.0
    optimal: float = 0.5
    weight: float = 1.0  # Importance weight

    def update(self, value: float):
        """Update vital sign value"""
        self.current_value = value
        self.history.append(value)

    @property
    def deviation(self) -> float:
        """Get deviation from optimal"""
        return abs(self.current_value - self.optimal)

    @property
    def in_safe_range(self) -> bool:
        """Check if value is in safe range"""
        return self.min_safe <= self.current_value <= self.max_safe

    @property
    def trend(self) -> float:
        """Get trend (positive = increasing)"""
        if len(self.history) < 2:
            return 0.0
        recent = list(self.history)[-10:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)


@dataclass
class Setpoint:
    """Target setpoint for regulation"""
    vital_sign: VitalSignType
    target: float
    tolerance: float = 0.1
    priority: int = 1
    adaptive: bool = True  # Can adjust based on conditions


@dataclass
class RegulationResponse:
    """Response to a regulation need"""
    vital_sign: VitalSignType
    action: str
    magnitude: float
    timestamp: float = field(default_factory=time.time)
    success: bool = False


@dataclass
class StressResponse:
    """Response to stress conditions"""
    level: StressLevel
    stressors: List[str]
    responses: List[str]
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# PID Controller
# ============================================================================

class PIDController:
    """
    PID (Proportional-Integral-Derivative) controller.

    Classic feedback control algorithm.
    """

    def __init__(
        self,
        kp: float = 1.0,  # Proportional gain
        ki: float = 0.1,  # Integral gain
        kd: float = 0.05, # Derivative gain
        output_limits: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()

    def compute(self, setpoint: float, measured: float) -> float:
        """
        Compute control output.

        Args:
            setpoint: Target value
            measured: Current measured value

        Returns:
            Control output
        """
        current_time = time.time()
        dt = current_time - self._last_time

        if dt <= 0:
            dt = 0.01

        # Calculate error
        error = setpoint - measured

        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            self.output_limits[0] / max(0.01, self.ki),
            self.output_limits[1] / max(0.01, self.ki)
        )
        i_term = self.ki * self._integral

        # Derivative term
        d_term = self.kd * (error - self._previous_error) / dt

        # Calculate output
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update state
        self._previous_error = error
        self._last_time = current_time

        return output

    def reset(self):
        """Reset controller state"""
        self._integral = 0.0
        self._previous_error = 0.0


# ============================================================================
# Homeostasis Controller
# ============================================================================

class HomeostasisController:
    """
    Maintains internal stability through feedback control.

    Monitors vital signs and adjusts internal parameters to
    maintain optimal operating conditions.

    Example:
        homeostasis = HomeostasisController()

        # Add vital signs to monitor
        homeostasis.add_vital_sign(VitalSignType.ENERGY_LEVEL, optimal=0.7)

        # Update readings
        homeostasis.update_vital_sign(VitalSignType.ENERGY_LEVEL, 0.4)

        # Get regulation responses
        responses = homeostasis.regulate()
    """

    def __init__(
        self,
        regulation_type: RegulationType = RegulationType.PID,
        stress_sensitivity: float = 1.0
    ):
        self.regulation_type = regulation_type
        self.stress_sensitivity = stress_sensitivity

        # Vital signs
        self.vital_signs: Dict[VitalSignType, VitalSign] = {}

        # Setpoints
        self.setpoints: Dict[VitalSignType, Setpoint] = {}

        # Controllers (for PID regulation)
        self.controllers: Dict[VitalSignType, PIDController] = {}

        # State
        self.stress_level = StressLevel.NONE
        self.overall_health: float = 1.0
        self.cycle_count: int = 0

        # History
        self.response_history: List[RegulationResponse] = []
        self.stress_history: List[StressResponse] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "stress_change": [],
            "regulation": [],
            "critical": []
        }

    def add_vital_sign(
        self,
        vital_type: VitalSignType,
        initial_value: float = 0.5,
        optimal: float = 0.5,
        min_safe: float = 0.1,
        max_safe: float = 0.9,
        weight: float = 1.0
    ):
        """Add a vital sign to monitor"""
        self.vital_signs[vital_type] = VitalSign(
            type=vital_type,
            current_value=initial_value,
            optimal=optimal,
            min_safe=min_safe,
            max_safe=max_safe,
            weight=weight
        )

        self.setpoints[vital_type] = Setpoint(
            vital_sign=vital_type,
            target=optimal,
            tolerance=0.1
        )

        if self.regulation_type == RegulationType.PID:
            self.controllers[vital_type] = PIDController()

    def update_vital_sign(self, vital_type: VitalSignType, value: float):
        """Update a vital sign reading"""
        if vital_type in self.vital_signs:
            self.vital_signs[vital_type].update(value)

    def regulate(self) -> List[RegulationResponse]:
        """
        Perform regulation cycle.

        Returns list of regulation responses.
        """
        self.cycle_count += 1
        responses = []

        for vital_type, vital in self.vital_signs.items():
            if vital_type not in self.setpoints:
                continue

            setpoint = self.setpoints[vital_type]

            # Check if regulation needed
            if abs(vital.current_value - setpoint.target) <= setpoint.tolerance:
                continue

            # Calculate regulation response
            if self.regulation_type == RegulationType.PID:
                controller = self.controllers.get(vital_type)
                if controller:
                    magnitude = controller.compute(setpoint.target, vital.current_value)
                else:
                    magnitude = setpoint.target - vital.current_value

            elif self.regulation_type == RegulationType.PROPORTIONAL:
                magnitude = (setpoint.target - vital.current_value) * 0.5

            elif self.regulation_type == RegulationType.ADAPTIVE:
                # Adjust based on trend
                trend = vital.trend
                base = setpoint.target - vital.current_value
                # If trending toward target, reduce correction
                if (trend > 0 and base > 0) or (trend < 0 and base < 0):
                    magnitude = base * 0.3
                else:
                    magnitude = base * 0.7

            else:  # PREDICTIVE
                # Predict future value and adjust
                trend = vital.trend
                predicted = vital.current_value + trend * 5
                magnitude = (setpoint.target - predicted) * 0.6

            # Determine action
            if magnitude > 0:
                action = f"increase_{vital_type.value}"
            else:
                action = f"decrease_{vital_type.value}"

            response = RegulationResponse(
                vital_sign=vital_type,
                action=action,
                magnitude=abs(magnitude)
            )
            responses.append(response)
            self.response_history.append(response)

            # Trigger callbacks
            for callback in self._callbacks["regulation"]:
                callback(response)

        # Update stress level
        self._update_stress_level()

        # Update overall health
        self._update_health()

        return responses

    def _update_stress_level(self):
        """Update stress level based on vital signs"""
        if not self.vital_signs:
            self.stress_level = StressLevel.NONE
            return

        # Calculate stress score
        stress_score = 0.0
        stressors = []

        for vital_type, vital in self.vital_signs.items():
            if not vital.in_safe_range:
                # Outside safe range is very stressful
                if vital.current_value < vital.min_safe:
                    stress = (vital.min_safe - vital.current_value) * 2
                else:
                    stress = (vital.current_value - vital.max_safe) * 2
                stress_score += stress * vital.weight
                stressors.append(f"{vital_type.value}_out_of_range")
            else:
                # Deviation from optimal causes mild stress
                stress_score += vital.deviation * vital.weight * 0.5

        stress_score *= self.stress_sensitivity

        # Determine stress level
        old_level = self.stress_level

        if stress_score <= 0.1:
            self.stress_level = StressLevel.NONE
        elif stress_score <= 0.3:
            self.stress_level = StressLevel.MILD
        elif stress_score <= 0.6:
            self.stress_level = StressLevel.MODERATE
        elif stress_score <= 0.9:
            self.stress_level = StressLevel.SEVERE
        else:
            self.stress_level = StressLevel.CRITICAL

        # Record stress response if level changed
        if self.stress_level != old_level:
            stress_response = StressResponse(
                level=self.stress_level,
                stressors=stressors,
                responses=self._get_stress_responses()
            )
            self.stress_history.append(stress_response)

            # Trigger callbacks
            for callback in self._callbacks["stress_change"]:
                callback(old_level, self.stress_level)

            if self.stress_level == StressLevel.CRITICAL:
                for callback in self._callbacks["critical"]:
                    callback(stress_response)

    def _get_stress_responses(self) -> List[str]:
        """Get appropriate stress responses"""
        responses = []

        if self.stress_level == StressLevel.MILD:
            responses = ["increase_monitoring", "optimize_resources"]
        elif self.stress_level == StressLevel.MODERATE:
            responses = ["reduce_non_essential", "request_resources", "alert_system"]
        elif self.stress_level == StressLevel.SEVERE:
            responses = ["emergency_conservation", "shed_load", "request_help"]
        elif self.stress_level == StressLevel.CRITICAL:
            responses = ["shutdown_non_critical", "emergency_protocols", "signal_distress"]

        return responses

    def _update_health(self):
        """Update overall health score"""
        if not self.vital_signs:
            self.overall_health = 1.0
            return

        health_scores = []
        for vital in self.vital_signs.values():
            if vital.in_safe_range:
                # Health based on closeness to optimal
                health = 1.0 - vital.deviation
            else:
                # Out of range is unhealthy
                health = max(0.0, 0.5 - vital.deviation)
            health_scores.append(health * vital.weight)

        total_weight = sum(v.weight for v in self.vital_signs.values())
        self.overall_health = sum(health_scores) / total_weight if total_weight > 0 else 0.5

    def adapt_setpoints(self, environment: Dict[str, float]):
        """
        Adapt setpoints based on environment.

        Organisms can adjust what's "optimal" based on conditions.
        """
        for vital_type, setpoint in self.setpoints.items():
            if not setpoint.adaptive:
                continue

            # Example adaptations
            if vital_type == VitalSignType.ENERGY_LEVEL:
                # Low resources = lower energy setpoint
                resource_scarcity = environment.get("resource_scarcity", 0)
                setpoint.target = 0.7 - resource_scarcity * 0.3

            elif vital_type == VitalSignType.PROCESSING_LOAD:
                # High demand = accept higher load
                demand = environment.get("demand", 0.5)
                setpoint.target = 0.5 + demand * 0.3

    def on_stress_change(self, callback: Callable[[StressLevel, StressLevel], None]):
        """Register callback for stress level changes"""
        self._callbacks["stress_change"].append(callback)

    def on_critical(self, callback: Callable[[StressResponse], None]):
        """Register callback for critical stress"""
        self._callbacks["critical"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get homeostasis status"""
        return {
            "overall_health": self.overall_health,
            "stress_level": self.stress_level.value,
            "vital_signs": {
                vt.value: {
                    "current": v.current_value,
                    "optimal": v.optimal,
                    "in_safe_range": v.in_safe_range,
                    "deviation": v.deviation,
                    "trend": v.trend
                }
                for vt, v in self.vital_signs.items()
            },
            "cycle_count": self.cycle_count,
            "regulation_responses": len(self.response_history),
            "stress_events": len(self.stress_history)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Homeostasis Controller Demo")
    print("=" * 50)

    # Create controller
    homeostasis = HomeostasisController(
        regulation_type=RegulationType.PID,
        stress_sensitivity=1.0
    )

    # Add vital signs
    homeostasis.add_vital_sign(
        VitalSignType.ENERGY_LEVEL,
        initial_value=0.7,
        optimal=0.7,
        min_safe=0.2,
        max_safe=0.95
    )
    homeostasis.add_vital_sign(
        VitalSignType.PROCESSING_LOAD,
        initial_value=0.5,
        optimal=0.5,
        min_safe=0.1,
        max_safe=0.9
    )
    homeostasis.add_vital_sign(
        VitalSignType.CONSCIOUSNESS_SCORE,
        initial_value=0.6,
        optimal=0.7,
        min_safe=0.3,
        max_safe=1.0
    )

    # Track stress changes
    stress_changes = []
    homeostasis.on_stress_change(lambda old, new: stress_changes.append((old, new)))

    print(f"\n1. Initial status:")
    status = homeostasis.get_status()
    print(f"   Health: {status['overall_health']:.3f}")
    print(f"   Stress: {status['stress_level']}")

    # Simulate normal operation
    print("\n2. Normal operation...")
    for cycle in range(20):
        # Simulate varying conditions
        energy = 0.7 + np.sin(cycle * 0.3) * 0.1
        load = 0.5 + np.cos(cycle * 0.2) * 0.15
        consciousness = 0.6 + np.random.uniform(-0.05, 0.05)

        homeostasis.update_vital_sign(VitalSignType.ENERGY_LEVEL, energy)
        homeostasis.update_vital_sign(VitalSignType.PROCESSING_LOAD, load)
        homeostasis.update_vital_sign(VitalSignType.CONSCIOUSNESS_SCORE, consciousness)

        responses = homeostasis.regulate()

        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: health={homeostasis.overall_health:.3f}, "
                  f"stress={homeostasis.stress_level.value}, "
                  f"responses={len(responses)}")

    # Simulate stress condition
    print("\n3. Simulating stress condition...")
    for cycle in range(15):
        # Declining energy
        energy = 0.5 - cycle * 0.03
        homeostasis.update_vital_sign(VitalSignType.ENERGY_LEVEL, energy)

        # High load
        homeostasis.update_vital_sign(VitalSignType.PROCESSING_LOAD, 0.85)

        responses = homeostasis.regulate()

        print(f"   Cycle {cycle}: energy={energy:.2f}, "
              f"health={homeostasis.overall_health:.3f}, "
              f"stress={homeostasis.stress_level.value}")

    # Recovery
    print("\n4. Recovery...")
    for cycle in range(10):
        # Recovering energy
        energy = 0.2 + cycle * 0.05
        homeostasis.update_vital_sign(VitalSignType.ENERGY_LEVEL, energy)
        homeostasis.update_vital_sign(VitalSignType.PROCESSING_LOAD, 0.5)

        homeostasis.regulate()

        if cycle % 3 == 0:
            print(f"   Cycle {cycle}: health={homeostasis.overall_health:.3f}, "
                  f"stress={homeostasis.stress_level.value}")

    # Final status
    print("\n5. Final status:")
    status = homeostasis.get_status()
    print(f"   Health: {status['overall_health']:.3f}")
    print(f"   Stress: {status['stress_level']}")
    print(f"   Stress events: {len(stress_changes)}")

    print("\n6. Vital signs:")
    for name, vs in status['vital_signs'].items():
        print(f"   {name}: {vs['current']:.3f} (optimal: {vs['optimal']:.1f}, safe: {vs['in_safe_range']})")
