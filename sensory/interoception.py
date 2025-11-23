"""
SYNAPSE-M: Interoception

Internal organ state awareness and visceral sensing.
Implements monitoring of internal body states including
energy, temperature, arousal, and homeostatic processes.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

from .modality_types import (
    ExtendedModality, ProcessedModality, ModalityInput,
    get_modality_characteristics
)


# ============================================================================
# Internal State Types
# ============================================================================

class OrganSystem(Enum):
    """Major organ systems for interoception"""
    CARDIOVASCULAR = auto()    # Heart, blood vessels
    RESPIRATORY = auto()       # Lungs, breathing
    DIGESTIVE = auto()         # Stomach, intestines
    METABOLIC = auto()         # Energy, metabolism
    THERMOREGULATORY = auto()  # Temperature control
    IMMUNE = auto()            # Immune response
    ENDOCRINE = auto()         # Hormones
    NERVOUS = auto()           # Autonomic nervous system


class HomeostasisState(Enum):
    """Homeostatic state levels"""
    OPTIMAL = auto()           # Within optimal range
    ELEVATED = auto()          # Above setpoint
    DEPRESSED = auto()         # Below setpoint
    CRITICAL_HIGH = auto()     # Dangerously high
    CRITICAL_LOW = auto()      # Dangerously low


class ArousalLevel(Enum):
    """Arousal/activation levels"""
    DORMANT = 0
    RELAXED = 1
    ALERT = 2
    AROUSED = 3
    STRESSED = 4
    PANIC = 5


# ============================================================================
# Internal State Data Structures
# ============================================================================

@dataclass
class OrganSystemState:
    """State of an organ system"""
    system: OrganSystem
    activity_level: float = 0.5           # 0-1 activity
    health: float = 1.0                   # 0-1 health status
    homeostasis: HomeostasisState = HomeostasisState.OPTIMAL
    setpoint: float = 0.5                 # Target value
    current_value: float = 0.5            # Current value
    deviation: float = 0.0                # Deviation from setpoint
    trend: float = 0.0                    # Rate of change
    last_update: float = field(default_factory=time.time)

    def update(self, new_value: float, dt: float):
        """Update state with new value"""
        self.trend = (new_value - self.current_value) / dt if dt > 0 else 0.0
        self.current_value = new_value
        self.deviation = self.current_value - self.setpoint
        self._update_homeostasis()
        self.last_update = time.time()

    def _update_homeostasis(self):
        """Update homeostasis classification"""
        if abs(self.deviation) < 0.1:
            self.homeostasis = HomeostasisState.OPTIMAL
        elif self.deviation > 0.3:
            self.homeostasis = HomeostasisState.CRITICAL_HIGH
        elif self.deviation < -0.3:
            self.homeostasis = HomeostasisState.CRITICAL_LOW
        elif self.deviation > 0:
            self.homeostasis = HomeostasisState.ELEVATED
        else:
            self.homeostasis = HomeostasisState.DEPRESSED


@dataclass
class VitalSign:
    """A vital sign measurement"""
    name: str
    value: float
    unit: str
    normal_range: Tuple[float, float]
    timestamp: float = field(default_factory=time.time)

    def is_normal(self) -> bool:
        """Check if within normal range"""
        return self.normal_range[0] <= self.value <= self.normal_range[1]

    def get_deviation(self) -> float:
        """Get deviation from normal (0 = center of range)"""
        mid = (self.normal_range[0] + self.normal_range[1]) / 2
        span = self.normal_range[1] - self.normal_range[0]
        return (self.value - mid) / (span / 2) if span > 0 else 0.0


@dataclass
class InternalStateSnapshot:
    """
    Complete snapshot of internal state.
    """
    organ_states: Dict[OrganSystem, OrganSystemState] = field(default_factory=dict)
    vital_signs: Dict[str, VitalSign] = field(default_factory=dict)

    # Global states
    energy_level: float = 1.0             # Overall energy (0-1)
    core_temperature: float = 37.0        # Core temperature (Celsius equivalent)
    arousal: ArousalLevel = ArousalLevel.ALERT
    stress_level: float = 0.0             # Stress indicator (0-1)
    comfort_level: float = 1.0            # Overall comfort (0-1)

    # Needs
    hunger: float = 0.0                   # Hunger level (0-1)
    thirst: float = 0.0                   # Thirst level (0-1)
    fatigue: float = 0.0                  # Fatigue level (0-1)
    pain: float = 0.0                     # Pain level (0-1)

    timestamp: float = field(default_factory=time.time)

    def get_overall_wellbeing(self) -> float:
        """Calculate overall wellbeing score"""
        # Combine factors
        need_penalty = (self.hunger + self.thirst + self.fatigue + self.pain) / 4
        health_score = np.mean([s.health for s in self.organ_states.values()]) if self.organ_states else 1.0
        homeostasis_score = self._get_homeostasis_score()

        wellbeing = (
            0.3 * (1 - need_penalty) +
            0.3 * health_score +
            0.2 * homeostasis_score +
            0.2 * self.comfort_level
        )
        return float(np.clip(wellbeing, 0.0, 1.0))

    def _get_homeostasis_score(self) -> float:
        """Get overall homeostasis score"""
        if not self.organ_states:
            return 1.0

        scores = []
        for state in self.organ_states.values():
            if state.homeostasis == HomeostasisState.OPTIMAL:
                scores.append(1.0)
            elif state.homeostasis in [HomeostasisState.ELEVATED, HomeostasisState.DEPRESSED]:
                scores.append(0.7)
            else:
                scores.append(0.3)

        return float(np.mean(scores))


@dataclass
class InteroceptivePerception:
    """
    Interoceptive perception output.
    """
    state_snapshot: InternalStateSnapshot
    signals: List[str]                    # Active interoceptive signals
    features: np.ndarray                  # Encoded features
    salience: float = 0.5                 # How attention-grabbing
    urgency: float = 0.0                  # Urgency of internal signals
    valence: float = 0.0                  # Pleasant/unpleasant (-1 to 1)
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Interoceptive Sensors
# ============================================================================

class InteroceptiveSensor:
    """Base class for interoceptive sensors"""

    def __init__(self, name: str, organ_system: OrganSystem):
        self.name = name
        self.organ_system = organ_system
        self.sensitivity = 1.0
        self.threshold = 0.1

    def sense(self, raw_value: float) -> float:
        """Process raw value through sensor"""
        if abs(raw_value) < self.threshold:
            return 0.0
        return raw_value * self.sensitivity


class BaroreceptorSensor(InteroceptiveSensor):
    """
    Baroreceptor - senses blood pressure changes.
    """

    def __init__(self):
        super().__init__("baroreceptor", OrganSystem.CARDIOVASCULAR)
        self.baseline_pressure = 100.0    # mmHg equivalent

    def sense(self, pressure: float) -> float:
        """Sense blood pressure deviation"""
        deviation = (pressure - self.baseline_pressure) / self.baseline_pressure
        return deviation * self.sensitivity


class ChemoreceptorSensor(InteroceptiveSensor):
    """
    Chemoreceptor - senses blood chemistry (O2, CO2, pH).
    """

    def __init__(self, target: str = "oxygen"):
        super().__init__(f"chemoreceptor_{target}", OrganSystem.RESPIRATORY)
        self.target = target
        self.optimal_value = 0.95 if target == "oxygen" else 0.04  # Normal ranges

    def sense(self, value: float) -> float:
        """Sense chemical deviation from optimal"""
        deviation = abs(value - self.optimal_value)
        return deviation * self.sensitivity


class ThermoreceptorSensor(InteroceptiveSensor):
    """
    Thermoreceptor - senses temperature.
    """

    def __init__(self, warm: bool = True):
        super().__init__(f"thermoreceptor_{'warm' if warm else 'cold'}", OrganSystem.THERMOREGULATORY)
        self.warm = warm
        self.setpoint = 37.0

    def sense(self, temperature: float) -> float:
        """Sense temperature deviation"""
        deviation = temperature - self.setpoint
        if (self.warm and deviation > 0) or (not self.warm and deviation < 0):
            return abs(deviation) * self.sensitivity
        return 0.0


class NociceptorSensor(InteroceptiveSensor):
    """
    Visceral nociceptor - senses internal pain.
    """

    def __init__(self, location: str = "general"):
        super().__init__(f"nociceptor_{location}", OrganSystem.NERVOUS)
        self.location = location
        self.threshold = 0.2
        self.sensitization = 1.0          # Can increase with repeated activation

    def sense(self, damage_signal: float) -> float:
        """Sense pain/damage"""
        if damage_signal < self.threshold:
            return 0.0
        pain = (damage_signal - self.threshold) * self.sensitivity * self.sensitization
        # Sensitization increases with activation
        self.sensitization = min(2.0, self.sensitization + 0.01)
        return pain


# ============================================================================
# Interoception System
# ============================================================================

class InteroceptionSystem:
    """
    Complete interoception system for internal state awareness.

    Provides:
    - Organ system monitoring
    - Vital signs tracking
    - Homeostasis detection
    - Need/drive signals
    - Internal state encoding
    """

    def __init__(self, feature_dim: int = 48):
        self.feature_dim = feature_dim

        # Initialize organ system states
        self.organ_states: Dict[OrganSystem, OrganSystemState] = {}
        for system in OrganSystem:
            self.organ_states[system] = OrganSystemState(system=system)

        # Sensors
        self.sensors: List[InteroceptiveSensor] = [
            BaroreceptorSensor(),
            ChemoreceptorSensor("oxygen"),
            ChemoreceptorSensor("carbon_dioxide"),
            ThermoreceptorSensor(warm=True),
            ThermoreceptorSensor(warm=False),
            NociceptorSensor("visceral"),
        ]

        # Vital signs
        self.vital_signs: Dict[str, VitalSign] = {
            "heart_rate": VitalSign("heart_rate", 70.0, "bpm", (60.0, 100.0)),
            "blood_pressure": VitalSign("blood_pressure", 120.0, "mmHg", (90.0, 140.0)),
            "respiratory_rate": VitalSign("respiratory_rate", 16.0, "bpm", (12.0, 20.0)),
            "temperature": VitalSign("temperature", 37.0, "C", (36.5, 37.5)),
            "blood_oxygen": VitalSign("blood_oxygen", 98.0, "%", (95.0, 100.0)),
        }

        # Current state
        self.current_state = InternalStateSnapshot(
            organ_states=self.organ_states.copy(),
            vital_signs=self.vital_signs.copy()
        )

        # History
        self._state_history: deque = deque(maxlen=100)

        # Feature encoder
        np.random.seed(52)
        self._encoder = np.random.randn(50, feature_dim) * 0.1

    def update(
        self,
        inputs: Dict[str, Any]
    ) -> InteroceptivePerception:
        """
        Update interoception with new internal state inputs.

        Args:
            inputs: Dict with keys like:
                - energy: float
                - temperature: float
                - heart_rate: float
                - hunger: float
                - fatigue: float
                - damage: float (triggers pain)
                - stress: float

        Returns:
            InteroceptivePerception
        """
        signals = []
        dt = 0.01  # Assumed time step

        # Update energy/metabolism
        if "energy" in inputs:
            energy = inputs["energy"]
            self.current_state.energy_level = np.clip(energy, 0.0, 1.0)
            self.organ_states[OrganSystem.METABOLIC].update(energy, dt)

            if energy < 0.3:
                signals.append("low_energy")

        # Update temperature
        if "temperature" in inputs:
            temp = inputs["temperature"]
            self.current_state.core_temperature = temp
            self.organ_states[OrganSystem.THERMOREGULATORY].update(temp / 40.0, dt)
            self.vital_signs["temperature"] = VitalSign(
                "temperature", temp, "C", (36.5, 37.5)
            )

            # Sense with thermoreceptors
            for sensor in self.sensors:
                if isinstance(sensor, ThermoreceptorSensor):
                    deviation = sensor.sense(temp)
                    if deviation > 0.1:
                        signals.append(f"{'warm' if sensor.warm else 'cold'}_signal")

        # Update cardiovascular
        if "heart_rate" in inputs:
            hr = inputs["heart_rate"]
            self.vital_signs["heart_rate"] = VitalSign(
                "heart_rate", hr, "bpm", (60.0, 100.0)
            )
            self.organ_states[OrganSystem.CARDIOVASCULAR].update(hr / 100.0, dt)

            if hr > 100:
                signals.append("elevated_heart_rate")
            elif hr < 60:
                signals.append("low_heart_rate")

        # Update needs
        if "hunger" in inputs:
            self.current_state.hunger = np.clip(inputs["hunger"], 0.0, 1.0)
            if self.current_state.hunger > 0.6:
                signals.append("hunger_signal")

        if "thirst" in inputs:
            self.current_state.thirst = np.clip(inputs["thirst"], 0.0, 1.0)
            if self.current_state.thirst > 0.6:
                signals.append("thirst_signal")

        if "fatigue" in inputs:
            self.current_state.fatigue = np.clip(inputs["fatigue"], 0.0, 1.0)
            if self.current_state.fatigue > 0.7:
                signals.append("fatigue_signal")

        # Update pain
        if "damage" in inputs:
            damage = inputs["damage"]
            for sensor in self.sensors:
                if isinstance(sensor, NociceptorSensor):
                    pain = sensor.sense(damage)
                    if pain > 0:
                        self.current_state.pain = np.clip(
                            self.current_state.pain + pain, 0.0, 1.0
                        )
                        signals.append("pain_signal")

        # Update stress
        if "stress" in inputs:
            stress = inputs["stress"]
            self.current_state.stress_level = np.clip(stress, 0.0, 1.0)
            self._update_arousal()

            if stress > 0.7:
                signals.append("high_stress")

        # Update overall comfort
        self._update_comfort()

        # Store state in history
        self.current_state.timestamp = time.time()
        self._state_history.append(self.current_state)

        # Encode features
        features = self._encode_features()

        # Calculate perception properties
        salience = self._compute_salience(signals)
        urgency = self._compute_urgency()
        valence = self._compute_valence()

        return InteroceptivePerception(
            state_snapshot=self.current_state,
            signals=signals,
            features=features,
            salience=salience,
            urgency=urgency,
            valence=valence
        )

    def _update_arousal(self):
        """Update arousal level based on internal state"""
        stress = self.current_state.stress_level
        energy = self.current_state.energy_level

        if stress > 0.8:
            self.current_state.arousal = ArousalLevel.PANIC
        elif stress > 0.6:
            self.current_state.arousal = ArousalLevel.STRESSED
        elif stress > 0.4 or energy > 0.7:
            self.current_state.arousal = ArousalLevel.AROUSED
        elif energy > 0.5:
            self.current_state.arousal = ArousalLevel.ALERT
        elif energy > 0.3:
            self.current_state.arousal = ArousalLevel.RELAXED
        else:
            self.current_state.arousal = ArousalLevel.DORMANT

    def _update_comfort(self):
        """Update overall comfort level"""
        discomfort = (
            self.current_state.hunger * 0.2 +
            self.current_state.thirst * 0.2 +
            self.current_state.fatigue * 0.2 +
            self.current_state.pain * 0.4 +
            self.current_state.stress_level * 0.3
        )
        self.current_state.comfort_level = np.clip(1.0 - discomfort, 0.0, 1.0)

    def _compute_salience(self, signals: List[str]) -> float:
        """Compute how attention-grabbing internal state is"""
        base_salience = 0.3

        # Pain is highly salient
        if self.current_state.pain > 0:
            base_salience += self.current_state.pain * 0.5

        # Strong needs are salient
        max_need = max(
            self.current_state.hunger,
            self.current_state.thirst,
            self.current_state.fatigue
        )
        base_salience += max_need * 0.3

        # Number of signals adds salience
        base_salience += len(signals) * 0.05

        return float(np.clip(base_salience, 0.0, 1.0))

    def _compute_urgency(self) -> float:
        """Compute urgency of internal signals"""
        urgency = 0.0

        # Check for critical states
        for state in self.organ_states.values():
            if state.homeostasis in [HomeostasisState.CRITICAL_HIGH, HomeostasisState.CRITICAL_LOW]:
                urgency += 0.3

        # Pain adds urgency
        urgency += self.current_state.pain * 0.4

        # Critical needs
        if self.current_state.hunger > 0.9:
            urgency += 0.2
        if self.current_state.thirst > 0.9:
            urgency += 0.3  # Thirst is more urgent

        return float(np.clip(urgency, 0.0, 1.0))

    def _compute_valence(self) -> float:
        """Compute pleasantness/unpleasantness (-1 to 1)"""
        # Start neutral
        valence = 0.0

        # Negative contributions
        valence -= self.current_state.pain * 0.5
        valence -= self.current_state.stress_level * 0.3
        valence -= max(self.current_state.hunger, self.current_state.thirst) * 0.2

        # Positive contributions
        valence += self.current_state.comfort_level * 0.3
        if self.current_state.energy_level > 0.7:
            valence += 0.2

        return float(np.clip(valence, -1.0, 1.0))

    def _encode_features(self) -> np.ndarray:
        """Encode internal state into feature vector"""
        raw_features = []

        # Organ system states
        for system in OrganSystem:
            state = self.organ_states.get(system)
            if state:
                raw_features.extend([
                    state.current_value,
                    state.deviation,
                    state.trend,
                    state.health
                ])

        # Global states
        raw_features.extend([
            self.current_state.energy_level,
            self.current_state.core_temperature / 40.0,
            self.current_state.arousal.value / 5.0,
            self.current_state.stress_level,
            self.current_state.comfort_level,
            self.current_state.hunger,
            self.current_state.thirst,
            self.current_state.fatigue,
            self.current_state.pain
        ])

        # Vital signs
        for vs in self.vital_signs.values():
            raw_features.append(vs.get_deviation())

        # Pad/truncate to 50
        raw_array = np.array(raw_features[:50])
        if len(raw_array) < 50:
            raw_array = np.pad(raw_array, (0, 50 - len(raw_array)))

        # Encode
        features = np.tanh(raw_array @ self._encoder)
        return features

    def to_processed_modality(
        self,
        perception: InteroceptivePerception
    ) -> ProcessedModality:
        """Convert to standard ProcessedModality format"""
        input_data = ModalityInput(
            modality=ExtendedModality.INTEROCEPTION,
            raw_data={
                "wellbeing": self.current_state.get_overall_wellbeing(),
                "arousal": self.current_state.arousal.name,
                "signals": perception.signals
            },
            intensity=perception.salience
        )

        chars = get_modality_characteristics(ExtendedModality.INTEROCEPTION)

        return ProcessedModality(
            modality=ExtendedModality.INTEROCEPTION,
            raw_input=input_data,
            features=perception.features,
            semantic_embedding=perception.features[:chars.semantic_dim],
            salience=perception.salience,
            confidence=0.9,
            metadata={
                "urgency": perception.urgency,
                "valence": perception.valence
            }
        )

    def get_wellbeing(self) -> float:
        """Get current overall wellbeing"""
        return self.current_state.get_overall_wellbeing()

    def get_statistics(self) -> Dict[str, Any]:
        """Get interoception statistics"""
        return {
            "organ_systems": len(self.organ_states),
            "vital_signs": len(self.vital_signs),
            "sensors": len(self.sensors),
            "energy_level": self.current_state.energy_level,
            "arousal": self.current_state.arousal.name,
            "wellbeing": self.current_state.get_overall_wellbeing(),
            "comfort": self.current_state.comfort_level,
            "needs": {
                "hunger": self.current_state.hunger,
                "thirst": self.current_state.thirst,
                "fatigue": self.current_state.fatigue
            },
            "pain": self.current_state.pain,
            "stress": self.current_state.stress_level
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Interoception Demo")
    print("=" * 50)

    # Create interoception system
    intero = InteroceptionSystem()

    np.random.seed(42)

    print("\n1. Initial state:")
    stats = intero.get_statistics()
    print(f"   Wellbeing: {stats['wellbeing']:.2f}")
    print(f"   Arousal: {stats['arousal']}")
    print(f"   Energy: {stats['energy_level']:.2f}")

    print("\n2. Simulating state changes...")

    # Scenario 1: Getting hungry
    perception = intero.update({"hunger": 0.4, "energy": 0.8})
    print(f"\n   After mild hunger:")
    print(f"   Signals: {perception.signals}")
    print(f"   Salience: {perception.salience:.2f}")
    print(f"   Valence: {perception.valence:.2f}")

    # Scenario 2: Stress
    perception = intero.update({"stress": 0.6, "heart_rate": 95})
    print(f"\n   After stress:")
    print(f"   Signals: {perception.signals}")
    print(f"   Arousal: {intero.current_state.arousal.name}")
    print(f"   Urgency: {perception.urgency:.2f}")

    # Scenario 3: Pain
    perception = intero.update({"damage": 0.5})
    print(f"\n   After damage:")
    print(f"   Signals: {perception.signals}")
    print(f"   Pain level: {intero.current_state.pain:.2f}")
    print(f"   Valence: {perception.valence:.2f}")

    print("\n3. Final state:")
    stats = intero.get_statistics()
    print(f"   Wellbeing: {stats['wellbeing']:.2f}")
    print(f"   Comfort: {stats['comfort']:.2f}")
    print(f"   Needs: {stats['needs']}")

    print("\n4. Feature encoding:")
    perception = intero.update({})
    print(f"   Feature shape: {perception.features.shape}")
    print(f"   Feature range: [{perception.features.min():.3f}, {perception.features.max():.3f}]")

    print("\n5. Converting to ProcessedModality:")
    processed = intero.to_processed_modality(perception)
    print(f"   Modality: {processed.modality.value}")
    print(f"   Salience: {processed.salience:.2f}")
    print(f"   Metadata: {processed.metadata}")
