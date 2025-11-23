"""
Receptor Binding and Sensitivity System

Implements hormone receptor dynamics for digital organisms including:
- Receptor types and binding affinities
- Sensitivity modulation (up/down regulation)
- Desensitization and recovery
- Receptor-mediated signal transduction
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

class ReceptorType(Enum):
    """Types of hormone receptors"""
    # Fast-acting membrane receptors
    ALPHA_ADRENERGIC = "alpha_adrenergic"     # Adrenaline (constriction)
    BETA_ADRENERGIC = "beta_adrenergic"       # Adrenaline (relaxation)
    DOPAMINE_D1 = "dopamine_d1"               # Reward, motivation (excitatory)
    DOPAMINE_D2 = "dopamine_d2"               # Reward modulation (inhibitory)
    SEROTONIN_5HT1 = "serotonin_5ht1"         # Mood regulation
    SEROTONIN_5HT2 = "serotonin_5ht2"         # Mood, cognition

    # Social/bonding receptors
    OXYTOCIN_R = "oxytocin_r"                 # Social bonding
    VASOPRESSIN_V1 = "vasopressin_v1"         # Social recognition

    # Metabolic receptors
    INSULIN_R = "insulin_r"                   # Glucose regulation
    GLUCAGON_R = "glucagon_r"                 # Energy mobilization
    GLUCOCORTICOID = "glucocorticoid"         # Cortisol (stress)
    MINERALOCORTICOID = "mineralocorticoid"   # Cortisol (homeostasis)

    # Growth/recovery receptors
    GROWTH_HORMONE_R = "growth_hormone_r"     # Growth signals
    MELATONIN_MT1 = "melatonin_mt1"           # Sleep/rest
    MELATONIN_MT2 = "melatonin_mt2"           # Circadian rhythm

    # Opioid receptors
    MU_OPIOID = "mu_opioid"                   # Endorphin (pleasure)
    DELTA_OPIOID = "delta_opioid"             # Endorphin (mood)

    # Thyroid receptor
    THYROID_R = "thyroid_r"                   # Metabolic rate


class ReceptorState(Enum):
    """Receptor activation states"""
    INACTIVE = "inactive"
    BOUND = "bound"
    ACTIVATED = "activated"
    DESENSITIZED = "desensitized"
    INTERNALIZED = "internalized"


class SignalType(Enum):
    """Types of downstream signals"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    METABOLIC = "metabolic"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ReceptorBinding:
    """Record of a hormone-receptor binding event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    receptor_type: ReceptorType = ReceptorType.DOPAMINE_D1
    hormone_type: HormoneType = HormoneType.DOPAMINE
    binding_affinity: float = 0.5          # 0-1 binding strength
    occupancy: float = 0.0                 # Fraction of receptors bound
    signal_strength: float = 0.0           # Downstream signal intensity
    timestamp: float = field(default_factory=time.time)


@dataclass
class Receptor:
    """A hormone receptor with dynamic properties"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ReceptorType = ReceptorType.DOPAMINE_D1

    # Binding properties
    target_hormones: List[HormoneType] = field(default_factory=list)
    base_affinity: float = 0.5             # Baseline binding affinity
    current_affinity: float = 0.5          # Current affinity (modulated)

    # State
    state: ReceptorState = ReceptorState.INACTIVE
    occupancy: float = 0.0                 # Fraction bound (0-1)

    # Sensitivity dynamics
    sensitivity: float = 1.0               # Current sensitivity multiplier
    max_sensitivity: float = 2.0           # Maximum upregulation
    min_sensitivity: float = 0.1           # Maximum downregulation

    # Desensitization parameters
    desensitization_rate: float = 0.02     # Rate of sensitivity loss
    recovery_rate: float = 0.01            # Rate of sensitivity recovery
    desensitization_threshold: float = 0.7  # Occupancy threshold

    # Signal properties
    signal_type: SignalType = SignalType.EXCITATORY
    signal_gain: float = 1.0               # Signal amplification factor
    signal_decay: float = 0.1              # How fast signal fades

    # Expression (receptor count)
    expression_level: float = 1.0          # Receptor density (0-2)
    internalization_rate: float = 0.01     # Rate of receptor removal
    synthesis_rate: float = 0.01           # Rate of new receptor synthesis

    # History
    last_update: float = field(default_factory=time.time)


@dataclass
class SignalCascade:
    """Downstream signaling cascade from receptor activation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_receptor: ReceptorType = ReceptorType.DOPAMINE_D1
    signal_type: SignalType = SignalType.EXCITATORY
    intensity: float = 0.0
    duration: float = 0.0                  # How long signal lasts
    targets: List[str] = field(default_factory=list)  # Target systems
    active: bool = True
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Receptor System
# ============================================================================

class ReceptorSystem:
    """
    Complete receptor binding and sensitivity system.

    Features:
    - Multiple receptor types for each hormone
    - Dynamic sensitivity (up/down regulation)
    - Desensitization under chronic stimulation
    - Signal transduction cascades
    - Receptor expression dynamics

    Example:
        receptor_system = ReceptorSystem()

        # Process hormone binding
        signals = receptor_system.process_hormone(HormoneType.DOPAMINE, 0.7)

        # Update dynamics
        receptor_system.tick()

        # Check sensitivity
        sensitivity = receptor_system.get_sensitivity(ReceptorType.DOPAMINE_D1)
    """

    def __init__(
        self,
        base_sensitivity: float = 1.0,
        enable_plasticity: bool = True
    ):
        self.base_sensitivity = base_sensitivity
        self.enable_plasticity = enable_plasticity

        # Receptors
        self.receptors: Dict[ReceptorType, Receptor] = self._initialize_receptors()

        # Hormone-receptor mappings
        self.hormone_receptor_map: Dict[HormoneType, List[ReceptorType]] = \
            self._initialize_mappings()

        # Active signals
        self.active_signals: List[SignalCascade] = []

        # Binding history
        self.binding_history: List[ReceptorBinding] = []
        self.max_history = 500

        # Cycle tracking
        self.cycle_count = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "receptor_activated": [],
            "receptor_desensitized": [],
            "sensitivity_changed": [],
            "signal_generated": []
        }

    def _initialize_receptors(self) -> Dict[ReceptorType, Receptor]:
        """Initialize all receptor types"""
        receptors = {}

        configs = {
            # Adrenergic receptors
            ReceptorType.ALPHA_ADRENERGIC: {
                "target_hormones": [HormoneType.ADRENALINE],
                "base_affinity": 0.6,
                "signal_type": SignalType.EXCITATORY,
                "desensitization_rate": 0.03
            },
            ReceptorType.BETA_ADRENERGIC: {
                "target_hormones": [HormoneType.ADRENALINE],
                "base_affinity": 0.7,
                "signal_type": SignalType.EXCITATORY,
                "desensitization_rate": 0.04
            },

            # Dopamine receptors
            ReceptorType.DOPAMINE_D1: {
                "target_hormones": [HormoneType.DOPAMINE],
                "base_affinity": 0.65,
                "signal_type": SignalType.EXCITATORY,
                "desensitization_rate": 0.025
            },
            ReceptorType.DOPAMINE_D2: {
                "target_hormones": [HormoneType.DOPAMINE],
                "base_affinity": 0.55,
                "signal_type": SignalType.INHIBITORY,
                "desensitization_rate": 0.02
            },

            # Serotonin receptors
            ReceptorType.SEROTONIN_5HT1: {
                "target_hormones": [HormoneType.SEROTONIN],
                "base_affinity": 0.6,
                "signal_type": SignalType.INHIBITORY,
                "desensitization_rate": 0.015
            },
            ReceptorType.SEROTONIN_5HT2: {
                "target_hormones": [HormoneType.SEROTONIN],
                "base_affinity": 0.5,
                "signal_type": SignalType.EXCITATORY,
                "desensitization_rate": 0.02
            },

            # Social receptors
            ReceptorType.OXYTOCIN_R: {
                "target_hormones": [HormoneType.OXYTOCIN],
                "base_affinity": 0.7,
                "signal_type": SignalType.MODULATORY,
                "desensitization_rate": 0.01
            },
            ReceptorType.VASOPRESSIN_V1: {
                "target_hormones": [HormoneType.VASOPRESSIN],
                "base_affinity": 0.65,
                "signal_type": SignalType.MODULATORY,
                "desensitization_rate": 0.015
            },

            # Metabolic receptors
            ReceptorType.INSULIN_R: {
                "target_hormones": [HormoneType.INSULIN],
                "base_affinity": 0.8,
                "signal_type": SignalType.METABOLIC,
                "desensitization_rate": 0.02
            },
            ReceptorType.GLUCAGON_R: {
                "target_hormones": [HormoneType.GLUCAGON],
                "base_affinity": 0.75,
                "signal_type": SignalType.METABOLIC,
                "desensitization_rate": 0.02
            },
            ReceptorType.GLUCOCORTICOID: {
                "target_hormones": [HormoneType.CORTISOL],
                "base_affinity": 0.6,
                "signal_type": SignalType.METABOLIC,
                "desensitization_rate": 0.01
            },
            ReceptorType.MINERALOCORTICOID: {
                "target_hormones": [HormoneType.CORTISOL],
                "base_affinity": 0.9,
                "signal_type": SignalType.MODULATORY,
                "desensitization_rate": 0.005
            },

            # Growth receptors
            ReceptorType.GROWTH_HORMONE_R: {
                "target_hormones": [HormoneType.GROWTH_HORMONE],
                "base_affinity": 0.7,
                "signal_type": SignalType.METABOLIC,
                "desensitization_rate": 0.01
            },
            ReceptorType.MELATONIN_MT1: {
                "target_hormones": [HormoneType.MELATONIN],
                "base_affinity": 0.8,
                "signal_type": SignalType.INHIBITORY,
                "desensitization_rate": 0.01
            },
            ReceptorType.MELATONIN_MT2: {
                "target_hormones": [HormoneType.MELATONIN],
                "base_affinity": 0.75,
                "signal_type": SignalType.MODULATORY,
                "desensitization_rate": 0.01
            },

            # Opioid receptors
            ReceptorType.MU_OPIOID: {
                "target_hormones": [HormoneType.ENDORPHIN],
                "base_affinity": 0.7,
                "signal_type": SignalType.INHIBITORY,
                "desensitization_rate": 0.03
            },
            ReceptorType.DELTA_OPIOID: {
                "target_hormones": [HormoneType.ENDORPHIN],
                "base_affinity": 0.5,
                "signal_type": SignalType.MODULATORY,
                "desensitization_rate": 0.02
            },

            # Thyroid receptor
            ReceptorType.THYROID_R: {
                "target_hormones": [HormoneType.THYROXINE],
                "base_affinity": 0.85,
                "signal_type": SignalType.METABOLIC,
                "desensitization_rate": 0.005
            }
        }

        for r_type in ReceptorType:
            config = configs.get(r_type, {})
            receptors[r_type] = Receptor(
                type=r_type,
                target_hormones=config.get("target_hormones", []),
                base_affinity=config.get("base_affinity", 0.5),
                current_affinity=config.get("base_affinity", 0.5),
                signal_type=config.get("signal_type", SignalType.EXCITATORY),
                desensitization_rate=config.get("desensitization_rate", 0.02),
                sensitivity=self.base_sensitivity
            )

        return receptors

    def _initialize_mappings(self) -> Dict[HormoneType, List[ReceptorType]]:
        """Map hormones to their receptors"""
        return {
            HormoneType.ADRENALINE: [
                ReceptorType.ALPHA_ADRENERGIC,
                ReceptorType.BETA_ADRENERGIC
            ],
            HormoneType.DOPAMINE: [
                ReceptorType.DOPAMINE_D1,
                ReceptorType.DOPAMINE_D2
            ],
            HormoneType.SEROTONIN: [
                ReceptorType.SEROTONIN_5HT1,
                ReceptorType.SEROTONIN_5HT2
            ],
            HormoneType.OXYTOCIN: [ReceptorType.OXYTOCIN_R],
            HormoneType.VASOPRESSIN: [ReceptorType.VASOPRESSIN_V1],
            HormoneType.INSULIN: [ReceptorType.INSULIN_R],
            HormoneType.GLUCAGON: [ReceptorType.GLUCAGON_R],
            HormoneType.CORTISOL: [
                ReceptorType.GLUCOCORTICOID,
                ReceptorType.MINERALOCORTICOID
            ],
            HormoneType.GROWTH_HORMONE: [ReceptorType.GROWTH_HORMONE_R],
            HormoneType.MELATONIN: [
                ReceptorType.MELATONIN_MT1,
                ReceptorType.MELATONIN_MT2
            ],
            HormoneType.ENDORPHIN: [
                ReceptorType.MU_OPIOID,
                ReceptorType.DELTA_OPIOID
            ],
            HormoneType.THYROXINE: [ReceptorType.THYROID_R]
        }

    def process_hormone(
        self,
        hormone_type: HormoneType,
        concentration: float
    ) -> List[SignalCascade]:
        """
        Process hormone binding to receptors.

        Returns list of generated signal cascades.
        """
        signals = []

        receptor_types = self.hormone_receptor_map.get(hormone_type, [])

        for r_type in receptor_types:
            receptor = self.receptors[r_type]

            # Calculate binding
            binding = self._calculate_binding(receptor, concentration)

            if binding.occupancy > 0.1:
                # Update receptor state
                receptor.occupancy = binding.occupancy
                receptor.state = ReceptorState.BOUND

                # Generate signal if threshold met
                if binding.signal_strength > 0.2:
                    receptor.state = ReceptorState.ACTIVATED

                    signal = SignalCascade(
                        source_receptor=r_type,
                        signal_type=receptor.signal_type,
                        intensity=binding.signal_strength,
                        duration=10.0 / receptor.signal_decay,
                        targets=self._get_signal_targets(r_type)
                    )
                    signals.append(signal)
                    self.active_signals.append(signal)

                    # Trigger callbacks
                    for callback in self._callbacks["receptor_activated"]:
                        callback(r_type, binding.signal_strength)
                    for callback in self._callbacks["signal_generated"]:
                        callback(signal)

                # Record binding
                self.binding_history.append(binding)
                if len(self.binding_history) > self.max_history:
                    self.binding_history.pop(0)

        return signals

    def _calculate_binding(
        self,
        receptor: Receptor,
        concentration: float
    ) -> ReceptorBinding:
        """Calculate hormone-receptor binding using Hill equation"""
        # Effective affinity = base * expression * sensitivity
        effective_affinity = (
            receptor.current_affinity *
            receptor.expression_level *
            receptor.sensitivity
        )

        # Hill equation for binding (simplified)
        # Occupancy = [H] / ([H] + Kd)
        kd = 1.0 - effective_affinity  # Dissociation constant
        kd = max(0.01, kd)  # Prevent division issues

        occupancy = concentration / (concentration + kd)
        occupancy = min(1.0, occupancy)

        # Signal strength depends on occupancy and receptor gain
        signal_strength = occupancy * receptor.signal_gain * receptor.sensitivity

        return ReceptorBinding(
            receptor_type=receptor.type,
            hormone_type=receptor.target_hormones[0] if receptor.target_hormones else HormoneType.DOPAMINE,
            binding_affinity=effective_affinity,
            occupancy=occupancy,
            signal_strength=signal_strength
        )

    def _get_signal_targets(self, receptor_type: ReceptorType) -> List[str]:
        """Get downstream targets for a receptor"""
        target_map = {
            ReceptorType.ALPHA_ADRENERGIC: ["motor", "metabolism"],
            ReceptorType.BETA_ADRENERGIC: ["motor", "metabolism", "sensory"],
            ReceptorType.DOPAMINE_D1: ["motor", "reward", "cognition"],
            ReceptorType.DOPAMINE_D2: ["motor", "reward"],
            ReceptorType.SEROTONIN_5HT1: ["mood", "homeostasis"],
            ReceptorType.SEROTONIN_5HT2: ["mood", "cognition", "sensory"],
            ReceptorType.OXYTOCIN_R: ["social", "reward", "stress"],
            ReceptorType.VASOPRESSIN_V1: ["social", "memory"],
            ReceptorType.INSULIN_R: ["metabolism", "growth"],
            ReceptorType.GLUCAGON_R: ["metabolism", "energy"],
            ReceptorType.GLUCOCORTICOID: ["metabolism", "immune", "stress"],
            ReceptorType.MINERALOCORTICOID: ["homeostasis", "memory"],
            ReceptorType.GROWTH_HORMONE_R: ["growth", "metabolism", "repair"],
            ReceptorType.MELATONIN_MT1: ["sleep", "circadian"],
            ReceptorType.MELATONIN_MT2: ["circadian", "mood"],
            ReceptorType.MU_OPIOID: ["pain", "reward", "mood"],
            ReceptorType.DELTA_OPIOID: ["mood", "stress"],
            ReceptorType.THYROID_R: ["metabolism", "growth", "cognition"]
        }
        return target_map.get(receptor_type, ["general"])

    def tick(self):
        """Process one receptor cycle"""
        self.cycle_count += 1
        current_time = time.time()

        for receptor in self.receptors.values():
            # Update based on state
            if receptor.state == ReceptorState.ACTIVATED:
                # Check for desensitization
                if receptor.occupancy > receptor.desensitization_threshold:
                    if self.enable_plasticity:
                        self._apply_desensitization(receptor)

            elif receptor.state == ReceptorState.DESENSITIZED:
                # Slow recovery
                if self.enable_plasticity:
                    self._apply_recovery(receptor)

            elif receptor.state in [ReceptorState.INACTIVE, ReceptorState.BOUND]:
                # Natural recovery toward baseline sensitivity
                if self.enable_plasticity:
                    self._natural_recovery(receptor)

            # Decay occupancy
            receptor.occupancy *= (1.0 - 0.1)
            if receptor.occupancy < 0.05:
                receptor.occupancy = 0.0
                if receptor.state == ReceptorState.BOUND:
                    receptor.state = ReceptorState.INACTIVE
                elif receptor.state == ReceptorState.ACTIVATED:
                    receptor.state = ReceptorState.INACTIVE

            # Update expression levels
            self._update_expression(receptor)

            receptor.last_update = current_time

        # Decay active signals
        self._decay_signals()

    def _apply_desensitization(self, receptor: Receptor):
        """Apply desensitization to chronically activated receptor"""
        receptor.sensitivity -= receptor.desensitization_rate
        receptor.sensitivity = max(receptor.min_sensitivity, receptor.sensitivity)

        if receptor.sensitivity < 0.5:
            receptor.state = ReceptorState.DESENSITIZED

            # Trigger callback
            for callback in self._callbacks["receptor_desensitized"]:
                callback(receptor.type, receptor.sensitivity)

    def _apply_recovery(self, receptor: Receptor):
        """Apply sensitivity recovery to desensitized receptor"""
        receptor.sensitivity += receptor.recovery_rate

        if receptor.sensitivity > 0.8:
            receptor.state = ReceptorState.INACTIVE

        receptor.sensitivity = min(receptor.max_sensitivity, receptor.sensitivity)

    def _natural_recovery(self, receptor: Receptor):
        """Natural drift toward baseline sensitivity"""
        diff = self.base_sensitivity - receptor.sensitivity
        receptor.sensitivity += diff * 0.01

    def _update_expression(self, receptor: Receptor):
        """Update receptor expression levels (up/down regulation)"""
        if receptor.state == ReceptorState.DESENSITIZED:
            # Internalization - reduce receptor count
            receptor.expression_level -= receptor.internalization_rate
            receptor.expression_level = max(0.1, receptor.expression_level)
        else:
            # Synthesis - restore receptor count
            if receptor.expression_level < 1.0:
                receptor.expression_level += receptor.synthesis_rate
                receptor.expression_level = min(2.0, receptor.expression_level)

    def _decay_signals(self):
        """Decay and remove old signals"""
        active = []
        current_time = time.time()

        for signal in self.active_signals:
            age = current_time - signal.timestamp
            if age < signal.duration and signal.active:
                # Decay intensity
                decay_factor = 1.0 - (age / signal.duration)
                signal.intensity *= decay_factor

                if signal.intensity > 0.01:
                    active.append(signal)

        self.active_signals = active

    def get_sensitivity(self, receptor_type: ReceptorType) -> float:
        """Get current sensitivity of a receptor"""
        if receptor_type in self.receptors:
            return self.receptors[receptor_type].sensitivity
        return 1.0

    def get_receptor_state(self, receptor_type: ReceptorType) -> Dict[str, Any]:
        """Get detailed state of a receptor"""
        if receptor_type not in self.receptors:
            return {}

        receptor = self.receptors[receptor_type]
        return {
            "type": receptor.type.value,
            "state": receptor.state.value,
            "sensitivity": receptor.sensitivity,
            "occupancy": receptor.occupancy,
            "expression_level": receptor.expression_level,
            "current_affinity": receptor.current_affinity,
            "signal_type": receptor.signal_type.value
        }

    def modulate_sensitivity(
        self,
        receptor_type: ReceptorType,
        factor: float
    ):
        """Externally modulate receptor sensitivity"""
        if receptor_type in self.receptors:
            receptor = self.receptors[receptor_type]
            receptor.sensitivity *= factor
            receptor.sensitivity = max(
                receptor.min_sensitivity,
                min(receptor.max_sensitivity, receptor.sensitivity)
            )

            for callback in self._callbacks["sensitivity_changed"]:
                callback(receptor_type, receptor.sensitivity)

    def get_total_signal(self, signal_type: SignalType) -> float:
        """Get total signal intensity of a type"""
        total = 0.0
        for signal in self.active_signals:
            if signal.signal_type == signal_type and signal.active:
                total += signal.intensity
        return total

    def get_signals_for_target(self, target: str) -> List[SignalCascade]:
        """Get all active signals targeting a system"""
        return [
            s for s in self.active_signals
            if target in s.targets and s.active
        ]

    def on_receptor_activated(self, callback: Callable):
        """Register callback for receptor activation"""
        self._callbacks["receptor_activated"].append(callback)

    def on_receptor_desensitized(self, callback: Callable):
        """Register callback for receptor desensitization"""
        self._callbacks["receptor_desensitized"].append(callback)

    def on_signal_generated(self, callback: Callable):
        """Register callback for signal generation"""
        self._callbacks["signal_generated"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get receptor system status"""
        receptor_summary = {}
        for r_type, receptor in self.receptors.items():
            receptor_summary[r_type.value] = {
                "sensitivity": round(receptor.sensitivity, 3),
                "state": receptor.state.value,
                "occupancy": round(receptor.occupancy, 3)
            }

        return {
            "cycle_count": self.cycle_count,
            "total_receptors": len(self.receptors),
            "active_signals": len(self.active_signals),
            "binding_events": len(self.binding_history),
            "receptors": receptor_summary,
            "signal_totals": {
                "excitatory": self.get_total_signal(SignalType.EXCITATORY),
                "inhibitory": self.get_total_signal(SignalType.INHIBITORY),
                "modulatory": self.get_total_signal(SignalType.MODULATORY),
                "metabolic": self.get_total_signal(SignalType.METABOLIC)
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Receptor Binding System Demo")
    print("=" * 50)

    # Create receptor system
    receptor_system = ReceptorSystem()

    print(f"\n1. Initialized {len(receptor_system.receptors)} receptor types")

    # Process dopamine binding
    print("\n2. Processing dopamine (0.7 concentration)...")
    signals = receptor_system.process_hormone(HormoneType.DOPAMINE, 0.7)
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.source_receptor.value}: intensity={signal.intensity:.2f}")

    # Check receptor states
    print("\n3. Dopamine receptor states:")
    for r_type in [ReceptorType.DOPAMINE_D1, ReceptorType.DOPAMINE_D2]:
        state = receptor_system.get_receptor_state(r_type)
        print(f"   {r_type.value}: sensitivity={state['sensitivity']:.2f}, "
              f"occupancy={state['occupancy']:.2f}")

    # Run cycles to show desensitization
    print("\n4. Chronic stimulation (50 cycles with high dopamine)...")
    for i in range(50):
        receptor_system.process_hormone(HormoneType.DOPAMINE, 0.8)
        receptor_system.tick()

        if i % 10 == 0:
            d1_sens = receptor_system.get_sensitivity(ReceptorType.DOPAMINE_D1)
            print(f"   Cycle {i}: D1 sensitivity = {d1_sens:.3f}")

    # Recovery period
    print("\n5. Recovery period (30 cycles without stimulation)...")
    for i in range(30):
        receptor_system.tick()

        if i % 10 == 0:
            d1_sens = receptor_system.get_sensitivity(ReceptorType.DOPAMINE_D1)
            print(f"   Cycle {i}: D1 sensitivity = {d1_sens:.3f}")

    # Final status
    print("\n6. Final status:")
    status = receptor_system.get_status()
    print(f"   Active signals: {status['active_signals']}")
    print(f"   Binding events recorded: {status['binding_events']}")
    print(f"   Signal totals: {status['signal_totals']}")
