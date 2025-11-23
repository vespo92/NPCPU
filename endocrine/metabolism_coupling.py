"""
Endocrine-Metabolism Coupling System

Implements the integration between endocrine and metabolic systems including:
- Glucose-insulin dynamics
- Energy homeostasis
- Metabolic rate modulation
- Hunger and satiety signaling
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

class MetabolicState(Enum):
    """Overall metabolic states"""
    ANABOLIC = "anabolic"           # Building, storing
    CATABOLIC = "catabolic"         # Breaking down, using
    BALANCED = "balanced"           # Homeostasis
    STRESSED = "stressed"           # Metabolic stress
    RECOVERY = "recovery"           # Post-stress recovery


class EnergyState(Enum):
    """Energy availability states"""
    SURPLUS = "surplus"             # Excess energy
    ADEQUATE = "adequate"           # Normal energy
    DEPLETING = "depleting"         # Using reserves
    DEPLETED = "depleted"           # Low reserves
    CRITICAL = "critical"           # Emergency low


class HungerLevel(Enum):
    """Hunger/satiety states"""
    OVERFED = "overfed"
    SATIATED = "satiated"
    NEUTRAL = "neutral"
    HUNGRY = "hungry"
    STARVING = "starving"


class MetabolicRate(Enum):
    """Metabolic rate states"""
    HYPERMETABOLIC = "hypermetabolic"   # High rate
    ELEVATED = "elevated"
    NORMAL = "normal"
    REDUCED = "reduced"
    HYPOMETABOLIC = "hypometabolic"     # Low rate


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GlucoseState:
    """Blood glucose homeostasis"""
    level: float = 0.5              # Current glucose (0-1)
    target: float = 0.5             # Target glucose
    rate_of_change: float = 0.0     # Rising/falling
    time_in_range: float = 0.0      # Time at target
    variability: float = 0.1        # Glucose variability


@dataclass
class EnergyReserves:
    """Energy storage and reserves"""
    immediate: float = 0.5          # Immediately available (ATP-like)
    short_term: float = 0.7         # Short-term (glycogen-like)
    long_term: float = 0.8          # Long-term (fat-like)
    total_capacity: float = 1.0     # Storage capacity


@dataclass
class MetabolicSignal:
    """A metabolic hormone signal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hormone: HormoneType = HormoneType.INSULIN
    target: str = ""                # Target tissue/process
    intensity: float = 0.5
    effect: str = ""                # "uptake", "release", "inhibit"
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Metabolism Coupling System
# ============================================================================

class MetabolismCouplingSystem:
    """
    Endocrine-metabolism integration system.

    Features:
    - Glucose homeostasis with insulin/glucagon
    - Energy reserve management
    - Hunger/satiety signaling (ghrelin/leptin-like)
    - Thyroid-mediated metabolic rate
    - Cortisol metabolic effects

    Example:
        coupling = MetabolismCouplingSystem()

        # Feed (increase glucose)
        coupling.consume_energy(0.3)

        # Process metabolism
        coupling.tick()

        # Get hormone requirements
        effects = coupling.get_hormone_effects()
    """

    def __init__(
        self,
        base_metabolic_rate: float = 0.5,
        insulin_sensitivity: float = 1.0
    ):
        self.base_metabolic_rate = base_metabolic_rate
        self.insulin_sensitivity = insulin_sensitivity

        # Glucose homeostasis
        self.glucose = GlucoseState()

        # Energy reserves
        self.reserves = EnergyReserves()

        # States
        self.metabolic_state = MetabolicState.BALANCED
        self.energy_state = EnergyState.ADEQUATE
        self.hunger_level = HungerLevel.NEUTRAL
        self.metabolic_rate_state = MetabolicRate.NORMAL

        # Current metabolic rate
        self.current_metabolic_rate: float = base_metabolic_rate

        # Hormone levels (local tracking)
        self.insulin_activity: float = 0.5
        self.glucagon_activity: float = 0.3
        self.ghrelin_level: float = 0.3      # Hunger hormone
        self.leptin_level: float = 0.5       # Satiety hormone

        # Energy flux
        self.energy_intake: float = 0.0      # Recent intake
        self.energy_expenditure: float = 0.0  # Recent expenditure

        # History
        self.glucose_history: List[float] = []
        self.max_history = 100

        # Cycle tracking
        self.cycle_count = 0
        self.cycles_since_feeding = 0

        # Hormone effects to request
        self._hormone_effects: Dict[HormoneType, float] = {}

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "glucose_high": [],
            "glucose_low": [],
            "hunger_signal": [],
            "metabolic_state_changed": [],
            "energy_critical": []
        }

    def consume_energy(self, amount: float, nutrient_type: str = "mixed"):
        """Consume energy (feeding)"""
        self.energy_intake = amount
        self.cycles_since_feeding = 0

        # Increase glucose
        glucose_increase = amount * 0.7
        self.glucose.level = min(1.0, self.glucose.level + glucose_increase)

        # Store excess as reserves
        excess = max(0, self.glucose.level - 0.6)
        if excess > 0:
            self.reserves.short_term = min(1.0, self.reserves.short_term + excess * 0.3)
            self.reserves.long_term = min(1.0, self.reserves.long_term + excess * 0.1)
            self.glucose.level -= excess * 0.4

        # Update leptin (satiety)
        self.leptin_level = min(1.0, self.leptin_level + amount * 0.3)

        # Reduce ghrelin (hunger)
        self.ghrelin_level = max(0.0, self.ghrelin_level - amount * 0.5)

    def expend_energy(self, amount: float, activity_type: str = "basal"):
        """Expend energy (metabolism, activity)"""
        self.energy_expenditure = amount

        # Use immediate reserves first
        if self.reserves.immediate >= amount:
            self.reserves.immediate -= amount
        else:
            shortfall = amount - self.reserves.immediate
            self.reserves.immediate = 0.0

            # Use glucose
            glucose_used = min(shortfall, self.glucose.level * 0.5)
            self.glucose.level -= glucose_used * 0.5
            shortfall -= glucose_used

            # Use short-term reserves
            if shortfall > 0:
                short_used = min(shortfall, self.reserves.short_term)
                self.reserves.short_term -= short_used
                self.glucose.level += short_used * 0.3  # Convert to glucose
                shortfall -= short_used

            # Use long-term reserves
            if shortfall > 0:
                long_used = min(shortfall, self.reserves.long_term * 0.1)
                self.reserves.long_term -= long_used

    def tick(self, external_hormones: Optional[Dict[HormoneType, float]] = None):
        """Process one metabolic cycle"""
        self.cycle_count += 1
        self.cycles_since_feeding += 1

        # Apply external hormone effects
        if external_hormones:
            self._apply_external_hormones(external_hormones)

        # Basal metabolism
        basal_expenditure = self.current_metabolic_rate * 0.01
        self.expend_energy(basal_expenditure)

        # Glucose homeostasis
        self._regulate_glucose()

        # Regenerate immediate reserves
        self._regenerate_reserves()

        # Update hunger/satiety signals
        self._update_hunger_signals()

        # Update metabolic rate
        self._update_metabolic_rate()

        # Update states
        self._update_states()

        # Generate hormone effects
        self._generate_hormone_effects()

        # Record history
        self.glucose_history.append(self.glucose.level)
        if len(self.glucose_history) > self.max_history:
            self.glucose_history.pop(0)

    def _apply_external_hormones(self, hormones: Dict[HormoneType, float]):
        """Apply effects of external hormone levels"""
        # Insulin effects
        if HormoneType.INSULIN in hormones:
            insulin = hormones[HormoneType.INSULIN]
            self.insulin_activity = insulin * self.insulin_sensitivity

        # Glucagon effects
        if HormoneType.GLUCAGON in hormones:
            self.glucagon_activity = hormones[HormoneType.GLUCAGON]

        # Cortisol effects (increases glucose)
        if HormoneType.CORTISOL in hormones:
            cortisol = hormones[HormoneType.CORTISOL]
            if cortisol > 0.5:
                self.glucose.level = min(1.0, self.glucose.level + cortisol * 0.02)

        # Thyroxine effects (metabolic rate)
        if HormoneType.THYROXINE in hormones:
            thyroxine = hormones[HormoneType.THYROXINE]
            rate_mod = (thyroxine - 0.5) * 0.5
            self.current_metabolic_rate = self.base_metabolic_rate * (1.0 + rate_mod)

        # Adrenaline effects (quick energy)
        if HormoneType.ADRENALINE in hormones:
            adrenaline = hormones[HormoneType.ADRENALINE]
            if adrenaline > 0.5:
                # Mobilize energy
                self.reserves.immediate = min(
                    1.0,
                    self.reserves.immediate + adrenaline * 0.1
                )
                self.glucose.level = min(1.0, self.glucose.level + adrenaline * 0.05)

    def _regulate_glucose(self):
        """Regulate blood glucose via insulin/glucagon"""
        target = self.glucose.target
        current = self.glucose.level
        error = current - target

        if error > 0.1:
            # High glucose: increase insulin, decrease glucagon
            self.insulin_activity = min(1.0, self.insulin_activity + 0.05)
            self.glucagon_activity = max(0.0, self.glucagon_activity - 0.05)

            # Store glucose
            storage = error * self.insulin_activity * self.insulin_sensitivity * 0.2
            self.glucose.level -= storage
            self.reserves.short_term = min(1.0, self.reserves.short_term + storage * 0.5)

            if current > 0.8:
                for callback in self._callbacks["glucose_high"]:
                    callback(current)

        elif error < -0.1:
            # Low glucose: increase glucagon, decrease insulin
            self.glucagon_activity = min(1.0, self.glucagon_activity + 0.05)
            self.insulin_activity = max(0.1, self.insulin_activity - 0.05)

            # Release glucose from reserves
            release = abs(error) * self.glucagon_activity * 0.2
            available = min(release, self.reserves.short_term)
            self.reserves.short_term -= available
            self.glucose.level += available

            if current < 0.2:
                for callback in self._callbacks["glucose_low"]:
                    callback(current)

        else:
            # Normal range: normalize hormones
            self.insulin_activity += (0.5 - self.insulin_activity) * 0.1
            self.glucagon_activity += (0.3 - self.glucagon_activity) * 0.1
            self.glucose.time_in_range += 1

        # Calculate rate of change
        if len(self.glucose_history) >= 2:
            self.glucose.rate_of_change = self.glucose.level - self.glucose_history[-1]

    def _regenerate_reserves(self):
        """Regenerate immediate energy reserves"""
        if self.reserves.immediate < 0.8 and self.glucose.level > 0.4:
            regen = 0.02 * (1.0 + self.insulin_activity)
            self.reserves.immediate = min(1.0, self.reserves.immediate + regen)
            self.glucose.level -= regen * 0.5

    def _update_hunger_signals(self):
        """Update hunger and satiety hormones"""
        # Ghrelin increases with time since feeding
        ghrelin_target = min(1.0, self.cycles_since_feeding / 50)
        self.ghrelin_level += (ghrelin_target - self.ghrelin_level) * 0.05

        # Leptin decreases over time
        self.leptin_level = max(0.0, self.leptin_level - 0.01)

        # Leptin correlates with long-term reserves
        self.leptin_level = max(self.leptin_level, self.reserves.long_term * 0.5)

        # Update hunger level
        hunger_signal = self.ghrelin_level - self.leptin_level

        if hunger_signal > 0.5:
            self.hunger_level = HungerLevel.STARVING
            for callback in self._callbacks["hunger_signal"]:
                callback(self.ghrelin_level)
        elif hunger_signal > 0.2:
            self.hunger_level = HungerLevel.HUNGRY
        elif hunger_signal > -0.2:
            self.hunger_level = HungerLevel.NEUTRAL
        elif hunger_signal > -0.4:
            self.hunger_level = HungerLevel.SATIATED
        else:
            self.hunger_level = HungerLevel.OVERFED

    def _update_metabolic_rate(self):
        """Update metabolic rate state"""
        rate = self.current_metabolic_rate

        if rate > self.base_metabolic_rate * 1.5:
            self.metabolic_rate_state = MetabolicRate.HYPERMETABOLIC
        elif rate > self.base_metabolic_rate * 1.2:
            self.metabolic_rate_state = MetabolicRate.ELEVATED
        elif rate > self.base_metabolic_rate * 0.8:
            self.metabolic_rate_state = MetabolicRate.NORMAL
        elif rate > self.base_metabolic_rate * 0.6:
            self.metabolic_rate_state = MetabolicRate.REDUCED
        else:
            self.metabolic_rate_state = MetabolicRate.HYPOMETABOLIC

    def _update_states(self):
        """Update metabolic and energy states"""
        old_metabolic = self.metabolic_state
        old_energy = self.energy_state

        # Energy state
        total_energy = (
            self.reserves.immediate * 0.2 +
            self.reserves.short_term * 0.3 +
            self.reserves.long_term * 0.5
        )

        if total_energy > 0.8:
            self.energy_state = EnergyState.SURPLUS
        elif total_energy > 0.5:
            self.energy_state = EnergyState.ADEQUATE
        elif total_energy > 0.3:
            self.energy_state = EnergyState.DEPLETING
        elif total_energy > 0.1:
            self.energy_state = EnergyState.DEPLETED
        else:
            self.energy_state = EnergyState.CRITICAL
            for callback in self._callbacks["energy_critical"]:
                callback(total_energy)

        # Metabolic state
        if self.energy_intake > self.energy_expenditure and self.insulin_activity > 0.6:
            self.metabolic_state = MetabolicState.ANABOLIC
        elif self.glucagon_activity > 0.6 or self.energy_state in [EnergyState.DEPLETED, EnergyState.CRITICAL]:
            self.metabolic_state = MetabolicState.CATABOLIC
        elif self.glucose.time_in_range > 50:
            self.metabolic_state = MetabolicState.BALANCED
        else:
            self.metabolic_state = MetabolicState.BALANCED

        if old_metabolic != self.metabolic_state:
            for callback in self._callbacks["metabolic_state_changed"]:
                callback(self.metabolic_state)

    def _generate_hormone_effects(self):
        """Generate hormone effect requests"""
        self._hormone_effects = {}

        # Insulin effect
        insulin_target = 0.5
        if self.glucose.level > 0.6:
            insulin_target = 0.7
        elif self.glucose.level < 0.4:
            insulin_target = 0.3
        self._hormone_effects[HormoneType.INSULIN] = insulin_target - 0.5

        # Glucagon effect
        glucagon_target = 0.4
        if self.glucose.level < 0.4:
            glucagon_target = 0.7
        elif self.glucose.level > 0.6:
            glucagon_target = 0.2
        self._hormone_effects[HormoneType.GLUCAGON] = glucagon_target - 0.4

        # Thyroxine effect (metabolic rate adjustment)
        if self.metabolic_rate_state == MetabolicRate.HYPERMETABOLIC:
            self._hormone_effects[HormoneType.THYROXINE] = 0.2
        elif self.metabolic_rate_state == MetabolicRate.HYPOMETABOLIC:
            self._hormone_effects[HormoneType.THYROXINE] = -0.2

        # Growth hormone (for repair/growth when fed)
        if self.metabolic_state == MetabolicState.ANABOLIC:
            self._hormone_effects[HormoneType.GROWTH_HORMONE] = 0.15

    def get_hormone_effects(self) -> Dict[HormoneType, float]:
        """Get hormone effects from metabolism"""
        return self._hormone_effects.copy()

    def get_energy_available(self) -> float:
        """Get total available energy"""
        return (
            self.reserves.immediate +
            self.reserves.short_term * 0.5 +
            self.reserves.long_term * 0.2
        )

    def get_hunger_signal(self) -> float:
        """Get hunger signal strength (0-1)"""
        return max(0, self.ghrelin_level - self.leptin_level)

    def on_hunger_signal(self, callback: Callable):
        """Register callback for hunger"""
        self._callbacks["hunger_signal"].append(callback)

    def on_energy_critical(self, callback: Callable):
        """Register callback for critical energy"""
        self._callbacks["energy_critical"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get metabolism coupling status"""
        return {
            "metabolic_state": self.metabolic_state.value,
            "energy_state": self.energy_state.value,
            "hunger_level": self.hunger_level.value,
            "metabolic_rate": self.metabolic_rate_state.value,
            "glucose": {
                "level": round(self.glucose.level, 3),
                "target": self.glucose.target,
                "rate_of_change": round(self.glucose.rate_of_change, 4)
            },
            "reserves": {
                "immediate": round(self.reserves.immediate, 3),
                "short_term": round(self.reserves.short_term, 3),
                "long_term": round(self.reserves.long_term, 3),
                "total": round(self.get_energy_available(), 3)
            },
            "hormones": {
                "insulin_activity": round(self.insulin_activity, 3),
                "glucagon_activity": round(self.glucagon_activity, 3),
                "ghrelin": round(self.ghrelin_level, 3),
                "leptin": round(self.leptin_level, 3)
            },
            "hunger_signal": round(self.get_hunger_signal(), 3),
            "cycle_count": self.cycle_count
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Metabolism Coupling System Demo")
    print("=" * 50)

    # Create system
    coupling = MetabolismCouplingSystem()

    print(f"\n1. Initial state:")
    status = coupling.get_status()
    print(f"   Glucose: {status['glucose']['level']:.2f}")
    print(f"   Energy state: {status['energy_state']}")
    print(f"   Hunger: {status['hunger_level']}")

    # Consume energy (feeding)
    print("\n2. Consuming energy (0.5)...")
    coupling.consume_energy(0.5)
    coupling.tick()

    status = coupling.get_status()
    print(f"   Glucose: {status['glucose']['level']:.2f}")
    print(f"   Insulin: {status['hormones']['insulin_activity']:.2f}")
    print(f"   Hunger: {status['hunger_level']}")

    # Run metabolism
    print("\n3. Running metabolism (20 cycles)...")
    for i in range(20):
        coupling.tick()

    status = coupling.get_status()
    print(f"   Glucose: {status['glucose']['level']:.2f}")
    print(f"   Reserves: {status['reserves']}")

    # Fasting period
    print("\n4. Fasting period (50 cycles)...")
    for i in range(50):
        coupling.tick()

        if i % 10 == 0:
            s = coupling.get_status()
            print(f"   Cycle {i}: glucose={s['glucose']['level']:.2f}, "
                  f"hunger={s['hunger_level']}")

    # High activity
    print("\n5. High energy expenditure...")
    for i in range(10):
        coupling.expend_energy(0.05, "exercise")
        coupling.tick()

    status = coupling.get_status()
    print(f"   Glucose: {status['glucose']['level']:.2f}")
    print(f"   Glucagon: {status['hormones']['glucagon_activity']:.2f}")
    print(f"   Reserves: {status['reserves']}")

    # Feed again
    print("\n6. Feeding again (0.4)...")
    coupling.consume_energy(0.4)
    coupling.tick()

    status = coupling.get_status()
    print(f"   Metabolic state: {status['metabolic_state']}")
    print(f"   Hunger: {status['hunger_level']}")

    # Final status
    print("\n7. Final status:")
    status = coupling.get_status()
    print(f"   Metabolic state: {status['metabolic_state']}")
    print(f"   Energy available: {status['reserves']['total']:.2f}")
    print(f"   Hormone effects: {coupling.get_hormone_effects()}")
