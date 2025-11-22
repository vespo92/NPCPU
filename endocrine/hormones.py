"""
Endocrine Hormone System

Manages hormonal regulation for digital organisms.
Implements biological-inspired hormonal control including:
- Hormone production and decay
- Mood and drive modulation
- Long-term state regulation
- Stress response
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class HormoneType(Enum):
    """Types of hormones (digital analogues)"""
    # Energy/Activity
    ADRENALINE = "adrenaline"         # Fight-or-flight, energy boost
    CORTISOL = "cortisol"             # Stress response
    THYROXINE = "thyroxine"           # Metabolic rate

    # Reward/Motivation
    DOPAMINE = "dopamine"             # Reward, motivation
    SEROTONIN = "serotonin"           # Well-being, mood
    ENDORPHIN = "endorphin"           # Pain relief, pleasure

    # Social/Bonding
    OXYTOCIN = "oxytocin"             # Social bonding
    VASOPRESSIN = "vasopressin"       # Social recognition

    # Growth/Repair
    GROWTH_HORMONE = "growth_hormone" # Growth and repair
    MELATONIN = "melatonin"           # Rest/recovery cycles

    # Regulatory
    INSULIN = "insulin"               # Resource regulation
    GLUCAGON = "glucagon"             # Resource mobilization


class MoodState(Enum):
    """Mood states influenced by hormones"""
    EUPHORIC = "euphoric"
    HAPPY = "happy"
    CONTENT = "content"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    STRESSED = "stressed"
    DEPRESSED = "depressed"


class DriveState(Enum):
    """Motivational drive states"""
    HIGH_ENERGY = "high_energy"
    MOTIVATED = "motivated"
    BALANCED = "balanced"
    FATIGUED = "fatigued"
    EXHAUSTED = "exhausted"


class GlandType(Enum):
    """Types of hormone-producing glands"""
    HYPOTHALAMUS = "hypothalamus"     # Master regulator
    PITUITARY = "pituitary"           # Growth, metabolism
    ADRENAL = "adrenal"               # Stress response
    THYROID = "thyroid"               # Metabolic rate
    REWARD_CENTER = "reward_center"   # Dopamine production


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Hormone:
    """A hormone in the system"""
    type: HormoneType
    level: float = 0.0                # Current level (0-1)
    baseline: float = 0.5             # Normal/resting level
    decay_rate: float = 0.05          # How fast it decays
    sensitivity: float = 1.0          # Response sensitivity
    last_update: float = field(default_factory=time.time)


@dataclass
class HormoneEffect:
    """Effect of a hormone on system parameters"""
    hormone_type: HormoneType
    target_system: str                # Which system is affected
    parameter: str                    # Which parameter
    effect_type: str                  # "multiply", "add", "set"
    magnitude: float                  # Effect strength
    threshold: float = 0.3            # Minimum hormone level to activate


@dataclass
class Gland:
    """A hormone-producing gland"""
    type: GlandType
    name: str
    produces: List[HormoneType] = field(default_factory=list)
    production_rate: float = 0.1
    active: bool = True
    health: float = 1.0


@dataclass
class HormoneSignal:
    """A signal to trigger hormone release"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hormone_type: HormoneType = HormoneType.DOPAMINE
    intensity: float = 0.5
    source: str = ""
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Endocrine System
# ============================================================================

class EndocrineSystem:
    """
    Complete endocrine system for digital organisms.

    Features:
    - Hormone production and regulation
    - Mood and drive state management
    - Effects on other systems
    - Circadian-like rhythms
    - Stress response

    Example:
        endocrine = EndocrineSystem()

        # Trigger hormone release
        endocrine.release_hormone(HormoneType.DOPAMINE, 0.3)

        # Process endocrine cycle
        endocrine.tick()

        # Check mood
        mood = endocrine.get_mood()
    """

    def __init__(
        self,
        baseline_mood: MoodState = MoodState.NEUTRAL,
        stress_threshold: float = 0.7
    ):
        self.baseline_mood = baseline_mood
        self.stress_threshold = stress_threshold

        # Hormones
        self.hormones: Dict[HormoneType, Hormone] = self._initialize_hormones()

        # Glands
        self.glands: Dict[GlandType, Gland] = self._initialize_glands()

        # Current states
        self.mood = baseline_mood
        self.drive = DriveState.BALANCED
        self.stress_level: float = 0.0

        # Effects registry
        self.effects: List[HormoneEffect] = self._initialize_effects()

        # Cycle tracking
        self.cycle_count = 0
        self.phase: float = 0.0  # 0-1 for circadian-like cycles

        # History
        self.hormone_history: Dict[HormoneType, List[float]] = defaultdict(list)

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "mood_changed": [],
            "drive_changed": [],
            "stress_high": []
        }

    def _initialize_hormones(self) -> Dict[HormoneType, Hormone]:
        """Initialize hormone system"""
        hormones = {}

        configs = {
            HormoneType.ADRENALINE: {"baseline": 0.2, "decay_rate": 0.15},
            HormoneType.CORTISOL: {"baseline": 0.3, "decay_rate": 0.05},
            HormoneType.THYROXINE: {"baseline": 0.5, "decay_rate": 0.02},
            HormoneType.DOPAMINE: {"baseline": 0.4, "decay_rate": 0.08},
            HormoneType.SEROTONIN: {"baseline": 0.5, "decay_rate": 0.03},
            HormoneType.ENDORPHIN: {"baseline": 0.3, "decay_rate": 0.1},
            HormoneType.OXYTOCIN: {"baseline": 0.3, "decay_rate": 0.06},
            HormoneType.VASOPRESSIN: {"baseline": 0.3, "decay_rate": 0.04},
            HormoneType.GROWTH_HORMONE: {"baseline": 0.4, "decay_rate": 0.02},
            HormoneType.MELATONIN: {"baseline": 0.3, "decay_rate": 0.05},
            HormoneType.INSULIN: {"baseline": 0.5, "decay_rate": 0.04},
            HormoneType.GLUCAGON: {"baseline": 0.4, "decay_rate": 0.06}
        }

        for h_type in HormoneType:
            config = configs.get(h_type, {})
            hormones[h_type] = Hormone(
                type=h_type,
                level=config.get("baseline", 0.5),
                baseline=config.get("baseline", 0.5),
                decay_rate=config.get("decay_rate", 0.05)
            )

        return hormones

    def _initialize_glands(self) -> Dict[GlandType, Gland]:
        """Initialize gland system"""
        return {
            GlandType.HYPOTHALAMUS: Gland(
                type=GlandType.HYPOTHALAMUS,
                name="Hypothalamus",
                produces=[HormoneType.OXYTOCIN, HormoneType.VASOPRESSIN],
                production_rate=0.1
            ),
            GlandType.PITUITARY: Gland(
                type=GlandType.PITUITARY,
                name="Pituitary",
                produces=[HormoneType.GROWTH_HORMONE, HormoneType.ENDORPHIN],
                production_rate=0.08
            ),
            GlandType.ADRENAL: Gland(
                type=GlandType.ADRENAL,
                name="Adrenal",
                produces=[HormoneType.ADRENALINE, HormoneType.CORTISOL],
                production_rate=0.15
            ),
            GlandType.THYROID: Gland(
                type=GlandType.THYROID,
                name="Thyroid",
                produces=[HormoneType.THYROXINE],
                production_rate=0.05
            ),
            GlandType.REWARD_CENTER: Gland(
                type=GlandType.REWARD_CENTER,
                name="Reward Center",
                produces=[HormoneType.DOPAMINE, HormoneType.SEROTONIN],
                production_rate=0.12
            )
        }

    def _initialize_effects(self) -> List[HormoneEffect]:
        """Initialize hormone effects on systems"""
        return [
            # Adrenaline effects
            HormoneEffect(
                hormone_type=HormoneType.ADRENALINE,
                target_system="motor",
                parameter="speed",
                effect_type="multiply",
                magnitude=1.5
            ),
            HormoneEffect(
                hormone_type=HormoneType.ADRENALINE,
                target_system="sensory",
                parameter="sensitivity",
                effect_type="multiply",
                magnitude=1.3
            ),

            # Cortisol effects
            HormoneEffect(
                hormone_type=HormoneType.CORTISOL,
                target_system="metabolism",
                parameter="energy_consumption",
                effect_type="multiply",
                magnitude=1.2
            ),

            # Dopamine effects
            HormoneEffect(
                hormone_type=HormoneType.DOPAMINE,
                target_system="motor",
                parameter="motivation",
                effect_type="multiply",
                magnitude=1.4
            ),

            # Serotonin effects
            HormoneEffect(
                hormone_type=HormoneType.SEROTONIN,
                target_system="homeostasis",
                parameter="stability",
                effect_type="multiply",
                magnitude=1.2
            ),

            # Growth hormone effects
            HormoneEffect(
                hormone_type=HormoneType.GROWTH_HORMONE,
                target_system="organism",
                parameter="growth_rate",
                effect_type="multiply",
                magnitude=1.3
            ),

            # Melatonin effects
            HormoneEffect(
                hormone_type=HormoneType.MELATONIN,
                target_system="metabolism",
                parameter="recovery_rate",
                effect_type="multiply",
                magnitude=1.5
            ),

            # Thyroxine effects
            HormoneEffect(
                hormone_type=HormoneType.THYROXINE,
                target_system="metabolism",
                parameter="metabolic_rate",
                effect_type="multiply",
                magnitude=1.2
            )
        ]

    def release_hormone(
        self,
        hormone_type: HormoneType,
        amount: float,
        source: str = ""
    ):
        """Release a hormone into the system"""
        if hormone_type in self.hormones:
            hormone = self.hormones[hormone_type]
            hormone.level = min(1.0, hormone.level + amount * hormone.sensitivity)
            hormone.last_update = time.time()

    def trigger_stress_response(self, intensity: float):
        """Trigger a stress response"""
        self.release_hormone(HormoneType.CORTISOL, intensity * 0.4)
        self.release_hormone(HormoneType.ADRENALINE, intensity * 0.6)

        self.stress_level = min(1.0, self.stress_level + intensity * 0.3)

        if self.stress_level > self.stress_threshold:
            for callback in self._callbacks["stress_high"]:
                callback(self.stress_level)

    def trigger_reward(self, intensity: float):
        """Trigger a reward response"""
        self.release_hormone(HormoneType.DOPAMINE, intensity * 0.5)
        self.release_hormone(HormoneType.SEROTONIN, intensity * 0.3)
        self.release_hormone(HormoneType.ENDORPHIN, intensity * 0.2)

    def trigger_social_bonding(self, intensity: float):
        """Trigger social bonding hormones"""
        self.release_hormone(HormoneType.OXYTOCIN, intensity * 0.6)
        self.release_hormone(HormoneType.VASOPRESSIN, intensity * 0.3)

    def trigger_rest_mode(self):
        """Trigger rest/recovery hormones"""
        self.release_hormone(HormoneType.MELATONIN, 0.4)
        self.release_hormone(HormoneType.GROWTH_HORMONE, 0.3)

    def tick(self):
        """Process one endocrine cycle"""
        self.cycle_count += 1

        # Update circadian phase
        self.phase = (self.phase + 0.01) % 1.0

        # Natural hormone decay toward baseline
        for hormone in self.hormones.values():
            diff = hormone.level - hormone.baseline
            decay = diff * hormone.decay_rate
            hormone.level -= decay
            hormone.level = max(0.0, min(1.0, hormone.level))

        # Gland production (circadian modulated)
        for gland in self.glands.values():
            if not gland.active:
                continue

            # Circadian modulation
            circadian_mod = 0.5 + 0.5 * np.sin(2 * np.pi * self.phase)

            for hormone_type in gland.produces:
                production = gland.production_rate * gland.health * circadian_mod * 0.1
                self.release_hormone(hormone_type, production)

        # Update mood and drive
        self._update_mood()
        self._update_drive()

        # Stress recovery
        self.stress_level = max(0, self.stress_level - 0.02)

        # Record history
        for h_type, hormone in self.hormones.items():
            self.hormone_history[h_type].append(hormone.level)
            if len(self.hormone_history[h_type]) > 100:
                self.hormone_history[h_type].pop(0)

    def _update_mood(self):
        """Update mood based on hormone levels"""
        old_mood = self.mood

        # Calculate mood score
        dopamine = self.hormones[HormoneType.DOPAMINE].level
        serotonin = self.hormones[HormoneType.SEROTONIN].level
        cortisol = self.hormones[HormoneType.CORTISOL].level
        endorphin = self.hormones[HormoneType.ENDORPHIN].level

        positive = (dopamine + serotonin + endorphin) / 3
        negative = cortisol * 0.5 + self.stress_level * 0.5

        mood_score = positive - negative

        if mood_score > 0.6:
            self.mood = MoodState.EUPHORIC
        elif mood_score > 0.4:
            self.mood = MoodState.HAPPY
        elif mood_score > 0.2:
            self.mood = MoodState.CONTENT
        elif mood_score > -0.1:
            self.mood = MoodState.NEUTRAL
        elif mood_score > -0.3:
            self.mood = MoodState.ANXIOUS
        elif mood_score > -0.5:
            self.mood = MoodState.STRESSED
        else:
            self.mood = MoodState.DEPRESSED

        if self.mood != old_mood:
            for callback in self._callbacks["mood_changed"]:
                callback(self.mood)

    def _update_drive(self):
        """Update drive state based on hormones"""
        old_drive = self.drive

        adrenaline = self.hormones[HormoneType.ADRENALINE].level
        thyroxine = self.hormones[HormoneType.THYROXINE].level
        dopamine = self.hormones[HormoneType.DOPAMINE].level
        cortisol = self.hormones[HormoneType.CORTISOL].level
        melatonin = self.hormones[HormoneType.MELATONIN].level

        # Energy drive calculation
        energy_boost = (adrenaline + thyroxine + dopamine) / 3
        energy_drain = (cortisol * 0.3 + melatonin * 0.5 + self.stress_level * 0.2)

        drive_score = energy_boost - energy_drain

        if drive_score > 0.5:
            self.drive = DriveState.HIGH_ENERGY
        elif drive_score > 0.2:
            self.drive = DriveState.MOTIVATED
        elif drive_score > -0.1:
            self.drive = DriveState.BALANCED
        elif drive_score > -0.3:
            self.drive = DriveState.FATIGUED
        else:
            self.drive = DriveState.EXHAUSTED

        if self.drive != old_drive:
            for callback in self._callbacks["drive_changed"]:
                callback(self.drive)

    def get_hormone_level(self, hormone_type: HormoneType) -> float:
        """Get current level of a hormone"""
        if hormone_type in self.hormones:
            return self.hormones[hormone_type].level
        return 0.0

    def get_mood(self) -> MoodState:
        """Get current mood state"""
        return self.mood

    def get_drive(self) -> DriveState:
        """Get current drive state"""
        return self.drive

    def get_active_effects(self) -> List[Dict[str, Any]]:
        """Get currently active hormone effects"""
        active = []

        for effect in self.effects:
            hormone_level = self.get_hormone_level(effect.hormone_type)
            if hormone_level >= effect.threshold:
                active.append({
                    "hormone": effect.hormone_type.value,
                    "target": effect.target_system,
                    "parameter": effect.parameter,
                    "effect": effect.effect_type,
                    "magnitude": effect.magnitude,
                    "intensity": hormone_level
                })

        return active

    def on_mood_changed(self, callback: Callable[[MoodState], None]):
        """Register callback for mood changes"""
        self._callbacks["mood_changed"].append(callback)

    def on_drive_changed(self, callback: Callable[[DriveState], None]):
        """Register callback for drive changes"""
        self._callbacks["drive_changed"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get endocrine system status"""
        return {
            "mood": self.mood.value,
            "drive": self.drive.value,
            "stress_level": self.stress_level,
            "phase": self.phase,
            "cycle_count": self.cycle_count,
            "hormones": {
                h_type.value: {
                    "level": h.level,
                    "baseline": h.baseline
                }
                for h_type, h in self.hormones.items()
            },
            "glands_active": len([g for g in self.glands.values() if g.active]),
            "active_effects": len(self.get_active_effects())
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Endocrine Hormone System Demo")
    print("=" * 50)

    # Create endocrine system
    endocrine = EndocrineSystem()

    print(f"\n1. Initial state:")
    status = endocrine.get_status()
    print(f"   Mood: {status['mood']}")
    print(f"   Drive: {status['drive']}")
    print(f"   Stress: {status['stress_level']:.2f}")

    # Normal operation
    print("\n2. Running normal cycles...")
    for cycle in range(20):
        endocrine.tick()

        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: mood={endocrine.mood.value}, "
                  f"drive={endocrine.drive.value}")

    # Trigger stress response
    print("\n3. Triggering stress response...")
    endocrine.trigger_stress_response(0.8)
    endocrine.tick()
    print(f"   Mood: {endocrine.mood.value}")
    print(f"   Stress level: {endocrine.stress_level:.2f}")
    print(f"   Cortisol: {endocrine.get_hormone_level(HormoneType.CORTISOL):.2f}")
    print(f"   Adrenaline: {endocrine.get_hormone_level(HormoneType.ADRENALINE):.2f}")

    # Recovery cycles
    print("\n4. Recovery cycles...")
    for cycle in range(15):
        endocrine.tick()

        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: stress={endocrine.stress_level:.2f}, "
                  f"mood={endocrine.mood.value}")

    # Trigger reward
    print("\n5. Triggering reward response...")
    endocrine.trigger_reward(0.7)
    endocrine.tick()
    print(f"   Mood: {endocrine.mood.value}")
    print(f"   Dopamine: {endocrine.get_hormone_level(HormoneType.DOPAMINE):.2f}")
    print(f"   Serotonin: {endocrine.get_hormone_level(HormoneType.SEROTONIN):.2f}")

    # Trigger social bonding
    print("\n6. Triggering social bonding...")
    endocrine.trigger_social_bonding(0.6)
    endocrine.tick()
    print(f"   Oxytocin: {endocrine.get_hormone_level(HormoneType.OXYTOCIN):.2f}")

    # Get active effects
    print("\n7. Active hormone effects:")
    effects = endocrine.get_active_effects()
    for effect in effects[:5]:
        print(f"   {effect['hormone']} -> {effect['target']}.{effect['parameter']}: "
              f"x{effect['magnitude']}")

    # Final status
    print("\n8. Final status:")
    status = endocrine.get_status()
    print(f"   Mood: {status['mood']}")
    print(f"   Drive: {status['drive']}")
    print(f"   Phase: {status['phase']:.2f}")

    print("\n   Key hormone levels:")
    for h_name in ["dopamine", "serotonin", "cortisol", "adrenaline"]:
        level = status["hormones"][h_name]["level"]
        baseline = status["hormones"][h_name]["baseline"]
        print(f"      {h_name}: {level:.2f} (baseline: {baseline:.2f})")
