"""
Circadian Rhythm System

Implements biological clock and day/night hormone cycling including:
- Core oscillator (master clock)
- Hormone production rhythms
- Sleep-wake cycle regulation
- Entrainment and phase shifting
"""

import time
import uuid
import math
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

class CircadianPhase(Enum):
    """Phases of the circadian cycle"""
    EARLY_MORNING = "early_morning"       # 0.0-0.125 (6-9 AM)
    LATE_MORNING = "late_morning"         # 0.125-0.25 (9 AM-12 PM)
    EARLY_AFTERNOON = "early_afternoon"   # 0.25-0.375 (12-3 PM)
    LATE_AFTERNOON = "late_afternoon"     # 0.375-0.5 (3-6 PM)
    EARLY_EVENING = "early_evening"       # 0.5-0.625 (6-9 PM)
    LATE_EVENING = "late_evening"         # 0.625-0.75 (9 PM-12 AM)
    EARLY_NIGHT = "early_night"           # 0.75-0.875 (12-3 AM)
    LATE_NIGHT = "late_night"             # 0.875-1.0 (3-6 AM)


class SleepState(Enum):
    """Sleep-wake states"""
    AWAKE = "awake"
    DROWSY = "drowsy"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM = "rem"


class ChronotypeBias(Enum):
    """Chronotype (morning/evening preference)"""
    EARLY_BIRD = "early_bird"         # Morning preference
    INTERMEDIATE = "intermediate"      # Neutral
    NIGHT_OWL = "night_owl"           # Evening preference


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CircadianOscillator:
    """Core circadian oscillator (biological clock)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "master_clock"

    # Phase tracking
    phase: float = 0.0                    # 0-1 representing 24-hour cycle
    period: float = 1.0                   # Natural period (1.0 = 24 hours)
    amplitude: float = 1.0                # Oscillation strength

    # Stability
    coupling_strength: float = 0.8        # How strongly it drives peripheral clocks
    entrainment_rate: float = 0.1         # How fast it responds to zeitgebers

    # State
    free_running: bool = False            # True if not entrained to external cues
    phase_locked: bool = True             # True if stably entrained


@dataclass
class HormoneRhythm:
    """Circadian rhythm profile for a hormone"""
    hormone_type: HormoneType = HormoneType.CORTISOL

    # Rhythm parameters
    peak_phase: float = 0.0               # Phase of peak production (0-1)
    trough_phase: float = 0.5             # Phase of minimum production
    amplitude: float = 0.3                # Amplitude of oscillation
    baseline: float = 0.5                 # Mean level

    # Waveform
    waveform: str = "cosine"              # "cosine", "square", "asymmetric"
    asymmetry: float = 0.0                # Asymmetry factor for waveform


@dataclass
class ZeitgeberEvent:
    """Time-giving signal that entrains the clock"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""                      # "light", "activity", "food", "social"
    intensity: float = 0.5                # Signal strength
    phase: float = 0.0                    # When the signal occurred
    direction: str = "advance"            # "advance" or "delay"
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Circadian System
# ============================================================================

class CircadianSystem:
    """
    Complete circadian rhythm system for digital organisms.

    Features:
    - Master oscillator (SCN-like)
    - Hormone-specific rhythms
    - Sleep-wake cycle regulation
    - Entrainment to external cues
    - Phase response curves
    - Chronotype variation

    Example:
        circadian = CircadianSystem()

        # Advance time
        circadian.tick()

        # Get current hormone modifiers
        modifiers = circadian.get_hormone_modifiers()

        # Check sleep pressure
        pressure = circadian.get_sleep_pressure()
    """

    def __init__(
        self,
        chronotype: ChronotypeBias = ChronotypeBias.INTERMEDIATE,
        initial_phase: float = 0.25,  # Start at 12 PM
        cycle_speed: float = 0.01     # How fast time passes
    ):
        self.chronotype = chronotype
        self.cycle_speed = cycle_speed

        # Master oscillator
        self.master_clock = CircadianOscillator(
            phase=initial_phase,
            period=self._get_chronotype_period()
        )

        # Hormone rhythms
        self.hormone_rhythms: Dict[HormoneType, HormoneRhythm] = \
            self._initialize_rhythms()

        # Sleep-wake state
        self.sleep_state = SleepState.AWAKE
        self.sleep_pressure: float = 0.0       # Homeostatic sleep drive
        self.sleep_debt: float = 0.0           # Accumulated sleep deficit
        self.awake_time: float = 0.0           # Time since last sleep

        # Zeitgeber history
        self.zeitgeber_history: List[ZeitgeberEvent] = []
        self.max_history = 100

        # Cycle tracking
        self.cycle_count = 0
        self.days_elapsed = 0

        # Phase history
        self.phase_history: List[float] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "phase_changed": [],
            "sleep_state_changed": [],
            "new_day": [],
            "high_sleep_pressure": []
        }

    def _get_chronotype_period(self) -> float:
        """Get natural period based on chronotype"""
        if self.chronotype == ChronotypeBias.EARLY_BIRD:
            return 0.98  # Slightly shorter than 24h
        elif self.chronotype == ChronotypeBias.NIGHT_OWL:
            return 1.02  # Slightly longer than 24h
        return 1.0

    def _initialize_rhythms(self) -> Dict[HormoneType, HormoneRhythm]:
        """Initialize circadian rhythm profiles for each hormone"""
        return {
            # Cortisol: peaks in early morning
            HormoneType.CORTISOL: HormoneRhythm(
                hormone_type=HormoneType.CORTISOL,
                peak_phase=0.0,      # 6 AM
                trough_phase=0.75,   # 12 AM
                amplitude=0.4,
                baseline=0.35,
                waveform="asymmetric",
                asymmetry=0.3        # Fast rise, slow fall
            ),

            # Melatonin: peaks at night
            HormoneType.MELATONIN: HormoneRhythm(
                hormone_type=HormoneType.MELATONIN,
                peak_phase=0.83,     # 2 AM
                trough_phase=0.33,   # 2 PM
                amplitude=0.5,
                baseline=0.25,
                waveform="cosine"
            ),

            # Growth hormone: peaks during deep sleep
            HormoneType.GROWTH_HORMONE: HormoneRhythm(
                hormone_type=HormoneType.GROWTH_HORMONE,
                peak_phase=0.79,     # 1 AM
                trough_phase=0.29,   # 1 PM
                amplitude=0.45,
                baseline=0.3,
                waveform="asymmetric",
                asymmetry=-0.2       # Fast peak during sleep
            ),

            # Thyroxine: peaks in late morning
            HormoneType.THYROXINE: HormoneRhythm(
                hormone_type=HormoneType.THYROXINE,
                peak_phase=0.17,     # 10 AM
                trough_phase=0.67,   # 10 PM
                amplitude=0.15,
                baseline=0.5,
                waveform="cosine"
            ),

            # Dopamine: peaks in afternoon
            HormoneType.DOPAMINE: HormoneRhythm(
                hormone_type=HormoneType.DOPAMINE,
                peak_phase=0.29,     # 1 PM
                trough_phase=0.79,   # 1 AM
                amplitude=0.2,
                baseline=0.45,
                waveform="cosine"
            ),

            # Serotonin: higher during day
            HormoneType.SEROTONIN: HormoneRhythm(
                hormone_type=HormoneType.SEROTONIN,
                peak_phase=0.25,     # 12 PM
                trough_phase=0.75,   # 12 AM
                amplitude=0.25,
                baseline=0.5,
                waveform="cosine"
            ),

            # Adrenaline: peaks in morning
            HormoneType.ADRENALINE: HormoneRhythm(
                hormone_type=HormoneType.ADRENALINE,
                peak_phase=0.08,     # 8 AM
                trough_phase=0.58,   # 8 PM
                amplitude=0.2,
                baseline=0.25,
                waveform="cosine"
            ),

            # Insulin sensitivity rhythm
            HormoneType.INSULIN: HormoneRhythm(
                hormone_type=HormoneType.INSULIN,
                peak_phase=0.17,     # 10 AM (peak sensitivity)
                trough_phase=0.75,   # 12 AM
                amplitude=0.15,
                baseline=0.5,
                waveform="cosine"
            ),

            # Endorphin: peaks in late afternoon
            HormoneType.ENDORPHIN: HormoneRhythm(
                hormone_type=HormoneType.ENDORPHIN,
                peak_phase=0.42,     # 4 PM
                trough_phase=0.92,   # 4 AM
                amplitude=0.15,
                baseline=0.35,
                waveform="cosine"
            )
        }

    def tick(self):
        """Advance the circadian clock by one cycle"""
        self.cycle_count += 1
        old_phase = self.master_clock.phase

        # Advance phase
        self.master_clock.phase += self.cycle_speed * self.master_clock.period

        # Check for new day
        if self.master_clock.phase >= 1.0:
            self.master_clock.phase -= 1.0
            self.days_elapsed += 1
            for callback in self._callbacks["new_day"]:
                callback(self.days_elapsed)

        # Record phase
        self.phase_history.append(self.master_clock.phase)
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)

        # Update sleep pressure
        self._update_sleep_pressure()

        # Update sleep state
        self._update_sleep_state()

        # Trigger phase change callback
        if self._get_phase_name(old_phase) != self._get_phase_name(self.master_clock.phase):
            for callback in self._callbacks["phase_changed"]:
                callback(self.get_current_phase())

    def _update_sleep_pressure(self):
        """Update homeostatic sleep pressure"""
        if self.sleep_state == SleepState.AWAKE:
            self.awake_time += self.cycle_speed
            # Sleep pressure builds exponentially while awake
            self.sleep_pressure += 0.02 * self.cycle_speed * (1.0 + self.sleep_pressure)
            self.sleep_pressure = min(1.0, self.sleep_pressure)

            if self.sleep_pressure > 0.8:
                for callback in self._callbacks["high_sleep_pressure"]:
                    callback(self.sleep_pressure)
        else:
            # Sleep pressure dissipates during sleep
            self.sleep_pressure -= 0.03 * self.cycle_speed
            self.sleep_pressure = max(0.0, self.sleep_pressure)
            self.awake_time = 0.0

    def _update_sleep_state(self):
        """Update sleep-wake state based on circadian phase and pressure"""
        old_state = self.sleep_state
        phase = self.master_clock.phase
        melatonin = self.get_hormone_modifier(HormoneType.MELATONIN)

        # Calculate sleep drive
        sleep_drive = (
            self.sleep_pressure * 0.5 +
            melatonin * 0.3 +
            (1.0 if 0.65 < phase < 0.95 else 0.0) * 0.2
        )

        if self.sleep_state == SleepState.AWAKE:
            if sleep_drive > 0.7:
                self.sleep_state = SleepState.DROWSY
            elif self.sleep_pressure > 0.9:
                self.sleep_state = SleepState.DROWSY
        elif self.sleep_state == SleepState.DROWSY:
            if sleep_drive > 0.8:
                self.sleep_state = SleepState.LIGHT_SLEEP
            elif sleep_drive < 0.5:
                self.sleep_state = SleepState.AWAKE
        elif self.sleep_state == SleepState.LIGHT_SLEEP:
            if sleep_drive > 0.9 and 0.7 < phase < 0.9:
                self.sleep_state = SleepState.DEEP_SLEEP
            elif sleep_drive < 0.6:
                self.sleep_state = SleepState.LIGHT_SLEEP
            elif 0.85 < phase or phase < 0.1:
                self.sleep_state = SleepState.REM
        elif self.sleep_state == SleepState.DEEP_SLEEP:
            if self.sleep_pressure < 0.3:
                self.sleep_state = SleepState.REM
        elif self.sleep_state == SleepState.REM:
            if self.sleep_pressure < 0.1 and phase > 0.9 or phase < 0.2:
                self.sleep_state = SleepState.AWAKE
            elif self.sleep_pressure > 0.4:
                self.sleep_state = SleepState.LIGHT_SLEEP

        if old_state != self.sleep_state:
            for callback in self._callbacks["sleep_state_changed"]:
                callback(self.sleep_state)

    def _get_phase_name(self, phase: float) -> CircadianPhase:
        """Convert phase to named phase"""
        if phase < 0.125:
            return CircadianPhase.EARLY_MORNING
        elif phase < 0.25:
            return CircadianPhase.LATE_MORNING
        elif phase < 0.375:
            return CircadianPhase.EARLY_AFTERNOON
        elif phase < 0.5:
            return CircadianPhase.LATE_AFTERNOON
        elif phase < 0.625:
            return CircadianPhase.EARLY_EVENING
        elif phase < 0.75:
            return CircadianPhase.LATE_EVENING
        elif phase < 0.875:
            return CircadianPhase.EARLY_NIGHT
        else:
            return CircadianPhase.LATE_NIGHT

    def get_current_phase(self) -> CircadianPhase:
        """Get current circadian phase"""
        return self._get_phase_name(self.master_clock.phase)

    def get_hormone_modifier(self, hormone_type: HormoneType) -> float:
        """Get circadian modifier for a hormone (0-1)"""
        if hormone_type not in self.hormone_rhythms:
            return 0.5  # No rhythm, neutral modifier

        rhythm = self.hormone_rhythms[hormone_type]
        phase = self.master_clock.phase

        # Calculate phase relative to peak
        relative_phase = phase - rhythm.peak_phase
        if relative_phase < 0:
            relative_phase += 1.0

        # Generate waveform
        if rhythm.waveform == "cosine":
            value = rhythm.baseline + rhythm.amplitude * math.cos(2 * math.pi * relative_phase)
        elif rhythm.waveform == "square":
            value = rhythm.baseline + rhythm.amplitude * (1 if relative_phase < 0.5 else -1)
        elif rhythm.waveform == "asymmetric":
            # Asymmetric waveform for hormones with fast rise/slow decay
            if rhythm.asymmetry >= 0:
                # Fast rise, slow fall
                if relative_phase < 0.25:
                    t = relative_phase / 0.25
                    value = rhythm.baseline + rhythm.amplitude * t
                else:
                    t = (relative_phase - 0.25) / 0.75
                    value = rhythm.baseline + rhythm.amplitude * (1 - t)
            else:
                # Slow rise, fast fall
                if relative_phase < 0.75:
                    t = relative_phase / 0.75
                    value = rhythm.baseline + rhythm.amplitude * t
                else:
                    t = (relative_phase - 0.75) / 0.25
                    value = rhythm.baseline + rhythm.amplitude * (1 - t)
        else:
            value = rhythm.baseline

        return max(0.0, min(1.0, value))

    def get_hormone_modifiers(self) -> Dict[HormoneType, float]:
        """Get circadian modifiers for all hormones"""
        return {
            h_type: self.get_hormone_modifier(h_type)
            for h_type in self.hormone_rhythms.keys()
        }

    def apply_zeitgeber(
        self,
        source: str,
        intensity: float,
        direction: str = "advance"
    ):
        """Apply a time-giving signal to entrain the clock"""
        event = ZeitgeberEvent(
            source=source,
            intensity=intensity,
            phase=self.master_clock.phase,
            direction=direction
        )
        self.zeitgeber_history.append(event)
        if len(self.zeitgeber_history) > self.max_history:
            self.zeitgeber_history.pop(0)

        # Calculate phase shift based on PRC (Phase Response Curve)
        shift = self._calculate_phase_shift(intensity, direction)
        self.master_clock.phase += shift
        self.master_clock.phase = self.master_clock.phase % 1.0

    def _calculate_phase_shift(
        self,
        intensity: float,
        direction: str
    ) -> float:
        """Calculate phase shift using simplified PRC"""
        phase = self.master_clock.phase
        rate = self.master_clock.entrainment_rate

        # PRC varies by current phase
        # Light in early night -> delays
        # Light in late night -> advances
        if direction == "advance":
            if 0.7 < phase < 0.95:
                shift = intensity * rate * 0.5
            elif phase > 0.95 or phase < 0.1:
                shift = intensity * rate
            else:
                shift = intensity * rate * 0.2
        else:  # delay
            if 0.6 < phase < 0.8:
                shift = -intensity * rate
            elif 0.8 < phase < 0.95:
                shift = -intensity * rate * 0.5
            else:
                shift = -intensity * rate * 0.2

        return shift

    def force_sleep(self):
        """Force transition to sleep state"""
        if self.sleep_state == SleepState.AWAKE:
            self.sleep_state = SleepState.DROWSY
        elif self.sleep_state == SleepState.DROWSY:
            self.sleep_state = SleepState.LIGHT_SLEEP

    def force_wake(self):
        """Force transition to awake state"""
        self.sleep_state = SleepState.AWAKE
        # Waking at wrong time increases sleep debt
        if 0.7 < self.master_clock.phase < 0.95:
            self.sleep_debt += 0.1

    def get_alertness(self) -> float:
        """Get current alertness level (0-1)"""
        if self.sleep_state != SleepState.AWAKE:
            return 0.0

        # Alertness is inverse of sleep pressure, modulated by circadian phase
        cortisol = self.get_hormone_modifier(HormoneType.CORTISOL)
        circadian_alertness = cortisol * 0.5 + 0.5

        alertness = circadian_alertness * (1.0 - self.sleep_pressure * 0.7)
        return max(0.0, min(1.0, alertness))

    def get_sleep_pressure(self) -> float:
        """Get current sleep pressure"""
        return self.sleep_pressure

    def on_phase_changed(self, callback: Callable):
        """Register callback for phase changes"""
        self._callbacks["phase_changed"].append(callback)

    def on_sleep_state_changed(self, callback: Callable):
        """Register callback for sleep state changes"""
        self._callbacks["sleep_state_changed"].append(callback)

    def on_new_day(self, callback: Callable):
        """Register callback for new day"""
        self._callbacks["new_day"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get circadian system status"""
        return {
            "phase": round(self.master_clock.phase, 4),
            "phase_name": self.get_current_phase().value,
            "cycle_count": self.cycle_count,
            "days_elapsed": self.days_elapsed,
            "chronotype": self.chronotype.value,
            "sleep_state": self.sleep_state.value,
            "sleep_pressure": round(self.sleep_pressure, 3),
            "sleep_debt": round(self.sleep_debt, 3),
            "alertness": round(self.get_alertness(), 3),
            "hormone_modifiers": {
                h.value: round(self.get_hormone_modifier(h), 3)
                for h in self.hormone_rhythms.keys()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Circadian Rhythm System Demo")
    print("=" * 50)

    # Create circadian system
    circadian = CircadianSystem(
        chronotype=ChronotypeBias.INTERMEDIATE,
        initial_phase=0.0,  # Start at 6 AM
        cycle_speed=0.02    # Fast simulation
    )

    print(f"\n1. Initial state:")
    print(f"   Phase: {circadian.master_clock.phase:.2f} ({circadian.get_current_phase().value})")
    print(f"   Sleep state: {circadian.sleep_state.value}")

    # Simulate a day
    print("\n2. Simulating 24-hour cycle:")
    phases_seen = set()

    for i in range(100):
        circadian.tick()

        phase_name = circadian.get_current_phase().value
        if phase_name not in phases_seen:
            phases_seen.add(phase_name)
            status = circadian.get_status()
            print(f"   {phase_name}:")
            print(f"      Alertness: {status['alertness']:.2f}")
            print(f"      Cortisol: {status['hormone_modifiers']['cortisol']:.2f}")
            print(f"      Melatonin: {status['hormone_modifiers']['melatonin']:.2f}")

    # Check hormone rhythms
    print("\n3. Hormone modifiers at current phase:")
    modifiers = circadian.get_hormone_modifiers()
    for hormone, value in sorted(modifiers.items(), key=lambda x: -x[1]):
        print(f"   {hormone.value}: {value:.3f}")

    # Simulate sleep
    print("\n4. Advancing to night (sleep):")
    while circadian.master_clock.phase < 0.75:
        circadian.tick()

    print(f"   Phase: {circadian.master_clock.phase:.2f}")
    print(f"   Sleep pressure: {circadian.sleep_pressure:.2f}")
    print(f"   Sleep state: {circadian.sleep_state.value}")

    # Continue through night
    for i in range(30):
        circadian.tick()

    print(f"\n5. During night:")
    print(f"   Phase: {circadian.master_clock.phase:.2f}")
    print(f"   Sleep state: {circadian.sleep_state.value}")
    print(f"   Sleep pressure: {circadian.sleep_pressure:.2f}")

    # Apply light zeitgeber
    print("\n6. Applying morning light zeitgeber:")
    circadian.apply_zeitgeber("light", 0.8, "advance")
    print(f"   Phase after light: {circadian.master_clock.phase:.2f}")

    # Final status
    print("\n7. Final status:")
    status = circadian.get_status()
    print(f"   Days elapsed: {status['days_elapsed']}")
    print(f"   Current phase: {status['phase_name']}")
    print(f"   Alertness: {status['alertness']:.2f}")
