"""
Neuroendocrine Interface System

Implements the neural-endocrine integration for digital organisms including:
- Hypothalamus-pituitary communication
- Neural control of hormone release
- Sensory-endocrine integration
- Emotional-hormonal coupling
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
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

class NeuralSignalType(Enum):
    """Types of neural signals to endocrine system"""
    EXCITATORY = "excitatory"       # Increases hormone release
    INHIBITORY = "inhibitory"       # Decreases hormone release
    MODULATORY = "modulatory"       # Modifies sensitivity
    PULSATILE = "pulsatile"         # Rhythmic release pattern


class ReleasingHormone(Enum):
    """Hypothalamic releasing hormones"""
    CRH = "crh"       # Corticotropin-releasing (stress)
    TRH = "trh"       # Thyrotropin-releasing (metabolism)
    GHRH = "ghrh"     # Growth hormone-releasing
    GnRH = "gnrh"     # Gonadotropin-releasing (not used)
    DOPAMINE = "da"   # Dopamine (inhibits prolactin)
    SOMATOSTATIN = "sst"  # Inhibits GH, TSH


class EmotionalState(Enum):
    """Emotional states that affect neuroendocrine output"""
    CALM = "calm"
    ALERT = "alert"
    ANXIOUS = "anxious"
    FEARFUL = "fearful"
    ANGRY = "angry"
    JOYFUL = "joyful"
    SAD = "sad"
    EXCITED = "excited"


class SensoryInput(Enum):
    """Sensory inputs that trigger neuroendocrine responses"""
    THREAT = "threat"               # Danger detection
    FOOD = "food"                   # Food-related
    SOCIAL = "social"               # Social signals
    TEMPERATURE = "temperature"     # Temperature changes
    PAIN = "pain"                   # Pain signals
    LIGHT = "light"                 # Light/dark
    NOVELTY = "novelty"             # New stimuli


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class NeuralSignal:
    """A neural signal to the endocrine system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: NeuralSignalType = NeuralSignalType.EXCITATORY
    source: str = ""                # Neural source
    target_gland: GlandType = GlandType.HYPOTHALAMUS
    intensity: float = 0.5
    duration: float = 5.0           # Signal duration in cycles
    timestamp: float = field(default_factory=time.time)


@dataclass
class HypothalamicOutput:
    """Output from the hypothalamus"""
    releasing_hormone: ReleasingHormone = ReleasingHormone.CRH
    level: float = 0.0              # Current release level
    pulsatile: bool = False         # Whether pulsatile release
    pulse_frequency: float = 0.0    # Pulses per cycle
    pulse_amplitude: float = 0.0    # Pulse height


@dataclass
class NeuroendocrineEvent:
    """Record of a neuroendocrine event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = ""               # What triggered it
    emotional_state: EmotionalState = EmotionalState.CALM
    sensory_input: Optional[SensoryInput] = None
    hormone_changes: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Neuroendocrine System
# ============================================================================

class NeuroendocrineSystem:
    """
    Complete neural-endocrine integration system.

    Features:
    - Hypothalamic releasing hormone control
    - Emotional-endocrine coupling
    - Sensory-triggered responses
    - Pulsatile hormone release patterns
    - Neural modulation of sensitivity

    Example:
        neuro = NeuroendocrineSystem()

        # Process emotional state
        neuro.set_emotional_state(EmotionalState.ANXIOUS)

        # Process sensory input
        neuro.process_sensory_input(SensoryInput.THREAT, 0.7)

        # Get hormone release signals
        effects = neuro.get_hormone_effects()

        # Update system
        neuro.tick()
    """

    def __init__(
        self,
        base_tone: float = 0.3,
        emotional_sensitivity: float = 1.0
    ):
        self.base_tone = base_tone
        self.emotional_sensitivity = emotional_sensitivity

        # Current emotional state
        self.emotional_state = EmotionalState.CALM
        self.emotional_intensity: float = 0.0

        # Hypothalamic outputs
        self.hypothalamic_outputs: Dict[ReleasingHormone, HypothalamicOutput] = \
            self._initialize_outputs()

        # Active neural signals
        self.active_signals: List[NeuralSignal] = []

        # Sensitivity modulation
        self.gland_sensitivity: Dict[GlandType, float] = {
            gland: 1.0 for gland in GlandType
        }

        # Event history
        self.event_history: List[NeuroendocrineEvent] = []
        self.max_history = 200

        # Hormone effects to apply
        self._hormone_effects: Dict[HormoneType, float] = {}

        # Pulsatile release state
        self._pulse_phase: Dict[ReleasingHormone, float] = {
            rh: 0.0 for rh in ReleasingHormone
        }

        # Cycle tracking
        self.cycle_count = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "emotional_change": [],
            "sensory_trigger": [],
            "hormone_pulse": [],
            "stress_activation": []
        }

    def _initialize_outputs(self) -> Dict[ReleasingHormone, HypothalamicOutput]:
        """Initialize hypothalamic releasing hormone outputs"""
        return {
            ReleasingHormone.CRH: HypothalamicOutput(
                releasing_hormone=ReleasingHormone.CRH,
                level=self.base_tone,
                pulsatile=True,
                pulse_frequency=0.05,
                pulse_amplitude=0.3
            ),
            ReleasingHormone.TRH: HypothalamicOutput(
                releasing_hormone=ReleasingHormone.TRH,
                level=self.base_tone,
                pulsatile=False
            ),
            ReleasingHormone.GHRH: HypothalamicOutput(
                releasing_hormone=ReleasingHormone.GHRH,
                level=self.base_tone * 0.8,
                pulsatile=True,
                pulse_frequency=0.03,
                pulse_amplitude=0.4
            ),
            ReleasingHormone.GnRH: HypothalamicOutput(
                releasing_hormone=ReleasingHormone.GnRH,
                level=0.0,
                pulsatile=True,
                pulse_frequency=0.02,
                pulse_amplitude=0.5
            ),
            ReleasingHormone.DOPAMINE: HypothalamicOutput(
                releasing_hormone=ReleasingHormone.DOPAMINE,
                level=self.base_tone,
                pulsatile=False
            ),
            ReleasingHormone.SOMATOSTATIN: HypothalamicOutput(
                releasing_hormone=ReleasingHormone.SOMATOSTATIN,
                level=self.base_tone * 0.5,
                pulsatile=False
            )
        }

    def set_emotional_state(
        self,
        state: EmotionalState,
        intensity: float = 0.5
    ):
        """Set current emotional state"""
        old_state = self.emotional_state
        self.emotional_state = state
        self.emotional_intensity = intensity * self.emotional_sensitivity

        # Map emotions to hormone effects
        self._process_emotional_state()

        if old_state != state:
            for callback in self._callbacks["emotional_change"]:
                callback(state, intensity)

    def _process_emotional_state(self):
        """Process emotional state effects on hormones"""
        intensity = self.emotional_intensity

        # Reset hypothalamic outputs to base
        for output in self.hypothalamic_outputs.values():
            output.level = self.base_tone

        if self.emotional_state == EmotionalState.CALM:
            # Baseline state
            pass

        elif self.emotional_state == EmotionalState.ALERT:
            # Mild activation
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = self.base_tone + intensity * 0.2

        elif self.emotional_state == EmotionalState.ANXIOUS:
            # Stress axis activation
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = self.base_tone + intensity * 0.5
            self.hypothalamic_outputs[ReleasingHormone.SOMATOSTATIN].level = self.base_tone + intensity * 0.2

        elif self.emotional_state == EmotionalState.FEARFUL:
            # Strong stress response
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = self.base_tone + intensity * 0.8
            self.hypothalamic_outputs[ReleasingHormone.TRH].level = self.base_tone + intensity * 0.3

            for callback in self._callbacks["stress_activation"]:
                callback(intensity)

        elif self.emotional_state == EmotionalState.ANGRY:
            # Activation with different profile
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = self.base_tone + intensity * 0.6
            self.hypothalamic_outputs[ReleasingHormone.DOPAMINE].level = self.base_tone + intensity * 0.3

        elif self.emotional_state == EmotionalState.JOYFUL:
            # Positive state
            self.hypothalamic_outputs[ReleasingHormone.DOPAMINE].level = self.base_tone + intensity * 0.4
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = max(0.1, self.base_tone - intensity * 0.2)

        elif self.emotional_state == EmotionalState.SAD:
            # Depressed state
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = self.base_tone + intensity * 0.3
            self.hypothalamic_outputs[ReleasingHormone.DOPAMINE].level = max(0.1, self.base_tone - intensity * 0.3)

        elif self.emotional_state == EmotionalState.EXCITED:
            # Arousal
            self.hypothalamic_outputs[ReleasingHormone.CRH].level = self.base_tone + intensity * 0.3
            self.hypothalamic_outputs[ReleasingHormone.DOPAMINE].level = self.base_tone + intensity * 0.4
            self.hypothalamic_outputs[ReleasingHormone.TRH].level = self.base_tone + intensity * 0.2

    def process_sensory_input(
        self,
        input_type: SensoryInput,
        intensity: float
    ):
        """Process sensory input and generate neuroendocrine response"""
        # Create neural signal
        signal = self._create_signal_from_sensory(input_type, intensity)
        if signal:
            self.active_signals.append(signal)

        # Map sensory inputs to hormone responses
        self._process_sensory_effects(input_type, intensity)

        # Record event
        event = NeuroendocrineEvent(
            trigger=f"sensory_{input_type.value}",
            emotional_state=self.emotional_state,
            sensory_input=input_type
        )
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        for callback in self._callbacks["sensory_trigger"]:
            callback(input_type, intensity)

    def _create_signal_from_sensory(
        self,
        input_type: SensoryInput,
        intensity: float
    ) -> Optional[NeuralSignal]:
        """Create neural signal from sensory input"""
        signal_configs = {
            SensoryInput.THREAT: {
                "signal_type": NeuralSignalType.EXCITATORY,
                "target_gland": GlandType.ADRENAL,
                "duration": 10.0
            },
            SensoryInput.FOOD: {
                "signal_type": NeuralSignalType.EXCITATORY,
                "target_gland": GlandType.PITUITARY,
                "duration": 15.0
            },
            SensoryInput.SOCIAL: {
                "signal_type": NeuralSignalType.MODULATORY,
                "target_gland": GlandType.HYPOTHALAMUS,
                "duration": 20.0
            },
            SensoryInput.PAIN: {
                "signal_type": NeuralSignalType.EXCITATORY,
                "target_gland": GlandType.ADRENAL,
                "duration": 8.0
            },
            SensoryInput.LIGHT: {
                "signal_type": NeuralSignalType.MODULATORY,
                "target_gland": GlandType.HYPOTHALAMUS,
                "duration": 30.0
            }
        }

        if input_type not in signal_configs:
            return None

        config = signal_configs[input_type]
        return NeuralSignal(
            signal_type=config["signal_type"],
            source=f"sensory_{input_type.value}",
            target_gland=config["target_gland"],
            intensity=intensity,
            duration=config["duration"]
        )

    def _process_sensory_effects(self, input_type: SensoryInput, intensity: float):
        """Process sensory input effects on hormones"""
        if input_type == SensoryInput.THREAT:
            # Immediate stress response
            self.hypothalamic_outputs[ReleasingHormone.CRH].level += intensity * 0.5
            self.gland_sensitivity[GlandType.ADRENAL] *= (1.0 + intensity * 0.3)

        elif input_type == SensoryInput.FOOD:
            # Metabolic preparation
            self.hypothalamic_outputs[ReleasingHormone.TRH].level += intensity * 0.2
            self.hypothalamic_outputs[ReleasingHormone.SOMATOSTATIN].level -= intensity * 0.1

        elif input_type == SensoryInput.SOCIAL:
            # Social hormone modulation (hypothalamus produces oxytocin)
            self.gland_sensitivity[GlandType.HYPOTHALAMUS] *= (1.0 + intensity * 0.2)

        elif input_type == SensoryInput.PAIN:
            # Pain response
            self.hypothalamic_outputs[ReleasingHormone.CRH].level += intensity * 0.4
            # Trigger endorphin release via pituitary
            self.hypothalamic_outputs[ReleasingHormone.GHRH].level += intensity * 0.2

        elif input_type == SensoryInput.LIGHT:
            # Circadian effects
            self.hypothalamic_outputs[ReleasingHormone.TRH].level += intensity * 0.1
            self.hypothalamic_outputs[ReleasingHormone.CRH].level += intensity * 0.1

        elif input_type == SensoryInput.NOVELTY:
            # Novelty response
            self.hypothalamic_outputs[ReleasingHormone.DOPAMINE].level += intensity * 0.3
            self.hypothalamic_outputs[ReleasingHormone.CRH].level += intensity * 0.1

    def add_neural_signal(self, signal: NeuralSignal):
        """Add a direct neural signal"""
        self.active_signals.append(signal)

    def tick(self):
        """Process one neuroendocrine cycle"""
        self.cycle_count += 1

        # Process pulsatile release
        self._process_pulsatile_release()

        # Decay active signals
        self._decay_signals()

        # Apply neural signals to glands
        self._apply_neural_signals()

        # Decay sensitivity modulations
        self._decay_sensitivity()

        # Generate hormone effects
        self._generate_hormone_effects()

        # Decay hypothalamic outputs toward baseline
        self._decay_outputs()

    def _process_pulsatile_release(self):
        """Process pulsatile hormone release patterns"""
        for rh, output in self.hypothalamic_outputs.items():
            if not output.pulsatile:
                continue

            # Advance phase
            self._pulse_phase[rh] += output.pulse_frequency
            if self._pulse_phase[rh] >= 1.0:
                self._pulse_phase[rh] = 0.0

                # Generate pulse
                pulse_output = output.level + output.pulse_amplitude
                output.level = min(1.0, pulse_output)

                for callback in self._callbacks["hormone_pulse"]:
                    callback(rh, pulse_output)

    def _decay_signals(self):
        """Decay and remove old neural signals"""
        current_time = time.time()
        active = []

        for signal in self.active_signals:
            age = current_time - signal.timestamp
            if age < signal.duration:
                active.append(signal)

        self.active_signals = active

    def _apply_neural_signals(self):
        """Apply active neural signals to glands"""
        for signal in self.active_signals:
            gland = signal.target_gland

            if signal.signal_type == NeuralSignalType.EXCITATORY:
                self.gland_sensitivity[gland] *= (1.0 + signal.intensity * 0.1)
            elif signal.signal_type == NeuralSignalType.INHIBITORY:
                self.gland_sensitivity[gland] *= (1.0 - signal.intensity * 0.1)
            elif signal.signal_type == NeuralSignalType.MODULATORY:
                # More subtle modulation
                self.gland_sensitivity[gland] *= (1.0 + signal.intensity * 0.05)

    def _decay_sensitivity(self):
        """Decay sensitivity modulations back to baseline"""
        for gland in self.gland_sensitivity:
            diff = 1.0 - self.gland_sensitivity[gland]
            self.gland_sensitivity[gland] += diff * 0.1

    def _decay_outputs(self):
        """Decay hypothalamic outputs toward baseline"""
        for output in self.hypothalamic_outputs.values():
            diff = self.base_tone - output.level
            output.level += diff * 0.05
            output.level = max(0.0, min(1.0, output.level))

    def _generate_hormone_effects(self):
        """Generate hormone release effects"""
        self._hormone_effects = {}

        # CRH -> Cortisol (via ACTH)
        crh = self.hypothalamic_outputs[ReleasingHormone.CRH].level
        adrenal_sens = self.gland_sensitivity[GlandType.ADRENAL]
        self._hormone_effects[HormoneType.CORTISOL] = (crh - self.base_tone) * adrenal_sens

        # Also adrenaline if high CRH
        if crh > 0.5:
            self._hormone_effects[HormoneType.ADRENALINE] = (crh - 0.5) * adrenal_sens * 0.8

        # TRH -> Thyroxine (via TSH)
        trh = self.hypothalamic_outputs[ReleasingHormone.TRH].level
        thyroid_sens = self.gland_sensitivity[GlandType.THYROID]
        self._hormone_effects[HormoneType.THYROXINE] = (trh - self.base_tone) * thyroid_sens * 0.5

        # GHRH -> Growth Hormone
        ghrh = self.hypothalamic_outputs[ReleasingHormone.GHRH].level
        sst = self.hypothalamic_outputs[ReleasingHormone.SOMATOSTATIN].level
        pituitary_sens = self.gland_sensitivity[GlandType.PITUITARY]
        gh_effect = (ghrh - sst) * pituitary_sens
        self._hormone_effects[HormoneType.GROWTH_HORMONE] = gh_effect

        # GHRH also triggers endorphin release
        if ghrh > self.base_tone:
            self._hormone_effects[HormoneType.ENDORPHIN] = (ghrh - self.base_tone) * 0.3

        # Dopamine effects
        da = self.hypothalamic_outputs[ReleasingHormone.DOPAMINE].level
        reward_sens = self.gland_sensitivity[GlandType.REWARD_CENTER]
        self._hormone_effects[HormoneType.DOPAMINE] = (da - self.base_tone) * reward_sens

        # Hypothalamus also controls oxytocin/vasopressin
        hypo_sens = self.gland_sensitivity[GlandType.HYPOTHALAMUS]
        if hypo_sens > 1.1:  # Social activation
            self._hormone_effects[HormoneType.OXYTOCIN] = (hypo_sens - 1.0) * 0.5
            self._hormone_effects[HormoneType.VASOPRESSIN] = (hypo_sens - 1.0) * 0.3

    def get_hormone_effects(self) -> Dict[HormoneType, float]:
        """Get hormone effects from neuroendocrine system"""
        return self._hormone_effects.copy()

    def get_releasing_hormone_level(self, rh: ReleasingHormone) -> float:
        """Get current level of a releasing hormone"""
        if rh in self.hypothalamic_outputs:
            return self.hypothalamic_outputs[rh].level
        return 0.0

    def get_gland_sensitivity(self, gland: GlandType) -> float:
        """Get current sensitivity of a gland"""
        return self.gland_sensitivity.get(gland, 1.0)

    def modulate_gland(self, gland: GlandType, factor: float):
        """Externally modulate gland sensitivity"""
        self.gland_sensitivity[gland] *= factor
        self.gland_sensitivity[gland] = max(0.1, min(3.0, self.gland_sensitivity[gland]))

    def on_emotional_change(self, callback: Callable):
        """Register callback for emotional state changes"""
        self._callbacks["emotional_change"].append(callback)

    def on_sensory_trigger(self, callback: Callable):
        """Register callback for sensory triggers"""
        self._callbacks["sensory_trigger"].append(callback)

    def on_stress_activation(self, callback: Callable):
        """Register callback for stress axis activation"""
        self._callbacks["stress_activation"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get neuroendocrine system status"""
        return {
            "emotional_state": self.emotional_state.value,
            "emotional_intensity": round(self.emotional_intensity, 3),
            "active_signals": len(self.active_signals),
            "cycle_count": self.cycle_count,
            "hypothalamic_outputs": {
                rh.value: round(output.level, 3)
                for rh, output in self.hypothalamic_outputs.items()
            },
            "gland_sensitivity": {
                g.value: round(s, 3)
                for g, s in self.gland_sensitivity.items()
            },
            "hormone_effects": {
                h.value: round(v, 3)
                for h, v in self._hormone_effects.items()
                if v != 0
            },
            "recent_events": len(self.event_history)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Neuroendocrine System Demo")
    print("=" * 50)

    # Create system
    neuro = NeuroendocrineSystem()

    print(f"\n1. Initial state:")
    status = neuro.get_status()
    print(f"   Emotional state: {status['emotional_state']}")
    print(f"   CRH level: {status['hypothalamic_outputs']['crh']:.2f}")

    # Set anxious state
    print("\n2. Setting anxious emotional state...")
    neuro.set_emotional_state(EmotionalState.ANXIOUS, 0.7)
    neuro.tick()

    status = neuro.get_status()
    print(f"   CRH level: {status['hypothalamic_outputs']['crh']:.2f}")
    print(f"   Hormone effects: {status['hormone_effects']}")

    # Process threat sensory input
    print("\n3. Processing threat sensory input...")
    neuro.process_sensory_input(SensoryInput.THREAT, 0.8)
    neuro.tick()

    status = neuro.get_status()
    print(f"   CRH level: {status['hypothalamic_outputs']['crh']:.2f}")
    print(f"   Adrenal sensitivity: {status['gland_sensitivity']['adrenal']:.2f}")
    print(f"   Hormone effects: {status['hormone_effects']}")

    # Calm down over time
    print("\n4. Calming down (20 cycles)...")
    neuro.set_emotional_state(EmotionalState.CALM, 0.0)

    for i in range(20):
        neuro.tick()

        if i % 5 == 0:
            crh = neuro.get_releasing_hormone_level(ReleasingHormone.CRH)
            print(f"   Cycle {i}: CRH = {crh:.3f}")

    # Joyful state
    print("\n5. Setting joyful state...")
    neuro.set_emotional_state(EmotionalState.JOYFUL, 0.8)
    neuro.tick()

    status = neuro.get_status()
    print(f"   Dopamine output: {status['hypothalamic_outputs']['da']:.2f}")
    print(f"   Hormone effects: {status['hormone_effects']}")

    # Social interaction
    print("\n6. Processing social sensory input...")
    neuro.process_sensory_input(SensoryInput.SOCIAL, 0.6)
    neuro.tick()

    status = neuro.get_status()
    print(f"   Hypothalamus sensitivity: {status['gland_sensitivity']['hypothalamus']:.2f}")
    print(f"   Hormone effects: {status['hormone_effects']}")

    # Pulsatile release
    print("\n7. Observing pulsatile release (30 cycles)...")
    for i in range(30):
        neuro.tick()

        if i % 10 == 0:
            ghrh = neuro.get_releasing_hormone_level(ReleasingHormone.GHRH)
            print(f"   Cycle {i}: GHRH = {ghrh:.3f}")

    # Final status
    print("\n8. Final status:")
    status = neuro.get_status()
    print(f"   Emotional state: {status['emotional_state']}")
    print(f"   Total events recorded: {status['recent_events']}")
    print(f"   Hypothalamic outputs: {status['hypothalamic_outputs']}")
