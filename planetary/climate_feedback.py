"""
Climate-Consciousness Feedback Loops

Implements bidirectional feedback between planetary climate systems
and collective consciousness. Climate affects consciousness emergence,
and consciousness can influence climate through coordinated action.

Features:
- Climate impact on consciousness (stress, awareness, behavior)
- Consciousness-driven climate adaptation
- Feedback loop detection and amplification
- Tipping point awareness
- Collective climate response coordination
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if TYPE_CHECKING:
    from planetary.biosphere import Biosphere
    from planetary.gaia_consciousness import GaiaConsciousness
    from planetary.resource_cycles import PlanetaryCycles


# ============================================================================
# Enums
# ============================================================================

class FeedbackType(Enum):
    """Types of climate-consciousness feedback"""
    POSITIVE = "positive"         # Amplifying
    NEGATIVE = "negative"         # Stabilizing
    NEUTRAL = "neutral"           # No effect
    NONLINEAR = "nonlinear"       # Threshold-dependent


class FeedbackDirection(Enum):
    """Direction of feedback effect"""
    CLIMATE_TO_CONSCIOUSNESS = "climate_to_consciousness"
    CONSCIOUSNESS_TO_CLIMATE = "consciousness_to_climate"
    BIDIRECTIONAL = "bidirectional"


class ClimateStressor(Enum):
    """Types of climate stressors affecting consciousness"""
    TEMPERATURE_EXTREME = "temperature_extreme"
    DROUGHT = "drought"
    FLOODING = "flooding"
    AIR_QUALITY = "air_quality"
    SEA_LEVEL_RISE = "sea_level_rise"
    BIODIVERSITY_LOSS = "biodiversity_loss"
    RESOURCE_SCARCITY = "resource_scarcity"


class ConsciousnessResponse(Enum):
    """Consciousness responses to climate"""
    ADAPTATION = "adaptation"           # Adjust to conditions
    MITIGATION = "mitigation"           # Reduce impacts
    MIGRATION = "migration"             # Move to better conditions
    INNOVATION = "innovation"           # Develop solutions
    COORDINATION = "coordination"       # Collective action
    DORMANCY = "dormancy"               # Reduce activity


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FeedbackLoop:
    """A climate-consciousness feedback loop"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    feedback_type: FeedbackType = FeedbackType.NEUTRAL
    direction: FeedbackDirection = FeedbackDirection.BIDIRECTIONAL
    strength: float = 0.5              # 0-1 feedback strength
    delay: int = 10                     # Ticks of delay
    threshold: float = 0.0             # Activation threshold
    active: bool = True
    description: str = ""

    # State
    input_history: List[float] = field(default_factory=list)
    output_history: List[float] = field(default_factory=list)


@dataclass
class ClimateState:
    """Current climate state for feedback processing"""
    temperature_anomaly: float = 0.0   # Deviation from baseline
    precipitation_anomaly: float = 0.0
    co2_level: float = 400.0
    sea_level: float = 0.0
    extreme_events: int = 0
    ecosystem_stress: float = 0.0
    resource_availability: float = 1.0


@dataclass
class ConsciousnessState:
    """Consciousness state for feedback processing"""
    awareness_level: float = 0.5
    stress_level: float = 0.0
    adaptation_capacity: float = 0.5
    collective_action: float = 0.0
    innovation_rate: float = 0.0
    migration_pressure: float = 0.0


@dataclass
class FeedbackConfig:
    """Configuration for feedback system"""
    enable_climate_to_consciousness: bool = True
    enable_consciousness_to_climate: bool = True
    feedback_delay: int = 5
    amplification_factor: float = 1.0
    damping_factor: float = 0.1
    threshold_sensitivity: float = 1.0


# ============================================================================
# Climate-Consciousness Feedback System
# ============================================================================

class ClimateFeedbackSystem:
    """
    Bidirectional feedback between climate and consciousness.

    Implements feedback loops where:
    1. Climate conditions affect consciousness emergence and behavior
    2. Collective consciousness can influence climate through action
    3. Feedback loops can be positive (amplifying) or negative (stabilizing)

    Example:
        feedback = ClimateFeedbackSystem()
        feedback.initialize_feedback_loops()

        # Connect to other systems
        feedback.set_biosphere(biosphere)
        feedback.set_consciousness(gaia)

        # Run feedback cycles
        for _ in range(100):
            feedback.process_tick()

        analysis = feedback.get_feedback_analysis()
    """

    def __init__(self, config: Optional[FeedbackConfig] = None):
        self.config = config or FeedbackConfig()

        # Connected systems
        self.biosphere: Optional['Biosphere'] = None
        self.consciousness: Optional['GaiaConsciousness'] = None
        self.cycles: Optional['PlanetaryCycles'] = None

        # Feedback loops
        self.feedback_loops: Dict[str, FeedbackLoop] = {}

        # Current states
        self.climate_state = ClimateState()
        self.consciousness_state = ConsciousnessState()

        # Effects accumulator
        self.climate_effects: Dict[str, float] = defaultdict(float)
        self.consciousness_effects: Dict[str, float] = defaultdict(float)

        # Tracking
        self.tick_count = 0
        self.active_stressors: List[ClimateStressor] = []
        self.active_responses: List[ConsciousnessResponse] = []

        # History
        self.feedback_history: List[Dict[str, Any]] = []

    def set_biosphere(self, biosphere: 'Biosphere'):
        """Connect to biosphere"""
        self.biosphere = biosphere

    def set_consciousness(self, consciousness: 'GaiaConsciousness'):
        """Connect to Gaia consciousness"""
        self.consciousness = consciousness

    def set_cycles(self, cycles: 'PlanetaryCycles'):
        """Connect to resource cycles"""
        self.cycles = cycles

    def initialize_feedback_loops(self):
        """Initialize default feedback loops"""
        loops = [
            # Climate -> Consciousness loops
            FeedbackLoop(
                name="temperature_stress",
                feedback_type=FeedbackType.NEGATIVE,
                direction=FeedbackDirection.CLIMATE_TO_CONSCIOUSNESS,
                strength=0.6,
                threshold=2.0,
                description="High temperatures reduce consciousness capacity"
            ),
            FeedbackLoop(
                name="resource_scarcity_awareness",
                feedback_type=FeedbackType.POSITIVE,
                direction=FeedbackDirection.CLIMATE_TO_CONSCIOUSNESS,
                strength=0.4,
                threshold=0.3,
                description="Resource scarcity increases collective awareness"
            ),
            FeedbackLoop(
                name="biodiversity_consciousness",
                feedback_type=FeedbackType.POSITIVE,
                direction=FeedbackDirection.CLIMATE_TO_CONSCIOUSNESS,
                strength=0.7,
                threshold=0.0,
                description="Biodiversity supports consciousness emergence"
            ),
            FeedbackLoop(
                name="extreme_event_awakening",
                feedback_type=FeedbackType.NONLINEAR,
                direction=FeedbackDirection.CLIMATE_TO_CONSCIOUSNESS,
                strength=0.8,
                threshold=3,
                description="Extreme events trigger consciousness awakening"
            ),

            # Consciousness -> Climate loops
            FeedbackLoop(
                name="conscious_mitigation",
                feedback_type=FeedbackType.NEGATIVE,
                direction=FeedbackDirection.CONSCIOUSNESS_TO_CLIMATE,
                strength=0.5,
                threshold=0.5,
                description="High consciousness enables climate mitigation"
            ),
            FeedbackLoop(
                name="collective_adaptation",
                feedback_type=FeedbackType.NEGATIVE,
                direction=FeedbackDirection.CONSCIOUSNESS_TO_CLIMATE,
                strength=0.4,
                threshold=0.4,
                description="Collective action reduces climate stress"
            ),
            FeedbackLoop(
                name="innovation_breakthrough",
                feedback_type=FeedbackType.NONLINEAR,
                direction=FeedbackDirection.CONSCIOUSNESS_TO_CLIMATE,
                strength=0.9,
                threshold=0.7,
                description="High consciousness enables breakthrough innovations"
            ),

            # Bidirectional loops
            FeedbackLoop(
                name="ecosystem_consciousness_coupling",
                feedback_type=FeedbackType.POSITIVE,
                direction=FeedbackDirection.BIDIRECTIONAL,
                strength=0.6,
                threshold=0.0,
                description="Healthy ecosystems and consciousness mutually reinforce"
            ),
            FeedbackLoop(
                name="crisis_coordination",
                feedback_type=FeedbackType.NONLINEAR,
                direction=FeedbackDirection.BIDIRECTIONAL,
                strength=0.7,
                threshold=0.6,
                description="Crisis triggers coordination which reduces crisis"
            )
        ]

        for loop in loops:
            self.feedback_loops[loop.id] = loop

    def process_tick(self):
        """Process one feedback cycle"""
        self.tick_count += 1

        # Update states from connected systems
        self._update_climate_state()
        self._update_consciousness_state()

        # Detect active stressors
        self._detect_stressors()

        # Process feedback loops
        for loop in self.feedback_loops.values():
            if loop.active:
                self._process_feedback_loop(loop)

        # Determine responses
        self._determine_responses()

        # Apply effects
        self._apply_effects()

        # Record history
        self._record_history()

    def _update_climate_state(self):
        """Update climate state from biosphere and cycles"""
        if self.biosphere:
            metrics = self.biosphere.get_global_metrics()
            self.climate_state.temperature_anomaly = metrics.get("temperature_anomaly", 0.0)
            self.climate_state.ecosystem_stress = 1.0 - metrics["ocean"]["health"]

            # Count stressed biomes as extreme events proxy
            stressed_biomes = len(self.biosphere.biomes) - metrics["healthy_biomes"]
            self.climate_state.extreme_events = stressed_biomes

        if self.cycles:
            self.climate_state.co2_level = self.cycles.get_atmospheric_co2()

    def _update_consciousness_state(self):
        """Update consciousness state from Gaia"""
        if self.consciousness:
            status = self.consciousness.get_status()
            self.consciousness_state.awareness_level = status["awareness_score"]
            self.consciousness_state.collective_action = len(status.get("active_actions", [])) * 0.1

            # Derive other states
            if status["awareness_score"] > 0.5:
                self.consciousness_state.adaptation_capacity = status["awareness_score"] * 0.8
                self.consciousness_state.innovation_rate = (status["awareness_score"] - 0.5) * 0.5

    def _detect_stressors(self):
        """Detect active climate stressors"""
        self.active_stressors = []

        if self.climate_state.temperature_anomaly > 2.0:
            self.active_stressors.append(ClimateStressor.TEMPERATURE_EXTREME)

        if self.climate_state.ecosystem_stress > 0.3:
            self.active_stressors.append(ClimateStressor.BIODIVERSITY_LOSS)

        if self.climate_state.resource_availability < 0.5:
            self.active_stressors.append(ClimateStressor.RESOURCE_SCARCITY)

        if self.climate_state.co2_level > 450:
            self.active_stressors.append(ClimateStressor.AIR_QUALITY)

    def _process_feedback_loop(self, loop: FeedbackLoop):
        """Process a single feedback loop"""
        # Determine input based on direction
        if loop.direction == FeedbackDirection.CLIMATE_TO_CONSCIOUSNESS:
            input_value = self._get_climate_input(loop)
            output_value = self._calculate_consciousness_effect(loop, input_value)
            self.consciousness_effects[loop.name] = output_value

        elif loop.direction == FeedbackDirection.CONSCIOUSNESS_TO_CLIMATE:
            input_value = self._get_consciousness_input(loop)
            output_value = self._calculate_climate_effect(loop, input_value)
            self.climate_effects[loop.name] = output_value

        elif loop.direction == FeedbackDirection.BIDIRECTIONAL:
            # Process both directions
            climate_input = self._get_climate_input(loop)
            consciousness_input = self._get_consciousness_input(loop)

            c2c_output = self._calculate_consciousness_effect(loop, climate_input)
            c2cl_output = self._calculate_climate_effect(loop, consciousness_input)

            self.consciousness_effects[loop.name] = c2c_output
            self.climate_effects[loop.name] = c2cl_output

        # Record in loop history
        loop.input_history.append(input_value if loop.direction != FeedbackDirection.BIDIRECTIONAL else climate_input)
        loop.output_history.append(output_value if loop.direction != FeedbackDirection.BIDIRECTIONAL else c2c_output)

        # Trim history
        if len(loop.input_history) > 100:
            loop.input_history = loop.input_history[-100:]
            loop.output_history = loop.output_history[-100:]

    def _get_climate_input(self, loop: FeedbackLoop) -> float:
        """Get climate-derived input for a feedback loop"""
        if "temperature" in loop.name:
            return abs(self.climate_state.temperature_anomaly)
        elif "resource" in loop.name:
            return 1.0 - self.climate_state.resource_availability
        elif "biodiversity" in loop.name or "ecosystem" in loop.name:
            return 1.0 - self.climate_state.ecosystem_stress
        elif "extreme" in loop.name:
            return self.climate_state.extreme_events
        else:
            return self.climate_state.ecosystem_stress

    def _get_consciousness_input(self, loop: FeedbackLoop) -> float:
        """Get consciousness-derived input for a feedback loop"""
        if "mitigation" in loop.name:
            return self.consciousness_state.awareness_level * self.consciousness_state.adaptation_capacity
        elif "adaptation" in loop.name or "collective" in loop.name:
            return self.consciousness_state.collective_action
        elif "innovation" in loop.name:
            return self.consciousness_state.innovation_rate
        elif "coordination" in loop.name or "crisis" in loop.name:
            return self.consciousness_state.awareness_level * self.consciousness_state.collective_action
        else:
            return self.consciousness_state.awareness_level

    def _calculate_consciousness_effect(self, loop: FeedbackLoop, input_value: float) -> float:
        """Calculate effect on consciousness"""
        if not self._check_threshold(loop, input_value):
            return 0.0

        base_effect = input_value * loop.strength * self.config.amplification_factor

        if loop.feedback_type == FeedbackType.POSITIVE:
            return base_effect
        elif loop.feedback_type == FeedbackType.NEGATIVE:
            return -base_effect
        elif loop.feedback_type == FeedbackType.NONLINEAR:
            # Sigmoid response
            return loop.strength * (1.0 / (1.0 + np.exp(-5 * (input_value - loop.threshold))))
        else:
            return 0.0

    def _calculate_climate_effect(self, loop: FeedbackLoop, input_value: float) -> float:
        """Calculate effect on climate"""
        if not self._check_threshold(loop, input_value):
            return 0.0

        base_effect = input_value * loop.strength * self.config.amplification_factor

        if loop.feedback_type == FeedbackType.POSITIVE:
            return base_effect
        elif loop.feedback_type == FeedbackType.NEGATIVE:
            return -base_effect  # Negative feedback stabilizes
        elif loop.feedback_type == FeedbackType.NONLINEAR:
            return loop.strength * (1.0 / (1.0 + np.exp(-5 * (input_value - loop.threshold))))
        else:
            return 0.0

    def _check_threshold(self, loop: FeedbackLoop, input_value: float) -> bool:
        """Check if input exceeds threshold"""
        return abs(input_value) >= loop.threshold

    def _determine_responses(self):
        """Determine consciousness responses to climate"""
        self.active_responses = []

        if self.consciousness_state.awareness_level > 0.5:
            if len(self.active_stressors) > 2:
                self.active_responses.append(ConsciousnessResponse.COORDINATION)

            if self.consciousness_state.adaptation_capacity > 0.3:
                self.active_responses.append(ConsciousnessResponse.ADAPTATION)

            if self.consciousness_state.innovation_rate > 0.2:
                self.active_responses.append(ConsciousnessResponse.INNOVATION)

            if ClimateStressor.TEMPERATURE_EXTREME in self.active_stressors:
                self.active_responses.append(ConsciousnessResponse.MITIGATION)

        elif self.consciousness_state.awareness_level < 0.2:
            self.active_responses.append(ConsciousnessResponse.DORMANCY)

    def _apply_effects(self):
        """Apply accumulated feedback effects"""
        # Apply consciousness effects
        if self.consciousness and self.config.enable_climate_to_consciousness:
            total_consciousness_effect = sum(self.consciousness_effects.values())
            # Effects would be applied to consciousness system
            # For now, we track them

        # Apply climate effects
        if self.biosphere and self.config.enable_consciousness_to_climate:
            total_climate_effect = sum(self.climate_effects.values())
            # Effects would be applied to biosphere
            # For now, we track them

        # Clear effects for next tick
        self.consciousness_effects.clear()
        self.climate_effects.clear()

    def _record_history(self):
        """Record feedback history"""
        record = {
            "tick": self.tick_count,
            "climate": {
                "temperature_anomaly": self.climate_state.temperature_anomaly,
                "co2": self.climate_state.co2_level,
                "stress": self.climate_state.ecosystem_stress
            },
            "consciousness": {
                "awareness": self.consciousness_state.awareness_level,
                "adaptation": self.consciousness_state.adaptation_capacity
            },
            "stressors": [s.value for s in self.active_stressors],
            "responses": [r.value for r in self.active_responses]
        }

        self.feedback_history.append(record)

        # Trim history
        if len(self.feedback_history) > 500:
            self.feedback_history = self.feedback_history[-500:]

    # ========================================================================
    # Public API
    # ========================================================================

    def get_feedback_analysis(self) -> Dict[str, Any]:
        """Get comprehensive feedback analysis"""
        active_loops = [
            loop for loop in self.feedback_loops.values()
            if loop.active and len(loop.output_history) > 0 and abs(loop.output_history[-1]) > 0.01
        ]

        return {
            "tick_count": self.tick_count,
            "climate_state": {
                "temperature_anomaly": self.climate_state.temperature_anomaly,
                "co2_level": self.climate_state.co2_level,
                "ecosystem_stress": self.climate_state.ecosystem_stress,
                "extreme_events": self.climate_state.extreme_events
            },
            "consciousness_state": {
                "awareness": self.consciousness_state.awareness_level,
                "adaptation_capacity": self.consciousness_state.adaptation_capacity,
                "collective_action": self.consciousness_state.collective_action
            },
            "active_stressors": [s.value for s in self.active_stressors],
            "active_responses": [r.value for r in self.active_responses],
            "active_feedback_loops": [
                {
                    "name": loop.name,
                    "type": loop.feedback_type.value,
                    "strength": loop.strength,
                    "recent_output": loop.output_history[-1] if loop.output_history else 0
                }
                for loop in active_loops
            ],
            "total_loops": len(self.feedback_loops),
            "history_length": len(self.feedback_history)
        }

    def get_loop_status(self, loop_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific feedback loop"""
        for loop in self.feedback_loops.values():
            if loop.name == loop_name:
                return {
                    "name": loop.name,
                    "type": loop.feedback_type.value,
                    "direction": loop.direction.value,
                    "strength": loop.strength,
                    "threshold": loop.threshold,
                    "active": loop.active,
                    "recent_inputs": loop.input_history[-10:] if loop.input_history else [],
                    "recent_outputs": loop.output_history[-10:] if loop.output_history else []
                }
        return None

    def set_loop_active(self, loop_name: str, active: bool):
        """Enable or disable a feedback loop"""
        for loop in self.feedback_loops.values():
            if loop.name == loop_name:
                loop.active = active
                break


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Climate-Consciousness Feedback Demo")
    print("=" * 50)

    # Create feedback system
    config = FeedbackConfig(
        amplification_factor=1.0,
        damping_factor=0.1
    )
    feedback = ClimateFeedbackSystem(config)
    feedback.initialize_feedback_loops()

    print(f"\n1. Initialized with {len(feedback.feedback_loops)} feedback loops")

    # Simulate some climate conditions
    print("\n2. Simulating feedback cycles...")
    for i in range(100):
        # Simulate changing conditions
        feedback.climate_state.temperature_anomaly = 1.0 + i * 0.02
        feedback.climate_state.co2_level = 400 + i * 0.5
        feedback.climate_state.ecosystem_stress = min(1.0, i * 0.01)

        # Simulate consciousness response
        feedback.consciousness_state.awareness_level = min(1.0, 0.3 + i * 0.005)

        feedback.process_tick()

        if i % 25 == 0:
            analysis = feedback.get_feedback_analysis()
            print(f"   Tick {i}: temp_anomaly={analysis['climate_state']['temperature_anomaly']:.2f}C, "
                  f"awareness={analysis['consciousness_state']['awareness']:.2f}, "
                  f"stressors={len(analysis['active_stressors'])}")

    # Final analysis
    print("\n3. Final feedback analysis:")
    analysis = feedback.get_feedback_analysis()
    print(f"   Active stressors: {analysis['active_stressors']}")
    print(f"   Active responses: {analysis['active_responses']}")
    print(f"   Active loops: {len(analysis['active_feedback_loops'])}")

    print("\n4. Active feedback loops:")
    for loop_info in analysis['active_feedback_loops'][:5]:
        print(f"   - {loop_info['name']}: {loop_info['type']}, "
              f"strength={loop_info['strength']:.2f}")
