"""
Metabolism-Consciousness Coupling System

Implements bidirectional coupling between organism metabolism and
consciousness systems, reflecting how energy states affect awareness
and cognitive function.

Features:
- Energy-dependent consciousness modulation
- Metabolic state influence on attention
- Hunger/satiation effects on emotional state
- Stress response coupling
- Homeostatic regulation of consciousness

Biological Inspiration:
- Glucose availability affects cognitive performance
- Sleep deprivation impairs consciousness
- Stress hormones modulate attention and memory
- Energy conservation modes reduce awareness

Usage:
    from consciousness.metabolism_coupling import MetabolismConsciousnessCoupler

    coupler = MetabolismConsciousnessCoupler(
        organism.metabolism,
        organism.consciousness
    )

    # Update coupling each tick
    coupler.update()

    # Get coupling state
    state = coupler.get_coupling_state()
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from organism.metabolism import Metabolism, EnergyState, MetabolicMode, ResourceType
from consciousness.neural_consciousness import NeuralConsciousness


# ============================================================================
# Enums and Data Structures
# ============================================================================

class CouplingMode(Enum):
    """Modes of metabolism-consciousness coupling"""
    FULL = auto()           # Full bidirectional coupling
    METABOLISM_TO_MIND = auto()  # Only metabolism affects consciousness
    MIND_TO_METABOLISM = auto()  # Only consciousness affects metabolism
    DECOUPLED = auto()      # No coupling


class ConsciousnessImpact(Enum):
    """Impact levels of metabolic state on consciousness"""
    ENHANCED = "enhanced"      # Energy surplus boosts consciousness
    NORMAL = "normal"          # Normal operation
    IMPAIRED = "impaired"      # Low energy impairs function
    CRITICAL = "critical"      # Severe impairment
    DORMANT = "dormant"        # Consciousness suppressed


@dataclass
class CouplingParameters:
    """Parameters controlling the coupling behavior"""
    # Energy thresholds (as fraction of max energy)
    enhancement_threshold: float = 0.8   # Above this, consciousness enhanced
    normal_threshold: float = 0.5        # Normal operation threshold
    impairment_threshold: float = 0.3    # Below this, impaired
    critical_threshold: float = 0.15     # Below this, critical
    dormancy_threshold: float = 0.05     # Below this, dormant

    # Coupling strengths (0-1)
    attention_coupling: float = 0.7      # How much energy affects attention
    memory_coupling: float = 0.5         # How much energy affects memory
    emotional_coupling: float = 0.8      # How much energy affects emotions
    consciousness_coupling: float = 0.6   # Overall consciousness modulation

    # Recovery rates
    attention_recovery_rate: float = 0.1  # How fast attention recovers
    cognitive_recovery_rate: float = 0.05 # How fast cognition recovers

    # Metabolic feedback from consciousness
    attention_cost: float = 0.5          # Energy cost of high attention
    memory_cost: float = 0.3             # Energy cost of memory operations
    emotional_cost: float = 0.2          # Energy cost of emotional processing


@dataclass
class CouplingState:
    """Current state of the metabolism-consciousness coupling"""
    impact: ConsciousnessImpact = ConsciousnessImpact.NORMAL
    attention_modifier: float = 1.0
    memory_modifier: float = 1.0
    emotional_modifier: float = 1.0
    consciousness_modifier: float = 1.0
    energy_drain_rate: float = 0.0
    tick_count: int = 0


# ============================================================================
# Coupling Functions
# ============================================================================

def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Smooth sigmoid transition function"""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def smooth_step(x: float, edge0: float, edge1: float) -> float:
    """Smooth step function between two edges"""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)


# ============================================================================
# Main Coupling System
# ============================================================================

class MetabolismConsciousnessCoupler:
    """
    Manages bidirectional coupling between metabolism and consciousness.

    This system models how energy availability affects cognitive function
    and how cognitive demands impact energy consumption.

    Key Relationships:
    1. Low energy -> reduced attention capacity
    2. Low energy -> impaired memory consolidation
    3. Low energy -> negative emotional bias
    4. High cognitive load -> increased energy consumption
    5. Stress -> altered consciousness and metabolism

    Example:
        metabolism = Metabolism()
        consciousness = NeuralConsciousness()
        coupler = MetabolismConsciousnessCoupler(metabolism, consciousness)

        # Each tick
        coupler.update()

        # Check impact
        if coupler.state.impact == ConsciousnessImpact.IMPAIRED:
            print("Low energy is affecting cognitive function!")
    """

    def __init__(
        self,
        metabolism: Metabolism,
        consciousness: NeuralConsciousness,
        mode: CouplingMode = CouplingMode.FULL,
        params: Optional[CouplingParameters] = None
    ):
        self.metabolism = metabolism
        self.consciousness = consciousness
        self.mode = mode
        self.params = params or CouplingParameters()
        self.state = CouplingState()

        # History for tracking trends
        self._energy_history: List[float] = []
        self._consciousness_history: List[float] = []
        self._impact_history: List[ConsciousnessImpact] = []

        # Cached original values
        self._original_attention_capacity = consciousness.attention_state.attention_capacity
        self._original_memory_capacity = consciousness.memory_capacity

    def update(self) -> CouplingState:
        """
        Update the coupling between metabolism and consciousness.

        This should be called each simulation tick to maintain
        the bidirectional relationship.

        Returns:
            Current coupling state
        """
        self.state.tick_count += 1

        # Record history
        energy_ratio = self.metabolism.energy / self.metabolism.max_energy
        self._energy_history.append(energy_ratio)
        if len(self._energy_history) > 100:
            self._energy_history.pop(0)

        # Update based on coupling mode
        if self.mode in [CouplingMode.FULL, CouplingMode.METABOLISM_TO_MIND]:
            self._apply_metabolism_to_consciousness(energy_ratio)

        if self.mode in [CouplingMode.FULL, CouplingMode.MIND_TO_METABOLISM]:
            self._apply_consciousness_to_metabolism()

        # Update impact level
        self.state.impact = self._calculate_impact(energy_ratio)
        self._impact_history.append(self.state.impact)
        if len(self._impact_history) > 100:
            self._impact_history.pop(0)

        return self.state

    def _apply_metabolism_to_consciousness(self, energy_ratio: float) -> None:
        """Apply metabolic state effects to consciousness"""
        params = self.params

        # Calculate base modifiers based on energy level
        attention_mod = self._calculate_attention_modifier(energy_ratio)
        memory_mod = self._calculate_memory_modifier(energy_ratio)
        emotional_mod = self._calculate_emotional_modifier(energy_ratio)
        consciousness_mod = self._calculate_consciousness_modifier(energy_ratio)

        # Apply smooth transitions
        self.state.attention_modifier = (
            self.state.attention_modifier * 0.8 +
            attention_mod * 0.2
        )
        self.state.memory_modifier = (
            self.state.memory_modifier * 0.9 +
            memory_mod * 0.1
        )
        self.state.emotional_modifier = (
            self.state.emotional_modifier * 0.8 +
            emotional_mod * 0.2
        )
        self.state.consciousness_modifier = (
            self.state.consciousness_modifier * 0.9 +
            consciousness_mod * 0.1
        )

        # Apply to consciousness system
        self._modulate_attention(self.state.attention_modifier)
        self._modulate_emotional_sensitivity(energy_ratio)
        self._modulate_consciousness_level(self.state.consciousness_modifier)

    def _calculate_attention_modifier(self, energy_ratio: float) -> float:
        """Calculate attention modifier based on energy"""
        params = self.params

        if energy_ratio >= params.enhancement_threshold:
            # Enhanced attention with surplus energy
            boost = (energy_ratio - params.enhancement_threshold) / (1.0 - params.enhancement_threshold)
            return 1.0 + boost * 0.2 * params.attention_coupling

        elif energy_ratio >= params.normal_threshold:
            # Normal operation
            return 1.0

        elif energy_ratio >= params.impairment_threshold:
            # Gradual impairment
            reduction = smooth_step(energy_ratio, params.impairment_threshold, params.normal_threshold)
            return 0.7 + reduction * 0.3

        elif energy_ratio >= params.critical_threshold:
            # Significant impairment
            reduction = smooth_step(energy_ratio, params.critical_threshold, params.impairment_threshold)
            return 0.4 + reduction * 0.3

        else:
            # Critical - minimal attention
            return 0.2

    def _calculate_memory_modifier(self, energy_ratio: float) -> float:
        """Calculate memory modifier based on energy"""
        params = self.params

        # Memory is more resistant to low energy than attention
        if energy_ratio >= params.normal_threshold:
            return 1.0

        elif energy_ratio >= params.impairment_threshold:
            reduction = smooth_step(energy_ratio, params.impairment_threshold, params.normal_threshold)
            return 0.8 + reduction * 0.2

        elif energy_ratio >= params.critical_threshold:
            reduction = smooth_step(energy_ratio, params.critical_threshold, params.impairment_threshold)
            return 0.5 + reduction * 0.3

        else:
            # Critical - memory consolidation impaired
            return 0.3

    def _calculate_emotional_modifier(self, energy_ratio: float) -> float:
        """Calculate emotional modifier based on energy"""
        params = self.params

        # Low energy creates negative emotional bias
        if energy_ratio >= params.enhancement_threshold:
            # Positive bias with high energy
            return 1.2

        elif energy_ratio >= params.normal_threshold:
            return 1.0

        elif energy_ratio >= params.impairment_threshold:
            # Slight negative bias
            return 0.9

        else:
            # Strong negative bias - stress response
            return 0.6

    def _calculate_consciousness_modifier(self, energy_ratio: float) -> float:
        """Calculate overall consciousness modifier"""
        params = self.params

        if energy_ratio >= params.enhancement_threshold:
            boost = (energy_ratio - params.enhancement_threshold) / (1.0 - params.enhancement_threshold)
            return 1.0 + boost * 0.15

        elif energy_ratio >= params.normal_threshold:
            return 1.0

        elif energy_ratio >= params.impairment_threshold:
            reduction = smooth_step(energy_ratio, params.impairment_threshold, params.normal_threshold)
            return 0.75 + reduction * 0.25

        elif energy_ratio >= params.critical_threshold:
            reduction = smooth_step(energy_ratio, params.critical_threshold, params.impairment_threshold)
            return 0.5 + reduction * 0.25

        elif energy_ratio >= params.dormancy_threshold:
            return 0.3

        else:
            # Dormancy - consciousness suppressed
            return 0.1

    def _modulate_attention(self, modifier: float) -> None:
        """Apply attention modulation to consciousness"""
        # Adjust attention capacity
        new_capacity = self._original_attention_capacity * modifier
        self.consciousness.attention_state.attention_capacity = max(0.2, min(1.5, new_capacity))

        # Low energy increases fatigue faster
        if modifier < 0.8:
            fatigue_increase = (1.0 - modifier) * 0.02
            self.consciousness.attention_state.fatigue_level = min(
                1.0,
                self.consciousness.attention_state.fatigue_level + fatigue_increase
            )

    def _modulate_emotional_sensitivity(self, energy_ratio: float) -> None:
        """Modulate emotional processing based on energy"""
        if energy_ratio < self.params.impairment_threshold:
            # Low energy creates negative valence shift
            negative_shift = (self.params.impairment_threshold - energy_ratio) * 0.5
            self.consciousness.emotional_state.valence -= negative_shift * 0.1

            # Increase arousal (stress response)
            self.consciousness.emotional_state.arousal += (1.0 - energy_ratio) * 0.05
            self.consciousness.emotional_state.arousal = min(1.0, self.consciousness.emotional_state.arousal)

    def _modulate_consciousness_level(self, modifier: float) -> None:
        """Modulate overall consciousness level"""
        # Apply consciousness modifier
        base_level = self.consciousness.consciousness_level
        modulated_level = base_level * modifier
        self.consciousness.consciousness_level = max(0.1, min(1.0, modulated_level))

    def _apply_consciousness_to_metabolism(self) -> None:
        """Apply consciousness state effects to metabolism"""
        params = self.params

        # Calculate energy drain from cognitive activity
        energy_drain = 0.0

        # Attention cost
        attention_focus = self.consciousness.attention_state.focus_strength
        if attention_focus > 0.5:
            energy_drain += (attention_focus - 0.5) * params.attention_cost

        # Memory operations cost
        memory_load = len(self.consciousness.working_memory) / self.consciousness.memory_capacity
        if memory_load > 0.5:
            energy_drain += (memory_load - 0.5) * params.memory_cost

        # Emotional processing cost
        emotional_arousal = self.consciousness.emotional_state.arousal
        if emotional_arousal > 0.6:
            energy_drain += (emotional_arousal - 0.6) * params.emotional_cost

        self.state.energy_drain_rate = energy_drain

        # Apply energy drain to metabolism
        if energy_drain > 0:
            self.metabolism.energy -= energy_drain * 0.1

    def _calculate_impact(self, energy_ratio: float) -> ConsciousnessImpact:
        """Calculate the overall impact level"""
        params = self.params

        if energy_ratio >= params.enhancement_threshold:
            return ConsciousnessImpact.ENHANCED
        elif energy_ratio >= params.normal_threshold:
            return ConsciousnessImpact.NORMAL
        elif energy_ratio >= params.impairment_threshold:
            return ConsciousnessImpact.IMPAIRED
        elif energy_ratio >= params.dormancy_threshold:
            return ConsciousnessImpact.CRITICAL
        else:
            return ConsciousnessImpact.DORMANT

    def get_coupling_state(self) -> Dict[str, Any]:
        """Get comprehensive coupling state"""
        energy_ratio = self.metabolism.energy / self.metabolism.max_energy

        return {
            "impact": self.state.impact.value,
            "energy_ratio": energy_ratio,
            "metabolic_state": self.metabolism.energy_state.value,
            "modifiers": {
                "attention": self.state.attention_modifier,
                "memory": self.state.memory_modifier,
                "emotional": self.state.emotional_modifier,
                "consciousness": self.state.consciousness_modifier
            },
            "energy_drain_rate": self.state.energy_drain_rate,
            "consciousness_level": self.consciousness.consciousness_level,
            "attention_capacity": self.consciousness.attention_state.attention_capacity,
            "tick_count": self.state.tick_count
        }

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends in the coupling relationship"""
        if len(self._energy_history) < 10:
            return {"status": "insufficient_data"}

        recent_energy = self._energy_history[-20:]
        energy_trend = (recent_energy[-1] - recent_energy[0]) / len(recent_energy)

        # Count impact transitions
        impact_changes = 0
        for i in range(1, len(self._impact_history)):
            if self._impact_history[i] != self._impact_history[i-1]:
                impact_changes += 1

        return {
            "energy_trend": energy_trend,
            "energy_stability": 1.0 - (max(recent_energy) - min(recent_energy)),
            "impact_volatility": impact_changes / len(self._impact_history) if self._impact_history else 0,
            "current_impact": self.state.impact.value,
            "avg_energy": sum(recent_energy) / len(recent_energy),
            "data_points": len(self._energy_history)
        }

    def set_coupling_mode(self, mode: CouplingMode) -> None:
        """Change the coupling mode"""
        self.mode = mode

    def reset(self) -> None:
        """Reset coupling state to defaults"""
        self.state = CouplingState()
        self._energy_history.clear()
        self._consciousness_history.clear()
        self._impact_history.clear()

        # Restore original values
        self.consciousness.attention_state.attention_capacity = self._original_attention_capacity


# ============================================================================
# Utility Functions
# ============================================================================

def create_coupler_for_organism(organism, mode: CouplingMode = CouplingMode.FULL) -> Optional[MetabolismConsciousnessCoupler]:
    """
    Create a metabolism-consciousness coupler for an organism.

    Automatically detects and connects metabolism and consciousness subsystems.

    Args:
        organism: An organism with metabolism and consciousness
        mode: Coupling mode to use

    Returns:
        MetabolismConsciousnessCoupler or None if components not found
    """
    metabolism = getattr(organism, 'metabolism', None)
    consciousness = organism.get_subsystem('neural_consciousness') if hasattr(organism, 'get_subsystem') else None

    if metabolism is None or consciousness is None:
        return None

    return MetabolismConsciousnessCoupler(
        metabolism=metabolism,
        consciousness=consciousness,
        mode=mode
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Metabolism-Consciousness Coupling Demo")
    print("=" * 60)

    # Create metabolism and consciousness
    metabolism = Metabolism()
    consciousness = NeuralConsciousness(
        attention_dim=32,
        memory_capacity=7,
        emotional_sensitivity=0.5
    )

    # Create coupler
    coupler = MetabolismConsciousnessCoupler(
        metabolism=metabolism,
        consciousness=consciousness,
        mode=CouplingMode.FULL
    )

    print("\n1. Normal Energy State")
    print("-" * 40)
    metabolism.energy = 70.0
    for _ in range(5):
        coupler.update()
    state = coupler.get_coupling_state()
    print(f"   Impact: {state['impact']}")
    print(f"   Attention modifier: {state['modifiers']['attention']:.2f}")
    print(f"   Consciousness level: {state['consciousness_level']:.2f}")

    print("\n2. Low Energy State")
    print("-" * 40)
    metabolism.energy = 20.0
    for _ in range(10):
        coupler.update()
    state = coupler.get_coupling_state()
    print(f"   Impact: {state['impact']}")
    print(f"   Attention modifier: {state['modifiers']['attention']:.2f}")
    print(f"   Consciousness level: {state['consciousness_level']:.2f}")
    print(f"   Emotional modifier: {state['modifiers']['emotional']:.2f}")

    print("\n3. Critical Energy State")
    print("-" * 40)
    metabolism.energy = 5.0
    for _ in range(10):
        coupler.update()
    state = coupler.get_coupling_state()
    print(f"   Impact: {state['impact']}")
    print(f"   Attention modifier: {state['modifiers']['attention']:.2f}")
    print(f"   Consciousness level: {state['consciousness_level']:.2f}")

    print("\n4. Enhanced Energy State")
    print("-" * 40)
    metabolism.energy = 95.0
    for _ in range(10):
        coupler.update()
    state = coupler.get_coupling_state()
    print(f"   Impact: {state['impact']}")
    print(f"   Attention modifier: {state['modifiers']['attention']:.2f}")
    print(f"   Consciousness level: {state['consciousness_level']:.2f}")

    print("\n5. Trend Analysis")
    print("-" * 40)
    trends = coupler.get_trend_analysis()
    print(f"   Energy trend: {trends['energy_trend']:.4f}")
    print(f"   Impact volatility: {trends['impact_volatility']:.2f}")
    print(f"   Data points: {trends['data_points']}")
