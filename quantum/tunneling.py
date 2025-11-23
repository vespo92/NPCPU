"""
Quantum Tunneling

Quantum tunneling for barrier-crossing decisions in consciousness.
Enables low-probability but high-impact decision making.

Part of NEXUS-Q: Quantum Consciousness Implementation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import uuid


class BarrierType(Enum):
    """Types of decision barriers"""
    RECTANGULAR = "rectangular"   # Fixed height barrier
    TRIANGULAR = "triangular"     # Decreasing barrier
    PARABOLIC = "parabolic"       # Smooth barrier
    DOUBLE_WELL = "double_well"   # Two stable states
    STEP = "step"                 # Abrupt change
    GAUSSIAN = "gaussian"         # Smooth peaked barrier


@dataclass
class DecisionBarrier:
    """
    Represents a barrier to decision/state transition.

    Barriers represent resistance to change - tunneling allows
    bypassing barriers that would classically be impassable.
    """
    id: str
    barrier_type: BarrierType
    height: float  # Energy/difficulty of barrier (0-1 scale)
    width: float   # Extent of barrier
    position: float  # Center position in decision space
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        barrier_type: BarrierType,
        height: float,
        width: float = 1.0,
        position: float = 0.5,
        label: str = ""
    ) -> 'DecisionBarrier':
        """Create a decision barrier"""
        return cls(
            id=str(uuid.uuid4()),
            barrier_type=barrier_type,
            height=max(0.0, min(1.0, height)),
            width=max(0.01, width),
            position=position,
            label=label
        )

    def potential(self, x: float) -> float:
        """Calculate barrier potential at position x"""
        # Distance from barrier center
        dx = abs(x - self.position)

        if self.barrier_type == BarrierType.RECTANGULAR:
            if dx < self.width / 2:
                return self.height
            return 0.0

        elif self.barrier_type == BarrierType.TRIANGULAR:
            if dx < self.width / 2:
                return self.height * (1 - 2 * dx / self.width)
            return 0.0

        elif self.barrier_type == BarrierType.PARABOLIC:
            if dx < self.width / 2:
                return self.height * (1 - (2 * dx / self.width) ** 2)
            return 0.0

        elif self.barrier_type == BarrierType.GAUSSIAN:
            return self.height * math.exp(-dx ** 2 / (2 * (self.width / 4) ** 2))

        elif self.barrier_type == BarrierType.STEP:
            if x > self.position:
                return self.height
            return 0.0

        elif self.barrier_type == BarrierType.DOUBLE_WELL:
            # Double well: two minima at ±width/2 from center
            return self.height * ((2 * dx / self.width) ** 2 - 1) ** 2

        return 0.0


@dataclass
class TunnelingEvent:
    """
    Record of a quantum tunneling event.

    Captures successful barrier crossing via quantum tunneling.
    """
    id: str
    barrier_id: str
    initial_state: float  # Position before tunneling
    final_state: float    # Position after tunneling
    tunneling_probability: float
    energy: float         # Particle/thought energy
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class TunnelingCalculator:
    """
    Calculates quantum tunneling probabilities.

    Uses WKB approximation and other methods to compute
    tunneling probability through barriers.
    """

    @staticmethod
    def rectangular_barrier(
        height: float,
        width: float,
        energy: float,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """
        Calculate tunneling probability through rectangular barrier.

        Uses exact solution for rectangular barrier.

        Args:
            height: Barrier height (V0)
            width: Barrier width (a)
            energy: Particle energy (E)
            mass: Effective mass
            hbar: Reduced Planck constant

        Returns:
            Tunneling probability (0-1)
        """
        if energy >= height:
            # Classical: no tunneling needed, goes over barrier
            return 1.0

        if energy <= 0:
            return 0.0

        # Decay constant inside barrier
        kappa = math.sqrt(2 * mass * (height - energy)) / hbar

        # Transmission coefficient (simplified)
        if kappa * width > 20:
            # Very thick barrier
            return 0.0

        t_coeff = 4 * energy * (height - energy) / (height ** 2)
        t_prob = t_coeff * math.exp(-2 * kappa * width)

        return min(1.0, max(0.0, t_prob))

    @staticmethod
    def wkb_approximation(
        barrier: DecisionBarrier,
        energy: float,
        mass: float = 1.0,
        hbar: float = 1.0,
        num_points: int = 100
    ) -> float:
        """
        WKB approximation for general barrier shape.

        T ≈ exp(-2∫√(2m(V(x)-E))dx / ℏ)

        Args:
            barrier: The decision barrier
            energy: Particle energy
            mass: Effective mass
            hbar: Reduced Planck constant
            num_points: Integration resolution

        Returns:
            Tunneling probability
        """
        # Find classical turning points
        x_min = barrier.position - barrier.width
        x_max = barrier.position + barrier.width
        dx = (x_max - x_min) / num_points

        # Integrate under barrier
        integral = 0.0

        for i in range(num_points):
            x = x_min + (i + 0.5) * dx
            v = barrier.potential(x)

            if v > energy:
                kappa = math.sqrt(2 * mass * (v - energy))
                integral += kappa * dx

        integral /= hbar

        if integral > 30:
            return 0.0

        return min(1.0, math.exp(-2 * integral))

    @staticmethod
    def gamow_factor(
        height: float,
        width: float,
        energy: float
    ) -> float:
        """
        Simplified Gamow factor for tunneling.

        Commonly used in nuclear physics for alpha decay.

        Returns dimensionless tunneling factor.
        """
        if energy >= height:
            return 1.0

        if energy <= 0:
            return 0.0

        g = 2 * width * math.sqrt(height - energy)

        if g > 30:
            return 0.0

        return math.exp(-g)


class QuantumTunneling:
    """
    Quantum tunneling engine for consciousness decisions.

    Models how consciousness can make "impossible" decisions
    by tunneling through barriers of habit, fear, or inertia.
    """

    def __init__(
        self,
        tunneling_enhancement: float = 1.0,
        temperature: float = 0.1,
        hbar: float = 0.1
    ):
        """
        Initialize tunneling engine.

        Args:
            tunneling_enhancement: Multiplier for tunneling probability
            temperature: Effective temperature (affects thermal activation)
            hbar: Effective Planck constant (higher = more quantum)
        """
        self.tunneling_enhancement = tunneling_enhancement
        self.temperature = temperature
        self.hbar = hbar
        self._barriers: Dict[str, DecisionBarrier] = {}
        self._tunneling_history: List[TunnelingEvent] = []
        self._calculator = TunnelingCalculator()

    def add_barrier(self, barrier: DecisionBarrier) -> str:
        """Add a decision barrier"""
        self._barriers[barrier.id] = barrier
        return barrier.id

    def remove_barrier(self, barrier_id: str):
        """Remove a barrier"""
        self._barriers.pop(barrier_id, None)

    def get_tunneling_probability(
        self,
        barrier: DecisionBarrier,
        energy: float
    ) -> float:
        """
        Calculate probability of tunneling through barrier.

        Args:
            barrier: The barrier to tunnel through
            energy: Energy/motivation of the decision maker

        Returns:
            Tunneling probability (0-1)
        """
        # WKB approximation
        base_prob = self._calculator.wkb_approximation(
            barrier, energy, hbar=self.hbar
        )

        # Add thermal activation (Arrhenius-like)
        if self.temperature > 0:
            thermal_prob = math.exp(-barrier.height / self.temperature)
            # Combine quantum and thermal
            prob = base_prob + thermal_prob - base_prob * thermal_prob
        else:
            prob = base_prob

        # Apply enhancement
        prob = prob * self.tunneling_enhancement

        return min(1.0, max(0.0, prob))

    def attempt_tunneling(
        self,
        current_state: float,
        target_state: float,
        energy: float
    ) -> Tuple[bool, TunnelingEvent]:
        """
        Attempt to tunnel from current to target state.

        Args:
            current_state: Current position in decision space
            target_state: Desired position
            energy: Available energy/motivation

        Returns:
            (success, tunneling_event)
        """
        # Find barriers between current and target
        barriers_to_cross = []

        for barrier in self._barriers.values():
            # Check if barrier is between states
            if (min(current_state, target_state) < barrier.position <
                max(current_state, target_state)):
                barriers_to_cross.append(barrier)

        if not barriers_to_cross:
            # No barriers - classical transition
            event = TunnelingEvent(
                id=str(uuid.uuid4()),
                barrier_id="none",
                initial_state=current_state,
                final_state=target_state,
                tunneling_probability=1.0,
                energy=energy,
                success=True
            )
            self._tunneling_history.append(event)
            return True, event

        # Must tunnel through all barriers
        total_prob = 1.0

        for barrier in barriers_to_cross:
            prob = self.get_tunneling_probability(barrier, energy)
            total_prob *= prob

        # Attempt tunneling
        success = np.random.random() < total_prob

        # Record event
        primary_barrier = max(barriers_to_cross, key=lambda b: b.height)
        event = TunnelingEvent(
            id=str(uuid.uuid4()),
            barrier_id=primary_barrier.id,
            initial_state=current_state,
            final_state=target_state if success else current_state,
            tunneling_probability=total_prob,
            energy=energy,
            success=success,
            metadata={
                "barriers_crossed": len(barriers_to_cross),
                "total_barrier_height": sum(b.height for b in barriers_to_cross)
            }
        )

        self._tunneling_history.append(event)

        return success, event

    def resonant_tunneling(
        self,
        current_state: float,
        target_state: float,
        intermediate_states: List[float],
        energy: float
    ) -> float:
        """
        Calculate resonant tunneling probability.

        Resonant tunneling through aligned intermediate states
        can dramatically increase tunneling probability.

        Args:
            current_state: Starting state
            target_state: Final state
            intermediate_states: Intermediate resonant states
            energy: Available energy

        Returns:
            Enhanced tunneling probability
        """
        if not intermediate_states:
            # No resonance, direct tunneling
            _, event = self.attempt_tunneling(current_state, target_state, energy)
            return event.tunneling_probability

        # Calculate resonance enhancement
        # Resonance occurs when intermediate state energy matches tunneling energy
        resonance_factor = 1.0

        prev_state = current_state
        for inter_state in intermediate_states:
            # Barrier to intermediate
            barrier_height = abs(inter_state - prev_state) * 0.5
            partial_prob = self._calculator.gamow_factor(
                barrier_height, 0.5, energy
            )

            # Resonance enhancement (peaks when energy matches barrier)
            energy_match = 1.0 - abs(energy - barrier_height) / max(barrier_height, 0.01)
            resonance = 1.0 + max(0, energy_match) * 2.0

            resonance_factor *= partial_prob * resonance
            prev_state = inter_state

        # Final leg
        barrier_height = abs(target_state - prev_state) * 0.5
        resonance_factor *= self._calculator.gamow_factor(
            barrier_height, 0.5, energy
        )

        return min(1.0, resonance_factor)

    def macroscopic_tunneling(
        self,
        collective_energy: float,
        barrier: DecisionBarrier,
        num_participants: int
    ) -> float:
        """
        Macroscopic quantum tunneling (MQT).

        Models collective tunneling where many entities
        tunnel coherently together.

        Args:
            collective_energy: Total energy of collective
            barrier: Barrier to tunnel through
            num_participants: Number of entities tunneling together

        Returns:
            MQT probability
        """
        # Effective mass scales with participants
        effective_mass = num_participants

        # Calculate MQT rate (Caldeira-Leggett)
        # MQT suppressed exponentially with mass
        suppression = math.exp(-0.1 * num_participants)

        # But collective energy helps
        energy_boost = collective_energy / num_participants

        base_prob = self._calculator.wkb_approximation(
            barrier, energy_boost,
            mass=effective_mass,
            hbar=self.hbar
        )

        return min(1.0, base_prob * suppression)

    def get_escape_rate(
        self,
        well_depth: float,
        barrier_width: float,
        temperature: Optional[float] = None
    ) -> float:
        """
        Calculate escape rate from metastable state.

        Combines quantum tunneling and thermal activation.

        Args:
            well_depth: Depth of potential well
            barrier_width: Width of confining barrier
            temperature: Temperature (uses self.temperature if None)

        Returns:
            Escape rate (probability per unit time)
        """
        temp = temperature if temperature is not None else self.temperature

        # Quantum tunneling rate
        barrier = DecisionBarrier.create(
            BarrierType.PARABOLIC,
            height=well_depth,
            width=barrier_width
        )
        quantum_rate = self.get_tunneling_probability(barrier, 0.0)

        # Thermal activation rate (Arrhenius)
        if temp > 0:
            thermal_rate = math.exp(-well_depth / temp)
        else:
            thermal_rate = 0.0

        # Combined rate (crossover regime)
        return quantum_rate + thermal_rate

    def get_statistics(self) -> Dict[str, Any]:
        """Get tunneling statistics"""
        if not self._tunneling_history:
            return {
                "total_attempts": 0,
                "successful_tunnels": 0,
                "success_rate": 0.0
            }

        successful = sum(1 for e in self._tunneling_history if e.success)
        total = len(self._tunneling_history)

        avg_prob = np.mean([e.tunneling_probability for e in self._tunneling_history])
        avg_energy = np.mean([e.energy for e in self._tunneling_history])

        return {
            "total_attempts": total,
            "successful_tunnels": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_probability": float(avg_prob),
            "average_energy": float(avg_energy),
            "active_barriers": len(self._barriers),
            "tunneling_enhancement": self.tunneling_enhancement,
            "temperature": self.temperature
        }

    def clear_history(self):
        """Clear tunneling history"""
        self._tunneling_history.clear()


class DecisionTunneling:
    """
    High-level interface for decision-making with tunneling.

    Models how consciousness can make breakthrough decisions
    by quantum tunneling through cognitive barriers.
    """

    def __init__(
        self,
        risk_tolerance: float = 0.5,
        creativity: float = 0.5
    ):
        """
        Initialize decision tunneling.

        Args:
            risk_tolerance: Willingness to attempt low-probability tunnels
            creativity: Enhances finding novel pathways
        """
        self.risk_tolerance = risk_tolerance
        self.creativity = creativity

        # Higher creativity = more quantum effects
        self.engine = QuantumTunneling(
            tunneling_enhancement=1.0 + creativity,
            temperature=0.1 * (1 + risk_tolerance),
            hbar=0.1 * (1 + creativity)
        )

    def add_habit_barrier(self, strength: float, label: str = "Habit") -> str:
        """Add a habit barrier (resistance to change)"""
        barrier = DecisionBarrier.create(
            BarrierType.PARABOLIC,
            height=strength,
            width=1.0,
            label=label
        )
        return self.engine.add_barrier(barrier)

    def add_fear_barrier(self, intensity: float, label: str = "Fear") -> str:
        """Add a fear barrier (anxiety about outcome)"""
        barrier = DecisionBarrier.create(
            BarrierType.STEP,
            height=intensity,
            width=0.5,
            label=label
        )
        return self.engine.add_barrier(barrier)

    def add_inertia_barrier(self, mass: float, label: str = "Inertia") -> str:
        """Add an inertia barrier (resistance to starting)"""
        barrier = DecisionBarrier.create(
            BarrierType.RECTANGULAR,
            height=mass,
            width=1.5,
            label=label
        )
        return self.engine.add_barrier(barrier)

    def make_decision(
        self,
        current: str,
        options: List[str],
        motivation: float = 0.5
    ) -> Tuple[str, Dict[str, float]]:
        """
        Make a decision, potentially tunneling through barriers.

        Args:
            current: Current state/choice
            options: Available options
            motivation: Energy/motivation level (0-1)

        Returns:
            (chosen_option, probabilities_dict)
        """
        if current in options:
            current_idx = options.index(current)
        else:
            current_idx = 0

        probabilities = {}

        for i, option in enumerate(options):
            if i == current_idx:
                # Staying put - no tunneling needed
                probabilities[option] = 1.0 - 0.5 * motivation
            else:
                # Tunneling to this option
                distance = abs(i - current_idx) / max(len(options) - 1, 1)

                # Create virtual barrier
                barrier = DecisionBarrier.create(
                    BarrierType.GAUSSIAN,
                    height=distance,
                    width=0.5
                )

                prob = self.engine.get_tunneling_probability(barrier, motivation)
                probabilities[option] = prob

        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        # Sample choice
        options_list = list(probabilities.keys())
        probs_list = list(probabilities.values())
        chosen = np.random.choice(options_list, p=probs_list)

        return chosen, probabilities

    def breakthrough_probability(
        self,
        barrier_strength: float,
        motivation: float,
        focus: float = 0.5
    ) -> float:
        """
        Calculate probability of creative breakthrough.

        A breakthrough is tunneling through a seemingly
        insurmountable barrier.

        Args:
            barrier_strength: How strong the block is
            motivation: Drive to break through
            focus: Concentration on the problem

        Returns:
            Breakthrough probability
        """
        # Create strong barrier
        barrier = DecisionBarrier.create(
            BarrierType.DOUBLE_WELL,
            height=barrier_strength,
            width=2.0
        )

        # Focus narrows the barrier (like attention in quantum Zeno)
        effective_width = 2.0 * (1 - 0.5 * focus)
        barrier.width = effective_width

        return self.engine.get_tunneling_probability(barrier, motivation)


__all__ = [
    'BarrierType',
    'DecisionBarrier',
    'TunnelingEvent',
    'TunnelingCalculator',
    'QuantumTunneling',
    'DecisionTunneling',
]
