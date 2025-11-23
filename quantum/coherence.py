"""
Quantum Coherence Metrics

Quantum coherence metrics and decoherence modeling for
consciousness stability and quantum effects preservation.

Part of NEXUS-Q: Quantum Consciousness Implementation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import time

from .quantum_state import QuantumStateVector, DensityMatrix


class DecoherenceChannel(Enum):
    """Types of quantum decoherence channels"""
    DEPHASING = "dephasing"           # Pure dephasing (T2)
    AMPLITUDE_DAMPING = "amplitude"   # Energy loss (T1)
    DEPOLARIZING = "depolarizing"     # Complete randomization
    BIT_FLIP = "bit_flip"             # Random bit flips
    PHASE_FLIP = "phase_flip"         # Random phase flips
    THERMAL = "thermal"               # Thermal equilibration


@dataclass
class CoherenceMetrics:
    """
    Comprehensive quantum coherence metrics.

    Tracks multiple aspects of quantum coherence relevant
    to consciousness modeling.
    """
    # L1 coherence (sum of off-diagonal magnitudes)
    l1_coherence: float = 0.0

    # Relative entropy of coherence
    relative_entropy: float = 0.0

    # Quantum purity
    purity: float = 1.0

    # Entanglement entropy (for multi-qubit)
    entanglement_entropy: float = 0.0

    # Decoherence time estimates
    t1_relaxation: float = float('inf')  # Amplitude damping time
    t2_dephasing: float = float('inf')   # Dephasing time

    # Coherence decay rate
    decay_rate: float = 0.0

    # Time of measurement
    timestamp: float = field(default_factory=time.time)

    def overall_coherence(self) -> float:
        """
        Combined coherence score (0-1).

        Higher values indicate more quantum coherence.
        """
        # Normalize L1 coherence (typically scales with dimension)
        normalized_l1 = min(1.0, self.l1_coherence / 10.0)

        # Combine metrics
        score = (
            0.3 * normalized_l1 +
            0.3 * min(1.0, self.relative_entropy) +
            0.2 * self.purity +
            0.2 * (1.0 - min(1.0, self.decay_rate))
        )

        return max(0.0, min(1.0, score))


class CoherenceCalculator:
    """
    Calculator for quantum coherence metrics.

    Provides various measures of quantum coherence from
    state vectors and density matrices.
    """

    @staticmethod
    def l1_coherence(rho: DensityMatrix) -> float:
        """
        Calculate L1-norm of coherence.

        C_l1(ρ) = Σ_{i≠j} |ρ_ij|

        This is the sum of absolute values of off-diagonal elements.
        """
        off_diag = rho.matrix - np.diag(np.diag(rho.matrix))
        return float(np.sum(np.abs(off_diag)))

    @staticmethod
    def relative_entropy_coherence(rho: DensityMatrix) -> float:
        """
        Calculate relative entropy of coherence.

        C_re(ρ) = S(ρ_diag) - S(ρ)

        Where S is von Neumann entropy and ρ_diag is diagonal part.
        """
        # Diagonal state entropy
        diag = np.real(np.diag(rho.matrix))
        diag = diag[diag > 1e-10]  # Filter zeros
        s_diag = -float(np.sum(diag * np.log2(diag))) if len(diag) > 0 else 0

        # Full state entropy
        s_full = rho.entropy()

        return max(0.0, s_diag - s_full)

    @staticmethod
    def robustness_of_coherence(rho: DensityMatrix) -> float:
        """
        Calculate robustness of coherence.

        Measures minimum mixing with incoherent state to destroy coherence.
        """
        # Simplified: use off-diagonal norm as proxy
        off_diag = rho.matrix - np.diag(np.diag(rho.matrix))
        norm = np.linalg.norm(off_diag, ord=2)
        return float(norm)

    @staticmethod
    def coherence_number(state: QuantumStateVector) -> int:
        """
        Count number of significant superposition terms.

        Coherence number indicates how many basis states
        have non-negligible probability amplitude.
        """
        probs = np.abs(state.amplitudes) ** 2
        return int(np.sum(probs > 1e-6))

    @staticmethod
    def phase_coherence(state: QuantumStateVector) -> float:
        """
        Measure phase coherence across superposition.

        High phase coherence indicates stable interference patterns.
        """
        # Get phases of non-zero amplitudes
        non_zero = state.amplitudes[np.abs(state.amplitudes) > 1e-10]
        if len(non_zero) < 2:
            return 1.0

        phases = np.angle(non_zero)

        # Phase coherence: how aligned are the phases?
        # Use circular variance
        mean_direction = np.mean(np.exp(1j * phases))
        coherence = float(np.abs(mean_direction))

        return coherence

    @staticmethod
    def measure_all(
        state: QuantumStateVector
    ) -> CoherenceMetrics:
        """Calculate all coherence metrics for a state"""
        rho = state.to_density_matrix()

        return CoherenceMetrics(
            l1_coherence=CoherenceCalculator.l1_coherence(rho),
            relative_entropy=CoherenceCalculator.relative_entropy_coherence(rho),
            purity=rho.purity(),
            entanglement_entropy=state.entropy() if state.num_qubits > 1 else 0.0,
        )


class DecoherenceModel:
    """
    Models quantum decoherence processes.

    Simulates environmental effects that destroy quantum coherence,
    modeling the transition from quantum to classical behavior.
    """

    def __init__(
        self,
        t1: float = 1000.0,  # Relaxation time (arbitrary units)
        t2: float = 500.0,   # Dephasing time
        temperature: float = 0.0  # Effective temperature
    ):
        """
        Initialize decoherence model.

        Args:
            t1: Amplitude damping time (T1 relaxation)
            t2: Dephasing time (T2 coherence)
            temperature: Effective temperature for thermal effects
        """
        self.t1 = t1
        self.t2 = t2
        self.temperature = temperature

    def apply_dephasing(
        self,
        rho: DensityMatrix,
        time_delta: float
    ) -> DensityMatrix:
        """
        Apply pure dephasing channel.

        Destroys off-diagonal elements while preserving populations.
        """
        gamma = 1.0 - math.exp(-time_delta / self.t2)

        dim = 2 ** rho.num_qubits
        new_matrix = rho.matrix.copy()

        # Decay off-diagonal elements
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    new_matrix[i, j] *= (1 - gamma)

        return DensityMatrix(num_qubits=rho.num_qubits, matrix=new_matrix)

    def apply_amplitude_damping(
        self,
        rho: DensityMatrix,
        time_delta: float
    ) -> DensityMatrix:
        """
        Apply amplitude damping channel.

        Models energy loss to environment (decay to ground state).
        """
        gamma = 1.0 - math.exp(-time_delta / self.t1)

        dim = 2 ** rho.num_qubits
        new_matrix = np.zeros((dim, dim), dtype=complex)

        # Kraus operators for single qubit amplitude damping
        # E0 = [[1, 0], [0, sqrt(1-gamma)]]
        # E1 = [[0, sqrt(gamma)], [0, 0]]

        # Simplified: apply to each qubit independently
        for i in range(dim):
            for j in range(dim):
                # Count number of 1s in each index
                n_excitations_i = bin(i).count('1')
                n_excitations_j = bin(j).count('1')

                # Diagonal elements: transition probability
                if i == j:
                    # Population transfer from excited to ground
                    new_matrix[i, j] = rho.matrix[i, j] * \
                        ((1 - gamma) ** n_excitations_i)

                    # Add population from higher excited states
                    for k in range(dim):
                        if k != i:
                            # Probability of k decaying to i
                            n_k = bin(k).count('1')
                            if n_k > n_excitations_i:
                                decay_prob = gamma ** (n_k - n_excitations_i) * \
                                    (1 - gamma) ** n_excitations_i
                                new_matrix[i, j] += rho.matrix[k, k] * decay_prob
                else:
                    # Off-diagonal: coherence decay
                    avg_excitations = (n_excitations_i + n_excitations_j) / 2
                    new_matrix[i, j] = rho.matrix[i, j] * \
                        ((1 - gamma) ** avg_excitations)

        # Ensure trace preservation
        rho_new = DensityMatrix(num_qubits=rho.num_qubits, matrix=new_matrix)
        rho_new.normalize()

        return rho_new

    def apply_depolarizing(
        self,
        rho: DensityMatrix,
        time_delta: float,
        strength: float = 0.01
    ) -> DensityMatrix:
        """
        Apply depolarizing channel.

        Mixes state towards maximally mixed state.
        """
        p = 1.0 - math.exp(-strength * time_delta)

        mixed = DensityMatrix.maximally_mixed(rho.num_qubits)
        new_matrix = (1 - p) * rho.matrix + p * mixed.matrix

        return DensityMatrix(num_qubits=rho.num_qubits, matrix=new_matrix)

    def apply_thermal(
        self,
        rho: DensityMatrix,
        time_delta: float
    ) -> DensityMatrix:
        """
        Apply thermal equilibration.

        Evolves state towards thermal equilibrium.
        """
        if self.temperature <= 0:
            return self.apply_amplitude_damping(rho, time_delta)

        # Thermal state
        beta = 1.0 / self.temperature
        dim = 2 ** rho.num_qubits

        # Energy eigenvalues (simplified: energy proportional to number of 1s)
        energies = np.array([bin(i).count('1') for i in range(dim)])
        boltzmann = np.exp(-beta * energies)
        partition = np.sum(boltzmann)
        thermal_pop = boltzmann / partition

        thermal_matrix = np.diag(thermal_pop.astype(complex))
        thermal_rho = DensityMatrix(num_qubits=rho.num_qubits, matrix=thermal_matrix)

        # Mix towards thermal state
        rate = 1.0 - math.exp(-time_delta / self.t1)
        new_matrix = (1 - rate) * rho.matrix + rate * thermal_rho.matrix

        return DensityMatrix(num_qubits=rho.num_qubits, matrix=new_matrix)

    def evolve(
        self,
        state: QuantumStateVector,
        time_delta: float,
        channels: Optional[List[DecoherenceChannel]] = None
    ) -> DensityMatrix:
        """
        Evolve state through decoherence channels.

        Args:
            state: Initial pure state
            time_delta: Time evolution duration
            channels: List of channels to apply (default: dephasing + damping)

        Returns:
            Evolved density matrix
        """
        if channels is None:
            channels = [DecoherenceChannel.DEPHASING,
                        DecoherenceChannel.AMPLITUDE_DAMPING]

        rho = state.to_density_matrix()

        for channel in channels:
            if channel == DecoherenceChannel.DEPHASING:
                rho = self.apply_dephasing(rho, time_delta)
            elif channel == DecoherenceChannel.AMPLITUDE_DAMPING:
                rho = self.apply_amplitude_damping(rho, time_delta)
            elif channel == DecoherenceChannel.DEPOLARIZING:
                rho = self.apply_depolarizing(rho, time_delta)
            elif channel == DecoherenceChannel.THERMAL:
                rho = self.apply_thermal(rho, time_delta)

        return rho


class CoherenceTracker:
    """
    Tracks coherence over time for consciousness monitoring.

    Maintains history of coherence metrics and detects
    coherence collapse events.
    """

    def __init__(
        self,
        history_size: int = 1000,
        collapse_threshold: float = 0.1
    ):
        """
        Initialize coherence tracker.

        Args:
            history_size: Number of measurements to retain
            collapse_threshold: Coherence level below which collapse is detected
        """
        self.history_size = history_size
        self.collapse_threshold = collapse_threshold
        self._history: List[CoherenceMetrics] = []
        self._collapse_events: List[Tuple[float, CoherenceMetrics]] = []

    def record(self, metrics: CoherenceMetrics):
        """Record coherence measurement"""
        self._history.append(metrics)

        # Check for collapse
        if metrics.overall_coherence() < self.collapse_threshold:
            self._collapse_events.append((time.time(), metrics))

        # Trim history
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

    def get_trend(self, window: int = 10) -> float:
        """
        Get coherence trend over recent window.

        Returns positive for increasing, negative for decreasing.
        """
        if len(self._history) < window:
            return 0.0

        recent = self._history[-window:]
        scores = [m.overall_coherence() for m in recent]

        # Linear regression slope
        x = np.arange(window)
        slope = np.polyfit(x, scores, 1)[0]

        return float(slope)

    def get_average_coherence(self, window: int = 100) -> float:
        """Get average coherence over window"""
        if not self._history:
            return 0.0

        recent = self._history[-window:]
        return float(np.mean([m.overall_coherence() for m in recent]))

    def get_stability(self, window: int = 100) -> float:
        """
        Get coherence stability (low variance = stable).

        Returns 0-1 where 1 is perfectly stable.
        """
        if len(self._history) < 2:
            return 1.0

        recent = self._history[-window:]
        scores = [m.overall_coherence() for m in recent]
        variance = np.var(scores)

        # Convert variance to stability score
        stability = 1.0 / (1.0 + 10 * variance)

        return float(stability)

    def time_since_collapse(self) -> float:
        """Get time since last collapse event"""
        if not self._collapse_events:
            return float('inf')

        return time.time() - self._collapse_events[-1][0]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coherence statistics"""
        if not self._history:
            return {
                "measurements": 0,
                "collapse_events": 0
            }

        scores = [m.overall_coherence() for m in self._history]

        return {
            "measurements": len(self._history),
            "current_coherence": self._history[-1].overall_coherence(),
            "average_coherence": float(np.mean(scores)),
            "max_coherence": float(np.max(scores)),
            "min_coherence": float(np.min(scores)),
            "coherence_variance": float(np.var(scores)),
            "trend": self.get_trend(),
            "stability": self.get_stability(),
            "collapse_events": len(self._collapse_events),
            "time_since_collapse": self.time_since_collapse()
        }


__all__ = [
    'DecoherenceChannel',
    'CoherenceMetrics',
    'CoherenceCalculator',
    'DecoherenceModel',
    'CoherenceTracker',
]
