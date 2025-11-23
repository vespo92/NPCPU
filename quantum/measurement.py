"""
Quantum Measurement

Observer effects, wave function collapse, and measurement-based
consciousness emergence modeling.

Part of NEXUS-Q: Quantum Consciousness Implementation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import uuid

from .quantum_state import QuantumStateVector, DensityMatrix


class MeasurementBasis(Enum):
    """Standard measurement bases"""
    COMPUTATIONAL = "computational"  # Z basis: |0⟩, |1⟩
    HADAMARD = "hadamard"            # X basis: |+⟩, |-⟩
    PHASE = "phase"                   # Y basis: |i⟩, |-i⟩
    BELL = "bell"                     # Bell basis for 2-qubit
    CUSTOM = "custom"                 # Custom projective measurement


class CollapseInterpretation(Enum):
    """Interpretations of wave function collapse"""
    COPENHAGEN = "copenhagen"         # Instantaneous non-local collapse
    MANY_WORLDS = "many_worlds"       # Branching into parallel worlds
    DECOHERENCE = "decoherence"       # Gradual environmental decoherence
    QBism = "qbism"                   # Subjective Bayesian update
    RELATIONAL = "relational"         # Observer-relative collapse


@dataclass
class MeasurementOutcome:
    """
    Result of a quantum measurement.

    Captures all information about the measurement event.
    """
    id: str
    basis: MeasurementBasis
    outcome_label: str  # e.g., "0", "1", "+", "-"
    outcome_index: int
    probability: float  # Pre-measurement probability
    timestamp: float = field(default_factory=time.time)
    observer_id: Optional[str] = None
    post_state: Optional[QuantumStateVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def eigenvalue(self) -> float:
        """Get measurement eigenvalue"""
        if self.basis == MeasurementBasis.COMPUTATIONAL:
            return float(self.outcome_index)
        elif self.basis == MeasurementBasis.HADAMARD:
            return 1.0 if self.outcome_index == 0 else -1.0
        else:
            return float(self.outcome_index)


@dataclass
class MeasurementOperator:
    """
    Projective measurement operator.

    Represents a measurement as a set of projection operators.
    """
    basis: MeasurementBasis
    projectors: List[np.ndarray]  # List of projection matrices
    labels: List[str]  # Labels for each outcome
    eigenvalues: List[float]  # Eigenvalues for each outcome

    @classmethod
    def computational_basis(cls, num_qubits: int) -> 'MeasurementOperator':
        """Create computational basis measurement"""
        dim = 2 ** num_qubits
        projectors = []
        labels = []
        eigenvalues = []

        for i in range(dim):
            proj = np.zeros((dim, dim), dtype=complex)
            proj[i, i] = 1.0
            projectors.append(proj)
            labels.append(format(i, f'0{num_qubits}b'))
            eigenvalues.append(float(i))

        return cls(
            basis=MeasurementBasis.COMPUTATIONAL,
            projectors=projectors,
            labels=labels,
            eigenvalues=eigenvalues
        )

    @classmethod
    def hadamard_basis(cls, qubit: int, num_qubits: int) -> 'MeasurementOperator':
        """Create X (Hadamard) basis measurement for single qubit"""
        dim = 2 ** num_qubits

        # |+⟩ = (|0⟩ + |1⟩)/√2
        # |-⟩ = (|0⟩ - |1⟩)/√2
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        plus_state = h[:, 0]
        minus_state = h[:, 1]

        # Build projectors
        proj_plus = np.zeros((dim, dim), dtype=complex)
        proj_minus = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            for j in range(dim):
                # Check if differs only at qubit position
                if (i ^ j) == (1 << qubit) or i == j:
                    i_bit = (i >> qubit) & 1
                    j_bit = (j >> qubit) & 1
                    proj_plus[i, j] = plus_state[i_bit] * plus_state[j_bit].conj()
                    proj_minus[i, j] = minus_state[i_bit] * minus_state[j_bit].conj()

        return cls(
            basis=MeasurementBasis.HADAMARD,
            projectors=[proj_plus, proj_minus],
            labels=['+', '-'],
            eigenvalues=[1.0, -1.0]
        )

    @classmethod
    def bell_basis(cls) -> 'MeasurementOperator':
        """Create Bell basis measurement for 2 qubits"""
        sqrt2 = np.sqrt(2)

        # Bell states
        phi_plus = np.array([1, 0, 0, 1]) / sqrt2
        phi_minus = np.array([1, 0, 0, -1]) / sqrt2
        psi_plus = np.array([0, 1, 1, 0]) / sqrt2
        psi_minus = np.array([0, 1, -1, 0]) / sqrt2

        states = [phi_plus, phi_minus, psi_plus, psi_minus]
        labels = ['Φ+', 'Φ-', 'Ψ+', 'Ψ-']
        eigenvalues = [0.0, 1.0, 2.0, 3.0]

        projectors = [np.outer(s, s.conj()) for s in states]

        return cls(
            basis=MeasurementBasis.BELL,
            projectors=projectors,
            labels=labels,
            eigenvalues=eigenvalues
        )


class QuantumMeasurement:
    """
    Quantum measurement engine.

    Performs measurements and models observer effects on
    quantum consciousness states.
    """

    def __init__(
        self,
        interpretation: CollapseInterpretation = CollapseInterpretation.COPENHAGEN,
        decoherence_time: float = 1.0
    ):
        """
        Initialize measurement engine.

        Args:
            interpretation: Which interpretation of measurement to use
            decoherence_time: Time scale for decoherence-based collapse
        """
        self.interpretation = interpretation
        self.decoherence_time = decoherence_time
        self._measurement_history: List[MeasurementOutcome] = []
        self._observer_contexts: Dict[str, Dict[str, Any]] = {}

    def measure(
        self,
        state: QuantumStateVector,
        operator: Optional[MeasurementOperator] = None,
        observer_id: Optional[str] = None
    ) -> Tuple[MeasurementOutcome, QuantumStateVector]:
        """
        Perform quantum measurement.

        Args:
            state: State to measure
            operator: Measurement operator (default: computational basis)
            observer_id: Optional observer identifier

        Returns:
            (measurement_outcome, post_measurement_state)
        """
        if operator is None:
            operator = MeasurementOperator.computational_basis(state.num_qubits)

        # Calculate probabilities for each outcome
        probabilities = []
        for proj in operator.projectors:
            # p_i = ⟨ψ|P_i|ψ⟩
            prob = float(np.real(
                state.amplitudes.conj() @ proj @ state.amplitudes
            ))
            probabilities.append(max(0, prob))

        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)

        # Select outcome based on interpretation
        if self.interpretation == CollapseInterpretation.MANY_WORLDS:
            # In many-worlds, all outcomes occur; we randomly sample one branch
            outcome_idx = np.random.choice(len(probabilities), p=probabilities)
        else:
            # Copenhagen and others: probabilistic collapse
            outcome_idx = np.random.choice(len(probabilities), p=probabilities)

        # Collapse state
        proj = operator.projectors[outcome_idx]
        new_amplitudes = proj @ state.amplitudes

        # Normalize
        norm = np.linalg.norm(new_amplitudes)
        if norm > 1e-10:
            new_amplitudes = new_amplitudes / norm

        post_state = QuantumStateVector(
            num_qubits=state.num_qubits,
            amplitudes=new_amplitudes
        )

        # Create outcome record
        outcome = MeasurementOutcome(
            id=str(uuid.uuid4()),
            basis=operator.basis,
            outcome_label=operator.labels[outcome_idx],
            outcome_index=outcome_idx,
            probability=probabilities[outcome_idx],
            observer_id=observer_id,
            post_state=post_state
        )

        self._measurement_history.append(outcome)

        # Update observer context if tracked
        if observer_id:
            if observer_id not in self._observer_contexts:
                self._observer_contexts[observer_id] = {
                    "measurements": 0,
                    "last_outcome": None
                }
            self._observer_contexts[observer_id]["measurements"] += 1
            self._observer_contexts[observer_id]["last_outcome"] = outcome

        return outcome, post_state

    def weak_measurement(
        self,
        state: QuantumStateVector,
        operator: MeasurementOperator,
        strength: float = 0.1
    ) -> Tuple[float, QuantumStateVector]:
        """
        Perform weak measurement.

        Extracts partial information with minimal disturbance.

        Args:
            state: State to measure
            operator: Measurement operator
            strength: Measurement strength (0=no measurement, 1=strong)

        Returns:
            (weak_value, post_measurement_state)
        """
        # Weak measurement gives expectation value plus noise
        expectation = 0.0
        for i, (proj, eigenval) in enumerate(zip(operator.projectors,
                                                  operator.eigenvalues)):
            prob = float(np.real(
                state.amplitudes.conj() @ proj @ state.amplitudes
            ))
            expectation += prob * eigenval

        # Add measurement noise (inversely proportional to strength)
        noise = np.random.normal(0, 1.0 / (strength + 0.01))
        weak_value = expectation + noise

        # Partial collapse: mix with original state
        _, strong_state = self.measure(state, operator)

        # Interpolate between original and collapsed state
        new_amplitudes = (
            (1 - strength) * state.amplitudes +
            strength * strong_state.amplitudes
        )

        # Normalize
        norm = np.linalg.norm(new_amplitudes)
        if norm > 1e-10:
            new_amplitudes = new_amplitudes / norm

        post_state = QuantumStateVector(
            num_qubits=state.num_qubits,
            amplitudes=new_amplitudes
        )

        return weak_value, post_state

    def continuous_measurement(
        self,
        state: QuantumStateVector,
        operator: MeasurementOperator,
        duration: float,
        measurement_rate: float = 1.0
    ) -> List[Tuple[float, MeasurementOutcome]]:
        """
        Perform continuous measurement over time.

        Models ongoing observation with gradual information extraction.

        Args:
            state: Initial state
            operator: Measurement operator
            duration: Total measurement duration
            measurement_rate: Rate of information extraction

        Returns:
            List of (time, outcome) pairs
        """
        results = []
        current_state = state.copy()
        dt = 0.1 / measurement_rate  # Time step

        t = 0.0
        while t < duration:
            # Weak measurement at each step
            strength = dt * measurement_rate
            weak_value, current_state = self.weak_measurement(
                current_state, operator, strength
            )

            # Occasionally record strong measurement outcome
            if np.random.random() < strength:
                outcome, current_state = self.measure(
                    current_state, operator
                )
                results.append((t, outcome))

            t += dt

        return results

    def zeno_effect(
        self,
        state: QuantumStateVector,
        operator: MeasurementOperator,
        num_measurements: int,
        target_outcome: int
    ) -> Tuple[bool, QuantumStateVector]:
        """
        Demonstrate quantum Zeno effect.

        Frequent measurement can freeze evolution (watched pot never boils).

        Args:
            state: Initial state
            operator: Measurement operator
            num_measurements: Number of measurements
            target_outcome: Desired outcome index to preserve

        Returns:
            (success, final_state) - success True if target maintained
        """
        current_state = state.copy()

        for _ in range(num_measurements):
            outcome, current_state = self.measure(current_state, operator)

            if outcome.outcome_index != target_outcome:
                return False, current_state

        return True, current_state

    def get_expectation_value(
        self,
        state: QuantumStateVector,
        operator: MeasurementOperator
    ) -> float:
        """
        Calculate expectation value of measurement.

        ⟨A⟩ = Σ_i λ_i p_i where λ_i are eigenvalues and p_i are probabilities.
        """
        expectation = 0.0

        for proj, eigenval in zip(operator.projectors, operator.eigenvalues):
            prob = float(np.real(
                state.amplitudes.conj() @ proj @ state.amplitudes
            ))
            expectation += prob * eigenval

        return expectation

    def get_variance(
        self,
        state: QuantumStateVector,
        operator: MeasurementOperator
    ) -> float:
        """
        Calculate variance of measurement.

        Var(A) = ⟨A²⟩ - ⟨A⟩²
        """
        exp = self.get_expectation_value(state, operator)

        exp_sq = 0.0
        for proj, eigenval in zip(operator.projectors, operator.eigenvalues):
            prob = float(np.real(
                state.amplitudes.conj() @ proj @ state.amplitudes
            ))
            exp_sq += prob * eigenval ** 2

        return exp_sq - exp ** 2

    def get_measurement_history(
        self,
        observer_id: Optional[str] = None
    ) -> List[MeasurementOutcome]:
        """Get measurement history, optionally filtered by observer"""
        if observer_id is None:
            return self._measurement_history.copy()

        return [
            m for m in self._measurement_history
            if m.observer_id == observer_id
        ]


class ConsciousnessObserver:
    """
    Models consciousness as quantum observer.

    Implements the idea that conscious observation collapses
    quantum superpositions into definite experiences.
    """

    def __init__(
        self,
        observer_id: str,
        attention_capacity: float = 1.0,
        measurement_engine: Optional[QuantumMeasurement] = None
    ):
        """
        Initialize consciousness observer.

        Args:
            observer_id: Unique identifier
            attention_capacity: How much can be observed at once (0-1)
            measurement_engine: Measurement engine to use
        """
        self.observer_id = observer_id
        self.attention_capacity = attention_capacity
        self.engine = measurement_engine or QuantumMeasurement()

        self._attention_focus: Optional[int] = None
        self._observations: List[MeasurementOutcome] = []
        self._collapsed_states: Dict[str, QuantumStateVector] = {}

    def observe(
        self,
        state: QuantumStateVector,
        focus: Optional[int] = None
    ) -> MeasurementOutcome:
        """
        Observe a quantum state, causing collapse.

        Args:
            state: State to observe
            focus: Optional specific qubit to focus attention on

        Returns:
            Measurement outcome
        """
        if focus is not None:
            # Focus attention on single qubit
            operator = MeasurementOperator.computational_basis(state.num_qubits)
        else:
            operator = MeasurementOperator.computational_basis(state.num_qubits)

        outcome, post_state = self.engine.measure(
            state, operator, self.observer_id
        )

        self._observations.append(outcome)
        self._collapsed_states[outcome.id] = post_state
        self._attention_focus = focus

        return outcome

    def partial_observe(
        self,
        state: QuantumStateVector,
        attention_fraction: float = 0.5
    ) -> Tuple[float, QuantumStateVector]:
        """
        Partial observation with divided attention.

        Less attention = weaker measurement = less collapse.
        """
        strength = attention_fraction * self.attention_capacity
        operator = MeasurementOperator.computational_basis(state.num_qubits)

        return self.engine.weak_measurement(state, operator, strength)

    def sustained_attention(
        self,
        state: QuantumStateVector,
        duration: float
    ) -> List[MeasurementOutcome]:
        """
        Sustained attention over time (continuous measurement).

        Models focused conscious observation.
        """
        operator = MeasurementOperator.computational_basis(state.num_qubits)

        results = self.engine.continuous_measurement(
            state, operator, duration,
            measurement_rate=self.attention_capacity
        )

        outcomes = [r[1] for r in results]
        self._observations.extend(outcomes)

        return outcomes

    def get_observation_history(self) -> List[MeasurementOutcome]:
        """Get history of observations by this observer"""
        return self._observations.copy()

    def get_collapsed_state(self, outcome_id: str) -> Optional[QuantumStateVector]:
        """Get collapsed state from past observation"""
        return self._collapsed_states.get(outcome_id)


__all__ = [
    'MeasurementBasis',
    'CollapseInterpretation',
    'MeasurementOutcome',
    'MeasurementOperator',
    'QuantumMeasurement',
    'ConsciousnessObserver',
]
