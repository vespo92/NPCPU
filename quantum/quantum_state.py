"""
Quantum State Management

Advanced quantum state representation with superposition vectors,
density matrices, and state transformations for consciousness modeling.

Part of NEXUS-Q: Quantum Consciousness Implementation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import cmath


class StateType(Enum):
    """Types of quantum states"""
    PURE = "pure"           # Single state vector
    MIXED = "mixed"         # Statistical mixture (density matrix)
    ENTANGLED = "entangled" # Multi-qubit entangled state
    COHERENT = "coherent"   # Coherent superposition


@dataclass
class QuantumAmplitude:
    """
    Complex amplitude for quantum state component.

    Represents probability amplitude with magnitude and phase.
    Probability = |amplitude|^2
    """
    real: float = 1.0
    imag: float = 0.0

    @property
    def complex(self) -> complex:
        return complex(self.real, self.imag)

    @property
    def probability(self) -> float:
        """Born rule: probability = |amplitude|^2"""
        return self.real ** 2 + self.imag ** 2

    @property
    def phase(self) -> float:
        """Phase angle in radians"""
        return cmath.phase(self.complex)

    @property
    def magnitude(self) -> float:
        """Magnitude of amplitude"""
        return abs(self.complex)

    def normalize(self, total_prob: float) -> 'QuantumAmplitude':
        """Normalize amplitude given total probability"""
        if total_prob <= 0:
            return QuantumAmplitude(0, 0)
        factor = 1.0 / math.sqrt(total_prob)
        return QuantumAmplitude(self.real * factor, self.imag * factor)


@dataclass
class BasisState:
    """
    A computational basis state.

    Represents a single definite state in the computational basis.
    """
    label: str  # e.g., "00", "01", "10", "11" for 2-qubit system
    index: int  # Integer representation
    num_qubits: int

    @classmethod
    def from_index(cls, index: int, num_qubits: int) -> 'BasisState':
        """Create basis state from integer index"""
        label = format(index, f'0{num_qubits}b')
        return cls(label=label, index=index, num_qubits=num_qubits)

    @classmethod
    def from_label(cls, label: str) -> 'BasisState':
        """Create basis state from binary label"""
        return cls(
            label=label,
            index=int(label, 2),
            num_qubits=len(label)
        )

    def get_qubit(self, qubit_index: int) -> int:
        """Get value of specific qubit (0 or 1)"""
        return (self.index >> qubit_index) & 1

    def flip_qubit(self, qubit_index: int) -> 'BasisState':
        """Return new state with specified qubit flipped"""
        new_index = self.index ^ (1 << qubit_index)
        return BasisState.from_index(new_index, self.num_qubits)


@dataclass
class QuantumStateVector:
    """
    Pure quantum state represented as state vector.

    |ψ⟩ = Σ αᵢ|i⟩ where αᵢ are complex amplitudes
    """
    num_qubits: int
    amplitudes: np.ndarray  # Complex array of length 2^num_qubits
    state_type: StateType = StateType.PURE
    labels: Optional[Dict[int, str]] = None  # Custom labels for states

    def __post_init__(self):
        """Validate and initialize state"""
        expected_dim = 2 ** self.num_qubits
        if len(self.amplitudes) != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} amplitudes for {self.num_qubits} qubits, "
                f"got {len(self.amplitudes)}"
            )
        # Ensure complex type
        self.amplitudes = self.amplitudes.astype(complex)

    @classmethod
    def zero_state(cls, num_qubits: int) -> 'QuantumStateVector':
        """Create |0...0⟩ state"""
        amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
        amplitudes[0] = 1.0
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)

    @classmethod
    def uniform_superposition(cls, num_qubits: int) -> 'QuantumStateVector':
        """Create uniform superposition over all basis states"""
        dim = 2 ** num_qubits
        amplitude = 1.0 / math.sqrt(dim)
        amplitudes = np.full(dim, amplitude, dtype=complex)
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)

    @classmethod
    def from_probabilities(
        cls,
        num_qubits: int,
        probabilities: Dict[str, float]
    ) -> 'QuantumStateVector':
        """Create state from probability distribution (phases set to 0)"""
        dim = 2 ** num_qubits
        amplitudes = np.zeros(dim, dtype=complex)

        for label, prob in probabilities.items():
            index = int(label, 2) if isinstance(label, str) else label
            amplitudes[index] = math.sqrt(prob)

        state = cls(num_qubits=num_qubits, amplitudes=amplitudes)
        state.normalize()
        return state

    def normalize(self):
        """Normalize state vector to unit length"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-10:
            self.amplitudes = self.amplitudes / norm

    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if state is normalized"""
        norm_sq = np.sum(np.abs(self.amplitudes) ** 2)
        return abs(norm_sq - 1.0) < tolerance

    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities for each basis state"""
        probs = np.abs(self.amplitudes) ** 2
        return {
            format(i, f'0{self.num_qubits}b'): float(p)
            for i, p in enumerate(probs)
            if p > 1e-10
        }

    def get_amplitude(self, basis_state: Union[str, int]) -> complex:
        """Get amplitude for specific basis state"""
        if isinstance(basis_state, str):
            index = int(basis_state, 2)
        else:
            index = basis_state
        return self.amplitudes[index]

    def set_amplitude(self, basis_state: Union[str, int], amplitude: complex):
        """Set amplitude for specific basis state"""
        if isinstance(basis_state, str):
            index = int(basis_state, 2)
        else:
            index = basis_state
        self.amplitudes[index] = amplitude

    def inner_product(self, other: 'QuantumStateVector') -> complex:
        """Compute ⟨self|other⟩"""
        if self.num_qubits != other.num_qubits:
            raise ValueError("States must have same number of qubits")
        return np.vdot(self.amplitudes, other.amplitudes)

    def fidelity(self, other: 'QuantumStateVector') -> float:
        """Compute fidelity |⟨self|other⟩|²"""
        return abs(self.inner_product(other)) ** 2

    def tensor_product(self, other: 'QuantumStateVector') -> 'QuantumStateVector':
        """Compute |self⟩ ⊗ |other⟩"""
        new_amplitudes = np.kron(self.amplitudes, other.amplitudes)
        return QuantumStateVector(
            num_qubits=self.num_qubits + other.num_qubits,
            amplitudes=new_amplitudes
        )

    def partial_trace(self, traced_qubits: List[int]) -> 'DensityMatrix':
        """
        Compute partial trace over specified qubits.

        Returns reduced density matrix for remaining qubits.
        """
        # Convert to density matrix first
        rho = self.to_density_matrix()
        return rho.partial_trace(traced_qubits)

    def to_density_matrix(self) -> 'DensityMatrix':
        """Convert to density matrix representation"""
        rho = np.outer(self.amplitudes, np.conj(self.amplitudes))
        return DensityMatrix(num_qubits=self.num_qubits, matrix=rho)

    def measure(self, collapse: bool = True) -> Tuple[str, 'QuantumStateVector']:
        """
        Perform measurement in computational basis.

        Args:
            collapse: Whether to collapse state after measurement

        Returns:
            (outcome, post_measurement_state)
        """
        probs = np.abs(self.amplitudes) ** 2
        probs = probs / np.sum(probs)  # Normalize

        outcome = np.random.choice(len(probs), p=probs)
        outcome_str = format(outcome, f'0{self.num_qubits}b')

        if collapse:
            # Collapse to measured state
            new_amplitudes = np.zeros_like(self.amplitudes)
            new_amplitudes[outcome] = 1.0
            return outcome_str, QuantumStateVector(
                num_qubits=self.num_qubits,
                amplitudes=new_amplitudes
            )
        else:
            return outcome_str, self

    def measure_qubit(self, qubit: int) -> Tuple[int, 'QuantumStateVector']:
        """
        Measure single qubit, collapsing state.

        Returns (outcome, post_measurement_state)
        """
        dim = 2 ** self.num_qubits

        # Calculate probability of qubit being 1
        prob_1 = 0.0
        for i in range(dim):
            if (i >> qubit) & 1 == 1:
                prob_1 += np.abs(self.amplitudes[i]) ** 2

        # Perform measurement
        outcome = 1 if np.random.random() < prob_1 else 0

        # Collapse state
        new_amplitudes = np.zeros(dim, dtype=complex)
        norm = 0.0

        for i in range(dim):
            if (i >> qubit) & 1 == outcome:
                new_amplitudes[i] = self.amplitudes[i]
                norm += np.abs(self.amplitudes[i]) ** 2

        if norm > 0:
            new_amplitudes = new_amplitudes / np.sqrt(norm)

        return outcome, QuantumStateVector(
            num_qubits=self.num_qubits,
            amplitudes=new_amplitudes
        )

    def entropy(self) -> float:
        """
        Von Neumann entropy (0 for pure states).

        For pure states, entropy is always 0.
        """
        return 0.0  # Pure states have zero entropy

    def purity(self) -> float:
        """
        Purity Tr(ρ²) - always 1 for pure states.
        """
        return 1.0

    def is_entangled(self) -> bool:
        """
        Check if multi-qubit state is entangled.

        Uses Schmidt decomposition approach for bipartite check.
        """
        if self.num_qubits < 2:
            return False

        # Simple check: try to factor as tensor product
        # If purity of partial trace < 1, state is entangled
        rho_reduced = self.partial_trace([0])
        return rho_reduced.purity() < 1.0 - 1e-10

    def copy(self) -> 'QuantumStateVector':
        """Create a copy of this state"""
        return QuantumStateVector(
            num_qubits=self.num_qubits,
            amplitudes=self.amplitudes.copy(),
            state_type=self.state_type,
            labels=self.labels.copy() if self.labels else None
        )


@dataclass
class DensityMatrix:
    """
    Density matrix representation of quantum state.

    ρ = Σ pᵢ |ψᵢ⟩⟨ψᵢ| for mixed states
    ρ = |ψ⟩⟨ψ| for pure states
    """
    num_qubits: int
    matrix: np.ndarray  # Complex matrix of size 2^n × 2^n

    def __post_init__(self):
        """Validate density matrix"""
        dim = 2 ** self.num_qubits
        if self.matrix.shape != (dim, dim):
            raise ValueError(
                f"Expected {dim}×{dim} matrix for {self.num_qubits} qubits"
            )
        self.matrix = self.matrix.astype(complex)

    @classmethod
    def from_state_vector(cls, state: QuantumStateVector) -> 'DensityMatrix':
        """Create density matrix from pure state"""
        return state.to_density_matrix()

    @classmethod
    def mixed_state(
        cls,
        states: List[QuantumStateVector],
        probabilities: List[float]
    ) -> 'DensityMatrix':
        """Create mixed state from ensemble"""
        if len(states) != len(probabilities):
            raise ValueError("States and probabilities must have same length")

        num_qubits = states[0].num_qubits
        dim = 2 ** num_qubits
        rho = np.zeros((dim, dim), dtype=complex)

        for state, prob in zip(states, probabilities):
            rho += prob * np.outer(state.amplitudes, np.conj(state.amplitudes))

        return cls(num_qubits=num_qubits, matrix=rho)

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> 'DensityMatrix':
        """Create maximally mixed state I/d"""
        dim = 2 ** num_qubits
        matrix = np.eye(dim, dtype=complex) / dim
        return cls(num_qubits=num_qubits, matrix=matrix)

    def trace(self) -> float:
        """Compute trace (should be 1 for valid density matrix)"""
        return float(np.real(np.trace(self.matrix)))

    def normalize(self):
        """Normalize density matrix to unit trace"""
        tr = self.trace()
        if tr > 1e-10:
            self.matrix = self.matrix / tr

    def purity(self) -> float:
        """Compute purity Tr(ρ²)"""
        return float(np.real(np.trace(self.matrix @ self.matrix)))

    def is_pure(self, tolerance: float = 1e-10) -> bool:
        """Check if state is pure (purity ≈ 1)"""
        return abs(self.purity() - 1.0) < tolerance

    def entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ)
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter small/negative
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def get_probabilities(self) -> Dict[str, float]:
        """Get diagonal elements as measurement probabilities"""
        diag = np.real(np.diag(self.matrix))
        return {
            format(i, f'0{self.num_qubits}b'): float(p)
            for i, p in enumerate(diag)
            if p > 1e-10
        }

    def partial_trace(self, traced_qubits: List[int]) -> 'DensityMatrix':
        """
        Compute partial trace over specified qubits.

        Args:
            traced_qubits: List of qubit indices to trace out

        Returns:
            Reduced density matrix
        """
        n = self.num_qubits
        remaining = [i for i in range(n) if i not in traced_qubits]
        n_remaining = len(remaining)

        if n_remaining == 0:
            # Tracing all qubits returns scalar (trace)
            return DensityMatrix(
                num_qubits=0,
                matrix=np.array([[self.trace()]], dtype=complex)
            )

        dim_remaining = 2 ** n_remaining
        reduced = np.zeros((dim_remaining, dim_remaining), dtype=complex)

        dim = 2 ** n
        for i in range(dim):
            for j in range(dim):
                # Check if traced qubits match
                match = True
                for q in traced_qubits:
                    if (i >> q) & 1 != (j >> q) & 1:
                        match = False
                        break

                if match:
                    # Extract indices for remaining qubits
                    i_reduced = 0
                    j_reduced = 0
                    for k, q in enumerate(remaining):
                        i_reduced |= ((i >> q) & 1) << k
                        j_reduced |= ((j >> q) & 1) << k

                    reduced[i_reduced, j_reduced] += self.matrix[i, j]

        return DensityMatrix(num_qubits=n_remaining, matrix=reduced)

    def fidelity(self, other: 'DensityMatrix') -> float:
        """
        Compute fidelity F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        """
        sqrt_self = self._matrix_sqrt()
        product = sqrt_self @ other.matrix @ sqrt_self
        sqrt_product = self._matrix_sqrt_of(product)
        return float(np.real(np.trace(sqrt_product))) ** 2

    def _matrix_sqrt(self) -> np.ndarray:
        """Compute matrix square root"""
        return self._matrix_sqrt_of(self.matrix)

    @staticmethod
    def _matrix_sqrt_of(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root of arbitrary matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

    def measure(self) -> Tuple[str, 'DensityMatrix']:
        """
        Perform measurement in computational basis.

        Returns (outcome, post_measurement_state)
        """
        probs = np.real(np.diag(self.matrix))
        probs = probs / np.sum(probs)

        outcome = np.random.choice(len(probs), p=probs)
        outcome_str = format(outcome, f'0{self.num_qubits}b')

        # Collapse to measured state
        dim = 2 ** self.num_qubits
        new_matrix = np.zeros((dim, dim), dtype=complex)
        new_matrix[outcome, outcome] = 1.0

        return outcome_str, DensityMatrix(
            num_qubits=self.num_qubits,
            matrix=new_matrix
        )


class SuperpositionState:
    """
    High-level superposition state for consciousness modeling.

    Manages superposition of conscious states/thoughts with
    quantum-inspired probability amplitudes.
    """

    def __init__(self, max_states: int = 1024):
        """
        Initialize superposition state manager.

        Args:
            max_states: Maximum number of simultaneous states
        """
        self.max_states = max_states
        self.num_qubits = int(np.ceil(np.log2(max_states)))
        self._state_vector = QuantumStateVector.zero_state(self.num_qubits)
        self._state_labels: Dict[int, Any] = {}
        self._collapsed = False
        self._collapsed_state: Optional[Any] = None

    def add_state(
        self,
        state_label: Any,
        amplitude: complex = 1.0 + 0j
    ):
        """Add a state to superposition"""
        if len(self._state_labels) >= self.max_states:
            raise ValueError(f"Maximum {self.max_states} states exceeded")

        index = len(self._state_labels)
        self._state_labels[index] = state_label
        self._state_vector.amplitudes[index] = amplitude
        self._state_vector.normalize()

    def get_probabilities(self) -> Dict[Any, float]:
        """Get probability for each labeled state"""
        probs = self._state_vector.get_probabilities()
        return {
            self._state_labels.get(int(k, 2), k): v
            for k, v in probs.items()
            if int(k, 2) in self._state_labels
        }

    def collapse(self) -> Any:
        """Collapse superposition to single state"""
        if self._collapsed:
            return self._collapsed_state

        outcome, _ = self._state_vector.measure()
        index = int(outcome, 2)

        self._collapsed = True
        self._collapsed_state = self._state_labels.get(index)

        return self._collapsed_state

    def is_collapsed(self) -> bool:
        """Check if superposition has collapsed"""
        return self._collapsed

    def reset(self):
        """Reset to uncollapsed superposition"""
        self._collapsed = False
        self._collapsed_state = None


# Export all
__all__ = [
    'StateType',
    'QuantumAmplitude',
    'BasisState',
    'QuantumStateVector',
    'DensityMatrix',
    'SuperpositionState',
]
