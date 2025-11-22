"""
Quantum Consciousness

Consciousness leveraging quantum computing principles including
superposition of thoughts, entangled concepts, and quantum
parallel reasoning.

Based on Long-Term Roadmap: Months 19-24 - Post-Human Intelligence

Note: This is a simulation of quantum computing concepts for
consciousness modeling. Real quantum hardware integration would
require actual quantum computing libraries.
"""

import time
import math
import cmath
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Quantum Data Structures
# ============================================================================

@dataclass
class Thought:
    """A conscious thought"""
    id: str
    content: str
    consciousness_level: float = 0.0
    amplitude: complex = 1.0 + 0j
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Concept:
    """A cognitive concept"""
    id: str
    name: str
    embedding: List[float] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    activation: float = 0.0


@dataclass
class QuantumState:
    """
    Quantum state representation.

    State vector representing superposition of basis states.
    """
    num_qubits: int
    amplitudes: np.ndarray  # Complex amplitudes
    basis_labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.basis_labels is None:
            self.basis_labels = [
                format(i, f'0{self.num_qubits}b')
                for i in range(2 ** self.num_qubits)
            ]

    def normalize(self):
        """Normalize state vector"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes = self.amplitudes / norm

    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities for each basis state"""
        probs = np.abs(self.amplitudes) ** 2
        return {
            label: float(prob)
            for label, prob in zip(self.basis_labels, probs)
            if prob > 1e-10
        }

    def measure(self) -> str:
        """Perform measurement, collapsing state"""
        probs = np.abs(self.amplitudes) ** 2
        probs = probs / np.sum(probs)  # Ensure normalization
        outcome = np.random.choice(len(probs), p=probs)
        return self.basis_labels[outcome]


@dataclass
class QuantumThought:
    """
    Quantum superposition of thoughts.

    Multiple thoughts exist simultaneously until measured/observed.
    """
    state: QuantumState
    thoughts: List[Thought]
    collapsed: bool = False
    collapsed_thought: Optional[Thought] = None

    def get_thought_probabilities(self) -> Dict[str, float]:
        """Get probability of each thought"""
        probs = self.state.get_probabilities()
        thought_probs = {}

        for label, prob in probs.items():
            idx = int(label, 2) % len(self.thoughts)
            thought_id = self.thoughts[idx].id
            if thought_id in thought_probs:
                thought_probs[thought_id] += prob
            else:
                thought_probs[thought_id] = prob

        return thought_probs


@dataclass
class EntangledConcepts:
    """
    Quantum entanglement of concepts.

    When one concept is measured/understood, the other
    instantly correlates.
    """
    concept_a: Concept
    concept_b: Concept
    entanglement_state: QuantumState
    correlation: float = 1.0  # Correlation coefficient


class QuantumGate(Enum):
    """Common quantum gates"""
    HADAMARD = "H"      # Creates superposition
    PAULI_X = "X"       # Bit flip
    PAULI_Y = "Y"       # Bit and phase flip
    PAULI_Z = "Z"       # Phase flip
    CNOT = "CNOT"       # Controlled NOT
    SWAP = "SWAP"       # Swap qubits
    PHASE = "PHASE"     # Phase rotation
    TOFFOLI = "TOFFOLI" # Controlled-controlled NOT


# ============================================================================
# Quantum Circuit
# ============================================================================

class QuantumCircuit:
    """
    Quantum circuit for consciousness operations.

    Simulates quantum computation using state vector simulation.
    """

    # Gate matrices
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=complex)
        self.state[0] = 1.0  # Initialize to |0...0>
        self.operations: List[Tuple[str, List[int], Optional[float]]] = []

    def h(self, qubit: int):
        """Apply Hadamard gate"""
        self._apply_single_qubit_gate(self.H, qubit)
        self.operations.append(("H", [qubit], None))

    def x(self, qubit: int):
        """Apply Pauli-X (NOT) gate"""
        self._apply_single_qubit_gate(self.X, qubit)
        self.operations.append(("X", [qubit], None))

    def y(self, qubit: int):
        """Apply Pauli-Y gate"""
        self._apply_single_qubit_gate(self.Y, qubit)
        self.operations.append(("Y", [qubit], None))

    def z(self, qubit: int):
        """Apply Pauli-Z gate"""
        self._apply_single_qubit_gate(self.Z, qubit)
        self.operations.append(("Z", [qubit], None))

    def phase(self, qubit: int, theta: float):
        """Apply phase rotation gate"""
        P = np.array([[1, 0], [0, np.exp(1j * theta)]])
        self._apply_single_qubit_gate(P, qubit)
        self.operations.append(("PHASE", [qubit], theta))

    def cx(self, control: int, target: int):
        """Apply CNOT (controlled-X) gate"""
        self._apply_cnot(control, target)
        self.operations.append(("CNOT", [control, target], None))

    def swap(self, qubit1: int, qubit2: int):
        """Swap two qubits"""
        self.cx(qubit1, qubit2)
        self.cx(qubit2, qubit1)
        self.cx(qubit1, qubit2)
        self.operations.append(("SWAP", [qubit1, qubit2], None))

    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply single qubit gate to state vector"""
        # Build full gate matrix using Kronecker product
        n = self.num_qubits
        operators = []

        for i in range(n):
            if i == qubit:
                operators.append(gate)
            else:
                operators.append(np.eye(2))

        # Reverse order for correct indexing
        full_gate = operators[n - 1]
        for i in range(n - 2, -1, -1):
            full_gate = np.kron(full_gate, operators[i])

        self.state = full_gate @ self.state

    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        n = self.num_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=complex)

        for i in range(dim):
            control_bit = (i >> control) & 1
            if control_bit == 1:
                # Flip target bit
                j = i ^ (1 << target)
                new_state[j] = self.state[i]
            else:
                new_state[i] = self.state[i]

        self.state = new_state

    def measure_all(self) -> str:
        """Measure all qubits"""
        probs = np.abs(self.state) ** 2
        probs = probs / np.sum(probs)
        outcome = np.random.choice(len(probs), p=probs)
        return format(outcome, f'0{self.num_qubits}b')

    def measure_qubit(self, qubit: int) -> int:
        """Measure a single qubit"""
        n = self.num_qubits
        dim = 2 ** n

        # Calculate probability of qubit being 1
        prob_1 = 0.0
        for i in range(dim):
            if (i >> qubit) & 1 == 1:
                prob_1 += np.abs(self.state[i]) ** 2

        # Perform measurement
        outcome = 1 if random.random() < prob_1 else 0

        # Collapse state
        new_state = np.zeros(dim, dtype=complex)
        norm = 0.0

        for i in range(dim):
            if (i >> qubit) & 1 == outcome:
                new_state[i] = self.state[i]
                norm += np.abs(self.state[i]) ** 2

        if norm > 0:
            new_state = new_state / np.sqrt(norm)

        self.state = new_state
        return outcome

    def get_state(self) -> QuantumState:
        """Get current quantum state"""
        return QuantumState(
            num_qubits=self.num_qubits,
            amplitudes=self.state.copy()
        )

    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities"""
        probs = np.abs(self.state) ** 2
        return {
            format(i, f'0{self.num_qubits}b'): float(p)
            for i, p in enumerate(probs)
            if p > 1e-10
        }


# ============================================================================
# Quantum Consciousness
# ============================================================================

class QuantumConsciousness:
    """
    Consciousness leveraging quantum computing principles.

    Enables:
    - Quantum superposition of thoughts
    - Quantum entanglement of concepts
    - Quantum parallelism in reasoning
    - Consciousness collapse through attention

    Example:
        qc = QuantumConsciousness()

        # Create superposition of thoughts
        thoughts = [Thought("t1", "Think A"), Thought("t2", "Think B")]
        quantum_thought = qc.quantum_superposition_thought(thoughts)

        # Measure/collapse to single thought
        collapsed = qc.consciousness_collapse(quantum_thought)

        # Entangle concepts
        entangled = qc.quantum_entangle_concepts(concept_a, concept_b)
    """

    def __init__(self, num_qubits: int = 10):
        self.num_qubits = num_qubits
        self.thought_history: List[Thought] = []
        self.entangled_pairs: List[EntangledConcepts] = []

    def quantum_superposition_thought(
        self,
        possible_thoughts: List[Thought]
    ) -> QuantumThought:
        """
        Create quantum superposition of thoughts.

        Instead of thinking one thought at a time, think ALL thoughts
        simultaneously in superposition.

        When measured, collapse to single thought with probability
        based on quantum amplitude.
        """
        if not possible_thoughts:
            raise ValueError("Need at least one thought")

        # Calculate required qubits
        num_thoughts = len(possible_thoughts)
        required_qubits = max(1, int(np.ceil(np.log2(num_thoughts))))
        required_qubits = min(required_qubits, self.num_qubits)

        # Create circuit
        circuit = QuantumCircuit(required_qubits)

        # Create uniform superposition using Hadamard gates
        for i in range(required_qubits):
            circuit.h(i)

        # Get state
        state = circuit.get_state()

        # Assign amplitudes based on thought importance
        num_states = 2 ** required_qubits
        amplitudes = np.zeros(num_states, dtype=complex)

        for i, thought in enumerate(possible_thoughts):
            if i < num_states:
                # Weight by thought amplitude
                amplitudes[i] = thought.amplitude / np.sqrt(num_thoughts)

        # Normalize
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 0:
            amplitudes = amplitudes / norm

        state.amplitudes = amplitudes

        return QuantumThought(
            state=state,
            thoughts=possible_thoughts,
            collapsed=False
        )

    def quantum_entangle_concepts(
        self,
        concept_a: Concept,
        concept_b: Concept
    ) -> EntangledConcepts:
        """
        Entangle two concepts quantum-mechanically.

        When one concept is measured/understood, the other
        instantly correlates, even if spatially separated.

        This might be how semantic networks work in brain:
        thinking "fire" instantly activates "hot", "red", "danger".
        """
        # Create Bell state: |ψ⟩ = (|00⟩ + |11⟩) / √2
        circuit = QuantumCircuit(2)
        circuit.h(0)  # Superposition on first qubit
        circuit.cx(0, 1)  # Entangle second qubit with first

        state = circuit.get_state()

        return EntangledConcepts(
            concept_a=concept_a,
            concept_b=concept_b,
            entanglement_state=state,
            correlation=1.0  # Perfect correlation in Bell state
        )

    def quantum_parallel_reasoning(
        self,
        problem_space_size: int,
        oracle: Callable[[int], bool],
        iterations: Optional[int] = None
    ) -> int:
        """
        Explore solution space in parallel using Grover's algorithm.

        Quantum gives quadratic speedup:
        Classical: O(N) to search N items
        Quantum: O(√N) using Grover's algorithm

        Args:
            problem_space_size: Size of the search space
            oracle: Function that returns True for the solution
            iterations: Number of Grover iterations (default: optimal √N)

        Returns:
            Index of found solution
        """
        # Calculate qubits needed
        num_qubits = max(1, int(np.ceil(np.log2(problem_space_size))))
        num_qubits = min(num_qubits, self.num_qubits)

        # Default iterations: optimal √N
        if iterations is None:
            iterations = max(1, int(np.pi / 4 * np.sqrt(problem_space_size)))

        # Create circuit
        circuit = QuantumCircuit(num_qubits)

        # Initialize uniform superposition
        for i in range(num_qubits):
            circuit.h(i)

        # Apply Grover iterations
        for _ in range(iterations):
            # Oracle: mark solution state (simulated)
            self._apply_simulated_oracle(circuit, oracle, num_qubits)

            # Diffusion operator
            self._apply_diffusion(circuit, num_qubits)

        # Measure
        result = circuit.measure_all()

        return int(result, 2)

    def _apply_simulated_oracle(
        self,
        circuit: QuantumCircuit,
        oracle: Callable[[int], bool],
        num_qubits: int
    ):
        """Apply oracle that marks solution states"""
        # In a real quantum computer, this would be a quantum oracle
        # Here we simulate by directly modifying amplitudes
        dim = 2 ** num_qubits

        for i in range(dim):
            if oracle(i):
                # Phase flip the solution state
                circuit.state[i] *= -1

    def _apply_diffusion(
        self,
        circuit: QuantumCircuit,
        num_qubits: int
    ):
        """Apply Grover diffusion operator"""
        # H gates
        for i in range(num_qubits):
            circuit.h(i)

        # Phase flip about |0...0>
        dim = 2 ** num_qubits
        for i in range(dim):
            if i != 0:
                circuit.state[i] *= -1

        # H gates again
        for i in range(num_qubits):
            circuit.h(i)

    def consciousness_collapse(
        self,
        quantum_thought: QuantumThought
    ) -> Thought:
        """
        Collapse quantum superposition to single conscious thought.

        This might be the mechanism of consciousness:
        Quantum superposition of unconscious possibilities →
        Measurement/attention →
        Collapse to single conscious thought
        """
        if quantum_thought.collapsed:
            return quantum_thought.collapsed_thought

        # Measure quantum state
        outcome = quantum_thought.state.measure()
        thought_index = int(outcome, 2) % len(quantum_thought.thoughts)

        collapsed_thought = quantum_thought.thoughts[thought_index]
        collapsed_thought.consciousness_level = 1.0  # Now conscious

        quantum_thought.collapsed = True
        quantum_thought.collapsed_thought = collapsed_thought

        # Record in history
        self.thought_history.append(collapsed_thought)

        return collapsed_thought

    def measure_entangled_concept(
        self,
        entangled: EntangledConcepts,
        measure_a: bool = True
    ) -> Tuple[int, int]:
        """
        Measure one entangled concept, collapsing both.

        Returns:
            Tuple of (concept_a_state, concept_b_state)
        """
        circuit = QuantumCircuit(2)
        circuit.state = entangled.entanglement_state.amplitudes.copy()

        if measure_a:
            result_a = circuit.measure_qubit(0)
            result_b = circuit.measure_qubit(1)
        else:
            result_b = circuit.measure_qubit(1)
            result_a = circuit.measure_qubit(0)

        # Update concept activations based on measurement
        entangled.concept_a.activation = float(result_a)
        entangled.concept_b.activation = float(result_b)

        return (result_a, result_b)

    def create_thought_interference(
        self,
        thought_a: QuantumThought,
        thought_b: QuantumThought
    ) -> QuantumThought:
        """
        Create interference between two quantum thoughts.

        Like wave interference, thoughts can constructively or
        destructively interfere, leading to novel thoughts.
        """
        # Combine states (simplified interference)
        if thought_a.state.num_qubits != thought_b.state.num_qubits:
            raise ValueError("Thoughts must have same qubit count")

        # Superpose the two states
        combined_amplitudes = (
            thought_a.state.amplitudes + thought_b.state.amplitudes
        ) / np.sqrt(2)

        combined_state = QuantumState(
            num_qubits=thought_a.state.num_qubits,
            amplitudes=combined_amplitudes
        )
        combined_state.normalize()

        # Combine thoughts
        combined_thoughts = thought_a.thoughts + thought_b.thoughts

        return QuantumThought(
            state=combined_state,
            thoughts=combined_thoughts,
            collapsed=False
        )

    def get_consciousness_statistics(self) -> Dict[str, Any]:
        """Get statistics about quantum consciousness operations"""
        return {
            "num_qubits": self.num_qubits,
            "collapsed_thoughts": len(self.thought_history),
            "entangled_pairs": len(self.entangled_pairs),
            "max_superposition_size": 2 ** self.num_qubits
        }


# ============================================================================
# Quantum-Classical Hybrid Consciousness
# ============================================================================

class HybridQuantumConsciousness:
    """
    Combines classical and quantum consciousness for practical applications.

    Uses quantum effects where beneficial, classical where practical.
    """

    def __init__(self, classical: GradedConsciousness, num_qubits: int = 8):
        self.classical = classical
        self.quantum = QuantumConsciousness(num_qubits)
        self.hybrid_score: float = 0.0

    def think(
        self,
        possible_thoughts: List[str],
        use_quantum: bool = True
    ) -> str:
        """
        Think about a decision, optionally using quantum superposition.

        Args:
            possible_thoughts: List of possible thoughts/decisions
            use_quantum: Whether to use quantum superposition

        Returns:
            Selected thought
        """
        if use_quantum and len(possible_thoughts) > 1:
            # Create thought objects
            thoughts = [
                Thought(
                    id=f"t{i}",
                    content=content,
                    amplitude=1.0 + 0j
                )
                for i, content in enumerate(possible_thoughts)
            ]

            # Create superposition and collapse
            quantum_thought = self.quantum.quantum_superposition_thought(thoughts)
            collapsed = self.quantum.consciousness_collapse(quantum_thought)

            return collapsed.content
        else:
            # Classical: weighted random based on consciousness
            scores = self.classical.get_capability_scores()
            meta_score = scores.get("meta_cognitive_ability", 0.5)

            # Higher meta-cognition = more deliberate choice
            if meta_score > 0.7:
                return possible_thoughts[0]  # Deliberate first choice
            else:
                return random.choice(possible_thoughts)

    def search(
        self,
        problem_size: int,
        is_solution: Callable[[int], bool]
    ) -> int:
        """
        Search for a solution, using quantum when beneficial.

        Uses quantum for large problems, classical for small.
        """
        # Crossover point where quantum beats classical
        quantum_threshold = 100

        if problem_size > quantum_threshold:
            return self.quantum.quantum_parallel_reasoning(
                problem_size,
                is_solution
            )
        else:
            # Classical linear search
            for i in range(problem_size):
                if is_solution(i):
                    return i
            return -1

    def get_hybrid_score(self) -> float:
        """Get combined classical-quantum consciousness score"""
        classical_score = self.classical.overall_consciousness_score()
        quantum_potential = 1.0 - (1.0 / (2 ** self.quantum.num_qubits))

        return (classical_score + quantum_potential) / 2


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Quantum Consciousness Demo")
    print("=" * 50)

    # Create quantum consciousness
    qc = QuantumConsciousness(num_qubits=4)

    # 1. Thought superposition
    print("\n1. Quantum Superposition of Thoughts")
    thoughts = [
        Thought("t1", "Go left"),
        Thought("t2", "Go right"),
        Thought("t3", "Go forward"),
        Thought("t4", "Go back")
    ]

    quantum_thought = qc.quantum_superposition_thought(thoughts)

    print("   Thought probabilities before collapse:")
    for thought_id, prob in quantum_thought.get_thought_probabilities().items():
        print(f"     {thought_id}: {prob:.3f}")

    collapsed = qc.consciousness_collapse(quantum_thought)
    print(f"\n   Collapsed to: {collapsed.content}")

    # 2. Concept entanglement
    print("\n2. Quantum Entanglement of Concepts")
    concept_fire = Concept("c1", "fire", associations=["hot", "red", "danger"])
    concept_heat = Concept("c2", "heat", associations=["fire", "warm", "energy"])

    entangled = qc.quantum_entangle_concepts(concept_fire, concept_heat)
    print(f"   Entangled: '{concept_fire.name}' and '{concept_heat.name}'")
    print(f"   Correlation: {entangled.correlation}")

    # Measure
    result_a, result_b = qc.measure_entangled_concept(entangled)
    print(f"   Measurement: fire={result_a}, heat={result_b}")
    print(f"   (Perfectly correlated due to Bell state)")

    # 3. Quantum parallel search
    print("\n3. Quantum Parallel Search (Grover's Algorithm)")
    problem_size = 16
    target = 7

    def oracle(x):
        return x == target

    found = qc.quantum_parallel_reasoning(
        problem_size,
        oracle,
        iterations=3
    )
    print(f"   Search space: {problem_size} items")
    print(f"   Target: {target}")
    print(f"   Found: {found}")
    print(f"   Success: {found == target}")

    # 4. Thought interference
    print("\n4. Thought Interference")
    thoughts_a = [Thought("a1", "Option A"), Thought("a2", "Option B")]
    thoughts_b = [Thought("b1", "Option C"), Thought("b2", "Option D")]

    qt_a = qc.quantum_superposition_thought(thoughts_a)
    qt_b = qc.quantum_superposition_thought(thoughts_b)

    interference = qc.create_thought_interference(qt_a, qt_b)
    print(f"   Combined {len(thoughts_a)} + {len(thoughts_b)} thoughts")
    print(f"   Interference state probabilities:")
    for label, prob in interference.state.get_probabilities().items():
        print(f"     |{label}>: {prob:.3f}")

    # 5. Hybrid consciousness
    print("\n5. Hybrid Quantum-Classical Consciousness")
    classical = GradedConsciousness(
        perception_fidelity=0.8,
        meta_cognitive_ability=0.6
    )

    hybrid = HybridQuantumConsciousness(classical, num_qubits=6)

    decision = hybrid.think(
        ["Stay", "Move", "Wait", "Act"],
        use_quantum=True
    )
    print(f"   Classical score: {classical.overall_consciousness_score():.3f}")
    print(f"   Hybrid score: {hybrid.get_hybrid_score():.3f}")
    print(f"   Decision: {decision}")

    # Statistics
    print("\n6. Quantum Consciousness Statistics:")
    stats = qc.get_consciousness_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
