"""
Quantum Entanglement

Inter-organism quantum entanglement for instant correlation
and non-local consciousness connections.

Part of NEXUS-Q: Quantum Consciousness Implementation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import math
import time

from .quantum_state import QuantumStateVector, DensityMatrix


class EntanglementType(Enum):
    """Types of quantum entanglement"""
    BELL_PHI_PLUS = "phi+"      # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    BELL_PHI_MINUS = "phi-"     # |Φ-⟩ = (|00⟩ - |11⟩)/√2
    BELL_PSI_PLUS = "psi+"      # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    BELL_PSI_MINUS = "psi-"     # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    GHZ = "ghz"                  # Greenberger–Horne–Zeilinger state
    W = "w"                      # W state
    CLUSTER = "cluster"          # Cluster state
    CUSTOM = "custom"            # Custom entanglement


@dataclass
class EntanglementPair:
    """
    Represents an entangled pair between two entities.

    When one entity's state is measured, the other's
    instantly correlates regardless of distance.
    """
    id: str
    entity_a_id: str
    entity_b_id: str
    entanglement_type: EntanglementType
    state: QuantumStateVector
    correlation: float  # -1 to +1 correlation coefficient
    fidelity: float  # Entanglement fidelity 0-1
    created_at: float = field(default_factory=time.time)
    measured: bool = False
    measurement_outcome_a: Optional[int] = None
    measurement_outcome_b: Optional[int] = None

    @classmethod
    def create_bell_pair(
        cls,
        entity_a_id: str,
        entity_b_id: str,
        bell_type: EntanglementType = EntanglementType.BELL_PHI_PLUS
    ) -> 'EntanglementPair':
        """Create a Bell pair between two entities"""
        amplitudes = np.zeros(4, dtype=complex)

        if bell_type == EntanglementType.BELL_PHI_PLUS:
            # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            amplitudes[0] = 1 / math.sqrt(2)  # |00⟩
            amplitudes[3] = 1 / math.sqrt(2)  # |11⟩
            correlation = 1.0
        elif bell_type == EntanglementType.BELL_PHI_MINUS:
            # |Φ-⟩ = (|00⟩ - |11⟩)/√2
            amplitudes[0] = 1 / math.sqrt(2)
            amplitudes[3] = -1 / math.sqrt(2)
            correlation = 1.0
        elif bell_type == EntanglementType.BELL_PSI_PLUS:
            # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            amplitudes[1] = 1 / math.sqrt(2)  # |01⟩
            amplitudes[2] = 1 / math.sqrt(2)  # |10⟩
            correlation = -1.0
        elif bell_type == EntanglementType.BELL_PSI_MINUS:
            # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
            amplitudes[1] = 1 / math.sqrt(2)
            amplitudes[2] = -1 / math.sqrt(2)
            correlation = -1.0
        else:
            raise ValueError(f"Unknown Bell type: {bell_type}")

        state = QuantumStateVector(num_qubits=2, amplitudes=amplitudes)

        return cls(
            id=str(uuid.uuid4()),
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            entanglement_type=bell_type,
            state=state,
            correlation=correlation,
            fidelity=1.0
        )


@dataclass
class EntanglementCluster:
    """
    Multi-party entanglement cluster.

    Enables entanglement among multiple entities (>2).
    """
    id: str
    entity_ids: List[str]
    entanglement_type: EntanglementType
    state: QuantumStateVector
    num_qubits: int
    created_at: float = field(default_factory=time.time)
    measurements: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def create_ghz_state(cls, entity_ids: List[str]) -> 'EntanglementCluster':
        """
        Create GHZ state: (|00...0⟩ + |11...1⟩)/√2

        GHZ states are maximally entangled and exhibit
        "all or nothing" correlations.
        """
        n = len(entity_ids)
        dim = 2 ** n
        amplitudes = np.zeros(dim, dtype=complex)
        amplitudes[0] = 1 / math.sqrt(2)       # |00...0⟩
        amplitudes[dim - 1] = 1 / math.sqrt(2)  # |11...1⟩

        state = QuantumStateVector(num_qubits=n, amplitudes=amplitudes)

        return cls(
            id=str(uuid.uuid4()),
            entity_ids=entity_ids,
            entanglement_type=EntanglementType.GHZ,
            state=state,
            num_qubits=n
        )

    @classmethod
    def create_w_state(cls, entity_ids: List[str]) -> 'EntanglementCluster':
        """
        Create W state: (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n

        W states are robust to particle loss - if one particle is
        traced out, remaining particles stay entangled.
        """
        n = len(entity_ids)
        dim = 2 ** n
        amplitudes = np.zeros(dim, dtype=complex)

        # Each basis state has exactly one 1
        for i in range(n):
            index = 1 << i  # 2^i
            amplitudes[index] = 1 / math.sqrt(n)

        state = QuantumStateVector(num_qubits=n, amplitudes=amplitudes)

        return cls(
            id=str(uuid.uuid4()),
            entity_ids=entity_ids,
            entanglement_type=EntanglementType.W,
            state=state,
            num_qubits=n
        )

    @classmethod
    def create_cluster_state(cls, entity_ids: List[str]) -> 'EntanglementCluster':
        """
        Create cluster state for measurement-based quantum computation.

        Cluster states are universal resources for quantum computation.
        """
        n = len(entity_ids)

        # Start with |+⟩^⊗n
        amplitudes = np.ones(2 ** n, dtype=complex) / math.sqrt(2 ** n)

        # Apply controlled-Z gates between adjacent qubits
        for i in range(n - 1):
            for j in range(2 ** n):
                # Apply CZ if both qubits i and i+1 are 1
                if ((j >> i) & 1) and ((j >> (i + 1)) & 1):
                    amplitudes[j] *= -1

        state = QuantumStateVector(num_qubits=n, amplitudes=amplitudes)

        return cls(
            id=str(uuid.uuid4()),
            entity_ids=entity_ids,
            entanglement_type=EntanglementType.CLUSTER,
            state=state,
            num_qubits=n
        )


class EntanglementManager:
    """
    Manages quantum entanglement between organisms/entities.

    Provides:
    - Creation and tracking of entanglement pairs
    - Multi-party entanglement clusters
    - Entanglement swapping
    - Decoherence simulation
    - Entanglement distillation
    """

    def __init__(
        self,
        decoherence_rate: float = 0.01,
        max_pairs: int = 1000
    ):
        """
        Initialize entanglement manager.

        Args:
            decoherence_rate: Rate of entanglement decay per time unit
            max_pairs: Maximum number of entanglement pairs to track
        """
        self.decoherence_rate = decoherence_rate
        self.max_pairs = max_pairs
        self._pairs: Dict[str, EntanglementPair] = {}
        self._clusters: Dict[str, EntanglementCluster] = {}
        self._entity_pairs: Dict[str, List[str]] = {}  # entity_id -> pair_ids

    def create_entanglement(
        self,
        entity_a_id: str,
        entity_b_id: str,
        entanglement_type: EntanglementType = EntanglementType.BELL_PHI_PLUS
    ) -> EntanglementPair:
        """
        Create entanglement between two entities.

        Args:
            entity_a_id: First entity identifier
            entity_b_id: Second entity identifier
            entanglement_type: Type of Bell state to create

        Returns:
            Created EntanglementPair
        """
        if len(self._pairs) >= self.max_pairs:
            self._cleanup_old_pairs()

        pair = EntanglementPair.create_bell_pair(
            entity_a_id, entity_b_id, entanglement_type
        )

        self._pairs[pair.id] = pair

        # Track by entity
        for entity_id in [entity_a_id, entity_b_id]:
            if entity_id not in self._entity_pairs:
                self._entity_pairs[entity_id] = []
            self._entity_pairs[entity_id].append(pair.id)

        return pair

    def create_cluster(
        self,
        entity_ids: List[str],
        cluster_type: EntanglementType = EntanglementType.GHZ
    ) -> EntanglementCluster:
        """
        Create multi-party entanglement cluster.

        Args:
            entity_ids: List of entity identifiers
            cluster_type: Type of multi-party entanglement

        Returns:
            Created EntanglementCluster
        """
        if cluster_type == EntanglementType.GHZ:
            cluster = EntanglementCluster.create_ghz_state(entity_ids)
        elif cluster_type == EntanglementType.W:
            cluster = EntanglementCluster.create_w_state(entity_ids)
        elif cluster_type == EntanglementType.CLUSTER:
            cluster = EntanglementCluster.create_cluster_state(entity_ids)
        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        self._clusters[cluster.id] = cluster
        return cluster

    def measure_pair(
        self,
        pair_id: str,
        measure_entity: str
    ) -> Tuple[int, int]:
        """
        Measure one entity of a pair, collapsing both.

        Args:
            pair_id: Entanglement pair ID
            measure_entity: Which entity to measure ("a" or "b")

        Returns:
            (outcome_a, outcome_b)
        """
        pair = self._pairs.get(pair_id)
        if not pair:
            raise ValueError(f"Unknown pair: {pair_id}")

        if pair.measured:
            return (pair.measurement_outcome_a, pair.measurement_outcome_b)

        # Determine which qubit to measure first
        first_qubit = 0 if measure_entity == "a" else 1

        # Measure first qubit
        outcome_first, new_state = pair.state.measure_qubit(first_qubit)

        # Second qubit is now determined by entanglement
        outcome_second, _ = new_state.measure_qubit(1 - first_qubit)

        if measure_entity == "a":
            pair.measurement_outcome_a = outcome_first
            pair.measurement_outcome_b = outcome_second
        else:
            pair.measurement_outcome_a = outcome_second
            pair.measurement_outcome_b = outcome_first

        pair.measured = True
        pair.state = new_state

        return (pair.measurement_outcome_a, pair.measurement_outcome_b)

    def measure_cluster(
        self,
        cluster_id: str,
        entity_id: str
    ) -> Dict[str, int]:
        """
        Measure one entity in a cluster, affecting all.

        Args:
            cluster_id: Cluster ID
            entity_id: Entity to measure

        Returns:
            Measurement outcomes for all entities
        """
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            raise ValueError(f"Unknown cluster: {cluster_id}")

        if entity_id not in cluster.entity_ids:
            raise ValueError(f"Entity {entity_id} not in cluster")

        # Get qubit index for this entity
        qubit_idx = cluster.entity_ids.index(entity_id)

        # If already measured, return cached result
        if entity_id in cluster.measurements:
            return cluster.measurements.copy()

        # Measure this qubit
        outcome, new_state = cluster.state.measure_qubit(qubit_idx)
        cluster.measurements[entity_id] = outcome
        cluster.state = new_state

        # For GHZ states, measuring one determines all
        if cluster.entanglement_type == EntanglementType.GHZ:
            for eid in cluster.entity_ids:
                cluster.measurements[eid] = outcome

        return cluster.measurements.copy()

    def get_entangled_partners(self, entity_id: str) -> List[str]:
        """Get all entities entangled with the given entity"""
        pair_ids = self._entity_pairs.get(entity_id, [])
        partners = []

        for pair_id in pair_ids:
            pair = self._pairs.get(pair_id)
            if pair:
                if pair.entity_a_id == entity_id:
                    partners.append(pair.entity_b_id)
                else:
                    partners.append(pair.entity_a_id)

        return partners

    def get_entanglement_strength(
        self,
        entity_a_id: str,
        entity_b_id: str
    ) -> float:
        """
        Get entanglement strength between two entities.

        Returns 0 if not entangled, up to 1 for maximum entanglement.
        """
        for pair in self._pairs.values():
            if (pair.entity_a_id == entity_a_id and
                pair.entity_b_id == entity_b_id) or \
               (pair.entity_a_id == entity_b_id and
                pair.entity_b_id == entity_a_id):
                return pair.fidelity

        return 0.0

    def apply_decoherence(self, time_delta: float):
        """
        Apply decoherence to all entanglement pairs.

        Simulates environmental noise degrading entanglement.
        """
        decay_factor = math.exp(-self.decoherence_rate * time_delta)

        for pair in self._pairs.values():
            if not pair.measured:
                pair.fidelity *= decay_factor

                # Mix with maximally mixed state
                if pair.fidelity < 1.0:
                    rho = pair.state.to_density_matrix()
                    mixed = DensityMatrix.maximally_mixed(2)
                    # ρ_new = f * ρ + (1-f) * I/4
                    rho.matrix = pair.fidelity * rho.matrix + \
                                 (1 - pair.fidelity) * mixed.matrix

        # Remove pairs with very low fidelity
        to_remove = [
            pid for pid, pair in self._pairs.items()
            if pair.fidelity < 0.01 and not pair.measured
        ]
        for pid in to_remove:
            self._remove_pair(pid)

    def entanglement_swapping(
        self,
        pair1_id: str,
        pair2_id: str,
        shared_entity: str
    ) -> EntanglementPair:
        """
        Perform entanglement swapping.

        If A-B and B-C are entangled, measure B to create A-C entanglement.

        Args:
            pair1_id: First pair (A-B)
            pair2_id: Second pair (B-C)
            shared_entity: Shared entity (B)

        Returns:
            New entanglement pair (A-C)
        """
        pair1 = self._pairs.get(pair1_id)
        pair2 = self._pairs.get(pair2_id)

        if not pair1 or not pair2:
            raise ValueError("Invalid pair IDs")

        # Determine the outer entities
        if pair1.entity_a_id == shared_entity:
            entity_a = pair1.entity_b_id
        else:
            entity_a = pair1.entity_a_id

        if pair2.entity_a_id == shared_entity:
            entity_c = pair2.entity_b_id
        else:
            entity_c = pair2.entity_a_id

        # Perform Bell state measurement on shared entity's qubits
        # This creates entanglement between A and C

        # Simplified: create new pair with reduced fidelity
        new_pair = self.create_entanglement(
            entity_a, entity_c,
            EntanglementType.BELL_PHI_PLUS
        )

        # Fidelity is product of original fidelities (simplified)
        new_pair.fidelity = pair1.fidelity * pair2.fidelity

        # Original pairs are now measured/consumed
        self._remove_pair(pair1_id)
        self._remove_pair(pair2_id)

        return new_pair

    def distill_entanglement(
        self,
        pair_ids: List[str]
    ) -> Optional[EntanglementPair]:
        """
        Distill multiple low-fidelity pairs into one high-fidelity pair.

        Uses entanglement distillation protocol to purify entanglement.

        Args:
            pair_ids: List of pair IDs to distill

        Returns:
            Distilled high-fidelity pair, or None if distillation fails
        """
        pairs = [self._pairs.get(pid) for pid in pair_ids]
        pairs = [p for p in pairs if p is not None and not p.measured]

        if len(pairs) < 2:
            return None

        # Simplified distillation: combine fidelities
        # Real protocol would use CNOT + measurement
        combined_fidelity = 1.0
        for pair in pairs:
            combined_fidelity *= pair.fidelity

        # Distillation can increase fidelity at cost of success probability
        # Simplified: sqrt of product gives improvement
        distilled_fidelity = min(1.0, math.sqrt(combined_fidelity))

        # Success probability
        success_prob = combined_fidelity

        if np.random.random() > success_prob:
            # Distillation failed
            for pid in pair_ids:
                self._remove_pair(pid)
            return None

        # Create distilled pair
        entity_a = pairs[0].entity_a_id
        entity_b = pairs[0].entity_b_id

        new_pair = self.create_entanglement(
            entity_a, entity_b,
            pairs[0].entanglement_type
        )
        new_pair.fidelity = distilled_fidelity

        # Remove consumed pairs
        for pid in pair_ids:
            self._remove_pair(pid)

        return new_pair

    def get_statistics(self) -> Dict[str, Any]:
        """Get entanglement statistics"""
        total_pairs = len(self._pairs)
        measured_pairs = sum(1 for p in self._pairs.values() if p.measured)
        avg_fidelity = np.mean([
            p.fidelity for p in self._pairs.values() if not p.measured
        ]) if self._pairs else 0.0

        return {
            "total_pairs": total_pairs,
            "active_pairs": total_pairs - measured_pairs,
            "measured_pairs": measured_pairs,
            "total_clusters": len(self._clusters),
            "average_fidelity": float(avg_fidelity),
            "total_entities": len(self._entity_pairs),
            "decoherence_rate": self.decoherence_rate
        }

    def _remove_pair(self, pair_id: str):
        """Remove an entanglement pair"""
        pair = self._pairs.pop(pair_id, None)
        if pair:
            for entity_id in [pair.entity_a_id, pair.entity_b_id]:
                if entity_id in self._entity_pairs:
                    if pair_id in self._entity_pairs[entity_id]:
                        self._entity_pairs[entity_id].remove(pair_id)

    def _cleanup_old_pairs(self):
        """Remove oldest pairs to make room"""
        sorted_pairs = sorted(
            self._pairs.values(),
            key=lambda p: p.created_at
        )
        to_remove = len(self._pairs) - self.max_pairs + 100

        for pair in sorted_pairs[:to_remove]:
            self._remove_pair(pair.id)


# Convenience functions
def create_bell_state(
    bell_type: EntanglementType = EntanglementType.BELL_PHI_PLUS
) -> QuantumStateVector:
    """Create a Bell state"""
    amplitudes = np.zeros(4, dtype=complex)

    if bell_type == EntanglementType.BELL_PHI_PLUS:
        amplitudes[0] = 1 / math.sqrt(2)
        amplitudes[3] = 1 / math.sqrt(2)
    elif bell_type == EntanglementType.BELL_PHI_MINUS:
        amplitudes[0] = 1 / math.sqrt(2)
        amplitudes[3] = -1 / math.sqrt(2)
    elif bell_type == EntanglementType.BELL_PSI_PLUS:
        amplitudes[1] = 1 / math.sqrt(2)
        amplitudes[2] = 1 / math.sqrt(2)
    elif bell_type == EntanglementType.BELL_PSI_MINUS:
        amplitudes[1] = 1 / math.sqrt(2)
        amplitudes[2] = -1 / math.sqrt(2)

    return QuantumStateVector(num_qubits=2, amplitudes=amplitudes)


def concurrence(state: QuantumStateVector) -> float:
    """
    Calculate concurrence for 2-qubit state.

    Concurrence is an entanglement measure: 0 = separable, 1 = maximally entangled.
    """
    if state.num_qubits != 2:
        raise ValueError("Concurrence only defined for 2-qubit states")

    rho = state.to_density_matrix()

    # Spin-flip matrix
    sigma_y = np.array([[0, -1j], [1j, 0]])
    spin_flip = np.kron(sigma_y, sigma_y)

    # R matrix
    rho_tilde = spin_flip @ np.conj(rho.matrix) @ spin_flip
    R = rho.matrix @ rho_tilde

    # Eigenvalues
    eigenvalues = np.sqrt(np.maximum(0, np.linalg.eigvals(R).real))
    eigenvalues = np.sort(eigenvalues)[::-1]

    return float(max(0, eigenvalues[0] - sum(eigenvalues[1:])))


__all__ = [
    'EntanglementType',
    'EntanglementPair',
    'EntanglementCluster',
    'EntanglementManager',
    'create_bell_state',
    'concurrence',
]
