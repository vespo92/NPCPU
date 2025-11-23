"""
Quantum Memory

Memory systems with quantum superposition, enabling
parallel memory search and quantum-enhanced recall.

Part of NEXUS-Q: Quantum Consciousness Implementation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import uuid
import hashlib

from .quantum_state import QuantumStateVector, DensityMatrix
from .coherence import CoherenceTracker, DecoherenceModel, DecoherenceChannel


T = TypeVar('T')


class MemoryType(Enum):
    """Types of quantum memory"""
    SUPERPOSITION = "superposition"  # Memories in superposition
    ENTANGLED = "entangled"          # Entangled memory pairs
    HOLOGRAPHIC = "holographic"       # Distributed holographic storage
    ASSOCIATIVE = "associative"       # Quantum associative memory


@dataclass
class QuantumMemoryItem:
    """
    A memory item stored in quantum memory.

    Can exist in superposition with other memories.
    """
    id: str
    content: Any
    amplitude: complex = 1.0 + 0j
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def probability(self) -> float:
        """Recall probability based on amplitude"""
        return abs(self.amplitude) ** 2

    def encode_to_index(self, num_qubits: int) -> int:
        """Encode memory to qubit index"""
        # Hash-based encoding
        hash_bytes = hashlib.sha256(str(self.id).encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], 'big')
        return hash_int % (2 ** num_qubits)


@dataclass
class MemoryRecall:
    """
    Result of memory recall operation.
    """
    memory_id: str
    content: Any
    recall_probability: float
    recall_time: float
    is_superposition: bool = False
    alternative_memories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumMemoryRegister:
    """
    Quantum register for memory storage.

    Stores memories in quantum superposition, enabling
    parallel access and interference effects.
    """

    def __init__(
        self,
        num_qubits: int = 10,
        decoherence_time: float = 1000.0
    ):
        """
        Initialize quantum memory register.

        Args:
            num_qubits: Number of qubits (2^n memory slots)
            decoherence_time: Memory coherence time
        """
        self.num_qubits = num_qubits
        self.capacity = 2 ** num_qubits
        self.decoherence_time = decoherence_time

        # Quantum state of memory
        self._state = QuantumStateVector.zero_state(num_qubits)

        # Memory contents
        self._memories: Dict[int, QuantumMemoryItem] = {}

        # Decoherence model
        self._decoherence = DecoherenceModel(
            t1=decoherence_time,
            t2=decoherence_time / 2
        )

        # Track coherence
        self._coherence_tracker = CoherenceTracker()

        # Last operation time
        self._last_time = time.time()

    def store(
        self,
        item: QuantumMemoryItem,
        superpose: bool = True
    ) -> int:
        """
        Store memory item.

        Args:
            item: Memory item to store
            superpose: Whether to superpose with existing memories

        Returns:
            Memory index
        """
        # Get index for this memory
        index = item.encode_to_index(self.num_qubits)

        # Store content
        self._memories[index] = item

        if superpose:
            # Add to superposition
            current_amp = self._state.amplitudes[index]
            new_amp = current_amp + item.amplitude
            self._state.amplitudes[index] = new_amp
            self._state.normalize()
        else:
            # Overwrite
            self._state.amplitudes[index] = item.amplitude
            self._state.normalize()

        return index

    def recall(
        self,
        query: Optional[str] = None,
        index: Optional[int] = None
    ) -> Optional[MemoryRecall]:
        """
        Recall memory from quantum storage.

        Measurement collapses superposition to single memory.

        Args:
            query: Query string (hashed to index)
            index: Direct memory index

        Returns:
            Recalled memory or None
        """
        # Apply decoherence since last access
        self._apply_decoherence()

        if index is None and query is not None:
            # Hash query to get most likely index
            hash_bytes = hashlib.sha256(query.encode()).digest()
            index = int.from_bytes(hash_bytes[:8], 'big') % self.capacity

        if index is not None:
            # Targeted recall - partial measurement
            prob = abs(self._state.amplitudes[index]) ** 2

            if np.random.random() < prob and index in self._memories:
                memory = self._memories[index]
                memory.access_count += 1

                return MemoryRecall(
                    memory_id=memory.id,
                    content=memory.content,
                    recall_probability=prob,
                    recall_time=time.time() - memory.timestamp,
                    is_superposition=False
                )

        # Full measurement - collapse to random memory
        outcome, new_state = self._state.measure()
        measured_index = int(outcome, 2)
        self._state = new_state

        if measured_index in self._memories:
            memory = self._memories[measured_index]
            memory.access_count += 1

            # Get alternative memories
            alternatives = [
                m.id for idx, m in self._memories.items()
                if idx != measured_index and abs(self._state.amplitudes[idx]) > 0.01
            ][:5]

            return MemoryRecall(
                memory_id=memory.id,
                content=memory.content,
                recall_probability=abs(self._state.amplitudes[measured_index]) ** 2,
                recall_time=time.time() - memory.timestamp,
                is_superposition=len(self._memories) > 1,
                alternative_memories=alternatives
            )

        return None

    def grover_search(
        self,
        predicate: callable,
        max_iterations: Optional[int] = None
    ) -> List[MemoryRecall]:
        """
        Quantum search for memories matching predicate.

        Uses Grover's algorithm for O(√N) search.

        Args:
            predicate: Function returning True for matching memories
            max_iterations: Maximum Grover iterations

        Returns:
            List of matching memories
        """
        # Find matching indices
        matching_indices = []
        for idx, memory in self._memories.items():
            if predicate(memory):
                matching_indices.append(idx)

        if not matching_indices:
            return []

        # Optimal number of iterations
        if max_iterations is None:
            num_solutions = len(matching_indices)
            num_items = len(self._memories)
            if num_items > 0 and num_solutions > 0:
                max_iterations = int(
                    math.pi / 4 * math.sqrt(num_items / num_solutions)
                )
            else:
                max_iterations = 1

        # Initialize uniform superposition
        search_state = QuantumStateVector.uniform_superposition(self.num_qubits)

        # Grover iterations
        for _ in range(max_iterations):
            # Oracle: flip phase of matching states
            for idx in matching_indices:
                search_state.amplitudes[idx] *= -1

            # Diffusion operator
            mean_amp = np.mean(search_state.amplitudes)
            search_state.amplitudes = 2 * mean_amp - search_state.amplitudes

        # Measure multiple times to find solutions
        results = []
        num_samples = min(5, len(matching_indices))

        for _ in range(num_samples):
            temp_state = search_state.copy()
            outcome, _ = temp_state.measure()
            idx = int(outcome, 2)

            if idx in self._memories and idx in matching_indices:
                memory = self._memories[idx]
                results.append(MemoryRecall(
                    memory_id=memory.id,
                    content=memory.content,
                    recall_probability=abs(search_state.amplitudes[idx]) ** 2,
                    recall_time=time.time() - memory.timestamp
                ))

        return results

    def get_memory_superposition(self) -> Dict[str, float]:
        """Get current probability distribution over memories"""
        probs = {}
        for idx, memory in self._memories.items():
            prob = abs(self._state.amplitudes[idx]) ** 2
            if prob > 1e-6:
                probs[memory.id] = prob
        return probs

    def reinforce(self, memory_id: str, factor: float = 1.5):
        """Reinforce a memory (increase amplitude)"""
        for idx, memory in self._memories.items():
            if memory.id == memory_id:
                self._state.amplitudes[idx] *= factor
                self._state.normalize()
                break

    def weaken(self, memory_id: str, factor: float = 0.5):
        """Weaken a memory (decrease amplitude)"""
        self.reinforce(memory_id, factor)

    def _apply_decoherence(self):
        """Apply decoherence based on elapsed time"""
        current_time = time.time()
        elapsed = current_time - self._last_time

        if elapsed > 0:
            rho = self._decoherence.evolve(
                self._state,
                elapsed,
                [DecoherenceChannel.DEPHASING]
            )

            # Track coherence
            from .coherence import CoherenceCalculator
            metrics = CoherenceCalculator.measure_all(self._state)
            self._coherence_tracker.record(metrics)

        self._last_time = current_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "capacity": self.capacity,
            "stored_memories": len(self._memories),
            "utilization": len(self._memories) / self.capacity,
            "num_qubits": self.num_qubits,
            "coherence": self._coherence_tracker.get_average_coherence(),
            "decoherence_time": self.decoherence_time
        }


class AssociativeQuantumMemory:
    """
    Quantum associative memory (Hopfield-like).

    Stores patterns that can be recalled from partial cues
    using quantum superposition for parallel pattern matching.
    """

    def __init__(
        self,
        pattern_size: int = 64,
        capacity: int = 10
    ):
        """
        Initialize associative memory.

        Args:
            pattern_size: Size of patterns (bits)
            capacity: Number of patterns to store
        """
        self.pattern_size = pattern_size
        self.capacity = capacity

        # Weight matrix (Hebbian learning)
        self._weights = np.zeros((pattern_size, pattern_size))

        # Stored patterns
        self._patterns: Dict[str, np.ndarray] = {}

        # Quantum register for superposition search
        self.num_qubits = int(np.ceil(np.log2(capacity + 1)))
        self._pattern_state = QuantumStateVector.zero_state(self.num_qubits)

    def store_pattern(
        self,
        pattern_id: str,
        pattern: np.ndarray
    ):
        """
        Store a pattern using Hebbian learning.

        Args:
            pattern_id: Pattern identifier
            pattern: Binary pattern array (-1 or +1)
        """
        if len(pattern) != self.pattern_size:
            raise ValueError(f"Pattern must be size {self.pattern_size}")

        # Normalize to -1, +1
        pattern = np.sign(pattern)
        pattern[pattern == 0] = 1

        # Hebbian update: W += pattern ⊗ pattern
        self._weights += np.outer(pattern, pattern)
        np.fill_diagonal(self._weights, 0)  # No self-connections

        # Store pattern
        idx = len(self._patterns)
        self._patterns[pattern_id] = pattern

        # Add to quantum superposition
        if idx < 2 ** self.num_qubits:
            self._pattern_state.amplitudes[idx] = 1.0
            self._pattern_state.normalize()

    def recall(
        self,
        cue: np.ndarray,
        iterations: int = 10,
        temperature: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Recall pattern from partial cue.

        Uses asynchronous update dynamics.

        Args:
            cue: Partial or noisy pattern
            iterations: Number of update iterations
            temperature: Noise for stochastic update (0 = deterministic)

        Returns:
            (recalled_pattern, overlap_with_stored)
        """
        state = np.sign(cue.copy())
        state[state == 0] = 1

        for _ in range(iterations):
            # Async update
            for i in np.random.permutation(self.pattern_size):
                field = np.dot(self._weights[i], state)

                if temperature > 0:
                    prob = 1 / (1 + np.exp(-2 * field / temperature))
                    state[i] = 1 if np.random.random() < prob else -1
                else:
                    state[i] = 1 if field >= 0 else -1

        # Find best matching pattern
        best_overlap = 0.0
        for pattern in self._patterns.values():
            overlap = np.dot(state, pattern) / self.pattern_size
            best_overlap = max(best_overlap, abs(overlap))

        return state, best_overlap

    def quantum_recall(
        self,
        cue: np.ndarray,
        use_grover: bool = True
    ) -> Tuple[str, np.ndarray, float]:
        """
        Quantum-enhanced recall using superposition.

        Args:
            cue: Partial pattern cue
            use_grover: Use Grover search for enhancement

        Returns:
            (pattern_id, pattern, confidence)
        """
        # Calculate overlaps with all stored patterns
        overlaps = {}
        for pid, pattern in self._patterns.items():
            overlap = np.dot(cue, pattern) / self.pattern_size
            overlaps[pid] = abs(overlap)

        if not overlaps:
            return "", np.zeros(self.pattern_size), 0.0

        if use_grover:
            # Amplitude amplification of high-overlap patterns
            threshold = 0.5
            good_patterns = [pid for pid, o in overlaps.items() if o > threshold]

            if good_patterns:
                # Grover iteration on pattern superposition
                for pid in good_patterns:
                    idx = list(self._patterns.keys()).index(pid)
                    if idx < len(self._pattern_state.amplitudes):
                        self._pattern_state.amplitudes[idx] *= 2

                self._pattern_state.normalize()

        # Measure to get pattern
        probs = np.abs(self._pattern_state.amplitudes) ** 2

        pattern_ids = list(self._patterns.keys())
        if len(probs) > len(pattern_ids):
            probs = probs[:len(pattern_ids)]
            probs = probs / np.sum(probs)

        chosen_idx = np.random.choice(len(pattern_ids), p=probs)
        chosen_id = pattern_ids[chosen_idx]

        return chosen_id, self._patterns[chosen_id], overlaps[chosen_id]


class WorkingQuantumMemory:
    """
    Working memory with quantum effects.

    Short-term memory buffer with limited capacity and
    quantum superposition of active items.
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        coherence_time: float = 30.0  # seconds
    ):
        """
        Initialize working memory.

        Args:
            capacity: Maximum items in working memory
            coherence_time: How long items stay coherent
        """
        self.capacity = capacity
        self.coherence_time = coherence_time

        # Active items in superposition
        self._items: List[QuantumMemoryItem] = []

        # Quantum state of attention
        self.num_qubits = int(np.ceil(np.log2(capacity + 1)))
        self._attention_state = QuantumStateVector.zero_state(self.num_qubits)

        # Focus index (-1 = distributed attention)
        self._focus: int = -1

    def hold(self, item: Any, priority: float = 1.0) -> QuantumMemoryItem:
        """
        Hold item in working memory.

        Args:
            item: Item to hold
            priority: Importance (affects amplitude)

        Returns:
            Created memory item
        """
        memory_item = QuantumMemoryItem(
            id=str(uuid.uuid4()),
            content=item,
            amplitude=complex(priority)
        )

        # If at capacity, remove weakest
        if len(self._items) >= self.capacity:
            # Find weakest item
            weakest_idx = min(
                range(len(self._items)),
                key=lambda i: abs(self._items[i].amplitude)
            )
            self._items.pop(weakest_idx)

        self._items.append(memory_item)
        self._update_attention_state()

        return memory_item

    def focus(self, index: int):
        """Focus attention on specific item"""
        if 0 <= index < len(self._items):
            self._focus = index

            # Increase focused item's amplitude
            self._items[index].amplitude *= 2.0

            self._update_attention_state()

    def unfocus(self):
        """Distribute attention across all items"""
        self._focus = -1
        self._update_attention_state()

    def rehearse(self, item_id: str):
        """Rehearse item to maintain in memory"""
        for item in self._items:
            if item.id == item_id:
                item.timestamp = time.time()
                item.amplitude *= 1.1  # Strengthen
                self._update_attention_state()
                break

    def access(self) -> Optional[Any]:
        """Access item from working memory (collapses superposition)"""
        if not self._items:
            return None

        # Apply decoherence
        self._apply_decay()

        # Measure attention state
        if self._focus >= 0 and self._focus < len(self._items):
            # Focused attention - deterministic
            item = self._items[self._focus]
            item.access_count += 1
            return item.content

        # Distributed attention - probabilistic
        probs = np.abs(self._attention_state.amplitudes[:len(self._items)]) ** 2

        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
            idx = np.random.choice(len(self._items), p=probs)
            item = self._items[idx]
            item.access_count += 1
            return item.content

        return None

    def get_contents(self) -> List[Tuple[Any, float]]:
        """Get all items with their current probabilities"""
        self._apply_decay()

        contents = []
        for i, item in enumerate(self._items):
            prob = abs(self._attention_state.amplitudes[i]) ** 2 if i < len(self._attention_state.amplitudes) else 0
            contents.append((item.content, prob))

        return contents

    def clear(self):
        """Clear working memory"""
        self._items.clear()
        self._focus = -1
        self._attention_state = QuantumStateVector.zero_state(self.num_qubits)

    def _update_attention_state(self):
        """Update quantum attention state from items"""
        self._attention_state = QuantumStateVector.zero_state(self.num_qubits)

        for i, item in enumerate(self._items):
            if i < len(self._attention_state.amplitudes):
                self._attention_state.amplitudes[i] = item.amplitude

        self._attention_state.normalize()

    def _apply_decay(self):
        """Apply temporal decay to items"""
        current_time = time.time()

        for item in self._items:
            age = current_time - item.timestamp
            decay = math.exp(-age / self.coherence_time)
            item.amplitude *= decay

        # Remove very weak items
        self._items = [
            item for item in self._items
            if abs(item.amplitude) > 0.01
        ]

        self._update_attention_state()

    def __len__(self) -> int:
        return len(self._items)


__all__ = [
    'MemoryType',
    'QuantumMemoryItem',
    'MemoryRecall',
    'QuantumMemoryRegister',
    'AssociativeQuantumMemory',
    'WorkingQuantumMemory',
]
