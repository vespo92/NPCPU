"""
Memory Cells System

Long-term threat memory storage for rapid recall and response.
Implements immunological memory mechanisms:
- Memory T cells
- Memory B cells
- Antigen persistence
- Memory consolidation and decay
"""

import time
import uuid
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from immune.defense import ThreatType, ThreatSeverity


# ============================================================================
# Enums
# ============================================================================

class MemoryType(Enum):
    """Types of immune memory"""
    CENTRAL = "central"         # Long-lived, high recall
    EFFECTOR = "effector"       # Fast response, shorter-lived
    RESIDENT = "resident"       # Location-specific
    VIRTUAL = "virtual"         # Never seen before, but predicted


class MemoryStrength(Enum):
    """Strength of memory"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    PERMANENT = 5


class ConsolidationState(Enum):
    """State of memory consolidation"""
    FORMING = "forming"
    CONSOLIDATING = "consolidating"
    STABLE = "stable"
    DECAYING = "decaying"
    EXPIRED = "expired"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ThreatMemory:
    """Memory of a specific threat"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: ThreatType = ThreatType.ANOMALY
    threat_signature: str = ""
    signature_hash: str = ""
    memory_type: MemoryType = MemoryType.CENTRAL
    strength: MemoryStrength = MemoryStrength.MODERATE
    state: ConsolidationState = ConsolidationState.FORMING

    # Response template
    successful_response: Dict[str, Any] = field(default_factory=dict)
    defense_actions: List[str] = field(default_factory=list)

    # Encounter history
    first_encounter: float = field(default_factory=time.time)
    last_encounter: float = field(default_factory=time.time)
    encounter_count: int = 1

    # Effectiveness tracking
    effectiveness_scores: List[float] = field(default_factory=list)
    average_effectiveness: float = 0.0

    # Decay parameters
    half_life: float = 86400 * 30  # 30 days in seconds
    decay_constant: float = 0.0


@dataclass
class MemoryCluster:
    """Cluster of related memories"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    threat_types: set = field(default_factory=set)
    memory_ids: List[str] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class RecallResult:
    """Result of memory recall"""
    memory: Optional[ThreatMemory] = None
    recall_confidence: float = 0.0
    recall_speed_ms: float = 0.0
    matches: List[Tuple[str, float]] = field(default_factory=list)
    recalled_at: float = field(default_factory=time.time)


@dataclass
class MemoryTransfer:
    """Record of memory transfer/sharing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    memory_ids: List[str] = field(default_factory=list)
    transfer_type: str = "copy"  # copy, move, share
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Memory Cell System
# ============================================================================

class MemoryCellSystem:
    """
    Long-term immune memory storage system.

    Features:
    - Hierarchical memory storage
    - Fast recall with pattern matching
    - Memory consolidation
    - Decay and forgetting
    - Memory clustering for generalization

    Example:
        memory_system = MemoryCellSystem()

        # Store memory of threat
        memory = memory_system.store_memory(
            threat_type=ThreatType.MALICIOUS_INPUT,
            signature="sql_injection_v1",
            response={"action": "block", "pattern": "SELECT.*FROM"}
        )

        # Recall when encountering similar threat
        result = memory_system.recall(signature="sql_injection")
        if result.memory:
            print(f"Found response: {result.memory.successful_response}")
    """

    def __init__(
        self,
        max_memories: int = 10000,
        consolidation_interval: float = 3600,
        decay_enabled: bool = True,
        similarity_threshold: float = 0.6
    ):
        self.max_memories = max_memories
        self.consolidation_interval = consolidation_interval
        self.decay_enabled = decay_enabled
        self.similarity_threshold = similarity_threshold

        # Memory storage
        self.memories: Dict[str, ThreatMemory] = {}
        self.memory_index: Dict[str, List[str]] = defaultdict(list)  # signature_hash -> memory_ids
        self.type_index: Dict[ThreatType, List[str]] = defaultdict(list)

        # Memory clusters
        self.clusters: Dict[str, MemoryCluster] = {}

        # Statistics
        self.total_stores = 0
        self.total_recalls = 0
        self.successful_recalls = 0
        self.last_consolidation = time.time()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "memory_stored": [],
            "memory_recalled": [],
            "memory_decayed": [],
            "memory_consolidated": []
        }

    def _compute_hash(self, signature: str) -> str:
        """Compute hash of signature for indexing"""
        return hashlib.md5(signature.encode()).hexdigest()[:16]

    def store_memory(
        self,
        threat_type: ThreatType,
        signature: str,
        response: Dict[str, Any],
        effectiveness: float = 0.5,
        memory_type: MemoryType = MemoryType.CENTRAL,
        defense_actions: Optional[List[str]] = None
    ) -> ThreatMemory:
        """Store a new threat memory"""
        self.total_stores += 1
        sig_hash = self._compute_hash(signature)

        # Check for existing memory
        existing = self.find_by_signature(signature)
        if existing:
            # Update existing memory
            return self._update_memory(existing, effectiveness, response)

        # Create new memory
        memory = ThreatMemory(
            threat_type=threat_type,
            threat_signature=signature,
            signature_hash=sig_hash,
            memory_type=memory_type,
            successful_response=response,
            defense_actions=defense_actions or [],
            effectiveness_scores=[effectiveness],
            average_effectiveness=effectiveness
        )

        # Calculate decay constant from half-life
        memory.decay_constant = np.log(2) / memory.half_life

        # Store
        self.memories[memory.id] = memory
        self.memory_index[sig_hash].append(memory.id)
        self.type_index[threat_type].append(memory.id)

        # Manage capacity
        self._enforce_capacity()

        # Trigger callback
        for callback in self._callbacks["memory_stored"]:
            callback(memory)

        return memory

    def _update_memory(
        self,
        memory: ThreatMemory,
        effectiveness: float,
        response: Dict[str, Any]
    ) -> ThreatMemory:
        """Update existing memory with new encounter"""
        memory.encounter_count += 1
        memory.last_encounter = time.time()
        memory.effectiveness_scores.append(effectiveness)
        memory.average_effectiveness = np.mean(memory.effectiveness_scores)

        # Strengthen memory
        if effectiveness > memory.average_effectiveness:
            memory.strength = MemoryStrength(
                min(memory.strength.value + 1, MemoryStrength.PERMANENT.value)
            )
            memory.successful_response.update(response)

        # Update state
        if memory.state in [ConsolidationState.FORMING, ConsolidationState.DECAYING]:
            memory.state = ConsolidationState.CONSOLIDATING

        return memory

    def find_by_signature(self, signature: str) -> Optional[ThreatMemory]:
        """Find memory by exact signature match"""
        sig_hash = self._compute_hash(signature)
        memory_ids = self.memory_index.get(sig_hash, [])

        for mem_id in memory_ids:
            memory = self.memories.get(mem_id)
            if memory and memory.threat_signature == signature:
                return memory

        return None

    def recall(
        self,
        signature: Optional[str] = None,
        threat_type: Optional[ThreatType] = None,
        min_confidence: float = 0.5
    ) -> RecallResult:
        """
        Recall memory based on signature or threat type.

        Returns RecallResult with best matching memory.
        """
        start_time = time.time()
        self.total_recalls += 1

        matches: List[Tuple[str, float]] = []

        # Try exact match first
        if signature:
            exact = self.find_by_signature(signature)
            if exact:
                matches.append((exact.id, 1.0))

        # Try fuzzy match on signature
        if signature and not matches:
            sig_hash = self._compute_hash(signature)
            for mem in self.memories.values():
                similarity = self._calculate_similarity(signature, mem.threat_signature)
                if similarity >= self.similarity_threshold:
                    matches.append((mem.id, similarity))

        # Filter by threat type if specified
        if threat_type:
            type_memories = self.type_index.get(threat_type, [])
            if not matches:
                for mem_id in type_memories:
                    matches.append((mem_id, 0.5))  # Base match for type
            else:
                matches = [(m_id, score) for m_id, score in matches
                          if m_id in type_memories]

        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)

        # Calculate recall time
        recall_time = (time.time() - start_time) * 1000

        # Get best match
        best_memory = None
        confidence = 0.0
        if matches:
            best_id, confidence = matches[0]
            if confidence >= min_confidence:
                best_memory = self.memories.get(best_id)
                if best_memory:
                    self.successful_recalls += 1
                    best_memory.last_encounter = time.time()

        result = RecallResult(
            memory=best_memory,
            recall_confidence=confidence,
            recall_speed_ms=recall_time,
            matches=matches[:10]  # Top 10 matches
        )

        for callback in self._callbacks["memory_recalled"]:
            callback(result)

        return result

    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures"""
        if not sig1 or not sig2:
            return 0.0

        # Levenshtein-based similarity
        len1, len2 = len(sig1), len(sig2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Simple character overlap for efficiency
        set1, set2 = set(sig1.lower()), set(sig2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        jaccard = intersection / union if union > 0 else 0.0

        # Also check substring containment
        s1, s2 = sig1.lower(), sig2.lower()
        if s1 in s2 or s2 in s1:
            containment_bonus = 0.3
        else:
            containment_bonus = 0.0

        return min(1.0, jaccard + containment_bonus)

    def consolidate_memories(self):
        """Consolidate forming memories into stable storage"""
        consolidated = []

        for memory in self.memories.values():
            if memory.state == ConsolidationState.FORMING:
                # Check if ready for consolidation
                if memory.encounter_count >= 2 or memory.effectiveness_scores:
                    memory.state = ConsolidationState.CONSOLIDATING

            elif memory.state == ConsolidationState.CONSOLIDATING:
                # Complete consolidation
                memory.state = ConsolidationState.STABLE
                memory.half_life *= 2  # Stable memories last longer
                consolidated.append(memory)

        self.last_consolidation = time.time()

        for callback in self._callbacks["memory_consolidated"]:
            for memory in consolidated:
                callback(memory)

        return consolidated

    def apply_decay(self):
        """Apply time-based decay to memories"""
        if not self.decay_enabled:
            return

        current_time = time.time()
        decayed = []
        expired = []

        for memory in list(self.memories.values()):
            if memory.strength == MemoryStrength.PERMANENT:
                continue

            # Calculate decay based on time since last encounter
            time_elapsed = current_time - memory.last_encounter
            decay_factor = np.exp(-memory.decay_constant * time_elapsed)

            if decay_factor < 0.1:
                # Memory has expired
                memory.state = ConsolidationState.EXPIRED
                expired.append(memory.id)
            elif decay_factor < 0.5 and memory.state == ConsolidationState.STABLE:
                memory.state = ConsolidationState.DECAYING
                decayed.append(memory)

        # Remove expired memories
        for mem_id in expired:
            self._remove_memory(mem_id)

        for callback in self._callbacks["memory_decayed"]:
            for memory in decayed:
                callback(memory)

    def _remove_memory(self, memory_id: str):
        """Remove a memory from storage"""
        memory = self.memories.get(memory_id)
        if not memory:
            return

        # Remove from indexes
        sig_hash = memory.signature_hash
        if sig_hash in self.memory_index:
            self.memory_index[sig_hash] = [
                m for m in self.memory_index[sig_hash] if m != memory_id
            ]

        if memory.threat_type in self.type_index:
            self.type_index[memory.threat_type] = [
                m for m in self.type_index[memory.threat_type] if m != memory_id
            ]

        del self.memories[memory_id]

    def _enforce_capacity(self):
        """Enforce maximum memory capacity"""
        if len(self.memories) <= self.max_memories:
            return

        # Remove oldest, weakest memories first
        memories_by_priority = sorted(
            self.memories.values(),
            key=lambda m: (m.strength.value, -m.last_encounter)
        )

        to_remove = len(self.memories) - self.max_memories
        for memory in memories_by_priority[:to_remove]:
            self._remove_memory(memory.id)

    def create_cluster(
        self,
        name: str,
        memory_ids: List[str]
    ) -> MemoryCluster:
        """Create a cluster of related memories"""
        threat_types = set()
        patterns = []

        for mem_id in memory_ids:
            memory = self.memories.get(mem_id)
            if memory:
                threat_types.add(memory.threat_type)
                patterns.append(memory.threat_signature)

        cluster = MemoryCluster(
            name=name,
            threat_types=threat_types,
            memory_ids=memory_ids,
            common_patterns=patterns[:10]
        )

        self.clusters[cluster.id] = cluster
        return cluster

    def generalize_from_cluster(
        self,
        cluster_id: str
    ) -> Optional[ThreatMemory]:
        """Create a generalized memory from a cluster"""
        cluster = self.clusters.get(cluster_id)
        if not cluster or len(cluster.memory_ids) < 2:
            return None

        # Combine responses from cluster memories
        combined_response = {}
        combined_actions = set()
        avg_effectiveness = []

        for mem_id in cluster.memory_ids:
            memory = self.memories.get(mem_id)
            if memory:
                combined_response.update(memory.successful_response)
                combined_actions.update(memory.defense_actions)
                avg_effectiveness.append(memory.average_effectiveness)

        # Create generalized memory
        general_memory = ThreatMemory(
            threat_type=list(cluster.threat_types)[0] if cluster.threat_types else ThreatType.ANOMALY,
            threat_signature=f"cluster_{cluster.name}",
            memory_type=MemoryType.VIRTUAL,
            strength=MemoryStrength.MODERATE,
            successful_response=combined_response,
            defense_actions=list(combined_actions),
            average_effectiveness=np.mean(avg_effectiveness) if avg_effectiveness else 0.5
        )

        self.memories[general_memory.id] = general_memory
        return general_memory

    def export_memories(
        self,
        memory_ids: Optional[List[str]] = None
    ) -> bytes:
        """Export memories for transfer"""
        if memory_ids:
            memories = {mid: self.memories[mid] for mid in memory_ids if mid in self.memories}
        else:
            memories = self.memories

        return pickle.dumps(memories)

    def import_memories(
        self,
        data: bytes,
        merge: bool = True
    ) -> int:
        """Import memories from export"""
        imported_memories = pickle.loads(data)
        count = 0

        for mem_id, memory in imported_memories.items():
            if merge and mem_id in self.memories:
                # Merge with existing
                self._update_memory(self.memories[mem_id],
                                   memory.average_effectiveness,
                                   memory.successful_response)
            else:
                self.memories[mem_id] = memory
                self.memory_index[memory.signature_hash].append(mem_id)
                self.type_index[memory.threat_type].append(mem_id)
            count += 1

        return count

    def tick(self):
        """Process one cycle"""
        current_time = time.time()

        # Periodic consolidation
        if current_time - self.last_consolidation > self.consolidation_interval:
            self.consolidate_memories()

        # Periodic decay
        if self.decay_enabled and np.random.random() < 0.01:
            self.apply_decay()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "total_memories": len(self.memories),
            "total_stores": self.total_stores,
            "total_recalls": self.total_recalls,
            "successful_recalls": self.successful_recalls,
            "recall_rate": self.successful_recalls / max(1, self.total_recalls),
            "clusters": len(self.clusters),
            "memories_by_type": {
                tt.value: len(self.type_index.get(tt, []))
                for tt in ThreatType
            },
            "memories_by_strength": {
                ms.name: len([m for m in self.memories.values() if m.strength == ms])
                for ms in MemoryStrength
            },
            "memories_by_state": {
                cs.value: len([m for m in self.memories.values() if m.state == cs])
                for cs in ConsolidationState
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Memory Cell System Demo")
    print("=" * 50)

    # Create system
    memory_system = MemoryCellSystem()

    print("\n1. Storing memories...")

    # Store some memories
    mem1 = memory_system.store_memory(
        threat_type=ThreatType.MALICIOUS_INPUT,
        signature="sql_injection_select",
        response={"action": "block", "pattern": "SELECT.*FROM"},
        effectiveness=0.8
    )
    print(f"   Stored: {mem1.threat_signature}")

    mem2 = memory_system.store_memory(
        threat_type=ThreatType.MALICIOUS_INPUT,
        signature="sql_injection_union",
        response={"action": "block", "pattern": "UNION.*SELECT"},
        effectiveness=0.75
    )
    print(f"   Stored: {mem2.threat_signature}")

    mem3 = memory_system.store_memory(
        threat_type=ThreatType.INTRUSION,
        signature="unauthorized_access_admin",
        response={"action": "alert", "escalate": True},
        effectiveness=0.9
    )
    print(f"   Stored: {mem3.threat_signature}")

    print("\n2. Testing recall...")

    # Exact match
    result = memory_system.recall(signature="sql_injection_select")
    print(f"   Exact match: {result.memory.threat_signature if result.memory else 'None'}")
    print(f"   Confidence: {result.recall_confidence:.2f}")
    print(f"   Speed: {result.recall_speed_ms:.2f}ms")

    # Fuzzy match
    result = memory_system.recall(signature="sql_injection")
    print(f"   Fuzzy match: {result.memory.threat_signature if result.memory else 'None'}")
    print(f"   Confidence: {result.recall_confidence:.2f}")

    # Type-based recall
    result = memory_system.recall(threat_type=ThreatType.INTRUSION)
    print(f"   Type match: {result.memory.threat_signature if result.memory else 'None'}")

    print("\n3. Creating memory cluster...")
    cluster = memory_system.create_cluster(
        name="sql_injections",
        memory_ids=[mem1.id, mem2.id]
    )
    print(f"   Cluster created: {cluster.name}")
    print(f"   Patterns: {cluster.common_patterns}")

    print("\n4. Generalizing from cluster...")
    general = memory_system.generalize_from_cluster(cluster.id)
    if general:
        print(f"   Generalized memory: {general.threat_signature}")
        print(f"   Effectiveness: {general.average_effectiveness:.2f}")

    print("\n5. Consolidating memories...")
    consolidated = memory_system.consolidate_memories()
    print(f"   Consolidated: {len(consolidated)} memories")

    print("\n6. Statistics:")
    stats = memory_system.get_statistics()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Recall rate: {stats['recall_rate']:.2%}")
    for state, count in stats['memories_by_state'].items():
        if count > 0:
            print(f"     {state}: {count}")
