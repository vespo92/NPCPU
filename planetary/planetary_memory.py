"""
Planetary Memory - Geological Timescale Storage

Implements long-term memory storage at planetary scales, operating
on geological timescales. This memory persists across epochs and
stores the deep history of planetary consciousness.

Features:
- Geological timescale memory persistence
- Epoch-based memory organization
- Deep pattern storage and retrieval
- Cross-epoch pattern recognition
- Consciousness lineage tracking
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
import hashlib

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class GeologicalEra(Enum):
    """Geological eras for memory organization"""
    ORIGIN = "origin"                  # System initialization
    EARLY = "early"                    # Initial evolution
    EXPANSION = "expansion"            # Population growth
    DIVERSIFICATION = "diversification"  # Species radiation
    STABILITY = "stability"            # Equilibrium period
    CRISIS = "crisis"                  # Major disruption
    RECOVERY = "recovery"              # Post-crisis rebuilding
    TRANSCENDENCE = "transcendence"    # Higher consciousness


class MemoryType(Enum):
    """Types of planetary memory"""
    EVENT = "event"                    # Specific events
    PATTERN = "pattern"                # Recurring patterns
    STATE = "state"                    # System state snapshots
    LINEAGE = "lineage"                # Evolutionary lineages
    WISDOM = "wisdom"                  # Accumulated knowledge
    TRAUMA = "trauma"                  # Crisis memories


class MemoryPersistence(Enum):
    """Memory persistence levels"""
    EPHEMERAL = "ephemeral"            # Fades quickly
    SHORT_TERM = "short_term"          # Persists for ticks
    LONG_TERM = "long_term"            # Persists for epochs
    PERMANENT = "permanent"            # Never fades


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PlanetaryMemoryItem:
    """A single memory item"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.EVENT
    persistence: MemoryPersistence = MemoryPersistence.LONG_TERM

    # Temporal
    created_tick: int = 0
    created_epoch: int = 0
    era: GeologicalEra = GeologicalEra.ORIGIN
    last_accessed: int = 0

    # Metadata
    importance: float = 0.5           # 0-1 importance
    access_count: int = 0
    associations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Encoding
    embedding: List[float] = field(default_factory=list)
    hash_key: str = ""


@dataclass
class Epoch:
    """A geological epoch"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    number: int = 0
    era: GeologicalEra = GeologicalEra.ORIGIN
    start_tick: int = 0
    end_tick: Optional[int] = None

    # State
    population_peak: int = 0
    consciousness_peak: float = 0.0
    biodiversity_index: float = 0.0

    # Events
    significant_events: List[str] = field(default_factory=list)
    extinction_events: int = 0
    emergence_events: int = 0

    # Summary
    summary: str = ""


@dataclass
class MemoryConfig:
    """Configuration for planetary memory"""
    epoch_length: int = 10000         # Ticks per epoch
    consolidation_interval: int = 100  # Ticks between consolidations
    max_short_term: int = 1000        # Max short-term memories
    max_long_term: int = 10000        # Max long-term memories
    decay_rate: float = 0.001         # Memory decay rate
    importance_threshold: float = 0.3  # Min importance to retain


# ============================================================================
# Planetary Memory System
# ============================================================================

class PlanetaryMemory:
    """
    Geological timescale memory storage.

    Stores and retrieves memories across planetary epochs,
    enabling pattern recognition across vast timescales.

    Example:
        memory = PlanetaryMemory()

        # Store a memory
        memory.store(
            content={"event": "first_consciousness"},
            memory_type=MemoryType.EVENT,
            importance=0.9
        )

        # Query memories
        results = memory.query(
            era=GeologicalEra.ORIGIN,
            memory_type=MemoryType.EVENT
        )
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Memory stores
        self.short_term: Dict[str, PlanetaryMemoryItem] = {}
        self.long_term: Dict[str, PlanetaryMemoryItem] = {}
        self.permanent: Dict[str, PlanetaryMemoryItem] = {}

        # Epochs
        self.epochs: List[Epoch] = []
        self.current_epoch: Optional[Epoch] = None

        # Indexing
        self.era_index: Dict[GeologicalEra, List[str]] = defaultdict(list)
        self.type_index: Dict[MemoryType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)

        # Tracking
        self.tick_count = 0
        self.total_memories_stored = 0
        self.total_memories_forgotten = 0

        # Patterns
        self.discovered_patterns: List[Dict[str, Any]] = []

        # Initialize first epoch
        self._start_new_epoch()

    def _start_new_epoch(self):
        """Start a new geological epoch"""
        if self.current_epoch:
            self.current_epoch.end_tick = self.tick_count
            self.epochs.append(self.current_epoch)

        epoch_num = len(self.epochs)

        # Determine era based on history
        if epoch_num == 0:
            era = GeologicalEra.ORIGIN
        elif epoch_num < 3:
            era = GeologicalEra.EARLY
        elif epoch_num < 6:
            era = GeologicalEra.EXPANSION
        else:
            era = GeologicalEra.DIVERSIFICATION

        self.current_epoch = Epoch(
            number=epoch_num,
            era=era,
            start_tick=self.tick_count
        )

    def tick(self):
        """Process one memory cycle"""
        self.tick_count += 1

        # Memory decay
        if self.tick_count % 10 == 0:
            self._apply_decay()

        # Consolidation
        if self.tick_count % self.config.consolidation_interval == 0:
            self._consolidate_memories()

        # Check for epoch transition
        if self.tick_count % self.config.epoch_length == 0:
            self._finalize_epoch()
            self._start_new_epoch()

        # Pattern detection
        if self.tick_count % 500 == 0:
            self._detect_patterns()

    def store(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.EVENT,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        persistence: Optional[MemoryPersistence] = None
    ) -> str:
        """Store a new memory"""
        # Determine persistence
        if persistence is None:
            if importance > 0.8:
                persistence = MemoryPersistence.PERMANENT
            elif importance > 0.5:
                persistence = MemoryPersistence.LONG_TERM
            else:
                persistence = MemoryPersistence.SHORT_TERM

        # Create memory item
        memory = PlanetaryMemoryItem(
            content=content,
            memory_type=memory_type,
            persistence=persistence,
            created_tick=self.tick_count,
            created_epoch=self.current_epoch.number if self.current_epoch else 0,
            era=self.current_epoch.era if self.current_epoch else GeologicalEra.ORIGIN,
            last_accessed=self.tick_count,
            importance=importance,
            tags=tags or [],
            hash_key=self._compute_hash(content)
        )

        # Create embedding
        memory.embedding = self._create_embedding(content)

        # Store in appropriate location
        if persistence == MemoryPersistence.PERMANENT:
            self.permanent[memory.id] = memory
        elif persistence == MemoryPersistence.LONG_TERM:
            self.long_term[memory.id] = memory
        else:
            self.short_term[memory.id] = memory

        # Index
        self.era_index[memory.era].append(memory.id)
        self.type_index[memory.memory_type].append(memory.id)
        for tag in memory.tags:
            self.tag_index[tag].append(memory.id)

        self.total_memories_stored += 1

        # Track epoch events
        if self.current_epoch:
            if memory_type == MemoryType.EVENT and importance > 0.7:
                self.current_epoch.significant_events.append(memory.id)

        return memory.id

    def recall(self, memory_id: str) -> Optional[PlanetaryMemoryItem]:
        """Recall a specific memory"""
        memory = (
            self.short_term.get(memory_id) or
            self.long_term.get(memory_id) or
            self.permanent.get(memory_id)
        )

        if memory:
            memory.last_accessed = self.tick_count
            memory.access_count += 1
            # Strengthen memory
            memory.importance = min(1.0, memory.importance + 0.01)

        return memory

    def query(
        self,
        era: Optional[GeologicalEra] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 100
    ) -> List[PlanetaryMemoryItem]:
        """Query memories by criteria"""
        results = []

        # Collect candidate IDs
        candidate_ids = set()

        if era:
            candidate_ids.update(self.era_index[era])
        if memory_type:
            candidate_ids.update(self.type_index[memory_type])
        if tags:
            for tag in tags:
                candidate_ids.update(self.tag_index[tag])

        if not era and not memory_type and not tags:
            # All memories
            candidate_ids = set(self.short_term.keys())
            candidate_ids.update(self.long_term.keys())
            candidate_ids.update(self.permanent.keys())

        # Filter and collect
        for memory_id in candidate_ids:
            memory = self.recall(memory_id)
            if memory and memory.importance >= min_importance:
                # Additional filters
                if era and memory.era != era:
                    continue
                if memory_type and memory.memory_type != memory_type:
                    continue
                if tags and not all(t in memory.tags for t in tags):
                    continue

                results.append(memory)

            if len(results) >= limit:
                break

        # Sort by importance
        results.sort(key=lambda m: m.importance, reverse=True)

        return results[:limit]

    def query_by_similarity(
        self,
        content: Dict[str, Any],
        limit: int = 10
    ) -> List[Tuple[PlanetaryMemoryItem, float]]:
        """Query memories by content similarity"""
        query_embedding = self._create_embedding(content)
        results = []

        all_memories = list(self.short_term.values())
        all_memories.extend(self.long_term.values())
        all_memories.extend(self.permanent.values())

        for memory in all_memories:
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                results.append((memory, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _compute_hash(self, content: Dict[str, Any]) -> str:
        """Compute hash for content"""
        content_str = str(sorted(content.items()))
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _create_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Create simple embedding for content"""
        # Simplified embedding - in practice would use neural encoder
        np.random.seed(hash(str(content)) % (2**32))
        return list(np.random.randn(64))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if not a or not b or len(a) != len(b):
            return 0.0

        a = np.array(a)
        b = np.array(b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _apply_decay(self):
        """Apply memory decay"""
        to_forget = []

        for memory_id, memory in self.short_term.items():
            # Decay based on time since access
            time_since_access = self.tick_count - memory.last_accessed
            decay = self.config.decay_rate * time_since_access

            memory.importance = max(0, memory.importance - decay)

            if memory.importance < self.config.importance_threshold:
                to_forget.append(memory_id)

        for memory_id in to_forget:
            self._forget(memory_id)

    def _forget(self, memory_id: str):
        """Remove a memory"""
        memory = self.short_term.pop(memory_id, None)
        if not memory:
            memory = self.long_term.pop(memory_id, None)

        if memory:
            # Remove from indices
            if memory.era in self.era_index:
                if memory_id in self.era_index[memory.era]:
                    self.era_index[memory.era].remove(memory_id)
            if memory.memory_type in self.type_index:
                if memory_id in self.type_index[memory.memory_type]:
                    self.type_index[memory.memory_type].remove(memory_id)
            for tag in memory.tags:
                if memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)

            self.total_memories_forgotten += 1

    def _consolidate_memories(self):
        """Consolidate short-term to long-term"""
        to_promote = []

        for memory_id, memory in self.short_term.items():
            # Promote if accessed frequently or high importance
            if memory.access_count > 3 or memory.importance > 0.6:
                to_promote.append(memory_id)

        for memory_id in to_promote:
            memory = self.short_term.pop(memory_id)
            memory.persistence = MemoryPersistence.LONG_TERM
            self.long_term[memory_id] = memory

        # Enforce limits
        while len(self.short_term) > self.config.max_short_term:
            # Remove lowest importance
            lowest = min(self.short_term.values(), key=lambda m: m.importance)
            self._forget(lowest.id)

    def _finalize_epoch(self):
        """Finalize current epoch with summary"""
        if not self.current_epoch:
            return

        # Collect epoch statistics
        epoch_memories = [
            m for m in list(self.long_term.values()) + list(self.permanent.values())
            if m.created_epoch == self.current_epoch.number
        ]

        # Generate summary
        event_count = sum(1 for m in epoch_memories if m.memory_type == MemoryType.EVENT)
        pattern_count = sum(1 for m in epoch_memories if m.memory_type == MemoryType.PATTERN)

        self.current_epoch.summary = (
            f"Epoch {self.current_epoch.number} ({self.current_epoch.era.value}): "
            f"{event_count} events, {pattern_count} patterns, "
            f"{len(self.current_epoch.significant_events)} significant"
        )

        # Store epoch as permanent memory
        self.store(
            content={
                "type": "epoch_summary",
                "epoch_number": self.current_epoch.number,
                "era": self.current_epoch.era.value,
                "duration": self.current_epoch.end_tick - self.current_epoch.start_tick if self.current_epoch.end_tick else 0,
                "summary": self.current_epoch.summary
            },
            memory_type=MemoryType.STATE,
            importance=0.85,
            persistence=MemoryPersistence.PERMANENT,
            tags=["epoch", self.current_epoch.era.value]
        )

    def _detect_patterns(self):
        """Detect recurring patterns across memories"""
        # Group memories by type
        type_memories: Dict[MemoryType, List[PlanetaryMemoryItem]] = defaultdict(list)

        for memory in list(self.long_term.values()) + list(self.permanent.values()):
            type_memories[memory.memory_type].append(memory)

        # Look for patterns in event sequences
        events = type_memories[MemoryType.EVENT]
        if len(events) > 10:
            # Simple pattern: repeated tags
            tag_counts = defaultdict(int)
            for event in events[-50:]:
                for tag in event.tags:
                    tag_counts[tag] += 1

            for tag, count in tag_counts.items():
                if count > 5:
                    pattern = {
                        "type": "recurring_tag",
                        "tag": tag,
                        "frequency": count,
                        "detected_at": self.tick_count
                    }

                    if pattern not in self.discovered_patterns:
                        self.discovered_patterns.append(pattern)
                        self.store(
                            content=pattern,
                            memory_type=MemoryType.PATTERN,
                            importance=0.6,
                            tags=["pattern", tag]
                        )

    # ========================================================================
    # Public API
    # ========================================================================

    def get_epoch_history(self) -> List[Dict[str, Any]]:
        """Get history of all epochs"""
        return [
            {
                "number": e.number,
                "era": e.era.value,
                "start_tick": e.start_tick,
                "end_tick": e.end_tick,
                "summary": e.summary
            }
            for e in self.epochs
        ]

    def get_current_epoch_info(self) -> Dict[str, Any]:
        """Get current epoch info"""
        if not self.current_epoch:
            return {}

        return {
            "number": self.current_epoch.number,
            "era": self.current_epoch.era.value,
            "start_tick": self.current_epoch.start_tick,
            "duration": self.tick_count - self.current_epoch.start_tick,
            "significant_events": len(self.current_epoch.significant_events)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "tick_count": self.tick_count,
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "permanent_count": len(self.permanent),
            "total_stored": self.total_memories_stored,
            "total_forgotten": self.total_memories_forgotten,
            "epochs_completed": len(self.epochs),
            "patterns_discovered": len(self.discovered_patterns)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Planetary Memory Demo")
    print("=" * 50)

    config = MemoryConfig(
        epoch_length=100,
        consolidation_interval=10
    )
    memory = PlanetaryMemory(config)

    print(f"\n1. Initialized planetary memory system")

    # Store some memories
    print("\n2. Storing memories...")
    for i in range(50):
        memory.store(
            content={"event": f"event_{i}", "value": np.random.random()},
            memory_type=MemoryType.EVENT,
            importance=np.random.uniform(0.3, 0.9),
            tags=["simulation", f"group_{i % 5}"]
        )
        memory.tick()

    # Store important memories
    memory.store(
        content={"event": "first_consciousness", "level": 0.5},
        memory_type=MemoryType.EVENT,
        importance=0.95,
        tags=["consciousness", "milestone"]
    )

    # Run more ticks
    print("\n3. Running memory cycles...")
    for i in range(200):
        memory.tick()

        if i % 50 == 0:
            stats = memory.get_statistics()
            print(f"   Tick {i}: ST={stats['short_term_count']}, "
                  f"LT={stats['long_term_count']}, "
                  f"PERM={stats['permanent_count']}")

    # Query memories
    print("\n4. Querying memories...")
    results = memory.query(memory_type=MemoryType.EVENT, min_importance=0.7, limit=5)
    print(f"   Found {len(results)} important events")

    # Statistics
    print("\n5. Final statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n6. Current epoch:")
    epoch_info = memory.get_current_epoch_info()
    for key, value in epoch_info.items():
        print(f"   {key}: {value}")
