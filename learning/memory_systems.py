"""
Memory Systems for NPCPU

A multi-tier memory system providing cognitive memory capabilities for organisms:
- Sensory memory (very short term, high capacity)
- Working memory (limited capacity, active processing)
- Long-term memory (consolidated experiences)
- Episodic vs semantic memory types

Example:
    from core.simple_organism import SimpleOrganism
    from learning.memory_systems import MemorySystem, MemoryType

    organism = SimpleOrganism("Learner")
    memory = MemorySystem()
    organism.add_subsystem(memory)

    # Store an episodic memory
    memory.store(
        memory_type=MemoryType.EPISODIC,
        content={"event": "found_food", "location": (10, 20), "time": 100},
        importance=0.8
    )

    # Recall related memories
    memories = memory.recall(query={"event": "found_food"}, memory_type=MemoryType.EPISODIC)

    # Consolidate important memories to long-term storage
    memory.consolidate()
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from collections import deque
import random
import hashlib
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.abstractions import BaseSubsystem, BaseOrganism
from core.events import get_event_bus


class MemoryType(Enum):
    """Types of memories based on content"""
    EPISODIC = auto()    # Specific events and experiences
    SEMANTIC = auto()    # General knowledge and facts
    PROCEDURAL = auto()  # Skills and how to do things
    EMOTIONAL = auto()   # Emotional associations


class MemoryTier(Enum):
    """Memory storage tiers based on duration"""
    SENSORY = auto()     # Very short-term (milliseconds to seconds)
    WORKING = auto()     # Short-term active processing (seconds to minutes)
    LONG_TERM = auto()   # Consolidated permanent storage


@dataclass
class Memory:
    """
    A single memory unit.

    Attributes:
        content: The actual memory content (dict or any serializable data)
        memory_type: Type of memory (episodic, semantic, etc.)
        tier: Current storage tier
        importance: Importance score (0.0 to 1.0)
        created_at: When the memory was formed
        last_accessed: When the memory was last recalled
        access_count: Number of times accessed
        decay_rate: Rate at which memory fades
        associations: Links to related memories (by content hash)
        emotional_valence: Emotional association (-1.0 negative to 1.0 positive)
    """
    content: Dict[str, Any]
    memory_type: MemoryType = MemoryType.EPISODIC
    tier: MemoryTier = MemoryTier.SENSORY
    importance: float = 0.5
    created_at: int = 0  # Simulation tick when created
    last_accessed: int = 0
    access_count: int = 0
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            # Generate unique ID based on content hash
            content_str = json.dumps(self.content, sort_keys=True, default=str)
            self.id = hashlib.md5(content_str.encode()).hexdigest()[:12]
        if self.last_accessed == 0:
            self.last_accessed = self.created_at

    def strength(self, current_tick: int) -> float:
        """
        Calculate current memory strength based on:
        - Importance
        - Recency (time since last access)
        - Access frequency
        - Decay rate
        """
        recency = max(1, current_tick - self.last_accessed)
        recency_factor = 1.0 / (1.0 + self.decay_rate * recency)
        frequency_factor = min(1.0, self.access_count / 10.0)

        strength = (
            0.4 * self.importance +
            0.3 * recency_factor +
            0.2 * frequency_factor +
            0.1 * (1.0 if self.tier == MemoryTier.LONG_TERM else 0.5)
        )
        return max(0.0, min(1.0, strength))

    def access(self, current_tick: int) -> None:
        """Record an access to this memory"""
        self.last_accessed = current_tick
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.name,
            "tier": self.tier.name,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "decay_rate": self.decay_rate,
            "associations": self.associations,
            "emotional_valence": self.emotional_valence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Deserialize memory from dictionary"""
        return cls(
            content=data["content"],
            memory_type=MemoryType[data.get("memory_type", "EPISODIC")],
            tier=MemoryTier[data.get("tier", "SENSORY")],
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", 0),
            last_accessed=data.get("last_accessed", 0),
            access_count=data.get("access_count", 0),
            decay_rate=data.get("decay_rate", 0.1),
            associations=data.get("associations", []),
            emotional_valence=data.get("emotional_valence", 0.0),
            id=data.get("id", ""),
        )


class MemorySystem(BaseSubsystem):
    """
    Multi-tier memory system for organisms.

    Features:
    - Sensory memory (very short term): High capacity, rapid decay
    - Working memory (limited capacity): Active processing, moderate decay
    - Long-term memory (consolidated experiences): Permanent storage

    Memory flows from sensory -> working -> long-term based on:
    - Importance scores
    - Emotional significance
    - Repetition and access patterns

    Example:
        from core.simple_organism import SimpleOrganism
        from learning.memory_systems import MemorySystem, MemoryType

        organism = SimpleOrganism("Learner")
        memory = MemorySystem(
            sensory_capacity=100,
            working_capacity=7,  # Miller's law
            consolidation_threshold=0.6
        )
        organism.add_subsystem(memory)

        # Store a memory
        memory.store(
            memory_type=MemoryType.EPISODIC,
            content={"predator_seen": True, "location": (5, 10)},
            importance=0.9  # High importance = likely to be consolidated
        )

        # Later, recall similar memories
        related = memory.recall({"predator_seen": True})
        for mem in related:
            print(f"Remembered: {mem.content}")
    """

    def __init__(
        self,
        owner: Optional[BaseOrganism] = None,
        sensory_capacity: int = 100,
        working_capacity: int = 7,
        long_term_capacity: int = 10000,
        consolidation_threshold: float = 0.6,
        sensory_decay_rate: float = 0.5,
        working_decay_rate: float = 0.1,
        long_term_decay_rate: float = 0.001,
    ):
        """
        Initialize memory system.

        Args:
            owner: The organism that owns this memory system
            sensory_capacity: Max items in sensory memory
            working_capacity: Max items in working memory (default 7, Miller's law)
            long_term_capacity: Max items in long-term memory
            consolidation_threshold: Minimum importance for consolidation
            sensory_decay_rate: How fast sensory memories fade
            working_decay_rate: How fast working memories fade
            long_term_decay_rate: How fast long-term memories fade
        """
        super().__init__("memory", owner)

        # Capacity settings
        self.sensory_capacity = sensory_capacity
        self.working_capacity = working_capacity
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold

        # Decay rates by tier
        self.decay_rates = {
            MemoryTier.SENSORY: sensory_decay_rate,
            MemoryTier.WORKING: working_decay_rate,
            MemoryTier.LONG_TERM: long_term_decay_rate,
        }

        # Memory stores (using deque for efficient FIFO operations)
        self._sensory: deque = deque(maxlen=sensory_capacity)
        self._working: List[Memory] = []
        self._long_term: Dict[str, Memory] = {}

        # Internal state
        self._current_tick: int = 0
        self._consolidation_interval: int = 10  # Ticks between consolidations

        # Statistics
        self._stats = {
            "total_stored": 0,
            "total_recalled": 0,
            "consolidations": 0,
            "forgotten": 0,
        }

    def tick(self) -> None:
        """Process one time step - handle memory decay and consolidation"""
        if not self.enabled:
            return

        self._current_tick += 1

        # Decay sensory memories (remove very weak ones)
        self._decay_sensory()

        # Decay working memories (remove weak ones, may demote to sensory)
        self._decay_working()

        # Periodic consolidation from working to long-term
        if self._current_tick % self._consolidation_interval == 0:
            self._auto_consolidate()

        # Emit tick event
        bus = get_event_bus()
        bus.emit("memory.tick", {
            "organism_id": self.owner.id if self.owner else "unknown",
            "sensory_count": len(self._sensory),
            "working_count": len(self._working),
            "long_term_count": len(self._long_term),
        }, source="memory_system")

    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        importance: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> Memory:
        """
        Store a new memory.

        Memories always start in sensory tier and may be promoted based on importance.

        Args:
            memory_type: Type of memory (episodic, semantic, etc.)
            content: The actual memory content
            importance: Importance score (0.0 to 1.0)
            emotional_valence: Emotional association (-1.0 to 1.0)

        Returns:
            The created Memory object
        """
        memory = Memory(
            content=content,
            memory_type=memory_type,
            tier=MemoryTier.SENSORY,
            importance=max(0.0, min(1.0, importance)),
            created_at=self._current_tick,
            decay_rate=self.decay_rates[MemoryTier.SENSORY],
            emotional_valence=max(-1.0, min(1.0, emotional_valence)),
        )

        # Add to sensory memory
        self._sensory.append(memory)
        self._stats["total_stored"] += 1

        # High importance memories go directly to working memory
        if importance >= self.consolidation_threshold:
            self._promote_to_working(memory)

        # Very high importance or strong emotional memories go to long-term
        if importance >= 0.9 or abs(emotional_valence) >= 0.9:
            self._promote_to_long_term(memory)

        # Find and create associations
        self._create_associations(memory)

        # Emit event
        bus = get_event_bus()
        bus.emit("memory.stored", {
            "organism_id": self.owner.id if self.owner else "unknown",
            "memory_id": memory.id,
            "memory_type": memory_type.name,
            "importance": importance,
        }, source="memory_system")

        return memory

    def recall(
        self,
        query: Dict[str, Any],
        memory_type: Optional[MemoryType] = None,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Retrieve relevant memories based on query.

        Uses content similarity matching to find related memories.

        Args:
            query: Dictionary with key-value pairs to match
            memory_type: Optional filter by memory type
            tier: Optional filter by memory tier
            limit: Maximum number of memories to return

        Returns:
            List of matching Memory objects, sorted by relevance
        """
        results: List[Tuple[float, Memory]] = []

        # Search all tiers unless specified
        tiers_to_search = [tier] if tier else list(MemoryTier)

        for search_tier in tiers_to_search:
            memories = self._get_memories_by_tier(search_tier)

            for memory in memories:
                # Filter by type if specified
                if memory_type and memory.memory_type != memory_type:
                    continue

                # Calculate relevance score
                relevance = self._calculate_relevance(query, memory)

                if relevance > 0:
                    # Update access stats
                    memory.access(self._current_tick)
                    results.append((relevance, memory))

        # Sort by relevance (descending) and return top results
        results.sort(key=lambda x: x[0], reverse=True)

        recalled = [mem for _, mem in results[:limit]]

        self._stats["total_recalled"] += len(recalled)

        # Emit event
        if recalled:
            bus = get_event_bus()
            bus.emit("memory.recalled", {
                "organism_id": self.owner.id if self.owner else "unknown",
                "query": query,
                "count": len(recalled),
            }, source="memory_system")

        return recalled

    def consolidate(self) -> int:
        """
        Manually trigger consolidation of important memories to long-term storage.

        Returns:
            Number of memories consolidated
        """
        return self._auto_consolidate()

    def forget(self, memory_id: str) -> bool:
        """
        Explicitly forget a memory by ID.

        Args:
            memory_id: The ID of the memory to forget

        Returns:
            True if memory was found and removed
        """
        # Check long-term
        if memory_id in self._long_term:
            del self._long_term[memory_id]
            self._stats["forgotten"] += 1
            return True

        # Check working
        for i, mem in enumerate(self._working):
            if mem.id == memory_id:
                self._working.pop(i)
                self._stats["forgotten"] += 1
                return True

        # Check sensory
        for i, mem in enumerate(self._sensory):
            if mem.id == memory_id:
                del self._sensory[i]
                self._stats["forgotten"] += 1
                return True

        return False

    def get_working_memory(self) -> List[Memory]:
        """Get current contents of working memory"""
        return self._working.copy()

    def get_long_term_memory(self, memory_type: Optional[MemoryType] = None) -> List[Memory]:
        """Get long-term memories, optionally filtered by type"""
        memories = list(self._long_term.values())
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        return memories

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _get_memories_by_tier(self, tier: MemoryTier) -> List[Memory]:
        """Get all memories in a specific tier"""
        if tier == MemoryTier.SENSORY:
            return list(self._sensory)
        elif tier == MemoryTier.WORKING:
            return self._working.copy()
        else:  # LONG_TERM
            return list(self._long_term.values())

    def _calculate_relevance(self, query: Dict[str, Any], memory: Memory) -> float:
        """
        Calculate relevance score between query and memory.

        Uses key overlap and value matching.
        """
        if not query:
            return memory.strength(self._current_tick)

        matches = 0
        total_keys = len(query)

        for key, value in query.items():
            if key in memory.content:
                if memory.content[key] == value:
                    matches += 1
                elif isinstance(value, (int, float)) and isinstance(memory.content[key], (int, float)):
                    # Fuzzy numeric matching
                    if abs(value - memory.content[key]) < abs(value) * 0.2:
                        matches += 0.5

        if total_keys == 0:
            return 0

        key_match_score = matches / total_keys
        strength_score = memory.strength(self._current_tick)

        return 0.7 * key_match_score + 0.3 * strength_score

    def _promote_to_working(self, memory: Memory) -> bool:
        """Promote a memory to working memory"""
        if memory.tier == MemoryTier.LONG_TERM:
            return False  # Already in long-term

        if len(self._working) >= self.working_capacity:
            # Remove weakest memory from working
            self._working.sort(key=lambda m: m.strength(self._current_tick))
            self._working.pop(0)

        memory.tier = MemoryTier.WORKING
        memory.decay_rate = self.decay_rates[MemoryTier.WORKING]

        if memory not in self._working:
            self._working.append(memory)

        return True

    def _promote_to_long_term(self, memory: Memory) -> bool:
        """Promote a memory to long-term storage"""
        if memory.id in self._long_term:
            return False  # Already stored

        if len(self._long_term) >= self.long_term_capacity:
            # Remove weakest long-term memory
            weakest_id = min(
                self._long_term.keys(),
                key=lambda k: self._long_term[k].strength(self._current_tick)
            )
            del self._long_term[weakest_id]

        memory.tier = MemoryTier.LONG_TERM
        memory.decay_rate = self.decay_rates[MemoryTier.LONG_TERM]
        self._long_term[memory.id] = memory

        # Remove from working if present
        self._working = [m for m in self._working if m.id != memory.id]

        self._stats["consolidations"] += 1
        return True

    def _create_associations(self, new_memory: Memory) -> None:
        """Create associations between new memory and existing memories"""
        # Find memories with overlapping content keys
        for memory in list(self._long_term.values())[:100]:  # Limit search
            overlap = set(new_memory.content.keys()) & set(memory.content.keys())
            if overlap:
                if memory.id not in new_memory.associations:
                    new_memory.associations.append(memory.id)
                if new_memory.id not in memory.associations:
                    memory.associations.append(new_memory.id)

    def _decay_sensory(self) -> None:
        """Remove very weak sensory memories"""
        threshold = 0.1
        to_remove = []

        for i, memory in enumerate(self._sensory):
            if memory.strength(self._current_tick) < threshold:
                to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(to_remove):
            del self._sensory[i]
            self._stats["forgotten"] += 1

    def _decay_working(self) -> None:
        """Decay working memories, removing weak ones"""
        threshold = 0.2
        surviving = []

        for memory in self._working:
            if memory.strength(self._current_tick) >= threshold:
                surviving.append(memory)
            else:
                self._stats["forgotten"] += 1

        self._working = surviving

    def _auto_consolidate(self) -> int:
        """Automatically consolidate important working memories"""
        count = 0

        for memory in self._working[:]:
            # Consolidate if:
            # - High importance
            # - Accessed multiple times
            # - Strong emotional valence
            should_consolidate = (
                memory.importance >= self.consolidation_threshold or
                memory.access_count >= 3 or
                abs(memory.emotional_valence) >= 0.7
            )

            if should_consolidate and self._promote_to_long_term(memory):
                count += 1

        return count

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Get current memory system state"""
        return {
            "current_tick": self._current_tick,
            "sensory": [m.to_dict() for m in self._sensory],
            "working": [m.to_dict() for m in self._working],
            "long_term": {k: v.to_dict() for k, v in self._long_term.items()},
            "stats": self._stats.copy(),
            "settings": {
                "sensory_capacity": self.sensory_capacity,
                "working_capacity": self.working_capacity,
                "long_term_capacity": self.long_term_capacity,
                "consolidation_threshold": self.consolidation_threshold,
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore memory system state"""
        self._current_tick = state.get("current_tick", 0)

        # Restore sensory memories
        self._sensory.clear()
        for mem_data in state.get("sensory", []):
            self._sensory.append(Memory.from_dict(mem_data))

        # Restore working memories
        self._working = [Memory.from_dict(m) for m in state.get("working", [])]

        # Restore long-term memories
        self._long_term = {
            k: Memory.from_dict(v)
            for k, v in state.get("long_term", {}).items()
        }

        # Restore stats
        self._stats = state.get("stats", self._stats)

        # Restore settings
        settings = state.get("settings", {})
        self.sensory_capacity = settings.get("sensory_capacity", self.sensory_capacity)
        self.working_capacity = settings.get("working_capacity", self.working_capacity)
        self.long_term_capacity = settings.get("long_term_capacity", self.long_term_capacity)
        self.consolidation_threshold = settings.get("consolidation_threshold", self.consolidation_threshold)

    def reset(self) -> None:
        """Reset memory system to initial state"""
        super().reset()
        self._sensory.clear()
        self._working.clear()
        self._long_term.clear()
        self._current_tick = 0
        self._stats = {
            "total_stored": 0,
            "total_recalled": 0,
            "consolidations": 0,
            "forgotten": 0,
        }
