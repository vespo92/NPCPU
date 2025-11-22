"""
Spatial and Experiential Memory System

Enables NPCPU organisms to remember and learn from experiences:
- Spatial memory: Remember locations (resources, dangers, shelter)
- Episodic memory: Remember events and outcomes
- Procedural memory: Learn action sequences
- Social memory: Remember other entities
"""

import time
import uuid
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minecraft.bridge import Position


# ============================================================================
# Enums
# ============================================================================

class MemoryType(Enum):
    """Types of memories"""
    LOCATION = "location"           # Places
    RESOURCE = "resource"           # Resource locations
    DANGER = "danger"               # Dangerous areas
    SHELTER = "shelter"             # Safe places
    ENTITY = "entity"               # Other beings
    EVENT = "event"                 # Things that happened
    SKILL = "skill"                 # Learned procedures


class EmotionalValence(Enum):
    """Emotional coloring of memories"""
    VERY_POSITIVE = 2
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    VERY_NEGATIVE = -2


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SpatialMemory:
    """Memory of a location"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position: Position = field(default_factory=lambda: Position(0, 0, 0))
    memory_type: MemoryType = MemoryType.LOCATION
    label: str = ""                  # What it is
    details: Dict[str, Any] = field(default_factory=dict)
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    strength: float = 1.0            # Memory strength (decays over time)
    access_count: int = 0            # Times recalled
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


@dataclass
class EpisodicMemory:
    """Memory of an event/episode"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""                # success, failure, neutral
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    lesson: str = ""                 # What was learned
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    location: Optional[Position] = None


@dataclass
class SocialMemory:
    """Memory of another entity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    entity_type: str = ""
    name: Optional[str] = None
    relationship: str = "stranger"   # friend, enemy, neutral, stranger
    trust_level: float = 0.5         # 0-1
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    last_seen_position: Optional[Position] = None
    last_seen_time: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)


@dataclass
class ProceduralMemory:
    """Memory of how to do something"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skill_name: str = ""
    action_sequence: List[str] = field(default_factory=list)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.5
    execution_count: int = 0
    last_executed: float = field(default_factory=time.time)


# ============================================================================
# Memory System
# ============================================================================

class MemorySystem:
    """
    Complete memory system for digital organisms.

    Features:
    - Spatial memory with decay
    - Episodic memory for learning
    - Social memory for relationships
    - Procedural memory for skills
    - Memory consolidation and forgetting

    Example:
        memory = MemorySystem()

        # Remember a resource location
        memory.remember_location(
            position=Position(100, 64, 200),
            memory_type=MemoryType.RESOURCE,
            label="iron_ore_deposit",
            valence=EmotionalValence.POSITIVE
        )

        # Recall nearby resources
        resources = memory.recall_nearby(
            position=current_pos,
            memory_type=MemoryType.RESOURCE,
            radius=100
        )
    """

    def __init__(
        self,
        decay_rate: float = 0.001,
        max_memories: int = 1000,
        consolidation_threshold: float = 0.3
    ):
        self.decay_rate = decay_rate
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold

        # Memory stores
        self.spatial_memories: Dict[str, SpatialMemory] = {}
        self.episodic_memories: List[EpisodicMemory] = []
        self.social_memories: Dict[str, SocialMemory] = {}
        self.procedural_memories: Dict[str, ProceduralMemory] = {}

        # Spatial index (chunked for efficiency)
        self._spatial_index: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self._chunk_size = 64

        # Statistics
        self.total_memories_created = 0
        self.total_memories_forgotten = 0

    def _get_chunk_key(self, pos: Position) -> Tuple[int, int]:
        """Get chunk key for a position"""
        return (int(pos.x // self._chunk_size), int(pos.z // self._chunk_size))

    # ========================================================================
    # Spatial Memory
    # ========================================================================

    def remember_location(
        self,
        position: Position,
        memory_type: MemoryType,
        label: str,
        details: Optional[Dict[str, Any]] = None,
        valence: EmotionalValence = EmotionalValence.NEUTRAL
    ) -> SpatialMemory:
        """Remember a location"""
        # Check for existing memory nearby
        existing = self.recall_at_position(position, memory_type, radius=5)
        if existing:
            # Reinforce existing memory
            existing.strength = min(1.0, existing.strength + 0.2)
            existing.access_count += 1
            existing.last_accessed = time.time()
            if details:
                existing.details.update(details)
            return existing

        # Create new memory
        memory = SpatialMemory(
            position=position,
            memory_type=memory_type,
            label=label,
            details=details or {},
            valence=valence
        )

        self.spatial_memories[memory.id] = memory
        self._spatial_index[self._get_chunk_key(position)].append(memory.id)
        self.total_memories_created += 1

        self._enforce_memory_limit()
        return memory

    def recall_at_position(
        self,
        position: Position,
        memory_type: Optional[MemoryType] = None,
        radius: float = 10
    ) -> Optional[SpatialMemory]:
        """Recall memory at a specific position"""
        chunk_key = self._get_chunk_key(position)

        # Search nearby chunks
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                nearby_key = (chunk_key[0] + dx, chunk_key[1] + dz)
                for memory_id in self._spatial_index.get(nearby_key, []):
                    memory = self.spatial_memories.get(memory_id)
                    if not memory:
                        continue

                    if memory_type and memory.memory_type != memory_type:
                        continue

                    distance = position.distance_to(memory.position)
                    if distance <= radius:
                        self._access_memory(memory)
                        return memory

        return None

    def recall_nearby(
        self,
        position: Position,
        memory_type: Optional[MemoryType] = None,
        radius: float = 100,
        limit: int = 10
    ) -> List[SpatialMemory]:
        """Recall memories near a position"""
        memories = []
        chunk_key = self._get_chunk_key(position)
        chunk_radius = int(radius / self._chunk_size) + 1

        for dx in range(-chunk_radius, chunk_radius + 1):
            for dz in range(-chunk_radius, chunk_radius + 1):
                nearby_key = (chunk_key[0] + dx, chunk_key[1] + dz)
                for memory_id in self._spatial_index.get(nearby_key, []):
                    memory = self.spatial_memories.get(memory_id)
                    if not memory:
                        continue

                    if memory_type and memory.memory_type != memory_type:
                        continue

                    distance = position.distance_to(memory.position)
                    if distance <= radius:
                        memories.append((distance, memory))

        # Sort by distance and return
        memories.sort(key=lambda x: x[0])
        result = [m[1] for m in memories[:limit]]

        for memory in result:
            self._access_memory(memory)

        return result

    def recall_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 20
    ) -> List[SpatialMemory]:
        """Recall all memories of a type, sorted by strength"""
        memories = [
            m for m in self.spatial_memories.values()
            if m.memory_type == memory_type
        ]
        memories.sort(key=lambda m: m.strength, reverse=True)
        return memories[:limit]

    # ========================================================================
    # Episodic Memory
    # ========================================================================

    def remember_event(
        self,
        event_type: str,
        context: Dict[str, Any],
        outcome: str,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        lesson: str = "",
        location: Optional[Position] = None
    ) -> EpisodicMemory:
        """Remember an event"""
        memory = EpisodicMemory(
            event_type=event_type,
            context=context,
            outcome=outcome,
            valence=valence,
            lesson=lesson,
            location=location
        )

        self.episodic_memories.append(memory)
        self.total_memories_created += 1

        # Keep episodic memories limited
        if len(self.episodic_memories) > self.max_memories // 2:
            self._consolidate_episodes()

        return memory

    def recall_similar_events(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[EpisodicMemory]:
        """Recall similar past events"""
        matches = []

        for memory in self.episodic_memories:
            if memory.event_type != event_type:
                continue

            score = memory.strength
            if context:
                # Simple context matching
                for key, value in context.items():
                    if key in memory.context and memory.context[key] == value:
                        score += 0.2

            matches.append((score, memory))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def learn_from_outcome(
        self,
        event_type: str,
        action: str,
        success: bool
    ) -> Optional[str]:
        """Learn from an outcome by recalling similar past events"""
        similar = self.recall_similar_events(event_type, {"action": action})

        if not similar:
            return None

        # Calculate success rate from past events
        successes = sum(1 for m in similar if m.outcome == "success")
        total = len(similar)

        if total < 2:
            return None

        success_rate = successes / total
        if success_rate > 0.7:
            return f"Action '{action}' usually works ({success_rate:.0%} success)"
        elif success_rate < 0.3:
            return f"Action '{action}' often fails ({success_rate:.0%} success)"

        return None

    # ========================================================================
    # Social Memory
    # ========================================================================

    def remember_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: Optional[str] = None,
        position: Optional[Position] = None
    ) -> SocialMemory:
        """Remember an entity"""
        if entity_id in self.social_memories:
            memory = self.social_memories[entity_id]
            if position:
                memory.last_seen_position = position
            memory.last_seen_time = time.time()
            return memory

        memory = SocialMemory(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            last_seen_position=position
        )
        self.social_memories[entity_id] = memory
        self.total_memories_created += 1

        return memory

    def record_interaction(
        self,
        entity_id: str,
        interaction_type: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record an interaction with an entity"""
        if entity_id not in self.social_memories:
            return

        memory = self.social_memories[entity_id]
        memory.interactions.append({
            "type": interaction_type,
            "outcome": outcome,
            "details": details or {},
            "time": time.time()
        })

        # Update trust based on interaction
        if outcome == "positive":
            memory.trust_level = min(1.0, memory.trust_level + 0.1)
            if memory.trust_level > 0.7:
                memory.relationship = "friend"
        elif outcome == "negative":
            memory.trust_level = max(0.0, memory.trust_level - 0.15)
            if memory.trust_level < 0.3:
                memory.relationship = "enemy"

    def get_relationship(self, entity_id: str) -> Tuple[str, float]:
        """Get relationship and trust with an entity"""
        if entity_id in self.social_memories:
            memory = self.social_memories[entity_id]
            return memory.relationship, memory.trust_level
        return "stranger", 0.5

    # ========================================================================
    # Procedural Memory
    # ========================================================================

    def learn_skill(
        self,
        skill_name: str,
        action_sequence: List[str],
        preconditions: Optional[Dict[str, Any]] = None
    ) -> ProceduralMemory:
        """Learn a new skill/procedure"""
        if skill_name in self.procedural_memories:
            # Reinforce existing skill
            skill = self.procedural_memories[skill_name]
            skill.execution_count += 1
            return skill

        skill = ProceduralMemory(
            skill_name=skill_name,
            action_sequence=action_sequence,
            preconditions=preconditions or {}
        )
        self.procedural_memories[skill_name] = skill
        return skill

    def execute_skill(self, skill_name: str, success: bool):
        """Record skill execution outcome"""
        if skill_name not in self.procedural_memories:
            return

        skill = self.procedural_memories[skill_name]
        skill.execution_count += 1
        skill.last_executed = time.time()

        # Update success rate (moving average)
        old_rate = skill.success_rate
        skill.success_rate = old_rate * 0.9 + (1.0 if success else 0.0) * 0.1

    def get_skill(self, skill_name: str) -> Optional[ProceduralMemory]:
        """Get a learned skill"""
        return self.procedural_memories.get(skill_name)

    # ========================================================================
    # Memory Management
    # ========================================================================

    def _access_memory(self, memory: SpatialMemory):
        """Update memory on access"""
        memory.access_count += 1
        memory.last_accessed = time.time()
        # Strengthen on recall
        memory.strength = min(1.0, memory.strength + 0.05)

    def decay_memories(self):
        """Apply memory decay"""
        current_time = time.time()

        # Decay spatial memories
        to_forget = []
        for memory_id, memory in self.spatial_memories.items():
            time_since_access = current_time - memory.last_accessed
            decay = self.decay_rate * time_since_access

            # Emotional memories decay slower
            if memory.valence in [EmotionalValence.VERY_POSITIVE, EmotionalValence.VERY_NEGATIVE]:
                decay *= 0.5

            memory.strength = max(0, memory.strength - decay)

            if memory.strength < self.consolidation_threshold:
                to_forget.append(memory_id)

        # Forget weak memories
        for memory_id in to_forget:
            self._forget_spatial(memory_id)

        # Decay episodic memories
        for memory in self.episodic_memories:
            time_since_creation = current_time - memory.created_at
            decay = self.decay_rate * time_since_creation * 0.5
            memory.strength = max(0, memory.strength - decay)

    def _forget_spatial(self, memory_id: str):
        """Forget a spatial memory"""
        if memory_id in self.spatial_memories:
            memory = self.spatial_memories.pop(memory_id)
            chunk_key = self._get_chunk_key(memory.position)
            if memory_id in self._spatial_index[chunk_key]:
                self._spatial_index[chunk_key].remove(memory_id)
            self.total_memories_forgotten += 1

    def _consolidate_episodes(self):
        """Consolidate episodic memories - keep important ones"""
        # Sort by strength and emotional importance
        scored = []
        for memory in self.episodic_memories:
            score = memory.strength
            score += abs(memory.valence.value) * 0.2
            if memory.lesson:
                score += 0.3
            scored.append((score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top half
        keep_count = len(scored) // 2
        self.episodic_memories = [m[1] for m in scored[:keep_count]]
        self.total_memories_forgotten += len(scored) - keep_count

    def _enforce_memory_limit(self):
        """Ensure total memories don't exceed limit"""
        total = len(self.spatial_memories) + len(self.episodic_memories)
        if total > self.max_memories:
            self.decay_memories()

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, filepath: str):
        """Save memories to file"""
        data = {
            "spatial": [
                {
                    "id": m.id,
                    "position": m.position.to_dict(),
                    "type": m.memory_type.value,
                    "label": m.label,
                    "details": m.details,
                    "valence": m.valence.value,
                    "strength": m.strength,
                    "access_count": m.access_count,
                    "created_at": m.created_at
                }
                for m in self.spatial_memories.values()
            ],
            "social": [
                {
                    "entity_id": m.entity_id,
                    "entity_type": m.entity_type,
                    "name": m.name,
                    "relationship": m.relationship,
                    "trust": m.trust_level,
                    "interactions": m.interactions[-10:]  # Keep last 10
                }
                for m in self.social_memories.values()
            ],
            "skills": [
                {
                    "name": s.skill_name,
                    "sequence": s.action_sequence,
                    "success_rate": s.success_rate,
                    "executions": s.execution_count
                }
                for s in self.procedural_memories.values()
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load memories from file"""
        if not Path(filepath).exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load spatial memories
        for m_data in data.get("spatial", []):
            memory = SpatialMemory(
                id=m_data["id"],
                position=Position.from_dict(m_data["position"]),
                memory_type=MemoryType(m_data["type"]),
                label=m_data["label"],
                details=m_data.get("details", {}),
                valence=EmotionalValence(m_data.get("valence", 0)),
                strength=m_data.get("strength", 0.5),
                access_count=m_data.get("access_count", 0),
                created_at=m_data.get("created_at", time.time())
            )
            self.spatial_memories[memory.id] = memory
            self._spatial_index[self._get_chunk_key(memory.position)].append(memory.id)

    def get_summary(self) -> Dict[str, Any]:
        """Get memory system summary"""
        return {
            "spatial_memories": len(self.spatial_memories),
            "episodic_memories": len(self.episodic_memories),
            "social_memories": len(self.social_memories),
            "skills_learned": len(self.procedural_memories),
            "total_created": self.total_memories_created,
            "total_forgotten": self.total_memories_forgotten,
            "resource_memories": len(self.recall_by_type(MemoryType.RESOURCE)),
            "danger_memories": len(self.recall_by_type(MemoryType.DANGER)),
            "shelter_memories": len(self.recall_by_type(MemoryType.SHELTER))
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Memory System Demo")
    print("=" * 50)

    memory = MemorySystem()

    # Remember some locations
    print("\n1. Remembering locations...")
    memory.remember_location(
        Position(100, 64, 200),
        MemoryType.RESOURCE,
        "iron_ore_vein",
        {"block_count": 8},
        EmotionalValence.POSITIVE
    )

    memory.remember_location(
        Position(50, 64, 50),
        MemoryType.DANGER,
        "creeper_spawn",
        {"mob_type": "creeper"},
        EmotionalValence.NEGATIVE
    )

    memory.remember_location(
        Position(0, 64, 0),
        MemoryType.SHELTER,
        "home_base",
        {"has_bed": True, "has_chest": True},
        EmotionalValence.VERY_POSITIVE
    )

    # Recall nearby
    print("\n2. Recalling nearby memories...")
    nearby = memory.recall_nearby(Position(75, 64, 100), radius=150)
    for mem in nearby:
        print(f"   {mem.label} ({mem.memory_type.value}) at {mem.position.x}, {mem.position.z}")

    # Remember an event
    print("\n3. Recording events...")
    memory.remember_event(
        "mining",
        {"target": "iron_ore", "tool": "stone_pickaxe"},
        "success",
        EmotionalValence.POSITIVE,
        "Stone pickaxe works for iron"
    )

    # Learn a skill
    print("\n4. Learning skills...")
    memory.learn_skill(
        "mine_iron",
        ["find_iron", "move_to_iron", "equip_pickaxe", "mine_block", "collect_drop"],
        {"requires_tool": "pickaxe"}
    )

    # Summary
    print("\n5. Memory summary:")
    summary = memory.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
