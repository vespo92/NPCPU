"""
Collective Behavior System for NPCPU

Implements shared consciousness and collective intelligence for organism groups:
- Knowledge sharing between group members
- Collective memory pool
- Group decision voting with weighted opinions
- Hive mind emergence from connected organisms

Example:
    from swarm.collective_behavior import CollectiveMind, KnowledgeType
    from core.simple_organism import SimpleOrganism

    # Create collective mind
    mind = CollectiveMind("hive")

    # Add organisms
    for i in range(5):
        org = SimpleOrganism(f"Drone_{i}")
        mind.add_member(org)

    # Share knowledge
    mind.share_knowledge(
        source_id=org1.id,
        knowledge_type=KnowledgeType.LOCATION,
        data={"food_source": (100, 200)},
        importance=0.8
    )

    # Group decision
    decision = mind.make_decision(
        question="Should we migrate?",
        options=["stay", "migrate_north", "migrate_south"]
    )
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
import uuid
import random
import math

from core.abstractions import BaseOrganism, BaseSubsystem
from core.events import get_event_bus, Event


# =============================================================================
# Enums and Types
# =============================================================================

class KnowledgeType(Enum):
    """Types of knowledge that can be shared"""
    LOCATION = "location"          # Places: food, water, danger
    THREAT = "threat"              # Predators, hazards
    RESOURCE = "resource"          # Resource availability
    SOCIAL = "social"              # Relationship info
    SKILL = "skill"                # Learned abilities
    EXPERIENCE = "experience"      # Past events


class DecisionMethod(Enum):
    """Methods for collective decision making"""
    MAJORITY = auto()      # Simple majority vote
    WEIGHTED = auto()      # Weighted by member influence
    CONSENSUS = auto()     # Require near-unanimous agreement
    LEADER = auto()        # Leader decides
    RANDOM = auto()        # Random selection


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Knowledge:
    """A piece of shared knowledge"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: KnowledgeType = KnowledgeType.EXPERIENCE
    data: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""                           # Originating organism
    importance: float = 0.5                       # 0.0 to 1.0
    confidence: float = 1.0                       # How certain the info is
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0                         # Times accessed
    decay_rate: float = 0.01                      # How fast it fades

    def decay(self) -> None:
        """Apply time-based decay to knowledge"""
        self.importance = max(0.0, self.importance - self.decay_rate)

    def access(self) -> Dict[str, Any]:
        """Access this knowledge (increases retention)"""
        self.access_count += 1
        # Accessing reinforces importance slightly
        self.importance = min(1.0, self.importance + 0.01)
        return self.data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "source_id": self.source_id,
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "decay_rate": self.decay_rate
        }


@dataclass
class Memory:
    """A collective memory entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    witnesses: Set[str] = field(default_factory=set)  # Organisms who experienced it
    significance: float = 0.5                          # How important
    emotional_valence: float = 0.0                     # -1.0 (negative) to 1.0 (positive)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "witnesses": list(self.witnesses),
            "significance": self.significance,
            "emotional_valence": self.emotional_valence
        }


@dataclass
class CollectiveMember:
    """Member of a collective mind"""
    organism: BaseOrganism
    influence: float = 1.0           # Weight in decisions
    knowledge_contributed: int = 0    # Knowledge items shared
    trust_level: float = 1.0          # How much others trust this member
    connected: bool = True            # Actively connected to hive

    @property
    def id(self) -> str:
        return self.organism.id


# =============================================================================
# Collective Mind
# =============================================================================

class CollectiveMind:
    """
    Shared consciousness for organism groups.

    Features:
    - Knowledge sharing between members
    - Collective memory pool
    - Group decision voting
    - Hive mind emergence

    Example:
        mind = CollectiveMind("pack")

        # Add members
        for org in organisms:
            mind.add_member(org)

        # Share knowledge
        mind.share_knowledge(
            source_id=scout.id,
            knowledge_type=KnowledgeType.LOCATION,
            data={"prey_location": (500, 300)},
            importance=0.9
        )

        # Query collective knowledge
        threats = mind.query_knowledge(KnowledgeType.THREAT)

        # Make group decision
        decision, confidence = mind.make_decision(
            question="attack_or_wait",
            options=["attack_now", "wait", "retreat"],
            method=DecisionMethod.WEIGHTED
        )
    """

    def __init__(
        self,
        name: str = "collective",
        knowledge_capacity: int = 1000,
        memory_capacity: int = 500,
        knowledge_spread_rate: float = 0.8,
        consensus_threshold: float = 0.7
    ):
        self._id = str(uuid.uuid4())
        self._name = name

        # Configuration
        self.knowledge_capacity = knowledge_capacity
        self.memory_capacity = memory_capacity
        self.knowledge_spread_rate = knowledge_spread_rate
        self.consensus_threshold = consensus_threshold

        # Members
        self._members: Dict[str, CollectiveMember] = {}
        self._leader_id: Optional[str] = None

        # Knowledge pool
        self._knowledge: Dict[str, Knowledge] = {}
        self._knowledge_by_type: Dict[KnowledgeType, Set[str]] = defaultdict(set)

        # Collective memory
        self._memories: List[Memory] = []

        # Event bus
        self._bus = get_event_bus()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return len(self._members)

    # -------------------------------------------------------------------------
    # Member Management
    # -------------------------------------------------------------------------

    def add_member(
        self,
        organism: BaseOrganism,
        influence: float = 1.0,
        trust_level: float = 1.0
    ) -> bool:
        """Add an organism to the collective"""
        if organism.id in self._members:
            return False

        member = CollectiveMember(
            organism=organism,
            influence=influence,
            trust_level=trust_level
        )
        self._members[organism.id] = member

        self._bus.emit("collective.member_joined", {
            "collective_id": self._id,
            "collective_name": self._name,
            "organism_id": organism.id,
            "organism_name": organism.name
        })

        return True

    def remove_member(self, organism_id: str) -> bool:
        """Remove an organism from the collective"""
        if organism_id not in self._members:
            return False

        member = self._members.pop(organism_id)

        # Update leader if removed
        if self._leader_id == organism_id:
            self._elect_new_leader()

        self._bus.emit("collective.member_left", {
            "collective_id": self._id,
            "organism_id": organism_id
        })

        return True

    def get_member(self, organism_id: str) -> Optional[CollectiveMember]:
        """Get a member by organism ID"""
        return self._members.get(organism_id)

    def get_members(self) -> List[BaseOrganism]:
        """Get all member organisms"""
        return [m.organism for m in self._members.values()]

    def set_leader(self, organism_id: str) -> bool:
        """Designate a leader for the collective"""
        if organism_id not in self._members:
            return False
        self._leader_id = organism_id

        self._bus.emit("collective.leader_changed", {
            "collective_id": self._id,
            "leader_id": organism_id
        })
        return True

    def get_leader(self) -> Optional[BaseOrganism]:
        """Get the leader organism"""
        if self._leader_id and self._leader_id in self._members:
            return self._members[self._leader_id].organism
        return None

    def _elect_new_leader(self) -> None:
        """Elect a new leader based on influence"""
        if not self._members:
            self._leader_id = None
            return

        # Elect member with highest influence
        best = max(self._members.values(), key=lambda m: m.influence)
        self._leader_id = best.id

    def set_member_influence(self, organism_id: str, influence: float) -> bool:
        """Update a member's influence level"""
        if organism_id in self._members:
            self._members[organism_id].influence = max(0.0, min(2.0, influence))
            return True
        return False

    def connect_member(self, organism_id: str) -> bool:
        """Connect a member to the hive mind"""
        if organism_id in self._members:
            self._members[organism_id].connected = True
            return True
        return False

    def disconnect_member(self, organism_id: str) -> bool:
        """Disconnect a member (they remain but can't participate)"""
        if organism_id in self._members:
            self._members[organism_id].connected = False
            return True
        return False

    # -------------------------------------------------------------------------
    # Knowledge Sharing
    # -------------------------------------------------------------------------

    def share_knowledge(
        self,
        source_id: str,
        knowledge_type: KnowledgeType,
        data: Dict[str, Any],
        importance: float = 0.5,
        confidence: float = 1.0,
        targets: Optional[List[str]] = None
    ) -> str:
        """
        Share knowledge from one member to the collective.

        Args:
            source_id: Organism sharing the knowledge
            knowledge_type: Category of knowledge
            data: The actual knowledge data
            importance: How important (0.0-1.0)
            confidence: How certain (0.0-1.0)
            targets: Specific members to share with (None = all)

        Returns:
            Knowledge ID

        Example:
            kid = mind.share_knowledge(
                source_id=scout.id,
                knowledge_type=KnowledgeType.THREAT,
                data={"predator_type": "wolf", "location": (100, 200)},
                importance=0.9
            )
        """
        # Verify source is a member
        if source_id not in self._members:
            return ""

        # Create knowledge entry
        knowledge = Knowledge(
            type=knowledge_type,
            data=data,
            source_id=source_id,
            importance=importance,
            confidence=confidence
        )

        # Apply spread rate (knowledge degrades as it spreads)
        knowledge.confidence *= self.knowledge_spread_rate

        # Add to pool
        self._knowledge[knowledge.id] = knowledge
        self._knowledge_by_type[knowledge_type].add(knowledge.id)

        # Update contributor stats
        self._members[source_id].knowledge_contributed += 1

        # Enforce capacity
        self._prune_knowledge()

        self._bus.emit("collective.knowledge_shared", {
            "collective_id": self._id,
            "knowledge_id": knowledge.id,
            "knowledge_type": knowledge_type.value,
            "source_id": source_id,
            "importance": importance
        })

        return knowledge.id

    def query_knowledge(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        min_importance: float = 0.0,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Knowledge]:
        """
        Query the collective knowledge pool.

        Args:
            knowledge_type: Filter by type (None = all)
            min_importance: Minimum importance threshold
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of matching Knowledge items

        Example:
            threats = mind.query_knowledge(
                knowledge_type=KnowledgeType.THREAT,
                min_importance=0.5
            )
        """
        if knowledge_type:
            knowledge_ids = self._knowledge_by_type.get(knowledge_type, set())
            candidates = [self._knowledge[kid] for kid in knowledge_ids
                         if kid in self._knowledge]
        else:
            candidates = list(self._knowledge.values())

        # Filter
        results = [
            k for k in candidates
            if k.importance >= min_importance and k.confidence >= min_confidence
        ]

        # Sort by importance
        results.sort(key=lambda k: -k.importance)

        # Access each result (reinforces memory)
        for k in results[:limit]:
            k.access()

        return results[:limit]

    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """Get specific knowledge by ID"""
        k = self._knowledge.get(knowledge_id)
        if k:
            k.access()
        return k

    def forget_knowledge(self, knowledge_id: str) -> bool:
        """Remove knowledge from the pool"""
        if knowledge_id in self._knowledge:
            k = self._knowledge.pop(knowledge_id)
            self._knowledge_by_type[k.type].discard(knowledge_id)
            return True
        return False

    def _prune_knowledge(self) -> None:
        """Remove least important knowledge if over capacity"""
        while len(self._knowledge) > self.knowledge_capacity:
            # Find lowest importance
            least = min(self._knowledge.values(), key=lambda k: k.importance)
            self.forget_knowledge(least.id)

    # -------------------------------------------------------------------------
    # Collective Memory
    # -------------------------------------------------------------------------

    def record_memory(
        self,
        event_type: str,
        data: Dict[str, Any],
        witnesses: Optional[Set[str]] = None,
        significance: float = 0.5,
        emotional_valence: float = 0.0
    ) -> str:
        """
        Record a shared memory for the collective.

        Args:
            event_type: Type of event being remembered
            data: Memory data
            witnesses: Set of organism IDs who witnessed
            significance: How significant (0.0-1.0)
            emotional_valence: -1.0 (negative) to 1.0 (positive)

        Returns:
            Memory ID

        Example:
            mid = mind.record_memory(
                event_type="attack_survived",
                data={"predator": "bear", "casualties": 2},
                witnesses={org1.id, org2.id},
                significance=0.9,
                emotional_valence=-0.7
            )
        """
        memory = Memory(
            event_type=event_type,
            data=data,
            witnesses=witnesses or set(),
            significance=significance,
            emotional_valence=emotional_valence
        )

        self._memories.append(memory)

        # Enforce capacity
        while len(self._memories) > self.memory_capacity:
            # Remove oldest, least significant
            self._memories.sort(key=lambda m: (m.significance, m.timestamp))
            self._memories.pop(0)

        self._bus.emit("collective.memory_recorded", {
            "collective_id": self._id,
            "memory_id": memory.id,
            "event_type": event_type,
            "significance": significance
        })

        return memory.id

    def recall_memories(
        self,
        event_type: Optional[str] = None,
        min_significance: float = 0.0,
        limit: int = 10,
        recent_first: bool = True
    ) -> List[Memory]:
        """
        Recall memories from collective pool.

        Args:
            event_type: Filter by event type
            min_significance: Minimum significance threshold
            limit: Maximum results
            recent_first: Sort by recency (True) or significance (False)

        Returns:
            List of Memory objects
        """
        candidates = self._memories

        if event_type:
            candidates = [m for m in candidates if m.event_type == event_type]

        candidates = [m for m in candidates if m.significance >= min_significance]

        if recent_first:
            candidates.sort(key=lambda m: m.timestamp, reverse=True)
        else:
            candidates.sort(key=lambda m: -m.significance)

        return candidates[:limit]

    def get_emotional_history(
        self,
        window: int = 10
    ) -> Tuple[float, List[float]]:
        """
        Get recent emotional trajectory of collective.

        Returns:
            Tuple of (average_valence, [recent_valences])
        """
        recent = self._memories[-window:] if self._memories else []
        if not recent:
            return (0.0, [])

        valences = [m.emotional_valence for m in recent]
        avg = sum(valences) / len(valences)

        return (avg, valences)

    # -------------------------------------------------------------------------
    # Group Decision Making
    # -------------------------------------------------------------------------

    def make_decision(
        self,
        question: str,
        options: List[str],
        method: DecisionMethod = DecisionMethod.WEIGHTED,
        voter_opinions: Optional[Dict[str, str]] = None
    ) -> Tuple[str, float]:
        """
        Make a collective decision.

        Args:
            question: The decision question
            options: Available choices
            method: Decision method to use
            voter_opinions: Pre-defined votes {organism_id: option}

        Returns:
            Tuple of (winning_option, confidence_score)

        Example:
            decision, confidence = mind.make_decision(
                question="What should we do?",
                options=["attack", "defend", "flee"],
                method=DecisionMethod.WEIGHTED
            )
        """
        if not options or not self._members:
            return ("", 0.0)

        connected = [m for m in self._members.values() if m.connected]
        if not connected:
            return ("", 0.0)

        # Collect votes
        votes: Dict[str, float] = {opt: 0.0 for opt in options}
        total_weight = 0.0

        if method == DecisionMethod.LEADER:
            # Leader decides
            leader = self.get_leader()
            if leader:
                if voter_opinions and leader.id in voter_opinions:
                    choice = voter_opinions[leader.id]
                else:
                    choice = random.choice(options)
                return (choice, 1.0)
            else:
                method = DecisionMethod.WEIGHTED

        if method == DecisionMethod.RANDOM:
            return (random.choice(options), 1.0 / len(options))

        # Voting methods
        for member in connected:
            # Determine vote
            if voter_opinions and member.id in voter_opinions:
                vote = voter_opinions[member.id]
            else:
                # Default: random vote weighted by organism traits
                # Could be extended to consider organism personality
                vote = random.choice(options)

            # Calculate weight
            if method == DecisionMethod.MAJORITY:
                weight = 1.0
            elif method == DecisionMethod.WEIGHTED:
                weight = member.influence * member.trust_level
            elif method == DecisionMethod.CONSENSUS:
                weight = 1.0
            else:
                weight = 1.0

            if vote in votes:
                votes[vote] += weight
                total_weight += weight

        # Determine winner
        if total_weight == 0:
            return (random.choice(options), 0.0)

        winner = max(votes, key=votes.get)
        winner_score = votes[winner] / total_weight

        # Check consensus requirement
        if method == DecisionMethod.CONSENSUS:
            if winner_score < self.consensus_threshold:
                # No consensus reached
                self._bus.emit("collective.decision_failed", {
                    "collective_id": self._id,
                    "question": question,
                    "reason": "no_consensus",
                    "best_score": winner_score
                })
                return ("", winner_score)

        self._bus.emit("collective.decision_made", {
            "collective_id": self._id,
            "question": question,
            "decision": winner,
            "confidence": winner_score,
            "method": method.name,
            "votes": votes
        })

        return (winner, winner_score)

    def vote(
        self,
        organism_id: str,
        question_id: str,
        choice: str
    ) -> bool:
        """
        Register a member's vote (for async voting).

        Returns True if vote was accepted.
        """
        if organism_id not in self._members:
            return False

        member = self._members[organism_id]
        if not member.connected:
            return False

        # Store vote (could be extended with question tracking)
        self._bus.emit("collective.vote_cast", {
            "collective_id": self._id,
            "organism_id": organism_id,
            "question_id": question_id,
            "choice": choice
        })

        return True

    # -------------------------------------------------------------------------
    # Hive Mind Emergence
    # -------------------------------------------------------------------------

    def calculate_cohesion(self) -> float:
        """
        Calculate how unified the collective mind is.

        Returns:
            Cohesion score from 0.0 (fragmented) to 1.0 (unified)
        """
        if not self._members:
            return 0.0

        connected = [m for m in self._members.values() if m.connected]
        if not connected:
            return 0.0

        # Factors:
        # 1. Proportion connected
        connection_ratio = len(connected) / len(self._members)

        # 2. Average trust level
        avg_trust = sum(m.trust_level for m in connected) / len(connected)

        # 3. Knowledge sharing activity
        total_contributions = sum(m.knowledge_contributed for m in connected)
        knowledge_factor = min(1.0, total_contributions / (len(connected) * 10))

        # 4. Shared memories
        if self._memories:
            # Average number of witnesses per memory
            avg_witnesses = sum(len(m.witnesses) for m in self._memories) / len(self._memories)
            memory_factor = min(1.0, avg_witnesses / len(connected))
        else:
            memory_factor = 0.0

        # Weighted combination
        cohesion = (
            connection_ratio * 0.3 +
            avg_trust * 0.3 +
            knowledge_factor * 0.2 +
            memory_factor * 0.2
        )

        return cohesion

    def get_emergence_level(self) -> float:
        """
        Calculate level of emergent hive mind behavior.

        Returns:
            Emergence level from 0.0 to 1.0

        Higher levels indicate:
        - Strong cohesion
        - Active knowledge sharing
        - Shared emotional experiences
        - Effective collective decisions
        """
        cohesion = self.calculate_cohesion()

        # Add bonuses for group size (larger groups can be more emergent)
        size_bonus = min(0.2, len(self._members) / 50)

        # Add bonus for emotional alignment
        avg_emotion, _ = self.get_emotional_history()
        # Strong shared emotions (positive or negative) increase emergence
        emotion_bonus = abs(avg_emotion) * 0.1

        emergence = min(1.0, cohesion + size_bonus + emotion_bonus)

        return emergence

    def broadcast_thought(
        self,
        source_id: str,
        thought: Dict[str, Any],
        strength: float = 1.0
    ) -> int:
        """
        Broadcast a thought to all connected members.

        Returns number of members who received it.
        """
        if source_id not in self._members:
            return 0

        source = self._members[source_id]
        if not source.connected:
            return 0

        received = 0
        for member in self._members.values():
            if member.id == source_id:
                continue
            if not member.connected:
                continue

            # Strength degraded by distance in trust network
            effective_strength = strength * source.trust_level * member.trust_level

            if random.random() < effective_strength:
                received += 1

                self._bus.emit("collective.thought_received", {
                    "collective_id": self._id,
                    "source_id": source_id,
                    "receiver_id": member.id,
                    "thought": thought,
                    "strength": effective_strength
                })

        return received

    # -------------------------------------------------------------------------
    # Tick and State
    # -------------------------------------------------------------------------

    def tick(self) -> None:
        """Process one time step for the collective"""
        # Decay knowledge
        for knowledge in list(self._knowledge.values()):
            knowledge.decay()
            if knowledge.importance <= 0:
                self.forget_knowledge(knowledge.id)

        # Update trust based on contribution
        for member in self._members.values():
            if member.knowledge_contributed > 0:
                # Slight trust increase for contributors
                member.trust_level = min(2.0, member.trust_level + 0.001)

        self._bus.emit("collective.tick", {
            "collective_id": self._id,
            "member_count": len(self._members),
            "knowledge_count": len(self._knowledge),
            "memory_count": len(self._memories),
            "cohesion": self.calculate_cohesion()
        })

    def get_state(self) -> Dict[str, Any]:
        """Get collective state for serialization"""
        return {
            "id": self._id,
            "name": self._name,
            "leader_id": self._leader_id,
            "members": {
                mid: {
                    "influence": m.influence,
                    "trust_level": m.trust_level,
                    "knowledge_contributed": m.knowledge_contributed,
                    "connected": m.connected
                }
                for mid, m in self._members.items()
            },
            "knowledge": [k.to_dict() for k in self._knowledge.values()],
            "memories": [m.to_dict() for m in self._memories],
            "settings": {
                "knowledge_capacity": self.knowledge_capacity,
                "memory_capacity": self.memory_capacity,
                "knowledge_spread_rate": self.knowledge_spread_rate,
                "consensus_threshold": self.consensus_threshold
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get collective statistics"""
        connected = sum(1 for m in self._members.values() if m.connected)

        return {
            "total_members": len(self._members),
            "connected_members": connected,
            "knowledge_count": len(self._knowledge),
            "memory_count": len(self._memories),
            "cohesion": self.calculate_cohesion(),
            "emergence_level": self.get_emergence_level(),
            "has_leader": self._leader_id is not None
        }


# =============================================================================
# Collective Mind Subsystem (for individual organisms)
# =============================================================================

class CollectiveMindSubsystem(BaseSubsystem):
    """
    Subsystem for organisms participating in a collective mind.

    Provides interface between organism and CollectiveMind.

    Example:
        org = SimpleOrganism("Alpha")
        collective_sub = CollectiveMindSubsystem(collective_mind)
        org.add_subsystem(collective_sub)

        # Share knowledge
        collective_sub.share(
            KnowledgeType.THREAT,
            {"predator": "hawk", "direction": "north"}
        )

        # Query collective knowledge
        threats = collective_sub.query(KnowledgeType.THREAT)
    """

    def __init__(self, collective: CollectiveMind):
        super().__init__("collective_mind")
        self._collective = collective
        self._joined = False
        self._received_thoughts: List[Dict[str, Any]] = []

    def tick(self) -> None:
        """Update collective participation"""
        if not self.enabled or not self._owner:
            return

        # Sync state
        self._state["joined"] = self._joined
        self._state["collective_id"] = self._collective.id if self._joined else None

    def join(self, influence: float = 1.0) -> bool:
        """Join the collective mind"""
        if not self._owner:
            return False

        if self._collective.add_member(self._owner, influence=influence):
            self._joined = True
            return True
        return False

    def leave(self) -> bool:
        """Leave the collective mind"""
        if not self._owner:
            return False

        if self._collective.remove_member(self._owner.id):
            self._joined = False
            return True
        return False

    def share(
        self,
        knowledge_type: KnowledgeType,
        data: Dict[str, Any],
        importance: float = 0.5
    ) -> str:
        """Share knowledge with the collective"""
        if not self._owner or not self._joined:
            return ""

        return self._collective.share_knowledge(
            source_id=self._owner.id,
            knowledge_type=knowledge_type,
            data=data,
            importance=importance
        )

    def query(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        min_importance: float = 0.0
    ) -> List[Knowledge]:
        """Query collective knowledge"""
        if not self._joined:
            return []

        return self._collective.query_knowledge(
            knowledge_type=knowledge_type,
            min_importance=min_importance
        )

    def broadcast(self, thought: Dict[str, Any], strength: float = 1.0) -> int:
        """Broadcast a thought to the collective"""
        if not self._owner or not self._joined:
            return 0

        return self._collective.broadcast_thought(
            source_id=self._owner.id,
            thought=thought,
            strength=strength
        )

    def get_cohesion(self) -> float:
        """Get collective cohesion level"""
        return self._collective.calculate_cohesion()

    def get_state(self) -> Dict[str, Any]:
        return {
            "joined": self._joined,
            "collective_id": self._collective.id if self._joined else None,
            **self._state
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._joined = state.get("joined", False)
        self._state = {k: v for k, v in state.items()
                      if k not in ("joined", "collective_id")}
