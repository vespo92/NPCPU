"""
Social Relationships Module for NPCPU

Implements graph-based social networks for organism relationships.
Provides relationship management, trust/reputation scoring,
group formation, and influence propagation.

Example:
    from social.relationships import SocialNetwork, RelationshipType
    from core.events import get_event_bus

    # Create a social network
    network = SocialNetwork()

    # Form relationships
    network.form_relationship("org_1", "org_2", RelationshipType.ALLY, strength=0.7)
    network.form_relationship("org_1", "org_3", RelationshipType.RIVAL, strength=0.5)

    # Check relationships
    rel = network.get_relationship("org_1", "org_2")
    print(f"Relationship: {rel.type.value}, Trust: {rel.trust}")

    # Calculate influence
    influence = network.get_influence("org_1")
    print(f"Influence score: {influence}")

    # Form groups
    group_id = network.form_group("hunting_party", ["org_1", "org_2", "org_3"])
"""

from typing import Dict, Any, List, Optional, Set, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid
import math

from core.events import get_event_bus, Event


class RelationshipType(Enum):
    """Types of relationships between organisms"""
    NEUTRAL = "neutral"
    ALLY = "ally"
    RIVAL = "rival"
    MATE = "mate"
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    MENTOR = "mentor"
    STUDENT = "student"
    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass
class Relationship:
    """
    Represents a directed relationship between two organisms.

    Attributes:
        source_id: ID of the organism that holds this view
        target_id: ID of the related organism
        type: Type of relationship
        strength: How strong the relationship is (0.0 to 1.0)
        trust: Trust level (-1.0 to 1.0, negative = distrust)
        familiarity: How well the source knows the target (0.0 to 1.0)
        interactions: Number of interactions
        last_interaction: Timestamp of last interaction
        metadata: Additional relationship data
    """
    source_id: str
    target_id: str
    type: RelationshipType = RelationshipType.NEUTRAL
    strength: float = 0.5
    trust: float = 0.0
    familiarity: float = 0.0
    interactions: int = 0
    last_interaction: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.trust = max(-1.0, min(1.0, self.trust))
        self.familiarity = max(0.0, min(1.0, self.familiarity))

    def record_interaction(self, positive: bool = True, weight: float = 0.1):
        """Record an interaction and update relationship metrics"""
        self.interactions += 1
        self.last_interaction = datetime.now()

        # Update familiarity (always increases with interaction)
        self.familiarity = min(1.0, self.familiarity + 0.05)

        # Update trust based on interaction quality
        trust_change = weight if positive else -weight
        self.trust = max(-1.0, min(1.0, self.trust + trust_change))

        # Positive interactions strengthen, negative weaken
        if positive:
            self.strength = min(1.0, self.strength + weight * 0.5)
        else:
            self.strength = max(0.0, self.strength - weight * 0.5)

    def decay(self, rate: float = 0.01):
        """Apply time-based decay to relationship strength"""
        self.strength = max(0.0, self.strength - rate)
        # Trust decays toward neutral
        if self.trust > 0:
            self.trust = max(0.0, self.trust - rate * 0.5)
        elif self.trust < 0:
            self.trust = min(0.0, self.trust + rate * 0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize relationship to dictionary"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "strength": self.strength,
            "trust": self.trust,
            "familiarity": self.familiarity,
            "interactions": self.interactions,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Deserialize relationship from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=RelationshipType(data.get("type", "neutral")),
            strength=data.get("strength", 0.5),
            trust=data.get("trust", 0.0),
            familiarity=data.get("familiarity", 0.0),
            interactions=data.get("interactions", 0),
            last_interaction=datetime.fromisoformat(data["last_interaction"]) if data.get("last_interaction") else None,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )


@dataclass
class Group:
    """
    Represents a social group/tribe of organisms.

    Attributes:
        name: Group name
        members: Set of organism IDs in the group
        leader_id: ID of the group leader (optional)
        formation_rules: Rules for joining/leaving
        metadata: Additional group data
    """
    name: str
    members: Set[str] = field(default_factory=set)
    leader_id: Optional[str] = None
    cohesion: float = 0.5  # How tightly bound the group is
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def size(self) -> int:
        return len(self.members)

    def add_member(self, organism_id: str) -> bool:
        """Add a member to the group"""
        if organism_id not in self.members:
            self.members.add(organism_id)
            return True
        return False

    def remove_member(self, organism_id: str) -> bool:
        """Remove a member from the group"""
        if organism_id in self.members:
            self.members.discard(organism_id)
            # If leader leaves, clear leadership
            if self.leader_id == organism_id:
                self.leader_id = None
            return True
        return False

    def set_leader(self, organism_id: str) -> bool:
        """Set the group leader"""
        if organism_id in self.members:
            self.leader_id = organism_id
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize group to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "members": list(self.members),
            "leader_id": self.leader_id,
            "cohesion": self.cohesion,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Group':
        """Deserialize group from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            members=set(data.get("members", [])),
            leader_id=data.get("leader_id"),
            cohesion=data.get("cohesion", 0.5),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )


class SocialNetwork:
    """
    Graph-based social network for organism relationships.

    Manages relationships between organisms, tracks trust/reputation,
    handles group formation, and calculates social influence.

    Example:
        network = SocialNetwork()

        # Form relationships
        network.form_relationship("alice", "bob", RelationshipType.ALLY)
        network.form_relationship("bob", "charlie", RelationshipType.MENTOR)

        # Query the network
        allies = network.get_relationships_by_type("alice", RelationshipType.ALLY)
        influence = network.get_influence("bob")

        # Form a group
        group_id = network.form_group("hunters", ["alice", "bob", "charlie"])
        network.set_group_leader(group_id, "alice")
    """

    def __init__(self, emit_events: bool = True):
        """
        Initialize the social network.

        Args:
            emit_events: Whether to emit events on relationship changes
        """
        # Adjacency list: source_id -> {target_id -> Relationship}
        self._relationships: Dict[str, Dict[str, Relationship]] = {}
        # Reverse index for faster lookups
        self._reverse_relationships: Dict[str, Set[str]] = {}
        # Groups
        self._groups: Dict[str, Group] = {}
        # Organism -> Groups mapping
        self._organism_groups: Dict[str, Set[str]] = {}
        # Reputation scores (aggregated from relationships)
        self._reputation: Dict[str, float] = {}
        # Event emission flag
        self._emit_events = emit_events

    # =========================================================================
    # Relationship Management
    # =========================================================================

    def form_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType = RelationshipType.NEUTRAL,
        strength: float = 0.5,
        trust: float = 0.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Relationship:
        """
        Create or update a relationship between two organisms.

        Args:
            source_id: ID of the source organism
            target_id: ID of the target organism
            rel_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            trust: Initial trust level (-1.0 to 1.0)
            bidirectional: If True, create relationship in both directions
            metadata: Additional relationship data

        Returns:
            The created/updated Relationship

        Example:
            rel = network.form_relationship("org_1", "org_2", RelationshipType.ALLY, strength=0.8)
        """
        # Ensure source has a relationship dict
        if source_id not in self._relationships:
            self._relationships[source_id] = {}

        # Create or update relationship
        existing = self._relationships[source_id].get(target_id)
        if existing:
            existing.type = rel_type
            existing.strength = strength
            existing.trust = trust
            if metadata:
                existing.metadata.update(metadata)
            relationship = existing
        else:
            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                strength=strength,
                trust=trust,
                metadata=metadata or {}
            )
            self._relationships[source_id][target_id] = relationship

        # Update reverse index
        if target_id not in self._reverse_relationships:
            self._reverse_relationships[target_id] = set()
        self._reverse_relationships[target_id].add(source_id)

        # Emit event
        if self._emit_events:
            bus = get_event_bus()
            bus.emit("relationship.formed", {
                "source_id": source_id,
                "target_id": target_id,
                "type": rel_type.value,
                "strength": strength,
                "trust": trust,
                "bidirectional": bidirectional
            })

        # Handle bidirectional
        if bidirectional:
            reciprocal_type = self._get_reciprocal_type(rel_type)
            self.form_relationship(
                target_id, source_id, reciprocal_type,
                strength, trust, bidirectional=False, metadata=metadata
            )

        return relationship

    def _get_reciprocal_type(self, rel_type: RelationshipType) -> RelationshipType:
        """Get the reciprocal relationship type"""
        reciprocals = {
            RelationshipType.PARENT: RelationshipType.CHILD,
            RelationshipType.CHILD: RelationshipType.PARENT,
            RelationshipType.MENTOR: RelationshipType.STUDENT,
            RelationshipType.STUDENT: RelationshipType.MENTOR,
            RelationshipType.LEADER: RelationshipType.FOLLOWER,
            RelationshipType.FOLLOWER: RelationshipType.LEADER,
        }
        return reciprocals.get(rel_type, rel_type)

    def get_relationship(self, source_id: str, target_id: str) -> Optional[Relationship]:
        """
        Get the relationship from source to target.

        Returns None if no relationship exists.
        """
        return self._relationships.get(source_id, {}).get(target_id)

    def has_relationship(self, source_id: str, target_id: str) -> bool:
        """Check if a relationship exists from source to target"""
        return target_id in self._relationships.get(source_id, {})

    def remove_relationship(self, source_id: str, target_id: str) -> Optional[Relationship]:
        """
        Remove a relationship between two organisms.

        Returns the removed relationship or None if it didn't exist.
        """
        if source_id in self._relationships:
            relationship = self._relationships[source_id].pop(target_id, None)
            if relationship:
                # Update reverse index
                if target_id in self._reverse_relationships:
                    self._reverse_relationships[target_id].discard(source_id)

                # Emit event
                if self._emit_events:
                    bus = get_event_bus()
                    bus.emit("relationship.removed", {
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": relationship.type.value
                    })

                return relationship
        return None

    def get_all_relationships(self, organism_id: str) -> List[Relationship]:
        """Get all relationships where organism is the source"""
        return list(self._relationships.get(organism_id, {}).values())

    def get_incoming_relationships(self, organism_id: str) -> List[Relationship]:
        """Get all relationships where organism is the target"""
        relationships = []
        for source_id in self._reverse_relationships.get(organism_id, set()):
            rel = self.get_relationship(source_id, organism_id)
            if rel:
                relationships.append(rel)
        return relationships

    def get_relationships_by_type(
        self,
        organism_id: str,
        rel_type: RelationshipType,
        include_incoming: bool = False
    ) -> List[Relationship]:
        """Get all relationships of a specific type for an organism"""
        relationships = [
            rel for rel in self.get_all_relationships(organism_id)
            if rel.type == rel_type
        ]

        if include_incoming:
            relationships.extend([
                rel for rel in self.get_incoming_relationships(organism_id)
                if rel.type == rel_type
            ])

        return relationships

    def get_connected_organisms(self, organism_id: str) -> Set[str]:
        """Get all organisms connected to this one (any direction)"""
        connected = set(self._relationships.get(organism_id, {}).keys())
        connected.update(self._reverse_relationships.get(organism_id, set()))
        return connected

    def record_interaction(
        self,
        source_id: str,
        target_id: str,
        positive: bool = True,
        weight: float = 0.1
    ) -> Optional[Relationship]:
        """
        Record an interaction between two organisms.

        Creates a neutral relationship if none exists.
        Updates trust and familiarity based on interaction quality.
        """
        relationship = self.get_relationship(source_id, target_id)

        if not relationship:
            relationship = self.form_relationship(source_id, target_id)

        relationship.record_interaction(positive, weight)

        # Emit event
        if self._emit_events:
            bus = get_event_bus()
            bus.emit("interaction.recorded", {
                "source_id": source_id,
                "target_id": target_id,
                "positive": positive,
                "trust": relationship.trust,
                "familiarity": relationship.familiarity
            })

        return relationship

    def decay_relationships(self, rate: float = 0.01):
        """Apply decay to all relationships"""
        for source_relationships in self._relationships.values():
            for relationship in source_relationships.values():
                relationship.decay(rate)

    # =========================================================================
    # Influence & Reputation
    # =========================================================================

    def get_influence(self, organism_id: str) -> float:
        """
        Calculate the social influence of an organism.

        Influence is based on:
        - Number of relationships
        - Strength of relationships
        - Trust from others
        - Leadership positions
        - Group memberships

        Returns a score from 0.0 to 1.0 (normalized).
        """
        influence = 0.0

        # Incoming relationships (how others view this organism)
        incoming = self.get_incoming_relationships(organism_id)
        for rel in incoming:
            # Positive trust adds to influence
            trust_factor = max(0, rel.trust)
            influence += rel.strength * trust_factor * 0.3

        # Outgoing relationships (connections)
        outgoing = self.get_all_relationships(organism_id)
        influence += len(outgoing) * 0.1

        # Leadership bonus
        leader_groups = [
            g for g in self._groups.values()
            if g.leader_id == organism_id
        ]
        for group in leader_groups:
            influence += group.size * 0.2

        # Group membership bonus
        member_groups = self._organism_groups.get(organism_id, set())
        influence += len(member_groups) * 0.05

        # Normalize to 0-1 range using sigmoid
        normalized = 1.0 / (1.0 + math.exp(-influence + 2))
        return normalized

    def get_reputation(self, organism_id: str) -> float:
        """
        Get the reputation score of an organism.

        Reputation is the aggregate trust from all relationships.
        Returns a value from -1.0 (terrible) to 1.0 (excellent).
        """
        incoming = self.get_incoming_relationships(organism_id)
        if not incoming:
            return 0.0

        total_trust = sum(rel.trust * rel.familiarity for rel in incoming)
        total_weight = sum(rel.familiarity for rel in incoming)

        if total_weight == 0:
            return 0.0

        return total_trust / total_weight

    def propagate_influence(
        self,
        source_id: str,
        event_type: str,
        data: Dict[str, Any],
        max_hops: int = 2,
        decay_factor: float = 0.5
    ) -> List[str]:
        """
        Propagate information/influence through the social network.

        Starting from source, spreads to connected organisms with
        decreasing influence based on relationship strength and hops.

        Args:
            source_id: Starting organism
            event_type: Type of event to propagate
            data: Event data
            max_hops: Maximum propagation distance
            decay_factor: How much influence decays per hop

        Returns:
            List of organism IDs that received the propagation
        """
        reached = []
        visited = {source_id}
        current_wave = [(source_id, 1.0)]  # (organism_id, influence_strength)

        for hop in range(max_hops):
            next_wave = []
            for organism_id, influence in current_wave:
                relationships = self.get_all_relationships(organism_id)
                for rel in relationships:
                    if rel.target_id not in visited:
                        visited.add(rel.target_id)
                        # Influence depends on relationship strength and trust
                        propagated_influence = influence * rel.strength * decay_factor
                        if rel.trust > 0:
                            propagated_influence *= (1 + rel.trust * 0.5)

                        if propagated_influence > 0.1:  # Threshold
                            reached.append(rel.target_id)
                            next_wave.append((rel.target_id, propagated_influence))

                            # Emit propagation event
                            if self._emit_events:
                                bus = get_event_bus()
                                bus.emit(f"influence.propagated.{event_type}", {
                                    "source_id": source_id,
                                    "recipient_id": rel.target_id,
                                    "hop": hop + 1,
                                    "influence": propagated_influence,
                                    "data": data
                                })

            current_wave = next_wave
            if not current_wave:
                break

        return reached

    # =========================================================================
    # Group Management
    # =========================================================================

    def form_group(
        self,
        name: str,
        members: List[str],
        leader_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Form a new social group.

        Args:
            name: Name of the group
            members: List of organism IDs to include
            leader_id: Optional leader organism ID
            metadata: Additional group data

        Returns:
            Group ID
        """
        group = Group(
            name=name,
            members=set(members),
            leader_id=leader_id,
            metadata=metadata or {}
        )

        self._groups[group.id] = group

        # Update organism -> groups mapping
        for member_id in members:
            if member_id not in self._organism_groups:
                self._organism_groups[member_id] = set()
            self._organism_groups[member_id].add(group.id)

        # Emit event
        if self._emit_events:
            bus = get_event_bus()
            bus.emit("group.formed", {
                "group_id": group.id,
                "name": name,
                "members": list(members),
                "leader_id": leader_id
            })

        return group.id

    def get_group(self, group_id: str) -> Optional[Group]:
        """Get a group by ID"""
        return self._groups.get(group_id)

    def dissolve_group(self, group_id: str) -> bool:
        """Dissolve a group"""
        group = self._groups.pop(group_id, None)
        if group:
            # Update organism -> groups mapping
            for member_id in group.members:
                if member_id in self._organism_groups:
                    self._organism_groups[member_id].discard(group_id)

            # Emit event
            if self._emit_events:
                bus = get_event_bus()
                bus.emit("group.dissolved", {
                    "group_id": group_id,
                    "name": group.name
                })

            return True
        return False

    def add_to_group(self, group_id: str, organism_id: str) -> bool:
        """Add an organism to a group"""
        group = self._groups.get(group_id)
        if group and group.add_member(organism_id):
            if organism_id not in self._organism_groups:
                self._organism_groups[organism_id] = set()
            self._organism_groups[organism_id].add(group_id)

            # Emit event
            if self._emit_events:
                bus = get_event_bus()
                bus.emit("group.member_added", {
                    "group_id": group_id,
                    "organism_id": organism_id
                })

            return True
        return False

    def remove_from_group(self, group_id: str, organism_id: str) -> bool:
        """Remove an organism from a group"""
        group = self._groups.get(group_id)
        if group and group.remove_member(organism_id):
            if organism_id in self._organism_groups:
                self._organism_groups[organism_id].discard(group_id)

            # Emit event
            if self._emit_events:
                bus = get_event_bus()
                bus.emit("group.member_removed", {
                    "group_id": group_id,
                    "organism_id": organism_id
                })

            return True
        return False

    def set_group_leader(self, group_id: str, organism_id: str) -> bool:
        """Set the leader of a group"""
        group = self._groups.get(group_id)
        if group and group.set_leader(organism_id):
            # Update leader relationships within group
            for member_id in group.members:
                if member_id != organism_id:
                    # Leader -> follower relationships
                    self.form_relationship(
                        organism_id, member_id,
                        RelationshipType.LEADER,
                        strength=0.6
                    )
                    self.form_relationship(
                        member_id, organism_id,
                        RelationshipType.FOLLOWER,
                        strength=0.6
                    )

            # Emit event
            if self._emit_events:
                bus = get_event_bus()
                bus.emit("group.leader_set", {
                    "group_id": group_id,
                    "leader_id": organism_id
                })

            return True
        return False

    def get_organism_groups(self, organism_id: str) -> List[Group]:
        """Get all groups an organism belongs to"""
        group_ids = self._organism_groups.get(organism_id, set())
        return [self._groups[gid] for gid in group_ids if gid in self._groups]

    def get_group_members(self, group_id: str) -> Set[str]:
        """Get all members of a group"""
        group = self._groups.get(group_id)
        return group.members.copy() if group else set()

    def calculate_group_cohesion(self, group_id: str) -> float:
        """
        Calculate how cohesive a group is based on internal relationships.

        Returns 0.0 to 1.0 where 1.0 is perfectly cohesive.
        """
        group = self._groups.get(group_id)
        if not group or group.size < 2:
            return 1.0

        members = list(group.members)
        total_strength = 0.0
        possible_connections = 0

        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                possible_connections += 1
                rel1 = self.get_relationship(m1, m2)
                rel2 = self.get_relationship(m2, m1)

                if rel1:
                    total_strength += rel1.strength * (1 + max(0, rel1.trust))
                if rel2:
                    total_strength += rel2.strength * (1 + max(0, rel2.trust))

        if possible_connections == 0:
            return 0.0

        # Normalize (max is 2 * possible_connections * 2 for perfect relationships)
        cohesion = total_strength / (possible_connections * 4)
        group.cohesion = min(1.0, cohesion)
        return group.cohesion

    # =========================================================================
    # Queries
    # =========================================================================

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 6
    ) -> Optional[List[str]]:
        """
        Find a path between two organisms in the social network.

        Uses BFS to find shortest path.
        Returns None if no path exists within max_depth.
        """
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = [(source_id, [source_id])]

        for _ in range(max_depth):
            if not queue:
                break

            next_queue = []
            for current_id, path in queue:
                for neighbor_id in self._relationships.get(current_id, {}).keys():
                    if neighbor_id == target_id:
                        return path + [neighbor_id]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_queue.append((neighbor_id, path + [neighbor_id]))

            queue = next_queue

        return None

    def get_common_connections(self, org1_id: str, org2_id: str) -> Set[str]:
        """Get organisms that both org1 and org2 have relationships with"""
        conn1 = self.get_connected_organisms(org1_id)
        conn2 = self.get_connected_organisms(org2_id)
        return conn1 & conn2

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the social network"""
        total_relationships = sum(
            len(rels) for rels in self._relationships.values()
        )

        relationship_types = {}
        for source_rels in self._relationships.values():
            for rel in source_rels.values():
                type_name = rel.type.value
                relationship_types[type_name] = relationship_types.get(type_name, 0) + 1

        return {
            "total_organisms": len(self._relationships),
            "total_relationships": total_relationships,
            "total_groups": len(self._groups),
            "relationship_types": relationship_types,
            "average_connections": total_relationships / max(1, len(self._relationships))
        }

    # =========================================================================
    # Organism Lifecycle
    # =========================================================================

    def remove_organism(self, organism_id: str):
        """
        Remove an organism from the network entirely.

        Removes all relationships and group memberships.
        """
        # Remove outgoing relationships
        if organism_id in self._relationships:
            for target_id in list(self._relationships[organism_id].keys()):
                self.remove_relationship(organism_id, target_id)
            del self._relationships[organism_id]

        # Remove incoming relationships
        for source_id in list(self._reverse_relationships.get(organism_id, set())):
            self.remove_relationship(source_id, organism_id)

        if organism_id in self._reverse_relationships:
            del self._reverse_relationships[organism_id]

        # Remove from groups
        for group_id in list(self._organism_groups.get(organism_id, set())):
            self.remove_from_group(group_id, organism_id)

        if organism_id in self._organism_groups:
            del self._organism_groups[organism_id]

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire social network"""
        relationships = []
        for source_rels in self._relationships.values():
            for rel in source_rels.values():
                relationships.append(rel.to_dict())

        groups = [g.to_dict() for g in self._groups.values()]

        return {
            "relationships": relationships,
            "groups": groups
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'SocialNetwork':
        """Deserialize a social network from dictionary"""
        network = cls(emit_events=emit_events)

        for rel_data in data.get("relationships", []):
            rel = Relationship.from_dict(rel_data)
            if rel.source_id not in network._relationships:
                network._relationships[rel.source_id] = {}
            network._relationships[rel.source_id][rel.target_id] = rel

            if rel.target_id not in network._reverse_relationships:
                network._reverse_relationships[rel.target_id] = set()
            network._reverse_relationships[rel.target_id].add(rel.source_id)

        for group_data in data.get("groups", []):
            group = Group.from_dict(group_data)
            network._groups[group.id] = group
            for member_id in group.members:
                if member_id not in network._organism_groups:
                    network._organism_groups[member_id] = set()
                network._organism_groups[member_id].add(group.id)

        return network
