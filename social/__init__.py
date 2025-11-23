"""
Social Module for NPCPU

Provides comprehensive social dynamics and communication systems for organisms.

Components:
- SocialNetwork: Graph-based social relationships
- CommunicationSystem: Inter-organism signaling and messaging
- SocialGraph: Advanced graph-based network representation
- ReputationEngine: Multi-dimensional reputation system
- CoalitionManager: Coalition formation and dynamics
- HierarchyManager: Dominance hierarchy modeling
- CulturalEngine: Meme and knowledge transmission
- ConflictResolver: Dispute resolution mechanisms
- KinshipTracker: Family and genetic relationship tracking
- CooperationEngine: Cooperation evolution and game theory

Example:
    from social import (
        SocialNetwork, RelationshipType, Relationship, Group,
        CommunicationSystem, SignalType, MessageType,
        SocialGraph, ReputationEngine, CoalitionManager,
        HierarchyManager, CulturalEngine, ConflictResolver,
        KinshipTracker, CooperationEngine
    )

    # Create social network
    network = SocialNetwork()
    network.form_relationship("org1", "org2", RelationshipType.ALLY)

    # Track kinship
    kinship = KinshipTracker()
    kinship.register_birth("child", "parent1", "parent2")

    # Model cooperation
    coop = CooperationEngine()
    coop.register_organism("player1", Strategy.TIT_FOR_TAT)
"""

from social.relationships import (
    SocialNetwork,
    RelationshipType,
    Relationship,
    Group,
)

from social.communication import (
    CommunicationSystem,
    CommunicationChannel,
    SignalType,
    MessageType,
    Signal,
    Message,
)

from social.social_graph import (
    SocialGraph,
    GraphNode,
    GraphEdge,
    GraphMetrics,
    Community,
    EdgeType,
)

from social.reputation import (
    ReputationEngine,
    ReputationDimension,
    ReputationObservation,
    ReputationScore,
    GossipMessage,
)

from social.coalition import (
    CoalitionManager,
    Coalition,
    CoalitionMember,
    CoalitionType,
    MembershipRole,
    CoalitionFormationStrategy,
)

from social.hierarchy import (
    HierarchyManager,
    DominanceScore,
    DominanceContest,
    HierarchyType,
    ContestType,
)

from social.cultural_transmission import (
    CulturalEngine,
    Meme,
    CulturalKnowledge,
    TransmissionEvent,
    TransmissionMode,
    MemeCategory,
)

from social.conflict_resolution import (
    ConflictResolver,
    Conflict,
    ConflictOutcome,
    ConflictType,
    Strategy as ConflictStrategy,
    OutcomeType,
)

from social.kinship import (
    KinshipTracker,
    KinRecord,
    KinLink,
    KinshipRelation,
)

from social.cooperation import (
    CooperationEngine,
    CooperationProfile,
    GameResult,
    Strategy,
    Action,
    Game,
)

__all__ = [
    # Relationships
    "SocialNetwork",
    "RelationshipType",
    "Relationship",
    "Group",
    # Communication
    "CommunicationSystem",
    "CommunicationChannel",
    "SignalType",
    "MessageType",
    "Signal",
    "Message",
    # Social Graph
    "SocialGraph",
    "GraphNode",
    "GraphEdge",
    "GraphMetrics",
    "Community",
    "EdgeType",
    # Reputation
    "ReputationEngine",
    "ReputationDimension",
    "ReputationObservation",
    "ReputationScore",
    "GossipMessage",
    # Coalition
    "CoalitionManager",
    "Coalition",
    "CoalitionMember",
    "CoalitionType",
    "MembershipRole",
    "CoalitionFormationStrategy",
    # Hierarchy
    "HierarchyManager",
    "DominanceScore",
    "DominanceContest",
    "HierarchyType",
    "ContestType",
    # Cultural Transmission
    "CulturalEngine",
    "Meme",
    "CulturalKnowledge",
    "TransmissionEvent",
    "TransmissionMode",
    "MemeCategory",
    # Conflict Resolution
    "ConflictResolver",
    "Conflict",
    "ConflictOutcome",
    "ConflictType",
    "ConflictStrategy",
    "OutcomeType",
    # Kinship
    "KinshipTracker",
    "KinRecord",
    "KinLink",
    "KinshipRelation",
    # Cooperation
    "CooperationEngine",
    "CooperationProfile",
    "GameResult",
    "Strategy",
    "Action",
    "Game",
]
