"""
Social Module for NPCPU

Provides social dynamics and communication systems for organisms.

Components:
- SocialNetwork: Graph-based social relationships
- CommunicationSystem: Inter-organism signaling and messaging

Example:
    from social import (
        SocialNetwork, RelationshipType, Relationship, Group,
        CommunicationSystem, SignalType, MessageType
    )

    # Create social network
    network = SocialNetwork()
    network.form_relationship("org1", "org2", RelationshipType.ALLY)

    # Add communication to an organism
    comm = CommunicationSystem("comm", owner=organism)
    organism.add_subsystem(comm)
    comm.broadcast_signal(SignalType.WARNING, {"threat": "predator"})
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
]
