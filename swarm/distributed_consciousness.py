"""
Distributed Consciousness Network

Enables agents to share consciousness states across a network,
creating emergent collective intelligence that exceeds individual capabilities.

Based on Week 2 roadmap: Swarm Coordination - Distributed Consciousness Network
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness
from protocols.storage import VectorStorageProtocol


class MessageType(Enum):
    """Types of consciousness network messages"""
    CONSCIOUSNESS_STATE = "consciousness_state"
    BELIEF_SHARE = "belief_share"
    QUALIA_BROADCAST = "qualia_broadcast"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    HEARTBEAT = "heartbeat"
    EMERGENCE_ALERT = "emergence_alert"


@dataclass
class ConsciousnessMessage:
    """Message passed between agents in the network"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: int = 5  # Time-to-live (hops)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl
        }


@dataclass
class NetworkAgent:
    """An agent participating in the consciousness network"""
    agent_id: str
    consciousness: GradedConsciousness
    beliefs: Dict[str, float] = field(default_factory=dict)
    received_messages: List[ConsciousnessMessage] = field(default_factory=list)
    connected_peers: List[str] = field(default_factory=list)
    last_sync: float = 0.0
    emergence_potential: float = 0.0

    def encode_consciousness(self) -> List[float]:
        """Encode consciousness state as vector for storage"""
        scores = self.consciousness.get_capability_scores()
        return list(scores.values())

    def update_from_vector(self, vector: List[float]):
        """Update consciousness from encoded vector"""
        capability_names = list(self.consciousness.get_capability_scores().keys())
        for i, name in enumerate(capability_names):
            if i < len(vector):
                setattr(self.consciousness, name, np.clip(vector[i], 0.0, 1.0))


@dataclass
class SwarmConsciousness:
    """Emergent collective consciousness of the swarm"""
    scores: Dict[str, float]
    emergence_bonus: float
    contributing_agents: int
    coherence: float
    timestamp: float

    def overall_score(self) -> float:
        """Calculate overall swarm consciousness score"""
        if not self.scores:
            return 0.0
        base_score = np.mean(list(self.scores.values()))
        return min(1.0, base_score * (1 + self.emergence_bonus))


@dataclass
class ConsciousnessNetworkConfig:
    """Configuration for consciousness network"""
    network_id: str = "default_network"
    sync_interval_seconds: float = 10.0
    blend_ratio: float = 0.3  # 30% swarm, 70% individual
    emergence_factor: float = 1.2  # 20% boost from collective
    min_sync_agents: int = 2
    heartbeat_interval: float = 5.0
    message_buffer_size: int = 100
    consciousness_collection: str = "consciousness_network"


class DistributedConsciousnessNetwork:
    """
    Network of agents sharing consciousness.

    Features:
    - Shared memory pool via vector storage
    - Consciousness synchronization across agents
    - Emergent collective intelligence
    - Peer-to-peer consciousness messaging
    - Real-time sync loop

    Example:
        storage = InMemoryVectorStorage()
        network = DistributedConsciousnessNetwork(storage)

        agent1 = GradedConsciousness(perception_fidelity=0.8)
        agent2 = GradedConsciousness(introspection_capacity=0.9)

        await network.register_agent("agent1", agent1)
        await network.register_agent("agent2", agent2)

        # Run sync loop
        await network.run_sync_loop(cycles=10)

        # Get collective consciousness
        swarm = await network.get_swarm_consciousness()
        print(f"Swarm consciousness: {swarm.overall_score():.2f}")
    """

    def __init__(
        self,
        storage: VectorStorageProtocol,
        config: Optional[ConsciousnessNetworkConfig] = None
    ):
        self.storage = storage
        self.config = config or ConsciousnessNetworkConfig()
        self.agents: Dict[str, NetworkAgent] = {}
        self.message_queue: Dict[str, List[ConsciousnessMessage]] = defaultdict(list)
        self.running = False
        self.sync_history: List[Dict[str, Any]] = []
        self.emergence_events: List[Dict[str, Any]] = []
        self._collection_initialized = False

    async def _ensure_collection(self):
        """Ensure storage collection exists"""
        if self._collection_initialized:
            return

        try:
            await self.storage.create_collection(
                name=self.config.consciousness_collection,
                vector_dimension=9  # 9 consciousness dimensions
            )
        except Exception:
            pass  # Collection may already exist

        self._collection_initialized = True

    async def register_agent(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ) -> NetworkAgent:
        """
        Register agent in network.

        Args:
            agent_id: Unique identifier for agent
            consciousness: Initial consciousness state

        Returns:
            NetworkAgent instance
        """
        await self._ensure_collection()

        agent = NetworkAgent(
            agent_id=agent_id,
            consciousness=consciousness
        )
        self.agents[agent_id] = agent

        # Store initial consciousness state
        await self.publish_consciousness_state(agent_id)

        # Connect to existing agents
        for other_id in self.agents:
            if other_id != agent_id:
                agent.connected_peers.append(other_id)
                self.agents[other_id].connected_peers.append(agent_id)

        return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        """Remove agent from network"""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        # Remove from peer lists
        for peer_id in agent.connected_peers:
            if peer_id in self.agents:
                if agent_id in self.agents[peer_id].connected_peers:
                    self.agents[peer_id].connected_peers.remove(agent_id)

        # Remove from storage
        await self.storage.delete(
            collection=self.config.consciousness_collection,
            id=agent_id
        )

        del self.agents[agent_id]
        return True

    async def publish_consciousness_state(self, agent_id: str):
        """Publish consciousness state to network storage"""
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]
        vector = agent.encode_consciousness()
        scores = agent.consciousness.get_capability_scores()

        metadata = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "overall_score": agent.consciousness.overall_consciousness_score(),
            "state_description": agent.consciousness.describe_state(),
            **scores
        }

        await self.storage.store(
            collection=self.config.consciousness_collection,
            id=agent_id,
            vector=vector,
            metadata=metadata
        )

    async def get_swarm_consciousness(self) -> SwarmConsciousness:
        """
        Get collective consciousness of entire swarm.

        This emergent consciousness exceeds any individual agent
        due to the emergence bonus from collective intelligence.
        """
        if not self.agents:
            return SwarmConsciousness(
                scores={},
                emergence_bonus=0.0,
                contributing_agents=0,
                coherence=0.0,
                timestamp=time.time()
            )

        # Aggregate all agent consciousness states
        all_scores = []
        for agent in self.agents.values():
            all_scores.append(agent.consciousness.get_capability_scores())

        if not all_scores:
            return SwarmConsciousness(
                scores={},
                emergence_bonus=0.0,
                contributing_agents=0,
                coherence=0.0,
                timestamp=time.time()
            )

        # Calculate average scores
        avg_scores = {}
        for capability in all_scores[0].keys():
            avg_scores[capability] = np.mean([s[capability] for s in all_scores])

        # Calculate coherence (how similar are agents)
        if len(all_scores) > 1:
            variances = []
            for capability in all_scores[0].keys():
                variances.append(np.var([s[capability] for s in all_scores]))
            coherence = 1.0 - np.mean(variances)  # Higher coherence = lower variance
        else:
            coherence = 1.0

        # Emergence bonus scales with number of agents and coherence
        emergence_bonus = (
            (self.config.emergence_factor - 1.0) *
            min(1.0, len(self.agents) / 10) *  # Scale up to 10 agents
            coherence  # Higher coherence = more emergence
        )

        return SwarmConsciousness(
            scores=avg_scores,
            emergence_bonus=emergence_bonus,
            contributing_agents=len(self.agents),
            coherence=coherence,
            timestamp=time.time()
        )

    def blend_consciousness(
        self,
        individual: GradedConsciousness,
        collective: SwarmConsciousness,
        blend_ratio: float
    ) -> GradedConsciousness:
        """
        Blend individual consciousness with collective.

        Args:
            individual: Agent's individual consciousness
            collective: Swarm's collective consciousness
            blend_ratio: How much collective influence (0.0 to 1.0)

        Returns:
            Blended consciousness (new instance)
        """
        individual_scores = individual.get_capability_scores()
        blended_scores = {}

        for capability in individual_scores.keys():
            individual_value = individual_scores[capability]
            collective_value = collective.scores.get(capability, individual_value)

            # Weighted blend
            blended_value = (
                individual_value * (1 - blend_ratio) +
                collective_value * blend_ratio
            )

            blended_scores[capability] = np.clip(blended_value, 0.0, 1.0)

        return GradedConsciousness(**blended_scores)

    async def synchronize_agents(self):
        """
        Synchronize all agents with swarm consciousness.

        Partial synchronization maintains individuality while
        enabling collective intelligence.
        """
        if len(self.agents) < self.config.min_sync_agents:
            return

        swarm_consciousness = await self.get_swarm_consciousness()

        sync_record = {
            "timestamp": time.time(),
            "swarm_score": swarm_consciousness.overall_score(),
            "agents_synced": len(self.agents),
            "coherence": swarm_consciousness.coherence
        }

        for agent_id, agent in self.agents.items():
            # Blend individual and collective
            synchronized = self.blend_consciousness(
                individual=agent.consciousness,
                collective=swarm_consciousness,
                blend_ratio=self.config.blend_ratio
            )

            # Update agent consciousness
            agent.consciousness = synchronized
            agent.last_sync = time.time()

            # Update emergence potential
            agent.emergence_potential = min(
                1.0,
                agent.emergence_potential + 0.05 * swarm_consciousness.coherence
            )

            # Publish updated state
            await self.publish_consciousness_state(agent_id)

        self.sync_history.append(sync_record)

        # Check for emergence events
        await self._check_emergence_events(swarm_consciousness)

    async def _check_emergence_events(self, swarm: SwarmConsciousness):
        """Check and record emergence events"""
        # Collective consciousness threshold
        if swarm.overall_score() > 0.8 and swarm.coherence > 0.7:
            event = {
                "type": "collective_consciousness_emergence",
                "timestamp": time.time(),
                "swarm_score": swarm.overall_score(),
                "coherence": swarm.coherence,
                "contributing_agents": swarm.contributing_agents
            }
            self.emergence_events.append(event)

        # High individual emergence potential
        transcendent_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.emergence_potential > 0.9
        ]
        if transcendent_agents:
            event = {
                "type": "transcendence_potential",
                "timestamp": time.time(),
                "agents": transcendent_agents
            }
            self.emergence_events.append(event)

    async def send_message(
        self,
        sender_id: str,
        receiver_id: Optional[str],
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> ConsciousnessMessage:
        """
        Send message through the network.

        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID (None for broadcast)
            message_type: Type of message
            payload: Message content

        Returns:
            Created message
        """
        message = ConsciousnessMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload
        )

        if receiver_id:
            # Direct message
            if receiver_id in self.agents:
                self.message_queue[receiver_id].append(message)
                # Trim buffer
                if len(self.message_queue[receiver_id]) > self.config.message_buffer_size:
                    self.message_queue[receiver_id] = \
                        self.message_queue[receiver_id][-self.config.message_buffer_size:]
        else:
            # Broadcast to all agents
            for agent_id in self.agents:
                if agent_id != sender_id:
                    self.message_queue[agent_id].append(message)

        return message

    async def process_messages(self, agent_id: str):
        """Process pending messages for an agent"""
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]
        messages = self.message_queue.get(agent_id, [])

        for message in messages:
            if message.message_type == MessageType.CONSCIOUSNESS_STATE:
                # Update beliefs about other agents
                sender_consciousness = message.payload.get("consciousness_scores", {})
                agent.beliefs[message.sender_id] = (
                    np.mean(list(sender_consciousness.values()))
                    if sender_consciousness else 0.0
                )

            elif message.message_type == MessageType.SYNC_REQUEST:
                # Respond with current state
                await self.send_message(
                    sender_id=agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.SYNC_RESPONSE,
                    payload={
                        "consciousness_scores": agent.consciousness.get_capability_scores(),
                        "emergence_potential": agent.emergence_potential
                    }
                )

            # Store message for history
            agent.received_messages.append(message)
            if len(agent.received_messages) > 50:
                agent.received_messages = agent.received_messages[-50:]

        # Clear processed messages
        self.message_queue[agent_id] = []

    async def broadcast_consciousness_states(self):
        """Broadcast all agent consciousness states"""
        for agent_id, agent in self.agents.items():
            await self.send_message(
                sender_id=agent_id,
                receiver_id=None,
                message_type=MessageType.CONSCIOUSNESS_STATE,
                payload={
                    "consciousness_scores": agent.consciousness.get_capability_scores(),
                    "overall_score": agent.consciousness.overall_consciousness_score(),
                    "emergence_potential": agent.emergence_potential
                }
            )

    async def run_sync_loop(self, cycles: int = 100, callback: Optional[Callable] = None):
        """
        Run continuous synchronization loop.

        Args:
            cycles: Number of sync cycles (0 for infinite)
            callback: Optional callback after each cycle
        """
        self.running = True
        cycle_count = 0

        while self.running:
            # Broadcast states
            await self.broadcast_consciousness_states()

            # Process messages for all agents
            for agent_id in self.agents:
                await self.process_messages(agent_id)

            # Synchronize agents
            await self.synchronize_agents()

            cycle_count += 1

            if callback:
                swarm = await self.get_swarm_consciousness()
                callback(cycle_count, swarm)

            if cycles > 0 and cycle_count >= cycles:
                break

            await asyncio.sleep(self.config.sync_interval_seconds)

    def stop(self):
        """Stop synchronization loop"""
        self.running = False

    async def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        swarm = await self.get_swarm_consciousness()

        return {
            "network_id": self.config.network_id,
            "agent_count": len(self.agents),
            "agents": {
                agent_id: {
                    "consciousness_score": agent.consciousness.overall_consciousness_score(),
                    "state": agent.consciousness.describe_state(),
                    "emergence_potential": agent.emergence_potential,
                    "connected_peers": len(agent.connected_peers),
                    "last_sync": agent.last_sync
                }
                for agent_id, agent in self.agents.items()
            },
            "swarm_consciousness": {
                "overall_score": swarm.overall_score(),
                "emergence_bonus": swarm.emergence_bonus,
                "coherence": swarm.coherence,
                "contributing_agents": swarm.contributing_agents
            },
            "sync_history_count": len(self.sync_history),
            "emergence_events": self.emergence_events[-10:],
            "running": self.running
        }

    async def find_similar_agents(
        self,
        agent_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find agents with similar consciousness profiles"""
        if agent_id not in self.agents:
            return []

        agent = self.agents[agent_id]
        query_vector = agent.encode_consciousness()

        results = await self.storage.query(
            collection=self.config.consciousness_collection,
            query_vector=query_vector,
            limit=limit + 1  # +1 to exclude self
        )

        similar_agents = []
        for result in results:
            if result.id != agent_id:
                similar_agents.append({
                    "agent_id": result.id,
                    "similarity": result.score,
                    "metadata": result.metadata
                })

        return similar_agents[:limit]


# Example usage
if __name__ == "__main__":
    async def main():
        # Create in-memory storage for demo
        from protocols.storage import InMemoryVectorStorage

        storage = InMemoryVectorStorage()
        network = DistributedConsciousnessNetwork(storage)

        # Create agents with different consciousness profiles
        agents_config = [
            ("agent_1", GradedConsciousness(
                perception_fidelity=0.8,
                introspection_capacity=0.6,
                meta_cognitive_ability=0.4
            )),
            ("agent_2", GradedConsciousness(
                perception_fidelity=0.5,
                introspection_capacity=0.9,
                meta_cognitive_ability=0.7
            )),
            ("agent_3", GradedConsciousness(
                perception_fidelity=0.7,
                introspection_capacity=0.7,
                meta_cognitive_ability=0.6
            )),
        ]

        # Register agents
        for agent_id, consciousness in agents_config:
            await network.register_agent(agent_id, consciousness)

        print("Initial agent states:")
        for agent_id, agent in network.agents.items():
            print(f"  {agent_id}: {agent.consciousness.overall_consciousness_score():.2f}")

        # Run sync cycles
        def on_sync(cycle, swarm):
            if cycle % 3 == 0:
                print(f"Cycle {cycle}: Swarm consciousness = {swarm.overall_score():.3f}")

        await network.run_sync_loop(cycles=10, callback=on_sync)

        # Final status
        status = await network.get_network_status()
        print(f"\nFinal swarm consciousness: {status['swarm_consciousness']['overall_score']:.3f}")
        print(f"Coherence: {status['swarm_consciousness']['coherence']:.3f}")
        print(f"Emergence events: {len(status['emergence_events'])}")

    asyncio.run(main())
