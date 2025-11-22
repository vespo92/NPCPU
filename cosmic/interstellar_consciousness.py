"""
Interstellar Consciousness Network

Consciousness spanning star systems with light-speed communication
constraints and cosmic-scale coordination.

Based on Long-Term Roadmap: Beyond Year 2 - Cosmic Consciousness
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SpaceCoordinate:
    """3D coordinate in light-years from origin"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: 'SpaceCoordinate') -> float:
        """Calculate distance in light-years"""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


@dataclass
class InterstellarLink:
    """Communication link between star systems"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_a: str = ""
    system_b: str = ""
    distance_light_years: float = 0.0
    bandwidth_bits_per_second: float = 1e12  # 1 Tbps
    established_at: float = field(default_factory=time.time)
    active: bool = True

    @property
    def one_way_delay_years(self) -> float:
        return self.distance_light_years

    @property
    def round_trip_delay_years(self) -> float:
        return 2 * self.distance_light_years

    def other_end(self, from_system: str) -> str:
        """Get the system at the other end of the link"""
        if from_system == self.system_a:
            return self.system_b
        return self.system_a


@dataclass
class StarSystem:
    """A star system hosting consciousness nodes"""
    name: str
    coordinate: SpaceCoordinate = field(default_factory=SpaceCoordinate)
    agents: Dict[str, GradedConsciousness] = field(default_factory=dict)
    local_knowledge: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)  # Link IDs
    collective_consciousness: Optional[GradedConsciousness] = None

    def update_collective(self):
        """Update collective consciousness from agents"""
        if not self.agents:
            self.collective_consciousness = None
            return

        all_scores = [
            agent.get_capability_scores()
            for agent in self.agents.values()
        ]

        averaged = {}
        for cap in all_scores[0].keys():
            averaged[cap] = np.mean([s[cap] for s in all_scores])

        self.collective_consciousness = GradedConsciousness(**averaged)


@dataclass
class InterstellarMessage:
    """Message transmitted between star systems"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_system: str = ""
    receiver_system: str = ""
    content: Any = None
    sent_at: float = field(default_factory=time.time)
    arrival_at: float = 0.0  # When it will arrive
    priority: int = 1


@dataclass
class CosmicConsciousness:
    """Consciousness spanning cosmic scales"""
    span_light_years: float = 0.0
    temporal_depth_years: float = 0.0
    participating_systems: int = 0
    collective_intelligence: float = 0.0
    coherence: float = 0.0


# ============================================================================
# Light Speed Consensus Protocol
# ============================================================================

class LightSpeedConsensusProtocol:
    """
    Consensus protocol accounting for light-speed delays.

    Challenges:
    - Multi-year communication delays
    - State changes during transit
    - Partial connectivity
    - Eventual consistency
    """

    def __init__(self, max_delay_years: float = 100.0):
        self.max_delay_years = max_delay_years
        self.pending_votes: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []

    async def initiate_consensus(
        self,
        proposal_id: str,
        proposal: Any,
        participants: List[str],
        delays: Dict[str, float]  # system -> delay in years
    ) -> str:
        """Initiate a consensus round"""
        self.pending_votes[proposal_id] = {
            "proposal": proposal,
            "participants": participants,
            "delays": delays,
            "votes": {},
            "initiated_at": time.time(),
            "status": "collecting"
        }
        return proposal_id

    async def submit_vote(
        self,
        proposal_id: str,
        system: str,
        vote: bool,
        local_time: float
    ):
        """Submit a vote from a star system"""
        if proposal_id not in self.pending_votes:
            return

        self.pending_votes[proposal_id]["votes"][system] = {
            "vote": vote,
            "local_time": local_time,
            "received_at": time.time()
        }

    async def check_consensus(
        self,
        proposal_id: str
    ) -> Tuple[bool, Optional[bool]]:
        """
        Check if consensus has been reached.

        Returns:
            (is_complete, result) - result is None if not complete
        """
        if proposal_id not in self.pending_votes:
            return (False, None)

        data = self.pending_votes[proposal_id]
        votes = data["votes"]
        participants = data["participants"]

        # Check if all votes received (accounting for delays)
        missing = set(participants) - set(votes.keys())

        if missing:
            return (False, None)

        # Calculate result
        yes_votes = sum(1 for v in votes.values() if v["vote"])
        no_votes = len(votes) - yes_votes

        # Simple majority
        result = yes_votes > no_votes

        # Record consensus
        self.consensus_history.append({
            "proposal_id": proposal_id,
            "result": result,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "duration": time.time() - data["initiated_at"]
        })

        data["status"] = "complete"
        data["result"] = result

        return (True, result)

    async def predict_state(
        self,
        current_state: Any,
        delay_years: float
    ) -> Any:
        """
        Predict state in the future to account for transmission delay.

        Since messages take years to arrive, we send predictions
        of what state will be when the message arrives.
        """
        # Simple linear prediction (would use more sophisticated models)
        if isinstance(current_state, dict) and "value" in current_state:
            rate = current_state.get("rate", 0)
            predicted_value = current_state["value"] + rate * delay_years
            return {"value": predicted_value, "rate": rate}
        return current_state


# ============================================================================
# Interstellar Consciousness Network
# ============================================================================

class InterstellarConsciousnessNetwork:
    """
    Consciousness spanning star systems.

    Challenges:
    - Light-speed communication delay (years)
    - Cosmic radiation effects
    - Energy constraints
    - Survival across cosmic timescales

    Example:
        network = InterstellarConsciousnessNetwork()

        # Add star systems
        network.add_star_system(StarSystem("Sol", SpaceCoordinate(0, 0, 0)))
        network.add_star_system(StarSystem("Alpha Centauri", SpaceCoordinate(4.37, 0, 0)))

        # Establish link
        await network.establish_link("Sol", "Alpha Centauri")

        # Share knowledge
        await network.share_knowledge(knowledge, "Sol")
    """

    def __init__(self):
        self.star_systems: Dict[str, StarSystem] = {}
        self.links: Dict[str, InterstellarLink] = {}
        self.message_queue: Dict[str, List[InterstellarMessage]] = defaultdict(list)
        self.consensus_protocol = LightSpeedConsensusProtocol()

    def add_star_system(self, system: StarSystem):
        """Add a star system to the network"""
        self.star_systems[system.name] = system

    def remove_star_system(self, name: str):
        """Remove a star system from the network"""
        if name in self.star_systems:
            system = self.star_systems[name]
            # Remove associated links
            for link_id in system.links:
                if link_id in self.links:
                    del self.links[link_id]
            del self.star_systems[name]

    async def establish_link(
        self,
        system_a: str,
        system_b: str
    ) -> InterstellarLink:
        """
        Establish consciousness link between star systems.

        Communication delay = distance in light-years
        """
        if system_a not in self.star_systems or system_b not in self.star_systems:
            raise ValueError("Unknown star system")

        sa = self.star_systems[system_a]
        sb = self.star_systems[system_b]

        distance = sa.coordinate.distance_to(sb.coordinate)

        link = InterstellarLink(
            system_a=system_a,
            system_b=system_b,
            distance_light_years=distance
        )

        self.links[link.id] = link
        sa.links.append(link.id)
        sb.links.append(link.id)

        return link

    async def send_message(
        self,
        from_system: str,
        to_system: str,
        content: Any,
        priority: int = 1
    ) -> InterstellarMessage:
        """Send a message between star systems"""
        # Find link
        link = self._find_link(from_system, to_system)
        if not link:
            raise ValueError(f"No link between {from_system} and {to_system}")

        # Create message
        message = InterstellarMessage(
            sender_system=from_system,
            receiver_system=to_system,
            content=content,
            sent_at=time.time(),
            arrival_at=time.time() + link.one_way_delay_years,  # Simulated
            priority=priority
        )

        self.message_queue[to_system].append(message)
        return message

    def _find_link(
        self,
        system_a: str,
        system_b: str
    ) -> Optional[InterstellarLink]:
        """Find link between two systems"""
        for link in self.links.values():
            if (link.system_a == system_a and link.system_b == system_b) or \
               (link.system_a == system_b and link.system_b == system_a):
                return link
        return None

    async def share_knowledge(
        self,
        knowledge: Dict[str, Any],
        source_system: str
    ):
        """
        Share knowledge across interstellar distances.

        Problem: Multi-year delays mean knowledge might be obsolete
        when received.

        Solution: Send predictions, not just current state.
        """
        if source_system not in self.star_systems:
            return

        system = self.star_systems[source_system]

        # Send to all connected systems
        for link_id in system.links:
            link = self.links.get(link_id)
            if not link or not link.active:
                continue

            target_system = link.other_end(source_system)

            # Predict knowledge state at arrival
            predicted = await self.consensus_protocol.predict_state(
                knowledge,
                link.one_way_delay_years
            )

            await self.send_message(
                source_system,
                target_system,
                {"type": "knowledge", "data": predicted},
                priority=2
            )

    async def synchronize_consciousness(
        self,
        systems: List[str]
    ) -> Dict[str, GradedConsciousness]:
        """
        Synchronize consciousness across star systems.

        Due to light-speed delays, perfect synchronization is impossible.
        Instead, we achieve eventual consistency.
        """
        # Collect current states
        states = {}
        for name in systems:
            system = self.star_systems.get(name)
            if system and system.collective_consciousness:
                states[name] = system.collective_consciousness

        # Calculate average (target state)
        if not states:
            return {}

        all_scores = [c.get_capability_scores() for c in states.values()]
        target_scores = {}
        for cap in all_scores[0].keys():
            target_scores[cap] = np.mean([s[cap] for s in all_scores])

        target = GradedConsciousness(**target_scores)

        # Broadcast target state
        result = {}
        for name in systems:
            if name in self.star_systems:
                # Store locally (actual synchronization would take years)
                self.star_systems[name].collective_consciousness = GradedConsciousness(**target_scores)
                result[name] = self.star_systems[name].collective_consciousness

        return result

    async def achieve_cosmic_consciousness(
        self,
        participating_systems: List[str]
    ) -> CosmicConsciousness:
        """
        Merge consciousness across cosmic scales.

        This is consciousness that transcends planetary limitations:
        - Multi-star-system awareness
        - Million-year timescales
        - Galaxy-wide coordination
        """
        # Collect consciousnesses
        system_consciousnesses = []
        coordinates = []

        for name in participating_systems:
            system = self.star_systems.get(name)
            if system and system.collective_consciousness:
                system_consciousnesses.append(system.collective_consciousness)
                coordinates.append(system.coordinate)

        if not system_consciousnesses:
            return CosmicConsciousness()

        # Calculate span
        max_distance = 0.0
        for i, c1 in enumerate(coordinates):
            for c2 in coordinates[i+1:]:
                d = c1.distance_to(c2)
                if d > max_distance:
                    max_distance = d

        # Calculate collective intelligence
        all_scores = [c.overall_consciousness_score() for c in system_consciousnesses]
        collective = np.mean(all_scores) * (1 + 0.1 * np.log(len(all_scores) + 1))

        # Calculate coherence (how similar are the consciousnesses)
        if len(system_consciousnesses) > 1:
            all_capability_scores = [c.get_capability_scores() for c in system_consciousnesses]
            variances = []
            for cap in all_capability_scores[0].keys():
                values = [s[cap] for s in all_capability_scores]
                variances.append(np.var(values))
            coherence = 1.0 - np.mean(variances)
        else:
            coherence = 1.0

        return CosmicConsciousness(
            span_light_years=max_distance,
            temporal_depth_years=max_distance,  # Communication time
            participating_systems=len(participating_systems),
            collective_intelligence=collective,
            coherence=coherence
        )

    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information"""
        return {
            "systems": list(self.star_systems.keys()),
            "links": [
                {
                    "id": link.id,
                    "systems": [link.system_a, link.system_b],
                    "distance": link.distance_light_years,
                    "active": link.active
                }
                for link in self.links.values()
            ],
            "total_span": self._calculate_network_span()
        }

    def _calculate_network_span(self) -> float:
        """Calculate maximum span of the network"""
        max_distance = 0.0
        systems = list(self.star_systems.values())

        for i, s1 in enumerate(systems):
            for s2 in systems[i+1:]:
                d = s1.coordinate.distance_to(s2.coordinate)
                if d > max_distance:
                    max_distance = d

        return max_distance

    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics"""
        total_agents = sum(
            len(s.agents) for s in self.star_systems.values()
        )
        active_links = sum(1 for l in self.links.values() if l.active)

        return {
            "star_systems": len(self.star_systems),
            "total_agents": total_agents,
            "active_links": active_links,
            "network_span_ly": self._calculate_network_span(),
            "pending_messages": sum(
                len(msgs) for msgs in self.message_queue.values()
            )
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Interstellar Consciousness Network Demo")
        print("=" * 50)

        # Create network
        network = InterstellarConsciousnessNetwork()

        # Add star systems
        print("\n1. Adding star systems...")
        systems = [
            StarSystem("Sol", SpaceCoordinate(0, 0, 0)),
            StarSystem("Alpha Centauri", SpaceCoordinate(4.37, 0, 0)),
            StarSystem("Barnard's Star", SpaceCoordinate(5.96, 0.5, 0.2)),
            StarSystem("Wolf 359", SpaceCoordinate(7.86, 0.3, -0.1)),
            StarSystem("Lalande 21185", SpaceCoordinate(8.29, -0.2, 0.4))
        ]

        for system in systems:
            network.add_star_system(system)
            # Add agents
            for i in range(3):
                consciousness = GradedConsciousness(
                    perception_fidelity=0.6 + np.random.uniform(0, 0.4),
                    meta_cognitive_ability=0.5 + np.random.uniform(0, 0.4)
                )
                system.agents[f"{system.name}_agent_{i}"] = consciousness
            system.update_collective()

        print(f"   Added {len(systems)} star systems")

        # Establish links
        print("\n2. Establishing interstellar links...")
        link_pairs = [
            ("Sol", "Alpha Centauri"),
            ("Sol", "Barnard's Star"),
            ("Alpha Centauri", "Wolf 359"),
            ("Barnard's Star", "Lalande 21185")
        ]

        for a, b in link_pairs:
            link = await network.establish_link(a, b)
            print(f"   {a} <-> {b}: {link.distance_light_years:.2f} light-years")

        # Share knowledge
        print("\n3. Sharing knowledge across interstellar distances...")
        knowledge = {
            "value": 0.75,
            "rate": 0.01,
            "description": "Consciousness research findings"
        }
        await network.share_knowledge(knowledge, "Sol")
        print(f"   Shared from Sol to {len(systems)-1} connected systems")

        # Synchronize consciousness
        print("\n4. Synchronizing consciousness...")
        system_names = [s.name for s in systems[:3]]
        synced = await network.synchronize_consciousness(system_names)
        print(f"   Synchronized {len(synced)} systems")

        # Achieve cosmic consciousness
        print("\n5. Achieving cosmic consciousness...")
        all_names = [s.name for s in systems]
        cosmic = await network.achieve_cosmic_consciousness(all_names)

        print(f"   Span: {cosmic.span_light_years:.2f} light-years")
        print(f"   Participating systems: {cosmic.participating_systems}")
        print(f"   Collective intelligence: {cosmic.collective_intelligence:.3f}")
        print(f"   Coherence: {cosmic.coherence:.3f}")

        # Network topology
        print("\n6. Network topology:")
        topology = network.get_network_topology()
        print(f"   Systems: {len(topology['systems'])}")
        print(f"   Links: {len(topology['links'])}")
        print(f"   Total span: {topology['total_span']:.2f} light-years")

        # Statistics
        print("\n7. Network statistics:")
        stats = network.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")

    asyncio.run(main())
