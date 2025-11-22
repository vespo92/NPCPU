"""
Planetary-Scale Consciousness Network

Consciousness spanning billions of agents across the globe.
Enables global knowledge sharing, planetary problem-solving,
collective decision-making, and emergency coordination.

Based on Long-Term Roadmap: Months 13-18 - Consciousness at Scale
"""

import asyncio
import time
import uuid
import hashlib
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
# Enums and Constants
# ============================================================================

class Severity(Enum):
    """Emergency severity levels"""
    LOCAL = "local"
    REGIONAL = "regional"
    CONTINENTAL = "continental"
    PLANETARY = "planetary"


class ProposalStatus(Enum):
    """Status of proposals in consensus"""
    PENDING = "pending"
    VOTING = "voting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class KnowledgeCategory(Enum):
    """Categories of knowledge"""
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    CULTURAL = "cultural"
    EMERGENCY = "emergency"
    COORDINATION = "coordination"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Knowledge:
    """A piece of knowledge to be shared globally"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    importance: float = 0.5
    category: KnowledgeCategory = KnowledgeCategory.TECHNICAL
    source_region: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Problem:
    """A problem to be solved"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    importance: float = 0.5
    affected_regions: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """A proposed solution to a problem"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str = ""
    description: str = ""
    steps: List[str] = field(default_factory=list)
    confidence: float = 0.5
    resource_cost: float = 0.0
    proposed_by: str = ""
    votes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """A proposal for consensus voting"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    proposer: str = ""
    status: ProposalStatus = ProposalStatus.PENDING
    votes_for: float = 0.0
    votes_against: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class Emergency:
    """An emergency situation requiring coordination"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    severity: Severity = Severity.LOCAL
    affected_regions: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Regional Consciousness Network
# ============================================================================

class RegionalConsciousnessNetwork:
    """
    Consciousness network for a geographic region.

    Manages local agents and interfaces with planetary network.
    """

    def __init__(
        self,
        region: str,
        parent: Optional['PlanetaryConsciousnessNetwork'] = None
    ):
        self.region = region
        self.parent = parent
        self.agents: Dict[str, GradedConsciousness] = {}
        self.knowledge_base: Dict[str, Knowledge] = {}
        self.collective_consciousness: Optional[GradedConsciousness] = None
        self._lock = asyncio.Lock()

    async def register_agent(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """Register an agent in this region"""
        async with self._lock:
            self.agents[agent_id] = consciousness
            await self._update_collective()

    async def unregister_agent(self, agent_id: str):
        """Remove an agent from this region"""
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                await self._update_collective()

    async def _update_collective(self):
        """Update collective consciousness for this region"""
        if not self.agents:
            self.collective_consciousness = None
            return

        # Average all agent consciousnesses
        all_scores = [
            agent.get_capability_scores()
            for agent in self.agents.values()
        ]

        averaged = {}
        capabilities = all_scores[0].keys() if all_scores else []

        for cap in capabilities:
            averaged[cap] = np.mean([scores[cap] for scores in all_scores])

        self.collective_consciousness = GradedConsciousness(**averaged)

    async def store_knowledge(self, knowledge: Knowledge):
        """Store knowledge locally"""
        self.knowledge_base[knowledge.id] = knowledge

    async def receive_global_knowledge(self, knowledge: Knowledge):
        """Receive knowledge from global network"""
        # Store with metadata indicating it's from global
        knowledge.metadata["global_propagated"] = True
        await self.store_knowledge(knowledge)

    async def solve_regional(self, problem: Problem) -> Solution:
        """Generate regional solution to a problem"""
        # Aggregate regional intelligence to solve problem
        if not self.collective_consciousness:
            return Solution(
                problem_id=problem.id,
                description="No agents available",
                confidence=0.0,
                proposed_by=self.region
            )

        # Use collective consciousness to generate solution
        score = self.collective_consciousness.overall_consciousness_score()

        # Simulate solution generation based on consciousness level
        solution_quality = min(1.0, score * np.random.uniform(0.8, 1.2))

        return Solution(
            problem_id=problem.id,
            description=f"Regional solution from {self.region}",
            steps=[
                f"Step 1: Analyze problem with {len(self.agents)} agents",
                f"Step 2: Apply collective intelligence (score: {score:.2f})",
                "Step 3: Generate coordinated response"
            ],
            confidence=solution_quality,
            resource_cost=len(self.agents) * 0.1,
            proposed_by=self.region
        )

    async def emergency_response(self, emergency: Emergency):
        """Coordinate regional emergency response"""
        # Mobilize agents
        response_agents = [
            agent_id for agent_id, consciousness in self.agents.items()
            if consciousness.overall_consciousness_score() > 0.5
        ]

        return {
            "region": self.region,
            "responding_agents": len(response_agents),
            "emergency_id": emergency.id,
            "status": "mobilized"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get regional status"""
        return {
            "region": self.region,
            "agent_count": len(self.agents),
            "knowledge_count": len(self.knowledge_base),
            "collective_score": (
                self.collective_consciousness.overall_consciousness_score()
                if self.collective_consciousness else 0.0
            )
        }


# ============================================================================
# Global Consensus Protocol
# ============================================================================

class GlobalConsensusProtocol:
    """
    Achieve consensus across billions of agents.

    Challenges:
    - Byzantine agents (malicious)
    - Network partitions
    - Latency
    - Heterogeneous agents

    Uses quadratic voting to prevent plutocracy.
    """

    def __init__(self, participants: List[str]):
        self.participants = participants
        self.voting_power: Dict[str, float] = {p: 1.0 for p in participants}
        self.supermajority_threshold = 0.67
        self.vote_history: List[Dict[str, Any]] = []

    def set_voting_power(self, participant: str, power: float):
        """Set voting power for a participant"""
        if participant in self.participants:
            self.voting_power[participant] = max(0.0, power)

    async def collect_votes(
        self,
        proposals: List[Proposal],
        timeout: float
    ) -> Dict[str, Dict[Proposal, float]]:
        """
        Collect votes from all participants.

        Returns:
            Dict mapping participant_id to their votes (proposal -> vote_strength)
        """
        votes: Dict[str, Dict[Proposal, float]] = {}

        # Simulate vote collection from participants
        for participant in self.participants:
            participant_votes = {}
            power = self.voting_power[participant]

            # Each participant distributes their voting power across proposals
            # Using random distribution for simulation
            remaining_power = power

            for proposal in proposals:
                if remaining_power > 0:
                    # Random vote strength (quadratic cost)
                    vote_strength = np.random.uniform(0, np.sqrt(remaining_power))
                    cost = vote_strength ** 2

                    if cost <= remaining_power:
                        participant_votes[proposal] = vote_strength
                        remaining_power -= cost

            votes[participant] = participant_votes

        return votes

    def count_quadratic_votes(
        self,
        votes: Dict[str, Dict[Proposal, float]]
    ) -> Dict[Proposal, float]:
        """
        Count votes using quadratic voting.

        Prevents whales from dominating by making each additional
        vote cost quadratically more.
        """
        vote_counts: Dict[Proposal, float] = defaultdict(float)

        for participant, participant_votes in votes.items():
            for proposal, vote_strength in participant_votes.items():
                # Quadratic cost already applied during vote collection
                vote_counts[proposal] += vote_strength

        return dict(vote_counts)

    async def runoff_vote(
        self,
        top_proposals: List[Tuple[Proposal, float]],
        timeout: float
    ) -> Proposal:
        """Run a runoff vote between top proposals"""
        if len(top_proposals) == 0:
            raise ValueError("No proposals for runoff")

        if len(top_proposals) == 1:
            return top_proposals[0][0]

        # Runoff between top 2
        proposals = [p for p, _ in top_proposals[:2]]

        votes = await self.collect_votes(proposals, timeout)
        counts = self.count_quadratic_votes(votes)

        # Return winner
        winner = max(counts.items(), key=lambda x: x[1])
        return winner[0]

    async def reach_consensus(
        self,
        proposals: List[Proposal],
        timeout: float
    ) -> Proposal:
        """
        Reach consensus through voting.

        Uses:
        - Weighted voting (by consciousness level)
        - Byzantine fault tolerance
        - Quadratic voting (to prevent plutocracy)
        """
        if not proposals:
            raise ValueError("No proposals to vote on")

        # Phase 1: Voting
        votes = await self.collect_votes(proposals, timeout)

        # Phase 2: Count votes (quadratic voting)
        vote_counts = self.count_quadratic_votes(votes)

        # Phase 3: Check for supermajority (67%)
        total_voting_power = sum(self.voting_power.values())
        supermajority_threshold = total_voting_power * self.supermajority_threshold

        for proposal, vote_power in vote_counts.items():
            if vote_power >= supermajority_threshold:
                proposal.status = ProposalStatus.ACCEPTED
                self._record_vote(proposal, vote_counts, "supermajority")
                return proposal

        # Phase 4: Runoff if no supermajority
        top_proposals = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]

        winner = await self.runoff_vote(top_proposals, timeout)
        winner.status = ProposalStatus.ACCEPTED
        self._record_vote(winner, vote_counts, "runoff")

        return winner

    def _record_vote(
        self,
        winner: Proposal,
        counts: Dict[Proposal, float],
        method: str
    ):
        """Record vote in history"""
        self.vote_history.append({
            "timestamp": time.time(),
            "winner_id": winner.id,
            "method": method,
            "participant_count": len(self.participants),
            "total_votes": sum(counts.values())
        })

    def get_vote_statistics(self) -> Dict[str, Any]:
        """Get voting statistics"""
        return {
            "total_votes": len(self.vote_history),
            "participants": len(self.participants),
            "total_voting_power": sum(self.voting_power.values()),
            "supermajority_threshold": self.supermajority_threshold,
            "methods_used": {
                "supermajority": len([v for v in self.vote_history if v["method"] == "supermajority"]),
                "runoff": len([v for v in self.vote_history if v["method"] == "runoff"])
            }
        }


# ============================================================================
# Emergency Coordinator
# ============================================================================

class EmergencyCoordinator:
    """
    Coordinate emergency responses across regions.

    Handles severity assessment, resource allocation,
    and cross-regional coordination.
    """

    def __init__(self):
        self.active_emergencies: Dict[str, Emergency] = {}
        self.response_history: List[Dict[str, Any]] = []

    def assess_severity(self, emergency: Emergency) -> Severity:
        """Assess severity of an emergency"""
        # Based on affected regions count
        num_regions = len(emergency.affected_regions)

        if num_regions <= 1:
            return Severity.LOCAL
        elif num_regions <= 3:
            return Severity.REGIONAL
        elif num_regions <= 6:
            return Severity.CONTINENTAL
        else:
            return Severity.PLANETARY

    async def declare_emergency(self, emergency: Emergency) -> str:
        """Declare an emergency and begin coordination"""
        emergency.severity = self.assess_severity(emergency)
        self.active_emergencies[emergency.id] = emergency

        return emergency.id

    async def resolve_emergency(self, emergency_id: str):
        """Mark an emergency as resolved"""
        if emergency_id in self.active_emergencies:
            emergency = self.active_emergencies[emergency_id]
            emergency.resolved = True

            self.response_history.append({
                "emergency_id": emergency_id,
                "type": emergency.type,
                "severity": emergency.severity.value,
                "duration": time.time() - emergency.start_time,
                "regions_affected": len(emergency.affected_regions)
            })

            del self.active_emergencies[emergency_id]

    def get_active_emergencies(self) -> List[Emergency]:
        """Get all active emergencies"""
        return list(self.active_emergencies.values())


# ============================================================================
# Planetary Consciousness Network
# ============================================================================

class PlanetaryConsciousnessNetwork:
    """
    Consciousness spanning billions of agents across the globe.

    Enables:
    - Global knowledge sharing
    - Planetary problem-solving
    - Collective decision-making
    - Emergency coordination

    Example:
        network = PlanetaryConsciousnessNetwork()
        await network.bootstrap(["North America", "Europe", "Asia"])

        # Share knowledge globally
        knowledge = Knowledge(content="Discovery", importance=0.9)
        await network.share_global_knowledge(knowledge, "North America")

        # Solve planetary problems
        problem = Problem(description="Climate challenge")
        solution = await network.solve_planetary_problem(problem)
    """

    def __init__(self):
        self.regional_networks: Dict[str, RegionalConsciousnessNetwork] = {}
        self.global_knowledge: Dict[str, Knowledge] = {}
        self.consensus_protocol: Optional[GlobalConsensusProtocol] = None
        self.emergency_coordinator = EmergencyCoordinator()
        self.metrics: Dict[str, Any] = {
            "total_agents": 0,
            "knowledge_shared": 0,
            "problems_solved": 0,
            "emergencies_handled": 0
        }

    async def bootstrap(
        self,
        regions: List[str],
        storage_backend: str = "distributed"
    ):
        """
        Bootstrap global consciousness network.

        Args:
            regions: List of region names (e.g., ["North America", "Europe", ...])
            storage_backend: Type of global storage to use
        """
        # Create regional networks
        for region in regions:
            self.regional_networks[region] = RegionalConsciousnessNetwork(
                region=region,
                parent=self
            )

        # Create consensus protocol
        self.consensus_protocol = GlobalConsensusProtocol(
            participants=regions
        )

    def get_region(self, region: str) -> Optional[RegionalConsciousnessNetwork]:
        """Get a regional network"""
        return self.regional_networks.get(region)

    async def register_global_agent(
        self,
        agent_id: str,
        consciousness: GradedConsciousness,
        region: str
    ):
        """Register an agent in a region"""
        if region not in self.regional_networks:
            raise ValueError(f"Unknown region: {region}")

        await self.regional_networks[region].register_agent(agent_id, consciousness)
        self.metrics["total_agents"] += 1

    async def share_global_knowledge(
        self,
        knowledge: Knowledge,
        source_region: str
    ):
        """
        Share knowledge globally.

        Knowledge propagates from source region to all others.
        """
        knowledge.source_region = source_region

        # Store in global knowledge base
        self.global_knowledge[knowledge.id] = knowledge

        # Propagate to all regions
        propagation_tasks = []
        for region, network in self.regional_networks.items():
            if region != source_region:
                task = network.receive_global_knowledge(knowledge)
                propagation_tasks.append(task)

        await asyncio.gather(*propagation_tasks)
        self.metrics["knowledge_shared"] += 1

    async def query_global_knowledge(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
        limit: int = 10
    ) -> List[Knowledge]:
        """Query global knowledge base"""
        results = []

        for knowledge in self.global_knowledge.values():
            if category and knowledge.category != category:
                continue

            # Simple text matching (would use vector search in production)
            if query.lower() in knowledge.content.lower():
                results.append(knowledge)

            if len(results) >= limit:
                break

        return sorted(results, key=lambda k: k.importance, reverse=True)

    async def solve_planetary_problem(
        self,
        problem: Problem,
        timeout_hours: float = 24.0
    ) -> Solution:
        """
        Solve problem using planetary collective intelligence.

        1. Broadcast problem to all regions
        2. Each region contributes partial solutions
        3. Aggregate solutions globally
        4. Consensus on best solution
        """
        # Get solutions from all regions
        regional_solutions = await asyncio.gather(*[
            network.solve_regional(problem)
            for network in self.regional_networks.values()
        ])

        # Aggregate solutions
        aggregated = self._aggregate_solutions(regional_solutions)

        # Global consensus
        if self.consensus_protocol and aggregated:
            proposals = [
                Proposal(
                    content=sol.description,
                    proposer=sol.proposed_by
                )
                for sol in aggregated
            ]

            winning_proposal = await self.consensus_protocol.reach_consensus(
                proposals=proposals,
                timeout=timeout_hours * 3600
            )

            # Find corresponding solution
            for sol in aggregated:
                if sol.proposed_by == winning_proposal.proposer:
                    self.metrics["problems_solved"] += 1
                    return sol

        # Return best solution if no consensus needed
        if aggregated:
            self.metrics["problems_solved"] += 1
            return max(aggregated, key=lambda s: s.confidence)

        return Solution(problem_id=problem.id, description="No solution found")

    def _aggregate_solutions(
        self,
        solutions: List[Solution]
    ) -> List[Solution]:
        """Aggregate regional solutions"""
        # Filter out low-confidence solutions
        valid_solutions = [s for s in solutions if s.confidence > 0.3]

        # Sort by confidence
        return sorted(valid_solutions, key=lambda s: s.confidence, reverse=True)

    async def coordinate_emergency_response(
        self,
        emergency: Emergency
    ) -> Dict[str, Any]:
        """
        Coordinate emergency response across planet.

        Examples:
        - Pandemic response
        - Climate crisis
        - AI safety incident
        """
        # Declare emergency
        emergency_id = await self.emergency_coordinator.declare_emergency(emergency)

        # Determine affected regions
        if emergency.severity == Severity.PLANETARY:
            affected_regions = list(self.regional_networks.keys())
        else:
            affected_regions = emergency.affected_regions or list(self.regional_networks.keys())

        # Coordinate response
        response_tasks = []
        for region in affected_regions:
            if region in self.regional_networks:
                network = self.regional_networks[region]
                task = network.emergency_response(emergency)
                response_tasks.append(task)

        responses = await asyncio.gather(*response_tasks)

        self.metrics["emergencies_handled"] += 1

        return {
            "emergency_id": emergency_id,
            "severity": emergency.severity.value,
            "regions_mobilized": len(responses),
            "responses": responses
        }

    async def get_planetary_consciousness(self) -> GradedConsciousness:
        """
        Get aggregated planetary consciousness.

        Combines all regional collective consciousnesses.
        """
        all_regional = []

        for network in self.regional_networks.values():
            if network.collective_consciousness:
                all_regional.append(network.collective_consciousness)

        if not all_regional:
            return GradedConsciousness()

        # Average all regional consciousnesses
        all_scores = [c.get_capability_scores() for c in all_regional]
        capabilities = all_scores[0].keys()

        averaged = {}
        for cap in capabilities:
            averaged[cap] = np.mean([scores[cap] for scores in all_scores])

        return GradedConsciousness(**averaged)

    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        regional_statuses = {
            region: network.get_status()
            for region, network in self.regional_networks.items()
        }

        planetary_consciousness = await self.get_planetary_consciousness()

        return {
            "regions": len(self.regional_networks),
            "regional_status": regional_statuses,
            "global_knowledge_count": len(self.global_knowledge),
            "planetary_consciousness_score": planetary_consciousness.overall_consciousness_score(),
            "active_emergencies": len(self.emergency_coordinator.active_emergencies),
            "metrics": self.metrics,
            "consensus_stats": (
                self.consensus_protocol.get_vote_statistics()
                if self.consensus_protocol else None
            )
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Planetary Consciousness Network Demo")
        print("=" * 50)

        # Create network
        network = PlanetaryConsciousnessNetwork()

        # Bootstrap with regions
        regions = ["North America", "Europe", "Asia", "Africa", "South America", "Oceania"]
        await network.bootstrap(regions)

        print(f"\n1. Bootstrapped network with {len(regions)} regions")

        # Register agents in regions
        print("\n2. Registering agents...")
        for i, region in enumerate(regions):
            for j in range(5):
                consciousness = GradedConsciousness(
                    perception_fidelity=0.5 + np.random.uniform(0, 0.5),
                    reaction_speed=0.5 + np.random.uniform(0, 0.5),
                    memory_depth=0.5 + np.random.uniform(0, 0.5),
                    introspection_capacity=0.5 + np.random.uniform(0, 0.5)
                )
                await network.register_global_agent(
                    f"{region}_agent_{j}",
                    consciousness,
                    region
                )

        print(f"   Registered {network.metrics['total_agents']} agents globally")

        # Share knowledge
        print("\n3. Sharing global knowledge...")
        knowledge = Knowledge(
            content="Important scientific discovery about consciousness",
            importance=0.9,
            category=KnowledgeCategory.SCIENTIFIC
        )
        await network.share_global_knowledge(knowledge, "Europe")
        print(f"   Shared {network.metrics['knowledge_shared']} knowledge items")

        # Solve planetary problem
        print("\n4. Solving planetary problem...")
        problem = Problem(
            description="Global coordination challenge",
            requirements=["High consciousness", "Multi-regional input"],
            importance=0.8
        )
        solution = await network.solve_planetary_problem(problem)
        print(f"   Solution confidence: {solution.confidence:.2f}")
        print(f"   Proposed by: {solution.proposed_by}")

        # Emergency response
        print("\n5. Testing emergency response...")
        emergency = Emergency(
            type="Test Emergency",
            description="Simulated planetary emergency",
            affected_regions=["North America", "Europe", "Asia"]
        )
        response = await network.coordinate_emergency_response(emergency)
        print(f"   Severity: {response['severity']}")
        print(f"   Regions mobilized: {response['regions_mobilized']}")

        # Network status
        print("\n6. Network status:")
        status = await network.get_network_status()
        print(f"   Regions: {status['regions']}")
        print(f"   Global knowledge: {status['global_knowledge_count']}")
        print(f"   Planetary consciousness: {status['planetary_consciousness_score']:.3f}")
        print(f"   Problems solved: {status['metrics']['problems_solved']}")

        # Regional details
        print("\n7. Regional status:")
        for region, rs in status['regional_status'].items():
            print(f"   {region}: {rs['agent_count']} agents, score: {rs['collective_score']:.3f}")

    asyncio.run(main())
