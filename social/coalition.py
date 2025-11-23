"""
Coalition Formation and Dynamics for NPCPU

Implements coalition behavior with:
- Coalition formation based on shared interests
- Coalition stability analysis
- Power dynamics within coalitions
- Coalition splitting and merging
- Betrayal and defection modeling

Example:
    from social.coalition import CoalitionManager, CoalitionFormationStrategy

    # Create coalition manager
    manager = CoalitionManager()

    # Form a coalition
    coalition_id = manager.form_coalition(
        founder_id="alpha",
        initial_members=["beta", "gamma"],
        purpose="hunting"
    )

    # Add member
    manager.request_membership(coalition_id, "delta")

    # Check stability
    stability = manager.calculate_stability(coalition_id)
"""

from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
import math
import uuid

from core.events import get_event_bus


class CoalitionType(Enum):
    """Types of coalitions"""
    TEMPORARY = "temporary"           # Short-term alliance
    PERMANENT = "permanent"           # Long-term commitment
    DEFENSIVE = "defensive"           # Mutual defense
    OFFENSIVE = "offensive"           # Joint aggression
    ECONOMIC = "economic"             # Resource sharing
    TERRITORIAL = "territorial"       # Territory control
    REPRODUCTIVE = "reproductive"     # Mating coalition


class MembershipRole(Enum):
    """Roles within a coalition"""
    FOUNDER = "founder"
    LEADER = "leader"
    CORE = "core"
    MEMBER = "member"
    PROBATION = "probation"
    AFFILIATE = "affiliate"


class CoalitionFormationStrategy(Enum):
    """Strategies for forming coalitions"""
    SIZE_MAXIMIZING = "size_maximizing"       # Recruit as many as possible
    QUALITY_FOCUSED = "quality_focused"       # Recruit only high-value members
    KIN_BASED = "kin_based"                   # Prefer relatives
    RECIPROCITY = "reciprocity"               # Prefer those who reciprocate
    BALANCED = "balanced"                     # Mix of size and quality


@dataclass
class CoalitionMember:
    """
    Represents a member of a coalition.

    Attributes:
        organism_id: Member identifier
        role: Role in coalition
        contribution: Contribution score
        loyalty: Loyalty to coalition
        join_time: When they joined
        metadata: Additional data
    """
    organism_id: str
    role: MembershipRole = MembershipRole.MEMBER
    contribution: float = 0.5
    loyalty: float = 0.5
    join_time: datetime = field(default_factory=datetime.now)
    last_contribution: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_contribution(self, amount: float) -> None:
        """Update contribution score"""
        self.contribution = max(0.0, min(1.0, self.contribution + amount))
        self.last_contribution = datetime.now()

    def update_loyalty(self, amount: float) -> None:
        """Update loyalty score"""
        self.loyalty = max(0.0, min(1.0, self.loyalty + amount))


@dataclass
class Coalition:
    """
    Represents a coalition of organisms.

    Attributes:
        id: Unique identifier
        name: Coalition name
        coalition_type: Type of coalition
        founder_id: Who founded it
        members: Dictionary of members
        purpose: Coalition's goal
        resources: Shared resources
        power: Collective power score
        stability: How stable the coalition is
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    coalition_type: CoalitionType = CoalitionType.TEMPORARY
    founder_id: str = ""
    members: Dict[str, CoalitionMember] = field(default_factory=dict)
    purpose: str = ""
    resources: Dict[str, float] = field(default_factory=dict)
    power: float = 0.0
    stability: float = 0.5
    formation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def leader_id(self) -> Optional[str]:
        for member_id, member in self.members.items():
            if member.role == MembershipRole.LEADER:
                return member_id
        return self.founder_id if self.founder_id in self.members else None

    def get_member(self, organism_id: str) -> Optional[CoalitionMember]:
        return self.members.get(organism_id)

    def has_member(self, organism_id: str) -> bool:
        return organism_id in self.members

    def get_members_by_role(self, role: MembershipRole) -> List[str]:
        return [
            member_id for member_id, member in self.members.items()
            if member.role == role
        ]


@dataclass
class CoalitionProposal:
    """
    Proposal to form or join a coalition.

    Attributes:
        proposer_id: Who is proposing
        target_id: Who is being invited (or coalition ID)
        proposal_type: "form", "join", "merge"
        terms: Proposed terms
        status: "pending", "accepted", "rejected"
    """
    proposer_id: str
    target_id: str
    proposal_type: str = "join"
    terms: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CoalitionManager:
    """
    Manages coalition formation and dynamics.

    Provides:
    - Coalition formation and dissolution
    - Membership management
    - Power and stability calculations
    - Coalition merging and splitting
    - Betrayal detection

    Example:
        manager = CoalitionManager()

        # Form coalition
        cid = manager.form_coalition("alpha", ["beta", "gamma"])

        # Add resources
        manager.contribute_resources(cid, "alpha", {"food": 10})

        # Check stability
        stability = manager.calculate_stability(cid)
    """

    def __init__(
        self,
        min_coalition_size: int = 2,
        max_coalition_size: int = 20,
        stability_threshold: float = 0.3,
        emit_events: bool = True
    ):
        """
        Initialize the coalition manager.

        Args:
            min_coalition_size: Minimum members for a valid coalition
            max_coalition_size: Maximum coalition size
            stability_threshold: Below this, coalition may dissolve
            emit_events: Whether to emit events
        """
        self._coalitions: Dict[str, Coalition] = {}
        self._organism_coalitions: Dict[str, Set[str]] = defaultdict(set)
        self._proposals: Dict[str, CoalitionProposal] = {}
        self._betrayal_history: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)

        self._min_size = min_coalition_size
        self._max_size = max_coalition_size
        self._stability_threshold = stability_threshold
        self._emit_events = emit_events

    # =========================================================================
    # Coalition Formation
    # =========================================================================

    def form_coalition(
        self,
        founder_id: str,
        initial_members: Optional[List[str]] = None,
        name: str = "",
        coalition_type: CoalitionType = CoalitionType.TEMPORARY,
        purpose: str = ""
    ) -> str:
        """
        Form a new coalition.

        Args:
            founder_id: Who is founding the coalition
            initial_members: Initial member IDs
            name: Coalition name
            coalition_type: Type of coalition
            purpose: Coalition's goal

        Returns:
            Coalition ID
        """
        coalition = Coalition(
            name=name or f"Coalition_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            coalition_type=coalition_type,
            founder_id=founder_id,
            purpose=purpose
        )

        # Add founder as leader
        founder_member = CoalitionMember(
            organism_id=founder_id,
            role=MembershipRole.FOUNDER,
            contribution=1.0,
            loyalty=1.0
        )
        coalition.members[founder_id] = founder_member
        self._organism_coalitions[founder_id].add(coalition.id)

        # Add initial members
        if initial_members:
            for member_id in initial_members:
                if member_id != founder_id:
                    member = CoalitionMember(
                        organism_id=member_id,
                        role=MembershipRole.CORE,
                        contribution=0.5,
                        loyalty=0.5
                    )
                    coalition.members[member_id] = member
                    self._organism_coalitions[member_id].add(coalition.id)

        # Calculate initial power
        coalition.power = self._calculate_power(coalition)

        self._coalitions[coalition.id] = coalition

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.formed", {
                "coalition_id": coalition.id,
                "founder_id": founder_id,
                "members": list(coalition.members.keys()),
                "purpose": purpose
            })

        return coalition.id

    def dissolve_coalition(self, coalition_id: str, reason: str = "") -> bool:
        """
        Dissolve a coalition.

        Returns:
            True if dissolved successfully
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition:
            return False

        # Remove all members
        for member_id in list(coalition.members.keys()):
            self._organism_coalitions[member_id].discard(coalition_id)

        del self._coalitions[coalition_id]

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.dissolved", {
                "coalition_id": coalition_id,
                "reason": reason
            })

        return True

    # =========================================================================
    # Membership Management
    # =========================================================================

    def request_membership(
        self,
        coalition_id: str,
        organism_id: str,
        terms: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Request to join a coalition.

        Returns:
            Proposal ID if request was made
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition:
            return None

        if organism_id in coalition.members:
            return None  # Already a member

        if coalition.size >= self._max_size:
            return None  # Coalition full

        proposal = CoalitionProposal(
            proposer_id=organism_id,
            target_id=coalition_id,
            proposal_type="join",
            terms=terms or {}
        )

        self._proposals[proposal.id] = proposal

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.membership_requested", {
                "coalition_id": coalition_id,
                "organism_id": organism_id
            })

        return proposal.id

    def accept_membership(self, proposal_id: str) -> bool:
        """Accept a membership proposal"""
        proposal = self._proposals.get(proposal_id)
        if not proposal or proposal.status != "pending":
            return False

        coalition = self._coalitions.get(proposal.target_id)
        if not coalition:
            return False

        member = CoalitionMember(
            organism_id=proposal.proposer_id,
            role=MembershipRole.PROBATION,
            contribution=0.3,
            loyalty=0.4
        )

        coalition.members[proposal.proposer_id] = member
        self._organism_coalitions[proposal.proposer_id].add(coalition.id)

        proposal.status = "accepted"
        coalition.power = self._calculate_power(coalition)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.member_joined", {
                "coalition_id": coalition.id,
                "organism_id": proposal.proposer_id
            })

        return True

    def reject_membership(self, proposal_id: str) -> bool:
        """Reject a membership proposal"""
        proposal = self._proposals.get(proposal_id)
        if not proposal or proposal.status != "pending":
            return False

        proposal.status = "rejected"
        return True

    def remove_member(
        self,
        coalition_id: str,
        organism_id: str,
        reason: str = ""
    ) -> bool:
        """
        Remove a member from a coalition.

        Returns:
            True if removed successfully
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition or organism_id not in coalition.members:
            return False

        del coalition.members[organism_id]
        self._organism_coalitions[organism_id].discard(coalition_id)

        # Check if coalition should dissolve
        if coalition.size < self._min_size:
            self.dissolve_coalition(coalition_id, "insufficient_members")
        else:
            coalition.power = self._calculate_power(coalition)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.member_removed", {
                "coalition_id": coalition_id,
                "organism_id": organism_id,
                "reason": reason
            })

        return True

    def defect(self, coalition_id: str, organism_id: str) -> bool:
        """
        Member defects from coalition (voluntary leave with penalty).

        Returns:
            True if defected successfully
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition or organism_id not in coalition.members:
            return False

        # Record betrayal
        self._betrayal_history[organism_id].append((coalition_id, datetime.now()))

        self.remove_member(coalition_id, organism_id, "defection")

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.defection", {
                "coalition_id": coalition_id,
                "organism_id": organism_id
            })

        return True

    def promote_member(
        self,
        coalition_id: str,
        organism_id: str,
        new_role: MembershipRole
    ) -> bool:
        """Promote a member to a new role"""
        coalition = self._coalitions.get(coalition_id)
        if not coalition or organism_id not in coalition.members:
            return False

        coalition.members[organism_id].role = new_role

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.member_promoted", {
                "coalition_id": coalition_id,
                "organism_id": organism_id,
                "new_role": new_role.value
            })

        return True

    # =========================================================================
    # Resource Management
    # =========================================================================

    def contribute_resources(
        self,
        coalition_id: str,
        organism_id: str,
        resources: Dict[str, float]
    ) -> bool:
        """
        Contribute resources to coalition.

        Returns:
            True if contribution accepted
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition or organism_id not in coalition.members:
            return False

        member = coalition.members[organism_id]

        # Add resources
        for resource_type, amount in resources.items():
            coalition.resources[resource_type] = (
                coalition.resources.get(resource_type, 0) + amount
            )

        # Update member contribution
        total_contribution = sum(resources.values())
        member.update_contribution(total_contribution * 0.1)

        # Increase loyalty from contribution
        member.update_loyalty(0.05)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.resources_contributed", {
                "coalition_id": coalition_id,
                "organism_id": organism_id,
                "resources": resources
            })

        return True

    def distribute_resources(
        self,
        coalition_id: str,
        resource_type: str,
        distribution: Dict[str, float]
    ) -> bool:
        """
        Distribute resources among members.

        Args:
            coalition_id: Coalition ID
            resource_type: Type of resource
            distribution: Member ID -> amount mapping
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition:
            return False

        total_needed = sum(distribution.values())
        available = coalition.resources.get(resource_type, 0)

        if total_needed > available:
            return False  # Not enough resources

        coalition.resources[resource_type] -= total_needed

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.resources_distributed", {
                "coalition_id": coalition_id,
                "resource_type": resource_type,
                "distribution": distribution
            })

        return True

    # =========================================================================
    # Power and Stability
    # =========================================================================

    def calculate_stability(self, coalition_id: str) -> float:
        """
        Calculate coalition stability.

        Stability is based on:
        - Average loyalty
        - Contribution variance
        - Size relative to ideal
        - Internal conflicts
        - Betrayal history of members
        """
        coalition = self._coalitions.get(coalition_id)
        if not coalition:
            return 0.0

        if coalition.size < self._min_size:
            return 0.0

        # Average loyalty
        loyalties = [m.loyalty for m in coalition.members.values()]
        avg_loyalty = sum(loyalties) / len(loyalties) if loyalties else 0

        # Contribution equity (low variance is good)
        contributions = [m.contribution for m in coalition.members.values()]
        avg_contrib = sum(contributions) / len(contributions) if contributions else 0
        variance = sum((c - avg_contrib) ** 2 for c in contributions) / len(contributions) if contributions else 0
        equity_score = 1.0 / (1.0 + variance * 10)

        # Size factor (prefer medium-sized coalitions)
        ideal_size = (self._min_size + self._max_size) / 2
        size_factor = 1.0 - abs(coalition.size - ideal_size) / ideal_size

        # Betrayal penalty
        betrayal_count = sum(
            len(self._betrayal_history.get(m_id, []))
            for m_id in coalition.members
        )
        betrayal_factor = 1.0 / (1.0 + betrayal_count * 0.2)

        # Combine factors
        stability = (
            0.4 * avg_loyalty +
            0.2 * equity_score +
            0.2 * size_factor +
            0.2 * betrayal_factor
        )

        coalition.stability = stability
        return stability

    def _calculate_power(self, coalition: Coalition) -> float:
        """Calculate collective power of a coalition"""
        if not coalition.members:
            return 0.0

        # Power from members
        member_power = len(coalition.members) * 0.1

        # Power from resources
        resource_power = sum(coalition.resources.values()) * 0.05

        # Power from cohesion
        avg_contrib = sum(
            m.contribution for m in coalition.members.values()
        ) / len(coalition.members)
        cohesion_power = avg_contrib * 0.3

        return min(1.0, member_power + resource_power + cohesion_power)

    def get_power_ranking(self) -> List[Tuple[str, float]]:
        """Get coalitions ranked by power"""
        rankings = [
            (cid, c.power) for cid, c in self._coalitions.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    # =========================================================================
    # Coalition Operations
    # =========================================================================

    def merge_coalitions(
        self,
        coalition1_id: str,
        coalition2_id: str,
        new_name: str = ""
    ) -> Optional[str]:
        """
        Merge two coalitions into one.

        Returns:
            New coalition ID if successful
        """
        c1 = self._coalitions.get(coalition1_id)
        c2 = self._coalitions.get(coalition2_id)

        if not c1 or not c2:
            return None

        combined_size = c1.size + c2.size
        if combined_size > self._max_size:
            return None

        # Create merged coalition
        new_coalition = Coalition(
            name=new_name or f"{c1.name}_{c2.name}",
            coalition_type=c1.coalition_type,
            founder_id=c1.founder_id,
            purpose=c1.purpose
        )

        # Merge members
        for member_id, member in c1.members.items():
            new_coalition.members[member_id] = member
            self._organism_coalitions[member_id].discard(coalition1_id)
            self._organism_coalitions[member_id].add(new_coalition.id)

        for member_id, member in c2.members.items():
            if member_id not in new_coalition.members:
                new_coalition.members[member_id] = member
            self._organism_coalitions[member_id].discard(coalition2_id)
            self._organism_coalitions[member_id].add(new_coalition.id)

        # Merge resources
        for rtype, amount in c1.resources.items():
            new_coalition.resources[rtype] = amount
        for rtype, amount in c2.resources.items():
            new_coalition.resources[rtype] = (
                new_coalition.resources.get(rtype, 0) + amount
            )

        new_coalition.power = self._calculate_power(new_coalition)

        # Remove old coalitions
        del self._coalitions[coalition1_id]
        del self._coalitions[coalition2_id]

        self._coalitions[new_coalition.id] = new_coalition

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.merged", {
                "coalition1_id": coalition1_id,
                "coalition2_id": coalition2_id,
                "new_coalition_id": new_coalition.id
            })

        return new_coalition.id

    def split_coalition(
        self,
        coalition_id: str,
        splinter_members: List[str]
    ) -> Optional[str]:
        """
        Split a coalition, forming a new one from splinter members.

        Returns:
            New coalition ID if successful
        """
        original = self._coalitions.get(coalition_id)
        if not original:
            return None

        if len(splinter_members) < self._min_size:
            return None

        remaining = original.size - len(splinter_members)
        if remaining < self._min_size:
            return None  # Would make original too small

        # Create splinter coalition
        leader = splinter_members[0]
        new_coalition_id = self.form_coalition(
            founder_id=leader,
            initial_members=splinter_members[1:],
            name=f"{original.name}_splinter",
            coalition_type=original.coalition_type
        )

        # Remove from original
        for member_id in splinter_members:
            if member_id in original.members:
                del original.members[member_id]
                self._organism_coalitions[member_id].discard(coalition_id)

        original.power = self._calculate_power(original)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("coalition.split", {
                "original_id": coalition_id,
                "splinter_id": new_coalition_id,
                "splinter_members": splinter_members
            })

        return new_coalition_id

    # =========================================================================
    # Queries
    # =========================================================================

    def get_coalition(self, coalition_id: str) -> Optional[Coalition]:
        """Get coalition by ID"""
        return self._coalitions.get(coalition_id)

    def get_organism_coalitions(self, organism_id: str) -> List[Coalition]:
        """Get all coalitions an organism belongs to"""
        coalition_ids = self._organism_coalitions.get(organism_id, set())
        return [
            self._coalitions[cid] for cid in coalition_ids
            if cid in self._coalitions
        ]

    def are_allies(self, org1_id: str, org2_id: str) -> bool:
        """Check if two organisms are in the same coalition"""
        coalitions1 = self._organism_coalitions.get(org1_id, set())
        coalitions2 = self._organism_coalitions.get(org2_id, set())
        return bool(coalitions1 & coalitions2)

    def get_betrayal_history(self, organism_id: str) -> List[Tuple[str, datetime]]:
        """Get betrayal history for an organism"""
        return self._betrayal_history.get(organism_id, [])

    def check_unstable_coalitions(self) -> List[str]:
        """Get list of unstable coalitions that might dissolve"""
        unstable = []
        for cid in self._coalitions:
            stability = self.calculate_stability(cid)
            if stability < self._stability_threshold:
                unstable.append(cid)
        return unstable

    def get_stats(self) -> Dict[str, Any]:
        """Get coalition statistics"""
        if not self._coalitions:
            return {
                "total_coalitions": 0,
                "total_members": 0,
                "avg_size": 0,
                "avg_stability": 0,
                "avg_power": 0
            }

        total_members = sum(c.size for c in self._coalitions.values())
        sizes = [c.size for c in self._coalitions.values()]
        stabilities = [c.stability for c in self._coalitions.values()]
        powers = [c.power for c in self._coalitions.values()]

        return {
            "total_coalitions": len(self._coalitions),
            "total_members": total_members,
            "avg_size": sum(sizes) / len(sizes),
            "avg_stability": sum(stabilities) / len(stabilities),
            "avg_power": sum(powers) / len(powers)
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize coalition manager"""
        coalitions = []
        for coalition in self._coalitions.values():
            members = [
                {
                    "organism_id": m.organism_id,
                    "role": m.role.value,
                    "contribution": m.contribution,
                    "loyalty": m.loyalty,
                    "join_time": m.join_time.isoformat()
                }
                for m in coalition.members.values()
            ]
            coalitions.append({
                "id": coalition.id,
                "name": coalition.name,
                "coalition_type": coalition.coalition_type.value,
                "founder_id": coalition.founder_id,
                "members": members,
                "purpose": coalition.purpose,
                "resources": coalition.resources,
                "power": coalition.power,
                "stability": coalition.stability
            })

        return {"coalitions": coalitions}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'CoalitionManager':
        """Deserialize coalition manager"""
        manager = cls(emit_events=emit_events)

        for c_data in data.get("coalitions", []):
            coalition = Coalition(
                id=c_data["id"],
                name=c_data["name"],
                coalition_type=CoalitionType(c_data["coalition_type"]),
                founder_id=c_data["founder_id"],
                purpose=c_data.get("purpose", ""),
                resources=c_data.get("resources", {}),
                power=c_data.get("power", 0),
                stability=c_data.get("stability", 0.5)
            )

            for m_data in c_data.get("members", []):
                member = CoalitionMember(
                    organism_id=m_data["organism_id"],
                    role=MembershipRole(m_data["role"]),
                    contribution=m_data.get("contribution", 0.5),
                    loyalty=m_data.get("loyalty", 0.5),
                    join_time=datetime.fromisoformat(m_data["join_time"])
                )
                coalition.members[member.organism_id] = member
                manager._organism_coalitions[member.organism_id].add(coalition.id)

            manager._coalitions[coalition.id] = coalition

        return manager
