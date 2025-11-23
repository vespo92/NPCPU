"""
Conflict Resolution System for NPCPU

Implements dispute resolution mechanisms with:
- Multiple resolution strategies (fight, flee, negotiate, submit)
- Mediation by third parties
- Cost-benefit analysis
- Escalation dynamics
- Reconciliation after conflict

Example:
    from social.conflict_resolution import ConflictResolver, ConflictType, Strategy

    # Create resolver
    resolver = ConflictResolver()

    # Register a conflict
    conflict = resolver.register_conflict(
        "org_a", "org_b",
        ConflictType.RESOURCE,
        resource_value=100
    )

    # Resolve
    outcome = resolver.resolve(conflict.id, Strategy.NEGOTIATE)
    print(f"Winner: {outcome.winner_id}, Cost: {outcome.total_cost}")
"""

from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
import random
import math
import uuid

from core.events import get_event_bus


class ConflictType(Enum):
    """Types of conflicts"""
    RESOURCE = "resource"           # Competition for resources
    TERRITORY = "territory"         # Territorial dispute
    MATE = "mate"                   # Mating competition
    DOMINANCE = "dominance"         # Status challenge
    REVENGE = "revenge"             # Retaliation
    IDEOLOGICAL = "ideological"     # Belief conflict


class Strategy(Enum):
    """Resolution strategies"""
    FIGHT = "fight"                 # Physical confrontation
    FLEE = "flee"                   # Withdraw from conflict
    SUBMIT = "submit"               # Accept lower status
    NEGOTIATE = "negotiate"         # Find compromise
    DISPLAY = "display"             # Threat display
    APPEAL = "appeal"               # Seek third-party judgment
    SHARE = "share"                 # Split the resource


class OutcomeType(Enum):
    """Types of conflict outcomes"""
    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"
    COMPROMISE = "compromise"
    WITHDRAWAL = "withdrawal"


@dataclass
class Conflict:
    """
    Represents an ongoing or past conflict.

    Attributes:
        id: Unique identifier
        party1_id: First party
        party2_id: Second party
        conflict_type: Type of conflict
        resource_value: Value at stake
        intensity: Current intensity level
        status: "pending", "active", "resolved"
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    party1_id: str = ""
    party2_id: str = ""
    conflict_type: ConflictType = ConflictType.RESOURCE
    resource_value: float = 100.0
    intensity: float = 0.5
    status: str = "pending"
    start_time: datetime = field(default_factory=datetime.now)
    escalations: int = 0
    mediator_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def parties(self) -> Tuple[str, str]:
        return (self.party1_id, self.party2_id)


@dataclass
class ConflictOutcome:
    """
    Outcome of a resolved conflict.

    Attributes:
        conflict_id: Which conflict was resolved
        winner_id: Who won (None for draw/compromise)
        loser_id: Who lost (None for draw/compromise)
        outcome_type: Type of outcome
        strategy_used: Resolution strategy
        party1_cost: Cost to party 1
        party2_cost: Cost to party 2
        resource_division: How resources were divided
    """
    conflict_id: str
    winner_id: Optional[str] = None
    loser_id: Optional[str] = None
    outcome_type: OutcomeType = OutcomeType.DRAW
    strategy_used: Strategy = Strategy.NEGOTIATE
    party1_cost: float = 0.0
    party2_cost: float = 0.0
    party1_gain: float = 0.0
    party2_gain: float = 0.0
    resource_division: Dict[str, float] = field(default_factory=dict)
    reconciled: bool = False
    resolution_time: datetime = field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        return self.party1_cost + self.party2_cost


@dataclass
class OrganismConflictProfile:
    """
    Conflict-related attributes of an organism.

    Attributes:
        organism_id: Organism identifier
        fighting_ability: Combat capability
        resource_holding_potential: Ability to defend resources
        aggression: Base aggression level
        conflict_history: Past conflict outcomes
    """
    organism_id: str
    fighting_ability: float = 0.5
    resource_holding_potential: float = 0.5
    aggression: float = 0.3
    submission_threshold: float = 0.3
    wins: int = 0
    losses: int = 0
    draws: int = 0
    injuries_received: float = 0.0
    injuries_inflicted: float = 0.0

    def update_from_outcome(self, outcome: ConflictOutcome, is_party1: bool) -> None:
        """Update profile based on conflict outcome"""
        if outcome.winner_id == self.organism_id:
            self.wins += 1
        elif outcome.loser_id == self.organism_id:
            self.losses += 1
        else:
            self.draws += 1

        cost = outcome.party1_cost if is_party1 else outcome.party2_cost
        self.injuries_received += cost


class ConflictResolver:
    """
    Manages conflict resolution between organisms.

    Provides:
    - Conflict registration and tracking
    - Multiple resolution strategies
    - Cost-benefit calculations
    - Mediation support
    - Reconciliation mechanics

    Example:
        resolver = ConflictResolver()

        # Register conflict
        conflict = resolver.register_conflict("alpha", "beta", ConflictType.TERRITORY)

        # Get recommended strategy
        strategy = resolver.recommend_strategy(conflict.id, "alpha")

        # Resolve
        outcome = resolver.resolve(conflict.id, strategy)
    """

    def __init__(
        self,
        injury_risk: float = 0.3,
        negotiation_success_rate: float = 0.6,
        emit_events: bool = True
    ):
        """
        Initialize the conflict resolver.

        Args:
            injury_risk: Base probability of injury in fights
            negotiation_success_rate: Base success rate for negotiations
            emit_events: Whether to emit events
        """
        self._injury_risk = injury_risk
        self._negotiation_rate = negotiation_success_rate
        self._emit_events = emit_events

        self._conflicts: Dict[str, Conflict] = {}
        self._outcomes: Dict[str, ConflictOutcome] = {}
        self._profiles: Dict[str, OrganismConflictProfile] = {}
        self._active_conflicts: Dict[str, Set[str]] = defaultdict(set)  # organism -> conflict IDs
        self._grudges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # =========================================================================
    # Profile Management
    # =========================================================================

    def register_organism(
        self,
        organism_id: str,
        fighting_ability: float = 0.5,
        resource_holding_potential: float = 0.5,
        aggression: float = 0.3
    ) -> OrganismConflictProfile:
        """Register an organism's conflict profile"""
        if organism_id in self._profiles:
            return self._profiles[organism_id]

        profile = OrganismConflictProfile(
            organism_id=organism_id,
            fighting_ability=fighting_ability,
            resource_holding_potential=resource_holding_potential,
            aggression=aggression
        )
        self._profiles[organism_id] = profile
        return profile

    def get_profile(self, organism_id: str) -> OrganismConflictProfile:
        """Get or create an organism's conflict profile"""
        if organism_id not in self._profiles:
            return self.register_organism(organism_id)
        return self._profiles[organism_id]

    # =========================================================================
    # Conflict Management
    # =========================================================================

    def register_conflict(
        self,
        party1_id: str,
        party2_id: str,
        conflict_type: ConflictType,
        resource_value: float = 100.0,
        intensity: float = 0.5
    ) -> Conflict:
        """
        Register a new conflict.

        Args:
            party1_id: First party
            party2_id: Second party
            conflict_type: Type of conflict
            resource_value: Value at stake
            intensity: Starting intensity

        Returns:
            The created Conflict
        """
        # Ensure profiles exist
        self.get_profile(party1_id)
        self.get_profile(party2_id)

        conflict = Conflict(
            party1_id=party1_id,
            party2_id=party2_id,
            conflict_type=conflict_type,
            resource_value=resource_value,
            intensity=intensity,
            status="active"
        )

        self._conflicts[conflict.id] = conflict
        self._active_conflicts[party1_id].add(conflict.id)
        self._active_conflicts[party2_id].add(conflict.id)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("conflict.registered", {
                "conflict_id": conflict.id,
                "party1": party1_id,
                "party2": party2_id,
                "type": conflict_type.value,
                "value": resource_value
            })

        return conflict

    def get_conflict(self, conflict_id: str) -> Optional[Conflict]:
        """Get a conflict by ID"""
        return self._conflicts.get(conflict_id)

    def get_active_conflicts(self, organism_id: str) -> List[Conflict]:
        """Get all active conflicts for an organism"""
        conflict_ids = self._active_conflicts.get(organism_id, set())
        return [
            self._conflicts[cid] for cid in conflict_ids
            if cid in self._conflicts and self._conflicts[cid].status == "active"
        ]

    def escalate(self, conflict_id: str, amount: float = 0.2) -> bool:
        """
        Escalate a conflict.

        Returns:
            True if escalation occurred
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict or conflict.status != "active":
            return False

        conflict.intensity = min(1.0, conflict.intensity + amount)
        conflict.escalations += 1

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("conflict.escalated", {
                "conflict_id": conflict_id,
                "new_intensity": conflict.intensity,
                "escalations": conflict.escalations
            })

        return True

    # =========================================================================
    # Resolution
    # =========================================================================

    def resolve(
        self,
        conflict_id: str,
        strategy: Strategy,
        party1_strategy: Optional[Strategy] = None,
        party2_strategy: Optional[Strategy] = None
    ) -> Optional[ConflictOutcome]:
        """
        Resolve a conflict using a strategy.

        Args:
            conflict_id: Conflict to resolve
            strategy: Primary resolution strategy
            party1_strategy: Party 1's strategy (optional)
            party2_strategy: Party 2's strategy (optional)

        Returns:
            ConflictOutcome if resolved
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict or conflict.status != "active":
            return None

        p1 = self.get_profile(conflict.party1_id)
        p2 = self.get_profile(conflict.party2_id)

        # Use same strategy if not specified
        if party1_strategy is None:
            party1_strategy = strategy
        if party2_strategy is None:
            party2_strategy = strategy

        # Resolve based on strategy combination
        outcome = self._resolve_strategy(conflict, p1, p2, party1_strategy, party2_strategy)

        # Update conflict
        conflict.status = "resolved"

        # Update profiles
        p1.update_from_outcome(outcome, is_party1=True)
        p2.update_from_outcome(outcome, is_party1=False)

        # Update grudges
        if outcome.loser_id:
            loser = outcome.loser_id
            winner = outcome.winner_id
            self._grudges[loser][winner] += conflict.intensity * 0.3

        # Clean up active conflicts
        self._active_conflicts[conflict.party1_id].discard(conflict_id)
        self._active_conflicts[conflict.party2_id].discard(conflict_id)

        self._outcomes[conflict_id] = outcome

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("conflict.resolved", {
                "conflict_id": conflict_id,
                "winner": outcome.winner_id,
                "outcome_type": outcome.outcome_type.value,
                "strategy": strategy.value,
                "total_cost": outcome.total_cost
            })

        return outcome

    def _resolve_strategy(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile,
        s1: Strategy,
        s2: Strategy
    ) -> ConflictOutcome:
        """Resolve based on both parties' strategies"""

        # Fight vs Fight - combat
        if s1 == Strategy.FIGHT and s2 == Strategy.FIGHT:
            return self._resolve_fight(conflict, p1, p2)

        # Fight vs Flee/Submit - clear winner
        if s1 == Strategy.FIGHT and s2 in (Strategy.FLEE, Strategy.SUBMIT):
            return self._resolve_dominance(conflict, p1, p2, winner_is_p1=True)
        if s2 == Strategy.FIGHT and s1 in (Strategy.FLEE, Strategy.SUBMIT):
            return self._resolve_dominance(conflict, p1, p2, winner_is_p1=False)

        # Both flee or submit - status quo
        if s1 in (Strategy.FLEE, Strategy.SUBMIT) and s2 in (Strategy.FLEE, Strategy.SUBMIT):
            return self._resolve_mutual_withdrawal(conflict, p1, p2)

        # Negotiate
        if s1 == Strategy.NEGOTIATE or s2 == Strategy.NEGOTIATE:
            return self._resolve_negotiation(conflict, p1, p2)

        # Share
        if s1 == Strategy.SHARE or s2 == Strategy.SHARE:
            return self._resolve_sharing(conflict, p1, p2)

        # Display
        if s1 == Strategy.DISPLAY or s2 == Strategy.DISPLAY:
            return self._resolve_display(conflict, p1, p2)

        # Default: negotiate
        return self._resolve_negotiation(conflict, p1, p2)

    def _resolve_fight(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile
    ) -> ConflictOutcome:
        """Resolve through physical combat"""
        # Calculate win probability
        ability_diff = p1.fighting_ability - p2.fighting_ability
        p1_win_prob = 0.5 + ability_diff * 0.5

        # Random outcome
        p1_wins = random.random() < p1_win_prob

        # Calculate costs (injuries)
        base_cost = conflict.intensity * conflict.resource_value * 0.2
        injury_modifier = self._injury_risk * (1 + conflict.intensity)

        p1_cost = base_cost * (1 - p1.fighting_ability + random.uniform(0, 0.2)) * injury_modifier
        p2_cost = base_cost * (1 - p2.fighting_ability + random.uniform(0, 0.2)) * injury_modifier

        # Winner takes all
        if p1_wins:
            return ConflictOutcome(
                conflict_id=conflict.id,
                winner_id=p1.organism_id,
                loser_id=p2.organism_id,
                outcome_type=OutcomeType.WIN,
                strategy_used=Strategy.FIGHT,
                party1_cost=p1_cost,
                party2_cost=p2_cost,
                party1_gain=conflict.resource_value,
                party2_gain=0,
                resource_division={p1.organism_id: conflict.resource_value}
            )
        else:
            return ConflictOutcome(
                conflict_id=conflict.id,
                winner_id=p2.organism_id,
                loser_id=p1.organism_id,
                outcome_type=OutcomeType.WIN,
                strategy_used=Strategy.FIGHT,
                party1_cost=p1_cost,
                party2_cost=p2_cost,
                party1_gain=0,
                party2_gain=conflict.resource_value,
                resource_division={p2.organism_id: conflict.resource_value}
            )

    def _resolve_dominance(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile,
        winner_is_p1: bool
    ) -> ConflictOutcome:
        """Resolve with clear dominance/submission"""
        if winner_is_p1:
            return ConflictOutcome(
                conflict_id=conflict.id,
                winner_id=p1.organism_id,
                loser_id=p2.organism_id,
                outcome_type=OutcomeType.WIN,
                strategy_used=Strategy.SUBMIT,
                party1_cost=0,
                party2_cost=conflict.intensity * 0.1,  # Social cost of submission
                party1_gain=conflict.resource_value,
                party2_gain=0,
                resource_division={p1.organism_id: conflict.resource_value}
            )
        else:
            return ConflictOutcome(
                conflict_id=conflict.id,
                winner_id=p2.organism_id,
                loser_id=p1.organism_id,
                outcome_type=OutcomeType.WIN,
                strategy_used=Strategy.SUBMIT,
                party1_cost=conflict.intensity * 0.1,
                party2_cost=0,
                party1_gain=0,
                party2_gain=conflict.resource_value,
                resource_division={p2.organism_id: conflict.resource_value}
            )

    def _resolve_mutual_withdrawal(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile
    ) -> ConflictOutcome:
        """Both parties withdraw"""
        return ConflictOutcome(
            conflict_id=conflict.id,
            outcome_type=OutcomeType.WITHDRAWAL,
            strategy_used=Strategy.FLEE,
            party1_cost=0,
            party2_cost=0,
            party1_gain=0,
            party2_gain=0,
            resource_division={}  # Resource goes unclaimed
        )

    def _resolve_negotiation(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile
    ) -> ConflictOutcome:
        """Resolve through negotiation"""
        success = random.random() < self._negotiation_rate

        if success:
            # Split based on relative power
            total_power = (
                p1.fighting_ability + p1.resource_holding_potential +
                p2.fighting_ability + p2.resource_holding_potential
            )
            p1_share = (p1.fighting_ability + p1.resource_holding_potential) / total_power
            p2_share = 1 - p1_share

            p1_gain = conflict.resource_value * p1_share
            p2_gain = conflict.resource_value * p2_share

            return ConflictOutcome(
                conflict_id=conflict.id,
                outcome_type=OutcomeType.COMPROMISE,
                strategy_used=Strategy.NEGOTIATE,
                party1_cost=conflict.intensity * 0.05,  # Time cost
                party2_cost=conflict.intensity * 0.05,
                party1_gain=p1_gain,
                party2_gain=p2_gain,
                resource_division={
                    p1.organism_id: p1_gain,
                    p2.organism_id: p2_gain
                }
            )
        else:
            # Negotiation failed, escalate
            self.escalate(conflict.id, 0.3)
            return self._resolve_fight(conflict, p1, p2)

    def _resolve_sharing(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile
    ) -> ConflictOutcome:
        """Equal split of resources"""
        half = conflict.resource_value / 2
        return ConflictOutcome(
            conflict_id=conflict.id,
            outcome_type=OutcomeType.COMPROMISE,
            strategy_used=Strategy.SHARE,
            party1_cost=0,
            party2_cost=0,
            party1_gain=half,
            party2_gain=half,
            resource_division={
                p1.organism_id: half,
                p2.organism_id: half
            }
        )

    def _resolve_display(
        self,
        conflict: Conflict,
        p1: OrganismConflictProfile,
        p2: OrganismConflictProfile
    ) -> ConflictOutcome:
        """Resolve through threat display"""
        # Winner based on display effectiveness (fighting ability + aggression)
        p1_display = p1.fighting_ability * 0.7 + p1.aggression * 0.3
        p2_display = p2.fighting_ability * 0.7 + p2.aggression * 0.3

        # If close, may need to fight
        if abs(p1_display - p2_display) < 0.1:
            # Escalate to fight with probability
            if random.random() < conflict.intensity:
                return self._resolve_fight(conflict, p1, p2)

        if p1_display > p2_display:
            return ConflictOutcome(
                conflict_id=conflict.id,
                winner_id=p1.organism_id,
                loser_id=p2.organism_id,
                outcome_type=OutcomeType.WIN,
                strategy_used=Strategy.DISPLAY,
                party1_cost=0.02,  # Energy for display
                party2_cost=0.02,
                party1_gain=conflict.resource_value,
                party2_gain=0,
                resource_division={p1.organism_id: conflict.resource_value}
            )
        else:
            return ConflictOutcome(
                conflict_id=conflict.id,
                winner_id=p2.organism_id,
                loser_id=p1.organism_id,
                outcome_type=OutcomeType.WIN,
                strategy_used=Strategy.DISPLAY,
                party1_cost=0.02,
                party2_cost=0.02,
                party1_gain=0,
                party2_gain=conflict.resource_value,
                resource_division={p2.organism_id: conflict.resource_value}
            )

    # =========================================================================
    # Strategy Recommendation
    # =========================================================================

    def recommend_strategy(
        self,
        conflict_id: str,
        organism_id: str
    ) -> Strategy:
        """
        Recommend optimal strategy for an organism.

        Uses game-theoretic analysis considering:
        - Relative fighting ability
        - Resource value vs injury cost
        - Past conflict history
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return Strategy.FLEE

        profile = self.get_profile(organism_id)

        # Determine opponent
        opponent_id = (
            conflict.party2_id if conflict.party1_id == organism_id
            else conflict.party1_id
        )
        opponent = self.get_profile(opponent_id)

        # Calculate factors
        ability_ratio = profile.fighting_ability / max(0.01, opponent.fighting_ability)
        value_cost_ratio = conflict.resource_value / (conflict.intensity * 50 + 1)
        grudge = self._grudges[organism_id][opponent_id]

        # Decision logic
        if ability_ratio > 1.5 and profile.aggression > 0.5:
            return Strategy.DISPLAY
        elif ability_ratio > 1.2:
            return Strategy.FIGHT if value_cost_ratio > 1 else Strategy.NEGOTIATE
        elif ability_ratio > 0.8:
            return Strategy.NEGOTIATE
        elif ability_ratio > 0.5:
            if grudge > 0.5:
                return Strategy.FIGHT  # Revenge despite odds
            return Strategy.SUBMIT if profile.submission_threshold > 0.4 else Strategy.NEGOTIATE
        else:
            return Strategy.FLEE

    # =========================================================================
    # Reconciliation
    # =========================================================================

    def reconcile(
        self,
        organism1_id: str,
        organism2_id: str,
        success_rate: float = 0.7
    ) -> bool:
        """
        Attempt reconciliation between two organisms.

        Returns:
            True if reconciliation was successful
        """
        if random.random() > success_rate:
            return False

        # Reduce grudges
        self._grudges[organism1_id][organism2_id] *= 0.5
        self._grudges[organism2_id][organism1_id] *= 0.5

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("conflict.reconciliation", {
                "organism1": organism1_id,
                "organism2": organism2_id
            })

        return True

    def get_grudge(self, organism1_id: str, organism2_id: str) -> float:
        """Get grudge level between two organisms"""
        return self._grudges[organism1_id][organism2_id]

    # =========================================================================
    # Mediation
    # =========================================================================

    def request_mediation(
        self,
        conflict_id: str,
        mediator_id: str
    ) -> bool:
        """
        Request a third party to mediate conflict.

        Returns:
            True if mediation was accepted
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict or conflict.status != "active":
            return False

        # Mediator should not be a party
        if mediator_id in (conflict.party1_id, conflict.party2_id):
            return False

        conflict.mediator_id = mediator_id

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("conflict.mediation_requested", {
                "conflict_id": conflict_id,
                "mediator_id": mediator_id
            })

        return True

    def mediated_resolution(
        self,
        conflict_id: str,
        division: Dict[str, float]
    ) -> Optional[ConflictOutcome]:
        """
        Resolve conflict with mediator-determined division.

        Args:
            conflict_id: Conflict to resolve
            division: Mediator's proposed resource division

        Returns:
            ConflictOutcome if accepted
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict or not conflict.mediator_id:
            return None

        # Both parties must accept (simplified - auto-accept for now)
        conflict.status = "resolved"

        p1_gain = division.get(conflict.party1_id, 0)
        p2_gain = division.get(conflict.party2_id, 0)

        outcome = ConflictOutcome(
            conflict_id=conflict_id,
            outcome_type=OutcomeType.COMPROMISE,
            strategy_used=Strategy.APPEAL,
            party1_cost=0,
            party2_cost=0,
            party1_gain=p1_gain,
            party2_gain=p2_gain,
            resource_division=division
        )

        self._outcomes[conflict_id] = outcome
        self._active_conflicts[conflict.party1_id].discard(conflict_id)
        self._active_conflicts[conflict.party2_id].discard(conflict_id)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("conflict.mediated_resolution", {
                "conflict_id": conflict_id,
                "mediator_id": conflict.mediator_id,
                "division": division
            })

        return outcome

    # =========================================================================
    # Queries
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get conflict statistics"""
        total_conflicts = len(self._conflicts)
        active = sum(1 for c in self._conflicts.values() if c.status == "active")
        resolved = sum(1 for c in self._conflicts.values() if c.status == "resolved")

        outcomes_by_type = defaultdict(int)
        total_cost = 0.0
        for outcome in self._outcomes.values():
            outcomes_by_type[outcome.outcome_type.value] += 1
            total_cost += outcome.total_cost

        return {
            "total_conflicts": total_conflicts,
            "active_conflicts": active,
            "resolved_conflicts": resolved,
            "outcomes_by_type": dict(outcomes_by_type),
            "total_cost": total_cost,
            "avg_cost_per_conflict": total_cost / resolved if resolved > 0 else 0
        }

    def remove_organism(self, organism_id: str) -> None:
        """Remove an organism from the conflict system"""
        # Cancel active conflicts
        for conflict_id in list(self._active_conflicts.get(organism_id, set())):
            conflict = self._conflicts.get(conflict_id)
            if conflict:
                conflict.status = "cancelled"

        self._active_conflicts.pop(organism_id, None)
        self._profiles.pop(organism_id, None)
        self._grudges.pop(organism_id, None)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conflict resolver"""
        conflicts = [
            {
                "id": c.id,
                "party1_id": c.party1_id,
                "party2_id": c.party2_id,
                "conflict_type": c.conflict_type.value,
                "resource_value": c.resource_value,
                "intensity": c.intensity,
                "status": c.status
            }
            for c in self._conflicts.values()
        ]

        profiles = [
            {
                "organism_id": p.organism_id,
                "fighting_ability": p.fighting_ability,
                "aggression": p.aggression,
                "wins": p.wins,
                "losses": p.losses
            }
            for p in self._profiles.values()
        ]

        return {
            "conflicts": conflicts,
            "profiles": profiles
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'ConflictResolver':
        """Deserialize conflict resolver"""
        resolver = cls(emit_events=emit_events)

        for c_data in data.get("conflicts", []):
            conflict = Conflict(
                id=c_data["id"],
                party1_id=c_data["party1_id"],
                party2_id=c_data["party2_id"],
                conflict_type=ConflictType(c_data["conflict_type"]),
                resource_value=c_data.get("resource_value", 100),
                intensity=c_data.get("intensity", 0.5),
                status=c_data.get("status", "pending")
            )
            resolver._conflicts[conflict.id] = conflict
            if conflict.status == "active":
                resolver._active_conflicts[conflict.party1_id].add(conflict.id)
                resolver._active_conflicts[conflict.party2_id].add(conflict.id)

        for p_data in data.get("profiles", []):
            profile = OrganismConflictProfile(
                organism_id=p_data["organism_id"],
                fighting_ability=p_data.get("fighting_ability", 0.5),
                aggression=p_data.get("aggression", 0.3),
                wins=p_data.get("wins", 0),
                losses=p_data.get("losses", 0)
            )
            resolver._profiles[profile.organism_id] = profile

        return resolver
