"""
Dominance Hierarchy Modeling for NPCPU

Implements social dominance structures with:
- Elo-based ranking system
- Dominance contests and outcomes
- Hierarchy stability tracking
- Linear and despotic hierarchies
- Rank-based resource access

Example:
    from social.hierarchy import HierarchyManager, DominanceContest

    # Create hierarchy manager
    manager = HierarchyManager()

    # Add organisms
    manager.add_organism("alpha", initial_rank=1.0)
    manager.add_organism("beta", initial_rank=0.5)

    # Record contest
    manager.record_contest("beta", "alpha", winner="beta")

    # Get rankings
    rankings = manager.get_rankings()
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
import math
import uuid

from core.events import get_event_bus


class HierarchyType(Enum):
    """Types of social hierarchies"""
    LINEAR = "linear"           # Strict A > B > C
    DESPOTIC = "despotic"       # One dominant, others equal
    EGALITARIAN = "egalitarian" # Minimal hierarchy
    NETWORK = "network"         # Complex, context-dependent


class ContestType(Enum):
    """Types of dominance contests"""
    DISPLAY = "display"         # Threat display
    PHYSICAL = "physical"       # Physical confrontation
    RESOURCE = "resource"       # Competition for resource
    SUBMISSION = "submission"   # Voluntary submission
    ALLIANCE = "alliance"       # Coalition-backed challenge


@dataclass
class DominanceScore:
    """
    Dominance score for an organism.

    Uses Elo-like rating system adapted for biological hierarchies.
    """
    organism_id: str
    rating: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    contests: int = 0
    last_contest: Optional[datetime] = None
    volatility: float = 350.0  # How much rating can change
    confidence: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.5

    def record_outcome(self, won: bool, drawn: bool = False) -> None:
        self.contests += 1
        if drawn:
            self.draws += 1
        elif won:
            self.wins += 1
        else:
            self.losses += 1
        self.last_contest = datetime.now()
        self.confidence = min(1.0, self.contests / 20)


@dataclass
class DominanceContest:
    """
    Record of a dominance contest between two organisms.

    Attributes:
        contestant1: First contestant
        contestant2: Second contestant
        winner_id: Who won (None for draw)
        contest_type: Type of contest
        intensity: How intense the contest was
        timestamp: When it occurred
    """
    contestant1: str
    contestant2: str
    winner_id: Optional[str] = None
    contest_type: ContestType = ContestType.DISPLAY
    intensity: float = 0.5
    injuries: bool = False
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def loser_id(self) -> Optional[str]:
        if self.winner_id is None:
            return None
        return self.contestant2 if self.winner_id == self.contestant1 else self.contestant1

    @property
    def is_draw(self) -> bool:
        return self.winner_id is None


class HierarchyManager:
    """
    Manages social dominance hierarchies.

    Provides:
    - Elo-based ranking system
    - Contest recording and outcome tracking
    - Hierarchy structure analysis
    - Rank-based benefit calculations
    - Stability monitoring

    Example:
        manager = HierarchyManager()
        manager.add_organism("alpha")
        manager.add_organism("beta")
        manager.record_contest("alpha", "beta", winner="alpha")
        rank = manager.get_rank("alpha")
    """

    def __init__(
        self,
        hierarchy_type: HierarchyType = HierarchyType.LINEAR,
        k_factor: float = 32.0,
        decay_rate: float = 0.01,
        emit_events: bool = True
    ):
        """
        Initialize the hierarchy manager.

        Args:
            hierarchy_type: Type of hierarchy to model
            k_factor: Elo K-factor (higher = more volatile)
            decay_rate: Rating decay rate per day
            emit_events: Whether to emit events
        """
        self._hierarchy_type = hierarchy_type
        self._k_factor = k_factor
        self._decay_rate = decay_rate
        self._emit_events = emit_events

        self._scores: Dict[str, DominanceScore] = {}
        self._contests: List[DominanceContest] = []
        self._dominance_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # =========================================================================
    # Organism Management
    # =========================================================================

    def add_organism(
        self,
        organism_id: str,
        initial_rating: float = 1000.0
    ) -> DominanceScore:
        """
        Add an organism to the hierarchy.

        Args:
            organism_id: Organism identifier
            initial_rating: Starting Elo rating

        Returns:
            DominanceScore for the organism
        """
        if organism_id in self._scores:
            return self._scores[organism_id]

        score = DominanceScore(
            organism_id=organism_id,
            rating=initial_rating
        )
        self._scores[organism_id] = score

        return score

    def remove_organism(self, organism_id: str) -> bool:
        """Remove an organism from the hierarchy"""
        if organism_id not in self._scores:
            return False

        del self._scores[organism_id]

        # Clean up dominance matrix
        if organism_id in self._dominance_matrix:
            del self._dominance_matrix[organism_id]
        for other_id in self._dominance_matrix:
            if organism_id in self._dominance_matrix[other_id]:
                del self._dominance_matrix[other_id][organism_id]

        return True

    # =========================================================================
    # Contest Management
    # =========================================================================

    def record_contest(
        self,
        contestant1: str,
        contestant2: str,
        winner: Optional[str] = None,
        contest_type: ContestType = ContestType.DISPLAY,
        intensity: float = 0.5
    ) -> DominanceContest:
        """
        Record a dominance contest.

        Args:
            contestant1: First contestant ID
            contestant2: Second contestant ID
            winner: Winner ID (None for draw)
            contest_type: Type of contest
            intensity: Contest intensity (0.0 to 1.0)

        Returns:
            The recorded contest
        """
        # Ensure organisms exist
        if contestant1 not in self._scores:
            self.add_organism(contestant1)
        if contestant2 not in self._scores:
            self.add_organism(contestant2)

        contest = DominanceContest(
            contestant1=contestant1,
            contestant2=contestant2,
            winner_id=winner,
            contest_type=contest_type,
            intensity=intensity
        )

        self._contests.append(contest)

        # Update ratings
        self._update_ratings(contest)

        # Update dominance matrix
        if winner:
            self._dominance_matrix[winner][contest.loser_id] += 1

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("hierarchy.contest", {
                "contestant1": contestant1,
                "contestant2": contestant2,
                "winner": winner,
                "contest_type": contest_type.value
            })

        return contest

    def _update_ratings(self, contest: DominanceContest) -> None:
        """Update Elo ratings based on contest outcome"""
        score1 = self._scores[contest.contestant1]
        score2 = self._scores[contest.contestant2]

        # Calculate expected scores
        expected1 = self._expected_score(score1.rating, score2.rating)
        expected2 = 1.0 - expected1

        # Actual scores
        if contest.is_draw:
            actual1 = 0.5
            actual2 = 0.5
            score1.record_outcome(won=False, drawn=True)
            score2.record_outcome(won=False, drawn=True)
        elif contest.winner_id == contest.contestant1:
            actual1 = 1.0
            actual2 = 0.0
            score1.record_outcome(won=True)
            score2.record_outcome(won=False)
        else:
            actual1 = 0.0
            actual2 = 1.0
            score1.record_outcome(won=False)
            score2.record_outcome(won=True)

        # Intensity-adjusted K factor
        k_adjusted = self._k_factor * (0.5 + 0.5 * contest.intensity)

        # Update ratings
        delta1 = k_adjusted * (actual1 - expected1)
        delta2 = k_adjusted * (actual2 - expected2)

        score1.rating += delta1
        score2.rating += delta2

        # Reduce volatility with more contests
        score1.volatility = max(100, score1.volatility - 5)
        score2.volatility = max(100, score2.volatility - 5)

    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score using Elo formula"""
        return 1.0 / (1.0 + math.pow(10, (rating2 - rating1) / 400))

    # =========================================================================
    # Ranking Queries
    # =========================================================================

    def get_rank(self, organism_id: str) -> int:
        """
        Get the rank of an organism (1 = highest).

        Returns:
            Rank position (1-indexed)
        """
        if organism_id not in self._scores:
            return -1

        rankings = self.get_rankings()
        for i, (oid, _) in enumerate(rankings):
            if oid == organism_id:
                return i + 1
        return -1

    def get_rating(self, organism_id: str) -> float:
        """Get the Elo rating of an organism"""
        if organism_id not in self._scores:
            return 0.0
        return self._scores[organism_id].rating

    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get all organisms ranked by dominance.

        Returns:
            List of (organism_id, rating) tuples, highest first
        """
        rankings = [
            (oid, score.rating)
            for oid, score in self._scores.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_dominance_score(self, organism_id: str) -> Optional[DominanceScore]:
        """Get full dominance score for an organism"""
        return self._scores.get(organism_id)

    def is_dominant(self, org1_id: str, org2_id: str) -> bool:
        """Check if org1 is dominant over org2"""
        if org1_id not in self._scores or org2_id not in self._scores:
            return False
        return self._scores[org1_id].rating > self._scores[org2_id].rating

    def dominance_probability(self, org1_id: str, org2_id: str) -> float:
        """
        Predict probability that org1 would beat org2.

        Returns:
            Probability from 0.0 to 1.0
        """
        if org1_id not in self._scores or org2_id not in self._scores:
            return 0.5

        return self._expected_score(
            self._scores[org1_id].rating,
            self._scores[org2_id].rating
        )

    def get_direct_wins(self, org1_id: str, org2_id: str) -> int:
        """Get number of times org1 directly beat org2"""
        return self._dominance_matrix[org1_id][org2_id]

    # =========================================================================
    # Hierarchy Analysis
    # =========================================================================

    def calculate_linearity(self) -> float:
        """
        Calculate hierarchy linearity (Landau's h).

        Returns:
            Linearity score from 0.0 (no hierarchy) to 1.0 (perfect linear)
        """
        n = len(self._scores)
        if n < 2:
            return 0.0

        organisms = list(self._scores.keys())
        h_sum = 0.0

        for org_id in organisms:
            wins = sum(self._dominance_matrix[org_id].values())
            h_sum += (wins - (n - 1) / 2) ** 2

        h = (12 * h_sum) / (n ** 3 - n) if n > 1 else 0
        return min(1.0, h)

    def calculate_steepness(self) -> float:
        """
        Calculate hierarchy steepness.

        Returns:
            Steepness from 0.0 (flat) to 1.0 (steep)
        """
        rankings = self.get_rankings()
        if len(rankings) < 2:
            return 0.0

        ratings = [r for _, r in rankings]
        avg_rating = sum(ratings) / len(ratings)

        if avg_rating == 0:
            return 0.0

        # Calculate coefficient of variation
        variance = sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)
        std_dev = math.sqrt(variance)
        cv = std_dev / avg_rating

        # Normalize to 0-1
        return min(1.0, cv / 0.5)

    def calculate_stability(self) -> float:
        """
        Calculate hierarchy stability based on recent rank changes.

        Returns:
            Stability from 0.0 (unstable) to 1.0 (very stable)
        """
        if len(self._contests) < 5:
            return 0.5  # Not enough data

        # Look at last 20 contests
        recent = self._contests[-20:]

        # Count upsets (lower-ranked beating higher-ranked)
        upsets = 0
        for contest in recent:
            if contest.winner_id:
                winner_rank = self.get_rank(contest.winner_id)
                loser_rank = self.get_rank(contest.loser_id)
                if winner_rank > loser_rank:  # Higher rank number = lower status
                    upsets += 1

        upset_rate = upsets / len(recent)
        return 1.0 - upset_rate

    def get_rank_correlation(self) -> float:
        """
        Calculate Spearman correlation between rank and win rate.

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(self._scores) < 2:
            return 0.0

        rankings = self.get_rankings()
        n = len(rankings)

        # Calculate rank differences
        rank_diffs_sq = 0.0
        for i, (org_id, _) in enumerate(rankings):
            score = self._scores[org_id]
            win_rate_rank = i + 1  # Already sorted by rating
            rank_diffs_sq += (i + 1 - win_rate_rank) ** 2

        # Spearman formula
        rho = 1 - (6 * rank_diffs_sq) / (n * (n ** 2 - 1))
        return rho

    # =========================================================================
    # Rank Benefits
    # =========================================================================

    def calculate_rank_benefit(
        self,
        organism_id: str,
        resource_amount: float = 1.0,
        despotism: float = 0.5
    ) -> float:
        """
        Calculate resource share based on rank.

        Args:
            organism_id: Target organism
            resource_amount: Total resource available
            despotism: How unequal distribution is (0 = equal, 1 = winner takes all)

        Returns:
            Resource share for this organism
        """
        rank = self.get_rank(organism_id)
        if rank < 0:
            return 0.0

        n = len(self._scores)
        if n == 0:
            return 0.0

        # Calculate share based on despotism level
        if despotism == 0:
            # Equal share
            return resource_amount / n
        else:
            # Rank-weighted share
            # Weight = (n - rank + 1)^despotism / sum of all weights
            weights = [(n - i) ** despotism for i in range(n)]
            total_weight = sum(weights)
            organism_weight = (n - rank + 1) ** despotism
            return resource_amount * organism_weight / total_weight

    # =========================================================================
    # Maintenance
    # =========================================================================

    def apply_decay(self, days: float = 1.0) -> None:
        """Apply rating decay over time"""
        for score in self._scores.values():
            decay = self._decay_rate * days
            # Decay toward average
            avg_rating = 1000.0
            score.rating = score.rating + (avg_rating - score.rating) * decay

    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchy statistics"""
        if not self._scores:
            return {
                "num_organisms": 0,
                "total_contests": 0,
                "linearity": 0,
                "steepness": 0,
                "stability": 0
            }

        ratings = [s.rating for s in self._scores.values()]

        return {
            "num_organisms": len(self._scores),
            "total_contests": len(self._contests),
            "avg_rating": sum(ratings) / len(ratings),
            "max_rating": max(ratings),
            "min_rating": min(ratings),
            "linearity": self.calculate_linearity(),
            "steepness": self.calculate_steepness(),
            "stability": self.calculate_stability()
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchy manager"""
        scores = [
            {
                "organism_id": s.organism_id,
                "rating": s.rating,
                "wins": s.wins,
                "losses": s.losses,
                "draws": s.draws,
                "volatility": s.volatility
            }
            for s in self._scores.values()
        ]

        contests = [
            {
                "contestant1": c.contestant1,
                "contestant2": c.contestant2,
                "winner_id": c.winner_id,
                "contest_type": c.contest_type.value,
                "intensity": c.intensity,
                "timestamp": c.timestamp.isoformat()
            }
            for c in self._contests[-100:]  # Keep last 100
        ]

        return {
            "hierarchy_type": self._hierarchy_type.value,
            "scores": scores,
            "contests": contests
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'HierarchyManager':
        """Deserialize hierarchy manager"""
        manager = cls(
            hierarchy_type=HierarchyType(data.get("hierarchy_type", "linear")),
            emit_events=emit_events
        )

        for s_data in data.get("scores", []):
            score = DominanceScore(
                organism_id=s_data["organism_id"],
                rating=s_data["rating"],
                wins=s_data.get("wins", 0),
                losses=s_data.get("losses", 0),
                draws=s_data.get("draws", 0),
                volatility=s_data.get("volatility", 350)
            )
            manager._scores[score.organism_id] = score

        for c_data in data.get("contests", []):
            contest = DominanceContest(
                contestant1=c_data["contestant1"],
                contestant2=c_data["contestant2"],
                winner_id=c_data.get("winner_id"),
                contest_type=ContestType(c_data["contest_type"]),
                intensity=c_data.get("intensity", 0.5),
                timestamp=datetime.fromisoformat(c_data["timestamp"])
            )
            manager._contests.append(contest)
            if contest.winner_id:
                manager._dominance_matrix[contest.winner_id][contest.loser_id] += 1

        return manager
