"""
Multi-dimensional Reputation System for NPCPU

Provides sophisticated reputation tracking with:
- Multiple reputation dimensions (trustworthiness, competence, benevolence)
- Temporal decay and recovery
- Gossip propagation
- Reputation inheritance from groups
- Context-dependent reputation

Example:
    from social.reputation import ReputationEngine, ReputationDimension

    # Create reputation engine
    engine = ReputationEngine()

    # Record observations
    engine.record_observation("alice", "bob", ReputationDimension.TRUSTWORTHY, 0.9)
    engine.record_observation("charlie", "bob", ReputationDimension.COMPETENT, 0.7)

    # Get reputation
    rep = engine.get_reputation("bob")
    print(f"Overall: {rep.overall_score}")
    print(f"Trustworthiness: {rep.dimensions[ReputationDimension.TRUSTWORTHY]}")

    # Propagate through gossip
    engine.propagate_gossip("alice", "bob", decay=0.9)
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict
import math
import uuid

from core.events import get_event_bus


class ReputationDimension(Enum):
    """Dimensions of reputation that can be tracked"""
    TRUSTWORTHY = "trustworthy"       # Keeps promises, honest
    COMPETENT = "competent"           # Skilled, capable
    BENEVOLENT = "benevolent"         # Kind, helpful
    AGGRESSIVE = "aggressive"         # Prone to conflict
    COOPERATIVE = "cooperative"       # Works well with others
    RELIABLE = "reliable"             # Consistent, predictable
    GENEROUS = "generous"             # Shares resources
    DOMINANT = "dominant"             # Assertive, leader-like
    SOCIAL = "social"                 # Good at relationships
    RESOURCEFUL = "resourceful"       # Good at acquiring resources


@dataclass
class ReputationObservation:
    """
    A single observation contributing to reputation.

    Attributes:
        observer_id: Who made the observation
        target_id: Who was observed
        dimension: Which reputation dimension
        value: Observed value (-1.0 to 1.0)
        weight: How much weight to give this observation
        context: Optional context tag
        timestamp: When observation was made
    """
    observer_id: str
    target_id: str
    dimension: ReputationDimension
    value: float
    weight: float = 1.0
    context: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        self.value = max(-1.0, min(1.0, self.value))
        self.weight = max(0.0, self.weight)


@dataclass
class ReputationScore:
    """
    Aggregated reputation score for an organism.

    Attributes:
        target_id: Whose reputation this is
        dimensions: Score for each dimension
        observation_count: Total observations
        last_update: Last update time
        confidence: How confident we are in this score
        overall_score: Weighted average across dimensions
    """
    target_id: str
    dimensions: Dict[ReputationDimension, float] = field(default_factory=dict)
    dimension_weights: Dict[ReputationDimension, float] = field(default_factory=dict)
    observation_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate weighted average reputation score"""
        if not self.dimensions:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for dim, score in self.dimensions.items():
            weight = self.dimension_weights.get(dim, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_dimension(self, dimension: ReputationDimension) -> float:
        """Get score for a specific dimension"""
        return self.dimensions.get(dimension, 0.0)


@dataclass
class GossipMessage:
    """
    A gossip message about reputation.

    Attributes:
        source_id: Who is spreading the gossip
        about_id: Who the gossip is about
        dimension: Which reputation dimension
        value: The gossip value
        original_observer: Original source of information
        hop_count: How many hops this has traveled
        credibility: How credible this gossip is
    """
    source_id: str
    about_id: str
    dimension: ReputationDimension
    value: float
    original_observer: str
    hop_count: int = 0
    credibility: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class ReputationEngine:
    """
    Multi-dimensional reputation tracking and gossip system.

    Provides:
    - Per-observer reputation views
    - Aggregated global reputation
    - Gossip-based information spread
    - Temporal decay
    - Context-dependent reputation

    Example:
        engine = ReputationEngine()

        # Record direct observations
        engine.record_observation("alice", "bob", ReputationDimension.TRUSTWORTHY, 0.8)

        # Get bob's reputation from alice's perspective
        score = engine.get_reputation("bob", observer_id="alice")

        # Get bob's global reputation
        global_score = engine.get_reputation("bob")

        # Propagate information through network
        engine.propagate_gossip("alice", "bob", max_hops=2)
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        gossip_decay: float = 0.8,
        emit_events: bool = True
    ):
        """
        Initialize the reputation engine.

        Args:
            decay_rate: Rate at which observations decay per day
            gossip_decay: How much credibility degrades per gossip hop
            emit_events: Whether to emit events
        """
        # Per-observer observations: observer -> target -> [observations]
        self._observations: Dict[str, Dict[str, List[ReputationObservation]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Cached scores: target -> score
        self._global_scores: Dict[str, ReputationScore] = {}

        # Per-observer scores: observer -> target -> score
        self._observer_scores: Dict[str, Dict[str, ReputationScore]] = defaultdict(dict)

        # Gossip messages in transit
        self._pending_gossip: List[GossipMessage] = []

        # Observer credibility (how much to trust their observations)
        self._observer_credibility: Dict[str, float] = defaultdict(lambda: 1.0)

        # Dimension weights for overall score calculation
        self._dimension_weights: Dict[ReputationDimension, float] = {
            ReputationDimension.TRUSTWORTHY: 1.5,
            ReputationDimension.COMPETENT: 1.2,
            ReputationDimension.BENEVOLENT: 1.0,
            ReputationDimension.COOPERATIVE: 1.1,
            ReputationDimension.RELIABLE: 1.0,
            ReputationDimension.GENEROUS: 0.8,
            ReputationDimension.AGGRESSIVE: -0.5,
            ReputationDimension.DOMINANT: 0.3,
            ReputationDimension.SOCIAL: 0.7,
            ReputationDimension.RESOURCEFUL: 0.9
        }

        self._decay_rate = decay_rate
        self._gossip_decay = gossip_decay
        self._emit_events = emit_events

    # =========================================================================
    # Observation Recording
    # =========================================================================

    def record_observation(
        self,
        observer_id: str,
        target_id: str,
        dimension: ReputationDimension,
        value: float,
        weight: float = 1.0,
        context: str = ""
    ) -> ReputationObservation:
        """
        Record a reputation observation.

        Args:
            observer_id: Who is observing
            target_id: Who is being observed
            dimension: Which reputation dimension
            value: Observed value (-1.0 to 1.0)
            weight: Observation weight
            context: Optional context tag

        Returns:
            The created observation
        """
        observation = ReputationObservation(
            observer_id=observer_id,
            target_id=target_id,
            dimension=dimension,
            value=value,
            weight=weight,
            context=context
        )

        self._observations[observer_id][target_id].append(observation)
        self._invalidate_cache(target_id, observer_id)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("reputation.observation", {
                "observer_id": observer_id,
                "target_id": target_id,
                "dimension": dimension.value,
                "value": value
            })

        return observation

    def record_interaction(
        self,
        actor_id: str,
        target_id: str,
        positive: bool = True,
        dimensions: Optional[List[ReputationDimension]] = None,
        weight: float = 0.5
    ) -> None:
        """
        Record a reputation-affecting interaction.

        Args:
            actor_id: Who performed the action
            target_id: Who observed/was affected
            positive: Whether interaction was positive
            dimensions: Which dimensions to affect (default: trustworthy, cooperative)
            weight: Interaction weight
        """
        if dimensions is None:
            dimensions = [ReputationDimension.TRUSTWORTHY, ReputationDimension.COOPERATIVE]

        value = weight if positive else -weight

        for dimension in dimensions:
            self.record_observation(
                observer_id=target_id,
                target_id=actor_id,
                dimension=dimension,
                value=value,
                context="interaction"
            )

    # =========================================================================
    # Reputation Retrieval
    # =========================================================================

    def get_reputation(
        self,
        target_id: str,
        observer_id: Optional[str] = None,
        dimensions: Optional[List[ReputationDimension]] = None
    ) -> ReputationScore:
        """
        Get reputation score for a target.

        Args:
            target_id: Whose reputation to get
            observer_id: From whose perspective (None for global)
            dimensions: Which dimensions to include (None for all)

        Returns:
            ReputationScore for the target
        """
        if observer_id:
            # Observer-specific reputation
            return self._calculate_observer_score(observer_id, target_id, dimensions)
        else:
            # Global reputation
            return self._calculate_global_score(target_id, dimensions)

    def _calculate_observer_score(
        self,
        observer_id: str,
        target_id: str,
        dimensions: Optional[List[ReputationDimension]] = None
    ) -> ReputationScore:
        """Calculate reputation from a specific observer's perspective"""
        observations = self._observations[observer_id][target_id]

        if not observations:
            return ReputationScore(target_id=target_id)

        # Filter by dimensions if specified
        if dimensions:
            observations = [o for o in observations if o.dimension in dimensions]

        return self._aggregate_observations(target_id, observations)

    def _calculate_global_score(
        self,
        target_id: str,
        dimensions: Optional[List[ReputationDimension]] = None
    ) -> ReputationScore:
        """Calculate global reputation across all observers"""
        all_observations = []

        for observer_id, targets in self._observations.items():
            if target_id in targets:
                observer_cred = self._observer_credibility[observer_id]
                for obs in targets[target_id]:
                    if dimensions is None or obs.dimension in dimensions:
                        # Weight by observer credibility
                        weighted_obs = ReputationObservation(
                            observer_id=obs.observer_id,
                            target_id=obs.target_id,
                            dimension=obs.dimension,
                            value=obs.value,
                            weight=obs.weight * observer_cred,
                            context=obs.context,
                            timestamp=obs.timestamp
                        )
                        all_observations.append(weighted_obs)

        if not all_observations:
            return ReputationScore(target_id=target_id)

        return self._aggregate_observations(target_id, all_observations)

    def _aggregate_observations(
        self,
        target_id: str,
        observations: List[ReputationObservation]
    ) -> ReputationScore:
        """Aggregate observations into a reputation score"""
        dimension_values: Dict[ReputationDimension, List[Tuple[float, float]]] = defaultdict(list)

        now = datetime.now()

        for obs in observations:
            # Apply temporal decay
            age_days = (now - obs.timestamp).total_seconds() / 86400
            decay = math.exp(-self._decay_rate * age_days)
            effective_weight = obs.weight * decay

            if effective_weight > 0.01:  # Threshold
                dimension_values[obs.dimension].append((obs.value, effective_weight))

        # Calculate weighted average for each dimension
        dimensions: Dict[ReputationDimension, float] = {}
        total_weight = 0.0

        for dim, values in dimension_values.items():
            weighted_sum = sum(v * w for v, w in values)
            weight_sum = sum(w for _, w in values)
            if weight_sum > 0:
                dimensions[dim] = weighted_sum / weight_sum
                total_weight += weight_sum

        # Calculate confidence based on observation count and recency
        observation_count = len(observations)
        confidence = min(1.0, observation_count / 10)  # Saturates at 10 observations

        return ReputationScore(
            target_id=target_id,
            dimensions=dimensions,
            dimension_weights=dict(self._dimension_weights),
            observation_count=observation_count,
            confidence=confidence,
            last_update=now
        )

    def get_dimension_reputation(
        self,
        target_id: str,
        dimension: ReputationDimension,
        observer_id: Optional[str] = None
    ) -> float:
        """Get reputation for a specific dimension"""
        score = self.get_reputation(target_id, observer_id)
        return score.get_dimension(dimension)

    def compare_reputation(
        self,
        target1_id: str,
        target2_id: str,
        dimension: Optional[ReputationDimension] = None
    ) -> float:
        """
        Compare reputation of two organisms.

        Returns:
            Positive if target1 > target2, negative otherwise
        """
        score1 = self.get_reputation(target1_id)
        score2 = self.get_reputation(target2_id)

        if dimension:
            return score1.get_dimension(dimension) - score2.get_dimension(dimension)
        else:
            return score1.overall_score - score2.overall_score

    # =========================================================================
    # Gossip Propagation
    # =========================================================================

    def propagate_gossip(
        self,
        source_id: str,
        about_id: str,
        recipients: Optional[List[str]] = None,
        max_hops: int = 2,
        decay: Optional[float] = None
    ) -> List[str]:
        """
        Propagate reputation information through gossip.

        Args:
            source_id: Who is sharing the information
            about_id: Who the information is about
            recipients: Who to share with (None = all relationships)
            max_hops: Maximum propagation distance
            decay: Credibility decay per hop (None = use default)

        Returns:
            List of organisms who received the gossip
        """
        if decay is None:
            decay = self._gossip_decay

        # Get source's observations about the target
        observations = self._observations[source_id][about_id]
        if not observations:
            return []

        reached = []
        messages = []

        # Create gossip messages from observations
        for obs in observations:
            # Recent observations only
            age_days = (datetime.now() - obs.timestamp).total_seconds() / 86400
            if age_days > 7:  # Only share recent info
                continue

            messages.append(GossipMessage(
                source_id=source_id,
                about_id=about_id,
                dimension=obs.dimension,
                value=obs.value,
                original_observer=source_id,
                credibility=1.0
            ))

        # Propagate to recipients
        if recipients is None:
            # Would integrate with social network to get connections
            recipients = []

        for recipient_id in recipients:
            if recipient_id != source_id and recipient_id != about_id:
                for msg in messages:
                    self._receive_gossip(recipient_id, msg, decay)
                    reached.append(recipient_id)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("reputation.gossip", {
                "source_id": source_id,
                "about_id": about_id,
                "recipients": recipients,
                "message_count": len(messages)
            })

        return list(set(reached))

    def _receive_gossip(
        self,
        recipient_id: str,
        gossip: GossipMessage,
        decay: float
    ) -> None:
        """Process received gossip and update reputation knowledge"""
        # Apply credibility decay
        effective_credibility = gossip.credibility * decay

        if effective_credibility < 0.1:  # Too unreliable
            return

        # Record as indirect observation with reduced weight
        self.record_observation(
            observer_id=recipient_id,
            target_id=gossip.about_id,
            dimension=gossip.dimension,
            value=gossip.value,
            weight=effective_credibility,
            context=f"gossip_from_{gossip.source_id}"
        )

    # =========================================================================
    # Reputation Management
    # =========================================================================

    def set_observer_credibility(self, observer_id: str, credibility: float) -> None:
        """Set how much to trust an observer's observations"""
        self._observer_credibility[observer_id] = max(0.0, min(1.0, credibility))

    def adjust_dimension_weight(self, dimension: ReputationDimension, weight: float) -> None:
        """Adjust the weight of a reputation dimension"""
        self._dimension_weights[dimension] = weight

    def decay_all_observations(self) -> None:
        """Apply decay to all observations (call periodically)"""
        now = datetime.now()
        cutoff = now - timedelta(days=30)  # Remove observations older than 30 days

        for observer_id in self._observations:
            for target_id in self._observations[observer_id]:
                self._observations[observer_id][target_id] = [
                    obs for obs in self._observations[observer_id][target_id]
                    if obs.timestamp > cutoff
                ]

        # Invalidate all caches
        self._global_scores.clear()
        self._observer_scores.clear()

    def remove_organism(self, organism_id: str) -> None:
        """Remove all observations about/from an organism"""
        # Remove as observer
        if organism_id in self._observations:
            del self._observations[organism_id]

        # Remove observations about this organism
        for observer_id in self._observations:
            if organism_id in self._observations[observer_id]:
                del self._observations[observer_id][organism_id]

        # Clear caches
        self._global_scores.pop(organism_id, None)
        self._observer_scores.pop(organism_id, None)
        self._observer_credibility.pop(organism_id, None)

    def _invalidate_cache(self, target_id: str, observer_id: Optional[str] = None) -> None:
        """Invalidate cached scores"""
        self._global_scores.pop(target_id, None)
        if observer_id and observer_id in self._observer_scores:
            self._observer_scores[observer_id].pop(target_id, None)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_top_reputation(
        self,
        dimension: Optional[ReputationDimension] = None,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get organisms with highest reputation.

        Args:
            dimension: Specific dimension (None for overall)
            limit: Maximum results

        Returns:
            List of (organism_id, score) tuples
        """
        # Get all organisms with observations
        all_targets = set()
        for observer_data in self._observations.values():
            all_targets.update(observer_data.keys())

        scores = []
        for target_id in all_targets:
            score = self.get_reputation(target_id)
            if dimension:
                scores.append((target_id, score.get_dimension(dimension)))
            else:
                scores.append((target_id, score.overall_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def get_reputation_history(
        self,
        target_id: str,
        dimension: Optional[ReputationDimension] = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get reputation history over time.

        Returns:
            List of (timestamp, score) tuples
        """
        all_observations = []

        for observer_data in self._observations.values():
            if target_id in observer_data:
                observations = observer_data[target_id]
                if dimension:
                    observations = [o for o in observations if o.dimension == dimension]
                all_observations.extend(observations)

        # Sort by timestamp
        all_observations.sort(key=lambda o: o.timestamp)

        # Calculate running average
        history = []
        running_sum = 0.0
        count = 0

        for obs in all_observations:
            running_sum += obs.value * obs.weight
            count += obs.weight
            avg = running_sum / count if count > 0 else 0
            history.append((obs.timestamp, avg))

        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get reputation system statistics"""
        total_observations = sum(
            len(obs)
            for targets in self._observations.values()
            for obs in targets.values()
        )

        num_observers = len(self._observations)
        num_targets = len(set(
            target
            for targets in self._observations.values()
            for target in targets.keys()
        ))

        return {
            "total_observations": total_observations,
            "num_observers": num_observers,
            "num_targets": num_targets,
            "avg_observations_per_target": total_observations / num_targets if num_targets > 0 else 0
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reputation engine to dictionary"""
        observations = []
        for observer_id, targets in self._observations.items():
            for target_id, obs_list in targets.items():
                for obs in obs_list:
                    observations.append({
                        "observer_id": obs.observer_id,
                        "target_id": obs.target_id,
                        "dimension": obs.dimension.value,
                        "value": obs.value,
                        "weight": obs.weight,
                        "context": obs.context,
                        "timestamp": obs.timestamp.isoformat()
                    })

        return {
            "observations": observations,
            "observer_credibility": dict(self._observer_credibility),
            "dimension_weights": {
                dim.value: weight for dim, weight in self._dimension_weights.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'ReputationEngine':
        """Deserialize reputation engine from dictionary"""
        engine = cls(emit_events=emit_events)

        for obs_data in data.get("observations", []):
            obs = ReputationObservation(
                observer_id=obs_data["observer_id"],
                target_id=obs_data["target_id"],
                dimension=ReputationDimension(obs_data["dimension"]),
                value=obs_data["value"],
                weight=obs_data.get("weight", 1.0),
                context=obs_data.get("context", ""),
                timestamp=datetime.fromisoformat(obs_data["timestamp"])
            )
            engine._observations[obs.observer_id][obs.target_id].append(obs)

        for observer_id, cred in data.get("observer_credibility", {}).items():
            engine._observer_credibility[observer_id] = cred

        for dim_str, weight in data.get("dimension_weights", {}).items():
            engine._dimension_weights[ReputationDimension(dim_str)] = weight

        return engine
