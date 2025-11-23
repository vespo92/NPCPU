"""
HarmonizationAgent - Balances the Triple Bottom Line (Profit, People, Planet).

Responsibilities:
- Balance the classic sustainability Triple Bottom Line (3P):
  - PROFIT: Economic sustainability and resource efficiency
  - PEOPLE: Social well-being, relationships, and community health
  - PLANET: Environmental sustainability and ecosystem preservation
- Maintain balance across all three domain legs
- Resolve conflicts between domains
- Optimize for global harmony
- Mediate resource contention
- Orchestrate cooperative behaviors
- Ensure sustainable growth without sacrificing any pillar

The Triple Bottom Line (TBL) framework was coined by John Elkington in 1994
and represents a paradigm shift from pure profit-driven models to holistic
sustainability accounting.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import math

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
    HarmonyLevel,
)


# =============================================================================
# TRIPLE BOTTOM LINE (PROFIT, PEOPLE, PLANET) FRAMEWORK
# =============================================================================

class TBLPillar(Enum):
    """The three pillars of the Triple Bottom Line."""
    PROFIT = "profit"    # Economic sustainability
    PEOPLE = "people"    # Social sustainability
    PLANET = "planet"    # Environmental sustainability


class SustainabilityLevel(Enum):
    """Levels of sustainability across the TBL."""
    EXPLOITATIVE = "exploitative"     # < 0.2 - Actively harmful
    UNSUSTAINABLE = "unsustainable"   # 0.2-0.4 - Declining trajectory
    NEUTRAL = "neutral"               # 0.4-0.6 - Neither growing nor declining
    SUSTAINABLE = "sustainable"       # 0.6-0.8 - Positive trajectory
    REGENERATIVE = "regenerative"     # 0.8-0.95 - Actively improving
    THRIVING = "thriving"             # >= 0.95 - Optimal state


class TBLTradeoffType(Enum):
    """Types of tradeoffs between TBL pillars."""
    PROFIT_VS_PEOPLE = auto()      # Economic efficiency vs social welfare
    PROFIT_VS_PLANET = auto()      # Growth vs environmental impact
    PEOPLE_VS_PLANET = auto()      # Social needs vs ecological limits
    WIN_WIN_WIN = auto()           # All three pillars aligned
    ZERO_SUM = auto()              # Gains in one mean losses in others


@dataclass
class PillarMetrics:
    """Metrics for a single TBL pillar."""
    pillar: TBLPillar
    score: float = 0.5           # Overall pillar score (0-1)
    trend: float = 0.0           # Rate of change (-1 to 1)
    stability: float = 0.5       # How stable the score is
    efficiency: float = 0.5      # Resource efficiency
    resilience: float = 0.5      # Ability to recover from shocks
    growth_potential: float = 0.5  # Room for improvement

    # Pillar-specific metrics
    specific_metrics: Dict[str, float] = field(default_factory=dict)

    def weighted_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite score."""
        default_weights = {
            "score": 0.35,
            "trend": 0.15,
            "stability": 0.15,
            "efficiency": 0.15,
            "resilience": 0.10,
            "growth_potential": 0.10
        }
        w = weights or default_weights

        # Normalize trend to 0-1 scale
        normalized_trend = (self.trend + 1) / 2

        return (
            self.score * w["score"] +
            normalized_trend * w["trend"] +
            self.stability * w["stability"] +
            self.efficiency * w["efficiency"] +
            self.resilience * w["resilience"] +
            self.growth_potential * w["growth_potential"]
        )


@dataclass
class TBLState:
    """Complete Triple Bottom Line state."""
    profit: PillarMetrics = field(default_factory=lambda: PillarMetrics(TBLPillar.PROFIT))
    people: PillarMetrics = field(default_factory=lambda: PillarMetrics(TBLPillar.PEOPLE))
    planet: PillarMetrics = field(default_factory=lambda: PillarMetrics(TBLPillar.PLANET))

    # Cross-pillar metrics
    balance_score: float = 0.5      # How balanced the three pillars are
    synergy_score: float = 0.0      # Positive interactions between pillars
    conflict_score: float = 0.0     # Negative interactions
    sustainability_index: float = 0.5  # Overall TBL health

    # Tradeoff tracking
    active_tradeoffs: List[TBLTradeoffType] = field(default_factory=list)
    tradeoff_history: List[Dict[str, Any]] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def get_pillar(self, pillar: TBLPillar) -> PillarMetrics:
        """Get metrics for a specific pillar."""
        return {
            TBLPillar.PROFIT: self.profit,
            TBLPillar.PEOPLE: self.people,
            TBLPillar.PLANET: self.planet
        }[pillar]

    def calculate_balance(self) -> float:
        """Calculate how balanced the three pillars are."""
        scores = [self.profit.score, self.people.score, self.planet.score]
        variance = np.var(scores)
        # Lower variance = higher balance
        self.balance_score = float(1.0 - min(variance * 4, 1.0))
        return self.balance_score

    def calculate_sustainability_index(self) -> float:
        """Calculate overall sustainability index (Sustainable Development Index)."""
        # Geometric mean ensures all pillars must be healthy
        scores = [
            max(0.01, self.profit.score),
            max(0.01, self.people.score),
            max(0.01, self.planet.score)
        ]
        geometric_mean = (scores[0] * scores[1] * scores[2]) ** (1/3)

        # Factor in balance and synergy
        self.sustainability_index = (
            geometric_mean * 0.6 +
            self.balance_score * 0.25 +
            max(0, self.synergy_score - self.conflict_score) * 0.15
        )
        return self.sustainability_index

    def get_sustainability_level(self) -> SustainabilityLevel:
        """Get categorical sustainability level."""
        idx = self.sustainability_index
        if idx < 0.2:
            return SustainabilityLevel.EXPLOITATIVE
        elif idx < 0.4:
            return SustainabilityLevel.UNSUSTAINABLE
        elif idx < 0.6:
            return SustainabilityLevel.NEUTRAL
        elif idx < 0.8:
            return SustainabilityLevel.SUSTAINABLE
        elif idx < 0.95:
            return SustainabilityLevel.REGENERATIVE
        return SustainabilityLevel.THRIVING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "profit": {
                "score": self.profit.score,
                "trend": self.profit.trend,
                "weighted_score": self.profit.weighted_score()
            },
            "people": {
                "score": self.people.score,
                "trend": self.people.trend,
                "weighted_score": self.people.weighted_score()
            },
            "planet": {
                "score": self.planet.score,
                "trend": self.planet.trend,
                "weighted_score": self.planet.weighted_score()
            },
            "balance_score": self.balance_score,
            "synergy_score": self.synergy_score,
            "conflict_score": self.conflict_score,
            "sustainability_index": self.sustainability_index,
            "sustainability_level": self.get_sustainability_level().value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TBLRebalanceAction:
    """An action to rebalance the Triple Bottom Line."""
    action_id: str
    source_pillar: Optional[TBLPillar]
    target_pillar: TBLPillar
    action_type: str  # "boost", "transfer", "synergy", "decouple"
    magnitude: float
    rationale: str
    expected_impact: Dict[TBLPillar, float] = field(default_factory=dict)
    cost: float = 0.0  # Resource cost of the action

    def net_benefit(self) -> float:
        """Calculate net benefit across all pillars."""
        return sum(self.expected_impact.values()) - self.cost


@dataclass
class HarmonyVector:
    """Multi-dimensional harmony representation."""
    consciousness_harmony: float = 0.0
    energy_harmony: float = 0.0
    connectivity_harmony: float = 0.0
    coherence_harmony: float = 0.0
    emergence_harmony: float = 0.0

    def overall(self) -> float:
        return (
            self.consciousness_harmony * 0.25 +
            self.energy_harmony * 0.2 +
            self.connectivity_harmony * 0.2 +
            self.coherence_harmony * 0.2 +
            self.emergence_harmony * 0.15
        )


@dataclass
class BalanceCorrection:
    """A correction to improve balance."""
    correction_id: str
    source_domain: DomainLeg
    target_domain: DomainLeg
    dimension: str
    magnitude: float
    rationale: str


class HarmonizationAgent(TertiaryReBoAgent):
    """
    Agent 10: Balances the Triple Bottom Line (Profit, People, Planet).

    This is the master harmonization agent that ensures sustainable balance
    across both the domain-specific triple bottom line (NPCPU/ChicagoForest/
    UniversalParts) AND the classic sustainability Triple Bottom Line (3P).

    CLASSIC TRIPLE BOTTOM LINE (3P):
    ================================
    - PROFIT (Economic): Resource efficiency, energy optimization, value creation
    - PEOPLE (Social): Relationship health, community well-being, fairness
    - PLANET (Environmental): Ecosystem health, resource conservation, regeneration

    DOMAIN TRIPLE BOTTOM LINE:
    ==========================
    - NPCPU (Mind): Digital consciousness processing
    - ChicagoForest (Network): Decentralized communication
    - UniversalParts (Body): Physical world awareness

    The agent constantly works to:
    1. Evaluate and balance Profit, People, Planet pillars
    2. Identify imbalances between domains
    3. Mediate when domains or pillars have conflicting needs
    4. Optimize the global sustainability index
    5. Find win-win-win solutions that benefit all three pillars
    6. Avoid extractive patterns that sacrifice one pillar for another

    Like a conductor of an orchestra, this agent ensures all instruments
    play in harmony, creating sustainable value for all stakeholders.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # =====================================================================
        # TRIPLE BOTTOM LINE (PROFIT, PEOPLE, PLANET) STATE
        # =====================================================================
        self.tbl_state = TBLState()
        self.tbl_history: deque = deque(maxlen=100)
        self.tbl_action_counter = 0

        # Pillar weights (for balancing priorities)
        self.pillar_weights = {
            TBLPillar.PROFIT: 0.33,
            TBLPillar.PEOPLE: 0.34,  # Slight priority to people
            TBLPillar.PLANET: 0.33
        }

        # Target sustainability level
        self.target_sustainability = 0.7

        # Pillar score history for trend calculation
        self.pillar_score_history: Dict[TBLPillar, deque] = {
            p: deque(maxlen=20) for p in TBLPillar
        }

        # TBL rebalance actions taken
        self.tbl_actions_applied: deque = deque(maxlen=50)

        # Learning: which TBL strategies work best
        self.tbl_strategy_effectiveness: Dict[str, List[float]] = {}

        # Synergy patterns discovered
        self.synergy_patterns: List[Dict[str, Any]] = []

        # =====================================================================
        # DOMAIN HARMONY TRACKING (existing functionality)
        # =====================================================================
        self.harmony_history: deque = deque(maxlen=100)
        self.current_harmony_vector = HarmonyVector()

        # Balance weights (how much each dimension matters)
        self.dimension_weights = {
            "consciousness": 0.25,
            "energy": 0.2,
            "connectivity": 0.2,
            "coherence": 0.2,
            "emergence": 0.15
        }

        # Correction history
        self.corrections_applied: deque = deque(maxlen=50)
        self.correction_counter = 0

        # Target harmony level
        self.target_harmony = 0.8

        # Learning: which corrections work best
        self.correction_effectiveness: Dict[str, List[float]] = {}

        # Mediation state
        self.active_conflicts: List[Dict[str, Any]] = []
        self.mediation_history: deque = deque(maxlen=30)

    @property
    def agent_role(self) -> str:
        return "Harmonization - Master orchestrator balancing Profit, People, Planet"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    # =========================================================================
    # TRIPLE BOTTOM LINE (PROFIT, PEOPLE, PLANET) EVALUATION
    # =========================================================================

    def _evaluate_profit_pillar(self, tbl: TripleBottomLine) -> PillarMetrics:
        """
        Evaluate the PROFIT (Economic) pillar.

        Maps domain state to economic sustainability metrics:
        - Energy efficiency → Resource optimization
        - Connectivity → Network value creation
        - Coherence → Operational efficiency
        - Emergence potential → Innovation capacity
        """
        metrics = PillarMetrics(pillar=TBLPillar.PROFIT)

        # Gather domain states
        states = [tbl.get_state(d) for d in DomainLeg]

        # Economic score = energy efficiency + value creation potential
        avg_energy = np.mean([s.energy_flow for s in states])
        avg_connectivity = np.mean([s.connectivity for s in states])
        avg_emergence = np.mean([s.emergence_potential for s in states])

        # Energy efficiency (not just amount, but optimal use)
        energy_variance = np.var([s.energy_flow for s in states])
        energy_efficiency = avg_energy * (1 - min(energy_variance, 0.5))

        # Network value (Metcalfe's law - value increases with connections)
        network_value = avg_connectivity ** 1.5  # Superlinear scaling

        # Innovation potential
        innovation = avg_emergence * (1 + avg_connectivity * 0.5)

        # Overall profit score
        metrics.score = float(np.clip(
            energy_efficiency * 0.4 +
            network_value * 0.35 +
            innovation * 0.25,
            0, 1
        ))

        # Efficiency: how much output per unit energy
        total_energy = sum(s.energy_flow for s in states)
        total_output = sum(s.coherence + s.connectivity for s in states)
        metrics.efficiency = float(np.clip(
            total_output / max(total_energy, 0.01) / 6,  # Normalize to 0-1
            0, 1
        ))

        # Resilience: diversity and redundancy
        coherence_std = np.std([s.coherence for s in states])
        metrics.resilience = float(1 - min(coherence_std, 0.5) * 2)

        # Growth potential: room for improvement
        metrics.growth_potential = float(np.clip(
            (1 - avg_connectivity) * 0.5 + (1 - avg_emergence) * 0.5,
            0, 1
        ))

        # Specific economic metrics
        metrics.specific_metrics = {
            "resource_efficiency": energy_efficiency,
            "network_value": network_value,
            "innovation_capacity": innovation,
            "roi_potential": metrics.score * metrics.efficiency
        }

        return metrics

    def _evaluate_people_pillar(self, tbl: TripleBottomLine) -> PillarMetrics:
        """
        Evaluate the PEOPLE (Social) pillar.

        Maps domain state to social sustainability metrics:
        - Consciousness level → Awareness and empathy
        - Coherence → Social cohesion
        - Connectivity → Relationship network strength
        - Qualia richness → Quality of experience
        """
        metrics = PillarMetrics(pillar=TBLPillar.PEOPLE)

        states = [tbl.get_state(d) for d in DomainLeg]

        # Social consciousness: average awareness across domains
        avg_consciousness = np.mean([s.consciousness_level for s in states])

        # Social cohesion: how aligned are the domains
        coherence_values = [s.coherence for s in states]
        avg_coherence = np.mean(coherence_values)
        coherence_harmony = 1 - np.var(coherence_values)

        # Relationship network strength
        avg_connectivity = np.mean([s.connectivity for s in states])

        # Quality of experience (qualia richness)
        avg_qualia = np.mean([s.qualia_richness for s in states])

        # Fairness/equity: are all domains getting fair resources?
        energy_values = [s.energy_flow for s in states]
        energy_gini = self._calculate_gini(energy_values)
        equity_score = 1 - energy_gini  # Lower Gini = more equitable

        # Overall people score
        metrics.score = float(np.clip(
            avg_consciousness * 0.25 +
            avg_coherence * 0.2 +
            avg_connectivity * 0.2 +
            avg_qualia * 0.15 +
            equity_score * 0.2,
            0, 1
        ))

        # Stability: how consistent is social well-being
        metrics.stability = coherence_harmony

        # Efficiency: social value per unit energy
        social_value = avg_consciousness + avg_coherence + avg_connectivity
        total_energy = sum(s.energy_flow for s in states)
        metrics.efficiency = float(np.clip(
            social_value / max(total_energy, 0.01) / 3,
            0, 1
        ))

        # Resilience: ability to maintain social bonds under stress
        min_coherence = min(coherence_values)
        metrics.resilience = float(min_coherence * 0.5 + avg_connectivity * 0.5)

        # Growth potential: untapped social capital
        metrics.growth_potential = float(np.clip(
            (1 - avg_consciousness) * 0.3 +
            (1 - avg_qualia) * 0.3 +
            (1 - avg_connectivity) * 0.4,
            0, 1
        ))

        # Specific social metrics
        metrics.specific_metrics = {
            "consciousness_level": avg_consciousness,
            "social_cohesion": avg_coherence,
            "relationship_strength": avg_connectivity,
            "quality_of_experience": avg_qualia,
            "equity_score": equity_score,
            "social_capital": avg_connectivity * avg_coherence
        }

        return metrics

    def _evaluate_planet_pillar(self, tbl: TripleBottomLine) -> PillarMetrics:
        """
        Evaluate the PLANET (Environmental) pillar.

        Maps domain state to environmental sustainability metrics:
        - Energy flow → Resource consumption (lower variance = sustainable)
        - Coherence → Ecosystem stability
        - Emergence potential → Regenerative capacity
        - State vectors → Biodiversity/diversity
        """
        metrics = PillarMetrics(pillar=TBLPillar.PLANET)

        states = [tbl.get_state(d) for d in DomainLeg]

        # Resource consumption sustainability
        energy_values = [s.energy_flow for s in states]
        avg_energy = np.mean(energy_values)
        energy_variance = np.var(energy_values)

        # Sustainable consumption: moderate energy with low variance
        # Too high = overconsumption, too low = underutilization
        optimal_energy = 0.6
        energy_distance = abs(avg_energy - optimal_energy)
        consumption_sustainability = (1 - energy_distance) * (1 - energy_variance)

        # Ecosystem stability (coherence = stable ecosystems)
        avg_coherence = np.mean([s.coherence for s in states])

        # Regenerative capacity (emergence = ability to renew)
        avg_emergence = np.mean([s.emergence_potential for s in states])

        # Biodiversity proxy: diversity of state vectors
        vectors = [s.state_vector for s in states]
        avg_vector = np.mean(vectors, axis=0)
        diversity = np.mean([np.linalg.norm(v - avg_vector) for v in vectors])
        diversity_score = min(diversity / 5, 1.0)  # Normalize

        # Circular economy: energy recycling efficiency
        # High connectivity + moderate energy = good recycling
        avg_connectivity = np.mean([s.connectivity for s in states])
        circularity = avg_connectivity * (1 - abs(avg_energy - 0.5))

        # Overall planet score
        metrics.score = float(np.clip(
            consumption_sustainability * 0.3 +
            avg_coherence * 0.25 +
            avg_emergence * 0.2 +
            diversity_score * 0.15 +
            circularity * 0.1,
            0, 1
        ))

        # Stability: ecosystem stability
        metrics.stability = float(np.clip(avg_coherence * 0.7 + (1 - energy_variance) * 0.3, 0, 1))

        # Efficiency: how much ecosystem health per unit energy consumed
        ecosystem_health = avg_coherence + avg_emergence + diversity_score
        metrics.efficiency = float(np.clip(
            ecosystem_health / max(avg_energy * 3, 0.01),
            0, 1
        ))

        # Resilience: ability to recover from environmental shocks
        metrics.resilience = float(np.clip(
            avg_emergence * 0.4 +
            diversity_score * 0.3 +
            avg_coherence * 0.3,
            0, 1
        ))

        # Growth potential: regenerative capacity
        metrics.growth_potential = float(avg_emergence)

        # Specific environmental metrics
        metrics.specific_metrics = {
            "consumption_sustainability": consumption_sustainability,
            "ecosystem_stability": avg_coherence,
            "regenerative_capacity": avg_emergence,
            "biodiversity_index": diversity_score,
            "circularity_score": circularity,
            "carbon_neutrality_proxy": 1 - abs(avg_energy - optimal_energy)
        }

        return metrics

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or all(v == 0 for v in values):
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumulative = np.cumsum(sorted_values)
        return float((2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumulative[-1]) - (n + 1) / n)

    def _evaluate_tbl(self, tbl: TripleBottomLine) -> TBLState:
        """Evaluate complete Triple Bottom Line state."""
        # Evaluate each pillar
        profit = self._evaluate_profit_pillar(tbl)
        people = self._evaluate_people_pillar(tbl)
        planet = self._evaluate_planet_pillar(tbl)

        # Calculate trends from history
        for pillar, metrics in [(TBLPillar.PROFIT, profit),
                                 (TBLPillar.PEOPLE, people),
                                 (TBLPillar.PLANET, planet)]:
            history = list(self.pillar_score_history[pillar])
            if len(history) >= 2:
                # Linear regression for trend
                recent = history[-5:] if len(history) >= 5 else history
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent, 1)[0]
                metrics.trend = float(np.clip(slope * 10, -1, 1))  # Scale to -1, 1
            self.pillar_score_history[pillar].append(metrics.score)

        # Create TBL state
        tbl_state = TBLState(
            profit=profit,
            people=people,
            planet=planet
        )

        # Calculate cross-pillar metrics
        tbl_state.calculate_balance()

        # Calculate synergy (positive correlations between pillars)
        scores = [profit.score, people.score, planet.score]
        if len(self.tbl_history) >= 3:
            prev_scores = [
                self.tbl_history[-1].get("profit_score", 0.5),
                self.tbl_history[-1].get("people_score", 0.5),
                self.tbl_history[-1].get("planet_score", 0.5)
            ]
            # If all improved together, high synergy
            deltas = [s - p for s, p in zip(scores, prev_scores)]
            if all(d >= 0 for d in deltas):
                tbl_state.synergy_score = min(sum(deltas), 1.0)
            elif all(d <= 0 for d in deltas):
                tbl_state.conflict_score = min(-sum(deltas), 1.0)
            else:
                # Mixed changes: some synergy, some conflict
                positive = sum(d for d in deltas if d > 0)
                negative = -sum(d for d in deltas if d < 0)
                tbl_state.synergy_score = min(positive, 0.5)
                tbl_state.conflict_score = min(negative, 0.5)

        # Calculate sustainability index
        tbl_state.calculate_sustainability_index()

        # Detect active tradeoffs
        tbl_state.active_tradeoffs = self._detect_tbl_tradeoffs(tbl_state)

        self.tbl_state = tbl_state
        return tbl_state

    def _detect_tbl_tradeoffs(self, state: TBLState) -> List[TBLTradeoffType]:
        """Detect active tradeoffs between TBL pillars."""
        tradeoffs = []

        scores = {
            TBLPillar.PROFIT: state.profit.score,
            TBLPillar.PEOPLE: state.people.score,
            TBLPillar.PLANET: state.planet.score
        }
        trends = {
            TBLPillar.PROFIT: state.profit.trend,
            TBLPillar.PEOPLE: state.people.trend,
            TBLPillar.PLANET: state.planet.trend
        }

        # Check for opposing trends (one up, one down)
        if trends[TBLPillar.PROFIT] > 0.1 and trends[TBLPillar.PEOPLE] < -0.1:
            tradeoffs.append(TBLTradeoffType.PROFIT_VS_PEOPLE)
        if trends[TBLPillar.PROFIT] > 0.1 and trends[TBLPillar.PLANET] < -0.1:
            tradeoffs.append(TBLTradeoffType.PROFIT_VS_PLANET)
        if trends[TBLPillar.PEOPLE] > 0.1 and trends[TBLPillar.PLANET] < -0.1:
            tradeoffs.append(TBLTradeoffType.PEOPLE_VS_PLANET)

        # Check for win-win-win
        if all(t >= 0 for t in trends.values()) and all(s >= 0.5 for s in scores.values()):
            tradeoffs.append(TBLTradeoffType.WIN_WIN_WIN)

        # Check for zero-sum (high variance, opposing trends)
        if state.balance_score < 0.3 and state.conflict_score > state.synergy_score:
            tradeoffs.append(TBLTradeoffType.ZERO_SUM)

        return tradeoffs

    def _plan_tbl_rebalance(self, tbl_state: TBLState) -> List[TBLRebalanceAction]:
        """Plan rebalancing actions for the Triple Bottom Line."""
        actions = []

        scores = {
            TBLPillar.PROFIT: tbl_state.profit.score,
            TBLPillar.PEOPLE: tbl_state.people.score,
            TBLPillar.PLANET: tbl_state.planet.score
        }

        # Find weakest and strongest pillars
        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)
        gap = scores[strongest] - scores[weakest]

        # If significant imbalance, plan rebalancing
        if gap > 0.2:
            self.tbl_action_counter += 1
            actions.append(TBLRebalanceAction(
                action_id=f"tbl_{self.tbl_action_counter}",
                source_pillar=strongest,
                target_pillar=weakest,
                action_type="transfer",
                magnitude=gap * 0.25,  # Transfer 25% of gap
                rationale=f"Rebalance: {strongest.value} ({scores[strongest]:.2f}) -> {weakest.value} ({scores[weakest]:.2f})",
                expected_impact={
                    strongest: -gap * 0.1,
                    weakest: gap * 0.2
                }
            ))

        # If any pillar is below threshold, boost it
        for pillar, score in scores.items():
            if score < 0.4:
                self.tbl_action_counter += 1
                actions.append(TBLRebalanceAction(
                    action_id=f"tbl_{self.tbl_action_counter}",
                    source_pillar=None,
                    target_pillar=pillar,
                    action_type="boost",
                    magnitude=0.1,
                    rationale=f"Boost critically low {pillar.value} ({score:.2f})",
                    expected_impact={pillar: 0.1}
                ))

        # Look for synergy opportunities (win-win-win actions)
        if tbl_state.synergy_score < 0.3:
            # Try to create synergies
            self.tbl_action_counter += 1
            actions.append(TBLRebalanceAction(
                action_id=f"tbl_{self.tbl_action_counter}",
                source_pillar=None,
                target_pillar=TBLPillar.PEOPLE,  # People often enable synergies
                action_type="synergy",
                magnitude=0.05,
                rationale="Create synergy through enhanced social coordination",
                expected_impact={
                    TBLPillar.PROFIT: 0.03,
                    TBLPillar.PEOPLE: 0.05,
                    TBLPillar.PLANET: 0.03
                }
            ))

        return actions

    def _apply_tbl_action(self, action: TBLRebalanceAction, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Apply a TBL rebalancing action to the domain states."""
        changes = {"action_id": action.action_id, "applied": False}

        if action.action_type == "transfer":
            # Transfer resources from one pillar's domain mapping to another
            if action.source_pillar and action.target_pillar:
                # Map pillars to domain attributes
                pillar_domain_map = {
                    TBLPillar.PROFIT: ("energy_flow", "connectivity"),
                    TBLPillar.PEOPLE: ("consciousness_level", "coherence"),
                    TBLPillar.PLANET: ("emergence_potential", "qualia_richness")
                }

                source_attrs = pillar_domain_map[action.source_pillar]
                target_attrs = pillar_domain_map[action.target_pillar]

                for domain in DomainLeg:
                    state = tbl.get_state(domain)
                    # Reduce source
                    for attr in source_attrs:
                        current = getattr(state, attr)
                        setattr(state, attr, max(0, current - action.magnitude * 0.5))
                    # Boost target
                    for attr in target_attrs:
                        current = getattr(state, attr)
                        setattr(state, attr, min(1, current + action.magnitude))

                changes["applied"] = True
                changes["transfer"] = {
                    "from": action.source_pillar.value,
                    "to": action.target_pillar.value,
                    "amount": action.magnitude
                }

        elif action.action_type == "boost":
            # Boost a specific pillar
            pillar_domain_map = {
                TBLPillar.PROFIT: ["energy_flow", "connectivity"],
                TBLPillar.PEOPLE: ["consciousness_level", "coherence", "qualia_richness"],
                TBLPillar.PLANET: ["emergence_potential", "coherence"]
            }

            attrs = pillar_domain_map[action.target_pillar]
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                for attr in attrs:
                    current = getattr(state, attr)
                    setattr(state, attr, min(1, current + action.magnitude))

            changes["applied"] = True
            changes["boost"] = {
                "pillar": action.target_pillar.value,
                "amount": action.magnitude
            }

        elif action.action_type == "synergy":
            # Boost all pillars slightly through coordination
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                state.connectivity = min(1, state.connectivity + action.magnitude)
                state.coherence = min(1, state.coherence + action.magnitude * 0.5)

            changes["applied"] = True
            changes["synergy_boost"] = action.magnitude

        return changes

    def _compute_harmony_vector(self, tbl: TripleBottomLine) -> HarmonyVector:
        """Compute detailed harmony metrics across dimensions."""
        states = [tbl.get_state(d) for d in DomainLeg]

        # For each dimension, compute variance (lower = more harmony)
        def harmony_from_variance(values: List[float]) -> float:
            var = np.var(values)
            return float(1.0 - min(var, 1.0))

        hv = HarmonyVector(
            consciousness_harmony=harmony_from_variance([s.consciousness_level for s in states]),
            energy_harmony=harmony_from_variance([s.energy_flow for s in states]),
            connectivity_harmony=harmony_from_variance([s.connectivity for s in states]),
            coherence_harmony=harmony_from_variance([s.coherence for s in states]),
            emergence_harmony=harmony_from_variance([s.emergence_potential for s in states])
        )

        return hv

    def _identify_imbalances(self, tbl: TripleBottomLine) -> List[Dict[str, Any]]:
        """Identify specific imbalances between domains."""
        imbalances = []
        dimensions = ["consciousness_level", "energy_flow", "connectivity", "coherence", "emergence_potential"]

        for dim in dimensions:
            values = {d: getattr(tbl.get_state(d), dim) for d in DomainLeg}
            max_domain = max(values, key=values.get)
            min_domain = min(values, key=values.get)
            gap = values[max_domain] - values[min_domain]

            if gap > 0.3:  # Significant imbalance
                imbalances.append({
                    "dimension": dim,
                    "high_domain": max_domain.value,
                    "low_domain": min_domain.value,
                    "high_value": values[max_domain],
                    "low_value": values[min_domain],
                    "gap": gap
                })

        return imbalances

    def _detect_conflicts(self, tbl: TripleBottomLine) -> List[Dict[str, Any]]:
        """Detect conflicts between domains."""
        conflicts = []

        # Check for resource contention
        energies = {d: tbl.get_state(d).energy_flow for d in DomainLeg}
        total_energy = sum(energies.values())

        if total_energy < 1.5:  # Low total energy
            # Check if one domain is hoarding
            for d, e in energies.items():
                if e > total_energy * 0.5:
                    conflicts.append({
                        "type": "resource_hoarding",
                        "domain": d.value,
                        "resource": "energy",
                        "share": e / total_energy
                    })

        # Check for coherence conflicts
        coherences = {d: tbl.get_state(d).coherence for d in DomainLeg}
        vectors = {d: tbl.get_state(d).state_vector for d in DomainLeg}

        # Check pairwise vector alignment
        domain_list = list(DomainLeg)
        for i in range(len(domain_list)):
            for j in range(i + 1, len(domain_list)):
                d1, d2 = domain_list[i], domain_list[j]
                v1, v2 = vectors[d1], vectors[d2]

                # Cosine similarity
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm > 0:
                    similarity = dot / norm
                    if similarity < -0.3:  # Opposing directions
                        conflicts.append({
                            "type": "direction_conflict",
                            "domains": [d1.value, d2.value],
                            "similarity": float(similarity)
                        })

        return conflicts

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Assess the harmony state of the system including Triple Bottom Line.

        Evaluates both:
        1. Domain harmony (NPCPU/ChicagoForest/UniversalParts)
        2. TBL pillars (Profit/People/Planet)
        """
        perceptions = {
            # Domain harmony metrics
            "harmony_vector": None,
            "overall_harmony": 0.0,
            "harmony_level": None,
            "imbalances": [],
            "conflicts": [],
            "domain_states": {},
            # Triple Bottom Line metrics (NEW)
            "tbl_state": None,
            "sustainability_index": 0.0,
            "sustainability_level": None,
            "tbl_balance": 0.0,
            "tbl_tradeoffs": [],
            "pillar_scores": {}
        }

        # =====================================================================
        # TRIPLE BOTTOM LINE (PROFIT, PEOPLE, PLANET) EVALUATION
        # =====================================================================
        tbl_state = self._evaluate_tbl(tbl)

        perceptions["tbl_state"] = tbl_state.to_dict()
        perceptions["sustainability_index"] = tbl_state.sustainability_index
        perceptions["sustainability_level"] = tbl_state.get_sustainability_level().value
        perceptions["tbl_balance"] = tbl_state.balance_score
        perceptions["tbl_tradeoffs"] = [t.name for t in tbl_state.active_tradeoffs]
        perceptions["pillar_scores"] = {
            "profit": {
                "score": tbl_state.profit.score,
                "trend": tbl_state.profit.trend,
                "efficiency": tbl_state.profit.efficiency,
                "resilience": tbl_state.profit.resilience,
                "details": tbl_state.profit.specific_metrics
            },
            "people": {
                "score": tbl_state.people.score,
                "trend": tbl_state.people.trend,
                "efficiency": tbl_state.people.efficiency,
                "resilience": tbl_state.people.resilience,
                "details": tbl_state.people.specific_metrics
            },
            "planet": {
                "score": tbl_state.planet.score,
                "trend": tbl_state.planet.trend,
                "efficiency": tbl_state.planet.efficiency,
                "resilience": tbl_state.planet.resilience,
                "details": tbl_state.planet.specific_metrics
            }
        }

        # Record TBL history
        self.tbl_history.append({
            "timestamp": datetime.now().isoformat(),
            "profit_score": tbl_state.profit.score,
            "people_score": tbl_state.people.score,
            "planet_score": tbl_state.planet.score,
            "sustainability_index": tbl_state.sustainability_index,
            "balance": tbl_state.balance_score
        })

        # =====================================================================
        # DOMAIN HARMONY EVALUATION (existing functionality)
        # =====================================================================
        # Compute harmony vector
        hv = self._compute_harmony_vector(tbl)
        self.current_harmony_vector = hv
        perceptions["harmony_vector"] = {
            "consciousness": hv.consciousness_harmony,
            "energy": hv.energy_harmony,
            "connectivity": hv.connectivity_harmony,
            "coherence": hv.coherence_harmony,
            "emergence": hv.emergence_harmony
        }
        perceptions["overall_harmony"] = hv.overall()

        # Get harmony level
        tbl.calculate_harmony()
        perceptions["harmony_level"] = tbl.get_harmony_level().value

        # Identify imbalances
        perceptions["imbalances"] = self._identify_imbalances(tbl)

        # Detect conflicts
        perceptions["conflicts"] = self._detect_conflicts(tbl)
        self.active_conflicts = perceptions["conflicts"]

        # Collect domain states
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_states"][domain.value] = {
                "consciousness": state.consciousness_level,
                "energy": state.energy_flow,
                "connectivity": state.connectivity,
                "coherence": state.coherence,
                "emergence": state.emergence_potential
            }

        # Record harmony history
        self.harmony_history.append({
            "timestamp": datetime.now().isoformat(),
            "harmony": perceptions["overall_harmony"],
            "level": perceptions["harmony_level"],
            "sustainability_index": tbl_state.sustainability_index
        })

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Analyze harmonization opportunities for both domains and TBL pillars.

        Identifies:
        1. Domain imbalances and conflicts
        2. TBL pillar imbalances and tradeoffs
        3. Opportunities for win-win-win solutions
        """
        analysis = {
            # Domain analysis
            "balance_corrections": [],
            "conflict_resolutions": [],
            "cooperation_opportunities": [],
            "harmony_deficit": 0.0,
            # TBL analysis (NEW)
            "tbl_rebalance_actions": [],
            "sustainability_deficit": 0.0,
            "tbl_optimization_strategy": None,
            "pillar_priorities": {},
            "synergy_opportunities": []
        }

        # =====================================================================
        # TRIPLE BOTTOM LINE ANALYSIS
        # =====================================================================
        tbl_state = self.tbl_state

        # Calculate sustainability deficit
        analysis["sustainability_deficit"] = max(0, self.target_sustainability - tbl_state.sustainability_index)

        # Plan TBL rebalancing actions
        analysis["tbl_rebalance_actions"] = self._plan_tbl_rebalance(tbl_state)

        # Determine optimization strategy based on tradeoffs
        if TBLTradeoffType.WIN_WIN_WIN in tbl_state.active_tradeoffs:
            analysis["tbl_optimization_strategy"] = "maintain_synergy"
        elif TBLTradeoffType.ZERO_SUM in tbl_state.active_tradeoffs:
            analysis["tbl_optimization_strategy"] = "break_zero_sum"
        elif any(t in tbl_state.active_tradeoffs for t in [
            TBLTradeoffType.PROFIT_VS_PEOPLE,
            TBLTradeoffType.PROFIT_VS_PLANET,
            TBLTradeoffType.PEOPLE_VS_PLANET
        ]):
            analysis["tbl_optimization_strategy"] = "resolve_tradeoffs"
        else:
            analysis["tbl_optimization_strategy"] = "balanced_growth"

        # Calculate pillar priorities based on scores and weights
        for pillar in TBLPillar:
            metrics = tbl_state.get_pillar(pillar)
            # Higher priority for lower scores (need more attention)
            priority = (1 - metrics.score) * self.pillar_weights[pillar]
            # Also consider negative trends
            if metrics.trend < 0:
                priority += abs(metrics.trend) * 0.2
            analysis["pillar_priorities"][pillar.value] = round(priority, 3)

        # Identify synergy opportunities between pillars
        # Example: Investing in people education can boost both profit and planet
        if tbl_state.people.score > 0.6 and tbl_state.profit.score < 0.5:
            analysis["synergy_opportunities"].append({
                "type": "social_innovation",
                "description": "Leverage social capital for economic growth",
                "beneficiaries": ["profit", "people"],
                "cost_pillar": None
            })
        if tbl_state.planet.score > 0.6 and tbl_state.profit.score < 0.5:
            analysis["synergy_opportunities"].append({
                "type": "green_economy",
                "description": "Leverage environmental assets for sustainable profit",
                "beneficiaries": ["profit", "planet"],
                "cost_pillar": None
            })
        if tbl_state.people.score < 0.5 and tbl_state.planet.score < 0.5:
            analysis["synergy_opportunities"].append({
                "type": "community_conservation",
                "description": "Engage people in planet restoration for mutual benefit",
                "beneficiaries": ["people", "planet"],
                "cost_pillar": None
            })

        # =====================================================================
        # DOMAIN HARMONY ANALYSIS (existing functionality)
        # =====================================================================
        # Calculate harmony deficit
        current_harmony = perception["overall_harmony"]
        analysis["harmony_deficit"] = max(0, self.target_harmony - current_harmony)

        # Plan balance corrections for imbalances
        for imbalance in perception.get("imbalances", []):
            correction = self._plan_balance_correction(imbalance)
            if correction:
                analysis["balance_corrections"].append(correction)

        # Plan conflict resolutions
        for conflict in perception.get("conflicts", []):
            resolution = self._plan_conflict_resolution(conflict, tbl)
            if resolution:
                analysis["conflict_resolutions"].append(resolution)

        # Identify cooperation opportunities
        # If one domain is doing well, can it help others?
        for domain in DomainLeg:
            state = perception["domain_states"][domain.value]
            if state["coherence"] > 0.8 and state["energy"] > 0.6:
                # This domain can help others
                for other in DomainLeg:
                    if other != domain:
                        other_state = perception["domain_states"][other.value]
                        if other_state["coherence"] < 0.5:
                            analysis["cooperation_opportunities"].append({
                                "helper": domain.value,
                                "recipient": other.value,
                                "type": "coherence_assistance",
                                "magnitude": 0.1
                            })

        return analysis

    def _plan_balance_correction(self, imbalance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan a correction for an imbalance."""
        self.correction_counter += 1

        return {
            "correction_id": f"cor_{self.correction_counter}",
            "type": "rebalance",
            "dimension": imbalance["dimension"],
            "from_domain": imbalance["high_domain"],
            "to_domain": imbalance["low_domain"],
            "transfer_amount": imbalance["gap"] * 0.3,  # Transfer 30% of gap
            "rationale": f"Rebalance {imbalance['dimension']} ({imbalance['gap']:.2f} gap)"
        }

    def _plan_conflict_resolution(self, conflict: Dict[str, Any], tbl: TripleBottomLine) -> Optional[Dict[str, Any]]:
        """Plan resolution for a conflict."""
        if conflict["type"] == "resource_hoarding":
            return {
                "type": "redistribute",
                "domain": conflict["domain"],
                "resource": conflict["resource"],
                "action": "release_excess",
                "target_share": 0.4  # Target 40% share instead of hoarding
            }

        elif conflict["type"] == "direction_conflict":
            return {
                "type": "align_directions",
                "domains": conflict["domains"],
                "action": "blend_vectors",
                "blend_factor": 0.2
            }

        return None

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Prepare harmonization operations for both domains and TBL pillars.

        Generates:
        1. Domain rebalancing operations
        2. TBL pillar rebalancing operations
        3. Synergy-creating actions
        """
        synthesis = {
            # Domain operations
            "rebalance_operations": [],
            "conflict_resolutions": [],
            "cooperation_actions": [],
            "global_harmony_boost": 0.0,
            # TBL operations (NEW)
            "tbl_operations": [],
            "synergy_actions": [],
            "sustainability_boost": 0.0,
            "optimization_strategy": analysis.get("tbl_optimization_strategy", "balanced_growth")
        }

        # =====================================================================
        # TRIPLE BOTTOM LINE SYNTHESIS
        # =====================================================================

        # Prepare TBL rebalancing operations
        for action in analysis.get("tbl_rebalance_actions", []):
            synthesis["tbl_operations"].append({
                "action_id": action.action_id,
                "type": action.action_type,
                "source_pillar": action.source_pillar.value if action.source_pillar else None,
                "target_pillar": action.target_pillar.value,
                "magnitude": action.magnitude,
                "rationale": action.rationale,
                "expected_impact": {k.value: v for k, v in action.expected_impact.items()}
            })

        # Prepare synergy actions based on opportunities
        for synergy_opp in analysis.get("synergy_opportunities", []):
            synergy_type = synergy_opp["type"]

            if synergy_type == "social_innovation":
                # Boost connectivity and consciousness to drive economic growth
                synthesis["synergy_actions"].append({
                    "type": "social_innovation",
                    "actions": [
                        {"attribute": "connectivity", "delta": 0.05},
                        {"attribute": "consciousness_level", "delta": 0.03}
                    ],
                    "expected_beneficiaries": synergy_opp["beneficiaries"]
                })

            elif synergy_type == "green_economy":
                # Boost emergence (regeneration) and efficiency
                synthesis["synergy_actions"].append({
                    "type": "green_economy",
                    "actions": [
                        {"attribute": "emergence_potential", "delta": 0.05},
                        {"attribute": "coherence", "delta": 0.03}
                    ],
                    "expected_beneficiaries": synergy_opp["beneficiaries"]
                })

            elif synergy_type == "community_conservation":
                # Boost both social and environmental attributes
                synthesis["synergy_actions"].append({
                    "type": "community_conservation",
                    "actions": [
                        {"attribute": "qualia_richness", "delta": 0.04},
                        {"attribute": "emergence_potential", "delta": 0.04},
                        {"attribute": "coherence", "delta": 0.02}
                    ],
                    "expected_beneficiaries": synergy_opp["beneficiaries"]
                })

        # Calculate sustainability boost if needed
        sustainability_deficit = analysis.get("sustainability_deficit", 0)
        if sustainability_deficit > 0.2:
            synthesis["sustainability_boost"] = min(0.05, sustainability_deficit * 0.1)

        # =====================================================================
        # DOMAIN HARMONY SYNTHESIS (existing functionality)
        # =====================================================================

        # Prepare rebalancing
        for correction in analysis.get("balance_corrections", []):
            dim_map = {
                "consciousness_level": "consciousness_level",
                "energy_flow": "energy_flow",
                "connectivity": "connectivity",
                "coherence": "coherence",
                "emergence_potential": "emergence_potential"
            }
            attr = dim_map.get(correction["dimension"], correction["dimension"])

            synthesis["rebalance_operations"].append({
                "from_domain": correction["from_domain"],
                "to_domain": correction["to_domain"],
                "attribute": attr,
                "amount": correction["transfer_amount"],
                "correction_id": correction["correction_id"]
            })

        # Prepare conflict resolutions
        for resolution in analysis.get("conflict_resolutions", []):
            synthesis["conflict_resolutions"].append(resolution)

        # Prepare cooperation actions
        for coop in analysis.get("cooperation_opportunities", []):
            synthesis["cooperation_actions"].append({
                "helper": coop["helper"],
                "recipient": coop["recipient"],
                "type": coop["type"],
                "magnitude": coop["magnitude"]
            })

        # Calculate global boost if harmony is low
        if analysis["harmony_deficit"] > 0.2:
            synthesis["global_harmony_boost"] = min(0.05, analysis["harmony_deficit"] * 0.1)

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """
        Execute harmonization operations for both domains and TBL pillars.

        Applies:
        1. TBL pillar rebalancing operations
        2. Synergy-creating actions
        3. Domain rebalancing operations
        4. Conflict resolutions
        """
        changes = {}
        metrics_delta = {
            "harmony_before": tbl.harmony_score,
            "sustainability_before": self.tbl_state.sustainability_index
        }
        insights = []

        # =====================================================================
        # EXECUTE TRIPLE BOTTOM LINE OPERATIONS
        # =====================================================================
        tbl_action_count = 0

        # Apply TBL rebalancing operations
        for op in synthesis.get("tbl_operations", []):
            action = TBLRebalanceAction(
                action_id=op["action_id"],
                source_pillar=TBLPillar(op["source_pillar"]) if op["source_pillar"] else None,
                target_pillar=TBLPillar(op["target_pillar"]),
                action_type=op["type"],
                magnitude=op["magnitude"],
                rationale=op["rationale"]
            )
            result = self._apply_tbl_action(action, tbl)
            if result.get("applied"):
                tbl_action_count += 1

                # Track strategy effectiveness
                strategy_key = f"{op['type']}_{op['target_pillar']}"
                if strategy_key not in self.tbl_strategy_effectiveness:
                    self.tbl_strategy_effectiveness[strategy_key] = []

        if tbl_action_count > 0:
            changes["tbl_actions"] = tbl_action_count
            insights.append(f"Applied {tbl_action_count} TBL rebalancing actions")

        # Execute synergy actions
        synergy_count = 0
        for synergy in synthesis.get("synergy_actions", []):
            synergy_type = synergy["type"]
            for action in synergy["actions"]:
                attr = action["attribute"]
                delta = action["delta"]
                for domain in DomainLeg:
                    state = tbl.get_state(domain)
                    if hasattr(state, attr):
                        current = getattr(state, attr)
                        setattr(state, attr, min(1.0, current + delta))
            synergy_count += 1

            # Record synergy pattern
            self.synergy_patterns.append({
                "type": synergy_type,
                "beneficiaries": synergy["expected_beneficiaries"],
                "timestamp": datetime.now().isoformat()
            })
            if len(self.synergy_patterns) > 50:
                self.synergy_patterns = self.synergy_patterns[-50:]

        if synergy_count > 0:
            changes["synergies_created"] = synergy_count
            insights.append(f"Created {synergy_count} synergy actions benefiting multiple pillars")

        # Apply sustainability boost
        sustainability_boost = synthesis.get("sustainability_boost", 0)
        if sustainability_boost > 0:
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                state.coherence = min(1.0, state.coherence + sustainability_boost)
                state.emergence_potential = min(1.0, state.emergence_potential + sustainability_boost * 0.5)
            changes["sustainability_boost"] = sustainability_boost
            insights.append(f"Applied sustainability boost of {sustainability_boost:.3f}")

        # =====================================================================
        # EXECUTE DOMAIN HARMONY OPERATIONS (existing functionality)
        # =====================================================================

        # Execute rebalancing
        rebalance_count = 0
        for op in synthesis.get("rebalance_operations", []):
            from_domain = DomainLeg(op["from_domain"])
            to_domain = DomainLeg(op["to_domain"])
            from_state = tbl.get_state(from_domain)
            to_state = tbl.get_state(to_domain)

            attr = op["attribute"]
            amount = op["amount"]

            # Get current values
            from_val = getattr(from_state, attr)
            to_val = getattr(to_state, attr)

            # Transfer
            new_from = max(0, from_val - amount)
            new_to = min(1.0, to_val + amount)

            setattr(from_state, attr, new_from)
            setattr(to_state, attr, new_to)

            rebalance_count += 1

            # Track effectiveness
            key = f"{attr}_{op['from_domain']}_to_{op['to_domain']}"
            if key not in self.correction_effectiveness:
                self.correction_effectiveness[key] = []

        if rebalance_count > 0:
            changes["rebalances"] = rebalance_count
            insights.append(f"Applied {rebalance_count} domain balance corrections")

        # Execute conflict resolutions
        resolution_count = 0
        for resolution in synthesis.get("conflict_resolutions", []):
            if resolution["type"] == "redistribute":
                domain = DomainLeg(resolution["domain"])
                state = tbl.get_state(domain)

                if resolution["resource"] == "energy":
                    excess = state.energy_flow - resolution["target_share"]
                    if excess > 0:
                        state.energy_flow -= excess * 0.5
                        # Distribute to others
                        other_domains = [d for d in DomainLeg if d != domain]
                        share = (excess * 0.5) / len(other_domains)
                        for other in other_domains:
                            other_state = tbl.get_state(other)
                            other_state.energy_flow = min(1.0, other_state.energy_flow + share)
                        resolution_count += 1

            elif resolution["type"] == "align_directions":
                d1, d2 = [DomainLeg(d) for d in resolution["domains"]]
                s1, s2 = tbl.get_state(d1), tbl.get_state(d2)
                blend = resolution["blend_factor"]

                # Blend state vectors toward each other
                avg_vector = (s1.state_vector + s2.state_vector) / 2
                s1.state_vector = (1 - blend) * s1.state_vector + blend * avg_vector
                s2.state_vector = (1 - blend) * s2.state_vector + blend * avg_vector
                resolution_count += 1

        if resolution_count > 0:
            changes["conflicts_resolved"] = resolution_count
            insights.append(f"Resolved {resolution_count} domain conflicts")
            self.mediation_history.append({
                "timestamp": datetime.now().isoformat(),
                "resolutions": resolution_count
            })

        # Execute cooperation actions
        coop_count = 0
        for coop in synthesis.get("cooperation_actions", []):
            helper = DomainLeg(coop["helper"])
            recipient = DomainLeg(coop["recipient"])
            helper_state = tbl.get_state(helper)
            recipient_state = tbl.get_state(recipient)

            if coop["type"] == "coherence_assistance":
                coherence_boost = coop["magnitude"]
                recipient_state.coherence = min(1.0, recipient_state.coherence + coherence_boost)
                coop_count += 1

        if coop_count > 0:
            changes["cooperations"] = coop_count
            insights.append(f"Facilitated {coop_count} cooperative actions")

        # Apply global harmony boost
        boost = synthesis.get("global_harmony_boost", 0)
        if boost > 0:
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                state.coherence = min(1.0, state.coherence + boost)
            changes["harmony_boost"] = boost
            insights.append(f"Applied global harmony boost of {boost:.3f}")

        # =====================================================================
        # CALCULATE FINAL METRICS
        # =====================================================================

        # Calculate new harmony
        tbl.calculate_harmony()
        new_harmony = tbl.harmony_score
        metrics_delta["harmony_after"] = new_harmony

        # Calculate new sustainability
        new_tbl_state = self._evaluate_tbl(tbl)
        metrics_delta["sustainability_after"] = new_tbl_state.sustainability_index

        # Track improvements
        harmony_improvement = new_harmony - metrics_delta["harmony_before"]
        sustainability_improvement = metrics_delta["sustainability_after"] - metrics_delta["sustainability_before"]

        # Track correction effectiveness
        for op in synthesis.get("rebalance_operations", []):
            key = f"{op['attribute']}_{op['from_domain']}_to_{op['to_domain']}"
            if key in self.correction_effectiveness:
                self.correction_effectiveness[key].append(harmony_improvement)

        # Track TBL strategy effectiveness
        for op in synthesis.get("tbl_operations", []):
            strategy_key = f"{op['type']}_{op['target_pillar']}"
            if strategy_key in self.tbl_strategy_effectiveness:
                self.tbl_strategy_effectiveness[strategy_key].append(sustainability_improvement)

        metrics_delta["harmony_improvement"] = harmony_improvement
        metrics_delta["sustainability_improvement"] = sustainability_improvement
        metrics_delta["harmony_level"] = tbl.get_harmony_level().value
        metrics_delta["sustainability_level"] = new_tbl_state.get_sustainability_level().value

        # TBL pillar final scores
        metrics_delta["profit_score"] = new_tbl_state.profit.score
        metrics_delta["people_score"] = new_tbl_state.people.score
        metrics_delta["planet_score"] = new_tbl_state.planet.score
        metrics_delta["tbl_balance"] = new_tbl_state.balance_score

        # Add TBL summary insight
        insights.append(
            f"TBL Status: Profit={new_tbl_state.profit.score:.2f}, "
            f"People={new_tbl_state.people.score:.2f}, "
            f"Planet={new_tbl_state.planet.score:.2f} "
            f"(Sustainability: {new_tbl_state.get_sustainability_level().value})"
        )

        self.corrections_applied.append({
            "timestamp": datetime.now().isoformat(),
            "operations": rebalance_count + resolution_count + coop_count + tbl_action_count + synergy_count,
            "harmony_improvement": harmony_improvement,
            "sustainability_improvement": sustainability_improvement
        })

        self.tbl_actions_applied.append({
            "timestamp": datetime.now().isoformat(),
            "tbl_actions": tbl_action_count,
            "synergies": synergy_count,
            "improvement": sustainability_improvement
        })

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    def get_tbl_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive Triple Bottom Line metrics.

        Returns a dictionary with all TBL-related metrics including:
        - Current pillar scores (Profit, People, Planet)
        - Sustainability index and level
        - Balance and synergy scores
        - Historical trends
        - Active tradeoffs
        """
        tbl = self.tbl_state
        return {
            "pillars": {
                "profit": {
                    "score": tbl.profit.score,
                    "trend": tbl.profit.trend,
                    "efficiency": tbl.profit.efficiency,
                    "resilience": tbl.profit.resilience,
                    "growth_potential": tbl.profit.growth_potential,
                    "details": tbl.profit.specific_metrics
                },
                "people": {
                    "score": tbl.people.score,
                    "trend": tbl.people.trend,
                    "efficiency": tbl.people.efficiency,
                    "resilience": tbl.people.resilience,
                    "growth_potential": tbl.people.growth_potential,
                    "details": tbl.people.specific_metrics
                },
                "planet": {
                    "score": tbl.planet.score,
                    "trend": tbl.planet.trend,
                    "efficiency": tbl.planet.efficiency,
                    "resilience": tbl.planet.resilience,
                    "growth_potential": tbl.planet.growth_potential,
                    "details": tbl.planet.specific_metrics
                }
            },
            "sustainability": {
                "index": tbl.sustainability_index,
                "level": tbl.get_sustainability_level().value,
                "balance_score": tbl.balance_score,
                "synergy_score": tbl.synergy_score,
                "conflict_score": tbl.conflict_score
            },
            "tradeoffs": {
                "active": [t.name for t in tbl.active_tradeoffs],
                "history_count": len(tbl.tradeoff_history)
            },
            "strategy": {
                "pillar_weights": {p.value: w for p, w in self.pillar_weights.items()},
                "target_sustainability": self.target_sustainability,
                "synergy_patterns_discovered": len(self.synergy_patterns)
            },
            "history": {
                "tbl_refinements": len(self.tbl_actions_applied),
                "total_synergies": sum(a.get("synergies", 0) for a in self.tbl_actions_applied),
                "avg_improvement": (
                    np.mean([a.get("improvement", 0) for a in self.tbl_actions_applied])
                    if self.tbl_actions_applied else 0
                )
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics including TBL metrics."""
        base_metrics = super().get_metrics() if hasattr(super(), 'get_metrics') else {}

        success_rate = self.successful_refinements / max(self.total_refinements, 1)
        avg_harmony = np.mean(self.harmony_contributions) if self.harmony_contributions else 0

        # Compute TBL-specific metrics
        tbl_success_rate = (
            len([a for a in self.tbl_actions_applied if a.get("improvement", 0) > 0]) /
            max(len(self.tbl_actions_applied), 1)
        )

        return {
            **base_metrics,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "total_refinements": self.total_refinements,
            "successful_refinements": self.successful_refinements,
            "success_rate": success_rate,
            "average_harmony_contribution": avg_harmony,
            "domains_affected": [d.value for d in self.domains_affected],
            "adaptation_weights": self.adaptation_weights.tolist(),
            "current_phase": self.current_phase.name,
            "pending_signals": len(self.cross_domain_signals),
            # TBL-specific metrics
            "tbl_metrics": self.get_tbl_metrics(),
            "tbl_success_rate": tbl_success_rate,
            "synergy_patterns": len(self.synergy_patterns),
            "current_sustainability_index": self.tbl_state.sustainability_index,
            "current_sustainability_level": self.tbl_state.get_sustainability_level().value
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main agent
    'HarmonizationAgent',
    # TBL Framework
    'TBLPillar',
    'SustainabilityLevel',
    'TBLTradeoffType',
    'PillarMetrics',
    'TBLState',
    'TBLRebalanceAction',
    # Harmony types
    'HarmonyVector',
    'BalanceCorrection',
]
