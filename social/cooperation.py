"""
Cooperation and Game Theory System for NPCPU

Implements cooperation evolution with:
- Classic game theory strategies (TFT, GRIM, etc.)
- Iterated prisoner's dilemma
- Public goods games
- Reciprocity tracking
- Altruism and kin selection

Example:
    from social.cooperation import CooperationEngine, Strategy, Game

    # Create engine
    engine = CooperationEngine()

    # Register organisms with strategies
    engine.register_organism("alice", Strategy.TIT_FOR_TAT)
    engine.register_organism("bob", Strategy.ALWAYS_COOPERATE)

    # Play a round
    result = engine.play_round("alice", "bob", Game.PRISONERS_DILEMMA)
    print(f"Alice: {result.payoff1}, Bob: {result.payoff2}")
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


class Action(Enum):
    """Actions in cooperative games"""
    COOPERATE = "cooperate"
    DEFECT = "defect"


class Strategy(Enum):
    """Cooperation strategies"""
    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    TIT_FOR_TAT = "tit_for_tat"
    GENEROUS_TIT_FOR_TAT = "generous_tit_for_tat"
    SUSPICIOUS_TIT_FOR_TAT = "suspicious_tit_for_tat"
    TIT_FOR_TWO_TATS = "tit_for_two_tats"
    GRIM_TRIGGER = "grim_trigger"
    PAVLOV = "pavlov"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class Game(Enum):
    """Types of cooperative games"""
    PRISONERS_DILEMMA = "prisoners_dilemma"
    STAG_HUNT = "stag_hunt"
    CHICKEN = "chicken"
    PUBLIC_GOODS = "public_goods"
    ULTIMATUM = "ultimatum"


@dataclass
class GamePayoffs:
    """
    Payoff matrix for a 2-player game.

    Payoffs: (row_player, column_player)
    """
    mutual_cooperate: Tuple[float, float] = (3.0, 3.0)     # R, R
    cooperate_defect: Tuple[float, float] = (0.0, 5.0)     # S, T
    defect_cooperate: Tuple[float, float] = (5.0, 0.0)     # T, S
    mutual_defect: Tuple[float, float] = (1.0, 1.0)        # P, P


@dataclass
class GameResult:
    """
    Result of a game round.

    Attributes:
        game: Type of game played
        player1_id: First player
        player2_id: Second player
        action1: Player 1's action
        action2: Player 2's action
        payoff1: Player 1's payoff
        payoff2: Player 2's payoff
        round_number: Which round this was
    """
    game: Game
    player1_id: str
    player2_id: str
    action1: Action
    action2: Action
    payoff1: float
    payoff2: float
    round_number: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CooperationProfile:
    """
    Cooperation profile for an organism.

    Tracks cooperation behavior and outcomes.
    """
    organism_id: str
    strategy: Strategy = Strategy.TIT_FOR_TAT
    cooperation_rate: float = 0.5
    total_games: int = 0
    total_payoff: float = 0.0
    cooperation_count: int = 0
    defection_count: int = 0
    reputation: float = 0.5  # How cooperative others think this organism is

    # History with specific partners
    partner_history: Dict[str, List[Action]] = field(default_factory=dict)

    # For GRIM strategy
    triggered_grim: Set[str] = field(default_factory=set)

    def record_action(self, action: Action, partner_id: str) -> None:
        """Record an action taken"""
        if partner_id not in self.partner_history:
            self.partner_history[partner_id] = []
        self.partner_history[partner_id].append(action)

        if action == Action.COOPERATE:
            self.cooperation_count += 1
        else:
            self.defection_count += 1

        total = self.cooperation_count + self.defection_count
        if total > 0:
            self.cooperation_rate = self.cooperation_count / total

    def get_partner_last_action(self, partner_id: str) -> Optional[Action]:
        """Get the last action a partner took"""
        history = self.partner_history.get(partner_id, [])
        return history[-1] if history else None

    def get_partner_history(self, partner_id: str) -> List[Action]:
        """Get full history with a partner"""
        return self.partner_history.get(partner_id, [])


class CooperationEngine:
    """
    Manages cooperative game interactions.

    Provides:
    - Classic game theory games
    - Strategy evolution
    - Reciprocity tracking
    - Reputation dynamics
    - Kin-based altruism

    Example:
        engine = CooperationEngine()

        # Register players
        engine.register_organism("p1", Strategy.TIT_FOR_TAT)
        engine.register_organism("p2", Strategy.GRIM_TRIGGER)

        # Play multiple rounds
        for i in range(10):
            result = engine.play_round("p1", "p2")

        # Check outcomes
        stats = engine.get_cooperation_stats("p1")
    """

    def __init__(
        self,
        default_game: Game = Game.PRISONERS_DILEMMA,
        mutation_rate: float = 0.01,
        emit_events: bool = True
    ):
        """
        Initialize the cooperation engine.

        Args:
            default_game: Default game type
            mutation_rate: Strategy mutation probability
            emit_events: Whether to emit events
        """
        self._default_game = default_game
        self._mutation_rate = mutation_rate
        self._emit_events = emit_events

        # Organism profiles
        self._profiles: Dict[str, CooperationProfile] = {}

        # Game history
        self._game_history: List[GameResult] = []

        # Payoff matrices for different games
        self._payoffs: Dict[Game, GamePayoffs] = {
            Game.PRISONERS_DILEMMA: GamePayoffs(
                mutual_cooperate=(3.0, 3.0),
                cooperate_defect=(0.0, 5.0),
                defect_cooperate=(5.0, 0.0),
                mutual_defect=(1.0, 1.0)
            ),
            Game.STAG_HUNT: GamePayoffs(
                mutual_cooperate=(4.0, 4.0),
                cooperate_defect=(0.0, 3.0),
                defect_cooperate=(3.0, 0.0),
                mutual_defect=(2.0, 2.0)
            ),
            Game.CHICKEN: GamePayoffs(
                mutual_cooperate=(3.0, 3.0),
                cooperate_defect=(1.0, 5.0),
                defect_cooperate=(5.0, 1.0),
                mutual_defect=(0.0, 0.0)
            )
        }

        # Round counter per pair
        self._round_counter: Dict[Tuple[str, str], int] = defaultdict(int)

    # =========================================================================
    # Registration
    # =========================================================================

    def register_organism(
        self,
        organism_id: str,
        strategy: Strategy = Strategy.TIT_FOR_TAT
    ) -> CooperationProfile:
        """
        Register an organism with a cooperation strategy.

        Args:
            organism_id: Organism identifier
            strategy: Cooperation strategy

        Returns:
            CooperationProfile for the organism
        """
        if organism_id in self._profiles:
            self._profiles[organism_id].strategy = strategy
            return self._profiles[organism_id]

        profile = CooperationProfile(
            organism_id=organism_id,
            strategy=strategy
        )
        self._profiles[organism_id] = profile
        return profile

    def get_profile(self, organism_id: str) -> CooperationProfile:
        """Get or create cooperation profile"""
        if organism_id not in self._profiles:
            return self.register_organism(organism_id)
        return self._profiles[organism_id]

    def set_strategy(self, organism_id: str, strategy: Strategy) -> None:
        """Set an organism's strategy"""
        profile = self.get_profile(organism_id)
        profile.strategy = strategy

    # =========================================================================
    # Game Play
    # =========================================================================

    def play_round(
        self,
        player1_id: str,
        player2_id: str,
        game: Optional[Game] = None
    ) -> GameResult:
        """
        Play one round of a game between two organisms.

        Args:
            player1_id: First player
            player2_id: Second player
            game: Game type (uses default if not specified)

        Returns:
            GameResult with outcomes
        """
        if game is None:
            game = self._default_game

        profile1 = self.get_profile(player1_id)
        profile2 = self.get_profile(player2_id)

        # Get pair key and increment round
        pair_key = tuple(sorted([player1_id, player2_id]))
        self._round_counter[pair_key] += 1
        round_num = self._round_counter[pair_key]

        # Determine actions based on strategies
        action1 = self._get_action(profile1, profile2)
        action2 = self._get_action(profile2, profile1)

        # Calculate payoffs
        payoff1, payoff2 = self._get_payoffs(game, action1, action2)

        # Record results
        profile1.record_action(action1, player2_id)
        profile2.record_action(action2, player1_id)
        profile1.total_games += 1
        profile2.total_games += 1
        profile1.total_payoff += payoff1
        profile2.total_payoff += payoff2

        # Update reputations
        self._update_reputation(profile1, action1)
        self._update_reputation(profile2, action2)

        # Handle GRIM trigger
        if action1 == Action.DEFECT:
            if profile2.strategy == Strategy.GRIM_TRIGGER:
                profile2.triggered_grim.add(player1_id)
        if action2 == Action.DEFECT:
            if profile1.strategy == Strategy.GRIM_TRIGGER:
                profile1.triggered_grim.add(player2_id)

        result = GameResult(
            game=game,
            player1_id=player1_id,
            player2_id=player2_id,
            action1=action1,
            action2=action2,
            payoff1=payoff1,
            payoff2=payoff2,
            round_number=round_num
        )

        self._game_history.append(result)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("cooperation.game_played", {
                "player1": player1_id,
                "player2": player2_id,
                "action1": action1.value,
                "action2": action2.value,
                "payoff1": payoff1,
                "payoff2": payoff2
            })

        return result

    def _get_action(
        self,
        actor: CooperationProfile,
        opponent: CooperationProfile
    ) -> Action:
        """Determine action based on strategy"""
        strategy = actor.strategy
        opponent_id = opponent.organism_id

        if strategy == Strategy.ALWAYS_COOPERATE:
            return Action.COOPERATE

        elif strategy == Strategy.ALWAYS_DEFECT:
            return Action.DEFECT

        elif strategy == Strategy.TIT_FOR_TAT:
            # Cooperate first, then mirror
            last = actor.get_partner_last_action(opponent_id)
            return last if last else Action.COOPERATE

        elif strategy == Strategy.GENEROUS_TIT_FOR_TAT:
            last = actor.get_partner_last_action(opponent_id)
            if last == Action.DEFECT:
                # 10% chance to forgive
                return Action.COOPERATE if random.random() < 0.1 else Action.DEFECT
            return Action.COOPERATE

        elif strategy == Strategy.SUSPICIOUS_TIT_FOR_TAT:
            # Defect first, then mirror
            last = actor.get_partner_last_action(opponent_id)
            return last if last else Action.DEFECT

        elif strategy == Strategy.TIT_FOR_TWO_TATS:
            history = actor.get_partner_history(opponent_id)
            if len(history) >= 2:
                if history[-1] == Action.DEFECT and history[-2] == Action.DEFECT:
                    return Action.DEFECT
            return Action.COOPERATE

        elif strategy == Strategy.GRIM_TRIGGER:
            if opponent_id in actor.triggered_grim:
                return Action.DEFECT
            return Action.COOPERATE

        elif strategy == Strategy.PAVLOV:
            # Win-stay, lose-shift
            last_own = actor.get_partner_history(opponent_id)
            last_opponent = opponent.get_partner_history(actor.organism_id)

            if not last_own or not last_opponent:
                return Action.COOPERATE

            # If we got a good payoff last time, repeat
            if last_own[-1] == last_opponent[-1]:  # Both same = mutual C or D
                return last_own[-1]
            else:
                # Switch
                return Action.COOPERATE if last_own[-1] == Action.DEFECT else Action.DEFECT

        elif strategy == Strategy.RANDOM:
            return Action.COOPERATE if random.random() < 0.5 else Action.DEFECT

        elif strategy == Strategy.ADAPTIVE:
            # Cooperate with cooperators, defect against defectors
            if opponent.cooperation_rate > 0.6:
                return Action.COOPERATE
            elif opponent.cooperation_rate < 0.4:
                return Action.DEFECT
            else:
                return Action.COOPERATE if random.random() < 0.5 else Action.DEFECT

        return Action.COOPERATE  # Default

    def _get_payoffs(
        self,
        game: Game,
        action1: Action,
        action2: Action
    ) -> Tuple[float, float]:
        """Get payoffs for action combination"""
        matrix = self._payoffs.get(game)
        if not matrix:
            matrix = self._payoffs[Game.PRISONERS_DILEMMA]

        if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
            return matrix.mutual_cooperate
        elif action1 == Action.COOPERATE and action2 == Action.DEFECT:
            return matrix.cooperate_defect
        elif action1 == Action.DEFECT and action2 == Action.COOPERATE:
            return matrix.defect_cooperate
        else:
            return matrix.mutual_defect

    def _update_reputation(self, profile: CooperationProfile, action: Action) -> None:
        """Update reputation based on action"""
        if action == Action.COOPERATE:
            profile.reputation = min(1.0, profile.reputation + 0.02)
        else:
            profile.reputation = max(0.0, profile.reputation - 0.03)

    # =========================================================================
    # Multi-player Games
    # =========================================================================

    def play_public_goods_game(
        self,
        players: List[str],
        multiplier: float = 2.0,
        endowment: float = 10.0
    ) -> Dict[str, float]:
        """
        Play a public goods game.

        Args:
            players: List of player IDs
            multiplier: Multiplier for public pool
            endowment: Starting amount each player has

        Returns:
            Dictionary of player_id -> final payoff
        """
        n = len(players)
        if n < 2:
            return {}

        contributions: Dict[str, float] = {}
        actions: Dict[str, Action] = {}

        for player_id in players:
            profile = self.get_profile(player_id)

            # Decide whether to contribute (cooperate = full, defect = nothing)
            action = self._decide_public_goods(profile)
            actions[player_id] = action

            if action == Action.COOPERATE:
                contributions[player_id] = endowment
            else:
                contributions[player_id] = 0.0

        # Calculate public pool
        total_contributed = sum(contributions.values())
        public_return = total_contributed * multiplier / n

        # Calculate individual payoffs
        payoffs: Dict[str, float] = {}
        for player_id in players:
            kept = endowment - contributions[player_id]
            payoffs[player_id] = kept + public_return

            # Update profile
            profile = self._profiles[player_id]
            profile.total_games += 1
            profile.total_payoff += payoffs[player_id]
            self._update_reputation(profile, actions[player_id])

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("cooperation.public_goods", {
                "players": players,
                "total_contributed": total_contributed,
                "public_return": public_return,
                "payoffs": payoffs
            })

        return payoffs

    def _decide_public_goods(self, profile: CooperationProfile) -> Action:
        """Decide whether to contribute in public goods game"""
        strategy = profile.strategy

        if strategy == Strategy.ALWAYS_COOPERATE:
            return Action.COOPERATE
        elif strategy == Strategy.ALWAYS_DEFECT:
            return Action.DEFECT
        elif strategy in (Strategy.TIT_FOR_TAT, Strategy.GENEROUS_TIT_FOR_TAT):
            # Base on recent cooperation rate of others
            return Action.COOPERATE if profile.reputation > 0.4 else Action.DEFECT
        elif strategy == Strategy.RANDOM:
            return Action.COOPERATE if random.random() < 0.5 else Action.DEFECT
        else:
            return Action.COOPERATE if random.random() < profile.cooperation_rate else Action.DEFECT

    # =========================================================================
    # Kin-based Altruism
    # =========================================================================

    def calculate_altruism_threshold(
        self,
        actor_id: str,
        recipient_id: str,
        relatedness: float,
        cost: float,
        benefit: float
    ) -> bool:
        """
        Check if altruistic act should occur (Hamilton's Rule).

        Hamilton's Rule: r*B > C

        Args:
            actor_id: Who would perform the act
            recipient_id: Who would benefit
            relatedness: Coefficient of relatedness (r)
            cost: Cost to actor (C)
            benefit: Benefit to recipient (B)

        Returns:
            True if altruism is beneficial
        """
        return relatedness * benefit > cost

    def perform_altruistic_act(
        self,
        actor_id: str,
        recipient_id: str,
        cost: float,
        benefit: float,
        relatedness: float = 0.0
    ) -> bool:
        """
        Perform an altruistic act if conditions are met.

        Returns:
            True if act was performed
        """
        should_act = self.calculate_altruism_threshold(
            actor_id, recipient_id, relatedness, cost, benefit
        )

        if should_act:
            actor = self.get_profile(actor_id)
            recipient = self.get_profile(recipient_id)

            actor.total_payoff -= cost
            recipient.total_payoff += benefit

            # Boost reputation
            actor.reputation = min(1.0, actor.reputation + 0.05)

            if self._emit_events:
                bus = get_event_bus()
                bus.emit("cooperation.altruism", {
                    "actor_id": actor_id,
                    "recipient_id": recipient_id,
                    "cost": cost,
                    "benefit": benefit,
                    "relatedness": relatedness
                })

        return should_act

    # =========================================================================
    # Reciprocity
    # =========================================================================

    def calculate_reciprocity_score(
        self,
        organism1_id: str,
        organism2_id: str
    ) -> float:
        """
        Calculate how reciprocal the relationship is.

        Returns:
            Score from -1 (one-sided exploitation) to 1 (perfect reciprocity)
        """
        p1 = self._profiles.get(organism1_id)
        p2 = self._profiles.get(organism2_id)

        if not p1 or not p2:
            return 0.0

        h1 = p1.get_partner_history(organism2_id)
        h2 = p2.get_partner_history(organism1_id)

        if not h1 or not h2:
            return 0.0

        coop1 = sum(1 for a in h1 if a == Action.COOPERATE) / len(h1)
        coop2 = sum(1 for a in h2 if a == Action.COOPERATE) / len(h2)

        # Perfect reciprocity when both have same cooperation rate
        diff = abs(coop1 - coop2)
        return 1.0 - diff

    def get_most_cooperative_partners(
        self,
        organism_id: str,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Get partners ranked by cooperation"""
        profile = self._profiles.get(organism_id)
        if not profile:
            return []

        partner_coop = []
        for partner_id, history in profile.partner_history.items():
            if history:
                coop_rate = sum(1 for a in history if a == Action.COOPERATE) / len(history)
                partner_coop.append((partner_id, coop_rate))

        partner_coop.sort(key=lambda x: x[1], reverse=True)
        return partner_coop[:limit]

    # =========================================================================
    # Strategy Evolution
    # =========================================================================

    def evolve_strategies(self, selection_pressure: float = 0.3) -> Dict[str, Strategy]:
        """
        Evolve strategies based on success.

        Low performers may adopt strategies of high performers.

        Returns:
            Dictionary of organism_id -> new_strategy
        """
        if not self._profiles:
            return {}

        # Calculate average payoffs
        avg_payoffs = []
        for org_id, profile in self._profiles.items():
            if profile.total_games > 0:
                avg = profile.total_payoff / profile.total_games
                avg_payoffs.append((org_id, avg, profile.strategy))

        if not avg_payoffs:
            return {}

        avg_payoffs.sort(key=lambda x: x[1], reverse=True)

        changes = {}
        n = len(avg_payoffs)

        # Bottom performers may switch
        bottom_third = n // 3
        top_strategies = [s for _, _, s in avg_payoffs[:max(1, n // 3)]]

        for i, (org_id, _, strategy) in enumerate(avg_payoffs):
            if i >= n - bottom_third:
                if random.random() < selection_pressure:
                    new_strategy = random.choice(top_strategies)
                    if new_strategy != strategy:
                        self._profiles[org_id].strategy = new_strategy
                        changes[org_id] = new_strategy

            # Random mutation
            if random.random() < self._mutation_rate:
                new_strategy = random.choice(list(Strategy))
                self._profiles[org_id].strategy = new_strategy
                changes[org_id] = new_strategy

        if changes and self._emit_events:
            bus = get_event_bus()
            bus.emit("cooperation.strategies_evolved", {
                "changes": {k: v.value for k, v in changes.items()}
            })

        return changes

    # =========================================================================
    # Queries
    # =========================================================================

    def get_cooperation_stats(self, organism_id: str) -> Dict[str, Any]:
        """Get cooperation statistics for an organism"""
        profile = self._profiles.get(organism_id)
        if not profile:
            return {}

        return {
            "strategy": profile.strategy.value,
            "cooperation_rate": profile.cooperation_rate,
            "total_games": profile.total_games,
            "total_payoff": profile.total_payoff,
            "avg_payoff": profile.total_payoff / profile.total_games if profile.total_games > 0 else 0,
            "reputation": profile.reputation,
            "unique_partners": len(profile.partner_history)
        }

    def get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies in population"""
        distribution = defaultdict(int)
        for profile in self._profiles.values():
            distribution[profile.strategy.value] += 1
        return dict(distribution)

    def get_overall_cooperation_rate(self) -> float:
        """Get overall cooperation rate across all games"""
        total_coop = sum(p.cooperation_count for p in self._profiles.values())
        total_actions = sum(
            p.cooperation_count + p.defection_count
            for p in self._profiles.values()
        )
        return total_coop / total_actions if total_actions > 0 else 0.5

    def remove_organism(self, organism_id: str) -> None:
        """Remove an organism from the cooperation system"""
        self._profiles.pop(organism_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get overall cooperation statistics"""
        return {
            "total_organisms": len(self._profiles),
            "total_games": len(self._game_history),
            "overall_cooperation_rate": self.get_overall_cooperation_rate(),
            "strategy_distribution": self.get_strategy_distribution(),
            "avg_reputation": (
                sum(p.reputation for p in self._profiles.values()) / len(self._profiles)
                if self._profiles else 0
            )
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cooperation engine"""
        profiles = [
            {
                "organism_id": p.organism_id,
                "strategy": p.strategy.value,
                "cooperation_rate": p.cooperation_rate,
                "total_games": p.total_games,
                "total_payoff": p.total_payoff,
                "reputation": p.reputation
            }
            for p in self._profiles.values()
        ]

        return {
            "profiles": profiles,
            "default_game": self._default_game.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'CooperationEngine':
        """Deserialize cooperation engine"""
        engine = cls(
            default_game=Game(data.get("default_game", "prisoners_dilemma")),
            emit_events=emit_events
        )

        for p_data in data.get("profiles", []):
            profile = CooperationProfile(
                organism_id=p_data["organism_id"],
                strategy=Strategy(p_data["strategy"]),
                cooperation_rate=p_data.get("cooperation_rate", 0.5),
                total_games=p_data.get("total_games", 0),
                total_payoff=p_data.get("total_payoff", 0),
                reputation=p_data.get("reputation", 0.5)
            )
            engine._profiles[profile.organism_id] = profile

        return engine
