"""
Sexual Selection System

Implements mate choice and sexual selection mechanisms:
- Intersexual selection (mate choice)
- Intrasexual selection (competition)
- Ornament evolution
- Handicap principle
- Good genes hypothesis
- Runaway selection
- Mate quality assessment

Example:
    from evolution.sexual_selection import SexualSelectionEngine, MatePreference

    engine = SexualSelectionEngine()

    # Define mate preferences
    engine.add_preference(MatePreference(
        trait="display_brightness",
        preference_strength=0.8,
        optimal_value=0.9
    ))

    # Select mates
    chosen_mate = engine.choose_mate(
        chooser=female_genome,
        candidates=male_population
    )
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import random
import math
import copy
import uuid

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.events import get_event_bus
from .genetic_engine import Genome, Gene


class SelectionType(Enum):
    """Types of sexual selection"""
    INTERSEXUAL = auto()     # Mate choice
    INTRASEXUAL = auto()     # Competition
    MUTUAL = auto()          # Both sexes choose


class PreferenceType(Enum):
    """Types of mate preferences"""
    DIRECTIONAL = auto()     # Higher is better
    STABILIZING = auto()     # Optimal value is best
    DISRUPTIVE = auto()      # Extremes are preferred
    RARE_MALE = auto()       # Rare phenotypes preferred


@dataclass
class MatePreference:
    """
    Defines a preference for a specific trait.
    """
    trait: str                           # Gene name to evaluate
    preference_strength: float = 0.5     # 0-1, how important this preference is
    preference_type: PreferenceType = PreferenceType.DIRECTIONAL
    optimal_value: float = 1.0           # For stabilizing selection
    variance_tolerance: float = 0.2      # Acceptable deviation from optimal

    def evaluate(
        self,
        trait_value: float,
        population_mean: Optional[float] = None
    ) -> float:
        """
        Evaluate attractiveness of a trait value.

        Returns:
            Attractiveness score (0-1)
        """
        if self.preference_type == PreferenceType.DIRECTIONAL:
            return trait_value * self.preference_strength

        elif self.preference_type == PreferenceType.STABILIZING:
            deviation = abs(trait_value - self.optimal_value)
            score = math.exp(-deviation / self.variance_tolerance)
            return score * self.preference_strength

        elif self.preference_type == PreferenceType.DISRUPTIVE:
            # Prefer extremes
            extremeness = abs(trait_value - 0.5) * 2  # 0 at middle, 1 at extremes
            return extremeness * self.preference_strength

        elif self.preference_type == PreferenceType.RARE_MALE:
            if population_mean is None:
                return 0.5 * self.preference_strength
            # Prefer values far from population mean
            rarity = abs(trait_value - population_mean)
            return min(1.0, rarity * 2) * self.preference_strength

        return 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait": self.trait,
            "strength": self.preference_strength,
            "type": self.preference_type.name,
            "optimal": self.optimal_value
        }


@dataclass
class Ornament:
    """
    Sexual ornament that signals mate quality.

    Ornaments are costly traits that honestly signal genetic quality
    (handicap principle) or have been amplified by runaway selection.
    """
    name: str
    display_gene: str           # Gene controlling ornament expression
    cost_gene: str              # Gene affected by ornament cost
    cost_coefficient: float = 0.1  # How much maintaining ornament costs
    condition_dependent: bool = True  # Only express when healthy
    minimum_condition: float = 0.5    # Min fitness to express ornament

    def calculate_display_value(
        self,
        genome: Genome,
        fitness: float = 1.0
    ) -> float:
        """Calculate effective display value considering costs"""
        if self.display_gene not in genome.genes:
            return 0.0

        base_display = genome.express(self.display_gene)

        if self.condition_dependent:
            if fitness < self.minimum_condition:
                return 0.0  # Can't afford to display
            # Scale display by condition
            return base_display * (fitness / 1.0)

        return base_display

    def calculate_cost(
        self,
        genome: Genome
    ) -> float:
        """Calculate fitness cost of maintaining ornament"""
        if self.display_gene not in genome.genes:
            return 0.0

        display = genome.express(self.display_gene)
        return display * self.cost_coefficient


@dataclass
class CompetitionResult:
    """Result of intrasexual competition"""
    winner_id: str
    loser_id: str
    competition_trait: str
    winner_score: float
    loser_score: float
    injuries: Dict[str, float] = field(default_factory=dict)


class SexualSelectionEngine:
    """
    Engine for managing sexual selection processes.

    Features:
    - Mate preferences and choice
    - Competition mechanics
    - Ornament evolution
    - Quality signaling
    - Assortative mating

    Example:
        engine = SexualSelectionEngine()

        # Add preferences
        engine.add_preference(MatePreference(
            trait="display_brightness",
            preference_strength=0.8
        ))

        # Mate selection
        mate = engine.choose_mate(chooser, candidates)
    """

    def __init__(
        self,
        selection_type: SelectionType = SelectionType.INTERSEXUAL,
        choosiness: float = 0.5,          # How selective choosers are
        competition_intensity: float = 0.5,  # Intensity of competition
        assortative_strength: float = 0.0    # Preference for similar mates
    ):
        self.selection_type = selection_type
        self.choosiness = choosiness
        self.competition_intensity = competition_intensity
        self.assortative_strength = assortative_strength

        # Preferences
        self.preferences: List[MatePreference] = []

        # Ornaments
        self.ornaments: List[Ornament] = []

        # Competition traits
        self.competition_traits: List[str] = []

        # Statistics
        self.mate_choices: int = 0
        self.competitions: int = 0
        self.rejected_mates: int = 0

    def add_preference(self, preference: MatePreference) -> None:
        """Add a mate preference"""
        self.preferences.append(preference)

    def add_ornament(self, ornament: Ornament) -> None:
        """Add a sexual ornament"""
        self.ornaments.append(ornament)

    def add_competition_trait(self, trait_name: str) -> None:
        """Add a trait used in intrasexual competition"""
        self.competition_traits.append(trait_name)

    def calculate_attractiveness(
        self,
        candidate: Genome,
        population: Optional[List[Genome]] = None,
        fitness: float = 1.0
    ) -> float:
        """
        Calculate overall attractiveness of a candidate.

        Args:
            candidate: Genome to evaluate
            population: Population for calculating means (for rare-male effect)
            fitness: Current fitness of candidate

        Returns:
            Attractiveness score (0-1)
        """
        if not self.preferences and not self.ornaments:
            return 0.5

        scores = []
        weights = []

        # Calculate population means if needed
        pop_means: Dict[str, float] = {}
        if population:
            for pref in self.preferences:
                if pref.preference_type == PreferenceType.RARE_MALE:
                    values = [
                        p.express(pref.trait) for p in population
                        if pref.trait in p.genes
                    ]
                    if values:
                        pop_means[pref.trait] = sum(values) / len(values)

        # Evaluate preferences
        for pref in self.preferences:
            if pref.trait in candidate.genes:
                trait_value = candidate.express(pref.trait)
                pop_mean = pop_means.get(pref.trait)
                score = pref.evaluate(trait_value, pop_mean)
                scores.append(score)
                weights.append(pref.preference_strength)

        # Evaluate ornaments
        for ornament in self.ornaments:
            display = ornament.calculate_display_value(candidate, fitness)
            scores.append(display)
            weights.append(0.5)  # Ornaments have moderate weight

        if not scores:
            return 0.5

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return weighted_score

    def calculate_competitiveness(
        self,
        contestant: Genome,
        fitness: float = 1.0
    ) -> float:
        """
        Calculate competitive ability for intrasexual selection.
        """
        if not self.competition_traits:
            return fitness

        scores = []
        for trait in self.competition_traits:
            if trait in contestant.genes:
                scores.append(contestant.express(trait))

        if not scores:
            return fitness

        avg_trait = sum(scores) / len(scores)
        return avg_trait * fitness

    def choose_mate(
        self,
        chooser: Genome,
        candidates: List[Genome],
        candidate_fitness: Optional[Dict[str, float]] = None
    ) -> Optional[Genome]:
        """
        Choose a mate from candidates.

        Args:
            chooser: The genome doing the choosing
            candidates: List of potential mates
            candidate_fitness: Optional fitness scores for candidates

        Returns:
            Chosen mate or None if all rejected
        """
        if not candidates:
            return None

        self.mate_choices += 1

        # Calculate attractiveness for each candidate
        attractiveness = []
        for candidate in candidates:
            fitness = candidate_fitness.get(candidate.id, 1.0) if candidate_fitness else 1.0

            score = self.calculate_attractiveness(
                candidate,
                population=candidates,
                fitness=fitness
            )

            # Apply assortative mating preference
            if self.assortative_strength > 0:
                similarity = 1.0 - chooser.distance_to(candidate)
                score = score * (1 - self.assortative_strength) + similarity * self.assortative_strength

            attractiveness.append((candidate, score))

        # Sort by attractiveness
        attractiveness.sort(key=lambda x: x[1], reverse=True)

        # Choose based on choosiness
        # Higher choosiness = more likely to pick top candidate
        threshold = self.choosiness * attractiveness[0][1]

        for candidate, score in attractiveness:
            if score >= threshold:
                # Emit event
                bus = get_event_bus()
                bus.emit("sexual_selection.mate_chosen", {
                    "chooser_id": chooser.id,
                    "chosen_id": candidate.id,
                    "attractiveness": score
                })
                return candidate
            else:
                self.rejected_mates += 1

        # If very choosy and no one meets threshold, return best anyway
        return attractiveness[0][0] if attractiveness else None

    def compete(
        self,
        contestant1: Genome,
        contestant2: Genome,
        fitness1: float = 1.0,
        fitness2: float = 1.0
    ) -> CompetitionResult:
        """
        Run competition between two contestants.

        Returns:
            CompetitionResult with winner and details
        """
        self.competitions += 1

        score1 = self.calculate_competitiveness(contestant1, fitness1)
        score2 = self.calculate_competitiveness(contestant2, fitness2)

        # Add some randomness
        score1 += random.gauss(0, 0.1)
        score2 += random.gauss(0, 0.1)

        if score1 > score2:
            winner_id = contestant1.id
            loser_id = contestant2.id
            winner_score = score1
            loser_score = score2
        else:
            winner_id = contestant2.id
            loser_id = contestant1.id
            winner_score = score2
            loser_score = score1

        # Calculate injuries (higher competition intensity = more injuries)
        injuries = {}
        if self.competition_intensity > 0.5:
            injury_prob = (self.competition_intensity - 0.5) * 2
            if random.random() < injury_prob:
                injuries[loser_id] = random.uniform(0.1, 0.3)
            if random.random() < injury_prob * 0.3:
                injuries[winner_id] = random.uniform(0.05, 0.15)

        result = CompetitionResult(
            winner_id=winner_id,
            loser_id=loser_id,
            competition_trait=self.competition_traits[0] if self.competition_traits else "fitness",
            winner_score=winner_score,
            loser_score=loser_score,
            injuries=injuries
        )

        # Emit event
        bus = get_event_bus()
        bus.emit("sexual_selection.competition", {
            "winner_id": winner_id,
            "loser_id": loser_id,
            "trait": result.competition_trait
        })

        return result

    def tournament_selection(
        self,
        contestants: List[Genome],
        num_rounds: int = 1,
        fitness: Optional[Dict[str, float]] = None
    ) -> List[Genome]:
        """
        Run tournament competition to determine mating access.

        Returns:
            List of winners (in order of success)
        """
        if len(contestants) < 2:
            return contestants

        # Score all contestants
        scores: Dict[str, float] = {}
        for c in contestants:
            fit = fitness.get(c.id, 1.0) if fitness else 1.0
            scores[c.id] = self.calculate_competitiveness(c, fit)

        # Run tournament rounds
        for _ in range(num_rounds):
            # Shuffle for random pairings
            shuffled = contestants.copy()
            random.shuffle(shuffled)

            for i in range(0, len(shuffled) - 1, 2):
                c1, c2 = shuffled[i], shuffled[i + 1]
                result = self.compete(
                    c1, c2,
                    fitness.get(c1.id, 1.0) if fitness else 1.0,
                    fitness.get(c2.id, 1.0) if fitness else 1.0
                )
                # Winners get score bonus
                scores[result.winner_id] = scores.get(result.winner_id, 0) + 0.2

        # Sort by score
        contestants.sort(key=lambda c: scores.get(c.id, 0), reverse=True)
        return contestants

    def calculate_ornament_costs(
        self,
        population: List[Genome]
    ) -> Dict[str, float]:
        """
        Calculate ornament maintenance costs for population.

        Returns:
            Dict mapping genome ID to total ornament cost
        """
        costs = {}
        for genome in population:
            total_cost = 0.0
            for ornament in self.ornaments:
                total_cost += ornament.calculate_cost(genome)
            costs[genome.id] = total_cost
        return costs

    def select_breeding_pairs(
        self,
        males: List[Genome],
        females: List[Genome],
        male_fitness: Optional[Dict[str, float]] = None,
        female_fitness: Optional[Dict[str, float]] = None,
        num_pairs: Optional[int] = None
    ) -> List[Tuple[Genome, Genome]]:
        """
        Select breeding pairs based on sexual selection.

        Implements both mate choice and competition.

        Returns:
            List of (male, female) pairs
        """
        if not males or not females:
            return []

        num_pairs = num_pairs or min(len(males), len(females))
        pairs = []

        # If intrasexual, run competition first
        if self.selection_type in [SelectionType.INTRASEXUAL, SelectionType.MUTUAL]:
            males = self.tournament_selection(males, num_rounds=2, fitness=male_fitness)

        # Females choose from males
        available_males = males.copy()
        for female in females[:num_pairs]:
            if not available_males:
                break

            chosen = self.choose_mate(
                female,
                available_males,
                male_fitness
            )

            if chosen:
                pairs.append((chosen, female))
                # Remove chosen male (or not, depending on mating system)
                if len(available_males) > len(females) // 2:
                    available_males.remove(chosen)

        return pairs

    def get_stats(self) -> Dict[str, Any]:
        """Get sexual selection statistics"""
        return {
            "selection_type": self.selection_type.name,
            "preferences": [p.to_dict() for p in self.preferences],
            "ornaments": len(self.ornaments),
            "competition_traits": self.competition_traits,
            "total_mate_choices": self.mate_choices,
            "total_competitions": self.competitions,
            "rejected_mates": self.rejected_mates,
            "rejection_rate": self.rejected_mates / max(1, self.mate_choices)
        }


# Convenience functions

def create_peacock_selection() -> SexualSelectionEngine:
    """Create engine mimicking peacock-style selection"""
    engine = SexualSelectionEngine(
        selection_type=SelectionType.INTERSEXUAL,
        choosiness=0.8
    )

    # Elaborate display preference
    engine.add_preference(MatePreference(
        trait="display_brightness",
        preference_strength=0.9,
        preference_type=PreferenceType.DIRECTIONAL
    ))

    # Add costly ornament
    engine.add_ornament(Ornament(
        name="tail_display",
        display_gene="display_brightness",
        cost_gene="mobility",
        cost_coefficient=0.15,
        condition_dependent=True
    ))

    return engine


def create_combat_selection() -> SexualSelectionEngine:
    """Create engine for combat-based selection (like elk)"""
    engine = SexualSelectionEngine(
        selection_type=SelectionType.INTRASEXUAL,
        competition_intensity=0.8
    )

    engine.add_competition_trait("strength")
    engine.add_competition_trait("size")
    engine.add_competition_trait("aggression")

    return engine
