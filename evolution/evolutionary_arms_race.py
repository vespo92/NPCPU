"""
Evolutionary Arms Race System

Implements co-evolutionary dynamics between competing species:
- Predator-prey coevolution
- Host-parasite dynamics
- Red Queen hypothesis
- Escalation and counter-adaptation
- Attack-defense trait coupling
- Oscillating selection

Example:
    from evolution.evolutionary_arms_race import ArmsRaceEngine, CompetitorRole

    engine = ArmsRaceEngine()

    # Define predator-prey relationship
    engine.add_relationship(
        predator_species="wolf",
        prey_species="deer",
        attack_trait="speed",
        defense_trait="speed"
    )

    # Process coevolution
    engine.process_coevolution(predator_pop, prey_pop, generation)
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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
from .genetic_engine import Genome


class CompetitorRole(Enum):
    """Role in an arms race"""
    PREDATOR = auto()
    PREY = auto()
    HOST = auto()
    PARASITE = auto()
    COMPETITOR = auto()


class InteractionType(Enum):
    """Type of antagonistic interaction"""
    PREDATION = auto()       # Predator-prey
    PARASITISM = auto()      # Host-parasite
    COMPETITION = auto()     # Interspecific competition
    MIMICRY = auto()         # Mimicry arms race


@dataclass
class TraitPair:
    """
    A pair of co-evolving traits.

    One trait attacks, one defends. They escalate in response to each other.
    """
    attack_trait: str
    defense_trait: str
    attack_effectiveness: float = 0.5  # How well attack overcomes defense
    escalation_rate: float = 0.1       # Rate of trait escalation
    cost_coefficient: float = 0.05     # Fitness cost of high trait values

    def calculate_outcome(
        self,
        attack_value: float,
        defense_value: float
    ) -> float:
        """
        Calculate interaction outcome.

        Returns:
            Probability that attack succeeds (0-1)
        """
        # Logistic function based on trait difference
        diff = (attack_value - defense_value) * self.attack_effectiveness
        return 1.0 / (1.0 + math.exp(-diff * 5))

    def calculate_trait_cost(self, trait_value: float) -> float:
        """Calculate fitness cost of maintaining high trait value"""
        return trait_value ** 2 * self.cost_coefficient


@dataclass
class ArmsRaceRelationship:
    """
    Defines an arms race between two species/groups.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    interaction_type: InteractionType = InteractionType.PREDATION
    species1_id: str = ""      # e.g., predator
    species2_id: str = ""      # e.g., prey
    role1: CompetitorRole = CompetitorRole.PREDATOR
    role2: CompetitorRole = CompetitorRole.PREY
    trait_pairs: List[TraitPair] = field(default_factory=list)
    generation_started: int = 0
    total_interactions: int = 0
    species1_wins: int = 0
    species2_wins: int = 0

    @property
    def win_ratio(self) -> float:
        """Ratio of species1 wins to total"""
        if self.total_interactions == 0:
            return 0.5
        return self.species1_wins / self.total_interactions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.interaction_type.name,
            "species1": self.species1_id,
            "species2": self.species2_id,
            "role1": self.role1.name,
            "role2": self.role2.name,
            "trait_pairs": [
                {"attack": tp.attack_trait, "defense": tp.defense_trait}
                for tp in self.trait_pairs
            ],
            "total_interactions": self.total_interactions,
            "win_ratio": self.win_ratio
        }


@dataclass
class InteractionRecord:
    """Record of a single interaction event"""
    relationship_id: str
    attacker_id: str
    defender_id: str
    success: bool
    attack_value: float
    defense_value: float
    generation: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship": self.relationship_id,
            "attacker": self.attacker_id,
            "defender": self.defender_id,
            "success": self.success,
            "attack": self.attack_value,
            "defense": self.defense_value,
            "generation": self.generation
        }


class ArmsRaceEngine:
    """
    Engine for managing evolutionary arms races.

    Features:
    - Multiple relationship types
    - Trait coevolution
    - Oscillating selection pressure
    - Escalation dynamics
    - Fitness cost tracking

    Example:
        engine = ArmsRaceEngine()

        # Create predator-prey relationship
        engine.add_predation_relationship(
            predator_species="wolf",
            prey_species="deer",
            trait_pairs=[
                TraitPair("speed", "speed"),
                TraitPair("stealth", "vigilance")
            ]
        )

        # Simulate interactions
        outcomes = engine.simulate_interactions(
            predator_pop, prey_pop, num_interactions=100
        )
    """

    def __init__(
        self,
        base_escalation_rate: float = 0.1,
        red_queen_intensity: float = 0.5,
        enable_oscillation: bool = True
    ):
        self.base_escalation_rate = base_escalation_rate
        self.red_queen_intensity = red_queen_intensity
        self.enable_oscillation = enable_oscillation

        # Relationships
        self.relationships: Dict[str, ArmsRaceRelationship] = {}

        # Interaction history
        self.interaction_history: List[InteractionRecord] = []

        # Trait escalation tracking
        self.escalation_history: Dict[str, List[Tuple[int, float]]] = {}

        # Statistics
        self.total_interactions = 0

    def add_relationship(
        self,
        species1_id: str,
        species2_id: str,
        interaction_type: InteractionType,
        role1: CompetitorRole,
        role2: CompetitorRole,
        trait_pairs: List[TraitPair],
        generation: int = 0
    ) -> ArmsRaceRelationship:
        """Add a new arms race relationship"""
        relationship = ArmsRaceRelationship(
            interaction_type=interaction_type,
            species1_id=species1_id,
            species2_id=species2_id,
            role1=role1,
            role2=role2,
            trait_pairs=trait_pairs,
            generation_started=generation
        )

        self.relationships[relationship.id] = relationship

        # Emit event
        bus = get_event_bus()
        bus.emit("arms_race.relationship_added", {
            "id": relationship.id,
            "species1": species1_id,
            "species2": species2_id,
            "type": interaction_type.name
        })

        return relationship

    def add_predation_relationship(
        self,
        predator_species: str,
        prey_species: str,
        trait_pairs: Optional[List[TraitPair]] = None,
        generation: int = 0
    ) -> ArmsRaceRelationship:
        """Convenience method to add predator-prey relationship"""
        if trait_pairs is None:
            trait_pairs = [
                TraitPair("speed", "speed", attack_effectiveness=0.5),
                TraitPair("stealth", "vigilance", attack_effectiveness=0.6)
            ]

        return self.add_relationship(
            species1_id=predator_species,
            species2_id=prey_species,
            interaction_type=InteractionType.PREDATION,
            role1=CompetitorRole.PREDATOR,
            role2=CompetitorRole.PREY,
            trait_pairs=trait_pairs,
            generation=generation
        )

    def add_parasitism_relationship(
        self,
        parasite_species: str,
        host_species: str,
        trait_pairs: Optional[List[TraitPair]] = None,
        generation: int = 0
    ) -> ArmsRaceRelationship:
        """Convenience method to add host-parasite relationship"""
        if trait_pairs is None:
            trait_pairs = [
                TraitPair("infectivity", "resistance", attack_effectiveness=0.6),
                TraitPair("evasion", "immunity", attack_effectiveness=0.5)
            ]

        return self.add_relationship(
            species1_id=parasite_species,
            species2_id=host_species,
            interaction_type=InteractionType.PARASITISM,
            role1=CompetitorRole.PARASITE,
            role2=CompetitorRole.HOST,
            trait_pairs=trait_pairs,
            generation=generation
        )

    def simulate_interaction(
        self,
        attacker: Genome,
        defender: Genome,
        relationship: ArmsRaceRelationship,
        generation: int = 0
    ) -> InteractionRecord:
        """
        Simulate a single interaction between attacker and defender.

        Returns:
            InteractionRecord with outcome
        """
        total_attack = 0.0
        total_defense = 0.0
        success_prob = 0.5

        for trait_pair in relationship.trait_pairs:
            # Get trait values
            attack_val = 0.5
            defense_val = 0.5

            if trait_pair.attack_trait in attacker.genes:
                attack_val = attacker.express(trait_pair.attack_trait)

            if trait_pair.defense_trait in defender.genes:
                defense_val = defender.express(trait_pair.defense_trait)

            total_attack += attack_val
            total_defense += defense_val

            # Calculate success probability
            prob = trait_pair.calculate_outcome(attack_val, defense_val)
            success_prob = success_prob * 0.3 + prob * 0.7  # Weighted average

        # Determine outcome
        success = random.random() < success_prob

        # Update statistics
        relationship.total_interactions += 1
        if success:
            relationship.species1_wins += 1
        else:
            relationship.species2_wins += 1

        record = InteractionRecord(
            relationship_id=relationship.id,
            attacker_id=attacker.id,
            defender_id=defender.id,
            success=success,
            attack_value=total_attack / max(1, len(relationship.trait_pairs)),
            defense_value=total_defense / max(1, len(relationship.trait_pairs)),
            generation=generation
        )

        self.interaction_history.append(record)
        self.total_interactions += 1

        return record

    def simulate_interactions(
        self,
        attackers: List[Genome],
        defenders: List[Genome],
        relationship_id: str,
        num_interactions: Optional[int] = None,
        generation: int = 0
    ) -> List[InteractionRecord]:
        """
        Simulate multiple interactions between populations.

        Returns:
            List of interaction records
        """
        if relationship_id not in self.relationships:
            return []

        relationship = self.relationships[relationship_id]

        if num_interactions is None:
            num_interactions = min(len(attackers), len(defenders)) * 2

        records = []
        for _ in range(num_interactions):
            if not attackers or not defenders:
                break

            attacker = random.choice(attackers)
            defender = random.choice(defenders)

            record = self.simulate_interaction(
                attacker, defender, relationship, generation
            )
            records.append(record)

        return records

    def calculate_selection_pressure(
        self,
        genome: Genome,
        role: CompetitorRole,
        relationship: ArmsRaceRelationship,
        win_rate: float
    ) -> Dict[str, float]:
        """
        Calculate selection pressure on traits based on recent outcomes.

        Returns:
            Dict mapping trait names to selection pressure values
        """
        pressures = {}

        for trait_pair in relationship.trait_pairs:
            if role in [CompetitorRole.PREDATOR, CompetitorRole.PARASITE]:
                # Attackers
                trait = trait_pair.attack_trait
                if trait in genome.genes:
                    # More pressure if losing
                    pressure = (0.5 - win_rate) * self.red_queen_intensity
                    pressures[trait] = pressure
            else:
                # Defenders
                trait = trait_pair.defense_trait
                if trait in genome.genes:
                    # More pressure if prey is losing (low win rate for predator)
                    pressure = (win_rate - 0.5) * self.red_queen_intensity
                    pressures[trait] = pressure

        return pressures

    def apply_escalation(
        self,
        population: List[Genome],
        pressures: Dict[str, float],
        generation: int
    ) -> int:
        """
        Apply escalation pressure to population traits.

        Traits under pressure have increased chance of beneficial mutations.

        Returns:
            Number of traits escalated
        """
        escalated = 0

        for genome in population:
            for trait, pressure in pressures.items():
                if trait not in genome.genes:
                    continue

                # Escalation probability based on pressure
                if random.random() < abs(pressure) * self.base_escalation_rate:
                    gene = genome._genes[trait]
                    current = gene.express()

                    # Direction based on pressure sign
                    delta = 0.05 if pressure > 0 else -0.05

                    # Apply to alleles
                    gene.allele1.value = max(0, min(1, gene.allele1.value + delta))
                    gene.allele2.value = max(0, min(1, gene.allele2.value + delta))

                    escalated += 1

                    # Track escalation
                    if trait not in self.escalation_history:
                        self.escalation_history[trait] = []
                    self.escalation_history[trait].append((generation, current + delta))

        return escalated

    def calculate_arms_race_costs(
        self,
        population: List[Genome]
    ) -> Dict[str, float]:
        """
        Calculate fitness costs from maintaining arms race traits.

        Returns:
            Dict mapping genome ID to total cost
        """
        costs = {}

        for genome in population:
            total_cost = 0.0

            for relationship in self.relationships.values():
                for trait_pair in relationship.trait_pairs:
                    # Attack trait cost
                    if trait_pair.attack_trait in genome.genes:
                        val = genome.express(trait_pair.attack_trait)
                        total_cost += trait_pair.calculate_trait_cost(val)

                    # Defense trait cost
                    if trait_pair.defense_trait in genome.genes:
                        val = genome.express(trait_pair.defense_trait)
                        total_cost += trait_pair.calculate_trait_cost(val)

            costs[genome.id] = total_cost

        return costs

    def detect_oscillation(
        self,
        trait: str,
        window: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Detect oscillating selection pattern in trait.

        Returns:
            Dict with oscillation info or None
        """
        if trait not in self.escalation_history:
            return None

        history = self.escalation_history[trait]
        if len(history) < window:
            return None

        recent = [v for _, v in history[-window:]]

        # Check for direction changes
        changes = 0
        for i in range(1, len(recent)):
            if (recent[i] - recent[i-1]) * (recent[i-1] - recent[i-2] if i > 1 else 1) < 0:
                changes += 1

        # High number of changes indicates oscillation
        oscillation_score = changes / window

        if oscillation_score > 0.3:
            return {
                "trait": trait,
                "oscillation_score": oscillation_score,
                "recent_values": recent[-5:],
                "period_estimate": window / max(1, changes)
            }

        return None

    def process_coevolution(
        self,
        populations: Dict[str, List[Genome]],
        generation: int
    ) -> Dict[str, Any]:
        """
        Process coevolutionary dynamics for a generation.

        Args:
            populations: Dict mapping species ID to population
            generation: Current generation

        Returns:
            Statistics about coevolution this generation
        """
        results = {
            "generation": generation,
            "interactions": [],
            "escalations": 0,
            "oscillations_detected": []
        }

        for relationship in self.relationships.values():
            sp1_pop = populations.get(relationship.species1_id, [])
            sp2_pop = populations.get(relationship.species2_id, [])

            if not sp1_pop or not sp2_pop:
                continue

            # Simulate interactions
            records = self.simulate_interactions(
                sp1_pop, sp2_pop,
                relationship.id,
                generation=generation
            )

            results["interactions"].extend([r.to_dict() for r in records[:5]])

            # Calculate selection pressure
            win_rate = relationship.win_ratio

            sp1_pressures = {}
            sp2_pressures = {}

            for genome in sp1_pop[:10]:  # Sample
                p = self.calculate_selection_pressure(
                    genome, relationship.role1, relationship, win_rate
                )
                for k, v in p.items():
                    sp1_pressures[k] = sp1_pressures.get(k, 0) + v

            for genome in sp2_pop[:10]:
                p = self.calculate_selection_pressure(
                    genome, relationship.role2, relationship, win_rate
                )
                for k, v in p.items():
                    sp2_pressures[k] = sp2_pressures.get(k, 0) + v

            # Apply escalation
            esc1 = self.apply_escalation(sp1_pop, sp1_pressures, generation)
            esc2 = self.apply_escalation(sp2_pop, sp2_pressures, generation)
            results["escalations"] += esc1 + esc2

            # Check for oscillations
            for trait_pair in relationship.trait_pairs:
                osc = self.detect_oscillation(trait_pair.attack_trait)
                if osc:
                    results["oscillations_detected"].append(osc)
                osc = self.detect_oscillation(trait_pair.defense_trait)
                if osc:
                    results["oscillations_detected"].append(osc)

        # Emit event
        bus = get_event_bus()
        bus.emit("arms_race.coevolution", {
            "generation": generation,
            "escalations": results["escalations"]
        })

        return results

    def get_relationship_summary(
        self,
        relationship_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary of a specific relationship"""
        if relationship_id not in self.relationships:
            return None

        r = self.relationships[relationship_id]

        return {
            **r.to_dict(),
            "escalation_status": {
                tp.attack_trait: len(self.escalation_history.get(tp.attack_trait, []))
                for tp in r.trait_pairs
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get arms race statistics"""
        return {
            "total_relationships": len(self.relationships),
            "total_interactions": self.total_interactions,
            "relationships": [r.to_dict() for r in self.relationships.values()],
            "escalation_summary": {
                trait: len(history)
                for trait, history in self.escalation_history.items()
            }
        }
