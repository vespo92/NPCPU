"""
Population Dynamics System

Manages populations of digital organisms with:
- Population growth and decline
- Inter-organism interactions
- Social networks and relationships
- Competition and cooperation
- Group behaviors
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class InteractionType(Enum):
    """Types of interactions between organisms"""
    COOPERATION = "cooperation"       # Working together
    COMPETITION = "competition"       # Competing for resources
    PREDATION = "predation"          # One consuming another
    MUTUALISM = "mutualism"          # Both benefit
    PARASITISM = "parasitism"         # One benefits, other harmed
    COMMENSALISM = "commensalism"     # One benefits, other neutral
    COMMUNICATION = "communication"   # Information exchange
    MATING = "mating"                # Reproduction
    CONFLICT = "conflict"            # Active fighting


class RelationshipType(Enum):
    """Types of relationships"""
    STRANGER = "stranger"            # No relationship
    ACQUAINTANCE = "acquaintance"    # Familiar
    ALLY = "ally"                    # Cooperative partner
    RIVAL = "rival"                  # Competitor
    MATE = "mate"                    # Reproductive partner
    OFFSPRING = "offspring"          # Child
    PARENT = "parent"                # Parent
    SIBLING = "sibling"              # Same generation, same parent
    ENEMY = "enemy"                  # Hostile


class PopulationTrend(Enum):
    """Population growth trends"""
    EXPLOSIVE = "explosive"          # Rapid growth
    GROWING = "growing"              # Steady growth
    STABLE = "stable"                # No change
    DECLINING = "declining"          # Steady decline
    CRASHING = "crashing"            # Rapid decline
    EXTINCT = "extinct"              # No organisms left


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Interaction:
    """A single interaction between organisms"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: InteractionType = InteractionType.COMMUNICATION
    initiator_id: str = ""
    target_id: str = ""
    outcome: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Relationship:
    """A relationship between two organisms"""
    organism_a: str
    organism_b: str
    type: RelationshipType = RelationshipType.STRANGER
    strength: float = 0.0            # -1 to 1
    trust: float = 0.5               # 0 to 1
    history: List[Interaction] = field(default_factory=list)
    formed_at: float = field(default_factory=time.time)


@dataclass
class GroupMembership:
    """Membership in a group/colony"""
    group_id: str
    organism_id: str
    role: str = "member"             # leader, member, etc.
    joined_at: float = field(default_factory=time.time)
    contribution: float = 0.0


@dataclass
class PopulationStats:
    """Statistics about a population"""
    total_count: int = 0
    birth_rate: float = 0.0
    death_rate: float = 0.0
    average_age: float = 0.0
    average_fitness: float = 0.0
    genetic_diversity: float = 0.0
    trend: PopulationTrend = PopulationTrend.STABLE


# ============================================================================
# Social Network
# ============================================================================

class SocialNetwork:
    """
    Social network of organism relationships.

    Tracks:
    - Individual relationships
    - Groups and colonies
    - Social hierarchy
    - Communication patterns
    """

    def __init__(self):
        # Relationships by pair
        self.relationships: Dict[Tuple[str, str], Relationship] = {}

        # Groups
        self.groups: Dict[str, Set[str]] = {}
        self.memberships: Dict[str, List[GroupMembership]] = defaultdict(list)

        # Interaction history
        self.interaction_count: Dict[str, int] = defaultdict(int)

    def _pair_key(self, id_a: str, id_b: str) -> Tuple[str, str]:
        """Get canonical pair key"""
        return (min(id_a, id_b), max(id_a, id_b))

    def get_relationship(self, id_a: str, id_b: str) -> Optional[Relationship]:
        """Get relationship between two organisms"""
        key = self._pair_key(id_a, id_b)
        return self.relationships.get(key)

    def set_relationship(
        self,
        id_a: str,
        id_b: str,
        rel_type: RelationshipType,
        strength: float = 0.5
    ):
        """Set or update relationship"""
        key = self._pair_key(id_a, id_b)

        if key in self.relationships:
            self.relationships[key].type = rel_type
            self.relationships[key].strength = strength
        else:
            self.relationships[key] = Relationship(
                organism_a=key[0],
                organism_b=key[1],
                type=rel_type,
                strength=strength
            )

    def record_interaction(
        self,
        initiator: str,
        target: str,
        interaction_type: InteractionType,
        outcome: Dict[str, Any]
    ) -> Interaction:
        """Record an interaction between organisms"""
        interaction = Interaction(
            type=interaction_type,
            initiator_id=initiator,
            target_id=target,
            outcome=outcome
        )

        key = self._pair_key(initiator, target)
        if key not in self.relationships:
            self.set_relationship(initiator, target, RelationshipType.ACQUAINTANCE)

        self.relationships[key].history.append(interaction)
        self.interaction_count[initiator] += 1
        self.interaction_count[target] += 1

        # Update relationship based on interaction
        self._update_relationship_from_interaction(key, interaction)

        return interaction

    def _update_relationship_from_interaction(
        self,
        key: Tuple[str, str],
        interaction: Interaction
    ):
        """Update relationship based on interaction"""
        rel = self.relationships[key]

        # Positive interactions increase strength
        if interaction.type in [InteractionType.COOPERATION, InteractionType.MUTUALISM]:
            rel.strength = min(1.0, rel.strength + 0.1)
            rel.trust = min(1.0, rel.trust + 0.05)
            if rel.strength > 0.5:
                rel.type = RelationshipType.ALLY

        # Negative interactions decrease strength
        elif interaction.type in [InteractionType.CONFLICT, InteractionType.PREDATION]:
            rel.strength = max(-1.0, rel.strength - 0.2)
            rel.trust = max(0.0, rel.trust - 0.1)
            if rel.strength < -0.3:
                rel.type = RelationshipType.ENEMY

        # Competition is mildly negative
        elif interaction.type == InteractionType.COMPETITION:
            rel.strength = max(-1.0, rel.strength - 0.05)
            if rel.strength < 0:
                rel.type = RelationshipType.RIVAL

    def create_group(self, name: str, founder_id: str) -> str:
        """Create a new group"""
        group_id = str(uuid.uuid4())
        self.groups[group_id] = {founder_id}

        membership = GroupMembership(
            group_id=group_id,
            organism_id=founder_id,
            role="leader"
        )
        self.memberships[founder_id].append(membership)

        return group_id

    def join_group(self, organism_id: str, group_id: str, role: str = "member"):
        """Join an existing group"""
        if group_id in self.groups:
            self.groups[group_id].add(organism_id)

            membership = GroupMembership(
                group_id=group_id,
                organism_id=organism_id,
                role=role
            )
            self.memberships[organism_id].append(membership)

    def leave_group(self, organism_id: str, group_id: str):
        """Leave a group"""
        if group_id in self.groups:
            self.groups[group_id].discard(organism_id)

        self.memberships[organism_id] = [
            m for m in self.memberships[organism_id]
            if m.group_id != group_id
        ]

    def get_allies(self, organism_id: str) -> List[str]:
        """Get all allies of an organism"""
        allies = []
        for key, rel in self.relationships.items():
            if organism_id in key and rel.type == RelationshipType.ALLY:
                other = key[1] if key[0] == organism_id else key[0]
                allies.append(other)
        return allies

    def get_enemies(self, organism_id: str) -> List[str]:
        """Get all enemies of an organism"""
        enemies = []
        for key, rel in self.relationships.items():
            if organism_id in key and rel.type == RelationshipType.ENEMY:
                other = key[1] if key[0] == organism_id else key[0]
                enemies.append(other)
        return enemies


# ============================================================================
# Population Dynamics
# ============================================================================

class PopulationDynamics:
    """
    Manages population-level dynamics.

    Features:
    - Birth/death tracking
    - Carrying capacity
    - Resource competition
    - Fitness calculation
    """

    def __init__(
        self,
        carrying_capacity: int = 1000,
        base_birth_rate: float = 0.1,
        base_death_rate: float = 0.05
    ):
        self.carrying_capacity = carrying_capacity
        self.base_birth_rate = base_birth_rate
        self.base_death_rate = base_death_rate

        # History
        self.population_history: List[int] = []
        self.birth_history: List[int] = []
        self.death_history: List[int] = []

    def calculate_growth_rate(
        self,
        current_population: int,
        resources_available: float
    ) -> float:
        """Calculate population growth rate (logistic growth)"""
        if current_population == 0:
            return 0.0

        # Logistic growth factor
        density_factor = 1 - (current_population / self.carrying_capacity)

        # Resource factor
        resource_factor = min(1.0, resources_available / current_population)

        # Combined rate
        birth_rate = self.base_birth_rate * density_factor * resource_factor
        death_rate = self.base_death_rate * (1 / max(0.1, resource_factor))

        return birth_rate - death_rate

    def calculate_fitness(
        self,
        organism_traits: Dict[str, float],
        environment_conditions: Dict[str, float]
    ) -> float:
        """Calculate organism fitness based on traits and environment"""
        fitness = 0.5  # Base fitness

        # Trait contributions
        for trait, value in organism_traits.items():
            if trait == "resilience":
                fitness += value * 0.1
            elif trait == "metabolism_rate":
                # Too high or too low is bad
                optimal = environment_conditions.get("optimal_metabolism", 1.0)
                diff = abs(value - optimal)
                fitness -= diff * 0.05
            elif trait == "sensitivity":
                # Good in dangerous environments
                danger = environment_conditions.get("danger_level", 0.0)
                fitness += value * danger * 0.1
            elif trait == "sociability":
                # Good in social environments
                social = environment_conditions.get("social_density", 0.0)
                fitness += value * social * 0.05

        return np.clip(fitness, 0.0, 1.0)

    def should_reproduce(
        self,
        organism_fitness: float,
        organism_energy: float,
        population_size: int
    ) -> bool:
        """Determine if organism should reproduce"""
        # Need sufficient energy
        if organism_energy < 50:
            return False

        # Need good fitness
        if organism_fitness < 0.3:
            return False

        # Density-dependent reproduction
        density_factor = 1 - (population_size / self.carrying_capacity)
        if density_factor < 0:
            return False

        # Probabilistic based on fitness
        return np.random.random() < organism_fitness * density_factor * 0.2

    def should_die(
        self,
        organism_age_ratio: float,
        organism_health: float,
        organism_energy: float
    ) -> bool:
        """Determine if organism should die"""
        # Natural death from old age
        if organism_age_ratio > 0.95:
            return np.random.random() < 0.5

        # Death from poor health
        if organism_health < 0.1:
            return np.random.random() < 0.8

        # Death from starvation
        if organism_energy <= 0:
            return True

        # Background mortality
        base_mortality = self.base_death_rate * (1 + organism_age_ratio)
        return np.random.random() < base_mortality * 0.01

    def record_population(self, count: int, births: int = 0, deaths: int = 0):
        """Record population snapshot"""
        self.population_history.append(count)
        self.birth_history.append(births)
        self.death_history.append(deaths)

        # Keep history manageable
        max_history = 1000
        if len(self.population_history) > max_history:
            self.population_history = self.population_history[-max_history:]
            self.birth_history = self.birth_history[-max_history:]
            self.death_history = self.death_history[-max_history:]

    def get_trend(self) -> PopulationTrend:
        """Get population trend"""
        if len(self.population_history) < 10:
            return PopulationTrend.STABLE

        recent = self.population_history[-10:]

        if recent[-1] == 0:
            return PopulationTrend.EXTINCT

        # Calculate trend
        start = np.mean(recent[:3])
        end = np.mean(recent[-3:])

        if start == 0:
            return PopulationTrend.GROWING if end > 0 else PopulationTrend.EXTINCT

        change_ratio = (end - start) / start

        if change_ratio > 0.3:
            return PopulationTrend.EXPLOSIVE
        elif change_ratio > 0.1:
            return PopulationTrend.GROWING
        elif change_ratio > -0.1:
            return PopulationTrend.STABLE
        elif change_ratio > -0.3:
            return PopulationTrend.DECLINING
        else:
            return PopulationTrend.CRASHING


# ============================================================================
# Population
# ============================================================================

class Population:
    """
    A population of digital organisms.

    Features:
    - Organism management
    - Interactions
    - Social dynamics
    - Population statistics

    Example:
        pop = Population()

        # Add organisms
        pop.add_organism(organism1)
        pop.add_organism(organism2)

        # Run population cycle
        pop.tick()

        # Get statistics
        stats = pop.get_stats()
    """

    def __init__(
        self,
        name: str = "default",
        carrying_capacity: int = 1000
    ):
        self.name = name

        # Organisms (by id)
        self.organisms: Dict[str, Any] = {}

        # Dynamics
        self.dynamics = PopulationDynamics(carrying_capacity)

        # Social network
        self.social = SocialNetwork()

        # Tick counter
        self.tick_count = 0

        # Statistics
        self.total_births = 0
        self.total_deaths = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "organism_added": [],
            "organism_died": [],
            "interaction": []
        }

    def add_organism(self, organism: Any):
        """Add an organism to the population"""
        self.organisms[organism.identity.id] = organism
        self.total_births += 1

        for callback in self._callbacks["organism_added"]:
            callback(organism)

    def remove_organism(self, organism_id: str, cause: str = "unknown"):
        """Remove an organism from the population"""
        if organism_id in self.organisms:
            organism = self.organisms[organism_id]
            del self.organisms[organism_id]
            self.total_deaths += 1

            for callback in self._callbacks["organism_died"]:
                callback(organism, cause)

    def get_organism(self, organism_id: str) -> Optional[Any]:
        """Get organism by ID"""
        return self.organisms.get(organism_id)

    def interact(
        self,
        initiator_id: str,
        target_id: str,
        interaction_type: InteractionType
    ) -> Optional[Interaction]:
        """Perform an interaction between two organisms"""
        initiator = self.organisms.get(initiator_id)
        target = self.organisms.get(target_id)

        if not initiator or not target:
            return None

        # Calculate interaction outcome
        outcome = self._calculate_interaction_outcome(
            initiator, target, interaction_type
        )

        # Record in social network
        interaction = self.social.record_interaction(
            initiator_id, target_id, interaction_type, outcome
        )

        # Apply effects
        self._apply_interaction_effects(initiator, target, interaction_type, outcome)

        for callback in self._callbacks["interaction"]:
            callback(interaction)

        return interaction

    def _calculate_interaction_outcome(
        self,
        initiator: Any,
        target: Any,
        interaction_type: InteractionType
    ) -> Dict[str, Any]:
        """Calculate the outcome of an interaction"""
        outcome = {"success": True}

        if interaction_type == InteractionType.COOPERATION:
            # Both gain resources
            combined_efficiency = (
                initiator.identity.traits.get("sociability", 0.5) +
                target.identity.traits.get("sociability", 0.5)
            ) / 2
            outcome["resource_gain"] = 10 * combined_efficiency
            outcome["both_benefit"] = True

        elif interaction_type == InteractionType.COMPETITION:
            # One wins, one loses
            initiator_power = (
                initiator.metabolism.energy * 0.5 +
                initiator.identity.traits.get("aggression", 0.3) * 50
            )
            target_power = (
                target.metabolism.energy * 0.5 +
                target.identity.traits.get("aggression", 0.3) * 50
            )

            if np.random.random() < initiator_power / (initiator_power + target_power):
                outcome["winner"] = initiator.identity.id
                outcome["loser"] = target.identity.id
            else:
                outcome["winner"] = target.identity.id
                outcome["loser"] = initiator.identity.id

        elif interaction_type == InteractionType.COMMUNICATION:
            # Information exchange
            initiator_sociability = initiator.identity.traits.get("sociability", 0.5)
            outcome["information_quality"] = initiator_sociability
            outcome["bond_increase"] = 0.05

        elif interaction_type == InteractionType.MATING:
            # Reproduction check
            can_mate = (
                initiator.lifecycle.stage.value >= 3 and  # Adult
                target.lifecycle.stage.value >= 3 and
                initiator.metabolism.energy > 30 and
                target.metabolism.energy > 30
            )
            outcome["successful_mating"] = can_mate and np.random.random() < 0.3

        return outcome

    def _apply_interaction_effects(
        self,
        initiator: Any,
        target: Any,
        interaction_type: InteractionType,
        outcome: Dict[str, Any]
    ):
        """Apply effects of interaction to organisms"""
        if interaction_type == InteractionType.COOPERATION:
            gain = outcome.get("resource_gain", 0)
            initiator.metabolism.energy = min(
                initiator.metabolism.max_energy,
                initiator.metabolism.energy + gain / 2
            )
            target.metabolism.energy = min(
                target.metabolism.max_energy,
                target.metabolism.energy + gain / 2
            )

            # Positive hormonal response
            initiator.endocrine.trigger_social_bonding(0.3)
            target.endocrine.trigger_social_bonding(0.3)

        elif interaction_type == InteractionType.COMPETITION:
            loser_id = outcome.get("loser")
            if loser_id:
                loser = initiator if initiator.identity.id == loser_id else target
                loser.metabolism.energy -= 5
                loser.endocrine.trigger_stress_response(0.2)

        elif interaction_type == InteractionType.COMMUNICATION:
            # Social bonding
            initiator.endocrine.trigger_social_bonding(
                outcome.get("bond_increase", 0.05)
            )
            target.endocrine.trigger_social_bonding(
                outcome.get("bond_increase", 0.05)
            )

    def find_interaction_partner(
        self,
        organism_id: str,
        interaction_type: InteractionType
    ) -> Optional[str]:
        """Find a suitable partner for an interaction"""
        organism = self.organisms.get(organism_id)
        if not organism:
            return None

        candidates = [
            oid for oid in self.organisms.keys()
            if oid != organism_id
        ]

        if not candidates:
            return None

        # Prefer allies for cooperation, rivals for competition
        relationship = self.social.get_relationship

        if interaction_type == InteractionType.COOPERATION:
            # Prefer allies
            allies = self.social.get_allies(organism_id)
            if allies:
                return np.random.choice(allies)

        elif interaction_type == InteractionType.COMPETITION:
            # Prefer rivals
            enemies = self.social.get_enemies(organism_id)
            if enemies:
                return np.random.choice(enemies)

        # Random otherwise
        return np.random.choice(candidates)

    def tick(self):
        """Process one population cycle"""
        self.tick_count += 1

        births = 0
        deaths = 0
        dead_ids = []

        # Process each organism
        for organism_id, organism in list(self.organisms.items()):
            # Check death
            vitals = organism.get_vital_signs()
            if self.dynamics.should_die(
                vitals.age_ratio,
                vitals.health,
                vitals.energy
            ):
                dead_ids.append(organism_id)
                deaths += 1
                continue

            # Check reproduction
            fitness = self.dynamics.calculate_fitness(
                organism.identity.traits,
                {"social_density": len(self.organisms) / self.dynamics.carrying_capacity}
            )

            if self.dynamics.should_reproduce(
                fitness,
                vitals.energy,
                len(self.organisms)
            ):
                offspring = organism.reproduce()
                if offspring:
                    self.add_organism(offspring)
                    births += 1

                    # Set family relationships
                    self.social.set_relationship(
                        organism_id,
                        offspring.identity.id,
                        RelationshipType.OFFSPRING,
                        0.8
                    )

        # Remove dead organisms
        for dead_id in dead_ids:
            self.remove_organism(dead_id, "natural")

        # Record statistics
        self.dynamics.record_population(len(self.organisms), births, deaths)

        # Random interactions
        self._process_random_interactions()

    def _process_random_interactions(self):
        """Process random interactions between organisms"""
        if len(self.organisms) < 2:
            return

        # Each organism has a chance to interact
        organism_ids = list(self.organisms.keys())

        for organism_id in organism_ids:
            if np.random.random() < 0.2:  # 20% chance to interact
                organism = self.organisms.get(organism_id)
                if not organism:
                    continue

                # Choose interaction type based on traits
                sociability = organism.identity.traits.get("sociability", 0.5)
                aggression = organism.identity.traits.get("aggression", 0.3)

                if np.random.random() < sociability:
                    interaction_type = np.random.choice([
                        InteractionType.COOPERATION,
                        InteractionType.COMMUNICATION
                    ])
                else:
                    interaction_type = InteractionType.COMPETITION

                # Find partner
                partner_id = self.find_interaction_partner(organism_id, interaction_type)
                if partner_id:
                    self.interact(organism_id, partner_id, interaction_type)

    def get_stats(self) -> PopulationStats:
        """Get population statistics"""
        if not self.organisms:
            return PopulationStats(trend=PopulationTrend.EXTINCT)

        ages = []
        fitnesses = []
        traits_collected: Dict[str, List[float]] = defaultdict(list)

        for organism in self.organisms.values():
            ages.append(organism.lifecycle.age)

            fitness = self.dynamics.calculate_fitness(
                organism.identity.traits,
                {}
            )
            fitnesses.append(fitness)

            for trait, value in organism.identity.traits.items():
                traits_collected[trait].append(value)

        # Calculate genetic diversity (variance in traits)
        diversity = np.mean([
            np.std(values) if len(values) > 1 else 0
            for values in traits_collected.values()
        ])

        # Calculate rates
        recent_births = sum(self.dynamics.birth_history[-10:]) if self.dynamics.birth_history else 0
        recent_deaths = sum(self.dynamics.death_history[-10:]) if self.dynamics.death_history else 0
        pop_size = len(self.organisms)

        return PopulationStats(
            total_count=pop_size,
            birth_rate=recent_births / max(1, pop_size * 10),
            death_rate=recent_deaths / max(1, pop_size * 10),
            average_age=np.mean(ages) if ages else 0,
            average_fitness=np.mean(fitnesses) if fitnesses else 0,
            genetic_diversity=diversity,
            trend=self.dynamics.get_trend()
        )

    def on_organism_added(self, callback: Callable):
        """Register callback for organism addition"""
        self._callbacks["organism_added"].append(callback)

    def on_organism_died(self, callback: Callable):
        """Register callback for organism death"""
        self._callbacks["organism_died"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get population status"""
        stats = self.get_stats()
        return {
            "name": self.name,
            "size": len(self.organisms),
            "tick_count": self.tick_count,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "trend": stats.trend.value,
            "average_age": stats.average_age,
            "average_fitness": stats.average_fitness,
            "genetic_diversity": stats.genetic_diversity,
            "groups": len(self.social.groups),
            "relationships": len(self.social.relationships)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Population Dynamics Demo")
    print("=" * 50)

    # Create population
    pop = Population(name="test_colony", carrying_capacity=100)

    print(f"\n1. Created population: {pop.name}")

    # We'd normally add DigitalBody organisms, but for demo we'll use mock objects
    @dataclass
    class MockOrganism:
        identity: Any
        lifecycle: Any
        metabolism: Any
        endocrine: Any

        def get_vital_signs(self):
            @dataclass
            class Vitals:
                age_ratio: float
                health: float
                energy: float
            return Vitals(0.3, 0.8, 50)

        def reproduce(self):
            return None

    @dataclass
    class MockIdentity:
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        traits: Dict[str, float] = field(default_factory=lambda: {
            "sociability": np.random.uniform(0.3, 0.8),
            "aggression": np.random.uniform(0.1, 0.5)
        })

    @dataclass
    class MockLifecycle:
        age: float = 50
        stage: Any = None

    @dataclass
    class MockMetabolism:
        energy: float = 50
        max_energy: float = 100

    @dataclass
    class MockEndocrine:
        def trigger_social_bonding(self, i): pass
        def trigger_stress_response(self, i): pass

    # Add mock organisms
    print("\n2. Adding organisms...")
    for i in range(10):
        mock_stage = type('MockStage', (), {'value': 4})()
        org = MockOrganism(
            identity=MockIdentity(),
            lifecycle=MockLifecycle(age=np.random.uniform(10, 80)),
            metabolism=MockMetabolism(energy=np.random.uniform(30, 80)),
            endocrine=MockEndocrine()
        )
        org.lifecycle.stage = mock_stage
        pop.add_organism(org)

    print(f"   Added {len(pop.organisms)} organisms")

    # Create some relationships
    print("\n3. Creating relationships...")
    ids = list(pop.organisms.keys())
    pop.social.set_relationship(ids[0], ids[1], RelationshipType.ALLY, 0.7)
    pop.social.set_relationship(ids[2], ids[3], RelationshipType.RIVAL, -0.3)

    # Perform interactions
    print("\n4. Processing interactions...")
    interaction = pop.interact(ids[0], ids[1], InteractionType.COOPERATION)
    if interaction:
        print(f"   Cooperation: {interaction.outcome}")

    interaction = pop.interact(ids[2], ids[3], InteractionType.COMPETITION)
    if interaction:
        print(f"   Competition: winner={interaction.outcome.get('winner', 'N/A')[:8]}...")

    # Get statistics
    print("\n5. Population statistics:")
    stats = pop.get_stats()
    print(f"   Total: {stats.total_count}")
    print(f"   Average fitness: {stats.average_fitness:.2f}")
    print(f"   Genetic diversity: {stats.genetic_diversity:.3f}")
    print(f"   Trend: {stats.trend.value}")

    # Get status
    print("\n6. Population status:")
    status = pop.get_status()
    for key, value in status.items():
        if not isinstance(value, (dict, list)):
            print(f"   {key}: {value}")
