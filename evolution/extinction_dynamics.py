"""
Extinction Dynamics System

Implements extinction and recovery mechanisms:
- Mass extinction events
- Background extinction rates
- Selective extinction patterns
- Recovery and radiation
- Extinction debt
- Rescue effects
- Bottleneck dynamics

Example:
    from evolution.extinction_dynamics import ExtinctionEngine

    engine = ExtinctionEngine(
        background_rate=0.01,
        mass_extinction_probability=0.001
    )

    # Check for extinctions
    casualties = engine.process_generation(population, generation)

    # Trigger catastrophic event
    survivors = engine.trigger_mass_extinction(
        population,
        severity=0.7,
        selective_traits=["temperature_tolerance"]
    )
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable
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
from .genetic_engine import Genome, Species


class ExtinctionType(Enum):
    """Types of extinction events"""
    BACKGROUND = auto()         # Normal turnover
    MASS = auto()               # Catastrophic mass extinction
    LOCAL = auto()              # Regional extinction
    COMPETITIVE = auto()        # Outcompeted by others
    HABITAT_LOSS = auto()       # Environment destruction
    DISEASE = auto()            # Pandemic


class RecoveryPhase(Enum):
    """Phases of post-extinction recovery"""
    IMMEDIATE = auto()          # Right after extinction
    SURVIVAL = auto()           # Survival phase
    RADIATION = auto()          # Diversification phase
    STABLE = auto()             # New equilibrium


@dataclass
class ExtinctionEvent:
    """Record of an extinction event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    extinction_type: ExtinctionType = ExtinctionType.BACKGROUND
    generation: int = 0
    casualties: int = 0
    species_lost: List[str] = field(default_factory=list)
    survivors: int = 0
    severity: float = 0.0  # 0-1, fraction eliminated
    selective_trait: Optional[str] = None
    cause: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.extinction_type.name,
            "generation": self.generation,
            "casualties": self.casualties,
            "species_lost": self.species_lost,
            "survivors": self.survivors,
            "severity": self.severity,
            "selective_trait": self.selective_trait,
            "cause": self.cause
        }


@dataclass
class PopulationBottleneck:
    """
    Represents a population bottleneck.

    Bottlenecks cause loss of genetic diversity
    and can lead to inbreeding depression.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    species_id: str = ""
    generation_start: int = 0
    minimum_population: int = 0
    original_population: int = 0
    genetic_diversity_before: float = 0.0
    genetic_diversity_after: float = 0.0
    recovery_generation: Optional[int] = None

    @property
    def is_recovered(self) -> bool:
        return self.recovery_generation is not None

    @property
    def severity(self) -> float:
        """Bottleneck severity (0-1)"""
        if self.original_population == 0:
            return 0.0
        return 1.0 - (self.minimum_population / self.original_population)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "species": self.species_id,
            "generation_start": self.generation_start,
            "min_population": self.minimum_population,
            "original_population": self.original_population,
            "severity": self.severity,
            "diversity_loss": self.genetic_diversity_before - self.genetic_diversity_after,
            "recovered": self.is_recovered
        }


@dataclass
class RecoveryState:
    """
    Tracks post-extinction recovery state.
    """
    phase: RecoveryPhase = RecoveryPhase.IMMEDIATE
    extinction_event_id: str = ""
    generation_started: int = 0
    current_diversity: float = 0.0
    target_diversity: float = 1.0
    recovery_rate: float = 0.1
    new_species_count: int = 0

    def update(self, current_pop: int, generation: int) -> None:
        """Update recovery phase based on population and time"""
        elapsed = generation - self.generation_started

        if self.phase == RecoveryPhase.IMMEDIATE and elapsed > 5:
            self.phase = RecoveryPhase.SURVIVAL
        elif self.phase == RecoveryPhase.SURVIVAL and elapsed > 20:
            self.phase = RecoveryPhase.RADIATION
        elif self.phase == RecoveryPhase.RADIATION:
            if self.current_diversity >= self.target_diversity * 0.8:
                self.phase = RecoveryPhase.STABLE


class ExtinctionEngine:
    """
    Engine for managing extinction dynamics.

    Features:
    - Background extinction modeling
    - Mass extinction events
    - Selective extinction
    - Bottleneck tracking
    - Recovery dynamics
    - Genetic rescue

    Example:
        engine = ExtinctionEngine(
            background_rate=0.01,
            mass_extinction_probability=0.001
        )

        # Process each generation
        result = engine.process_generation(population, generation)

        # Trigger specific extinction
        survivors = engine.trigger_mass_extinction(
            population, severity=0.6
        )
    """

    def __init__(
        self,
        background_rate: float = 0.01,
        mass_extinction_probability: float = 0.001,
        minimum_viable_population: int = 10,
        enable_rescue_effect: bool = True,
        bottleneck_threshold: float = 0.3  # Fraction of original pop
    ):
        self.background_rate = background_rate
        self.mass_extinction_probability = mass_extinction_probability
        self.minimum_viable_population = minimum_viable_population
        self.enable_rescue_effect = enable_rescue_effect
        self.bottleneck_threshold = bottleneck_threshold

        # Extinction events
        self.extinction_events: List[ExtinctionEvent] = []

        # Bottlenecks
        self.bottlenecks: Dict[str, PopulationBottleneck] = {}

        # Recovery state
        self.recovery_states: Dict[str, RecoveryState] = {}

        # Species tracking
        self.species_history: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_casualties = 0
        self.total_species_extinct = 0

    def calculate_extinction_probability(
        self,
        genome: Genome,
        population_size: int,
        fitness: float = 1.0,
        environmental_stress: float = 0.0
    ) -> float:
        """
        Calculate extinction probability for an individual.

        Factors:
        - Background rate
        - Population size (small = higher risk)
        - Individual fitness
        - Environmental stress
        """
        base_prob = self.background_rate

        # Small population effect (Allee effect)
        if population_size < self.minimum_viable_population:
            base_prob *= 2.0 + (self.minimum_viable_population - population_size) * 0.1

        # Fitness effect
        base_prob *= (2.0 - fitness)

        # Environmental stress
        base_prob *= (1.0 + environmental_stress)

        return min(0.99, base_prob)

    def apply_background_extinction(
        self,
        population: List[Genome],
        fitness_scores: Optional[Dict[str, float]] = None,
        generation: int = 0
    ) -> Tuple[List[Genome], List[str]]:
        """
        Apply background extinction to population.

        Returns:
            Tuple of (survivors, casualty IDs)
        """
        survivors = []
        casualties = []

        for genome in population:
            fitness = fitness_scores.get(genome.id, 1.0) if fitness_scores else 1.0

            prob = self.calculate_extinction_probability(
                genome, len(population), fitness
            )

            if random.random() < prob:
                casualties.append(genome.id)
            else:
                survivors.append(genome)

        if casualties:
            event = ExtinctionEvent(
                extinction_type=ExtinctionType.BACKGROUND,
                generation=generation,
                casualties=len(casualties),
                survivors=len(survivors),
                severity=len(casualties) / max(1, len(population))
            )
            self.extinction_events.append(event)
            self.total_casualties += len(casualties)

        return survivors, casualties

    def trigger_mass_extinction(
        self,
        population: List[Genome],
        severity: float,
        selective_trait: Optional[str] = None,
        trait_survival_direction: float = 1.0,  # +1 = high trait survives
        generation: int = 0,
        cause: str = "catastrophe"
    ) -> Tuple[List[Genome], ExtinctionEvent]:
        """
        Trigger a mass extinction event.

        Args:
            population: Current population
            severity: Fraction to eliminate (0-1)
            selective_trait: Trait that affects survival
            trait_survival_direction: +1 = high trait values survive
            generation: Current generation
            cause: Description of extinction cause

        Returns:
            Tuple of (survivors, extinction_event)
        """
        if not population:
            return [], ExtinctionEvent(extinction_type=ExtinctionType.MASS)

        survivors = []
        species_affected: Set[str] = set()

        for genome in population:
            # Calculate survival probability
            base_survival = 1.0 - severity

            if selective_trait and selective_trait in genome.genes:
                trait_value = genome.express(selective_trait)
                # Modify survival based on trait
                trait_effect = trait_value if trait_survival_direction > 0 else (1 - trait_value)
                survival_prob = base_survival * (0.5 + trait_effect * 0.5)
            else:
                survival_prob = base_survival

            # Add some randomness
            survival_prob += random.gauss(0, 0.1)
            survival_prob = max(0.01, min(0.99, survival_prob))

            if random.random() < survival_prob:
                survivors.append(genome)
            else:
                if genome.species_id:
                    species_affected.add(genome.species_id)

        # Determine which species went extinct
        surviving_species = {g.species_id for g in survivors if g.species_id}
        extinct_species = list(species_affected - surviving_species)

        event = ExtinctionEvent(
            extinction_type=ExtinctionType.MASS,
            generation=generation,
            casualties=len(population) - len(survivors),
            species_lost=extinct_species,
            survivors=len(survivors),
            severity=severity,
            selective_trait=selective_trait,
            cause=cause
        )

        self.extinction_events.append(event)
        self.total_casualties += event.casualties
        self.total_species_extinct += len(extinct_species)

        # Start recovery tracking
        self.recovery_states[event.id] = RecoveryState(
            phase=RecoveryPhase.IMMEDIATE,
            extinction_event_id=event.id,
            generation_started=generation,
            target_diversity=len(population)
        )

        # Emit event
        bus = get_event_bus()
        bus.emit("extinction.mass_event", {
            "event_id": event.id,
            "severity": severity,
            "casualties": event.casualties,
            "species_lost": len(extinct_species),
            "cause": cause
        })

        return survivors, event

    def check_for_bottleneck(
        self,
        species_id: str,
        current_pop: int,
        original_pop: int,
        diversity: float,
        generation: int
    ) -> Optional[PopulationBottleneck]:
        """
        Check if species is in a population bottleneck.

        Returns:
            PopulationBottleneck if detected, None otherwise
        """
        if current_pop >= original_pop * self.bottleneck_threshold:
            # Check if recovering from existing bottleneck
            if species_id in self.bottlenecks:
                bn = self.bottlenecks[species_id]
                if not bn.is_recovered and current_pop >= original_pop * 0.8:
                    bn.recovery_generation = generation
            return None

        # Create or update bottleneck
        if species_id not in self.bottlenecks:
            bn = PopulationBottleneck(
                species_id=species_id,
                generation_start=generation,
                minimum_population=current_pop,
                original_population=original_pop,
                genetic_diversity_before=diversity
            )
            self.bottlenecks[species_id] = bn

            # Emit event
            bus = get_event_bus()
            bus.emit("extinction.bottleneck", {
                "species": species_id,
                "population": current_pop,
                "severity": bn.severity
            })
        else:
            bn = self.bottlenecks[species_id]
            bn.minimum_population = min(bn.minimum_population, current_pop)
            bn.genetic_diversity_after = diversity

        return bn

    def apply_genetic_rescue(
        self,
        struggling_pop: List[Genome],
        donor_pop: List[Genome],
        rescue_rate: float = 0.1
    ) -> int:
        """
        Apply genetic rescue from donor population.

        This simulates immigration or assisted gene flow
        to boost genetic diversity in a bottlenecked population.

        Returns:
            Number of rescuing gene flow events
        """
        if not self.enable_rescue_effect:
            return 0

        if not donor_pop or not struggling_pop:
            return 0

        rescue_events = 0

        for recipient in struggling_pop:
            if random.random() < rescue_rate:
                donor = random.choice(donor_pop)

                # Transfer some genetic variation
                for gene_name, gene in donor._genes.items():
                    if gene_name in recipient._genes and random.random() < 0.3:
                        # Mix in donor allele
                        recipient._genes[gene_name].allele2 = copy.deepcopy(gene.allele1)
                        recipient._genes[gene_name].allele2.origin = f"rescue:{donor.id}"
                        rescue_events += 1

        return rescue_events

    def calculate_extinction_debt(
        self,
        habitat_loss: float,
        current_species: int
    ) -> int:
        """
        Calculate extinction debt - future extinctions
        from past habitat loss.

        Returns:
            Estimated number of species committed to extinction
        """
        # Simple species-area relationship
        # S = cA^z where z ≈ 0.25
        z = 0.25

        # Current species based on remaining habitat
        remaining_habitat = 1.0 - habitat_loss
        expected_species = current_species * (remaining_habitat ** z)

        debt = int(current_species - expected_species)
        return max(0, debt)

    def check_random_mass_extinction(
        self,
        population: List[Genome],
        generation: int
    ) -> Optional[Tuple[List[Genome], ExtinctionEvent]]:
        """
        Check for random mass extinction event.

        Returns:
            Tuple of (survivors, event) if extinction occurs, None otherwise
        """
        if random.random() < self.mass_extinction_probability:
            severity = random.uniform(0.3, 0.9)  # Variable severity
            cause = random.choice([
                "asteroid_impact",
                "volcanic_winter",
                "climate_shift",
                "pandemic",
                "gamma_ray_burst"
            ])

            # Sometimes selective
            selective_trait = None
            if random.random() < 0.5:
                # Pick random trait to be selective
                if population and population[0].gene_names:
                    selective_trait = random.choice(population[0].gene_names)

            return self.trigger_mass_extinction(
                population,
                severity=severity,
                selective_trait=selective_trait,
                generation=generation,
                cause=cause
            )

        return None

    def update_recovery_states(
        self,
        population_size: int,
        species_count: int,
        generation: int
    ) -> Dict[str, Any]:
        """
        Update all recovery states.

        Returns:
            Summary of recovery progress
        """
        updates = []

        for state in self.recovery_states.values():
            state.current_diversity = species_count
            state.update(population_size, generation)
            updates.append({
                "event_id": state.extinction_event_id,
                "phase": state.phase.name,
                "diversity": state.current_diversity
            })

        return {
            "active_recoveries": len(self.recovery_states),
            "updates": updates
        }

    def process_generation(
        self,
        population: List[Genome],
        generation: int,
        fitness_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Process extinction dynamics for a generation.

        Returns:
            Summary of extinction activity
        """
        result = {
            "generation": generation,
            "initial_population": len(population),
            "background_casualties": 0,
            "mass_extinction": None,
            "bottlenecks": [],
            "recovery": {}
        }

        survivors = population

        # Check for mass extinction
        mass_result = self.check_random_mass_extinction(survivors, generation)
        if mass_result:
            survivors, event = mass_result
            result["mass_extinction"] = event.to_dict()

        # Apply background extinction
        survivors, casualties = self.apply_background_extinction(
            survivors, fitness_scores, generation
        )
        result["background_casualties"] = len(casualties)

        # Check for bottlenecks
        species_pops: Dict[str, List[Genome]] = {}
        for genome in survivors:
            if genome.species_id:
                if genome.species_id not in species_pops:
                    species_pops[genome.species_id] = []
                species_pops[genome.species_id].append(genome)

        for species_id, members in species_pops.items():
            original = len([g for g in population if g.species_id == species_id])
            bn = self.check_for_bottleneck(
                species_id,
                len(members),
                original,
                0.5,  # Placeholder diversity
                generation
            )
            if bn:
                result["bottlenecks"].append(bn.to_dict())

        # Update recovery states
        species_count = len(species_pops)
        result["recovery"] = self.update_recovery_states(
            len(survivors), species_count, generation
        )

        result["final_population"] = len(survivors)
        result["survivors"] = survivors

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get extinction statistics"""
        mass_events = [e for e in self.extinction_events
                       if e.extinction_type == ExtinctionType.MASS]

        return {
            "total_casualties": self.total_casualties,
            "total_species_extinct": self.total_species_extinct,
            "total_events": len(self.extinction_events),
            "mass_extinctions": len(mass_events),
            "active_bottlenecks": len([b for b in self.bottlenecks.values()
                                       if not b.is_recovered]),
            "recent_events": [e.to_dict() for e in self.extinction_events[-5:]],
            "recovery_phases": {
                state.extinction_event_id: state.phase.name
                for state in self.recovery_states.values()
            }
        }


# Convenience functions

def create_stable_world() -> ExtinctionEngine:
    """Create engine with low extinction rates"""
    return ExtinctionEngine(
        background_rate=0.005,
        mass_extinction_probability=0.0001,
        minimum_viable_population=5
    )


def create_volatile_world() -> ExtinctionEngine:
    """Create engine with high extinction rates"""
    return ExtinctionEngine(
        background_rate=0.02,
        mass_extinction_probability=0.005,
        minimum_viable_population=20
    )
