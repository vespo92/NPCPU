"""
Evolutionary Innovation System

Implements major evolutionary transitions and innovations:
- Key innovations (new adaptive zones)
- Major transitions (complexity increases)
- Exaptation (co-option of existing structures)
- Adaptive radiation
- Punctuated equilibrium
- Developmental constraint relaxation

Example:
    from evolution.evolutionary_innovation import InnovationEngine

    engine = InnovationEngine()

    # Define potential innovations
    engine.add_innovation_pathway(
        name="flight",
        prerequisite_genes=["limb_structure", "lightweight_bones"],
        threshold_expressions={"limb_structure": 0.8, "lightweight_bones": 0.7},
        new_gene="flight_capability"
    )

    # Check for innovations
    innovations = engine.check_for_innovations(population, generation)
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
from .genetic_engine import Genome, Gene, Allele


class InnovationType(Enum):
    """Types of evolutionary innovations"""
    KEY_INNOVATION = auto()      # Opens new adaptive zone
    MAJOR_TRANSITION = auto()    # Increase in organizational complexity
    EXAPTATION = auto()          # Co-option of existing structure
    DEVELOPMENTAL = auto()       # New developmental pathway
    BEHAVIORAL = auto()          # Novel behavioral capability


class TransitionType(Enum):
    """Major evolutionary transitions"""
    MULTICELLULARITY = auto()
    SEXUAL_REPRODUCTION = auto()
    SOCIALITY = auto()
    LANGUAGE = auto()
    CONSCIOUSNESS = auto()
    TOOL_USE = auto()


@dataclass
class InnovationPathway:
    """
    Defines requirements for an evolutionary innovation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    innovation_type: InnovationType = InnovationType.KEY_INNOVATION
    prerequisite_genes: List[str] = field(default_factory=list)
    threshold_expressions: Dict[str, float] = field(default_factory=dict)
    probability_modifier: float = 1.0  # Multiplier for base innovation probability
    new_gene: Optional[str] = None     # Gene created by innovation
    new_gene_initial_value: float = 0.5
    fitness_boost: float = 0.2         # Immediate fitness benefit
    radiation_potential: float = 0.5   # Potential for adaptive radiation
    achieved_by: List[str] = field(default_factory=list)  # Genome IDs that achieved this

    def check_prerequisites(self, genome: Genome) -> bool:
        """Check if genome meets prerequisites"""
        for gene in self.prerequisite_genes:
            if gene not in genome.genes:
                return False

        for gene, threshold in self.threshold_expressions.items():
            if gene not in genome.genes:
                return False
            if genome.express(gene) < threshold:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.innovation_type.name,
            "prerequisites": self.prerequisite_genes,
            "thresholds": self.threshold_expressions,
            "new_gene": self.new_gene,
            "achieved_count": len(self.achieved_by)
        }


@dataclass
class InnovationEvent:
    """Record of an innovation event"""
    pathway_id: str
    genome_id: str
    generation: int
    innovation_type: InnovationType
    new_genes: List[str]
    fitness_effect: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pathway": self.pathway_id,
            "genome": self.genome_id,
            "generation": self.generation,
            "type": self.innovation_type.name,
            "new_genes": self.new_genes,
            "fitness_effect": self.fitness_effect
        }


@dataclass
class AdaptiveRadiation:
    """
    Represents an adaptive radiation event.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger_innovation: str = ""
    source_genome_id: str = ""
    generation_started: int = 0
    descendant_species: List[str] = field(default_factory=list)
    niche_diversity: float = 0.0
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trigger": self.trigger_innovation,
            "source": self.source_genome_id,
            "generation": self.generation_started,
            "descendants": len(self.descendant_species),
            "diversity": self.niche_diversity,
            "active": self.is_active
        }


class InnovationEngine:
    """
    Engine for managing evolutionary innovations.

    Features:
    - Innovation pathway tracking
    - Prerequisite checking
    - Major transition modeling
    - Exaptation detection
    - Adaptive radiation triggering
    - Punctuated equilibrium dynamics

    Example:
        engine = InnovationEngine(
            base_innovation_rate=0.001,
            enable_radiation=True
        )

        # Add innovation pathways
        engine.add_innovation_pathway(
            name="endothermy",
            prerequisite_genes=["metabolism", "insulation"],
            threshold_expressions={"metabolism": 0.8, "insulation": 0.7}
        )

        # Process each generation
        events = engine.process_generation(population, generation)
    """

    def __init__(
        self,
        base_innovation_rate: float = 0.001,
        enable_radiation: bool = True,
        stasis_duration: int = 50,     # Generations of evolutionary stasis
        burst_duration: int = 10,       # Generations of rapid change
        enable_punctuated: bool = True
    ):
        self.base_innovation_rate = base_innovation_rate
        self.enable_radiation = enable_radiation
        self.stasis_duration = stasis_duration
        self.burst_duration = burst_duration
        self.enable_punctuated = enable_punctuated

        # Innovation pathways
        self.pathways: Dict[str, InnovationPathway] = {}

        # Innovation events
        self.innovation_events: List[InnovationEvent] = []

        # Adaptive radiations
        self.radiations: Dict[str, AdaptiveRadiation] = {}

        # Punctuated equilibrium state
        self.in_stasis: bool = True
        self.stasis_counter: int = 0
        self.burst_counter: int = 0
        self.evolution_rate_multiplier: float = 0.1  # Low during stasis

        # Statistics
        self.total_innovations = 0

    def add_innovation_pathway(
        self,
        name: str,
        prerequisite_genes: List[str],
        threshold_expressions: Optional[Dict[str, float]] = None,
        innovation_type: InnovationType = InnovationType.KEY_INNOVATION,
        new_gene: Optional[str] = None,
        new_gene_value: float = 0.5,
        fitness_boost: float = 0.2
    ) -> InnovationPathway:
        """Add a potential innovation pathway"""
        threshold_expressions = threshold_expressions or {}

        # Default thresholds for prerequisites
        for gene in prerequisite_genes:
            if gene not in threshold_expressions:
                threshold_expressions[gene] = 0.7

        pathway = InnovationPathway(
            name=name,
            innovation_type=innovation_type,
            prerequisite_genes=prerequisite_genes,
            threshold_expressions=threshold_expressions,
            new_gene=new_gene,
            new_gene_initial_value=new_gene_value,
            fitness_boost=fitness_boost
        )

        self.pathways[pathway.id] = pathway
        return pathway

    def add_major_transition(
        self,
        name: str,
        transition_type: TransitionType,
        prerequisites: List[str],
        thresholds: Dict[str, float]
    ) -> InnovationPathway:
        """Add a major evolutionary transition pathway"""
        return self.add_innovation_pathway(
            name=name,
            prerequisite_genes=prerequisites,
            threshold_expressions=thresholds,
            innovation_type=InnovationType.MAJOR_TRANSITION,
            fitness_boost=0.3  # Major transitions have larger effects
        )

    def check_for_innovations(
        self,
        genome: Genome,
        generation: int
    ) -> List[InnovationEvent]:
        """
        Check if genome achieves any innovations.

        Returns:
            List of innovation events
        """
        events = []

        for pathway in self.pathways.values():
            # Skip if already achieved by this genome
            if genome.id in pathway.achieved_by:
                continue

            # Check prerequisites
            if not pathway.check_prerequisites(genome):
                continue

            # Innovation probability
            rate = self.base_innovation_rate * pathway.probability_modifier
            rate *= self.evolution_rate_multiplier  # Punctuated equilibrium effect

            if random.random() < rate:
                event = self._trigger_innovation(genome, pathway, generation)
                events.append(event)

        return events

    def _trigger_innovation(
        self,
        genome: Genome,
        pathway: InnovationPathway,
        generation: int
    ) -> InnovationEvent:
        """Trigger an innovation event"""
        new_genes = []

        # Add new gene if specified
        if pathway.new_gene:
            genome.add_gene(
                pathway.new_gene,
                pathway.new_gene_initial_value,
                dominant=True
            )
            new_genes.append(pathway.new_gene)

        # Record achievement
        pathway.achieved_by.append(genome.id)
        self.total_innovations += 1

        event = InnovationEvent(
            pathway_id=pathway.id,
            genome_id=genome.id,
            generation=generation,
            innovation_type=pathway.innovation_type,
            new_genes=new_genes,
            fitness_effect=pathway.fitness_boost
        )

        self.innovation_events.append(event)

        # Emit event
        bus = get_event_bus()
        bus.emit("innovation.achieved", {
            "pathway": pathway.name,
            "genome_id": genome.id,
            "type": pathway.innovation_type.name,
            "generation": generation
        })

        # Check for adaptive radiation
        if self.enable_radiation and pathway.radiation_potential > 0.3:
            self._trigger_radiation(genome, pathway, generation)

        return event

    def _trigger_radiation(
        self,
        genome: Genome,
        pathway: InnovationPathway,
        generation: int
    ) -> AdaptiveRadiation:
        """Trigger an adaptive radiation"""
        radiation = AdaptiveRadiation(
            trigger_innovation=pathway.name,
            source_genome_id=genome.id,
            generation_started=generation
        )

        self.radiations[radiation.id] = radiation

        # Switch to burst mode (punctuated equilibrium)
        if self.enable_punctuated and self.in_stasis:
            self.in_stasis = False
            self.burst_counter = 0
            self.evolution_rate_multiplier = 2.0

        # Emit event
        bus = get_event_bus()
        bus.emit("innovation.radiation_started", {
            "radiation_id": radiation.id,
            "trigger": pathway.name,
            "generation": generation
        })

        return radiation

    def detect_exaptation(
        self,
        genome: Genome,
        original_function: str,
        new_function: str
    ) -> Optional[InnovationEvent]:
        """
        Detect if a gene has been co-opted for a new function.

        This is a simplified model - in reality, exaptation detection
        would require tracking gene usage over time.
        """
        if original_function not in genome.genes:
            return None

        original_expr = genome.express(original_function)

        # Exaptation more likely if original function is highly expressed
        if original_expr > 0.7 and random.random() < 0.01:
            # Create exaptation pathway on the fly
            pathway = InnovationPathway(
                name=f"Exaptation_{original_function}_to_{new_function}",
                innovation_type=InnovationType.EXAPTATION,
                prerequisite_genes=[original_function],
                new_gene=new_function,
                fitness_boost=0.15
            )

            event = self._trigger_innovation(genome, pathway, genome.generation)
            return event

        return None

    def update_punctuated_equilibrium(self, generation: int) -> Dict[str, Any]:
        """
        Update punctuated equilibrium state.

        Returns:
            State information
        """
        if not self.enable_punctuated:
            return {"enabled": False}

        if self.in_stasis:
            self.stasis_counter += 1
            self.evolution_rate_multiplier = 0.1

            # Random chance to enter burst
            if random.random() < 0.01:
                self.in_stasis = False
                self.burst_counter = 0
                self.evolution_rate_multiplier = 1.5
        else:
            self.burst_counter += 1
            self.evolution_rate_multiplier = max(1.0, 2.0 - self.burst_counter * 0.1)

            # Return to stasis after burst duration
            if self.burst_counter >= self.burst_duration:
                self.in_stasis = True
                self.stasis_counter = 0
                self.evolution_rate_multiplier = 0.1

        return {
            "enabled": True,
            "in_stasis": self.in_stasis,
            "stasis_generations": self.stasis_counter if self.in_stasis else 0,
            "burst_generations": self.burst_counter if not self.in_stasis else 0,
            "rate_multiplier": self.evolution_rate_multiplier
        }

    def update_radiations(
        self,
        species_diversity: int,
        generation: int
    ) -> Dict[str, Any]:
        """
        Update active adaptive radiations.

        Args:
            species_diversity: Current number of species
            generation: Current generation
        """
        updates = []

        for radiation in self.radiations.values():
            if not radiation.is_active:
                continue

            # Update diversity
            radiation.niche_diversity = species_diversity * 0.1

            # Check if radiation is slowing
            generations_active = generation - radiation.generation_started
            if generations_active > 50:  # Radiations slow after 50 generations
                radiation.is_active = False

            updates.append({
                "id": radiation.id,
                "active": radiation.is_active,
                "generations": generations_active
            })

        return {
            "active_radiations": len([r for r in self.radiations.values() if r.is_active]),
            "updates": updates
        }

    def process_generation(
        self,
        population: List[Genome],
        generation: int
    ) -> Dict[str, Any]:
        """
        Process innovations for a generation.

        Returns:
            Summary of innovation activity
        """
        # Update punctuated equilibrium
        punctuated_state = self.update_punctuated_equilibrium(generation)

        # Check for innovations in population
        all_events = []
        for genome in population:
            events = self.check_for_innovations(genome, generation)
            all_events.extend(events)

        # Update radiations
        species_count = len(set(g.species_id for g in population if g.species_id))
        radiation_updates = self.update_radiations(species_count, generation)

        return {
            "generation": generation,
            "innovations": len(all_events),
            "innovation_events": [e.to_dict() for e in all_events],
            "punctuated_state": punctuated_state,
            "radiations": radiation_updates,
            "total_innovations_ever": self.total_innovations
        }

    def get_innovation_fitness_bonus(
        self,
        genome: Genome
    ) -> float:
        """
        Calculate total fitness bonus from innovations.
        """
        bonus = 0.0

        for pathway in self.pathways.values():
            if genome.id in pathway.achieved_by:
                bonus += pathway.fitness_boost

        return bonus

    def get_pathway_progress(
        self,
        genome: Genome
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get progress toward each innovation pathway.
        """
        progress = {}

        for pathway in self.pathways.values():
            achieved = genome.id in pathway.achieved_by
            if achieved:
                progress[pathway.name] = {"achieved": True, "progress": 1.0}
                continue

            # Calculate progress
            prereq_met = 0
            total_prereq = len(pathway.prerequisite_genes) + len(pathway.threshold_expressions)

            for gene in pathway.prerequisite_genes:
                if gene in genome.genes:
                    prereq_met += 1

            for gene, threshold in pathway.threshold_expressions.items():
                if gene in genome.genes:
                    expr = genome.express(gene)
                    if expr >= threshold:
                        prereq_met += 1
                    else:
                        prereq_met += expr / threshold * 0.5

            progress_value = prereq_met / max(1, total_prereq)

            progress[pathway.name] = {
                "achieved": False,
                "progress": progress_value,
                "missing_genes": [g for g in pathway.prerequisite_genes if g not in genome.genes]
            }

        return progress

    def get_stats(self) -> Dict[str, Any]:
        """Get innovation statistics"""
        achieved = [p for p in self.pathways.values() if p.achieved_by]

        return {
            "total_pathways": len(self.pathways),
            "achieved_pathways": len(achieved),
            "total_innovations": self.total_innovations,
            "active_radiations": len([r for r in self.radiations.values() if r.is_active]),
            "punctuated_state": "stasis" if self.in_stasis else "burst",
            "evolution_rate": self.evolution_rate_multiplier,
            "recent_innovations": [e.to_dict() for e in self.innovation_events[-5:]]
        }


# Pre-defined innovation pathways

def create_standard_pathways() -> List[InnovationPathway]:
    """Create standard evolutionary innovation pathways"""
    pathways = []

    # Multicellularity
    pathways.append(InnovationPathway(
        name="multicellularity",
        innovation_type=InnovationType.MAJOR_TRANSITION,
        prerequisite_genes=["cell_adhesion", "cell_signaling"],
        threshold_expressions={"cell_adhesion": 0.7, "cell_signaling": 0.6},
        new_gene="differentiation",
        fitness_boost=0.25,
        radiation_potential=0.8
    ))

    # Endothermy
    pathways.append(InnovationPathway(
        name="endothermy",
        innovation_type=InnovationType.KEY_INNOVATION,
        prerequisite_genes=["metabolism", "insulation"],
        threshold_expressions={"metabolism": 0.8, "insulation": 0.7},
        new_gene="thermoregulation",
        fitness_boost=0.2,
        radiation_potential=0.6
    ))

    # Flight
    pathways.append(InnovationPathway(
        name="flight",
        innovation_type=InnovationType.KEY_INNOVATION,
        prerequisite_genes=["limb_structure", "lightweight_skeleton"],
        threshold_expressions={"limb_structure": 0.8, "lightweight_skeleton": 0.8},
        new_gene="flight_capability",
        fitness_boost=0.3,
        radiation_potential=0.7
    ))

    # Intelligence
    pathways.append(InnovationPathway(
        name="advanced_intelligence",
        innovation_type=InnovationType.MAJOR_TRANSITION,
        prerequisite_genes=["brain_size", "neural_plasticity", "social_cognition"],
        threshold_expressions={
            "brain_size": 0.8,
            "neural_plasticity": 0.7,
            "social_cognition": 0.7
        },
        new_gene="abstract_reasoning",
        fitness_boost=0.35,
        radiation_potential=0.9
    ))

    return pathways
