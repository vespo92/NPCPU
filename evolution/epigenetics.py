"""
Epigenetic Inheritance System

Implements non-genetic inheritance mechanisms that affect gene expression
without altering the underlying DNA sequence:
- DNA methylation patterns
- Histone modifications
- Environmental imprinting
- Transgenerational memory
- Gene silencing/activation

Epigenetics enables adaptive responses to environmental conditions
that can persist across generations.

Example:
    from evolution.epigenetics import EpigeneticLayer, EnvironmentalStressor

    layer = EpigeneticLayer()

    # Apply environmental stress
    stressor = EnvironmentalStressor(
        name="drought",
        affected_genes=["water_retention", "metabolism"],
        intensity=0.7
    )

    layer.apply_stressor(genome, stressor)

    # Epigenetic marks affect gene expression
    modified_expression = layer.get_modified_expression(genome, "water_retention")
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
from .genetic_engine import Genome, Gene


class EpigeneticMarkType(Enum):
    """Types of epigenetic modifications"""
    METHYLATION = auto()      # DNA methylation (typically silencing)
    ACETYLATION = auto()      # Histone acetylation (activation)
    PHOSPHORYLATION = auto()  # Histone phosphorylation
    UBIQUITINATION = auto()   # Protein ubiquitination
    IMPRINTING = auto()       # Parental imprinting
    ENVIRONMENTAL = auto()    # Environment-induced marks


class ExpressionModifier(Enum):
    """How epigenetic marks modify expression"""
    SILENCED = auto()         # Gene completely silenced
    REDUCED = auto()          # Expression reduced
    NORMAL = auto()           # No modification
    ENHANCED = auto()         # Expression enhanced
    OVEREXPRESSED = auto()    # Gene highly overexpressed


@dataclass
class EpigeneticMark:
    """
    Represents an epigenetic modification on a gene.

    Marks can persist across generations with decay.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mark_type: EpigeneticMarkType = EpigeneticMarkType.METHYLATION
    target_gene: str = ""
    intensity: float = 0.5  # 0.0 to 1.0
    modifier: ExpressionModifier = ExpressionModifier.NORMAL
    generations_remaining: int = 3  # How many generations mark persists
    origin_generation: int = 0
    cause: str = ""  # What triggered this mark

    @property
    def expression_multiplier(self) -> float:
        """Calculate expression multiplier based on modifier"""
        multipliers = {
            ExpressionModifier.SILENCED: 0.0,
            ExpressionModifier.REDUCED: 0.5,
            ExpressionModifier.NORMAL: 1.0,
            ExpressionModifier.ENHANCED: 1.5,
            ExpressionModifier.OVEREXPRESSED: 2.0
        }
        base = multipliers.get(self.modifier, 1.0)
        # Scale by intensity
        if self.modifier in [ExpressionModifier.SILENCED, ExpressionModifier.REDUCED]:
            return 1.0 - (1.0 - base) * self.intensity
        elif self.modifier in [ExpressionModifier.ENHANCED, ExpressionModifier.OVEREXPRESSED]:
            return 1.0 + (base - 1.0) * self.intensity
        return base

    def decay(self) -> bool:
        """
        Decay the mark by one generation.

        Returns:
            True if mark should be removed
        """
        self.generations_remaining -= 1
        self.intensity *= 0.8  # Marks fade over time
        return self.generations_remaining <= 0 or self.intensity < 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.mark_type.name,
            "target_gene": self.target_gene,
            "intensity": self.intensity,
            "modifier": self.modifier.name,
            "generations_remaining": self.generations_remaining,
            "expression_multiplier": self.expression_multiplier,
            "cause": self.cause
        }


@dataclass
class EnvironmentalStressor:
    """
    Environmental condition that can induce epigenetic changes.

    Examples: drought, heat, toxins, social stress, abundance
    """
    name: str
    affected_genes: List[str]
    intensity: float = 0.5
    activation_genes: Optional[List[str]] = None  # Genes to activate
    silencing_genes: Optional[List[str]] = None   # Genes to silence
    persistence: int = 3  # Generations of effect

    def get_mark_for_gene(self, gene_name: str) -> Optional[EpigeneticMark]:
        """Generate appropriate mark for affected gene"""
        if gene_name not in self.affected_genes:
            return None

        # Determine if activating or silencing
        if self.activation_genes and gene_name in self.activation_genes:
            modifier = (
                ExpressionModifier.OVEREXPRESSED if self.intensity > 0.7
                else ExpressionModifier.ENHANCED
            )
            mark_type = EpigeneticMarkType.ACETYLATION
        elif self.silencing_genes and gene_name in self.silencing_genes:
            modifier = (
                ExpressionModifier.SILENCED if self.intensity > 0.7
                else ExpressionModifier.REDUCED
            )
            mark_type = EpigeneticMarkType.METHYLATION
        else:
            # Default to environmental mark with reduced expression
            modifier = ExpressionModifier.REDUCED
            mark_type = EpigeneticMarkType.ENVIRONMENTAL

        return EpigeneticMark(
            mark_type=mark_type,
            target_gene=gene_name,
            intensity=self.intensity,
            modifier=modifier,
            generations_remaining=self.persistence,
            cause=self.name
        )


@dataclass
class Epigenome:
    """
    Complete epigenetic state of a genome.

    Tracks all epigenetic marks and provides expression modification.
    """
    genome_id: str = ""
    marks: Dict[str, List[EpigeneticMark]] = field(default_factory=dict)
    generation: int = 0
    parent_epigenome_id: Optional[str] = None

    def add_mark(self, mark: EpigeneticMark) -> None:
        """Add an epigenetic mark"""
        if mark.target_gene not in self.marks:
            self.marks[mark.target_gene] = []
        self.marks[mark.target_gene].append(mark)

    def remove_mark(self, mark_id: str) -> bool:
        """Remove a specific mark"""
        for gene, marks in self.marks.items():
            for i, mark in enumerate(marks):
                if mark.id == mark_id:
                    marks.pop(i)
                    return True
        return False

    def get_marks_for_gene(self, gene_name: str) -> List[EpigeneticMark]:
        """Get all marks affecting a gene"""
        return self.marks.get(gene_name, [])

    def get_expression_modifier(self, gene_name: str) -> float:
        """
        Calculate combined expression modifier for a gene.

        Multiple marks combine multiplicatively.
        """
        marks = self.get_marks_for_gene(gene_name)
        if not marks:
            return 1.0

        combined = 1.0
        for mark in marks:
            combined *= mark.expression_multiplier

        return max(0.0, min(2.5, combined))  # Clamp to reasonable range

    def decay_marks(self) -> List[str]:
        """
        Decay all marks by one generation.

        Returns:
            List of mark IDs that were removed
        """
        removed = []
        for gene in list(self.marks.keys()):
            surviving = []
            for mark in self.marks[gene]:
                if mark.decay():
                    removed.append(mark.id)
                else:
                    surviving.append(mark)
            self.marks[gene] = surviving
            if not surviving:
                del self.marks[gene]
        return removed

    def inherit_from(
        self,
        parent: 'Epigenome',
        inheritance_rate: float = 0.7
    ) -> int:
        """
        Inherit epigenetic marks from parent.

        Args:
            parent: Parent epigenome
            inheritance_rate: Probability of inheriting each mark

        Returns:
            Number of marks inherited
        """
        inherited = 0
        self.parent_epigenome_id = parent.genome_id

        for gene, marks in parent.marks.items():
            for mark in marks:
                if random.random() < inheritance_rate:
                    # Create inherited copy with reduced intensity
                    new_mark = copy.deepcopy(mark)
                    new_mark.id = str(uuid.uuid4())[:8]
                    new_mark.intensity *= 0.7  # Dilution effect
                    new_mark.generations_remaining -= 1
                    if new_mark.generations_remaining > 0 and new_mark.intensity > 0.1:
                        self.add_mark(new_mark)
                        inherited += 1

        return inherited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_id": self.parent_epigenome_id,
            "marks": {
                gene: [m.to_dict() for m in marks]
                for gene, marks in self.marks.items()
            },
            "total_marks": sum(len(m) for m in self.marks.values())
        }


class EpigeneticLayer:
    """
    Manages epigenetic inheritance for a population.

    Features:
    - Tracks epigenomes for all organisms
    - Applies environmental stressors
    - Handles transgenerational inheritance
    - Modifies gene expression
    - Integrates with evolution engine

    Example:
        layer = EpigeneticLayer()

        # Register genomes
        layer.register_genome(genome1)
        layer.register_genome(genome2)

        # Apply environmental stress
        stressor = EnvironmentalStressor(
            name="heat_shock",
            affected_genes=["heat_tolerance", "metabolism"],
            intensity=0.8,
            activation_genes=["heat_tolerance"],
            silencing_genes=["metabolism"]
        )

        layer.apply_stressor(genome1, stressor)

        # Get modified expression
        expr = layer.get_modified_expression(genome1, "heat_tolerance")
    """

    def __init__(
        self,
        inheritance_rate: float = 0.7,
        spontaneous_rate: float = 0.01,
        max_marks_per_gene: int = 5
    ):
        self.inheritance_rate = inheritance_rate
        self.spontaneous_rate = spontaneous_rate
        self.max_marks_per_gene = max_marks_per_gene

        # Genome ID -> Epigenome
        self.epigenomes: Dict[str, Epigenome] = {}

        # Active stressors in environment
        self.active_stressors: List[EnvironmentalStressor] = []

        # Statistics
        self.total_marks_applied = 0
        self.total_marks_inherited = 0

    def register_genome(self, genome: Genome) -> Epigenome:
        """Register a genome and create its epigenome"""
        if genome.id not in self.epigenomes:
            self.epigenomes[genome.id] = Epigenome(
                genome_id=genome.id,
                generation=genome.generation
            )
        return self.epigenomes[genome.id]

    def get_epigenome(self, genome: Genome) -> Epigenome:
        """Get or create epigenome for a genome"""
        return self.register_genome(genome)

    def apply_mark(
        self,
        genome: Genome,
        mark: EpigeneticMark,
        generation: int = 0
    ) -> bool:
        """
        Apply an epigenetic mark to a genome.

        Returns:
            True if mark was applied
        """
        epigenome = self.get_epigenome(genome)

        # Check mark limit
        existing = epigenome.get_marks_for_gene(mark.target_gene)
        if len(existing) >= self.max_marks_per_gene:
            # Remove oldest mark
            if existing:
                epigenome.remove_mark(existing[0].id)

        mark.origin_generation = generation
        epigenome.add_mark(mark)
        self.total_marks_applied += 1

        # Emit event
        bus = get_event_bus()
        bus.emit("epigenetics.mark_applied", {
            "genome_id": genome.id,
            "mark": mark.to_dict()
        })

        return True

    def apply_stressor(
        self,
        genome: Genome,
        stressor: EnvironmentalStressor,
        generation: int = 0
    ) -> List[EpigeneticMark]:
        """
        Apply environmental stressor to a genome.

        Returns:
            List of marks that were applied
        """
        applied_marks = []

        for gene_name in stressor.affected_genes:
            if gene_name in genome.genes:
                mark = stressor.get_mark_for_gene(gene_name)
                if mark:
                    self.apply_mark(genome, mark, generation)
                    applied_marks.append(mark)

        return applied_marks

    def apply_stressor_to_population(
        self,
        population: List[Genome],
        stressor: EnvironmentalStressor,
        exposure_rate: float = 1.0,
        generation: int = 0
    ) -> Dict[str, Any]:
        """
        Apply stressor to entire population.

        Args:
            population: List of genomes
            stressor: Stressor to apply
            exposure_rate: Fraction of population exposed
            generation: Current generation

        Returns:
            Statistics about stressor application
        """
        affected = 0
        total_marks = 0

        for genome in population:
            if random.random() < exposure_rate:
                marks = self.apply_stressor(genome, stressor, generation)
                if marks:
                    affected += 1
                    total_marks += len(marks)

        return {
            "stressor": stressor.name,
            "population_size": len(population),
            "affected": affected,
            "marks_applied": total_marks
        }

    def add_environmental_stressor(self, stressor: EnvironmentalStressor) -> None:
        """Add a persistent environmental stressor"""
        self.active_stressors.append(stressor)

    def remove_environmental_stressor(self, stressor_name: str) -> bool:
        """Remove a stressor by name"""
        for i, stressor in enumerate(self.active_stressors):
            if stressor.name == stressor_name:
                self.active_stressors.pop(i)
                return True
        return False

    def get_modified_expression(
        self,
        genome: Genome,
        gene_name: str
    ) -> float:
        """
        Get modified gene expression value.

        Combines genetic expression with epigenetic modifications.
        """
        if gene_name not in genome.genes:
            return 0.0

        base_expression = genome.express(gene_name)
        epigenome = self.get_epigenome(genome)
        modifier = epigenome.get_expression_modifier(gene_name)

        return max(0.0, min(1.0, base_expression * modifier))

    def get_all_modified_expressions(
        self,
        genome: Genome
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get all gene expressions with epigenetic modifications.

        Returns:
            Dict mapping gene name to (base_expression, modified_expression)
        """
        result = {}
        for gene_name in genome.gene_names:
            base = genome.express(gene_name)
            modified = self.get_modified_expression(genome, gene_name)
            result[gene_name] = (base, modified)
        return result

    def handle_reproduction(
        self,
        parent1: Genome,
        parent2: Optional[Genome],
        offspring: Genome,
        generation: int = 0
    ) -> int:
        """
        Handle epigenetic inheritance during reproduction.

        Returns:
            Number of marks inherited by offspring
        """
        offspring_epigenome = self.register_genome(offspring)
        offspring_epigenome.generation = generation

        inherited = 0

        # Inherit from parent 1
        if parent1.id in self.epigenomes:
            parent1_epi = self.epigenomes[parent1.id]
            inherited += offspring_epigenome.inherit_from(
                parent1_epi,
                self.inheritance_rate
            )

        # Inherit from parent 2 (if exists)
        if parent2 and parent2.id in self.epigenomes:
            parent2_epi = self.epigenomes[parent2.id]
            # Lower rate for second parent to avoid mark overload
            inherited += offspring_epigenome.inherit_from(
                parent2_epi,
                self.inheritance_rate * 0.5
            )

        self.total_marks_inherited += inherited

        # Emit event
        bus = get_event_bus()
        bus.emit("epigenetics.inheritance", {
            "offspring_id": offspring.id,
            "marks_inherited": inherited,
            "generation": generation
        })

        return inherited

    def apply_spontaneous_marks(
        self,
        genome: Genome,
        generation: int = 0
    ) -> List[EpigeneticMark]:
        """
        Apply random spontaneous epigenetic changes.

        These represent stochastic epigenetic drift.
        """
        applied = []

        for gene_name in genome.gene_names:
            if random.random() < self.spontaneous_rate:
                # Random mark type and effect
                mark_type = random.choice(list(EpigeneticMarkType))
                modifier = random.choice([
                    ExpressionModifier.REDUCED,
                    ExpressionModifier.ENHANCED
                ])
                intensity = random.uniform(0.2, 0.5)

                mark = EpigeneticMark(
                    mark_type=mark_type,
                    target_gene=gene_name,
                    intensity=intensity,
                    modifier=modifier,
                    generations_remaining=random.randint(1, 3),
                    cause="spontaneous"
                )

                self.apply_mark(genome, mark, generation)
                applied.append(mark)

        return applied

    def process_generation(
        self,
        population: List[Genome],
        generation: int
    ) -> Dict[str, Any]:
        """
        Process epigenetic changes for a generation.

        Should be called each generation.
        """
        # Decay existing marks
        total_decayed = 0
        for genome in population:
            if genome.id in self.epigenomes:
                decayed = self.epigenomes[genome.id].decay_marks()
                total_decayed += len(decayed)

        # Apply active environmental stressors
        stressor_stats = []
        for stressor in self.active_stressors:
            stats = self.apply_stressor_to_population(
                population, stressor, generation=generation
            )
            stressor_stats.append(stats)

        # Apply spontaneous marks
        spontaneous = 0
        for genome in population:
            marks = self.apply_spontaneous_marks(genome, generation)
            spontaneous += len(marks)

        # Clean up epigenomes for dead genomes
        active_ids = {g.id for g in population}
        dead_ids = [gid for gid in self.epigenomes.keys() if gid not in active_ids]
        for gid in dead_ids:
            del self.epigenomes[gid]

        return {
            "generation": generation,
            "marks_decayed": total_decayed,
            "spontaneous_marks": spontaneous,
            "active_stressors": len(self.active_stressors),
            "stressor_effects": stressor_stats,
            "tracked_genomes": len(self.epigenomes)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get epigenetic layer statistics"""
        total_marks = sum(
            sum(len(marks) for marks in epi.marks.values())
            for epi in self.epigenomes.values()
        )

        marks_by_type: Dict[str, int] = {}
        for epi in self.epigenomes.values():
            for marks in epi.marks.values():
                for mark in marks:
                    type_name = mark.mark_type.name
                    marks_by_type[type_name] = marks_by_type.get(type_name, 0) + 1

        return {
            "tracked_genomes": len(self.epigenomes),
            "total_active_marks": total_marks,
            "marks_by_type": marks_by_type,
            "total_marks_applied": self.total_marks_applied,
            "total_marks_inherited": self.total_marks_inherited,
            "active_stressors": [s.name for s in self.active_stressors]
        }


# Convenience functions

def create_common_stressors() -> Dict[str, EnvironmentalStressor]:
    """Create a dictionary of common environmental stressors"""
    return {
        "drought": EnvironmentalStressor(
            name="drought",
            affected_genes=["water_retention", "metabolism", "growth_rate"],
            intensity=0.7,
            activation_genes=["water_retention"],
            silencing_genes=["growth_rate"],
            persistence=4
        ),
        "heat_shock": EnvironmentalStressor(
            name="heat_shock",
            affected_genes=["heat_tolerance", "protein_folding", "metabolism"],
            intensity=0.8,
            activation_genes=["heat_tolerance", "protein_folding"],
            silencing_genes=["metabolism"],
            persistence=3
        ),
        "nutrient_abundance": EnvironmentalStressor(
            name="nutrient_abundance",
            affected_genes=["growth_rate", "reproduction_rate", "storage"],
            intensity=0.6,
            activation_genes=["growth_rate", "reproduction_rate"],
            persistence=2
        ),
        "social_stress": EnvironmentalStressor(
            name="social_stress",
            affected_genes=["aggression", "cooperation", "stress_response"],
            intensity=0.5,
            activation_genes=["stress_response"],
            silencing_genes=["cooperation"],
            persistence=3
        ),
        "toxin_exposure": EnvironmentalStressor(
            name="toxin_exposure",
            affected_genes=["detoxification", "metabolism", "reproduction_rate"],
            intensity=0.7,
            activation_genes=["detoxification"],
            silencing_genes=["reproduction_rate"],
            persistence=5
        )
    }
