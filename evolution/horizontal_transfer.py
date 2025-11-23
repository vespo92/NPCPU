"""
Horizontal Gene Transfer (HGT) System

Implements mechanisms for non-vertical inheritance of genetic material:
- Transformation: Uptake of free genetic material from environment
- Transduction: Gene transfer via viral-like vectors
- Conjugation: Direct organism-to-organism transfer
- Gene flow: Transfer between populations/species

Horizontal gene transfer enables rapid adaptation and innovation
by allowing beneficial traits to spread across lineage boundaries.

Example:
    from evolution.horizontal_transfer import HorizontalTransferEngine

    engine = HorizontalTransferEngine(
        transformation_rate=0.01,
        transduction_rate=0.005,
        conjugation_rate=0.02
    )

    # Transfer genes between organisms
    recipient_genome = engine.attempt_transfer(
        donor_genome,
        recipient_genome,
        transfer_type=TransferType.CONJUGATION
    )
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import math
import copy
import uuid

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.events import get_event_bus
from .genetic_engine import Genome, Gene, Allele


class TransferType(Enum):
    """Types of horizontal gene transfer"""
    TRANSFORMATION = auto()   # Uptake of free DNA from environment
    TRANSDUCTION = auto()     # Transfer via viral vectors
    CONJUGATION = auto()      # Direct cell-to-cell transfer
    GENE_FLOW = auto()        # Transfer between populations


class TransferResult(Enum):
    """Outcome of a transfer attempt"""
    SUCCESS = auto()          # Gene(s) successfully transferred
    REJECTED = auto()         # Recipient rejected foreign DNA
    INCOMPATIBLE = auto()     # Genetic incompatibility
    NO_DONOR = auto()         # No suitable donor found
    RATE_LIMIT = auto()       # Transfer rate not triggered


@dataclass
class GeneticElement:
    """
    A transferable genetic element (mobile genetic element).

    Represents a segment of DNA that can be transferred between organisms.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    genes: Dict[str, Gene] = field(default_factory=dict)
    origin_genome_id: str = ""
    origin_species: str = ""
    transfer_count: int = 0
    fitness_benefit: float = 0.0
    integration_cost: float = 0.0  # Cost to recipient

    @property
    def size(self) -> int:
        """Number of genes in this element"""
        return len(self.genes)

    def get_gene_names(self) -> List[str]:
        return list(self.genes.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genes": {name: gene.to_dict() for name, gene in self.genes.items()},
            "origin_genome_id": self.origin_genome_id,
            "origin_species": self.origin_species,
            "transfer_count": self.transfer_count,
            "fitness_benefit": self.fitness_benefit,
            "integration_cost": self.integration_cost
        }


@dataclass
class TransferRecord:
    """Record of a horizontal gene transfer event"""
    transfer_type: TransferType
    donor_id: str
    recipient_id: str
    genes_transferred: List[str]
    generation: int
    success: bool
    result: TransferResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.transfer_type.name,
            "donor_id": self.donor_id,
            "recipient_id": self.recipient_id,
            "genes": self.genes_transferred,
            "generation": self.generation,
            "success": self.success,
            "result": self.result.name
        }


@dataclass
class GenePool:
    """
    Environmental gene pool for transformation events.

    Contains free genetic material that can be taken up by organisms.
    Simulates the "gene soup" in microbial environments.
    """
    elements: List[GeneticElement] = field(default_factory=list)
    max_size: int = 1000
    decay_rate: float = 0.1  # Fraction of elements degraded per generation

    def add_element(self, element: GeneticElement) -> None:
        """Add genetic element to the pool"""
        self.elements.append(element)
        # Enforce max size
        while len(self.elements) > self.max_size:
            # Remove oldest element
            self.elements.pop(0)

    def decay(self) -> int:
        """Degrade genetic elements (called each generation)"""
        decay_count = int(len(self.elements) * self.decay_rate)
        for _ in range(decay_count):
            if self.elements:
                # Remove random element
                idx = random.randint(0, len(self.elements) - 1)
                self.elements.pop(idx)
        return decay_count

    def sample(self, n: int = 1) -> List[GeneticElement]:
        """Sample random genetic elements from pool"""
        if not self.elements:
            return []
        return random.sample(self.elements, min(n, len(self.elements)))

    def get_elements_with_gene(self, gene_name: str) -> List[GeneticElement]:
        """Find elements containing a specific gene"""
        return [e for e in self.elements if gene_name in e.genes]


class CompatibilityChecker:
    """
    Checks genetic compatibility between donor and recipient.

    Factors affecting compatibility:
    - Genetic distance
    - Species relationship
    - Gene function compatibility
    - Regulatory compatibility
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        same_species_bonus: float = 0.3,
        essential_genes: Optional[Set[str]] = None
    ):
        self.distance_threshold = distance_threshold
        self.same_species_bonus = same_species_bonus
        self.essential_genes = essential_genes or set()

    def check_compatibility(
        self,
        donor: Genome,
        recipient: Genome,
        genes_to_transfer: List[str]
    ) -> Tuple[bool, float]:
        """
        Check if gene transfer is compatible.

        Returns:
            Tuple of (is_compatible, compatibility_score)
        """
        # Calculate genetic distance
        distance = donor.distance_to(recipient)

        # Base compatibility from distance
        base_compat = max(0.0, 1.0 - distance / self.distance_threshold)

        # Same species bonus
        if donor.species_id and donor.species_id == recipient.species_id:
            base_compat += self.same_species_bonus

        # Check for essential gene conflicts
        essential_conflict = False
        for gene in genes_to_transfer:
            if gene in self.essential_genes and gene in recipient.genes:
                # Can't replace essential genes easily
                base_compat *= 0.5
                essential_conflict = True

        # Normalize compatibility score
        compat_score = min(1.0, base_compat)

        # Determine compatibility threshold
        is_compatible = compat_score > 0.3 and not essential_conflict

        return is_compatible, compat_score


class HorizontalTransferEngine:
    """
    Engine for managing horizontal gene transfer events.

    Features:
    - Multiple transfer mechanisms
    - Environmental gene pool
    - Compatibility checking
    - Transfer history tracking
    - Integration with evolution events

    Example:
        engine = HorizontalTransferEngine(
            transformation_rate=0.01,
            conjugation_rate=0.02
        )

        # Attempt conjugation between two organisms
        success, record = engine.conjugation_transfer(
            donor_genome,
            recipient_genome,
            generation=10
        )
    """

    def __init__(
        self,
        transformation_rate: float = 0.01,
        transduction_rate: float = 0.005,
        conjugation_rate: float = 0.02,
        gene_flow_rate: float = 0.01,
        max_genes_per_transfer: int = 3,
        compatibility_threshold: float = 0.5
    ):
        self.transformation_rate = transformation_rate
        self.transduction_rate = transduction_rate
        self.conjugation_rate = conjugation_rate
        self.gene_flow_rate = gene_flow_rate
        self.max_genes_per_transfer = max_genes_per_transfer

        # Components
        self.gene_pool = GenePool()
        self.compatibility_checker = CompatibilityChecker(
            distance_threshold=compatibility_threshold
        )

        # Tracking
        self.transfer_history: List[TransferRecord] = []
        self.total_transfers = 0
        self.successful_transfers = 0

    def release_genes_to_pool(
        self,
        genome: Genome,
        gene_names: Optional[List[str]] = None,
        all_genes: bool = False
    ) -> GeneticElement:
        """
        Release genetic material into the environmental gene pool.

        This simulates cell lysis, secretion, or death releasing DNA.

        Args:
            genome: Source genome
            gene_names: Specific genes to release (None = random selection)
            all_genes: If True, release entire genome

        Returns:
            The genetic element added to the pool
        """
        if all_genes:
            genes_to_release = list(genome.genes.keys())
        elif gene_names:
            genes_to_release = [g for g in gene_names if g in genome.genes]
        else:
            # Random selection
            available = list(genome.genes.keys())
            count = random.randint(1, min(3, len(available)))
            genes_to_release = random.sample(available, count)

        # Create genetic element
        element = GeneticElement(
            genes={name: copy.deepcopy(genome.genes[name])
                   for name in genes_to_release if name in genome.genes},
            origin_genome_id=genome.id,
            origin_species=genome.species_id or ""
        )

        # Calculate fitness benefit (based on gene expression values)
        if element.genes:
            avg_expression = sum(
                g.express() for g in element.genes.values()
            ) / len(element.genes)
            element.fitness_benefit = avg_expression * 0.1

        self.gene_pool.add_element(element)

        # Emit event
        bus = get_event_bus()
        bus.emit("hgt.genes_released", {
            "genome_id": genome.id,
            "genes": genes_to_release,
            "element_id": element.id
        })

        return element

    def transformation_transfer(
        self,
        recipient: Genome,
        generation: int = 0
    ) -> Tuple[bool, Optional[TransferRecord]]:
        """
        Attempt transformation - uptake of environmental DNA.

        The recipient takes up free genetic material from the gene pool.

        Returns:
            Tuple of (success, transfer_record)
        """
        # Check rate
        if random.random() > self.transformation_rate:
            return False, None

        # Sample from gene pool
        elements = self.gene_pool.sample(1)
        if not elements:
            record = TransferRecord(
                transfer_type=TransferType.TRANSFORMATION,
                donor_id="environment",
                recipient_id=recipient.id,
                genes_transferred=[],
                generation=generation,
                success=False,
                result=TransferResult.NO_DONOR
            )
            return False, record

        element = elements[0]
        genes_to_transfer = list(element.genes.keys())[:self.max_genes_per_transfer]

        # Create dummy genome for compatibility check
        dummy_donor = Genome(genes=element.genes)

        is_compatible, compat_score = self.compatibility_checker.check_compatibility(
            dummy_donor, recipient, genes_to_transfer
        )

        if not is_compatible:
            record = TransferRecord(
                transfer_type=TransferType.TRANSFORMATION,
                donor_id=element.origin_genome_id,
                recipient_id=recipient.id,
                genes_transferred=genes_to_transfer,
                generation=generation,
                success=False,
                result=TransferResult.INCOMPATIBLE
            )
            self.transfer_history.append(record)
            return False, record

        # Perform transfer
        transferred = self._integrate_genes(recipient, element.genes, genes_to_transfer)
        element.transfer_count += 1

        record = TransferRecord(
            transfer_type=TransferType.TRANSFORMATION,
            donor_id=element.origin_genome_id,
            recipient_id=recipient.id,
            genes_transferred=transferred,
            generation=generation,
            success=True,
            result=TransferResult.SUCCESS
        )

        self._record_transfer(record)
        return True, record

    def conjugation_transfer(
        self,
        donor: Genome,
        recipient: Genome,
        gene_names: Optional[List[str]] = None,
        generation: int = 0
    ) -> Tuple[bool, Optional[TransferRecord]]:
        """
        Attempt conjugation - direct organism-to-organism transfer.

        Requires physical proximity (simulated by being in same population).

        Args:
            donor: Source genome
            recipient: Target genome
            gene_names: Specific genes to transfer (None = random)
            generation: Current generation

        Returns:
            Tuple of (success, transfer_record)
        """
        # Check rate
        if random.random() > self.conjugation_rate:
            return False, None

        # Select genes to transfer
        if gene_names:
            genes_to_transfer = [g for g in gene_names
                                 if g in donor.genes][:self.max_genes_per_transfer]
        else:
            available = list(donor.genes.keys())
            count = random.randint(1, min(self.max_genes_per_transfer, len(available)))
            genes_to_transfer = random.sample(available, count)

        if not genes_to_transfer:
            return False, None

        # Check compatibility
        is_compatible, compat_score = self.compatibility_checker.check_compatibility(
            donor, recipient, genes_to_transfer
        )

        if not is_compatible:
            record = TransferRecord(
                transfer_type=TransferType.CONJUGATION,
                donor_id=donor.id,
                recipient_id=recipient.id,
                genes_transferred=genes_to_transfer,
                generation=generation,
                success=False,
                result=TransferResult.INCOMPATIBLE
            )
            self.transfer_history.append(record)
            return False, record

        # Transfer with probability based on compatibility
        if random.random() > compat_score:
            record = TransferRecord(
                transfer_type=TransferType.CONJUGATION,
                donor_id=donor.id,
                recipient_id=recipient.id,
                genes_transferred=genes_to_transfer,
                generation=generation,
                success=False,
                result=TransferResult.REJECTED
            )
            self.transfer_history.append(record)
            return False, record

        # Perform transfer
        genes_dict = {name: copy.deepcopy(donor.genes[name])
                      for name in genes_to_transfer}
        transferred = self._integrate_genes(recipient, genes_dict, genes_to_transfer)

        record = TransferRecord(
            transfer_type=TransferType.CONJUGATION,
            donor_id=donor.id,
            recipient_id=recipient.id,
            genes_transferred=transferred,
            generation=generation,
            success=True,
            result=TransferResult.SUCCESS
        )

        self._record_transfer(record)
        return True, record

    def transduction_transfer(
        self,
        donor: Genome,
        recipient: Genome,
        generation: int = 0
    ) -> Tuple[bool, Optional[TransferRecord]]:
        """
        Attempt transduction - viral-mediated gene transfer.

        Simulates phage/virus picking up donor genes and
        injecting them into recipient.

        More random than conjugation, can transfer any genes.
        """
        # Check rate
        if random.random() > self.transduction_rate:
            return False, None

        # Random gene selection (viruses don't choose)
        available = list(donor.genes.keys())
        if not available:
            return False, None

        count = random.randint(1, min(2, len(available)))  # Viruses carry less
        genes_to_transfer = random.sample(available, count)

        # Transduction has lower compatibility requirements
        # (viruses inject DNA more forcefully)
        donor_distance = donor.distance_to(recipient)
        success_prob = max(0.1, 0.8 - donor_distance)

        if random.random() > success_prob:
            record = TransferRecord(
                transfer_type=TransferType.TRANSDUCTION,
                donor_id=donor.id,
                recipient_id=recipient.id,
                genes_transferred=genes_to_transfer,
                generation=generation,
                success=False,
                result=TransferResult.REJECTED
            )
            self.transfer_history.append(record)
            return False, record

        # Perform transfer
        genes_dict = {name: copy.deepcopy(donor.genes[name])
                      for name in genes_to_transfer}
        transferred = self._integrate_genes(recipient, genes_dict, genes_to_transfer)

        record = TransferRecord(
            transfer_type=TransferType.TRANSDUCTION,
            donor_id=donor.id,
            recipient_id=recipient.id,
            genes_transferred=transferred,
            generation=generation,
            success=True,
            result=TransferResult.SUCCESS
        )

        self._record_transfer(record)
        return True, record

    def gene_flow_transfer(
        self,
        source_population: List[Genome],
        target_population: List[Genome],
        generation: int = 0
    ) -> List[TransferRecord]:
        """
        Simulate gene flow between populations.

        Represents migration and interbreeding between
        distinct population groups.

        Returns:
            List of transfer records
        """
        records = []

        for target in target_population:
            if random.random() > self.gene_flow_rate:
                continue

            # Select random donor from source population
            if not source_population:
                continue
            donor = random.choice(source_population)

            # Gene flow is more permissive than other mechanisms
            available = list(donor.genes.keys())
            if not available:
                continue

            count = random.randint(1, min(self.max_genes_per_transfer, len(available)))
            genes_to_transfer = random.sample(available, count)

            # Transfer with high probability (populations are already mixing)
            genes_dict = {name: copy.deepcopy(donor.genes[name])
                          for name in genes_to_transfer}
            transferred = self._integrate_genes(target, genes_dict, genes_to_transfer)

            record = TransferRecord(
                transfer_type=TransferType.GENE_FLOW,
                donor_id=donor.id,
                recipient_id=target.id,
                genes_transferred=transferred,
                generation=generation,
                success=True,
                result=TransferResult.SUCCESS
            )

            self._record_transfer(record)
            records.append(record)

        return records

    def _integrate_genes(
        self,
        recipient: Genome,
        genes: Dict[str, Gene],
        gene_names: List[str]
    ) -> List[str]:
        """
        Integrate transferred genes into recipient genome.

        Handles both new genes and gene replacement.

        Returns:
            List of successfully integrated gene names
        """
        integrated = []

        for name in gene_names:
            if name not in genes:
                continue

            gene = genes[name]

            if name in recipient._genes:
                # Replace one allele (recombination)
                existing = recipient._genes[name]
                if random.random() < 0.5:
                    existing.allele1 = copy.deepcopy(gene.allele1)
                    existing.allele1.origin = f"HGT:{gene.allele1.origin}"
                else:
                    existing.allele2 = copy.deepcopy(gene.allele2)
                    existing.allele2.origin = f"HGT:{gene.allele2.origin}"
            else:
                # Add new gene
                new_gene = copy.deepcopy(gene)
                new_gene.allele1.origin = f"HGT:{gene.allele1.origin}"
                new_gene.allele2.origin = f"HGT:{gene.allele2.origin}"
                recipient._genes[name] = new_gene

            integrated.append(name)

        return integrated

    def _record_transfer(self, record: TransferRecord) -> None:
        """Record a transfer event"""
        self.transfer_history.append(record)
        self.total_transfers += 1
        if record.success:
            self.successful_transfers += 1

        # Emit event
        bus = get_event_bus()
        bus.emit("hgt.transfer", record.to_dict())

    def process_generation(
        self,
        population: List[Genome],
        generation: int
    ) -> Dict[str, Any]:
        """
        Process HGT events for a generation.

        Should be called each generation during evolution.

        Returns:
            Statistics about HGT events
        """
        # Decay gene pool
        decayed = self.gene_pool.decay()

        # Some organisms release genes (simulating death/lysis)
        release_rate = 0.05
        for genome in population:
            if random.random() < release_rate:
                self.release_genes_to_pool(genome)

        # Attempt transfers
        transformations = 0
        transductions = 0
        conjugations = 0

        for recipient in population:
            # Transformation from environment
            success, _ = self.transformation_transfer(recipient, generation)
            if success:
                transformations += 1

            # Transduction and conjugation require donor
            if len(population) < 2:
                continue

            # Pick random potential donor
            donor = random.choice([g for g in population if g.id != recipient.id])

            success, _ = self.transduction_transfer(donor, recipient, generation)
            if success:
                transductions += 1

            success, _ = self.conjugation_transfer(donor, recipient, generation=generation)
            if success:
                conjugations += 1

        return {
            "generation": generation,
            "gene_pool_size": len(self.gene_pool.elements),
            "elements_decayed": decayed,
            "transformations": transformations,
            "transductions": transductions,
            "conjugations": conjugations,
            "total_hgt_events": transformations + transductions + conjugations
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get HGT statistics"""
        by_type: Dict[str, int] = {}
        for record in self.transfer_history:
            type_name = record.transfer_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1

        success_rate = (
            self.successful_transfers / self.total_transfers
            if self.total_transfers > 0 else 0
        )

        return {
            "total_transfers": self.total_transfers,
            "successful_transfers": self.successful_transfers,
            "success_rate": success_rate,
            "by_type": by_type,
            "gene_pool_size": len(self.gene_pool.elements),
            "recent_transfers": [r.to_dict() for r in self.transfer_history[-10:]]
        }


# Convenience functions

def create_hgt_engine(
    mode: str = "bacterial"
) -> HorizontalTransferEngine:
    """
    Create an HGT engine with preset configurations.

    Modes:
    - bacterial: High transformation and conjugation rates
    - archaeal: Moderate rates
    - eukaryotic: Low rates, mostly via viruses
    - minimal: Very low rates for stable evolution
    """
    configs = {
        "bacterial": {
            "transformation_rate": 0.02,
            "transduction_rate": 0.01,
            "conjugation_rate": 0.03,
            "gene_flow_rate": 0.02
        },
        "archaeal": {
            "transformation_rate": 0.01,
            "transduction_rate": 0.005,
            "conjugation_rate": 0.015,
            "gene_flow_rate": 0.01
        },
        "eukaryotic": {
            "transformation_rate": 0.001,
            "transduction_rate": 0.005,
            "conjugation_rate": 0.001,
            "gene_flow_rate": 0.005
        },
        "minimal": {
            "transformation_rate": 0.001,
            "transduction_rate": 0.001,
            "conjugation_rate": 0.002,
            "gene_flow_rate": 0.001
        }
    }

    config = configs.get(mode, configs["bacterial"])
    return HorizontalTransferEngine(**config)
