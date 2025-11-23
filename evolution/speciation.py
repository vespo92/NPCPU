"""
Advanced Speciation Engine

Implements mechanisms for species formation and divergence:
- Allopatric speciation (geographic isolation)
- Sympatric speciation (same area, different niches)
- Parapatric speciation (partial isolation)
- Peripatric speciation (peripheral populations)
- Reproductive isolation barriers
- Hybrid zones
- Species tracking and phylogeny

Example:
    from evolution.speciation import SpeciationEngine, IsolationBarrier

    engine = SpeciationEngine(
        genetic_distance_threshold=0.4,
        isolation_threshold=0.7
    )

    # Check for speciation events
    new_species = engine.check_speciation(population, generation=50)

    # Track species relationships
    engine.build_phylogeny()
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import math
import copy
import uuid
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.events import get_event_bus
from .genetic_engine import Genome, Gene, Species


class SpeciationType(Enum):
    """Types of speciation events"""
    ALLOPATRIC = auto()   # Geographic isolation
    SYMPATRIC = auto()    # Same area, different niches
    PARAPATRIC = auto()   # Partial isolation
    PERIPATRIC = auto()   # Peripheral population
    HYBRIDIZATION = auto() # New species from hybrid


class IsolationType(Enum):
    """Types of reproductive isolation barriers"""
    PREZYGOTIC_TEMPORAL = auto()    # Different breeding times
    PREZYGOTIC_BEHAVIORAL = auto()  # Different mating behaviors
    PREZYGOTIC_MECHANICAL = auto()  # Physical incompatibility
    PREZYGOTIC_GAMETIC = auto()     # Gamete incompatibility
    POSTZYGOTIC_INVIABILITY = auto()  # Hybrid inviability
    POSTZYGOTIC_STERILITY = auto()    # Hybrid sterility
    POSTZYGOTIC_BREAKDOWN = auto()    # F2 hybrid breakdown


@dataclass
class IsolationBarrier:
    """
    Reproductive isolation barrier between populations/species.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    isolation_type: IsolationType = IsolationType.PREZYGOTIC_BEHAVIORAL
    strength: float = 0.5  # 0.0 = no barrier, 1.0 = complete isolation
    species1_id: str = ""
    species2_id: str = ""
    generation_formed: int = 0
    cause: str = ""

    def get_reproduction_probability(self) -> float:
        """Probability that cross-species reproduction succeeds"""
        return 1.0 - self.strength

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.isolation_type.name,
            "strength": self.strength,
            "species1": self.species1_id,
            "species2": self.species2_id,
            "generation": self.generation_formed,
            "cause": self.cause
        }


@dataclass
class SpeciesRecord:
    """
    Extended species information for tracking.
    """
    species: Species
    parent_species_id: Optional[str] = None
    speciation_type: Optional[SpeciationType] = None
    extinction_generation: Optional[int] = None
    peak_population: int = 0
    total_generations: int = 0
    niche: str = ""
    geographic_region: str = ""
    distinctive_traits: List[str] = field(default_factory=list)

    @property
    def is_extinct(self) -> bool:
        return self.extinction_generation is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.species.id,
            "name": self.species.name,
            "parent_species": self.parent_species_id,
            "speciation_type": self.speciation_type.name if self.speciation_type else None,
            "created_generation": self.species.created_generation,
            "extinction_generation": self.extinction_generation,
            "is_extinct": self.is_extinct,
            "peak_population": self.peak_population,
            "total_generations": self.total_generations,
            "niche": self.niche,
            "region": self.geographic_region,
            "distinctive_traits": self.distinctive_traits
        }


@dataclass
class HybridZone:
    """
    Zone where two species can hybridize.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    species1_id: str = ""
    species2_id: str = ""
    hybrid_viability: float = 0.5  # Probability hybrid survives
    hybrid_fertility: float = 0.5   # Probability hybrid can reproduce
    generation_formed: int = 0
    hybrid_genome_ids: List[str] = field(default_factory=list)

    def can_produce_viable_hybrid(self) -> bool:
        """Check if viable hybrid can be produced"""
        return random.random() < self.hybrid_viability

    def is_hybrid_fertile(self) -> bool:
        """Check if hybrid can reproduce"""
        return random.random() < self.hybrid_fertility


class SpeciationEngine:
    """
    Engine for managing speciation events and species tracking.

    Features:
    - Automatic speciation detection
    - Multiple speciation modes
    - Reproductive isolation modeling
    - Hybrid zone management
    - Phylogenetic tracking
    - Species diversity metrics

    Example:
        engine = SpeciationEngine(
            genetic_distance_threshold=0.4,
            min_population_for_species=5
        )

        # Check for speciation each generation
        events = engine.check_speciation(population, generation)

        # Get species diversity
        diversity = engine.calculate_species_diversity()
    """

    def __init__(
        self,
        genetic_distance_threshold: float = 0.4,
        isolation_threshold: float = 0.7,
        min_population_for_species: int = 5,
        speciation_cooldown: int = 10,  # Generations between speciation events
        enable_hybridization: bool = True
    ):
        self.genetic_distance_threshold = genetic_distance_threshold
        self.isolation_threshold = isolation_threshold
        self.min_population_for_species = min_population_for_species
        self.speciation_cooldown = speciation_cooldown
        self.enable_hybridization = enable_hybridization

        # Species tracking
        self.species_records: Dict[str, SpeciesRecord] = {}
        self.active_species: Set[str] = set()

        # Isolation barriers
        self.barriers: Dict[Tuple[str, str], IsolationBarrier] = {}

        # Hybrid zones
        self.hybrid_zones: Dict[Tuple[str, str], HybridZone] = {}

        # Speciation history
        self.speciation_events: List[Dict[str, Any]] = []
        self.last_speciation_generation: int = -100

        # Statistics
        self.total_species_created = 0
        self.total_extinctions = 0

    def register_species(
        self,
        species: Species,
        parent_id: Optional[str] = None,
        speciation_type: Optional[SpeciationType] = None,
        niche: str = "",
        region: str = ""
    ) -> SpeciesRecord:
        """Register a new species"""
        record = SpeciesRecord(
            species=species,
            parent_species_id=parent_id,
            speciation_type=speciation_type,
            niche=niche,
            geographic_region=region
        )

        self.species_records[species.id] = record
        self.active_species.add(species.id)
        self.total_species_created += 1

        # Emit event
        bus = get_event_bus()
        bus.emit("speciation.new_species", {
            "species_id": species.id,
            "species_name": species.name,
            "parent_id": parent_id,
            "type": speciation_type.name if speciation_type else None
        })

        return record

    def check_speciation(
        self,
        population: List[Genome],
        generation: int
    ) -> List[Dict[str, Any]]:
        """
        Check for potential speciation events in population.

        Returns:
            List of speciation event records
        """
        # Check cooldown
        if generation - self.last_speciation_generation < self.speciation_cooldown:
            return []

        events = []

        # Group by current species
        species_groups: Dict[str, List[Genome]] = defaultdict(list)
        for genome in population:
            species_id = genome.species_id or "unassigned"
            species_groups[species_id].append(genome)

        # Check each species for potential splits
        for species_id, members in species_groups.items():
            if len(members) < self.min_population_for_species * 2:
                continue

            # Find genetic clusters
            clusters = self._find_genetic_clusters(members)

            if len(clusters) >= 2:
                # Check if clusters are sufficiently divergent
                for i, cluster1 in enumerate(clusters):
                    for cluster2 in clusters[i + 1:]:
                        distance = self._cluster_distance(cluster1, cluster2)

                        if distance > self.genetic_distance_threshold:
                            # Potential speciation event!
                            event = self._create_speciation_event(
                                parent_species_id=species_id,
                                cluster=cluster2,  # Smaller cluster becomes new species
                                generation=generation,
                                speciation_type=self._determine_speciation_type(
                                    cluster1, cluster2
                                )
                            )
                            if event:
                                events.append(event)
                                self.last_speciation_generation = generation

        return events

    def _find_genetic_clusters(
        self,
        genomes: List[Genome],
        max_clusters: int = 3
    ) -> List[List[Genome]]:
        """
        Find genetic clusters in population using simple clustering.
        """
        if len(genomes) < 2:
            return [genomes]

        # Simple distance-based clustering
        clusters: List[List[Genome]] = []
        assigned = set()

        for genome in genomes:
            if genome.id in assigned:
                continue

            # Start new cluster
            cluster = [genome]
            assigned.add(genome.id)

            # Find nearby genomes
            for other in genomes:
                if other.id in assigned:
                    continue

                distance = genome.distance_to(other)
                if distance < self.genetic_distance_threshold * 0.5:
                    cluster.append(other)
                    assigned.add(other.id)

            if len(cluster) >= self.min_population_for_species:
                clusters.append(cluster)

            if len(clusters) >= max_clusters:
                break

        # Assign remaining to nearest cluster
        for genome in genomes:
            if genome.id not in assigned:
                best_cluster = None
                best_distance = float('inf')
                for cluster in clusters:
                    dist = genome.distance_to(cluster[0])
                    if dist < best_distance:
                        best_distance = dist
                        best_cluster = cluster
                if best_cluster:
                    best_cluster.append(genome)

        return clusters

    def _cluster_distance(
        self,
        cluster1: List[Genome],
        cluster2: List[Genome]
    ) -> float:
        """Calculate average distance between clusters"""
        distances = []
        sample_size = min(10, len(cluster1), len(cluster2))

        sample1 = random.sample(cluster1, sample_size)
        sample2 = random.sample(cluster2, sample_size)

        for g1 in sample1:
            for g2 in sample2:
                distances.append(g1.distance_to(g2))

        return sum(distances) / len(distances) if distances else 0.0

    def _determine_speciation_type(
        self,
        cluster1: List[Genome],
        cluster2: List[Genome]
    ) -> SpeciationType:
        """Determine likely speciation type based on clusters"""
        # Simple heuristic - could be enhanced with more data
        size_ratio = len(cluster2) / len(cluster1)

        if size_ratio < 0.2:
            return SpeciationType.PERIPATRIC
        elif size_ratio < 0.5:
            return SpeciationType.PARAPATRIC
        else:
            return SpeciationType.SYMPATRIC

    def _create_speciation_event(
        self,
        parent_species_id: str,
        cluster: List[Genome],
        generation: int,
        speciation_type: SpeciationType
    ) -> Optional[Dict[str, Any]]:
        """Create a new species from a cluster"""
        if len(cluster) < self.min_population_for_species:
            return None

        # Create new species
        new_species = Species(
            name=f"Species_{self.total_species_created + 1}",
            representative=copy.deepcopy(cluster[0]),
            created_generation=generation
        )

        # Assign genomes to new species
        for genome in cluster:
            genome.species_id = new_species.id
            new_species.add_member(genome.id)

        # Register species
        record = self.register_species(
            new_species,
            parent_id=parent_species_id,
            speciation_type=speciation_type
        )

        # Find distinctive traits
        record.distinctive_traits = self._find_distinctive_traits(cluster)

        # Create isolation barrier with parent species
        if parent_species_id and parent_species_id != "unassigned":
            barrier = IsolationBarrier(
                isolation_type=IsolationType.PREZYGOTIC_BEHAVIORAL,
                strength=self.isolation_threshold * 0.5,  # Partial isolation initially
                species1_id=parent_species_id,
                species2_id=new_species.id,
                generation_formed=generation,
                cause="speciation"
            )
            key = (min(parent_species_id, new_species.id),
                   max(parent_species_id, new_species.id))
            self.barriers[key] = barrier

        event = {
            "generation": generation,
            "type": speciation_type.name,
            "new_species_id": new_species.id,
            "new_species_name": new_species.name,
            "parent_species_id": parent_species_id,
            "initial_population": len(cluster),
            "distinctive_traits": record.distinctive_traits
        }

        self.speciation_events.append(event)
        return event

    def _find_distinctive_traits(
        self,
        genomes: List[Genome],
        threshold: float = 0.2
    ) -> List[str]:
        """Find genes that are distinctive in this group"""
        if not genomes:
            return []

        # Calculate average expression for each gene
        gene_avgs: Dict[str, float] = defaultdict(float)
        gene_counts: Dict[str, int] = defaultdict(int)

        for genome in genomes:
            for gene_name, gene in genome.genes.items():
                gene_avgs[gene_name] += gene.express()
                gene_counts[gene_name] += 1

        distinctive = []
        for gene_name, total in gene_avgs.items():
            avg = total / gene_counts[gene_name]
            # High or low expression is distinctive
            if avg > 0.8 or avg < 0.2:
                distinctive.append(gene_name)

        return distinctive[:5]  # Top 5 distinctive traits

    def check_hybridization(
        self,
        genome1: Genome,
        genome2: Genome
    ) -> Tuple[bool, Optional[HybridZone]]:
        """
        Check if two genomes from different species can hybridize.

        Returns:
            Tuple of (can_hybridize, hybrid_zone)
        """
        if not self.enable_hybridization:
            return False, None

        sp1 = genome1.species_id
        sp2 = genome2.species_id

        if not sp1 or not sp2 or sp1 == sp2:
            return True, None  # Same species, normal reproduction

        # Check for barrier
        key = (min(sp1, sp2), max(sp1, sp2))
        barrier = self.barriers.get(key)

        if barrier and random.random() > barrier.get_reproduction_probability():
            return False, None

        # Check/create hybrid zone
        zone = self.hybrid_zones.get(key)
        if not zone:
            # Create new hybrid zone
            zone = HybridZone(
                species1_id=sp1,
                species2_id=sp2,
                hybrid_viability=0.5,
                hybrid_fertility=0.3
            )
            self.hybrid_zones[key] = zone

        if not zone.can_produce_viable_hybrid():
            return False, zone

        return True, zone

    def mark_hybrid(
        self,
        offspring: Genome,
        zone: HybridZone
    ) -> None:
        """Mark a genome as a hybrid and track it"""
        zone.hybrid_genome_ids.append(offspring.id)
        # Hybrids get no species initially
        offspring.species_id = None

    def check_extinction(
        self,
        population: List[Genome],
        generation: int
    ) -> List[str]:
        """
        Check for extinct species.

        Returns:
            List of extinct species IDs
        """
        extinct = []

        # Count members per species
        species_counts: Dict[str, int] = defaultdict(int)
        for genome in population:
            if genome.species_id:
                species_counts[genome.species_id] += 1

        # Check for extinctions
        for species_id in list(self.active_species):
            if species_counts[species_id] == 0:
                if species_id in self.species_records:
                    record = self.species_records[species_id]
                    record.extinction_generation = generation
                    self.active_species.remove(species_id)
                    self.total_extinctions += 1
                    extinct.append(species_id)

                    # Emit event
                    bus = get_event_bus()
                    bus.emit("speciation.extinction", {
                        "species_id": species_id,
                        "generation": generation
                    })
            else:
                # Update population stats
                if species_id in self.species_records:
                    record = self.species_records[species_id]
                    record.peak_population = max(
                        record.peak_population,
                        species_counts[species_id]
                    )
                    record.total_generations += 1

        return extinct

    def strengthen_isolation(
        self,
        species1_id: str,
        species2_id: str,
        amount: float = 0.1
    ) -> None:
        """Strengthen isolation barrier between species"""
        key = (min(species1_id, species2_id), max(species1_id, species2_id))

        if key not in self.barriers:
            self.barriers[key] = IsolationBarrier(
                species1_id=species1_id,
                species2_id=species2_id,
                strength=amount
            )
        else:
            self.barriers[key].strength = min(
                1.0,
                self.barriers[key].strength + amount
            )

    def calculate_species_diversity(self) -> Dict[str, Any]:
        """Calculate species diversity metrics"""
        active_count = len(self.active_species)
        total_count = len(self.species_records)

        # Calculate phylogenetic diversity
        lineage_depths = []
        for species_id in self.active_species:
            depth = self._get_lineage_depth(species_id)
            lineage_depths.append(depth)

        avg_depth = sum(lineage_depths) / len(lineage_depths) if lineage_depths else 0

        # Simpson's diversity index (simplified)
        # Would need population counts for full calculation
        diversity_index = 1.0 - (1.0 / active_count) if active_count > 0 else 0

        return {
            "active_species": active_count,
            "total_species_ever": total_count,
            "extinctions": self.total_extinctions,
            "extinction_rate": self.total_extinctions / total_count if total_count > 0 else 0,
            "avg_lineage_depth": avg_depth,
            "diversity_index": diversity_index
        }

    def _get_lineage_depth(self, species_id: str) -> int:
        """Get depth of lineage (generations since common ancestor)"""
        depth = 0
        current = species_id

        while current and current in self.species_records:
            record = self.species_records[current]
            current = record.parent_species_id
            depth += 1
            if depth > 100:  # Prevent infinite loops
                break

        return depth

    def build_phylogeny(self) -> Dict[str, Any]:
        """
        Build phylogenetic tree structure.

        Returns:
            Nested dict representing the tree
        """
        # Find root species (no parents)
        roots = []
        for species_id, record in self.species_records.items():
            if not record.parent_species_id:
                roots.append(species_id)

        def build_subtree(species_id: str) -> Dict[str, Any]:
            record = self.species_records.get(species_id)
            if not record:
                return {}

            children = []
            for sid, rec in self.species_records.items():
                if rec.parent_species_id == species_id:
                    children.append(build_subtree(sid))

            return {
                "id": species_id,
                "name": record.species.name,
                "created": record.species.created_generation,
                "extinct": record.is_extinct,
                "children": children
            }

        tree = {
            "roots": [build_subtree(r) for r in roots]
        }

        return tree

    def process_generation(
        self,
        population: List[Genome],
        generation: int
    ) -> Dict[str, Any]:
        """
        Process speciation-related events for a generation.
        """
        # Check for speciation
        speciation_events = self.check_speciation(population, generation)

        # Check for extinctions
        extinctions = self.check_extinction(population, generation)

        # Gradually strengthen barriers (reinforcement)
        for key, barrier in self.barriers.items():
            if barrier.strength < 0.9:
                barrier.strength = min(1.0, barrier.strength + 0.01)

        return {
            "generation": generation,
            "speciation_events": len(speciation_events),
            "extinctions": len(extinctions),
            "active_species": len(self.active_species),
            "diversity": self.calculate_species_diversity()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get speciation statistics"""
        return {
            "total_species_created": self.total_species_created,
            "active_species": len(self.active_species),
            "total_extinctions": self.total_extinctions,
            "isolation_barriers": len(self.barriers),
            "hybrid_zones": len(self.hybrid_zones),
            "speciation_events": len(self.speciation_events),
            "diversity": self.calculate_species_diversity()
        }
