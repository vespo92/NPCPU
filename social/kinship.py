"""
Kinship System for NPCPU

Implements family and genetic relationship tracking with:
- Pedigree/family tree structures
- Kinship coefficient calculations (r)
- Kin recognition and kin selection
- Incest avoidance mechanisms
- Multi-generational tracking

Example:
    from social.kinship import KinshipTracker, KinshipRelation

    # Create tracker
    tracker = KinshipTracker()

    # Register birth
    tracker.register_birth(
        child_id="offspring_1",
        parent1_id="parent_a",
        parent2_id="parent_b"
    )

    # Calculate relatedness
    r = tracker.calculate_relatedness("sibling_1", "sibling_2")
    print(f"Relatedness coefficient: {r}")

    # Check kin
    kin = tracker.get_kin("organism_1", max_degree=2)
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict, deque
import math
import uuid

from core.events import get_event_bus


class KinshipRelation(Enum):
    """Types of kinship relationships"""
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    HALF_SIBLING = "half_sibling"
    GRANDPARENT = "grandparent"
    GRANDCHILD = "grandchild"
    AUNT_UNCLE = "aunt_uncle"
    NIECE_NEPHEW = "niece_nephew"
    COUSIN = "cousin"
    MATE = "mate"


@dataclass
class KinRecord:
    """
    Record of an organism's kinship information.

    Attributes:
        organism_id: Organism identifier
        parent1_id: First parent (if known)
        parent2_id: Second parent (if known)
        birth_time: When organism was born
        generation: Generation number
        children: Set of child IDs
        mates: Set of mate IDs
    """
    organism_id: str
    parent1_id: Optional[str] = None
    parent2_id: Optional[str] = None
    birth_time: datetime = field(default_factory=datetime.now)
    generation: int = 0
    children: Set[str] = field(default_factory=set)
    mates: Set[str] = field(default_factory=set)
    genome_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def parents(self) -> List[str]:
        """Get list of known parents"""
        return [p for p in [self.parent1_id, self.parent2_id] if p]

    @property
    def has_known_parents(self) -> bool:
        """Check if any parents are known"""
        return self.parent1_id is not None or self.parent2_id is not None


@dataclass
class KinLink:
    """
    A link between two related organisms.

    Attributes:
        organism1_id: First organism
        organism2_id: Second organism
        relation: Type of relation (from org1's perspective)
        relatedness: Coefficient of relatedness (r)
        path_length: Generational distance
    """
    organism1_id: str
    organism2_id: str
    relation: KinshipRelation
    relatedness: float
    path_length: int = 1


class KinshipTracker:
    """
    Tracks kinship and genetic relationships.

    Provides:
    - Family tree construction
    - Relatedness calculations
    - Kin recognition
    - Incest avoidance checking
    - Multi-generational queries

    Example:
        tracker = KinshipTracker()

        # Register founding organisms
        tracker.register_organism("founder1")
        tracker.register_organism("founder2")

        # Register offspring
        tracker.register_birth("child1", "founder1", "founder2")

        # Check relatedness
        r = tracker.calculate_relatedness("child1", "founder1")  # 0.5
    """

    def __init__(
        self,
        incest_threshold: float = 0.25,
        max_tracked_generations: int = 10,
        emit_events: bool = True
    ):
        """
        Initialize the kinship tracker.

        Args:
            incest_threshold: Relatedness threshold for incest warning
            max_tracked_generations: Maximum generations to track
            emit_events: Whether to emit events
        """
        self._incest_threshold = incest_threshold
        self._max_generations = max_tracked_generations
        self._emit_events = emit_events

        # Organism kinship records
        self._records: Dict[str, KinRecord] = {}

        # Cached relatedness coefficients
        self._relatedness_cache: Dict[Tuple[str, str], float] = {}

        # Generation index
        self._generation_index: Dict[int, Set[str]] = defaultdict(set)

        # Statistics
        self._current_generation = 0
        self._total_births = 0

    # =========================================================================
    # Registration
    # =========================================================================

    def register_organism(
        self,
        organism_id: str,
        parent1_id: Optional[str] = None,
        parent2_id: Optional[str] = None,
        generation: Optional[int] = None,
        genome_id: Optional[str] = None
    ) -> KinRecord:
        """
        Register an organism in the kinship system.

        Args:
            organism_id: Organism identifier
            parent1_id: First parent ID
            parent2_id: Second parent ID
            generation: Generation number (auto-calculated if not provided)
            genome_id: Associated genome ID

        Returns:
            KinRecord for the organism
        """
        if organism_id in self._records:
            return self._records[organism_id]

        # Calculate generation
        if generation is None:
            if parent1_id and parent1_id in self._records:
                gen1 = self._records[parent1_id].generation
            else:
                gen1 = 0

            if parent2_id and parent2_id in self._records:
                gen2 = self._records[parent2_id].generation
            else:
                gen2 = 0

            generation = max(gen1, gen2) + 1 if (parent1_id or parent2_id) else 0

        record = KinRecord(
            organism_id=organism_id,
            parent1_id=parent1_id,
            parent2_id=parent2_id,
            generation=generation,
            genome_id=genome_id
        )

        self._records[organism_id] = record
        self._generation_index[generation].add(organism_id)
        self._current_generation = max(self._current_generation, generation)

        # Update parent records
        if parent1_id and parent1_id in self._records:
            self._records[parent1_id].children.add(organism_id)
        if parent2_id and parent2_id in self._records:
            self._records[parent2_id].children.add(organism_id)

        # Invalidate cache
        self._relatedness_cache.clear()

        return record

    def register_birth(
        self,
        child_id: str,
        parent1_id: str,
        parent2_id: Optional[str] = None,
        genome_id: Optional[str] = None
    ) -> KinRecord:
        """
        Register a birth event.

        Args:
            child_id: New organism ID
            parent1_id: First parent
            parent2_id: Second parent (optional for asexual)
            genome_id: Child's genome ID

        Returns:
            KinRecord for the child
        """
        # Ensure parents exist
        if parent1_id not in self._records:
            self.register_organism(parent1_id)
        if parent2_id and parent2_id not in self._records:
            self.register_organism(parent2_id)

        record = self.register_organism(
            child_id,
            parent1_id=parent1_id,
            parent2_id=parent2_id,
            genome_id=genome_id
        )

        self._total_births += 1

        # Record mate relationship
        if parent2_id:
            self._records[parent1_id].mates.add(parent2_id)
            self._records[parent2_id].mates.add(parent1_id)

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("kinship.birth", {
                "child_id": child_id,
                "parent1_id": parent1_id,
                "parent2_id": parent2_id,
                "generation": record.generation
            })

        return record

    def register_mating(self, organism1_id: str, organism2_id: str) -> bool:
        """
        Register a mating event between two organisms.

        Returns:
            True if registered (not incestuous)
        """
        # Ensure both exist
        if organism1_id not in self._records:
            self.register_organism(organism1_id)
        if organism2_id not in self._records:
            self.register_organism(organism2_id)

        # Check for incest
        r = self.calculate_relatedness(organism1_id, organism2_id)
        if r >= self._incest_threshold:
            if self._emit_events:
                bus = get_event_bus()
                bus.emit("kinship.incest_warning", {
                    "organism1": organism1_id,
                    "organism2": organism2_id,
                    "relatedness": r
                })
            return False

        self._records[organism1_id].mates.add(organism2_id)
        self._records[organism2_id].mates.add(organism1_id)

        return True

    # =========================================================================
    # Relatedness Calculation
    # =========================================================================

    def calculate_relatedness(self, organism1_id: str, organism2_id: str) -> float:
        """
        Calculate coefficient of relatedness (r) between two organisms.

        Uses path-counting method through common ancestors.

        Returns:
            Relatedness coefficient (0.0 to 1.0)
            - Parent-child: 0.5
            - Full siblings: 0.5
            - Half siblings: 0.25
            - Grandparent-grandchild: 0.25
            - First cousins: 0.125
        """
        if organism1_id == organism2_id:
            return 1.0

        # Check cache
        cache_key = tuple(sorted([organism1_id, organism2_id]))
        if cache_key in self._relatedness_cache:
            return self._relatedness_cache[cache_key]

        if organism1_id not in self._records or organism2_id not in self._records:
            return 0.0

        # Find common ancestors and calculate r
        r = self._calculate_r_via_ancestors(organism1_id, organism2_id)

        self._relatedness_cache[cache_key] = r
        return r

    def _calculate_r_via_ancestors(self, org1_id: str, org2_id: str) -> float:
        """Calculate relatedness by finding common ancestors"""
        # Get ancestors with path lengths for both organisms
        ancestors1 = self._get_ancestors_with_depth(org1_id)
        ancestors2 = self._get_ancestors_with_depth(org2_id)

        total_r = 0.0

        # Check if one is a direct ancestor of the other
        # If org2 is an ancestor of org1, r = 0.5^depth
        if org2_id in ancestors1:
            for depth in ancestors1[org2_id]:
                total_r += math.pow(0.5, depth)

        # If org1 is an ancestor of org2, r = 0.5^depth
        if org1_id in ancestors2:
            for depth in ancestors2[org1_id]:
                total_r += math.pow(0.5, depth)

        # Find common ancestors (for siblings, cousins, etc.)
        common = set(ancestors1.keys()) & set(ancestors2.keys())

        for ancestor in common:
            depths1 = ancestors1[ancestor]  # List of depths from org1 to ancestor
            depths2 = ancestors2[ancestor]  # List of depths from org2 to ancestor

            for d1 in depths1:
                for d2 in depths2:
                    # Each path contributes (0.5)^path_length
                    total_r += math.pow(0.5, d1 + d2)

        return min(1.0, total_r)

    def _get_ancestors_with_depth(
        self,
        organism_id: str,
        max_depth: Optional[int] = None
    ) -> Dict[str, List[int]]:
        """
        Get all ancestors with their path depths.

        Returns:
            Dict mapping ancestor_id -> [list of depths]
        """
        if max_depth is None:
            max_depth = self._max_generations

        ancestors: Dict[str, List[int]] = defaultdict(list)
        queue = deque([(organism_id, 0)])
        visited = set()

        while queue:
            current, depth = queue.popleft()

            if depth > max_depth:
                continue

            if (current, depth) in visited:
                continue
            visited.add((current, depth))

            if current != organism_id:
                ancestors[current].append(depth)

            record = self._records.get(current)
            if record:
                for parent_id in record.parents:
                    queue.append((parent_id, depth + 1))

        return dict(ancestors)

    # =========================================================================
    # Kin Queries
    # =========================================================================

    def get_kin(
        self,
        organism_id: str,
        max_degree: int = 3,
        min_relatedness: float = 0.0
    ) -> List[KinLink]:
        """
        Get all kin of an organism.

        Args:
            organism_id: Target organism
            max_degree: Maximum generational distance
            min_relatedness: Minimum relatedness to include

        Returns:
            List of KinLink objects
        """
        if organism_id not in self._records:
            return []

        kin = []
        record = self._records[organism_id]

        # Parents
        for parent_id in record.parents:
            kin.append(KinLink(
                organism1_id=organism_id,
                organism2_id=parent_id,
                relation=KinshipRelation.PARENT,
                relatedness=0.5,
                path_length=1
            ))

        # Children
        for child_id in record.children:
            kin.append(KinLink(
                organism1_id=organism_id,
                organism2_id=child_id,
                relation=KinshipRelation.CHILD,
                relatedness=0.5,
                path_length=1
            ))

        # Siblings
        for parent_id in record.parents:
            parent = self._records.get(parent_id)
            if parent:
                for sibling_id in parent.children:
                    if sibling_id != organism_id:
                        # Check if full or half sibling
                        sib_record = self._records.get(sibling_id)
                        if sib_record:
                            shared_parents = set(record.parents) & set(sib_record.parents)
                            if len(shared_parents) == 2:
                                rel = KinshipRelation.SIBLING
                                r = 0.5
                            else:
                                rel = KinshipRelation.HALF_SIBLING
                                r = 0.25

                            kin.append(KinLink(
                                organism1_id=organism_id,
                                organism2_id=sibling_id,
                                relation=rel,
                                relatedness=r,
                                path_length=2
                            ))

        # Extended kin if requested
        if max_degree >= 2:
            kin.extend(self._get_extended_kin(organism_id, max_degree))

        # Filter by relatedness and deduplicate
        seen = set()
        filtered = []
        for link in kin:
            if link.organism2_id not in seen and link.relatedness >= min_relatedness:
                seen.add(link.organism2_id)
                filtered.append(link)

        return filtered

    def _get_extended_kin(self, organism_id: str, max_degree: int) -> List[KinLink]:
        """Get extended family members"""
        extended = []
        record = self._records[organism_id]

        # Grandparents
        for parent_id in record.parents:
            parent = self._records.get(parent_id)
            if parent:
                for gp_id in parent.parents:
                    extended.append(KinLink(
                        organism1_id=organism_id,
                        organism2_id=gp_id,
                        relation=KinshipRelation.GRANDPARENT,
                        relatedness=0.25,
                        path_length=2
                    ))

        # Grandchildren
        for child_id in record.children:
            child = self._records.get(child_id)
            if child:
                for gc_id in child.children:
                    extended.append(KinLink(
                        organism1_id=organism_id,
                        organism2_id=gc_id,
                        relation=KinshipRelation.GRANDCHILD,
                        relatedness=0.25,
                        path_length=2
                    ))

        # Aunts/Uncles (parents' siblings)
        for parent_id in record.parents:
            parent = self._records.get(parent_id)
            if parent:
                for gp_id in parent.parents:
                    gp = self._records.get(gp_id)
                    if gp:
                        for au_id in gp.children:
                            if au_id != parent_id:
                                extended.append(KinLink(
                                    organism1_id=organism_id,
                                    organism2_id=au_id,
                                    relation=KinshipRelation.AUNT_UNCLE,
                                    relatedness=0.25,
                                    path_length=3
                                ))

        # Cousins
        if max_degree >= 3:
            extended.extend(self._get_cousins(organism_id))

        return extended

    def _get_cousins(self, organism_id: str) -> List[KinLink]:
        """Get cousins of an organism"""
        cousins = []
        record = self._records[organism_id]

        for parent_id in record.parents:
            parent = self._records.get(parent_id)
            if not parent:
                continue

            for gp_id in parent.parents:
                gp = self._records.get(gp_id)
                if not gp:
                    continue

                for au_id in gp.children:
                    if au_id == parent_id:
                        continue

                    au = self._records.get(au_id)
                    if au:
                        for cousin_id in au.children:
                            cousins.append(KinLink(
                                organism1_id=organism_id,
                                organism2_id=cousin_id,
                                relation=KinshipRelation.COUSIN,
                                relatedness=0.125,
                                path_length=4
                            ))

        return cousins

    def get_relation(
        self,
        organism1_id: str,
        organism2_id: str
    ) -> Optional[KinshipRelation]:
        """
        Get the kinship relation between two organisms.

        Returns:
            KinshipRelation if related, None otherwise
        """
        if organism1_id not in self._records or organism2_id not in self._records:
            return None

        rec1 = self._records[organism1_id]
        rec2 = self._records[organism2_id]

        # Direct relations
        if organism2_id in rec1.parents:
            return KinshipRelation.PARENT
        if organism2_id in rec1.children:
            return KinshipRelation.CHILD
        if organism2_id in rec1.mates:
            return KinshipRelation.MATE

        # Siblings
        shared_parents = set(rec1.parents) & set(rec2.parents)
        if len(shared_parents) == 2:
            return KinshipRelation.SIBLING
        elif len(shared_parents) == 1:
            return KinshipRelation.HALF_SIBLING

        # Check further relations by relatedness
        r = self.calculate_relatedness(organism1_id, organism2_id)
        if r >= 0.25 - 0.01:
            # Could be grandparent/grandchild or aunt/uncle/niece/nephew
            if rec1.generation < rec2.generation - 1:
                return KinshipRelation.GRANDPARENT
            elif rec1.generation > rec2.generation + 1:
                return KinshipRelation.GRANDCHILD
            else:
                return KinshipRelation.AUNT_UNCLE
        elif r >= 0.125 - 0.01:
            return KinshipRelation.COUSIN

        return None

    # =========================================================================
    # Incest Avoidance
    # =========================================================================

    def is_incestuous(self, organism1_id: str, organism2_id: str) -> bool:
        """
        Check if mating would be considered incestuous.

        Returns:
            True if relatedness exceeds threshold
        """
        r = self.calculate_relatedness(organism1_id, organism2_id)
        return r >= self._incest_threshold

    def get_suitable_mates(
        self,
        organism_id: str,
        candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Filter candidates for suitable (non-incestuous) mates.

        Returns:
            List of (candidate_id, relatedness) for suitable mates
        """
        suitable = []
        for candidate_id in candidates:
            if candidate_id == organism_id:
                continue
            r = self.calculate_relatedness(organism_id, candidate_id)
            if r < self._incest_threshold:
                suitable.append((candidate_id, r))

        # Sort by relatedness (prefer less related)
        suitable.sort(key=lambda x: x[1])
        return suitable

    # =========================================================================
    # Lineage Queries
    # =========================================================================

    def get_ancestors(
        self,
        organism_id: str,
        max_generations: int = 5
    ) -> Set[str]:
        """Get all ancestors up to max generations"""
        ancestors = set()
        current = {organism_id}

        for _ in range(max_generations):
            next_gen = set()
            for org_id in current:
                record = self._records.get(org_id)
                if record:
                    for parent_id in record.parents:
                        if parent_id not in ancestors:
                            next_gen.add(parent_id)
                            ancestors.add(parent_id)
            current = next_gen
            if not current:
                break

        return ancestors

    def get_descendants(
        self,
        organism_id: str,
        max_generations: int = 5
    ) -> Set[str]:
        """Get all descendants up to max generations"""
        descendants = set()
        current = {organism_id}

        for _ in range(max_generations):
            next_gen = set()
            for org_id in current:
                record = self._records.get(org_id)
                if record:
                    for child_id in record.children:
                        if child_id not in descendants:
                            next_gen.add(child_id)
                            descendants.add(child_id)
            current = next_gen
            if not current:
                break

        return descendants

    def get_common_ancestors(
        self,
        organism1_id: str,
        organism2_id: str
    ) -> Set[str]:
        """Get common ancestors of two organisms"""
        ancestors1 = self.get_ancestors(organism1_id, self._max_generations)
        ancestors2 = self.get_ancestors(organism2_id, self._max_generations)
        return ancestors1 & ancestors2

    def get_lineage_depth(self, organism_id: str) -> int:
        """Get the depth of an organism's lineage (generations to founders)"""
        record = self._records.get(organism_id)
        if not record or not record.has_known_parents:
            return 0

        max_depth = 0
        for parent_id in record.parents:
            depth = self.get_lineage_depth(parent_id)
            max_depth = max(max_depth, depth + 1)

        return max_depth

    def get_generation_members(self, generation: int) -> Set[str]:
        """Get all organisms in a specific generation"""
        return self._generation_index.get(generation, set()).copy()

    def remove_organism(self, organism_id: str) -> None:
        """Remove an organism from tracking"""
        if organism_id not in self._records:
            return

        record = self._records[organism_id]

        # Remove from parents' children lists
        for parent_id in record.parents:
            if parent_id in self._records:
                self._records[parent_id].children.discard(organism_id)

        # Remove from mates
        for mate_id in record.mates:
            if mate_id in self._records:
                self._records[mate_id].mates.discard(organism_id)

        # Remove from generation index
        self._generation_index[record.generation].discard(organism_id)

        del self._records[organism_id]
        self._relatedness_cache.clear()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get kinship statistics"""
        if not self._records:
            return {
                "total_organisms": 0,
                "current_generation": 0,
                "total_births": 0,
                "avg_offspring": 0
            }

        total = len(self._records)
        offspring_counts = [len(r.children) for r in self._records.values()]
        with_parents = sum(1 for r in self._records.values() if r.has_known_parents)

        return {
            "total_organisms": total,
            "current_generation": self._current_generation,
            "total_births": self._total_births,
            "organisms_with_known_parents": with_parents,
            "avg_offspring": sum(offspring_counts) / total if total > 0 else 0,
            "max_offspring": max(offspring_counts) if offspring_counts else 0,
            "generation_sizes": {
                gen: len(members)
                for gen, members in self._generation_index.items()
            }
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize kinship tracker"""
        records = [
            {
                "organism_id": r.organism_id,
                "parent1_id": r.parent1_id,
                "parent2_id": r.parent2_id,
                "generation": r.generation,
                "children": list(r.children),
                "mates": list(r.mates),
                "genome_id": r.genome_id
            }
            for r in self._records.values()
        ]

        return {
            "records": records,
            "incest_threshold": self._incest_threshold,
            "current_generation": self._current_generation,
            "total_births": self._total_births
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'KinshipTracker':
        """Deserialize kinship tracker"""
        tracker = cls(
            incest_threshold=data.get("incest_threshold", 0.25),
            emit_events=emit_events
        )

        tracker._current_generation = data.get("current_generation", 0)
        tracker._total_births = data.get("total_births", 0)

        for r_data in data.get("records", []):
            record = KinRecord(
                organism_id=r_data["organism_id"],
                parent1_id=r_data.get("parent1_id"),
                parent2_id=r_data.get("parent2_id"),
                generation=r_data.get("generation", 0),
                children=set(r_data.get("children", [])),
                mates=set(r_data.get("mates", [])),
                genome_id=r_data.get("genome_id")
            )
            tracker._records[record.organism_id] = record
            tracker._generation_index[record.generation].add(record.organism_id)

        return tracker
