"""
Cultural Transmission System for NPCPU

Implements meme and knowledge spread with:
- Horizontal transmission (peer-to-peer)
- Vertical transmission (parent-to-child)
- Oblique transmission (elders-to-young)
- Innovation and mutation of memes
- Cultural fitness and selection

Example:
    from social.cultural_transmission import CulturalEngine, Meme, TransmissionMode

    # Create engine
    engine = CulturalEngine()

    # Create a meme (cultural unit)
    meme = engine.create_meme(
        creator_id="inventor",
        content="tool_use",
        category="technology"
    )

    # Transmit to others
    engine.transmit(meme.id, "inventor", "learner", TransmissionMode.HORIZONTAL)

    # Check spread
    carriers = engine.get_meme_carriers(meme.id)
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


class TransmissionMode(Enum):
    """Modes of cultural transmission"""
    VERTICAL = "vertical"       # Parent to child
    HORIZONTAL = "horizontal"   # Peer to peer
    OBLIQUE = "oblique"         # Older to younger non-parent
    BROADCAST = "broadcast"     # One to many (teaching)
    OBSERVATION = "observation" # Learning by watching


class MemeCategory(Enum):
    """Categories of cultural information"""
    TECHNOLOGY = "technology"   # Tool use, techniques
    SOCIAL = "social"           # Social norms, customs
    BELIEF = "belief"           # Beliefs, superstitions
    LANGUAGE = "language"       # Communication patterns
    FORAGING = "foraging"       # Food finding strategies
    DEFENSE = "defense"         # Predator avoidance
    TRADITION = "tradition"     # Rituals, ceremonies


@dataclass
class Meme:
    """
    A unit of cultural information (meme).

    Attributes:
        id: Unique identifier
        content: The cultural content (symbolic)
        category: Type of cultural information
        creator_id: Original creator
        fitness: Cultural fitness (how likely to spread)
        complexity: How hard to learn
        variants: Related meme variants
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    category: MemeCategory = MemeCategory.SOCIAL
    creator_id: str = ""
    fitness: float = 0.5
    complexity: float = 0.3
    mutation_rate: float = 0.05
    creation_time: datetime = field(default_factory=datetime.now)
    parent_meme_id: Optional[str] = None
    variants: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mutate(self) -> 'Meme':
        """Create a mutated variant of this meme"""
        variant = Meme(
            content=f"{self.content}_variant",
            category=self.category,
            creator_id=self.creator_id,
            fitness=self.fitness + random.uniform(-0.1, 0.1),
            complexity=self.complexity,
            mutation_rate=self.mutation_rate,
            parent_meme_id=self.id
        )
        self.variants.add(variant.id)
        return variant


@dataclass
class CulturalKnowledge:
    """
    An organism's cultural knowledge.

    Attributes:
        organism_id: Knowledge holder
        memes: Set of known meme IDs
        proficiency: How well each meme is known
        acquired_from: Who each meme was learned from
    """
    organism_id: str
    memes: Set[str] = field(default_factory=set)
    proficiency: Dict[str, float] = field(default_factory=dict)
    acquired_from: Dict[str, str] = field(default_factory=dict)
    acquisition_time: Dict[str, datetime] = field(default_factory=dict)

    def add_meme(
        self,
        meme_id: str,
        proficiency: float = 0.5,
        source_id: str = ""
    ) -> None:
        """Add a meme to knowledge"""
        self.memes.add(meme_id)
        self.proficiency[meme_id] = max(0.0, min(1.0, proficiency))
        self.acquired_from[meme_id] = source_id
        self.acquisition_time[meme_id] = datetime.now()

    def improve_proficiency(self, meme_id: str, amount: float = 0.1) -> None:
        """Improve proficiency in a meme"""
        if meme_id in self.proficiency:
            self.proficiency[meme_id] = min(1.0, self.proficiency[meme_id] + amount)

    def knows_meme(self, meme_id: str) -> bool:
        """Check if organism knows a meme"""
        return meme_id in self.memes


@dataclass
class TransmissionEvent:
    """
    Record of a cultural transmission.

    Attributes:
        meme_id: What was transmitted
        source_id: Who transmitted
        target_id: Who received
        mode: How it was transmitted
        success: Whether transmission succeeded
        fidelity: How accurate the transmission was
    """
    meme_id: str
    source_id: str
    target_id: str
    mode: TransmissionMode
    success: bool = True
    fidelity: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CulturalEngine:
    """
    Manages cultural transmission and evolution.

    Provides:
    - Meme creation and mutation
    - Transmission between organisms
    - Cultural fitness tracking
    - Knowledge diffusion analysis
    - Innovation modeling

    Example:
        engine = CulturalEngine()

        # Create knowledge
        meme = engine.create_meme("inventor", "fire_making", MemeCategory.TECHNOLOGY)

        # Spread knowledge
        engine.transmit(meme.id, "inventor", "student", TransmissionMode.VERTICAL)

        # Check cultural landscape
        popular = engine.get_most_common_memes(10)
    """

    def __init__(
        self,
        base_transmission_rate: float = 0.7,
        base_mutation_rate: float = 0.05,
        complexity_penalty: float = 0.3,
        emit_events: bool = True
    ):
        """
        Initialize the cultural engine.

        Args:
            base_transmission_rate: Base probability of successful transmission
            base_mutation_rate: Base probability of mutation during transmission
            complexity_penalty: How much complexity reduces transmission
            emit_events: Whether to emit events
        """
        self._base_transmission_rate = base_transmission_rate
        self._base_mutation_rate = base_mutation_rate
        self._complexity_penalty = complexity_penalty
        self._emit_events = emit_events

        # Memes by ID
        self._memes: Dict[str, Meme] = {}

        # Organism knowledge
        self._knowledge: Dict[str, CulturalKnowledge] = {}

        # Transmission history
        self._transmissions: List[TransmissionEvent] = []

        # Meme popularity (number of carriers)
        self._popularity: Dict[str, int] = defaultdict(int)

    # =========================================================================
    # Meme Management
    # =========================================================================

    def create_meme(
        self,
        creator_id: str,
        content: str,
        category: MemeCategory = MemeCategory.SOCIAL,
        fitness: float = 0.5,
        complexity: float = 0.3
    ) -> Meme:
        """
        Create a new meme.

        Args:
            creator_id: Who created it
            content: Symbolic content
            category: Meme category
            fitness: Initial fitness
            complexity: Learning complexity

        Returns:
            The created Meme
        """
        meme = Meme(
            content=content,
            category=category,
            creator_id=creator_id,
            fitness=fitness,
            complexity=complexity,
            mutation_rate=self._base_mutation_rate
        )

        self._memes[meme.id] = meme

        # Creator knows the meme
        self._ensure_knowledge(creator_id)
        self._knowledge[creator_id].add_meme(meme.id, proficiency=1.0, source_id=creator_id)
        self._popularity[meme.id] = 1

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("culture.meme_created", {
                "meme_id": meme.id,
                "creator_id": creator_id,
                "content": content,
                "category": category.value
            })

        return meme

    def get_meme(self, meme_id: str) -> Optional[Meme]:
        """Get a meme by ID"""
        return self._memes.get(meme_id)

    def _ensure_knowledge(self, organism_id: str) -> CulturalKnowledge:
        """Ensure organism has a knowledge record"""
        if organism_id not in self._knowledge:
            self._knowledge[organism_id] = CulturalKnowledge(organism_id=organism_id)
        return self._knowledge[organism_id]

    # =========================================================================
    # Transmission
    # =========================================================================

    def transmit(
        self,
        meme_id: str,
        source_id: str,
        target_id: str,
        mode: TransmissionMode = TransmissionMode.HORIZONTAL,
        force: bool = False
    ) -> TransmissionEvent:
        """
        Attempt to transmit a meme from source to target.

        Args:
            meme_id: Meme to transmit
            source_id: Who is transmitting
            target_id: Who is receiving
            mode: Transmission mode
            force: Skip probability check

        Returns:
            TransmissionEvent recording the attempt
        """
        meme = self._memes.get(meme_id)
        if not meme:
            return TransmissionEvent(
                meme_id=meme_id,
                source_id=source_id,
                target_id=target_id,
                mode=mode,
                success=False,
                fidelity=0.0
            )

        source_knowledge = self._knowledge.get(source_id)
        if not source_knowledge or meme_id not in source_knowledge.memes:
            return TransmissionEvent(
                meme_id=meme_id,
                source_id=source_id,
                target_id=target_id,
                mode=mode,
                success=False,
                fidelity=0.0
            )

        self._ensure_knowledge(target_id)
        target_knowledge = self._knowledge[target_id]

        # Calculate transmission probability
        source_proficiency = source_knowledge.proficiency.get(meme_id, 0.5)
        probability = self._calculate_transmission_probability(
            meme, source_proficiency, mode
        )

        success = force or random.random() < probability

        # Calculate fidelity (how accurately it's transmitted)
        fidelity = source_proficiency * (1 - meme.complexity * 0.5)

        if success and meme_id not in target_knowledge.memes:
            # Check for mutation
            if random.random() < meme.mutation_rate:
                # Create mutated variant
                variant = meme.mutate()
                self._memes[variant.id] = variant
                target_knowledge.add_meme(variant.id, fidelity * 0.8, source_id)
                self._popularity[variant.id] = 1

                if self._emit_events:
                    bus = get_event_bus()
                    bus.emit("culture.meme_mutated", {
                        "original_id": meme_id,
                        "variant_id": variant.id,
                        "target_id": target_id
                    })
            else:
                target_knowledge.add_meme(meme_id, fidelity, source_id)
                self._popularity[meme_id] += 1

        event = TransmissionEvent(
            meme_id=meme_id,
            source_id=source_id,
            target_id=target_id,
            mode=mode,
            success=success,
            fidelity=fidelity if success else 0.0
        )

        self._transmissions.append(event)

        if self._emit_events and success:
            bus = get_event_bus()
            bus.emit("culture.transmission", {
                "meme_id": meme_id,
                "source_id": source_id,
                "target_id": target_id,
                "mode": mode.value,
                "fidelity": fidelity
            })

        return event

    def _calculate_transmission_probability(
        self,
        meme: Meme,
        source_proficiency: float,
        mode: TransmissionMode
    ) -> float:
        """Calculate probability of successful transmission"""
        base = self._base_transmission_rate

        # Mode modifiers
        mode_modifier = {
            TransmissionMode.VERTICAL: 1.2,      # Parents teach well
            TransmissionMode.HORIZONTAL: 1.0,
            TransmissionMode.OBLIQUE: 1.1,
            TransmissionMode.BROADCAST: 0.8,     # Less effective
            TransmissionMode.OBSERVATION: 0.6   # Hardest
        }.get(mode, 1.0)

        # Factors
        fitness_factor = 0.5 + 0.5 * meme.fitness
        proficiency_factor = 0.5 + 0.5 * source_proficiency
        complexity_factor = 1 - meme.complexity * self._complexity_penalty

        probability = base * mode_modifier * fitness_factor * proficiency_factor * complexity_factor
        return max(0.0, min(1.0, probability))

    def broadcast(
        self,
        meme_id: str,
        source_id: str,
        targets: List[str]
    ) -> List[TransmissionEvent]:
        """
        Broadcast a meme to multiple targets.

        Returns:
            List of transmission events
        """
        events = []
        for target_id in targets:
            if target_id != source_id:
                event = self.transmit(
                    meme_id, source_id, target_id,
                    TransmissionMode.BROADCAST
                )
                events.append(event)
        return events

    def vertical_transmission(
        self,
        parent_id: str,
        child_id: str,
        selection: str = "all"
    ) -> List[TransmissionEvent]:
        """
        Transmit culture from parent to child.

        Args:
            parent_id: Parent organism
            child_id: Child organism
            selection: "all", "high_fitness", or "random"

        Returns:
            List of transmission events
        """
        parent_knowledge = self._knowledge.get(parent_id)
        if not parent_knowledge:
            return []

        events = []
        memes_to_transmit = list(parent_knowledge.memes)

        if selection == "high_fitness":
            # Prioritize high-fitness memes
            memes_to_transmit.sort(
                key=lambda m: self._memes.get(m, Meme()).fitness,
                reverse=True
            )
        elif selection == "random":
            random.shuffle(memes_to_transmit)

        for meme_id in memes_to_transmit:
            event = self.transmit(
                meme_id, parent_id, child_id,
                TransmissionMode.VERTICAL
            )
            events.append(event)

        return events

    # =========================================================================
    # Innovation
    # =========================================================================

    def innovate(
        self,
        organism_id: str,
        category: MemeCategory,
        base_fitness: float = 0.5
    ) -> Optional[Meme]:
        """
        Organism creates a new innovation.

        The innovation may be based on existing knowledge.
        """
        knowledge = self._knowledge.get(organism_id)

        # Innovation quality based on existing knowledge
        existing_in_category = []
        if knowledge:
            for meme_id in knowledge.memes:
                meme = self._memes.get(meme_id)
                if meme and meme.category == category:
                    existing_in_category.append(meme)

        # More knowledge = better innovations
        knowledge_bonus = len(existing_in_category) * 0.05
        fitness = min(1.0, base_fitness + knowledge_bonus + random.uniform(-0.1, 0.2))

        # Generate content
        content = f"innovation_{category.value}_{len(self._memes)}"

        meme = self.create_meme(
            creator_id=organism_id,
            content=content,
            category=category,
            fitness=fitness,
            complexity=random.uniform(0.2, 0.6)
        )

        if self._emit_events:
            bus = get_event_bus()
            bus.emit("culture.innovation", {
                "meme_id": meme.id,
                "creator_id": organism_id,
                "category": category.value,
                "fitness": fitness
            })

        return meme

    def recombine_memes(
        self,
        organism_id: str,
        meme_ids: List[str]
    ) -> Optional[Meme]:
        """
        Create a new meme by combining existing ones.

        Returns:
            New combined meme if successful
        """
        knowledge = self._knowledge.get(organism_id)
        if not knowledge:
            return None

        # Verify organism knows all memes
        memes = []
        for meme_id in meme_ids:
            if meme_id not in knowledge.memes:
                return None
            meme = self._memes.get(meme_id)
            if meme:
                memes.append(meme)

        if len(memes) < 2:
            return None

        # Create combined meme
        avg_fitness = sum(m.fitness for m in memes) / len(memes)
        combined_complexity = max(m.complexity for m in memes) + 0.1

        content = "_".join(m.content for m in memes[:2])
        category = memes[0].category  # Use first meme's category

        new_meme = self.create_meme(
            creator_id=organism_id,
            content=content,
            category=category,
            fitness=avg_fitness + random.uniform(-0.1, 0.1),
            complexity=min(1.0, combined_complexity)
        )

        return new_meme

    # =========================================================================
    # Queries
    # =========================================================================

    def get_meme_carriers(self, meme_id: str) -> Set[str]:
        """Get all organisms that know a meme"""
        carriers = set()
        for org_id, knowledge in self._knowledge.items():
            if meme_id in knowledge.memes:
                carriers.add(org_id)
        return carriers

    def get_organism_knowledge(self, organism_id: str) -> Optional[CulturalKnowledge]:
        """Get an organism's cultural knowledge"""
        return self._knowledge.get(organism_id)

    def get_most_common_memes(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most widely known memes"""
        sorted_memes = sorted(
            self._popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_memes[:limit]

    def get_memes_by_category(self, category: MemeCategory) -> List[Meme]:
        """Get all memes in a category"""
        return [m for m in self._memes.values() if m.category == category]

    def calculate_cultural_diversity(self) -> float:
        """
        Calculate cultural diversity (Shannon index).

        Returns:
            Diversity score (higher = more diverse)
        """
        if not self._popularity:
            return 0.0

        total = sum(self._popularity.values())
        if total == 0:
            return 0.0

        diversity = 0.0
        for count in self._popularity.values():
            if count > 0:
                p = count / total
                diversity -= p * math.log(p)

        return diversity

    def get_transmission_network(self) -> Dict[str, List[str]]:
        """
        Get the cultural transmission network.

        Returns:
            Mapping of source_id -> [target_ids] for successful transmissions
        """
        network: Dict[str, List[str]] = defaultdict(list)
        for event in self._transmissions:
            if event.success:
                network[event.source_id].append(event.target_id)
        return dict(network)

    def get_cultural_lineage(self, meme_id: str) -> List[str]:
        """
        Trace the lineage of a meme back to its origin.

        Returns:
            List of meme IDs from oldest to newest
        """
        lineage = [meme_id]
        current = self._memes.get(meme_id)

        while current and current.parent_meme_id:
            lineage.insert(0, current.parent_meme_id)
            current = self._memes.get(current.parent_meme_id)

        return lineage

    def remove_organism(self, organism_id: str) -> None:
        """Remove an organism from the cultural system"""
        if organism_id in self._knowledge:
            knowledge = self._knowledge[organism_id]
            for meme_id in knowledge.memes:
                self._popularity[meme_id] = max(0, self._popularity[meme_id] - 1)
            del self._knowledge[organism_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get cultural statistics"""
        return {
            "total_memes": len(self._memes),
            "total_transmissions": len(self._transmissions),
            "successful_transmissions": sum(
                1 for t in self._transmissions if t.success
            ),
            "cultural_diversity": self.calculate_cultural_diversity(),
            "avg_meme_popularity": (
                sum(self._popularity.values()) / len(self._popularity)
                if self._popularity else 0
            ),
            "organisms_with_culture": len(self._knowledge)
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cultural engine"""
        memes = [
            {
                "id": m.id,
                "content": m.content,
                "category": m.category.value,
                "creator_id": m.creator_id,
                "fitness": m.fitness,
                "complexity": m.complexity,
                "parent_meme_id": m.parent_meme_id,
                "variants": list(m.variants)
            }
            for m in self._memes.values()
        ]

        knowledge = [
            {
                "organism_id": k.organism_id,
                "memes": list(k.memes),
                "proficiency": k.proficiency
            }
            for k in self._knowledge.values()
        ]

        return {
            "memes": memes,
            "knowledge": knowledge,
            "popularity": dict(self._popularity)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], emit_events: bool = False) -> 'CulturalEngine':
        """Deserialize cultural engine"""
        engine = cls(emit_events=emit_events)

        for m_data in data.get("memes", []):
            meme = Meme(
                id=m_data["id"],
                content=m_data["content"],
                category=MemeCategory(m_data["category"]),
                creator_id=m_data["creator_id"],
                fitness=m_data.get("fitness", 0.5),
                complexity=m_data.get("complexity", 0.3),
                parent_meme_id=m_data.get("parent_meme_id"),
                variants=set(m_data.get("variants", []))
            )
            engine._memes[meme.id] = meme

        for k_data in data.get("knowledge", []):
            knowledge = CulturalKnowledge(
                organism_id=k_data["organism_id"],
                memes=set(k_data.get("memes", [])),
                proficiency=k_data.get("proficiency", {})
            )
            engine._knowledge[knowledge.organism_id] = knowledge

        engine._popularity = defaultdict(int, data.get("popularity", {}))

        return engine
