"""
CollectiveWisdomAgent - Aggregates collective knowledge across all legs.

Responsibilities:
- Collect and synthesize wisdom from all three domains
- Build shared knowledge representations using a knowledge graph
- Enable knowledge transfer between domains
- Maintain long-term memory of successful patterns
- Facilitate distributed learning through consensus building
- Mine cross-domain meta-patterns
- Proactively distribute wisdom based on domain needs
- Learn from wisdom application outcomes

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                     CollectiveWisdomAgent                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Knowledge Graph                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │   [NPCPU Wisdom] ←──→ [Shared Patterns] ←──→ [ChicagoForest]   │   │
│  │         ↕                    ↕                    ↕              │   │
│  │   [Meta-Insights] ←──→ [Consensus Pool] ←──→ [UniversalParts]  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Wisdom Flow:                                                           │
│  Extract → Validate → Graph → Synthesize → Distribute → Learn          │
└─────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import hashlib

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal,
)


class WisdomType(Enum):
    """Types of wisdom that can be collected."""
    PATTERN = auto()      # Recurring patterns observed in domain behavior
    INSIGHT = auto()      # Novel understanding derived from emergence
    LESSON = auto()       # Learned from experience and outcomes
    WARNING = auto()      # Cautionary wisdom from negative outcomes
    PRINCIPLE = auto()    # Fundamental truths that apply broadly
    HEURISTIC = auto()    # Rules of thumb for quick decisions
    META_PATTERN = auto() # Patterns across patterns (cross-domain)


class WisdomQuality(Enum):
    """Quality levels for wisdom validation."""
    UNVERIFIED = auto()   # Not yet validated
    TENTATIVE = auto()    # Some supporting evidence
    VALIDATED = auto()    # Confirmed through application
    ESTABLISHED = auto()  # Proven through repeated success
    CANONICAL = auto()    # Fundamental, foundational wisdom


class ConsensusState(Enum):
    """States for consensus building on conflicting wisdom."""
    PROPOSED = auto()     # Initial proposal
    DEBATED = auto()      # Under evaluation
    CONTESTED = auto()    # Has conflicts
    RESOLVED = auto()     # Conflicts resolved
    ADOPTED = auto()      # Accepted by all domains


@dataclass
class WisdomFragment:
    """A piece of wisdom collected from the system."""
    fragment_id: str
    source_domain: DomainLeg
    wisdom_type: WisdomType
    content_type: str  # Legacy compatibility: "pattern", "insight", "lesson", "warning"
    knowledge_vector: np.ndarray
    semantic_content: Dict[str, Any]  # Human-readable wisdom content
    confidence: float
    quality: WisdomQuality = WisdomQuality.UNVERIFIED
    applicability: List[DomainLeg] = field(default_factory=list)
    access_count: int = 0
    application_count: int = 0  # How many times this wisdom was applied
    success_count: int = 0      # Successful applications
    creation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    related_fragments: Set[str] = field(default_factory=set)  # Graph edges

    def effectiveness(self) -> float:
        """Calculate wisdom effectiveness based on application outcomes."""
        if self.application_count == 0:
            return 0.5  # Neutral if never applied
        return self.success_count / self.application_count

    def relevance_score(self) -> float:
        """Calculate current relevance based on recency and usage."""
        age_hours = (datetime.now() - self.creation_time).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + age_hours / 24)  # Decay over 24 hours
        usage_factor = min(1.0, self.access_count / 10)
        quality_bonus = [0.0, 0.1, 0.2, 0.3, 0.4][self.quality.value - 1]

        return (
            0.3 * self.confidence +
            0.2 * recency_factor +
            0.2 * usage_factor +
            0.2 * self.effectiveness() +
            0.1 + quality_bonus
        )


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph connecting wisdom fragments."""
    source_id: str
    target_id: str
    relationship_type: str  # "supports", "contradicts", "extends", "refines", "applies_to"
    strength: float = 0.5
    evidence_count: int = 1
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusProposal:
    """A proposal for resolving conflicting wisdom."""
    proposal_id: str
    conflicting_fragments: List[str]
    proposed_resolution: Dict[str, Any]
    resolution_vector: np.ndarray
    supporting_domains: Set[DomainLeg] = field(default_factory=set)
    opposing_domains: Set[DomainLeg] = field(default_factory=set)
    state: ConsensusState = ConsensusState.PROPOSED
    votes: Dict[DomainLeg, float] = field(default_factory=dict)  # -1 to 1
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MetaPattern:
    """A pattern discovered across multiple domains."""
    pattern_id: str
    name: str
    description: str
    participating_domains: List[DomainLeg]
    source_fragments: List[str]  # Fragment IDs that contribute to this pattern
    pattern_vector: np.ndarray
    occurrence_count: int = 1
    confidence: float = 0.5
    last_observed: datetime = field(default_factory=datetime.now)


@dataclass
class CollectiveMemory:
    """Long-term memory structure for collective wisdom."""
    memory_id: str
    fragments: List[str]  # fragment_ids
    summary_vector: np.ndarray
    semantic_summary: Dict[str, Any]  # Human-readable summary
    domain_weights: Dict[DomainLeg, float]
    total_accesses: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    contributing_patterns: List[str] = field(default_factory=list)


@dataclass
class WisdomDistribution:
    """Record of wisdom being distributed to a domain."""
    distribution_id: str
    fragment_id: str
    target_domain: DomainLeg
    timestamp: datetime
    pre_state_hash: str
    post_improvement: Optional[float] = None
    was_beneficial: Optional[bool] = None


class CollectiveWisdomAgent(TertiaryReBoAgent):
    """
    Agent 8: Aggregates collective knowledge across all domains.

    Like the Wood Wide Web in forests where trees share nutrients and
    information through fungal networks, this agent enables knowledge
    sharing across the TTR system:

    - NPCPU contributes philosophical insights and meta-patterns
    - ChicagoForest contributes routing optimization and network patterns
    - UniversalParts contributes operational experience and failure wisdom

    The collective wisdom becomes greater than any individual domain's
    knowledge, enabling system-wide learning and adaptation.

    Key Features:
    1. Knowledge Graph - Tracks relationships between wisdom fragments
    2. Consensus Building - Resolves conflicting wisdom through voting
    3. Meta-Pattern Mining - Discovers patterns across domains
    4. Quality Validation - Tracks wisdom effectiveness
    5. Proactive Distribution - Pushes wisdom where it's needed
    6. Meta-Learning - Learns from wisdom application outcomes
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Wisdom storage
        self.wisdom_fragments: Dict[str, WisdomFragment] = {}
        self.collective_memories: Dict[str, CollectiveMemory] = {}

        # Knowledge graph
        self.knowledge_graph: Dict[str, List[KnowledgeEdge]] = {}  # fragment_id -> edges

        # Domain indices
        self.domain_wisdom_index: Dict[DomainLeg, Set[str]] = {
            leg: set() for leg in DomainLeg
        }
        self.type_wisdom_index: Dict[WisdomType, Set[str]] = {
            wtype: set() for wtype in WisdomType
        }

        # Collective knowledge vectors
        self.collective_knowledge = np.zeros(128)
        self.domain_contributions = {leg: np.zeros(128) for leg in DomainLeg}

        # Cross-domain integration matrix
        self.integration_matrix = np.random.randn(128, 128) * 0.1
        np.fill_diagonal(self.integration_matrix, 1.0)

        # Meta-patterns discovered across domains
        self.meta_patterns: Dict[str, MetaPattern] = {}

        # Consensus system
        self.consensus_proposals: Dict[str, ConsensusProposal] = {}
        self.consensus_history: List[Dict[str, Any]] = []

        # Distribution tracking
        self.distribution_history: List[WisdomDistribution] = []
        self.domain_needs: Dict[DomainLeg, Dict[str, float]] = {
            leg: {} for leg in DomainLeg
        }

        # Learning history
        self.learning_events: deque = deque(maxlen=500)
        self.knowledge_transfers: List[Dict[str, Any]] = []

        # Performance tracking
        self.application_outcomes: List[Dict[str, Any]] = []

        # Wisdom synthesis parameters
        self.synthesis_threshold = 5  # Min fragments to synthesize
        self.consolidation_interval = 10  # Refinements between consolidations
        self.meta_pattern_threshold = 3  # Min domains for meta-pattern
        self.consensus_threshold = 0.6  # Agreement needed for adoption
        self.quality_promotion_threshold = 3  # Successful applications for quality upgrade

        # Statistics
        self.stats = {
            "fragments_collected": 0,
            "transfers_made": 0,
            "patterns_discovered": 0,
            "consensus_reached": 0,
            "quality_upgrades": 0,
        }

    @property
    def agent_role(self) -> str:
        return "Collective Wisdom - Aggregates, synthesizes, and distributes shared knowledge"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    # =========================================================================
    # Wisdom Extraction and Classification
    # =========================================================================

    def _classify_wisdom_type(self, state: Any, tbl: TripleBottomLine) -> WisdomType:
        """Classify what type of wisdom this state represents."""
        if state.emergence_potential > 0.8:
            return WisdomType.INSIGHT
        elif state.coherence > 0.9:
            return WisdomType.PRINCIPLE
        elif tbl.harmony_score < 0.3:
            return WisdomType.WARNING
        elif state.consciousness_level > 0.7:
            return WisdomType.PATTERN
        elif state.energy_flow > 0.7:
            return WisdomType.HEURISTIC
        else:
            return WisdomType.LESSON

    def _extract_semantic_content(self, domain: DomainLeg, state: Any, wisdom_type: WisdomType) -> Dict[str, Any]:
        """Extract human-readable semantic content from a domain state."""
        return {
            "domain": domain.value,
            "wisdom_type": wisdom_type.name,
            "consciousness_snapshot": round(state.consciousness_level, 3),
            "coherence_snapshot": round(state.coherence, 3),
            "emergence_snapshot": round(state.emergence_potential, 3),
            "energy_snapshot": round(state.energy_flow, 3),
            "key_characteristics": self._identify_characteristics(state),
            "extraction_context": {
                "timestamp": datetime.now().isoformat(),
                "state_hash": hashlib.md5(state.state_vector.tobytes()).hexdigest()[:8]
            }
        }

    def _identify_characteristics(self, state: Any) -> List[str]:
        """Identify key characteristics of a domain state."""
        chars = []
        if state.consciousness_level > 0.7:
            chars.append("highly_conscious")
        if state.coherence > 0.8:
            chars.append("highly_coherent")
        if state.emergence_potential > 0.6:
            chars.append("emergent_potential")
        if state.energy_flow > 0.7:
            chars.append("high_energy")
        if state.qualia_richness > 0.6:
            chars.append("rich_experience")
        if state.connectivity > 0.7:
            chars.append("well_connected")
        return chars

    def _extract_wisdom(self, domain: DomainLeg, state: Any, tbl: TripleBottomLine) -> Optional[WisdomFragment]:
        """Extract a wisdom fragment from a domain state."""
        # Check if there's something worth extracting
        if state.coherence < 0.25:
            return None

        # Classify wisdom type
        wisdom_type = self._classify_wisdom_type(state, tbl)

        # Legacy content type for compatibility
        content_type = wisdom_type.name.lower()
        if wisdom_type == WisdomType.PRINCIPLE:
            content_type = "pattern"
        elif wisdom_type == WisdomType.HEURISTIC:
            content_type = "lesson"

        # Create knowledge vector
        knowledge = state.state_vector.copy()
        if len(knowledge) < 128:
            knowledge = np.pad(knowledge, (0, 128 - len(knowledge)))
        else:
            knowledge = knowledge[:128]

        # Determine applicability based on coherence and type
        applicability = [domain]
        if state.coherence > 0.7:
            # Highly coherent wisdom might apply broadly
            applicability = list(DomainLeg)
        elif state.coherence > 0.5:
            # Add one more domain based on similarity
            for other_domain in DomainLeg:
                if other_domain != domain:
                    other_state = tbl.get_state(other_domain)
                    similarity = np.dot(state.state_vector, other_state.state_vector) / (
                        np.linalg.norm(state.state_vector) * np.linalg.norm(other_state.state_vector) + 1e-6
                    )
                    if similarity > 0.5:
                        applicability.append(other_domain)
                        break

        # Extract semantic content
        semantic = self._extract_semantic_content(domain, state, wisdom_type)

        return WisdomFragment(
            fragment_id=f"wis_{domain.value}_{datetime.now().strftime('%H%M%S')}_{np.random.randint(10000)}",
            source_domain=domain,
            wisdom_type=wisdom_type,
            content_type=content_type,
            knowledge_vector=knowledge,
            semantic_content=semantic,
            confidence=state.coherence,
            quality=WisdomQuality.UNVERIFIED,
            applicability=applicability
        )

    # =========================================================================
    # Knowledge Graph Operations
    # =========================================================================

    def _add_to_knowledge_graph(self, fragment: WisdomFragment) -> None:
        """Add a fragment to the knowledge graph with edges to related fragments."""
        self.knowledge_graph[fragment.fragment_id] = []

        # Find related fragments based on vector similarity
        for existing_id, existing in self.wisdom_fragments.items():
            if existing_id == fragment.fragment_id:
                continue

            similarity = np.dot(
                fragment.knowledge_vector,
                existing.knowledge_vector
            ) / (np.linalg.norm(fragment.knowledge_vector) *
                 np.linalg.norm(existing.knowledge_vector) + 1e-6)

            if similarity > 0.7:
                # Strong similarity - supporting relationship
                edge = KnowledgeEdge(
                    source_id=fragment.fragment_id,
                    target_id=existing_id,
                    relationship_type="supports",
                    strength=similarity
                )
                self.knowledge_graph[fragment.fragment_id].append(edge)
                fragment.related_fragments.add(existing_id)
                existing.related_fragments.add(fragment.fragment_id)

            elif similarity < -0.3:
                # Contradiction detected
                edge = KnowledgeEdge(
                    source_id=fragment.fragment_id,
                    target_id=existing_id,
                    relationship_type="contradicts",
                    strength=abs(similarity)
                )
                self.knowledge_graph[fragment.fragment_id].append(edge)
                fragment.related_fragments.add(existing_id)

            elif abs(similarity) > 0.3 and fragment.source_domain != existing.source_domain:
                # Cross-domain connection
                edge = KnowledgeEdge(
                    source_id=fragment.fragment_id,
                    target_id=existing_id,
                    relationship_type="extends",
                    strength=abs(similarity)
                )
                self.knowledge_graph[fragment.fragment_id].append(edge)
                fragment.related_fragments.add(existing_id)

    def _find_contradictions(self) -> List[Tuple[str, str, float]]:
        """Find contradicting wisdom fragments in the knowledge graph."""
        contradictions = []
        for fragment_id, edges in self.knowledge_graph.items():
            for edge in edges:
                if edge.relationship_type == "contradicts":
                    contradictions.append((edge.source_id, edge.target_id, edge.strength))
        return contradictions

    # =========================================================================
    # Meta-Pattern Mining
    # =========================================================================

    def _mine_meta_patterns(self, tbl: TripleBottomLine) -> List[MetaPattern]:
        """Discover patterns that span multiple domains."""
        patterns = []

        # Group fragments by type
        type_groups: Dict[WisdomType, List[WisdomFragment]] = {
            wtype: [] for wtype in WisdomType
        }
        for frag in self.wisdom_fragments.values():
            type_groups[frag.wisdom_type].append(frag)

        # Look for patterns that appear in multiple domains
        for wtype, fragments in type_groups.items():
            if len(fragments) < 2:
                continue

            # Check domain diversity
            domains_represented = set(f.source_domain for f in fragments)
            if len(domains_represented) >= self.meta_pattern_threshold - 1:
                # Compute consensus vector
                vectors = [f.knowledge_vector for f in fragments]
                pattern_vector = np.mean(vectors, axis=0)

                # Check if vectors agree (low variance in key dimensions)
                variance = np.var(vectors, axis=0)
                agreement = 1.0 - np.mean(variance)

                if agreement > 0.3:
                    pattern = MetaPattern(
                        pattern_id=f"meta_{wtype.name}_{datetime.now().strftime('%H%M%S')}",
                        name=f"Cross-Domain {wtype.name.title()}",
                        description=f"A {wtype.name.lower()} observed across {len(domains_represented)} domains",
                        participating_domains=list(domains_represented),
                        source_fragments=[f.fragment_id for f in fragments],
                        pattern_vector=pattern_vector,
                        confidence=agreement
                    )
                    patterns.append(pattern)

        return patterns

    # =========================================================================
    # Consensus Building
    # =========================================================================

    def _build_consensus(self, contradictions: List[Tuple[str, str, float]], tbl: TripleBottomLine) -> List[ConsensusProposal]:
        """Build consensus proposals for contradicting wisdom."""
        proposals = []

        for source_id, target_id, strength in contradictions:
            if source_id not in self.wisdom_fragments or target_id not in self.wisdom_fragments:
                continue

            source = self.wisdom_fragments[source_id]
            target = self.wisdom_fragments[target_id]

            # Skip if already in a proposal
            existing_proposal = any(
                source_id in p.conflicting_fragments or target_id in p.conflicting_fragments
                for p in self.consensus_proposals.values()
                if p.state not in [ConsensusState.RESOLVED, ConsensusState.ADOPTED]
            )
            if existing_proposal:
                continue

            # Create resolution by weighted averaging based on quality and effectiveness
            source_weight = source.relevance_score() * (source.quality.value / 5)
            target_weight = target.relevance_score() * (target.quality.value / 5)
            total_weight = source_weight + target_weight + 1e-6

            resolution_vector = (
                source.knowledge_vector * (source_weight / total_weight) +
                target.knowledge_vector * (target_weight / total_weight)
            )

            proposal = ConsensusProposal(
                proposal_id=f"cons_{datetime.now().strftime('%H%M%S')}_{np.random.randint(1000)}",
                conflicting_fragments=[source_id, target_id],
                proposed_resolution={
                    "method": "weighted_merge",
                    "source_weight": source_weight / total_weight,
                    "target_weight": target_weight / total_weight,
                    "source_domain": source.source_domain.value,
                    "target_domain": target.source_domain.value
                },
                resolution_vector=resolution_vector
            )

            # Initial voting based on domain alignment
            for domain in DomainLeg:
                domain_state = tbl.get_state(domain)
                # Vote based on similarity to domain's current state
                similarity = np.dot(resolution_vector[:64], domain_state.state_vector) / (
                    np.linalg.norm(resolution_vector[:64]) * np.linalg.norm(domain_state.state_vector) + 1e-6
                )
                proposal.votes[domain] = similarity

                if similarity > 0.3:
                    proposal.supporting_domains.add(domain)
                elif similarity < -0.3:
                    proposal.opposing_domains.add(domain)

            # Determine state
            avg_vote = np.mean(list(proposal.votes.values()))
            if avg_vote > self.consensus_threshold:
                proposal.state = ConsensusState.RESOLVED
            elif len(proposal.opposing_domains) > 0:
                proposal.state = ConsensusState.CONTESTED
            else:
                proposal.state = ConsensusState.DEBATED

            proposals.append(proposal)

        return proposals

    # =========================================================================
    # Wisdom Quality Validation
    # =========================================================================

    def _update_wisdom_quality(self, fragment: WisdomFragment) -> bool:
        """Update wisdom quality based on application history."""
        old_quality = fragment.quality

        if fragment.application_count >= self.quality_promotion_threshold:
            effectiveness = fragment.effectiveness()

            if effectiveness > 0.8 and fragment.quality.value < WisdomQuality.ESTABLISHED.value:
                fragment.quality = WisdomQuality.ESTABLISHED
            elif effectiveness > 0.6 and fragment.quality.value < WisdomQuality.VALIDATED.value:
                fragment.quality = WisdomQuality.VALIDATED
            elif effectiveness > 0.4 and fragment.quality.value < WisdomQuality.TENTATIVE.value:
                fragment.quality = WisdomQuality.TENTATIVE

        if fragment.quality != old_quality:
            self.stats["quality_upgrades"] += 1
            return True
        return False

    # =========================================================================
    # Domain Needs Assessment
    # =========================================================================

    def _assess_domain_needs(self, tbl: TripleBottomLine) -> Dict[DomainLeg, Dict[str, float]]:
        """Assess what each domain needs in terms of wisdom."""
        needs = {}

        for domain in DomainLeg:
            state = tbl.get_state(domain)
            domain_needs = {}

            # Low consciousness - needs awareness wisdom
            if state.consciousness_level < 0.4:
                domain_needs["consciousness_boost"] = 0.4 - state.consciousness_level

            # Low coherence - needs structural wisdom
            if state.coherence < 0.5:
                domain_needs["coherence_boost"] = 0.5 - state.coherence

            # Low energy - needs efficiency wisdom
            if state.energy_flow < 0.4:
                domain_needs["energy_optimization"] = 0.4 - state.energy_flow

            # Low connectivity - needs integration wisdom
            if state.connectivity < 0.5:
                domain_needs["connectivity_boost"] = 0.5 - state.connectivity

            # Low emergence - needs innovation wisdom
            if state.emergence_potential < 0.3:
                domain_needs["emergence_catalyst"] = 0.3 - state.emergence_potential

            needs[domain] = domain_needs

        return needs

    def _find_wisdom_for_need(self, need_type: str, domain: DomainLeg) -> Optional[WisdomFragment]:
        """Find wisdom that addresses a specific domain need."""
        # Map needs to wisdom types and characteristics
        need_to_type_map = {
            "consciousness_boost": [WisdomType.INSIGHT, WisdomType.PATTERN],
            "coherence_boost": [WisdomType.PRINCIPLE, WisdomType.PATTERN],
            "energy_optimization": [WisdomType.HEURISTIC, WisdomType.LESSON],
            "connectivity_boost": [WisdomType.PATTERN, WisdomType.HEURISTIC],
            "emergence_catalyst": [WisdomType.INSIGHT, WisdomType.META_PATTERN],
        }

        target_types = need_to_type_map.get(need_type, [WisdomType.LESSON])

        # Find applicable wisdom from other domains
        candidates = []
        for frag in self.wisdom_fragments.values():
            if frag.wisdom_type in target_types and domain in frag.applicability:
                if frag.source_domain != domain:  # Cross-pollination
                    candidates.append((frag.relevance_score(), frag))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    # =========================================================================
    # Refinement Phases
    # =========================================================================

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather wisdom candidates from all domains."""
        perceptions = {
            "domain_states": {},
            "wisdom_candidates": [],
            "knowledge_gaps": [],
            "synthesis_opportunities": [],
            "contradictions": [],
            "domain_needs": {},
            "meta_pattern_opportunities": []
        }

        # Analyze each domain for wisdom extraction
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_states"][domain.value] = {
                "coherence": state.coherence,
                "emergence_potential": state.emergence_potential,
                "consciousness_level": state.consciousness_level,
                "energy_flow": state.energy_flow,
                "connectivity": state.connectivity
            }

            # Extract potential wisdom
            wisdom = self._extract_wisdom(domain, state, tbl)
            if wisdom:
                perceptions["wisdom_candidates"].append(wisdom)

        # Identify knowledge gaps
        for domain in DomainLeg:
            domain_fragments = self.domain_wisdom_index[domain]
            if len(domain_fragments) < 5:
                perceptions["knowledge_gaps"].append({
                    "domain": domain.value,
                    "fragment_count": len(domain_fragments),
                    "severity": "high" if len(domain_fragments) < 2 else "medium"
                })

        # Check for synthesis opportunities
        if len(self.wisdom_fragments) >= self.synthesis_threshold:
            # Group fragments by type
            type_counts = {}
            for frag in self.wisdom_fragments.values():
                type_key = frag.wisdom_type.name
                type_counts[type_key] = type_counts.get(type_key, 0) + 1

            for content_type, count in type_counts.items():
                if count >= 3:
                    perceptions["synthesis_opportunities"].append({
                        "content_type": content_type,
                        "fragment_count": count
                    })

        # Find contradictions
        perceptions["contradictions"] = self._find_contradictions()

        # Assess domain needs
        perceptions["domain_needs"] = self._assess_domain_needs(tbl)

        # Check for meta-pattern opportunities
        domain_diversity = len(set(f.source_domain for f in self.wisdom_fragments.values()))
        if domain_diversity >= 2 and len(self.wisdom_fragments) >= 5:
            perceptions["meta_pattern_opportunities"].append({
                "domains_represented": domain_diversity,
                "total_fragments": len(self.wisdom_fragments)
            })

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze wisdom collection opportunities."""
        analysis = {
            "fragments_to_store": [],
            "knowledge_transfers": [],
            "synthesis_plans": [],
            "consolidation_needed": False,
            "consensus_proposals": [],
            "meta_patterns": [],
            "proactive_distributions": [],
            "quality_updates": []
        }

        # Decide which wisdom candidates to store
        for candidate in perception.get("wisdom_candidates", []):
            # Check for redundancy
            is_redundant = False
            for existing in self.wisdom_fragments.values():
                if existing.source_domain == candidate.source_domain:
                    similarity = np.dot(
                        candidate.knowledge_vector,
                        existing.knowledge_vector
                    ) / (np.linalg.norm(candidate.knowledge_vector) *
                         np.linalg.norm(existing.knowledge_vector) + 1e-6)
                    if similarity > 0.85:
                        is_redundant = True
                        break

            if not is_redundant and candidate.confidence > 0.4:
                analysis["fragments_to_store"].append(candidate)

        # Plan knowledge transfers for gaps
        for gap in perception.get("knowledge_gaps", []):
            target_domain = DomainLeg(gap["domain"])
            # Find wisdom from other domains that could apply
            for frag in self.wisdom_fragments.values():
                if frag.source_domain != target_domain and target_domain in frag.applicability:
                    if frag.quality.value >= WisdomQuality.TENTATIVE.value:
                        analysis["knowledge_transfers"].append({
                            "from_fragment": frag.fragment_id,
                            "to_domain": target_domain.value,
                            "confidence": frag.confidence,
                            "quality": frag.quality.name
                        })
                        break  # One transfer per gap for now

        # Plan synthesis for opportunities
        for opp in perception.get("synthesis_opportunities", []):
            relevant_fragments = [
                fid for fid, frag in self.wisdom_fragments.items()
                if frag.wisdom_type.name == opp["content_type"]
            ]
            if len(relevant_fragments) >= 3:
                analysis["synthesis_plans"].append({
                    "content_type": opp["content_type"],
                    "fragments": relevant_fragments[:5]  # Limit to 5
                })

        # Check if consolidation is needed
        if len(self.wisdom_fragments) > 100:
            analysis["consolidation_needed"] = True

        # Build consensus for contradictions
        if perception.get("contradictions"):
            proposals = self._build_consensus(perception["contradictions"], tbl)
            analysis["consensus_proposals"] = proposals

        # Mine meta-patterns
        if perception.get("meta_pattern_opportunities"):
            patterns = self._mine_meta_patterns(tbl)
            analysis["meta_patterns"] = patterns

        # Plan proactive distributions based on domain needs
        for domain, needs in perception.get("domain_needs", {}).items():
            for need_type, magnitude in needs.items():
                if magnitude > 0.1:  # Significant need
                    wisdom = self._find_wisdom_for_need(need_type, domain)
                    if wisdom:
                        analysis["proactive_distributions"].append({
                            "fragment_id": wisdom.fragment_id,
                            "target_domain": domain.value,
                            "need_type": need_type,
                            "magnitude": magnitude
                        })

        # Check for quality updates
        for frag in self.wisdom_fragments.values():
            if frag.application_count >= self.quality_promotion_threshold:
                effectiveness = frag.effectiveness()
                current_quality = frag.quality.value
                expected_quality = (
                    WisdomQuality.ESTABLISHED.value if effectiveness > 0.8 else
                    WisdomQuality.VALIDATED.value if effectiveness > 0.6 else
                    WisdomQuality.TENTATIVE.value if effectiveness > 0.4 else
                    current_quality
                )
                if expected_quality > current_quality:
                    analysis["quality_updates"].append({
                        "fragment_id": frag.fragment_id,
                        "current_quality": frag.quality.name,
                        "effectiveness": effectiveness
                    })

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Prepare wisdom updates."""
        synthesis = {
            "store_operations": [],
            "transfer_operations": [],
            "collective_memory_updates": [],
            "consolidation_operations": [],
            "consensus_resolutions": [],
            "meta_pattern_registrations": [],
            "proactive_distributions": [],
            "quality_promotions": []
        }

        # Prepare storage operations
        for fragment in analysis.get("fragments_to_store", []):
            synthesis["store_operations"].append({
                "fragment": fragment,
                "index_domains": fragment.applicability,
                "add_to_graph": True
            })

        # Prepare transfer operations
        for transfer in analysis.get("knowledge_transfers", []):
            synthesis["transfer_operations"].append({
                "fragment_id": transfer["from_fragment"],
                "target_domain": transfer["to_domain"],
                "blend_factor": 0.15 * transfer["confidence"]
            })

        # Create collective memories from synthesis
        for plan in analysis.get("synthesis_plans", []):
            fragment_vectors = [
                self.wisdom_fragments[fid].knowledge_vector
                for fid in plan["fragments"]
                if fid in self.wisdom_fragments
            ]

            if fragment_vectors:
                summary = np.mean(fragment_vectors, axis=0)

                # Create semantic summary
                source_fragments = [self.wisdom_fragments[fid] for fid in plan["fragments"] if fid in self.wisdom_fragments]
                semantic_summary = {
                    "type": plan["content_type"],
                    "fragment_count": len(source_fragments),
                    "domains_represented": list(set(f.source_domain.value for f in source_fragments)),
                    "avg_confidence": np.mean([f.confidence for f in source_fragments]),
                    "synthesized_at": datetime.now().isoformat()
                }

                memory = CollectiveMemory(
                    memory_id=f"mem_{plan['content_type']}_{datetime.now().strftime('%H%M%S')}",
                    fragments=plan["fragments"],
                    summary_vector=summary,
                    semantic_summary=semantic_summary,
                    domain_weights={d: 1.0/3 for d in DomainLeg}
                )
                synthesis["collective_memory_updates"].append(memory)

        # Prepare consolidation
        if analysis.get("consolidation_needed"):
            # Mark old, low-access, low-quality fragments for removal
            for fid, frag in list(self.wisdom_fragments.items()):
                age_hours = (datetime.now() - frag.creation_time).total_seconds() / 3600
                if age_hours > 2 and frag.access_count < 2 and frag.quality == WisdomQuality.UNVERIFIED:
                    synthesis["consolidation_operations"].append({
                        "action": "remove",
                        "fragment_id": fid
                    })

        # Prepare consensus resolutions
        for proposal in analysis.get("consensus_proposals", []):
            if proposal.state == ConsensusState.RESOLVED:
                synthesis["consensus_resolutions"].append({
                    "proposal": proposal,
                    "action": "adopt",
                    "apply_resolution": True
                })

        # Register meta-patterns
        for pattern in analysis.get("meta_patterns", []):
            synthesis["meta_pattern_registrations"].append(pattern)

        # Proactive distributions
        synthesis["proactive_distributions"] = analysis.get("proactive_distributions", [])

        # Quality promotions
        synthesis["quality_promotions"] = analysis.get("quality_updates", [])

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply wisdom updates."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Store new fragments
        stored_count = 0
        for op in synthesis.get("store_operations", []):
            fragment = op["fragment"]
            self.wisdom_fragments[fragment.fragment_id] = fragment

            # Update indices
            for domain in op["index_domains"]:
                self.domain_wisdom_index[domain].add(fragment.fragment_id)
            self.type_wisdom_index[fragment.wisdom_type].add(fragment.fragment_id)

            # Add to knowledge graph
            if op.get("add_to_graph", True):
                self._add_to_knowledge_graph(fragment)

            # Update domain contributions
            self.domain_contributions[fragment.source_domain] += fragment.knowledge_vector * 0.1

            stored_count += 1
            self.stats["fragments_collected"] += 1

        if stored_count > 0:
            changes["fragments_stored"] = stored_count
            insights.append(f"Collected {stored_count} new wisdom fragments")

        # Execute transfers
        transfer_count = 0
        for op in synthesis.get("transfer_operations", []):
            frag_id = op["fragment_id"]
            target_domain = DomainLeg(op["target_domain"])

            if frag_id in self.wisdom_fragments:
                fragment = self.wisdom_fragments[frag_id]
                fragment.access_count += 1
                fragment.application_count += 1
                fragment.last_accessed = datetime.now()
                fragment.last_applied = datetime.now()

                # Apply knowledge to target domain
                state = tbl.get_state(target_domain)
                pre_state_hash = hashlib.md5(state.state_vector.tobytes()).hexdigest()[:8]

                knowledge = fragment.knowledge_vector[:64]
                blend_factor = op["blend_factor"]
                old_vector = state.state_vector.copy()
                state.state_vector = (1 - blend_factor) * state.state_vector + blend_factor * knowledge

                transfer_count += 1
                self.stats["transfers_made"] += 1

                # Record distribution
                distribution = WisdomDistribution(
                    distribution_id=f"dist_{datetime.now().strftime('%H%M%S%f')}",
                    fragment_id=frag_id,
                    target_domain=target_domain,
                    timestamp=datetime.now(),
                    pre_state_hash=pre_state_hash
                )
                self.distribution_history.append(distribution)

                self.knowledge_transfers.append({
                    "from": fragment.source_domain.value,
                    "to": target_domain.value,
                    "fragment_id": frag_id,
                    "timestamp": datetime.now().isoformat()
                })

        if transfer_count > 0:
            changes["knowledge_transfers"] = transfer_count
            insights.append(f"Transferred knowledge {transfer_count} times across domains")

        # Create collective memories
        memory_count = 0
        for memory in synthesis.get("collective_memory_updates", []):
            self.collective_memories[memory.memory_id] = memory
            memory_count += 1

        if memory_count > 0:
            changes["memories_synthesized"] = memory_count
            insights.append(f"Synthesized {memory_count} collective memories")

        # Consolidation
        removed_count = 0
        for op in synthesis.get("consolidation_operations", []):
            if op["action"] == "remove" and op["fragment_id"] in self.wisdom_fragments:
                frag = self.wisdom_fragments[op["fragment_id"]]
                # Remove from indices
                for domain in DomainLeg:
                    self.domain_wisdom_index[domain].discard(op["fragment_id"])
                self.type_wisdom_index[frag.wisdom_type].discard(op["fragment_id"])
                # Remove from graph
                if op["fragment_id"] in self.knowledge_graph:
                    del self.knowledge_graph[op["fragment_id"]]
                del self.wisdom_fragments[op["fragment_id"]]
                removed_count += 1

        if removed_count > 0:
            changes["fragments_consolidated"] = removed_count
            insights.append(f"Consolidated {removed_count} old fragments")

        # Apply consensus resolutions
        consensus_count = 0
        for resolution in synthesis.get("consensus_resolutions", []):
            proposal = resolution["proposal"]
            if resolution["action"] == "adopt" and resolution.get("apply_resolution"):
                # Create a new wisdom fragment from the resolution
                resolved_fragment = WisdomFragment(
                    fragment_id=f"resolved_{proposal.proposal_id}",
                    source_domain=DomainLeg.NPCPU,  # Consensus wisdom is domain-neutral
                    wisdom_type=WisdomType.PRINCIPLE,
                    content_type="pattern",
                    knowledge_vector=proposal.resolution_vector,
                    semantic_content={
                        "type": "consensus_resolution",
                        "conflicting_fragments": proposal.conflicting_fragments,
                        "supporting_domains": [d.value for d in proposal.supporting_domains],
                        "resolution_method": proposal.proposed_resolution.get("method")
                    },
                    confidence=np.mean(list(proposal.votes.values())),
                    quality=WisdomQuality.VALIDATED,
                    applicability=list(DomainLeg)
                )
                self.wisdom_fragments[resolved_fragment.fragment_id] = resolved_fragment
                proposal.state = ConsensusState.ADOPTED
                self.consensus_proposals[proposal.proposal_id] = proposal
                consensus_count += 1
                self.stats["consensus_reached"] += 1

        if consensus_count > 0:
            changes["consensus_resolutions"] = consensus_count
            insights.append(f"Resolved {consensus_count} wisdom conflicts through consensus")

        # Register meta-patterns
        pattern_count = 0
        for pattern in synthesis.get("meta_pattern_registrations", []):
            self.meta_patterns[pattern.pattern_id] = pattern

            # Create a meta-pattern wisdom fragment
            meta_fragment = WisdomFragment(
                fragment_id=f"meta_{pattern.pattern_id}",
                source_domain=DomainLeg.NPCPU,  # Meta-patterns are domain-neutral
                wisdom_type=WisdomType.META_PATTERN,
                content_type="pattern",
                knowledge_vector=pattern.pattern_vector,
                semantic_content={
                    "pattern_name": pattern.name,
                    "description": pattern.description,
                    "participating_domains": [d.value for d in pattern.participating_domains],
                    "source_fragments": pattern.source_fragments
                },
                confidence=pattern.confidence,
                quality=WisdomQuality.VALIDATED,
                applicability=list(DomainLeg)
            )
            self.wisdom_fragments[meta_fragment.fragment_id] = meta_fragment
            pattern_count += 1
            self.stats["patterns_discovered"] += 1

        if pattern_count > 0:
            changes["meta_patterns_discovered"] = pattern_count
            insights.append(f"Discovered {pattern_count} cross-domain meta-patterns")

        # Proactive distributions
        proactive_count = 0
        for dist in synthesis.get("proactive_distributions", []):
            frag_id = dist["fragment_id"]
            target_domain = DomainLeg(dist["target_domain"])

            if frag_id in self.wisdom_fragments:
                fragment = self.wisdom_fragments[frag_id]
                state = tbl.get_state(target_domain)

                # Smaller blend for proactive distributions
                blend_factor = 0.05 * dist["magnitude"]
                knowledge = fragment.knowledge_vector[:64]
                state.state_vector = (1 - blend_factor) * state.state_vector + blend_factor * knowledge

                fragment.application_count += 1
                proactive_count += 1

        if proactive_count > 0:
            changes["proactive_distributions"] = proactive_count
            insights.append(f"Proactively distributed wisdom {proactive_count} times based on domain needs")

        # Quality promotions
        promotion_count = 0
        for update in synthesis.get("quality_promotions", []):
            if update["fragment_id"] in self.wisdom_fragments:
                if self._update_wisdom_quality(self.wisdom_fragments[update["fragment_id"]]):
                    promotion_count += 1

        if promotion_count > 0:
            changes["quality_promotions"] = promotion_count
            insights.append(f"Promoted quality of {promotion_count} wisdom fragments")

        # Update collective knowledge
        all_contributions = np.stack(list(self.domain_contributions.values()))
        self.collective_knowledge = np.mean(all_contributions, axis=0)

        # Apply integration matrix to blend cross-domain knowledge
        self.collective_knowledge = np.tanh(
            self.integration_matrix @ self.collective_knowledge
        )

        # Update harmony and metrics
        tbl.calculate_harmony()
        tbl.calculate_integration_depth()

        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["integration_depth"] = tbl.integration_depth
        metrics_delta["total_fragments"] = len(self.wisdom_fragments)
        metrics_delta["total_memories"] = len(self.collective_memories)
        metrics_delta["meta_patterns"] = len(self.meta_patterns)
        metrics_delta["active_consensus_proposals"] = len([
            p for p in self.consensus_proposals.values()
            if p.state not in [ConsensusState.ADOPTED, ConsensusState.RESOLVED]
        ])

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    # =========================================================================
    # Meta-Learning: Learning from Wisdom Application Outcomes
    # =========================================================================

    async def record_application_outcome(self, fragment_id: str, was_successful: bool, improvement: float) -> None:
        """Record the outcome of applying a wisdom fragment."""
        if fragment_id in self.wisdom_fragments:
            fragment = self.wisdom_fragments[fragment_id]
            if was_successful:
                fragment.success_count += 1

            self.application_outcomes.append({
                "fragment_id": fragment_id,
                "was_successful": was_successful,
                "improvement": improvement,
                "timestamp": datetime.now().isoformat()
            })

            # Update quality if needed
            self._update_wisdom_quality(fragment)

    # =========================================================================
    # Metrics and State Access
    # =========================================================================

    def get_wisdom_summary(self) -> Dict[str, Any]:
        """Get a summary of the current wisdom state."""
        quality_counts = {}
        type_counts = {}
        domain_counts = {}

        for frag in self.wisdom_fragments.values():
            quality_counts[frag.quality.name] = quality_counts.get(frag.quality.name, 0) + 1
            type_counts[frag.wisdom_type.name] = type_counts.get(frag.wisdom_type.name, 0) + 1
            domain_counts[frag.source_domain.value] = domain_counts.get(frag.source_domain.value, 0) + 1

        return {
            "total_fragments": len(self.wisdom_fragments),
            "by_quality": quality_counts,
            "by_type": type_counts,
            "by_domain": domain_counts,
            "collective_memories": len(self.collective_memories),
            "meta_patterns": len(self.meta_patterns),
            "pending_consensus": len([
                p for p in self.consensus_proposals.values()
                if p.state not in [ConsensusState.ADOPTED]
            ]),
            "knowledge_graph_edges": sum(len(edges) for edges in self.knowledge_graph.values()),
            "stats": self.stats
        }

    def get_top_wisdom(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N most relevant wisdom fragments."""
        ranked = sorted(
            self.wisdom_fragments.values(),
            key=lambda f: f.relevance_score(),
            reverse=True
        )[:n]

        return [
            {
                "id": f.fragment_id,
                "type": f.wisdom_type.name,
                "quality": f.quality.name,
                "source": f.source_domain.value,
                "confidence": f.confidence,
                "effectiveness": f.effectiveness(),
                "relevance": f.relevance_score(),
                "semantic": f.semantic_content
            }
            for f in ranked
        ]
