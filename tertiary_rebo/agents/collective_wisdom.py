"""
CollectiveWisdomAgent - Aggregates collective knowledge across all legs.

Responsibilities:
- Collect and synthesize wisdom from all three domains
- Build shared knowledge representations
- Enable knowledge transfer between domains
- Maintain long-term memory of successful patterns
- Facilitate distributed learning
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
)


@dataclass
class WisdomFragment:
    """A piece of wisdom collected from the system."""
    fragment_id: str
    source_domain: DomainLeg
    content_type: str  # "pattern", "insight", "lesson", "warning"
    knowledge_vector: np.ndarray
    confidence: float
    applicability: List[DomainLeg]
    access_count: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class CollectiveMemory:
    """Long-term memory structure for collective wisdom."""
    memory_id: str
    fragments: List[str]  # fragment_ids
    summary_vector: np.ndarray
    domain_weights: Dict[DomainLeg, float]
    total_accesses: int = 0
    creation_time: datetime = field(default_factory=datetime.now)


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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Wisdom storage
        self.wisdom_fragments: Dict[str, WisdomFragment] = {}
        self.collective_memories: Dict[str, CollectiveMemory] = {}

        # Knowledge indices
        self.domain_wisdom_index: Dict[DomainLeg, Set[str]] = {
            leg: set() for leg in DomainLeg
        }

        # Collective knowledge vectors
        self.collective_knowledge = np.zeros(128)
        self.domain_contributions = {leg: np.zeros(128) for leg in DomainLeg}

        # Learning history
        self.learning_events: deque = deque(maxlen=200)
        self.knowledge_transfers: List[Dict[str, Any]] = []

        # Wisdom synthesis parameters
        self.synthesis_threshold = 5  # Min fragments to synthesize
        self.consolidation_interval = 10  # Refinements between consolidations

    @property
    def agent_role(self) -> str:
        return "Collective Wisdom - Aggregates and distributes shared knowledge"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def _extract_wisdom(self, domain: DomainLeg, state: Any, tbl: TripleBottomLine) -> Optional[WisdomFragment]:
        """Extract a wisdom fragment from a domain state."""
        # Check if there's something worth extracting
        if state.coherence < 0.3:
            return None

        # Create knowledge vector
        knowledge = state.state_vector.copy()

        # Determine content type based on state characteristics
        if state.emergence_potential > 0.7:
            content_type = "insight"
        elif state.consciousness_level > 0.8:
            content_type = "pattern"
        elif tbl.harmony_score < 0.4:
            content_type = "warning"
        else:
            content_type = "lesson"

        # Determine applicability
        applicability = [domain]
        if state.coherence > 0.7:
            # Highly coherent wisdom might apply broadly
            applicability = list(DomainLeg)

        return WisdomFragment(
            fragment_id=f"wis_{domain.value}_{datetime.now().strftime('%H%M%S')}_{np.random.randint(1000)}",
            source_domain=domain,
            content_type=content_type,
            knowledge_vector=knowledge[:128] if len(knowledge) >= 128 else np.pad(knowledge, (0, 128-len(knowledge))),
            confidence=state.coherence,
            applicability=applicability
        )

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather wisdom candidates from all domains."""
        perceptions = {
            "domain_states": {},
            "wisdom_candidates": [],
            "knowledge_gaps": [],
            "synthesis_opportunities": []
        }

        # Analyze each domain for wisdom extraction
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_states"][domain.value] = {
                "coherence": state.coherence,
                "emergence_potential": state.emergence_potential,
                "consciousness_level": state.consciousness_level
            }

            # Extract potential wisdom
            wisdom = self._extract_wisdom(domain, state, tbl)
            if wisdom:
                perceptions["wisdom_candidates"].append(wisdom)

        # Identify knowledge gaps
        for domain in DomainLeg:
            domain_fragments = self.domain_wisdom_index[domain]
            if len(domain_fragments) < 3:
                perceptions["knowledge_gaps"].append({
                    "domain": domain.value,
                    "fragment_count": len(domain_fragments),
                    "severity": "high" if len(domain_fragments) == 0 else "medium"
                })

        # Check for synthesis opportunities
        if len(self.wisdom_fragments) >= self.synthesis_threshold:
            # Group fragments by type
            type_counts = {}
            for frag in self.wisdom_fragments.values():
                type_counts[frag.content_type] = type_counts.get(frag.content_type, 0) + 1

            for content_type, count in type_counts.items():
                if count >= 3:
                    perceptions["synthesis_opportunities"].append({
                        "content_type": content_type,
                        "fragment_count": count
                    })

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze wisdom collection opportunities."""
        analysis = {
            "fragments_to_store": [],
            "knowledge_transfers": [],
            "synthesis_plans": [],
            "consolidation_needed": False
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
                    if similarity > 0.9:
                        is_redundant = True
                        break

            if not is_redundant and candidate.confidence > 0.5:
                analysis["fragments_to_store"].append(candidate)

        # Plan knowledge transfers for gaps
        for gap in perception.get("knowledge_gaps", []):
            target_domain = DomainLeg(gap["domain"])
            # Find wisdom from other domains that could apply
            for frag in self.wisdom_fragments.values():
                if frag.source_domain != target_domain and target_domain in frag.applicability:
                    analysis["knowledge_transfers"].append({
                        "from_fragment": frag.fragment_id,
                        "to_domain": target_domain.value,
                        "confidence": frag.confidence
                    })
                    break  # One transfer per gap for now

        # Plan synthesis for opportunities
        for opp in perception.get("synthesis_opportunities", []):
            relevant_fragments = [
                fid for fid, frag in self.wisdom_fragments.items()
                if frag.content_type == opp["content_type"]
            ]
            if len(relevant_fragments) >= 3:
                analysis["synthesis_plans"].append({
                    "content_type": opp["content_type"],
                    "fragments": relevant_fragments[:5]  # Limit to 5
                })

        # Check if consolidation is needed
        if len(self.wisdom_fragments) > 50:
            analysis["consolidation_needed"] = True

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Prepare wisdom updates."""
        synthesis = {
            "store_operations": [],
            "transfer_operations": [],
            "collective_memory_updates": [],
            "consolidation_operations": []
        }

        # Prepare storage operations
        for fragment in analysis.get("fragments_to_store", []):
            synthesis["store_operations"].append({
                "fragment": fragment,
                "index_domains": fragment.applicability
            })

        # Prepare transfer operations
        for transfer in analysis.get("knowledge_transfers", []):
            synthesis["transfer_operations"].append({
                "fragment_id": transfer["from_fragment"],
                "target_domain": transfer["to_domain"],
                "blend_factor": 0.2 * transfer["confidence"]
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
                memory = CollectiveMemory(
                    memory_id=f"mem_{plan['content_type']}_{datetime.now().strftime('%H%M%S')}",
                    fragments=plan["fragments"],
                    summary_vector=summary,
                    domain_weights={d: 1.0/3 for d in DomainLeg}
                )
                synthesis["collective_memory_updates"].append(memory)

        # Prepare consolidation
        if analysis.get("consolidation_needed"):
            # Mark old, low-access fragments for removal
            for fid, frag in list(self.wisdom_fragments.items()):
                age_hours = (datetime.now() - frag.creation_time).total_seconds() / 3600
                if age_hours > 1 and frag.access_count < 2:
                    synthesis["consolidation_operations"].append({
                        "action": "remove",
                        "fragment_id": fid
                    })

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

            # Update domain contributions
            self.domain_contributions[fragment.source_domain] += fragment.knowledge_vector * 0.1

            stored_count += 1

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
                fragment.last_accessed = datetime.now()

                # Apply knowledge to target domain
                state = tbl.get_state(target_domain)
                knowledge = fragment.knowledge_vector[:64]
                state.state_vector = (1 - op["blend_factor"]) * state.state_vector + \
                                    op["blend_factor"] * knowledge

                transfer_count += 1
                self.knowledge_transfers.append({
                    "from": fragment.source_domain.value,
                    "to": target_domain.value,
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
                del self.wisdom_fragments[op["fragment_id"]]
                removed_count += 1

        if removed_count > 0:
            changes["fragments_consolidated"] = removed_count
            insights.append(f"Consolidated {removed_count} old fragments")

        # Update collective knowledge
        all_contributions = np.stack(list(self.domain_contributions.values()))
        self.collective_knowledge = np.mean(all_contributions, axis=0)

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["total_fragments"] = len(self.wisdom_fragments)
        metrics_delta["total_memories"] = len(self.collective_memories)

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
