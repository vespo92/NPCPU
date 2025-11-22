"""
SemanticBridgeAgent - Bridges semantic understanding between domains.

Responsibilities:
- Translate concepts between NPCPU (digital), ChicagoForest (network),
  and UniversalParts (physical) semantic spaces
- Maintain cross-domain ontologies
- Resolve semantic conflicts and ambiguities
- Enable meaningful communication between heterogeneous systems
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
)


@dataclass
class SemanticConcept:
    """A concept with domain-specific representations."""
    concept_id: str
    universal_name: str
    domain_representations: Dict[DomainLeg, str]
    embedding: np.ndarray
    confidence: float = 1.0
    usage_count: int = 0


@dataclass
class SemanticMapping:
    """A mapping between concepts in different domains."""
    source_domain: DomainLeg
    target_domain: DomainLeg
    source_concept: str
    target_concept: str
    translation_confidence: float
    context_requirements: List[str] = field(default_factory=list)


class SemanticBridgeAgent(TertiaryReBoAgent):
    """
    Agent 6: Bridges semantic understanding between domains.

    Each domain has its own "language" and concepts:
    - NPCPU speaks of consciousness, qualia, beliefs, philosophical stances
    - ChicagoForest speaks of nodes, routes, nutrients, spores, mycelium
    - UniversalParts speaks of stress, wear, mating, cycles, failure

    This agent creates and maintains translations so that meaningful
    information can flow across domain boundaries. For example:
    - "Consciousness level" (NPCPU) ↔ "Node activity" (CF) ↔ "Part awareness" (UP)
    - "Energy flow" (NPCPU) ↔ "Nutrient distribution" (CF) ↔ "Power consumption" (UP)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Semantic concept registry
        self.concepts: Dict[str, SemanticConcept] = {}
        self.mappings: List[SemanticMapping] = []

        # Domain vocabularies
        self.domain_vocab: Dict[DomainLeg, Set[str]] = {
            DomainLeg.NPCPU: {
                "consciousness", "qualia", "belief", "memory", "attention",
                "coherence", "emergence", "philosophical_stance", "introspection",
                "meta_cognition", "awareness", "perception", "intention"
            },
            DomainLeg.CHICAGO_FOREST: {
                "node", "route", "nutrient", "spore", "mycelium", "hyphae",
                "connectivity", "propagation", "mesh", "bandwidth", "latency",
                "topology", "broadcast", "peer"
            },
            DomainLeg.UNIVERSAL_PARTS: {
                "stress", "wear", "mating", "cycles", "failure", "torque",
                "temperature", "vibration", "degradation", "assembly",
                "component", "sensor", "actuator", "tolerance"
            }
        }

        # Embedding space for semantic similarity
        self.embedding_dim = 32
        self.concept_embeddings: Dict[str, np.ndarray] = {}

        # Translation matrices between domains
        self.translation_matrices = {
            (DomainLeg.NPCPU, DomainLeg.CHICAGO_FOREST): np.random.randn(32, 32) * 0.1,
            (DomainLeg.CHICAGO_FOREST, DomainLeg.UNIVERSAL_PARTS): np.random.randn(32, 32) * 0.1,
            (DomainLeg.NPCPU, DomainLeg.UNIVERSAL_PARTS): np.random.randn(32, 32) * 0.1,
        }

        # Initialize core concept mappings
        self._initialize_core_mappings()

    def _initialize_core_mappings(self):
        """Set up foundational cross-domain concept mappings."""
        core_mappings = [
            # Consciousness ↔ Activity ↔ Awareness
            ("consciousness", {
                DomainLeg.NPCPU: "consciousness_level",
                DomainLeg.CHICAGO_FOREST: "node_activity",
                DomainLeg.UNIVERSAL_PARTS: "part_awareness"
            }),
            # Energy ↔ Nutrients ↔ Power
            ("energy", {
                DomainLeg.NPCPU: "computational_energy",
                DomainLeg.CHICAGO_FOREST: "nutrient_flow",
                DomainLeg.UNIVERSAL_PARTS: "power_consumption"
            }),
            # Communication ↔ Signal ↔ Interaction
            ("communication", {
                DomainLeg.NPCPU: "message_passing",
                DomainLeg.CHICAGO_FOREST: "signal_propagation",
                DomainLeg.UNIVERSAL_PARTS: "mechanical_coupling"
            }),
            # Health ↔ Connectivity ↔ Integrity
            ("health", {
                DomainLeg.NPCPU: "coherence",
                DomainLeg.CHICAGO_FOREST: "connectivity_score",
                DomainLeg.UNIVERSAL_PARTS: "structural_integrity"
            }),
            # Learning ↔ Adaptation ↔ Wear
            ("adaptation", {
                DomainLeg.NPCPU: "learning_rate",
                DomainLeg.CHICAGO_FOREST: "route_optimization",
                DomainLeg.UNIVERSAL_PARTS: "wear_pattern"
            }),
        ]

        for universal_name, domain_reps in core_mappings:
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)

            concept = SemanticConcept(
                concept_id=f"core_{universal_name}",
                universal_name=universal_name,
                domain_representations=domain_reps,
                embedding=embedding,
                confidence=1.0
            )
            self.concepts[concept.concept_id] = concept

            # Create pairwise mappings
            for source_domain in DomainLeg:
                for target_domain in DomainLeg:
                    if source_domain != target_domain:
                        mapping = SemanticMapping(
                            source_domain=source_domain,
                            target_domain=target_domain,
                            source_concept=domain_reps[source_domain],
                            target_concept=domain_reps[target_domain],
                            translation_confidence=0.9
                        )
                        self.mappings.append(mapping)

    @property
    def agent_role(self) -> str:
        return "Semantic Bridge - Translates meaning between Mind, Network, and Body domains"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather semantic state from all domains."""
        perceptions = {
            "domain_semantic_states": {},
            "cross_domain_coherence": 0.0,
            "untranslated_concepts": [],
            "semantic_conflicts": []
        }

        # Extract semantic indicators from each domain
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_semantic_states"][domain.value] = {
                "state_vector": state.state_vector,
                "active_concepts": self._extract_active_concepts(domain, state.state_vector),
                "semantic_density": len(self.domain_vocab[domain])
            }

        # Check cross-domain semantic coherence
        coherence = self._measure_semantic_coherence(tbl)
        perceptions["cross_domain_coherence"] = coherence

        # Identify concepts that lack cross-domain mappings
        for domain in DomainLeg:
            for term in self.domain_vocab[domain]:
                has_mapping = any(
                    m.source_concept == term or m.target_concept == term
                    for m in self.mappings
                )
                if not has_mapping:
                    perceptions["untranslated_concepts"].append({
                        "domain": domain.value,
                        "term": term
                    })

        # Detect semantic conflicts (same concept, different meanings)
        perceptions["semantic_conflicts"] = self._detect_conflicts()

        return perceptions

    def _extract_active_concepts(self, domain: DomainLeg, state_vector: np.ndarray) -> List[str]:
        """Determine which concepts are currently "active" in a domain."""
        active = []
        vocab_list = list(self.domain_vocab[domain])

        # Use state vector components to simulate concept activation
        for i, term in enumerate(vocab_list[:len(state_vector) // 5]):
            activation = state_vector[i * 5]  # Sample from state vector
            if activation > 0.3:
                active.append(term)

        return active

    def _measure_semantic_coherence(self, tbl: TripleBottomLine) -> float:
        """Measure how well semantic concepts align across domains."""
        vectors = [tbl.get_state(d).state_vector for d in DomainLeg]

        # Apply translation matrices and measure alignment
        alignments = []

        for (src, tgt), matrix in self.translation_matrices.items():
            src_idx = list(DomainLeg).index(src)
            tgt_idx = list(DomainLeg).index(tgt)

            src_vec = vectors[src_idx][:32]
            tgt_vec = vectors[tgt_idx][:32]

            translated = matrix @ src_vec
            # Measure how well translation matches target
            alignment = np.dot(translated, tgt_vec) / (
                np.linalg.norm(translated) * np.linalg.norm(tgt_vec) + 1e-6
            )
            alignments.append((alignment + 1) / 2)  # Normalize to [0, 1]

        return float(np.mean(alignments)) if alignments else 0.5

    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect semantic conflicts in the mapping space."""
        conflicts = []

        # Check for mappings that contradict each other
        for i, m1 in enumerate(self.mappings):
            for m2 in self.mappings[i+1:]:
                if (m1.source_domain == m2.source_domain and
                    m1.source_concept == m2.source_concept and
                    m1.target_domain == m2.target_domain and
                    m1.target_concept != m2.target_concept):

                    conflicts.append({
                        "type": "ambiguous_translation",
                        "source": m1.source_concept,
                        "targets": [m1.target_concept, m2.target_concept],
                        "domain_pair": (m1.source_domain.value, m1.target_domain.value)
                    })

        return conflicts[:10]  # Limit to top 10

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze semantic gaps and opportunities."""
        analysis = {
            "semantic_health": "good",
            "bridge_opportunities": [],
            "conflict_resolutions": [],
            "vocabulary_expansions": []
        }

        # Assess overall semantic health
        coherence = perception["cross_domain_coherence"]
        if coherence < 0.3:
            analysis["semantic_health"] = "poor"
        elif coherence < 0.6:
            analysis["semantic_health"] = "fair"

        # Identify bridging opportunities for untranslated concepts
        for untranslated in perception.get("untranslated_concepts", [])[:10]:
            domain = DomainLeg(untranslated["domain"])
            term = untranslated["term"]

            # Find potential translations by semantic similarity
            candidates = self._find_translation_candidates(domain, term)
            if candidates:
                analysis["bridge_opportunities"].append({
                    "source_domain": domain.value,
                    "term": term,
                    "candidates": candidates
                })

        # Plan conflict resolutions
        for conflict in perception.get("semantic_conflicts", []):
            analysis["conflict_resolutions"].append({
                "conflict": conflict,
                "strategy": "merge_meanings",
                "confidence_adjustment": -0.1
            })

        # Suggest vocabulary expansions
        for domain in DomainLeg:
            active = perception["domain_semantic_states"][domain.value]["active_concepts"]
            # Look for combinations that might represent new concepts
            if len(active) >= 2:
                analysis["vocabulary_expansions"].append({
                    "domain": domain.value,
                    "base_concepts": active[:2],
                    "potential_compound": f"{active[0]}_{active[1]}"
                })

        return analysis

    def _find_translation_candidates(
        self, source_domain: DomainLeg, term: str
    ) -> List[Dict[str, Any]]:
        """Find potential translation candidates for a term."""
        candidates = []

        # Simple heuristic: look for similar-sounding or semantically related terms
        for target_domain in DomainLeg:
            if target_domain == source_domain:
                continue

            for target_term in self.domain_vocab[target_domain]:
                # Check for substring matches or semantic relatedness
                similarity = self._term_similarity(term, target_term)
                if similarity > 0.3:
                    candidates.append({
                        "target_domain": target_domain.value,
                        "target_term": target_term,
                        "similarity": similarity
                    })

        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:3]

    def _term_similarity(self, term1: str, term2: str) -> float:
        """Compute similarity between two terms."""
        # Simple character-based similarity
        set1 = set(term1.lower())
        set2 = set(term2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Generate semantic bridge refinements."""
        synthesis = {
            "new_mappings": [],
            "mapping_updates": [],
            "translation_matrix_updates": [],
            "new_concepts": []
        }

        # Create new mappings for bridging opportunities
        for opportunity in analysis.get("bridge_opportunities", []):
            if opportunity["candidates"]:
                best_candidate = opportunity["candidates"][0]
                synthesis["new_mappings"].append({
                    "source_domain": opportunity["source_domain"],
                    "target_domain": best_candidate["target_domain"],
                    "source_concept": opportunity["term"],
                    "target_concept": best_candidate["target_term"],
                    "confidence": best_candidate["similarity"]
                })

        # Apply conflict resolutions
        for resolution in analysis.get("conflict_resolutions", []):
            conflict = resolution["conflict"]
            synthesis["mapping_updates"].append({
                "action": "reduce_confidence",
                "affected_mappings": conflict["targets"],
                "adjustment": resolution["confidence_adjustment"]
            })

        # Update translation matrices based on current alignment
        for (src, tgt) in self.translation_matrices.keys():
            src_vec = tbl.get_state(src).state_vector[:32]
            tgt_vec = tbl.get_state(tgt).state_vector[:32]

            # Compute gradient for better alignment
            outer_product = np.outer(src_vec, tgt_vec)
            synthesis["translation_matrix_updates"].append({
                "pair": (src.value, tgt.value),
                "gradient": outer_product * 0.01
            })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply semantic bridge updates."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Add new mappings
        new_mapping_count = 0
        for mapping_data in synthesis.get("new_mappings", []):
            mapping = SemanticMapping(
                source_domain=DomainLeg(mapping_data["source_domain"]),
                target_domain=DomainLeg(mapping_data["target_domain"]),
                source_concept=mapping_data["source_concept"],
                target_concept=mapping_data["target_concept"],
                translation_confidence=mapping_data["confidence"]
            )
            self.mappings.append(mapping)
            new_mapping_count += 1
            insights.append(
                f"Bridged '{mapping_data['source_concept']}' ({mapping_data['source_domain']}) "
                f"↔ '{mapping_data['target_concept']}' ({mapping_data['target_domain']})"
            )

        if new_mapping_count > 0:
            changes["new_mappings"] = new_mapping_count

        # Update translation matrices
        matrix_updates = 0
        for update in synthesis.get("translation_matrix_updates", []):
            pair_key = (DomainLeg(update["pair"][0]), DomainLeg(update["pair"][1]))
            if pair_key in self.translation_matrices:
                self.translation_matrices[pair_key] += update["gradient"]
                # Normalize
                norm = np.linalg.norm(self.translation_matrices[pair_key])
                if norm > 0:
                    self.translation_matrices[pair_key] /= norm
                matrix_updates += 1

        if matrix_updates > 0:
            changes["translation_updates"] = matrix_updates
            insights.append(f"Refined {matrix_updates} cross-domain translation matrices")

        # Measure new coherence
        new_coherence = self._measure_semantic_coherence(tbl)
        metrics_delta["semantic_coherence"] = new_coherence

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
