"""
SemanticBridgeAgent - Bridges semantic understanding between domains.

Responsibilities:
- Translate concepts between NPCPU (digital), ChicagoForest (network),
  and UniversalParts (physical) semantic spaces
- Maintain cross-domain ontologies with real semantic embeddings
- Resolve semantic conflicts and ambiguities using neural similarity
- Enable meaningful communication between heterogeneous systems
- Discover emergent concepts from cross-domain patterns
- Support multi-hop translation paths for complex concepts
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal,
)


@dataclass
class SemanticConcept:
    """A concept with domain-specific representations and neural embeddings."""
    concept_id: str
    universal_name: str
    domain_representations: Dict[DomainLeg, str]
    embedding: np.ndarray
    confidence: float = 1.0
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    emergence_source: Optional[str] = None  # If discovered from cross-domain patterns
    context_vectors: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SemanticMapping:
    """A mapping between concepts in different domains."""
    mapping_id: str
    source_domain: DomainLeg
    target_domain: DomainLeg
    source_concept: str
    target_concept: str
    translation_confidence: float
    context_requirements: List[str] = field(default_factory=list)
    bidirectional: bool = True
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class TranslationPath:
    """A multi-hop translation path between concepts."""
    path_id: str
    source_domain: DomainLeg
    target_domain: DomainLeg
    hops: List[Tuple[DomainLeg, str]]  # List of (domain, concept) pairs
    total_confidence: float
    path_length: int


@dataclass
class SemanticConflict:
    """A detected conflict in semantic space."""
    conflict_id: str
    conflict_type: str  # ambiguous, contradictory, polysemous
    concepts_involved: List[str]
    domains_involved: List[DomainLeg]
    severity: float  # 0-1, higher = more severe
    resolution_candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EmergentConcept:
    """A concept discovered from cross-domain pattern analysis."""
    concept_id: str
    source_patterns: List[Dict[str, Any]]
    proposed_name: str
    domain_manifestations: Dict[DomainLeg, str]
    confidence: float
    discovery_timestamp: datetime = field(default_factory=datetime.now)


class SemanticEmbedder:
    """
    Neural semantic embedder using lightweight embedding techniques.
    Falls back to heuristic embeddings if transformers unavailable.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.use_neural = False
        self.model = None
        self._try_initialize_neural()
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def _try_initialize_neural(self):
        """Try to initialize neural embeddings, fall back to heuristic if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.use_neural = True
        except (ImportError, Exception):
            self.use_neural = False

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.use_neural and self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._heuristic_embed(text)

        self._embedding_cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if self.use_neural and self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)
        return np.array([self._heuristic_embed(t) for t in texts])

    def _heuristic_embed(self, text: str) -> np.ndarray:
        """Generate heuristic embedding based on character/word features."""
        # Use hash-based features for reproducibility
        text_lower = text.lower()
        words = text_lower.replace("_", " ").split()

        # Character n-gram features
        ngrams = set()
        for n in range(2, 4):
            for i in range(len(text_lower) - n + 1):
                ngrams.add(text_lower[i:i+n])

        # Word-level features
        word_hashes = [int(hashlib.md5(w.encode()).hexdigest()[:8], 16) for w in words]
        ngram_hashes = [int(hashlib.md5(ng.encode()).hexdigest()[:8], 16) for ng in ngrams]

        # Build deterministic embedding
        np.random.seed(sum(word_hashes + ngram_hashes) % (2**31))
        embedding = np.random.randn(self.embedding_dim)

        # Add semantic features
        semantic_markers = {
            "conscious": 0, "aware": 1, "think": 2, "memory": 3,
            "node": 10, "network": 11, "route": 12, "connect": 13,
            "part": 20, "wear": 21, "stress": 22, "physical": 23,
            "energy": 30, "flow": 31, "transfer": 32,
            "signal": 40, "propagate": 41, "broadcast": 42
        }

        for word in words:
            for marker, idx in semantic_markers.items():
                if marker in word:
                    embedding[idx * 10:(idx + 1) * 10] += 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class SemanticBridgeAgent(TertiaryReBoAgent):
    """
    Agent 6: Bridges semantic understanding between domains.

    Each domain has its own "language" and concepts:
    - NPCPU speaks of consciousness, qualia, beliefs, philosophical stances
    - ChicagoForest speaks of nodes, routes, nutrients, spores, mycelium
    - UniversalParts speaks of stress, wear, mating, cycles, failure

    This agent creates and maintains translations so that meaningful
    information can flow across domain boundaries. It uses neural embeddings
    for semantic similarity, supports multi-hop translations, and can
    discover emergent concepts from cross-domain patterns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Neural embedder for semantic similarity
        self.embedder = SemanticEmbedder()
        self.embedding_dim = self.embedder.embedding_dim

        # Semantic concept registry
        self.concepts: Dict[str, SemanticConcept] = {}
        self.mappings: Dict[str, SemanticMapping] = {}
        self.translation_paths: Dict[str, TranslationPath] = {}

        # Emergent concept tracking
        self.emergent_concepts: List[EmergentConcept] = []
        self.emergence_threshold: float = 0.7

        # Conflict tracking
        self.active_conflicts: List[SemanticConflict] = []

        # Domain vocabularies - expanded with rich semantic terms
        self.domain_vocab: Dict[DomainLeg, Set[str]] = {
            DomainLeg.NPCPU: {
                # Core consciousness
                "consciousness", "qualia", "awareness", "perception", "cognition",
                # Mental processes
                "belief", "memory", "attention", "intention", "expectation",
                "reasoning", "inference", "abstraction", "imagination",
                # Meta-cognition
                "introspection", "meta_cognition", "self_model", "reflection",
                # States
                "coherence", "emergence", "philosophical_stance", "mental_state",
                "arousal", "valence", "engagement", "focus",
                # Learning
                "learning", "adaptation", "plasticity", "consolidation",
                # Communication
                "message", "signal", "broadcast", "query", "response"
            },
            DomainLeg.CHICAGO_FOREST: {
                # Network topology
                "node", "edge", "route", "path", "topology", "mesh",
                "hub", "spoke", "cluster", "subnetwork",
                # Biological network metaphors
                "mycelium", "hyphae", "spore", "fruiting_body",
                "nutrient", "symbiosis", "decomposition",
                # Communication
                "signal_propagation", "broadcast", "unicast", "multicast",
                "packet", "bandwidth", "latency", "throughput",
                # Health
                "connectivity", "resilience", "redundancy", "failover",
                "partition", "healing", "growth",
                # Resources
                "nutrient_flow", "resource_allocation", "load_balance"
            },
            DomainLeg.UNIVERSAL_PARTS: {
                # Physical properties
                "stress", "strain", "torque", "force", "pressure",
                "temperature", "vibration", "friction",
                # Wear and degradation
                "wear", "degradation", "fatigue", "corrosion", "erosion",
                "failure_mode", "lifetime", "cycles",
                # Assembly
                "component", "assembly", "mating", "tolerance", "fit",
                "clearance", "interference",
                # Sensing
                "sensor", "actuator", "feedback", "calibration",
                # Physical state
                "structural_integrity", "alignment", "balance",
                "part_awareness", "physical_state", "operational_status"
            }
        }

        # Domain embeddings cache
        self._domain_term_embeddings: Dict[DomainLeg, Dict[str, np.ndarray]] = {}

        # Translation matrices (learned alignment between domains)
        self.translation_matrices: Dict[Tuple[DomainLeg, DomainLeg], np.ndarray] = {}

        # Cross-domain concept graphs for multi-hop translation
        self.concept_graph: Dict[str, Set[str]] = defaultdict(set)

        # Initialize core mappings and embeddings
        self._initialize_domain_embeddings()
        self._initialize_core_mappings()
        self._initialize_translation_matrices()
        self._build_concept_graph()

    def _initialize_domain_embeddings(self):
        """Pre-compute embeddings for all domain vocabulary terms."""
        for domain in DomainLeg:
            self._domain_term_embeddings[domain] = {}
            for term in self.domain_vocab[domain]:
                # Create contextual term by adding domain prefix
                contextual_term = f"{domain.value} domain concept: {term.replace('_', ' ')}"
                embedding = self.embedder.embed(contextual_term)
                self._domain_term_embeddings[domain][term] = embedding

    def _initialize_core_mappings(self):
        """Set up foundational cross-domain concept mappings with neural embeddings."""
        core_mappings = [
            # Consciousness ↔ Activity ↔ Awareness
            ("consciousness", {
                DomainLeg.NPCPU: "consciousness",
                DomainLeg.CHICAGO_FOREST: "node",
                DomainLeg.UNIVERSAL_PARTS: "part_awareness"
            }, ["state", "level", "active"]),

            # Energy/Resource flow
            ("energy_flow", {
                DomainLeg.NPCPU: "attention",
                DomainLeg.CHICAGO_FOREST: "nutrient_flow",
                DomainLeg.UNIVERSAL_PARTS: "force"
            }, ["transfer", "distribution", "consumption"]),

            # Communication/Signal
            ("communication", {
                DomainLeg.NPCPU: "message",
                DomainLeg.CHICAGO_FOREST: "signal_propagation",
                DomainLeg.UNIVERSAL_PARTS: "sensor"
            }, ["send", "receive", "propagate"]),

            # Health/Integrity
            ("health", {
                DomainLeg.NPCPU: "coherence",
                DomainLeg.CHICAGO_FOREST: "connectivity",
                DomainLeg.UNIVERSAL_PARTS: "structural_integrity"
            }, ["status", "quality", "degradation"]),

            # Adaptation/Learning
            ("adaptation", {
                DomainLeg.NPCPU: "learning",
                DomainLeg.CHICAGO_FOREST: "growth",
                DomainLeg.UNIVERSAL_PARTS: "wear"
            }, ["change", "adjust", "evolve"]),

            # Memory/Storage
            ("persistence", {
                DomainLeg.NPCPU: "memory",
                DomainLeg.CHICAGO_FOREST: "mycelium",
                DomainLeg.UNIVERSAL_PARTS: "component"
            }, ["store", "retain", "history"]),

            # Structure/Topology
            ("structure", {
                DomainLeg.NPCPU: "self_model",
                DomainLeg.CHICAGO_FOREST: "topology",
                DomainLeg.UNIVERSAL_PARTS: "assembly"
            }, ["organization", "architecture", "hierarchy"]),

            # Processing/Transformation
            ("processing", {
                DomainLeg.NPCPU: "reasoning",
                DomainLeg.CHICAGO_FOREST: "path",
                DomainLeg.UNIVERSAL_PARTS: "actuator"
            }, ["transform", "compute", "act"]),

            # Resilience/Robustness
            ("resilience", {
                DomainLeg.NPCPU: "plasticity",
                DomainLeg.CHICAGO_FOREST: "redundancy",
                DomainLeg.UNIVERSAL_PARTS: "tolerance"
            }, ["recover", "fault", "robust"]),

            # Emergence/Novelty
            ("emergence", {
                DomainLeg.NPCPU: "emergence",
                DomainLeg.CHICAGO_FOREST: "fruiting_body",
                DomainLeg.UNIVERSAL_PARTS: "failure_mode"
            }, ["novel", "unexpected", "threshold"]),

            # Focus/Attention
            ("focus", {
                DomainLeg.NPCPU: "focus",
                DomainLeg.CHICAGO_FOREST: "hub",
                DomainLeg.UNIVERSAL_PARTS: "calibration"
            }, ["concentrate", "center", "priority"]),

            # Distribution/Spread
            ("distribution", {
                DomainLeg.NPCPU: "broadcast",
                DomainLeg.CHICAGO_FOREST: "multicast",
                DomainLeg.UNIVERSAL_PARTS: "stress"
            }, ["spread", "distribute", "disperse"]),

            # State/Condition
            ("state", {
                DomainLeg.NPCPU: "mental_state",
                DomainLeg.CHICAGO_FOREST: "cluster",
                DomainLeg.UNIVERSAL_PARTS: "physical_state"
            }, ["condition", "mode", "phase"]),

            # Boundary/Interface
            ("boundary", {
                DomainLeg.NPCPU: "perception",
                DomainLeg.CHICAGO_FOREST: "edge",
                DomainLeg.UNIVERSAL_PARTS: "mating"
            }, ["interface", "connection", "transition"]),

            # Feedback/Response
            ("feedback", {
                DomainLeg.NPCPU: "response",
                DomainLeg.CHICAGO_FOREST: "latency",
                DomainLeg.UNIVERSAL_PARTS: "feedback"
            }, ["react", "adjust", "loop"])
        ]

        for universal_name, domain_reps, contexts in core_mappings:
            # Generate embedding from universal name and contexts
            embedding_text = f"{universal_name}: {' '.join(contexts)}"
            embedding = self.embedder.embed(embedding_text)

            concept = SemanticConcept(
                concept_id=f"core_{universal_name}",
                universal_name=universal_name,
                domain_representations=domain_reps,
                embedding=embedding,
                confidence=1.0
            )

            # Add context vectors for each context word
            for ctx in contexts:
                concept.context_vectors[ctx] = self.embedder.embed(ctx)

            self.concepts[concept.concept_id] = concept

            # Create pairwise mappings
            for source_domain in DomainLeg:
                for target_domain in DomainLeg:
                    if source_domain != target_domain:
                        mapping_id = f"map_{universal_name}_{source_domain.value}_{target_domain.value}"
                        mapping = SemanticMapping(
                            mapping_id=mapping_id,
                            source_domain=source_domain,
                            target_domain=target_domain,
                            source_concept=domain_reps[source_domain],
                            target_concept=domain_reps[target_domain],
                            translation_confidence=0.95,
                            context_requirements=contexts
                        )
                        self.mappings[mapping_id] = mapping

    def _initialize_translation_matrices(self):
        """Initialize learnable translation matrices between domain embedding spaces."""
        for src in DomainLeg:
            for tgt in DomainLeg:
                if src != tgt:
                    # Initialize with identity-like transformation
                    matrix = np.eye(self.embedding_dim) + np.random.randn(
                        self.embedding_dim, self.embedding_dim
                    ) * 0.01
                    self.translation_matrices[(src, tgt)] = matrix

    def _build_concept_graph(self):
        """Build a graph connecting related concepts across domains."""
        # Connect concepts through their universal mappings
        for concept in self.concepts.values():
            universal_id = concept.concept_id
            for domain, rep in concept.domain_representations.items():
                domain_concept_id = f"{domain.value}:{rep}"
                self.concept_graph[universal_id].add(domain_concept_id)
                self.concept_graph[domain_concept_id].add(universal_id)

                # Also connect domain concepts that share a universal concept
                for other_domain, other_rep in concept.domain_representations.items():
                    if domain != other_domain:
                        other_id = f"{other_domain.value}:{other_rep}"
                        self.concept_graph[domain_concept_id].add(other_id)

    @property
    def agent_role(self) -> str:
        return "Semantic Bridge - Translates meaning between Mind, Network, and Body domains using neural embeddings"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather semantic state from all domains using neural analysis."""
        perceptions = {
            "domain_semantic_states": {},
            "cross_domain_coherence": 0.0,
            "untranslated_concepts": [],
            "semantic_conflicts": [],
            "active_concept_clusters": {},
            "translation_usage_stats": {},
            "emergent_pattern_candidates": []
        }

        # Extract semantic indicators from each domain
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            active_concepts = self._extract_active_concepts(domain, state.state_vector)

            # Compute semantic density as coverage of vocabulary
            translated_count = sum(
                1 for term in self.domain_vocab[domain]
                if any(m.source_concept == term or m.target_concept == term
                       for m in self.mappings.values())
            )
            semantic_coverage = translated_count / len(self.domain_vocab[domain])

            perceptions["domain_semantic_states"][domain.value] = {
                "state_vector": state.state_vector,
                "active_concepts": active_concepts,
                "semantic_coverage": semantic_coverage,
                "vocabulary_size": len(self.domain_vocab[domain]),
                "domain_coherence": state.coherence
            }

            # Find concepts that cluster together (potential emergent patterns)
            if len(active_concepts) >= 3:
                clusters = self._cluster_active_concepts(domain, active_concepts)
                perceptions["active_concept_clusters"][domain.value] = clusters

        # Check cross-domain semantic coherence using neural similarity
        coherence = await self._measure_semantic_coherence(tbl)
        perceptions["cross_domain_coherence"] = coherence

        # Identify concepts that lack cross-domain mappings
        perceptions["untranslated_concepts"] = self._find_untranslated_concepts()

        # Detect semantic conflicts
        perceptions["semantic_conflicts"] = self._detect_conflicts()

        # Look for emergent pattern candidates
        perceptions["emergent_pattern_candidates"] = await self._detect_emergent_patterns(
            perceptions["active_concept_clusters"]
        )

        # Track translation usage
        perceptions["translation_usage_stats"] = {
            mapping_id: {
                "usage_count": m.usage_count,
                "success_rate": m.success_rate
            }
            for mapping_id, m in self.mappings.items()
        }

        return perceptions

    def _extract_active_concepts(self, domain: DomainLeg, state_vector: np.ndarray) -> List[Dict[str, Any]]:
        """Determine which concepts are currently "active" using embedding similarity."""
        active = []
        vocab_list = list(self.domain_vocab[domain])

        # Use first portion of state vector as activation pattern
        activation_pattern = state_vector[:min(64, len(state_vector))]
        activation_pattern = activation_pattern / (np.linalg.norm(activation_pattern) + 1e-6)

        for term in vocab_list:
            if domain in self._domain_term_embeddings and term in self._domain_term_embeddings[domain]:
                term_emb = self._domain_term_embeddings[domain][term][:len(activation_pattern)]

                # Compute activation score
                activation = np.dot(activation_pattern, term_emb[:len(activation_pattern)])
                activation = (activation + 1) / 2  # Normalize to [0, 1]

                if activation > 0.3:
                    active.append({
                        "term": term,
                        "activation": float(activation),
                        "domain": domain.value
                    })

        # Sort by activation and return top concepts
        active.sort(key=lambda x: x["activation"], reverse=True)
        return active[:15]

    def _cluster_active_concepts(self, domain: DomainLeg, active_concepts: List[Dict[str, Any]]) -> List[List[str]]:
        """Cluster active concepts by semantic similarity."""
        if len(active_concepts) < 2:
            return [[c["term"] for c in active_concepts]]

        terms = [c["term"] for c in active_concepts]
        embeddings = [self._domain_term_embeddings[domain].get(
            t, self.embedder.embed(t)
        ) for t in terms]

        # Simple hierarchical clustering based on similarity
        clusters = []
        used = set()

        for i, (term, emb) in enumerate(zip(terms, embeddings)):
            if term in used:
                continue

            cluster = [term]
            used.add(term)

            for j, (other_term, other_emb) in enumerate(zip(terms, embeddings)):
                if j <= i or other_term in used:
                    continue

                sim = self.embedder.similarity(emb, other_emb)
                if sim > 0.6:
                    cluster.append(other_term)
                    used.add(other_term)

            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    async def _measure_semantic_coherence(self, tbl: TripleBottomLine) -> float:
        """Measure semantic coherence across domains using neural embeddings."""
        domain_embeddings = []

        for domain in DomainLeg:
            state = tbl.get_state(domain)
            # Create domain summary embedding from active concepts
            active = self._extract_active_concepts(domain, state.state_vector)

            if active:
                concept_embeddings = [
                    self._domain_term_embeddings[domain].get(c["term"], np.zeros(self.embedding_dim))
                    for c in active[:5]
                ]
                domain_emb = np.mean(concept_embeddings, axis=0)
            else:
                domain_emb = np.zeros(self.embedding_dim)

            domain_embeddings.append(domain_emb)

        # Compute pairwise similarities
        similarities = []
        for i, emb1 in enumerate(domain_embeddings):
            for emb2 in domain_embeddings[i+1:]:
                sim = self.embedder.similarity(emb1, emb2)
                similarities.append((sim + 1) / 2)  # Normalize to [0, 1]

        # Also check translation matrix quality
        matrix_coherence = []
        for (src, tgt), matrix in self.translation_matrices.items():
            # Check if matrix preserves norms (good translation matrices should)
            test_vec = np.random.randn(self.embedding_dim)
            test_vec = test_vec / np.linalg.norm(test_vec)
            transformed = matrix @ test_vec
            norm_preservation = 1.0 - abs(1.0 - np.linalg.norm(transformed))
            matrix_coherence.append(max(0, norm_preservation))

        avg_similarity = np.mean(similarities) if similarities else 0.5
        avg_matrix = np.mean(matrix_coherence) if matrix_coherence else 0.5

        return float(avg_similarity * 0.7 + avg_matrix * 0.3)

    def _find_untranslated_concepts(self) -> List[Dict[str, Any]]:
        """Find concepts without cross-domain translations."""
        untranslated = []

        for domain in DomainLeg:
            for term in self.domain_vocab[domain]:
                has_mapping = any(
                    m.source_concept == term or m.target_concept == term
                    for m in self.mappings.values()
                )
                if not has_mapping:
                    # Find closest existing concept for translation candidate
                    term_emb = self._domain_term_embeddings[domain].get(
                        term, self.embedder.embed(term)
                    )

                    best_match = None
                    best_sim = 0.0

                    for concept in self.concepts.values():
                        sim = self.embedder.similarity(term_emb, concept.embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_match = concept.universal_name

                    untranslated.append({
                        "domain": domain.value,
                        "term": term,
                        "closest_universal": best_match,
                        "similarity": float(best_sim)
                    })

        return sorted(untranslated, key=lambda x: x["similarity"], reverse=True)[:20]

    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect semantic conflicts in the mapping space."""
        conflicts = []

        # Check for ambiguous translations (one source -> multiple targets)
        source_targets: Dict[Tuple[DomainLeg, str, DomainLeg], List[str]] = defaultdict(list)

        for m in self.mappings.values():
            key = (m.source_domain, m.source_concept, m.target_domain)
            source_targets[key].append(m.target_concept)

        for (src_domain, src_concept, tgt_domain), targets in source_targets.items():
            if len(set(targets)) > 1:
                conflicts.append({
                    "conflict_id": f"ambig_{src_concept}_{tgt_domain.value}",
                    "type": "ambiguous_translation",
                    "source": src_concept,
                    "source_domain": src_domain.value,
                    "target_domain": tgt_domain.value,
                    "targets": list(set(targets)),
                    "severity": min(1.0, len(set(targets)) * 0.3)
                })

        # Check for semantic distance conflicts (mapped concepts too different)
        for m in self.mappings.values():
            src_emb = self._domain_term_embeddings.get(m.source_domain, {}).get(
                m.source_concept, self.embedder.embed(m.source_concept)
            )
            tgt_emb = self._domain_term_embeddings.get(m.target_domain, {}).get(
                m.target_concept, self.embedder.embed(m.target_concept)
            )

            sim = self.embedder.similarity(src_emb, tgt_emb)

            if sim < 0.2 and m.translation_confidence > 0.5:
                conflicts.append({
                    "conflict_id": f"dist_{m.mapping_id}",
                    "type": "semantic_distance",
                    "source": m.source_concept,
                    "target": m.target_concept,
                    "similarity": float(sim),
                    "confidence": m.translation_confidence,
                    "severity": 1.0 - sim
                })

        return conflicts[:10]

    async def _detect_emergent_patterns(
        self,
        clusters: Dict[str, List[List[str]]]
    ) -> List[Dict[str, Any]]:
        """Detect potential emergent concepts from cross-domain clusters."""
        emergent_candidates = []

        # Look for similar clusters across different domains
        all_clusters = []
        for domain_name, domain_clusters in clusters.items():
            domain = DomainLeg(domain_name)
            for cluster in domain_clusters:
                if len(cluster) >= 2:
                    # Compute cluster centroid
                    embeddings = [
                        self._domain_term_embeddings[domain].get(t, self.embedder.embed(t))
                        for t in cluster
                    ]
                    centroid = np.mean(embeddings, axis=0)
                    all_clusters.append({
                        "domain": domain,
                        "terms": cluster,
                        "centroid": centroid
                    })

        # Find cross-domain cluster pairs
        for i, c1 in enumerate(all_clusters):
            for c2 in all_clusters[i+1:]:
                if c1["domain"] != c2["domain"]:
                    sim = self.embedder.similarity(c1["centroid"], c2["centroid"])

                    if sim > self.emergence_threshold:
                        # Potential emergent concept connecting these clusters
                        emergent_candidates.append({
                            "domains": [c1["domain"].value, c2["domain"].value],
                            "terms_1": c1["terms"],
                            "terms_2": c2["terms"],
                            "similarity": float(sim),
                            "proposed_name": f"emergent_{c1['terms'][0]}_{c2['terms'][0]}"
                        })

        return emergent_candidates[:5]

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze semantic gaps and opportunities using neural analysis."""
        analysis = {
            "semantic_health": "good",
            "health_score": 0.0,
            "bridge_opportunities": [],
            "conflict_resolutions": [],
            "emergent_concepts": [],
            "translation_improvements": [],
            "vocabulary_expansions": []
        }

        # Assess overall semantic health
        coherence = perception["cross_domain_coherence"]
        coverage_scores = [
            s["semantic_coverage"]
            for s in perception["domain_semantic_states"].values()
        ]
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.5

        health_score = coherence * 0.6 + avg_coverage * 0.4
        analysis["health_score"] = float(health_score)

        if health_score < 0.3:
            analysis["semantic_health"] = "poor"
        elif health_score < 0.5:
            analysis["semantic_health"] = "fair"
        elif health_score < 0.7:
            analysis["semantic_health"] = "good"
        else:
            analysis["semantic_health"] = "excellent"

        # Identify high-priority bridging opportunities
        for untranslated in perception.get("untranslated_concepts", [])[:10]:
            if untranslated["similarity"] > 0.4:
                # Good candidate for automatic mapping
                candidates = await self._find_translation_candidates(
                    DomainLeg(untranslated["domain"]),
                    untranslated["term"]
                )

                if candidates:
                    analysis["bridge_opportunities"].append({
                        "source_domain": untranslated["domain"],
                        "term": untranslated["term"],
                        "candidates": candidates,
                        "universal_anchor": untranslated["closest_universal"],
                        "priority": untranslated["similarity"]
                    })

        # Plan conflict resolutions
        for conflict in perception.get("semantic_conflicts", []):
            resolution = await self._plan_conflict_resolution(conflict)
            if resolution:
                analysis["conflict_resolutions"].append(resolution)

        # Process emergent pattern candidates
        for candidate in perception.get("emergent_pattern_candidates", []):
            if candidate["similarity"] > 0.75:
                emergent = await self._synthesize_emergent_concept(candidate)
                if emergent:
                    analysis["emergent_concepts"].append(emergent)

        # Identify translation matrix improvements
        analysis["translation_improvements"] = await self._analyze_translation_quality(tbl)

        return analysis

    async def _find_translation_candidates(
        self,
        source_domain: DomainLeg,
        term: str
    ) -> List[Dict[str, Any]]:
        """Find potential translation candidates using neural similarity."""
        candidates = []

        source_emb = self._domain_term_embeddings.get(source_domain, {}).get(
            term, self.embedder.embed(f"{source_domain.value}: {term}")
        )

        for target_domain in DomainLeg:
            if target_domain == source_domain:
                continue

            # Try direct embedding similarity
            for target_term in self.domain_vocab[target_domain]:
                target_emb = self._domain_term_embeddings.get(target_domain, {}).get(
                    target_term, self.embedder.embed(f"{target_domain.value}: {target_term}")
                )

                similarity = self.embedder.similarity(source_emb, target_emb)

                if similarity > 0.35:
                    # Also try translation matrix
                    if (source_domain, target_domain) in self.translation_matrices:
                        matrix = self.translation_matrices[(source_domain, target_domain)]
                        translated = matrix @ source_emb
                        translated_sim = self.embedder.similarity(translated, target_emb)
                        combined_sim = (similarity + translated_sim) / 2
                    else:
                        combined_sim = similarity

                    candidates.append({
                        "target_domain": target_domain.value,
                        "target_term": target_term,
                        "direct_similarity": float(similarity),
                        "combined_similarity": float(combined_sim)
                    })

        return sorted(candidates, key=lambda x: x["combined_similarity"], reverse=True)[:5]

    async def _plan_conflict_resolution(self, conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan resolution for a semantic conflict."""
        if conflict["type"] == "ambiguous_translation":
            # Resolve by keeping highest confidence mapping
            return {
                "conflict": conflict,
                "strategy": "select_best",
                "action": "keep_highest_similarity",
                "confidence_adjustment": -0.1
            }
        elif conflict["type"] == "semantic_distance":
            # May need to remove or reduce confidence
            return {
                "conflict": conflict,
                "strategy": "reduce_confidence",
                "action": "lower_translation_confidence",
                "new_confidence": max(0.1, conflict.get("confidence", 0.5) - 0.2)
            }
        return None

    async def _synthesize_emergent_concept(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Synthesize a new concept from emergent patterns."""
        domains = [DomainLeg(d) for d in candidate["domains"]]

        # Create universal embedding from the cluster centroids
        embeddings = []
        for domain in domains:
            terms = candidate.get(f"terms_{domains.index(domain) + 1}", [])
            for term in terms:
                emb = self._domain_term_embeddings.get(domain, {}).get(term)
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            return None

        universal_embedding = np.mean(embeddings, axis=0)

        # Generate proposed domain representations
        domain_reps = {}
        for domain in DomainLeg:
            best_term = None
            best_sim = 0.0

            for term in self.domain_vocab[domain]:
                term_emb = self._domain_term_embeddings.get(domain, {}).get(term)
                if term_emb is not None:
                    sim = self.embedder.similarity(universal_embedding, term_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_term = term

            if best_term:
                domain_reps[domain] = {"term": best_term, "similarity": float(best_sim)}

        return {
            "proposed_name": candidate["proposed_name"],
            "source_domains": candidate["domains"],
            "source_terms": {
                candidate["domains"][0]: candidate["terms_1"],
                candidate["domains"][1]: candidate["terms_2"]
            },
            "domain_representations": domain_reps,
            "confidence": candidate["similarity"],
            "universal_embedding": universal_embedding.tolist()
        }

    async def _analyze_translation_quality(self, tbl: TripleBottomLine) -> List[Dict[str, Any]]:
        """Analyze translation matrix quality and suggest improvements."""
        improvements = []

        for (src, tgt), matrix in self.translation_matrices.items():
            # Test translation quality on known mappings
            test_pairs = [
                (m.source_concept, m.target_concept)
                for m in self.mappings.values()
                if m.source_domain == src and m.target_domain == tgt
            ][:5]

            if not test_pairs:
                continue

            errors = []
            for src_concept, tgt_concept in test_pairs:
                src_emb = self._domain_term_embeddings.get(src, {}).get(
                    src_concept, self.embedder.embed(src_concept)
                )
                tgt_emb = self._domain_term_embeddings.get(tgt, {}).get(
                    tgt_concept, self.embedder.embed(tgt_concept)
                )

                translated = matrix @ src_emb
                error = np.linalg.norm(translated - tgt_emb)
                errors.append(error)

            avg_error = np.mean(errors)
            if avg_error > 0.5:
                improvements.append({
                    "pair": (src.value, tgt.value),
                    "current_error": float(avg_error),
                    "recommendation": "retrain_matrix",
                    "priority": min(1.0, avg_error)
                })

        return sorted(improvements, key=lambda x: x["priority"], reverse=True)

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Generate semantic bridge refinements."""
        synthesis = {
            "new_mappings": [],
            "mapping_updates": [],
            "translation_matrix_updates": [],
            "new_concepts": [],
            "conflict_resolutions_applied": []
        }

        # Create new mappings for high-priority bridge opportunities
        for opportunity in analysis.get("bridge_opportunities", []):
            if opportunity["candidates"]:
                best_candidate = max(
                    opportunity["candidates"],
                    key=lambda x: x["combined_similarity"]
                )

                if best_candidate["combined_similarity"] > 0.5:
                    synthesis["new_mappings"].append({
                        "source_domain": opportunity["source_domain"],
                        "target_domain": best_candidate["target_domain"],
                        "source_concept": opportunity["term"],
                        "target_concept": best_candidate["target_term"],
                        "confidence": best_candidate["combined_similarity"],
                        "universal_anchor": opportunity.get("universal_anchor")
                    })

        # Apply conflict resolutions
        for resolution in analysis.get("conflict_resolutions", []):
            if resolution["strategy"] == "reduce_confidence":
                synthesis["conflict_resolutions_applied"].append({
                    "conflict_id": resolution["conflict"]["conflict_id"],
                    "action": resolution["action"],
                    "new_confidence": resolution.get("new_confidence", 0.3)
                })

        # Create new concepts from emergent patterns
        for emergent in analysis.get("emergent_concepts", []):
            if emergent["confidence"] > 0.7:
                synthesis["new_concepts"].append({
                    "concept_id": f"emergent_{len(self.emergent_concepts)}",
                    "universal_name": emergent["proposed_name"],
                    "domain_representations": {
                        domain: info["term"]
                        for domain, info in emergent["domain_representations"].items()
                        if info["similarity"] > 0.4
                    },
                    "embedding": emergent.get("universal_embedding"),
                    "confidence": emergent["confidence"],
                    "emergence_source": f"cross_domain_cluster_{emergent['source_domains']}"
                })

        # Update translation matrices based on gradient
        for (src, tgt) in self.translation_matrices.keys():
            src_vec = tbl.get_state(src).state_vector[:self.embedding_dim]
            tgt_vec = tbl.get_state(tgt).state_vector[:self.embedding_dim]

            # Pad if necessary
            if len(src_vec) < self.embedding_dim:
                src_vec = np.pad(src_vec, (0, self.embedding_dim - len(src_vec)))
            if len(tgt_vec) < self.embedding_dim:
                tgt_vec = np.pad(tgt_vec, (0, self.embedding_dim - len(tgt_vec)))

            # Compute alignment gradient
            outer_product = np.outer(src_vec, tgt_vec)
            synthesis["translation_matrix_updates"].append({
                "pair": (src.value, tgt.value),
                "gradient": outer_product * 0.001  # Small learning rate
            })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply semantic bridge updates to the system."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Add new mappings
        new_mapping_count = 0
        for mapping_data in synthesis.get("new_mappings", []):
            mapping_id = f"map_{mapping_data['source_concept']}_{mapping_data['target_domain']}"

            if mapping_id not in self.mappings:
                mapping = SemanticMapping(
                    mapping_id=mapping_id,
                    source_domain=DomainLeg(mapping_data["source_domain"]),
                    target_domain=DomainLeg(mapping_data["target_domain"]),
                    source_concept=mapping_data["source_concept"],
                    target_concept=mapping_data["target_concept"],
                    translation_confidence=mapping_data["confidence"]
                )
                self.mappings[mapping_id] = mapping
                new_mapping_count += 1

                # Update concept graph
                src_id = f"{mapping_data['source_domain']}:{mapping_data['source_concept']}"
                tgt_id = f"{mapping_data['target_domain']}:{mapping_data['target_concept']}"
                self.concept_graph[src_id].add(tgt_id)
                self.concept_graph[tgt_id].add(src_id)

                insights.append(
                    f"Bridged '{mapping_data['source_concept']}' ({mapping_data['source_domain']}) "
                    f"↔ '{mapping_data['target_concept']}' ({mapping_data['target_domain']}) "
                    f"[confidence: {mapping_data['confidence']:.2f}]"
                )

        if new_mapping_count > 0:
            changes["new_mappings"] = new_mapping_count

        # Add new concepts
        new_concept_count = 0
        for concept_data in synthesis.get("new_concepts", []):
            concept_id = concept_data["concept_id"]

            if concept_id not in self.concepts:
                embedding = np.array(concept_data.get("embedding", np.zeros(self.embedding_dim)))
                if len(embedding) != self.embedding_dim:
                    embedding = np.zeros(self.embedding_dim)

                concept = SemanticConcept(
                    concept_id=concept_id,
                    universal_name=concept_data["universal_name"],
                    domain_representations={
                        DomainLeg(d) if isinstance(d, str) else d: rep
                        for d, rep in concept_data["domain_representations"].items()
                    },
                    embedding=embedding,
                    confidence=concept_data["confidence"],
                    emergence_source=concept_data.get("emergence_source")
                )
                self.concepts[concept_id] = concept

                # Track as emergent
                emergent = EmergentConcept(
                    concept_id=concept_id,
                    source_patterns=[],
                    proposed_name=concept_data["universal_name"],
                    domain_manifestations=concept.domain_representations,
                    confidence=concept_data["confidence"]
                )
                self.emergent_concepts.append(emergent)

                new_concept_count += 1
                insights.append(
                    f"Discovered emergent concept '{concept_data['universal_name']}' "
                    f"[confidence: {concept_data['confidence']:.2f}]"
                )

        if new_concept_count > 0:
            changes["new_concepts"] = new_concept_count

        # Apply conflict resolutions
        resolution_count = 0
        for resolution in synthesis.get("conflict_resolutions_applied", []):
            for mapping in self.mappings.values():
                if resolution["action"] == "lower_translation_confidence":
                    if mapping.translation_confidence > resolution["new_confidence"]:
                        mapping.translation_confidence = resolution["new_confidence"]
                        resolution_count += 1

        if resolution_count > 0:
            changes["conflict_resolutions"] = resolution_count
            insights.append(f"Resolved {resolution_count} semantic conflicts")

        # Update translation matrices
        matrix_updates = 0
        for update in synthesis.get("translation_matrix_updates", []):
            pair_key = (DomainLeg(update["pair"][0]), DomainLeg(update["pair"][1]))
            if pair_key in self.translation_matrices:
                self.translation_matrices[pair_key] += update["gradient"]

                # Regularize to prevent explosion
                matrix = self.translation_matrices[pair_key]
                norm = np.linalg.norm(matrix)
                if norm > 10:
                    self.translation_matrices[pair_key] = matrix / norm * 10

                matrix_updates += 1

        if matrix_updates > 0:
            changes["translation_matrix_updates"] = matrix_updates

        # Measure new coherence
        new_coherence = await self._measure_semantic_coherence(tbl)
        metrics_delta["semantic_coherence"] = new_coherence
        metrics_delta["coherence_delta"] = new_coherence - metrics_delta.get("harmony_before", 0.5)

        # Update stats
        metrics_delta["total_mappings"] = len(self.mappings)
        metrics_delta["total_concepts"] = len(self.concepts)
        metrics_delta["emergent_concepts"] = len(self.emergent_concepts)

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score

        # Emit cross-domain signal about semantic updates
        if new_mapping_count > 0 or new_concept_count > 0:
            self.emit_cross_domain_signal(
                signal_type="semantic_bridge_update",
                payload={
                    "new_mappings": new_mapping_count,
                    "new_concepts": new_concept_count,
                    "coherence": new_coherence
                },
                strength=0.8
            )

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    # Public translation API methods

    async def translate_concept(
        self,
        concept: str,
        source_domain: DomainLeg,
        target_domain: DomainLeg,
        context: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Translate a concept from one domain to another.

        Args:
            concept: The concept to translate
            source_domain: The domain the concept is from
            target_domain: The domain to translate to
            context: Optional context words to improve translation

        Returns:
            Translation result with target concept and confidence
        """
        # Check direct mappings first
        for mapping in self.mappings.values():
            if (mapping.source_domain == source_domain and
                mapping.target_domain == target_domain and
                mapping.source_concept == concept):

                mapping.usage_count += 1
                return {
                    "source": concept,
                    "target": mapping.target_concept,
                    "confidence": mapping.translation_confidence,
                    "method": "direct_mapping"
                }

        # Try neural translation
        source_emb = self._domain_term_embeddings.get(source_domain, {}).get(
            concept, self.embedder.embed(f"{source_domain.value}: {concept}")
        )

        # Apply translation matrix
        if (source_domain, target_domain) in self.translation_matrices:
            matrix = self.translation_matrices[(source_domain, target_domain)]
            translated_emb = matrix @ source_emb
        else:
            translated_emb = source_emb

        # Add context if provided
        if context:
            context_embs = [self.embedder.embed(c) for c in context]
            context_avg = np.mean(context_embs, axis=0)
            translated_emb = translated_emb * 0.7 + context_avg * 0.3

        # Find closest target concept
        best_match = None
        best_sim = 0.0

        for term in self.domain_vocab[target_domain]:
            term_emb = self._domain_term_embeddings.get(target_domain, {}).get(term)
            if term_emb is not None:
                sim = self.embedder.similarity(translated_emb, term_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = term

        if best_match and best_sim > 0.3:
            return {
                "source": concept,
                "target": best_match,
                "confidence": float(best_sim),
                "method": "neural_translation"
            }

        return None

    async def find_multi_hop_path(
        self,
        source_concept: str,
        source_domain: DomainLeg,
        target_concept: str,
        target_domain: DomainLeg,
        max_hops: int = 3
    ) -> Optional[TranslationPath]:
        """Find a multi-hop translation path between concepts."""
        if source_domain == target_domain:
            return None

        # BFS through concept graph
        start_id = f"{source_domain.value}:{source_concept}"
        end_id = f"{target_domain.value}:{target_concept}"

        queue = [(start_id, [start_id], 1.0)]
        visited = {start_id}

        while queue:
            current, path, confidence = queue.pop(0)

            if current == end_id:
                hops = []
                for node in path:
                    parts = node.split(":", 1)
                    if len(parts) == 2:
                        hops.append((DomainLeg(parts[0]), parts[1]))

                return TranslationPath(
                    path_id=f"path_{source_concept}_{target_concept}",
                    source_domain=source_domain,
                    target_domain=target_domain,
                    hops=hops,
                    total_confidence=confidence,
                    path_length=len(hops)
                )

            if len(path) >= max_hops:
                continue

            for neighbor in self.concept_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Estimate confidence decay per hop
                    new_confidence = confidence * 0.9
                    queue.append((neighbor, path + [neighbor], new_confidence))

        return None

    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the semantic bridge."""
        return {
            "total_concepts": len(self.concepts),
            "total_mappings": len(self.mappings),
            "emergent_concepts": len(self.emergent_concepts),
            "vocabulary_sizes": {
                domain.value: len(vocab)
                for domain, vocab in self.domain_vocab.items()
            },
            "concept_graph_nodes": len(self.concept_graph),
            "concept_graph_edges": sum(len(neighbors) for neighbors in self.concept_graph.values()) // 2,
            "translation_matrices": len(self.translation_matrices),
            "embedding_dimension": self.embedding_dim,
            "neural_embeddings_active": self.embedder.use_neural,
            "active_conflicts": len(self.active_conflicts)
        }
