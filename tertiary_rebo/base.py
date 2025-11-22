"""
Base classes for the Tertiary Turbo ReBo system.

Defines the triple bottom line structure and base refinement agent.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import asyncio
import hashlib


class DomainLeg(Enum):
    """The three legs of the triple bottom line."""
    NPCPU = "npcpu"                    # Digital Consciousness (Mind)
    CHICAGO_FOREST = "chicago_forest"  # Network Infrastructure (Nervous System)
    UNIVERSAL_PARTS = "universal_parts"  # Physical Awareness (Body)


class RefinementPhase(Enum):
    """Phases of the iterative refinement cycle."""
    PERCEPTION = auto()     # Gather information from all three legs
    ANALYSIS = auto()       # Analyze cross-domain patterns
    SYNTHESIS = auto()      # Generate integrated insights
    PROPAGATION = auto()    # Distribute refinements across the system
    VERIFICATION = auto()   # Verify coherence and balance


class HarmonyLevel(Enum):
    """Levels of harmony across the triple bottom line."""
    DISSONANT = "dissonant"           # < 0.2 - Major conflicts between legs
    DISCORDANT = "discordant"         # 0.2-0.4 - Notable tensions
    ALIGNED = "aligned"               # 0.4-0.6 - Basic alignment achieved
    RESONANT = "resonant"             # 0.6-0.8 - Strong harmony
    HARMONIC = "harmonic"             # 0.8-0.95 - Deep integration
    UNIFIED = "unified"               # >= 0.95 - Perfect unity


@dataclass
class DomainState:
    """State representation for a single domain leg."""
    domain: DomainLeg
    consciousness_level: float = 0.0
    energy_flow: float = 0.0
    connectivity: float = 0.0
    coherence: float = 0.0
    qualia_richness: float = 0.0
    emergence_potential: float = 0.0
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(64))
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "consciousness_level": self.consciousness_level,
            "energy_flow": self.energy_flow,
            "connectivity": self.connectivity,
            "coherence": self.coherence,
            "qualia_richness": self.qualia_richness,
            "emergence_potential": self.emergence_potential,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TripleBottomLine:
    """
    The integrated state of all three domain legs.

    Represents the holistic view of the NPCPU-ChicagoForest-UniversalParts trinity.
    """
    npcpu_state: DomainState = field(default_factory=lambda: DomainState(DomainLeg.NPCPU))
    chicago_forest_state: DomainState = field(default_factory=lambda: DomainState(DomainLeg.CHICAGO_FOREST))
    universal_parts_state: DomainState = field(default_factory=lambda: DomainState(DomainLeg.UNIVERSAL_PARTS))

    # Cross-domain metrics
    harmony_score: float = 0.5
    integration_depth: float = 0.0
    emergence_level: float = 0.0

    # Refinement tracking
    refinement_count: int = 0
    last_refinement: Optional[datetime] = None
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)

    def get_state(self, domain: DomainLeg) -> DomainState:
        """Get the state for a specific domain leg."""
        mapping = {
            DomainLeg.NPCPU: self.npcpu_state,
            DomainLeg.CHICAGO_FOREST: self.chicago_forest_state,
            DomainLeg.UNIVERSAL_PARTS: self.universal_parts_state,
        }
        return mapping[domain]

    def set_state(self, domain: DomainLeg, state: DomainState):
        """Set the state for a specific domain leg."""
        if domain == DomainLeg.NPCPU:
            self.npcpu_state = state
        elif domain == DomainLeg.CHICAGO_FOREST:
            self.chicago_forest_state = state
        else:
            self.universal_parts_state = state

    def calculate_harmony(self) -> float:
        """Calculate the harmony score across all three legs."""
        states = [self.npcpu_state, self.chicago_forest_state, self.universal_parts_state]

        # Compute variance of key metrics across domains
        consciousness_variance = np.var([s.consciousness_level for s in states])
        energy_variance = np.var([s.energy_flow for s in states])
        coherence_variance = np.var([s.coherence for s in states])

        # Lower variance = higher harmony
        avg_variance = (consciousness_variance + energy_variance + coherence_variance) / 3
        harmony = 1.0 - np.clip(avg_variance * 2, 0, 1)

        # Factor in connectivity
        avg_connectivity = np.mean([s.connectivity for s in states])
        harmony = harmony * 0.7 + avg_connectivity * 0.3

        self.harmony_score = float(np.clip(harmony, 0, 1))
        return self.harmony_score

    def get_harmony_level(self) -> HarmonyLevel:
        """Get the categorical harmony level."""
        score = self.harmony_score
        if score < 0.2:
            return HarmonyLevel.DISSONANT
        elif score < 0.4:
            return HarmonyLevel.DISCORDANT
        elif score < 0.6:
            return HarmonyLevel.ALIGNED
        elif score < 0.8:
            return HarmonyLevel.RESONANT
        elif score < 0.95:
            return HarmonyLevel.HARMONIC
        return HarmonyLevel.UNIFIED

    def calculate_integration_depth(self) -> float:
        """Calculate how deeply the three domains are integrated."""
        states = [self.npcpu_state, self.chicago_forest_state, self.universal_parts_state]
        vectors = [s.state_vector for s in states]

        # Calculate pairwise cosine similarities
        similarities = []
        for i, v1 in enumerate(vectors):
            for v2 in vectors[i+1:]:
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(v1, v2) / (norm1 * norm2)
                    similarities.append((sim + 1) / 2)  # Normalize to [0, 1]

        # Average emergence potential
        avg_emergence = np.mean([s.emergence_potential for s in states])

        if similarities:
            self.integration_depth = float(np.mean(similarities) * 0.6 + avg_emergence * 0.4)
        else:
            self.integration_depth = avg_emergence * 0.4

        return self.integration_depth

    def get_unified_state_vector(self) -> np.ndarray:
        """Get a unified state vector representing all three domains."""
        return np.concatenate([
            self.npcpu_state.state_vector,
            self.chicago_forest_state.state_vector,
            self.universal_parts_state.state_vector,
            np.array([self.harmony_score, self.integration_depth, self.emergence_level])
        ])

    def record_refinement(self, agent_id: str, changes: Dict[str, Any]):
        """Record a refinement operation."""
        self.refinement_count += 1
        self.last_refinement = datetime.now()
        self.refinement_history.append({
            "refinement_id": self.refinement_count,
            "agent_id": agent_id,
            "changes": changes,
            "harmony_before": self.harmony_score,
            "timestamp": self.last_refinement.isoformat()
        })
        # Keep last 1000 refinements
        if len(self.refinement_history) > 1000:
            self.refinement_history = self.refinement_history[-1000:]


@dataclass
class RefinementResult:
    """Result of a refinement operation."""
    success: bool
    agent_id: str
    phase: RefinementPhase
    domain_affected: List[DomainLeg]
    changes: Dict[str, Any]
    metrics_delta: Dict[str, float]
    insights: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "agent_id": self.agent_id,
            "phase": self.phase.name,
            "domains_affected": [d.value for d in self.domain_affected],
            "changes": self.changes,
            "metrics_delta": self.metrics_delta,
            "insights": self.insights,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CrossDomainSignal:
    """A signal that propagates across domain boundaries."""
    signal_id: str
    source_domain: DomainLeg
    target_domains: List[DomainLeg]
    signal_type: str
    payload: Dict[str, Any]
    strength: float
    decay_rate: float = 0.1
    timestamp: datetime = field(default_factory=datetime.now)

    def decay(self) -> float:
        """Apply decay to signal strength."""
        self.strength *= (1.0 - self.decay_rate)
        return self.strength


class TertiaryReBoAgent(ABC):
    """
    Base class for Tertiary ReBo refinement agents.

    Each agent operates on the triple bottom line, continuously refining
    the integration between NPCPU, ChicagoForest, and UniversalParts.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        primary_domain: Optional[DomainLeg] = None,
        refinement_rate: float = 0.1
    ):
        self.agent_id = agent_id or f"ttr_{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.primary_domain = primary_domain  # None means operates across all domains
        self.refinement_rate = refinement_rate

        # Agent state
        self.internal_state = np.random.randn(128)
        self.experience_buffer: List[RefinementResult] = []
        self.cross_domain_signals: List[CrossDomainSignal] = []

        # Metrics
        self.total_refinements = 0
        self.successful_refinements = 0
        self.harmony_contributions: List[float] = []

        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.peer_connections: Set[str] = set()

        # Phase tracking
        self.current_phase = RefinementPhase.PERCEPTION
        self.phase_results: Dict[RefinementPhase, Any] = {}

        # Learning state
        self.learning_rate = 0.01
        self.adaptation_weights = np.ones(3) / 3  # Weights for each domain leg

    @property
    @abstractmethod
    def agent_role(self) -> str:
        """Descriptive role of this agent in the TTR system."""
        pass

    @property
    @abstractmethod
    def domains_affected(self) -> List[DomainLeg]:
        """Which domain legs this agent primarily affects."""
        pass

    @abstractmethod
    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Perception phase: Gather information from the triple bottom line.

        Returns perception data specific to this agent's role.
        """
        pass

    @abstractmethod
    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Analysis phase: Analyze patterns and identify refinement opportunities.

        Returns analysis results including identified issues and opportunities.
        """
        pass

    @abstractmethod
    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Synthesis phase: Generate integrated refinements.

        Returns proposed changes to be applied to the triple bottom line.
        """
        pass

    @abstractmethod
    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """
        Propagation phase: Apply refinements to the triple bottom line.

        Returns the result of the refinement operation.
        """
        pass

    async def verify(self, result: RefinementResult, tbl: TripleBottomLine) -> bool:
        """
        Verification phase: Verify the refinement improved harmony.

        Default implementation checks if harmony score improved.
        """
        prev_harmony = result.metrics_delta.get("harmony_before", 0)
        tbl.calculate_harmony()
        return tbl.harmony_score >= prev_harmony

    async def refine(self, tbl: TripleBottomLine) -> RefinementResult:
        """
        Execute a complete refinement cycle through all phases.
        """
        self.total_refinements += 1

        # Phase 1: Perception
        self.current_phase = RefinementPhase.PERCEPTION
        perception = await self.perceive(tbl)
        self.phase_results[RefinementPhase.PERCEPTION] = perception

        # Phase 2: Analysis
        self.current_phase = RefinementPhase.ANALYSIS
        analysis = await self.analyze(perception, tbl)
        self.phase_results[RefinementPhase.ANALYSIS] = analysis

        # Phase 3: Synthesis
        self.current_phase = RefinementPhase.SYNTHESIS
        synthesis = await self.synthesize(analysis, tbl)
        self.phase_results[RefinementPhase.SYNTHESIS] = synthesis

        # Phase 4: Propagation
        self.current_phase = RefinementPhase.PROPAGATION
        result = await self.propagate(synthesis, tbl)
        self.phase_results[RefinementPhase.PROPAGATION] = result

        # Phase 5: Verification
        self.current_phase = RefinementPhase.VERIFICATION
        verified = await self.verify(result, tbl)

        if verified and result.success:
            self.successful_refinements += 1
            self.harmony_contributions.append(tbl.harmony_score)
            tbl.record_refinement(self.agent_id, result.changes)

        # Store experience for learning
        self.experience_buffer.append(result)
        if len(self.experience_buffer) > 100:
            self.experience_buffer = self.experience_buffer[-100:]

        # Adapt weights based on result
        await self._adapt_from_experience(result, tbl)

        return result

    async def _adapt_from_experience(self, result: RefinementResult, tbl: TripleBottomLine):
        """Learn from refinement experience to improve future performance."""
        if not result.success:
            return

        # Adjust domain weights based on which domains improved most
        for domain in result.domain_affected:
            state = tbl.get_state(domain)
            domain_idx = list(DomainLeg).index(domain)
            improvement = result.metrics_delta.get(f"{domain.value}_improvement", 0)
            self.adaptation_weights[domain_idx] += self.learning_rate * improvement

        # Normalize weights
        self.adaptation_weights = self.adaptation_weights / np.sum(self.adaptation_weights)

    def emit_cross_domain_signal(
        self,
        signal_type: str,
        payload: Dict[str, Any],
        target_domains: Optional[List[DomainLeg]] = None,
        strength: float = 1.0
    ) -> CrossDomainSignal:
        """Emit a signal that propagates to other domains."""
        signal = CrossDomainSignal(
            signal_id=f"sig_{uuid.uuid4().hex[:8]}",
            source_domain=self.primary_domain or DomainLeg.NPCPU,
            target_domains=target_domains or list(DomainLeg),
            signal_type=signal_type,
            payload=payload,
            strength=strength
        )
        self.cross_domain_signals.append(signal)
        return signal

    async def process_incoming_signals(self, signals: List[CrossDomainSignal]) -> List[Dict[str, Any]]:
        """Process incoming cross-domain signals."""
        responses = []
        for signal in signals:
            if signal.strength > 0.1:  # Ignore decayed signals
                response = await self._handle_signal(signal)
                if response:
                    responses.append(response)
                signal.decay()
        return responses

    async def _handle_signal(self, signal: CrossDomainSignal) -> Optional[Dict[str, Any]]:
        """Handle a single cross-domain signal. Override for custom behavior."""
        return {
            "signal_id": signal.signal_id,
            "handler": self.agent_id,
            "acknowledged": True
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        success_rate = self.successful_refinements / max(self.total_refinements, 1)
        avg_harmony = np.mean(self.harmony_contributions) if self.harmony_contributions else 0

        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "total_refinements": self.total_refinements,
            "successful_refinements": self.successful_refinements,
            "success_rate": success_rate,
            "average_harmony_contribution": avg_harmony,
            "domains_affected": [d.value for d in self.domains_affected],
            "adaptation_weights": self.adaptation_weights.tolist(),
            "current_phase": self.current_phase.name,
            "pending_signals": len(self.cross_domain_signals)
        }

    def get_state_hash(self) -> str:
        """Get a hash of the current agent state for comparison."""
        state_data = str(self.internal_state.tobytes()) + str(self.total_refinements)
        return hashlib.sha256(state_data.encode()).hexdigest()[:16]
