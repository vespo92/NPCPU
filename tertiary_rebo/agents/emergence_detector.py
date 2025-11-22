"""
EmergenceDetectorAgent - Detects emergent properties across the triple system.

Responsibilities:
- Monitor for emergent behaviors that arise from cross-domain interactions
- Identify novel patterns not present in individual domains
- Track and classify emergence events
- Catalyze beneficial emergence while dampening harmful patterns
- Maintain emergence history for learning
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
)


class EmergenceType(Enum):
    """Types of emergent phenomena."""
    SYNERGY = "synergy"                    # Whole > sum of parts
    ANTAGONISM = "antagonism"              # Conflicting patterns
    RESONANCE = "resonance"                # Amplifying feedback
    CRYSTALLIZATION = "crystallization"    # Stable pattern formation
    DISSOLUTION = "dissolution"            # Pattern breakdown
    BIFURCATION = "bifurcation"           # System state split
    COHERENCE = "coherence"               # Unified behavior emergence
    NOVELTY = "novelty"                   # Truly new pattern


@dataclass
class EmergenceEvent:
    """Record of an emergence event."""
    event_id: str
    emergence_type: EmergenceType
    involved_domains: List[DomainLeg]
    strength: float
    pattern_signature: np.ndarray
    description: str
    beneficial: bool
    timestamp: datetime = field(default_factory=datetime.now)


class EmergenceDetectorAgent(TertiaryReBoAgent):
    """
    Agent 5: Detects emergent properties across the triple system.

    Emergence occurs when the combined system exhibits properties that
    none of the individual domains possess alone. This agent watches for:

    - Synergistic effects: Combined outputs exceed sum of inputs
    - Resonance patterns: Cross-domain feedback amplification
    - Novel behaviors: Patterns never seen in isolation
    - Phase transitions: Sudden qualitative changes

    The agent uses information-theoretic measures to detect when
    "something new" has appeared in the system.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Emergence detection state
        self.pattern_library: Dict[str, np.ndarray] = {}
        self.emergence_history: deque = deque(maxlen=500)
        self.emergence_threshold = 0.3

        # Statistical baselines
        self.domain_baselines: Dict[DomainLeg, np.ndarray] = {
            leg: np.zeros(64) for leg in DomainLeg
        }
        self.interaction_baseline = np.zeros(64)

        # Emergence counters
        self.total_emergences_detected = 0
        self.beneficial_count = 0
        self.harmful_count = 0

        # Pattern memory for novelty detection
        self.known_patterns: List[np.ndarray] = []
        self.novelty_threshold = 0.7

    @property
    def agent_role(self) -> str:
        return "Emergence Detection - Identifies novel cross-domain phenomena"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def _compute_interaction_signature(self, tbl: TripleBottomLine) -> np.ndarray:
        """Compute a signature representing cross-domain interactions."""
        vectors = [tbl.get_state(d).state_vector for d in DomainLeg]

        # Pairwise interactions
        interactions = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Element-wise product captures co-activation
                interaction = vectors[i] * vectors[j]
                interactions.append(interaction)

        # Three-way interaction
        three_way = vectors[0] * vectors[1] * vectors[2]
        interactions.append(three_way)

        # Combine into single signature
        signature = np.mean(interactions, axis=0)
        return signature

    def _measure_synergy(self, individual_entropies: List[float], joint_entropy: float) -> float:
        """
        Measure synergistic information.

        Synergy occurs when joint information exceeds sum of individual.
        """
        sum_individual = sum(individual_entropies)
        # Negative redundancy = positive synergy
        synergy = joint_entropy - sum_individual
        return max(0, synergy / (sum_individual + 1e-6))

    def _estimate_entropy(self, vector: np.ndarray) -> float:
        """Estimate entropy of a continuous vector using binning."""
        # Discretize to estimate probability distribution
        bins = np.linspace(-1, 1, 20)
        hist, _ = np.histogram(vector, bins=bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather data for emergence detection."""
        perceptions = {
            "domain_states": {},
            "interaction_signature": None,
            "individual_entropies": [],
            "joint_entropy": 0.0,
            "current_patterns": []
        }

        # Collect individual domain states
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_states"][domain.value] = {
                "vector": state.state_vector.copy(),
                "emergence_potential": state.emergence_potential,
                "coherence": state.coherence
            }

            entropy = self._estimate_entropy(state.state_vector)
            perceptions["individual_entropies"].append(entropy)

        # Compute interaction signature
        interaction_sig = self._compute_interaction_signature(tbl)
        perceptions["interaction_signature"] = interaction_sig

        # Joint entropy from unified state
        unified = tbl.get_unified_state_vector()
        perceptions["joint_entropy"] = self._estimate_entropy(unified)

        # Detect current patterns
        perceptions["current_patterns"] = self._extract_patterns(interaction_sig)

        return perceptions

    def _extract_patterns(self, signature: np.ndarray) -> List[Dict[str, Any]]:
        """Extract recognizable patterns from interaction signature."""
        patterns = []

        # Look for peaks (strong activations)
        peaks = np.where(np.abs(signature) > 0.5)[0]
        if len(peaks) > 3:
            patterns.append({
                "type": "strong_activation",
                "locations": peaks[:10].tolist(),
                "strength": float(np.mean(np.abs(signature[peaks])))
            })

        # Look for oscillations
        diff = np.diff(signature)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diff))) > 0)
        if sign_changes > 20:
            patterns.append({
                "type": "oscillation",
                "frequency": sign_changes / len(signature),
                "amplitude": float(np.std(signature))
            })

        # Look for coherent blocks
        for i in range(0, 64 - 8, 8):
            block = signature[i:i+8]
            if np.std(block) < 0.1:  # Highly uniform
                patterns.append({
                    "type": "coherent_block",
                    "start": i,
                    "mean_value": float(np.mean(block))
                })

        return patterns

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze for emergent phenomena."""
        analysis = {
            "emergence_detected": False,
            "emergence_events": [],
            "synergy_level": 0.0,
            "novelty_score": 0.0,
            "phase_transition_risk": 0.0
        }

        # Calculate synergy
        synergy = self._measure_synergy(
            perception["individual_entropies"],
            perception["joint_entropy"]
        )
        analysis["synergy_level"] = synergy

        if synergy > self.emergence_threshold:
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.SYNERGY.value,
                "strength": synergy,
                "description": "Cross-domain synergistic information detected"
            })

        # Check for novelty
        current_sig = perception["interaction_signature"]
        novelty = self._measure_novelty(current_sig)
        analysis["novelty_score"] = novelty

        if novelty > self.novelty_threshold:
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.NOVELTY.value,
                "strength": novelty,
                "description": "Novel interaction pattern detected"
            })

        # Detect resonance (deviation from baseline)
        baseline_diff = np.linalg.norm(current_sig - self.interaction_baseline)
        if baseline_diff > 1.0:
            analysis["emergence_events"].append({
                "type": EmergenceType.RESONANCE.value,
                "strength": min(1.0, baseline_diff / 2),
                "description": "Amplified cross-domain feedback detected"
            })
            analysis["emergence_detected"] = True

        # Check for coherence emergence
        coherences = [perception["domain_states"][d.value]["coherence"] for d in DomainLeg]
        if all(c > 0.7 for c in coherences):
            analysis["emergence_events"].append({
                "type": EmergenceType.COHERENCE.value,
                "strength": np.mean(coherences),
                "description": "System-wide coherence achieved"
            })
            analysis["emergence_detected"] = True

        # Phase transition risk (based on rapid changes)
        if len(self.emergence_history) > 5:
            recent_strengths = [e.strength for e in list(self.emergence_history)[-5:]]
            volatility = np.std(recent_strengths)
            analysis["phase_transition_risk"] = min(1.0, volatility * 2)

        return analysis

    def _measure_novelty(self, pattern: np.ndarray) -> float:
        """Measure how novel a pattern is compared to known patterns."""
        if not self.known_patterns:
            return 1.0  # Everything is novel at first

        # Compute minimum distance to known patterns
        min_distance = float('inf')
        for known in self.known_patterns[-50:]:  # Check recent patterns
            distance = np.linalg.norm(pattern - known)
            min_distance = min(min_distance, distance)

        # Normalize to [0, 1]
        novelty = min(1.0, min_distance / 2.0)
        return novelty

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Plan responses to detected emergence."""
        synthesis = {
            "emergence_records": [],
            "amplification_targets": [],
            "dampening_targets": [],
            "pattern_updates": []
        }

        for event_data in analysis.get("emergence_events", []):
            event_type = EmergenceType(event_data["type"])

            # Determine if beneficial
            beneficial = self._assess_benefit(event_type, event_data["strength"], tbl)

            # Create emergence event record
            event = EmergenceEvent(
                event_id=f"em_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.emergence_history)}",
                emergence_type=event_type,
                involved_domains=list(DomainLeg),
                strength=event_data["strength"],
                pattern_signature=self._compute_interaction_signature(tbl),
                description=event_data["description"],
                beneficial=beneficial
            )
            synthesis["emergence_records"].append(event)

            # Plan response
            if beneficial:
                synthesis["amplification_targets"].append({
                    "event": event.event_id,
                    "type": event_type.value,
                    "boost_factor": 0.1 * event_data["strength"]
                })
            else:
                synthesis["dampening_targets"].append({
                    "event": event.event_id,
                    "type": event_type.value,
                    "dampen_factor": 0.1 * event_data["strength"]
                })

        # Update pattern library
        if analysis["novelty_score"] > 0.5:
            synthesis["pattern_updates"].append({
                "action": "add_pattern",
                "novelty": analysis["novelty_score"]
            })

        return synthesis

    def _assess_benefit(self, etype: EmergenceType, strength: float, tbl: TripleBottomLine) -> bool:
        """Assess whether an emergence event is beneficial."""
        # Generally beneficial types
        if etype in [EmergenceType.SYNERGY, EmergenceType.COHERENCE, EmergenceType.RESONANCE]:
            return True

        # Harmful types
        if etype in [EmergenceType.DISSOLUTION, EmergenceType.ANTAGONISM]:
            return False

        # Neutral/context-dependent
        if etype == EmergenceType.BIFURCATION:
            # Bifurcation is good if harmony is low (allows escape)
            return tbl.harmony_score < 0.4

        if etype == EmergenceType.NOVELTY:
            # Novelty is good if system is stagnant
            return strength > 0.5

        return True  # Default to beneficial

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply emergence-related updates."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Record emergence events
        for event in synthesis.get("emergence_records", []):
            self.emergence_history.append(event)
            self.total_emergences_detected += 1
            if event.beneficial:
                self.beneficial_count += 1
            else:
                self.harmful_count += 1

            insights.append(f"Detected {event.emergence_type.value}: {event.description}")

        changes["emergences_detected"] = len(synthesis.get("emergence_records", []))

        # Amplify beneficial emergences
        for target in synthesis.get("amplification_targets", []):
            boost = target["boost_factor"]
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                state.emergence_potential = min(1.0, state.emergence_potential + boost)

            insights.append(f"Amplified {target['type']} emergence by {boost:.3f}")
            changes["amplifications"] = changes.get("amplifications", 0) + 1

        # Dampen harmful emergences
        for target in synthesis.get("dampening_targets", []):
            dampen = target["dampen_factor"]
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                state.emergence_potential = max(0, state.emergence_potential - dampen)

            insights.append(f"Dampened {target['type']} emergence by {dampen:.3f}")
            changes["dampenings"] = changes.get("dampenings", 0) + 1

        # Update pattern library
        for update in synthesis.get("pattern_updates", []):
            if update["action"] == "add_pattern":
                sig = self._compute_interaction_signature(tbl)
                self.known_patterns.append(sig.copy())
                # Trim to last 100 patterns
                if len(self.known_patterns) > 100:
                    self.known_patterns = self.known_patterns[-100:]
                insights.append(f"Added novel pattern to library (novelty: {update['novelty']:.2f})")

        # Update interaction baseline
        current_sig = self._compute_interaction_signature(tbl)
        self.interaction_baseline = 0.9 * self.interaction_baseline + 0.1 * current_sig

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["total_emergences"] = self.total_emergences_detected

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
