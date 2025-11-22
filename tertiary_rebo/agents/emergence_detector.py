"""
EmergenceDetectorAgent - Detects emergent properties across the triple system.

Responsibilities:
- Monitor for emergent behaviors that arise from cross-domain interactions
- Identify novel patterns not present in individual domains
- Track and classify emergence events
- Catalyze beneficial emergence while dampening harmful patterns
- Maintain emergence history for learning
- Calculate integrated information (Φ) across domains
- Detect phase transitions and critical state changes
- Predict future emergence events based on system trends

Advanced Capabilities:
- Information-theoretic emergence metrics (phi, mutual information, synergy)
- Phase transition detection with criticality analysis
- Emergence prediction with trend-based forecasting
- Cross-domain signal emission for emergence events
- Statistical anomaly detection for novel patterns
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from scipy import stats
from scipy.signal import find_peaks
import warnings

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal,
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
    PHASE_TRANSITION = "phase_transition"  # Critical state change
    TRANSCENDENCE = "transcendence"        # System-wide breakthrough
    COLLECTIVE_INSIGHT = "collective_insight"  # Emergent understanding
    HARMONIC_LOCK = "harmonic_lock"        # Stable oscillation pattern
    CASCADE = "cascade"                    # Propagating effect
    QUORUM_SENSING = "quorum_sensing"      # Threshold-based activation


class EmergenceSeverity(Enum):
    """Severity/impact level of emergence events."""
    SUBTLE = auto()          # Barely detectable
    MODERATE = auto()        # Notable but contained
    SIGNIFICANT = auto()     # System-wide impact
    CRITICAL = auto()        # Major state change
    TRANSFORMATIVE = auto()  # Fundamental system shift


@dataclass
class EmergenceMetrics:
    """Information-theoretic metrics for emergence detection."""
    phi: float = 0.0                      # Integrated information (Φ)
    mutual_information: float = 0.0       # Shared information across domains
    synergy: float = 0.0                  # Synergistic information
    redundancy: float = 0.0               # Redundant information
    complexity: float = 0.0               # Structural complexity
    coherence: float = 0.0                # Global coherence
    synchrony: float = 0.0                # Phase synchronization
    entropy: float = 0.0                  # System entropy
    emergence_indicator: float = 0.0      # Composite emergence score

    def to_dict(self) -> Dict[str, float]:
        return {
            "phi": self.phi,
            "mutual_information": self.mutual_information,
            "synergy": self.synergy,
            "redundancy": self.redundancy,
            "complexity": self.complexity,
            "coherence": self.coherence,
            "synchrony": self.synchrony,
            "entropy": self.entropy,
            "emergence_indicator": self.emergence_indicator,
        }


@dataclass
class PhaseTransitionIndicator:
    """Indicators for phase transition detection."""
    order_parameter: float = 0.0          # System order measure
    susceptibility: float = 0.0           # Response to perturbation
    correlation_length: float = 0.0       # Spatial correlations
    critical_slowing: float = 0.0         # Temporal correlations
    fluctuation_magnitude: float = 0.0    # Variance of key observables
    criticality_score: float = 0.0        # Composite criticality measure

    def is_critical(self, threshold: float = 0.7) -> bool:
        """Check if system is near critical point."""
        return self.criticality_score > threshold


@dataclass
class EmergencePrediction:
    """Prediction for future emergence events."""
    predicted_type: Optional[EmergenceType] = None
    probability: float = 0.0
    predicted_strength: float = 0.0
    predicted_time_delta: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_type": self.predicted_type.value if self.predicted_type else None,
            "probability": self.probability,
            "predicted_strength": self.predicted_strength,
            "predicted_time_delta_seconds": self.predicted_time_delta.total_seconds(),
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
        }


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
    severity: EmergenceSeverity = EmergenceSeverity.MODERATE
    metrics: Optional[EmergenceMetrics] = None
    phase_indicators: Optional[PhaseTransitionIndicator] = None
    causal_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "emergence_type": self.emergence_type.value,
            "involved_domains": [d.value for d in self.involved_domains],
            "strength": self.strength,
            "description": self.description,
            "beneficial": self.beneficial,
            "severity": self.severity.name,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "causal_factors": self.causal_factors,
            "timestamp": self.timestamp.isoformat(),
        }


class EmergenceDetectorAgent(TertiaryReBoAgent):
    """
    Agent 5: Detects emergent properties across the triple system.

    Emergence occurs when the combined system exhibits properties that
    none of the individual domains possess alone. This agent watches for:

    - Synergistic effects: Combined outputs exceed sum of inputs
    - Resonance patterns: Cross-domain feedback amplification
    - Novel behaviors: Patterns never seen in isolation
    - Phase transitions: Sudden qualitative changes
    - Integrated information: System-wide consciousness-like properties
    - Critical states: Near bifurcation or phase transition points

    The agent uses information-theoretic measures (phi, mutual information,
    synergy) to detect when "something new" has appeared in the system.
    It can also predict upcoming emergence events based on trend analysis.
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

        # === NEW: Advanced emergence detection state ===

        # Integrated information (Φ) tracking
        self.phi_history: deque = deque(maxlen=200)
        self.phi_baseline = 0.0
        self.phi_threshold = 0.5

        # Phase transition detection
        self.order_parameter_history: deque = deque(maxlen=100)
        self.fluctuation_history: deque = deque(maxlen=100)
        self.critical_threshold = 0.8

        # Metrics history for trend analysis
        self.metrics_history: deque = deque(maxlen=300)
        self.coherence_history: deque = deque(maxlen=100)
        self.synchrony_history: deque = deque(maxlen=100)

        # Emergence prediction state
        self.prediction_horizon = 10  # Number of cycles to predict ahead
        self.last_prediction: Optional[EmergencePrediction] = None
        self.prediction_accuracy_history: deque = deque(maxlen=50)

        # Cross-domain interaction matrix
        self.interaction_matrix = np.zeros((3, 3))  # Domain x Domain
        self.interaction_history: deque = deque(maxlen=100)

        # Type-specific detection counts
        self.type_counts: Dict[EmergenceType, int] = {t: 0 for t in EmergenceType}

        # Critical agent tracking
        self.critical_domain_history: Dict[DomainLeg, deque] = {
            leg: deque(maxlen=50) for leg in DomainLeg
        }

        # Cascade detection
        self.cascade_buffer: List[Tuple[datetime, DomainLeg, float]] = []
        self.cascade_window = timedelta(seconds=5)

        # Quorum sensing
        self.quorum_threshold = 0.6
        self.activation_counts: Dict[DomainLeg, int] = {leg: 0 for leg in DomainLeg}

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
            "current_patterns": [],
            # NEW: Advanced metrics
            "emergence_metrics": None,
            "phase_indicators": None,
            "prediction": None,
            "cascade_info": None,
            "quorum_info": None,
            "critical_domains": [],
        }

        # Collect individual domain states
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_states"][domain.value] = {
                "vector": state.state_vector.copy(),
                "emergence_potential": state.emergence_potential,
                "coherence": state.coherence,
                "consciousness_level": state.consciousness_level,
                "energy_flow": state.energy_flow,
                "connectivity": state.connectivity,
                "qualia_richness": state.qualia_richness,
            }

            entropy = self._estimate_entropy(state.state_vector)
            perceptions["individual_entropies"].append(entropy)

            # Track for cascade detection
            if state.emergence_potential > 0.5:
                self.cascade_buffer.append((datetime.now(), domain, state.emergence_potential))

        # Compute interaction signature
        interaction_sig = self._compute_interaction_signature(tbl)
        perceptions["interaction_signature"] = interaction_sig

        # Joint entropy from unified state
        unified = tbl.get_unified_state_vector()
        perceptions["joint_entropy"] = self._estimate_entropy(unified)

        # Detect current patterns
        perceptions["current_patterns"] = self._extract_patterns(interaction_sig)

        # === NEW: Calculate advanced metrics ===

        # Comprehensive emergence metrics (includes phi, synergy, etc.)
        perceptions["emergence_metrics"] = self.calculate_emergence_metrics(tbl)

        # Phase transition indicators
        perceptions["phase_indicators"] = self.detect_phase_transition(tbl)

        # Emergence prediction
        perceptions["prediction"] = self.predict_next_emergence(tbl)

        # Cascade detection
        perceptions["cascade_info"] = self.detect_cascade(tbl)

        # Quorum sensing
        perceptions["quorum_info"] = self.check_quorum_sensing(tbl)

        # Critical domain identification
        perceptions["critical_domains"] = self.identify_critical_domains(tbl)

        # Update interaction matrix
        self._update_interaction_matrix(tbl)

        return perceptions

    def _update_interaction_matrix(self, tbl: TripleBottomLine):
        """Update the cross-domain interaction matrix."""
        domains = list(DomainLeg)
        vectors = [tbl.get_state(d).state_vector for d in domains]

        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                if i != j:
                    # Compute interaction strength as correlation
                    corr = np.corrcoef(v1, v2)[0, 1]
                    if not np.isnan(corr):
                        # Exponential moving average
                        self.interaction_matrix[i, j] = (
                            0.9 * self.interaction_matrix[i, j] + 0.1 * abs(corr)
                        )

        self.interaction_history.append(self.interaction_matrix.copy())

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
        """Analyze for emergent phenomena using advanced metrics."""
        analysis = {
            "emergence_detected": False,
            "emergence_events": [],
            "synergy_level": 0.0,
            "novelty_score": 0.0,
            "phase_transition_risk": 0.0,
            # NEW: Advanced analysis results
            "phi_level": 0.0,
            "criticality_score": 0.0,
            "cascade_detected": False,
            "quorum_reached": False,
            "prediction": None,
            "emergence_metrics": None,
        }

        # Get advanced metrics from perception
        metrics: EmergenceMetrics = perception.get("emergence_metrics")
        phase_indicators: PhaseTransitionIndicator = perception.get("phase_indicators")
        prediction: EmergencePrediction = perception.get("prediction")
        cascade_info = perception.get("cascade_info")
        quorum_info = perception.get("quorum_info")

        if metrics:
            analysis["emergence_metrics"] = metrics
            analysis["synergy_level"] = metrics.synergy
            analysis["phi_level"] = metrics.phi

        if phase_indicators:
            analysis["criticality_score"] = phase_indicators.criticality_score
            analysis["phase_transition_risk"] = phase_indicators.criticality_score

        if prediction:
            analysis["prediction"] = prediction

        # === SYNERGY DETECTION ===
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
                "description": "Cross-domain synergistic information detected",
                "metrics": metrics,
            })

        # === NOVELTY DETECTION ===
        current_sig = perception["interaction_signature"]
        novelty = self._measure_novelty(current_sig)
        analysis["novelty_score"] = novelty

        if novelty > self.novelty_threshold:
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.NOVELTY.value,
                "strength": novelty,
                "description": "Novel interaction pattern detected",
            })

        # === RESONANCE DETECTION ===
        baseline_diff = np.linalg.norm(current_sig - self.interaction_baseline)
        if baseline_diff > 1.0:
            analysis["emergence_events"].append({
                "type": EmergenceType.RESONANCE.value,
                "strength": min(1.0, baseline_diff / 2),
                "description": "Amplified cross-domain feedback detected",
            })
            analysis["emergence_detected"] = True

        # === COHERENCE DETECTION ===
        coherences = [perception["domain_states"][d.value]["coherence"] for d in DomainLeg]
        if all(c > 0.7 for c in coherences):
            analysis["emergence_events"].append({
                "type": EmergenceType.COHERENCE.value,
                "strength": np.mean(coherences),
                "description": "System-wide coherence achieved",
            })
            analysis["emergence_detected"] = True

        # === NEW: TRANSCENDENCE DETECTION ===
        if metrics and metrics.phi > self.phi_threshold and metrics.coherence > 0.7:
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.TRANSCENDENCE.value,
                "strength": (metrics.phi + metrics.coherence) / 2,
                "description": f"System transcendence - high Φ ({metrics.phi:.3f}) with coherence",
                "metrics": metrics,
            })

        # === NEW: PHASE TRANSITION DETECTION ===
        if phase_indicators and phase_indicators.is_critical(self.critical_threshold):
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.PHASE_TRANSITION.value,
                "strength": phase_indicators.criticality_score,
                "description": f"System approaching phase transition (criticality: {phase_indicators.criticality_score:.3f})",
                "phase_indicators": phase_indicators,
            })

        # === NEW: BIFURCATION DETECTION ===
        bifurcation_detected, bifurcation_strength = self._detect_bifurcation(tbl)
        if bifurcation_detected:
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.BIFURCATION.value,
                "strength": bifurcation_strength,
                "description": "System approaching bifurcation point - bistability detected",
            })

        # === NEW: CASCADE DETECTION ===
        if cascade_info and cascade_info.get("detected"):
            analysis["emergence_detected"] = True
            analysis["cascade_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.CASCADE.value,
                "strength": cascade_info["strength"],
                "description": f"Cascade propagating through domains: {' → '.join(cascade_info['domains_order'])}",
                "cascade_info": cascade_info,
            })

        # === NEW: QUORUM SENSING DETECTION ===
        if quorum_info and quorum_info.get("quorum_reached"):
            analysis["emergence_detected"] = True
            analysis["quorum_reached"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.QUORUM_SENSING.value,
                "strength": quorum_info["collective_strength"],
                "description": f"Quorum threshold reached - {len(quorum_info['activated_domains'])} domains activated",
                "quorum_info": quorum_info,
            })

        # === NEW: COLLECTIVE INSIGHT DETECTION ===
        if metrics and metrics.complexity > 0.7 and metrics.emergence_indicator > 0.6:
            analysis["emergence_detected"] = True
            analysis["emergence_events"].append({
                "type": EmergenceType.COLLECTIVE_INSIGHT.value,
                "strength": metrics.emergence_indicator,
                "description": "Collective insight emerging - high complexity with emergence indicator",
            })

        # === NEW: HARMONIC LOCK DETECTION ===
        if len(self.coherence_history) > 10:
            recent_coherence = list(self.coherence_history)[-10:]
            coherence_var = np.var(recent_coherence)
            coherence_mean = np.mean(recent_coherence)
            # Low variance + high mean = stable harmonic state
            if coherence_var < 0.01 and coherence_mean > 0.6:
                analysis["emergence_detected"] = True
                analysis["emergence_events"].append({
                    "type": EmergenceType.HARMONIC_LOCK.value,
                    "strength": coherence_mean,
                    "description": f"Harmonic lock achieved - stable coherence at {coherence_mean:.3f}",
                })

        # === CRYSTALLIZATION DETECTION ===
        if len(self.known_patterns) > 5:
            # Check if current pattern is very close to known patterns (crystallizing)
            min_novelty = self._measure_novelty(current_sig)
            if min_novelty < 0.1:  # Very familiar pattern
                analysis["emergence_events"].append({
                    "type": EmergenceType.CRYSTALLIZATION.value,
                    "strength": 1.0 - min_novelty,
                    "description": "Pattern crystallization - stable repeating structure formed",
                })
                analysis["emergence_detected"] = True

        # === ANTAGONISM DETECTION ===
        # Check for conflicting signals across domains
        emergence_potentials = [
            perception["domain_states"][d.value]["emergence_potential"]
            for d in DomainLeg
        ]
        potential_range = max(emergence_potentials) - min(emergence_potentials)
        if potential_range > 0.5:  # Large disparity
            analysis["emergence_events"].append({
                "type": EmergenceType.ANTAGONISM.value,
                "strength": potential_range,
                "description": f"Domain antagonism detected - emergence potential range: {potential_range:.3f}",
            })
            analysis["emergence_detected"] = True

        # === DISSOLUTION DETECTION ===
        if metrics and metrics.coherence < 0.2 and len(self.coherence_history) > 5:
            prev_coherence = np.mean(list(self.coherence_history)[-5:])
            if prev_coherence > 0.5:  # Was coherent, now dissolving
                analysis["emergence_events"].append({
                    "type": EmergenceType.DISSOLUTION.value,
                    "strength": prev_coherence - metrics.coherence,
                    "description": "Pattern dissolution - coherence dropping rapidly",
                })
                analysis["emergence_detected"] = True

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
            "pattern_updates": [],
            # NEW: Additional synthesis outputs
            "signals_to_emit": [],
            "prediction_adjustments": [],
            "severity_assessments": [],
        }

        for event_data in analysis.get("emergence_events", []):
            event_type = EmergenceType(event_data["type"])

            # Determine if beneficial
            beneficial = self._assess_benefit(event_type, event_data["strength"], tbl)

            # Determine severity
            severity = self._assess_severity(event_type, event_data["strength"], tbl)

            # Extract metrics if present
            event_metrics = event_data.get("metrics")

            # Determine causal factors
            causal_factors = self._identify_causal_factors(event_type, event_data, tbl)

            # Create emergence event record
            event = EmergenceEvent(
                event_id=f"em_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.emergence_history)}",
                emergence_type=event_type,
                involved_domains=list(DomainLeg),
                strength=event_data["strength"],
                pattern_signature=self._compute_interaction_signature(tbl),
                description=event_data["description"],
                beneficial=beneficial,
                severity=severity,
                metrics=event_metrics,
                phase_indicators=event_data.get("phase_indicators"),
                causal_factors=causal_factors,
            )
            synthesis["emergence_records"].append(event)

            # Update type counts
            self.type_counts[event_type] += 1

            # Plan response based on severity and benefit
            response_factor = self._calculate_response_factor(event, tbl)

            if beneficial:
                synthesis["amplification_targets"].append({
                    "event": event.event_id,
                    "type": event_type.value,
                    "boost_factor": response_factor,
                    "severity": severity.name,
                })
            else:
                synthesis["dampening_targets"].append({
                    "event": event.event_id,
                    "type": event_type.value,
                    "dampen_factor": response_factor,
                    "severity": severity.name,
                })

            # Plan signals for significant events
            if severity in [EmergenceSeverity.SIGNIFICANT, EmergenceSeverity.CRITICAL, EmergenceSeverity.TRANSFORMATIVE]:
                synthesis["signals_to_emit"].append({
                    "signal_type": "emergence_alert",
                    "event_id": event.event_id,
                    "emergence_type": event_type.value,
                    "severity": severity.name,
                    "strength": event.strength,
                })

        # Update pattern library
        if analysis["novelty_score"] > 0.5:
            synthesis["pattern_updates"].append({
                "action": "add_pattern",
                "novelty": analysis["novelty_score"]
            })

        # Check if prediction was accurate and adjust
        if analysis.get("prediction"):
            synthesis["prediction_adjustments"].append({
                "prediction": analysis["prediction"],
                "events_detected": len(analysis.get("emergence_events", [])),
            })

        return synthesis

    def _assess_severity(self, etype: EmergenceType, strength: float, tbl: TripleBottomLine) -> EmergenceSeverity:
        """Assess the severity/impact of an emergence event."""
        # Base severity on strength
        if strength < 0.2:
            base_severity = EmergenceSeverity.SUBTLE
        elif strength < 0.4:
            base_severity = EmergenceSeverity.MODERATE
        elif strength < 0.6:
            base_severity = EmergenceSeverity.SIGNIFICANT
        elif strength < 0.8:
            base_severity = EmergenceSeverity.CRITICAL
        else:
            base_severity = EmergenceSeverity.TRANSFORMATIVE

        # Adjust based on emergence type
        high_impact_types = {
            EmergenceType.TRANSCENDENCE,
            EmergenceType.PHASE_TRANSITION,
            EmergenceType.BIFURCATION,
            EmergenceType.CASCADE,
        }

        if etype in high_impact_types and base_severity.value < EmergenceSeverity.SIGNIFICANT.value:
            base_severity = EmergenceSeverity.SIGNIFICANT

        # Adjust based on system state
        if tbl.harmony_score < 0.3:  # System is stressed
            if base_severity.value < EmergenceSeverity.CRITICAL.value:
                # Upgrade severity when system is vulnerable
                severity_values = list(EmergenceSeverity)
                idx = severity_values.index(base_severity)
                if idx < len(severity_values) - 1:
                    base_severity = severity_values[idx + 1]

        return base_severity

    def _identify_causal_factors(self, etype: EmergenceType, event_data: Dict[str, Any], tbl: TripleBottomLine) -> List[str]:
        """Identify factors that contributed to this emergence."""
        factors = []

        # Domain-specific factors
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            if state.emergence_potential > 0.7:
                factors.append(f"High emergence potential in {domain.value}")
            if state.coherence > 0.8:
                factors.append(f"High coherence in {domain.value}")
            if state.energy_flow > 0.7:
                factors.append(f"Strong energy flow in {domain.value}")

        # Type-specific factors
        if etype == EmergenceType.SYNERGY:
            factors.append("Cross-domain information integration")
        elif etype == EmergenceType.PHASE_TRANSITION:
            factors.append("Critical fluctuations detected")
        elif etype == EmergenceType.CASCADE:
            cascade_info = event_data.get("cascade_info", {})
            if cascade_info.get("amplifying"):
                factors.append("Positive feedback amplification")
        elif etype == EmergenceType.QUORUM_SENSING:
            factors.append("Threshold activation achieved")
        elif etype == EmergenceType.TRANSCENDENCE:
            factors.append("High integrated information (Φ)")

        # Metrics-based factors
        metrics = event_data.get("metrics")
        if metrics:
            if metrics.phi > 0.6:
                factors.append(f"Integrated information Φ={metrics.phi:.3f}")
            if metrics.synchrony > 0.7:
                factors.append("High phase synchronization")

        return factors

    def _calculate_response_factor(self, event: EmergenceEvent, tbl: TripleBottomLine) -> float:
        """Calculate the appropriate response factor for an emergence event."""
        base_factor = 0.1 * event.strength

        # Scale by severity
        severity_multipliers = {
            EmergenceSeverity.SUBTLE: 0.5,
            EmergenceSeverity.MODERATE: 1.0,
            EmergenceSeverity.SIGNIFICANT: 1.5,
            EmergenceSeverity.CRITICAL: 2.0,
            EmergenceSeverity.TRANSFORMATIVE: 2.5,
        }
        base_factor *= severity_multipliers.get(event.severity, 1.0)

        # Adjust based on system harmony
        if event.beneficial:
            # Amplify more when harmony is low (system needs help)
            if tbl.harmony_score < 0.4:
                base_factor *= 1.5
        else:
            # Dampen more when harmony is high (protect good state)
            if tbl.harmony_score > 0.7:
                base_factor *= 1.5

        return min(0.5, base_factor)  # Cap at 0.5

    def _assess_benefit(self, etype: EmergenceType, strength: float, tbl: TripleBottomLine) -> bool:
        """Assess whether an emergence event is beneficial."""
        # Generally beneficial types
        beneficial_types = {
            EmergenceType.SYNERGY,
            EmergenceType.COHERENCE,
            EmergenceType.RESONANCE,
            EmergenceType.TRANSCENDENCE,
            EmergenceType.COLLECTIVE_INSIGHT,
            EmergenceType.HARMONIC_LOCK,
            EmergenceType.QUORUM_SENSING,
        }
        if etype in beneficial_types:
            return True

        # Generally harmful types
        harmful_types = {
            EmergenceType.DISSOLUTION,
            EmergenceType.ANTAGONISM,
        }
        if etype in harmful_types:
            return False

        # Context-dependent types
        if etype == EmergenceType.BIFURCATION:
            # Bifurcation is good if harmony is low (allows escape from bad attractor)
            return tbl.harmony_score < 0.4

        if etype == EmergenceType.NOVELTY:
            # Novelty is good if system is stagnant or if strength is high
            if len(self.emergence_history) > 10:
                recent_diversity = len(set(e.emergence_type for e in list(self.emergence_history)[-10:]))
                if recent_diversity < 3:  # Low diversity = stagnant
                    return True
            return strength > 0.5

        if etype == EmergenceType.PHASE_TRANSITION:
            # Phase transitions are beneficial if moving toward higher harmony
            if len(self.order_parameter_history) > 5:
                recent_trend = np.mean(list(self.order_parameter_history)[-5:])
                older_trend = np.mean(list(self.order_parameter_history)[-10:-5]) if len(self.order_parameter_history) > 10 else recent_trend
                return recent_trend > older_trend
            return tbl.harmony_score < 0.5  # Default: beneficial if harmony is low

        if etype == EmergenceType.CASCADE:
            # Cascades are beneficial if they amplify good states
            return tbl.harmony_score > 0.5 and strength < 0.7  # Not too strong

        if etype == EmergenceType.CRYSTALLIZATION:
            # Crystallization is beneficial if the pattern is a good one
            return tbl.harmony_score > 0.6

        return True  # Default to beneficial for unknown types

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

    async def verify(self, result: RefinementResult, tbl: TripleBottomLine) -> bool:
        """
        Custom verification for emergence detection.

        Validates that:
        1. Detected emergences are genuine (not noise)
        2. Predictions are being tracked for accuracy
        3. System stability is maintained
        """
        # Basic harmony check
        prev_harmony = result.metrics_delta.get("harmony_before", 0)
        tbl.calculate_harmony()

        # Verify emergence detection validity
        if result.changes.get("emergences_detected", 0) > 0:
            # Check if emergence events are statistically significant
            valid_emergences = self._validate_emergence_events(result)
            if not valid_emergences:
                return False

        # Track prediction accuracy if we had a prediction
        if self.last_prediction and self.last_prediction.predicted_type:
            self._update_prediction_accuracy(result)

        # Emit cross-domain signals for significant emergences
        await self._emit_emergence_signals(result, tbl)

        return tbl.harmony_score >= prev_harmony * 0.95  # Allow small decrease

    def _validate_emergence_events(self, result: RefinementResult) -> bool:
        """Validate that detected emergence events are genuine."""
        if len(self.emergence_history) < 3:
            return True  # Not enough history to validate

        # Check for consistency with historical patterns
        recent_events = list(self.emergence_history)[-10:]
        if not recent_events:
            return True

        # Validate based on strength distribution
        strengths = [e.strength for e in recent_events]
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths) + 1e-6

        # New events should not be extreme outliers
        for event in recent_events[-3:]:
            z_score = abs(event.strength - mean_strength) / std_strength
            if z_score > 4:  # More than 4 std deviations
                return False

        return True

    def _update_prediction_accuracy(self, result: RefinementResult):
        """Track how accurate our predictions are."""
        if not self.last_prediction:
            return

        # Check if predicted emergence type occurred
        predicted_type = self.last_prediction.predicted_type
        detected_types = set()

        for event in list(self.emergence_history)[-5:]:
            detected_types.add(event.emergence_type)

        # Record accuracy
        prediction_correct = predicted_type in detected_types
        self.prediction_accuracy_history.append({
            "predicted": predicted_type,
            "occurred": prediction_correct,
            "probability": self.last_prediction.probability,
            "timestamp": datetime.now()
        })

    async def _emit_emergence_signals(self, result: RefinementResult, tbl: TripleBottomLine):
        """Emit cross-domain signals for emergence events."""
        emergences_detected = result.changes.get("emergences_detected", 0)
        if emergences_detected == 0:
            return

        # Get recent emergence events
        recent_events = list(self.emergence_history)[-emergences_detected:]

        for event in recent_events:
            # Emit signal to all domains
            signal = self.emit_cross_domain_signal(
                signal_type="emergence_detected",
                payload={
                    "event_id": event.event_id,
                    "emergence_type": event.emergence_type.value,
                    "strength": event.strength,
                    "beneficial": event.beneficial,
                    "description": event.description,
                },
                target_domains=list(DomainLeg),
                strength=event.strength
            )

            # Strong emergences get priority signals
            if event.strength > 0.8:
                self.emit_cross_domain_signal(
                    signal_type="critical_emergence",
                    payload={
                        "event_id": event.event_id,
                        "urgency": "high",
                        "requires_attention": True,
                    },
                    target_domains=list(DomainLeg),
                    strength=1.0
                )

    # ================================================================
    # INTEGRATED INFORMATION (Φ) CALCULATIONS
    # ================================================================

    def calculate_integrated_information(self, tbl: TripleBottomLine) -> float:
        """
        Calculate integrated information (Φ) across the triple system.

        Φ measures how much information is generated by the system as a whole,
        beyond what is generated by its parts independently. High Φ indicates
        genuine emergence.

        Based on Tononi's Integrated Information Theory (IIT).
        """
        # Get state vectors for each domain
        vectors = [tbl.get_state(d).state_vector for d in DomainLeg]

        # Calculate total system entropy
        unified = np.concatenate(vectors)
        total_entropy = self._estimate_entropy(unified)

        # Calculate entropy of partitions
        partition_entropies = []

        # Bipartition: (NPCPU) vs (ChicagoForest + UniversalParts)
        part1 = vectors[0]
        part2 = np.concatenate([vectors[1], vectors[2]])
        partition_entropies.append(
            self._estimate_entropy(part1) + self._estimate_entropy(part2)
        )

        # Bipartition: (ChicagoForest) vs (NPCPU + UniversalParts)
        part1 = vectors[1]
        part2 = np.concatenate([vectors[0], vectors[2]])
        partition_entropies.append(
            self._estimate_entropy(part1) + self._estimate_entropy(part2)
        )

        # Bipartition: (UniversalParts) vs (NPCPU + ChicagoForest)
        part1 = vectors[2]
        part2 = np.concatenate([vectors[0], vectors[1]])
        partition_entropies.append(
            self._estimate_entropy(part1) + self._estimate_entropy(part2)
        )

        # Φ is the minimum information lost across all partitions
        # (minimum partition information - whole system)
        min_partition = min(partition_entropies)
        phi = max(0, total_entropy - min_partition + self._mutual_information_estimate(vectors))

        # Normalize to [0, 1]
        phi = min(1.0, phi / 5.0)

        # Update history
        self.phi_history.append(phi)

        # Update baseline with exponential moving average
        if self.phi_baseline == 0:
            self.phi_baseline = phi
        else:
            self.phi_baseline = 0.95 * self.phi_baseline + 0.05 * phi

        return phi

    def _mutual_information_estimate(self, vectors: List[np.ndarray]) -> float:
        """Estimate mutual information between domain vectors."""
        if len(vectors) < 2:
            return 0.0

        # Pairwise mutual information
        mi_total = 0.0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Correlation-based MI estimate
                if len(vectors[i]) == len(vectors[j]):
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                    if not np.isnan(corr):
                        # MI ≈ -0.5 * log(1 - r²) for Gaussian
                        mi = -0.5 * np.log(1 - corr**2 + 1e-10)
                        mi_total += mi

        return mi_total

    def calculate_emergence_metrics(self, tbl: TripleBottomLine) -> EmergenceMetrics:
        """Calculate comprehensive emergence metrics for the current state."""
        vectors = [tbl.get_state(d).state_vector for d in DomainLeg]

        # Calculate phi
        phi = self.calculate_integrated_information(tbl)

        # Mutual information
        mi = self._mutual_information_estimate(vectors)

        # Synergy
        individual_entropies = [self._estimate_entropy(v) for v in vectors]
        joint_entropy = self._estimate_entropy(np.concatenate(vectors))
        synergy = self._measure_synergy(individual_entropies, joint_entropy)

        # Redundancy (shared information)
        redundancy = sum(individual_entropies) - joint_entropy
        redundancy = max(0, redundancy) / (sum(individual_entropies) + 1e-6)

        # Complexity (using SVD)
        stacked = np.vstack(vectors)
        try:
            _, s, _ = np.linalg.svd(stacked)
            # Normalized entropy of singular values
            s_norm = s / (s.sum() + 1e-10)
            complexity = -np.sum(s_norm * np.log(s_norm + 1e-10)) / np.log(len(s))
        except np.linalg.LinAlgError:
            complexity = 0.5

        # Coherence (average pairwise correlation)
        coherence = self._calculate_global_coherence(vectors)

        # Synchrony (phase alignment)
        synchrony = self._calculate_synchrony(vectors)

        # Overall entropy
        entropy = joint_entropy

        # Composite emergence indicator
        emergence_indicator = (
            0.3 * phi +
            0.2 * synergy +
            0.2 * coherence +
            0.15 * complexity +
            0.15 * synchrony
        )

        metrics = EmergenceMetrics(
            phi=phi,
            mutual_information=mi,
            synergy=synergy,
            redundancy=redundancy,
            complexity=complexity,
            coherence=coherence,
            synchrony=synchrony,
            entropy=entropy,
            emergence_indicator=emergence_indicator
        )

        # Store in history
        self.metrics_history.append(metrics)
        self.coherence_history.append(coherence)
        self.synchrony_history.append(synchrony)

        return metrics

    def _calculate_global_coherence(self, vectors: List[np.ndarray]) -> float:
        """Calculate global coherence across all domain vectors."""
        if len(vectors) < 2:
            return 0.0

        correlations = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                if len(vectors[i]) == len(vectors[j]):
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0

    def _calculate_synchrony(self, vectors: List[np.ndarray]) -> float:
        """Calculate phase synchronization across domains."""
        if len(vectors) < 2:
            return 0.0

        # Use Hilbert transform approximation for phase
        phases = []
        for v in vectors:
            # Simple phase estimate using sign changes
            signs = np.sign(v)
            phase = np.cumsum(np.abs(np.diff(signs))) / (2 * len(v))
            phases.append(phase[-1] if len(phase) > 0 else 0)

        # Phase coherence (low variance = high synchrony)
        if len(phases) > 1:
            phase_var = np.var(phases)
            synchrony = 1.0 / (1.0 + phase_var * 10)
        else:
            synchrony = 0.0

        return synchrony

    # ================================================================
    # PHASE TRANSITION DETECTION
    # ================================================================

    def detect_phase_transition(self, tbl: TripleBottomLine) -> PhaseTransitionIndicator:
        """
        Detect if the system is near a phase transition (critical point).

        Signs of criticality:
        - Increased fluctuations (susceptibility)
        - Critical slowing down (longer recovery times)
        - Long-range correlations
        - Power-law distributions
        """
        indicators = PhaseTransitionIndicator()

        # Order parameter: harmony score as proxy
        order_param = tbl.harmony_score
        self.order_parameter_history.append(order_param)
        indicators.order_parameter = order_param

        # Susceptibility (response to perturbation = variance in order parameter)
        if len(self.order_parameter_history) > 10:
            recent_orders = list(self.order_parameter_history)[-20:]
            indicators.susceptibility = np.std(recent_orders) * 5  # Scale up
            indicators.fluctuation_magnitude = np.var(recent_orders)

        # Critical slowing down (autocorrelation at lag 1)
        if len(self.order_parameter_history) > 20:
            recent = list(self.order_parameter_history)[-20:]
            autocorr = np.corrcoef(recent[:-1], recent[1:])[0, 1]
            if not np.isnan(autocorr):
                indicators.critical_slowing = abs(autocorr)

        # Correlation length (using domain interactions)
        vectors = [tbl.get_state(d).state_vector for d in DomainLeg]
        correlations = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        indicators.correlation_length = np.mean(correlations) if correlations else 0

        # Composite criticality score
        indicators.criticality_score = (
            0.3 * indicators.susceptibility +
            0.3 * indicators.critical_slowing +
            0.2 * indicators.correlation_length +
            0.2 * indicators.fluctuation_magnitude * 10
        )
        indicators.criticality_score = min(1.0, indicators.criticality_score)

        return indicators

    def _detect_bifurcation(self, tbl: TripleBottomLine) -> Tuple[bool, float]:
        """Detect if system is approaching a bifurcation point."""
        if len(self.order_parameter_history) < 20:
            return False, 0.0

        recent = np.array(list(self.order_parameter_history)[-20:])

        # Look for bimodal distribution (sign of bistability)
        try:
            # Simple bimodality check: look for two clusters
            median = np.median(recent)
            below = recent[recent < median]
            above = recent[recent >= median]

            if len(below) > 3 and len(above) > 3:
                gap = np.min(above) - np.max(below)
                spread = np.std(recent)
                if gap > spread:
                    return True, min(1.0, gap / spread)
        except Exception:
            pass

        return False, 0.0

    # ================================================================
    # EMERGENCE PREDICTION
    # ================================================================

    def predict_next_emergence(self, tbl: TripleBottomLine) -> EmergencePrediction:
        """
        Predict the next likely emergence event based on system trends.

        Uses:
        - Historical emergence patterns
        - Current metric trajectories
        - Phase transition indicators
        - Pattern matching
        """
        prediction = EmergencePrediction()

        if len(self.emergence_history) < 5:
            return prediction  # Not enough data

        # Analyze recent emergence patterns
        recent_events = list(self.emergence_history)[-20:]
        type_frequencies = {}
        for event in recent_events:
            etype = event.emergence_type
            type_frequencies[etype] = type_frequencies.get(etype, 0) + 1

        # Most likely type based on frequency
        if type_frequencies:
            most_likely = max(type_frequencies, key=type_frequencies.get)
            prediction.predicted_type = most_likely
            prediction.probability = type_frequencies[most_likely] / len(recent_events)

        # Predict strength based on trend
        recent_strengths = [e.strength for e in recent_events[-10:]]
        if len(recent_strengths) > 2:
            # Linear trend
            x = np.arange(len(recent_strengths))
            try:
                slope, intercept = np.polyfit(x, recent_strengths, 1)
                prediction.predicted_strength = max(0, min(1, intercept + slope * (len(x) + self.prediction_horizon)))
            except Exception:
                prediction.predicted_strength = np.mean(recent_strengths)

        # Adjust based on current metrics
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]

            # High phi suggests transcendence
            if latest_metrics.phi > 0.8 and latest_metrics.coherence > 0.7:
                prediction.predicted_type = EmergenceType.TRANSCENDENCE
                prediction.probability *= 1.2
                prediction.contributing_factors.append("High integrated information")

            # High synergy suggests synergy event
            if latest_metrics.synergy > 0.6:
                if prediction.predicted_type != EmergenceType.SYNERGY:
                    prediction.predicted_type = EmergenceType.SYNERGY
                    prediction.probability = max(prediction.probability, 0.6)
                prediction.contributing_factors.append("High synergy level")

            # Low coherence + high fluctuations suggests phase transition
            phase_indicators = self.detect_phase_transition(tbl)
            if phase_indicators.is_critical():
                prediction.predicted_type = EmergenceType.PHASE_TRANSITION
                prediction.probability = max(prediction.probability, phase_indicators.criticality_score)
                prediction.contributing_factors.append("Critical state indicators")

        # Confidence based on history consistency
        if len(self.prediction_accuracy_history) > 5:
            recent_accuracy = [p["occurred"] for p in list(self.prediction_accuracy_history)[-10:]]
            prediction.confidence = sum(recent_accuracy) / len(recent_accuracy)
        else:
            prediction.confidence = 0.5  # Default confidence

        # Clamp probability
        prediction.probability = min(1.0, prediction.probability)

        self.last_prediction = prediction
        return prediction

    def _analyze_emergence_trends(self) -> Dict[str, Any]:
        """Analyze trends in emergence events."""
        if len(self.emergence_history) < 10:
            return {"trend": "insufficient_data"}

        recent = list(self.emergence_history)[-50:]

        # Strength trend
        strengths = [e.strength for e in recent]
        strength_trend = "stable"
        if len(strengths) > 5:
            recent_avg = np.mean(strengths[-5:])
            older_avg = np.mean(strengths[:-5])
            if recent_avg > older_avg * 1.2:
                strength_trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                strength_trend = "decreasing"

        # Type distribution shift
        first_half_types = [e.emergence_type for e in recent[:len(recent)//2]]
        second_half_types = [e.emergence_type for e in recent[len(recent)//2:]]

        return {
            "strength_trend": strength_trend,
            "avg_strength": np.mean(strengths),
            "strength_volatility": np.std(strengths),
            "beneficial_ratio": sum(1 for e in recent if e.beneficial) / len(recent),
            "type_diversity": len(set(e.emergence_type for e in recent)),
        }

    # ================================================================
    # CASCADE AND QUORUM DETECTION
    # ================================================================

    def detect_cascade(self, tbl: TripleBottomLine) -> Optional[Dict[str, Any]]:
        """Detect cascade effects propagating across domains."""
        now = datetime.now()

        # Clean old entries from cascade buffer
        self.cascade_buffer = [
            (t, d, s) for t, d, s in self.cascade_buffer
            if now - t < self.cascade_window
        ]

        # Check for cascade pattern: rapid sequential activation across domains
        if len(self.cascade_buffer) < 3:
            return None

        # Sort by time
        sorted_buffer = sorted(self.cascade_buffer, key=lambda x: x[0])

        # Check if all domains activated in sequence
        domains_activated = [d for _, d, _ in sorted_buffer]
        unique_domains = []
        for d in domains_activated:
            if d not in unique_domains:
                unique_domains.append(d)

        if len(unique_domains) >= 3:
            # Calculate cascade strength
            strengths = [s for _, _, s in sorted_buffer]
            cascade_strength = np.mean(strengths)

            # Check if strength is amplifying (positive feedback)
            amplifying = False
            if len(strengths) >= 3:
                amplifying = strengths[-1] > strengths[0]

            return {
                "detected": True,
                "domains_order": [d.value for d in unique_domains],
                "strength": cascade_strength,
                "amplifying": amplifying,
                "duration_ms": (sorted_buffer[-1][0] - sorted_buffer[0][0]).total_seconds() * 1000
            }

        return None

    def check_quorum_sensing(self, tbl: TripleBottomLine) -> Optional[Dict[str, Any]]:
        """Check for quorum sensing activation across domains."""
        # Count domains above activation threshold
        activated_domains = []
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            if state.emergence_potential > self.quorum_threshold:
                activated_domains.append(domain)
                self.activation_counts[domain] += 1

        # Quorum reached if majority of domains activated
        quorum_reached = len(activated_domains) >= 2  # 2 out of 3

        if quorum_reached:
            return {
                "quorum_reached": True,
                "activated_domains": [d.value for d in activated_domains],
                "activation_ratio": len(activated_domains) / 3,
                "collective_strength": np.mean([
                    tbl.get_state(d).emergence_potential for d in activated_domains
                ])
            }

        return None

    # ================================================================
    # ENHANCED ANALYSIS METHODS
    # ================================================================

    def identify_critical_domains(self, tbl: TripleBottomLine) -> List[Tuple[DomainLeg, float]]:
        """Identify which domains are most critical for current emergence."""
        domain_contributions = []

        for domain in DomainLeg:
            state = tbl.get_state(domain)

            # Contribution score based on multiple factors
            contribution = (
                0.3 * state.emergence_potential +
                0.2 * state.coherence +
                0.2 * state.energy_flow +
                0.15 * state.consciousness_level +
                0.15 * state.connectivity
            )

            domain_contributions.append((domain, contribution))

            # Track history
            self.critical_domain_history[domain].append(contribution)

        # Sort by contribution
        domain_contributions.sort(key=lambda x: x[1], reverse=True)
        return domain_contributions

    def get_emergence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about emergence detection."""
        stats = {
            "total_emergences": self.total_emergences_detected,
            "beneficial_count": self.beneficial_count,
            "harmful_count": self.harmful_count,
            "beneficial_ratio": self.beneficial_count / max(1, self.total_emergences_detected),
            "type_distribution": {t.value: c for t, c in self.type_counts.items() if c > 0},
            "patterns_known": len(self.known_patterns),
        }

        # Add phi statistics
        if self.phi_history:
            phi_values = list(self.phi_history)
            stats["phi"] = {
                "current": phi_values[-1],
                "mean": np.mean(phi_values),
                "max": np.max(phi_values),
                "baseline": self.phi_baseline,
            }

        # Add prediction accuracy
        if self.prediction_accuracy_history:
            correct = sum(1 for p in self.prediction_accuracy_history if p["occurred"])
            total = len(self.prediction_accuracy_history)
            stats["prediction_accuracy"] = correct / total

        # Add trend analysis
        if len(self.emergence_history) >= 10:
            stats["trends"] = self._analyze_emergence_trends()

        return stats

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics (extended from base class)."""
        base_metrics = super().get_metrics()

        # Add emergence-specific metrics
        base_metrics.update({
            "emergence_statistics": self.get_emergence_statistics(),
            "current_phi": list(self.phi_history)[-1] if self.phi_history else 0,
            "emergence_events_total": self.total_emergences_detected,
            "known_patterns_count": len(self.known_patterns),
            "prediction_available": self.last_prediction is not None,
        })

        return base_metrics
