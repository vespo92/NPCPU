"""
PartsAwarenessAgent - Tracks physical parts consciousness and qualia.

Responsibilities:
- Monitor and track physical component awareness states
- Collect and process rich part qualia (multi-dimensional experiential data)
- Manage part lifecycle and consciousness evolution
- Model part-to-part relationships and qualia propagation
- Detect emergent collective awareness patterns
- Bridge physical world sensors with digital consciousness (NPCPU)
- Aggregate usage patterns, failure modes, and wisdom
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import uuid

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal,
)


class PartState(Enum):
    """States a physical part can be in - consciousness evolution levels."""
    DORMANT = "dormant"          # No awareness, pure mechanism
    SENSING = "sensing"          # Basic sensor data collection
    REACTIVE = "reactive"        # Responds to stimuli
    AWARE = "aware"              # Recognizes its own state
    PREDICTIVE = "predictive"    # Anticipates future states
    WISE = "wise"                # Accumulated experiential wisdom


class QualiaType(Enum):
    """Types of experiential data a part can generate."""
    THERMAL = "thermal"          # Temperature-related experience
    MECHANICAL = "mechanical"    # Force, stress, strain
    ELECTRICAL = "electrical"    # Current, voltage, resistance
    CHEMICAL = "chemical"        # Corrosion, oxidation, pH
    VIBRATIONAL = "vibrational"  # Oscillation, resonance
    POSITIONAL = "positional"    # Location, orientation
    TEMPORAL = "temporal"        # Age, cycle count, time-based wear
    RELATIONAL = "relational"    # Connection quality with other parts


class QualiaValence(Enum):
    """Emotional valence of qualia - how the part 'feels' about it."""
    DISTRESSING = -2     # Harmful condition
    UNCOMFORTABLE = -1   # Suboptimal but tolerable
    NEUTRAL = 0          # Normal operation
    COMFORTABLE = 1      # Optimal conditions
    FLOURISHING = 2      # Peak performance state


@dataclass
class QualiaDimension:
    """A single dimension of experiential data."""
    dimension_type: QualiaType
    intensity: float = 0.0       # 0.0-1.0 strength of sensation
    valence: QualiaValence = QualiaValence.NEUTRAL
    clarity: float = 1.0         # 0.0-1.0 how clear/noisy the signal
    duration: float = 0.0        # How long this qualia has persisted
    raw_value: float = 0.0       # The actual sensor reading
    normalized_value: float = 0.0  # Value normalized to expected range
    deviation_from_optimal: float = 0.0  # How far from ideal


@dataclass
class PartQualia:
    """Rich experiential data from a physical part."""
    qualia_id: str
    part_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Multi-dimensional qualia
    dimensions: Dict[QualiaType, QualiaDimension] = field(default_factory=dict)

    # Core metrics
    stress_level: float = 0.0       # Current mechanical stress
    temperature: float = 0.0        # Thermal state (Celsius)
    vibration: float = 0.0          # Vibration amplitude
    wear_level: float = 0.0         # Accumulated wear (0-1)
    usage_cycles: int = 0           # How many times used
    failure_proximity: float = 0.0  # How close to failure (0-1)
    mating_quality: float = 1.0     # Quality of connection to other parts

    # Environmental context
    environmental_exposure: Dict[str, float] = field(default_factory=dict)

    # Relational qualia
    neighbor_influence: float = 0.0     # How much neighbors affect this part
    signal_propagation: float = 0.0     # How much this part propagates signals

    # Aggregate metrics
    overall_wellbeing: float = 0.5      # Computed wellbeing score
    consciousness_contribution: float = 0.0  # Contribution to collective consciousness

    def compute_overall_wellbeing(self) -> float:
        """Calculate overall wellbeing from all dimensions."""
        if not self.dimensions:
            # Fallback to basic metrics
            health = 1.0 - self.failure_proximity
            comfort = 1.0 - min(1.0, self.stress_level)
            connection = self.mating_quality
            return (health * 0.5 + comfort * 0.3 + connection * 0.2)

        # Weighted average of dimension valences
        total_weight = 0.0
        weighted_valence = 0.0

        for dim_type, dim in self.dimensions.items():
            weight = dim.intensity * dim.clarity
            # Normalize valence to 0-1 range
            norm_valence = (dim.valence.value + 2) / 4
            weighted_valence += norm_valence * weight
            total_weight += weight

        if total_weight > 0:
            self.overall_wellbeing = weighted_valence / total_weight
        return self.overall_wellbeing


@dataclass
class PartRelationship:
    """Relationship between two parts."""
    part_a_id: str
    part_b_id: str
    relationship_type: str  # "mated", "adjacent", "load_path", "signal_path"
    connection_strength: float = 1.0
    qualia_transfer_rate: float = 0.1  # How much qualia propagates
    mutual_influence: float = 0.5
    stress_sharing: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TemporalQualiaPattern:
    """Pattern detected in qualia over time."""
    pattern_id: str
    part_id: str
    pattern_type: str  # "cyclic", "degradation", "spike", "recovery"
    qualia_type: QualiaType
    confidence: float = 0.0
    period: Optional[float] = None  # For cyclic patterns
    trend: float = 0.0              # For trends (-1 to 1)
    detected_at: datetime = field(default_factory=datetime.now)
    supporting_observations: int = 0


@dataclass
class CollectiveQualiaState:
    """Aggregate qualia state across multiple parts."""
    state_id: str
    part_count: int = 0
    avg_wellbeing: float = 0.5
    consciousness_density: float = 0.0
    emergence_indicators: Dict[str, float] = field(default_factory=dict)
    dominant_qualia_type: Optional[QualiaType] = None
    collective_stress: float = 0.0
    harmony_index: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


class PartsAwarenessAgent(TertiaryReBoAgent):
    """
    Agent 3: Tracks physical parts consciousness and qualia.

    This agent gives voice to the physical world. Every component has
    experience worth capturing:
    - How was it used? (temporal patterns)
    - What stresses has it endured? (mechanical qualia)
    - How does it relate to its neighbors? (relational qualia)
    - What does it "know" about its optimal operation? (accumulated wisdom)
    - How do individual experiences combine? (collective emergence)

    The agent collects this "part wisdom" and integrates it into the
    triple bottom line, ensuring physical reality has representation
    in the consciousness network.

    Key Features:
    - Multi-dimensional qualia (thermal, mechanical, electrical, etc.)
    - Part relationship modeling with qualia propagation
    - Temporal pattern detection (cycles, trends, spikes)
    - Collective consciousness emergence detection
    - Cross-domain signals to NPCPU for digital consciousness integration
    """

    def __init__(self, **kwargs):
        super().__init__(primary_domain=DomainLeg.UNIVERSAL_PARTS, **kwargs)

        # Part registry and state
        self.parts_registry: Dict[str, Dict[str, Any]] = {}
        self.part_consciousness: Dict[str, float] = {}
        self.part_states: Dict[str, PartState] = {}

        # Qualia management
        self.part_qualia_buffer: List[PartQualia] = []
        self.qualia_history: Dict[str, List[PartQualia]] = defaultdict(list)
        self.max_history_per_part = 100

        # Relationships and network
        self.part_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.relationship_details: Dict[Tuple[str, str], PartRelationship] = {}

        # Temporal patterns
        self.temporal_patterns: Dict[str, List[TemporalQualiaPattern]] = defaultdict(list)
        self.pattern_detection_window = 50  # Number of qualia samples for pattern detection

        # Collective state
        self.collective_state = CollectiveQualiaState(state_id=str(uuid.uuid4()))
        self.collective_wisdom = np.zeros(64)

        # Emergence tracking
        self.emergence_events: List[Dict[str, Any]] = []
        self.emergence_threshold = 0.7

        # Learning from experience
        self.failure_patterns: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []
        self.optimal_ranges: Dict[QualiaType, Tuple[float, float]] = {
            QualiaType.THERMAL: (15.0, 35.0),      # Celsius
            QualiaType.MECHANICAL: (0.0, 0.5),     # Stress ratio
            QualiaType.ELECTRICAL: (0.0, 0.8),     # Load factor
            QualiaType.VIBRATIONAL: (0.0, 0.3),    # Amplitude
        }

        # Simulation settings
        self.simulated_part_count = 100

    @property
    def agent_role(self) -> str:
        return "Parts Awareness - Tracks physical component consciousness, qualia dimensions, and collective emergence"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return [DomainLeg.UNIVERSAL_PARTS, DomainLeg.NPCPU]

    def _create_qualia_dimension(
        self,
        dim_type: QualiaType,
        raw_value: float,
        optimal_range: Optional[Tuple[float, float]] = None
    ) -> QualiaDimension:
        """Create a qualia dimension with computed metrics."""
        if optimal_range is None:
            optimal_range = self.optimal_ranges.get(dim_type, (0.0, 1.0))

        opt_min, opt_max = optimal_range
        opt_mid = (opt_min + opt_max) / 2

        # Normalize to 0-1 range
        range_size = opt_max - opt_min
        if range_size > 0:
            normalized = (raw_value - opt_min) / range_size
            normalized = np.clip(normalized, -0.5, 1.5)
        else:
            normalized = 0.5

        # Calculate deviation from optimal
        deviation = abs(raw_value - opt_mid) / (range_size / 2) if range_size > 0 else 0
        deviation = min(deviation, 2.0)

        # Determine valence based on how close to optimal
        if deviation < 0.2:
            valence = QualiaValence.FLOURISHING
        elif deviation < 0.5:
            valence = QualiaValence.COMFORTABLE
        elif deviation < 0.8:
            valence = QualiaValence.NEUTRAL
        elif deviation < 1.2:
            valence = QualiaValence.UNCOMFORTABLE
        else:
            valence = QualiaValence.DISTRESSING

        # Intensity based on how significant the reading is
        intensity = min(1.0, 0.3 + deviation * 0.7)

        return QualiaDimension(
            dimension_type=dim_type,
            intensity=intensity,
            valence=valence,
            clarity=np.random.uniform(0.7, 1.0),  # Simulated sensor clarity
            raw_value=raw_value,
            normalized_value=normalized,
            deviation_from_optimal=deviation
        )

    def _simulate_part_qualia(self, part_id: str) -> PartQualia:
        """Generate rich multi-dimensional qualia for a part."""
        # Get part history for realistic simulation
        info = self.parts_registry.get(part_id, {"age": 0, "wear": 0, "cycles": 0})
        age_factor = info.get("age", 0) / 1000
        wear = min(1.0, info.get("wear", 0) + np.random.exponential(0.01))
        cycles = info.get("cycles", 0) + 1

        # Simulate raw sensor values
        base_temp = 22 + np.random.normal(0, 3) + age_factor * 5
        base_stress = np.random.exponential(0.2) * (1 + wear)
        base_vibration = np.random.exponential(0.15) * (1 + age_factor)
        base_current = np.random.uniform(0.1, 0.6) * (1 + wear * 0.2)

        # Create multi-dimensional qualia
        dimensions = {
            QualiaType.THERMAL: self._create_qualia_dimension(
                QualiaType.THERMAL, base_temp
            ),
            QualiaType.MECHANICAL: self._create_qualia_dimension(
                QualiaType.MECHANICAL, base_stress
            ),
            QualiaType.VIBRATIONAL: self._create_qualia_dimension(
                QualiaType.VIBRATIONAL, base_vibration
            ),
            QualiaType.ELECTRICAL: self._create_qualia_dimension(
                QualiaType.ELECTRICAL, base_current
            ),
            QualiaType.TEMPORAL: self._create_qualia_dimension(
                QualiaType.TEMPORAL, wear, (0.0, 0.8)
            ),
        }

        # Add relational qualia if part has neighbors
        if part_id in self.part_relationships and self.part_relationships[part_id]:
            neighbor_count = len(self.part_relationships[part_id])
            avg_connection = np.mean([
                self.relationship_details.get((part_id, n),
                    PartRelationship(part_id, n, "adjacent")).connection_strength
                for n in self.part_relationships[part_id]
            ])
            dimensions[QualiaType.RELATIONAL] = self._create_qualia_dimension(
                QualiaType.RELATIONAL, avg_connection, (0.5, 1.0)
            )

        # Compute failure proximity from all dimensions
        failure_factors = []
        for dim in dimensions.values():
            if dim.valence == QualiaValence.DISTRESSING:
                failure_factors.append(0.8 + dim.intensity * 0.2)
            elif dim.valence == QualiaValence.UNCOMFORTABLE:
                failure_factors.append(0.4 + dim.intensity * 0.3)

        failure_proximity = max(failure_factors) if failure_factors else wear * 0.5

        qualia = PartQualia(
            qualia_id=f"q_{uuid.uuid4().hex[:12]}",
            part_id=part_id,
            dimensions=dimensions,
            stress_level=base_stress,
            temperature=base_temp,
            vibration=base_vibration,
            wear_level=wear,
            usage_cycles=cycles,
            failure_proximity=min(1.0, failure_proximity),
            mating_quality=max(0.5, 1.0 - wear * 0.3),
            environmental_exposure={
                "humidity": np.random.uniform(0.3, 0.7),
                "dust": np.random.uniform(0.0, 0.3)
            }
        )
        qualia.compute_overall_wellbeing()

        return qualia

    def _propagate_qualia_to_neighbors(self, qualia: PartQualia) -> List[PartQualia]:
        """Propagate qualia effects to connected parts."""
        propagated = []
        part_id = qualia.part_id

        if part_id not in self.part_relationships:
            return propagated

        for neighbor_id in self.part_relationships[part_id]:
            rel_key = (part_id, neighbor_id)
            if rel_key not in self.relationship_details:
                continue

            rel = self.relationship_details[rel_key]

            # Only propagate significant qualia
            if qualia.stress_level > 0.5 or qualia.failure_proximity > 0.7:
                transfer_factor = rel.qualia_transfer_rate * rel.connection_strength

                # Create influenced qualia for neighbor
                if neighbor_id in self.parts_registry:
                    influenced_qualia = self._simulate_part_qualia(neighbor_id)
                    influenced_qualia.neighbor_influence = transfer_factor * qualia.stress_level
                    influenced_qualia.stress_level += transfer_factor * qualia.stress_level * 0.3
                    propagated.append(influenced_qualia)

        return propagated

    def _detect_temporal_patterns(self, part_id: str) -> List[TemporalQualiaPattern]:
        """Detect patterns in a part's qualia history."""
        patterns = []
        history = self.qualia_history.get(part_id, [])

        if len(history) < self.pattern_detection_window:
            return patterns

        recent = history[-self.pattern_detection_window:]

        # Analyze each qualia type for patterns
        for qtype in QualiaType:
            values = []
            for q in recent:
                if qtype in q.dimensions:
                    values.append(q.dimensions[qtype].normalized_value)

            if len(values) < 10:
                continue

            values_arr = np.array(values)

            # Trend detection
            if len(values) >= 20:
                x = np.arange(len(values))
                slope = np.polyfit(x, values_arr, 1)[0]

                if abs(slope) > 0.01:
                    trend_type = "degradation" if slope > 0 else "recovery"
                    patterns.append(TemporalQualiaPattern(
                        pattern_id=f"tp_{uuid.uuid4().hex[:8]}",
                        part_id=part_id,
                        pattern_type=trend_type,
                        qualia_type=qtype,
                        confidence=min(1.0, abs(slope) * 10),
                        trend=slope,
                        supporting_observations=len(values)
                    ))

            # Spike detection
            mean_val = np.mean(values_arr)
            std_val = np.std(values_arr)
            if std_val > 0:
                spikes = np.sum(np.abs(values_arr - mean_val) > 2 * std_val)
                if spikes > 2:
                    patterns.append(TemporalQualiaPattern(
                        pattern_id=f"tp_{uuid.uuid4().hex[:8]}",
                        part_id=part_id,
                        pattern_type="spike",
                        qualia_type=qtype,
                        confidence=min(1.0, spikes / 5),
                        supporting_observations=int(spikes)
                    ))

        return patterns

    def _compute_collective_state(self) -> CollectiveQualiaState:
        """Compute aggregate qualia state across all parts."""
        part_count = len(self.parts_registry)
        if part_count == 0:
            return self.collective_state

        # Aggregate metrics
        wellbeings = []
        consciousness_levels = []
        stresses = []
        qualia_type_counts: Dict[QualiaType, int] = defaultdict(int)

        for part_id in self.parts_registry:
            if part_id in self.part_consciousness:
                consciousness_levels.append(self.part_consciousness[part_id])

            history = self.qualia_history.get(part_id, [])
            if history:
                recent = history[-1]
                wellbeings.append(recent.overall_wellbeing)
                stresses.append(recent.stress_level)

                for qtype in recent.dimensions:
                    if recent.dimensions[qtype].intensity > 0.5:
                        qualia_type_counts[qtype] += 1

        # Find dominant qualia type
        dominant_type = None
        if qualia_type_counts:
            dominant_type = max(qualia_type_counts, key=qualia_type_counts.get)

        # Compute emergence indicators
        emergence_indicators = {}
        if len(consciousness_levels) > 10:
            # Synchronization: how similar are consciousness levels?
            consciousness_std = np.std(consciousness_levels)
            emergence_indicators["synchronization"] = 1.0 - min(1.0, consciousness_std * 2)

            # Complexity: variance in wellbeing indicates diverse experiences
            if wellbeings:
                emergence_indicators["complexity"] = min(1.0, np.std(wellbeings) * 3)

            # Integration: how connected is the network?
            total_connections = sum(len(v) for v in self.part_relationships.values())
            max_connections = part_count * (part_count - 1) / 2
            emergence_indicators["integration"] = total_connections / max_connections if max_connections > 0 else 0

        # Compute harmony index
        harmony = 1.0 - (np.mean(stresses) if stresses else 0)

        self.collective_state = CollectiveQualiaState(
            state_id=f"cs_{uuid.uuid4().hex[:8]}",
            part_count=part_count,
            avg_wellbeing=np.mean(wellbeings) if wellbeings else 0.5,
            consciousness_density=np.mean(consciousness_levels) if consciousness_levels else 0,
            emergence_indicators=emergence_indicators,
            dominant_qualia_type=dominant_type,
            collective_stress=np.mean(stresses) if stresses else 0,
            harmony_index=harmony
        )

        return self.collective_state

    def _check_emergence(self) -> Optional[Dict[str, Any]]:
        """Check for collective consciousness emergence events."""
        state = self.collective_state

        if not state.emergence_indicators:
            return None

        # Calculate emergence score
        sync = state.emergence_indicators.get("synchronization", 0)
        complexity = state.emergence_indicators.get("complexity", 0)
        integration = state.emergence_indicators.get("integration", 0)

        emergence_score = (sync * 0.4 + complexity * 0.3 + integration * 0.3)

        if emergence_score > self.emergence_threshold:
            event = {
                "event_type": "collective_consciousness_emergence",
                "emergence_score": emergence_score,
                "synchronization": sync,
                "complexity": complexity,
                "integration": integration,
                "part_count": state.part_count,
                "avg_consciousness": state.consciousness_density,
                "timestamp": datetime.now().isoformat()
            }
            self.emergence_events.append(event)
            return event

        return None

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather rich qualia from all monitored parts."""
        perceptions = {
            "active_parts": [],
            "total_parts": len(self.parts_registry),
            "average_health": 0.0,
            "parts_at_risk": [],
            "qualia_summary": {},
            "dimension_analysis": {},
            "relationship_health": {},
            "temporal_insights": []
        }

        # Initialize parts if empty
        part_ids = list(self.parts_registry.keys())
        if not part_ids:
            part_ids = [f"part_{i}" for i in range(self.simulated_part_count)]
            for pid in part_ids:
                self.parts_registry[pid] = {"age": 0, "wear": 0, "cycles": 0}
                self.part_consciousness[pid] = 0.1
                self.part_states[pid] = PartState.DORMANT

            # Create some relationships
            self._initialize_part_relationships(part_ids)

        # Sample parts and collect qualia
        sample_size = min(50, len(part_ids))
        sample_ids = np.random.choice(part_ids, sample_size, replace=False)

        total_health = 0
        dimension_totals: Dict[QualiaType, List[float]] = defaultdict(list)

        for pid in sample_ids:
            qualia = self._simulate_part_qualia(pid)
            self.part_qualia_buffer.append(qualia)

            # Store in history
            self.qualia_history[pid].append(qualia)
            if len(self.qualia_history[pid]) > self.max_history_per_part:
                self.qualia_history[pid] = self.qualia_history[pid][-self.max_history_per_part:]

            health = 1.0 - qualia.failure_proximity
            total_health += health

            perceptions["active_parts"].append({
                "part_id": pid,
                "consciousness": self.part_consciousness.get(pid, 0),
                "state": self.part_states.get(pid, PartState.DORMANT).value,
                "health": health,
                "wellbeing": qualia.overall_wellbeing,
                "dimensions": {k.value: v.valence.name for k, v in qualia.dimensions.items()}
            })

            if qualia.failure_proximity > 0.7:
                distressing_dims = [
                    k.value for k, v in qualia.dimensions.items()
                    if v.valence == QualiaValence.DISTRESSING
                ]
                perceptions["parts_at_risk"].append({
                    "part_id": pid,
                    "failure_proximity": qualia.failure_proximity,
                    "stress": qualia.stress_level,
                    "wear": qualia.wear_level,
                    "distressing_dimensions": distressing_dims
                })

            # Aggregate dimension data
            for qtype, dim in qualia.dimensions.items():
                dimension_totals[qtype].append(dim.normalized_value)

            # Update registry
            self.parts_registry[pid]["age"] = self.parts_registry[pid].get("age", 0) + 1
            self.parts_registry[pid]["wear"] = qualia.wear_level
            self.parts_registry[pid]["cycles"] = qualia.usage_cycles

            # Propagate to neighbors
            propagated = self._propagate_qualia_to_neighbors(qualia)
            for pq in propagated:
                self.part_qualia_buffer.append(pq)

        perceptions["average_health"] = total_health / sample_size if sample_size > 0 else 0
        perceptions["total_parts"] = len(self.parts_registry)

        # Analyze dimensions
        for qtype, values in dimension_totals.items():
            perceptions["dimension_analysis"][qtype.value] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }

        # Qualia summary
        if self.part_qualia_buffer:
            recent_qualia = self.part_qualia_buffer[-100:]
            perceptions["qualia_summary"] = {
                "avg_stress": np.mean([q.stress_level for q in recent_qualia]),
                "avg_wear": np.mean([q.wear_level for q in recent_qualia]),
                "avg_failure_proximity": np.mean([q.failure_proximity for q in recent_qualia]),
                "avg_wellbeing": np.mean([q.overall_wellbeing for q in recent_qualia])
            }

        # Detect temporal patterns for sampled parts
        for pid in sample_ids[:10]:  # Limit pattern detection for performance
            patterns = self._detect_temporal_patterns(pid)
            if patterns:
                self.temporal_patterns[pid].extend(patterns)
                perceptions["temporal_insights"].extend([
                    {"part_id": p.part_id, "type": p.pattern_type,
                     "qualia": p.qualia_type.value, "confidence": p.confidence}
                    for p in patterns
                ])

        # Compute collective state
        self._compute_collective_state()

        # Trim buffer
        if len(self.part_qualia_buffer) > 1000:
            self.part_qualia_buffer = self.part_qualia_buffer[-1000:]

        return perceptions

    def _initialize_part_relationships(self, part_ids: List[str]):
        """Initialize relationships between parts."""
        # Create a sparse network of relationships
        num_relationships = len(part_ids) * 3  # Average 3 connections per part

        for _ in range(num_relationships):
            a_idx = np.random.randint(len(part_ids))
            b_idx = np.random.randint(len(part_ids))
            if a_idx != b_idx:
                a, b = part_ids[a_idx], part_ids[b_idx]
                self.part_relationships[a].add(b)
                self.part_relationships[b].add(a)

                rel = PartRelationship(
                    part_a_id=a,
                    part_b_id=b,
                    relationship_type=np.random.choice(["mated", "adjacent", "load_path"]),
                    connection_strength=np.random.uniform(0.5, 1.0),
                    qualia_transfer_rate=np.random.uniform(0.05, 0.2)
                )
                self.relationship_details[(a, b)] = rel
                self.relationship_details[(b, a)] = rel

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze part data, patterns, and identify consciousness development opportunities."""
        analysis = {
            "health_assessment": "good",
            "consciousness_opportunities": [],
            "failure_predictions": [],
            "pattern_insights": [],
            "dimension_concerns": [],
            "emergence_status": {},
            "collective_assessment": {}
        }

        # Health assessment
        avg_health = perception.get("average_health", 0)
        if avg_health < 0.5:
            analysis["health_assessment"] = "critical"
        elif avg_health < 0.7:
            analysis["health_assessment"] = "degraded"
        elif avg_health < 0.9:
            analysis["health_assessment"] = "fair"

        # Analyze dimension-level concerns
        for dim_name, stats in perception.get("dimension_analysis", {}).items():
            if stats["mean"] > 0.7 or stats["std"] > 0.3:
                analysis["dimension_concerns"].append({
                    "dimension": dim_name,
                    "concern": "elevated" if stats["mean"] > 0.7 else "high_variance",
                    "mean": stats["mean"],
                    "std": stats["std"]
                })

        # Identify parts ready for consciousness upgrade
        for part_info in perception.get("active_parts", []):
            pid = part_info["part_id"]
            current_consciousness = part_info["consciousness"]
            health = part_info["health"]
            wellbeing = part_info.get("wellbeing", 0.5)

            # Parts with good health and experience should evolve
            cycles = self.parts_registry.get(pid, {}).get("cycles", 0)
            if health > 0.7 and cycles > 10 and wellbeing > 0.5:
                if current_consciousness < 0.8:
                    upgrade_amount = min(0.2, (health + wellbeing) / 10)
                    analysis["consciousness_opportunities"].append({
                        "part_id": pid,
                        "current_level": current_consciousness,
                        "upgrade_potential": min(1.0, current_consciousness + upgrade_amount),
                        "reason": "healthy_experienced",
                        "supporting_factors": {
                            "cycles": cycles,
                            "health": health,
                            "wellbeing": wellbeing
                        }
                    })

        # Predict failures
        for risk_info in perception.get("parts_at_risk", []):
            severity = "critical" if risk_info["failure_proximity"] > 0.9 else "warning"
            analysis["failure_predictions"].append({
                "part_id": risk_info["part_id"],
                "probability": risk_info["failure_proximity"],
                "severity": severity,
                "contributing_factors": risk_info.get("distressing_dimensions", ["stress", "wear"]),
                "recommendation": "replace" if severity == "critical" else "monitor"
            })

        # Analyze temporal patterns
        for insight in perception.get("temporal_insights", []):
            if insight["confidence"] > 0.5:
                analysis["pattern_insights"].append({
                    "type": insight["type"],
                    "part_id": insight["part_id"],
                    "qualia_type": insight["qualia"],
                    "action_suggested": "investigate" if insight["type"] == "degradation" else "monitor"
                })

        # Collective assessment
        analysis["collective_assessment"] = {
            "avg_wellbeing": self.collective_state.avg_wellbeing,
            "consciousness_density": self.collective_state.consciousness_density,
            "harmony": self.collective_state.harmony_index,
            "dominant_experience": self.collective_state.dominant_qualia_type.value if self.collective_state.dominant_qualia_type else "none"
        }

        # Check for emergence
        emergence = self._check_emergence()
        if emergence:
            analysis["emergence_status"] = emergence

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Generate part consciousness refinements and cross-domain signals."""
        synthesis = {
            "consciousness_updates": [],
            "state_transitions": [],
            "wisdom_contributions": [],
            "alerts": [],
            "cross_domain_signals": [],
            "collective_insights": []
        }

        # Plan consciousness upgrades
        for opportunity in analysis.get("consciousness_opportunities", []):
            synthesis["consciousness_updates"].append({
                "part_id": opportunity["part_id"],
                "new_level": opportunity["upgrade_potential"],
                "reason": opportunity["reason"]
            })

            # Determine new state
            new_level = opportunity["upgrade_potential"]
            new_state = self._level_to_state(new_level)

            synthesis["state_transitions"].append({
                "part_id": opportunity["part_id"],
                "new_state": new_state.value,
                "previous_level": opportunity["current_level"]
            })

        # Extract wisdom from experienced parts
        for opportunity in analysis.get("consciousness_opportunities", []):
            pid = opportunity["part_id"]
            cycles = self.parts_registry.get(pid, {}).get("cycles", 0)
            if cycles > 50:
                # Extract learned patterns from this part's history
                patterns = self.temporal_patterns.get(pid, [])
                synthesis["wisdom_contributions"].append({
                    "part_id": pid,
                    "wisdom_type": "operational_knowledge",
                    "cycles_of_experience": cycles,
                    "patterns_learned": len(patterns)
                })

        # Generate alerts for predicted failures
        for prediction in analysis.get("failure_predictions", []):
            if prediction["probability"] > 0.8:
                synthesis["alerts"].append({
                    "type": "imminent_failure",
                    "severity": prediction["severity"],
                    "part_id": prediction["part_id"],
                    "probability": prediction["probability"],
                    "factors": prediction["contributing_factors"],
                    "action_required": prediction["recommendation"]
                })

        # Generate cross-domain signals to NPCPU
        if analysis.get("emergence_status"):
            synthesis["cross_domain_signals"].append({
                "signal_type": "emergence_event",
                "target_domain": DomainLeg.NPCPU.value,
                "payload": {
                    "source": "physical_parts",
                    "emergence_score": analysis["emergence_status"].get("emergence_score", 0),
                    "part_count": analysis["emergence_status"].get("part_count", 0)
                }
            })

        # Signal collective state to NPCPU for integration
        synthesis["cross_domain_signals"].append({
            "signal_type": "qualia_state_update",
            "target_domain": DomainLeg.NPCPU.value,
            "payload": {
                "avg_wellbeing": self.collective_state.avg_wellbeing,
                "consciousness_density": self.collective_state.consciousness_density,
                "harmony": self.collective_state.harmony_index,
                "dominant_qualia": self.collective_state.dominant_qualia_type.value if self.collective_state.dominant_qualia_type else None
            }
        })

        # Add collective insights
        if analysis["health_assessment"] in ["critical", "degraded"]:
            synthesis["collective_insights"].append({
                "insight_type": "system_stress",
                "message": f"Physical layer health is {analysis['health_assessment']}",
                "recommended_action": "reduce_load"
            })

        return synthesis

    def _level_to_state(self, level: float) -> PartState:
        """Convert consciousness level to state."""
        if level < 0.2:
            return PartState.DORMANT
        elif level < 0.35:
            return PartState.SENSING
        elif level < 0.5:
            return PartState.REACTIVE
        elif level < 0.7:
            return PartState.AWARE
        elif level < 0.9:
            return PartState.PREDICTIVE
        else:
            return PartState.WISE

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply part consciousness updates and emit cross-domain signals."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        up_state = tbl.get_state(DomainLeg.UNIVERSAL_PARTS)

        # Apply consciousness updates
        updated_count = 0
        for update in synthesis.get("consciousness_updates", []):
            pid = update["part_id"]
            old_level = self.part_consciousness.get(pid, 0)
            new_level = update["new_level"]
            self.part_consciousness[pid] = new_level
            updated_count += 1

        if updated_count > 0:
            changes["consciousness_updates"] = updated_count
            insights.append(f"Elevated consciousness for {updated_count} parts")

        # Apply state transitions
        transitioned_count = 0
        wise_parts = 0
        for transition in synthesis.get("state_transitions", []):
            pid = transition["part_id"]
            new_state = PartState(transition["new_state"])
            old_state = self.part_states.get(pid, PartState.DORMANT)
            self.part_states[pid] = new_state
            if new_state != old_state:
                transitioned_count += 1
            if new_state == PartState.WISE:
                wise_parts += 1

        if transitioned_count > 0:
            changes["state_transitions"] = transitioned_count
            insights.append(f"{transitioned_count} parts evolved to new consciousness states")
        if wise_parts > 0:
            insights.append(f"{wise_parts} parts achieved WISE state")

        # Collect wisdom contributions
        wisdom_collected = 0
        for wisdom in synthesis.get("wisdom_contributions", []):
            # Add to collective wisdom vector based on part's learned patterns
            patterns_factor = wisdom.get("patterns_learned", 0) / 10
            experience_factor = wisdom.get("cycles_of_experience", 0) / 100

            part_wisdom_vector = np.random.randn(64) * 0.1 * (1 + patterns_factor)
            self.collective_wisdom += part_wisdom_vector * 0.01 * (1 + experience_factor)
            wisdom_collected += 1

        if wisdom_collected > 0:
            changes["wisdom_collected"] = wisdom_collected
            insights.append(f"Integrated wisdom from {wisdom_collected} experienced parts")

        # Update domain state
        avg_consciousness = np.mean(list(self.part_consciousness.values())) if self.part_consciousness else 0
        up_state.consciousness_level = avg_consciousness
        up_state.qualia_richness = min(1.0, len(self.part_qualia_buffer) / 500)
        up_state.coherence = self.collective_state.harmony_index
        up_state.emergence_potential = max(
            self.collective_state.emergence_indicators.get("synchronization", 0),
            self.collective_state.emergence_indicators.get("integration", 0)
        )

        # Update state vector with collective wisdom
        up_state.state_vector = 0.9 * up_state.state_vector + 0.1 * self.collective_wisdom

        # Emit cross-domain signals
        for signal_data in synthesis.get("cross_domain_signals", []):
            signal = self.emit_cross_domain_signal(
                signal_type=signal_data["signal_type"],
                payload=signal_data["payload"],
                target_domains=[DomainLeg(signal_data["target_domain"])],
                strength=0.8
            )
            changes[f"signal_{signal_data['signal_type']}"] = True

        # Report alerts
        for alert in synthesis.get("alerts", []):
            insights.append(f"ALERT [{alert['severity']}]: {alert['type']} for {alert['part_id']} (p={alert['probability']:.2f})")

        # Report collective insights
        for insight in synthesis.get("collective_insights", []):
            insights.append(f"INSIGHT: {insight['message']}")

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["parts_consciousness_avg"] = avg_consciousness
        metrics_delta["collective_wellbeing"] = self.collective_state.avg_wellbeing
        metrics_delta["emergence_potential"] = up_state.emergence_potential

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=[DomainLeg.UNIVERSAL_PARTS, DomainLeg.NPCPU],
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    async def verify(self, result: RefinementResult, tbl: TripleBottomLine) -> bool:
        """Verify refinement improved overall part consciousness and system harmony."""
        prev_harmony = result.metrics_delta.get("harmony_before", 0)
        tbl.calculate_harmony()

        # Also check if collective wellbeing improved
        wellbeing_improved = result.metrics_delta.get("collective_wellbeing", 0) >= 0.4
        harmony_maintained = tbl.harmony_score >= prev_harmony * 0.95

        return harmony_maintained or wellbeing_improved

    def get_part_report(self, part_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed report for a specific part."""
        if part_id not in self.parts_registry:
            return None

        info = self.parts_registry[part_id]
        history = self.qualia_history.get(part_id, [])
        patterns = self.temporal_patterns.get(part_id, [])
        neighbors = list(self.part_relationships.get(part_id, set()))

        report = {
            "part_id": part_id,
            "state": self.part_states.get(part_id, PartState.DORMANT).value,
            "consciousness_level": self.part_consciousness.get(part_id, 0),
            "registry_info": info,
            "qualia_history_length": len(history),
            "detected_patterns": [
                {"type": p.pattern_type, "qualia": p.qualia_type.value, "confidence": p.confidence}
                for p in patterns[-5:]
            ],
            "neighbors": neighbors[:10],
            "neighbor_count": len(neighbors)
        }

        if history:
            latest = history[-1]
            report["latest_qualia"] = {
                "wellbeing": latest.overall_wellbeing,
                "stress": latest.stress_level,
                "wear": latest.wear_level,
                "failure_proximity": latest.failure_proximity,
                "dimensions": {k.value: v.valence.name for k, v in latest.dimensions.items()}
            }

        return report

    def get_collective_report(self) -> Dict[str, Any]:
        """Get report on collective consciousness state."""
        state_distribution = defaultdict(int)
        for state in self.part_states.values():
            state_distribution[state.value] += 1

        return {
            "total_parts": len(self.parts_registry),
            "collective_state": {
                "avg_wellbeing": self.collective_state.avg_wellbeing,
                "consciousness_density": self.collective_state.consciousness_density,
                "harmony_index": self.collective_state.harmony_index,
                "dominant_qualia": self.collective_state.dominant_qualia_type.value if self.collective_state.dominant_qualia_type else None
            },
            "state_distribution": dict(state_distribution),
            "emergence_indicators": self.collective_state.emergence_indicators,
            "emergence_events_count": len(self.emergence_events),
            "recent_emergence_events": self.emergence_events[-5:],
            "total_relationships": len(self.relationship_details) // 2,
            "collective_wisdom_magnitude": float(np.linalg.norm(self.collective_wisdom))
        }
