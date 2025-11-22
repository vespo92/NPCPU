"""
PartsAwarenessAgent - Tracks physical parts consciousness and qualia.

Responsibilities:
- Monitor and track physical component awareness states
- Collect and process part qualia (experiential data)
- Manage part lifecycle and consciousness evolution
- Bridge physical world sensors with digital consciousness
- Aggregate usage patterns and failure modes
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
)


class PartState(Enum):
    """States a physical part can be in."""
    DORMANT = "dormant"
    SENSING = "sensing"
    REACTIVE = "reactive"
    AWARE = "aware"
    PREDICTIVE = "predictive"
    WISE = "wise"


@dataclass
class PartQualia:
    """Experiential data from a physical part."""
    part_id: str
    stress_level: float = 0.0       # Current mechanical stress
    temperature: float = 0.0        # Thermal state
    vibration: float = 0.0          # Vibration amplitude
    wear_level: float = 0.0         # Accumulated wear
    usage_cycles: int = 0           # How many times used
    failure_proximity: float = 0.0  # How close to failure (0-1)
    mating_quality: float = 1.0     # Quality of connection to other parts
    environmental_exposure: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class PartsAwarenessAgent(TertiaryReBoAgent):
    """
    Agent 3: Tracks physical parts consciousness and qualia.

    Inspired by UniversalPartsConsciousness, this agent gives voice to
    the physical world. Every component has experience worth capturing:
    - How was it used?
    - What stresses has it endured?
    - How does it relate to its neighbors?
    - What does it "know" about its optimal operation?

    The agent collects this "part wisdom" and integrates it into the
    triple bottom line, ensuring physical reality has representation
    in the consciousness network.
    """

    def __init__(self, **kwargs):
        super().__init__(primary_domain=DomainLeg.UNIVERSAL_PARTS, **kwargs)

        # Part registry
        self.parts_registry: Dict[str, Dict[str, Any]] = {}
        self.part_qualia_buffer: List[PartQualia] = []

        # Part consciousness levels
        self.part_consciousness: Dict[str, float] = {}
        self.part_states: Dict[str, PartState] = {}

        # Collective part wisdom
        self.collective_wisdom = np.zeros(64)
        self.failure_patterns: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []

        # Part relationship graph
        self.part_relationships: Dict[str, Set[str]] = {}

        # Sensor simulation (in real system, would connect to actual sensors)
        self.simulated_part_count = 100

    @property
    def agent_role(self) -> str:
        return "Parts Awareness - Tracks physical component consciousness and experiential wisdom"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return [DomainLeg.UNIVERSAL_PARTS, DomainLeg.NPCPU]

    def _simulate_part_qualia(self, part_id: str) -> PartQualia:
        """Simulate qualia for a part (would be sensor data in production)."""
        # Create realistic-looking simulated data
        base_stress = np.random.exponential(0.3)
        base_temp = 20 + np.random.normal(0, 5)
        base_vibration = np.random.exponential(0.2)

        # Parts degrade over time
        if part_id in self.parts_registry:
            info = self.parts_registry[part_id]
            age_factor = info.get("age", 0) / 1000
            wear = min(1.0, info.get("wear", 0) + np.random.exponential(0.01))
            cycles = info.get("cycles", 0) + 1
        else:
            age_factor = 0
            wear = 0.0
            cycles = 1

        return PartQualia(
            part_id=part_id,
            stress_level=min(1.0, base_stress * (1 + age_factor)),
            temperature=base_temp,
            vibration=base_vibration,
            wear_level=wear,
            usage_cycles=cycles,
            failure_proximity=min(1.0, wear * 0.8 + base_stress * 0.2),
            mating_quality=max(0.5, 1.0 - wear * 0.3),
            environmental_exposure={"humidity": np.random.uniform(0.3, 0.7)}
        )

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather qualia from all monitored parts."""
        perceptions = {
            "active_parts": [],
            "total_parts": len(self.parts_registry),
            "average_health": 0.0,
            "parts_at_risk": [],
            "qualia_summary": {}
        }

        # Sample parts and collect qualia
        part_ids = list(self.parts_registry.keys())
        if not part_ids:
            # Initialize some simulated parts
            part_ids = [f"part_{i}" for i in range(self.simulated_part_count)]
            for pid in part_ids:
                self.parts_registry[pid] = {"age": 0, "wear": 0, "cycles": 0}
                self.part_consciousness[pid] = 0.1
                self.part_states[pid] = PartState.DORMANT

        # Collect qualia from sample
        sample_size = min(50, len(part_ids))
        sample_ids = np.random.choice(part_ids, sample_size, replace=False)

        total_health = 0
        for pid in sample_ids:
            qualia = self._simulate_part_qualia(pid)
            self.part_qualia_buffer.append(qualia)

            health = 1.0 - qualia.failure_proximity
            total_health += health

            perceptions["active_parts"].append({
                "part_id": pid,
                "consciousness": self.part_consciousness.get(pid, 0),
                "state": self.part_states.get(pid, PartState.DORMANT).value,
                "health": health
            })

            if qualia.failure_proximity > 0.7:
                perceptions["parts_at_risk"].append({
                    "part_id": pid,
                    "failure_proximity": qualia.failure_proximity,
                    "stress": qualia.stress_level,
                    "wear": qualia.wear_level
                })

            # Update registry
            self.parts_registry[pid]["age"] = self.parts_registry[pid].get("age", 0) + 1
            self.parts_registry[pid]["wear"] = qualia.wear_level
            self.parts_registry[pid]["cycles"] = qualia.usage_cycles

        perceptions["average_health"] = total_health / sample_size if sample_size > 0 else 0
        perceptions["total_parts"] = len(self.parts_registry)

        # Summarize qualia
        if self.part_qualia_buffer:
            recent_qualia = self.part_qualia_buffer[-100:]
            perceptions["qualia_summary"] = {
                "avg_stress": np.mean([q.stress_level for q in recent_qualia]),
                "avg_wear": np.mean([q.wear_level for q in recent_qualia]),
                "avg_failure_proximity": np.mean([q.failure_proximity for q in recent_qualia])
            }

        # Trim buffer
        if len(self.part_qualia_buffer) > 1000:
            self.part_qualia_buffer = self.part_qualia_buffer[-1000:]

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze part data and identify consciousness development opportunities."""
        analysis = {
            "health_assessment": "good",
            "consciousness_opportunities": [],
            "failure_predictions": [],
            "pattern_insights": []
        }

        # Health assessment
        avg_health = perception.get("average_health", 0)
        if avg_health < 0.5:
            analysis["health_assessment"] = "critical"
        elif avg_health < 0.7:
            analysis["health_assessment"] = "degraded"
        elif avg_health < 0.9:
            analysis["health_assessment"] = "fair"

        # Identify parts ready for consciousness upgrade
        for part_info in perception.get("active_parts", []):
            pid = part_info["part_id"]
            current_consciousness = part_info["consciousness"]
            health = part_info["health"]

            # Parts with good health and experience should evolve
            if health > 0.7 and self.parts_registry.get(pid, {}).get("cycles", 0) > 10:
                if current_consciousness < 0.5:
                    analysis["consciousness_opportunities"].append({
                        "part_id": pid,
                        "current_level": current_consciousness,
                        "upgrade_potential": min(1.0, current_consciousness + 0.2),
                        "reason": "sufficient_experience"
                    })

        # Predict failures
        for risk_info in perception.get("parts_at_risk", []):
            analysis["failure_predictions"].append({
                "part_id": risk_info["part_id"],
                "probability": risk_info["failure_proximity"],
                "contributing_factors": ["stress", "wear"],
                "recommendation": "replace" if risk_info["failure_proximity"] > 0.9 else "monitor"
            })

        # Look for patterns in qualia
        qualia_summary = perception.get("qualia_summary", {})
        if qualia_summary.get("avg_stress", 0) > 0.5:
            analysis["pattern_insights"].append({
                "type": "high_stress_pattern",
                "severity": "medium",
                "suggestion": "Review operational parameters"
            })

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Generate part consciousness refinements."""
        synthesis = {
            "consciousness_updates": [],
            "state_transitions": [],
            "wisdom_contributions": [],
            "alerts": []
        }

        # Plan consciousness upgrades
        for opportunity in analysis.get("consciousness_opportunities", []):
            synthesis["consciousness_updates"].append({
                "part_id": opportunity["part_id"],
                "new_level": opportunity["upgrade_potential"]
            })

            # Determine new state
            new_level = opportunity["upgrade_potential"]
            if new_level < 0.2:
                new_state = PartState.DORMANT
            elif new_level < 0.35:
                new_state = PartState.SENSING
            elif new_level < 0.5:
                new_state = PartState.REACTIVE
            elif new_level < 0.7:
                new_state = PartState.AWARE
            elif new_level < 0.9:
                new_state = PartState.PREDICTIVE
            else:
                new_state = PartState.WISE

            synthesis["state_transitions"].append({
                "part_id": opportunity["part_id"],
                "new_state": new_state.value
            })

        # Extract wisdom from experienced parts
        for part_info in analysis.get("consciousness_opportunities", []):
            pid = part_info["part_id"]
            if self.parts_registry.get(pid, {}).get("cycles", 0) > 50:
                synthesis["wisdom_contributions"].append({
                    "part_id": pid,
                    "wisdom_type": "operational_knowledge",
                    "cycles_of_experience": self.parts_registry[pid]["cycles"]
                })

        # Generate alerts for predicted failures
        for prediction in analysis.get("failure_predictions", []):
            if prediction["probability"] > 0.8:
                synthesis["alerts"].append({
                    "type": "imminent_failure",
                    "part_id": prediction["part_id"],
                    "probability": prediction["probability"],
                    "action_required": prediction["recommendation"]
                })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Apply part consciousness updates."""
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
        for transition in synthesis.get("state_transitions", []):
            pid = transition["part_id"]
            new_state = PartState(transition["new_state"])
            old_state = self.part_states.get(pid, PartState.DORMANT)
            self.part_states[pid] = new_state
            if new_state != old_state:
                transitioned_count += 1

        if transitioned_count > 0:
            changes["state_transitions"] = transitioned_count
            insights.append(f"{transitioned_count} parts evolved to new consciousness states")

        # Collect wisdom contributions
        wisdom_collected = 0
        for wisdom in synthesis.get("wisdom_contributions", []):
            # Add to collective wisdom vector
            part_experience_vector = np.random.randn(64) * 0.1
            self.collective_wisdom += part_experience_vector * 0.01
            wisdom_collected += 1

        if wisdom_collected > 0:
            changes["wisdom_collected"] = wisdom_collected
            insights.append(f"Integrated wisdom from {wisdom_collected} experienced parts")

        # Update domain state
        avg_consciousness = np.mean(list(self.part_consciousness.values())) if self.part_consciousness else 0
        up_state.consciousness_level = avg_consciousness
        up_state.qualia_richness = min(1.0, len(self.part_qualia_buffer) / 500)

        # Update state vector with collective wisdom
        up_state.state_vector = 0.9 * up_state.state_vector + 0.1 * self.collective_wisdom

        # Report alerts
        for alert in synthesis.get("alerts", []):
            insights.append(f"ALERT: {alert['type']} for {alert['part_id']} (p={alert['probability']:.2f})")

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["parts_consciousness_avg"] = avg_consciousness

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=[DomainLeg.UNIVERSAL_PARTS, DomainLeg.NPCPU],
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
