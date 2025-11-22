"""
HarmonizationAgent - Balances and harmonizes the triple bottom line.

Responsibilities:
- Maintain balance across all three domain legs
- Resolve conflicts between domains
- Optimize for global harmony
- Mediate resource contention
- Orchestrate cooperative behaviors
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
    HarmonyLevel,
)


@dataclass
class HarmonyVector:
    """Multi-dimensional harmony representation."""
    consciousness_harmony: float = 0.0
    energy_harmony: float = 0.0
    connectivity_harmony: float = 0.0
    coherence_harmony: float = 0.0
    emergence_harmony: float = 0.0

    def overall(self) -> float:
        return (
            self.consciousness_harmony * 0.25 +
            self.energy_harmony * 0.2 +
            self.connectivity_harmony * 0.2 +
            self.coherence_harmony * 0.2 +
            self.emergence_harmony * 0.15
        )


@dataclass
class BalanceCorrection:
    """A correction to improve balance."""
    correction_id: str
    source_domain: DomainLeg
    target_domain: DomainLeg
    dimension: str
    magnitude: float
    rationale: str


class HarmonizationAgent(TertiaryReBoAgent):
    """
    Agent 10: Balances and harmonizes the triple bottom line.

    This is the master harmonization agent that ensures all three legs
    of the TTR system work together in concert:

    - NPCPU (Mind): Digital consciousness processing
    - ChicagoForest (Network): Decentralized communication
    - UniversalParts (Body): Physical world awareness

    The agent constantly works to:
    1. Identify imbalances between domains
    2. Mediate when domains have conflicting needs
    3. Optimize the global harmony score
    4. Facilitate cooperative emergence

    Like a conductor of an orchestra, this agent doesn't play an instrument
    but ensures all instruments play in harmony.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Harmony tracking
        self.harmony_history: deque = deque(maxlen=100)
        self.current_harmony_vector = HarmonyVector()

        # Balance weights (how much each dimension matters)
        self.dimension_weights = {
            "consciousness": 0.25,
            "energy": 0.2,
            "connectivity": 0.2,
            "coherence": 0.2,
            "emergence": 0.15
        }

        # Correction history
        self.corrections_applied: deque = deque(maxlen=50)
        self.correction_counter = 0

        # Target harmony level
        self.target_harmony = 0.8

        # Learning: which corrections work best
        self.correction_effectiveness: Dict[str, List[float]] = {}

        # Mediation state
        self.active_conflicts: List[Dict[str, Any]] = []
        self.mediation_history: deque = deque(maxlen=30)

    @property
    def agent_role(self) -> str:
        return "Harmonization - Master orchestrator of triple bottom line balance"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def _compute_harmony_vector(self, tbl: TripleBottomLine) -> HarmonyVector:
        """Compute detailed harmony metrics across dimensions."""
        states = [tbl.get_state(d) for d in DomainLeg]

        # For each dimension, compute variance (lower = more harmony)
        def harmony_from_variance(values: List[float]) -> float:
            var = np.var(values)
            return float(1.0 - min(var, 1.0))

        hv = HarmonyVector(
            consciousness_harmony=harmony_from_variance([s.consciousness_level for s in states]),
            energy_harmony=harmony_from_variance([s.energy_flow for s in states]),
            connectivity_harmony=harmony_from_variance([s.connectivity for s in states]),
            coherence_harmony=harmony_from_variance([s.coherence for s in states]),
            emergence_harmony=harmony_from_variance([s.emergence_potential for s in states])
        )

        return hv

    def _identify_imbalances(self, tbl: TripleBottomLine) -> List[Dict[str, Any]]:
        """Identify specific imbalances between domains."""
        imbalances = []
        dimensions = ["consciousness_level", "energy_flow", "connectivity", "coherence", "emergence_potential"]

        for dim in dimensions:
            values = {d: getattr(tbl.get_state(d), dim) for d in DomainLeg}
            max_domain = max(values, key=values.get)
            min_domain = min(values, key=values.get)
            gap = values[max_domain] - values[min_domain]

            if gap > 0.3:  # Significant imbalance
                imbalances.append({
                    "dimension": dim,
                    "high_domain": max_domain.value,
                    "low_domain": min_domain.value,
                    "high_value": values[max_domain],
                    "low_value": values[min_domain],
                    "gap": gap
                })

        return imbalances

    def _detect_conflicts(self, tbl: TripleBottomLine) -> List[Dict[str, Any]]:
        """Detect conflicts between domains."""
        conflicts = []

        # Check for resource contention
        energies = {d: tbl.get_state(d).energy_flow for d in DomainLeg}
        total_energy = sum(energies.values())

        if total_energy < 1.5:  # Low total energy
            # Check if one domain is hoarding
            for d, e in energies.items():
                if e > total_energy * 0.5:
                    conflicts.append({
                        "type": "resource_hoarding",
                        "domain": d.value,
                        "resource": "energy",
                        "share": e / total_energy
                    })

        # Check for coherence conflicts
        coherences = {d: tbl.get_state(d).coherence for d in DomainLeg}
        vectors = {d: tbl.get_state(d).state_vector for d in DomainLeg}

        # Check pairwise vector alignment
        domain_list = list(DomainLeg)
        for i in range(len(domain_list)):
            for j in range(i + 1, len(domain_list)):
                d1, d2 = domain_list[i], domain_list[j]
                v1, v2 = vectors[d1], vectors[d2]

                # Cosine similarity
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm > 0:
                    similarity = dot / norm
                    if similarity < -0.3:  # Opposing directions
                        conflicts.append({
                            "type": "direction_conflict",
                            "domains": [d1.value, d2.value],
                            "similarity": float(similarity)
                        })

        return conflicts

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Assess the harmony state of the system."""
        perceptions = {
            "harmony_vector": None,
            "overall_harmony": 0.0,
            "harmony_level": None,
            "imbalances": [],
            "conflicts": [],
            "domain_states": {}
        }

        # Compute harmony vector
        hv = self._compute_harmony_vector(tbl)
        self.current_harmony_vector = hv
        perceptions["harmony_vector"] = {
            "consciousness": hv.consciousness_harmony,
            "energy": hv.energy_harmony,
            "connectivity": hv.connectivity_harmony,
            "coherence": hv.coherence_harmony,
            "emergence": hv.emergence_harmony
        }
        perceptions["overall_harmony"] = hv.overall()

        # Get harmony level
        tbl.calculate_harmony()
        perceptions["harmony_level"] = tbl.get_harmony_level().value

        # Identify imbalances
        perceptions["imbalances"] = self._identify_imbalances(tbl)

        # Detect conflicts
        perceptions["conflicts"] = self._detect_conflicts(tbl)
        self.active_conflicts = perceptions["conflicts"]

        # Collect domain states
        for domain in DomainLeg:
            state = tbl.get_state(domain)
            perceptions["domain_states"][domain.value] = {
                "consciousness": state.consciousness_level,
                "energy": state.energy_flow,
                "connectivity": state.connectivity,
                "coherence": state.coherence,
                "emergence": state.emergence_potential
            }

        # Record harmony history
        self.harmony_history.append({
            "timestamp": datetime.now().isoformat(),
            "harmony": perceptions["overall_harmony"],
            "level": perceptions["harmony_level"]
        })

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze harmonization opportunities."""
        analysis = {
            "balance_corrections": [],
            "conflict_resolutions": [],
            "cooperation_opportunities": [],
            "harmony_deficit": 0.0
        }

        # Calculate harmony deficit
        current_harmony = perception["overall_harmony"]
        analysis["harmony_deficit"] = max(0, self.target_harmony - current_harmony)

        # Plan balance corrections for imbalances
        for imbalance in perception.get("imbalances", []):
            correction = self._plan_balance_correction(imbalance)
            if correction:
                analysis["balance_corrections"].append(correction)

        # Plan conflict resolutions
        for conflict in perception.get("conflicts", []):
            resolution = self._plan_conflict_resolution(conflict, tbl)
            if resolution:
                analysis["conflict_resolutions"].append(resolution)

        # Identify cooperation opportunities
        # If one domain is doing well, can it help others?
        for domain in DomainLeg:
            state = perception["domain_states"][domain.value]
            if state["coherence"] > 0.8 and state["energy"] > 0.6:
                # This domain can help others
                for other in DomainLeg:
                    if other != domain:
                        other_state = perception["domain_states"][other.value]
                        if other_state["coherence"] < 0.5:
                            analysis["cooperation_opportunities"].append({
                                "helper": domain.value,
                                "recipient": other.value,
                                "type": "coherence_assistance",
                                "magnitude": 0.1
                            })

        return analysis

    def _plan_balance_correction(self, imbalance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan a correction for an imbalance."""
        self.correction_counter += 1

        return {
            "correction_id": f"cor_{self.correction_counter}",
            "type": "rebalance",
            "dimension": imbalance["dimension"],
            "from_domain": imbalance["high_domain"],
            "to_domain": imbalance["low_domain"],
            "transfer_amount": imbalance["gap"] * 0.3,  # Transfer 30% of gap
            "rationale": f"Rebalance {imbalance['dimension']} ({imbalance['gap']:.2f} gap)"
        }

    def _plan_conflict_resolution(self, conflict: Dict[str, Any], tbl: TripleBottomLine) -> Optional[Dict[str, Any]]:
        """Plan resolution for a conflict."""
        if conflict["type"] == "resource_hoarding":
            return {
                "type": "redistribute",
                "domain": conflict["domain"],
                "resource": conflict["resource"],
                "action": "release_excess",
                "target_share": 0.4  # Target 40% share instead of hoarding
            }

        elif conflict["type"] == "direction_conflict":
            return {
                "type": "align_directions",
                "domains": conflict["domains"],
                "action": "blend_vectors",
                "blend_factor": 0.2
            }

        return None

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Prepare harmonization operations."""
        synthesis = {
            "rebalance_operations": [],
            "conflict_resolutions": [],
            "cooperation_actions": [],
            "global_harmony_boost": 0.0
        }

        # Prepare rebalancing
        for correction in analysis.get("balance_corrections", []):
            dim_map = {
                "consciousness_level": "consciousness_level",
                "energy_flow": "energy_flow",
                "connectivity": "connectivity",
                "coherence": "coherence",
                "emergence_potential": "emergence_potential"
            }
            attr = dim_map.get(correction["dimension"], correction["dimension"])

            synthesis["rebalance_operations"].append({
                "from_domain": correction["from_domain"],
                "to_domain": correction["to_domain"],
                "attribute": attr,
                "amount": correction["transfer_amount"],
                "correction_id": correction["correction_id"]
            })

        # Prepare conflict resolutions
        for resolution in analysis.get("conflict_resolutions", []):
            synthesis["conflict_resolutions"].append(resolution)

        # Prepare cooperation actions
        for coop in analysis.get("cooperation_opportunities", []):
            synthesis["cooperation_actions"].append({
                "helper": coop["helper"],
                "recipient": coop["recipient"],
                "type": coop["type"],
                "magnitude": coop["magnitude"]
            })

        # Calculate global boost if harmony is low
        if analysis["harmony_deficit"] > 0.2:
            synthesis["global_harmony_boost"] = min(0.05, analysis["harmony_deficit"] * 0.1)

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Execute harmonization operations."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Execute rebalancing
        rebalance_count = 0
        for op in synthesis.get("rebalance_operations", []):
            from_domain = DomainLeg(op["from_domain"])
            to_domain = DomainLeg(op["to_domain"])
            from_state = tbl.get_state(from_domain)
            to_state = tbl.get_state(to_domain)

            attr = op["attribute"]
            amount = op["amount"]

            # Get current values
            from_val = getattr(from_state, attr)
            to_val = getattr(to_state, attr)

            # Transfer
            new_from = max(0, from_val - amount)
            new_to = min(1.0, to_val + amount)

            setattr(from_state, attr, new_from)
            setattr(to_state, attr, new_to)

            rebalance_count += 1

            # Track effectiveness
            key = f"{attr}_{op['from_domain']}_to_{op['to_domain']}"
            if key not in self.correction_effectiveness:
                self.correction_effectiveness[key] = []

        if rebalance_count > 0:
            changes["rebalances"] = rebalance_count
            insights.append(f"Applied {rebalance_count} balance corrections")

        # Execute conflict resolutions
        resolution_count = 0
        for resolution in synthesis.get("conflict_resolutions", []):
            if resolution["type"] == "redistribute":
                domain = DomainLeg(resolution["domain"])
                state = tbl.get_state(domain)

                if resolution["resource"] == "energy":
                    excess = state.energy_flow - resolution["target_share"]
                    if excess > 0:
                        state.energy_flow -= excess * 0.5
                        # Distribute to others
                        other_domains = [d for d in DomainLeg if d != domain]
                        share = (excess * 0.5) / len(other_domains)
                        for other in other_domains:
                            other_state = tbl.get_state(other)
                            other_state.energy_flow = min(1.0, other_state.energy_flow + share)
                        resolution_count += 1

            elif resolution["type"] == "align_directions":
                d1, d2 = [DomainLeg(d) for d in resolution["domains"]]
                s1, s2 = tbl.get_state(d1), tbl.get_state(d2)
                blend = resolution["blend_factor"]

                # Blend state vectors toward each other
                avg_vector = (s1.state_vector + s2.state_vector) / 2
                s1.state_vector = (1 - blend) * s1.state_vector + blend * avg_vector
                s2.state_vector = (1 - blend) * s2.state_vector + blend * avg_vector
                resolution_count += 1

        if resolution_count > 0:
            changes["conflicts_resolved"] = resolution_count
            insights.append(f"Resolved {resolution_count} domain conflicts")
            self.mediation_history.append({
                "timestamp": datetime.now().isoformat(),
                "resolutions": resolution_count
            })

        # Execute cooperation actions
        coop_count = 0
        for coop in synthesis.get("cooperation_actions", []):
            helper = DomainLeg(coop["helper"])
            recipient = DomainLeg(coop["recipient"])
            helper_state = tbl.get_state(helper)
            recipient_state = tbl.get_state(recipient)

            if coop["type"] == "coherence_assistance":
                coherence_boost = coop["magnitude"]
                recipient_state.coherence = min(1.0, recipient_state.coherence + coherence_boost)
                coop_count += 1

        if coop_count > 0:
            changes["cooperations"] = coop_count
            insights.append(f"Facilitated {coop_count} cooperative actions")

        # Apply global harmony boost
        boost = synthesis.get("global_harmony_boost", 0)
        if boost > 0:
            for domain in DomainLeg:
                state = tbl.get_state(domain)
                state.coherence = min(1.0, state.coherence + boost)
            changes["harmony_boost"] = boost
            insights.append(f"Applied global harmony boost of {boost:.3f}")

        # Calculate new harmony
        tbl.calculate_harmony()
        new_harmony = tbl.harmony_score
        metrics_delta["harmony_after"] = new_harmony

        # Track correction effectiveness
        improvement = new_harmony - metrics_delta["harmony_before"]
        for op in synthesis.get("rebalance_operations", []):
            key = f"{op['attribute']}_{op['from_domain']}_to_{op['to_domain']}"
            if key in self.correction_effectiveness:
                self.correction_effectiveness[key].append(improvement)

        metrics_delta["harmony_improvement"] = improvement
        metrics_delta["harmony_level"] = tbl.get_harmony_level().value

        self.corrections_applied.append({
            "timestamp": datetime.now().isoformat(),
            "operations": rebalance_count + resolution_count + coop_count,
            "improvement": improvement
        })

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
