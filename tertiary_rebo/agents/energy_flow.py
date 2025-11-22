"""
EnergyFlowAgent - Manages energy/resource flows across the system.

Responsibilities:
- Balance energy distribution across all three domain legs
- Implement energy democracy principles from ChicagoForest
- Track resource consumption and production
- Optimize flow paths for efficiency
- Prevent energy bottlenecks and starvation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import deque

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    RefinementResult,
    RefinementPhase,
)


class EnergyFlowAgent(TertiaryReBoAgent):
    """
    Agent 4: Manages energy/resource flows across the triple bottom line.

    Energy democracy means no single domain should hoard resources while
    others starve. This agent ensures:
    - Fair distribution based on need
    - Efficient flow from sources to sinks
    - Emergency reserves for critical operations
    - Sustainable consumption patterns

    The three legs have different energy characteristics:
    - NPCPU: Computational energy (processing power, attention)
    - ChicagoForest: Network energy (bandwidth, routing capacity)
    - UniversalParts: Physical energy (mechanical, thermal)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Energy state for each domain
        self.domain_energy = {
            DomainLeg.NPCPU: 1.0,
            DomainLeg.CHICAGO_FOREST: 1.0,
            DomainLeg.UNIVERSAL_PARTS: 1.0,
        }

        # Energy production/consumption rates
        self.production_rates = {leg: 0.1 for leg in DomainLeg}
        self.consumption_rates = {leg: 0.05 for leg in DomainLeg}

        # Flow tracking
        self.flow_history: deque = deque(maxlen=100)
        self.flow_matrix = np.zeros((3, 3))  # Transfer between domains

        # Energy reserves
        self.reserve_threshold = 0.2
        self.reserves = {leg: 0.3 for leg in DomainLeg}

        # Efficiency metrics
        self.transfer_efficiency = 0.9
        self.total_energy_transferred = 0.0

        # Fair share calculation
        self.fair_share = 1.0 / 3.0

    @property
    def agent_role(self) -> str:
        return "Energy Flow - Manages democratic resource distribution across all domains"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)  # Affects all domains

    def _domain_to_index(self, domain: DomainLeg) -> int:
        return list(DomainLeg).index(domain)

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Gather energy state from all domains."""
        perceptions = {
            "domain_energies": {},
            "total_energy": 0.0,
            "energy_distribution": [],
            "flow_rates": {},
            "imbalances": [],
            "critical_domains": []
        }

        total = 0
        energies = []

        for domain in DomainLeg:
            state = tbl.get_state(domain)
            energy = state.energy_flow

            # Update our tracking
            self.domain_energy[domain] = energy

            perceptions["domain_energies"][domain.value] = energy
            total += energy
            energies.append(energy)

            # Check for critical state
            if energy < self.reserve_threshold:
                perceptions["critical_domains"].append({
                    "domain": domain.value,
                    "energy": energy,
                    "shortfall": self.reserve_threshold - energy
                })

        perceptions["total_energy"] = total
        perceptions["energy_distribution"] = energies

        # Calculate Gini coefficient for energy inequality
        sorted_energies = sorted(energies)
        n = len(sorted_energies)
        if sum(sorted_energies) > 0:
            gini = sum((2 * i - n + 1) * e for i, e in enumerate(sorted_energies))
            gini = gini / (n * sum(sorted_energies))
        else:
            gini = 0

        perceptions["gini_coefficient"] = gini

        # Identify imbalances
        mean_energy = np.mean(energies)
        for domain, energy in zip(DomainLeg, energies):
            if abs(energy - mean_energy) > 0.2:
                perceptions["imbalances"].append({
                    "domain": domain.value,
                    "deviation": energy - mean_energy,
                    "type": "surplus" if energy > mean_energy else "deficit"
                })

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze energy flows and identify optimization opportunities."""
        analysis = {
            "redistribution_needed": False,
            "emergency_transfers": [],
            "optimization_opportunities": [],
            "sustainability_assessment": "sustainable",
            "recommended_flows": []
        }

        # Check if redistribution is needed
        gini = perception.get("gini_coefficient", 0)
        if gini > 0.2:
            analysis["redistribution_needed"] = True
            analysis["optimization_opportunities"].append({
                "type": "reduce_inequality",
                "gini_before": gini,
                "target_gini": 0.1
            })

        # Handle critical domains
        for critical in perception.get("critical_domains", []):
            # Find domains with surplus
            surplus_domains = [
                imb for imb in perception.get("imbalances", [])
                if imb["type"] == "surplus"
            ]

            for surplus in surplus_domains:
                analysis["emergency_transfers"].append({
                    "from": surplus["domain"],
                    "to": critical["domain"],
                    "amount": min(surplus["deviation"] * 0.5, critical["shortfall"]),
                    "priority": "emergency"
                })

        # Plan regular redistribution
        for imbalance in perception.get("imbalances", []):
            if imbalance["type"] == "surplus":
                # Find deficit domains
                deficit_domains = [
                    d for d in perception.get("imbalances", [])
                    if d["type"] == "deficit"
                ]

                for deficit in deficit_domains:
                    amount = min(
                        abs(imbalance["deviation"]) * 0.3,
                        abs(deficit["deviation"]) * 0.3
                    )
                    analysis["recommended_flows"].append({
                        "from": imbalance["domain"],
                        "to": deficit["domain"],
                        "amount": amount * self.transfer_efficiency,
                        "priority": "balance"
                    })

        # Sustainability check
        total_production = sum(self.production_rates.values())
        total_consumption = sum(self.consumption_rates.values())

        if total_consumption > total_production * 1.2:
            analysis["sustainability_assessment"] = "unsustainable"
            analysis["optimization_opportunities"].append({
                "type": "reduce_consumption",
                "current_ratio": total_consumption / total_production,
                "target_ratio": 0.9
            })
        elif total_consumption > total_production:
            analysis["sustainability_assessment"] = "marginal"

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Create energy flow plan."""
        synthesis = {
            "transfers": [],
            "production_adjustments": {},
            "consumption_adjustments": {},
            "reserve_operations": []
        }

        # Prioritize emergency transfers
        for transfer in analysis.get("emergency_transfers", []):
            synthesis["transfers"].append({
                "from": transfer["from"],
                "to": transfer["to"],
                "amount": transfer["amount"],
                "type": "emergency"
            })

        # Add balance transfers
        for flow in analysis.get("recommended_flows", []):
            # Check if not redundant with emergency
            is_duplicate = any(
                t["from"] == flow["from"] and t["to"] == flow["to"]
                for t in synthesis["transfers"]
            )
            if not is_duplicate:
                synthesis["transfers"].append({
                    "from": flow["from"],
                    "to": flow["to"],
                    "amount": flow["amount"],
                    "type": "balance"
                })

        # Handle sustainability issues
        for opp in analysis.get("optimization_opportunities", []):
            if opp["type"] == "reduce_consumption":
                # Reduce consumption rates proportionally
                for domain in DomainLeg:
                    current_rate = self.consumption_rates[domain]
                    synthesis["consumption_adjustments"][domain.value] = current_rate * 0.9

        # Reserve management
        for domain in DomainLeg:
            if self.reserves[domain] < 0.1 and self.domain_energy[domain] > 0.5:
                synthesis["reserve_operations"].append({
                    "domain": domain.value,
                    "operation": "deposit",
                    "amount": 0.05
                })
            elif self.reserves[domain] > 0.4 and self.domain_energy[domain] < 0.3:
                synthesis["reserve_operations"].append({
                    "domain": domain.value,
                    "operation": "withdraw",
                    "amount": 0.1
                })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Execute energy flow operations."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Execute transfers
        transfer_count = 0
        total_transferred = 0

        for transfer in synthesis.get("transfers", []):
            from_domain = DomainLeg(transfer["from"])
            to_domain = DomainLeg(transfer["to"])
            amount = transfer["amount"]

            from_state = tbl.get_state(from_domain)
            to_state = tbl.get_state(to_domain)

            # Only transfer if source has enough
            if from_state.energy_flow >= amount:
                from_state.energy_flow -= amount
                to_state.energy_flow += amount * self.transfer_efficiency
                transfer_count += 1
                total_transferred += amount

                # Update flow matrix
                from_idx = self._domain_to_index(from_domain)
                to_idx = self._domain_to_index(to_domain)
                self.flow_matrix[from_idx, to_idx] += amount

        if transfer_count > 0:
            changes["transfers_executed"] = transfer_count
            changes["total_transferred"] = total_transferred
            insights.append(f"Executed {transfer_count} energy transfers totaling {total_transferred:.3f}")
            self.total_energy_transferred += total_transferred

        # Apply consumption adjustments
        for domain_name, new_rate in synthesis.get("consumption_adjustments", {}).items():
            domain = DomainLeg(domain_name)
            old_rate = self.consumption_rates[domain]
            self.consumption_rates[domain] = new_rate
            changes[f"consumption_rate_{domain_name}"] = new_rate
            insights.append(f"Adjusted {domain_name} consumption: {old_rate:.3f} -> {new_rate:.3f}")

        # Handle reserve operations
        for reserve_op in synthesis.get("reserve_operations", []):
            domain = DomainLeg(reserve_op["domain"])
            state = tbl.get_state(domain)

            if reserve_op["operation"] == "deposit":
                amount = min(reserve_op["amount"], state.energy_flow * 0.5)
                state.energy_flow -= amount
                self.reserves[domain] += amount
                insights.append(f"Deposited {amount:.3f} to {domain.value} reserves")
            elif reserve_op["operation"] == "withdraw":
                amount = min(reserve_op["amount"], self.reserves[domain])
                self.reserves[domain] -= amount
                state.energy_flow += amount
                insights.append(f"Withdrew {amount:.3f} from {domain.value} reserves")

        # Record flow history
        flow_record = {
            "timestamp": datetime.now().isoformat(),
            "transfers": transfer_count,
            "total_amount": total_transferred
        }
        self.flow_history.append(flow_record)

        # Calculate new energy distribution metrics
        energies = [tbl.get_state(d).energy_flow for d in DomainLeg]
        new_gini = self._calculate_gini(energies)

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["gini_after"] = new_gini
        metrics_delta["total_energy"] = sum(energies)

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient."""
        sorted_values = sorted(values)
        n = len(sorted_values)
        total = sum(sorted_values)
        if total == 0:
            return 0
        gini = sum((2 * i - n + 1) * v for i, v in enumerate(sorted_values))
        return gini / (n * total)
