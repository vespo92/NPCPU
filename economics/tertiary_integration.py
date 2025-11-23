"""
Integration between Economics System and Tertiary ReBo

Bridges the economic subsystem with the Triple Bottom Line
energy flow and refinement system.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tertiary_rebo.base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal
)
from economics.resources import (
    ResourcePool,
    ResourceRegistry,
    ResourceCategory,
    get_resource_registry
)
from economics.currency import CurrencySystem, get_currency_system
from economics.scarcity import ScarcityModel, ScarcityLevel, get_scarcity_model
from economics.wealth_distribution import (
    WealthDistributionTracker,
    get_wealth_tracker
)
from economics.markets import MarketManager, get_market_manager


@dataclass
class EconomicDomainMetrics:
    """
    Economic metrics for a domain leg.
    """
    domain: DomainLeg
    total_resources: float = 0.0
    resource_flow_rate: float = 0.0
    scarcity_index: float = 0.0
    wealth_gini: float = 0.0
    market_activity: float = 0.0
    trade_volume: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "total_resources": self.total_resources,
            "resource_flow_rate": self.resource_flow_rate,
            "scarcity_index": self.scarcity_index,
            "wealth_gini": self.wealth_gini,
            "market_activity": self.market_activity,
            "trade_volume": self.trade_volume,
            "timestamp": self.timestamp
        }


class EconomicRefinementAgent(TertiaryReBoAgent):
    """
    Refinement agent that optimizes economic flows across the Triple Bottom Line.

    This agent monitors economic activity across all three domain legs and
    makes refinements to improve resource distribution, reduce inequality,
    and maintain healthy market dynamics.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        refinement_rate: float = 0.1
    ):
        super().__init__(
            agent_id=agent_id,
            primary_domain=DomainLeg.UNIVERSAL_PARTS,  # Economics tied to physical resources
            refinement_rate=refinement_rate
        )

        # Economic components
        self.resource_registry = get_resource_registry()
        self.currency_system = get_currency_system()
        self.scarcity_model = get_scarcity_model()
        self.wealth_tracker = get_wealth_tracker()
        self.market_manager = get_market_manager()

        # Domain-specific resource pools
        self.domain_pools: Dict[DomainLeg, ResourcePool] = {}

        # Tracking
        self.domain_metrics: Dict[DomainLeg, EconomicDomainMetrics] = {}
        self.flow_history: List[Dict[str, Any]] = []

    @property
    def agent_role(self) -> str:
        return "Economic Resource Optimizer"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def register_domain_pool(self, domain: DomainLeg, pool: ResourcePool) -> None:
        """Register a resource pool for a domain."""
        self.domain_pools[domain] = pool
        self.domain_metrics[domain] = EconomicDomainMetrics(domain=domain)

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """
        Perception phase: Gather economic information from all domains.
        """
        perception = {
            "domains": {},
            "cross_domain_flows": [],
            "market_state": {},
            "scarcity_alerts": [],
            "wealth_distribution": {}
        }

        # Gather domain-specific economic metrics
        for domain in DomainLeg:
            domain_data = self._perceive_domain(domain, tbl)
            perception["domains"][domain.value] = domain_data

            # Update metrics
            if domain in self.domain_metrics:
                metrics = self.domain_metrics[domain]
                metrics.total_resources = domain_data.get("total_resources", 0)
                metrics.scarcity_index = domain_data.get("scarcity_index", 0)

        # Gather market state
        perception["market_state"] = self._perceive_markets()

        # Gather scarcity information
        perception["scarcity_alerts"] = self._perceive_scarcity()

        # Gather wealth distribution
        perception["wealth_distribution"] = self._perceive_wealth()

        return perception

    def _perceive_domain(self, domain: DomainLeg, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Perceive economic state of a single domain."""
        domain_state = tbl.get_state(domain)
        pool = self.domain_pools.get(domain)

        data = {
            "energy_flow": domain_state.energy_flow,
            "consciousness_level": domain_state.consciousness_level,
            "total_resources": 0.0,
            "resource_breakdown": {},
            "scarcity_index": 0.0
        }

        if pool:
            data["total_resources"] = pool.get_total_value()
            for category in ResourceCategory:
                data["resource_breakdown"][category.value] = pool.get_total_by_category(category)

        return data

    def _perceive_markets(self) -> Dict[str, Any]:
        """Perceive market state."""
        markets = self.market_manager.list_markets()
        return {
            "market_count": len(markets),
            "total_volume": sum(len(m.price_history.prices) for m in markets),
            "markets": {
                m.id: {
                    "name": m.name,
                    "last_price": m.last_price,
                    "spread": m.order_book.spread()
                }
                for m in markets
            }
        }

    def _perceive_scarcity(self) -> List[Dict[str, Any]]:
        """Perceive scarcity alerts."""
        alerts = []
        most_scarce = self.scarcity_model.get_most_scarce(5)

        for resource_type, level in most_scarce:
            if level in (ScarcityLevel.SCARCE, ScarcityLevel.CRITICAL, ScarcityLevel.DEPLETED):
                alerts.append({
                    "resource_type": resource_type,
                    "level": level.value,
                    "urgency": {
                        ScarcityLevel.SCARCE: 0.5,
                        ScarcityLevel.CRITICAL: 0.8,
                        ScarcityLevel.DEPLETED: 1.0
                    }.get(level, 0.3)
                })

        return alerts

    def _perceive_wealth(self) -> Dict[str, Any]:
        """Perceive wealth distribution."""
        stats = self.wealth_tracker.get_stats()
        return {
            "gini": stats.get("distribution", {}).get("gini", 0.0),
            "population": stats.get("distribution", {}).get("population", 0),
            "total_wealth": stats.get("distribution", {}).get("total_wealth", 0.0),
            "inequality_trend": stats.get("trend", {})
        }

    async def analyze(
        self,
        perception: Dict[str, Any],
        tbl: TripleBottomLine
    ) -> Dict[str, Any]:
        """
        Analysis phase: Identify economic imbalances and opportunities.
        """
        analysis = {
            "imbalances": [],
            "opportunities": [],
            "risks": [],
            "recommended_actions": []
        }

        # Analyze domain imbalances
        domain_resources = {}
        for domain_name, data in perception["domains"].items():
            domain_resources[domain_name] = data.get("total_resources", 0)

        if domain_resources:
            mean_resources = np.mean(list(domain_resources.values()))
            std_resources = np.std(list(domain_resources.values()))

            for domain_name, resources in domain_resources.items():
                if abs(resources - mean_resources) > std_resources:
                    analysis["imbalances"].append({
                        "type": "resource_imbalance",
                        "domain": domain_name,
                        "current": resources,
                        "target": mean_resources,
                        "severity": abs(resources - mean_resources) / max(mean_resources, 1)
                    })

        # Analyze scarcity alerts
        for alert in perception["scarcity_alerts"]:
            if alert["urgency"] > 0.5:
                analysis["risks"].append({
                    "type": "scarcity_risk",
                    "resource": alert["resource_type"],
                    "level": alert["level"],
                    "urgency": alert["urgency"]
                })
                analysis["recommended_actions"].append({
                    "action": "increase_production",
                    "resource": alert["resource_type"],
                    "priority": alert["urgency"]
                })

        # Analyze wealth distribution
        wealth_data = perception["wealth_distribution"]
        gini = wealth_data.get("gini", 0)
        if gini > 0.5:
            analysis["risks"].append({
                "type": "inequality_risk",
                "gini": gini,
                "severity": (gini - 0.5) * 2
            })
            analysis["recommended_actions"].append({
                "action": "redistribute_wealth",
                "urgency": (gini - 0.5) * 2
            })

        return analysis

    async def synthesize(
        self,
        analysis: Dict[str, Any],
        tbl: TripleBottomLine
    ) -> Dict[str, Any]:
        """
        Synthesis phase: Generate refinement proposals.
        """
        synthesis = {
            "refinements": [],
            "resource_transfers": [],
            "market_interventions": [],
            "policy_changes": []
        }

        # Generate resource balance refinements
        for imbalance in analysis["imbalances"]:
            if imbalance["type"] == "resource_imbalance":
                target = imbalance["target"]
                current = imbalance["current"]
                domain = imbalance["domain"]

                if current < target:
                    # Domain needs more resources
                    synthesis["refinements"].append({
                        "type": "resource_injection",
                        "domain": domain,
                        "amount": (target - current) * self.refinement_rate
                    })
                else:
                    # Domain has excess resources
                    synthesis["resource_transfers"].append({
                        "from_domain": domain,
                        "amount": (current - target) * self.refinement_rate
                    })

        # Generate scarcity responses
        for action in analysis["recommended_actions"]:
            if action["action"] == "increase_production":
                synthesis["refinements"].append({
                    "type": "production_boost",
                    "resource": action["resource"],
                    "priority": action["priority"]
                })

        # Generate inequality responses
        for action in analysis["recommended_actions"]:
            if action["action"] == "redistribute_wealth":
                synthesis["policy_changes"].append({
                    "type": "progressive_taxation",
                    "rate": action["urgency"] * 0.1
                })

        return synthesis

    async def propagate(
        self,
        synthesis: Dict[str, Any],
        tbl: TripleBottomLine
    ) -> RefinementResult:
        """
        Propagation phase: Apply economic refinements.
        """
        changes = {"applied": [], "failed": []}
        metrics_delta = {}
        insights = []

        # Apply resource refinements
        for refinement in synthesis["refinements"]:
            try:
                result = self._apply_refinement(refinement, tbl)
                if result:
                    changes["applied"].append(refinement)
                    insights.append(f"Applied {refinement['type']} refinement")
                else:
                    changes["failed"].append(refinement)
            except Exception as e:
                changes["failed"].append({"refinement": refinement, "error": str(e)})

        # Apply resource transfers
        for transfer in synthesis["resource_transfers"]:
            try:
                self._apply_transfer(transfer, tbl)
                changes["applied"].append(transfer)
            except Exception as e:
                changes["failed"].append({"transfer": transfer, "error": str(e)})

        # Calculate metrics changes
        prev_harmony = tbl.harmony_score
        tbl.calculate_harmony()
        metrics_delta["harmony_before"] = prev_harmony
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["harmony_change"] = tbl.harmony_score - prev_harmony

        return RefinementResult(
            success=len(changes["applied"]) > 0,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=self.domains_affected,
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    def _apply_refinement(self, refinement: Dict[str, Any], tbl: TripleBottomLine) -> bool:
        """Apply a single refinement."""
        refinement_type = refinement.get("type")

        if refinement_type == "resource_injection":
            domain = DomainLeg(refinement["domain"])
            amount = refinement["amount"]
            domain_state = tbl.get_state(domain)
            domain_state.energy_flow += amount * 0.1
            return True

        elif refinement_type == "production_boost":
            # Signal production increase
            self.emit_cross_domain_signal(
                signal_type="production_request",
                payload={
                    "resource": refinement["resource"],
                    "priority": refinement["priority"]
                },
                strength=refinement["priority"]
            )
            return True

        return False

    def _apply_transfer(self, transfer: Dict[str, Any], tbl: TripleBottomLine) -> bool:
        """Apply a resource transfer between domains."""
        from_domain = DomainLeg(transfer["from_domain"])
        amount = transfer["amount"]

        from_state = tbl.get_state(from_domain)
        from_state.energy_flow -= amount * 0.05

        # Distribute to other domains
        other_domains = [d for d in DomainLeg if d != from_domain]
        per_domain = amount * 0.05 / len(other_domains)

        for domain in other_domains:
            state = tbl.get_state(domain)
            state.energy_flow += per_domain

        return True

    def get_economic_summary(self) -> Dict[str, Any]:
        """Get a summary of the economic state."""
        return {
            "agent_id": self.agent_id,
            "domain_metrics": {
                d.value: m.to_dict() for d, m in self.domain_metrics.items()
            },
            "total_refinements": self.total_refinements,
            "successful_refinements": self.successful_refinements,
            "success_rate": self.successful_refinements / max(self.total_refinements, 1),
            "active_signals": len(self.cross_domain_signals)
        }


class EconomicSubsystemBridge:
    """
    Bridge connecting the Economics subsystem to organism subsystems.

    Provides utilities for integrating economic functionality with
    the broader NPCPU organism architecture.
    """

    def __init__(self):
        self.refinement_agent = EconomicRefinementAgent()
        self.registered_organisms: Dict[str, Dict[str, Any]] = {}

    def register_organism(
        self,
        organism_id: str,
        domain: DomainLeg,
        resource_pool: ResourcePool
    ) -> None:
        """Register an organism for economic tracking."""
        self.registered_organisms[organism_id] = {
            "domain": domain,
            "resource_pool": resource_pool,
            "registered_at": time.time()
        }

        # Register the pool with the refinement agent
        self.refinement_agent.register_domain_pool(domain, resource_pool)

        # Register with wealth tracker
        total_value = resource_pool.get_total_value()
        get_wealth_tracker().register_entity(organism_id, total_value)

    def update_organism_wealth(self, organism_id: str) -> None:
        """Update organism's wealth in the tracker."""
        if organism_id not in self.registered_organisms:
            return

        pool = self.registered_organisms[organism_id]["resource_pool"]
        total_value = pool.get_total_value()
        get_wealth_tracker().update_wealth(organism_id, total_wealth=total_value)

    async def run_economic_refinement(self, tbl: TripleBottomLine) -> RefinementResult:
        """Run a full economic refinement cycle."""
        return await self.refinement_agent.refine(tbl)

    def get_organism_economic_state(self, organism_id: str) -> Optional[Dict[str, Any]]:
        """Get economic state for an organism."""
        if organism_id not in self.registered_organisms:
            return None

        data = self.registered_organisms[organism_id]
        pool = data["resource_pool"]
        wealth_record = get_wealth_tracker().get_record(organism_id)

        return {
            "organism_id": organism_id,
            "domain": data["domain"].value,
            "total_resources": pool.get_total_value(),
            "wealth_class": wealth_record.wealth_class.value if wealth_record else "unknown",
            "wealth_percentile": wealth_record.percentile if wealth_record else 50.0
        }


# Singleton bridge instance
_bridge_instance = None


def get_economic_bridge() -> EconomicSubsystemBridge:
    """Get the economic subsystem bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = EconomicSubsystemBridge()
    return _bridge_instance
