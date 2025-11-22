"""
ResilienceAgent - Ensures system resilience through self-healing and redundancy.

Responsibilities:
- Monitor system health across all domains
- Detect failures and anomalies
- Implement self-healing mechanisms
- Maintain redundancy and backup states
- Coordinate graceful degradation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
)


class HealthStatus(Enum):
    """Health status levels."""
    OPTIMAL = "optimal"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class FailureType(Enum):
    """Types of system failures."""
    CONNECTIVITY_LOSS = "connectivity_loss"
    ENERGY_DEPLETION = "energy_depletion"
    COHERENCE_COLLAPSE = "coherence_collapse"
    CONSCIOUSNESS_DROP = "consciousness_drop"
    EMERGENCE_STALL = "emergence_stall"
    CASCADE_FAILURE = "cascade_failure"


@dataclass
class HealthSnapshot:
    """Snapshot of system health."""
    timestamp: datetime
    domain_health: Dict[DomainLeg, float]
    overall_health: float
    active_issues: List[Dict[str, Any]]


@dataclass
class BackupState:
    """Backup of domain state for recovery."""
    backup_id: str
    domain: DomainLeg
    state_vector: np.ndarray
    health_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class ResilienceAgent(TertiaryReBoAgent):
    """
    Agent 9: Ensures system resilience through self-healing and redundancy.

    Like an immune system, this agent constantly monitors for threats,
    maintains backups, and orchestrates recovery when problems occur.

    Key capabilities:
    - Health monitoring: Continuous assessment of all domains
    - Anomaly detection: Identify unusual patterns that might indicate problems
    - Self-healing: Automatic recovery from detected failures
    - Redundancy management: Maintain backup states for critical components
    - Graceful degradation: Maintain core functions when resources are limited
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Health tracking
        self.health_history: deque = deque(maxlen=100)
        self.current_status: Dict[DomainLeg, HealthStatus] = {
            leg: HealthStatus.HEALTHY for leg in DomainLeg
        }

        # Backup management
        self.backups: Dict[DomainLeg, List[BackupState]] = {
            leg: [] for leg in DomainLeg
        }
        self.max_backups = 5
        self.backup_interval = 10  # Refinements between backups

        # Anomaly detection
        self.baseline_metrics: Dict[DomainLeg, Dict[str, float]] = {}
        self.anomaly_threshold = 2.0  # Standard deviations

        # Recovery tracking
        self.active_recoveries: List[Dict[str, Any]] = []
        self.recovery_history: deque = deque(maxlen=50)

        # Health thresholds
        self.thresholds = {
            "optimal": 0.9,
            "healthy": 0.7,
            "degraded": 0.5,
            "critical": 0.3
        }

        # Refinement counter for backup scheduling
        self.refinement_counter = 0

    @property
    def agent_role(self) -> str:
        return "Resilience - Self-healing and redundancy management"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    def _calculate_health(self, state: DomainState) -> float:
        """Calculate health score for a domain state."""
        health_factors = [
            state.consciousness_level * 0.2,
            state.coherence * 0.25,
            state.connectivity * 0.2,
            state.energy_flow * 0.2,
            (1.0 - np.var(state.state_vector)) * 0.15  # Stability
        ]
        return float(np.clip(sum(health_factors), 0, 1))

    def _get_health_status(self, health_score: float) -> HealthStatus:
        """Convert health score to status."""
        if health_score >= self.thresholds["optimal"]:
            return HealthStatus.OPTIMAL
        elif health_score >= self.thresholds["healthy"]:
            return HealthStatus.HEALTHY
        elif health_score >= self.thresholds["degraded"]:
            return HealthStatus.DEGRADED
        elif health_score >= self.thresholds["critical"]:
            return HealthStatus.CRITICAL
        return HealthStatus.FAILED

    def _detect_anomalies(self, domain: DomainLeg, state: DomainState) -> List[Dict[str, Any]]:
        """Detect anomalies in domain state."""
        anomalies = []

        if domain not in self.baseline_metrics:
            # Initialize baseline
            self.baseline_metrics[domain] = {
                "consciousness": state.consciousness_level,
                "coherence": state.coherence,
                "energy": state.energy_flow,
                "state_mean": float(np.mean(state.state_vector)),
                "state_std": float(np.std(state.state_vector))
            }
            return anomalies

        baseline = self.baseline_metrics[domain]

        # Check for deviations
        checks = [
            ("consciousness", state.consciousness_level, baseline["consciousness"]),
            ("coherence", state.coherence, baseline["coherence"]),
            ("energy", state.energy_flow, baseline["energy"]),
        ]

        for metric_name, current, expected in checks:
            if expected > 0:
                deviation = abs(current - expected) / expected
                if deviation > 0.5:  # 50% deviation
                    anomalies.append({
                        "type": "metric_deviation",
                        "metric": metric_name,
                        "expected": expected,
                        "actual": current,
                        "deviation": deviation
                    })

        # Check state vector distribution
        current_std = float(np.std(state.state_vector))
        if baseline["state_std"] > 0:
            std_ratio = current_std / baseline["state_std"]
            if std_ratio > 2.0 or std_ratio < 0.5:
                anomalies.append({
                    "type": "distribution_anomaly",
                    "metric": "state_variance",
                    "expected_std": baseline["state_std"],
                    "actual_std": current_std
                })

        # Update baseline with exponential moving average
        alpha = 0.1
        self.baseline_metrics[domain] = {
            "consciousness": alpha * state.consciousness_level + (1-alpha) * baseline["consciousness"],
            "coherence": alpha * state.coherence + (1-alpha) * baseline["coherence"],
            "energy": alpha * state.energy_flow + (1-alpha) * baseline["energy"],
            "state_mean": alpha * float(np.mean(state.state_vector)) + (1-alpha) * baseline["state_mean"],
            "state_std": alpha * current_std + (1-alpha) * baseline["state_std"]
        }

        return anomalies

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Monitor system health."""
        self.refinement_counter += 1

        perceptions = {
            "domain_health": {},
            "overall_health": 0.0,
            "anomalies": {},
            "failure_risks": [],
            "backup_status": {}
        }

        total_health = 0
        active_issues = []

        for domain in DomainLeg:
            state = tbl.get_state(domain)
            health = self._calculate_health(state)
            status = self._get_health_status(health)

            self.current_status[domain] = status

            perceptions["domain_health"][domain.value] = {
                "score": health,
                "status": status.value
            }
            total_health += health

            # Detect anomalies
            anomalies = self._detect_anomalies(domain, state)
            perceptions["anomalies"][domain.value] = anomalies

            if anomalies:
                active_issues.extend([{**a, "domain": domain.value} for a in anomalies])

            # Assess failure risk
            if status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                failure_type = self._identify_failure_risk(state, status)
                perceptions["failure_risks"].append({
                    "domain": domain.value,
                    "status": status.value,
                    "failure_type": failure_type.value if failure_type else None,
                    "risk_level": 1.0 - health
                })

            # Check backup status
            perceptions["backup_status"][domain.value] = {
                "count": len(self.backups[domain]),
                "latest": self.backups[domain][-1].timestamp.isoformat() if self.backups[domain] else None
            }

        perceptions["overall_health"] = total_health / 3

        # Record health snapshot
        snapshot = HealthSnapshot(
            timestamp=datetime.now(),
            domain_health={d: perceptions["domain_health"][d.value]["score"] for d in DomainLeg},
            overall_health=perceptions["overall_health"],
            active_issues=active_issues
        )
        self.health_history.append(snapshot)

        return perceptions

    def _identify_failure_risk(self, state: DomainState, status: HealthStatus) -> Optional[FailureType]:
        """Identify the type of failure risk."""
        if state.connectivity < 0.3:
            return FailureType.CONNECTIVITY_LOSS
        elif state.energy_flow < 0.2:
            return FailureType.ENERGY_DEPLETION
        elif state.coherence < 0.3:
            return FailureType.COHERENCE_COLLAPSE
        elif state.consciousness_level < 0.2:
            return FailureType.CONSCIOUSNESS_DROP
        elif state.emergence_potential < 0.1:
            return FailureType.EMERGENCE_STALL
        return None

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze health data and plan interventions."""
        analysis = {
            "backup_needed": [],
            "recovery_actions": [],
            "preventive_actions": [],
            "cascade_risk": False
        }

        # Check if backups are needed
        if self.refinement_counter % self.backup_interval == 0:
            for domain in DomainLeg:
                health = perception["domain_health"][domain.value]["score"]
                if health > 0.6:  # Only backup healthy states
                    analysis["backup_needed"].append({
                        "domain": domain.value,
                        "health": health
                    })

        # Plan recovery for failures
        for risk in perception.get("failure_risks", []):
            if risk["risk_level"] > 0.5:
                domain = DomainLeg(risk["domain"])
                recovery = self._plan_recovery(domain, risk["failure_type"], tbl)
                if recovery:
                    analysis["recovery_actions"].append(recovery)

        # Plan preventive actions for anomalies
        for domain_name, anomalies in perception.get("anomalies", {}).items():
            for anomaly in anomalies:
                if anomaly["type"] == "metric_deviation" and anomaly["deviation"] > 0.3:
                    analysis["preventive_actions"].append({
                        "domain": domain_name,
                        "action": "stabilize_metric",
                        "metric": anomaly["metric"],
                        "target": anomaly["expected"]
                    })

        # Check for cascade risk
        critical_count = sum(
            1 for d in DomainLeg
            if self.current_status[d] in [HealthStatus.CRITICAL, HealthStatus.FAILED]
        )
        if critical_count >= 2:
            analysis["cascade_risk"] = True
            analysis["recovery_actions"].append({
                "type": "cascade_prevention",
                "action": "emergency_stabilization",
                "priority": "critical"
            })

        return analysis

    def _plan_recovery(self, domain: DomainLeg, failure_type: Optional[str], tbl: TripleBottomLine) -> Optional[Dict[str, Any]]:
        """Plan recovery action for a domain."""
        if not failure_type:
            return None

        # Check if we have a backup to restore
        backups = self.backups[domain]
        usable_backup = None
        for backup in reversed(backups):  # Start with most recent
            if backup.health_score > 0.6:
                usable_backup = backup
                break

        if usable_backup:
            return {
                "type": "restore_backup",
                "domain": domain.value,
                "backup_id": usable_backup.backup_id,
                "failure_type": failure_type
            }
        else:
            return {
                "type": "regenerate",
                "domain": domain.value,
                "failure_type": failure_type,
                "strategy": "gradual_recovery"
            }

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Prepare resilience operations."""
        synthesis = {
            "backup_operations": [],
            "recovery_operations": [],
            "stabilization_operations": []
        }

        # Prepare backups
        for backup_req in analysis.get("backup_needed", []):
            domain = DomainLeg(backup_req["domain"])
            state = tbl.get_state(domain)

            backup = BackupState(
                backup_id=f"bak_{domain.value}_{datetime.now().strftime('%H%M%S')}",
                domain=domain,
                state_vector=state.state_vector.copy(),
                health_score=backup_req["health"]
            )
            synthesis["backup_operations"].append(backup)

        # Prepare recovery operations
        for recovery in analysis.get("recovery_actions", []):
            if recovery.get("type") == "restore_backup":
                synthesis["recovery_operations"].append({
                    "action": "restore",
                    "domain": recovery["domain"],
                    "backup_id": recovery["backup_id"]
                })
            elif recovery.get("type") == "regenerate":
                synthesis["recovery_operations"].append({
                    "action": "regenerate",
                    "domain": recovery["domain"],
                    "strategy": recovery["strategy"]
                })
            elif recovery.get("type") == "cascade_prevention":
                synthesis["recovery_operations"].append({
                    "action": "emergency_stabilize",
                    "all_domains": True
                })

        # Prepare stabilization operations
        for action in analysis.get("preventive_actions", []):
            synthesis["stabilization_operations"].append({
                "domain": action["domain"],
                "metric": action["metric"],
                "target": action["target"],
                "blend_factor": 0.2
            })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Execute resilience operations."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Execute backups
        backup_count = 0
        for backup in synthesis.get("backup_operations", []):
            domain = backup.domain
            self.backups[domain].append(backup)

            # Trim old backups
            while len(self.backups[domain]) > self.max_backups:
                self.backups[domain].pop(0)

            backup_count += 1

        if backup_count > 0:
            changes["backups_created"] = backup_count
            insights.append(f"Created {backup_count} state backups")

        # Execute recovery operations
        recovery_count = 0
        for recovery in synthesis.get("recovery_operations", []):
            if recovery["action"] == "restore":
                domain = DomainLeg(recovery["domain"])
                backup_id = recovery["backup_id"]

                # Find backup
                backup = next(
                    (b for b in self.backups[domain] if b.backup_id == backup_id),
                    None
                )
                if backup:
                    state = tbl.get_state(domain)
                    # Blend restoration (not full replacement to preserve recent progress)
                    state.state_vector = 0.3 * state.state_vector + 0.7 * backup.state_vector
                    recovery_count += 1
                    insights.append(f"Restored {domain.value} from backup {backup_id}")

            elif recovery["action"] == "regenerate":
                domain = DomainLeg(recovery["domain"])
                state = tbl.get_state(domain)

                # Gradual regeneration: reset toward neutral state
                neutral = np.tanh(np.random.randn(64) * 0.3)
                state.state_vector = 0.5 * state.state_vector + 0.5 * neutral
                state.coherence = max(state.coherence, 0.4)
                state.energy_flow = max(state.energy_flow, 0.3)
                recovery_count += 1
                insights.append(f"Regenerated {domain.value} state")

            elif recovery["action"] == "emergency_stabilize":
                for domain in DomainLeg:
                    state = tbl.get_state(domain)
                    # Emergency stabilization: boost all critical metrics
                    state.coherence = max(state.coherence, 0.5)
                    state.energy_flow = max(state.energy_flow, 0.4)
                    state.connectivity = max(state.connectivity, 0.4)
                recovery_count += 1
                insights.append("Executed emergency stabilization across all domains")

        if recovery_count > 0:
            changes["recoveries_executed"] = recovery_count

        # Execute stabilization
        stabilization_count = 0
        for stab in synthesis.get("stabilization_operations", []):
            domain = DomainLeg(stab["domain"])
            state = tbl.get_state(domain)
            metric = stab["metric"]
            target = stab["target"]
            blend = stab["blend_factor"]

            if metric == "consciousness":
                state.consciousness_level = (1 - blend) * state.consciousness_level + blend * target
            elif metric == "coherence":
                state.coherence = (1 - blend) * state.coherence + blend * target
            elif metric == "energy":
                state.energy_flow = (1 - blend) * state.energy_flow + blend * target

            stabilization_count += 1

        if stabilization_count > 0:
            changes["stabilizations"] = stabilization_count
            insights.append(f"Applied {stabilization_count} stabilization operations")

        # Log recovery
        if recovery_count > 0:
            self.recovery_history.append({
                "timestamp": datetime.now().isoformat(),
                "recoveries": recovery_count,
                "type": "automatic"
            })

        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["overall_health"] = np.mean([
            self._calculate_health(tbl.get_state(d)) for d in DomainLeg
        ])

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )
