"""
Consciousness Monitoring

Production monitoring and observability for NPCPU consciousness systems.
Provides metrics collection, alerting, and dashboarding support.

Based on Month 6 roadmap: Production Deployment - Monitoring & Observability
"""

import time
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Metric Types
# ============================================================================

class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"           # Current value
    COUNTER = "counter"       # Cumulative count
    HISTOGRAM = "histogram"   # Distribution
    SUMMARY = "summary"       # Statistical summary


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ConsciousnessMetrics:
    """Metrics for a consciousness instance"""
    timestamp: float
    overall_score: float
    capability_scores: Dict[str, float]
    state_description: str
    emergence_potential: float = 0.0
    adaptation_count: int = 0


@dataclass
class AgentMetrics:
    """Metrics for a single agent"""
    agent_id: str
    consciousness_metrics: ConsciousnessMetrics
    memory_usage: int = 0
    message_count: int = 0
    uptime_seconds: float = 0.0
    last_activity: float = 0.0


@dataclass
class AlertRule:
    """Rule for triggering alerts"""
    name: str
    metric: str
    operator: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: float = 60.0

    def check(self, value: float) -> bool:
        """Check if alert should trigger"""
        if self.operator == "gt":
            return value > self.threshold
        elif self.operator == "lt":
            return value < self.threshold
        elif self.operator == "eq":
            return value == self.threshold
        elif self.operator == "ne":
            return value != self.threshold
        return False


@dataclass
class Alert:
    """An alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    agent_id: Optional[str] = None
    metric_value: float = 0.0
    resolved: bool = False
    resolved_at: Optional[float] = None


# ============================================================================
# Consciousness Monitor
# ============================================================================

class ConsciousnessMonitor:
    """
    Monitor consciousness metrics for observability.

    Features:
    - Real-time metric collection
    - Historical metric storage
    - Alert rules and notifications
    - Prometheus-compatible metric export
    - Metric aggregation across agents
    - Performance tracking

    Example:
        monitor = ConsciousnessMonitor()

        # Register agents
        monitor.register_agent("agent_1", agent_consciousness)

        # Record metrics
        monitor.record_experience("agent_1")

        # Add alert rules
        monitor.add_alert_rule(AlertRule(
            name="low_consciousness",
            metric="consciousness_score",
            operator="lt",
            threshold=0.3,
            severity=AlertSeverity.WARNING,
            message_template="Agent {agent_id} consciousness dropped to {value:.2f}"
        ))

        # Get metrics
        metrics = monitor.get_prometheus_metrics()
    """

    def __init__(
        self,
        history_size: int = 1000,
        export_interval: float = 10.0
    ):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, float] = {}

        self.history_size = history_size
        self.export_interval = export_interval
        self._lock = threading.Lock()

        # Aggregated metrics
        self.global_metrics = {
            "total_agents": 0,
            "avg_consciousness_score": 0.0,
            "experiences_stored": 0,
            "queries_executed": 0,
            "total_uptime": 0.0
        }

    def register_agent(
        self,
        agent_id: str,
        consciousness: GradedConsciousness,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register an agent for monitoring"""
        with self._lock:
            self.agents[agent_id] = {
                "consciousness": consciousness,
                "registered_at": time.time(),
                "metadata": metadata or {},
                "message_count": 0,
                "experience_count": 0
            }
            self.global_metrics["total_agents"] = len(self.agents)

            # Record initial metrics
            self._record_consciousness_metrics(agent_id, consciousness)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                self.global_metrics["total_agents"] = len(self.agents)

    def update_consciousness(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """Update consciousness metrics for an agent"""
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id]["consciousness"] = consciousness
                self._record_consciousness_metrics(agent_id, consciousness)
                self._check_alerts(agent_id, consciousness)

    def _record_consciousness_metrics(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """Record consciousness metrics internally"""
        metrics = ConsciousnessMetrics(
            timestamp=time.time(),
            overall_score=consciousness.overall_consciousness_score(),
            capability_scores=consciousness.get_capability_scores(),
            state_description=consciousness.describe_state()
        )

        self.metrics_history[f"{agent_id}_consciousness"].append(metrics)

        # Update histograms
        self.histograms["consciousness_score"].append(metrics.overall_score)
        for cap, score in metrics.capability_scores.items():
            self.histograms[f"capability_{cap}"].append(score)

    def record_experience(self, agent_id: str):
        """Record an experience storage event"""
        with self._lock:
            self.counters["experiences_stored"] += 1
            self.global_metrics["experiences_stored"] += 1
            if agent_id in self.agents:
                self.agents[agent_id]["experience_count"] += 1

    def record_query(self, agent_id: str, latency_ms: float):
        """Record a query with latency"""
        with self._lock:
            self.counters["queries_executed"] += 1
            self.global_metrics["queries_executed"] += 1
            self.histograms["query_latency_ms"].append(latency_ms)

    def record_message(self, agent_id: str):
        """Record a message sent/received"""
        with self._lock:
            self.counters["messages_total"] += 1
            if agent_id in self.agents:
                self.agents[agent_id]["message_count"] += 1

    def add_alert_rule(self, rule: AlertRule):
        """Add an alerting rule"""
        self.alert_rules.append(rule)

    def _check_alerts(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """Check alert rules against current metrics"""
        now = time.time()
        score = consciousness.overall_consciousness_score()
        scores = consciousness.get_capability_scores()

        for rule in self.alert_rules:
            # Check cooldown
            last_time = self.last_alert_times.get(f"{rule.name}_{agent_id}", 0)
            if now - last_time < rule.cooldown_seconds:
                continue

            # Get metric value
            if rule.metric == "consciousness_score":
                value = score
            elif rule.metric.startswith("capability_"):
                cap_name = rule.metric.replace("capability_", "")
                value = scores.get(cap_name, 0)
            else:
                continue

            # Check rule
            if rule.check(value):
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        agent_id=agent_id,
                        value=value
                    ),
                    timestamp=now,
                    agent_id=agent_id,
                    metric_value=value
                )
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                self.last_alert_times[f"{rule.name}_{agent_id}"] = now

    def resolve_alert(self, rule_name: str, agent_id: Optional[str] = None):
        """Resolve an active alert"""
        now = time.time()
        for alert in self.active_alerts:
            if alert.rule_name == rule_name:
                if agent_id is None or alert.agent_id == agent_id:
                    alert.resolved = True
                    alert.resolved_at = now

        self.active_alerts = [a for a in self.active_alerts if not a.resolved]

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get current metrics for an agent"""
        if agent_id not in self.agents:
            return None

        agent_data = self.agents[agent_id]
        consciousness = agent_data["consciousness"]

        return AgentMetrics(
            agent_id=agent_id,
            consciousness_metrics=ConsciousnessMetrics(
                timestamp=time.time(),
                overall_score=consciousness.overall_consciousness_score(),
                capability_scores=consciousness.get_capability_scores(),
                state_description=consciousness.describe_state()
            ),
            message_count=agent_data.get("message_count", 0),
            uptime_seconds=time.time() - agent_data["registered_at"],
            last_activity=time.time()
        )

    def get_all_agent_metrics(self) -> List[AgentMetrics]:
        """Get metrics for all agents"""
        return [
            self.get_agent_metrics(agent_id)
            for agent_id in self.agents
            if self.get_agent_metrics(agent_id) is not None
        ]

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all agents"""
        if not self.agents:
            return self.global_metrics.copy()

        scores = [
            self.agents[aid]["consciousness"].overall_consciousness_score()
            for aid in self.agents
        ]

        return {
            **self.global_metrics,
            "avg_consciousness_score": sum(scores) / len(scores) if scores else 0,
            "min_consciousness_score": min(scores) if scores else 0,
            "max_consciousness_score": max(scores) if scores else 0,
            "active_alerts": len(self.active_alerts)
        }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # Global metrics
        for metric, value in self.global_metrics.items():
            lines.append(f"npcpu_{metric} {value}")

        # Counters
        for counter, value in self.counters.items():
            lines.append(f"npcpu_{counter}_total {value}")

        # Agent metrics
        for agent_id in self.agents:
            metrics = self.get_agent_metrics(agent_id)
            if metrics:
                lines.append(
                    f'npcpu_agent_consciousness_score{{agent_id="{agent_id}"}} '
                    f'{metrics.consciousness_metrics.overall_score}'
                )
                lines.append(
                    f'npcpu_agent_uptime_seconds{{agent_id="{agent_id}"}} '
                    f'{metrics.uptime_seconds}'
                )
                lines.append(
                    f'npcpu_agent_message_count{{agent_id="{agent_id}"}} '
                    f'{metrics.message_count}'
                )

                # Capability scores
                for cap, score in metrics.consciousness_metrics.capability_scores.items():
                    lines.append(
                        f'npcpu_agent_capability_score{{agent_id="{agent_id}",capability="{cap}"}} '
                        f'{score}'
                    )

        # Histogram summaries
        for histogram, values in self.histograms.items():
            if values:
                import numpy as np
                lines.append(f'npcpu_{histogram}_count {len(values)}')
                lines.append(f'npcpu_{histogram}_sum {sum(values)}')
                lines.append(f'npcpu_{histogram}_p50 {np.percentile(values, 50)}')
                lines.append(f'npcpu_{histogram}_p95 {np.percentile(values, 95)}')
                lines.append(f'npcpu_{histogram}_p99 {np.percentile(values, 99)}')

        return '\n'.join(lines)

    def get_json_metrics(self) -> str:
        """Export metrics as JSON"""
        import numpy as np

        data = {
            "timestamp": time.time(),
            "global_metrics": self.get_aggregated_metrics(),
            "counters": dict(self.counters),
            "agents": {},
            "alerts": {
                "active": len(self.active_alerts),
                "total_triggered": len(self.alert_history)
            }
        }

        for agent_id in self.agents:
            metrics = self.get_agent_metrics(agent_id)
            if metrics:
                data["agents"][agent_id] = {
                    "consciousness_score": metrics.consciousness_metrics.overall_score,
                    "state": metrics.consciousness_metrics.state_description,
                    "uptime_seconds": metrics.uptime_seconds,
                    "capabilities": metrics.consciousness_metrics.capability_scores
                }

        # Histogram stats
        data["histograms"] = {}
        for histogram, values in self.histograms.items():
            if values:
                data["histograms"][histogram] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95))
                }

        return json.dumps(data, indent=2)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        critical_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.WARNING]

        if critical_alerts:
            status = "critical"
        elif warning_alerts:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "total_agents": len(self.agents),
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "uptime": time.time() - min(
                (self.agents[aid]["registered_at"] for aid in self.agents),
                default=time.time()
            )
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Consciousness Monitor Demo")
    print("=" * 50)

    # Create monitor
    monitor = ConsciousnessMonitor()

    # Add alert rules
    monitor.add_alert_rule(AlertRule(
        name="low_consciousness",
        metric="consciousness_score",
        operator="lt",
        threshold=0.3,
        severity=AlertSeverity.WARNING,
        message_template="Agent {agent_id} consciousness dropped to {value:.2f}"
    ))

    monitor.add_alert_rule(AlertRule(
        name="critical_consciousness",
        metric="consciousness_score",
        operator="lt",
        threshold=0.1,
        severity=AlertSeverity.CRITICAL,
        message_template="CRITICAL: Agent {agent_id} consciousness at {value:.2f}"
    ))

    # Register agents
    print("\n1. Registering agents...")
    for i in range(3):
        consciousness = GradedConsciousness(
            perception_fidelity=0.5 + i * 0.1,
            introspection_capacity=0.6 + i * 0.1
        )
        monitor.register_agent(f"agent_{i}", consciousness)

    # Record some activity
    print("\n2. Recording activity...")
    for i in range(10):
        monitor.record_experience(f"agent_{i % 3}")
        monitor.record_query(f"agent_{i % 3}", latency_ms=50 + i * 5)
        monitor.record_message(f"agent_{i % 3}")

    # Update consciousness (trigger alert)
    print("\n3. Updating consciousness (triggering alert)...")
    low_consciousness = GradedConsciousness(
        perception_fidelity=0.1,
        introspection_capacity=0.1
    )
    monitor.update_consciousness("agent_0", low_consciousness)

    # Get metrics
    print("\n4. Getting metrics...")
    print("   Aggregated metrics:")
    for key, value in monitor.get_aggregated_metrics().items():
        print(f"     {key}: {value}")

    print("\n   Health status:")
    health = monitor.get_health_status()
    for key, value in health.items():
        print(f"     {key}: {value}")

    print("\n   Active alerts:")
    for alert in monitor.active_alerts:
        print(f"     [{alert.severity.value}] {alert.message}")

    print("\n5. Prometheus metrics (sample):")
    prom_metrics = monitor.get_prometheus_metrics()
    for line in prom_metrics.split('\n')[:10]:
        print(f"   {line}")
