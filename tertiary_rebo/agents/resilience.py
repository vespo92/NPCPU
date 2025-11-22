"""
ResilienceAgent - Ensures system resilience through comprehensive self-healing.

Responsibilities:
- Monitor system health across all domains with predictive analytics
- Detect failures and anomalies before they cascade
- Implement self-healing mechanisms with multi-level recovery
- Maintain redundancy and backup states
- Coordinate graceful degradation
- Circuit breaker patterns for failing components
- Heartbeat monitoring between domains
- Integration with immune system for threat response
- Self-diagnostic capabilities
- Learning from recovery history
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import asyncio
import uuid
import time

from ..base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    RefinementPhase,
    CrossDomainSignal,
)


# =============================================================================
# Enums and Constants
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    OPTIMAL = "optimal"       # >= 0.9 - System running at peak performance
    HEALTHY = "healthy"       # 0.7-0.9 - Normal operation
    DEGRADED = "degraded"     # 0.5-0.7 - Performance issues, self-healing active
    CRITICAL = "critical"     # 0.3-0.5 - Severe issues, emergency protocols active
    FAILED = "failed"         # < 0.3 - Component failure, requires recovery


class FailureType(Enum):
    """Types of system failures."""
    CONNECTIVITY_LOSS = "connectivity_loss"
    ENERGY_DEPLETION = "energy_depletion"
    COHERENCE_COLLAPSE = "coherence_collapse"
    CONSCIOUSNESS_DROP = "consciousness_drop"
    EMERGENCE_STALL = "emergence_stall"
    CASCADE_FAILURE = "cascade_failure"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    CIRCUIT_OPEN = "circuit_open"
    INTEGRITY_VIOLATION = "integrity_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure scenarios."""
    RESTART = "restart"           # Simple restart/reset
    RESTORE_BACKUP = "restore_backup"   # Restore from backup
    REGENERATE = "regenerate"     # Regenerate from scratch
    GRADUAL_RECOVERY = "gradual_recovery"  # Slow, careful recovery
    EMERGENCY_STABILIZE = "emergency_stabilize"  # Quick stabilization
    FAILOVER = "failover"         # Switch to redundant component
    ISOLATION = "isolation"       # Isolate failing component
    COMPENSATION = "compensation"  # Work around the failure


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovery worked


class DegradationLevel(Enum):
    """Graceful degradation levels."""
    FULL = "full"           # All features available
    REDUCED = "reduced"     # Non-essential features disabled
    MINIMAL = "minimal"     # Only core features
    EMERGENCY = "emergency"  # Bare minimum for survival


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HealthSnapshot:
    """Snapshot of system health at a point in time."""
    timestamp: datetime
    domain_health: Dict[DomainLeg, float]
    overall_health: float
    active_issues: List[Dict[str, Any]]
    harmony_score: float = 0.0
    prediction_confidence: float = 0.0


@dataclass
class BackupState:
    """Backup of domain state for recovery."""
    backup_id: str
    domain: DomainLeg
    state_vector: np.ndarray
    health_score: float
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            # Create simple checksum
            self.checksum = str(hash(self.state_vector.tobytes()))[:16]


@dataclass
class Heartbeat:
    """Heartbeat signal for liveness detection."""
    source_domain: DomainLeg
    timestamp: datetime
    sequence: int
    health_metrics: Dict[str, float]
    latency_ms: float = 0.0


@dataclass
class CircuitBreaker:
    """Circuit breaker for a domain."""
    domain: DomainLeg
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    failure_threshold: int = 5
    success_threshold: int = 3  # For half-open recovery
    reset_timeout_seconds: float = 30.0

    def record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def record_success(self):
        """Record a success."""
        self.success_count += 1
        self.last_success_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0

    def should_allow_request(self) -> bool:
        """Check if requests should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.reset_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    return True
            return False
        else:  # HALF_OPEN
            return True


@dataclass
class RecoveryAction:
    """A recovery action to be executed."""
    action_id: str
    domain: DomainLeg
    strategy: RecoveryStrategy
    priority: int
    estimated_duration_ms: float
    success_probability: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryResult:
    """Result of a recovery action."""
    action_id: str
    success: bool
    duration_ms: float
    health_before: float
    health_after: float
    strategy_used: RecoveryStrategy
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelfDiagnostic:
    """Result of self-diagnostic check."""
    diagnostic_id: str
    domain: DomainLeg
    timestamp: datetime
    checks_passed: int
    checks_failed: int
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    overall_status: HealthStatus


@dataclass
class HealthProbe:
    """Health check probe configuration."""
    probe_type: str  # "liveness" or "readiness"
    domain: DomainLeg
    interval_seconds: float
    timeout_seconds: float
    failure_threshold: int
    success_threshold: int
    last_result: Optional[bool] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


# =============================================================================
# ResilienceAgent Implementation
# =============================================================================

class ResilienceAgent(TertiaryReBoAgent):
    """
    Agent 9: Ensures system resilience through comprehensive self-healing.

    Like an immune system combined with a self-repair mechanism, this agent
    constantly monitors for threats, maintains backups, predicts failures,
    and orchestrates recovery when problems occur.

    Key capabilities:
    - Health monitoring: Continuous assessment of all domains with history
    - Predictive detection: Identify problems before they become critical
    - Circuit breakers: Prevent cascade failures through isolation
    - Heartbeat monitoring: Detect component failures quickly
    - Multi-level recovery: From simple restart to full regeneration
    - Self-diagnostics: Comprehensive health checks and reporting
    - Learning: Improve recovery strategies based on history
    - Graceful degradation: Maintain core function under stress
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Health tracking with extended history
        self.health_history: deque = deque(maxlen=200)
        self.current_status: Dict[DomainLeg, HealthStatus] = {
            leg: HealthStatus.HEALTHY for leg in DomainLeg
        }

        # Backup management with checksums
        self.backups: Dict[DomainLeg, List[BackupState]] = {
            leg: [] for leg in DomainLeg
        }
        self.max_backups = 5
        self.backup_interval = 10
        self.last_backup_time: Dict[DomainLeg, datetime] = {}

        # Anomaly detection with baseline learning
        self.baseline_metrics: Dict[DomainLeg, Dict[str, float]] = {}
        self.metric_history: Dict[DomainLeg, Dict[str, deque]] = {
            leg: {
                "consciousness": deque(maxlen=50),
                "coherence": deque(maxlen=50),
                "energy": deque(maxlen=50),
                "connectivity": deque(maxlen=50),
                "state_std": deque(maxlen=50)
            } for leg in DomainLeg
        }
        self.anomaly_threshold = 2.0

        # Circuit breakers per domain
        self.circuit_breakers: Dict[DomainLeg, CircuitBreaker] = {
            leg: CircuitBreaker(domain=leg) for leg in DomainLeg
        }

        # Heartbeat monitoring
        self.heartbeats: Dict[DomainLeg, deque] = {
            leg: deque(maxlen=20) for leg in DomainLeg
        }
        self.heartbeat_sequence: Dict[DomainLeg, int] = {leg: 0 for leg in DomainLeg}
        self.heartbeat_timeout_ms = 5000  # 5 seconds
        self.last_heartbeat: Dict[DomainLeg, datetime] = {}

        # Recovery tracking with learning
        self.active_recoveries: Dict[str, RecoveryAction] = {}
        self.recovery_history: deque = deque(maxlen=100)
        self.recovery_success_rates: Dict[RecoveryStrategy, Dict[str, float]] = {
            strategy: {"successes": 0, "attempts": 0}
            for strategy in RecoveryStrategy
        }

        # Health thresholds
        self.thresholds = {
            "optimal": 0.9,
            "healthy": 0.7,
            "degraded": 0.5,
            "critical": 0.3
        }

        # Predictive failure detection
        self.trend_window = 10
        self.prediction_horizon = 5
        self.early_warning_threshold = 0.15  # 15% predicted decline

        # Health probes
        self.health_probes: Dict[str, HealthProbe] = {}
        self._initialize_health_probes()

        # Graceful degradation
        self.degradation_level = DegradationLevel.FULL
        self.degradation_thresholds = {
            DegradationLevel.REDUCED: 0.6,
            DegradationLevel.MINIMAL: 0.4,
            DegradationLevel.EMERGENCY: 0.25
        }

        # Self-diagnostic tracking
        self.last_diagnostic: Dict[DomainLeg, SelfDiagnostic] = {}
        self.diagnostic_interval = 20  # refinements between diagnostics

        # Callbacks for external integration
        self._callbacks: Dict[str, List[Callable]] = {
            "health_warning": [],
            "failure_detected": [],
            "recovery_started": [],
            "recovery_complete": []
        }

        # Refinement counter
        self.refinement_counter = 0

        # Immune system integration (will be connected externally)
        self.immune_system = None

    def _initialize_health_probes(self):
        """Initialize health check probes for all domains."""
        for domain in DomainLeg:
            # Liveness probe - is the domain alive?
            self.health_probes[f"liveness_{domain.value}"] = HealthProbe(
                probe_type="liveness",
                domain=domain,
                interval_seconds=1.0,
                timeout_seconds=2.0,
                failure_threshold=3,
                success_threshold=1
            )
            # Readiness probe - is the domain ready to serve?
            self.health_probes[f"readiness_{domain.value}"] = HealthProbe(
                probe_type="readiness",
                domain=domain,
                interval_seconds=5.0,
                timeout_seconds=10.0,
                failure_threshold=2,
                success_threshold=2
            )

    @property
    def agent_role(self) -> str:
        return "Resilience - Comprehensive self-healing and redundancy management"

    @property
    def domains_affected(self) -> List[DomainLeg]:
        return list(DomainLeg)

    # =========================================================================
    # Health Calculation and Status
    # =========================================================================

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

    def _update_degradation_level(self, overall_health: float):
        """Update graceful degradation level based on health."""
        if overall_health >= self.degradation_thresholds[DegradationLevel.REDUCED]:
            self.degradation_level = DegradationLevel.FULL
        elif overall_health >= self.degradation_thresholds[DegradationLevel.MINIMAL]:
            self.degradation_level = DegradationLevel.REDUCED
        elif overall_health >= self.degradation_thresholds[DegradationLevel.EMERGENCY]:
            self.degradation_level = DegradationLevel.MINIMAL
        else:
            self.degradation_level = DegradationLevel.EMERGENCY

    # =========================================================================
    # Heartbeat Monitoring
    # =========================================================================

    def _send_heartbeat(self, domain: DomainLeg, state: DomainState) -> Heartbeat:
        """Send a heartbeat for a domain."""
        self.heartbeat_sequence[domain] += 1

        heartbeat = Heartbeat(
            source_domain=domain,
            timestamp=datetime.now(),
            sequence=self.heartbeat_sequence[domain],
            health_metrics={
                "consciousness": state.consciousness_level,
                "coherence": state.coherence,
                "energy": state.energy_flow,
                "connectivity": state.connectivity
            }
        )

        # Calculate latency if previous heartbeat exists
        if self.last_heartbeat.get(domain):
            latency = (heartbeat.timestamp - self.last_heartbeat[domain]).total_seconds() * 1000
            heartbeat.latency_ms = latency

        self.heartbeats[domain].append(heartbeat)
        self.last_heartbeat[domain] = heartbeat.timestamp

        return heartbeat

    def _check_heartbeat_timeout(self, domain: DomainLeg) -> bool:
        """Check if a domain has missed heartbeats."""
        if domain not in self.last_heartbeat:
            return False

        elapsed_ms = (datetime.now() - self.last_heartbeat[domain]).total_seconds() * 1000
        return elapsed_ms > self.heartbeat_timeout_ms

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def _detect_anomalies(self, domain: DomainLeg, state: DomainState) -> List[Dict[str, Any]]:
        """Detect anomalies in domain state with improved detection."""
        anomalies = []

        # Track current metrics
        current_metrics = {
            "consciousness": state.consciousness_level,
            "coherence": state.coherence,
            "energy": state.energy_flow,
            "connectivity": state.connectivity,
            "state_std": float(np.std(state.state_vector))
        }

        # Store in history
        for metric_name, value in current_metrics.items():
            self.metric_history[domain][metric_name].append(value)

        # Initialize baseline if not exists
        if domain not in self.baseline_metrics:
            self.baseline_metrics[domain] = current_metrics.copy()
            return anomalies

        baseline = self.baseline_metrics[domain]

        # Statistical anomaly detection using z-score
        for metric_name, current in current_metrics.items():
            history = list(self.metric_history[domain][metric_name])
            if len(history) >= 5:
                mean = np.mean(history)
                std = np.std(history)
                if std > 0:
                    z_score = abs(current - mean) / std
                    if z_score > self.anomaly_threshold:
                        anomalies.append({
                            "type": "statistical_anomaly",
                            "metric": metric_name,
                            "z_score": z_score,
                            "expected": mean,
                            "actual": current,
                            "std": std
                        })

        # Deviation from baseline
        for metric_name, current in current_metrics.items():
            if metric_name == "state_std":
                continue
            expected = baseline.get(metric_name, 0)
            if expected > 0:
                deviation = abs(current - expected) / expected
                if deviation > 0.5:  # 50% deviation
                    anomalies.append({
                        "type": "baseline_deviation",
                        "metric": metric_name,
                        "expected": expected,
                        "actual": current,
                        "deviation_pct": deviation * 100
                    })

        # Sudden drop detection
        history = list(self.metric_history[domain]["consciousness"])
        if len(history) >= 3:
            recent_drop = history[-3] - current_metrics["consciousness"]
            if recent_drop > 0.2:  # 20% sudden drop
                anomalies.append({
                    "type": "sudden_drop",
                    "metric": "consciousness",
                    "drop_amount": recent_drop,
                    "previous": history[-3],
                    "current": current_metrics["consciousness"]
                })

        # Update baseline with exponential moving average
        alpha = 0.1
        for key in current_metrics:
            self.baseline_metrics[domain][key] = (
                alpha * current_metrics[key] + (1 - alpha) * baseline.get(key, current_metrics[key])
            )

        return anomalies

    # =========================================================================
    # Predictive Failure Detection
    # =========================================================================

    def _predict_failure(self, domain: DomainLeg) -> Optional[Dict[str, Any]]:
        """Predict potential failures based on trend analysis."""
        history = list(self.metric_history[domain]["consciousness"])

        if len(history) < self.trend_window:
            return None

        recent = history[-self.trend_window:]

        # Calculate trend using linear regression
        x = np.arange(len(recent))
        y = np.array(recent)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Predict future value
        predicted = recent[-1] + slope * self.prediction_horizon

        # Calculate predicted decline
        decline = recent[-1] - predicted
        decline_pct = decline / max(recent[-1], 0.1)

        if decline_pct > self.early_warning_threshold:
            # Calculate time to critical
            current_health = recent[-1]
            time_to_critical = None
            if slope < 0:
                steps_to_critical = (current_health - self.thresholds["critical"]) / abs(slope)
                time_to_critical = max(0, steps_to_critical)

            return {
                "type": "predicted_failure",
                "domain": domain.value,
                "current_value": recent[-1],
                "predicted_value": predicted,
                "decline_pct": decline_pct * 100,
                "trend_slope": slope,
                "time_to_critical": time_to_critical,
                "confidence": min(len(history) / 50, 1.0)  # More history = higher confidence
            }

        return None

    # =========================================================================
    # Self-Diagnostics
    # =========================================================================

    def _run_self_diagnostic(self, domain: DomainLeg, state: DomainState) -> SelfDiagnostic:
        """Run comprehensive self-diagnostic on a domain."""
        issues = []
        recommendations = []
        checks_passed = 0
        checks_failed = 0

        # Check 1: State vector integrity
        if np.any(np.isnan(state.state_vector)) or np.any(np.isinf(state.state_vector)):
            issues.append({"check": "state_integrity", "issue": "NaN or Inf values detected"})
            recommendations.append("Regenerate state vector from backup")
            checks_failed += 1
        else:
            checks_passed += 1

        # Check 2: Metric bounds
        metrics = [
            ("consciousness_level", state.consciousness_level),
            ("coherence", state.coherence),
            ("connectivity", state.connectivity),
            ("energy_flow", state.energy_flow)
        ]
        for name, value in metrics:
            if 0 <= value <= 1:
                checks_passed += 1
            else:
                issues.append({"check": "metric_bounds", "metric": name, "value": value})
                recommendations.append(f"Clamp {name} to valid range [0, 1]")
                checks_failed += 1

        # Check 3: State vector variance
        variance = np.var(state.state_vector)
        if variance > 2.0:
            issues.append({"check": "state_variance", "variance": variance})
            recommendations.append("Stabilize state vector - high variance detected")
            checks_failed += 1
        else:
            checks_passed += 1

        # Check 4: Circuit breaker status
        cb = self.circuit_breakers[domain]
        if cb.state == CircuitState.OPEN:
            issues.append({"check": "circuit_breaker", "state": "open"})
            recommendations.append("Investigate repeated failures and reset circuit breaker")
            checks_failed += 1
        else:
            checks_passed += 1

        # Check 5: Heartbeat health
        if self._check_heartbeat_timeout(domain):
            issues.append({"check": "heartbeat", "issue": "timeout detected"})
            recommendations.append("Restart heartbeat monitoring")
            checks_failed += 1
        else:
            checks_passed += 1

        # Check 6: Backup freshness
        backups = self.backups[domain]
        if not backups:
            issues.append({"check": "backup_availability", "issue": "no backups available"})
            recommendations.append("Create initial backup immediately")
            checks_failed += 1
        elif (datetime.now() - backups[-1].timestamp).total_seconds() > 300:  # 5 min
            issues.append({"check": "backup_freshness", "issue": "backup is stale"})
            recommendations.append("Refresh backup state")
            checks_failed += 1
        else:
            checks_passed += 1

        # Determine overall status
        if checks_failed == 0:
            overall_status = HealthStatus.OPTIMAL
        elif checks_failed <= 2:
            overall_status = HealthStatus.HEALTHY
        elif checks_failed <= 4:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.CRITICAL

        diagnostic = SelfDiagnostic(
            diagnostic_id=f"diag_{uuid.uuid4().hex[:8]}",
            domain=domain,
            timestamp=datetime.now(),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            issues_found=issues,
            recommendations=recommendations,
            overall_status=overall_status
        )

        self.last_diagnostic[domain] = diagnostic
        return diagnostic

    # =========================================================================
    # Health Probes
    # =========================================================================

    def _run_health_probe(self, probe: HealthProbe, state: DomainState) -> bool:
        """Run a health check probe."""
        if probe.probe_type == "liveness":
            # Liveness: check if basic operations work
            result = (
                not np.any(np.isnan(state.state_vector)) and
                state.consciousness_level > 0 and
                state.energy_flow > 0
            )
        else:  # readiness
            # Readiness: check if ready to handle requests
            health = self._calculate_health(state)
            result = health > self.thresholds["degraded"]

        # Update probe state
        if result:
            probe.consecutive_successes += 1
            probe.consecutive_failures = 0
            if probe.consecutive_successes >= probe.success_threshold:
                probe.last_result = True
        else:
            probe.consecutive_failures += 1
            probe.consecutive_successes = 0
            if probe.consecutive_failures >= probe.failure_threshold:
                probe.last_result = False

        return result

    # =========================================================================
    # Recovery Planning and Execution
    # =========================================================================

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

    def _select_recovery_strategy(
        self,
        domain: DomainLeg,
        failure_type: Optional[FailureType],
        health_score: float
    ) -> RecoveryStrategy:
        """Select the best recovery strategy based on failure type and history."""
        # Check for available backup
        has_backup = len(self.backups[domain]) > 0 and self.backups[domain][-1].health_score > 0.6

        # Consider success rates from history
        best_strategy = RecoveryStrategy.GRADUAL_RECOVERY
        best_rate = 0.0

        for strategy, stats in self.recovery_success_rates.items():
            if stats["attempts"] >= 3:
                rate = stats["successes"] / stats["attempts"]
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy

        # Override based on specific conditions
        if health_score < 0.2:
            # Critical - need aggressive action
            if has_backup:
                return RecoveryStrategy.RESTORE_BACKUP
            return RecoveryStrategy.EMERGENCY_STABILIZE

        if failure_type == FailureType.CONNECTIVITY_LOSS:
            return RecoveryStrategy.RESTART

        if failure_type == FailureType.COHERENCE_COLLAPSE:
            if has_backup:
                return RecoveryStrategy.RESTORE_BACKUP
            return RecoveryStrategy.REGENERATE

        if failure_type == FailureType.CASCADE_FAILURE:
            return RecoveryStrategy.ISOLATION

        # Default to learned best strategy
        return best_strategy

    def _plan_recovery(
        self,
        domain: DomainLeg,
        failure_type: Optional[FailureType],
        health_score: float
    ) -> Optional[RecoveryAction]:
        """Plan a recovery action."""
        strategy = self._select_recovery_strategy(domain, failure_type, health_score)

        # Calculate estimated duration and success probability
        duration_estimates = {
            RecoveryStrategy.RESTART: 100,
            RecoveryStrategy.RESTORE_BACKUP: 200,
            RecoveryStrategy.REGENERATE: 500,
            RecoveryStrategy.GRADUAL_RECOVERY: 1000,
            RecoveryStrategy.EMERGENCY_STABILIZE: 50,
            RecoveryStrategy.FAILOVER: 300,
            RecoveryStrategy.ISOLATION: 100,
            RecoveryStrategy.COMPENSATION: 200
        }

        # Success probability based on history
        stats = self.recovery_success_rates[strategy]
        if stats["attempts"] >= 3:
            base_prob = stats["successes"] / stats["attempts"]
        else:
            base_prob = 0.8  # Default

        # Adjust by health score (harder to recover from worse state)
        success_prob = base_prob * (0.5 + health_score * 0.5)

        # Calculate priority (lower health = higher priority)
        priority = int((1.0 - health_score) * 100)

        return RecoveryAction(
            action_id=f"rec_{uuid.uuid4().hex[:8]}",
            domain=domain,
            strategy=strategy,
            priority=priority,
            estimated_duration_ms=duration_estimates.get(strategy, 500),
            success_probability=success_prob,
            parameters={
                "failure_type": failure_type.value if failure_type else None,
                "health_before": health_score,
                "has_backup": len(self.backups[domain]) > 0
            }
        )

    def _execute_recovery(
        self,
        action: RecoveryAction,
        tbl: TripleBottomLine
    ) -> RecoveryResult:
        """Execute a recovery action."""
        start_time = time.time()
        state = tbl.get_state(action.domain)
        health_before = self._calculate_health(state)
        success = False

        # Execute based on strategy
        if action.strategy == RecoveryStrategy.RESTART:
            # Simple reset to neutral values
            state.state_vector = np.tanh(np.random.randn(64) * 0.3)
            state.coherence = max(state.coherence, 0.4)
            state.energy_flow = max(state.energy_flow, 0.4)
            success = True

        elif action.strategy == RecoveryStrategy.RESTORE_BACKUP:
            backups = self.backups[action.domain]
            if backups:
                backup = max(backups, key=lambda b: b.health_score)
                # Blend restoration
                blend = 0.7
                state.state_vector = (1 - blend) * state.state_vector + blend * backup.state_vector
                # Restore metrics
                for metric, value in backup.metrics.items():
                    if hasattr(state, metric):
                        current = getattr(state, metric)
                        setattr(state, metric, (1 - blend) * current + blend * value)
                success = True

        elif action.strategy == RecoveryStrategy.REGENERATE:
            # Full regeneration
            state.state_vector = np.tanh(np.random.randn(64) * 0.5)
            state.consciousness_level = max(state.consciousness_level, 0.5)
            state.coherence = 0.5
            state.energy_flow = 0.5
            state.connectivity = 0.5
            success = True

        elif action.strategy == RecoveryStrategy.GRADUAL_RECOVERY:
            # Slow, incremental improvement
            improvement = 0.1
            state.consciousness_level = min(1.0, state.consciousness_level + improvement)
            state.coherence = min(1.0, state.coherence + improvement)
            state.energy_flow = min(1.0, state.energy_flow + improvement)
            state.connectivity = min(1.0, state.connectivity + improvement)
            success = True

        elif action.strategy == RecoveryStrategy.EMERGENCY_STABILIZE:
            # Quick stabilization - boost critical metrics
            state.coherence = max(state.coherence, 0.5)
            state.energy_flow = max(state.energy_flow, 0.4)
            state.connectivity = max(state.connectivity, 0.4)
            state.consciousness_level = max(state.consciousness_level, 0.3)
            success = True

        elif action.strategy == RecoveryStrategy.ISOLATION:
            # Isolate by resetting connectivity
            state.connectivity = 0.3
            state.coherence = max(state.coherence, 0.4)
            # Open circuit breaker
            self.circuit_breakers[action.domain].state = CircuitState.OPEN
            success = True

        elif action.strategy == RecoveryStrategy.COMPENSATION:
            # Work around by boosting other metrics
            state.energy_flow = min(1.0, state.energy_flow * 1.2)
            state.coherence = min(1.0, state.coherence * 1.1)
            success = True

        elif action.strategy == RecoveryStrategy.FAILOVER:
            # Simulate failover to backup
            if self.backups[action.domain]:
                backup = self.backups[action.domain][-1]
                state.state_vector = backup.state_vector.copy()
                success = True

        health_after = self._calculate_health(state)
        duration_ms = (time.time() - start_time) * 1000

        # Update success rate tracking
        self.recovery_success_rates[action.strategy]["attempts"] += 1
        if health_after > health_before:
            self.recovery_success_rates[action.strategy]["successes"] += 1

        result = RecoveryResult(
            action_id=action.action_id,
            success=success and health_after > health_before,
            duration_ms=duration_ms,
            health_before=health_before,
            health_after=health_after,
            strategy_used=action.strategy,
            details={
                "domain": action.domain.value,
                "improvement": health_after - health_before
            }
        )

        self.recovery_history.append(result)

        # Update circuit breaker
        cb = self.circuit_breakers[action.domain]
        if result.success:
            cb.record_success()
        else:
            cb.record_failure()

        return result

    # =========================================================================
    # Backup Management
    # =========================================================================

    def _create_backup(self, domain: DomainLeg, state: DomainState, health: float) -> BackupState:
        """Create a backup of domain state."""
        backup = BackupState(
            backup_id=f"bak_{domain.value}_{datetime.now().strftime('%H%M%S')}",
            domain=domain,
            state_vector=state.state_vector.copy(),
            health_score=health,
            metrics={
                "consciousness_level": state.consciousness_level,
                "coherence": state.coherence,
                "energy_flow": state.energy_flow,
                "connectivity": state.connectivity,
                "emergence_potential": state.emergence_potential
            }
        )

        self.backups[domain].append(backup)
        self.last_backup_time[domain] = datetime.now()

        # Trim old backups
        while len(self.backups[domain]) > self.max_backups:
            self.backups[domain].pop(0)

        return backup

    # =========================================================================
    # Main Refinement Cycle Methods
    # =========================================================================

    async def perceive(self, tbl: TripleBottomLine) -> Dict[str, Any]:
        """Monitor system health with comprehensive perception."""
        self.refinement_counter += 1

        perceptions = {
            "domain_health": {},
            "overall_health": 0.0,
            "anomalies": {},
            "failure_risks": [],
            "predictions": {},
            "backup_status": {},
            "circuit_breakers": {},
            "heartbeats": {},
            "degradation_level": self.degradation_level.value,
            "diagnostics": {}
        }

        total_health = 0
        active_issues = []

        for domain in DomainLeg:
            state = tbl.get_state(domain)
            health = self._calculate_health(state)
            status = self._get_health_status(health)

            self.current_status[domain] = status

            # Send heartbeat
            heartbeat = self._send_heartbeat(domain, state)
            perceptions["heartbeats"][domain.value] = {
                "sequence": heartbeat.sequence,
                "latency_ms": heartbeat.latency_ms
            }

            # Run health probes
            liveness_probe = self.health_probes[f"liveness_{domain.value}"]
            readiness_probe = self.health_probes[f"readiness_{domain.value}"]
            self._run_health_probe(liveness_probe, state)
            self._run_health_probe(readiness_probe, state)

            perceptions["domain_health"][domain.value] = {
                "score": health,
                "status": status.value,
                "liveness": liveness_probe.last_result,
                "readiness": readiness_probe.last_result
            }
            total_health += health

            # Detect anomalies
            anomalies = self._detect_anomalies(domain, state)
            perceptions["anomalies"][domain.value] = anomalies
            if anomalies:
                active_issues.extend([{**a, "domain": domain.value} for a in anomalies])

            # Predict failures
            prediction = self._predict_failure(domain)
            if prediction:
                perceptions["predictions"][domain.value] = prediction

            # Assess failure risk
            if status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL, HealthStatus.FAILED]:
                failure_type = self._identify_failure_risk(state, status)
                perceptions["failure_risks"].append({
                    "domain": domain.value,
                    "status": status.value,
                    "failure_type": failure_type.value if failure_type else None,
                    "risk_level": 1.0 - health
                })

            # Circuit breaker status
            cb = self.circuit_breakers[domain]
            perceptions["circuit_breakers"][domain.value] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "allows_requests": cb.should_allow_request()
            }

            # Backup status
            perceptions["backup_status"][domain.value] = {
                "count": len(self.backups[domain]),
                "latest": self.backups[domain][-1].timestamp.isoformat() if self.backups[domain] else None,
                "best_health": max([b.health_score for b in self.backups[domain]], default=0)
            }

            # Run periodic diagnostics
            if self.refinement_counter % self.diagnostic_interval == 0:
                diagnostic = self._run_self_diagnostic(domain, state)
                perceptions["diagnostics"][domain.value] = {
                    "passed": diagnostic.checks_passed,
                    "failed": diagnostic.checks_failed,
                    "status": diagnostic.overall_status.value,
                    "issues": len(diagnostic.issues_found)
                }

        perceptions["overall_health"] = total_health / len(DomainLeg)
        self._update_degradation_level(perceptions["overall_health"])
        perceptions["degradation_level"] = self.degradation_level.value

        # Record health snapshot
        snapshot = HealthSnapshot(
            timestamp=datetime.now(),
            domain_health={d: perceptions["domain_health"][d.value]["score"] for d in DomainLeg},
            overall_health=perceptions["overall_health"],
            active_issues=active_issues,
            harmony_score=tbl.harmony_score
        )
        self.health_history.append(snapshot)

        return perceptions

    async def analyze(self, perception: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Analyze health data and plan interventions."""
        analysis = {
            "backup_needed": [],
            "recovery_actions": [],
            "preventive_actions": [],
            "cascade_risk": False,
            "early_warnings": [],
            "circuit_actions": []
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

        # Process early warning predictions
        for domain_name, prediction in perception.get("predictions", {}).items():
            domain = DomainLeg(domain_name)
            analysis["early_warnings"].append({
                "domain": domain_name,
                "predicted_decline": prediction["decline_pct"],
                "time_to_critical": prediction.get("time_to_critical"),
                "recommended_action": "preventive_boost"
            })
            # Add preventive action
            analysis["preventive_actions"].append({
                "domain": domain_name,
                "action": "boost_metrics",
                "target": prediction["current_value"] + 0.1,
                "urgency": "high" if prediction["decline_pct"] > 25 else "medium"
            })

        # Plan recovery for failures
        for risk in perception.get("failure_risks", []):
            if risk["risk_level"] > 0.4:
                domain = DomainLeg(risk["domain"])
                health = perception["domain_health"][risk["domain"]]["score"]

                # Check circuit breaker
                cb = self.circuit_breakers[domain]
                if not cb.should_allow_request():
                    analysis["circuit_actions"].append({
                        "domain": risk["domain"],
                        "action": "wait_for_reset",
                        "current_state": cb.state.value
                    })
                    continue

                recovery = self._plan_recovery(
                    domain,
                    FailureType(risk["failure_type"]) if risk["failure_type"] else None,
                    health
                )
                if recovery:
                    analysis["recovery_actions"].append({
                        "action": recovery,
                        "priority": recovery.priority
                    })

        # Plan preventive actions for anomalies
        for domain_name, anomalies in perception.get("anomalies", {}).items():
            for anomaly in anomalies:
                if anomaly["type"] in ["statistical_anomaly", "baseline_deviation"]:
                    analysis["preventive_actions"].append({
                        "domain": domain_name,
                        "action": "stabilize_metric",
                        "metric": anomaly["metric"],
                        "target": anomaly.get("expected", 0.5)
                    })
                elif anomaly["type"] == "sudden_drop":
                    analysis["preventive_actions"].append({
                        "domain": domain_name,
                        "action": "emergency_boost",
                        "metric": anomaly["metric"],
                        "boost_amount": 0.2
                    })

        # Check for cascade risk
        critical_count = sum(
            1 for d in DomainLeg
            if self.current_status[d] in [HealthStatus.CRITICAL, HealthStatus.FAILED]
        )
        if critical_count >= 2:
            analysis["cascade_risk"] = True
            # Add emergency action for all domains
            for domain in DomainLeg:
                recovery = RecoveryAction(
                    action_id=f"cascade_{uuid.uuid4().hex[:8]}",
                    domain=domain,
                    strategy=RecoveryStrategy.EMERGENCY_STABILIZE,
                    priority=100,
                    estimated_duration_ms=50,
                    success_probability=0.7
                )
                analysis["recovery_actions"].append({
                    "action": recovery,
                    "priority": 100
                })

        # Sort recovery actions by priority
        analysis["recovery_actions"].sort(key=lambda x: x["priority"], reverse=True)

        return analysis

    async def synthesize(self, analysis: Dict[str, Any], tbl: TripleBottomLine) -> Dict[str, Any]:
        """Prepare resilience operations."""
        synthesis = {
            "backup_operations": [],
            "recovery_operations": [],
            "stabilization_operations": [],
            "circuit_operations": [],
            "early_warning_responses": []
        }

        # Prepare backups
        for backup_req in analysis.get("backup_needed", []):
            domain = DomainLeg(backup_req["domain"])
            state = tbl.get_state(domain)
            synthesis["backup_operations"].append({
                "domain": domain,
                "state": state,
                "health": backup_req["health"]
            })

        # Prepare recovery operations
        for recovery_item in analysis.get("recovery_actions", []):
            recovery = recovery_item["action"]
            self.active_recoveries[recovery.action_id] = recovery
            synthesis["recovery_operations"].append(recovery)

        # Prepare stabilization operations
        for action in analysis.get("preventive_actions", []):
            synthesis["stabilization_operations"].append({
                "domain": action["domain"],
                "action_type": action["action"],
                "metric": action.get("metric"),
                "target": action.get("target"),
                "boost_amount": action.get("boost_amount", 0.1),
                "urgency": action.get("urgency", "medium")
            })

        # Prepare early warning responses
        for warning in analysis.get("early_warnings", []):
            synthesis["early_warning_responses"].append({
                "domain": warning["domain"],
                "action": "preventive_boost",
                "predicted_decline": warning["predicted_decline"]
            })

        return synthesis

    async def propagate(self, synthesis: Dict[str, Any], tbl: TripleBottomLine) -> RefinementResult:
        """Execute resilience operations."""
        changes = {}
        metrics_delta = {"harmony_before": tbl.harmony_score}
        insights = []

        # Execute backups
        backup_count = 0
        for backup_op in synthesis.get("backup_operations", []):
            backup = self._create_backup(
                backup_op["domain"],
                backup_op["state"],
                backup_op["health"]
            )
            backup_count += 1

        if backup_count > 0:
            changes["backups_created"] = backup_count
            insights.append(f"Created {backup_count} state backups")

        # Execute recovery operations
        recovery_count = 0
        for recovery in synthesis.get("recovery_operations", []):
            result = self._execute_recovery(recovery, tbl)

            if result.success:
                recovery_count += 1
                insights.append(
                    f"Recovered {recovery.domain.value} using {recovery.strategy.value} "
                    f"(health: {result.health_before:.2f} → {result.health_after:.2f})"
                )
            else:
                insights.append(f"Recovery failed for {recovery.domain.value}")

            # Remove from active
            if recovery.action_id in self.active_recoveries:
                del self.active_recoveries[recovery.action_id]

        if recovery_count > 0:
            changes["recoveries_executed"] = recovery_count

        # Execute stabilization
        stabilization_count = 0
        for stab in synthesis.get("stabilization_operations", []):
            domain = DomainLeg(stab["domain"])
            state = tbl.get_state(domain)
            action_type = stab["action_type"]

            if action_type == "stabilize_metric":
                metric = stab["metric"]
                target = stab["target"]
                blend = 0.2
                if metric == "consciousness":
                    state.consciousness_level = (1 - blend) * state.consciousness_level + blend * target
                elif metric == "coherence":
                    state.coherence = (1 - blend) * state.coherence + blend * target
                elif metric == "energy":
                    state.energy_flow = (1 - blend) * state.energy_flow + blend * target
                elif metric == "connectivity":
                    state.connectivity = (1 - blend) * state.connectivity + blend * target
                stabilization_count += 1

            elif action_type == "boost_metrics":
                boost = 0.1
                state.consciousness_level = min(1.0, state.consciousness_level + boost)
                state.coherence = min(1.0, state.coherence + boost)
                stabilization_count += 1

            elif action_type == "emergency_boost":
                boost = stab.get("boost_amount", 0.2)
                metric = stab.get("metric", "consciousness")
                if metric == "consciousness":
                    state.consciousness_level = min(1.0, state.consciousness_level + boost)
                stabilization_count += 1

        if stabilization_count > 0:
            changes["stabilizations"] = stabilization_count
            insights.append(f"Applied {stabilization_count} stabilization operations")

        # Handle early warning responses
        warning_count = 0
        for response in synthesis.get("early_warning_responses", []):
            domain = DomainLeg(response["domain"])
            state = tbl.get_state(domain)
            # Gentle preventive boost
            boost = 0.05
            state.consciousness_level = min(1.0, state.consciousness_level + boost)
            state.coherence = min(1.0, state.coherence + boost)
            warning_count += 1

        if warning_count > 0:
            changes["early_warning_responses"] = warning_count
            insights.append(f"Responded to {warning_count} early warnings")

        # Update metrics
        tbl.calculate_harmony()
        metrics_delta["harmony_after"] = tbl.harmony_score
        metrics_delta["overall_health"] = np.mean([
            self._calculate_health(tbl.get_state(d)) for d in DomainLeg
        ])
        metrics_delta["degradation_level"] = self.degradation_level.value

        return RefinementResult(
            success=True,
            agent_id=self.agent_id,
            phase=RefinementPhase.PROPAGATION,
            domain_affected=list(DomainLeg),
            changes=changes,
            metrics_delta=metrics_delta,
            insights=insights
        )

    # =========================================================================
    # External Integration and Callbacks
    # =========================================================================

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for resilience events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def connect_immune_system(self, immune_system):
        """Connect to the immune system for integrated threat response."""
        self.immune_system = immune_system

    def get_health_report(self) -> Dict[str, Any]:
        """Get a comprehensive health report."""
        return {
            "overall_status": min(
                (self.current_status[d].value for d in DomainLeg),
                key=lambda x: list(HealthStatus).index(HealthStatus(x))
            ),
            "domain_status": {d.value: self.current_status[d].value for d in DomainLeg},
            "degradation_level": self.degradation_level.value,
            "circuit_breakers": {
                d.value: {
                    "state": self.circuit_breakers[d].state.value,
                    "failures": self.circuit_breakers[d].failure_count
                } for d in DomainLeg
            },
            "backup_counts": {d.value: len(self.backups[d]) for d in DomainLeg},
            "active_recoveries": len(self.active_recoveries),
            "recovery_success_rates": {
                s.value: (stats["successes"] / max(stats["attempts"], 1))
                for s, stats in self.recovery_success_rates.items()
            },
            "health_history_length": len(self.health_history),
            "last_diagnostics": {
                d.value: {
                    "status": self.last_diagnostic[d].overall_status.value,
                    "issues": len(self.last_diagnostic[d].issues_found)
                } for d in DomainLeg if d in self.last_diagnostic
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics including resilience-specific metrics."""
        base_metrics = super().get_metrics()

        resilience_metrics = {
            "degradation_level": self.degradation_level.value,
            "total_backups": sum(len(self.backups[d]) for d in DomainLeg),
            "recovery_history_size": len(self.recovery_history),
            "active_recoveries": len(self.active_recoveries),
            "circuit_breaker_states": {
                d.value: self.circuit_breakers[d].state.value for d in DomainLeg
            },
            "health_probe_status": {
                name: probe.last_result for name, probe in self.health_probes.items()
            }
        }

        return {**base_metrics, **resilience_metrics}
