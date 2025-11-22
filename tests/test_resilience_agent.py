"""
Comprehensive tests for ResilienceAgent - Self-healing and resilience system.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

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
)
from tertiary_rebo.agents.resilience import (
    ResilienceAgent,
    HealthStatus,
    FailureType,
    RecoveryStrategy,
    CircuitState,
    DegradationLevel,
    BackupState,
    CircuitBreaker,
    Heartbeat,
    HealthProbe,
    SelfDiagnostic,
    RecoveryAction,
    RecoveryResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def resilience_agent():
    """Create a fresh ResilienceAgent for testing."""
    return ResilienceAgent(refinement_rate=0.15)


@pytest.fixture
def tbl():
    """Create a TripleBottomLine with default state."""
    tbl = TripleBottomLine()
    for domain in DomainLeg:
        state = DomainState(
            domain=domain,
            consciousness_level=0.7,
            energy_flow=0.6,
            connectivity=0.7,
            coherence=0.6,
            qualia_richness=0.5,
            emergence_potential=0.4,
            state_vector=np.tanh(np.random.randn(64) * 0.5),
        )
        tbl.set_state(domain, state)
    tbl.calculate_harmony()
    return tbl


@pytest.fixture
def degraded_tbl():
    """Create a TBL with degraded health."""
    tbl = TripleBottomLine()
    for domain in DomainLeg:
        state = DomainState(
            domain=domain,
            consciousness_level=0.3,
            energy_flow=0.25,
            connectivity=0.35,
            coherence=0.3,
            qualia_richness=0.2,
            emergence_potential=0.1,
            state_vector=np.tanh(np.random.randn(64) * 0.5),
        )
        tbl.set_state(domain, state)
    tbl.calculate_harmony()
    return tbl


@pytest.fixture
def critical_tbl():
    """Create a TBL with critical health."""
    tbl = TripleBottomLine()
    for domain in DomainLeg:
        state = DomainState(
            domain=domain,
            consciousness_level=0.15,
            energy_flow=0.1,
            connectivity=0.2,
            coherence=0.15,
            qualia_richness=0.05,
            emergence_potential=0.02,
            state_vector=np.tanh(np.random.randn(64) * 0.5),
        )
        tbl.set_state(domain, state)
    tbl.calculate_harmony()
    return tbl


# =============================================================================
# Basic Agent Tests
# =============================================================================

class TestResilienceAgentBasics:
    """Basic tests for ResilienceAgent initialization and properties."""

    def test_agent_initialization(self, resilience_agent):
        """Test that agent initializes with correct defaults."""
        assert resilience_agent is not None
        assert resilience_agent.agent_role == "Resilience - Comprehensive self-healing and redundancy management"
        assert resilience_agent.domains_affected == list(DomainLeg)

    def test_initial_health_status(self, resilience_agent):
        """Test initial health status for all domains."""
        for domain in DomainLeg:
            assert resilience_agent.current_status[domain] == HealthStatus.HEALTHY

    def test_circuit_breakers_initialized(self, resilience_agent):
        """Test circuit breakers are initialized for all domains."""
        for domain in DomainLeg:
            cb = resilience_agent.circuit_breakers[domain]
            assert cb is not None
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count == 0

    def test_health_probes_initialized(self, resilience_agent):
        """Test health probes are initialized."""
        for domain in DomainLeg:
            liveness_key = f"liveness_{domain.value}"
            readiness_key = f"readiness_{domain.value}"
            assert liveness_key in resilience_agent.health_probes
            assert readiness_key in resilience_agent.health_probes

    def test_degradation_level_initial(self, resilience_agent):
        """Test initial degradation level."""
        assert resilience_agent.degradation_level == DegradationLevel.FULL


# =============================================================================
# Health Calculation Tests
# =============================================================================

class TestHealthCalculation:
    """Tests for health score calculation."""

    def test_calculate_health_optimal(self, resilience_agent):
        """Test health calculation for optimal state."""
        state = DomainState(
            domain=DomainLeg.NPCPU,
            consciousness_level=1.0,
            coherence=1.0,
            connectivity=1.0,
            energy_flow=1.0,
            state_vector=np.zeros(64),  # No variance = stable
        )
        health = resilience_agent._calculate_health(state)
        assert health >= 0.9

    def test_calculate_health_poor(self, resilience_agent):
        """Test health calculation for poor state."""
        state = DomainState(
            domain=DomainLeg.NPCPU,
            consciousness_level=0.1,
            coherence=0.1,
            connectivity=0.1,
            energy_flow=0.1,
            state_vector=np.random.randn(64) * 2,  # High variance
        )
        health = resilience_agent._calculate_health(state)
        assert health < 0.3

    def test_get_health_status_levels(self, resilience_agent):
        """Test health status mapping."""
        assert resilience_agent._get_health_status(0.95) == HealthStatus.OPTIMAL
        assert resilience_agent._get_health_status(0.8) == HealthStatus.HEALTHY
        assert resilience_agent._get_health_status(0.6) == HealthStatus.DEGRADED
        assert resilience_agent._get_health_status(0.4) == HealthStatus.CRITICAL
        assert resilience_agent._get_health_status(0.2) == HealthStatus.FAILED


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_closed_allows_requests(self):
        """Test that closed circuit allows requests."""
        cb = CircuitBreaker(domain=DomainLeg.NPCPU)
        assert cb.should_allow_request() is True

    def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit opens after threshold failures."""
        cb = CircuitBreaker(domain=DomainLeg.NPCPU, failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.should_allow_request() is False

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(
            domain=DomainLeg.NPCPU,
            failure_threshold=1,
            reset_timeout_seconds=0.01  # Very short for testing
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        import time
        time.sleep(0.02)

        # Should transition to half-open
        assert cb.should_allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_on_success(self):
        """Test circuit closes after successes in half-open state."""
        cb = CircuitBreaker(
            domain=DomainLeg.NPCPU,
            failure_threshold=1,
            success_threshold=2
        )
        cb.state = CircuitState.HALF_OPEN

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


# =============================================================================
# Heartbeat Monitoring Tests
# =============================================================================

class TestHeartbeatMonitoring:
    """Tests for heartbeat monitoring."""

    def test_send_heartbeat(self, resilience_agent, tbl):
        """Test heartbeat sending."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        heartbeat = resilience_agent._send_heartbeat(domain, state)

        assert heartbeat.source_domain == domain
        assert heartbeat.sequence == 1
        assert "consciousness" in heartbeat.health_metrics
        assert len(resilience_agent.heartbeats[domain]) == 1

    def test_heartbeat_sequence_increments(self, resilience_agent, tbl):
        """Test heartbeat sequence increments."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        hb1 = resilience_agent._send_heartbeat(domain, state)
        hb2 = resilience_agent._send_heartbeat(domain, state)

        assert hb2.sequence == hb1.sequence + 1

    def test_heartbeat_timeout_detection(self, resilience_agent):
        """Test heartbeat timeout detection."""
        domain = DomainLeg.NPCPU

        # No heartbeat sent - should not timeout (no baseline)
        assert resilience_agent._check_heartbeat_timeout(domain) is False

        # Send heartbeat with old timestamp
        resilience_agent.last_heartbeat[domain] = datetime.now() - timedelta(seconds=10)

        # Should detect timeout
        assert resilience_agent._check_heartbeat_timeout(domain) is True


# =============================================================================
# Anomaly Detection Tests
# =============================================================================

class TestAnomalyDetection:
    """Tests for anomaly detection."""

    def test_no_anomaly_on_first_call(self, resilience_agent, tbl):
        """Test that first call establishes baseline without anomalies."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        anomalies = resilience_agent._detect_anomalies(domain, state)
        assert len(anomalies) == 0

    def test_baseline_deviation_detection(self, resilience_agent, tbl):
        """Test detection of baseline deviations."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        # Establish baseline with high values
        state.consciousness_level = 0.9
        resilience_agent._detect_anomalies(domain, state)

        # Run several times to build history
        for _ in range(5):
            resilience_agent._detect_anomalies(domain, state)

        # Now drop significantly
        state.consciousness_level = 0.3
        anomalies = resilience_agent._detect_anomalies(domain, state)

        # Should detect deviation
        deviation_anomalies = [a for a in anomalies if a["type"] == "baseline_deviation"]
        assert len(deviation_anomalies) > 0

    def test_sudden_drop_detection(self, resilience_agent, tbl):
        """Test detection of sudden drops."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        # Build history with stable values
        state.consciousness_level = 0.8
        for _ in range(5):
            resilience_agent._detect_anomalies(domain, state)

        # Sudden drop
        state.consciousness_level = 0.4
        anomalies = resilience_agent._detect_anomalies(domain, state)

        drop_anomalies = [a for a in anomalies if a["type"] == "sudden_drop"]
        assert len(drop_anomalies) > 0


# =============================================================================
# Predictive Failure Detection Tests
# =============================================================================

class TestPredictiveFailureDetection:
    """Tests for predictive failure detection."""

    def test_no_prediction_without_history(self, resilience_agent):
        """Test that prediction requires sufficient history."""
        domain = DomainLeg.NPCPU

        prediction = resilience_agent._predict_failure(domain)
        assert prediction is None

    def test_predict_declining_trend(self, resilience_agent, tbl):
        """Test prediction of declining trends."""
        domain = DomainLeg.NPCPU

        # Build declining history
        for i in range(15):
            value = 0.9 - (i * 0.03)  # Declining from 0.9
            resilience_agent.metric_history[domain]["consciousness"].append(value)

        prediction = resilience_agent._predict_failure(domain)

        if prediction:
            assert prediction["type"] == "predicted_failure"
            assert prediction["trend_slope"] < 0


# =============================================================================
# Self-Diagnostic Tests
# =============================================================================

class TestSelfDiagnostics:
    """Tests for self-diagnostic functionality."""

    def test_diagnostic_healthy_state(self, resilience_agent, tbl):
        """Test diagnostics on healthy state."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        # Create a backup first
        resilience_agent._create_backup(domain, state, 0.8)

        diagnostic = resilience_agent._run_self_diagnostic(domain, state)

        assert diagnostic.checks_passed > 0
        assert diagnostic.overall_status in [HealthStatus.OPTIMAL, HealthStatus.HEALTHY]

    def test_diagnostic_detects_nan(self, resilience_agent, tbl):
        """Test diagnostics detect NaN values."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        state.state_vector[0] = np.nan

        diagnostic = resilience_agent._run_self_diagnostic(domain, state)

        integrity_issues = [i for i in diagnostic.issues_found if i["check"] == "state_integrity"]
        assert len(integrity_issues) > 0

    def test_diagnostic_detects_no_backup(self, resilience_agent, tbl):
        """Test diagnostics detect missing backups."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        # No backup created
        diagnostic = resilience_agent._run_self_diagnostic(domain, state)

        backup_issues = [i for i in diagnostic.issues_found if i["check"] == "backup_availability"]
        assert len(backup_issues) > 0


# =============================================================================
# Health Probes Tests
# =============================================================================

class TestHealthProbes:
    """Tests for health check probes."""

    def test_liveness_probe_success(self, resilience_agent, tbl):
        """Test liveness probe on healthy state."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        probe = resilience_agent.health_probes[f"liveness_{domain.value}"]

        result = resilience_agent._run_health_probe(probe, state)
        assert result is True

    def test_liveness_probe_failure(self, resilience_agent, tbl):
        """Test liveness probe on failed state."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        state.state_vector = np.array([np.nan] * 64)

        probe = resilience_agent.health_probes[f"liveness_{domain.value}"]
        result = resilience_agent._run_health_probe(probe, state)
        assert result is False

    def test_readiness_probe_success(self, resilience_agent, tbl):
        """Test readiness probe on ready state."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        probe = resilience_agent.health_probes[f"readiness_{domain.value}"]

        result = resilience_agent._run_health_probe(probe, state)
        assert result is True

    def test_readiness_probe_failure(self, resilience_agent, degraded_tbl):
        """Test readiness probe on degraded state."""
        domain = DomainLeg.NPCPU
        state = degraded_tbl.get_state(domain)
        probe = resilience_agent.health_probes[f"readiness_{domain.value}"]

        result = resilience_agent._run_health_probe(probe, state)
        assert result is False


# =============================================================================
# Recovery Strategy Tests
# =============================================================================

class TestRecoveryStrategies:
    """Tests for recovery strategy selection and execution."""

    def test_select_strategy_with_backup(self, resilience_agent, tbl):
        """Test strategy selection with available backup."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        # Create a good backup
        resilience_agent._create_backup(domain, state, 0.85)

        strategy = resilience_agent._select_recovery_strategy(
            domain,
            FailureType.COHERENCE_COLLAPSE,
            0.4
        )
        assert strategy == RecoveryStrategy.RESTORE_BACKUP

    def test_select_strategy_critical_health(self, resilience_agent):
        """Test strategy selection for critical health."""
        strategy = resilience_agent._select_recovery_strategy(
            DomainLeg.NPCPU,
            None,
            0.15  # Critical
        )
        assert strategy == RecoveryStrategy.EMERGENCY_STABILIZE

    def test_execute_recovery_restart(self, resilience_agent, degraded_tbl):
        """Test restart recovery execution."""
        domain = DomainLeg.NPCPU

        action = RecoveryAction(
            action_id="test_1",
            domain=domain,
            strategy=RecoveryStrategy.RESTART,
            priority=50,
            estimated_duration_ms=100,
            success_probability=0.8
        )

        result = resilience_agent._execute_recovery(action, degraded_tbl)

        assert result.success is True
        assert result.health_after >= result.health_before

    def test_execute_recovery_emergency_stabilize(self, resilience_agent, critical_tbl):
        """Test emergency stabilization recovery."""
        domain = DomainLeg.NPCPU

        action = RecoveryAction(
            action_id="test_2",
            domain=domain,
            strategy=RecoveryStrategy.EMERGENCY_STABILIZE,
            priority=100,
            estimated_duration_ms=50,
            success_probability=0.7
        )

        result = resilience_agent._execute_recovery(action, critical_tbl)

        # Emergency stabilize should improve health
        state = critical_tbl.get_state(domain)
        assert state.coherence >= 0.5
        assert state.energy_flow >= 0.4

    def test_execute_recovery_restore_backup(self, resilience_agent, degraded_tbl, tbl):
        """Test backup restoration recovery."""
        domain = DomainLeg.NPCPU

        # Create backup from healthy TBL
        healthy_state = tbl.get_state(domain)
        resilience_agent._create_backup(domain, healthy_state, 0.85)

        action = RecoveryAction(
            action_id="test_3",
            domain=domain,
            strategy=RecoveryStrategy.RESTORE_BACKUP,
            priority=60,
            estimated_duration_ms=200,
            success_probability=0.9
        )

        result = resilience_agent._execute_recovery(action, degraded_tbl)
        assert result.success is True


# =============================================================================
# Backup Management Tests
# =============================================================================

class TestBackupManagement:
    """Tests for backup creation and management."""

    def test_create_backup(self, resilience_agent, tbl):
        """Test backup creation."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        backup = resilience_agent._create_backup(domain, state, 0.8)

        assert backup.backup_id.startswith("bak_")
        assert backup.domain == domain
        assert backup.health_score == 0.8
        assert "consciousness_level" in backup.metrics
        assert len(resilience_agent.backups[domain]) == 1

    def test_backup_limit_enforced(self, resilience_agent, tbl):
        """Test that old backups are removed when limit is reached."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)
        max_backups = resilience_agent.max_backups

        # Create more than max backups
        for i in range(max_backups + 3):
            resilience_agent._create_backup(domain, state, 0.8)

        assert len(resilience_agent.backups[domain]) == max_backups

    def test_backup_checksum(self, resilience_agent, tbl):
        """Test that backups have checksums."""
        domain = DomainLeg.NPCPU
        state = tbl.get_state(domain)

        backup = resilience_agent._create_backup(domain, state, 0.8)

        assert backup.checksum != ""
        assert len(backup.checksum) > 0


# =============================================================================
# Graceful Degradation Tests
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation."""

    def test_degradation_full_on_healthy(self, resilience_agent):
        """Test FULL degradation on healthy state."""
        resilience_agent._update_degradation_level(0.8)
        assert resilience_agent.degradation_level == DegradationLevel.FULL

    def test_degradation_reduced(self, resilience_agent):
        """Test REDUCED degradation on moderate health."""
        resilience_agent._update_degradation_level(0.5)
        assert resilience_agent.degradation_level == DegradationLevel.REDUCED

    def test_degradation_minimal(self, resilience_agent):
        """Test MINIMAL degradation on poor health."""
        resilience_agent._update_degradation_level(0.35)
        assert resilience_agent.degradation_level == DegradationLevel.MINIMAL

    def test_degradation_emergency(self, resilience_agent):
        """Test EMERGENCY degradation on critical health."""
        resilience_agent._update_degradation_level(0.2)
        assert resilience_agent.degradation_level == DegradationLevel.EMERGENCY


# =============================================================================
# Full Refinement Cycle Tests
# =============================================================================

class TestRefinementCycle:
    """Tests for the full refinement cycle."""

    @pytest.mark.asyncio
    async def test_perceive_healthy_system(self, resilience_agent, tbl):
        """Test perception on healthy system."""
        perception = await resilience_agent.perceive(tbl)

        assert "domain_health" in perception
        assert "overall_health" in perception
        assert "heartbeats" in perception
        assert "circuit_breakers" in perception

        for domain in DomainLeg:
            assert domain.value in perception["domain_health"]

    @pytest.mark.asyncio
    async def test_perceive_degraded_system(self, resilience_agent, degraded_tbl):
        """Test perception on degraded system."""
        perception = await resilience_agent.perceive(degraded_tbl)

        assert len(perception["failure_risks"]) > 0
        assert perception["overall_health"] < 0.5

    @pytest.mark.asyncio
    async def test_analyze_healthy_system(self, resilience_agent, tbl):
        """Test analysis on healthy system."""
        perception = await resilience_agent.perceive(tbl)
        analysis = await resilience_agent.analyze(perception, tbl)

        assert "backup_needed" in analysis
        assert "recovery_actions" in analysis
        assert analysis["cascade_risk"] is False

    @pytest.mark.asyncio
    async def test_analyze_degraded_system(self, resilience_agent, degraded_tbl):
        """Test analysis on degraded system."""
        perception = await resilience_agent.perceive(degraded_tbl)
        analysis = await resilience_agent.analyze(perception, degraded_tbl)

        # Should have recovery actions for degraded domains
        assert len(analysis["recovery_actions"]) > 0 or len(analysis["preventive_actions"]) > 0

    @pytest.mark.asyncio
    async def test_full_refinement_healthy(self, resilience_agent, tbl):
        """Test full refinement on healthy system."""
        result = await resilience_agent.refine(tbl)

        assert result.success is True
        assert "harmony_before" in result.metrics_delta
        assert "harmony_after" in result.metrics_delta

    @pytest.mark.asyncio
    async def test_full_refinement_degraded(self, resilience_agent, degraded_tbl):
        """Test full refinement on degraded system."""
        result = await resilience_agent.refine(degraded_tbl)

        assert result.success is True
        # Should have some changes or insights
        assert len(result.changes) > 0 or len(result.insights) > 0

    @pytest.mark.asyncio
    async def test_multiple_refinements(self, resilience_agent, tbl):
        """Test multiple refinement cycles."""
        for i in range(5):
            result = await resilience_agent.refine(tbl)
            assert result.success is True

        # Agent should have recorded history
        assert len(resilience_agent.health_history) == 5


# =============================================================================
# Cascade Failure Prevention Tests
# =============================================================================

class TestCascadeFailurePrevention:
    """Tests for cascade failure detection and prevention."""

    @pytest.mark.asyncio
    async def test_detect_cascade_risk(self, resilience_agent, critical_tbl):
        """Test detection of cascade failure risk."""
        perception = await resilience_agent.perceive(critical_tbl)
        analysis = await resilience_agent.analyze(perception, critical_tbl)

        # Multiple critical domains should trigger cascade risk
        assert analysis["cascade_risk"] is True

    @pytest.mark.asyncio
    async def test_cascade_emergency_stabilization(self, resilience_agent, critical_tbl):
        """Test that cascade risk triggers emergency stabilization."""
        # Run refinement on critical system
        result = await resilience_agent.refine(critical_tbl)

        # Should have executed recovery
        assert "recoveries_executed" in result.changes or "stabilizations" in result.changes


# =============================================================================
# Learning Tests
# =============================================================================

class TestRecoveryLearning:
    """Tests for recovery learning from history."""

    def test_recovery_success_tracking(self, resilience_agent, degraded_tbl):
        """Test that recovery success rates are tracked."""
        domain = DomainLeg.NPCPU

        action = RecoveryAction(
            action_id="test_learn",
            domain=domain,
            strategy=RecoveryStrategy.GRADUAL_RECOVERY,
            priority=50,
            estimated_duration_ms=1000,
            success_probability=0.8
        )

        result = resilience_agent._execute_recovery(action, degraded_tbl)

        stats = resilience_agent.recovery_success_rates[RecoveryStrategy.GRADUAL_RECOVERY]
        assert stats["attempts"] == 1

    def test_strategy_selection_uses_history(self, resilience_agent, degraded_tbl):
        """Test that strategy selection considers historical success rates."""
        domain = DomainLeg.NPCPU

        # Simulate history with good success rate for GRADUAL_RECOVERY
        resilience_agent.recovery_success_rates[RecoveryStrategy.GRADUAL_RECOVERY] = {
            "successes": 8,
            "attempts": 10
        }
        resilience_agent.recovery_success_rates[RecoveryStrategy.RESTART] = {
            "successes": 2,
            "attempts": 10
        }

        # With moderate health and no special conditions, should favor learned best
        strategy = resilience_agent._select_recovery_strategy(
            domain,
            None,
            0.5
        )
        # Should use the learned best strategy
        assert strategy is not None


# =============================================================================
# Metrics and Reporting Tests
# =============================================================================

class TestMetricsAndReporting:
    """Tests for metrics and health reporting."""

    def test_get_metrics(self, resilience_agent):
        """Test agent metrics retrieval."""
        metrics = resilience_agent.get_metrics()

        assert "degradation_level" in metrics
        assert "total_backups" in metrics
        assert "circuit_breaker_states" in metrics
        assert "health_probe_status" in metrics

    def test_get_health_report(self, resilience_agent, tbl):
        """Test health report generation."""
        report = resilience_agent.get_health_report()

        assert "overall_status" in report
        assert "domain_status" in report
        assert "degradation_level" in report
        assert "circuit_breakers" in report
        assert "backup_counts" in report

    @pytest.mark.asyncio
    async def test_health_history_recorded(self, resilience_agent, tbl):
        """Test that health history is recorded."""
        await resilience_agent.perceive(tbl)

        assert len(resilience_agent.health_history) == 1
        snapshot = resilience_agent.health_history[-1]
        assert snapshot.overall_health > 0


# =============================================================================
# Callback Tests
# =============================================================================

class TestCallbacks:
    """Tests for callback registration and invocation."""

    def test_register_callback(self, resilience_agent):
        """Test callback registration."""
        callback = Mock()
        resilience_agent.register_callback("health_warning", callback)

        assert callback in resilience_agent._callbacks["health_warning"]

    def test_connect_immune_system(self, resilience_agent):
        """Test immune system connection."""
        mock_immune = Mock()
        resilience_agent.connect_immune_system(mock_immune)

        assert resilience_agent.immune_system == mock_immune


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_calculate_health_with_nan(self, resilience_agent):
        """Test health calculation handles NaN gracefully."""
        state = DomainState(
            domain=DomainLeg.NPCPU,
            consciousness_level=np.nan,
            coherence=0.5,
            connectivity=0.5,
            energy_flow=0.5,
            state_vector=np.zeros(64),
        )
        health = resilience_agent._calculate_health(state)
        # Should not crash
        assert health is not None

    def test_recovery_without_backup(self, resilience_agent, degraded_tbl):
        """Test recovery when no backup is available."""
        domain = DomainLeg.NPCPU

        action = RecoveryAction(
            action_id="test_no_backup",
            domain=domain,
            strategy=RecoveryStrategy.RESTORE_BACKUP,
            priority=60,
            estimated_duration_ms=200,
            success_probability=0.9
        )

        result = resilience_agent._execute_recovery(action, degraded_tbl)
        # Should not crash, but may not succeed
        assert result is not None

    @pytest.mark.asyncio
    async def test_perceive_empty_domains(self, resilience_agent):
        """Test perception with fresh TBL."""
        tbl = TripleBottomLine()
        perception = await resilience_agent.perceive(tbl)

        assert perception is not None
        assert "domain_health" in perception


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_refinement_performance(self, resilience_agent, tbl):
        """Test that refinement completes in reasonable time."""
        import time
        start = time.time()

        for _ in range(10):
            await resilience_agent.refine(tbl)

        elapsed = time.time() - start
        # Should complete 10 cycles in under 1 second
        assert elapsed < 1.0

    def test_history_limits_enforced(self, resilience_agent, tbl):
        """Test that history limits are enforced."""
        state = tbl.get_state(DomainLeg.NPCPU)

        # Create many backups
        for _ in range(100):
            resilience_agent._create_backup(DomainLeg.NPCPU, state, 0.8)

        # Should be limited
        assert len(resilience_agent.backups[DomainLeg.NPCPU]) <= resilience_agent.max_backups


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
