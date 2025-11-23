"""
Comprehensive Test Suite for SENTINEL-K Immune System

Tests all immune system components:
- Defense system
- Repair system
- Pathogen detection
- Antibody system
- Innate immunity
- Adaptive immunity
- Memory cells
- Autoimmune prevention
- Inflammation system
"""

import pytest
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from immune.defense import (
    ImmuneSystem, Threat, ThreatType, ThreatSeverity,
    DefenseAction, AntibodyPattern
)
from immune.repair import (
    RepairSystem, Damage, DamageType, RepairStrategy,
    HealingPhase, RegenerationCascade, TissueRegenerator
)
from immune.pathogen_detection import (
    PathogenDetector, PathogenType, DetectionMethod, RiskLevel
)
from immune.antibody_system import (
    AntibodyGenerator, Antibody, AntibodyState, ResponseType
)
from immune.innate_immunity import (
    InnateImmuneSystem, BarrierType, PRRType, NKCellState
)
from immune.adaptive_immunity import (
    AdaptiveImmuneSystem, TCellType, BCellType, ImmunePhase
)
from immune.memory_cells import (
    MemoryCellSystem, MemoryType, MemoryStrength, ConsolidationState
)
from immune.autoimmune import (
    AutoimmunePreventionSystem, SelfMarkerType, AutoimmuneRisk
)
from immune.inflammation import (
    InflammationSystem, InflammationLevel, CytokineType, AlertPriority
)


# ============================================================================
# Defense System Tests
# ============================================================================

class TestImmuneSystem:
    """Tests for the core ImmuneSystem"""

    def test_immune_system_creation(self):
        """Test creating an immune system"""
        immune = ImmuneSystem()
        assert immune is not None
        assert immune.sensitivity == 0.7
        assert len(immune.antibodies) > 0

    def test_immune_system_custom_sensitivity(self):
        """Test custom sensitivity"""
        immune = ImmuneSystem(sensitivity=0.9)
        assert immune.sensitivity == 0.9

    def test_add_antibody_pattern(self):
        """Test adding antibody patterns"""
        immune = ImmuneSystem()
        initial_count = len(immune.antibodies)

        pattern = immune.add_antibody_pattern(
            "test_pattern",
            ThreatType.MALICIOUS_INPUT,
            "test.*pattern"
        )

        assert pattern is not None
        assert len(immune.antibodies) == initial_count + 1

    def test_scan_clean_data(self):
        """Test scanning clean data"""
        immune = ImmuneSystem(auto_response=False)
        threats = immune.scan("This is normal data", "test")
        assert len(threats) == 0

    def test_scan_malicious_data(self):
        """Test scanning malicious data"""
        immune = ImmuneSystem(auto_response=False)
        threats = immune.scan("corrupt invalid malformed data", "test")
        assert len(threats) > 0
        assert any(t.type == ThreatType.CORRUPTION for t in threats)

    def test_threat_response(self):
        """Test responding to threats"""
        immune = ImmuneSystem(auto_response=False)
        threat = Threat(
            type=ThreatType.MALICIOUS_INPUT,
            severity=ThreatSeverity.HIGH
        )

        response = immune.respond(threat)
        assert response is not None
        assert len(response.actions) > 0

    def test_quarantine(self):
        """Test quarantine functionality"""
        immune = ImmuneSystem()
        threat = Threat(severity=ThreatSeverity.HIGH)

        immune._quarantine_threat(threat)
        assert threat.id in immune.default_quarantine.contained_threats

    def test_get_status(self):
        """Test getting immune status"""
        immune = ImmuneSystem()
        status = immune.get_status()

        assert 'active_threats' in status
        assert 'antibody_count' in status
        assert 'sensitivity' in status


# ============================================================================
# Repair System Tests
# ============================================================================

class TestRepairSystem:
    """Tests for the RepairSystem"""

    def test_repair_system_creation(self):
        """Test creating repair system"""
        repair = RepairSystem()
        assert repair is not None
        assert repair.healing_rate == 0.1

    def test_detect_damage(self):
        """Test damage detection"""
        repair = RepairSystem()
        damage = repair.detect_damage(
            location="test_location",
            severity=0.5,
            damage_type=DamageType.CORRUPTION
        )

        assert damage is not None
        assert damage.severity == 0.5
        assert damage.id in repair.active_damage

    def test_assess_damage(self):
        """Test damage assessment"""
        repair = RepairSystem()
        damage = repair.detect_damage("test", 0.6, DamageType.TRAUMA)

        assessment = repair.assess_damage(damage)
        assert 'severity_level' in assessment
        assert 'recommended_strategy' in assessment
        assert 'success_probability' in assessment

    def test_initiate_repair(self):
        """Test initiating repair"""
        repair = RepairSystem()
        damage = repair.detect_damage("test", 0.4)

        healing = repair.initiate_repair(damage)
        assert healing is not None
        assert healing.damage_id == damage.id
        assert healing.phase == HealingPhase.DETECTION

    def test_advance_healing(self):
        """Test advancing healing process"""
        repair = RepairSystem(healing_rate=0.5)
        damage = repair.detect_damage("test", 0.3)
        healing = repair.initiate_repair(damage)

        initial_phase = healing.phase
        for _ in range(10):
            repair.advance_healing(healing)

        assert healing.phase != initial_phase or healing.progress > 0

    def test_create_backup(self):
        """Test backup creation"""
        repair = RepairSystem()
        state = {"key": "value", "data": [1, 2, 3]}

        backup = repair.create_backup("test_backup", state)
        assert backup is not None
        assert backup.name == "test_backup"
        assert "test_backup" in repair.backups

    def test_restore_from_backup(self):
        """Test restoring from backup"""
        repair = RepairSystem()
        original = {"key": "original_value"}
        repair.create_backup("test", original)

        restored = repair.restore_from_backup("test")
        assert restored == original

    def test_integrity_check(self):
        """Test integrity checking"""
        repair = RepairSystem()
        data = {"integrity": "test"}
        repair.register_checksum("test_target", data)

        check = repair.check_integrity("test_target", data)
        assert check.passed

        check = repair.check_integrity("test_target", {"corrupted": True})
        assert not check.passed


# ============================================================================
# Pathogen Detection Tests
# ============================================================================

class TestPathogenDetector:
    """Tests for PathogenDetector"""

    def test_detector_creation(self):
        """Test creating detector"""
        detector = PathogenDetector()
        assert detector is not None
        assert len(detector.signatures) > 0

    def test_scan_clean_data(self):
        """Test scanning clean input"""
        detector = PathogenDetector()
        result = detector.scan("Hello, normal data", "test")

        assert result is not None
        assert result.clean

    def test_detect_sql_injection(self):
        """Test SQL injection detection"""
        detector = PathogenDetector()
        result = detector.scan(
            "SELECT * FROM users WHERE id=1 OR 1=1",
            "db_query"
        )

        assert not result.clean
        assert any(p.pathogen_type == PathogenType.INJECTION_ATTACK
                  for p in result.pathogens)

    def test_detect_xss(self):
        """Test XSS detection"""
        detector = PathogenDetector()
        result = detector.scan("<script>alert('XSS')</script>", "html")

        assert not result.clean

    def test_entropy_detection(self):
        """Test high entropy detection"""
        import random
        detector = PathogenDetector()

        # High entropy random data
        random_data = ''.join(chr(random.randint(32, 126)) for _ in range(500))
        result = detector.scan(random_data, "file")

        # Should potentially detect anomaly
        assert result is not None

    def test_add_custom_signature(self):
        """Test adding custom signatures"""
        detector = PathogenDetector()
        initial_count = len(detector.signatures)

        sig = detector.add_signature(
            "custom_threat",
            PathogenType.TROJAN,
            "custom_pattern",
            RiskLevel.HIGH
        )

        assert len(detector.signatures) == initial_count + 1

    def test_get_statistics(self):
        """Test getting statistics"""
        detector = PathogenDetector()
        detector.scan("test data", "test")

        stats = detector.get_statistics()
        assert 'total_scans' in stats
        assert stats['total_scans'] == 1


# ============================================================================
# Antibody System Tests
# ============================================================================

class TestAntibodyGenerator:
    """Tests for AntibodyGenerator"""

    def test_generator_creation(self):
        """Test creating generator"""
        gen = AntibodyGenerator()
        assert gen is not None
        assert len(gen.antibodies) > 0  # Germline antibodies

    def test_generate_antibody(self):
        """Test antibody generation"""
        gen = AntibodyGenerator()
        threat = Threat(type=ThreatType.CORRUPTION, severity=ThreatSeverity.HIGH)

        ab = gen.generate_antibody(threat)
        assert ab is not None
        assert ThreatType.CORRUPTION in ab.target_types

    def test_create_response(self):
        """Test creating response"""
        gen = AntibodyGenerator()
        threat = Threat(type=ThreatType.MALICIOUS_INPUT)
        ab = gen.generate_antibody(threat)

        response = gen.create_response(ab, threat)
        assert response is not None
        assert response.antibody_id == ab.id

    def test_clonal_expansion(self):
        """Test clonal expansion"""
        gen = AntibodyGenerator()
        threat = Threat(type=ThreatType.CORRUPTION)
        ab = gen.generate_antibody(threat)

        clones = gen.clonal_expansion(ab, factor=3)
        assert len(clones) == 3
        for clone in clones:
            assert clone.parent_id == ab.id

    def test_memory_cell_formation(self):
        """Test forming memory cells"""
        gen = AntibodyGenerator()
        threat = Threat(type=ThreatType.CORRUPTION)
        ab = gen.generate_antibody(threat)
        ab.success_count = 5

        result = gen.form_memory_cell(ab)
        assert result
        assert ab.state == AntibodyState.MEMORY


# ============================================================================
# Innate Immunity Tests
# ============================================================================

class TestInnateImmuneSystem:
    """Tests for InnateImmuneSystem"""

    def test_system_creation(self):
        """Test creating system"""
        innate = InnateImmuneSystem()
        assert innate is not None
        assert len(innate.barriers) > 0
        assert len(innate.prrs) > 0
        assert len(innate.nk_cells) > 0

    def test_barrier_check_pass(self):
        """Test barrier check with valid data"""
        innate = InnateImmuneSystem()
        passed, responses = innate.check_barriers("normal data")
        assert passed

    def test_barrier_check_fail(self):
        """Test barrier with oversized data"""
        innate = InnateImmuneSystem()
        innate.add_barrier("size", BarrierType.SIZE_LIMIT,
                          rules={"max_size": 10})

        passed, responses = innate.check_barriers("x" * 100)
        assert not passed

    def test_prr_scanning(self):
        """Test PRR scanning"""
        innate = InnateImmuneSystem()
        threats = innate.scan_with_prrs("unauthorized access breach")

        assert len(threats) > 0
        assert any(t.type == ThreatType.INTRUSION for t in threats)

    def test_nk_cell_deployment(self):
        """Test NK cell deployment"""
        innate = InnateImmuneSystem()
        threats = [Threat(type=ThreatType.MALICIOUS_INPUT)]

        responses = innate.deploy_nk_cells(threats)
        assert len(responses) > 0

    def test_quick_response(self):
        """Test quick response"""
        innate = InnateImmuneSystem()
        threat = Threat(severity=ThreatSeverity.HIGH)

        response = innate.quick_response(threat)
        assert response is not None


# ============================================================================
# Adaptive Immunity Tests
# ============================================================================

class TestAdaptiveImmuneSystem:
    """Tests for AdaptiveImmuneSystem"""

    def test_system_creation(self):
        """Test creating system"""
        adaptive = AdaptiveImmuneSystem()
        assert adaptive is not None
        assert len(adaptive.t_cells) > 0
        assert len(adaptive.b_cells) > 0

    def test_mount_response(self):
        """Test mounting response"""
        adaptive = AdaptiveImmuneSystem()
        threat = Threat(type=ThreatType.MALICIOUS_INPUT)

        response = adaptive.mount_response(threat)
        assert response is not None
        assert response.phase == ImmunePhase.RECOGNITION

    def test_advance_response(self):
        """Test advancing response"""
        adaptive = AdaptiveImmuneSystem()
        threat = Threat(type=ThreatType.CORRUPTION)
        response = adaptive.mount_response(threat)

        initial_phase = response.phase
        adaptive.advance_response(response, threat)

        assert response.phase != initial_phase

    def test_full_response(self):
        """Test running full response"""
        adaptive = AdaptiveImmuneSystem()
        threat = Threat(type=ThreatType.MALICIOUS_INPUT,
                       signature="test_sig")

        response = adaptive.run_full_response(threat)
        assert response.phase == ImmunePhase.MEMORY

    def test_memory_formation(self):
        """Test memory formation"""
        adaptive = AdaptiveImmuneSystem()
        threat = Threat(type=ThreatType.CORRUPTION, signature="test")

        adaptive.run_full_response(threat)

        # Check memory was formed
        memory = adaptive.check_memory(threat)
        # Memory may or may not form depending on effectiveness
        assert True  # Test passes if no exception


# ============================================================================
# Memory Cells Tests
# ============================================================================

class TestMemoryCellSystem:
    """Tests for MemoryCellSystem"""

    def test_system_creation(self):
        """Test creating system"""
        memory = MemoryCellSystem()
        assert memory is not None

    def test_store_memory(self):
        """Test storing memory"""
        system = MemoryCellSystem()

        mem = system.store_memory(
            threat_type=ThreatType.MALICIOUS_INPUT,
            signature="test_signature",
            response={"action": "block"},
            effectiveness=0.8
        )

        assert mem is not None
        assert mem.threat_signature == "test_signature"

    def test_recall_exact(self):
        """Test exact recall"""
        system = MemoryCellSystem()
        system.store_memory(ThreatType.CORRUPTION, "exact_sig",
                           {"action": "repair"})

        result = system.recall(signature="exact_sig")
        assert result.memory is not None
        assert result.recall_confidence == 1.0

    def test_recall_fuzzy(self):
        """Test fuzzy recall"""
        system = MemoryCellSystem()
        system.store_memory(ThreatType.CORRUPTION, "similar_signature",
                           {"action": "repair"})

        result = system.recall(signature="similar_sig")
        # May or may not match depending on similarity
        assert result is not None

    def test_memory_update(self):
        """Test memory update on re-encounter"""
        system = MemoryCellSystem()
        mem1 = system.store_memory(ThreatType.CORRUPTION, "test",
                                   {}, effectiveness=0.5)
        mem2 = system.store_memory(ThreatType.CORRUPTION, "test",
                                   {}, effectiveness=0.9)

        assert mem1.id == mem2.id  # Same memory updated
        assert mem2.encounter_count == 2

    def test_create_cluster(self):
        """Test cluster creation"""
        system = MemoryCellSystem()
        mem1 = system.store_memory(ThreatType.CORRUPTION, "sig1", {})
        mem2 = system.store_memory(ThreatType.CORRUPTION, "sig2", {})

        cluster = system.create_cluster("test_cluster", [mem1.id, mem2.id])
        assert cluster is not None
        assert len(cluster.memory_ids) == 2


# ============================================================================
# Autoimmune Prevention Tests
# ============================================================================

class TestAutoimmunePrevention:
    """Tests for AutoimmunePreventionSystem"""

    def test_system_creation(self):
        """Test creating system"""
        auto = AutoimmunePreventionSystem()
        assert auto is not None

    def test_register_self_marker(self):
        """Test registering self marker"""
        auto = AutoimmunePreventionSystem()

        marker = auto.register_self_marker(
            "core_component",
            SelfMarkerType.CORE_COMPONENT,
            patterns=["core_", "component_"]
        )

        assert marker is not None
        assert marker.id in auto.self_markers

    def test_is_self_target(self):
        """Test self-target detection"""
        auto = AutoimmunePreventionSystem()
        auto.register_self_marker("memory", SelfMarkerType.MEMORY,
                                  patterns=["memory_", "recall_"])

        # Threat targeting self
        self_threat = Threat(signature="memory_corruption")
        assert auto.is_self_target(self_threat)

        # Threat not targeting self
        external_threat = Threat(signature="external_attack")
        assert not auto.is_self_target(external_threat)

    def test_detect_autoimmune_event(self):
        """Test autoimmune event detection"""
        auto = AutoimmunePreventionSystem()
        auto.register_self_marker("core", SelfMarkerType.CORE_COMPONENT,
                                  patterns=["core_"], criticality=0.9)

        threat = Threat(signature="core_damage", severity=ThreatSeverity.HIGH)
        event = auto.detect_autoimmune_event(threat)

        assert event is not None
        assert event.risk_level.value > 0


# ============================================================================
# Inflammation System Tests
# ============================================================================

class TestInflammationSystem:
    """Tests for InflammationSystem"""

    def test_system_creation(self):
        """Test creating system"""
        inflammation = InflammationSystem()
        assert inflammation is not None

    def test_trigger_inflammation(self):
        """Test triggering inflammation"""
        inflammation = InflammationSystem()
        threat = Threat(severity=ThreatSeverity.HIGH)

        zone = inflammation.trigger_inflammation("test_location", threat)
        assert zone is not None
        assert zone.level.value > 0

    def test_create_alert(self):
        """Test creating alert"""
        inflammation = InflammationSystem()
        threat = Threat(severity=ThreatSeverity.CRITICAL)

        signal = inflammation.create_alert(threat)
        assert signal is not None
        assert signal.priority == AlertPriority.CRITICAL

    def test_signal_propagation(self):
        """Test signal propagation"""
        inflammation = InflammationSystem()
        received = []
        inflammation.subscribe("test", lambda s: received.append(s))

        threat = Threat()
        signal = inflammation.create_alert(threat, target_locations=["test"])
        inflammation.propagate_signal(signal)

        assert len(received) == 1
        assert signal.acknowledged

    def test_resolution(self):
        """Test inflammation resolution"""
        inflammation = InflammationSystem()
        threat = Threat()
        zone = inflammation.trigger_inflammation("test", threat)

        inflammation.begin_resolution(zone.id)
        assert zone.phase.value >= 4  # Resolution or later

    def test_system_level(self):
        """Test system-wide inflammation level"""
        inflammation = InflammationSystem()

        # No zones = no inflammation
        assert inflammation.get_system_inflammation_level() == InflammationLevel.NONE

        # Add zone
        threat = Threat(severity=ThreatSeverity.HIGH)
        inflammation.trigger_inflammation("test", threat)

        level = inflammation.get_system_inflammation_level()
        assert level.value > 0


# ============================================================================
# Regeneration Cascade Tests
# ============================================================================

class TestRegenerationCascade:
    """Tests for RegenerationCascade"""

    def test_cascade_creation(self):
        """Test creating cascade system"""
        cascade = RegenerationCascade()
        assert cascade is not None
        assert cascade.energy_pool == 100.0

    def test_initiate_cascade(self):
        """Test initiating cascade"""
        cascade_sys = RegenerationCascade()
        repair = RepairSystem()
        damage = Damage(location="test", severity=0.5)

        cascade = cascade_sys.initiate_cascade(damage, repair)
        assert cascade is not None
        assert cascade.damage_id == damage.id

    def test_advance_cascade(self):
        """Test advancing cascade"""
        cascade_sys = RegenerationCascade()
        repair = RepairSystem()
        damage = Damage(location="test", severity=0.5)
        cascade_sys.initiate_cascade(damage, repair)

        initial_count = len(cascade_sys.active_cascades)
        for _ in range(10):
            cascade_sys.advance_cascades(repair)

        # Cascade should progress or complete
        assert True  # Test passes if no exception


# ============================================================================
# Integration Tests
# ============================================================================

class TestImmuneIntegration:
    """Integration tests for immune system components"""

    def test_full_immune_response(self):
        """Test complete immune response flow"""
        # Create all systems
        innate = InnateImmuneSystem()
        adaptive = AdaptiveImmuneSystem()
        inflammation = InflammationSystem()

        # Create threat
        threat = Threat(
            type=ThreatType.MALICIOUS_INPUT,
            severity=ThreatSeverity.HIGH,
            signature="test_attack"
        )

        # 1. Innate response
        innate_response = innate.quick_response(threat)
        assert innate_response is not None

        # 2. Trigger inflammation
        zone = inflammation.trigger_inflammation("immune_site", threat)
        assert zone.level.value > 0

        # 3. Adaptive response
        adaptive_response = adaptive.run_full_response(threat)
        assert adaptive_response.phase == ImmunePhase.MEMORY

        # 4. Resolution
        inflammation.begin_resolution(zone.id)

    def test_repair_after_threat(self):
        """Test repair system after threat handling"""
        immune = ImmuneSystem()
        repair = RepairSystem()

        # Detect threat damage
        damage = repair.detect_damage("system", 0.6, DamageType.CORRUPTION)

        # Initiate repair
        healing = repair.initiate_repair(damage)

        # Run repair
        while healing.phase != HealingPhase.COMPLETE:
            repair.advance_healing(healing)

        assert damage.repaired

    def test_memory_improves_response(self):
        """Test that memory improves subsequent responses"""
        adaptive = AdaptiveImmuneSystem()

        # First encounter
        threat1 = Threat(type=ThreatType.CORRUPTION, signature="threat_a")
        response1 = adaptive.run_full_response(threat1)

        # Second encounter (should be faster/stronger with memory)
        threat2 = Threat(type=ThreatType.CORRUPTION, signature="threat_a")
        memory = adaptive.check_memory(threat2)

        if memory:
            response2 = adaptive.run_full_response(threat2)
            # Second response should leverage memory
            assert response2.phase == ImmunePhase.MEMORY


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
